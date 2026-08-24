from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

import numpy as np
import pandas as pd

from src.economics import (
    SELECTION_RULE,
    TARGET_NOBODY_POLICY,
    EconomicScenario,
    evaluate_economic_scenario,
    evaluate_frozen_economic_decision,
    select_and_freeze_economic_policy,
    select_economic_policy,
    summarize_paired_net_value_bootstrap,
    transform_paired_gain_bootstrap,
)


def qini_curve(
    rows: list[tuple[str, float, float]],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "policy": policy,
                "targeting_fraction": budget,
                "actual_targeting_fraction": budget,
                "targeted_count": int(1_000 * budget),
                "cumulative_incremental_rate": gain,
            }
            for policy, budget, gain in rows
        ]
    )


class EconomicScenarioTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scenario = EconomicScenario(
            label="Base margin / assigned-target cost",
            margin_per_conversion=100.0,
            cost_per_assigned_target=0.5,
            eligible_population=1_000,
            currency="USD",
            horizon="one campaign",
        )

    def test_scenario_contract_rejects_ambiguous_or_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "label"):
            EconomicScenario(" ", 10.0, 1.0, 100, "USD", "month")
        with self.assertRaisesRegex(ValueError, "margin_per_conversion"):
            EconomicScenario("bad", -1.0, 1.0, 100, "USD", "month")
        with self.assertRaisesRegex(ValueError, "cost_per_assigned_target"):
            EconomicScenario("bad", 1.0, np.inf, 100, "USD", "month")
        with self.assertRaisesRegex(ValueError, "eligible_population"):
            EconomicScenario("bad", 1.0, 1.0, 100.5, "USD", "month")
        with self.assertRaisesRegex(ValueError, "currency"):
            EconomicScenario("bad", 1.0, 1.0, 100, "", "month")
        with self.assertRaisesRegex(ValueError, "horizon"):
            EconomicScenario("bad", 1.0, 1.0, 100, "USD", "")

    def test_qini_curve_is_monetized_with_one_explicit_nobody_row(self) -> None:
        curve = qini_curve(
            [
                ("response", 0.0, 0.0),
                ("response", 0.2, 0.0020),
                ("response", 0.4, 0.0025),
                ("t_learner", 0.0, 0.0),
                ("t_learner", 0.2, 0.0018),
            ]
        )
        result = evaluate_economic_scenario(curve, self.scenario)

        nobody = result[result["policy"].eq(TARGET_NOBODY_POLICY)]
        self.assertEqual(len(nobody), 1)
        self.assertEqual(float(nobody.iloc[0]["actual_budget"]), 0.0)
        self.assertEqual(
            float(nobody.iloc[0]["incremental_net_value_vs_target_nobody"]), 0.0
        )
        self.assertTrue(
            np.isnan(nobody.iloc[0]["break_even_cost_per_assigned_target"])
        )

        response = result[
            result["policy"].eq("response") & np.isclose(result["actual_budget"], 0.2)
        ].iloc[0]
        self.assertEqual(response["scenario_label"], self.scenario.label)
        self.assertEqual(response["currency"], "USD")
        self.assertEqual(response["horizon"], "one campaign")
        self.assertAlmostEqual(response["assigned_targets"], 200.0)
        self.assertAlmostEqual(response["estimated_incremental_conversions"], 2.0)
        self.assertAlmostEqual(response["incremental_conversion_margin"], 200.0)
        self.assertAlmostEqual(response["targeting_cost"], 100.0)
        self.assertAlmostEqual(
            response["incremental_net_value_vs_target_nobody"], 100.0
        )
        self.assertAlmostEqual(
            response["incremental_net_value_per_1000_eligible"], 100.0
        )
        self.assertAlmostEqual(
            response["break_even_cost_to_margin_ratio"], 0.01
        )
        self.assertAlmostEqual(
            response["break_even_cost_per_assigned_target"], 1.0
        )

    def test_budget_metrics_schema_is_inferred(self) -> None:
        metrics = pd.DataFrame(
            {
                "policy": ["response"],
                "budget": [0.2],
                "actual_budget": [0.2],
                "gain_vs_treat_none": [0.002],
            }
        )
        result = evaluate_economic_scenario(metrics, self.scenario)
        self.assertEqual(set(result["policy"]), {TARGET_NOBODY_POLICY, "response"})
        self.assertAlmostEqual(
            result[result["policy"].eq("response")].iloc[0][
                "incremental_net_value_vs_target_nobody"
            ],
            100.0,
        )

    def test_gain_curve_validation_catches_zero_gain_and_duplicate_candidates(self) -> None:
        nonzero_at_zero = qini_curve([("response", 0.0, 0.001), ("response", 0.2, 0.002)])
        with self.assertRaisesRegex(ValueError, "zero-budget gain"):
            evaluate_economic_scenario(nonzero_at_zero, self.scenario)

        duplicate = qini_curve(
            [("response", 0.2, 0.001), ("response", 0.2, 0.002)]
        )
        with self.assertRaisesRegex(ValueError, "duplicate policy/actual-budget"):
            evaluate_economic_scenario(duplicate, self.scenario)


class EconomicDecisionTests(unittest.TestCase):
    def test_tie_break_prefers_lower_budget_before_champion(self) -> None:
        scenario = EconomicScenario("ties", 100.0, 0.0, 1_000, "USD", "campaign")
        result = evaluate_economic_scenario(
            qini_curve(
                [
                    ("response", 0.2, 0.002),
                    ("t_learner", 0.1, 0.002),
                ]
            ),
            scenario,
        )
        selected = select_economic_policy(result, champion_policy="response")
        self.assertEqual(selected["policy"], "t_learner")
        self.assertEqual(float(selected["actual_budget"]), 0.1)
        self.assertEqual(selected["selection_rule"], SELECTION_RULE)

    def test_tie_break_prefers_champion_at_same_budget(self) -> None:
        scenario = EconomicScenario("ties", 100.0, 0.0, 1_000, "USD", "campaign")
        result = evaluate_economic_scenario(
            qini_curve(
                [
                    ("response", 0.2, 0.002),
                    ("t_learner", 0.2, 0.002),
                ]
            ),
            scenario,
        )
        selected = select_economic_policy(result, champion_policy="response")
        self.assertEqual(selected["policy"], "response")

    def test_target_nobody_wins_and_frozen_record_is_immutable(self) -> None:
        scenario = EconomicScenario(
            "high assignment cost", 10.0, 1.0, 1_000, "USD", "campaign"
        )
        result = evaluate_economic_scenario(
            qini_curve(
                [
                    ("response", 0.1, 0.001),
                    ("t_learner", 0.1, -0.001),
                ]
            ),
            scenario,
        )
        decision = select_and_freeze_economic_policy(
            result, scenario, champion_policy="response"
        )
        self.assertTrue(decision.targets_nobody)
        self.assertEqual(decision.selected_policy, TARGET_NOBODY_POLICY)
        self.assertEqual(decision.validation_incremental_net_value, 0.0)
        record = decision.to_record()
        self.assertEqual(record["scenario_label"], "high assignment cost")
        self.assertEqual(record["selected_on"], "validation")
        self.assertEqual(record["selection_rule"], SELECTION_RULE)
        with self.assertRaises(FrozenInstanceError):
            decision.selected_policy = "response"  # type: ignore[misc]

    def test_final_evaluation_uses_frozen_policy_instead_of_reselecting(self) -> None:
        scenario = EconomicScenario("freeze", 100.0, 0.2, 1_000, "USD", "campaign")
        validation = evaluate_economic_scenario(
            qini_curve(
                [
                    ("response", 0.2, 0.0020),
                    ("t_learner", 0.2, 0.0010),
                ]
            ),
            scenario,
        )
        decision = select_and_freeze_economic_policy(validation, scenario)
        self.assertEqual(decision.selected_policy, "response")

        final_curve = qini_curve(
            [
                ("response", 0.2, 0.0005),
                ("t_learner", 0.2, 0.0100),
            ]
        )
        final = evaluate_frozen_economic_decision(final_curve, decision)
        self.assertEqual(final["policy"], "response")
        self.assertAlmostEqual(final["gain_vs_treat_none"], 0.0005)
        self.assertAlmostEqual(
            final["incremental_net_value_vs_target_nobody"], 10.0
        )
        self.assertTrue(final["decision_frozen"])
        self.assertAlmostEqual(final["validation_incremental_net_value"], 160.0)


class EconomicBootstrapTests(unittest.TestCase):
    def test_paired_gain_replicates_transform_to_net_value(self) -> None:
        scenario = EconomicScenario(
            "bootstrap", 100.0, 0.5, 1_000, "USD", "campaign"
        )
        candidate = np.array([0.002, 0.003, np.nan])
        transformed = transform_paired_gain_bootstrap(candidate, scenario, 0.2)
        np.testing.assert_allclose(transformed[:2], [100.0, 200.0])
        self.assertTrue(np.isnan(transformed[2]))

        reference = np.array([0.001, 0.001, np.nan])
        paired = transform_paired_gain_bootstrap(
            candidate,
            scenario,
            0.2,
            reference_gain_replicates=reference,
            reference_actual_budget=0.1,
        )
        np.testing.assert_allclose(paired[:2], [50.0, 150.0])

        summary = summarize_paired_net_value_bootstrap(
            candidate, scenario, 0.2, confidence_level=0.95
        )
        self.assertEqual(summary["valid_paired_bootstrap_replicates"], 2)
        self.assertAlmostEqual(
            summary["bootstrap_median_net_value_difference"], 150.0
        )
        self.assertAlmostEqual(
            summary["bootstrap_probability_net_value_difference_positive"], 1.0
        )
        self.assertAlmostEqual(summary["net_value_difference_ci_low"], 102.5)
        self.assertAlmostEqual(summary["net_value_difference_ci_high"], 197.5)

    def test_paired_bootstrap_requires_aligned_finite_pairs(self) -> None:
        scenario = EconomicScenario(
            "bootstrap", 100.0, 0.5, 1_000, "USD", "campaign"
        )
        with self.assertRaisesRegex(ValueError, "equal shape"):
            transform_paired_gain_bootstrap(
                [0.1, 0.2],
                scenario,
                0.2,
                reference_gain_replicates=[0.1],
            )
        with self.assertRaisesRegex(ValueError, "no finite paired"):
            transform_paired_gain_bootstrap([np.nan], scenario, 0.2)


if __name__ == "__main__":
    unittest.main()
