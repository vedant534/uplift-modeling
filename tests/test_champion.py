from __future__ import annotations

import unittest

import numpy as np

from src.champion import assess_promotion, champion_challenger_diagnostics


class ChampionChallengerDiagnosticTests(unittest.TestCase):
    def test_overlap_uplift_and_exact_ht_decomposition(self) -> None:
        n = 12
        treatment = np.tile(np.array([1.0, 0.0]), n // 2)
        outcome = np.zeros(n)
        outcome[0] = 1.0
        outcome[7] = 1.0
        response = np.array([12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1])
        challenger = np.array([2, 1, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3])

        result = champion_challenger_diagnostics(
            outcome,
            treatment,
            {"response": response, "t_learner": challenger},
            budgets=(0.5,),
            challengers=("t_learner",),
            n_bootstrap=120,
            random_state=17,
            propensity=0.5,
        )
        row = result.iloc[0]
        self.assertEqual(row["targeted_count"], 6)
        self.assertEqual(row["intersection_count"], 4)
        self.assertEqual(row["union_count"], 8)
        self.assertAlmostEqual(row["jaccard"], 0.5)
        self.assertEqual(row["champion_only_count"], 2)
        self.assertEqual(row["challenger_only_count"], 2)
        self.assertEqual(row["total_switched_count"], 4)
        self.assertAlmostEqual(row["champion_only_ht_uplift_per_1000"], 1000.0)
        self.assertAlmostEqual(row["challenger_only_ht_uplift_per_1000"], -1000.0)
        self.assertAlmostEqual(row["switched_ht_uplift_delta_per_1000"], -2000.0)
        self.assertAlmostEqual(
            row["challenger_minus_champion_ht_policy_value"], -1.0 / 3.0
        )
        self.assertAlmostEqual(
            row["challenger_minus_champion_ht_per_1000_nominal_targeted"],
            -2000.0 / 3.0,
        )
        self.assertAlmostEqual(
            row["decomposed_ht_per_1000_nominal_targeted"], -2000.0 / 3.0
        )
        self.assertTrue(row["ht_decomposition_holds"])
        self.assertAlmostEqual(row["ht_policy_value_decomposition_residual"], 0.0)
        self.assertGreater(row["valid_ht_cohort_bootstrap_reps"], 0)
        self.assertGreater(row["valid_hajek_cohort_bootstrap_reps"], 0)
        self.assertTrue(
            np.isfinite(row["switched_ht_uplift_delta_per_1000_ci_low"])
        )

    def test_bootstrap_is_deterministic_and_ties_use_stable_row_order(self) -> None:
        outcome = np.array([1, 0, 0, 0, 0, 1, 0, 0], dtype=float)
        treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        scores = {
            "response": np.ones(8),
            "challenger": np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float),
        }
        kwargs = {
            "budgets": (0.25,),
            "challengers": ("challenger",),
            "n_bootstrap": 50,
            "random_state": 9,
        }
        first = champion_challenger_diagnostics(
            outcome, treatment, scores, **kwargs
        )
        second = champion_challenger_diagnostics(
            outcome, treatment, scores, **kwargs
        )
        self.assertEqual(first.loc[0, "intersection_count"], 1)
        self.assertEqual(first.loc[0, "champion_only_count"], 1)
        self.assertEqual(first.loc[0, "challenger_only_count"], 1)
        for suffix in ("ci_low", "ci_high"):
            column = f"challenger_minus_champion_ht_policy_value_{suffix}"
            self.assertEqual(first.loc[0, column], second.loc[0, column])

    def test_identical_rankings_handle_empty_exclusive_cohorts(self) -> None:
        outcome = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        treatment = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=float)
        score = np.arange(8, 0, -1, dtype=float)
        result = champion_challenger_diagnostics(
            outcome,
            treatment,
            {"response": score, "copy": score.copy()},
            budgets=(0.5,),
            n_bootstrap=30,
            random_state=4,
        )
        row = result.iloc[0]
        self.assertEqual(row["jaccard"], 1.0)
        self.assertEqual(row["total_switched_count"], 0)
        self.assertTrue(np.isnan(row["champion_only_ht_uplift_per_1000"]))
        self.assertTrue(np.isnan(row["challenger_only_hajek_uplift_per_1000"]))
        self.assertEqual(row["challenger_minus_champion_ht_policy_value"], 0.0)
        self.assertEqual(
            row["challenger_minus_champion_ht_policy_value_ci_low"], 0.0
        )
        self.assertEqual(
            row["challenger_minus_champion_ht_policy_value_ci_high"], 0.0
        )
        self.assertTrue(row["ht_decomposition_holds"])
        self.assertEqual(row["valid_ht_cohort_bootstrap_reps"], 0)

    def test_input_validation_rejects_unevaluable_contracts(self) -> None:
        outcome = np.array([0, 1, 0, 1], dtype=float)
        treatment = np.array([0, 1, 0, 1], dtype=float)
        scores = {"response": np.arange(4), "t": np.arange(4)}
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            champion_challenger_diagnostics(
                outcome,
                treatment,
                scores,
                budgets=(0.5, 0.25),
                n_bootstrap=2,
            )
        with self.assertRaisesRegex(ValueError, "champion cannot"):
            champion_challenger_diagnostics(
                outcome,
                treatment,
                scores,
                budgets=(0.5,),
                challengers=("response",),
                n_bootstrap=2,
            )
        with self.assertRaisesRegex(ValueError, "both treatment arms"):
            champion_challenger_diagnostics(
                outcome,
                np.ones(4),
                scores,
                budgets=(0.5,),
                n_bootstrap=2,
                propensity=0.5,
            )


class PromotionAssessmentTests(unittest.TestCase):
    def test_superiority_rule_requires_positive_lower_bound(self) -> None:
        passed = assess_promotion(0.2, 0.05, 0.35, metric="paired_net_value")
        self.assertTrue(passed["rule_passed"])
        self.assertTrue(passed["superiority_demonstrated"])
        self.assertIn("superiority_rule", passed["decision"])

        inconclusive = assess_promotion(0.2, -0.05, 0.35)
        self.assertFalse(inconclusive["rule_passed"])
        self.assertEqual(inconclusive["evidence_status"], "difference_inconclusive")
        self.assertIn("retain_champion", inconclusive["decision"])

    def test_noninferiority_is_not_mislabeled_superiority(self) -> None:
        assessment = assess_promotion(
            -0.02,
            -0.05,
            0.01,
            metric="paired_policy_value_delta",
            rule="noninferiority",
            noninferiority_margin=0.10,
        )
        self.assertTrue(assessment["rule_passed"])
        self.assertTrue(assessment["noninferiority_demonstrated"])
        self.assertFalse(assessment["superiority_demonstrated"])
        self.assertEqual(assessment["evidence_status"], "difference_inconclusive")
        self.assertIn("noninferiority_rule", assessment["decision"])

        with self.assertRaisesRegex(ValueError, "required"):
            assess_promotion(0.0, -0.1, 0.1, rule="noninferiority")


if __name__ == "__main__":
    unittest.main()
