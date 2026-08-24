from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.governance import derive_evaluation_paths, verify_completion_receipt


ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "metrics"
FIGURES = ROOT / "reports" / "figures"
PROTOCOL = ROOT / "decision_protocol.json"


class FinalReportContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.summary = json.loads((METRICS / "experiment_summary.json").read_text())
        cls.budgets = pd.read_csv(METRICS / "budget_metrics.csv")
        cls.comparisons = pd.read_csv(METRICS / "policy_comparisons.csv")
        cls.source = pd.read_csv(METRICS / "source_metadata.csv")
        cls.validation_optima = pd.read_csv(
            METRICS / "validation_economic_optima.csv"
        )
        cls.final_economics = pd.read_csv(
            METRICS / "final_frozen_economic_decisions.csv"
        )
        cls.champion = pd.read_csv(
            METRICS / "champion_challenger_diagnostics.csv"
        )
        config = cls.summary["configuration"]
        cls.freeze_path = ROOT / config["frozen_decisions_path"]
        cls.claim_path = ROOT / config["final_access_claim_path"]
        cls.receipt_path = ROOT / config["completion_receipt_path"]
        cls.frozen = json.loads(cls.freeze_path.read_text())
        cls.claim = json.loads(cls.claim_path.read_text())
        cls.receipt = json.loads(cls.receipt_path.read_text())
        cls.promotion = json.loads((METRICS / "promotion_decision.json").read_text())

    def test_source_sampling_and_fresh_final_contract_are_recorded(self) -> None:
        config = self.summary["configuration"]
        self.assertEqual(config["source_row_count"], 13_979_592)
        self.assertEqual(config["development_sample_size"], 2_000_000)
        self.assertEqual(config["final_evaluation_sample_size"], 400_000)
        self.assertTrue(config["source_row_overlap_verified"])
        self.assertEqual(config["source_row_overlap_count"], 0)
        self.assertEqual(
            set(self.source["sample_role"]),
            {"development", "rehearsal_final_20260817", "final_evaluation"},
        )
        development = self.source.set_index("sample_role").loc["development"]
        rehearsal = self.source.set_index("sample_role").loc[
            "rehearsal_final_20260817"
        ]
        final = self.source.set_index("sample_role").loc["final_evaluation"]
        self.assertEqual(
            development["sampling_method"],
            "uniform_without_replacement_over_source_row_indices",
        )
        self.assertEqual(
            final["sampling_method"],
            "uniform_without_replacement_over_complement_of_all_pinned_source_row_indices",
        )
        self.assertNotEqual(
            development["selected_row_index_sha256"],
            final["selected_row_index_sha256"],
        )
        self.assertEqual(rehearsal["loaded_sample_size_in_current_run"], 0)
        self.assertEqual(
            final["access_status"], "loaded_after_immutable_final_access_claim"
        )
        self.assertEqual(
            config["final_overlap_with_pre_final_excluded_union_count"], 0
        )
        self.assertEqual(config["pre_final_excluded_union_row_count"], 2_400_000)
        self.assertEqual(config["continuous_features"], ["f0", "f2", "f7", "f10"])
        self.assertEqual(
            config["categorical_features"],
            ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"],
        )

    def test_protocol_and_validation_decisions_were_frozen(self) -> None:
        config = self.summary["configuration"]
        protocol_digest = hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
        frozen_digest = hashlib.sha256(self.freeze_path.read_bytes()).hexdigest()
        claim_digest = hashlib.sha256(self.claim_path.read_bytes()).hexdigest()
        receipt_digest = hashlib.sha256(self.receipt_path.read_bytes()).hexdigest()
        self.assertEqual(config["decision_protocol_sha256"], protocol_digest)
        self.assertEqual(config["frozen_decisions_sha256"], frozen_digest)
        self.assertEqual(config["final_access_claim_sha256"], claim_digest)
        self.assertEqual(
            config["legacy_result_status"],
            "exploratory_not_eligible_for_new_budget_selection",
        )
        decision = self.frozen["decision"]
        self.assertTrue(decision["frozen_before_final_outcome_access"])
        self.assertEqual(
            decision["sample_plan"]["final_overlap_with_excluded_union_count"], 0
        )
        self.assertEqual(len(decision["frozen_decisions"]), 12)
        self.assertEqual(self.claim["outcome"], "conversion")
        self.assertEqual(
            self.claim["final_index_sha256"],
            config["final_evaluation_selected_row_index_sha256"],
        )
        verified = verify_completion_receipt(
            derive_evaluation_paths(METRICS, config["evaluation_id"]),
            protocol_path=PROTOCOL,
            code_root=ROOT,
            output_root=ROOT / "reports",
            expected_freeze_sha256=frozen_digest,
            expected_claim_sha256=claim_digest,
            expected_receipt_sha256=receipt_digest,
        )
        self.assertEqual(verified["receipt_sha256"], receipt_digest)
        self.assertTrue(self.validation_optima["selected_on"].eq("validation").all())
        self.assertEqual(len(self.validation_optima), 12)
        self.assertTrue(self.final_economics["decision_frozen"].all())
        self.assertTrue(self.final_economics["final_evaluation_policy_frozen"].all())
        merged = self.validation_optima.merge(
            self.final_economics,
            on=["scenario_label", "decision_scope"],
            suffixes=("_validation", "_final"),
        )
        self.assertTrue(merged["selected_policy"].eq(merged["policy"]).all())
        self.assertTrue(
            np.allclose(
                merged["requested_budget_validation"],
                merged["requested_budget_final"],
            )
        )

    def test_final_models_bootstraps_and_invariance_pass(self) -> None:
        self.assertEqual(self.summary["configuration"]["bootstrap_repetitions"], 1_000)
        models = pd.read_csv(METRICS / "model_diagnostics.csv")
        self.assertTrue(models["fit_converged"].all())
        treatment = pd.read_csv(METRICS / "treatment_diagnostics.csv")
        self.assertTrue(treatment["fit_converged"].all())
        self.assertEqual(
            treatment.iloc[0]["feature_columns"], ",".join(f"f{i}" for i in range(12))
        )
        invariance = pd.read_csv(METRICS / "score_invariance.csv")
        self.assertTrue(invariance["scores_exactly_unchanged"].all())
        self.assertTrue((invariance["max_absolute_score_change"] == 0.0).all())
        data = pd.read_csv(METRICS / "data_summary.csv").set_index("sample")
        self.assertEqual(data.loc["final_evaluation", "bootstrap_repetitions"], 1_000)

    def test_analytical_random_and_paired_comparison_contract(self) -> None:
        policies = set(self.budgets["policy"])
        self.assertIn("expected_random", policies)
        self.assertNotIn("random", policies)
        direct = self.comparisons[self.comparisons["reference_policy"].eq("response")]
        budget_count = self.budgets[self.budgets["policy"].eq("response")][
            "budget"
        ].nunique()
        self.assertEqual(len(direct), 2 * budget_count)
        self.assertTrue(direct["delta_policy_value_ci_low"].notna().all())
        self.assertTrue(direct["delta_hajek_policy_value_ci_low"].notna().all())

    def test_ht_hajek_and_full_budget_identities(self) -> None:
        twenty = self.budgets[np.isclose(self.budgets["budget"], 0.2)].set_index("policy")
        self.assertNotAlmostEqual(
            twenty.loc["s_learner", "incremental_outcomes_per_1000_targeted"],
            twenty.loc[
                "s_learner", "hajek_incremental_outcomes_per_1000_targeted"
            ],
        )
        full = self.budgets[np.isclose(self.budgets["budget"], 1.0)]
        data = pd.read_csv(METRICS / "data_summary.csv").set_index("sample")
        final = data.loc["final_evaluation"]
        self.assertTrue(
            np.allclose(full["policy_value"], final["outcome_rate_treated"], atol=1e-12)
        )
        self.assertTrue(
            np.allclose(
                full["hajek_policy_value"], final["outcome_rate_treated"], atol=1e-12
            )
        )

    def test_champion_challenger_and_promotion_contract(self) -> None:
        response_t = self.champion[self.champion["challenger_policy"].eq("t_learner")]
        top = response_t[response_t["budget"].isin((0.1, 0.2))]
        self.assertEqual(set(top["budget"]), {0.1, 0.2})
        self.assertTrue(top["jaccard"].between(0.0, 1.0).all())
        self.assertTrue(
            top["total_switched_count"].eq(
                top["champion_only_count"] + top["challenger_only_count"]
            ).all()
        )
        self.assertTrue(top["ht_decomposition_holds"].all())
        self.assertTrue(
            np.allclose(
                top["ht_policy_value_decomposition_residual"], 0.0, atol=1e-12
            )
        )
        self.assertEqual(self.promotion["champion_policy"], "response")
        self.assertEqual(self.promotion["challenger_policy"], "t_learner")
        self.assertIsNone(self.promotion["noninferiority_rule"])
        self.assertTrue(
            np.isfinite(
                [
                    self.promotion["delta_estimate"],
                    self.promotion["ci_low"],
                    self.promotion["ci_high"],
                    self.promotion["hajek_net_value_delta"],
                ]
            ).all()
        )
        self.assertLessEqual(self.promotion["ci_low"], self.promotion["ci_high"])
        expected_pass = (
            self.promotion["ci_low"] > 0.0
            and self.promotion["hajek_net_value_delta"] > 0.0
        )
        self.assertEqual(self.promotion["rule_passed"], expected_pass)
        self.assertEqual(
            self.promotion["superiority_demonstrated"],
            self.promotion["ci_low"] > 0.0,
        )

    def test_result_card_keeps_each_conclusion_with_its_estimand(self) -> None:
        card = json.loads((METRICS / "result_card.json").read_text())
        diagnostic = card["fresh_final_20_percent_diagnostic"]
        for policy in ("s_learner", "t_learner"):
            report = diagnostic[policy]
            comparison = self.comparisons[
                self.comparisons["candidate_policy"].eq(policy)
                & self.comparisons["reference_policy"].eq("response")
                & np.isclose(self.comparisons["candidate_budget"], 0.2)
            ].iloc[0]
            self.assertEqual(
                report["ht_minus_response_conclusion"],
                comparison["incremental_conclusion"],
            )
            self.assertEqual(
                report["primary_policy_value_conclusion"],
                comparison["policy_value_conclusion"],
            )
            self.assertTrue(
                np.allclose(
                    report["ht_minus_response_ci"],
                    [
                        comparison["delta_incremental_per_1000_ci_low"],
                        comparison["delta_incremental_per_1000_ci_high"],
                    ],
                )
            )
            self.assertTrue(
                np.allclose(
                    report["primary_policy_value_minus_response_ci"],
                    [
                        comparison["delta_policy_value_ci_low"],
                        comparison["delta_policy_value_ci_high"],
                    ],
                )
            )

    def test_economic_scenarios_include_nobody_optima_and_break_even(self) -> None:
        grid = pd.read_csv(METRICS / "validation_economic_grid.csv")
        nobody = grid[grid["policy"].eq("target_nobody")]
        self.assertEqual(len(nobody), grid["scenario_label"].nunique())
        self.assertTrue(nobody["actual_budget"].eq(0.0).all())
        self.assertTrue(nobody["incremental_net_value_vs_target_nobody"].eq(0.0).all())
        global_optima = self.validation_optima[
            self.validation_optima["decision_scope"].eq("global_response_vs_t")
        ]
        self.assertEqual(len(global_optima), 4)
        positive = self.final_economics[~self.final_economics["is_target_nobody"]]
        expected_break_even = (
            positive["margin_per_conversion"]
            * positive["gain_vs_treat_none"]
            / positive["actual_budget"]
        )
        self.assertTrue(
            np.allclose(
                positive["break_even_cost_per_assigned_target"], expected_break_even
            )
        )
        primary = self.final_economics[
            self.final_economics["scenario_role"].eq("primary")
            & self.final_economics["decision_scope"].eq("global_response_vs_t")
        ].iloc[0]
        self.assertTrue(
            np.isfinite(
                [
                    primary["final_incremental_net_value_ci_low"],
                    primary["final_incremental_net_value_ci_high"],
                ]
            ).all()
        )
        self.assertLessEqual(
            primary["final_incremental_net_value_ci_low"],
            primary["final_incremental_net_value_ci_high"],
        )
        expected_conclusion = (
            "baseline_target_nobody"
            if primary["is_target_nobody"]
            else "positive"
            if primary["final_incremental_net_value_ci_low"] > 0.0
            else "negative"
            if primary["final_incremental_net_value_ci_high"] < 0.0
            else "inconclusive"
        )
        self.assertEqual(
            primary["final_net_value_conclusion_vs_target_nobody"],
            expected_conclusion,
        )

    def test_all_declared_artifacts_exist_and_are_nonempty(self) -> None:
        artifacts = self.summary["artifacts"]
        paths = [ROOT / path for path in artifacts["metric_tables"]]
        paths.extend(ROOT / path for path in artifacts["figures"])
        paths.extend(
            ROOT / artifacts[key]
            for key in (
                "result_card",
                "frozen_decisions",
                "final_access_claim",
                "completion_receipt",
                "promotion_decision",
            )
        )
        for path in paths:
            self.assertTrue(path.is_file(), path)
            self.assertGreater(path.stat().st_size, 0, path)
        self.assertEqual(len(list(FIGURES.glob("*.png"))), 7)

    def test_active_credibility_errata_uses_bounded_terminology(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        errata = (ROOT / "ERRATA.md").read_text(encoding="utf-8")
        presentation_readme = (ROOT / "presentation" / "README.md").read_text(
            encoding="utf-8"
        )
        active_text = " ".join(
            "\n".join((readme, errata, presentation_readme)).split()
        )
        for required in (
            "conditional row-level paired-bootstrap",
            "empirical-marginal-propensity inverse-probability weighting",
            "S-learner was not excluded through an executable screening test",
            "benchmark champion among the evaluated fixed linear baseline models",
            "2p + 1 = 10,461",
            "offline causal targeting and policy-evaluation case study",
        ):
            self.assertIn(required.lower(), active_text.lower())


if __name__ == "__main__":
    unittest.main()
