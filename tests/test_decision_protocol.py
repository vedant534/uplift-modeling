from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_experiment import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_FINAL_EVALUATION_SAMPLE_SIZE,
    DEFAULT_FINAL_EVALUATION_SEED,
    DEFAULT_SAMPLE_SIZE,
    _load_decision_protocol,
    _protocol_budget_grid,
    _protocol_scenarios,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "decision_protocol.json"


class DecisionProtocolTests(unittest.TestCase):
    def _args(self) -> argparse.Namespace:
        return argparse.Namespace(
            sample_size=DEFAULT_SAMPLE_SIZE,
            seed=42,
            final_evaluation_sample_size=DEFAULT_FINAL_EVALUATION_SAMPLE_SIZE,
            final_evaluation_seed=DEFAULT_FINAL_EVALUATION_SEED,
            bootstrap_reps=DEFAULT_BOOTSTRAP_REPS,
            validation_bootstrap_reps=100,
            outcome="conversion",
            output_dir=ROOT / "reports",
        )

    def test_protocol_freezes_honest_selection_and_final_sample_contract(self) -> None:
        protocol, digest = _load_decision_protocol(PROTOCOL, self._args())
        self.assertEqual(len(digest), 64)
        self.assertEqual(protocol["protocol_version"], 2)
        self.assertEqual(protocol["data_contract"]["outcome"], "conversion")
        self.assertEqual(
            protocol["legacy_result_status"],
            "exploratory_not_eligible_for_new_budget_selection",
        )
        self.assertEqual(protocol["selection"]["sample"], "development_validation")
        self.assertIn("target_nobody", protocol["selection"]["candidate_policies"])
        self.assertIn(
            "previously accessed final-evaluation sample",
            protocol["data_contract"]["final_evaluation_requirement"],
        )
        self.assertEqual(
            protocol["rehearsal_result_status"],
            "invalidated_for_confirmatory_use_final_frame_was_materialized_before_freeze",
        )
        self.assertTrue(protocol["execution_contract"]["one_shot_final_evaluation"])
        self.assertEqual(
            protocol["execution_contract"]["source_access_order"][-1],
            "load_final_evaluation_only",
        )
        self.assertEqual(
            protocol["champion_challenger"]["overlap_budgets"], [0.1, 0.2]
        )
        self.assertIsNone(
            protocol["champion_challenger"]["noninferiority_rule"]
        )

    def test_budget_grid_and_scenarios_are_explicit_and_unique(self) -> None:
        protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        budgets = _protocol_budget_grid(protocol)
        self.assertEqual(len(budgets), 100)
        self.assertTrue(np.isclose(budgets[0], 0.01))
        self.assertTrue(np.isclose(budgets[-1], 1.0))
        scenarios = _protocol_scenarios(protocol)
        self.assertEqual(sum(role == "primary" for role, _ in scenarios), 1)
        self.assertEqual(len({scenario.label for _, scenario in scenarios}), 4)
        for _, scenario in scenarios:
            self.assertTrue(scenario.currency)
            self.assertTrue(scenario.horizon)
            self.assertGreater(scenario.eligible_population, 0)

    def test_protocol_rejects_endpoint_runtime_and_semantic_drift(self) -> None:
        endpoint_args = self._args()
        endpoint_args.outcome = "visit"
        with self.assertRaisesRegex(ValueError, "Runtime arguments"):
            _load_decision_protocol(PROTOCOL, endpoint_args)

        output_args = self._args()
        output_args.output_dir = ROOT / "different-reports"
        with self.assertRaisesRegex(ValueError, "Runtime arguments"):
            _load_decision_protocol(PROTOCOL, output_args)

        original = json.loads(PROTOCOL.read_text(encoding="utf-8"))
        mutations = (
            ("primary estimator", ("selection", "primary_estimator"), "hajek"),
            ("tie break", ("selection", "tie_break_order"), ["policy_name"]),
            (
                "promotion rule",
                ("champion_challenger", "promotion_rule_id"),
                "noninferiority",
            ),
        )
        for label, keys, replacement in mutations:
            mutated = copy.deepcopy(original)
            mutated[keys[0]][keys[1]] = replacement
            with self.subTest(label=label), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "protocol.json"
                path.write_text(json.dumps(mutated), encoding="utf-8")
                with self.assertRaises(ValueError):
                    _load_decision_protocol(path, self._args())


if __name__ == "__main__":
    unittest.main()
