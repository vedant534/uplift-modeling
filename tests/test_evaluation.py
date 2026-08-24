from __future__ import annotations

import unittest

import numpy as np

from src.diagnostics import score_invariance_table
from src.evaluation import evaluate_policies


def exact_randomized_fixture() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cell = 5_000
    high = np.r_[np.ones(2 * cell), np.zeros(2 * cell)]
    treatment = np.tile(np.r_[np.ones(cell), np.zeros(cell)], 2)
    outcome = np.zeros(4 * cell)
    outcome[:1_500] = 1
    outcome[cell : cell + 500] = 1
    outcome[2 * cell : 2 * cell + 500] = 1
    outcome[3 * cell : 3 * cell + 500] = 1
    rng = np.random.default_rng(20260817)
    permutation = rng.permutation(len(outcome))
    high = high[permutation]
    treatment = treatment[permutation]
    outcome = outcome[permutation]
    oracle = high + rng.uniform(0.0, 0.01, len(high))
    reverse = -high + rng.uniform(0.0, 0.01, len(high))
    return outcome, treatment, {
        "response": reverse,
        "s_learner": oracle,
        "t_learner": oracle,
    }


class EvaluationTests(unittest.TestCase):
    def test_known_ate_qini_orientation_policy_order_and_full_budget(self) -> None:
        outcome, treatment, scores = exact_randomized_fixture()
        results = evaluate_policies(
            outcome,
            treatment,
            scores,
            budgets=(0.5, 1.0),
            n_bootstrap=40,
            random_state=11,
            curve_points=11,
        )
        ranking = results["ranking_metrics"].set_index("policy")
        self.assertAlmostEqual(ranking.loc["expected_random", "qini_coefficient"], 0.0)
        self.assertGreater(ranking.loc["s_learner", "qini_coefficient"], 0.0)
        self.assertLess(ranking.loc["response", "qini_coefficient"], 0.0)
        self.assertAlmostEqual(ranking.loc["s_learner", "ate_curve_endpoint"], 0.1)

        half = results["budget_metrics"]
        half = half[np.isclose(half["budget"], 0.5)].set_index("policy")
        self.assertAlmostEqual(half.loc["expected_random", "policy_value"], 0.15)
        self.assertAlmostEqual(half.loc["response", "policy_value"], 0.10)
        self.assertAlmostEqual(half.loc["s_learner", "policy_value"], 0.20)
        self.assertAlmostEqual(
            half.loc["s_learner", "incremental_outcomes_per_1000_targeted"],
            200.0,
        )
        self.assertAlmostEqual(
            half.loc[
                "s_learner", "hajek_incremental_outcomes_per_1000_targeted"
            ],
            200.0,
        )
        full = results["budget_metrics"]
        full = full[np.isclose(full["budget"], 1.0)]
        self.assertTrue(np.allclose(full["policy_value"], 0.20, atol=1e-12))
        self.assertTrue(np.allclose(full["hajek_policy_value"], 0.20, atol=1e-12))

    def test_ht_and_within_selected_hajek_are_distinct_and_correct(self) -> None:
        n = 1_000
        treatment = np.zeros(n)
        treatment[:172] = 1
        treatment[200:878] = 1
        outcome = np.zeros(n)
        outcome[:20] = 1
        outcome[172:175] = 1
        outcome[200:210] = 1
        outcome[878:888] = 1
        score = np.r_[np.ones(200), np.zeros(800)]
        scores = {name: score.copy() for name in ("response", "s_learner", "t_learner")}
        results = evaluate_policies(
            outcome,
            treatment,
            scores,
            budgets=(0.2, 1.0),
            n_bootstrap=20,
            random_state=5,
            curve_points=11,
        )
        row = results["budget_metrics"]
        row = row[(row["policy"] == "s_learner") & np.isclose(row["budget"], 0.2)].iloc[0]
        expected_ht = 1000.0 * ((20 / 0.85) - (3 / 0.15)) / 200
        expected_hajek = 1000.0 * (20 / 172 - 3 / 28)
        self.assertAlmostEqual(row["incremental_outcomes_per_1000_targeted"], expected_ht)
        self.assertAlmostEqual(
            row["hajek_incremental_outcomes_per_1000_targeted"], expected_hajek
        )
        self.assertNotAlmostEqual(expected_ht, expected_hajek)
        self.assertNotIn("random", set(results["budget_metrics"]["policy"]))
        self.assertIn("expected_random", set(results["budget_metrics"]["policy"]))

    def test_score_invariance_requires_exact_equality(self) -> None:
        original = {
            "response": np.array([0.1, 0.2]),
            "s_learner": np.array([0.0, 0.3]),
        }
        table = score_invariance_table(original, {k: v.copy() for k, v in original.items()})
        self.assertTrue(table["scores_exactly_unchanged"].all())
        changed = {k: v.copy() for k, v in original.items()}
        changed["response"][0] += 1e-12
        with self.assertRaisesRegex(AssertionError, "scores changed"):
            score_invariance_table(original, changed)


if __name__ == "__main__":
    unittest.main()
