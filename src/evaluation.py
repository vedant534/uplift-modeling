"""Held-out causal policy evaluation for randomized advertising experiments.

Horvitz--Thompson (HT) is the pre-specified primary policy estimator.  Hájek
self-normalized policy values and within-selected arm-rate contrasts are
reported as sensitivity estimates.  Expected random allocation is analytical,
so inferential comparisons do not depend on one arbitrary permutation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_BUDGETS = (0.10, 0.20, 0.40, 0.60, 0.80, 1.00)
SCORED_POLICIES = ("response", "s_learner", "t_learner")
POLICIES = ("expected_random", *SCORED_POLICIES)
UPLIFT_POLICIES = ("s_learner", "t_learner")
N_DECILES = 10


def _binary_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")
    if not np.isin(np.unique(array), (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.float64, copy=False)


def _score_array(values: Sequence[float], name: str, n_rows: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or len(array) != n_rows:
        raise ValueError(f"score '{name}' must be one-dimensional with {n_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"score '{name}' contains NaN or infinite values")
    return array


def _validated_budgets(
    budgets: Sequence[float], n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(tuple(budgets), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("budgets must be a non-empty one-dimensional sequence")
    if not np.isfinite(values).all() or np.any(values <= 0.0) or np.any(values > 1.0):
        raise ValueError("every budget must be finite and in (0, 1]")
    if np.any(np.diff(values) <= 0.0):
        raise ValueError("budgets must be strictly increasing")
    counts = np.maximum(1, np.floor(values * n_rows).astype(np.int64))
    counts = np.minimum(counts, n_rows)
    if np.any(np.diff(counts) <= 0):
        raise ValueError("budgets select duplicate row counts; use more evaluation rows")
    return values, counts


def transformed_outcome(
    outcome: Sequence[float],
    treatment: Sequence[float],
    propensity: float | None = None,
) -> tuple[np.ndarray, float]:
    """Return psi=T*Y/e-(1-T)*Y/(1-e) and the assignment propensity."""

    y = _binary_array(outcome, "outcome")
    t = _binary_array(treatment, "treatment")
    if len(y) != len(t):
        raise ValueError("outcome and treatment must have equal length")
    e = float(t.mean()) if propensity is None else float(propensity)
    if not np.isfinite(e) or not 0.0 < e < 1.0:
        raise ValueError("treatment propensity must be strictly between 0 and 1")
    return t * y / e - (1.0 - t) * y / (1.0 - e), e


def _prefix_shell_sums(shell: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.cumsum(np.bincount(shell, weights=weights, minlength=n + 1)[:n])


def _group_sums(group: np.ndarray, weights: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(group, weights=weights, minlength=n)[:n]


def _interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    low, high = np.percentile(finite, (2.5, 97.5))
    return float(low), float(high), int(finite.size)


def _comparison_label(low: float, high: float) -> str:
    if np.isfinite(low) and low > 0.0:
        return "beats"
    if np.isfinite(high) and high < 0.0:
        return "worse"
    return "inconclusive"


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        numerator,
        denominator,
        out=np.full_like(np.asarray(numerator, dtype=np.float64), np.nan),
        where=np.asarray(denominator) > 0,
    )


def _rank_decile_groups(order: np.ndarray) -> np.ndarray:
    groups = np.empty(len(order), dtype=np.int8)
    groups[order] = np.minimum(
        N_DECILES - 1, np.arange(len(order)) * N_DECILES // len(order)
    )
    return groups


def evaluate_policies(
    outcome: Sequence[float],
    treatment: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    *,
    budgets: Sequence[float] = DEFAULT_BUDGETS,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    curve_points: int = 101,
) -> dict[str, pd.DataFrame]:
    """Evaluate fixed response and uplift rankings on a randomized test set."""

    y = _binary_array(outcome, "outcome")
    t = _binary_array(treatment, "treatment")
    if len(y) != len(t):
        raise ValueError("outcome and treatment must have equal length")
    n_rows = len(y)
    if not isinstance(n_bootstrap, (int, np.integer)) or n_bootstrap < 1:
        raise ValueError("n_bootstrap must be a positive integer")
    if not isinstance(curve_points, (int, np.integer)) or curve_points < 2:
        raise ValueError("curve_points must be at least 2")
    budget_values, target_counts = _validated_budgets(budgets, n_rows)
    missing = [name for name in SCORED_POLICIES if name not in scores]
    if missing:
        raise ValueError(f"missing required policy scores: {', '.join(missing)}")
    policy_scores = {
        name: _score_array(scores[name], name, n_rows) for name in SCORED_POLICIES
    }

    psi, propensity = transformed_outcome(y, t)
    treated = t == 1.0
    control = ~treated
    treated_outcome = treated & (y == 1.0)
    control_outcome = control & (y == 1.0)
    n_treated = int(treated.sum())
    n_control = int(control.sum())
    if n_treated == 0 or n_control == 0:
        raise ValueError("the evaluation set must contain treatment and control rows")
    treated_rate = float(y[treated].mean())
    control_rate = float(y[control].mean())
    value_none = control_rate
    value_all = treated_rate
    ate = float(psi.mean())

    policy_to_index = {name: index for index, name in enumerate(POLICIES)}
    expected_index = policy_to_index["expected_random"]
    n_policies = len(POLICIES)
    n_budgets = len(budget_values)

    orders: dict[str, np.ndarray] = {}
    shells: dict[str, np.ndarray] = {}
    rank_multipliers: dict[str, np.ndarray] = {}
    decile_groups: dict[str, np.ndarray] = {}
    for name in SCORED_POLICIES:
        order = np.argsort(-policy_scores[name], kind="mergesort")
        orders[name] = order
        shell = np.full(n_rows, n_budgets, dtype=np.int16)
        start = 0
        for shell_index, end in enumerate(target_counts):
            shell[order[start:end]] = shell_index
            start = int(end)
        shells[name] = shell
        multiplier = np.empty(n_rows, dtype=np.float64)
        multiplier[order] = (n_rows - np.arange(1, n_rows + 1) + 0.5) / n_rows
        rank_multipliers[name] = multiplier
        decile_groups[name] = _rank_decile_groups(order)

    shape = (n_policies, n_budgets)
    point_values = np.full(shape, np.nan)
    point_gains = np.full(shape, np.nan)
    point_per_1000 = np.full(shape, np.nan)
    point_treated_rates = np.full(shape, np.nan)
    point_targeted_treated = np.full(shape, np.nan)
    point_hajek_values = np.full(shape, np.nan)
    point_hajek_gains = np.full(shape, np.nan)
    point_hajek_per_1000 = np.full(shape, np.nan)

    point_gains[expected_index] = budget_values * ate
    point_values[expected_index] = value_none + point_gains[expected_index]
    point_per_1000[expected_index] = 1000.0 * ate
    point_hajek_values[expected_index] = point_values[expected_index]
    point_hajek_gains[expected_index] = point_gains[expected_index]
    point_hajek_per_1000[expected_index] = 1000.0 * ate

    ranking_rows: list[dict[str, float | str]] = [
        {
            "policy": "expected_random",
            "auuc": 0.5 * ate,
            "qini_coefficient": 0.0,
            "ate_curve_endpoint": ate,
        }
    ]
    total_control_outcomes = float(control_outcome.sum())
    for name in SCORED_POLICIES:
        policy_index = policy_to_index[name]
        order = orders[name]
        cumulative_gain = np.cumsum(psi[order], dtype=np.float64)
        selected_gain_total = cumulative_gain[target_counts - 1]
        point_gains[policy_index] = selected_gain_total / n_rows
        point_values[policy_index] = value_none + point_gains[policy_index]
        point_per_1000[policy_index] = 1000.0 * selected_gain_total / target_counts

        cumulative_treated = np.cumsum(t[order], dtype=np.float64)
        cumulative_control = np.cumsum((1.0 - t)[order], dtype=np.float64)
        cumulative_treated_outcomes = np.cumsum((t * y)[order], dtype=np.float64)
        cumulative_control_outcomes = np.cumsum(((1.0 - t) * y)[order])
        selected_treated = cumulative_treated[target_counts - 1]
        selected_control = cumulative_control[target_counts - 1]
        selected_treated_outcomes = cumulative_treated_outcomes[target_counts - 1]
        selected_control_outcomes = cumulative_control_outcomes[target_counts - 1]
        point_targeted_treated[policy_index] = selected_treated
        point_treated_rates[policy_index] = _safe_divide(
            selected_treated_outcomes, selected_treated
        )
        point_hajek_per_1000[policy_index] = 1000.0 * (
            _safe_divide(selected_treated_outcomes, selected_treated)
            - _safe_divide(selected_control_outcomes, selected_control)
        )
        nonselected_control = n_control - selected_control
        nonselected_control_outcomes = total_control_outcomes - selected_control_outcomes
        hajek_numerator = (
            selected_treated_outcomes / propensity
            + nonselected_control_outcomes / (1.0 - propensity)
        )
        hajek_denominator = (
            selected_treated / propensity
            + nonselected_control / (1.0 - propensity)
        )
        point_hajek_values[policy_index] = _safe_divide(
            hajek_numerator, hajek_denominator
        )
        point_hajek_gains[policy_index] = (
            point_hajek_values[policy_index] - value_none
        )

        auuc = float(np.mean(psi * rank_multipliers[name]))
        ranking_rows.append(
            {
                "policy": name,
                "auuc": auuc,
                "qini_coefficient": auuc - 0.5 * ate,
                "ate_curve_endpoint": ate,
            }
        )

    curve_fractions = np.unique(
        np.concatenate((np.linspace(0.0, 1.0, curve_points), budget_values))
    )
    curve_counts = np.floor(curve_fractions * n_rows).astype(np.int64)
    curve_rows: list[dict[str, float | str | int]] = []
    for name in SCORED_POLICIES:
        cumulative = np.concatenate(
            (np.array([0.0]), np.cumsum(psi[orders[name]], dtype=np.float64))
        )
        gains = cumulative[curve_counts] / n_rows
        for requested_fraction, count, gain in zip(
            curve_fractions, curve_counts, gains, strict=True
        ):
            curve_rows.append(
                {
                    "policy": name,
                    "targeting_fraction": float(requested_fraction),
                    "actual_targeting_fraction": float(count / n_rows),
                    "targeted_count": int(count),
                    "cumulative_incremental_rate": float(gain),
                    "cumulative_incremental_per_1000_eligible": float(1000.0 * gain),
                }
            )
    for requested_fraction, count in zip(curve_fractions, curve_counts, strict=True):
        actual_fraction = count / n_rows
        gain = actual_fraction * ate
        curve_rows.append(
            {
                "policy": "expected_random",
                "targeting_fraction": float(requested_fraction),
                "actual_targeting_fraction": float(actual_fraction),
                "targeted_count": int(count),
                "cumulative_incremental_rate": float(gain),
                "cumulative_incremental_per_1000_eligible": float(1000.0 * gain),
            }
        )

    boot_values = np.full((n_bootstrap, *shape), np.nan)
    boot_gains = np.full_like(boot_values, np.nan)
    boot_per_1000 = np.full_like(boot_values, np.nan)
    boot_treated_rates = np.full_like(boot_values, np.nan)
    boot_hajek_values = np.full_like(boot_values, np.nan)
    boot_hajek_gains = np.full_like(boot_values, np.nan)
    boot_hajek_per_1000 = np.full_like(boot_values, np.nan)
    boot_none = np.full(n_bootstrap, np.nan)
    boot_all = np.full(n_bootstrap, np.nan)
    boot_ate = np.full(n_bootstrap, np.nan)
    boot_auuc = np.full((n_bootstrap, n_policies), np.nan)
    boot_qini = np.full_like(boot_auuc, np.nan)
    boot_decile_ht = np.full(
        (n_bootstrap, len(SCORED_POLICIES), N_DECILES), np.nan
    )
    boot_decile_hajek = np.full_like(boot_decile_ht, np.nan)

    control_indices = np.flatnonzero(control)
    treated_indices = np.flatnonzero(treated)
    treated_outcome_indices = np.flatnonzero(treated_outcome)
    control_outcome_indices = np.flatnonzero(control_outcome)
    bootstrap_rng = np.random.default_rng(random_state)
    for replicate in range(n_bootstrap):
        sampled = bootstrap_rng.integers(0, n_rows, size=n_rows)
        row_counts = np.bincount(sampled, minlength=n_rows).astype(np.float64, copy=False)
        bootstrap_treated = float(row_counts[treated].sum())
        e_boot = bootstrap_treated / n_rows
        if not 0.0 < e_boot < 1.0:
            continue
        bootstrap_control = n_rows - bootstrap_treated
        total_treated_outcomes = float(row_counts[treated_outcome].sum())
        total_control_outcomes_boot = float(row_counts[control_outcome].sum())
        none_boot = total_control_outcomes_boot / bootstrap_control
        all_boot = total_treated_outcomes / bootstrap_treated
        ate_boot = all_boot - none_boot
        boot_none[replicate] = none_boot
        boot_all[replicate] = all_boot
        boot_ate[replicate] = ate_boot

        boot_gains[replicate, expected_index] = budget_values * ate_boot
        boot_values[replicate, expected_index] = (
            none_boot + boot_gains[replicate, expected_index]
        )
        boot_per_1000[replicate, expected_index] = 1000.0 * ate_boot
        boot_hajek_values[replicate, expected_index] = boot_values[
            replicate, expected_index
        ]
        boot_hajek_gains[replicate, expected_index] = boot_gains[
            replicate, expected_index
        ]
        boot_hajek_per_1000[replicate, expected_index] = 1000.0 * ate_boot
        boot_auuc[replicate, expected_index] = 0.5 * ate_boot
        boot_qini[replicate, expected_index] = 0.0

        for scored_index, name in enumerate(SCORED_POLICIES):
            policy_index = policy_to_index[name]
            shell = shells[name]
            selected_total = _prefix_shell_sums(shell, row_counts, n_budgets)
            selected_control = _prefix_shell_sums(
                shell[control_indices], row_counts[control_indices], n_budgets
            )
            selected_treated = selected_total - selected_control
            selected_treated_outcomes = _prefix_shell_sums(
                shell[treated_outcome_indices],
                row_counts[treated_outcome_indices],
                n_budgets,
            )
            selected_control_outcomes = _prefix_shell_sums(
                shell[control_outcome_indices],
                row_counts[control_outcome_indices],
                n_budgets,
            )
            selected_gain_total = (
                selected_treated_outcomes / e_boot
                - selected_control_outcomes / (1.0 - e_boot)
            )
            boot_gains[replicate, policy_index] = selected_gain_total / n_rows
            boot_values[replicate, policy_index] = (
                none_boot + boot_gains[replicate, policy_index]
            )
            boot_per_1000[replicate, policy_index] = _safe_divide(
                1000.0 * selected_gain_total, selected_total
            )
            boot_treated_rates[replicate, policy_index] = _safe_divide(
                selected_treated_outcomes, selected_treated
            )
            boot_hajek_per_1000[replicate, policy_index] = 1000.0 * (
                _safe_divide(selected_treated_outcomes, selected_treated)
                - _safe_divide(selected_control_outcomes, selected_control)
            )
            nonselected_control = bootstrap_control - selected_control
            nonselected_control_outcomes = (
                total_control_outcomes_boot - selected_control_outcomes
            )
            hajek_numerator = (
                selected_treated_outcomes / e_boot
                + nonselected_control_outcomes / (1.0 - e_boot)
            )
            hajek_denominator = (
                selected_treated / e_boot + nonselected_control / (1.0 - e_boot)
            )
            boot_hajek_values[replicate, policy_index] = _safe_divide(
                hajek_numerator, hajek_denominator
            )
            boot_hajek_gains[replicate, policy_index] = (
                boot_hajek_values[replicate, policy_index] - none_boot
            )

            multiplier = rank_multipliers[name]
            treated_ranked = float(
                np.dot(
                    row_counts[treated_outcome_indices],
                    multiplier[treated_outcome_indices],
                )
            )
            control_ranked = float(
                np.dot(
                    row_counts[control_outcome_indices],
                    multiplier[control_outcome_indices],
                )
            )
            auuc_boot = (
                treated_ranked / e_boot - control_ranked / (1.0 - e_boot)
            ) / n_rows
            boot_auuc[replicate, policy_index] = auuc_boot
            boot_qini[replicate, policy_index] = auuc_boot - 0.5 * ate_boot

            group = decile_groups[name]
            decile_total = _group_sums(group, row_counts, N_DECILES)
            decile_treated = _group_sums(
                group[treated_indices], row_counts[treated_indices], N_DECILES
            )
            decile_control = decile_total - decile_treated
            decile_treated_outcomes = _group_sums(
                group[treated_outcome_indices],
                row_counts[treated_outcome_indices],
                N_DECILES,
            )
            decile_control_outcomes = _group_sums(
                group[control_outcome_indices],
                row_counts[control_outcome_indices],
                N_DECILES,
            )
            decile_gain = (
                decile_treated_outcomes / e_boot
                - decile_control_outcomes / (1.0 - e_boot)
            )
            boot_decile_ht[replicate, scored_index] = _safe_divide(
                1000.0 * decile_gain, decile_total
            )
            boot_decile_hajek[replicate, scored_index] = 1000.0 * (
                _safe_divide(decile_treated_outcomes, decile_treated)
                - _safe_divide(decile_control_outcomes, decile_control)
            )

    if not np.isfinite(boot_none).any():
        raise ValueError("no valid bootstrap replicate contained both treatment arms")

    budget_rows: list[dict[str, float | str | int]] = []
    for policy_index, name in enumerate(POLICIES):
        for budget_index, (budget, count) in enumerate(
            zip(budget_values, target_counts, strict=True)
        ):
            value_low, value_high, value_reps = _interval(
                boot_values[:, policy_index, budget_index]
            )
            gain_low, gain_high, gain_reps = _interval(
                boot_gains[:, policy_index, budget_index]
            )
            per_low, per_high, per_reps = _interval(
                boot_per_1000[:, policy_index, budget_index]
            )
            rate_low, rate_high, rate_reps = _interval(
                boot_treated_rates[:, policy_index, budget_index]
            )
            h_value_low, h_value_high, h_value_reps = _interval(
                boot_hajek_values[:, policy_index, budget_index]
            )
            h_gain_low, h_gain_high, h_gain_reps = _interval(
                boot_hajek_gains[:, policy_index, budget_index]
            )
            h_per_low, h_per_high, h_per_reps = _interval(
                boot_hajek_per_1000[:, policy_index, budget_index]
            )
            budget_rows.append(
                {
                    "policy": name,
                    "budget": float(budget),
                    "targeted_count": int(count),
                    "actual_budget": float(count / n_rows),
                    "targeted_treated_count": point_targeted_treated[
                        policy_index, budget_index
                    ],
                    "targeted_treated_outcome_rate": point_treated_rates[
                        policy_index, budget_index
                    ],
                    "targeted_treated_outcome_rate_ci_low": rate_low,
                    "targeted_treated_outcome_rate_ci_high": rate_high,
                    "policy_value_estimator": "horvitz_thompson",
                    "policy_value": point_values[policy_index, budget_index],
                    "policy_value_ci_low": value_low,
                    "policy_value_ci_high": value_high,
                    "gain_vs_treat_none": point_gains[policy_index, budget_index],
                    "gain_ci_low": gain_low,
                    "gain_ci_high": gain_high,
                    "incremental_outcomes_per_1000_targeted": point_per_1000[
                        policy_index, budget_index
                    ],
                    "incremental_per_1000_ci_low": per_low,
                    "incremental_per_1000_ci_high": per_high,
                    "hajek_policy_value": point_hajek_values[
                        policy_index, budget_index
                    ],
                    "hajek_policy_value_ci_low": h_value_low,
                    "hajek_policy_value_ci_high": h_value_high,
                    "hajek_gain_vs_treat_none": point_hajek_gains[
                        policy_index, budget_index
                    ],
                    "hajek_gain_ci_low": h_gain_low,
                    "hajek_gain_ci_high": h_gain_high,
                    "hajek_incremental_outcomes_per_1000_targeted": point_hajek_per_1000[
                        policy_index, budget_index
                    ],
                    "hajek_incremental_per_1000_ci_low": h_per_low,
                    "hajek_incremental_per_1000_ci_high": h_per_high,
                    "treat_none_value": value_none,
                    "treat_all_value": value_all,
                    "valid_value_bootstrap_reps": value_reps,
                    "valid_gain_bootstrap_reps": gain_reps,
                    "valid_incremental_bootstrap_reps": per_reps,
                    "valid_rate_bootstrap_reps": rate_reps,
                    "valid_hajek_value_bootstrap_reps": h_value_reps,
                    "valid_hajek_gain_bootstrap_reps": h_gain_reps,
                    "valid_hajek_incremental_bootstrap_reps": h_per_reps,
                }
            )

    ranking_metrics = pd.DataFrame(ranking_rows)
    for policy_index, name in enumerate(POLICIES):
        auuc_low, auuc_high, auuc_reps = _interval(boot_auuc[:, policy_index])
        qini_low, qini_high, qini_reps = _interval(boot_qini[:, policy_index])
        row = ranking_metrics["policy"] == name
        ranking_metrics.loc[row, "auuc_ci_low"] = auuc_low
        ranking_metrics.loc[row, "auuc_ci_high"] = auuc_high
        ranking_metrics.loc[row, "qini_ci_low"] = qini_low
        ranking_metrics.loc[row, "qini_ci_high"] = qini_high
        ranking_metrics.loc[row, "valid_bootstrap_reps"] = min(auuc_reps, qini_reps)

    none_low, none_high, none_reps = _interval(boot_none)
    all_low, all_high, all_reps = _interval(boot_all)
    ate_low, ate_high, ate_reps = _interval(boot_ate)
    reference_metrics = pd.DataFrame(
        [
            {
                "policy": "treat_none",
                "targeting_fraction": 0.0,
                "policy_value": value_none,
                "policy_value_ci_low": none_low,
                "policy_value_ci_high": none_high,
                "gain_vs_treat_none": 0.0,
                "gain_ci_low": 0.0,
                "gain_ci_high": 0.0,
                "valid_bootstrap_reps": none_reps,
            },
            {
                "policy": "treat_all",
                "targeting_fraction": 1.0,
                "policy_value": value_all,
                "policy_value_ci_low": all_low,
                "policy_value_ci_high": all_high,
                "gain_vs_treat_none": ate,
                "gain_ci_low": ate_low,
                "gain_ci_high": ate_high,
                "valid_bootstrap_reps": min(all_reps, ate_reps),
            },
        ]
    )

    comparison_rows: list[dict[str, float | str | int]] = []
    for candidate in UPLIFT_POLICIES:
        candidate_index = policy_to_index[candidate]
        for reference in ("response", "expected_random", "treat_all"):
            for budget_index, budget in enumerate(budget_values):
                candidate_value = point_values[candidate_index, budget_index]
                candidate_per = point_per_1000[candidate_index, budget_index]
                candidate_value_boot = boot_values[:, candidate_index, budget_index]
                candidate_per_boot = boot_per_1000[:, candidate_index, budget_index]
                candidate_h_value = point_hajek_values[candidate_index, budget_index]
                candidate_h_per = point_hajek_per_1000[candidate_index, budget_index]
                candidate_h_value_boot = boot_hajek_values[
                    :, candidate_index, budget_index
                ]
                candidate_h_per_boot = boot_hajek_per_1000[
                    :, candidate_index, budget_index
                ]
                if reference == "treat_all":
                    reference_budget = 1.0
                    reference_value = value_all
                    reference_per = 1000.0 * ate
                    reference_value_boot = boot_all
                    reference_per_boot = 1000.0 * boot_ate
                    reference_h_value = value_all
                    reference_h_per = 1000.0 * ate
                    reference_h_value_boot = boot_all
                    reference_h_per_boot = 1000.0 * boot_ate
                else:
                    reference_budget = float(budget)
                    reference_index = policy_to_index[reference]
                    reference_value = point_values[reference_index, budget_index]
                    reference_per = point_per_1000[reference_index, budget_index]
                    reference_value_boot = boot_values[
                        :, reference_index, budget_index
                    ]
                    reference_per_boot = boot_per_1000[
                        :, reference_index, budget_index
                    ]
                    reference_h_value = point_hajek_values[
                        reference_index, budget_index
                    ]
                    reference_h_per = point_hajek_per_1000[
                        reference_index, budget_index
                    ]
                    reference_h_value_boot = boot_hajek_values[
                        :, reference_index, budget_index
                    ]
                    reference_h_per_boot = boot_hajek_per_1000[
                        :, reference_index, budget_index
                    ]

                value_delta = float(candidate_value - reference_value)
                value_delta_boot = candidate_value_boot - reference_value_boot
                value_low, value_high, value_reps = _interval(value_delta_boot)
                per_delta = float(candidate_per - reference_per)
                per_delta_boot = candidate_per_boot - reference_per_boot
                per_low, per_high, per_reps = _interval(per_delta_boot)
                h_value_delta = float(candidate_h_value - reference_h_value)
                h_value_delta_boot = candidate_h_value_boot - reference_h_value_boot
                h_value_low, h_value_high, h_value_reps = _interval(h_value_delta_boot)
                h_per_delta = float(candidate_h_per - reference_h_per)
                h_per_delta_boot = candidate_h_per_boot - reference_h_per_boot
                h_per_low, h_per_high, h_per_reps = _interval(h_per_delta_boot)
                finite_value = value_delta_boot[np.isfinite(value_delta_boot)]
                finite_per = per_delta_boot[np.isfinite(per_delta_boot)]
                comparison_rows.append(
                    {
                        "candidate_policy": candidate,
                        "reference_policy": reference,
                        "candidate_budget": float(budget),
                        "reference_budget": reference_budget,
                        "delta_policy_value": value_delta,
                        "delta_policy_value_ci_low": value_low,
                        "delta_policy_value_ci_high": value_high,
                        "policy_value_bootstrap_fraction_positive": (
                            float(np.mean(finite_value > 0.0))
                            if finite_value.size
                            else np.nan
                        ),
                        "policy_value_conclusion": _comparison_label(
                            value_low, value_high
                        ),
                        "delta_incremental_per_1000_targeted": per_delta,
                        "delta_incremental_per_1000_ci_low": per_low,
                        "delta_incremental_per_1000_ci_high": per_high,
                        "incremental_bootstrap_fraction_positive": (
                            float(np.mean(finite_per > 0.0))
                            if finite_per.size
                            else np.nan
                        ),
                        "incremental_conclusion": _comparison_label(per_low, per_high),
                        "delta_hajek_policy_value": h_value_delta,
                        "delta_hajek_policy_value_ci_low": h_value_low,
                        "delta_hajek_policy_value_ci_high": h_value_high,
                        "hajek_policy_value_conclusion": _comparison_label(
                            h_value_low, h_value_high
                        ),
                        "delta_hajek_incremental_per_1000_targeted": h_per_delta,
                        "delta_hajek_incremental_per_1000_ci_low": h_per_low,
                        "delta_hajek_incremental_per_1000_ci_high": h_per_high,
                        "hajek_incremental_conclusion": _comparison_label(
                            h_per_low, h_per_high
                        ),
                        "valid_policy_value_bootstrap_reps": value_reps,
                        "valid_incremental_bootstrap_reps": per_reps,
                        "valid_hajek_policy_value_bootstrap_reps": h_value_reps,
                        "valid_hajek_incremental_bootstrap_reps": h_per_reps,
                    }
                )

    decile_rows: list[dict[str, float | str | int]] = []
    for scored_index, name in enumerate(SCORED_POLICIES):
        group = decile_groups[name]
        totals = _group_sums(group, np.ones(n_rows), N_DECILES)
        treated_counts = _group_sums(group, t, N_DECILES)
        control_counts = totals - treated_counts
        treated_outcomes = _group_sums(group, t * y, N_DECILES)
        control_outcomes = _group_sums(group, (1.0 - t) * y, N_DECILES)
        ht = 1000.0 * _safe_divide(
            treated_outcomes / propensity
            - control_outcomes / (1.0 - propensity),
            totals,
        )
        hajek = 1000.0 * (
            _safe_divide(treated_outcomes, treated_counts)
            - _safe_divide(control_outcomes, control_counts)
        )
        treated_rates = _safe_divide(treated_outcomes, treated_counts)
        control_rates = _safe_divide(control_outcomes, control_counts)
        for decile_index in range(N_DECILES):
            ht_low, ht_high, ht_reps = _interval(
                boot_decile_ht[:, scored_index, decile_index]
            )
            h_low, h_high, h_reps = _interval(
                boot_decile_hajek[:, scored_index, decile_index]
            )
            decile_rows.append(
                {
                    "policy": name,
                    "score_decile": decile_index + 1,
                    "rows": int(totals[decile_index]),
                    "treated": int(treated_counts[decile_index]),
                    "control": int(control_counts[decile_index]),
                    "treatment_share": float(
                        treated_counts[decile_index] / totals[decile_index]
                    ),
                    "treated_outcome_rate": float(treated_rates[decile_index]),
                    "control_outcome_rate": float(control_rates[decile_index]),
                    "ht_uplift_per_1000": float(ht[decile_index]),
                    "ht_uplift_ci_low": ht_low,
                    "ht_uplift_ci_high": ht_high,
                    "hajek_uplift_per_1000": float(hajek[decile_index]),
                    "hajek_uplift_ci_low": h_low,
                    "hajek_uplift_ci_high": h_high,
                    "valid_ht_bootstrap_reps": ht_reps,
                    "valid_hajek_bootstrap_reps": h_reps,
                }
            )

    full_budget_index = np.flatnonzero(np.isclose(budget_values, 1.0))
    if full_budget_index.size:
        index = full_budget_index[0]
        if not np.allclose(point_values[:, index], value_all, rtol=0.0, atol=1e-12):
            raise AssertionError("all policies must equal treat-all at a 100% budget")
        if not np.allclose(
            point_hajek_values[:, index], value_all, rtol=0.0, atol=1e-12
        ):
            raise AssertionError(
                "all Hájek policy values must equal treat-all at a 100% budget"
            )

    data_summary = pd.DataFrame(
        [
            {
                "sample": "test",
                "n": n_rows,
                "treated": n_treated,
                "control": n_control,
                "treatment_ratio": propensity,
                "estimated_propensity": propensity,
                "positive_outcomes": int(y.sum()),
                "treated_positive_outcomes": int(y[treated].sum()),
                "control_positive_outcomes": int(y[control].sum()),
                "outcome_rate_treated": treated_rate,
                "outcome_rate_control": control_rate,
                "naive_ate": treated_rate - control_rate,
                "bootstrap_repetitions": int(n_bootstrap),
                "valid_bootstrap_repetitions": int(np.isfinite(boot_none).sum()),
            }
        ]
    )
    return {
        "data_summary": data_summary,
        "reference_metrics": reference_metrics,
        "ranking_metrics": ranking_metrics,
        "qini_curve": pd.DataFrame(curve_rows),
        "budget_metrics": pd.DataFrame(budget_rows),
        "policy_comparisons": pd.DataFrame(comparison_rows),
        "decile_metrics": pd.DataFrame(decile_rows),
    }


def save_evaluation_plots(
    results: Mapping[str, pd.DataFrame],
    scores: Mapping[str, Sequence[float]],
    output_dir: str | Path,
    *,
    random_state: int = 42,
    max_distribution_rows: int = 100_000,
) -> list[Path]:
    """Save decision-focused evaluation figures."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    labels = {
        "expected_random": "Expected random allocation",
        "response": "Response model",
        "s_learner": "S-learner",
        "t_learner": "T-learner",
    }
    colors = {"response": "#4C78A8", "s_learner": "#F58518", "t_learner": "#54A24B"}

    curve = results["qini_curve"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in (*SCORED_POLICIES, "expected_random"):
        subset = curve[curve["policy"] == policy]
        style = (
            {"linestyle": "--", "color": "black", "linewidth": 1.2}
            if policy == "expected_random"
            else {"color": colors[policy]}
        )
        ax.plot(
            100.0 * subset["actual_targeting_fraction"],
            subset["cumulative_incremental_per_1000_eligible"],
            label=labels[policy],
            **style,
        )
    ax.axhline(0.0, color="0.75", linewidth=0.8)
    ax.set(
        title="Cumulative incremental outcomes (HT Qini curve)",
        xlabel="Users targeted (%)",
        ylabel="Incremental outcomes per 1,000 eligible users",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_path / "qini_curve.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    budgets = results["budget_metrics"]
    references = results["reference_metrics"].set_index("policy")
    fig, ax = plt.subplots(figsize=(8, 5))
    for policy in POLICIES:
        subset = budgets[budgets["policy"] == policy].sort_values("budget")
        x = 100.0 * subset["actual_budget"].to_numpy()
        value = subset["policy_value"].to_numpy()
        low = subset["policy_value_ci_low"].to_numpy()
        high = subset["policy_value_ci_high"].to_numpy()
        style = (
            {"color": "black", "linestyle": "--"}
            if policy == "expected_random"
            else {"color": colors[policy]}
        )
        ax.plot(x, value, marker="o", markersize=3, label=labels[policy], **style)
        ax.fill_between(x, low, high, alpha=0.08)
    ax.axhline(
        references.loc["treat_none", "policy_value"],
        color="0.5",
        linestyle=":",
        label="Treat none",
    )
    ax.axhline(
        references.loc["treat_all", "policy_value"],
        color="0.2",
        linestyle="-.",
        label="Treat all",
    )
    ax.set(
        title="Horvitz–Thompson policy value by targeting budget",
        xlabel="Users targeted (%)",
        ylabel="Expected outcome rate",
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_path / "policy_value.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    comparisons = results["policy_comparisons"]
    subset = comparisons[comparisons["reference_policy"] == "response"].copy()
    fig, ax = plt.subplots(figsize=(8, 5))
    offsets = {"s_learner": -0.7, "t_learner": 0.7}
    for policy in UPLIFT_POLICIES:
        rows = subset[subset["candidate_policy"] == policy].sort_values(
            "candidate_budget"
        )
        x = 100.0 * rows["candidate_budget"].to_numpy() + offsets[policy]
        value = rows["delta_incremental_per_1000_targeted"].to_numpy()
        low = rows["delta_incremental_per_1000_ci_low"].to_numpy()
        high = rows["delta_incremental_per_1000_ci_high"].to_numpy()
        ax.errorbar(
            x,
            value,
            # Percentile intervals need not contain the original point estimate,
            # especially in tiny smoke runs. Matplotlib requires nonnegative
            # error lengths, so clip only the rendered distance, not the table.
            yerr=np.vstack(
                (np.maximum(0.0, value - low), np.maximum(0.0, high - value))
            ),
            marker="o",
            capsize=3,
            linewidth=1.3,
            color=colors[policy],
            label=f"{labels[policy]} minus response",
        )
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set(
        title="Incremental outcomes versus response targeting (paired 95% CIs)",
        xlabel="Users targeted (%)",
        ylabel="HT difference per 1,000 targeted users",
        xticks=100.0 * np.asarray(DEFAULT_BUDGETS),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_path / "uplift_vs_response.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    deciles = results["decile_metrics"]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    offsets = {"response": -0.18, "s_learner": 0.0, "t_learner": 0.18}
    for policy in SCORED_POLICIES:
        rows = deciles[deciles["policy"] == policy].sort_values("score_decile")
        x = rows["score_decile"].to_numpy() + offsets[policy]
        value = rows["hajek_uplift_per_1000"].to_numpy()
        low = rows["hajek_uplift_ci_low"].to_numpy()
        high = rows["hajek_uplift_ci_high"].to_numpy()
        ax.errorbar(
            x,
            value,
            yerr=np.vstack(
                (np.maximum(0.0, value - low), np.maximum(0.0, high - value))
            ),
            marker="o",
            capsize=2,
            linewidth=1.1,
            color=colors[policy],
            label=labels[policy],
        )
    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set(
        title="Within-decile uplift (Hájek estimates with 95% CIs)",
        xlabel="Score decile (1 = highest priority)",
        ylabel="Incremental outcomes per 1,000 users",
        xticks=np.arange(1, 11),
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_path / "uplift_by_decile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    s_score = np.asarray(scores["s_learner"], dtype=np.float64)
    t_score = np.asarray(scores["t_learner"], dtype=np.float64)
    if s_score.ndim != 1 or t_score.ndim != 1 or len(s_score) != len(t_score):
        raise ValueError("S-learner and T-learner scores must be equal-length vectors")
    if max_distribution_rows < 1:
        raise ValueError("max_distribution_rows must be positive")
    if len(s_score) > max_distribution_rows:
        rng = np.random.default_rng(random_state)
        sample = rng.choice(len(s_score), size=max_distribution_rows, replace=False)
        s_plot, t_plot = s_score[sample], t_score[sample]
    else:
        s_plot, t_plot = s_score, t_score
    lower = float(min(np.quantile(s_plot, 0.005), np.quantile(t_plot, 0.005)))
    upper = float(max(np.quantile(s_plot, 0.995), np.quantile(t_plot, 0.995)))
    if np.isclose(lower, upper):
        lower -= 1e-6
        upper += 1e-6
    bins = np.linspace(lower, upper, 51)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s_plot, bins=bins, density=True, alpha=0.55, label="S-learner")
    ax.hist(t_plot, bins=bins, density=True, alpha=0.55, label="T-learner")
    ax.axvline(0.0, color="black", linewidth=0.9, linestyle="--")
    ax.set(
        title="Distribution of predicted individual uplift",
        xlabel="Predicted treatment effect",
        ylabel="Density",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path = output_path / "uplift_distribution.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)
    return written
