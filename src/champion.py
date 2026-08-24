"""Champion/challenger diagnostics for fixed equal-budget targeting policies.

The rankings passed to this module are assumed to have been frozen before the
outcomes used here were inspected.  Bootstrap intervals condition on those
rankings and therefore do not include model fitting or cutoff-selection
uncertainty.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class _FixedComparison:
    """Membership information for one equal-budget policy comparison."""

    challenger: str
    budget: float
    targeted_count: int
    champion_only: np.ndarray
    challenger_only: np.ndarray
    intersection_count: int
    union_count: int


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
        raise ValueError(f"score {name!r} must be one-dimensional with {n_rows} rows")
    if not np.isfinite(array).all():
        raise ValueError(f"score {name!r} contains NaN or infinite values")
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
        raise ValueError("budgets select duplicate row counts; use more rows")
    return values, counts


def _interval(values: np.ndarray) -> tuple[float, float, int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.nan, np.nan, 0
    low, high = np.percentile(finite, (2.5, 97.5))
    return float(low), float(high), int(finite.size)


def _cohort_uplift(
    indices: np.ndarray,
    outcome: np.ndarray,
    treatment: np.ndarray,
    transformed: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Return HT and within-cohort Hajek uplift per 1,000 rows."""

    if indices.size == 0:
        return np.nan, np.nan
    cohort_weights = (
        np.ones(indices.size, dtype=np.float64)
        if weights is None
        else weights[indices]
    )
    total = float(cohort_weights.sum())
    if total <= 0.0:
        return np.nan, np.nan

    ht = 1000.0 * float(np.dot(cohort_weights, transformed[indices])) / total
    cohort_treatment = treatment[indices]
    cohort_outcome = outcome[indices]
    treated = float(np.dot(cohort_weights, cohort_treatment))
    control = total - treated
    if treated <= 0.0 or control <= 0.0:
        return ht, np.nan
    treated_outcomes = float(
        np.dot(cohort_weights, cohort_treatment * cohort_outcome)
    )
    control_outcomes = float(
        np.dot(cohort_weights, (1.0 - cohort_treatment) * cohort_outcome)
    )
    hajek = 1000.0 * (treated_outcomes / treated - control_outcomes / control)
    return ht, hajek


def _comparison_memberships(
    policy_scores: Mapping[str, np.ndarray],
    champion: str,
    challengers: tuple[str, ...],
    budgets: np.ndarray,
    target_counts: np.ndarray,
) -> list[_FixedComparison]:
    n_rows = len(policy_scores[champion])
    orders = {
        name: np.argsort(-policy_scores[name], kind="mergesort")
        for name in (champion, *challengers)
    }
    comparisons: list[_FixedComparison] = []
    for challenger in challengers:
        for budget, count in zip(budgets, target_counts, strict=True):
            champion_selected = np.zeros(n_rows, dtype=bool)
            challenger_selected = np.zeros(n_rows, dtype=bool)
            champion_selected[orders[champion][:count]] = True
            challenger_selected[orders[challenger][:count]] = True
            comparisons.append(
                _FixedComparison(
                    challenger=challenger,
                    budget=float(budget),
                    targeted_count=int(count),
                    champion_only=np.flatnonzero(
                        champion_selected & ~challenger_selected
                    ),
                    challenger_only=np.flatnonzero(
                        challenger_selected & ~champion_selected
                    ),
                    intersection_count=int(
                        np.count_nonzero(champion_selected & challenger_selected)
                    ),
                    union_count=int(
                        np.count_nonzero(champion_selected | challenger_selected)
                    ),
                )
            )
    return comparisons


def champion_challenger_diagnostics(
    outcome: Sequence[float],
    treatment: Sequence[float],
    scores: Mapping[str, Sequence[float]],
    budgets: Sequence[float],
    *,
    champion: str = "response",
    challengers: Sequence[str] | None = None,
    n_bootstrap: int = 1000,
    random_state: int = 42,
    propensity: float | None = None,
) -> pd.DataFrame:
    """Compare fixed top-k policies at identical targeting budgets.

    The primary exact decomposition is Horvitz--Thompson (HT).  If ``m`` rows
    switch in and ``k`` rows are targeted, then

    ``challenger_minus_champion_per_1000 = (m / k) * switched_uplift_delta``.

    Exclusive-cohort Hajek estimates are included as sensitivity diagnostics,
    but no exact full-policy Hajek decomposition is claimed.  Empty exclusive
    cohorts produce ``NaN`` cohort uplifts and an exact zero policy difference.
    Ties are resolved deterministically by original row order using stable
    mergesort ranking.
    """

    y = _binary_array(outcome, "outcome")
    t = _binary_array(treatment, "treatment")
    if len(y) != len(t):
        raise ValueError("outcome and treatment must have equal length")
    if np.unique(t).size != 2:
        raise ValueError("treatment must contain both treatment arms")
    if not isinstance(scores, Mapping) or not scores:
        raise ValueError("scores must be a non-empty policy-to-score mapping")
    if champion not in scores:
        raise ValueError(f"champion policy {champion!r} is missing from scores")
    if challengers is None:
        challenger_names = tuple(name for name in scores if name != champion)
    else:
        if isinstance(challengers, (str, bytes)):
            raise TypeError("challengers must be a sequence of policy names")
        challenger_names = tuple(challengers)
    if not challenger_names:
        raise ValueError("at least one challenger policy is required")
    if champion in challenger_names:
        raise ValueError("the champion cannot also be a challenger")
    if len(set(challenger_names)) != len(challenger_names):
        raise ValueError("challenger policy names must be unique")
    missing = [name for name in challenger_names if name not in scores]
    if missing:
        raise ValueError(f"challenger policies are missing from scores: {missing}")
    if (
        not isinstance(n_bootstrap, (int, np.integer))
        or isinstance(n_bootstrap, (bool, np.bool_))
        or n_bootstrap < 1
    ):
        raise ValueError("n_bootstrap must be a positive integer")

    n_rows = len(y)
    budget_values, target_counts = _validated_budgets(budgets, n_rows)
    required_names = (champion, *challenger_names)
    policy_scores = {
        name: _score_array(scores[name], name, n_rows) for name in required_names
    }
    empirical_propensity = float(t.mean())
    point_propensity = (
        empirical_propensity if propensity is None else float(propensity)
    )
    if not np.isfinite(point_propensity) or not 0.0 < point_propensity < 1.0:
        raise ValueError("propensity must be strictly between 0 and 1")
    transformed = (
        t * y / point_propensity
        - (1.0 - t) * y / (1.0 - point_propensity)
    )

    comparisons = _comparison_memberships(
        policy_scores,
        champion,
        challenger_names,
        budget_values,
        target_counts,
    )
    bootstrap_names = (
        "champion_ht",
        "challenger_ht",
        "switched_ht_delta",
        "champion_hajek",
        "challenger_hajek",
        "switched_hajek_delta",
        "policy_value_delta",
        "nominal_per_1000_delta",
    )
    bootstrap = {
        name: np.full((len(comparisons), n_bootstrap), np.nan)
        for name in bootstrap_names
    }
    rng = np.random.default_rng(random_state)
    for replicate in range(n_bootstrap):
        sampled = rng.integers(0, n_rows, size=n_rows)
        weights = np.bincount(sampled, minlength=n_rows).astype(
            np.float64, copy=False
        )
        bootstrap_propensity = (
            float(np.dot(weights, t) / n_rows)
            if propensity is None
            else point_propensity
        )
        if not 0.0 < bootstrap_propensity < 1.0:
            continue
        transformed_bootstrap = (
            t * y / bootstrap_propensity
            - (1.0 - t) * y / (1.0 - bootstrap_propensity)
        )
        for comparison_index, comparison in enumerate(comparisons):
            champion_ht, champion_hajek = _cohort_uplift(
                comparison.champion_only,
                y,
                t,
                transformed_bootstrap,
                weights,
            )
            challenger_ht, challenger_hajek = _cohort_uplift(
                comparison.challenger_only,
                y,
                t,
                transformed_bootstrap,
                weights,
            )
            bootstrap["champion_ht"][comparison_index, replicate] = champion_ht
            bootstrap["challenger_ht"][comparison_index, replicate] = challenger_ht
            bootstrap["champion_hajek"][comparison_index, replicate] = champion_hajek
            bootstrap["challenger_hajek"][comparison_index, replicate] = challenger_hajek
            if np.isfinite(champion_ht) and np.isfinite(challenger_ht):
                bootstrap["switched_ht_delta"][comparison_index, replicate] = (
                    challenger_ht - champion_ht
                )
            if np.isfinite(champion_hajek) and np.isfinite(challenger_hajek):
                bootstrap["switched_hajek_delta"][comparison_index, replicate] = (
                    challenger_hajek - champion_hajek
                )

            challenger_total = float(
                np.dot(
                    weights[comparison.challenger_only],
                    transformed_bootstrap[comparison.challenger_only],
                )
            )
            champion_total = float(
                np.dot(
                    weights[comparison.champion_only],
                    transformed_bootstrap[comparison.champion_only],
                )
            )
            policy_delta = (challenger_total - champion_total) / n_rows
            bootstrap["policy_value_delta"][comparison_index, replicate] = policy_delta
            actual_budget = comparison.targeted_count / n_rows
            bootstrap["nominal_per_1000_delta"][comparison_index, replicate] = (
                1000.0 * policy_delta / actual_budget
            )

    rows: list[dict[str, Any]] = []
    for comparison_index, comparison in enumerate(comparisons):
        champion_ht, champion_hajek = _cohort_uplift(
            comparison.champion_only, y, t, transformed
        )
        challenger_ht, challenger_hajek = _cohort_uplift(
            comparison.challenger_only, y, t, transformed
        )
        exclusive_count = int(comparison.challenger_only.size)
        if exclusive_count != int(comparison.champion_only.size):
            raise AssertionError("equal-budget policies must have equal switch counts")
        switched_ht_delta = (
            float(challenger_ht - champion_ht)
            if np.isfinite(champion_ht) and np.isfinite(challenger_ht)
            else np.nan
        )
        switched_hajek_delta = (
            float(challenger_hajek - champion_hajek)
            if np.isfinite(champion_hajek) and np.isfinite(challenger_hajek)
            else np.nan
        )
        challenger_total = float(transformed[comparison.challenger_only].sum())
        champion_total = float(transformed[comparison.champion_only].sum())
        policy_delta = (challenger_total - champion_total) / n_rows
        actual_budget = comparison.targeted_count / n_rows
        nominal_per_1000_delta = 1000.0 * policy_delta / actual_budget
        switch_share = exclusive_count / comparison.targeted_count
        if exclusive_count == 0:
            decomposed_policy_delta = 0.0
            decomposed_per_1000_delta = 0.0
        else:
            decomposed_policy_delta = (
                exclusive_count / n_rows * switched_ht_delta / 1000.0
            )
            decomposed_per_1000_delta = switch_share * switched_ht_delta
        policy_residual = policy_delta - decomposed_policy_delta
        per_1000_residual = nominal_per_1000_delta - decomposed_per_1000_delta
        decomposition_holds = bool(
            np.isclose(policy_residual, 0.0, rtol=0.0, atol=1e-12)
            and np.isclose(per_1000_residual, 0.0, rtol=0.0, atol=1e-9)
        )
        if not decomposition_holds:
            raise AssertionError("HT switched-cohort decomposition failed")

        row: dict[str, Any] = {
            "champion_policy": champion,
            "challenger_policy": comparison.challenger,
            "budget": comparison.budget,
            "targeted_count": comparison.targeted_count,
            "actual_budget": actual_budget,
            "intersection_count": comparison.intersection_count,
            "union_count": comparison.union_count,
            "jaccard": comparison.intersection_count / comparison.union_count,
            "champion_only_count": exclusive_count,
            "challenger_only_count": exclusive_count,
            "total_switched_count": 2 * exclusive_count,
            "switch_share_of_targeted": switch_share,
            "champion_only_ht_uplift_per_1000": champion_ht,
            "challenger_only_ht_uplift_per_1000": challenger_ht,
            "switched_ht_uplift_delta_per_1000": switched_ht_delta,
            "champion_only_hajek_uplift_per_1000": champion_hajek,
            "challenger_only_hajek_uplift_per_1000": challenger_hajek,
            "switched_hajek_uplift_delta_per_1000": switched_hajek_delta,
            "challenger_minus_champion_ht_policy_value": policy_delta,
            "challenger_minus_champion_ht_per_1000_nominal_targeted": (
                nominal_per_1000_delta
            ),
            "decomposed_ht_policy_value_delta": decomposed_policy_delta,
            "decomposed_ht_per_1000_nominal_targeted": decomposed_per_1000_delta,
            "ht_policy_value_decomposition_residual": policy_residual,
            "ht_per_1000_decomposition_residual": per_1000_residual,
            "ht_decomposition_holds": decomposition_holds,
        }
        interval_columns = {
            "champion_ht": "champion_only_ht_uplift_per_1000",
            "challenger_ht": "challenger_only_ht_uplift_per_1000",
            "switched_ht_delta": "switched_ht_uplift_delta_per_1000",
            "champion_hajek": "champion_only_hajek_uplift_per_1000",
            "challenger_hajek": "challenger_only_hajek_uplift_per_1000",
            "switched_hajek_delta": "switched_hajek_uplift_delta_per_1000",
            "policy_value_delta": "challenger_minus_champion_ht_policy_value",
            "nominal_per_1000_delta": (
                "challenger_minus_champion_ht_per_1000_nominal_targeted"
            ),
        }
        valid_counts: dict[str, int] = {}
        for bootstrap_name, column_prefix in interval_columns.items():
            low, high, valid = _interval(
                bootstrap[bootstrap_name][comparison_index]
            )
            row[f"{column_prefix}_ci_low"] = low
            row[f"{column_prefix}_ci_high"] = high
            valid_counts[bootstrap_name] = valid
        row["valid_ht_cohort_bootstrap_reps"] = min(
            valid_counts["champion_ht"], valid_counts["challenger_ht"]
        )
        row["valid_hajek_cohort_bootstrap_reps"] = min(
            valid_counts["champion_hajek"], valid_counts["challenger_hajek"]
        )
        row["valid_policy_delta_bootstrap_reps"] = valid_counts[
            "policy_value_delta"
        ]
        rows.append(row)

    return pd.DataFrame(rows)


def assess_promotion(
    delta_estimate: float,
    ci_low: float,
    ci_high: float,
    *,
    metric: str = "paired_policy_value_delta",
    rule: str = "superiority",
    noninferiority_margin: float | None = None,
) -> dict[str, Any]:
    """Apply an explicit offline promotion rule to challenger-minus-champion CI.

    Positive deltas favor the challenger.  Passing this offline rule means only
    that the challenger is eligible for promotion review; it is not evidence of
    production effectiveness.  Noninferiority is reported separately and is
    never described as superiority.
    """

    estimate = float(delta_estimate)
    low = float(ci_low)
    high = float(ci_high)
    if not metric or not isinstance(metric, str):
        raise ValueError("metric must be a non-empty string")
    if not np.isfinite((estimate, low, high)).all():
        raise ValueError("delta estimate and confidence bounds must be finite")
    if low > high:
        raise ValueError("ci_low cannot exceed ci_high")
    if rule not in {"superiority", "noninferiority"}:
        raise ValueError("rule must be 'superiority' or 'noninferiority'")

    superiority_demonstrated = low > 0.0
    if high < 0.0:
        evidence_status = "challenger_inferior"
    elif superiority_demonstrated:
        evidence_status = "challenger_superior"
    else:
        evidence_status = "difference_inconclusive"

    if rule == "superiority":
        if noninferiority_margin is not None:
            raise ValueError(
                "noninferiority_margin is only valid for the noninferiority rule"
            )
        threshold = 0.0
        rule_passed = superiority_demonstrated
        noninferiority_demonstrated: bool | None = None
        if rule_passed:
            decision = "eligible_for_promotion_review_under_superiority_rule"
        else:
            decision = "retain_champion_superiority_not_demonstrated"
    else:
        if noninferiority_margin is None:
            raise ValueError(
                "noninferiority_margin is required for the noninferiority rule"
            )
        margin = float(noninferiority_margin)
        if not np.isfinite(margin) or margin < 0.0:
            raise ValueError("noninferiority_margin must be finite and nonnegative")
        threshold = -margin
        noninferiority_demonstrated = low > threshold
        rule_passed = noninferiority_demonstrated
        if rule_passed:
            decision = "eligible_for_promotion_review_under_noninferiority_rule"
        else:
            decision = "retain_champion_noninferiority_not_demonstrated"

    return {
        "metric": metric,
        "delta_direction": "challenger_minus_champion_positive_favors_challenger",
        "delta_estimate": estimate,
        "ci_low": low,
        "ci_high": high,
        "rule": rule,
        "decision_threshold": threshold,
        "rule_passed": bool(rule_passed),
        "superiority_demonstrated": bool(superiority_demonstrated),
        "noninferiority_demonstrated": noninferiority_demonstrated,
        "evidence_status": evidence_status,
        "decision": decision,
        "scope": "offline_fixed_policy_evidence_only",
    }
