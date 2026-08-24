"""Economic policy evaluation for fixed causal targeting gain curves.

The causal evaluator reports incremental outcome rates relative to treating
nobody.  This module converts those rates into *incremental* net value under an
explicitly labelled, deliberately narrow business assumption:

* one constant contribution margin per incremental conversion; and
* one constant cost per user assigned to treatment/targeting.

It does not interpret that cost as an impression or exposure cost.  Those
quantities require a different cost estimand because exposure is post-treatment.
Policy and budget selection is intentionally separated from final evaluation so
that a validation decision can be frozen before an untouched test evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


TARGET_NOBODY_POLICY = "target_nobody"
ECONOMIC_ASSUMPTION = "constant_margin_and_cost_per_assigned_target"
SELECTION_RULE = (
    "max_validation_net_value_then_lower_actual_budget_then_champion_preference"
    "_then_policy_name"
)

_POLICY_COLUMN = "policy"
_REQUESTED_BUDGET_COLUMN = "requested_budget"
_ACTUAL_BUDGET_COLUMN = "actual_budget"
_GAIN_COLUMN = "gain_vs_treat_none"
_NET_VALUE_COLUMN = "incremental_net_value_vs_target_nobody"


def _nonempty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _finite_nonnegative(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite non-negative number")
    number = float(value)
    if not np.isfinite(number) or number < 0.0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return number


def _positive_population(value: object) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("eligible_population must be a positive integer")
    population = int(value)
    if population <= 0:
        raise ValueError("eligible_population must be a positive integer")
    return population


def _fraction(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    fraction = float(value)
    if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return fraction


@dataclass(frozen=True)
class EconomicScenario:
    """A labelled constant-margin, constant-assignment-cost scenario.

    ``eligible_population`` defines the N used for total-value columns.  The
    same calculations are also returned per 1,000 eligible users so scenarios
    can be compared without treating a benchmark sample as business volume.
    """

    label: str
    margin_per_conversion: float
    cost_per_assigned_target: float
    eligible_population: int
    currency: str
    horizon: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", _nonempty_text(self.label, "label"))
        object.__setattr__(
            self,
            "margin_per_conversion",
            _finite_nonnegative(self.margin_per_conversion, "margin_per_conversion"),
        )
        object.__setattr__(
            self,
            "cost_per_assigned_target",
            _finite_nonnegative(
                self.cost_per_assigned_target, "cost_per_assigned_target"
            ),
        )
        object.__setattr__(
            self, "eligible_population", _positive_population(self.eligible_population)
        )
        object.__setattr__(
            self, "currency", _nonempty_text(self.currency, "currency")
        )
        object.__setattr__(self, "horizon", _nonempty_text(self.horizon, "horizon"))

    def to_record(self) -> dict[str, str | float | int]:
        """Return a flat, JSON-serializable scenario record."""

        return {
            "scenario_label": self.label,
            "currency": self.currency,
            "horizon": self.horizon,
            "margin_per_conversion": self.margin_per_conversion,
            "cost_per_assigned_target": self.cost_per_assigned_target,
            "eligible_population": self.eligible_population,
            "economic_assumption": ECONOMIC_ASSUMPTION,
        }


@dataclass(frozen=True)
class FrozenEconomicDecision:
    """Immutable record of a policy and cutoff selected on validation data."""

    scenario: EconomicScenario
    selected_policy: str
    requested_budget: float
    actual_budget: float
    validation_gain_vs_treat_none: float
    validation_incremental_net_value: float
    champion_policy: str
    selection_rule: str = SELECTION_RULE

    def __post_init__(self) -> None:
        if not isinstance(self.scenario, EconomicScenario):
            raise TypeError("scenario must be an EconomicScenario")
        policy = _nonempty_text(self.selected_policy, "selected_policy")
        requested = _fraction(self.requested_budget, "requested_budget")
        actual = _fraction(self.actual_budget, "actual_budget")
        champion = _nonempty_text(self.champion_policy, "champion_policy")
        rule = _nonempty_text(self.selection_rule, "selection_rule")
        gain = float(self.validation_gain_vs_treat_none)
        net_value = float(self.validation_incremental_net_value)
        if not np.isfinite(gain):
            raise ValueError("validation_gain_vs_treat_none must be finite")
        if not np.isfinite(net_value):
            raise ValueError("validation_incremental_net_value must be finite")
        if policy == TARGET_NOBODY_POLICY:
            if requested != 0.0 or actual != 0.0 or gain != 0.0 or net_value != 0.0:
                raise ValueError(
                    "a target-nobody decision must have zero budgets, gain and net value"
                )
        elif requested <= 0.0 or actual <= 0.0:
            raise ValueError("a targeting decision must have positive budgets")
        object.__setattr__(self, "selected_policy", policy)
        object.__setattr__(self, "requested_budget", requested)
        object.__setattr__(self, "actual_budget", actual)
        object.__setattr__(self, "validation_gain_vs_treat_none", gain)
        object.__setattr__(self, "validation_incremental_net_value", net_value)
        object.__setattr__(self, "champion_policy", champion)
        object.__setattr__(self, "selection_rule", rule)

    @property
    def targets_nobody(self) -> bool:
        return self.selected_policy == TARGET_NOBODY_POLICY

    def to_record(self) -> dict[str, str | float | int | bool]:
        """Return a flat, JSON-serializable frozen-decision record."""

        return {
            **self.scenario.to_record(),
            "selected_on": "validation",
            "selected_policy": self.selected_policy,
            "requested_budget": self.requested_budget,
            "actual_budget": self.actual_budget,
            "targets_nobody": self.targets_nobody,
            "validation_gain_vs_treat_none": self.validation_gain_vs_treat_none,
            "validation_incremental_net_value": self.validation_incremental_net_value,
            "champion_policy": self.champion_policy,
            "selection_rule": self.selection_rule,
        }


def _resolve_column(
    frame: pd.DataFrame,
    explicit: str | None,
    candidates: Sequence[str],
    role: str,
    *,
    required: bool = True,
) -> str | None:
    if explicit is not None:
        name = _nonempty_text(explicit, f"{role}_column")
        if name not in frame.columns:
            raise ValueError(f"missing {role} column: {name!r}")
        return name
    for name in candidates:
        if name in frame.columns:
            return name
    if required:
        raise ValueError(
            f"could not infer {role} column; expected one of {list(candidates)}"
        )
    return None


def _economic_row(
    *,
    scenario: EconomicScenario,
    policy: str,
    requested_budget: float,
    actual_budget: float,
    gain: float,
) -> dict[str, str | float | int | bool]:
    population = scenario.eligible_population
    assigned_targets = population * actual_budget
    incremental_conversions = population * gain
    incremental_margin = scenario.margin_per_conversion * incremental_conversions
    targeting_cost = scenario.cost_per_assigned_target * assigned_targets
    net_value = incremental_margin - targeting_cost
    is_nobody = policy == TARGET_NOBODY_POLICY
    break_even_ratio = np.nan if actual_budget == 0.0 else gain / actual_budget
    break_even_cost = (
        np.nan
        if actual_budget == 0.0
        else scenario.margin_per_conversion * break_even_ratio
    )
    return {
        **scenario.to_record(),
        _POLICY_COLUMN: policy,
        _REQUESTED_BUDGET_COLUMN: requested_budget,
        _ACTUAL_BUDGET_COLUMN: actual_budget,
        "is_target_nobody": is_nobody,
        _GAIN_COLUMN: gain,
        "assigned_targets": assigned_targets,
        "estimated_incremental_conversions": incremental_conversions,
        "incremental_conversion_margin": incremental_margin,
        "targeting_cost": targeting_cost,
        _NET_VALUE_COLUMN: net_value,
        "incremental_net_value_per_1000_eligible": 1000.0
        * (
            scenario.margin_per_conversion * gain
            - scenario.cost_per_assigned_target * actual_budget
        ),
        "break_even_cost_to_margin_ratio": break_even_ratio,
        "break_even_cost_per_assigned_target": break_even_cost,
    }


def evaluate_economic_scenario(
    gain_curve: pd.DataFrame,
    scenario: EconomicScenario,
    *,
    candidate_policies: Sequence[str] | None = None,
    policy_column: str | None = None,
    budget_column: str | None = None,
    actual_budget_column: str | None = None,
    gain_column: str | None = None,
) -> pd.DataFrame:
    """Monetize a fixed policy gain curve and add an explicit no-action row.

    The function accepts either the project's Qini-curve schema
    (``targeting_fraction``, ``actual_targeting_fraction``,
    ``cumulative_incremental_rate``) or its budget-metrics schema (``budget``,
    ``actual_budget``, ``gain_vs_treat_none``).  Gain must be incremental
    conversions per eligible user relative to targeting nobody.
    """

    if not isinstance(gain_curve, pd.DataFrame):
        raise TypeError("gain_curve must be a pandas DataFrame")
    if gain_curve.empty:
        raise ValueError("gain_curve must not be empty")
    if not isinstance(scenario, EconomicScenario):
        raise TypeError("scenario must be an EconomicScenario")

    policy_name = _resolve_column(
        gain_curve, policy_column, ("policy",), "policy"
    )
    budget_name = _resolve_column(
        gain_curve,
        budget_column,
        ("targeting_fraction", "budget"),
        "budget",
    )
    actual_name = _resolve_column(
        gain_curve,
        actual_budget_column,
        ("actual_targeting_fraction", "actual_budget"),
        "actual budget",
        required=False,
    )
    gain_name = _resolve_column(
        gain_curve,
        gain_column,
        ("cumulative_incremental_rate", "gain_vs_treat_none"),
        "gain",
    )
    assert policy_name is not None and budget_name is not None and gain_name is not None

    work = pd.DataFrame(
        {
            _POLICY_COLUMN: gain_curve[policy_name],
            _REQUESTED_BUDGET_COLUMN: gain_curve[budget_name],
            _ACTUAL_BUDGET_COLUMN: (
                gain_curve[actual_name]
                if actual_name is not None
                else gain_curve[budget_name]
            ),
            _GAIN_COLUMN: gain_curve[gain_name],
        }
    )
    if work.isna().any().any():
        raise ValueError("gain_curve policy, budget and gain values must not be missing")
    work[_POLICY_COLUMN] = work[_POLICY_COLUMN].map(
        lambda value: _nonempty_text(value, "policy")
    )
    for column in (_REQUESTED_BUDGET_COLUMN, _ACTUAL_BUDGET_COLUMN, _GAIN_COLUMN):
        try:
            work[column] = pd.to_numeric(work[column], errors="raise").astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"gain_curve {column} values must be numeric") from exc
        if not np.isfinite(work[column].to_numpy()).all():
            raise ValueError(f"gain_curve {column} values must be finite")
    if (
        (work[_REQUESTED_BUDGET_COLUMN] < 0.0)
        | (work[_REQUESTED_BUDGET_COLUMN] > 1.0)
        | (work[_ACTUAL_BUDGET_COLUMN] < 0.0)
        | (work[_ACTUAL_BUDGET_COLUMN] > 1.0)
    ).any():
        raise ValueError("gain_curve budgets must be in [0, 1]")
    inconsistent_zero = (
        (work[_REQUESTED_BUDGET_COLUMN] == 0.0)
        != (work[_ACTUAL_BUDGET_COLUMN] == 0.0)
    )
    if inconsistent_zero.any():
        raise ValueError("requested and actual budgets must agree on zero targeting")
    zero_rows = work[_ACTUAL_BUDGET_COLUMN] == 0.0
    if not np.allclose(
        work.loc[zero_rows, _GAIN_COLUMN].to_numpy(), 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("zero-budget gain must be zero")
    if (work.loc[~zero_rows, _POLICY_COLUMN] == TARGET_NOBODY_POLICY).any():
        raise ValueError("target_nobody cannot have a positive budget")

    if candidate_policies is not None:
        if isinstance(candidate_policies, str):
            raise ValueError("candidate_policies must be a sequence, not a string")
        policies = tuple(
            dict.fromkeys(
                _nonempty_text(policy, "candidate policy")
                for policy in candidate_policies
            )
        )
        if not policies:
            raise ValueError("candidate_policies must not be empty")
        missing = sorted(set(policies) - set(work[_POLICY_COLUMN]))
        if missing:
            raise ValueError(f"candidate policies are absent from gain_curve: {missing}")
        work = work[work[_POLICY_COLUMN].isin(policies)]

    positive = work[work[_ACTUAL_BUDGET_COLUMN] > 0.0].copy()
    if positive.empty:
        raise ValueError("gain_curve must contain at least one positive-budget candidate")
    duplicates = positive.duplicated(
        subset=[_POLICY_COLUMN, _ACTUAL_BUDGET_COLUMN], keep=False
    )
    if duplicates.any():
        duplicate_keys = positive.loc[
            duplicates, [_POLICY_COLUMN, _ACTUAL_BUDGET_COLUMN]
        ].drop_duplicates()
        raise ValueError(
            "gain_curve has duplicate policy/actual-budget candidates: "
            f"{duplicate_keys.to_dict(orient='records')}"
        )

    rows = [
        _economic_row(
            scenario=scenario,
            policy=TARGET_NOBODY_POLICY,
            requested_budget=0.0,
            actual_budget=0.0,
            gain=0.0,
        )
    ]
    positive = positive.sort_values(
        [_POLICY_COLUMN, _ACTUAL_BUDGET_COLUMN, _REQUESTED_BUDGET_COLUMN],
        kind="mergesort",
    )
    for row in positive.itertuples(index=False):
        rows.append(
            _economic_row(
                scenario=scenario,
                policy=getattr(row, _POLICY_COLUMN),
                requested_budget=float(getattr(row, _REQUESTED_BUDGET_COLUMN)),
                actual_budget=float(getattr(row, _ACTUAL_BUDGET_COLUMN)),
                gain=float(getattr(row, _GAIN_COLUMN)),
            )
        )
    return pd.DataFrame(rows)


def select_economic_policy(
    validation_economics: pd.DataFrame,
    *,
    champion_policy: str = "response",
    tie_atol: float = 1e-12,
    tie_rtol: float = 1e-12,
) -> pd.Series:
    """Select one validation candidate with a deterministic tie-break.

    Order: maximum incremental net value, lower *actual* budget, champion
    preference, then lexicographically smaller policy name.  The final name
    fallback makes selection reproducible when non-champion policies still tie.
    """

    if not isinstance(validation_economics, pd.DataFrame):
        raise TypeError("validation_economics must be a pandas DataFrame")
    if validation_economics.empty:
        raise ValueError("validation_economics must not be empty")
    champion = _nonempty_text(champion_policy, "champion_policy")
    atol = _finite_nonnegative(tie_atol, "tie_atol")
    rtol = _finite_nonnegative(tie_rtol, "tie_rtol")
    required = {
        "scenario_label",
        "currency",
        "horizon",
        "margin_per_conversion",
        "cost_per_assigned_target",
        "eligible_population",
        "economic_assumption",
        _POLICY_COLUMN,
        _REQUESTED_BUDGET_COLUMN,
        _ACTUAL_BUDGET_COLUMN,
        _GAIN_COLUMN,
        _NET_VALUE_COLUMN,
    }
    missing = sorted(required - set(validation_economics.columns))
    if missing:
        raise ValueError(f"validation_economics is missing columns: {missing}")
    scenario_columns = (
        "scenario_label",
        "currency",
        "horizon",
        "margin_per_conversion",
        "cost_per_assigned_target",
        "eligible_population",
        "economic_assumption",
    )
    if any(
        validation_economics[column].nunique(dropna=False) != 1
        for column in scenario_columns
    ):
        raise ValueError("select one economic scenario at a time")
    if validation_economics["economic_assumption"].iloc[0] != ECONOMIC_ASSUMPTION:
        raise ValueError(
            f"economic_assumption must be {ECONOMIC_ASSUMPTION!r}"
        )

    candidates = validation_economics.copy()
    for column in (
        _REQUESTED_BUDGET_COLUMN,
        _ACTUAL_BUDGET_COLUMN,
        _GAIN_COLUMN,
        _NET_VALUE_COLUMN,
    ):
        values = pd.to_numeric(candidates[column], errors="coerce").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"validation_economics {column} must be finite")
        candidates[column] = values
    if (
        (candidates[_ACTUAL_BUDGET_COLUMN] < 0.0)
        | (candidates[_ACTUAL_BUDGET_COLUMN] > 1.0)
    ).any():
        raise ValueError("validation_economics actual budgets must be in [0, 1]")

    max_value = float(candidates[_NET_VALUE_COLUMN].max())
    close_value = np.isclose(
        candidates[_NET_VALUE_COLUMN].to_numpy(),
        max_value,
        rtol=rtol,
        atol=atol,
    )
    finalists = candidates.loc[close_value].copy()
    min_budget = float(finalists[_ACTUAL_BUDGET_COLUMN].min())
    close_budget = np.isclose(
        finalists[_ACTUAL_BUDGET_COLUMN].to_numpy(),
        min_budget,
        rtol=rtol,
        atol=atol,
    )
    finalists = finalists.loc[close_budget].copy()
    champion_rows = finalists[finalists[_POLICY_COLUMN] == champion]
    if not champion_rows.empty:
        finalists = champion_rows
    finalists = finalists.sort_values(
        [_POLICY_COLUMN, _REQUESTED_BUDGET_COLUMN], kind="mergesort"
    )
    selected = finalists.iloc[0].copy()
    selected["selected_on"] = "validation"
    selected["champion_policy"] = champion
    selected["selection_rule"] = SELECTION_RULE
    return selected


def freeze_economic_decision(
    selected_validation_row: Mapping[str, object] | pd.Series,
    scenario: EconomicScenario,
    *,
    champion_policy: str = "response",
) -> FrozenEconomicDecision:
    """Freeze a selected validation row after checking its scenario contract."""

    if not isinstance(selected_validation_row, Mapping) and not isinstance(
        selected_validation_row, pd.Series
    ):
        raise TypeError("selected_validation_row must be a mapping or pandas Series")
    if not isinstance(scenario, EconomicScenario):
        raise TypeError("scenario must be an EconomicScenario")
    champion = _nonempty_text(champion_policy, "champion_policy")
    row = dict(selected_validation_row)
    required = {
        "scenario_label",
        "currency",
        "horizon",
        "margin_per_conversion",
        "cost_per_assigned_target",
        "eligible_population",
        "economic_assumption",
        _POLICY_COLUMN,
        _REQUESTED_BUDGET_COLUMN,
        _ACTUAL_BUDGET_COLUMN,
        _GAIN_COLUMN,
        _NET_VALUE_COLUMN,
    }
    missing = sorted(required - set(row))
    if missing:
        raise ValueError(f"selected validation row is missing fields: {missing}")
    scenario_record = scenario.to_record()
    for key in (
        "scenario_label",
        "currency",
        "horizon",
        "margin_per_conversion",
        "cost_per_assigned_target",
        "eligible_population",
        "economic_assumption",
    ):
        if row[key] != scenario_record[key]:
            raise ValueError(f"selected validation row does not match scenario field {key!r}")
    if "champion_policy" in row and row["champion_policy"] != champion:
        raise ValueError("selected validation row champion_policy does not match")
    return FrozenEconomicDecision(
        scenario=scenario,
        selected_policy=str(row[_POLICY_COLUMN]),
        requested_budget=float(row[_REQUESTED_BUDGET_COLUMN]),
        actual_budget=float(row[_ACTUAL_BUDGET_COLUMN]),
        validation_gain_vs_treat_none=float(row[_GAIN_COLUMN]),
        validation_incremental_net_value=float(row[_NET_VALUE_COLUMN]),
        champion_policy=champion,
        selection_rule=str(row.get("selection_rule", SELECTION_RULE)),
    )


def select_and_freeze_economic_policy(
    validation_economics: pd.DataFrame,
    scenario: EconomicScenario,
    *,
    champion_policy: str = "response",
    tie_atol: float = 1e-12,
    tie_rtol: float = 1e-12,
) -> FrozenEconomicDecision:
    """Select on validation economics and return an immutable decision record."""

    selected = select_economic_policy(
        validation_economics,
        champion_policy=champion_policy,
        tie_atol=tie_atol,
        tie_rtol=tie_rtol,
    )
    return freeze_economic_decision(
        selected, scenario, champion_policy=champion_policy
    )


def evaluate_frozen_economic_decision(
    final_gain_curve: pd.DataFrame,
    decision: FrozenEconomicDecision,
    *,
    policy_column: str | None = None,
    budget_column: str | None = None,
    actual_budget_column: str | None = None,
    gain_column: str | None = None,
    budget_atol: float = 1e-12,
) -> pd.Series:
    """Evaluate exactly one frozen decision without re-selecting on final data."""

    if not isinstance(decision, FrozenEconomicDecision):
        raise TypeError("decision must be a FrozenEconomicDecision")
    atol = _finite_nonnegative(budget_atol, "budget_atol")
    if decision.targets_nobody:
        row = pd.Series(
            _economic_row(
                scenario=decision.scenario,
                policy=TARGET_NOBODY_POLICY,
                requested_budget=0.0,
                actual_budget=0.0,
                gain=0.0,
            )
        )
    else:
        evaluated = evaluate_economic_scenario(
            final_gain_curve,
            decision.scenario,
            candidate_policies=(decision.selected_policy,),
            policy_column=policy_column,
            budget_column=budget_column,
            actual_budget_column=actual_budget_column,
            gain_column=gain_column,
        )
        match = (
            evaluated[_POLICY_COLUMN].eq(decision.selected_policy)
            & np.isclose(
                evaluated[_REQUESTED_BUDGET_COLUMN].to_numpy(),
                decision.requested_budget,
                rtol=0.0,
                atol=atol,
            )
        )
        matches = evaluated.loc[match]
        if len(matches) != 1:
            raise ValueError(
                "final gain curve must contain exactly one row for the frozen "
                f"policy/budget {decision.selected_policy!r}/{decision.requested_budget}"
            )
        row = matches.iloc[0].copy()
        if not np.isclose(
            float(row[_ACTUAL_BUDGET_COLUMN]),
            decision.actual_budget,
            rtol=0.0,
            atol=atol,
        ):
            raise ValueError("final actual budget does not match the frozen decision")
    row["decision_frozen"] = True
    row["selected_on"] = "validation"
    row["validation_gain_vs_treat_none"] = decision.validation_gain_vs_treat_none
    row["validation_incremental_net_value"] = (
        decision.validation_incremental_net_value
    )
    row["champion_policy"] = decision.champion_policy
    row["selection_rule"] = decision.selection_rule
    return row


def _bootstrap_array(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if np.isinf(array).any():
        raise ValueError(f"{name} must not contain infinite values")
    return array


def transform_paired_gain_bootstrap(
    candidate_gain_replicates: Sequence[float],
    scenario: EconomicScenario,
    candidate_actual_budget: float,
    *,
    reference_gain_replicates: Sequence[float] | None = None,
    reference_actual_budget: float = 0.0,
) -> np.ndarray:
    """Transform paired causal-gain replicates into total net-value differences.

    With no reference replicates, the reference is target nobody (zero gain and
    zero budget).  NaN pairs are retained for the summary helper to omit, matching
    the causal evaluator's handling of invalid bootstrap replicates.
    """

    if not isinstance(scenario, EconomicScenario):
        raise TypeError("scenario must be an EconomicScenario")
    candidate = _bootstrap_array(
        candidate_gain_replicates, "candidate_gain_replicates"
    )
    candidate_budget = _fraction(candidate_actual_budget, "candidate_actual_budget")
    reference_budget = _fraction(reference_actual_budget, "reference_actual_budget")
    if reference_gain_replicates is None:
        if reference_budget != 0.0:
            raise ValueError(
                "reference_actual_budget must be zero when the reference is target nobody"
            )
        reference = np.zeros_like(candidate)
    else:
        reference = _bootstrap_array(
            reference_gain_replicates, "reference_gain_replicates"
        )
        if reference.shape != candidate.shape:
            raise ValueError("candidate and reference bootstrap arrays must have equal shape")
    paired_finite = np.isfinite(candidate) & np.isfinite(reference)
    if not paired_finite.any():
        raise ValueError("no finite paired bootstrap replicates")
    if candidate_budget == 0.0 and not np.allclose(
        candidate[paired_finite], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("zero-budget candidate gain replicates must be zero")
    if reference_budget == 0.0 and not np.allclose(
        reference[paired_finite], 0.0, rtol=0.0, atol=1e-12
    ):
        raise ValueError("zero-budget reference gain replicates must be zero")
    return scenario.eligible_population * (
        scenario.margin_per_conversion * (candidate - reference)
        - scenario.cost_per_assigned_target
        * (candidate_budget - reference_budget)
    )


def summarize_paired_net_value_bootstrap(
    candidate_gain_replicates: Sequence[float],
    scenario: EconomicScenario,
    candidate_actual_budget: float,
    *,
    reference_gain_replicates: Sequence[float] | None = None,
    reference_actual_budget: float = 0.0,
    confidence_level: float = 0.95,
) -> dict[str, str | float | int]:
    """Return percentile uncertainty for a paired net-value comparison."""

    if isinstance(confidence_level, (bool, np.bool_)) or not isinstance(
        confidence_level, Real
    ):
        raise ValueError("confidence_level must be finite and in (0, 1)")
    confidence = float(confidence_level)
    if not np.isfinite(confidence) or not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must be finite and in (0, 1)")
    transformed = transform_paired_gain_bootstrap(
        candidate_gain_replicates,
        scenario,
        candidate_actual_budget,
        reference_gain_replicates=reference_gain_replicates,
        reference_actual_budget=reference_actual_budget,
    )
    finite = transformed[np.isfinite(transformed)]
    alpha = 1.0 - confidence
    low, high = np.percentile(finite, (100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)))
    return {
        **scenario.to_record(),
        "candidate_actual_budget": _fraction(
            candidate_actual_budget, "candidate_actual_budget"
        ),
        "reference_actual_budget": _fraction(
            reference_actual_budget, "reference_actual_budget"
        ),
        "confidence_level": confidence,
        "bootstrap_median_net_value_difference": float(np.median(finite)),
        "net_value_difference_ci_low": float(low),
        "net_value_difference_ci_high": float(high),
        "bootstrap_probability_net_value_difference_positive": float(
            np.mean(finite > 0.0)
        ),
        "valid_paired_bootstrap_replicates": int(finite.size),
    }


__all__ = [
    "ECONOMIC_ASSUMPTION",
    "SELECTION_RULE",
    "TARGET_NOBODY_POLICY",
    "EconomicScenario",
    "FrozenEconomicDecision",
    "evaluate_economic_scenario",
    "evaluate_frozen_economic_decision",
    "freeze_economic_decision",
    "select_and_freeze_economic_policy",
    "select_economic_policy",
    "summarize_paired_net_value_bootstrap",
    "transform_paired_gain_bootstrap",
]
