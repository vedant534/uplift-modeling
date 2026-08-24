#!/usr/bin/env python3
"""Run the complete causal ad-targeting experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path.cwd() / ".cache"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.champion import assess_promotion, champion_challenger_diagnostics
from src.data import (
    CATEGORICAL_FEATURE_COLUMNS,
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
    TREATMENT_COLUMN,
    DevelopmentSplits,
    DisjointSampleMetadata,
    ReservedSampleSpec,
    StagedSamplePlan,
    download_criteo,
    load_pinned_criteo_sample_with_metadata,
    plan_staged_criteo_samples,
    split_development_data,
    summarize_data,
)
from src.diagnostics import (
    save_treatment_diagnostic_plot,
    score_invariance_table,
    treatment_leakage_diagnostics,
)
from src.evaluation import DEFAULT_BUDGETS, evaluate_policies, save_evaluation_plots
from src.economics import (
    TARGET_NOBODY_POLICY,
    EconomicScenario,
    FrozenEconomicDecision,
    evaluate_economic_scenario,
    evaluate_frozen_economic_decision,
    select_and_freeze_economic_policy,
)
from src.governance import (
    CodeManifest,
    EvaluationPaths,
    build_code_manifest,
    create_completion_receipt,
    create_decision_freeze,
    create_final_access_claim,
    derive_evaluation_paths,
    preflight_evaluation,
    sanitize_evaluation_id,
    verify_completion_receipt,
    verify_final_access_claim,
)
from src.models import (
    EncodedLogisticModel,
    LOGISTIC_ALPHA,
    classification_diagnostics,
    factual_probability,
    fit_response_model,
    fit_s_learner,
    fit_t_learner,
    predict_response,
    predict_s_learner,
    predict_t_learner,
)


DEFAULT_DATA_PATH = Path("data/criteo-research-uplift-v2.1.csv.gz")
DEFAULT_SAMPLE_SIZE = 2_000_000
DEFAULT_FINAL_EVALUATION_SAMPLE_SIZE = 400_000
DEFAULT_FINAL_EVALUATION_SEED = 20_260_818
DEFAULT_BOOTSTRAP_REPS = 1_000
DEFAULT_VALIDATION_BOOTSTRAP_REPS = 100
DEFAULT_OUTPUT_DIR = Path("reports")
DEFAULT_PROTOCOL_PATH = Path("decision_protocol.json")
SUPPORTED_PROTOCOL_VERSION = 2
SUPPORTED_PRIMARY_ESTIMATOR = "horvitz_thompson"
SUPPORTED_PROMOTION_RULE_ID = "paired_ht_net_value_superiority_95_hajek_direction"
SUPPORTED_TIE_BREAK_ORDER = [
    "higher_incremental_net_value",
    "lower_targeting_budget",
    "champion_response_before_t_learner",
]
SUPPORTED_SOURCE_ACCESS_ORDER = [
    "plan_source_indices_only",
    "load_development_only",
    "freeze_validation_decisions",
    "create_immutable_final_access_claim",
    "load_final_evaluation_only",
]
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_MANIFEST_FILES = (
    "run_experiment.py",
    "src/__init__.py",
    "src/champion.py",
    "src/data.py",
    "src/diagnostics.py",
    "src/economics.py",
    "src/evaluation.py",
    "src/governance.py",
    "src/models.py",
)

CSV_FILENAMES = {
    "source_metadata": "source_metadata.csv",
    "data_summary": "data_summary.csv",
    "model_diagnostics": "model_diagnostics.csv",
    "treatment_diagnostics": "treatment_diagnostics.csv",
    "treatment_score_deciles": "treatment_score_deciles.csv",
    "score_invariance": "score_invariance.csv",
    "ranking_metrics": "ranking_metrics.csv",
    "qini_curve": "qini_curve.csv",
    "budget_metrics": "budget_metrics.csv",
    "policy_comparisons": "policy_comparisons.csv",
    "decile_metrics": "decile_metrics.csv",
    "validation_qini_curve": "validation_qini_curve.csv",
    "validation_economic_grid": "validation_economic_grid.csv",
    "validation_economic_optima": "validation_economic_optima.csv",
    "final_frozen_economic_decisions": "final_frozen_economic_decisions.csv",
    "champion_challenger_diagnostics": "champion_challenger_diagnostics.csv",
}
EXPECTED_FIGURES = {
    "qini_curve.png",
    "policy_value.png",
    "uplift_vs_response.png",
    "uplift_by_decile.png",
    "uplift_distribution.png",
    "treatment_balance_by_score_decile.png",
    "economic_net_value_validation.png",
}
EVALUATION_KEYS = {
    "data_summary",
    "reference_metrics",
    "ranking_metrics",
    "qini_curve",
    "budget_metrics",
    "policy_comparisons",
    "decile_metrics",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare response and uplift targeting on the corrected Criteo "
            "randomized advertising dataset."
        )
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download or verify the corrected Criteo v2.1 file at --data-path.",
    )
    parser.add_argument(
        "--outcome", choices=("conversion", "visit"), default="conversion"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Exact seeded sample size (default: {DEFAULT_SAMPLE_SIZE:,}).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--final-evaluation-sample-size",
        type=int,
        default=DEFAULT_FINAL_EVALUATION_SAMPLE_SIZE,
        help=(
            "Fresh source-index-disjoint final-evaluation rows "
            f"(default: {DEFAULT_FINAL_EVALUATION_SAMPLE_SIZE:,})."
        ),
    )
    parser.add_argument(
        "--final-evaluation-seed",
        type=int,
        default=DEFAULT_FINAL_EVALUATION_SEED,
    )
    parser.add_argument(
        "--bootstrap-reps",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPS,
        help=(
            "Paired held-out row-bootstrap repetitions "
            f"(default: {DEFAULT_BOOTSTRAP_REPS:,})."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-path", type=Path, default=DEFAULT_PROTOCOL_PATH)
    parser.add_argument(
        "--validation-bootstrap-reps",
        type=int,
        default=DEFAULT_VALIDATION_BOOTSTRAP_REPS,
        help="Bootstrap repetitions for validation diagnostics; never used to select on final data.",
    )
    args = parser.parse_args(argv)
    if args.sample_size <= 0:
        parser.error("--sample-size must be positive")
    if args.bootstrap_reps <= 0:
        parser.error("--bootstrap-reps must be positive")
    if args.final_evaluation_sample_size <= 0:
        parser.error("--final-evaluation-sample-size must be positive")
    if args.validation_bootstrap_reps <= 0:
        parser.error("--validation-bootstrap-reps must be positive")
    return args


def _resolve_dataset(path: Path, allow_download: bool) -> Path:
    if allow_download:
        print(f"Downloading or verifying dataset at {path} ...", flush=True)
        return download_criteo(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Pass --download to fetch corrected v2.1."
        )
    return path


def _sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_decision_protocol(
    path: Path, args: argparse.Namespace
) -> tuple[dict[str, Any], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Decision protocol not found: {path}")
    raw = path.read_bytes()
    try:
        protocol = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Decision protocol is not valid JSON: {path}") from exc
    if not isinstance(protocol, dict):
        raise ValueError("Decision protocol must be a JSON object")
    required = {
        "evaluation_id",
        "protocol_version",
        "legacy_result_status",
        "data_contract",
        "model_contract",
        "policy_roles",
        "selection",
        "economic_scenarios",
        "champion_challenger",
        "final_reporting",
        "execution_contract",
    }
    missing = sorted(required - set(protocol))
    if missing:
        raise ValueError(f"Decision protocol is missing sections: {missing}")
    if protocol["protocol_version"] != SUPPORTED_PROTOCOL_VERSION:
        raise ValueError(
            f"Only decision protocol version {SUPPORTED_PROTOCOL_VERSION} is supported"
        )
    sanitize_evaluation_id(protocol["evaluation_id"])
    data_contract = protocol["data_contract"]
    model_contract = protocol["model_contract"]
    selection = protocol["selection"]
    policy_roles = protocol["policy_roles"]
    champion_challenger = protocol["champion_challenger"]
    final_reporting = protocol["final_reporting"]
    execution_contract = protocol["execution_contract"]
    expected_runtime = {
        "outcome": args.outcome,
        "development_sample_size": args.sample_size,
        "development_sampling_seed": args.seed,
        "final_evaluation_sample_size": args.final_evaluation_sample_size,
        "final_evaluation_sampling_seed": args.final_evaluation_seed,
    }
    mismatches = {
        key: {"protocol": data_contract.get(key), "runtime": value}
        for key, value in expected_runtime.items()
        if data_contract.get(key) != value
    }
    if final_reporting.get("bootstrap_repetitions") != args.bootstrap_reps:
        mismatches["bootstrap_repetitions"] = {
            "protocol": final_reporting.get("bootstrap_repetitions"),
            "runtime": args.bootstrap_reps,
        }
    if (
        final_reporting.get("validation_bootstrap_repetitions")
        != args.validation_bootstrap_reps
    ):
        mismatches["validation_bootstrap_repetitions"] = {
            "protocol": final_reporting.get("validation_bootstrap_repetitions"),
            "runtime": args.validation_bootstrap_reps,
        }
    expected_output = (PROJECT_ROOT / execution_contract.get("output_directory", "")).resolve()
    runtime_output = args.output_dir.resolve()
    if expected_output != runtime_output:
        mismatches["output_directory"] = {
            "protocol": str(expected_output),
            "runtime": str(runtime_output),
        }
    if mismatches:
        raise ValueError(
            "Runtime arguments do not match the frozen decision protocol: "
            f"{mismatches}"
        )
    if protocol["legacy_result_status"] != "exploratory_not_eligible_for_new_budget_selection":
        raise ValueError("Legacy multi-budget results must remain explicitly exploratory")
    if selection.get("sample") != "development_validation":
        raise ValueError("Policy and budget selection must use development_validation")
    if data_contract.get("development_validation_fraction") != 0.20:
        raise ValueError("Only the frozen 80/20 development split is supported")
    if selection.get("primary_estimator") != SUPPORTED_PRIMARY_ESTIMATOR:
        raise ValueError(
            f"selection.primary_estimator must be {SUPPORTED_PRIMARY_ESTIMATOR!r}"
        )
    if selection.get("objective") != "maximize_incremental_net_value_vs_target_nobody":
        raise ValueError("Unsupported economic selection objective")
    if selection.get("tie_break_order") != SUPPORTED_TIE_BREAK_ORDER:
        raise ValueError("Protocol tie-break order does not match executable semantics")
    expected_roles = {
        "champion": "response",
        "challenger": "t_learner",
        "development_rejected_candidate": "s_learner",
    }
    for field, expected in expected_roles.items():
        if policy_roles.get(field) != expected:
            raise ValueError(f"policy_roles.{field} must be {expected!r}")
    if selection.get("candidate_policies") != [
        "response",
        "t_learner",
        TARGET_NOBODY_POLICY,
    ]:
        raise ValueError("Protocol candidate policies do not match executable selection")
    if (
        champion_challenger.get("promotion_rule_id")
        != SUPPORTED_PROMOTION_RULE_ID
    ):
        raise ValueError("Protocol promotion rule does not match executable semantics")
    if champion_challenger.get("noninferiority_rule") is not None:
        raise ValueError("This protocol must not silently introduce noninferiority")
    if final_reporting.get("confidence_level") != 0.95:
        raise ValueError("Only the frozen 95 percent interval contract is supported")
    if execution_contract.get("source_access_order") != SUPPORTED_SOURCE_ACCESS_ORDER:
        raise ValueError("Protocol source-access order does not match staged execution")
    if execution_contract.get("one_shot_final_evaluation") is not True:
        raise ValueError("Protocol must require a one-shot final evaluation")
    expected_model_contract = {
        "outcome": args.outcome,
        "feature_columns": list(FEATURE_COLUMNS),
        "random_state": args.seed,
        "logistic_estimator": "SGDClassifier(loss=log_loss, penalty=l2)",
        "logistic_alpha": LOGISTIC_ALPHA,
    }
    if model_contract != expected_model_contract:
        raise ValueError(
            "Protocol model contract does not match the executable model configuration"
        )
    manifest = CodeManifest.from_record(execution_contract.get("code_manifest"))
    actual_manifest = build_code_manifest(PROJECT_ROOT, CODE_MANIFEST_FILES)
    if manifest != actual_manifest:
        raise ValueError("Protocol code manifest does not match current executable bytes")
    if execution_contract.get("code_manifest_sha256") != manifest.sha256:
        raise ValueError("Protocol code-manifest digest is invalid")
    return protocol, hashlib.sha256(raw).hexdigest()


def _protocol_budget_grid(protocol: dict[str, Any]) -> np.ndarray:
    grid = protocol["selection"]["candidate_budget_grid"]
    start = float(grid["start"])
    stop = float(grid["stop"])
    step = float(grid["step"])
    if not (
        np.isfinite((start, stop, step)).all()
        and 0.0 < start <= stop <= 1.0
        and step > 0.0
    ):
        raise ValueError("Protocol budget grid must be finite within (0, 1]")
    count = int(round((stop - start) / step)) + 1
    values = start + step * np.arange(count, dtype=np.float64)
    if not np.isclose(values[-1], stop, rtol=0.0, atol=1e-12):
        raise ValueError("Protocol budget grid stop is not reachable by its step")
    return np.round(values, 12)


def _protocol_scenarios(
    protocol: dict[str, Any],
) -> list[tuple[str, EconomicScenario]]:
    specifications = protocol["economic_scenarios"]
    if not isinstance(specifications, list) or not specifications:
        raise ValueError("Decision protocol must define economic scenarios")
    scenarios: list[tuple[str, EconomicScenario]] = []
    labels: set[str] = set()
    primary_count = 0
    for spec in specifications:
        label = str(spec["name"])
        if label in labels:
            raise ValueError(f"Duplicate economic scenario: {label}")
        labels.add(label)
        role = str(spec["role"])
        if role not in {"primary", "sensitivity"}:
            raise ValueError(f"Unknown scenario role {role!r}")
        primary_count += int(role == "primary")
        scenarios.append(
            (
                role,
                EconomicScenario(
                    label=label,
                    margin_per_conversion=float(
                        spec["conversion_contribution_margin"]
                    ),
                    cost_per_assigned_target=float(spec["cost_per_assigned_target"]),
                    eligible_population=int(spec["eligible_population"]),
                    currency=str(spec["currency"]),
                    horizon=str(spec["horizon"]),
                ),
            )
        )
    if primary_count != 1:
        raise ValueError("Decision protocol must define exactly one primary scenario")
    return scenarios


def _protocol_reserved_samples(
    protocol: dict[str, Any],
) -> tuple[ReservedSampleSpec, ...]:
    specifications = protocol["data_contract"].get(
        "reserved_previously_accessed_samples"
    )
    if not isinstance(specifications, list) or not specifications:
        raise ValueError(
            "Protocol must reserve every previously accessed final-evaluation sample"
        )
    reserved: list[ReservedSampleSpec] = []
    for spec in specifications:
        if not isinstance(spec, dict):
            raise ValueError("Reserved-sample specifications must be JSON objects")
        required = {
            "sample_role",
            "sample_size",
            "sampling_seed",
            "selected_row_index_sha256",
            "excluded_sample_roles",
        }
        if set(spec) != required:
            raise ValueError(
                "Reserved-sample specification fields must be exactly "
                f"{sorted(required)}"
            )
        if not isinstance(spec["excluded_sample_roles"], list):
            raise ValueError("excluded_sample_roles must be a JSON array")
        reserved.append(
            ReservedSampleSpec(
                sample_role=spec["sample_role"],
                sample_size=spec["sample_size"],
                sampling_seed=spec["sampling_seed"],
                expected_selected_row_index_sha256=(
                    spec["selected_row_index_sha256"]
                ),
                excluded_sample_roles=tuple(spec["excluded_sample_roles"]),
            )
        )
    if len({sample.sample_role for sample in reserved}) != len(reserved):
        raise ValueError("Reserved-sample roles must be unique")
    return tuple(reserved)


def _plan_staged_samples(
    dataset_path: Path,
    args: argparse.Namespace,
    protocol: dict[str, Any],
) -> StagedSamplePlan:
    contract = protocol["data_contract"]
    plan = plan_staged_criteo_samples(
        dataset_path,
        development_sample_size=args.sample_size,
        final_evaluation_sample_size=args.final_evaluation_sample_size,
        development_seed=args.seed,
        final_evaluation_seed=args.final_evaluation_seed,
        reserved_accessed_samples=_protocol_reserved_samples(protocol),
        expected_development_row_index_sha256=str(
            contract["development_selected_row_index_sha256"]
        ),
    )
    expected = {
        "source_sha256": plan.development.source_sha256,
        "development_selected_row_index_sha256": (
            plan.development.selected_row_index_sha256
        ),
        "pre_final_excluded_union_row_index_sha256": (
            plan.pre_final_excluded_union_row_index_sha256
        ),
        "pre_final_excluded_union_row_count": (
            plan.pre_final_excluded_union_row_count
        ),
        "final_evaluation_selected_row_index_sha256": (
            plan.final_evaluation.selected_row_index_sha256
        ),
        "all_pinned_union_row_index_sha256": (
            plan.all_pinned_union_row_index_sha256
        ),
        "all_pinned_union_row_count": plan.all_pinned_union_row_count,
    }
    mismatches = {
        key: {"protocol": contract.get(key), "planned": value}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise ValueError(f"Staged source-index plan disagrees with protocol: {mismatches}")
    if plan.final_overlap_with_excluded_union_count != 0:
        raise AssertionError("Final sample overlaps a pre-final accessed sample")
    return plan


def _protocol_code_manifest(protocol: dict[str, Any]) -> CodeManifest:
    return CodeManifest.from_record(
        protocol["execution_contract"]["code_manifest"]
    )


def _reserved_union_index_sha256(sample_plan: StagedSamplePlan) -> str:
    rows = np.empty(0, dtype=np.int64)
    for pin in sample_plan.reserved_accessed_samples:
        rows = np.union1d(rows, pin.selected_row_indices)
    if rows.size == 0:
        raise ValueError("At least one previously accessed sample must be reserved")
    canonical = np.asarray(rows, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _split_summaries(
    development: pd.DataFrame,
    splits: DevelopmentSplits,
    final_evaluation: pd.DataFrame,
    outcome: str,
    validation_evaluation_summary: pd.DataFrame,
    final_evaluation_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, frame in (
        ("development_full", development),
        ("development_train", splits.train),
        ("development_validation", splits.validation),
        ("final_evaluation", final_evaluation),
    ):
        rows.append({"sample": name, "outcome": outcome, **summarize_data(frame, outcome)})
    summaries = pd.DataFrame(rows)
    for column in (
        "estimated_propensity",
        "treated_positive_outcomes",
        "control_positive_outcomes",
        "bootstrap_repetitions",
        "valid_bootstrap_repetitions",
    ):
        summaries[column] = np.nan
        for sample_name, evaluation_summary in (
            ("development_validation", validation_evaluation_summary),
            ("final_evaluation", final_evaluation_summary),
        ):
            details = evaluation_summary.iloc[0]
            summaries.loc[summaries["sample"] == sample_name, column] = details[
                column
            ]
    return summaries


def _one_diagnostic(
    model_name: str,
    split_name: str,
    scope: str,
    outcomes: np.ndarray,
    probabilities: np.ndarray,
    encoded_feature_count: int,
    fit_iterations: int,
    fit_converged: bool,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "split": split_name,
        "scope": scope,
        "encoded_feature_count": encoded_feature_count,
        "fit_iterations": fit_iterations,
        "fit_converged": fit_converged,
        "evaluation_rows": int(len(outcomes)),
        "positive_outcomes": int(outcomes.sum()),
        "outcome_rate": float(outcomes.mean()),
        **classification_diagnostics(outcomes, probabilities),
    }


def _model_diagnostics(
    splits: DevelopmentSplits,
    outcome: str,
    response_model: EncodedLogisticModel,
    s_model: EncodedLogisticModel,
    control_model: EncodedLogisticModel,
    treatment_model: EncodedLogisticModel,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name, frame in (
        ("train", splits.train),
        ("validation", splits.validation),
    ):
        X = frame.loc[:, FEATURE_COLUMNS]
        y = frame[outcome].to_numpy(dtype=np.int8, copy=False)
        treatment = frame[TREATMENT_COLUMN].to_numpy(dtype=np.int8, copy=False)
        treated = treatment == 1
        rows.append(
            _one_diagnostic(
                "response",
                split_name,
                "treated_rows",
                y[treated],
                predict_response(response_model, X.loc[treated, :]),
                response_model.encoded_feature_count,
                response_model.fit_iterations,
                response_model.fit_converged,
            )
        )
        s_mu0, s_mu1, _ = predict_s_learner(s_model, X)
        rows.append(
            _one_diagnostic(
                "s_learner",
                split_name,
                "observed_treatment_factual",
                y,
                factual_probability(s_mu0, s_mu1, treatment),
                s_model.encoded_feature_count,
                s_model.fit_iterations,
                s_model.fit_converged,
            )
        )
        t_mu0, t_mu1, _ = predict_t_learner(control_model, treatment_model, X)
        rows.append(
            _one_diagnostic(
                "t_learner",
                split_name,
                "observed_treatment_factual",
                y,
                factual_probability(t_mu0, t_mu1, treatment),
                max(
                    control_model.encoded_feature_count,
                    treatment_model.encoded_feature_count,
                ),
                max(control_model.fit_iterations, treatment_model.fit_iterations),
                control_model.fit_converged and treatment_model.fit_converged,
            )
        )
    return pd.DataFrame(rows)


def _policy_scores(
    frame: pd.DataFrame,
    response_model: EncodedLogisticModel,
    s_model: EncodedLogisticModel,
    control_model: EncodedLogisticModel,
    treatment_model: EncodedLogisticModel,
) -> dict[str, np.ndarray]:
    if tuple(frame.loc[:, FEATURE_COLUMNS].columns) != FEATURE_COLUMNS:
        raise AssertionError("policy-score feature contract is not exactly f0 through f11")
    X = frame.loc[:, FEATURE_COLUMNS]
    scores = {
        "response": predict_response(response_model, X),
        "s_learner": predict_s_learner(s_model, X)[2],
        "t_learner": predict_t_learner(control_model, treatment_model, X)[2],
    }
    for name, score in scores.items():
        if score.ndim != 1 or len(score) != len(frame):
            raise AssertionError(f"{name} score has the wrong shape")
        if not np.isfinite(score).all():
            raise AssertionError(f"{name} score contains NaN or infinite values")
    return scores


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _validation_gain_curve(
    curve: pd.DataFrame,
    protocol: dict[str, Any],
) -> pd.DataFrame:
    budget_grid = _protocol_budget_grid(protocol)
    candidate_policies = tuple(
        policy
        for policy in protocol["selection"]["candidate_policies"]
        if policy != TARGET_NOBODY_POLICY
    )
    if not candidate_policies:
        raise ValueError("Protocol must include at least one scored candidate policy")
    missing = sorted(set(candidate_policies) - set(curve["policy"]))
    if missing:
        raise ValueError(f"Validation gain curve omitted candidate policies: {missing}")
    requested = curve["targeting_fraction"].to_numpy(dtype=np.float64)
    on_grid = np.any(
        np.isclose(
            requested[:, None],
            budget_grid[None, :],
            rtol=0.0,
            atol=1e-12,
        ),
        axis=1,
    )
    selected = curve[curve["policy"].isin(candidate_policies) & on_grid].copy()
    expected_rows = len(candidate_policies) * len(budget_grid)
    if len(selected) != expected_rows:
        raise AssertionError(
            f"Expected {expected_rows} validation policy/budget candidates, got {len(selected)}"
        )
    return selected


def _select_and_freeze_economic_decisions(
    validation_curve: pd.DataFrame,
    protocol: dict[str, Any],
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[tuple[str, str], FrozenEconomicDecision],
]:
    champion = str(protocol["policy_roles"]["champion"])
    challenger = str(protocol["policy_roles"]["challenger"])
    candidate_curve = _validation_gain_curve(validation_curve, protocol)
    economic_tables: list[pd.DataFrame] = []
    optimum_records: list[dict[str, Any]] = []
    decisions: dict[tuple[str, str], FrozenEconomicDecision] = {}
    for role, scenario in _protocol_scenarios(protocol):
        economics = evaluate_economic_scenario(
            candidate_curve,
            scenario,
            candidate_policies=(champion, challenger),
        )
        economics.insert(1, "scenario_role", role)
        economic_tables.append(economics)
        scopes = {
            "global_response_vs_t": economics,
            f"policy_{champion}": economics[
                economics["policy"].isin((TARGET_NOBODY_POLICY, champion))
            ],
            f"policy_{challenger}": economics[
                economics["policy"].isin((TARGET_NOBODY_POLICY, challenger))
            ],
        }
        for scope, candidates in scopes.items():
            decision = select_and_freeze_economic_policy(
                candidates,
                scenario,
                champion_policy=champion,
            )
            decisions[(scenario.label, scope)] = decision
            selected = candidates[
                candidates["policy"].eq(decision.selected_policy)
                & np.isclose(
                    candidates["requested_budget"],
                    decision.requested_budget,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(selected) != 1:
                raise AssertionError("Frozen validation decision is not uniquely recoverable")
            selected_row = selected.iloc[0]
            optimum_records.append(
                {
                    **decision.to_record(),
                    "scenario_role": role,
                    "decision_scope": scope,
                    "validation_break_even_cost_to_margin_ratio": selected_row[
                        "break_even_cost_to_margin_ratio"
                    ],
                    "validation_break_even_cost_per_assigned_target": selected_row[
                        "break_even_cost_per_assigned_target"
                    ],
                }
            )
    return (
        pd.concat(economic_tables, ignore_index=True),
        pd.DataFrame(optimum_records),
        decisions,
    )


def _evaluate_frozen_economic_decisions(
    final_curve: pd.DataFrame,
    final_budget_metrics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    decisions: dict[tuple[str, str], FrozenEconomicDecision],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for optimum in validation_optima.to_dict(orient="records"):
        key = (str(optimum["scenario_label"]), str(optimum["decision_scope"]))
        decision = decisions[key]
        final = evaluate_frozen_economic_decision(final_curve, decision).to_dict()
        final["scenario_role"] = optimum["scenario_role"]
        final["decision_scope"] = optimum["decision_scope"]
        if decision.targets_nobody:
            gain_low = gain_high = 0.0
            net_low = net_high = 0.0
            break_even_low = break_even_high = np.nan
        else:
            match = final_budget_metrics[
                final_budget_metrics["policy"].eq(decision.selected_policy)
                & np.isclose(
                    final_budget_metrics["budget"],
                    decision.requested_budget,
                    rtol=0.0,
                    atol=1e-12,
                )
            ]
            if len(match) != 1:
                raise AssertionError(
                    "Final budget metrics omitted a frozen policy/budget decision"
                )
            metric = match.iloc[0]
            if not np.isclose(
                float(final["gain_vs_treat_none"]),
                float(metric["gain_vs_treat_none"]),
                rtol=0.0,
                atol=1e-12,
            ):
                raise AssertionError("Final economic and causal gain estimates disagree")
            gain_low = float(metric["gain_ci_low"])
            gain_high = float(metric["gain_ci_high"])
            scenario = decision.scenario
            population = scenario.eligible_population
            margin = scenario.margin_per_conversion
            cost = scenario.cost_per_assigned_target
            budget = decision.actual_budget
            net_low = population * (margin * gain_low - cost * budget)
            net_high = population * (margin * gain_high - cost * budget)
            break_even_low = margin * gain_low / budget
            break_even_high = margin * gain_high / budget
        final.update(
            {
                "final_gain_ci_low": gain_low,
                "final_gain_ci_high": gain_high,
                "final_incremental_net_value_ci_low": net_low,
                "final_incremental_net_value_ci_high": net_high,
                "final_break_even_cost_per_assigned_target_ci_low": break_even_low,
                "final_break_even_cost_per_assigned_target_ci_high": break_even_high,
                "final_net_value_conclusion_vs_target_nobody": (
                    "baseline_target_nobody"
                    if decision.targets_nobody
                    else "positive"
                    if net_low > 0.0
                    else "negative"
                    if net_high < 0.0
                    else "inconclusive"
                ),
                "final_evaluation_policy_frozen": True,
            }
        )
        rows.append(final)
    return pd.DataFrame(rows)


def _primary_scenario_label(validation_optima: pd.DataFrame) -> str:
    labels = validation_optima.loc[
        validation_optima["scenario_role"].eq("primary"), "scenario_label"
    ].unique()
    if len(labels) != 1:
        raise AssertionError("Expected exactly one primary economic scenario")
    return str(labels[0])


def _promotion_assessment(
    results: dict[str, pd.DataFrame],
    champion_diagnostics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    decisions: dict[tuple[str, str], FrozenEconomicDecision],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    champion = str(protocol["policy_roles"]["champion"])
    challenger = str(protocol["policy_roles"]["challenger"])
    primary_label = _primary_scenario_label(validation_optima)
    champion_decision = decisions[(primary_label, f"policy_{champion}")]
    base: dict[str, Any] = {
        "champion_policy": champion,
        "challenger_policy": challenger,
        "scenario_label": primary_label,
        "promotion_rule": protocol["champion_challenger"]["promotion_rule"],
        "noninferiority_rule": protocol["champion_challenger"][
            "noninferiority_rule"
        ],
        "scope": "offline_fixed_policy_evidence_only",
    }
    if champion_decision.targets_nobody:
        return {
            **base,
            "promotion_budget": 0.0,
            "rule_passed": False,
            "decision": "retain_champion_primary_scenario_targets_nobody",
            "online_requirement": protocol["champion_challenger"][
                "online_requirement"
            ],
        }
    budget = champion_decision.requested_budget
    diagnostic = champion_diagnostics[
        champion_diagnostics["challenger_policy"].eq(challenger)
        & np.isclose(
            champion_diagnostics["budget"], budget, rtol=0.0, atol=1e-12
        )
    ]
    if len(diagnostic) != 1:
        raise AssertionError("Champion diagnostics omitted the promotion budget")
    diagnostic_row = diagnostic.iloc[0]
    scenario = champion_decision.scenario
    scale = scenario.eligible_population * scenario.margin_per_conversion
    delta = scale * float(
        diagnostic_row["challenger_minus_champion_ht_policy_value"]
    )
    low = scale * float(
        diagnostic_row["challenger_minus_champion_ht_policy_value_ci_low"]
    )
    high = scale * float(
        diagnostic_row["challenger_minus_champion_ht_policy_value_ci_high"]
    )
    assessment = assess_promotion(
        delta,
        low,
        high,
        metric="paired_incremental_net_value_challenger_minus_champion",
        rule="superiority",
    )
    comparisons = results["policy_comparisons"]
    sensitivity = comparisons[
        comparisons["candidate_policy"].eq(challenger)
        & comparisons["reference_policy"].eq(champion)
        & np.isclose(
            comparisons["candidate_budget"], budget, rtol=0.0, atol=1e-12
        )
    ]
    if len(sensitivity) != 1:
        raise AssertionError("Final policy comparisons omitted promotion sensitivity")
    sensitivity_row = sensitivity.iloc[0]
    hajek_delta = scale * float(sensitivity_row["delta_hajek_policy_value"])
    direction_consistent = bool(hajek_delta > 0.0)
    if assessment["rule_passed"] and not direction_consistent:
        assessment["rule_passed"] = False
        assessment["decision"] = "retain_champion_hajek_direction_not_consistent"
    return {
        **base,
        **assessment,
        "promotion_budget": budget,
        "cost_difference_at_equal_budget": 0.0,
        "hajek_net_value_delta": hajek_delta,
        "hajek_direction_consistent": direction_consistent,
        "online_requirement": protocol["champion_challenger"][
            "online_requirement"
        ],
    }


def _save_economic_plot(
    validation_economics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    output_dir: Path,
) -> Path:
    primary = _primary_scenario_label(validation_optima)
    scenarios = list(
        validation_economics[["scenario_label", "scenario_role"]]
        .drop_duplicates()
        .sort_values(["scenario_role", "scenario_label"])["scenario_label"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    colors = {"response": "#4C78A8", "t_learner": "#54A24B"}
    for ax, label in zip(axes.flat, scenarios, strict=True):
        rows = validation_economics[validation_economics["scenario_label"].eq(label)]
        currencies = rows["currency"].unique()
        horizons = rows["horizon"].unique()
        if len(currencies) != 1 or len(horizons) != 1:
            raise AssertionError("Each economic scenario must have one currency and horizon")
        for policy in ("response", "t_learner"):
            policy_rows = rows[rows["policy"].eq(policy)].sort_values("actual_budget")
            ax.plot(
                100.0 * policy_rows["actual_budget"],
                policy_rows["incremental_net_value_vs_target_nobody"],
                color=colors[policy],
                label=policy.replace("_", " ").title(),
            )
        optimum = validation_optima[
            validation_optima["scenario_label"].eq(label)
            & validation_optima["decision_scope"].eq("global_response_vs_t")
        ].iloc[0]
        ax.scatter(
            100.0 * float(optimum["actual_budget"]),
            float(optimum["validation_incremental_net_value"]),
            color="black",
            marker="*",
            s=90,
            zorder=5,
            label="Validation-selected optimum",
        )
        ax.axhline(
            0.0,
            color="0.4",
            linestyle=":",
            linewidth=1.0,
            label="Target nobody (zero net value)",
        )
        display_label = label.replace("_", " ")
        role = " (primary)" if label == primary else ""
        ax.set_title(
            f"{display_label}{role}\n{currencies[0]}; {horizons[0]}",
            fontsize=9,
        )
        ax.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Eligible records targeted (%)")
    axes[1, 1].set_xlabel("Eligible records targeted (%)")
    axes[0, 0].set_ylabel("Incremental net value")
    axes[1, 0].set_ylabel("Incremental net value")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    fig.suptitle("Validation-only economic policy and budget selection", fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    path = output_dir / "economic_net_value_validation.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _validate_results(results: dict[str, pd.DataFrame]) -> None:
    missing = EVALUATION_KEYS - set(results)
    if missing:
        raise AssertionError(f"evaluation omitted required tables: {sorted(missing)}")
    for name in EVALUATION_KEYS:
        table = results[name]
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise AssertionError(f"evaluation table {name!r} is missing or empty")


def _at_budget(frame: pd.DataFrame, budget: float) -> pd.DataFrame:
    return frame[np.isclose(frame["budget"], budget)]


def _result_card(
    results: dict[str, pd.DataFrame],
    treatment_diagnostics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    final_economic_decisions: pd.DataFrame,
    champion_diagnostics: pd.DataFrame,
    promotion: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    legacy_budget = 0.20
    budget_rows = _at_budget(results["budget_metrics"], legacy_budget).set_index("policy")
    comparisons = results["policy_comparisons"]
    s_response = comparisons[
        (comparisons["candidate_policy"] == "s_learner")
        & (comparisons["reference_policy"] == "response")
        & np.isclose(comparisons["candidate_budget"], legacy_budget)
    ].iloc[0]
    t_response = comparisons[
        (comparisons["candidate_policy"] == "t_learner")
        & (comparisons["reference_policy"] == "response")
        & np.isclose(comparisons["candidate_budget"], legacy_budget)
    ].iloc[0]
    primary_final = final_economic_decisions[
        final_economic_decisions["scenario_role"].eq("primary")
        & final_economic_decisions["decision_scope"].eq("global_response_vs_t")
    ].iloc[0]
    primary_validation = validation_optima[
        validation_optima["scenario_role"].eq("primary")
        & validation_optima["decision_scope"].eq("global_response_vs_t")
    ].iloc[0]
    challenger = str(protocol["policy_roles"]["challenger"])
    overlap = champion_diagnostics[
        champion_diagnostics["challenger_policy"].eq(challenger)
        & champion_diagnostics["budget"].isin((0.10, 0.20))
    ].sort_values("budget")
    headline = (
        f"Validation selected {primary_validation['selected_policy']} at "
        f"{100.0 * float(primary_validation['actual_budget']):.0f}% under the "
        f"explicitly hypothetical primary economic scenario. On the fresh disjoint "
        f"final sample, its incremental net value versus target nobody was "
        f"{float(primary_final['incremental_net_value_vs_target_nobody']):.1f} "
        f"{primary_final['currency']}; {promotion['decision']}."
    )
    return {
        "headline": headline,
        "legacy_20_percent_result_status": protocol["legacy_result_status"],
        "primary_estimator": "horvitz_thompson",
        "primary_economic_scenario": {
            "scenario_label": primary_final["scenario_label"],
            "currency": primary_final["currency"],
            "horizon": primary_final["horizon"],
            "margin_per_conversion": float(primary_final["margin_per_conversion"]),
            "cost_per_assigned_target": float(
                primary_final["cost_per_assigned_target"]
            ),
            "validation_selected_policy": primary_validation["selected_policy"],
            "validation_selected_budget": float(primary_validation["actual_budget"]),
            "final_incremental_net_value_vs_target_nobody": float(
                primary_final["incremental_net_value_vs_target_nobody"]
            ),
            "final_incremental_net_value_ci": [
                float(primary_final["final_incremental_net_value_ci_low"]),
                float(primary_final["final_incremental_net_value_ci_high"]),
            ],
            "final_break_even_cost_per_assigned_target": (
                None
                if pd.isna(primary_final["break_even_cost_per_assigned_target"])
                else float(primary_final["break_even_cost_per_assigned_target"])
            ),
        },
        "champion_challenger": {
            "champion": protocol["policy_roles"]["champion"],
            "challenger": challenger,
            "development_rejected_candidate": protocol["policy_roles"][
                "development_rejected_candidate"
            ],
            "promotion": promotion,
            "overlap_diagnostics": _json_records(
                overlap[
                    [
                        "budget",
                        "targeted_count",
                        "jaccard",
                        "intersection_count",
                        "champion_only_count",
                        "challenger_only_count",
                        "total_switched_count",
                        "switched_ht_uplift_delta_per_1000",
                        "switched_ht_uplift_delta_per_1000_ci_low",
                        "switched_ht_uplift_delta_per_1000_ci_high",
                    ]
                ]
            ),
        },
        "fresh_final_20_percent_diagnostic": {
            "s_learner": {
                "ht_incremental_per_1000_targeted": float(
                    budget_rows.loc[
                        "s_learner", "incremental_outcomes_per_1000_targeted"
                    ]
                ),
                "ht_minus_response_per_1000": float(
                    s_response["delta_incremental_per_1000_targeted"]
                ),
                "ht_minus_response_ci": [
                    float(s_response["delta_incremental_per_1000_ci_low"]),
                    float(s_response["delta_incremental_per_1000_ci_high"]),
                ],
                "ht_minus_response_conclusion": s_response[
                    "incremental_conclusion"
                ],
                "primary_policy_value_minus_response": float(
                    s_response["delta_policy_value"]
                ),
                "primary_policy_value_minus_response_ci": [
                    float(s_response["delta_policy_value_ci_low"]),
                    float(s_response["delta_policy_value_ci_high"]),
                ],
                "primary_policy_value_conclusion": s_response[
                    "policy_value_conclusion"
                ],
            },
            "t_learner": {
                "ht_incremental_per_1000_targeted": float(
                    budget_rows.loc[
                        "t_learner", "incremental_outcomes_per_1000_targeted"
                    ]
                ),
                "ht_minus_response_per_1000": float(
                    t_response["delta_incremental_per_1000_targeted"]
                ),
                "ht_minus_response_ci": [
                    float(t_response["delta_incremental_per_1000_ci_low"]),
                    float(t_response["delta_incremental_per_1000_ci_high"]),
                ],
                "ht_minus_response_conclusion": t_response[
                    "incremental_conclusion"
                ],
                "primary_policy_value_minus_response": float(
                    t_response["delta_policy_value"]
                ),
                "primary_policy_value_minus_response_ci": [
                    float(t_response["delta_policy_value_ci_low"]),
                    float(t_response["delta_policy_value_ci_high"]),
                ],
                "primary_policy_value_conclusion": t_response[
                    "policy_value_conclusion"
                ],
            },
        },
        "treatment_leakage_diagnostic_auc": float(
            treatment_diagnostics.iloc[0]["roc_auc"]
        ),
        "bootstrap_repetitions": int(
            results["data_summary"].loc[
                results["data_summary"]["sample"] == "final_evaluation",
                "bootstrap_repetitions",
            ].iloc[0]
        ),
    }


def _source_metadata_table(
    sample_plan: StagedSamplePlan,
    metadata: DisjointSampleMetadata,
) -> pd.DataFrame:
    loaded = {
        "development": metadata.development,
        "final_evaluation": metadata.final_evaluation,
    }
    rows: list[dict[str, Any]] = []
    for pin in sample_plan.all_samples:
        loaded_metadata = loaded.get(pin.sample_role)
        if pin.sample_role == "development":
            access_status = "loaded_before_freeze_for_development_only"
        elif pin.sample_role == "final_evaluation":
            access_status = "loaded_after_immutable_final_access_claim"
        else:
            access_status = "reserved_previously_accessed_not_loaded_in_current_run"
        rows.append(
            {
                "sample_role": pin.sample_role,
                "access_status": access_status,
                "dataset_version": pin.dataset_version,
                "source_path": pin.source_path,
                "source_sha256": pin.source_sha256,
                "source_sha256_verification": pin.source_sha256_verification,
                "source_row_count": pin.source_row_count,
                "requested_sample_size": pin.sample_size,
                "loaded_sample_size_in_current_run": (
                    0
                    if loaded_metadata is None
                    else loaded_metadata.loaded_sample_size
                ),
                "sampling_method": pin.sampling_method,
                "sampling_seed": pin.sampling_seed,
                "selected_row_index_sha256": pin.selected_row_index_sha256,
                "selected_min_source_row": pin.selected_min_source_row,
                "selected_max_source_row": pin.selected_max_source_row,
                "excluded_sample_roles": ",".join(pin.excluded_sample_roles),
                "excluded_union_row_count": pin.excluded_union_row_count,
                "excluded_union_row_index_sha256": (
                    pin.excluded_union_row_index_sha256
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_outputs(
    results: dict[str, pd.DataFrame],
    validation_results: dict[str, pd.DataFrame],
    model_diagnostics: pd.DataFrame,
    treatment_diagnostics: pd.DataFrame,
    treatment_deciles: pd.DataFrame,
    score_invariance: pd.DataFrame,
    scores: dict[str, np.ndarray],
    validation_economics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    final_economic_decisions: pd.DataFrame,
    champion_diagnostics: pd.DataFrame,
    promotion: dict[str, Any],
    args: argparse.Namespace,
    metadata: DisjointSampleMetadata,
    sample_plan: StagedSamplePlan,
    protocol: dict[str, Any],
    protocol_sha256: str,
    governance_paths: EvaluationPaths,
    frozen_decisions_path: Path,
    expected_frozen_decisions_sha256: str,
    final_access_claim_sha256: str,
) -> tuple[list[Path], list[Path], list[Path]]:
    metrics_dir = args.output_dir / "metrics"
    figures_dir = args.output_dir / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    source_metadata = _source_metadata_table(sample_plan, metadata)
    tables = {
        **results,
        "source_metadata": source_metadata,
        "model_diagnostics": model_diagnostics,
        "treatment_diagnostics": treatment_diagnostics,
        "treatment_score_deciles": treatment_deciles,
        "score_invariance": score_invariance,
        "validation_qini_curve": validation_results["qini_curve"],
        "validation_economic_grid": validation_economics,
        "validation_economic_optima": validation_optima,
        "final_frozen_economic_decisions": final_economic_decisions,
        "champion_challenger_diagnostics": champion_diagnostics,
    }
    csv_paths: list[Path] = []
    for key, filename in CSV_FILENAMES.items():
        path = metrics_dir / filename
        tables[key].to_csv(path, index=False)
        csv_paths.append(path)

    figure_paths = save_evaluation_plots(
        results, scores, figures_dir, random_state=args.seed
    )
    figure_paths.append(save_treatment_diagnostic_plot(treatment_deciles, figures_dir))
    figure_paths.append(
        _save_economic_plot(validation_economics, validation_optima, figures_dir)
    )
    if {path.name for path in figure_paths} != EXPECTED_FIGURES:
        raise AssertionError(
            f"expected figures {sorted(EXPECTED_FIGURES)}, got "
            f"{sorted(path.name for path in figure_paths)}"
        )

    result_card = _result_card(
        results,
        treatment_diagnostics,
        validation_optima,
        final_economic_decisions,
        champion_diagnostics,
        promotion,
        protocol,
    )
    result_card_path = metrics_dir / "result_card.json"
    result_card_path.write_text(
        json.dumps(result_card, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    promotion_path = metrics_dir / "promotion_decision.json"
    promotion_path.write_text(
        json.dumps(promotion, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if not frozen_decisions_path.is_file():
        raise AssertionError("Frozen validation decision artifact is missing")
    frozen_decisions_sha256 = _sha256_bytes(frozen_decisions_path)
    if frozen_decisions_sha256 != expected_frozen_decisions_sha256:
        raise AssertionError("Frozen validation decisions changed during final evaluation")
    if frozen_decisions_path != governance_paths.freeze_path:
        raise AssertionError("Frozen decision path is not the governed evaluation path")
    verify_final_access_claim(
        governance_paths,
        protocol_path=args.protocol_path,
        code_root=PROJECT_ROOT,
        expected_freeze_sha256=expected_frozen_decisions_sha256,
        expected_claim_sha256=final_access_claim_sha256,
        require_receipt_absent=True,
    )
    summary = {
        "configuration": {
            "evaluation_id": protocol["evaluation_id"],
            "dataset_path": metadata.development.source_path,
            "dataset_version": metadata.development.dataset_version,
            "source_row_count": metadata.development.source_row_count,
            "source_sha256": metadata.development.source_sha256,
            "source_sha256_verification": metadata.development.source_sha256_verification,
            "development_sampling_method": metadata.development.sampling_method,
            "development_sampling_seed": metadata.development.sampling_seed,
            "development_selected_row_index_sha256": (
                metadata.development.selected_row_index_sha256
            ),
            "final_evaluation_sampling_method": (
                metadata.final_evaluation.sampling_method
            ),
            "final_evaluation_sampling_seed": (
                metadata.final_evaluation.sampling_seed
            ),
            "final_evaluation_selected_row_index_sha256": (
                metadata.final_evaluation.selected_row_index_sha256
            ),
            "source_row_overlap_count": metadata.source_row_overlap_count,
            "source_row_overlap_verified": metadata.source_row_overlap_verified,
            "pre_final_excluded_sample_roles": list(
                sample_plan.pre_final_excluded_sample_roles
            ),
            "pre_final_excluded_union_row_count": (
                sample_plan.pre_final_excluded_union_row_count
            ),
            "pre_final_excluded_union_row_index_sha256": (
                sample_plan.pre_final_excluded_union_row_index_sha256
            ),
            "final_overlap_with_pre_final_excluded_union_count": (
                sample_plan.final_overlap_with_excluded_union_count
            ),
            "all_pinned_union_row_count": sample_plan.all_pinned_union_row_count,
            "all_pinned_union_row_index_sha256": (
                sample_plan.all_pinned_union_row_index_sha256
            ),
            "outcome": args.outcome,
            "development_sample_size": metadata.development.loaded_sample_size,
            "final_evaluation_sample_size": (
                metadata.final_evaluation.loaded_sample_size
            ),
            "seed": args.seed,
            "bootstrap_repetitions": args.bootstrap_reps,
            "validation_bootstrap_repetitions": args.validation_bootstrap_reps,
            "final_evaluation_budgets": sorted(
                float(value) for value in results["budget_metrics"]["budget"].unique()
            ),
            "selection_budget_grid": [
                float(value) for value in _protocol_budget_grid(protocol)
            ],
            "split_fractions": {
                "development_train": 0.80,
                "development_validation": 0.20,
                "final_evaluation": (
                    "fresh_sample_disjoint_from_development_and_all_previously_"
                    "accessed_final_samples"
                ),
            },
            "continuous_features": list(CONTINUOUS_FEATURE_COLUMNS),
            "categorical_features": list(CATEGORICAL_FEATURE_COLUMNS),
            "categorical_encoding": "sparse_one_hot_handle_unknown_ignore",
            "logistic_optimizer": "SGDClassifier(loss=log_loss, penalty=l2)",
            "logistic_alpha": LOGISTIC_ALPHA,
            "primary_policy_estimator": "horvitz_thompson",
            "sensitivity_policy_estimator": "hajek_self_normalized",
            "random_baseline": "analytical_expected_random",
            "decision_protocol_path": str(args.protocol_path),
            "decision_protocol_sha256": protocol_sha256,
            "frozen_decisions_path": str(frozen_decisions_path),
            "frozen_decisions_sha256": frozen_decisions_sha256,
            "final_access_claim_path": str(governance_paths.claim_path),
            "final_access_claim_sha256": final_access_claim_sha256,
            "completion_receipt_path": str(governance_paths.receipt_path),
            "legacy_result_status": protocol["legacy_result_status"],
        },
        "decision_protocol": protocol,
        "result_card": result_card,
        "data_summary": _json_records(results["data_summary"]),
        "model_diagnostics": _json_records(model_diagnostics),
        "treatment_diagnostics": _json_records(treatment_diagnostics),
        "treatment_score_deciles": _json_records(treatment_deciles),
        "score_invariance": _json_records(score_invariance),
        "reference_metrics": _json_records(results["reference_metrics"]),
        "ranking_metrics": _json_records(results["ranking_metrics"]),
        "budget_metrics": _json_records(results["budget_metrics"]),
        "policy_comparisons": _json_records(results["policy_comparisons"]),
        "decile_metrics": _json_records(results["decile_metrics"]),
        "validation_economic_optima": _json_records(validation_optima),
        "final_frozen_economic_decisions": _json_records(final_economic_decisions),
        "champion_challenger_diagnostics": _json_records(champion_diagnostics),
        "promotion_decision": promotion,
        "artifacts": {
            "metric_tables": [str(path) for path in csv_paths],
            "figures": [str(path) for path in figure_paths],
            "result_card": str(result_card_path),
            "frozen_decisions": str(frozen_decisions_path),
            "final_access_claim": str(governance_paths.claim_path),
            "completion_receipt": str(governance_paths.receipt_path),
            "promotion_decision": str(promotion_path),
        },
    }
    summary_path = metrics_dir / "experiment_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    json_paths = [
        result_card_path,
        frozen_decisions_path,
        governance_paths.claim_path,
        promotion_path,
        summary_path,
    ]
    expected_paths = [*csv_paths, *figure_paths, *json_paths]
    missing_or_empty = [
        path for path in expected_paths if not path.is_file() or path.stat().st_size == 0
    ]
    if missing_or_empty:
        raise AssertionError(f"missing or empty output artifacts: {missing_or_empty}")
    return csv_paths, figure_paths, json_paths


def _print_summary(
    results: dict[str, pd.DataFrame],
    diagnostics: pd.DataFrame,
    treatment_diagnostics: pd.DataFrame,
    validation_optima: pd.DataFrame,
    promotion: dict[str, Any],
    output_dir: Path,
) -> None:
    print("\nValidation outcome-model diagnostics:")
    validation = diagnostics[diagnostics["split"] == "validation"]
    print(
        validation[["model", "encoded_feature_count", "roc_auc", "log_loss"]].to_string(
            index=False, float_format=lambda value: f"{value:.6f}"
        )
    )
    print("\nFresh final-sample treatment leakage diagnostic:")
    print(
        treatment_diagnostics[
            ["roc_auc", "log_loss", "constant_log_loss", "log_loss_improvement_vs_constant"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nFresh final-sample uplift ranking metrics:")
    print(
        results["ranking_metrics"][
            ["policy", "auuc", "qini_coefficient", "qini_ci_low", "qini_ci_high"]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    comparisons = results["policy_comparisons"]
    versus_response = comparisons[comparisons["reference_policy"] == "response"]
    print("\nUplift policies versus response targeting:")
    print(
        versus_response[
            [
                "candidate_policy",
                "candidate_budget",
                "delta_policy_value",
                "delta_policy_value_ci_low",
                "delta_policy_value_ci_high",
                "policy_value_conclusion",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print("\nValidation-selected economic optima:")
    global_optima = validation_optima[
        validation_optima["decision_scope"].eq("global_response_vs_t")
    ]
    print(
        global_optima[
            [
                "scenario_label",
                "scenario_role",
                "selected_policy",
                "actual_budget",
                "validation_incremental_net_value",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
    print(f"\nPromotion decision: {promotion['decision']}")
    print(f"\nSaved metrics and figures under {output_dir}")


def run(args: argparse.Namespace) -> dict[str, pd.DataFrame]:
    protocol, protocol_sha256 = _load_decision_protocol(args.protocol_path, args)
    governance_paths = derive_evaluation_paths(
        args.output_dir / "metrics", protocol["evaluation_id"]
    )
    preflight_evaluation(governance_paths)
    dataset_path = _resolve_dataset(args.data_path, args.download)
    print(
        f"Planning {args.sample_size:,} development rows and a new "
        f"{args.final_evaluation_sample_size:,}-row final sample without reading "
        f"final outcomes for outcome={args.outcome!r} ...",
        flush=True,
    )
    sample_plan = _plan_staged_samples(
        dataset_path,
        args,
        protocol,
    )
    print(
        f"Pinned final index digest "
        f"{sample_plan.final_evaluation.selected_row_index_sha256[:12]}... outside "
        f"{sample_plan.pre_final_excluded_union_row_count:,} development/rehearsal "
        f"rows; overlap={sample_plan.final_overlap_with_excluded_union_count}.",
        flush=True,
    )
    print("Loading the development pin only ...", flush=True)
    development_loaded = load_pinned_criteo_sample_with_metadata(
        dataset_path,
        sample_plan.development,
        outcome=args.outcome,
    )
    development = development_loaded.frame
    print(
        f"Verified {development_loaded.metadata.source_row_count:,} source rows via "
        f"{development_loaded.metadata.source_sha256_verification}; final rows remain "
        "unmaterialized.",
        flush=True,
    )
    validation_fraction = float(
        protocol["data_contract"]["development_validation_fraction"]
    )
    splits = split_development_data(
        development,
        outcome=args.outcome,
        seed=args.seed,
        validation_fraction=validation_fraction,
    )
    print(
        f"Split development rows: train={len(splits.train):,}, "
        f"validation={len(splits.validation):,}",
        flush=True,
    )
    X_train = splits.train.loc[:, FEATURE_COLUMNS]
    y_train = splits.train[args.outcome]
    treatment_train = splits.train[TREATMENT_COLUMN]

    print("Fitting sparse categorical response model ...", flush=True)
    response_model = fit_response_model(
        X_train, y_train, treatment_train, random_state=args.seed
    )
    print("Fitting sparse categorical interaction S-learner ...", flush=True)
    s_model = fit_s_learner(X_train, y_train, treatment_train, random_state=args.seed)
    print("Fitting sparse categorical T-learner outcome models ...", flush=True)
    control_model, treatment_model = fit_t_learner(
        X_train, y_train, treatment_train, random_state=args.seed
    )
    fitted_models = {
        "response": response_model,
        "s_learner": s_model,
        "t_learner_control": control_model,
        "t_learner_treatment": treatment_model,
    }
    nonconverged = [
        name for name, model in fitted_models.items() if not model.fit_converged
    ]
    if nonconverged:
        raise RuntimeError(f"refusing to publish non-converged models: {nonconverged}")
    diagnostics = _model_diagnostics(
        splits,
        args.outcome,
        response_model,
        s_model,
        control_model,
        treatment_model,
    )

    validation_scores = _policy_scores(
        splits.validation, response_model, s_model, control_model, treatment_model
    )
    print(
        "Selecting policy and budget decisions on development validation data ...",
        flush=True,
    )
    validation_results = evaluate_policies(
        splits.validation[args.outcome].to_numpy(copy=False),
        splits.validation[TREATMENT_COLUMN].to_numpy(copy=False),
        validation_scores,
        budgets=DEFAULT_BUDGETS,
        n_bootstrap=args.validation_bootstrap_reps,
        random_state=args.seed,
        curve_points=101,
    )
    _validate_results(validation_results)
    validation_economics, validation_optima, frozen_decisions = (
        _select_and_freeze_economic_decisions(
            validation_results["qini_curve"], protocol
        )
    )

    frozen_records: list[dict[str, Any]] = []
    for optimum in validation_optima.to_dict(orient="records"):
        key = (str(optimum["scenario_label"]), str(optimum["decision_scope"]))
        frozen_records.append(
            {
                **frozen_decisions[key].to_record(),
                "scenario_role": optimum["scenario_role"],
                "decision_scope": optimum["decision_scope"],
            }
        )
    frozen_payload = {
        "frozen_before_final_outcome_access": True,
        "decision_protocol_path": str(args.protocol_path),
        "decision_protocol_sha256": protocol_sha256,
        "evaluation_id": protocol["evaluation_id"],
        "outcome": args.outcome,
        "selection_sample": "development_validation",
        "sample_plan": sample_plan.to_dict(),
        "model_configuration": protocol["model_contract"],
        "code_manifest": _protocol_code_manifest(protocol).to_record(),
        "code_manifest_sha256": _protocol_code_manifest(protocol).sha256,
        "frozen_decisions": frozen_records,
    }
    frozen_record = create_decision_freeze(
        governance_paths,
        frozen_payload,
    )
    frozen_decisions_path = frozen_record.path
    frozen_decisions_sha256 = frozen_record.sha256
    print(
        f"Frozen {len(frozen_records)} validation decisions at "
        f"{frozen_decisions_path} ({frozen_decisions_sha256[:12]}...).",
        flush=True,
    )

    code_manifest = _protocol_code_manifest(protocol)
    claim_record = create_final_access_claim(
        governance_paths,
        protocol_path=args.protocol_path,
        protocol_sha256=protocol_sha256,
        freeze_sha256=frozen_decisions_sha256,
        outcome=args.outcome,
        source_sha256=sample_plan.development.source_sha256,
        development_index_sha256=(
            sample_plan.development.selected_row_index_sha256
        ),
        reserved_index_sha256=_reserved_union_index_sha256(sample_plan),
        final_index_sha256=(
            sample_plan.final_evaluation.selected_row_index_sha256
        ),
        code_manifest=code_manifest,
        code_root=PROJECT_ROOT,
        caller_metadata={
            "source_access_order": SUPPORTED_SOURCE_ACCESS_ORDER,
            "pre_final_excluded_union_row_count": (
                sample_plan.pre_final_excluded_union_row_count
            ),
            "pre_final_excluded_union_row_index_sha256": (
                sample_plan.pre_final_excluded_union_row_index_sha256
            ),
            "final_overlap_with_excluded_union_count": (
                sample_plan.final_overlap_with_excluded_union_count
            ),
        },
    )
    verify_final_access_claim(
        governance_paths,
        protocol_path=args.protocol_path,
        code_root=PROJECT_ROOT,
        expected_freeze_sha256=frozen_decisions_sha256,
        expected_claim_sha256=claim_record.sha256,
        require_receipt_absent=True,
    )
    print(
        f"Created immutable one-shot final-access claim "
        f"{claim_record.path} ({claim_record.sha256[:12]}...).",
        flush=True,
    )
    print("Loading the pinned final-evaluation rows only after the claim ...", flush=True)
    final_loaded = load_pinned_criteo_sample_with_metadata(
        dataset_path,
        sample_plan.final_evaluation,
        outcome=args.outcome,
    )
    final_evaluation = final_loaded.frame
    metadata = DisjointSampleMetadata(
        development=development_loaded.metadata,
        final_evaluation=final_loaded.metadata,
        source_row_overlap_count=int(
            np.intersect1d(
                sample_plan.development.selected_row_indices,
                sample_plan.final_evaluation.selected_row_indices,
                assume_unique=True,
            ).size
        ),
    )
    if not metadata.source_row_overlap_verified:
        raise AssertionError("Development and final loaded samples overlap")

    print("Fitting fresh final-sample treatment-leakage diagnostic ...", flush=True)
    treatment_diagnostics, treatment_deciles = treatment_leakage_diagnostics(
        splits.train, final_evaluation, random_state=args.seed
    )
    if not bool(treatment_diagnostics.iloc[0]["fit_converged"]):
        raise RuntimeError("refusing to publish non-converged treatment diagnostic")
    scores = _policy_scores(
        final_evaluation, response_model, s_model, control_model, treatment_model
    )
    shuffled_final = final_evaluation.copy()
    shuffled_final[TREATMENT_COLUMN] = np.random.default_rng(args.seed).permutation(
        shuffled_final[TREATMENT_COLUMN].to_numpy(copy=True)
    )
    shuffled_scores = _policy_scores(
        shuffled_final, response_model, s_model, control_model, treatment_model
    )
    invariance = score_invariance_table(scores, shuffled_scores)
    print(
        "Verified exact score invariance after final treatment shuffling.",
        flush=True,
    )

    frozen_budgets = {
        round(decision.requested_budget, 12)
        for decision in frozen_decisions.values()
        if not decision.targets_nobody
    }
    overlap_budgets = {
        round(float(value), 12)
        for value in protocol["champion_challenger"]["overlap_budgets"]
    }
    final_budgets = tuple(
        sorted(
            {
                *(round(float(value), 12) for value in DEFAULT_BUDGETS),
                *frozen_budgets,
                *overlap_budgets,
            }
        )
    )
    print(
        f"Evaluating frozen policies on fresh final rows with "
        f"{args.bootstrap_reps:,} paired bootstrap repetitions ...",
        flush=True,
    )
    results = evaluate_policies(
        final_evaluation[args.outcome].to_numpy(copy=False),
        final_evaluation[TREATMENT_COLUMN].to_numpy(copy=False),
        scores,
        budgets=final_budgets,
        n_bootstrap=args.bootstrap_reps,
        random_state=args.seed,
        curve_points=101,
    )
    _validate_results(results)
    primary_label = _primary_scenario_label(validation_optima)
    champion = str(protocol["policy_roles"]["champion"])
    primary_champion = frozen_decisions[(primary_label, f"policy_{champion}")]
    champion_budgets = set(overlap_budgets)
    if not primary_champion.targets_nobody:
        champion_budgets.add(round(primary_champion.requested_budget, 12))
    champion_diagnostics = champion_challenger_diagnostics(
        final_evaluation[args.outcome].to_numpy(copy=False),
        final_evaluation[TREATMENT_COLUMN].to_numpy(copy=False),
        scores,
        budgets=tuple(sorted(champion_budgets)),
        champion=champion,
        challengers=(
            str(protocol["policy_roles"]["challenger"]),
            str(protocol["policy_roles"]["development_rejected_candidate"]),
        ),
        n_bootstrap=args.bootstrap_reps,
        random_state=args.seed,
    )
    promotion = _promotion_assessment(
        results,
        champion_diagnostics,
        validation_optima,
        frozen_decisions,
        protocol,
    )
    final_economic_decisions = _evaluate_frozen_economic_decisions(
        results["qini_curve"],
        results["budget_metrics"],
        validation_optima,
        frozen_decisions,
    )
    results["data_summary"] = _split_summaries(
        development,
        splits,
        final_evaluation,
        args.outcome,
        validation_results["data_summary"],
        results["data_summary"],
    )
    csv_paths, figure_paths, json_paths = _write_outputs(
        results,
        validation_results,
        diagnostics,
        treatment_diagnostics,
        treatment_deciles,
        invariance,
        scores,
        validation_economics,
        validation_optima,
        final_economic_decisions,
        champion_diagnostics,
        promotion,
        args,
        metadata,
        sample_plan,
        protocol,
        protocol_sha256,
        governance_paths,
        frozen_decisions_path,
        frozen_decisions_sha256,
        claim_record.sha256,
    )
    governed_paths = {governance_paths.freeze_path, governance_paths.claim_path}
    output_paths = [
        path
        for path in (*csv_paths, *figure_paths, *json_paths)
        if path not in governed_paths
    ]
    output_artifacts = {
        f"{path.parent.name}-{path.stem}": path.resolve() for path in output_paths
    }
    if len(output_artifacts) != len(output_paths):
        raise AssertionError("Output artifact governance names are not unique")
    receipt_record = create_completion_receipt(
        governance_paths,
        protocol_path=args.protocol_path,
        protocol_sha256=protocol_sha256,
        freeze_sha256=frozen_decisions_sha256,
        claim_sha256=claim_record.sha256,
        code_root=PROJECT_ROOT,
        output_root=args.output_dir,
        output_artifacts=output_artifacts,
        caller_metadata={
            "final_evaluation_completed": True,
            "secondary_budget_curves": protocol["final_reporting"][
                "secondary_budget_curves"
            ],
        },
    )
    verify_completion_receipt(
        governance_paths,
        protocol_path=args.protocol_path,
        code_root=PROJECT_ROOT,
        output_root=args.output_dir,
        expected_freeze_sha256=frozen_decisions_sha256,
        expected_claim_sha256=claim_record.sha256,
        expected_receipt_sha256=receipt_record.sha256,
    )
    print(
        f"Sealed completion receipt {receipt_record.path} "
        f"({receipt_record.sha256[:12]}...).",
        flush=True,
    )
    _print_summary(
        results,
        diagnostics,
        treatment_diagnostics,
        validation_optima,
        promotion,
        args.output_dir,
    )
    return results


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
