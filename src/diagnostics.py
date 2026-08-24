"""Treatment-leakage diagnostics for the randomized experiment."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.data import FEATURE_COLUMNS, TREATMENT_COLUMN
from src.models import (
    classification_diagnostics,
    fit_treatment_model,
    predict_treatment,
)


def _rank_deciles(score: np.ndarray) -> np.ndarray:
    """Assign deterministic equal-count deciles, with 1 the highest score."""

    order = np.argsort(-score, kind="mergesort")
    decile = np.empty(len(score), dtype=np.int8)
    decile[order] = np.minimum(9, np.arange(len(score)) * 10 // len(score)) + 1
    return decile


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total**2))
        / denominator
    )
    return float(center - half_width), float(center + half_width)


def treatment_leakage_diagnostics(
    train: pd.DataFrame,
    test: pd.DataFrame,
    *,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit P(T=1|X) on train and report held-out score-decile balance."""

    X_train = train.loc[:, FEATURE_COLUMNS]
    t_train = train[TREATMENT_COLUMN].to_numpy(dtype=np.int8, copy=False)
    X_test = test.loc[:, FEATURE_COLUMNS]
    t_test = test[TREATMENT_COLUMN].to_numpy(dtype=np.int8, copy=False)
    model = fit_treatment_model(X_train, t_train, random_state=random_state)
    probability = predict_treatment(model, X_test)
    metrics = classification_diagnostics(t_test, probability)
    train_share = float(t_train.mean())
    constant_probability = np.full(len(t_test), train_share, dtype=np.float64)
    constant_loss = float(log_loss(t_test, constant_probability, labels=[0, 1]))

    diagnostics = pd.DataFrame(
        [
            {
                "model": "sparse_categorical_logistic",
                "feature_columns": ",".join(FEATURE_COLUMNS),
                "encoded_feature_count": model.encoded_feature_count,
                "fit_iterations": model.fit_iterations,
                "fit_converged": model.fit_converged,
                "train_rows": len(train),
                "test_rows": len(test),
                "train_treatment_share": train_share,
                "test_treatment_share": float(t_test.mean()),
                "roc_auc": metrics["roc_auc"],
                "log_loss": metrics["log_loss"],
                "constant_log_loss": constant_loss,
                "log_loss_improvement_vs_constant": constant_loss
                - float(metrics["log_loss"]),
            }
        ]
    )

    decile = _rank_deciles(probability)
    rows: list[dict[str, float | int]] = []
    overall_share = float(t_test.mean())
    for value in range(1, 11):
        selected = decile == value
        count = int(selected.sum())
        treated = int(t_test[selected].sum())
        low, high = _wilson_interval(treated, count)
        rows.append(
            {
                "treatment_score_decile": value,
                "rows": count,
                "score_min": float(probability[selected].min()),
                "score_mean": float(probability[selected].mean()),
                "score_max": float(probability[selected].max()),
                "treated": treated,
                "treatment_share": float(treated / count),
                "treatment_share_ci_low": low,
                "treatment_share_ci_high": high,
                "overall_test_treatment_share": overall_share,
                "share_minus_overall": float(treated / count - overall_share),
            }
        )
    return diagnostics, pd.DataFrame(rows)


def score_invariance_table(
    original_scores: Mapping[str, Sequence[float]],
    shuffled_treatment_scores: Mapping[str, Sequence[float]],
) -> pd.DataFrame:
    """Prove that observed test treatment is not used during policy scoring."""

    if set(original_scores) != set(shuffled_treatment_scores):
        raise AssertionError("score mappings differ after treatment shuffling")
    rows: list[dict[str, float | str | bool]] = []
    for policy in sorted(original_scores):
        original = np.asarray(original_scores[policy], dtype=np.float64)
        shuffled = np.asarray(shuffled_treatment_scores[policy], dtype=np.float64)
        if original.shape != shuffled.shape:
            raise AssertionError(f"{policy} score shape changed after treatment shuffling")
        maximum = float(np.max(np.abs(original - shuffled), initial=0.0))
        unchanged = bool(np.array_equal(original, shuffled))
        if not unchanged:
            raise AssertionError(
                f"{policy} scores changed after observed test treatment was shuffled; "
                f"max absolute change={maximum}"
            )
        rows.append(
            {
                "policy": policy,
                "observed_test_treatment_shuffled": True,
                "scores_exactly_unchanged": unchanged,
                "max_absolute_score_change": maximum,
            }
        )
    return pd.DataFrame(rows)


def save_treatment_diagnostic_plot(
    deciles: pd.DataFrame, output_dir: str | Path
) -> Path:
    """Plot held-out treatment share across predicted-treatment deciles."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    x = deciles["treatment_score_decile"].to_numpy()
    share = 100.0 * deciles["treatment_share"].to_numpy()
    low = 100.0 * deciles["treatment_share_ci_low"].to_numpy()
    high = 100.0 * deciles["treatment_share_ci_high"].to_numpy()
    overall = 100.0 * float(deciles["overall_test_treatment_share"].iloc[0])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.errorbar(
        x,
        share,
        yerr=np.vstack((share - low, high - share)),
        marker="o",
        capsize=3,
        linewidth=1.4,
        label="Held-out share (95% Wilson CI)",
    )
    ax.axhline(overall, color="black", linestyle="--", label="Overall test share")
    ax.set(
        title="Treatment balance by predicted-treatment score decile",
        xlabel="Predicted-treatment decile (1 = highest score)",
        ylabel="Observed treatment share (%)",
        xticks=x,
    )
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = output_path / "treatment_balance_by_score_decile.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
