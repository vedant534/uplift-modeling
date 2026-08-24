"""Sparse, type-correct logistic baselines for response and uplift modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import (
    CATEGORICAL_FEATURE_COLUMNS,
    CONTINUOUS_FEATURE_COLUMNS,
    FEATURE_COLUMNS,
)


def _feature_frame(X: Any) -> pd.DataFrame:
    """Require the exact pre-treatment feature contract in canonical order."""

    if not isinstance(X, pd.DataFrame):
        raise TypeError("X must be a pandas DataFrame with columns f0 through f11")
    actual = tuple(X.columns)
    if actual != FEATURE_COLUMNS:
        raise ValueError(
            "X columns must be exactly f0 through f11 in canonical order; "
            f"observed {list(actual)}"
        )
    values = X.to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("X contains NaN or infinite values")
    return X


def _as_binary(values: Any, name: str) -> np.ndarray:
    """Return a validated one-dimensional binary array."""

    array = np.asarray(values).reshape(-1)
    if not np.isin(array, (0, 1)).all():
        raise ValueError(f"{name} must contain only 0 and 1")
    return array.astype(np.int8, copy=False)


def _check_lengths(X: pd.DataFrame, y: np.ndarray, treatment: np.ndarray) -> None:
    if len(X) != len(y) or len(X) != len(treatment):
        raise ValueError("X, y, and treatment must contain the same number of rows")


def _feature_encoder() -> ColumnTransformer:
    """Scale four continuous fields and sparsely encode eight categoricals."""

    return ColumnTransformer(
        transformers=[
            ("continuous", StandardScaler(), list(CONTINUOUS_FEATURE_COLUMNS)),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                list(CATEGORICAL_FEATURE_COLUMNS),
            ),
        ],
        sparse_threshold=1.0,
        verbose_feature_names_out=False,
    )


LOGISTIC_ALPHA = 1e-4


def _logistic_estimator(random_state: int, max_iter: int) -> SGDClassifier:
    """Build scalable regularized logistic regression for sparse large data.

    Fixed ``alpha=1e-4`` supplies stable regularization for rare categorical
    modalities while keeping the reported test split out of optimizer choices.
    """

    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=LOGISTIC_ALPHA,
        learning_rate="optimal",
        average=False,
        max_iter=max_iter,
        tol=1e-4,
        random_state=random_state,
    )


def _as_csr(values: Any) -> sparse.csr_matrix:
    if sparse.issparse(values):
        return values.tocsr().astype(np.float32, copy=False)
    return sparse.csr_matrix(np.asarray(values, dtype=np.float32))


def _sparse_s_design(
    encoded: sparse.csr_matrix, treatment: np.ndarray
) -> sparse.csr_matrix:
    """Create sparse [Z, T, Z*T] after type-correct feature encoding."""

    treatment_column = sparse.csr_matrix(
        treatment.astype(np.float32, copy=False).reshape(-1, 1)
    )
    interactions = encoded.multiply(treatment_column)
    return sparse.hstack(
        (encoded, treatment_column, interactions), format="csr", dtype=np.float32
    )


@dataclass
class EncodedLogisticModel:
    """A fitted sparse encoder and logistic estimator."""

    encoder: ColumnTransformer
    estimator: SGDClassifier
    treatment_interactions: bool = False

    @property
    def encoded_feature_count(self) -> int:
        return int(len(self.encoder.get_feature_names_out()))

    @property
    def fit_iterations(self) -> int:
        return int(np.max(self.estimator.n_iter_))

    @property
    def fit_converged(self) -> bool:
        return self.fit_iterations < int(self.estimator.max_iter)

    def _design(self, X: Any, treatment: Any | None = None) -> sparse.csr_matrix:
        frame = _feature_frame(X)
        encoded = _as_csr(self.encoder.transform(frame))
        if not self.treatment_interactions:
            if treatment is not None:
                raise ValueError("treatment is only valid for an interaction model")
            return encoded
        if treatment is None:
            raise ValueError("the S-learner design requires a treatment vector")
        assignments = _as_binary(treatment, "treatment")
        if len(assignments) != len(frame):
            raise ValueError("X and treatment must contain the same number of rows")
        return _sparse_s_design(encoded, assignments)

    def positive_probability(
        self, X: Any, treatment: Any | None = None
    ) -> np.ndarray:
        design = self._design(X, treatment)
        probabilities = np.asarray(
            self.estimator.predict_proba(design)[:, 1], dtype=np.float64
        )
        if not np.isfinite(probabilities).all():
            raise RuntimeError("model produced NaN or infinite probabilities")
        if np.any((probabilities < 0.0) | (probabilities > 1.0)):
            raise RuntimeError("model produced probabilities outside [0, 1]")
        return probabilities


def _fit_plain_model(
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    random_state: int,
    max_iter: int,
) -> EncodedLogisticModel:
    encoder = _feature_encoder()
    encoded = _as_csr(encoder.fit_transform(X))
    estimator = _logistic_estimator(random_state, max_iter)
    estimator.fit(encoded, y)
    return EncodedLogisticModel(encoder=encoder, estimator=estimator)


def fit_response_model(
    X: Any,
    y: Any,
    treatment: Any,
    *,
    random_state: int = 42,
    max_iter: int = 500,
) -> EncodedLogisticModel:
    """Fit P(Y=1 | X, T=1), the conventional response-targeting model."""

    features = _feature_frame(X)
    outcomes = _as_binary(y, "y")
    assignments = _as_binary(treatment, "treatment")
    _check_lengths(features, outcomes, assignments)
    treated = assignments == 1
    if not treated.any():
        raise ValueError("the response model requires treated training rows")
    return _fit_plain_model(
        features.loc[treated, :],
        outcomes[treated],
        random_state=random_state,
        max_iter=max_iter,
    )


def predict_response(model: EncodedLogisticModel, X: Any) -> np.ndarray:
    """Predict treated-outcome probability for response-model ranking."""

    return model.positive_probability(X)


def fit_s_learner(
    X: Any,
    y: Any,
    treatment: Any,
    *,
    random_state: int = 42,
    max_iter: int = 500,
) -> EncodedLogisticModel:
    """Fit a sparse logistic S-learner with treatment-feature interactions."""

    features = _feature_frame(X)
    outcomes = _as_binary(y, "y")
    assignments = _as_binary(treatment, "treatment")
    _check_lengths(features, outcomes, assignments)
    encoder = _feature_encoder()
    encoded = _as_csr(encoder.fit_transform(features))
    design = _sparse_s_design(encoded, assignments)
    estimator = _logistic_estimator(random_state, max_iter)
    estimator.fit(design, outcomes)
    return EncodedLogisticModel(
        encoder=encoder,
        estimator=estimator,
        treatment_interactions=True,
    )


def predict_s_learner(
    model: EncodedLogisticModel, X: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu0, mu1, uplift) from S-learner counterfactual scoring."""

    features = _feature_frame(X)
    control = np.zeros(len(features), dtype=np.int8)
    treated = np.ones(len(features), dtype=np.int8)
    mu0 = model.positive_probability(features, control)
    mu1 = model.positive_probability(features, treated)
    uplift = mu1 - mu0
    if not np.isfinite(uplift).all():
        raise RuntimeError("S-learner produced NaN or infinite uplift predictions")
    return mu0, mu1, uplift


def fit_t_learner(
    X: Any,
    y: Any,
    treatment: Any,
    *,
    random_state: int = 42,
    max_iter: int = 500,
) -> tuple[EncodedLogisticModel, EncodedLogisticModel]:
    """Fit separate control and treatment sparse logistic outcome models."""

    features = _feature_frame(X)
    outcomes = _as_binary(y, "y")
    assignments = _as_binary(treatment, "treatment")
    _check_lengths(features, outcomes, assignments)
    control = assignments == 0
    treated = assignments == 1
    if not control.any() or not treated.any():
        raise ValueError("the T-learner requires both control and treated rows")
    control_model = _fit_plain_model(
        features.loc[control, :],
        outcomes[control],
        random_state=random_state,
        max_iter=max_iter,
    )
    treatment_model = _fit_plain_model(
        features.loc[treated, :],
        outcomes[treated],
        random_state=random_state,
        max_iter=max_iter,
    )
    return control_model, treatment_model


def predict_t_learner(
    control_model: EncodedLogisticModel,
    treatment_model: EncodedLogisticModel,
    X: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mu0, mu1, uplift) from T-learner counterfactual scoring."""

    features = _feature_frame(X)
    mu0 = control_model.positive_probability(features)
    mu1 = treatment_model.positive_probability(features)
    uplift = mu1 - mu0
    if not np.isfinite(uplift).all():
        raise RuntimeError("T-learner produced NaN or infinite uplift predictions")
    return mu0, mu1, uplift


def fit_treatment_model(
    X: Any,
    treatment: Any,
    *,
    random_state: int = 42,
    max_iter: int = 500,
) -> EncodedLogisticModel:
    """Fit a held-out leakage diagnostic P(T=1 | X)."""

    features = _feature_frame(X)
    assignments = _as_binary(treatment, "treatment")
    if len(features) != len(assignments):
        raise ValueError("X and treatment must contain the same number of rows")
    if np.unique(assignments).size != 2:
        raise ValueError("the treatment diagnostic requires both treatment arms")
    return _fit_plain_model(
        features,
        assignments,
        random_state=random_state,
        max_iter=max_iter,
    )


def predict_treatment(model: EncodedLogisticModel, X: Any) -> np.ndarray:
    """Predict treatment probability for the leakage diagnostic."""

    return model.positive_probability(X)


def factual_probability(mu0: Any, mu1: Any, treatment: Any) -> np.ndarray:
    """Select each row's prediction for its observed treatment assignment."""

    control_probability = np.asarray(mu0, dtype=np.float64).reshape(-1)
    treatment_probability = np.asarray(mu1, dtype=np.float64).reshape(-1)
    assignments = _as_binary(treatment, "treatment")
    if not (
        len(control_probability) == len(treatment_probability) == len(assignments)
    ):
        raise ValueError("mu0, mu1, and treatment must have the same length")
    if not (
        np.isfinite(control_probability).all()
        and np.isfinite(treatment_probability).all()
    ):
        raise ValueError("potential-outcome predictions must be finite")
    return np.where(assignments == 1, treatment_probability, control_probability)


def classification_diagnostics(y_true: Any, probability: Any) -> dict[str, float | None]:
    """Compute outcome or treatment classification diagnostics."""

    outcomes = _as_binary(y_true, "y_true")
    probabilities = np.asarray(probability, dtype=np.float64).reshape(-1)
    if len(outcomes) != len(probabilities):
        raise ValueError("y_true and probability must have the same length")
    if not np.isfinite(probabilities).all():
        raise ValueError("probability contains NaN or infinite values")
    if np.any((probabilities < 0.0) | (probabilities > 1.0)):
        raise ValueError("probability must lie in [0, 1]")
    auc = (
        float(roc_auc_score(outcomes, probabilities))
        if np.unique(outcomes).size == 2
        else None
    )
    return {
        "roc_auc": auc,
        "log_loss": float(log_loss(outcomes, probabilities, labels=[0, 1])),
    }
