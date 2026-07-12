"""Probabilistic forecast metrics used by baselines and models."""

from __future__ import annotations

import numpy as np


def _validated_quantile_inputs(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and normalize arrays shared by quantile diagnostics."""

    truth = np.asarray(y_true, dtype=float)
    predictions = np.asarray(y_quantiles, dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    if truth.ndim != 1:
        raise ValueError("y_true must be one-dimensional")
    if levels.ndim != 1:
        raise ValueError("quantiles must be one-dimensional")
    if predictions.shape != (truth.size, levels.size):
        raise ValueError(
            f"Expected prediction shape {(truth.size, levels.size)}, got {predictions.shape}"
        )
    if not np.isfinite(truth).all() or not np.isfinite(predictions).all():
        raise ValueError("Metric inputs must contain only finite values")
    if not np.isfinite(levels).all() or not np.all((levels > 0) & (levels < 1)):
        raise ValueError("Quantile levels must be finite and lie strictly between 0 and 1")
    if levels.size > 1 and not np.all(np.diff(levels) > 0):
        raise ValueError("Quantile levels must be strictly increasing")
    return truth, predictions, levels


def per_observation_pinball_loss(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Return pinball loss averaged over quantiles for every observation."""

    truth, predictions, levels = _validated_quantile_inputs(
        y_true, y_quantiles, quantiles
    )

    residual = truth[:, None] - predictions
    loss = np.maximum(levels * residual, (levels - 1.0) * residual)
    return loss.mean(axis=1)


def pinball_loss(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> float:
    """Mean pinball loss across observations and requested quantiles."""

    return float(per_observation_pinball_loss(y_true, y_quantiles, quantiles).mean())


def central_interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Observed inclusive coverage of a central predictive interval."""

    truth = np.asarray(y_true, dtype=float)
    lower_bound = np.asarray(lower, dtype=float)
    upper_bound = np.asarray(upper, dtype=float)
    if not (truth.shape == lower_bound.shape == upper_bound.shape):
        raise ValueError("Truth and interval bounds must have identical shapes")
    if truth.ndim != 1:
        raise ValueError("Truth and interval bounds must be one-dimensional")
    if not (
        np.isfinite(truth).all()
        and np.isfinite(lower_bound).all()
        and np.isfinite(upper_bound).all()
    ):
        raise ValueError("Truth and interval bounds must contain only finite values")
    if np.any(lower_bound > upper_bound):
        raise ValueError("Lower interval bounds cannot exceed upper bounds")
    return float(np.mean((truth >= lower_bound) & (truth <= upper_bound)))


def quantile_coverage(
    y_true: np.ndarray,
    y_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Return empirical ``P(Y <= q_tau)`` at every requested quantile."""

    truth, predictions, _ = _validated_quantile_inputs(
        y_true, y_quantiles, quantiles
    )
    return np.mean(truth[:, None] <= predictions, axis=0)


def quantile_crossing_count(
    y_quantiles: np.ndarray,
    quantiles: np.ndarray,
) -> int:
    """Count adjacent quantile pairs that decrease within a forecast row."""

    predictions = np.asarray(y_quantiles, dtype=float)
    levels = np.asarray(quantiles, dtype=float)
    dummy_truth = np.zeros(predictions.shape[0], dtype=float)
    _, validated, _ = _validated_quantile_inputs(dummy_truth, predictions, levels)
    return int(np.sum(np.diff(validated, axis=1) < 0))
