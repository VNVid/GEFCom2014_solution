"""CatBoost MultiQuantile training and prediction contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from catboost import CatBoostRegressor
import numpy as np
import pandas as pd


def build_multi_quantile_loss(quantiles: np.ndarray) -> str:
    """Return CatBoost's comma-separated MultiQuantile objective string."""

    levels = np.asarray(quantiles, dtype=float)
    if levels.ndim != 1 or levels.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional array")
    if not np.isfinite(levels).all() or not np.all((levels > 0) & (levels < 1)):
        raise ValueError("quantiles must be finite and strictly inside (0, 1)")
    if levels.size > 1 and not np.all(np.diff(levels) > 0):
        raise ValueError("quantiles must be strictly increasing")
    alpha = ",".join(f"{level:.10g}" for level in levels)
    return f"MultiQuantile:alpha={alpha}"


def fit_catboost_quantiles(
    features: pd.DataFrame,
    target: pd.Series | np.ndarray,
    quantiles: np.ndarray,
    categorical_features: Sequence[str],
    parameters: Mapping[str, Any],
    *,
    verbose: bool | int = False,
) -> CatBoostRegressor:
    """Fit one CPU CatBoost model that predicts all requested quantiles."""

    if features.empty:
        raise ValueError("Training features must contain at least one row")
    if not features.columns.is_unique:
        raise ValueError("Training feature names must be unique")
    categorical = tuple(categorical_features)
    missing_categorical = sorted(set(categorical) - set(features.columns))
    if missing_categorical:
        raise ValueError(f"Categorical features are missing {missing_categorical}")
    labels = np.asarray(target, dtype=float)
    if labels.ndim != 1 or labels.size != len(features):
        raise ValueError("Target must contain one value per training row")
    if not np.isfinite(labels).all():
        raise ValueError("Training target must contain only finite values")

    configured = dict(parameters)
    reserved = {"loss_function", "allow_writing_files", "task_type", "verbose"}
    overridden = sorted(reserved & set(configured))
    if overridden:
        raise ValueError(f"CatBoost parameters override managed values {overridden}")
    model = CatBoostRegressor(
        loss_function=build_multi_quantile_loss(quantiles),
        task_type="CPU",
        allow_writing_files=False,
        **configured,
    )
    model.fit(
        features,
        labels,
        cat_features=list(categorical),
        verbose=verbose,
    )
    return model


def predict_catboost_quantiles(
    model: CatBoostRegressor,
    features: pd.DataFrame,
    quantiles: np.ndarray,
    *,
    tree_count: int | None = None,
) -> np.ndarray:
    """Predict and validate CatBoost's multi-dimensional quantile output."""

    levels = np.asarray(quantiles, dtype=float)
    if tree_count is not None:
        if not isinstance(tree_count, int) or not 1 <= tree_count <= model.tree_count_:
            raise ValueError(
                f"tree_count must be inside [1, {model.tree_count_}]"
            )
        raw_predictions = model.predict(features, ntree_end=tree_count)
    else:
        raw_predictions = model.predict(features)
    predictions = np.asarray(raw_predictions, dtype=float)
    if predictions.ndim == 1 and levels.size == 1:
        predictions = predictions[:, None]
    expected = (len(features), len(levels))
    if predictions.shape != expected:
        raise ValueError(
            f"Expected CatBoost prediction shape {expected}, got {predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("CatBoost predictions must contain only finite values")
    return predictions
