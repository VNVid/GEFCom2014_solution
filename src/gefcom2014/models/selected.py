"""Configuration contracts for refitting selected model candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd


def resolve_selected_features(
    available_features: Sequence[str],
    available_categorical: Sequence[str],
    manifest: Mapping[str, Any] | None,
) -> tuple[list[str], list[str]]:
    """Validate a selected-feature manifest against cached model data."""

    available = list(available_features)
    categorical_pool = list(available_categorical)
    if manifest is None:
        return available, categorical_pool
    features = [str(value) for value in manifest.get("features", [])]
    if not features or len(features) != len(set(features)):
        raise ValueError("Selected-feature manifest must contain unique features")
    missing = sorted(set(features) - set(available))
    if missing:
        raise ValueError(f"Selected features are absent from model data: {missing}")
    categorical = [feature for feature in categorical_pool if feature in features]
    declared = [str(value) for value in manifest.get("categorical_features", [])]
    if set(categorical) != set(declared):
        raise ValueError("Selected categorical features do not match model data")
    return features, categorical


def resolve_candidate_parameters(
    summary: pd.DataFrame,
    candidate: str,
    common_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one exact candidate row into direct CatBoost fit parameters."""

    rows = summary.loc[summary["candidate"].eq(candidate)]
    if len(rows) != 1:
        raise ValueError("Selected candidate must occur exactly once in search summary")
    row = rows.iloc[0]
    return {
        **common_parameters,
        "depth": int(row["depth"]),
        "learning_rate": float(row["learning_rate"]),
        "l2_leaf_reg": float(row["l2_leaf_reg"]),
        "iterations": int(row["iterations"]),
    }
