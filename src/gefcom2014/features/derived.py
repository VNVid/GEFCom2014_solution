"""Cheap target-specific transformations of already constructed features."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


def build_horizon_decay_features(
    features: pd.DataFrame,
    horizon_hours: pd.Series,
    *,
    source_scales_days: Mapping[str, float],
) -> pd.DataFrame:
    """Exponentially decay pre-origin signals through the forecast month."""

    if not source_scales_days:
        raise ValueError("source_scales_days must contain at least one feature")
    if not features.index.equals(horizon_hours.index):
        raise ValueError("Features and horizon_hours must have identical indices")
    horizon = horizon_hours.to_numpy(dtype=float)
    if not np.isfinite(horizon).all() or np.any(horizon < 0):
        raise ValueError("horizon_hours must be finite and non-negative")

    output = pd.DataFrame(index=features.index)
    for source, raw_scale in source_scales_days.items():
        if source not in features:
            raise ValueError(f"Horizon-decay source feature {source!r} is unavailable")
        if not pd.api.types.is_numeric_dtype(features[source]):
            raise TypeError(f"Horizon-decay source feature {source!r} must be numeric")
        scale = float(raw_scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Horizon-decay scales must be finite and positive")
        label = f"{scale:g}".replace(".", "p")
        output[f"{source}_decay_{label}d"] = (
            features[source].to_numpy(dtype=float)
            * np.exp(-horizon / (24.0 * scale))
        )
    return output
