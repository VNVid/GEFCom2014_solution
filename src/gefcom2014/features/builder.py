"""Configuration-driven composition of independently testable feature blocks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from .adjustment import build_seasonal_load_adjustment_features
from .annual import build_annual_load_anchor_features
from .load import build_recent_load_features
from .profiles import build_recent_load_profile_features
from .seasonal import build_seasonal_load_features
from .target import TARGET_CATEGORICAL_FEATURES, build_target_time_features
from .temperature import (
    build_recent_temperature_features,
    build_temperature_climatology_features,
)


SUPPORTED_FEATURE_GROUPS = (
    "target_time",
    "recent_load",
    "recent_load_profile",
    "seasonal_load_profile",
    "seasonal_load_adjustment",
    "annual_load_anchors",
    "temperature_climatology",
    "recent_temperature",
)


def build_feature_matrix(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    feature_groups: Mapping[str, Mapping[str, Any] | None],
    *,
    weather_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build selected feature groups with stable target-row alignment.

    ``weather_history`` may contain a longer pre-origin record than ``history``
    so weather-only years can support climatology without entering load-derived
    feature blocks. When omitted, all groups use ``history``.
    """

    unknown = sorted(set(feature_groups) - set(SUPPORTED_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups {unknown}")
    if not feature_groups:
        raise ValueError("At least one feature group must be selected")

    # Validate the monthly origin/target contract even when an alternative
    # feature set deliberately omits target-time columns.
    target_time = build_target_time_features(target, origin)
    blocks: list[pd.DataFrame] = []
    if "target_time" in feature_groups:
        parameters = feature_groups["target_time"] or {}
        if parameters:
            raise ValueError("target_time does not accept configuration parameters")
        blocks.append(target_time)
    if "recent_load" in feature_groups:
        parameters = dict(feature_groups["recent_load"] or {})
        blocks.append(build_recent_load_features(history, target, origin, **parameters))
    if "recent_load_profile" in feature_groups:
        parameters = dict(feature_groups["recent_load_profile"] or {})
        blocks.append(
            build_recent_load_profile_features(
                history, target, origin, **parameters
            )
        )
    if "seasonal_load_profile" in feature_groups:
        parameters = dict(feature_groups["seasonal_load_profile"] or {})
        blocks.append(
            build_seasonal_load_features(history, target, origin, **parameters)
        )
    if "seasonal_load_adjustment" in feature_groups:
        parameters = dict(feature_groups["seasonal_load_adjustment"] or {})
        blocks.append(
            build_seasonal_load_adjustment_features(
                history, target, origin, **parameters
            )
        )
    if "annual_load_anchors" in feature_groups:
        parameters = dict(feature_groups["annual_load_anchors"] or {})
        blocks.append(
            build_annual_load_anchor_features(
                history, target, origin, **parameters
            )
        )
    if "temperature_climatology" in feature_groups:
        parameters = dict(feature_groups["temperature_climatology"] or {})
        blocks.append(
            build_temperature_climatology_features(
                history if weather_history is None else weather_history,
                target,
                origin,
                **parameters,
            )
        )
    if "recent_temperature" in feature_groups:
        parameters = dict(feature_groups["recent_temperature"] or {})
        blocks.append(
            build_recent_temperature_features(
                history if weather_history is None else weather_history,
                target,
                origin,
                **parameters,
            )
        )

    features = pd.concat(blocks, axis=1)
    duplicate_columns = features.columns[features.columns.duplicated()].tolist()
    if duplicate_columns:
        raise ValueError(
            f"Feature groups produced duplicate columns {duplicate_columns}"
        )
    if not features.index.equals(target.index):
        raise AssertionError("Feature construction changed target-row alignment")
    return features


def categorical_feature_names(
    feature_groups: Mapping[str, Mapping[str, Any] | None],
) -> tuple[str, ...]:
    """Return semantic categorical columns present in a configured feature set."""

    unknown = sorted(set(feature_groups) - set(SUPPORTED_FEATURE_GROUPS))
    if unknown:
        raise ValueError(f"Unknown feature groups {unknown}")
    return TARGET_CATEGORICAL_FEATURES if "target_time" in feature_groups else ()
