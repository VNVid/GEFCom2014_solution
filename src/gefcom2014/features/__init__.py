"""Composable leakage-safe feature blocks for forecasting models."""

from .adjustment import build_seasonal_load_adjustment_features
from .annual import build_annual_load_anchor_features
from .builder import build_feature_matrix, categorical_feature_names
from .derived import build_horizon_decay_features
from .holiday import (
    HOLIDAY_CATEGORICAL_FEATURES,
    HOLIDAY_FEATURES,
    build_holiday_features,
)
from .load import build_recent_load_features
from .profiles import build_recent_load_profile_features
from .seasonal import build_seasonal_load_features
from .temperature import (
    build_recent_temperature_features,
    build_temperature_climatology_features,
)
from .target import (
    TARGET_CATEGORICAL_FEATURES,
    TARGET_TIME_FEATURES,
    build_target_time_features,
)

__all__ = [
    "TARGET_CATEGORICAL_FEATURES",
    "TARGET_TIME_FEATURES",
    "HOLIDAY_CATEGORICAL_FEATURES",
    "HOLIDAY_FEATURES",
    "build_annual_load_anchor_features",
    "build_feature_matrix",
    "build_holiday_features",
    "build_horizon_decay_features",
    "build_recent_load_features",
    "build_recent_load_profile_features",
    "build_recent_temperature_features",
    "build_seasonal_load_adjustment_features",
    "build_seasonal_load_features",
    "build_temperature_climatology_features",
    "build_target_time_features",
    "categorical_feature_names",
]
