"""Deterministic target calendar and forecast-horizon features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


TARGET_TIME_FEATURES = (
    "hour",
    "day_of_week",
    "hour_of_week",
    "month",
    "is_weekend",
    "hour_sin",
    "hour_cos",
    "seasonal_day_sin",
    "seasonal_day_cos",
    "horizon_hours",
    "forecast_week",
)

TARGET_CATEGORICAL_FEATURES = (
    "hour",
    "day_of_week",
    "hour_of_week",
    "month",
    "is_weekend",
    "forecast_week",
)


def build_target_time_features(
    target: pd.DataFrame, origin: str | pd.Timestamp
) -> pd.DataFrame:
    """Build deterministic calendar and horizon features for one target month.

    ``origin`` is the first operating interval of the forecast month. The
    returned rows retain ``target``'s index and use only ``period_start``;
    realized target load and weather columns are never read.

    The seasonal encoding maps every month/day to the fixed leap year 2000.
    This keeps the same calendar date aligned across leap and non-leap years
    while preserving February 29 as a distinct seasonal position.
    """

    required = {"zone_id", "period_start"}
    missing = sorted(required - set(target.columns))
    if missing:
        raise ValueError(f"Target frame is missing columns {missing}")
    if target.empty:
        raise ValueError("Target frame must contain at least one row")
    if not target.index.is_unique:
        raise ValueError("Target frame index must be unique for feature alignment")
    if not is_datetime64_any_dtype(target["period_start"]):
        raise TypeError("period_start must have a pandas datetime64 dtype")
    if target["period_start"].isna().any():
        raise ValueError("period_start must not contain missing values")
    if target.duplicated(["zone_id", "period_start"]).any():
        raise ValueError("Target rows must be unique by zone and operating hour")

    forecast_origin = pd.Timestamp(origin)
    if forecast_origin.tz is not None:
        raise ValueError("origin must be timezone-naive")
    if forecast_origin != forecast_origin.normalize() or forecast_origin.day != 1:
        raise ValueError("origin must be midnight on the first day of a month")

    periods = target["period_start"]
    if periods.dt.tz is not None:
        raise ValueError("period_start must be timezone-naive")
    month_end = forecast_origin + pd.offsets.MonthBegin(1)
    if not periods.ge(forecast_origin).all() or not periods.lt(month_end).all():
        raise ValueError("Every target period_start must lie inside the origin month")
    if not (
        periods.dt.minute.eq(0)
        & periods.dt.second.eq(0)
        & periods.dt.microsecond.eq(0)
    ).all():
        raise ValueError("Target period_start values must lie on exact hourly boundaries")

    hour = periods.dt.hour.to_numpy()
    day_of_week = periods.dt.dayofweek.to_numpy()
    horizon_hours = ((periods - forecast_origin) / pd.Timedelta(hours=1)).to_numpy(
        dtype=np.int16
    )

    canonical_dates = pd.to_datetime(
        "2000-" + periods.dt.strftime("%m-%d"), format="%Y-%m-%d", errors="raise"
    )
    seasonal_day = canonical_dates.dt.dayofyear.to_numpy() - 1
    hour_angle = 2.0 * np.pi * hour / 24.0
    seasonal_angle = 2.0 * np.pi * seasonal_day / 366.0

    features = pd.DataFrame(index=target.index)
    features["hour"] = hour.astype(np.int8)
    features["day_of_week"] = day_of_week.astype(np.int8)
    features["hour_of_week"] = (24 * day_of_week + hour).astype(np.int16)
    features["month"] = periods.dt.month.to_numpy(dtype=np.int8)
    features["is_weekend"] = (day_of_week >= 5).astype(np.int8)
    features["hour_sin"] = np.sin(hour_angle)
    features["hour_cos"] = np.cos(hour_angle)
    features["seasonal_day_sin"] = np.sin(seasonal_angle)
    features["seasonal_day_cos"] = np.cos(seasonal_angle)
    features["horizon_hours"] = horizon_hours
    features["forecast_week"] = (horizon_hours // (7 * 24)).astype(np.int8)
    return features.loc[:, list(TARGET_TIME_FEATURES)]
