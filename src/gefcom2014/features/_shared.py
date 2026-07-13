"""Internal validation and window utilities shared by historical features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def validate_pre_origin_history(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    value_columns: Iterable[str],
) -> pd.Timestamp:
    """Validate the common leakage boundary and return normalized origin."""

    values = tuple(value_columns)
    required_history = {"zone_id", "period_start", *values}
    missing_history = sorted(required_history - set(history.columns))
    if missing_history:
        raise ValueError(f"History frame is missing columns {missing_history}")
    if "zone_id" not in target:
        raise ValueError("Target frame is missing column 'zone_id'")
    if target.empty:
        raise ValueError("Target frame must contain at least one row")
    if not target.index.is_unique:
        raise ValueError("Target frame index must be unique for feature alignment")
    if not is_datetime64_any_dtype(history["period_start"]):
        raise TypeError("History period_start must have a pandas datetime64 dtype")
    if history["period_start"].dt.tz is not None:
        raise ValueError("History period_start must be timezone-naive")
    if history["period_start"].isna().any() or history[list(values)].isna().any().any():
        raise ValueError("History timestamps and required values must not be missing")
    if history.duplicated(["zone_id", "period_start"]).any():
        raise ValueError("History rows must be unique by zone and operating hour")
    if not np.isfinite(history[list(values)].to_numpy(dtype=float)).all():
        raise ValueError("History feature values must contain only finite numbers")

    forecast_origin = pd.Timestamp(origin)
    if forecast_origin.tz is not None:
        raise ValueError("origin must be timezone-naive")
    if not history.empty and history["period_start"].max() >= forecast_origin:
        raise ValueError("History must contain only observations strictly before origin")
    return forecast_origin


def complete_hourly_window(
    zone_history: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame | None:
    """Return a sorted half-open window only when every expected hour exists."""

    selected = zone_history.loc[
        zone_history["period_start"].ge(start)
        & zone_history["period_start"].lt(end)
    ].sort_values("period_start")
    expected = pd.date_range(start, end, freq="h", inclusive="left")
    actual = pd.DatetimeIndex(selected["period_start"])
    return selected if actual.equals(expected) else None
