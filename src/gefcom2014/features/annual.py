"""Previous-year load anchors available at a monthly forecast origin."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ..analogues import load_values_at_periods
from ._shared import validate_pre_origin_history


def build_annual_load_anchor_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    calendar_year_lags: Iterable[int] = (1,),
    fixed_day_lags: Iterable[int] = (364,),
    aggregate_statistics: Iterable[str] = (),
) -> pd.DataFrame:
    """Build calendar-aligned and weekday-aligned annual load anchors.

    A calendar-year lag preserves month, day, and hour but can change weekday.
    A 364-day lag is exactly 52 weeks, so it preserves weekday and hour while
    allowing the calendar date to shift. Missing reference observations remain
    ``NaN`` rather than being replaced with a different historical value.
    """

    validate_pre_origin_history(history, target, origin, value_columns=("load",))
    if "period_start" not in target:
        raise ValueError("Target frame is missing column 'period_start'")
    if not is_datetime64_any_dtype(target["period_start"]):
        raise TypeError("Target period_start must have a pandas datetime64 dtype")
    if target["period_start"].isna().any():
        raise ValueError("Target timestamps must not be missing")
    if target["period_start"].dt.tz is not None:
        raise ValueError("Target period_start must be timezone-naive")

    year_lags = _positive_unique_lags(calendar_year_lags, "calendar_year_lags")
    day_lags = _positive_unique_lags(fixed_day_lags, "fixed_day_lags")
    statistics = tuple(str(value) for value in aggregate_statistics)
    unknown_statistics = sorted(set(statistics) - {"mean", "std"})
    if unknown_statistics:
        raise ValueError(f"Unknown annual aggregate statistics {unknown_statistics}")
    if len(statistics) != len(set(statistics)):
        raise ValueError("aggregate_statistics must not contain duplicates")
    if not year_lags and not day_lags:
        raise ValueError("annual_load_anchors configuration must produce a feature")

    features = pd.DataFrame(index=target.index)
    anchor_names: list[str] = []
    for years in year_lags:
        references = target["period_start"] - pd.DateOffset(years=years)
        name = f"load_lag_calendar_{years}y"
        features[name] = load_values_at_periods(
            history, target, references
        )
        anchor_names.append(name)
    for days in day_lags:
        references = target["period_start"] - pd.Timedelta(days=days)
        name = f"load_lag_{days}d"
        features[name] = load_values_at_periods(
            history, target, references
        )
        anchor_names.append(name)

    anchors = features.loc[:, anchor_names].to_numpy(dtype=float)
    for statistic in statistics:
        if statistic == "mean":
            values = np.mean(anchors, axis=1)
        else:
            values = np.std(anchors, axis=1, ddof=0)
        features[f"load_annual_anchor_{statistic}"] = values
    return features


def _positive_unique_lags(values: Iterable[int], name: str) -> tuple[int, ...]:
    """Normalize configured positive annual lag lengths."""

    lags = tuple(int(value) for value in values)
    if any(value <= 0 for value in lags):
        raise ValueError(f"{name} must contain only positive integers")
    if len(lags) != len(set(lags)):
        raise ValueError(f"{name} must not contain duplicate lags")
    return lags
