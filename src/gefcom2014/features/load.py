"""Load-derived feature blocks frozen at a monthly forecast origin."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ._shared import complete_hourly_window, validate_pre_origin_history


def build_recent_load_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    mean_windows_days: Iterable[int] = (7, 28, 365),
    std_windows_days: Iterable[int] = (28,),
    difference_windows_days: Iterable[Iterable[int]] = ((7, 28),),
    trend_windows_days: Iterable[int] = (28,),
    yoy_windows_days: Iterable[int] = (28, 365),
) -> pd.DataFrame:
    """Build compact recent load level and trend features for every target row.

    Windows are half-open intervals ending at ``origin`` and must contain a
    complete hourly history for a zone. Incomplete windows produce ``NaN``
    rather than a statistic based on silently shortened history. Year-over-year
    features compare a recent window with the same window ending one calendar
    year before the origin.
    """

    forecast_origin = validate_pre_origin_history(
        history, target, origin, value_columns=("load",)
    )

    mean_windows = _positive_unique_days(mean_windows_days, "mean_windows_days")
    std_windows = _positive_unique_days(std_windows_days, "std_windows_days")
    trend_windows = _positive_unique_days(trend_windows_days, "trend_windows_days")
    if any(days < 2 for days in trend_windows):
        raise ValueError("trend_windows_days must contain windows of at least two days")
    yoy_windows = _positive_unique_days(yoy_windows_days, "yoy_windows_days")
    difference_windows = tuple(tuple(int(day) for day in pair) for pair in difference_windows_days)
    if any(len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0 for pair in difference_windows):
        raise ValueError("difference_windows_days must contain pairs of positive integers")

    feature_names = [f"load_mean_{days}d" for days in mean_windows]
    feature_names += [f"load_std_{days}d" for days in std_windows]
    feature_names += [
        f"load_mean_{short}d_minus_{long}d" for short, long in difference_windows
    ]
    feature_names += [f"load_daily_slope_{days}d" for days in trend_windows]
    feature_names += [f"load_yoy_ratio_{days}d" for days in yoy_windows]
    if not feature_names:
        raise ValueError("recent_load configuration must produce at least one feature")

    zone_records: dict[object, dict[str, float]] = {}
    for zone_id in target["zone_id"].unique():
        zone_history = history.loc[history["zone_id"] == zone_id].sort_values(
            "period_start"
        )
        if zone_history.empty:
            raise ValueError(f"No pre-origin history for target zone {zone_id}")

        window_cache: dict[tuple[pd.Timestamp, pd.Timestamp], np.ndarray | None] = {}

        def values_between(start: pd.Timestamp, end: pd.Timestamp) -> np.ndarray | None:
            key = (start, end)
            if key not in window_cache:
                selected = complete_hourly_window(zone_history, start, end)
                window_cache[key] = (
                    selected["load"].to_numpy(dtype=float)
                    if selected is not None
                    else None
                )
            return window_cache[key]

        def recent_values(days: int) -> np.ndarray | None:
            return values_between(
                forecast_origin - pd.Timedelta(days=days), forecast_origin
            )

        means: dict[int, float] = {}
        all_mean_windows = set(mean_windows)
        all_mean_windows.update(day for pair in difference_windows for day in pair)
        for days in all_mean_windows:
            values = recent_values(days)
            means[days] = float(np.mean(values)) if values is not None else np.nan

        record: dict[str, float] = {
            f"load_mean_{days}d": means[days] for days in mean_windows
        }
        for days in std_windows:
            values = recent_values(days)
            record[f"load_std_{days}d"] = (
                float(np.std(values, ddof=0)) if values is not None else np.nan
            )
        for short, long in difference_windows:
            record[f"load_mean_{short}d_minus_{long}d"] = means[short] - means[long]
        for days in trend_windows:
            values = recent_values(days)
            if values is None:
                slope = np.nan
            else:
                daily_means = values.reshape(days, 24).mean(axis=1)
                slope = float(np.polyfit(np.arange(days), daily_means, deg=1)[0])
            record[f"load_daily_slope_{days}d"] = slope
        for days in yoy_windows:
            current = recent_values(days)
            reference_end = forecast_origin - pd.DateOffset(years=1)
            reference = values_between(
                reference_end - pd.Timedelta(days=days), reference_end
            )
            if current is None or reference is None:
                ratio = np.nan
            else:
                reference_mean = float(np.mean(reference))
                ratio = (
                    float(np.mean(current)) / reference_mean
                    if reference_mean != 0.0
                    else np.nan
                )
            record[f"load_yoy_ratio_{days}d"] = ratio
        zone_records[zone_id] = record

    features = pd.DataFrame(index=target.index)
    for name in feature_names:
        by_zone = {zone_id: record[name] for zone_id, record in zone_records.items()}
        features[name] = target["zone_id"].map(by_zone).to_numpy(dtype=float)
    return features


def _positive_unique_days(values: Iterable[int], name: str) -> tuple[int, ...]:
    """Normalize a configured collection of positive day-window lengths."""

    days = tuple(int(value) for value in values)
    if any(value <= 0 for value in days):
        raise ValueError(f"{name} must contain only positive integers")
    if len(days) != len(set(days)):
        raise ValueError(f"{name} must not contain duplicate windows")
    return days
