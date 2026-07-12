"""Target-specific recent load-profile features."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ._shared import complete_hourly_window, validate_pre_origin_history


def build_recent_load_profile_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    mean_windows_weeks: Iterable[int] = (4, 12),
    std_windows_weeks: Iterable[int] = (12,),
    include_last_day: bool = True,
    include_last_hour_of_week: bool = True,
) -> pd.DataFrame:
    """Build recent load profiles matched to each target hour.

    The last-day feature matches operating hour. Hour-of-week features match
    both weekday and hour over complete pre-origin windows. All values are
    frozen at ``origin`` and may therefore repeat within the target month.
    """

    forecast_origin = validate_pre_origin_history(
        history, target, origin, value_columns=("load",)
    )
    if "period_start" not in target:
        raise ValueError("Target frame is missing column 'period_start'")
    if not is_datetime64_any_dtype(target["period_start"]):
        raise TypeError("Target period_start must have a pandas datetime64 dtype")
    if target["period_start"].isna().any():
        raise ValueError("Target period_start must not contain missing values")

    mean_windows = _positive_unique_weeks(mean_windows_weeks, "mean_windows_weeks")
    std_windows = _positive_unique_weeks(std_windows_weeks, "std_windows_weeks")
    if not (include_last_day or include_last_hour_of_week or mean_windows or std_windows):
        raise ValueError("recent_load_profile configuration must produce a feature")

    feature_names: list[str] = []
    if include_last_day:
        feature_names.append("load_last_day_same_hour")
    if include_last_hour_of_week:
        feature_names.append("load_last_same_hour_of_week")
    feature_names += [f"load_how_mean_{weeks}w" for weeks in mean_windows]
    feature_names += [f"load_how_std_{weeks}w" for weeks in std_windows]

    target_hour = target["period_start"].dt.hour
    target_how = 24 * target["period_start"].dt.dayofweek + target_hour
    features = pd.DataFrame(index=target.index)
    for name in feature_names:
        features[name] = np.nan

    for zone_id in target["zone_id"].unique():
        zone_mask = target["zone_id"].eq(zone_id)
        zone_history = history.loc[history["zone_id"] == zone_id].sort_values(
            "period_start"
        )
        if zone_history.empty:
            raise ValueError(f"No pre-origin history for target zone {zone_id}")

        if include_last_day:
            last_day = complete_hourly_window(
                zone_history,
                forecast_origin - pd.Timedelta(days=1),
                forecast_origin,
            )
            by_hour = (
                last_day.set_index(last_day["period_start"].dt.hour)["load"]
                if last_day is not None
                else pd.Series(dtype=float)
            )
            features.loc[zone_mask, "load_last_day_same_hour"] = target_hour.loc[
                zone_mask
            ].map(by_hour)

        required_weeks = set(mean_windows) | set(std_windows)
        if include_last_hour_of_week:
            required_weeks.add(1)
        profiles: dict[int, pd.DataFrame | None] = {}
        for weeks in required_weeks:
            window = complete_hourly_window(
                zone_history,
                forecast_origin - pd.Timedelta(weeks=weeks),
                forecast_origin,
            )
            if window is None:
                profiles[weeks] = None
                continue
            with_how = window.assign(
                hour_of_week=(
                    24 * window["period_start"].dt.dayofweek
                    + window["period_start"].dt.hour
                )
            )
            profiles[weeks] = with_how.groupby("hour_of_week")["load"].agg(
                mean="mean", std=lambda values: values.std(ddof=0), last="last"
            )

        if include_last_hour_of_week:
            profile = profiles[1]
            if profile is not None:
                features.loc[zone_mask, "load_last_same_hour_of_week"] = target_how.loc[
                    zone_mask
                ].map(profile["last"])
        for weeks in mean_windows:
            profile = profiles[weeks]
            if profile is not None:
                features.loc[zone_mask, f"load_how_mean_{weeks}w"] = target_how.loc[
                    zone_mask
                ].map(profile["mean"])
        for weeks in std_windows:
            profile = profiles[weeks]
            if profile is not None:
                features.loc[zone_mask, f"load_how_std_{weeks}w"] = target_how.loc[
                    zone_mask
                ].map(profile["std"])

    return features.loc[:, feature_names]


def _positive_unique_weeks(values: Iterable[int], name: str) -> tuple[int, ...]:
    """Normalize a configured collection of positive week-window lengths."""

    weeks = tuple(int(value) for value in values)
    if any(value <= 0 for value in weeks):
        raise ValueError(f"{name} must contain only positive integers")
    if len(weeks) != len(set(weeks)):
        raise ValueError(f"{name} must not contain duplicate windows")
    return weeks
