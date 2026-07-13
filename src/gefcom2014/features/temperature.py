"""Forecast-time-safe temperature climatology for target hours."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ..analogues import previous_seasonal_samples
from ..data import WEATHER_COLUMNS
from ._shared import complete_hourly_window, validate_pre_origin_history


def build_temperature_climatology_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    seasonal_window_days: int = 15,
    statistics: Iterable[str] = (
        "mean",
        "min_station",
        "max_station",
        "station_std",
        "temporal_std",
    ),
    quantiles: Iterable[float] = (),
    degree_thresholds: Iterable[float] = (),
    station_columns: Iterable[str] = WEATHER_COLUMNS,
) -> pd.DataFrame:
    """Summarize expected weather from completed prior seasonal cycles.

    Analogues match target zone and operating hour within a calendar window;
    weekdays are deliberately not filtered because weather does not follow the
    working-week pattern. First, each station is averaged over the historical
    analogue rows. Aggregate statistics then describe expected temperature and
    its spatial spread. ``temporal_std`` instead measures variation over the
    analogue rows after averaging stations, providing a weather-uncertainty
    signal without observing the target month's realized temperatures.
    Optional quantiles summarize the historical distribution of station-mean
    temperature, while degree transforms expose its nonlinear heating/cooling
    relationship with load.
    """

    stations = _station_names(station_columns)
    validate_pre_origin_history(
        history, target, origin, value_columns=stations
    )
    if "period_start" not in target:
        raise ValueError("Target frame is missing column 'period_start'")
    if not is_datetime64_any_dtype(target["period_start"]):
        raise TypeError("Target period_start must have a pandas datetime64 dtype")
    if target["period_start"].isna().any():
        raise ValueError("Target timestamps must not be missing")
    if target["period_start"].dt.tz is not None:
        raise ValueError("Target period_start must be timezone-naive")
    if not isinstance(seasonal_window_days, int) or seasonal_window_days < 0:
        raise ValueError("seasonal_window_days must be a non-negative integer")

    requested = tuple(str(statistic) for statistic in statistics)
    allowed = {"mean", "min_station", "max_station", "station_std", "temporal_std"}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ValueError(f"Unknown temperature climatology statistics {unknown}")
    if len(requested) != len(set(requested)):
        raise ValueError(
            "temperature climatology statistics must not contain duplicates"
        )

    raw_quantiles = tuple(quantiles)
    levels = np.asarray(raw_quantiles, dtype=float)
    if levels.ndim != 1 or not np.isfinite(levels).all():
        raise ValueError("Temperature quantiles must be a finite sequence")
    if levels.size and (
        not np.all((levels > 0) & (levels < 1))
        or not np.all(np.diff(levels) > 0)
    ):
        raise ValueError("Temperature quantiles must increase strictly inside (0, 1)")
    percentiles = np.rint(100 * levels).astype(int)
    if levels.size and not np.allclose(100 * levels, percentiles):
        raise ValueError("Temperature quantiles must be whole percentiles")
    if len(percentiles) != len(set(percentiles)):
        raise ValueError("Temperature quantiles produce duplicate feature names")

    thresholds = tuple(float(value) for value in degree_thresholds)
    if not np.isfinite(thresholds).all():
        raise ValueError("Temperature degree thresholds must be finite")
    threshold_labels = tuple(_number_label(value) for value in thresholds)
    if len(threshold_labels) != len(set(threshold_labels)):
        raise ValueError("Temperature degree thresholds produce duplicate names")
    if not requested and not raw_quantiles and not thresholds:
        raise ValueError("temperature_climatology configuration must produce a feature")

    samples_by_target = previous_seasonal_samples(
        history,
        target,
        value_columns=stations,
        window_days=seasonal_window_days,
        day_match="none",
    )
    value_names = [*requested, *(f"q{value:02d}" for value in percentiles)]
    for threshold in threshold_labels:
        value_names.extend((f"hdd{threshold}", f"cdd{threshold}"))
    feature_values: dict[str, list[float]] = {name: [] for name in value_names}
    for samples in samples_by_target:
        if samples.size == 0:
            for name in value_names:
                feature_values[name].append(np.nan)
            continue

        station_climatology = np.mean(samples, axis=0)
        temporal_means = np.mean(samples, axis=1)
        expected_mean = float(np.mean(station_climatology))
        values = {
            "mean": expected_mean,
            "min_station": float(np.min(station_climatology)),
            "max_station": float(np.max(station_climatology)),
            "station_std": float(np.std(station_climatology, ddof=0)),
            "temporal_std": float(np.std(temporal_means, ddof=0)),
        }
        for name in requested:
            feature_values[name].append(values[name])
        for level, percentile in zip(levels, percentiles):
            feature_values[f"q{percentile:02d}"].append(
                float(np.quantile(temporal_means, level, method="linear"))
            )
        for raw_threshold, label in zip(thresholds, threshold_labels):
            feature_values[f"hdd{label}"].append(
                max(raw_threshold - expected_mean, 0.0)
            )
            feature_values[f"cdd{label}"].append(
                max(expected_mean - raw_threshold, 0.0)
            )

    prefix = f"temperature_clim_{seasonal_window_days}d"
    return pd.DataFrame(
        {
            f"{prefix}_{name}": np.asarray(feature_values[name], dtype=float)
            for name in value_names
        },
        index=target.index,
    )


def _number_label(value: float) -> str:
    """Create a stable feature-name component for a configured threshold."""

    return f"{value:g}".replace("-", "m").replace(".", "p")


def build_recent_temperature_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    mean_windows_days: Iterable[int] = (1, 7, 28),
    std_windows_days: Iterable[int] = (7,),
    anomaly_window_days: int | None = 7,
    anomaly_seasonal_window_days: int = 15,
    station_columns: Iterable[str] = WEATHER_COLUMNS,
) -> pd.DataFrame:
    """Summarize the observed weather regime immediately before ``origin``.

    Each hourly temperature is first averaged across stations. Fixed recent
    windows must be complete; otherwise their statistics remain ``NaN``. The
    optional anomaly compares the recent observed mean with seasonal estimates
    for the same recent hours, built only from completed earlier annual cycles.
    All outputs are frozen at the monthly origin and repeat across target rows
    belonging to the same zone.
    """

    stations = _station_names(station_columns)
    forecast_origin = validate_pre_origin_history(
        history, target, origin, value_columns=stations
    )
    mean_windows = _positive_unique_days(mean_windows_days, "mean_windows_days")
    std_windows = _positive_unique_days(std_windows_days, "std_windows_days")
    if anomaly_window_days is not None:
        if not isinstance(anomaly_window_days, int) or anomaly_window_days <= 0:
            raise ValueError("anomaly_window_days must be a positive integer or null")
        if (
            not isinstance(anomaly_seasonal_window_days, int)
            or anomaly_seasonal_window_days < 0
        ):
            raise ValueError(
                "anomaly_seasonal_window_days must be a non-negative integer"
            )
    if not mean_windows and not std_windows and anomaly_window_days is None:
        raise ValueError("recent_temperature configuration must produce a feature")

    feature_names = [f"temperature_recent_mean_{days}d" for days in mean_windows]
    feature_names += [f"temperature_recent_std_{days}d" for days in std_windows]
    if anomaly_window_days is not None:
        feature_names.append(f"temperature_recent_anomaly_{anomaly_window_days}d")

    zone_records: dict[object, dict[str, float]] = {}
    for zone_id in target["zone_id"].unique():
        zone_history = history.loc[history["zone_id"] == zone_id].sort_values(
            "period_start"
        )
        if zone_history.empty:
            raise ValueError(f"No pre-origin weather history for target zone {zone_id}")

        windows: dict[int, pd.DataFrame | None] = {}

        def recent_window(days: int) -> pd.DataFrame | None:
            if days not in windows:
                windows[days] = complete_hourly_window(
                    zone_history,
                    forecast_origin - pd.Timedelta(days=days),
                    forecast_origin,
                )
            return windows[days]

        record: dict[str, float] = {}
        for days in mean_windows:
            window = recent_window(days)
            record[f"temperature_recent_mean_{days}d"] = (
                float(window.loc[:, stations].to_numpy(dtype=float).mean())
                if window is not None
                else np.nan
            )
        for days in std_windows:
            window = recent_window(days)
            if window is None:
                value = np.nan
            else:
                hourly_mean = window.loc[:, stations].mean(axis=1).to_numpy(dtype=float)
                value = float(np.std(hourly_mean, ddof=0))
            record[f"temperature_recent_std_{days}d"] = value

        if anomaly_window_days is not None:
            window = recent_window(anomaly_window_days)
            anomaly = np.nan
            if window is not None:
                recent_start = forecast_origin - pd.Timedelta(
                    days=anomaly_window_days
                )
                earlier_cycles = zone_history.loc[
                    zone_history["period_start"] < recent_start
                ]
                samples_by_hour = previous_seasonal_samples(
                    earlier_cycles,
                    window.loc[:, ["zone_id", "period_start"]],
                    value_columns=stations,
                    window_days=anomaly_seasonal_window_days,
                    day_match="none",
                )
                expected = np.asarray(
                    [
                        float(np.mean(samples)) if samples.size else np.nan
                        for samples in samples_by_hour
                    ]
                )
                if np.isfinite(expected).all():
                    anomaly = float(
                        window.loc[:, stations].to_numpy(dtype=float).mean()
                        - expected.mean()
                    )
            record[f"temperature_recent_anomaly_{anomaly_window_days}d"] = anomaly
        zone_records[zone_id] = record

    features = pd.DataFrame(index=target.index)
    for name in feature_names:
        values_by_zone = {
            zone_id: record[name] for zone_id, record in zone_records.items()
        }
        features[name] = target["zone_id"].map(values_by_zone).to_numpy(dtype=float)
    return features


def _station_names(values: Iterable[str]) -> tuple[str, ...]:
    """Normalize the weather columns used by a temperature feature block."""

    stations = tuple(str(column) for column in values)
    if not stations:
        raise ValueError("station_columns must contain at least one column")
    if len(stations) != len(set(stations)):
        raise ValueError("station_columns must not contain duplicates")
    return stations


def _positive_unique_days(values: Iterable[int], name: str) -> tuple[int, ...]:
    """Normalize configured positive temperature-window lengths."""

    days = tuple(int(value) for value in values)
    if any(value <= 0 for value in days):
        raise ValueError(f"{name} must contain only positive integers")
    if len(days) != len(set(days)):
        raise ValueError(f"{name} must not contain duplicate windows")
    return days
