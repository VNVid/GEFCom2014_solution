"""Descriptive analyses used by the EDA report and figures.

Each domain has one entry point: load quality, weather relationships, and
serial dependence. Small calculations stay inline so readers can follow an
analysis without jumping through layers of one-use helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from gefcom2014.data import (
    WEATHER_COLUMNS,
    actuals_for_round,
    discover_round_files,
    interval_start,
    load_complete_history,
    read_benchmark_file,
    read_train_file,
)


DAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


@dataclass(frozen=True)
class WeatherResults:
    """Weather tables plus matrices needed directly by figures and the summary."""

    tables: dict[str, pd.DataFrame]
    station_correlations: pd.DataFrame
    response_by_hour: pd.DataFrame
    temperature_proxy: pd.DataFrame
    correlation_summary: dict[str, float]


@dataclass(frozen=True)
class EdaResults:
    """Complete non-plotting output of the EDA workflow."""

    history: pd.DataFrame
    observed: pd.DataFrame
    manifest: pd.DataFrame
    tables: dict[str, pd.DataFrame]
    summary: dict[str, Any]
    station_correlations: pd.DataFrame
    temperature_response_by_hour: pd.DataFrame


def analyze(load_dir: str | Path, config: dict[str, Any]) -> EdaResults:
    """Run the descriptive analyses and assemble their named artifacts."""

    analysis_config = config["analysis"]
    history = prepare_history(
        load_complete_history(load_dir, include_solution=True),
        weather_aggregate=analysis_config["weather_aggregate"],
    )
    observed = history.dropna(subset=["load"]).copy()

    load_tables = analyze_load(history, observed, analysis_config)
    weather = analyze_weather(load_dir, history, observed, analysis_config)
    dependence_tables = analyze_dependence(observed, analysis_config)
    forecast_periods = describe_forecast_periods(load_dir)

    manifest_export = forecast_periods.copy()
    for column in ["training_cutoff", "forecast_start", "forecast_end"]:
        manifest_export[column] = manifest_export[column].dt.strftime("%Y-%m-%d %H:%M:%S")

    tables = {
        **load_tables,
        "round_manifest.csv": manifest_export,
        **weather.tables,
        **dependence_tables,
    }
    summary = build_summary(history, observed, weather, forecast_periods, config)

    return EdaResults(
        history=history,
        observed=observed,
        manifest=forecast_periods,
        tables=tables,
        summary=summary,
        station_correlations=weather.station_correlations,
        temperature_response_by_hour=weather.response_by_hour,
    )


def prepare_history(history: pd.DataFrame, weather_aggregate: str) -> pd.DataFrame:
    """Add calendar and compact weather fields shared by all analyses."""

    frame = history.copy()
    if weather_aggregate == "mean":
        frame["temperature"] = frame[list(WEATHER_COLUMNS)].mean(axis=1)
    elif weather_aggregate == "median":
        frame["temperature"] = frame[list(WEATHER_COLUMNS)].median(axis=1)
    else:
        raise ValueError("weather_aggregate must be 'mean' or 'median'")

    frame["temperature_spread"] = frame[list(WEATHER_COLUMNS)].max(axis=1) - frame[
        list(WEATHER_COLUMNS)
    ].min(axis=1)
    frame["period_start"] = interval_start(frame["timestamp"])
    frame["date"] = frame["period_start"].dt.normalize()
    frame["year"] = frame["period_start"].dt.year
    frame["month"] = frame["period_start"].dt.month
    frame["month_name"] = pd.Categorical(
        frame["period_start"].dt.strftime("%b"), categories=MONTH_ORDER, ordered=True
    )
    frame["day_of_week"] = frame["period_start"].dt.dayofweek
    frame["day_name"] = pd.Categorical(
        frame["period_start"].dt.day_name(), categories=DAY_ORDER, ordered=True
    )
    frame["hour"] = frame["period_start"].dt.hour
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["day_type"] = np.where(frame["is_weekend"], "Weekend", "Weekday")
    frame["hour_block"] = pd.cut(
        frame["hour"],
        bins=[-1, 5, 9, 15, 19, 23],
        labels=[
            "Night (00-05)",
            "Morning (06-09)",
            "Day (10-15)",
            "Evening (16-19)",
            "Late (20-23)",
        ],
    )
    return frame


def describe_forecast_periods(load_dir: str | Path) -> pd.DataFrame:
    """Describe forecast-window coverage without evaluating any forecasts."""

    available_load_rows = 0
    records: list[dict[str, Any]] = []
    for item in discover_round_files(load_dir):
        released = read_train_file(item.train_path, item.task_id)
        forecast = read_benchmark_file(item.benchmark_path, item.task_id)
        available_load_rows += int(released["load"].notna().sum())
        records.append(
            {
                "task_id": item.task_id,
                "training_cutoff": released["timestamp"].max(),
                "forecast_start": forecast["timestamp"].min(),
                "forecast_end": forecast["timestamp"].max(),
                "horizon_hours": len(forecast),
                "available_load_rows": available_load_rows,
            }
        )

    periods = pd.DataFrame(records)
    origin_gaps = (
        periods["forecast_start"] - periods["training_cutoff"]
    ).dt.total_seconds() / 3600
    if not (origin_gaps == 1).all():
        raise ValueError(
            f"Expected a one-hour step at every origin, observed {origin_gaps.tolist()}"
        )
    return periods


def analyze_load(
    history: pd.DataFrame, observed: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Describe data integrity, load distribution, drift, and extremes."""

    timestamp_steps = history["timestamp"].sort_values().diff().dropna()
    operating_day_counts = history["date"].value_counts()
    missing_load = history["load"].isna().to_numpy()
    missing_load_blocks = int(np.sum(missing_load & ~np.r_[False, missing_load[:-1]]))
    integrity = pd.DataFrame(
        [
            ("rows_total", len(history)),
            ("weather_start_hour_ending", history["timestamp"].min().isoformat()),
            ("weather_end_hour_ending", history["timestamp"].max().isoformat()),
            ("load_start_hour_ending", observed["timestamp"].min().isoformat()),
            ("load_end_hour_ending", observed["timestamp"].max().isoformat()),
            ("observed_load_rows", len(observed)),
            ("missing_load_rows", int(missing_load.sum())),
            ("missing_load_blocks", missing_load_blocks),
            ("missing_weather_values", int(history[list(WEATHER_COLUMNS)].isna().sum().sum())),
            ("duplicate_timestamps", int(history["timestamp"].duplicated().sum())),
            ("non_hourly_steps", int((timestamp_steps != pd.Timedelta(hours=1)).sum())),
            ("operating_days_not_24_hours", int((operating_day_counts != 24).sum())),
            ("zones", int(history["zone_id"].nunique())),
            ("weather_stations", len(WEATHER_COLUMNS)),
            ("nonpositive_load_rows", int((observed["load"] <= 0).sum())),
        ],
        columns=["check", "value"],
    )

    quantile_levels = [0.0, 0.001, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 0.999, 1.0]
    load_quantiles = observed["load"].quantile(quantile_levels)
    distribution = pd.DataFrame(
        [
            {"statistic": "count", "value": float(len(observed))},
            *[
                {"statistic": f"q{level:g}", "value": float(value)}
                for level, value in load_quantiles.items()
            ],
            {"statistic": "mean", "value": float(observed["load"].mean())},
            {"statistic": "std", "value": float(observed["load"].std())},
            {"statistic": "skew", "value": float(observed["load"].skew())},
        ]
    )

    years = (
        observed.groupby("year", observed=True)
        .agg(
            hours=("load", "size"),
            load_mean=("load", "mean"),
            load_std=("load", "std"),
            load_min=("load", "min"),
            load_q05=("load", lambda values: values.quantile(0.05)),
            load_median=("load", "median"),
            load_q95=("load", lambda values: values.quantile(0.95)),
            load_max=("load", "max"),
            temperature_mean=("temperature", "mean"),
        )
        .reset_index()
    )

    lower_q, upper_q = (float(value) for value in config["extreme_quantiles"])
    lower_load = observed["load"].quantile(lower_q)
    upper_load = observed["load"].quantile(upper_q)
    extremes = observed.loc[
        (observed["load"] <= lower_load) | (observed["load"] >= upper_load),
        ["timestamp", "period_start", "load", "temperature", "day_name", "hour"],
    ].copy()
    extremes["tail"] = np.where(extremes["load"] <= lower_load, "low", "high")
    extremes = extremes.sort_values("load", ignore_index=True)

    ordered = observed.sort_values("timestamp").copy()
    ordered["load_change"] = ordered["load"].diff()
    ordered["absolute_load_change"] = ordered["load_change"].abs()
    changes = ordered.nlargest(20, "absolute_load_change")[
        ["timestamp", "period_start", "load", "load_change", "absolute_load_change", "temperature"]
    ].sort_values("absolute_load_change", ascending=False, ignore_index=True)

    return {
        "dataset_integrity.csv": integrity,
        "load_distribution.csv": distribution,
        "yearly_summary.csv": years,
        "extreme_load_hours.csv": extremes,
        "largest_hourly_load_changes.csv": changes,
    }


def analyze_weather(
    load_dir: str | Path,
    history: pd.DataFrame,
    observed: pd.DataFrame,
    config: dict[str, Any],
) -> WeatherResults:
    """Describe station quality, redundancy, and nonlinear load response."""

    cold = observed["temperature"] <= observed["temperature"].quantile(0.25)
    hot = observed["temperature"] >= observed["temperature"].quantile(0.75)
    station_records: list[dict[str, Any]] = []
    for station in WEATHER_COLUMNS:
        values = history[station]
        constant_runs = values.ne(values.shift()).cumsum()
        station_records.append(
            {
                "station": station,
                "count": int(values.count()),
                "missing": int(values.isna().sum()),
                "n_unique": int(values.nunique()),
                "mean": values.mean(),
                "std": values.std(),
                "min": values.min(),
                "q01": values.quantile(0.01),
                "q99": values.quantile(0.99),
                "max": values.max(),
                "longest_constant_run_hours": int(values.groupby(constant_runs).size().max()),
                "load_corr_all": observed[[station, "load"]].corr().iloc[0, 1],
                "load_corr_cold_quartile": observed.loc[cold, [station, "load"]].corr().iloc[0, 1],
                "load_corr_hot_quartile": observed.loc[hot, [station, "load"]].corr().iloc[0, 1],
            }
        )
    stations = pd.DataFrame(station_records)
    station_correlations = history[list(WEATHER_COLUMNS)].corr()

    temperature_bins = pd.cut(
        observed["temperature"], bins=int(config["temperature_bins"]), duplicates="drop"
    )
    binned = observed.assign(temperature_bin=temperature_bins)
    response = (
        binned.groupby("temperature_bin", observed=True)
        .agg(
            temperature_mean=("temperature", "mean"),
            count=("load", "size"),
            load_mean=("load", "mean"),
            load_q10=("load", lambda values: values.quantile(0.10)),
            load_median=("load", "median"),
            load_q90=("load", lambda values: values.quantile(0.90)),
        )
        .reset_index()
    )
    response["temperature_bin"] = response["temperature_bin"].astype("string")
    response_by_hour = (
        binned.groupby(["hour_block", "temperature_bin"], observed=True)
        .agg(
            temperature_mean=("temperature", "mean"),
            load_mean=("load", "mean"),
            count=("load", "size"),
        )
        .reset_index()
    )

    weather_history = history.set_index("timestamp")[list(WEATHER_COLUMNS)].sort_index()
    temperature_records: list[dict[str, Any]] = []
    for task_id in range(1, 16):
        actual = actuals_for_round(load_dir, task_id)
        target_weather = actual[list(WEATHER_COLUMNS)].to_numpy(dtype=float)
        previous_timestamps = actual["timestamp"] - pd.DateOffset(years=1)
        previous_weather = weather_history.reindex(
            pd.DatetimeIndex(previous_timestamps)
        ).to_numpy(dtype=float)
        if np.isnan(previous_weather).any():
            raise ValueError(
                f"Previous-year temperature proxy is incomplete for Task {task_id}"
            )

        target_temperature = target_weather.mean(axis=1)
        previous_temperature = previous_weather.mean(axis=1)
        residual = target_temperature - previous_temperature
        temperature_records.append(
            {
                "task_id": task_id,
                "aggregate_temperature_mae": np.mean(np.abs(residual)),
                "aggregate_temperature_rmse": np.sqrt(np.mean(np.square(residual))),
                "aggregate_temperature_bias": residual.mean(),
                "aggregate_temperature_correlation": np.corrcoef(
                    target_temperature, previous_temperature
                )[0, 1],
                "station_level_temperature_mae": np.mean(
                    np.abs(target_weather - previous_weather)
                ),
            }
        )
    temperature_proxy = pd.DataFrame(temperature_records)

    correlation_values = station_correlations.to_numpy(dtype=float)
    pairwise = correlation_values[np.triu_indices_from(correlation_values, k=1)]
    eigenvalues = np.linalg.eigvalsh(correlation_values)[::-1]
    explained_variance = eigenvalues / eigenvalues.sum()
    correlation_summary = {
        "median_pairwise_station_correlation": float(np.median(pairwise)),
        "min_pairwise_station_correlation": float(np.min(pairwise)),
        "max_pairwise_station_correlation": float(np.max(pairwise)),
        "weather_pc1_variance_share": float(explained_variance[0]),
        "weather_first_3_pc_variance_share": float(explained_variance[:3].sum()),
        "weather_first_5_pc_variance_share": float(explained_variance[:5].sum()),
    }
    return WeatherResults(
        tables={
            "weather_station_summary.csv": stations,
            "weather_station_correlations.csv": station_correlations.reset_index(names="station"),
            "temperature_response_bins.csv": response,
            "previous_year_temperature_proxy.csv": temperature_proxy,
        },
        station_correlations=station_correlations,
        response_by_hour=response_by_hour,
        temperature_proxy=temperature_proxy,
        correlation_summary=correlation_summary,
    )


def analyze_dependence(
    observed: pd.DataFrame, config: dict[str, Any]
) -> dict[str, pd.DataFrame]:
    """Calculate correlations and lagged-value errors for selected lags."""

    load = observed.set_index("timestamp")["load"].sort_index()
    lag_records: list[dict[str, Any]] = []
    for lag in [int(value) for value in config["selected_lags_hours"]]:
        prediction = load.shift(lag)
        valid = load.notna() & prediction.notna()
        residual = load[valid] - prediction[valid]
        lag_records.append(
            {
                "lag_hours": lag,
                "pairs": int(valid.sum()),
                "correlation": load[valid].corr(prediction[valid]),
                "mae": residual.abs().mean(),
                "rmse": np.sqrt(np.mean(np.square(residual))),
                "bias": residual.mean(),
            }
        )

    return {"lag_diagnostics.csv": pd.DataFrame(lag_records)}


def build_summary(
    history: pd.DataFrame,
    observed: pd.DataFrame,
    weather: WeatherResults,
    forecast_periods: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Collect the small set of headline values printed after an EDA run."""

    return {
        "weather_hours": int(len(history)),
        "observed_load_hours": int(len(observed)),
        "weather_only_hours": int(history["load"].isna().sum()),
        "load_start": observed["period_start"].min().isoformat(),
        "load_end": observed["period_start"].max().isoformat(),
        "load_mean": float(observed["load"].mean()),
        "load_std": float(observed["load"].std()),
        "load_min": float(observed["load"].min()),
        "load_max": float(observed["load"].max()),
        "temperature_mean": float(history["temperature"].mean()),
        "temperature_min_station_value": float(history[list(WEATHER_COLUMNS)].min().min()),
        "temperature_max_station_value": float(history[list(WEATHER_COLUMNS)].max().max()),
        "rounds": int(len(forecast_periods)),
        "forecast_hours": int(forecast_periods["horizon_hours"].sum()),
        "previous_year_temperature_proxy_mae_mean": float(
            weather.temperature_proxy["aggregate_temperature_mae"].mean()
        ),
        **weather.correlation_summary,
        "config": config,
    }
