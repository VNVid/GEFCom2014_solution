"""Figures describing load coverage, seasonality, and drift."""

from __future__ import annotations

from pathlib import Path

# This import must precede Matplotlib and seaborn imports.
from .plot_utils import plt, save_figure

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import seaborn as sns

from .analysis import DAY_ORDER, MONTH_ORDER


def plot_data_coverage(
    history: pd.DataFrame, manifest: pd.DataFrame, output: Path, dpi: int
) -> None:
    daily = (
        history.set_index("period_start")[["load", "temperature"]]
        .resample("D")
        .agg({"load": "mean", "temperature": "mean"})
    )
    forecast_start = manifest["forecast_start"].min()
    forecast_end = manifest["forecast_end"].max()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"hspace": 0.12})
    axes[0].plot(daily.index, daily["temperature"], color="#4C78A8", linewidth=0.65)
    axes[0].set_ylabel("Mean of 25 stations")
    axes[0].set_title("Weather covers four years before load labels")
    axes[0].axvspan(
        forecast_start,
        forecast_end,
        color="#F58518",
        alpha=0.13,
        label="15 forecast rounds",
    )
    axes[0].legend(loc="upper right")

    axes[1].plot(daily.index, daily["load"], color="#E45756", linewidth=0.75)
    axes[1].set_ylabel("Daily mean load")
    axes[1].set_xlabel("Operating date (official timestamp shifted back one hour)")
    axes[1].set_title("Observed load history and rolling-origin evaluation window")
    axes[1].axvspan(forecast_start, forecast_end, color="#F58518", alpha=0.13)
    for start in manifest["forecast_start"]:
        axes[1].axvline(start, color="#F58518", alpha=0.25, linewidth=0.5)
    axes[1].xaxis.set_major_locator(mdates.YearLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    save_figure(fig, output, dpi)


def plot_load_overview(observed: pd.DataFrame, output: Path, dpi: int) -> None:
    daily = observed.set_index("period_start")["load"].resample("D").agg(["min", "mean", "max"])
    rolling = daily["mean"].rolling(30, center=True, min_periods=15).mean()

    fig = plt.figure(figsize=(14, 8))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.34, wspace=0.25)
    ax_time = fig.add_subplot(grid[0, :])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_year = fig.add_subplot(grid[1, 1])

    ax_time.fill_between(
        daily.index,
        daily["min"],
        daily["max"],
        color="#9D755D",
        alpha=0.16,
        linewidth=0,
    )
    ax_time.plot(
        daily.index,
        daily["mean"],
        color="#9D755D",
        linewidth=0.45,
        alpha=0.65,
        label="Daily mean",
    )
    ax_time.plot(
        rolling.index,
        rolling,
        color="#E45756",
        linewidth=1.5,
        label="Centered 30-day mean",
    )
    ax_time.set_title("Load has strong yearly seasonality and changing peak intensity")
    ax_time.set_ylabel("Load")
    ax_time.legend(ncol=2, loc="upper left")

    sns.histplot(observed["load"], bins=60, stat="density", kde=True, ax=ax_hist, color="#4C78A8")
    ax_hist.set_title("Hourly load distribution")
    ax_hist.set_xlabel("Load")

    sns.boxplot(data=observed, x="year", y="load", showfliers=False, color="#72B7B2", ax=ax_year)
    ax_year.set_title("Distribution drift by operating year")
    ax_year.set_xlabel("Year")
    ax_year.tick_params(axis="x", rotation=30)
    save_figure(fig, output, dpi)


def plot_calendar_seasonality(observed: pd.DataFrame, output: Path, dpi: int) -> None:
    hourly_type = observed.groupby(["hour", "day_type"], observed=True)["load"].mean().reset_index()
    day_hour = observed.pivot_table(
        index="day_name", columns="hour", values="load", aggfunc="mean", observed=False
    ).reindex(DAY_ORDER)
    month_stats = (
        observed.groupby("month_name", observed=True)["load"]
        .agg(mean="mean", q10=lambda x: x.quantile(0.1), q90=lambda x: x.quantile(0.9))
        .reindex(MONTH_ORDER)
    )
    month_hour = observed.pivot_table(
        index="month_name", columns="hour", values="load", aggfunc="mean", observed=False
    ).reindex(MONTH_ORDER)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    sns.lineplot(data=hourly_type, x="hour", y="load", hue="day_type", marker="o", ax=axes[0, 0])
    axes[0, 0].set_title("Weekday/weekend effects depend strongly on operating hour")
    axes[0, 0].set_xticks(range(0, 24, 3))
    axes[0, 0].set_ylabel("Mean load")

    sns.heatmap(day_hour, cmap="viridis", ax=axes[0, 1], cbar_kws={"label": "Mean load"})
    axes[0, 1].set_title("Hour-of-week structure")
    axes[0, 1].set_xlabel("Operating hour")
    axes[0, 1].set_ylabel("")

    x = np.arange(12)
    axes[1, 0].fill_between(
        x,
        month_stats["q10"],
        month_stats["q90"],
        alpha=0.2,
        color="#4C78A8",
        label="10th–90th percentile",
    )
    axes[1, 0].plot(x, month_stats["mean"], marker="o", color="#4C78A8", label="Mean")
    axes[1, 0].set_xticks(x, MONTH_ORDER)
    axes[1, 0].set_title("Winter and summer both carry elevated demand")
    axes[1, 0].set_ylabel("Load")
    axes[1, 0].legend()

    sns.heatmap(month_hour, cmap="mako", ax=axes[1, 1], cbar_kws={"label": "Mean load"})
    axes[1, 1].set_title("Season and hour interact strongly")
    axes[1, 1].set_xlabel("Operating hour")
    axes[1, 1].set_ylabel("")
    save_figure(fig, output, dpi)


def plot_year_month_drift(observed: pd.DataFrame, output: Path, dpi: int) -> None:
    monthly = observed.groupby(["year", "month"], observed=True)["load"].mean().reset_index()
    yearly = (
        observed.groupby("year", observed=True)
        .agg(load_mean=("load", "mean"), temperature_mean=("temperature", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    sns.lineplot(
        data=monthly,
        x="month",
        y="load",
        hue="year",
        marker="o",
        palette="viridis",
        ax=axes[0],
    )
    axes[0].set_xticks(range(1, 13), MONTH_ORDER)
    axes[0].set_title("The seasonal shape is stable, but amplitudes drift")
    axes[0].set_xlabel("Operating month")
    axes[0].set_ylabel("Monthly mean load")
    axes[0].legend(title="Year", ncol=2, fontsize=8)

    axes[1].plot(
        yearly["year"],
        yearly["load_mean"],
        marker="o",
        color="#E45756",
        label="Mean load",
    )
    ax_temp = axes[1].twinx()
    ax_temp.plot(
        yearly["year"],
        yearly["temperature_mean"],
        marker="s",
        color="#4C78A8",
        label="Mean temperature",
    )
    axes[1].set_title("Annual load shifts are not just annual temperature shifts")
    axes[1].set_xlabel("Operating year")
    axes[1].set_ylabel("Annual mean load")
    ax_temp.set_ylabel("Annual mean temperature")
    handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = ax_temp.get_legend_handles_labels()
    axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper left")
    save_figure(fig, output, dpi)
