"""Figures describing weather stations and their load relationship."""

from __future__ import annotations

from pathlib import Path

# This import must precede seaborn's Matplotlib import.
from .plot_utils import plt, save_figure

import pandas as pd
import seaborn as sns

from gefcom2014.data import WEATHER_COLUMNS


def plot_temperature_diagnostics(
    history: pd.DataFrame,
    station_summary: pd.DataFrame,
    station_correlations: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    weather = history.set_index("period_start")[list(WEATHER_COLUMNS)]
    monthly = weather.resample("MS").mean()
    monthly_mean = monthly.mean(axis=1)
    monthly_min = monthly.min(axis=1)
    monthly_max = monthly.max(axis=1)

    fig = plt.figure(figsize=(15, 10))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.85, 1.25], wspace=0.27, hspace=0.32)
    ax_time = fig.add_subplot(grid[0, :])
    ax_corr = fig.add_subplot(grid[1, 0])
    ax_load = fig.add_subplot(grid[1, 1])

    ax_time.fill_between(
        monthly.index,
        monthly_min,
        monthly_max,
        color="#4C78A8",
        alpha=0.18,
        label="Station min–max",
    )
    ax_time.plot(monthly.index, monthly_mean, color="#4C78A8", linewidth=1.2, label="Station mean")
    ax_time.set_title("Weather stations share seasonality but retain location-specific levels")
    ax_time.set_ylabel("Monthly temperature")
    ax_time.legend(ncol=2)

    sns.heatmap(
        station_correlations,
        cmap="rocket",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax_corr,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax_corr.set_title("Pairwise station correlation")
    ax_corr.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax_corr.tick_params(axis="y", labelsize=7)

    correlations_long = station_summary.melt(
        id_vars="station",
        value_vars=["load_corr_cold_quartile", "load_corr_hot_quartile"],
        var_name="regime",
        value_name="correlation",
    )
    correlations_long["regime"] = correlations_long["regime"].map(
        {
            "load_corr_cold_quartile": "Coldest temperature quartile",
            "load_corr_hot_quartile": "Hottest temperature quartile",
        }
    )
    sns.barplot(data=correlations_long, x="station", y="correlation", hue="regime", ax=ax_load)
    ax_load.axhline(0, color="black", linewidth=0.7)
    ax_load.set_title("Temperature–load direction reverses by regime")
    ax_load.set_ylabel("Pearson correlation with load")
    ax_load.set_xlabel("Station")
    ax_load.tick_params(axis="x", rotation=90, labelsize=8)
    ax_load.legend(fontsize=8)
    save_figure(fig, output, dpi)


def plot_temperature_load_response(
    observed: pd.DataFrame,
    bins: pd.DataFrame,
    block_bins: pd.DataFrame,
    output: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True)
    hexbin = axes[0].hexbin(
        observed["temperature"],
        observed["load"],
        gridsize=55,
        mincnt=1,
        cmap="viridis",
        bins="log",
    )
    axes[0].plot(
        bins["temperature_mean"],
        bins["load_mean"],
        color="#E45756",
        linewidth=2.3,
        label="Binned mean",
    )
    axes[0].set_title("Load response to contemporaneous observed temperature")
    axes[0].set_xlabel("Mean temperature across stations")
    axes[0].set_ylabel("Load")
    axes[0].legend()
    fig.colorbar(hexbin, ax=axes[0], label="log10 hourly count")

    sns.lineplot(
        data=block_bins,
        x="temperature_mean",
        y="load_mean",
        hue="hour_block",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Temperature response changes across the daily cycle")
    axes[1].set_xlabel("Mean temperature across stations")
    axes[1].set_ylabel("Binned mean load")
    axes[1].legend(title="Operating hours", fontsize=8, title_fontsize=8)
    save_figure(fig, output, dpi)
