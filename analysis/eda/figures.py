"""Configure and generate the ordered EDA figure set."""

from __future__ import annotations

from pathlib import Path

# This import must precede seaborn so the writable Matplotlib cache is set.
from .plot_utils import plt

import seaborn as sns

from .analysis import EdaResults
from .load_plots import (
    plot_calendar_seasonality,
    plot_data_coverage,
    plot_load_overview,
    plot_year_month_drift,
)
from .weather_plots import plot_temperature_diagnostics, plot_temperature_load_response


def generate_figures(results: EdaResults, output_dir: Path, config: dict) -> None:
    """Create all report figures from named analysis results."""

    sns.set_theme(
        style=config.get("style", "whitegrid"),
        context=config.get("context", "notebook"),
        palette=config.get("palette", "colorblind"),
    )
    plt.rcParams.update(
        {
            "figure.dpi": config.get("dpi", 150),
            "savefig.dpi": config.get("dpi", 150),
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )

    dpi = int(config["dpi"])
    extension = config.get("format", "png")
    plot_data_coverage(
        results.history,
        results.manifest,
        output_dir / f"01_data_coverage.{extension}",
        dpi,
    )
    plot_load_overview(results.observed, output_dir / f"02_load_overview.{extension}", dpi)
    plot_calendar_seasonality(
        results.observed,
        output_dir / f"03_calendar_seasonality.{extension}",
        dpi,
    )
    plot_temperature_diagnostics(
        results.history,
        results.tables["weather_station_summary.csv"],
        results.station_correlations,
        output_dir / f"04_temperature_diagnostics.{extension}",
        dpi,
    )
    plot_temperature_load_response(
        results.observed,
        results.tables["temperature_response_bins.csv"],
        results.temperature_response_by_hour,
        output_dir / f"05_temperature_load_response.{extension}",
        dpi,
    )
    plot_year_month_drift(
        results.observed,
        output_dir / f"09_year_month_drift.{extension}",
        dpi,
    )
