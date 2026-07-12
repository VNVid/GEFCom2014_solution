"""Compact diagnostics for the baseline experiment report."""

from __future__ import annotations

from pathlib import Path

from analysis.eda.plot_utils import plt, save_figure
import pandas as pd
import seaborn as sns


def generate_figures(
    fold_metrics: pd.DataFrame,
    quantile_calibration: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    output_dir: Path,
    config: dict,
) -> None:
    """Write fold-score and calibration figures."""

    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(
        style=config.get("style", "whitegrid"),
        context=config.get("context", "notebook"),
        palette=config.get("palette", "colorblind"),
    )
    dpi = int(config.get("dpi", 150))
    extension = config.get("format", "png")

    fig, axis = plt.subplots(figsize=(11, 4.5))
    sns.lineplot(
        data=fold_metrics,
        x="origin",
        y="pinball_loss",
        hue="model",
        marker="o",
        ax=axis,
    )
    axis.set(
        title="Monthly rolling-origin pinball loss",
        xlabel="Forecast origin",
        ylabel="Mean pinball loss",
    )
    axis.tick_params(axis="x", rotation=45)
    save_figure(fig, output_dir / f"01_fold_pinball.{extension}", dpi)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sns.lineplot(
        data=quantile_calibration,
        x="quantile",
        y="empirical_coverage",
        hue="model",
        ax=axes[0],
    )
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set(
        title="Marginal quantile calibration",
        xlabel="Nominal quantile",
        ylabel="Empirical P(Y ≤ forecast)",
        xlim=(0, 1),
        ylim=(0, 1),
    )

    sns.lineplot(
        data=interval_calibration,
        x="nominal_coverage",
        y="empirical_coverage",
        hue="model",
        marker="o",
        ax=axes[1],
    )
    axes[1].plot([0.45, 1], [0.45, 1], linestyle="--", color="black", linewidth=1)
    axes[1].set(
        title="Central interval coverage",
        xlabel="Nominal coverage",
        ylabel="Empirical coverage",
        xlim=(0.45, 1),
        ylim=(0, 1),
    )
    fig.tight_layout()
    save_figure(fig, output_dir / f"02_calibration.{extension}", dpi)
