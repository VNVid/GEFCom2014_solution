"""Create the first-round CatBoost validation report and comparison artifacts."""

from __future__ import annotations

import argparse
from math import comb
import os
from pathlib import Path
import tempfile

temporary_root = Path(tempfile.gettempdir())
os.environ.setdefault(
    "MPLCONFIGDIR", str(temporary_root / "gefcom2014-matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", str(temporary_root / "gefcom2014-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import t as student_t


CATBOOST_LABEL = "CatBoost round 1"
EMPIRICAL_LABEL = "Seasonal empirical"
NAIVE_LABEL = "Seasonal naive"
MODEL_LABELS = {
    "catboost_round1": CATBOOST_LABEL,
    "seasonal_empirical": EMPIRICAL_LABEL,
    "seasonal_naive": NAIVE_LABEL,
}


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    return float(np.average(values.to_numpy(dtype=float), weights=weights))


def _exact_sign_test(differences: np.ndarray) -> tuple[int, int, int, float]:
    """Return wins, losses, ties and an exact two-sided paired sign-test p-value."""

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Paired differences must be a finite one-dimensional array")
    ties = int(np.isclose(values, 0.0).sum())
    wins = int((values < 0).sum())
    losses = int((values > 0).sum())
    trials = wins + losses
    if trials == 0:
        return wins, losses, ties, 1.0
    tail = min(wins, losses)
    probability = 2.0 * sum(comb(trials, k) for k in range(tail + 1)) / 2**trials
    return wins, losses, ties, float(min(1.0, probability))


def _hac_mean_loss_test(
    differences: np.ndarray, lag: int
) -> tuple[float, float, float, float, float]:
    """Test a zero mean loss differential using a Bartlett HAC variance.

    The returned values are the mean difference, test statistic, two-sided
    p-value, and lower/upper 95% confidence limits. A negative difference favors
    the first model in the pair.
    """

    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or values.size < 3 or not np.isfinite(values).all():
        raise ValueError("At least three finite paired differences are required")
    if lag < 0 or lag >= values.size:
        raise ValueError("HAC lag must lie between zero and n - 1")

    count = values.size
    mean = float(values.mean())
    centered = values - mean
    long_run_variance = float(centered @ centered / count)
    for offset in range(1, lag + 1):
        covariance = float(centered[offset:] @ centered[:-offset] / count)
        weight = 1.0 - offset / (lag + 1.0)
        long_run_variance += 2.0 * weight * covariance
    long_run_variance = max(long_run_variance, 0.0)
    standard_error = float(np.sqrt(long_run_variance / count))
    if np.isclose(standard_error, 0.0):
        statistic = 0.0 if np.isclose(mean, 0.0) else np.sign(mean) * np.inf
        pvalue = 1.0 if np.isclose(mean, 0.0) else 0.0
        return mean, float(statistic), pvalue, mean, mean

    statistic = mean / standard_error
    pvalue = float(2.0 * student_t.sf(abs(statistic), df=count - 1))
    critical = float(student_t.ppf(0.975, df=count - 1))
    return (
        mean,
        float(statistic),
        pvalue,
        mean - critical * standard_error,
        mean + critical * standard_error,
    )


def _within_month_calibration(
    predictions: pd.DataFrame, quantile_columns: list[str], levels: np.ndarray
) -> dict[str, float]:
    """Calculate hour-weighted means of fold-level calibration error."""

    records: list[dict[str, float | str]] = []
    for (model, _), group in predictions.groupby(["model", "origin"], sort=True):
        actual = group["actual"].to_numpy(dtype=float)
        forecast = group[quantile_columns].to_numpy(dtype=float)
        coverage = np.mean(actual[:, None] <= forecast, axis=0)
        records.append(
            {
                "model": str(model),
                "hours": len(group),
                "calibration_mae": float(np.mean(np.abs(coverage - levels))),
            }
        )
    fold_calibration = pd.DataFrame(records)
    return {
        str(model): _weighted_average(group["calibration_mae"], group["hours"])
        for model, group in fold_calibration.groupby("model", sort=True)
    }


def _build_model_comparison(
    selected: pd.Series,
    baseline_aggregate: pd.DataFrame,
    baseline_intervals: pd.DataFrame,
    baseline_calibration: dict[str, float],
) -> pd.DataFrame:
    empirical_loss = float(
        baseline_aggregate.set_index("model").loc[
            "seasonal_empirical", "pinball_loss"
        ]
    )
    records: list[dict[str, float | int | str]] = [
        {
            "model": CATBOOST_LABEL,
            "pinball_loss": float(selected["pinball_loss"]),
            "relative_improvement_vs_empirical": 1.0
            - float(selected["pinball_loss"]) / empirical_loss,
            "monthly_pinball_mean": float(selected["fold_pinball_mean"]),
            "monthly_pinball_std": float(selected["fold_pinball_std"]),
            "median_mae": float(selected["median_mae"]),
            "median_bias_actual_minus_forecast": float(selected["median_bias"]),
            "within_month_calibration_mae": float(
                selected["mean_absolute_calibration_error"]
            ),
            "coverage_90": float(selected["coverage_90"]),
            "mean_width_90": float(selected["mean_width_90"]),
            "invalid_90_intervals": int(selected["invalid_90_intervals"]),
            "adjacent_quantile_crossings": int(selected["quantile_crossings"]),
            "adjacent_crossing_rate": float(selected["quantile_crossings"])
            / (int(selected["forecast_hours"]) * 98),
        }
    ]

    aggregate_by_model = baseline_aggregate.set_index("model")
    interval_90 = baseline_intervals.loc[
        np.isclose(baseline_intervals["nominal_coverage"], 0.9)
    ].set_index("model")
    for model in ("seasonal_empirical", "seasonal_naive"):
        aggregate = aggregate_by_model.loc[model]
        interval = interval_90.loc[model]
        records.append(
            {
                "model": MODEL_LABELS[model],
                "pinball_loss": float(aggregate["pinball_loss"]),
                "relative_improvement_vs_empirical": 1.0
                - float(aggregate["pinball_loss"]) / empirical_loss,
                "monthly_pinball_mean": float(aggregate["fold_pinball_mean"]),
                "monthly_pinball_std": float(aggregate["fold_pinball_std"]),
                "median_mae": float(aggregate["median_mae"]),
                "median_bias_actual_minus_forecast": float(
                    aggregate["median_bias"]
                ),
                "within_month_calibration_mae": baseline_calibration[model],
                "coverage_90": float(interval["empirical_coverage"]),
                "mean_width_90": float(interval["mean_width"]),
                "invalid_90_intervals": 0,
                "adjacent_quantile_crossings": int(
                    aggregate["quantile_crossings"]
                ),
                "adjacent_crossing_rate": 0.0,
            }
        )
    return pd.DataFrame(records)


def _build_fold_table(
    selected_folds: pd.DataFrame, baseline_folds: pd.DataFrame
) -> pd.DataFrame:
    baseline_wide = baseline_folds.pivot(
        index="origin", columns="model", values="pinball_loss"
    )
    selected_by_origin = selected_folds.set_index("origin")
    table = baseline_wide.join(
        selected_by_origin[["pinball_loss", "evaluation_rows"]].rename(
            columns={"pinball_loss": "catboost_round1", "evaluation_rows": "hours"}
        ),
        how="inner",
        validate="one_to_one",
    ).reset_index()
    table["catboost_minus_empirical"] = (
        table["catboost_round1"] - table["seasonal_empirical"]
    )
    table["catboost_minus_naive"] = (
        table["catboost_round1"] - table["seasonal_naive"]
    )
    return table


def _build_paired_tests(
    fold_table: pd.DataFrame, search_candidates: int, period: str
) -> tuple[pd.DataFrame, int]:
    count = len(fold_table)
    hac_lag = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    pairs = (
        ("catboost_round1", "seasonal_empirical"),
        ("catboost_round1", "seasonal_naive"),
        ("seasonal_empirical", "seasonal_naive"),
    )
    records: list[dict[str, float | int | str]] = []
    weights = fold_table["hours"].to_numpy(dtype=float)
    for model, reference in pairs:
        differences = (
            fold_table[model] - fold_table[reference]
        ).to_numpy(dtype=float)
        wins, losses, ties, sign_pvalue = _exact_sign_test(differences)
        mean, statistic, hac_pvalue, lower, upper = _hac_mean_loss_test(
            differences, hac_lag
        )
        adjustment = search_candidates if model == "catboost_round1" else 1
        model_loss = float(np.average(fold_table[model], weights=weights))
        reference_loss = float(np.average(fold_table[reference], weights=weights))
        records.append(
            {
                "period": period,
                "model": MODEL_LABELS[model],
                "reference_model": MODEL_LABELS[reference],
                "paired_months": count,
                "hour_weighted_relative_improvement": 1.0
                - model_loss / reference_loss,
                "mean_monthly_loss_difference": mean,
                "std_monthly_loss_difference": float(
                    np.std(differences, ddof=1)
                ),
                "hac_lag": hac_lag,
                "hac_mean_test_statistic": statistic,
                "hac_mean_test_pvalue": hac_pvalue,
                "hac_mean_difference_ci95_lower": lower,
                "hac_mean_difference_ci95_upper": upper,
                "folds_won": wins,
                "folds_lost": losses,
                "folds_tied": ties,
                "exact_sign_test_pvalue": sign_pvalue,
                "selection_bonferroni_factor": adjustment,
                "selection_adjusted_hac_pvalue": min(1.0, adjustment * hac_pvalue),
                "selection_adjusted_sign_pvalue": min(
                    1.0, adjustment * sign_pvalue
                ),
            }
        )
    return pd.DataFrame(records), hac_lag


def _build_period_comparison(fold_table: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, float | int | str]] = []
    periods = [
        ("2009", fold_table["origin"].dt.year.eq(2009)),
        ("2010", fold_table["origin"].dt.year.eq(2010)),
    ]
    periods.extend(
        (f"Q{quarter} (both years)", fold_table["origin"].dt.quarter.eq(quarter))
        for quarter in range(1, 5)
    )
    for period, mask in periods:
        group = fold_table.loc[mask]
        weights = group["hours"].to_numpy(dtype=float)
        catboost = float(np.average(group["catboost_round1"], weights=weights))
        empirical = float(
            np.average(group["seasonal_empirical"], weights=weights)
        )
        naive = float(np.average(group["seasonal_naive"], weights=weights))
        records.append(
            {
                "period": period,
                "months": len(group),
                "catboost_pinball": catboost,
                "empirical_pinball": empirical,
                "naive_pinball": naive,
                "catboost_relative_improvement_vs_empirical": 1.0
                - catboost / empirical,
                "catboost_wins_vs_empirical": int(
                    (group["catboost_round1"] < group["seasonal_empirical"]).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def _plot_monthly_scores(fold_table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    styles = (
        ("catboost_round1", CATBOOST_LABEL, "#2563eb", "o"),
        ("seasonal_empirical", EMPIRICAL_LABEL, "#d97706", "s"),
        ("seasonal_naive", NAIVE_LABEL, "#6b7280", "^"),
    )
    for column, label, color, marker in styles:
        axis.plot(
            fold_table["origin"],
            fold_table[column],
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
        )
    axis.set(title="Monthly rolling-origin pinball loss", ylabel="Pinball loss")
    axis.set_xlabel("")
    axis.legend(frameon=False, ncol=3)
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_fold_differences(fold_table: pd.DataFrame, path: Path) -> None:
    differences = fold_table["catboost_minus_empirical"]
    colors = np.where(differences < 0, "#15803d", "#dc2626")
    labels = fold_table["origin"].dt.strftime("%Y-%m")
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(labels, differences, color=colors, width=0.8)
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.set(
        title="CatBoost minus seasonal-empirical pinball loss by fold",
        ylabel="Loss difference (negative favors CatBoost)",
        xlabel="Forecast month",
    )
    axis.tick_params(axis="x", rotation=55)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_search_landscape(
    candidate_summary: pd.DataFrame, empirical_loss: float, path: Path
) -> None:
    figure, axis = plt.subplots(figsize=(10.5, 6))
    colors = {4: "#2563eb", 6: "#d97706", 8: "#7c3aed"}
    line_styles = {0.04: "-", 0.08: "--"}
    for (depth, learning_rate), group in candidate_summary.groupby(
        ["depth", "learning_rate"], sort=True
    ):
        ordered = group.sort_values("iterations")
        axis.plot(
            ordered["iterations"],
            ordered["pinball_loss"],
            color=colors[int(depth)],
            linestyle=line_styles[float(learning_rate)],
            marker="o",
            linewidth=1.8,
            label=f"depth {int(depth)}, lr {learning_rate:g}",
        )
    axis.axhline(
        empirical_loss,
        color="#111827",
        linestyle=":",
        linewidth=1.5,
        label="Seasonal empirical",
    )
    axis.set(
        title="Round-1 search: validation score versus tree count",
        xlabel="Trees",
        ylabel="Hour-weighted pinball loss",
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_calibration_sharpness(comparison: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    colors = ["#2563eb", "#d97706", "#6b7280"]
    sns.barplot(
        data=comparison,
        x="model",
        y="coverage_90",
        hue="model",
        palette=colors,
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(0.9, color="#111827", linestyle=":", linewidth=1.4)
    axes[0].set(title="Nominal 90% interval coverage", xlabel="", ylabel="Coverage")
    axes[0].set_ylim(0, 1)
    sns.barplot(
        data=comparison,
        x="model",
        y="mean_width_90",
        hue="model",
        palette=colors,
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="90% interval sharpness", xlabel="", ylabel="Mean width (MW)")
    for axis in axes:
        axis.tick_params(axis="x", rotation=18)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run(search_dir: Path, baseline_dir: Path, output_dir: Path) -> Path:
    """Build deterministic tables and figures from saved validation results."""

    candidate_summary = pd.read_csv(search_dir / "candidate_summary.csv")
    fold_results = pd.read_csv(search_dir / "fold_results.csv", parse_dates=["origin"])
    baseline_folds = pd.read_csv(
        baseline_dir / "fold_metrics.csv", parse_dates=["origin"]
    )
    baseline_aggregate = pd.read_csv(baseline_dir / "aggregate_metrics.csv")
    baseline_intervals = pd.read_csv(baseline_dir / "interval_calibration.csv")
    baseline_predictions = pd.read_csv(
        baseline_dir / "predictions.csv.gz", parse_dates=["origin"]
    )

    if candidate_summary.empty:
        raise ValueError("Candidate summary is empty")
    complete_folds = int(candidate_summary["folds"].max())
    complete = (
        candidate_summary.loc[candidate_summary["folds"].eq(complete_folds)]
        .sort_values("pinball_loss")
        .reset_index(drop=True)
    )
    if complete_folds != 24 or len(complete) != 18:
        raise ValueError("Round-1 report requires 18 candidates with all 24 folds")
    selected = complete.iloc[0]
    selected_folds = fold_results.loc[
        fold_results["candidate"].eq(selected["candidate"])
    ].sort_values("origin")
    if len(selected_folds) != complete_folds:
        raise ValueError("Selected candidate does not have one row per fold")

    levels = np.arange(1, 100, dtype=float) / 100.0
    quantile_columns = [f"q{level:.2f}" for level in levels]
    baseline_calibration = _within_month_calibration(
        baseline_predictions, quantile_columns, levels
    )
    comparison = _build_model_comparison(
        selected,
        baseline_aggregate,
        baseline_intervals,
        baseline_calibration,
    )
    fold_table = _build_fold_table(selected_folds, baseline_folds)
    full_tests, _ = _build_paired_tests(
        fold_table, len(complete), "2009-2010 validation"
    )
    tests_2010, _ = _build_paired_tests(
        fold_table.loc[fold_table["origin"].dt.year.eq(2010)],
        len(complete),
        "2010 only",
    )
    paired_tests = pd.concat([full_tests, tests_2010], ignore_index=True)
    period_comparison = _build_period_comparison(fold_table)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_comparison.csv": comparison,
        "paired_tests.csv": paired_tests,
        "period_comparison.csv": period_comparison,
        "selected_fold_comparison.csv": fold_table,
        "candidate_shortlist.csv": complete.head(8),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_monthly_scores(fold_table, figures_dir / "01_monthly_pinball.png")
    _plot_fold_differences(fold_table, figures_dir / "02_fold_differences.png")
    empirical_loss = float(
        comparison.set_index("model").loc[EMPIRICAL_LABEL, "pinball_loss"]
    )
    _plot_search_landscape(
        complete, empirical_loss, figures_dir / "03_search_landscape.png"
    )
    _plot_calibration_sharpness(
        comparison, figures_dir / "04_calibration_sharpness.png"
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path("artifacts/catboost/search/phase1"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("artifacts/baseline/validation"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catboost/round1"),
    )
    args = parser.parse_args()
    output_dir = run(args.search_dir, args.baseline_dir, args.output_dir)
    print(f"Wrote comparison artifacts to {output_dir}")


if __name__ == "__main__":
    main()
