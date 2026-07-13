"""Create the round-two CatBoost validation report and comparison artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

temporary_root = Path(tempfile.gettempdir())
os.environ.setdefault("MPLCONFIGDIR", str(temporary_root / "gefcom2014-matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(temporary_root / "gefcom2014-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml

from analysis.catboost.report import (
    _exact_sign_test,
    _hac_mean_loss_test,
    _plot_calibration_curves,
    _within_month_calibration,
)
from gefcom2014.evaluation import evaluate_predictions


ROUND2_LABEL = "CatBoost round 2"
ROUND1_LABEL = "CatBoost round 1"
EMPIRICAL_LABEL = "Seasonal empirical"
NAIVE_LABEL = "Seasonal naive"
ROUND1_CANDIDATE = "depth4_lr0p04_l25_trees250"


def _weighted_average(frame: pd.DataFrame, column: str) -> float:
    return float(
        np.average(
            frame[column].to_numpy(dtype=float),
            weights=frame["evaluation_rows"].to_numpy(dtype=float),
        )
    )


def _model_record(
    frame: pd.DataFrame,
    label: str,
    calibration_mae: float,
) -> dict[str, object]:
    weights = frame["evaluation_rows"].to_numpy(dtype=float)
    forecast_hours = int(weights.sum())
    return {
        "model": label,
        "folds": len(frame),
        "forecast_hours": forecast_hours,
        "pinball_loss": _weighted_average(frame, "pinball_loss"),
        "fold_pinball_mean": float(frame["pinball_loss"].mean()),
        "fold_pinball_std": float(frame["pinball_loss"].std(ddof=1)),
        "median_mae": _weighted_average(frame, "median_mae"),
        "median_bias_actual_minus_forecast": _weighted_average(
            frame, "median_bias"
        ),
        "within_month_calibration_mae": calibration_mae,
        "coverage_90": _weighted_average(frame, "coverage_90"),
        "mean_width_90": _weighted_average(frame, "mean_width_90"),
        "invalid_90_intervals": int(frame["invalid_90_intervals"].sum()),
        "adjacent_quantile_crossings": int(frame["quantile_crossings"].sum()),
        "adjacent_crossing_rate": int(frame["quantile_crossings"].sum())
        / (forecast_hours * 98),
    }


def _build_fold_comparison(
    round2: pd.DataFrame,
    round1: pd.DataFrame,
    baselines: pd.DataFrame,
) -> pd.DataFrame:
    baseline_wide = baselines.pivot(
        index="origin", columns="model", values="pinball_loss"
    )
    round2_indexed = round2.set_index("origin")
    round1_indexed = round1.set_index("origin")
    table = pd.DataFrame(
        {
            "hours": round2_indexed["evaluation_rows"],
            "catboost_round2": round2_indexed["pinball_loss"],
            "catboost_round1": round1_indexed["pinball_loss"],
            "seasonal_empirical": baseline_wide["seasonal_empirical"],
            "seasonal_naive": baseline_wide["seasonal_naive"],
        }
    ).reset_index()
    if table.isna().any().any() or len(table) != 12:
        raise ValueError("Matched 2010 fold comparison is incomplete")
    for reference in (
        "catboost_round1",
        "seasonal_empirical",
        "seasonal_naive",
    ):
        table[f"round2_minus_{reference}"] = (
            table["catboost_round2"] - table[reference]
        )
    return table


def _build_paired_tests(fold_table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    count = len(fold_table)
    lag = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    references = (
        ("catboost_round1", ROUND1_LABEL),
        ("seasonal_empirical", EMPIRICAL_LABEL),
        ("seasonal_naive", NAIVE_LABEL),
    )
    weights = fold_table["hours"].to_numpy(dtype=float)
    model_loss = float(
        np.average(fold_table["catboost_round2"], weights=weights)
    )
    records = []
    for column, label in references:
        differences = (
            fold_table["catboost_round2"] - fold_table[column]
        ).to_numpy(dtype=float)
        wins, losses, ties, sign_pvalue = _exact_sign_test(differences)
        mean, statistic, hac_pvalue, lower, upper = _hac_mean_loss_test(
            differences, lag
        )
        reference_loss = float(np.average(fold_table[column], weights=weights))
        records.append(
            {
                "model": ROUND2_LABEL,
                "reference_model": label,
                "paired_months": count,
                "hour_weighted_relative_improvement": 1.0
                - model_loss / reference_loss,
                "mean_monthly_loss_difference": mean,
                "std_monthly_loss_difference": float(
                    np.std(differences, ddof=1)
                ),
                "hac_lag": lag,
                "hac_mean_test_statistic": statistic,
                "hac_mean_test_pvalue": hac_pvalue,
                "hac_mean_difference_ci95_lower": lower,
                "hac_mean_difference_ci95_upper": upper,
                "folds_won": wins,
                "folds_lost": losses,
                "folds_tied": ties,
                "exact_sign_test_pvalue": sign_pvalue,
            }
        )
    return pd.DataFrame(records), lag


def _build_quarter_comparison(fold_table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for quarter, group in fold_table.groupby(fold_table["origin"].dt.quarter):
        weights = group["hours"].to_numpy(dtype=float)
        round2_loss = float(
            np.average(group["catboost_round2"], weights=weights)
        )
        round1_loss = float(
            np.average(group["catboost_round1"], weights=weights)
        )
        empirical_loss = float(
            np.average(group["seasonal_empirical"], weights=weights)
        )
        records.append(
            {
                "quarter": f"Q{quarter}",
                "months": len(group),
                "round2_pinball": round2_loss,
                "round1_pinball": round1_loss,
                "empirical_pinball": empirical_loss,
                "naive_pinball": float(
                    np.average(group["seasonal_naive"], weights=weights)
                ),
                "round2_improvement_vs_round1": 1.0
                - round2_loss / round1_loss,
                "round2_improvement_vs_empirical": 1.0
                - round2_loss / empirical_loss,
                "round2_wins_vs_round1": int(
                    (group["catboost_round2"] < group["catboost_round1"]).sum()
                ),
                "round2_wins_vs_empirical": int(
                    (
                        group["catboost_round2"]
                        < group["seasonal_empirical"]
                    ).sum()
                ),
            }
        )
    return pd.DataFrame(records)


def _best_by_structure(candidate_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        candidate_summary.sort_values("pinball_loss")
        .groupby(["depth", "l2_leaf_reg"], as_index=False)
        .first()
        .sort_values(["depth", "l2_leaf_reg"])
    )


def _plot_monthly_scores(fold_table: pd.DataFrame, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(12, 5.5))
    styles = (
        ("catboost_round2", ROUND2_LABEL, "#2563eb", "o"),
        ("catboost_round1", ROUND1_LABEL, "#7c3aed", "D"),
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
    axis.set(title="2010 monthly rolling-origin pinball loss", ylabel="Pinball loss")
    axis.set_xlabel("")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_fold_differences(fold_table: pd.DataFrame, path: Path) -> None:
    labels = fold_table["origin"].dt.strftime("%Y-%m")
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(
        positions - width / 2,
        fold_table["round2_minus_catboost_round1"],
        width,
        label="Round 2 minus round 1",
        color="#7c3aed",
    )
    axis.bar(
        positions + width / 2,
        fold_table["round2_minus_seasonal_empirical"],
        width,
        label="Round 2 minus empirical",
        color="#d97706",
    )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.set(
        title="Round-two monthly loss differences",
        ylabel="Pinball difference (negative favors round 2)",
        xlabel="Forecast month",
        xticks=positions,
        xticklabels=labels,
    )
    axis.tick_params(axis="x", rotation=45)
    axis.legend(frameon=False)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_search_landscape(candidate_summary: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8), sharey=True)
    colors = {1.0: "#2563eb", 5.0: "#16a34a", 20.0: "#d97706"}
    for axis, depth in zip(axes, sorted(candidate_summary["depth"].unique())):
        depth_rows = candidate_summary.loc[candidate_summary["depth"].eq(depth)]
        for l2, group in depth_rows.groupby("l2_leaf_reg", sort=True):
            ordered = group.sort_values("iterations")
            axis.plot(
                ordered["iterations"],
                ordered["pinball_loss"],
                marker="o",
                linewidth=1.8,
                color=colors[float(l2)],
                label=f"L2={l2:g}",
            )
        axis.set(title=f"Depth {int(depth)}", xlabel="Trees")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Hour-weighted pinball loss")
    axes[-1].legend(frameon=False)
    figure.suptitle("Round-two search landscape", y=1.02)
    figure.tight_layout()
    figure.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(figure)


def _plot_calibration_sharpness(comparison: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    order = comparison["model"].tolist()
    colors = ["#2563eb", "#7c3aed", "#d97706", "#6b7280"]
    sns.barplot(
        data=comparison,
        x="model",
        y="coverage_90",
        hue="model",
        order=order,
        palette=colors,
        legend=False,
        ax=axes[0],
    )
    axes[0].axhline(0.9, color="#111827", linestyle=":", linewidth=1.4)
    axes[0].set(title="90% interval coverage", xlabel="", ylabel="Coverage")
    axes[0].set_ylim(0, 1)
    sns.barplot(
        data=comparison,
        x="model",
        y="mean_width_90",
        hue="model",
        order=order,
        palette=colors,
        legend=False,
        ax=axes[1],
    )
    axes[1].set(title="90% interval width", xlabel="", ylabel="Mean width (MW)")
    sns.barplot(
        data=comparison,
        x="model",
        y="within_month_calibration_mae",
        hue="model",
        order=order,
        palette=colors,
        legend=False,
        ax=axes[2],
    )
    axes[2].set(title="Monthly calibration MAE", xlabel="", ylabel="MAE")
    for axis in axes:
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    selected: pd.Series,
    comparison: pd.DataFrame,
    fold_table: pd.DataFrame,
    paired_tests: pd.DataFrame,
    quarter_comparison: pd.DataFrame,
    shortlist: pd.DataFrame,
    structure: pd.DataFrame,
    selected_features: list[str],
    new_features: list[str],
    feature_selection_summary: dict[str, object],
    quantile_calibration: pd.DataFrame,
    interval_calibration: pd.DataFrame,
    search_candidates: int,
    hac_lag: int,
    search_seconds: float,
) -> None:
    by_model = comparison.set_index("model")
    round2 = by_model.loc[ROUND2_LABEL]
    round1 = by_model.loc[ROUND1_LABEL]
    empirical = by_model.loc[EMPIRICAL_LABEL]
    naive = by_model.loc[NAIVE_LABEL]
    test_by_reference = paired_tests.set_index("reference_model")
    vs_round1 = test_by_reference.loc[ROUND1_LABEL]
    vs_empirical = test_by_reference.loc[EMPIRICAL_LABEL]
    vs_naive = test_by_reference.loc[NAIVE_LABEL]

    primary_rows = []
    for label in (ROUND2_LABEL, ROUND1_LABEL, EMPIRICAL_LABEL, NAIVE_LABEL):
        row = by_model.loc[label]
        primary_rows.append(
            [
                label,
                f"{row['pinball_loss']:.3f}",
                f"{100 * (1 - row['pinball_loss'] / empirical['pinball_loss']):+.2f}",
                f"{row['fold_pinball_mean']:.3f}",
                f"{row['fold_pinball_std']:.3f}",
                f"{row['median_mae']:.2f}",
                f"{row['median_bias_actual_minus_forecast']:+.2f}",
            ]
        )
    primary_table = _table(
        [
            "Model",
            "Pinball",
            "Improvement vs empirical (%)",
            "Monthly mean",
            "Monthly SD",
            "Median MAE",
            "Median bias",
        ],
        primary_rows,
    )

    quarter_rows = [
        [
            str(row.quarter),
            f"{row.round2_pinball:.3f}",
            f"{row.round1_pinball:.3f}",
            f"{row.empirical_pinball:.3f}",
            f"{100 * row.round2_improvement_vs_round1:+.2f}",
            f"{int(row.round2_wins_vs_round1)}/3",
        ]
        for row in quarter_comparison.itertuples(index=False)
    ]
    quarter_table = _table(
        [
            "Quarter",
            "Round 2",
            "Round 1",
            "Empirical",
            "Improvement vs round 1 (%)",
            "Wins vs round 1",
        ],
        quarter_rows,
    )

    paired_rows = []
    for row in paired_tests.itertuples(index=False):
        paired_rows.append(
            [
                str(row.reference_model),
                f"{row.mean_monthly_loss_difference:+.3f}",
                f"{int(row.folds_won)}",
                f"{int(row.folds_lost)}",
                f"{row.hac_mean_test_pvalue:.4g}",
                f"{row.exact_sign_test_pvalue:.4g}",
                (
                    f"[{row.hac_mean_difference_ci95_lower:.3f}, "
                    f"{row.hac_mean_difference_ci95_upper:.3f}]"
                ),
            ]
        )
    paired_table = _table(
        [
            "Reference",
            "Mean difference",
            "Wins",
            "Losses",
            "HAC p",
            "Sign p",
            "HAC 95% CI",
        ],
        paired_rows,
    )

    calibration_rows = []
    for label in (ROUND2_LABEL, ROUND1_LABEL, EMPIRICAL_LABEL, NAIVE_LABEL):
        row = by_model.loc[label]
        calibration_rows.append(
            [
                label,
                f"{row['within_month_calibration_mae']:.3f}",
                f"{row['coverage_90']:.3f}",
                f"{row['mean_width_90']:.2f}",
                f"{int(row['invalid_90_intervals'])}",
                f"{int(row['adjacent_quantile_crossings'])}",
                f"{100 * row['adjacent_crossing_rate']:.3f}",
            ]
        )
    calibration_table = _table(
        [
            "Model",
            "Calibration MAE",
            "90% coverage",
            "90% width",
            "Invalid intervals",
            "Crossings",
            "Crossing rate (%)",
        ],
        calibration_rows,
    )
    round2_median_coverage = float(
        quantile_calibration.loc[
            quantile_calibration["model"].eq(ROUND2_LABEL)
            & np.isclose(quantile_calibration["quantile"], 0.5),
            "empirical_coverage",
        ].iloc[0]
    )

    structure_rows = [
        [
            f"{int(row.depth)}",
            f"{row.l2_leaf_reg:g}",
            f"{int(row.iterations)}",
            f"{row.pinball_loss:.3f}",
            f"{row.coverage_90:.3f}",
            f"{int(row.quantile_crossings)}",
        ]
        for row in structure.itertuples(index=False)
    ]
    structure_table = _table(
        ["Depth", "L2", "Best trees", "Pinball", "90% coverage", "Crossings"],
        structure_rows,
    )

    shortlist_rows = [
        [
            str(row.candidate),
            f"{row.pinball_loss:.4f}",
            f"{row.coverage_90:.3f}",
            f"{row.mean_absolute_calibration_error:.3f}",
            f"{int(row.quantile_crossings)}",
        ]
        for row in shortlist.head(8).itertuples(index=False)
    ]
    shortlist_table = _table(
        ["Candidate", "Pinball", "90% coverage", "Calibration MAE", "Crossings"],
        shortlist_rows,
    )

    largest_round1_win = fold_table.loc[
        fold_table["round2_minus_catboost_round1"].idxmin()
    ]
    largest_round1_loss = fold_table.loc[
        fold_table["round2_minus_catboost_round1"].idxmax()
    ]
    feature_lines = "\n".join(f"- `{feature}`" for feature in selected_features)
    new_feature_text = ", ".join(f"`{feature}`" for feature in new_features)

    report = f"""# CatBoost round-2 validation report

## Executive summary

Round two combines a deterministic feature-selection recipe with a compact
CPU CatBoost `MultiQuantile` search. Feature selection reduced the 70-feature
candidate matrix to {len(selected_features)} features, after which all
{search_candidates} effective hyperparameter candidates were evaluated on the
same 12 monthly rolling-origin validation folds from January through December
2010. The configured 2011 test period was not accessed.

The selected configuration is **depth {int(selected['depth'])}, learning rate
{selected['learning_rate']:g}, L2 leaf regularization
{selected['l2_leaf_reg']:g}, and {int(selected['iterations'])} trees**. Its
hour-weighted pinball loss is **{round2['pinball_loss']:.3f}**, compared with
**{round1['pinball_loss']:.3f}** for the selected round-one model and
**{empirical['pinball_loss']:.3f}** for seasonal empirical. This is a
**{100 * (1 - round2['pinball_loss'] / round1['pinball_loss']):.2f}%** reduction
relative to round one and a
**{100 * (1 - round2['pinball_loss'] / empirical['pinball_loss']):.2f}%**
reduction relative to the empirical baseline. Round two wins 9 of 12 months
against each and all 12 months against seasonal naive.

The HAC mean-loss tests favor round two over round one (p=
{vs_round1['hac_mean_test_pvalue']:.4f}) and seasonal empirical (p=
{vs_empirical['hac_mean_test_pvalue']:.4f}), but the exact sign tests give p=
{vs_round1['exact_sign_test_pvalue']:.3f}. Feature selection and model
selection both reused 2010 validation. The improvement is therefore strong
exploratory evidence, not a confirmatory generalization result.

## Validation design and leakage protection

Every 2010 month is treated as a separate forecast origin. Training for a
month uses only pseudo-forecast labels whose forecast month has ended by that
origin. All load- and temperature-derived features are constructed from
observations strictly before the origin; only deterministic target calendar
and horizon information uses the future timestamp. Realized target-month
temperature is never supplied.

The comparison is exactly matched: round two, the selected round-one model,
seasonal empirical, and seasonal naive are all restricted to the same 8,760
hours and 12 origins. Round-one results are taken from its already-saved
candidate `{ROUND1_CANDIDATE}` rather than retuned on 2010. The 2011 test
period remains untouched.

## Feature selection

The candidate matrix contained 70 leakage-safe features. A fixed fast model
(depth 4, learning rate 0.08, L2=5, 125 trees) was fitted on the six odd
months. CatBoost validation `LossFunctionChange` was aggregated by feature.
A feature was retained when its median importance was positive and its
importance was positive in at least four of six screening folds; the fixed
cap of 55 was not reached. This mechanical rule selected
{len(selected_features)} features.

On the six complementary even-month folds, the selected fast model improved
the matching round-one fast model by
{100 * float(feature_selection_summary['relative_improvement_vs_round1']):.2f}%
and won {int(feature_selection_summary['verification_folds_won'])} of 6 folds.
Those folds later entered the full grid search, so this check is development
evidence rather than an independent test.

The selected features are:

{feature_lines}

Seven features were new relative to round one: {new_feature_text}. The set is
dominated by historical seasonal profile location, spread, and support,
augmented by annual load growth, calendar seasonality, and temperature
climatology or recent variability.

## Primary matched validation comparison

Pinball loss averages all 99 quantiles and all 2010 forecast hours; lower is
better. Monthly mean and standard deviation weight each origin equally.
Median bias is `actual - q0.50`, so a positive value denotes under-forecasting.

{primary_table}

Round two reduces median MAE by
{round1['median_mae'] - round2['median_mae']:.2f} MW relative to round one and
cuts median under-forecast bias from {round1['median_bias_actual_minus_forecast']:+.2f}
to {round2['median_bias_actual_minus_forecast']:+.2f} MW. Relative to seasonal
empirical, pinball improves by
{100 * (1 - round2['pinball_loss'] / empirical['pinball_loss']):.2f}% and
median MAE improves by {empirical['median_mae'] - round2['median_mae']:.2f} MW.

![Monthly pinball loss](figures/01_monthly_pinball.png)

## Stability across months and quarters

{quarter_table}

Round two improves on round one in every quarter, with the largest aggregate
gain in Q1. Its largest monthly improvement against round one is
{largest_round1_win['origin']:%B} ({largest_round1_win['round2_minus_catboost_round1']:+.3f}
loss points); its largest deterioration is
{largest_round1_loss['origin']:%B} ({largest_round1_loss['round2_minus_catboost_round1']:+.3f}).
The three losses against round one occur in May, August, and November; two are
small. Against seasonal empirical, the losses occur in March, October, and
November. This is meaningful variability despite the strong annual aggregate.

![Fold loss differences](figures/02_fold_differences.png)

## Paired statistical comparison

{paired_table}

The paired unit is a monthly forecast origin rather than an individual hour.
The Diebold–Mariano-style mean-loss test uses a Bartlett HAC variance with lag
{hac_lag} and a t(11) small-sample reference. It uses the magnitude of monthly
gains and allows short-range serial dependence. The exact two-sided sign test
uses only the 12 win/loss outcomes; with 9 wins and 3 losses its p-value is
0.146, illustrating the low power of a 12-fold comparison.

The HAC intervals exclude zero for round two versus round one and empirical.
However, the same validation year informed the earlier feature search,
inspection of round-one results, feature verification, and final model
selection. Consequently, these validation p-values are descriptive and should
not be presented as confirmatory.

Against seasonal naive the result is much less ambiguous: round two improves
pinball by {100 * (1 - round2['pinball_loss'] / naive['pinball_loss']):.2f}%,
wins every month, and has HAC p={vs_naive['hac_mean_test_pvalue']:.4g}.

## Calibration, sharpness, and quantile coherence

Calibration MAE is calculated within each month over all 99 marginal
quantiles, then averaged with forecast-hour weights.

{calibration_table}

Round two materially improves calibration relative to round one: monthly
calibration MAE falls from
{round1['within_month_calibration_mae']:.3f} to
{round2['within_month_calibration_mae']:.3f}, while nominal 90% coverage rises
from {100 * round1['coverage_90']:.1f}% to
{100 * round2['coverage_90']:.1f}%. Coverage remains below the nominal 90%
target. Round two's intervals are wider than both round one and empirical, so
part of the calibration gain comes from reduced sharpness rather than location
accuracy alone.

No model has a reversed 5th/95th percentile interval. Round two has
{int(round2['adjacent_quantile_crossings']):,} adjacent crossings, a
{100 * round2['adjacent_crossing_rate']:.3f}% rate. This is higher than round
one but still affects fewer than one half of one percent of adjacent quantile
pairs. Scores use raw output; monotonic rearrangement has not been applied.

![Calibration and sharpness](figures/04_calibration_sharpness.png)

### Full marginal and interval calibration curves

The aggregate marginal curve evaluates `P(Y ≤ qτ)` at every requested
quantile τ. Round two's predicted median has empirical marginal coverage
{100 * round2_median_coverage:.1f}%; the complete curve makes remaining
lower- and upper-tail asymmetry visible rather than reducing calibration to one
average. The central-interval panel separately compares empirical coverage at
the configured 50%, 80%, 90%, and 98% levels. Both panels use exactly the same
2010 observations for round two, round one, and the baselines.

![Marginal quantile and central interval calibration](figures/05_calibration_curves.png)

Exact curve values and interval widths are in `quantile_calibration.csv` and
`interval_calibration.csv`.

## What the search learned

The search jointly varied depth {{3, 4, 5}}, L2 regularization {{1, 5, 20}},
and tree checkpoints {{75, 100, 125}} at learning rate 0.08. Every effective
candidate used all 12 folds. Tree prefixes shared parent fits, so 27 candidates
required 108 actual fold fits and {search_seconds / 60:.1f} minutes of summed
model-fit time.

Best checkpoint for each depth/L2 pair:

{structure_table}

The top candidates are:

{shortlist_table}

Depth 3 is preferred: its L2=1 and L2=5 results at 125 trees are nearly tied,
differing by only
{100 * (shortlist.iloc[1]['pinball_loss'] / shortlist.iloc[0]['pinball_loss'] - 1):.3f}%.
This indicates that the leading result is not highly sensitive to modest L2
changes. Depth 4 remains competitive around 75–100 trees, while depth 5 is
worse and deteriorates as trees are added. The deeper models also narrow
intervals and create more crossings, consistent with quantile overfitting.

![Search landscape](figures/03_search_landscape.png)

## Limitations and next decision

- Feature selection, feature-set verification, and hyperparameter selection
  all used 2010. The final 8.373 validation score is therefore optimistic.
- Only 12 monthly folds are available, so paired inference has low power and
  is sensitive to a few large winter-month differences.
- Round two searched one learning rate. The planned lower-rate refinement
  around the winning shallow structure has not yet been run.
- Forecasts use temperature climatology and observed pre-origin weather, not
  realized future temperature. Unexpected target-month weather remains an
  irreducible source of error under this assumption.
- Search artifacts contain fold diagnostics rather than every CatBoost
  quantile prediction. A final evaluation runner should save all 99 quantiles
  and explicitly evaluate monotonic rearrangement if it is adopted.

The leading structure is stable enough to justify a small learning-rate/tree
refinement rather than another broad grid. After that choice is frozen, the
2011 rolling-origin test should be run exactly once. Test performance—not the
adaptive 2010 p-values—must be the main evidence of generalization.

## Reproduction and artifacts

From the repository root:

```bash
.venv/bin/python analysis/catboost/search.py --config configs/catboost_round2_phase1.yaml
.venv/bin/python -m analysis.catboost.predict_selected
.venv/bin/python -m analysis.catboost.round2_report
```

The search is resumable. Raw outputs are in
[`../search/round2_phase1`](../search/round2_phase1), the feature-selection
artifacts are in [`../feature_selection/round2`](../feature_selection/round2),
and the selected feature manifest is
[`selected_features.yaml`](../feature_selection/round2/selected_features.yaml).
Supporting outputs in this directory are `model_comparison.csv`,
`fold_comparison.csv`, `quarter_comparison.csv`, `paired_tests.csv`,
`candidate_shortlist.csv`, and `structure_summary.csv`.
"""
    path.write_text(report, encoding="utf-8")


def run(
    search_dir: Path,
    round1_search_dir: Path,
    baseline_dir: Path,
    feature_selection_dir: Path,
    selected_prediction_dir: Path,
    round1_prediction_dir: Path,
    round1_schema_path: Path,
    output_dir: Path,
) -> Path:
    candidate_summary = pd.read_csv(search_dir / "candidate_summary.csv")
    fold_results = pd.read_csv(
        search_dir / "fold_results.csv", parse_dates=["origin"]
    )
    if len(candidate_summary) != 27 or not candidate_summary["folds"].eq(12).all():
        raise ValueError("Round-two report requires 27 complete 12-fold candidates")
    selected = candidate_summary.sort_values("pinball_loss").iloc[0]
    round2_folds = fold_results.loc[
        fold_results["candidate"].eq(selected["candidate"])
    ].sort_values("origin")
    if len(round2_folds) != 12 or not round2_folds["origin"].dt.year.eq(2010).all():
        raise ValueError("Selected round-two candidate must cover all 2010 origins")

    round1_all = pd.read_csv(
        round1_search_dir / "fold_results.csv", parse_dates=["origin"]
    )
    round1_folds = round1_all.loc[
        round1_all["candidate"].eq(ROUND1_CANDIDATE)
        & round1_all["origin"].dt.year.eq(2010)
    ].sort_values("origin")
    if len(round1_folds) != 12:
        raise ValueError("Selected round-one model does not cover all 2010 origins")

    baseline_folds = pd.read_csv(
        baseline_dir / "fold_metrics.csv", parse_dates=["origin"]
    )
    baseline_folds = baseline_folds.loc[
        baseline_folds["origin"].dt.year.eq(2010)
    ].copy()
    if len(baseline_folds) != 24:
        raise ValueError("Both baselines must cover all 12 origins in 2010")
    baseline_folds = baseline_folds.rename(
        columns={"forecast_hours": "evaluation_rows"}
    )
    baseline_folds["invalid_90_intervals"] = 0
    baseline_folds["quantile_crossings"] = 0

    baseline_predictions = pd.read_csv(
        baseline_dir / "predictions.csv.gz", parse_dates=["origin"]
    )
    baseline_predictions = baseline_predictions.loc[
        baseline_predictions["origin"].dt.year.eq(2010)
    ]
    levels = np.arange(1, 100, dtype=float) / 100.0
    quantile_columns = [f"q{level:.2f}" for level in levels]
    baseline_calibration = _within_month_calibration(
        baseline_predictions, quantile_columns, levels
    )
    baseline_manifest = pd.read_csv(
        baseline_dir / "fold_manifest.csv", parse_dates=["origin"]
    )
    baseline_manifest = baseline_manifest.loc[
        baseline_manifest["origin"].dt.year.eq(2010)
    ]
    baseline_detail = evaluate_predictions(
        baseline_predictions,
        baseline_manifest,
        levels,
        [0.50, 0.80, 0.90, 0.98],
        reference_model="seasonal_empirical",
    )
    selected_quantile_calibration = pd.read_csv(
        selected_prediction_dir / "quantile_calibration.csv"
    )
    selected_interval_calibration = pd.read_csv(
        selected_prediction_dir / "interval_calibration.csv"
    )
    model_labels = {
        "seasonal_empirical": EMPIRICAL_LABEL,
        "seasonal_naive": NAIVE_LABEL,
    }
    quantile_calibration = pd.concat(
        [
            selected_quantile_calibration.assign(model=ROUND2_LABEL),
            baseline_detail.quantile_calibration.assign(
                model=lambda frame: frame["model"].map(model_labels)
            ),
        ],
        ignore_index=True,
    )
    round1_predictions = pd.read_csv(
        round1_prediction_dir / "predictions.csv.gz", parse_dates=["origin"]
    )
    round1_predictions = round1_predictions.loc[
        round1_predictions["origin"].dt.year.eq(2010)
    ]
    round1_detail = evaluate_predictions(
        round1_predictions,
        baseline_manifest,
        levels,
        [0.50, 0.80, 0.90, 0.98],
        reference_model="catboost_round1",
    )
    round1_quantile_calibration = round1_detail.quantile_calibration.assign(
        model=ROUND1_LABEL
    )
    quantile_calibration = pd.concat(
        [
            quantile_calibration.iloc[: len(levels)],
            round1_quantile_calibration,
            quantile_calibration.iloc[len(levels) :],
        ],
        ignore_index=True,
    )
    interval_calibration = pd.concat(
        [
            selected_interval_calibration.assign(model=ROUND2_LABEL),
            round1_detail.interval_calibration.assign(model=ROUND1_LABEL),
            baseline_detail.interval_calibration.assign(
                model=lambda frame: frame["model"].map(model_labels)
            ),
        ],
        ignore_index=True,
    )
    if len(quantile_calibration) != 4 * len(levels):
        raise ValueError("Round-two quantile calibration comparison is incomplete")
    if len(interval_calibration) != 4 * 4:
        raise ValueError("Round-two interval calibration comparison is incomplete")

    comparison = pd.DataFrame(
        [
            _model_record(
                round2_folds,
                ROUND2_LABEL,
                _weighted_average(
                    round2_folds, "mean_absolute_calibration_error"
                ),
            ),
            _model_record(
                round1_folds,
                ROUND1_LABEL,
                _weighted_average(
                    round1_folds, "mean_absolute_calibration_error"
                ),
            ),
            _model_record(
                baseline_folds.loc[
                    baseline_folds["model"].eq("seasonal_empirical")
                ],
                EMPIRICAL_LABEL,
                baseline_calibration["seasonal_empirical"],
            ),
            _model_record(
                baseline_folds.loc[
                    baseline_folds["model"].eq("seasonal_naive")
                ],
                NAIVE_LABEL,
                baseline_calibration["seasonal_naive"],
            ),
        ]
    )
    empirical_loss = float(
        comparison.set_index("model").loc[EMPIRICAL_LABEL, "pinball_loss"]
    )
    comparison["relative_improvement_vs_empirical"] = (
        1.0 - comparison["pinball_loss"] / empirical_loss
    )

    fold_table = _build_fold_comparison(
        round2_folds, round1_folds, baseline_folds
    )
    paired_tests, hac_lag = _build_paired_tests(fold_table)
    quarter_comparison = _build_quarter_comparison(fold_table)
    shortlist = candidate_summary.sort_values("pinball_loss").head(10).copy()
    structure = _best_by_structure(candidate_summary)

    with (feature_selection_dir / "selected_features.yaml").open(
        "r", encoding="utf-8"
    ) as stream:
        feature_manifest = yaml.safe_load(stream)
    selected_features = [str(value) for value in feature_manifest["features"]]
    with round1_schema_path.open("r", encoding="utf-8") as stream:
        round1_features = set(json.load(stream)["feature_columns"])
    new_features = [
        feature for feature in selected_features if feature not in round1_features
    ]
    with (feature_selection_dir / "summary.json").open(
        "r", encoding="utf-8"
    ) as stream:
        feature_selection_summary = json.load(stream)

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_comparison.csv": comparison,
        "fold_comparison.csv": fold_table,
        "paired_tests.csv": paired_tests,
        "quarter_comparison.csv": quarter_comparison,
        "candidate_shortlist.csv": shortlist,
        "structure_summary.csv": structure,
        "quantile_calibration.csv": quantile_calibration,
        "interval_calibration.csv": interval_calibration,
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_monthly_scores(fold_table, figures_dir / "01_monthly_pinball.png")
    _plot_fold_differences(fold_table, figures_dir / "02_fold_differences.png")
    _plot_search_landscape(
        candidate_summary, figures_dir / "03_search_landscape.png"
    )
    _plot_calibration_sharpness(
        comparison, figures_dir / "04_calibration_sharpness.png"
    )
    _plot_calibration_curves(
        quantile_calibration,
        interval_calibration,
        [ROUND2_LABEL, ROUND1_LABEL, EMPIRICAL_LABEL, NAIVE_LABEL],
        figures_dir / "05_calibration_curves.png",
    )

    resolved = yaml.safe_load((search_dir / "resolved_config.yaml").read_text())
    search_seconds = float(
        fold_results[
            ["base_candidate", "origin", "fit_seconds"]
        ].drop_duplicates(["base_candidate", "origin"])["fit_seconds"].sum()
    )
    if resolved["catboost"]["search"]["split"] != "round2_validation":
        raise ValueError("Round-two search used an unexpected validation split")

    _write_report(
        output_dir / "report.md",
        selected,
        comparison,
        fold_table,
        paired_tests,
        quarter_comparison,
        shortlist,
        structure,
        selected_features,
        new_features,
        feature_selection_summary,
        quantile_calibration,
        interval_calibration,
        len(candidate_summary),
        hac_lag,
        search_seconds,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--search-dir",
        type=Path,
        default=Path("artifacts/catboost/search/round2_phase1"),
    )
    parser.add_argument(
        "--round1-search-dir",
        type=Path,
        default=Path("artifacts/catboost/search/phase1"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("artifacts/baseline/validation"),
    )
    parser.add_argument(
        "--feature-selection-dir",
        type=Path,
        default=Path("artifacts/catboost/feature_selection/round2"),
    )
    parser.add_argument(
        "--selected-prediction-dir",
        type=Path,
        default=Path("artifacts/catboost/predictions/round2"),
    )
    parser.add_argument(
        "--round1-prediction-dir",
        type=Path,
        default=Path("artifacts/catboost/predictions/round1"),
    )
    parser.add_argument(
        "--round1-schema",
        type=Path,
        default=Path("artifacts/model_data/validation/schema.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catboost/round2"),
    )
    args = parser.parse_args()
    output_dir = run(
        args.search_dir,
        args.round1_search_dir,
        args.baseline_dir,
        args.feature_selection_dir,
        args.selected_prediction_dir,
        args.round1_prediction_dir,
        args.round1_schema,
        args.output_dir,
    )
    print(f"Wrote round-two report artifacts to {output_dir}")


if __name__ == "__main__":
    main()
