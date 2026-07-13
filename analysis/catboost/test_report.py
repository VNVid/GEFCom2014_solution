"""Create the locked four-method 2011 rolling-origin test report."""

from __future__ import annotations

import argparse
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

from analysis.catboost.report import (
    _exact_sign_test,
    _hac_mean_loss_test,
    _plot_calibration_curves,
)


ROUND2_LABEL = "CatBoost round 2"
ROUND1_LABEL = "CatBoost round 1"
EMPIRICAL_LABEL = "Seasonal empirical"
NAIVE_LABEL = "Seasonal naive"
MODEL_ORDER = [ROUND2_LABEL, ROUND1_LABEL, EMPIRICAL_LABEL, NAIVE_LABEL]
MODEL_NAMES = {
    "catboost_round2": ROUND2_LABEL,
    "catboost_round1": ROUND1_LABEL,
    "seasonal_empirical": EMPIRICAL_LABEL,
    "seasonal_naive": NAIVE_LABEL,
}
PAIRWISE_COMPARISONS = (
    (ROUND2_LABEL, EMPIRICAL_LABEL, "primary"),
    (ROUND2_LABEL, ROUND1_LABEL, "secondary"),
    (ROUND2_LABEL, NAIVE_LABEL, "secondary"),
    (ROUND1_LABEL, EMPIRICAL_LABEL, "secondary"),
    (ROUND1_LABEL, NAIVE_LABEL, "secondary"),
    (EMPIRICAL_LABEL, NAIVE_LABEL, "secondary"),
)


def _load_method_tables(directory: Path) -> dict[str, pd.DataFrame]:
    return {
        "aggregate": pd.read_csv(directory / "aggregate_metrics.csv"),
        "fold": pd.read_csv(directory / "fold_metrics.csv", parse_dates=["origin"]),
        "quantile": pd.read_csv(directory / "quantile_calibration.csv"),
        "interval": pd.read_csv(directory / "interval_calibration.csv"),
    }


def _relabel(frame: pd.DataFrame) -> pd.DataFrame:
    relabeled = frame.copy()
    relabeled["model"] = relabeled["model"].map(MODEL_NAMES)
    if relabeled["model"].isna().any():
        raise ValueError("Unknown model name in test artifacts")
    return relabeled


def _build_comparison(
    aggregate: pd.DataFrame, interval: pd.DataFrame
) -> pd.DataFrame:
    interval_90 = interval.loc[np.isclose(interval["nominal_coverage"], 0.9)].set_index(
        "model"
    )
    records = []
    for model in MODEL_ORDER:
        row = aggregate.loc[aggregate["model"].eq(model)].iloc[0]
        central = interval_90.loc[model]
        hours = int(row["forecast_hours"])
        records.append(
            {
                "model": model,
                "folds": int(row["folds"]),
                "forecast_hours": hours,
                "pinball_loss": float(row["pinball_loss"]),
                "fold_pinball_mean": float(row["fold_pinball_mean"]),
                "fold_pinball_std": float(row["fold_pinball_std"]),
                "median_mae": float(row["median_mae"]),
                "median_bias_actual_minus_forecast": float(row["median_bias"]),
                "global_calibration_mae": float(
                    row["mean_absolute_calibration_error"]
                ),
                "max_absolute_calibration_error": float(
                    row["max_absolute_calibration_error"]
                ),
                "coverage_90": float(central["empirical_coverage"]),
                "mean_width_90": float(central["mean_width"]),
                "quantile_crossings": int(row["quantile_crossings"]),
                "crossing_rate": int(row["quantile_crossings"]) / (hours * 98),
            }
        )
    comparison = pd.DataFrame(records)
    empirical_loss = float(
        comparison.set_index("model").loc[EMPIRICAL_LABEL, "pinball_loss"]
    )
    comparison["relative_improvement_vs_empirical"] = (
        1.0 - comparison["pinball_loss"] / empirical_loss
    )
    return comparison


def _build_fold_table(folds: pd.DataFrame) -> pd.DataFrame:
    scores = folds.pivot(index="origin", columns="model", values="pinball_loss")
    hours = folds.pivot(index="origin", columns="model", values="forecast_hours")
    if not hours.nunique(axis=1).eq(1).all():
        raise ValueError("Test methods do not cover identical hours by origin")
    table = scores.loc[:, MODEL_ORDER].copy()
    table.insert(0, "hours", hours.iloc[:, 0])
    table = table.reset_index()
    if len(table) != 12 or not table["origin"].dt.year.eq(2011).all():
        raise ValueError("Test comparison must contain all 12 origins in 2011")
    return table


def _build_paired_tests(fold_table: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    count = len(fold_table)
    lag = int(np.floor(4.0 * (count / 100.0) ** (2.0 / 9.0)))
    weights = fold_table["hours"].to_numpy(dtype=float)
    records = []
    for model, reference, role in PAIRWISE_COMPARISONS:
        differences = (fold_table[model] - fold_table[reference]).to_numpy(dtype=float)
        wins, losses, ties, sign_pvalue = _exact_sign_test(differences)
        mean, statistic, hac_pvalue, lower, upper = _hac_mean_loss_test(
            differences, lag
        )
        model_loss = float(np.average(fold_table[model], weights=weights))
        reference_loss = float(np.average(fold_table[reference], weights=weights))
        records.append(
            {
                "comparison_role": role,
                "model": model,
                "reference_model": reference,
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
        record: dict[str, object] = {"quarter": f"Q{quarter}", "months": len(group)}
        for model in MODEL_ORDER:
            record[model] = float(np.average(group[model], weights=weights))
        record["round2_improvement_vs_empirical"] = 1.0 - (
            float(record[ROUND2_LABEL]) / float(record[EMPIRICAL_LABEL])
        )
        record["round2_improvement_vs_round1"] = 1.0 - (
            float(record[ROUND2_LABEL]) / float(record[ROUND1_LABEL])
        )
        records.append(record)
    return pd.DataFrame(records)


def _plot_monthly(fold_table: pd.DataFrame, path: Path) -> None:
    colors = ["#2563eb", "#7c3aed", "#d97706", "#6b7280"]
    markers = ["o", "D", "s", "^"]
    figure, axis = plt.subplots(figsize=(12, 5.5))
    for model, color, marker in zip(MODEL_ORDER, colors, markers):
        axis.plot(
            fold_table["origin"],
            fold_table[model],
            label=model,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
        )
    axis.set(title="2011 monthly rolling-origin pinball loss", ylabel="Pinball loss")
    axis.set_xlabel("")
    axis.legend(frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.25)
    figure.autofmt_xdate()
    figure.tight_layout()
    figure.savefig(path, dpi=170)
    plt.close(figure)


def _plot_primary_differences(fold_table: pd.DataFrame, path: Path) -> None:
    labels = fold_table["origin"].dt.strftime("%Y-%m")
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5.5))
    axis.bar(
        positions - width / 2,
        fold_table[ROUND2_LABEL] - fold_table[ROUND1_LABEL],
        width,
        label="Round 2 minus round 1",
        color="#7c3aed",
    )
    axis.bar(
        positions + width / 2,
        fold_table[ROUND2_LABEL] - fold_table[EMPIRICAL_LABEL],
        width,
        label="Round 2 minus empirical",
        color="#d97706",
    )
    axis.axhline(0.0, color="#111827", linewidth=1)
    axis.set(
        title="Primary test loss differences",
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


def _plot_calibration_summary(comparison: pd.DataFrame, path: Path) -> None:
    colors = ["#2563eb", "#7c3aed", "#d97706", "#6b7280"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.8))
    for axis, column, title, ylabel in (
        (axes[0], "coverage_90", "90% interval coverage", "Coverage"),
        (axes[1], "mean_width_90", "90% interval width", "Mean width (MW)"),
        (axes[2], "global_calibration_mae", "Global calibration MAE", "MAE"),
    ):
        sns.barplot(
            data=comparison,
            x="model",
            y=column,
            hue="model",
            order=MODEL_ORDER,
            palette=colors,
            legend=False,
            ax=axis,
        )
        axis.set(title=title, xlabel="", ylabel=ylabel)
        axis.tick_params(axis="x", rotation=24)
        axis.grid(axis="y", alpha=0.2)
    axes[0].axhline(0.9, color="#111827", linestyle=":", linewidth=1.4)
    axes[0].set_ylim(0, 1)
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
    comparison: pd.DataFrame,
    folds: pd.DataFrame,
    paired: pd.DataFrame,
    quarters: pd.DataFrame,
    intervals: pd.DataFrame,
    quantiles: pd.DataFrame,
    validation_comparison: pd.DataFrame,
    hac_lag: int,
) -> None:
    models = comparison.set_index("model")
    round2 = models.loc[ROUND2_LABEL]
    round1 = models.loc[ROUND1_LABEL]
    empirical = models.loc[EMPIRICAL_LABEL]
    naive = models.loc[NAIVE_LABEL]
    primary = paired.loc[
        paired["model"].eq(ROUND2_LABEL)
        & paired["reference_model"].eq(EMPIRICAL_LABEL)
    ].iloc[0]
    secondary = paired.loc[
        paired["model"].eq(ROUND2_LABEL)
        & paired["reference_model"].eq(ROUND1_LABEL)
    ].iloc[0]

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
        [
            [
                model,
                f"{models.loc[model, 'pinball_loss']:.3f}",
                f"{100 * models.loc[model, 'relative_improvement_vs_empirical']:+.2f}",
                f"{models.loc[model, 'fold_pinball_mean']:.3f}",
                f"{models.loc[model, 'fold_pinball_std']:.3f}",
                f"{models.loc[model, 'median_mae']:.2f}",
                f"{models.loc[model, 'median_bias_actual_minus_forecast']:+.2f}",
            ]
            for model in MODEL_ORDER
        ],
    )
    quarter_table = _table(
        ["Quarter", "Round 2", "Round 1", "Empirical", "Naive", "R2 vs empirical (%)"],
        [
            [
                str(row["quarter"]),
                f"{row[ROUND2_LABEL]:.3f}",
                f"{row[ROUND1_LABEL]:.3f}",
                f"{row[EMPIRICAL_LABEL]:.3f}",
                f"{row[NAIVE_LABEL]:.3f}",
                f"{100 * row['round2_improvement_vs_empirical']:+.2f}",
            ]
            for _, row in quarters.iterrows()
        ],
    )
    paired_table = _table(
        ["Role", "Model", "Reference", "Mean diff", "Wins", "Losses", "HAC p", "Sign p", "HAC 95% CI"],
        [
            [
                str(row.comparison_role),
                str(row.model),
                str(row.reference_model),
                f"{row.mean_monthly_loss_difference:+.3f}",
                str(int(row.folds_won)),
                str(int(row.folds_lost)),
                f"{row.hac_mean_test_pvalue:.4g}",
                f"{row.exact_sign_test_pvalue:.4g}",
                f"[{row.hac_mean_difference_ci95_lower:.3f}, {row.hac_mean_difference_ci95_upper:.3f}]",
            ]
            for row in paired.itertuples(index=False)
        ],
    )
    interval_table = _table(
        ["Model", "Nominal", "Coverage", "Error", "Mean width"],
        [
            [
                str(row.model),
                f"{100 * row.nominal_coverage:.0f}%",
                f"{100 * row.empirical_coverage:.1f}%",
                f"{100 * row.coverage_error:+.1f} pp",
                f"{row.mean_width:.2f}",
            ]
            for row in intervals.itertuples(index=False)
        ],
    )
    coherence_table = _table(
        ["Model", "Calibration MAE", "Max calibration error", "Crossings", "Crossing rate (%)"],
        [
            [
                model,
                f"{models.loc[model, 'global_calibration_mae']:.3f}",
                f"{models.loc[model, 'max_absolute_calibration_error']:.3f}",
                f"{int(models.loc[model, 'quantile_crossings'])}",
                f"{100 * models.loc[model, 'crossing_rate']:.3f}",
            ]
            for model in MODEL_ORDER
        ],
    )
    validation = validation_comparison.set_index("model")
    validation_table = _table(
        ["Model", "2010 validation", "2011 test", "Test change (%)"],
        [
            [
                model,
                f"{validation.loc[model, 'pinball_loss']:.3f}",
                f"{models.loc[model, 'pinball_loss']:.3f}",
                f"{100 * (models.loc[model, 'pinball_loss'] / validation.loc[model, 'pinball_loss'] - 1):+.2f}",
            ]
            for model in MODEL_ORDER
        ],
    )
    median_coverage = quantiles.loc[np.isclose(quantiles["quantile"], 0.5)].set_index(
        "model"
    )["empirical_coverage"]
    primary_differences = folds[ROUND2_LABEL] - folds[EMPIRICAL_LABEL]
    secondary_differences = folds[ROUND2_LABEL] - folds[ROUND1_LABEL]

    report = f"""# Locked 2011 rolling-origin test report

## Executive summary

This report is the first and only evaluation on the configured 2011 test
period. The specifications of all four methods were frozen before test access:
seasonal naive, seasonal empirical, the selected round-one CatBoost, and the
selected round-two CatBoost. Round two was declared the primary final model;
round one was retained as a pre-specified comparator rather than selected
after viewing test results.

The primary round-two model obtains hour-weighted pinball loss
**{round2['pinball_loss']:.3f}**, compared with
**{empirical['pinball_loss']:.3f}** for seasonal empirical,
**{round1['pinball_loss']:.3f}** for round one, and
**{naive['pinball_loss']:.3f}** for seasonal naive. Round two is
**{100 * (round2['pinball_loss'] / empirical['pinball_loss'] - 1):.2f}% worse**
than the primary empirical reference and
**{100 * (1 - round2['pinball_loss'] / round1['pinball_loss']):.2f}% better**
than round one.

Against seasonal empirical, round two wins {int(primary['folds_won'])} of 12
months; the HAC mean-loss p-value is {primary['hac_mean_test_pvalue']:.4g} and
the exact sign-test p-value is {primary['exact_sign_test_pvalue']:.4g}.
Against round one it wins {int(secondary['folds_won'])} months, with HAC p=
{secondary['hac_mean_test_pvalue']:.4g}. These test estimates were computed
only after model and hyperparameter selection was completed on validation.
The report nevertheless emphasizes effect sizes and monthly stability because
only 12 test folds are available.

## Locked evaluation protocol

Each 2011 month is forecast as a separate rolling origin. January is trained
using pseudo-origin labels available through December 2010. After January's
predictions are saved, January outcomes become eligible for the February fit,
and so on through December. Training therefore grows from 35,064 labeled
pseudo-forecast rows in January to 43,080 in December.

This is a production-style sequence of twelve one-month-ahead forecasts, not
one forecast made in January for the entire year. The forecasting procedure is
frozen, while CatBoost is refitted at each origin using only information then
available. Thus a revealed test month may train a later-origin model, but can
never affect its own forecast or an earlier forecast.

For every origin, all load and temperature features use observations strictly
before that origin. Realized target-month temperature is never supplied. The
round-one model uses its frozen 45 features with depth 4, learning rate 0.04,
L2=5, and 250 trees. Round two uses its frozen 17 selected features with depth
3, learning rate 0.08, L2=5, and 125 trees. Baseline definitions are unchanged.
No test score was used for feature selection, hyperparameter selection,
calibration correction, ensembling, or any other tuning decision. Earlier
2011 observations enter later fits only through the predeclared rolling-update
rule described above.

## Primary metrics

Pinball loss averages all 99 quantiles and all 8,760 test hours. Monthly mean
and standard deviation weight each origin equally. Median bias is
`actual - q0.50`, so positive values denote under-forecasting.

{primary_table}

On the realized 2011 sample, seasonal empirical has the lowest aggregate loss
and median MAE. The two CatBoost models are practically tied: round two lowers
annual loss by only {100 * (1 - round2['pinball_loss'] / round1['pinball_loss']):.2f}%.
The primary statistical comparison does not establish a difference between
round two and empirical at conventional significance levels, so the result is
best described as a test-set reversal in observed ranking rather than evidence
that one method is uniformly superior.

![Monthly test pinball loss](figures/01_monthly_pinball.png)

## Monthly and seasonal stability

{quarter_table}

Round two's largest monthly gain against empirical is
{primary_differences.min():+.3f} pinball points in
{folds.loc[primary_differences.idxmin(), 'origin']:%B}; its largest loss is
{primary_differences.max():+.3f} in
{folds.loc[primary_differences.idxmax(), 'origin']:%B}. Against round one, the
range is {secondary_differences.min():+.3f} to
{secondary_differences.max():+.3f}. This fold-level variation is essential
context for the annual aggregate.

![Primary paired differences](figures/02_primary_differences.png)

## Paired statistical comparisons

{paired_table}

The paired unit is one monthly origin. Each Diebold–Mariano-style mean-loss
test uses a Bartlett HAC variance with lag {hac_lag} and a t(11) small-sample
reference. The exact two-sided sign test discards magnitude and uses only
monthly win/loss direction. The round-two versus empirical comparison was
predeclared as primary and is interpreted separately. Model selection was
completed without 2011; the other displayed comparisons are secondary.

## Marginal calibration, intervals, and quantile coherence

Global marginal calibration MAE averages `|P(Y ≤ qτ) - τ|` across all 99
quantiles. The predicted-median empirical coverages are
{100 * median_coverage[ROUND2_LABEL]:.1f}% for round two,
{100 * median_coverage[ROUND1_LABEL]:.1f}% for round one,
{100 * median_coverage[EMPIRICAL_LABEL]:.1f}% for empirical, and
{100 * median_coverage[NAIVE_LABEL]:.1f}% for naive.

{coherence_table}

Central interval coverage and sharpness:

{interval_table}

Seasonal empirical is also the strongest calibrated probabilistic forecast on
this test year: its 90% interval covers {100 * empirical['coverage_90']:.1f}%
of outcomes and its global calibration MAE is
{empirical['global_calibration_mae']:.3f}. Round two improves materially on
round one's calibration and interval coverage, but remains under-dispersed.
That calibration gain comes with more raw adjacent quantile crossings
({100 * round2['crossing_rate']:.3f}% versus
{100 * round1['crossing_rate']:.3f}%); crossings are reported rather than
silently repaired. Seasonal naive repeats one point across all quantiles, so
its interval width is zero and its probabilistic calibration is correspondingly
poor.

![Calibration summary](figures/03_calibration_summary.png)

The full marginal curve shows calibration at every requested quantile; the
central-interval panel shows the configured 50%, 80%, 90%, and 98% levels.
All curves use the same 8,760 test outcomes.

![Marginal and central-interval calibration](figures/04_calibration_curves.png)

Exact values are in `quantile_calibration.csv` and
`interval_calibration.csv`. Scores and calibration use raw model output; no
monotonic rearrangement is applied.

## Validation-to-test comparison

{validation_table}

This table is diagnostic, not a new selection step. The 2010 column is the
matched validation period used in the round-two report; the 2011 column is the
locked test. Changes may reflect realized weather, load drift, and finite-fold
variation as well as generalization error.

## Limitations

- The test contains only 12 monthly origins, so confidence intervals and
  p-values remain sensitive to individual seasons.
- Historical and pre-origin temperature features cannot anticipate unusual
  target-month weather.
- Quantile forecasts are marginal by hour and do not define coherent temporal
  scenarios.
- Four pre-specified methods are displayed. Round two versus empirical is the
  primary comparison; other pairwise tests should be interpreted as secondary.
- The test must not be reused to tune a lower learning rate, change features,
  select post-processing, or fit ensemble weights.

## Reproduction and artifacts

From the repository root:

```bash
.venv/bin/python -m analysis.model_data.run --feature-config configs/features/round2_candidates.yaml --split test
.venv/bin/python -m analysis.baseline.run --split test
.venv/bin/python -m analysis.catboost.predict_selected --experiment round1_test --experiment round2_test
.venv/bin/python -m analysis.catboost.test_report
```

The CatBoost prediction runner is resumable by monthly origin. Complete
per-hour forecasts are under `artifacts/catboost/predictions/test/`; baseline
forecasts are under `artifacts/baseline/test/`. Supporting tables in this
directory are `model_comparison.csv`, `fold_comparison.csv`,
`quarter_comparison.csv`, `paired_tests.csv`, `quantile_calibration.csv`,
`interval_calibration.csv`, and `validation_to_test.csv`.
"""
    path.write_text(report, encoding="utf-8")


def run(
    round1_dir: Path,
    round2_dir: Path,
    baseline_dir: Path,
    validation_comparison_path: Path,
    output_dir: Path,
) -> Path:
    tables = [
        _load_method_tables(round2_dir),
        _load_method_tables(round1_dir),
        _load_method_tables(baseline_dir),
    ]
    aggregate = _relabel(pd.concat([table["aggregate"] for table in tables]))
    folds = _relabel(pd.concat([table["fold"] for table in tables]))
    quantiles = _relabel(pd.concat([table["quantile"] for table in tables]))
    intervals = _relabel(pd.concat([table["interval"] for table in tables]))
    if set(aggregate["model"]) != set(MODEL_ORDER) or len(aggregate) != 4:
        raise ValueError("Test report requires exactly four aggregate model rows")
    if len(folds) != 48 or len(quantiles) != 4 * 99 or len(intervals) != 4 * 4:
        raise ValueError("Test evaluation tables are incomplete")

    comparison = _build_comparison(aggregate, intervals)
    fold_table = _build_fold_table(folds)
    paired, hac_lag = _build_paired_tests(fold_table)
    quarters = _build_quarter_comparison(fold_table)
    validation_comparison = pd.read_csv(validation_comparison_path)
    if set(validation_comparison["model"]) != set(MODEL_ORDER):
        raise ValueError("Validation-to-test comparison is missing a method")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_comparison.csv": comparison,
        "fold_comparison.csv": fold_table,
        "quarter_comparison.csv": quarters,
        "paired_tests.csv": paired,
        "quantile_calibration.csv": quantiles,
        "interval_calibration.csv": intervals,
        "validation_to_test.csv": validation_comparison.loc[
            :, ["model", "pinball_loss"]
        ].merge(
            comparison.loc[:, ["model", "pinball_loss"]],
            on="model",
            suffixes=("_validation", "_test"),
            validate="one_to_one",
        ),
    }
    for filename, frame in outputs.items():
        frame.to_csv(output_dir / filename, index=False, float_format="%.10g")

    sns.set_theme(style="whitegrid", context="notebook")
    _plot_monthly(fold_table, figures_dir / "01_monthly_pinball.png")
    _plot_primary_differences(
        fold_table, figures_dir / "02_primary_differences.png"
    )
    _plot_calibration_summary(
        comparison, figures_dir / "03_calibration_summary.png"
    )
    _plot_calibration_curves(
        quantiles,
        intervals,
        MODEL_ORDER,
        figures_dir / "04_calibration_curves.png",
    )
    _write_report(
        output_dir / "report.md",
        comparison,
        fold_table,
        paired,
        quarters,
        intervals,
        quantiles,
        validation_comparison,
        hac_lag,
    )
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--round1-dir",
        type=Path,
        default=Path("artifacts/catboost/predictions/test/round1"),
    )
    parser.add_argument(
        "--round2-dir",
        type=Path,
        default=Path("artifacts/catboost/predictions/test/round2"),
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("artifacts/baseline/test"),
    )
    parser.add_argument(
        "--validation-comparison",
        type=Path,
        default=Path("artifacts/catboost/round2/model_comparison.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/catboost/test"),
    )
    args = parser.parse_args()
    output = run(
        args.round1_dir,
        args.round2_dir,
        args.baseline_dir,
        args.validation_comparison,
        args.output_dir,
    )
    print(f"Wrote locked test report artifacts to {output}")


if __name__ == "__main__":
    main()
