"""Aggregate probabilistic forecast diagnostics over identical monthly folds."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb

import numpy as np
import pandas as pd

from .metrics import (
    per_observation_pinball_loss,
    quantile_coverage,
    quantile_crossing_count,
)


@dataclass(frozen=True)
class EvaluationTables:
    """Named tables written by an experiment runner."""

    predictions: pd.DataFrame
    fold_metrics: pd.DataFrame
    aggregate_metrics: pd.DataFrame
    quantile_calibration: pd.DataFrame
    interval_calibration: pd.DataFrame
    paired_comparison: pd.DataFrame


def evaluate_predictions(
    predictions: pd.DataFrame,
    fold_manifest: pd.DataFrame,
    quantiles: np.ndarray,
    central_intervals: list[float],
    reference_model: str,
) -> EvaluationTables:
    """Evaluate all models on shared observations and monthly origins.

    Required prediction columns are ``model``, ``origin``, ``actual``, and one
    column per requested quantile. ``sample_size`` is an optional model-specific
    diagnostic used by empirical forecasts; other approaches may omit it.
    """

    levels = np.asarray(quantiles, dtype=float)
    quantile_columns = [f"q{level:.2f}" for level in levels]
    required = {"model", "origin", "actual", *quantile_columns}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Prediction table is missing columns {missing}")

    evaluated = predictions.copy()
    evaluated["pinball_loss"] = np.nan
    evaluated["median_absolute_error"] = np.nan
    evaluated["median_error"] = np.nan
    calibration_records: list[dict[str, float | str]] = []
    crossing_counts: dict[str, int] = {}

    median_positions = np.flatnonzero(np.isclose(levels, 0.5))
    if len(median_positions) != 1:
        raise ValueError("Exactly one 0.50 quantile is required")
    median_column = quantile_columns[int(median_positions[0])]

    interval_columns: dict[float, tuple[str, str]] = {}
    for nominal in central_intervals:
        tail = (1.0 - float(nominal)) / 2.0
        lower = np.flatnonzero(np.isclose(levels, tail))
        upper = np.flatnonzero(np.isclose(levels, 1.0 - tail))
        if len(lower) != 1 or len(upper) != 1:
            raise ValueError(f"Quantile grid cannot represent central interval {nominal}")
        suffix = f"{int(round(100 * nominal)):02d}"
        coverage_column = f"coverage_{suffix}"
        width_column = f"interval_width_{suffix}"
        interval_columns[float(nominal)] = (coverage_column, width_column)
        evaluated[coverage_column] = False
        evaluated[width_column] = np.nan

    for model, indices in evaluated.groupby("model", sort=True).groups.items():
        model_rows = evaluated.loc[indices]
        truth = model_rows["actual"].to_numpy(dtype=float)
        forecast = model_rows[quantile_columns].to_numpy(dtype=float)
        evaluated.loc[indices, "pinball_loss"] = per_observation_pinball_loss(
            truth, forecast, levels
        )
        evaluated.loc[indices, "median_absolute_error"] = np.abs(
            truth - model_rows[median_column].to_numpy(dtype=float)
        )
        evaluated.loc[indices, "median_error"] = (
            truth - model_rows[median_column].to_numpy(dtype=float)
        )

        coverage = quantile_coverage(truth, forecast, levels)
        for level, observed in zip(levels, coverage):
            calibration_records.append(
                {
                    "model": model,
                    "quantile": level,
                    "empirical_coverage": observed,
                    "calibration_error": observed - level,
                    "absolute_calibration_error": abs(observed - level),
                }
            )
        crossing_counts[model] = quantile_crossing_count(forecast, levels)

        for nominal, (coverage_column, width_column) in interval_columns.items():
            tail = (1.0 - nominal) / 2.0
            lower_column = quantile_columns[int(np.flatnonzero(np.isclose(levels, tail))[0])]
            upper_column = quantile_columns[
                int(np.flatnonzero(np.isclose(levels, 1.0 - tail))[0])
            ]
            lower = model_rows[lower_column].to_numpy(dtype=float)
            upper = model_rows[upper_column].to_numpy(dtype=float)
            evaluated.loc[indices, coverage_column] = (truth >= lower) & (truth <= upper)
            evaluated.loc[indices, width_column] = upper - lower

    fold_records: list[dict[str, float | int | str | pd.Timestamp]] = []
    for (model, origin), group in evaluated.groupby(["model", "origin"], sort=True):
        record: dict[str, float | int | str | pd.Timestamp] = {
            "model": model,
            "origin": origin,
            "forecast_hours": len(group),
            "pinball_loss": group["pinball_loss"].mean(),
            "median_mae": group["median_absolute_error"].mean(),
            "median_bias": group["median_error"].mean(),
        }
        sample_sizes = (
            group["sample_size"].dropna()
            if "sample_size" in group.columns
            else pd.Series(dtype=float)
        )
        if sample_sizes.empty:
            record.update(
                sample_size_min=np.nan,
                sample_size_median=np.nan,
                sample_size_max=np.nan,
            )
        else:
            record.update(
                sample_size_min=int(sample_sizes.min()),
                sample_size_median=float(sample_sizes.median()),
                sample_size_max=int(sample_sizes.max()),
            )
        for nominal, (coverage_column, width_column) in interval_columns.items():
            suffix = f"{int(round(100 * nominal)):02d}"
            record[f"coverage_{suffix}"] = group[coverage_column].mean()
            record[f"mean_width_{suffix}"] = group[width_column].mean()
        fold_records.append(record)
    fold_metrics = pd.DataFrame(fold_records).merge(
        fold_manifest, on="origin", how="left", validate="many_to_one"
    )
    if fold_metrics["training_hours"].isna().any():
        raise ValueError("Fold manifest does not cover every prediction origin")

    quantile_calibration = pd.DataFrame(calibration_records)
    interval_records: list[dict[str, float | str]] = []
    aggregate_records: list[dict[str, float | int | str]] = []
    for model, group in evaluated.groupby("model", sort=True):
        model_folds = fold_metrics.loc[fold_metrics["model"] == model]
        model_calibration = quantile_calibration.loc[
            quantile_calibration["model"] == model
        ]
        aggregate_records.append(
            {
                "model": model,
                "forecast_hours": len(group),
                "folds": len(model_folds),
                "pinball_loss": group["pinball_loss"].mean(),
                "fold_pinball_mean": model_folds["pinball_loss"].mean(),
                "fold_pinball_std": model_folds["pinball_loss"].std(ddof=1),
                "fold_pinball_median": model_folds["pinball_loss"].median(),
                "fold_pinball_min": model_folds["pinball_loss"].min(),
                "fold_pinball_max": model_folds["pinball_loss"].max(),
                "median_mae": group["median_absolute_error"].mean(),
                "median_bias": group["median_error"].mean(),
                "mean_absolute_calibration_error": model_calibration[
                    "absolute_calibration_error"
                ].mean(),
                "max_absolute_calibration_error": model_calibration[
                    "absolute_calibration_error"
                ].max(),
                "quantile_crossings": crossing_counts[model],
            }
        )
        for nominal, (coverage_column, width_column) in interval_columns.items():
            coverage = group[coverage_column].mean()
            interval_records.append(
                {
                    "model": model,
                    "nominal_coverage": nominal,
                    "empirical_coverage": coverage,
                    "coverage_error": coverage - nominal,
                    "absolute_coverage_error": abs(coverage - nominal),
                    "mean_width": group[width_column].mean(),
                }
            )

    aggregate_metrics = pd.DataFrame(aggregate_records)
    interval_calibration = pd.DataFrame(interval_records)
    if reference_model not in set(aggregate_metrics["model"]):
        raise ValueError(f"Unknown reference model {reference_model!r}")

    score_by_model = aggregate_metrics.set_index("model")["pinball_loss"]
    fold_scores = fold_metrics.pivot(index="origin", columns="model", values="pinball_loss")
    comparison_records: list[dict[str, float | int | str]] = []
    for model in sorted(set(fold_scores.columns) - {reference_model}):
        difference = (fold_scores[model] - fold_scores[reference_model]).dropna()
        non_ties = difference.loc[~np.isclose(difference, 0.0)]
        wins = int((non_ties < 0).sum())
        losses = int((non_ties > 0).sum())
        comparison_records.append(
            {
                "model": model,
                "reference_model": reference_model,
                "paired_folds": len(difference),
                "mean_pinball_difference": difference.mean(),
                "std_pinball_difference": difference.std(ddof=1),
                "median_pinball_difference": difference.median(),
                "relative_pinball_improvement": 1.0
                - score_by_model[model] / score_by_model[reference_model],
                "folds_won": wins,
                "folds_lost": losses,
                "folds_tied": int(len(difference) - len(non_ties)),
                "paired_sign_test_pvalue": _two_sided_sign_test(wins, losses),
            }
        )
    paired_comparison = pd.DataFrame(comparison_records)

    return EvaluationTables(
        predictions=evaluated,
        fold_metrics=fold_metrics,
        aggregate_metrics=aggregate_metrics,
        quantile_calibration=quantile_calibration,
        interval_calibration=interval_calibration,
        paired_comparison=paired_comparison,
    )


def _two_sided_sign_test(wins: int, losses: int) -> float:
    """Exact two-sided binomial sign test, excluding tied folds."""

    trials = wins + losses
    if trials == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2.0 * sum(comb(trials, k) for k in range(tail + 1)) / (2**trials)
    return float(min(1.0, probability))
