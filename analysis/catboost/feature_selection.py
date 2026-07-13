"""Run bounded rolling-origin feature screening and complementary verification."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
import json
from pathlib import Path
from time import perf_counter
from typing import Any

from catboost import Pool
import numpy as np
import pandas as pd
import yaml

from gefcom2014.data import QUANTILES
from gefcom2014.metrics import pinball_loss, quantile_coverage, quantile_crossing_count
from gefcom2014.model_data import load_monthly_modeling_dataset, rolling_origin_masks
from gefcom2014.models import (
    fit_catboost_quantiles,
    predict_catboost_quantiles,
    select_features_from_loss_change,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    temporary.replace(path)


def _atomic_yaml(data: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)
    temporary.replace(path)


def _log(message: str, path: Path) -> None:
    stamped = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(stamped, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stamped + "\n")


def _evaluate_predictions(actual: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    median = predictions[:, 49]
    coverage = quantile_coverage(actual, predictions, QUANTILES)
    lower = predictions[:, 4]
    upper = predictions[:, 94]
    invalid = lower > upper
    valid_widths = (upper - lower)[~invalid]
    return {
        "pinball_loss": pinball_loss(actual, predictions, QUANTILES),
        "median_mae": float(np.mean(np.abs(actual - median))),
        "median_bias": float(np.mean(actual - median)),
        "coverage_90": float(
            np.mean(~invalid & (actual >= lower) & (actual <= upper))
        ),
        "mean_width_90": float(
            np.mean(valid_widths) if valid_widths.size else np.nan
        ),
        "invalid_90_intervals": int(invalid.sum()),
        "mean_absolute_calibration_error": float(
            np.mean(np.abs(coverage - QUANTILES))
        ),
        "quantile_crossings": quantile_crossing_count(predictions, QUANTILES),
    }


def _check_candidate_matrix(features: pd.DataFrame) -> None:
    constants = features.columns[features.nunique(dropna=False).le(1)].tolist()
    if constants:
        raise ValueError(f"Candidate matrix contains constant features {constants}")
    duplicates: list[tuple[str, str]] = []
    for position, left in enumerate(features.columns):
        for right in features.columns[position + 1 :]:
            if features[left].equals(features[right]):
                duplicates.append((left, right))
    if duplicates:
        raise ValueError(f"Candidate matrix contains exact duplicate features {duplicates}")


def _aggregate_metrics(frame: pd.DataFrame, model: str, stage: str) -> dict[str, object]:
    weights = frame["evaluation_rows"].to_numpy(dtype=float)
    record: dict[str, object] = {
        "stage": stage,
        "model": model,
        "folds": len(frame),
        "forecast_hours": int(weights.sum()),
    }
    averaged = (
        "pinball_loss",
        "median_mae",
        "median_bias",
        "coverage_90",
        "mean_width_90",
        "mean_absolute_calibration_error",
    )
    for column in averaged:
        record[column] = (
            float(np.average(frame[column], weights=weights))
            if column in frame and frame[column].notna().any()
            else np.nan
        )
    record["invalid_90_intervals"] = (
        int(frame["invalid_90_intervals"].sum())
        if "invalid_90_intervals" in frame
        else 0
    )
    record["quantile_crossings"] = (
        int(frame["quantile_crossings"].sum())
        if "quantile_crossings" in frame
        else 0
    )
    return record


def _reference_rows(
    round1: pd.DataFrame,
    baselines: pd.DataFrame,
    origins: pd.DatetimeIndex,
    stage: str,
) -> list[tuple[str, pd.DataFrame]]:
    round1_rows = round1.loc[round1["origin"].isin(origins)].copy()
    baseline_rows = baselines.loc[baselines["origin"].isin(origins)].copy()
    renamed: list[tuple[str, pd.DataFrame]] = [("round1_fast", round1_rows)]
    for model in ("seasonal_empirical", "seasonal_naive"):
        rows = baseline_rows.loc[baseline_rows["model"].eq(model)].copy()
        rows = rows.rename(
            columns={
                "forecast_hours": "evaluation_rows",
                "coverage_90": "coverage_90",
                "mean_width_90": "mean_width_90",
            }
        )
        renamed.append((model, rows))
    for model, rows in renamed:
        if len(rows) != len(origins):
            raise ValueError(f"Reference {model} does not cover every {stage} origin")
    return renamed


def run(backtest_config_path: Path, selection_config_path: Path) -> dict[str, object]:
    """Fit six screening and six verification folds with atomic checkpoints."""

    backtest = _read_yaml(backtest_config_path)
    config = _read_yaml(selection_config_path)
    experiment = config["experiment"]
    split_name = str(experiment["split"])
    split = backtest["backtest"]["splits"][split_name]
    origins = pd.date_range(split["start"], split["end"], freq="MS", inclusive="left")
    screening_months = tuple(int(value) for value in experiment["screening_months"])
    verification_months = tuple(int(value) for value in experiment["verification_months"])
    if set(screening_months) & set(verification_months):
        raise ValueError("Screening and verification months must be disjoint")
    if set(screening_months) | set(verification_months) != set(origins.month):
        raise ValueError("Screening and verification months must partition the split")
    screening_origins = origins[origins.month.isin(screening_months)]
    verification_origins = origins[origins.month.isin(verification_months)]

    dataset = load_monthly_modeling_dataset(config["data"]["model_data_dir"])
    candidate_features = dataset.features.columns.tolist()
    expected_features = int(experiment["expected_candidate_features"])
    if len(candidate_features) != expected_features:
        raise ValueError(
            f"Expected {expected_features} candidate features, got {len(candidate_features)}"
        )
    _check_candidate_matrix(dataset.features)

    output_dir = Path(config["outputs"]["root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "fold_metrics.csv"
    importance_path = output_dir / "fold_feature_importance.csv"
    ranking_path = output_dir / "feature_ranking.csv"
    selected_path = output_dir / "selected_features.yaml"
    comparison_path = output_dir / "comparison.csv"
    summary_path = output_dir / "summary.json"
    resolved_path = output_dir / "resolved_config.yaml"
    log_path = output_dir / "progress.log"
    resolved = {
        "backtest": backtest,
        "feature_selection": config,
        "candidate_features": candidate_features,
        "categorical_features": list(dataset.categorical_features),
    }
    if resolved_path.is_file():
        existing = _read_yaml(resolved_path)
        if existing != resolved:
            raise ValueError(
                "Existing feature-selection artifacts use a different configuration"
            )
    else:
        _atomic_yaml(resolved, resolved_path)

    metrics = (
        pd.read_csv(metrics_path, parse_dates=["origin"])
        if metrics_path.is_file()
        else pd.DataFrame()
    )
    importance = (
        pd.read_csv(importance_path, parse_dates=["origin"])
        if importance_path.is_file()
        else pd.DataFrame()
    )
    baseline = pd.read_csv(
        config["references"]["baseline_fold_metrics"], parse_dates=["origin"]
    )
    empirical = baseline.loc[baseline["model"].eq("seasonal_empirical")].set_index(
        "origin"
    )["pinball_loss"]

    _log(
        f"SELECTION START candidates={len(candidate_features)} "
        f"screening_folds={len(screening_origins)} "
        f"verification_folds={len(verification_origins)}",
        log_path,
    )
    start = perf_counter()
    for number, origin in enumerate(screening_origins, start=1):
        metric_complete = (
            not metrics.empty
            and bool(
                (
                    metrics["stage"].eq("screening")
                    & metrics["origin"].eq(origin)
                ).any()
            )
        )
        importance_complete = (
            not importance.empty
            and len(importance.loc[importance["origin"].eq(origin)])
            == len(candidate_features)
        )
        if metric_complete and importance_complete:
            _log(f"SCREEN SKIP {origin.date()} already complete", log_path)
            continue

        train_mask, evaluation_mask = rolling_origin_masks(dataset.metadata, origin)
        train_features = dataset.features.loc[train_mask].reset_index(drop=True)
        train_target = dataset.target.loc[train_mask].reset_index(drop=True)
        evaluation_features = dataset.features.loc[evaluation_mask].reset_index(drop=True)
        actual = dataset.target.loc[evaluation_mask].to_numpy(dtype=float)
        _log(
            f"SCREEN FIT {number}/{len(screening_origins)} origin={origin.date()} "
            f"train={len(train_features)} eval={len(evaluation_features)}",
            log_path,
        )
        fit_start = perf_counter()
        model = fit_catboost_quantiles(
            train_features,
            train_target,
            QUANTILES,
            dataset.categorical_features,
            config["model"],
            verbose=False,
        )
        fit_seconds = perf_counter() - fit_start
        predictions = predict_catboost_quantiles(model, evaluation_features, QUANTILES)
        diagnostics = _evaluate_predictions(actual, predictions)
        importance_start = perf_counter()
        pool = Pool(
            evaluation_features,
            actual,
            cat_features=list(dataset.categorical_features),
        )
        loss_change = np.asarray(
            model.get_feature_importance(pool, type="LossFunctionChange"), dtype=float
        )
        importance_seconds = perf_counter() - importance_start
        if loss_change.shape != (len(candidate_features),) or not np.isfinite(
            loss_change
        ).all():
            raise ValueError("CatBoost returned invalid LossFunctionChange values")

        record = {
            "stage": "screening",
            "origin": origin,
            "feature_count": len(candidate_features),
            "training_rows": int(train_mask.sum()),
            "evaluation_rows": int(evaluation_mask.sum()),
            "fit_seconds": fit_seconds,
            "importance_seconds": importance_seconds,
            "baseline_pinball_loss": float(empirical.loc[origin]),
            **diagnostics,
        }
        new_importance = pd.DataFrame(
            {
                "origin": origin,
                "feature": candidate_features,
                "loss_function_change": loss_change,
            }
        )
        if not metrics.empty:
            metrics = metrics.loc[
                ~(metrics["stage"].eq("screening") & metrics["origin"].eq(origin))
            ]
        if not importance.empty:
            importance = importance.loc[~importance["origin"].eq(origin)]
        metrics = pd.concat([metrics, pd.DataFrame([record])], ignore_index=True)
        importance = pd.concat([importance, new_importance], ignore_index=True)
        _atomic_csv(metrics.sort_values(["stage", "origin"]), metrics_path)
        _atomic_csv(importance.sort_values(["origin", "feature"]), importance_path)
        top = candidate_features[int(np.argmax(loss_change))]
        _log(
            f"SCREEN RESULT origin={origin.date()} loss={diagnostics['pinball_loss']:.4f} "
            f"baseline={record['baseline_pinball_loss']:.4f} top={top} "
            f"fit_seconds={fit_seconds:.1f} importance_seconds={importance_seconds:.1f}",
            log_path,
        )
        del model, pool
        gc.collect()

    screening_importance = importance.loc[
        importance["origin"].isin(screening_origins)
    ]
    policy = config["selection"]
    ranking, selected_features = select_features_from_loss_change(
        screening_importance,
        expected_folds=len(screening_origins),
        minimum_positive_folds=int(policy["minimum_positive_folds"]),
        maximum_features=int(policy["maximum_features"]),
    )
    _atomic_csv(ranking, ranking_path)
    selected_categorical = [
        feature
        for feature in dataset.categorical_features
        if feature in selected_features
    ]
    _atomic_yaml(
        {
            "source_model_data": str(config["data"]["model_data_dir"]),
            "selection_policy": policy,
            "features": list(selected_features),
            "categorical_features": selected_categorical,
        },
        selected_path,
    )
    _log(
        f"FEATURES SELECTED count={len(selected_features)} "
        f"categorical={len(selected_categorical)}",
        log_path,
    )

    for number, origin in enumerate(verification_origins, start=1):
        complete = (
            not metrics.empty
            and bool(
                (
                    metrics["stage"].eq("verification")
                    & metrics["origin"].eq(origin)
                ).any()
            )
        )
        if complete:
            _log(f"VERIFY SKIP {origin.date()} already complete", log_path)
            continue
        train_mask, evaluation_mask = rolling_origin_masks(dataset.metadata, origin)
        train_features = dataset.features.loc[train_mask, list(selected_features)].reset_index(
            drop=True
        )
        train_target = dataset.target.loc[train_mask].reset_index(drop=True)
        evaluation_features = dataset.features.loc[
            evaluation_mask, list(selected_features)
        ].reset_index(drop=True)
        actual = dataset.target.loc[evaluation_mask].to_numpy(dtype=float)
        _log(
            f"VERIFY FIT {number}/{len(verification_origins)} origin={origin.date()} "
            f"features={len(selected_features)} train={len(train_features)} "
            f"eval={len(evaluation_features)}",
            log_path,
        )
        fit_start = perf_counter()
        model = fit_catboost_quantiles(
            train_features,
            train_target,
            QUANTILES,
            selected_categorical,
            config["model"],
            verbose=False,
        )
        fit_seconds = perf_counter() - fit_start
        predictions = predict_catboost_quantiles(model, evaluation_features, QUANTILES)
        diagnostics = _evaluate_predictions(actual, predictions)
        record = {
            "stage": "verification",
            "origin": origin,
            "feature_count": len(selected_features),
            "training_rows": int(train_mask.sum()),
            "evaluation_rows": int(evaluation_mask.sum()),
            "fit_seconds": fit_seconds,
            "importance_seconds": 0.0,
            "baseline_pinball_loss": float(empirical.loc[origin]),
            **diagnostics,
        }
        metrics = pd.concat([metrics, pd.DataFrame([record])], ignore_index=True)
        _atomic_csv(metrics.sort_values(["stage", "origin"]), metrics_path)
        _log(
            f"VERIFY RESULT origin={origin.date()} loss={diagnostics['pinball_loss']:.4f} "
            f"baseline={record['baseline_pinball_loss']:.4f} "
            f"fit_seconds={fit_seconds:.1f}",
            log_path,
        )
        del model
        gc.collect()

    round1_all = pd.read_csv(
        config["references"]["round1_fold_results"], parse_dates=["origin"]
    )
    round1 = round1_all.loc[
        round1_all["candidate"].eq(config["references"]["round1_candidate"])
    ].copy()
    comparison_records: list[dict[str, object]] = []
    stage_definitions = (
        ("screening", screening_origins, "round2_superset_fast"),
        ("verification", verification_origins, "round2_selected_fast"),
    )
    for stage, stage_origins, label in stage_definitions:
        model_rows = metrics.loc[
            metrics["stage"].eq(stage) & metrics["origin"].isin(stage_origins)
        ]
        if len(model_rows) != len(stage_origins):
            raise ValueError(f"Model results do not cover every {stage} origin")
        comparison_records.append(_aggregate_metrics(model_rows, label, stage))
        for model, reference in _reference_rows(
            round1, baseline, stage_origins, stage
        ):
            comparison_records.append(_aggregate_metrics(reference, model, stage))
    comparison = pd.DataFrame(comparison_records)
    _atomic_csv(comparison, comparison_path)

    verification_metrics = metrics.loc[metrics["stage"].eq("verification")].set_index(
        "origin"
    )
    verification_round1 = round1.loc[
        round1["origin"].isin(verification_origins)
    ].set_index("origin")
    differences = (
        verification_metrics["pinball_loss"]
        - verification_round1["pinball_loss"]
    )
    selected_aggregate = comparison.loc[
        comparison["stage"].eq("verification")
        & comparison["model"].eq("round2_selected_fast"),
        "pinball_loss",
    ].iloc[0]
    round1_aggregate = comparison.loc[
        comparison["stage"].eq("verification")
        & comparison["model"].eq("round1_fast"),
        "pinball_loss",
    ].iloc[0]
    relative_improvement = 1.0 - selected_aggregate / round1_aggregate
    tolerance = float(
        config["verification"]["maximum_relative_degradation_vs_round1"]
    )
    accepted = relative_improvement >= -tolerance
    summary: dict[str, object] = {
        "candidate_features": len(candidate_features),
        "selected_features": len(selected_features),
        "screening_origins": [str(value.date()) for value in screening_origins],
        "verification_origins": [str(value.date()) for value in verification_origins],
        "verification_pinball_loss": float(selected_aggregate),
        "round1_reference_pinball_loss": float(round1_aggregate),
        "relative_improvement_vs_round1": float(relative_improvement),
        "verification_folds_won": int((differences < 0).sum()),
        "verification_folds_lost": int((differences > 0).sum()),
        "maximum_allowed_relative_degradation": tolerance,
        "accepted": bool(accepted),
        "elapsed_seconds": perf_counter() - start,
    }
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _log(
        f"SELECTION COMPLETE selected={len(selected_features)} "
        f"verification_loss={selected_aggregate:.4f} "
        f"round1={round1_aggregate:.4f} "
        f"improvement={100 * relative_improvement:+.2f}% accepted={accepted}",
        log_path,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-config", type=Path, default=Path("configs/backtest.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/feature_selection.yaml")
    )
    args = parser.parse_args()
    run(args.backtest_config, args.config)


if __name__ == "__main__":
    main()
