"""Refit selected CatBoost models and save all rolling-origin quantiles."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gefcom2014.data import QUANTILES
from gefcom2014.evaluation import evaluate_predictions
from gefcom2014.model_data import load_monthly_modeling_dataset, rolling_origin_masks
from gefcom2014.models import (
    fit_catboost_quantiles,
    predict_catboost_quantiles,
    resolve_candidate_parameters,
    resolve_selected_features,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False, float_format="%.10g")
    temporary.replace(path)


def _atomic_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(
        temporary,
        index=False,
        float_format="%.6f",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    temporary.replace(path)


def _atomic_yaml(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)
    temporary.replace(path)


def _log(message: str, path: Path) -> None:
    line = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(line, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def _selected_features(dataset, experiment: dict[str, Any]) -> tuple[list[str], list[str]]:
    manifest_path = experiment.get("selected_features_path")
    manifest = _read_yaml(Path(manifest_path)) if manifest_path is not None else None
    return resolve_selected_features(
        dataset.features.columns,
        dataset.categorical_features,
        manifest,
    )


def _model_parameters(
    experiment: dict[str, Any], common: dict[str, Any]
) -> dict[str, Any]:
    summary = pd.read_csv(experiment["search_summary_path"])
    return resolve_candidate_parameters(summary, experiment["candidate"], common)


def _run_experiment(
    name: str,
    experiment: dict[str, Any],
    common: dict[str, Any],
    backtest: dict[str, Any],
) -> None:
    split = backtest["backtest"]["splits"][experiment["split"]]
    origins = pd.date_range(split["start"], split["end"], freq="MS", inclusive="left")
    dataset = load_monthly_modeling_dataset(experiment["model_data_dir"])
    features, categorical = _selected_features(dataset, experiment)
    parameters = _model_parameters(experiment, common)
    output_dir = Path(experiment["output_dir"])
    folds_dir = output_dir / "folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "progress.log"
    quantile_columns = [f"q{level:.2f}" for level in QUANTILES]
    expected_columns = ["model", "origin", "period_start", "actual", *quantile_columns]
    resolved = {
        "experiment_name": name,
        "experiment": experiment,
        "backtest_split": split,
        "features": features,
        "categorical_features": categorical,
        "model_parameters": parameters,
    }
    resolved_path = output_dir / "resolved_config.yaml"
    if resolved_path.is_file():
        if _read_yaml(resolved_path) != resolved:
            raise ValueError(f"Existing {name} predictions use a different configuration")
    else:
        _atomic_yaml(resolved, resolved_path)

    manifest_records = []
    _log(
        f"PREDICTION START experiment={name} folds={len(origins)} "
        f"features={len(features)} candidate={experiment['candidate']}",
        log_path,
    )
    for number, origin in enumerate(origins, start=1):
        training_mask, evaluation_mask = rolling_origin_masks(dataset.metadata, origin)
        checkpoint = folds_dir / f"{origin:%Y-%m}.csv.gz"
        expected_rows = int(evaluation_mask.sum())
        if checkpoint.is_file():
            saved = pd.read_csv(checkpoint, parse_dates=["origin", "period_start"])
            if saved.columns.tolist() != expected_columns or len(saved) != expected_rows:
                raise ValueError(f"Invalid saved prediction fold {checkpoint}")
            if not saved["origin"].eq(origin).all():
                raise ValueError(f"Saved prediction origin mismatch in {checkpoint}")
            _log(f"SKIP {number}/{len(origins)} origin={origin.date()}", log_path)
        else:
            _log(
                f"FIT {number}/{len(origins)} origin={origin.date()} "
                f"train={training_mask.sum()} eval={expected_rows}",
                log_path,
            )
            model = fit_catboost_quantiles(
                dataset.features.loc[training_mask, features].reset_index(drop=True),
                dataset.target.loc[training_mask].reset_index(drop=True),
                QUANTILES,
                categorical,
                parameters,
                verbose=False,
            )
            predictions = predict_catboost_quantiles(
                model,
                dataset.features.loc[evaluation_mask, features].reset_index(drop=True),
                QUANTILES,
            )
            saved = pd.DataFrame(
                {
                    "model": experiment["model_name"],
                    "origin": origin,
                    "period_start": dataset.metadata.loc[
                        evaluation_mask, "period_start"
                    ].to_numpy(),
                    "actual": dataset.target.loc[evaluation_mask].to_numpy(dtype=float),
                }
            )
            saved = pd.concat(
                [saved, pd.DataFrame(predictions, columns=quantile_columns)], axis=1
            )
            _atomic_gzip_csv(saved.loc[:, expected_columns], checkpoint)
            _log(f"RESULT origin={origin.date()} saved_rows={len(saved)}", log_path)
            del model
            gc.collect()

        training_periods = dataset.metadata.loc[training_mask, "period_start"]
        forecast_ends = dataset.metadata.loc[evaluation_mask, "forecast_end"].unique()
        if len(forecast_ends) != 1:
            raise ValueError(f"Evaluation origin {origin} has multiple forecast ends")
        manifest_records.append(
            {
                "origin": origin,
                "forecast_end": pd.Timestamp(forecast_ends[0]),
                "training_start": training_periods.min(),
                "training_end": training_periods.max(),
                "training_hours": int(training_mask.sum()),
            }
        )

    prediction_frames = [
        pd.read_csv(
            folds_dir / f"{origin:%Y-%m}.csv.gz",
            parse_dates=["origin", "period_start"],
        )
        for origin in origins
    ]
    predictions = pd.concat(prediction_frames, ignore_index=True)
    manifest = pd.DataFrame(manifest_records)
    tables = evaluate_predictions(
        predictions,
        manifest,
        QUANTILES,
        [float(value) for value in backtest["evaluation"]["central_intervals"]],
        reference_model=experiment["model_name"],
    )
    outputs = {
        "fold_manifest.csv": manifest,
        "fold_metrics.csv": tables.fold_metrics,
        "aggregate_metrics.csv": tables.aggregate_metrics,
        "quantile_calibration.csv": tables.quantile_calibration,
        "interval_calibration.csv": tables.interval_calibration,
    }
    for filename, frame in outputs.items():
        _atomic_csv(frame, output_dir / filename)
    _atomic_gzip_csv(tables.predictions, output_dir / "predictions.csv.gz")
    _log(
        f"PREDICTION COMPLETE experiment={name} rows={len(predictions)}",
        log_path,
    )


def run(backtest_path: Path, config_path: Path, experiment_names: list[str]) -> None:
    backtest = _read_yaml(backtest_path)
    config = _read_yaml(config_path)
    experiments = config["experiments"]
    selected_names = experiment_names or list(experiments)
    unknown = sorted(set(selected_names) - set(experiments))
    if unknown:
        raise ValueError(f"Unknown selected-prediction experiments: {unknown}")
    for name in selected_names:
        _run_experiment(name, experiments[name], config["common_parameters"], backtest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-config", type=Path, default=Path("configs/backtest.yaml")
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/catboost_selected_predictions.yaml"),
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment name to run; repeat to select multiple (default: all)",
    )
    args = parser.parse_args()
    run(args.backtest_config, args.config, args.experiment)


if __name__ == "__main__":
    main()
