"""Run both load baselines over a configured rolling-origin split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from gefcom2014.backtesting import data_for_fold, monthly_folds, prepare_backtest_frame
from gefcom2014.baselines import seasonal_empirical_forecast, seasonal_naive_forecast
from gefcom2014.data import load_backtest_actuals
from gefcom2014.evaluation import evaluate_predictions

from .plots import generate_figures


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def _quantile_grid(config: dict[str, float]) -> np.ndarray:
    start = float(config["start"])
    stop = float(config["stop"])
    step = float(config["step"])
    if step <= 0 or start <= 0 or stop >= 1 or start > stop:
        raise ValueError("Invalid quantile grid configuration")
    count = int(round((stop - start) / step)) + 1
    levels = np.round(start + step * np.arange(count), 10)
    if not np.isclose(levels[-1], stop):
        raise ValueError("Quantile stop is not reachable from start and step")
    if len(levels) != 99 or not np.allclose(levels, np.arange(1, 100) / 100):
        raise ValueError("GEFCom2014 evaluation requires quantiles 0.01 through 0.99")
    return levels


def run(
    backtest_config_path: Path,
    baseline_config_path: Path,
    split_override: str | None = None,
) -> dict[str, Any]:
    """Execute the configured split and write reproducible artifacts."""

    backtest_config = _read_yaml(backtest_config_path)
    baseline_config = _read_yaml(baseline_config_path)
    split_name = split_override or backtest_config["backtest"]["default_split"]
    splits = backtest_config["backtest"]["splits"]
    if split_name not in splits:
        raise ValueError(f"Unknown split {split_name!r}; expected one of {sorted(splits)}")

    split_config = splits[split_name]
    folds = monthly_folds(split_config["start"], split_config["end"])
    quantiles = _quantile_grid(backtest_config["evaluation"]["quantiles"])
    actuals = load_backtest_actuals(backtest_config["data"]["load_dir"])
    frame = prepare_backtest_frame(actuals)

    empirical_config = baseline_config["model"]["seasonal_empirical"]
    prediction_frames: list[pd.DataFrame] = []
    manifest_records: list[dict[str, Any]] = []
    quantile_columns = [f"q{level:.2f}" for level in quantiles]

    for fold in folds:
        training, target = data_for_fold(frame, fold)
        # Models receive calendar covariates only; withheld target loads remain
        # solely in this runner for evaluation.
        target_features = target.loc[:, ["zone_id", "period_start"]]
        forecasts = {
            "seasonal_naive": seasonal_naive_forecast(
                training, target_features, quantiles
            ),
            "seasonal_empirical": seasonal_empirical_forecast(
                training,
                target_features,
                quantiles,
                window_days=int(empirical_config["window_days"]),
                quantile_method=str(empirical_config["quantile_method"]),
            ),
        }

        metadata = target.loc[:, ["zone_id", "timestamp", "period_start"]].reset_index(
            drop=True
        )
        metadata["actual"] = target["load"].to_numpy(dtype=float)
        metadata["origin"] = fold.origin
        metadata["forecast_end"] = fold.end
        for model, (values, sample_sizes) in forecasts.items():
            model_frame = metadata.copy()
            model_frame.insert(0, "model", model)
            model_frame["sample_size"] = sample_sizes
            forecast_frame = pd.DataFrame(values, columns=quantile_columns)
            prediction_frames.append(pd.concat([model_frame, forecast_frame], axis=1))

        manifest_records.append(
            {
                "origin": fold.origin,
                "forecast_end": fold.end,
                "training_start": training["period_start"].min(),
                "training_end": training["period_start"].max(),
                "training_hours": len(training),
            }
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_manifest = pd.DataFrame(manifest_records)
    tables = evaluate_predictions(
        predictions,
        fold_manifest,
        quantiles,
        [float(value) for value in backtest_config["evaluation"]["central_intervals"]],
        reference_model=baseline_config["evaluation"]["reference_model"],
    )

    output_dir = Path(baseline_config["outputs"]["root_dir"]) / split_name
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    table_outputs = {
        "fold_manifest.csv": fold_manifest,
        "fold_metrics.csv": tables.fold_metrics,
        "aggregate_metrics.csv": tables.aggregate_metrics,
        "quantile_calibration.csv": tables.quantile_calibration,
        "interval_calibration.csv": tables.interval_calibration,
        "paired_comparison.csv": tables.paired_comparison,
    }
    for filename, table in table_outputs.items():
        table.to_csv(output_dir / filename, index=False, float_format="%.10g")
    if baseline_config["outputs"].get("save_predictions", True):
        tables.predictions.to_csv(
            output_dir / "predictions.csv.gz",
            index=False,
            float_format="%.6f",
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )

    resolved_config = {
        "selected_split": split_name,
        "backtest": backtest_config,
        "baseline": baseline_config,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False)

    generate_figures(
        tables.fold_metrics,
        tables.quantile_calibration,
        tables.interval_calibration,
        figures_dir,
        baseline_config["plot"],
    )
    summary = {
        "split": split_name,
        "folds": len(folds),
        "forecast_start": str(folds[0].origin),
        "forecast_end": str(folds[-1].end),
        "aggregate_metrics": json.loads(
            tables.aggregate_metrics.to_json(orient="records")
        ),
        "paired_comparison": json.loads(
            tables.paired_comparison.to_json(orient="records")
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-config", type=Path, default=Path("configs/backtest.yaml")
    )
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument(
        "--split",
        help="Configured split to run; defaults to backtest.default_split (validation)",
    )
    args = parser.parse_args()
    run(args.backtest_config, args.config, args.split)


if __name__ == "__main__":
    main()
