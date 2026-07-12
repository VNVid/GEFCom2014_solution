"""Time one CatBoost MultiQuantile fit on a configured rolling-origin fold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import catboost
import numpy as np
import pandas as pd
import yaml

from gefcom2014.data import QUANTILES
from gefcom2014.metrics import (
    pinball_loss,
    quantile_coverage,
    quantile_crossing_count,
)
from gefcom2014.model_data import (
    load_monthly_modeling_dataset,
    rolling_origin_masks,
)
from gefcom2014.models import fit_catboost_quantiles, predict_catboost_quantiles


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def run(
    backtest_config_path: Path,
    catboost_config_path: Path,
) -> dict[str, Any]:
    """Fit, predict, time, and diagnose one configured validation fold."""

    backtest_config = _read_yaml(backtest_config_path)
    config = _read_yaml(catboost_config_path)
    dataset = load_monthly_modeling_dataset(config["data"]["model_data_dir"])
    origin = pd.Timestamp(config["benchmark"]["origin"])
    training_mask, evaluation_mask = rolling_origin_masks(dataset.metadata, origin)

    train_features = dataset.features.loc[training_mask].reset_index(drop=True)
    train_target = dataset.target.loc[training_mask].reset_index(drop=True)
    evaluation_features = dataset.features.loc[evaluation_mask].reset_index(drop=True)
    actual = dataset.target.loc[evaluation_mask].to_numpy(dtype=float)
    model_parameters = dict(config["model"])

    fit_start = perf_counter()
    model = fit_catboost_quantiles(
        train_features,
        train_target,
        QUANTILES,
        dataset.categorical_features,
        model_parameters,
    )
    fit_seconds = perf_counter() - fit_start

    prediction_start = perf_counter()
    predictions = predict_catboost_quantiles(
        model, evaluation_features, QUANTILES
    )
    prediction_seconds = perf_counter() - prediction_start

    median_position = int(np.flatnonzero(np.isclose(QUANTILES, 0.50))[0])
    lower_90 = int(np.flatnonzero(np.isclose(QUANTILES, 0.05))[0])
    upper_90 = int(np.flatnonzero(np.isclose(QUANTILES, 0.95))[0])
    coverage = quantile_coverage(actual, predictions, QUANTILES)
    lower_90_values = predictions[:, lower_90]
    upper_90_values = predictions[:, upper_90]
    invalid_90 = lower_90_values > upper_90_values
    valid_widths = (upper_90_values - lower_90_values)[~invalid_90]
    summary = {
        "catboost_version": catboost.__version__,
        "origin": str(origin),
        "training_rows": int(training_mask.sum()),
        "evaluation_rows": int(evaluation_mask.sum()),
        "features": dataset.features.shape[1],
        "quantiles": len(QUANTILES),
        "trees": int(model.tree_count_),
        "fit_seconds": fit_seconds,
        "prediction_seconds": prediction_seconds,
        "pinball_loss": pinball_loss(actual, predictions, QUANTILES),
        "median_mae": float(
            np.mean(np.abs(actual - predictions[:, median_position]))
        ),
        "median_bias": float(
            np.mean(actual - predictions[:, median_position])
        ),
        "coverage_90": float(
            np.mean(
                ~invalid_90
                & (actual >= lower_90_values)
                & (actual <= upper_90_values)
            )
        ),
        "mean_width_90": float(
            np.mean(valid_widths) if valid_widths.size else np.nan
        ),
        "invalid_90_intervals": int(invalid_90.sum()),
        "mean_absolute_calibration_error": float(
            np.mean(np.abs(coverage - QUANTILES))
        ),
        "quantile_crossings": quantile_crossing_count(predictions, QUANTILES),
        "model_parameters": model_parameters,
    }

    output_dir = Path(config["outputs"]["root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    quantile_columns = [f"q{level:.2f}" for level in QUANTILES]
    prediction_frame = dataset.metadata.loc[evaluation_mask].reset_index(drop=True)
    prediction_frame["actual"] = actual
    prediction_frame = pd.concat(
        [
            prediction_frame,
            pd.DataFrame(predictions, columns=quantile_columns),
        ],
        axis=1,
    )
    prediction_frame.to_csv(
        output_dir / "predictions.csv.gz",
        index=False,
        float_format="%.10g",
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    resolved = {
        "backtest": backtest_config,
        "catboost": config,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backtest-config", type=Path, default=Path("configs/backtest.yaml")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/catboost.yaml")
    )
    args = parser.parse_args()
    run(args.backtest_config, args.config)


if __name__ == "__main__":
    main()
