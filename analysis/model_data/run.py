"""Build and cache leakage-safe monthly pseudo-origin model data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from gefcom2014.backtesting import (
    monthly_folds,
    prepare_backtest_frame,
    prepare_weather_frame,
)
from gefcom2014.data import load_backtest_actuals, load_complete_history
from gefcom2014.model_data import (
    build_monthly_modeling_dataset,
    rolling_origin_masks,
)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return config


def run(
    backtest_config_path: Path,
    feature_config_path: Path,
    model_data_config_path: Path,
    split_override: str | None = None,
) -> dict[str, Any]:
    """Build the configured development or test modeling-data cache."""

    backtest_config = _read_yaml(backtest_config_path)
    feature_config = _read_yaml(feature_config_path)
    model_data_config = _read_yaml(model_data_config_path)

    split_name = split_override or backtest_config["backtest"]["default_split"]
    splits = backtest_config["backtest"]["splits"]
    if split_name not in splits:
        raise ValueError(
            f"Unknown split {split_name!r}; expected one of {sorted(splits)}"
        )
    split = splits[split_name]
    evaluation_start = pd.Timestamp(split["start"])
    evaluation_end = pd.Timestamp(split["end"])
    first_origin = pd.Timestamp(model_data_config["dataset"]["first_origin"])
    if first_origin >= evaluation_start:
        raise ValueError("dataset.first_origin must precede the evaluation split")

    folds = monthly_folds(first_origin, evaluation_end)
    load_dir = backtest_config["data"]["load_dir"]
    load_frame = prepare_backtest_frame(load_backtest_actuals(load_dir))
    weather_frame = prepare_weather_frame(load_complete_history(load_dir))
    dataset = build_monthly_modeling_dataset(
        load_frame,
        folds,
        feature_config["feature_groups"],
        weather_frame=weather_frame,
        require_complete_features=bool(
            model_data_config["dataset"]["require_complete_features"]
        ),
    )

    first_training, _ = rolling_origin_masks(dataset.metadata, evaluation_start)
    last_origin = evaluation_end - pd.offsets.MonthBegin(1)
    last_training, _ = rolling_origin_masks(dataset.metadata, last_origin)

    output_dir = Path(model_data_config["outputs"]["root_dir"]) / split_name
    output_dir.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(
        [dataset.metadata, dataset.target, dataset.features], axis=1
    )
    float_format = str(model_data_config["outputs"]["float_format"])
    combined.to_csv(
        output_dir / "dataset.csv.gz",
        index=False,
        float_format=float_format,
        compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
    )
    dataset.manifest.to_csv(output_dir / "fold_manifest.csv", index=False)

    schema = {
        "metadata_columns": dataset.metadata.columns.tolist(),
        "target_column": dataset.target.name,
        "feature_columns": dataset.features.columns.tolist(),
        "feature_dtypes": {
            column: str(dtype)
            for column, dtype in dataset.features.dtypes.items()
        },
        "categorical_features": list(dataset.categorical_features),
    }
    with (output_dir / "schema.json").open("w", encoding="utf-8") as stream:
        json.dump(schema, stream, indent=2, sort_keys=True)
        stream.write("\n")

    resolved_config = {
        "selected_split": split_name,
        "backtest": backtest_config,
        "features": feature_config,
        "model_data": model_data_config,
    }
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved_config, stream, sort_keys=False)

    summary = {
        "split": split_name,
        "dataset_first_origin": str(first_origin),
        "dataset_end": str(evaluation_end),
        "evaluation_start": str(evaluation_start),
        "folds": len(dataset.manifest),
        "rows": len(dataset.features),
        "features": dataset.features.shape[1],
        "missing_feature_values": int(dataset.features.isna().sum().sum()),
        "training_rows_at_first_evaluation_origin": int(first_training.sum()),
        "training_rows_at_last_evaluation_origin": int(last_training.sum()),
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
    parser.add_argument(
        "--feature-config",
        type=Path,
        default=Path("configs/features/first_round.yaml"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/model_data.yaml")
    )
    parser.add_argument(
        "--split",
        help="Configured split to build; defaults to backtest.default_split",
    )
    args = parser.parse_args()
    run(args.backtest_config, args.feature_config, args.config, args.split)


if __name__ == "__main__":
    main()
