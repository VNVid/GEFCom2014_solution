"""Run a resumable rolling-origin CatBoost hyperparameter search."""

from __future__ import annotations

import argparse
from datetime import datetime
import gc
from pathlib import Path
from time import perf_counter
from typing import Any

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
from gefcom2014.models import (
    build_catboost_candidates,
    effective_candidate_name,
    fit_catboost_quantiles,
    predict_catboost_quantiles,
    summarize_search_results,
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


def _log(message: str, path: Path) -> None:
    stamped = f"[{datetime.now().isoformat(timespec='seconds')}] {message}"
    print(stamped, flush=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(stamped + "\n")


def run(backtest_config_path: Path, catboost_config_path: Path) -> pd.DataFrame:
    """Run or resume every configured candidate over all validation origins."""

    backtest = _read_yaml(backtest_config_path)
    config = _read_yaml(catboost_config_path)
    search = config["search"]
    split_name = str(search["split"])
    split = backtest["backtest"]["splits"][split_name]
    origins = pd.date_range(split["start"], split["end"], freq="MS", inclusive="left")
    dataset = load_monthly_modeling_dataset(config["data"]["model_data_dir"])

    baseline = pd.read_csv(search["baseline_fold_metrics"], parse_dates=["origin"])
    baseline = baseline.loc[
        baseline["model"].eq(search["reference_model"])
        & baseline["origin"].isin(origins),
        ["origin", "pinball_loss"],
    ]
    if len(baseline) != len(origins) or baseline["origin"].duplicated().any():
        raise ValueError("Baseline fold metrics do not cover every search origin")
    baseline_by_origin = baseline.set_index("origin")["pinball_loss"].to_dict()

    candidates = build_catboost_candidates(search)
    l2 = float(search["l2_leaf_reg"])
    common_parameters = dict(search["common_parameters"])
    managed = {"depth", "learning_rate", "iterations", "l2_leaf_reg"}
    overlap = sorted(managed & set(common_parameters))
    if overlap:
        raise ValueError(f"common_parameters override search values {overlap}")

    output_dir = Path(config["outputs"]["search_root_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "fold_results.csv"
    summary_path = output_dir / "candidate_summary.csv"
    log_path = output_dir / "progress.log"
    with (output_dir / "resolved_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"backtest": backtest, "catboost": config}, stream, sort_keys=False
        )

    if progress_path.is_file():
        results = pd.read_csv(progress_path, parse_dates=["origin"])
        if "invalid_90_intervals" not in results:
            # Earlier saved rows passed strict interval validation, so none of
            # their 5th/95th percentile bounds were reversed.
            results["invalid_90_intervals"] = 0
    else:
        results = pd.DataFrame()
    total_fits = len(candidates) * len(origins)
    _log(
        f"SEARCH START base_candidates={len(candidates)} origins={len(origins)} "
        f"actual_fits={total_fits} resumed_rows={len(results)}",
        log_path,
    )

    run_start = perf_counter()
    for base in candidates:
        for origin in origins:
            if not results.empty:
                completed = results.loc[
                    results["base_candidate"].eq(base.name)
                    & results["origin"].eq(origin),
                    "iterations",
                ]
                if set(completed.astype(int)) == set(base.iteration_counts):
                    _log(
                        f"SKIP {base.name} origin={origin.date()} already complete",
                        log_path,
                    )
                    continue

            training_mask, evaluation_mask = rolling_origin_masks(
                dataset.metadata, origin
            )
            train_features = dataset.features.loc[training_mask].reset_index(drop=True)
            train_target = dataset.target.loc[training_mask].reset_index(drop=True)
            evaluation_features = dataset.features.loc[evaluation_mask].reset_index(
                drop=True
            )
            actual = dataset.target.loc[evaluation_mask].to_numpy(dtype=float)
            parameters = {
                **common_parameters,
                "depth": base.depth,
                "learning_rate": base.learning_rate,
                "iterations": base.max_iterations,
                "l2_leaf_reg": l2,
            }

            completed_fits = (
                0
                if results.empty
                else len(results[["base_candidate", "origin"]].drop_duplicates())
            )
            _log(
                f"FIT {completed_fits + 1}/{total_fits} {base.name} "
                f"origin={origin.date()} train={training_mask.sum()} "
                f"eval={evaluation_mask.sum()} max_trees={base.max_iterations}",
                log_path,
            )
            fit_start = perf_counter()
            model = fit_catboost_quantiles(
                train_features,
                train_target,
                QUANTILES,
                dataset.categorical_features,
                parameters,
                verbose=int(search["training_log_period"]),
            )
            fit_seconds = perf_counter() - fit_start
            fold_records: list[dict[str, object]] = []
            for iterations in base.iteration_counts:
                prediction_start = perf_counter()
                predictions = predict_catboost_quantiles(
                    model,
                    evaluation_features,
                    QUANTILES,
                    tree_count=iterations,
                )
                prediction_seconds = perf_counter() - prediction_start
                median = predictions[:, 49]
                coverage = quantile_coverage(actual, predictions, QUANTILES)
                candidate = effective_candidate_name(base, iterations, l2)
                loss = pinball_loss(actual, predictions, QUANTILES)
                baseline_loss = float(baseline_by_origin[origin])
                lower_90 = predictions[:, 4]
                upper_90 = predictions[:, 94]
                invalid_90 = lower_90 > upper_90
                valid_widths = (upper_90 - lower_90)[~invalid_90]
                record = {
                    "base_candidate": base.name,
                    "candidate": candidate,
                    "origin": origin,
                    "depth": base.depth,
                    "learning_rate": base.learning_rate,
                    "l2_leaf_reg": l2,
                    "max_iterations": base.max_iterations,
                    "iterations": iterations,
                    "training_rows": int(training_mask.sum()),
                    "evaluation_rows": int(evaluation_mask.sum()),
                    "fit_seconds": fit_seconds,
                    "prediction_seconds": prediction_seconds,
                    "pinball_loss": loss,
                    "baseline_pinball_loss": baseline_loss,
                    "pinball_difference": loss - baseline_loss,
                    "median_mae": float(np.mean(np.abs(actual - median))),
                    "median_bias": float(np.mean(actual - median)),
                    "coverage_90": float(
                        np.mean(
                            ~invalid_90
                            & (actual >= lower_90)
                            & (actual <= upper_90)
                        )
                    ),
                    "mean_width_90": float(
                        np.mean(valid_widths) if valid_widths.size else np.nan
                    ),
                    "invalid_90_intervals": int(invalid_90.sum()),
                    "mean_absolute_calibration_error": float(
                        np.mean(np.abs(coverage - QUANTILES))
                    ),
                    "quantile_crossings": quantile_crossing_count(
                        predictions, QUANTILES
                    ),
                }
                fold_records.append(record)
                _log(
                    f"RESULT {candidate} origin={origin.date()} loss={loss:.4f} "
                    f"baseline={baseline_loss:.4f} delta={loss - baseline_loss:+.4f} "
                    f"coverage90={record['coverage_90']:.3f} "
                    f"invalid90={record['invalid_90_intervals']} "
                    f"crossings={record['quantile_crossings']}",
                    log_path,
                )

            new_rows = pd.DataFrame(fold_records)
            results = pd.concat([results, new_rows], ignore_index=True)
            results = results.drop_duplicates(
                ["base_candidate", "origin", "iterations"], keep="last"
            ).sort_values(["base_candidate", "origin", "iterations"])
            _atomic_csv(results, progress_path)
            candidate_summary = summarize_search_results(results)
            _atomic_csv(candidate_summary, summary_path)

            for candidate in new_rows["candidate"]:
                row = candidate_summary.loc[
                    candidate_summary["candidate"].eq(candidate)
                ].iloc[0]
                _log(
                    f"RUNNING {candidate} folds={int(row['folds'])}/24 "
                    f"loss={row['pinball_loss']:.4f} "
                    f"baseline={row['baseline_pinball_loss']:.4f} "
                    f"improvement={100 * row['relative_improvement']:+.2f}%",
                    log_path,
                )
                if (
                    int(row["folds"]) >= int(search["alert_after_folds"])
                    and float(row["relative_improvement"])
                    < -float(search["alert_relative_degradation"])
                ):
                    _log(
                        f"ALERT {candidate} is persistently worse than baseline; "
                        "inspect before continuing if this affects all candidates",
                        log_path,
                    )

            unique_fits = results[
                ["base_candidate", "origin", "fit_seconds"]
            ].drop_duplicates(
                ["base_candidate", "origin"]
            )
            average_fit = float(unique_fits["fit_seconds"].mean())
            remaining = total_fits - len(unique_fits)
            _log(
                f"FIT DONE {base.name} origin={origin.date()} "
                f"seconds={fit_seconds:.1f} "
                f"completed={len(unique_fits)}/{total_fits} "
                f"rough_remaining_hours={remaining * average_fit / 3600:.2f}",
                log_path,
            )
            del model
            gc.collect()

    final_summary = summarize_search_results(results)
    _atomic_csv(final_summary, summary_path)
    elapsed = perf_counter() - run_start
    best = final_summary.iloc[0]
    _log(
        f"SEARCH COMPLETE seconds={elapsed:.1f} best={best['candidate']} "
        f"loss={best['pinball_loss']:.4f} "
        f"improvement={100 * best['relative_improvement']:+.2f}%",
        log_path,
    )
    return final_summary


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
