import numpy as np
import pandas as pd
import pytest

from gefcom2014.evaluation import evaluate_predictions


def test_evaluation_aggregates_shared_folds_and_performs_paired_sign_test() -> None:
    quantiles = np.array([0.25, 0.5, 0.75])
    origins = [pd.Timestamp("2010-01-01"), pd.Timestamp("2010-02-01")]
    records = []
    for model in ["reference", "candidate"]:
        for origin, actuals in zip(origins, [[10.0, 11.0], [12.0, 13.0]]):
            for actual in actuals:
                forecast = (
                    [actual - 1.0, actual, actual + 1.0]
                    if model == "candidate"
                    else [0.0, 0.0, 0.0]
                )
                records.append(
                    {
                        "model": model,
                        "origin": origin,
                        "actual": actual,
                        "sample_size": 3,
                        "q0.25": forecast[0],
                        "q0.50": forecast[1],
                        "q0.75": forecast[2],
                    }
                )
    predictions = pd.DataFrame(records)
    manifest = pd.DataFrame(
        {
            "origin": origins,
            "training_hours": [100, 200],
        }
    )

    tables = evaluate_predictions(
        predictions, manifest, quantiles, [0.5], reference_model="reference"
    )

    candidate = tables.aggregate_metrics.set_index("model").loc["candidate"]
    assert candidate["folds"] == 2
    assert candidate["median_mae"] == pytest.approx(0.0)
    assert candidate["median_bias"] == pytest.approx(0.0)
    assert candidate["quantile_crossings"] == 0
    comparison = tables.paired_comparison.iloc[0]
    assert comparison["folds_won"] == 2
    assert comparison["folds_lost"] == 0
    assert comparison["paired_sign_test_pvalue"] == pytest.approx(0.5)

    without_sample_sizes = evaluate_predictions(
        predictions.drop(columns="sample_size"),
        manifest,
        quantiles,
        [0.5],
        reference_model="reference",
    )
    assert without_sample_sizes.fold_metrics["sample_size_min"].isna().all()
