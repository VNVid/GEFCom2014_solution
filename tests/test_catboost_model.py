from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.models import (
    build_multi_quantile_loss,
    fit_catboost_quantiles,
    predict_catboost_quantiles,
)


def test_multi_quantile_loss_encodes_ordered_levels() -> None:
    loss = build_multi_quantile_loss(np.array([0.01, 0.50, 0.99]))
    assert loss == "MultiQuantile:alpha=0.01,0.5,0.99"

    with pytest.raises(ValueError, match="strictly increasing"):
        build_multi_quantile_loss(np.array([0.50, 0.50]))


def test_catboost_wrapper_fits_and_returns_one_column_per_quantile() -> None:
    hours = np.tile(np.arange(24), 4)
    features = pd.DataFrame(
        {
            "hour": hours,
            "level": 100.0 + 10.0 * np.sin(2.0 * np.pi * hours / 24.0),
        }
    )
    target = features["level"] + np.tile(np.arange(4), 24)
    quantiles = np.array([0.10, 0.50, 0.90])

    model = fit_catboost_quantiles(
        features,
        target,
        quantiles,
        categorical_features=("hour",),
        parameters={
            "iterations": 5,
            "depth": 2,
            "learning_rate": 0.1,
            "random_seed": 42,
            "thread_count": 1,
        },
    )
    predictions = predict_catboost_quantiles(model, features.iloc[:7], quantiles)
    prefix_predictions = predict_catboost_quantiles(
        model, features.iloc[:7], quantiles, tree_count=3
    )

    assert predictions.shape == (7, 3)
    assert prefix_predictions.shape == (7, 3)
    assert np.isfinite(predictions).all()
    assert np.isfinite(prefix_predictions).all()


def test_catboost_wrapper_rejects_managed_parameter_overrides() -> None:
    with pytest.raises(ValueError, match="managed values"):
        fit_catboost_quantiles(
            pd.DataFrame({"x": [1.0, 2.0]}),
            np.array([1.0, 2.0]),
            np.array([0.5]),
            categorical_features=(),
            parameters={"loss_function": "RMSE"},
        )
