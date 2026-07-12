from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.backtesting import monthly_folds
from gefcom2014.features import TARGET_CATEGORICAL_FEATURES
from gefcom2014.model_data import (
    build_monthly_modeling_dataset,
    rolling_origin_masks,
)


def _load_frame(start: str, end: str) -> pd.DataFrame:
    periods = pd.date_range(start, end, freq="h", inclusive="left")
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": periods,
            "load": np.arange(len(periods), dtype=float) + 100.0,
        }
    )


def test_monthly_modeling_dataset_preserves_features_labels_and_metadata() -> None:
    folds = monthly_folds("2009-01-01", "2009-03-01")
    dataset = build_monthly_modeling_dataset(
        _load_frame("2008-12-31", "2009-03-01"),
        folds,
        {"target_time": {}},
    )

    assert dataset.features.shape == (744 + 672, 11)
    assert len(dataset.target) == len(dataset.features)
    assert len(dataset.metadata) == len(dataset.features)
    assert dataset.target.name == "load"
    assert dataset.metadata.columns.tolist() == [
        "zone_id",
        "period_start",
        "origin",
        "forecast_end",
    ]
    assert dataset.metadata["origin"].value_counts().to_dict() == {
        pd.Timestamp("2009-01-01"): 744,
        pd.Timestamp("2009-02-01"): 672,
    }
    assert dataset.manifest["feature_count"].tolist() == [11, 11]
    assert dataset.manifest["missing_feature_values"].tolist() == [0, 0]
    assert dataset.categorical_features == TARGET_CATEGORICAL_FEATURES


def test_rolling_origin_masks_require_labels_to_have_been_revealed() -> None:
    dataset = build_monthly_modeling_dataset(
        _load_frame("2008-12-31", "2009-04-01"),
        monthly_folds("2009-01-01", "2009-04-01"),
        {"target_time": {}},
    )

    training, evaluation = rolling_origin_masks(
        dataset.metadata, "2009-02-01"
    )

    assert training.sum() == 744
    assert evaluation.sum() == 672
    assert not np.logical_and(training, evaluation).any()
    assert dataset.metadata.loc[training, "forecast_end"].max() == pd.Timestamp(
        "2009-02-01"
    )
    assert dataset.metadata.loc[evaluation, "origin"].nunique() == 1


def test_modeling_dataset_can_retain_or_reject_missing_features() -> None:
    frame = _load_frame("2008-12-31", "2009-02-01")
    folds = monthly_folds("2009-01-01", "2009-02-01")
    groups = {
        "recent_load": {
            "mean_windows_days": [7],
            "std_windows_days": [],
            "difference_windows_days": [],
            "trend_windows_days": [],
            "yoy_windows_days": [],
        }
    }

    with pytest.raises(ValueError, match="missing values"):
        build_monthly_modeling_dataset(frame, folds, groups)

    dataset = build_monthly_modeling_dataset(
        frame,
        folds,
        groups,
        require_complete_features=False,
    )
    assert dataset.features["load_mean_7d"].isna().all()
    assert dataset.manifest.loc[0, "missing_feature_values"] == 744


def test_rolling_origin_masks_reject_unknown_evaluation_origin() -> None:
    metadata = pd.DataFrame(
        {
            "origin": [pd.Timestamp("2009-01-01")],
            "forecast_end": [pd.Timestamp("2009-02-01")],
        }
    )
    with pytest.raises(ValueError, match="No evaluation rows"):
        rolling_origin_masks(metadata, "2010-01-01")
