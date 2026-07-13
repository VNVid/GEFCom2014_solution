from __future__ import annotations

import pandas as pd
import pytest

from gefcom2014.models import resolve_candidate_parameters, resolve_selected_features


def test_model_parameters_are_resolved_from_exact_search_candidate() -> None:
    summary = pd.DataFrame(
        [
            {
                "candidate": "winner",
                "depth": 3,
                "learning_rate": 0.08,
                "l2_leaf_reg": 5,
                "iterations": 125,
            }
        ]
    )

    parameters = resolve_candidate_parameters(
        summary,
        "winner",
        {"random_seed": 42, "thread_count": -1},
    )

    assert parameters == {
        "random_seed": 42,
        "thread_count": -1,
        "depth": 3,
        "learning_rate": 0.08,
        "l2_leaf_reg": 5.0,
        "iterations": 125,
    }


def test_selected_features_validate_categorical_manifest() -> None:
    manifest = {
        "features": ["hour", "seasonal_mean"],
        "categorical_features": ["hour"],
    }
    features, categorical = resolve_selected_features(
        ["hour", "seasonal_mean"],
        ["hour"],
        manifest,
    )

    assert features == ["hour", "seasonal_mean"]
    assert categorical == ["hour"]

    manifest["categorical_features"] = []
    with pytest.raises(ValueError, match="categorical"):
        resolve_selected_features(
            ["hour", "seasonal_mean"], ["hour"], manifest
        )
