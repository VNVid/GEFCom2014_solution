from __future__ import annotations

import pandas as pd
import pytest

from gefcom2014.models import select_features_from_loss_change


def _importance(values: dict[str, list[float]]) -> pd.DataFrame:
    records = []
    for feature, feature_values in values.items():
        for fold, value in enumerate(feature_values, start=1):
            records.append(
                {
                    "origin": f"fold_{fold}",
                    "feature": feature,
                    "loss_function_change": value,
                }
            )
    return pd.DataFrame(records)


def test_selection_requires_positive_median_and_fold_consistency() -> None:
    summary, selected = select_features_from_loss_change(
        _importance(
            {
                "stable": [3, 2, 1, 2, 3, -1],
                "unstable": [5, 4, 3, -1, -2, -3],
                "harmful": [-1, -1, -1, -1, 1, 1],
            }
        ),
        expected_folds=6,
        minimum_positive_folds=4,
        maximum_features=5,
    )

    assert selected == ("stable",)
    reasons = summary.set_index("feature")["selection_reason"]
    assert reasons["unstable"] == "insufficient_positive_folds"
    assert reasons["harmful"] == "non_positive_median"


def test_selection_applies_fixed_feature_cap_after_ranking() -> None:
    summary, selected = select_features_from_loss_change(
        _importance(
            {
                "best": [3, 3, 3, 3, 3, 3],
                "second": [2, 2, 2, 2, 2, 2],
                "third": [1, 1, 1, 1, 1, 1],
            }
        ),
        expected_folds=6,
        minimum_positive_folds=4,
        maximum_features=2,
    )

    assert selected == ("best", "second")
    assert summary.set_index("feature").loc["third", "selection_reason"] == "feature_cap"


def test_selection_rejects_incomplete_fold_coverage() -> None:
    with pytest.raises(ValueError, match="exactly one value"):
        select_features_from_loss_change(
            _importance({"feature": [1, 2, 3, 4, 5]}),
            expected_folds=6,
            minimum_positive_folds=4,
            maximum_features=5,
        )
