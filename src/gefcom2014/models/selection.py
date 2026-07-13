"""Deterministic aggregation rules for rolling-origin feature screening."""

from __future__ import annotations

import numpy as np
import pandas as pd


def select_features_from_loss_change(
    importance: pd.DataFrame,
    *,
    expected_folds: int,
    minimum_positive_folds: int,
    maximum_features: int,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    """Aggregate fold-level LossFunctionChange values and select features.

    Every feature follows the same rule: its median importance must be positive
    and it must be positive in at least ``minimum_positive_folds``. Eligible
    features are ordered by median importance and truncated only by the fixed
    feature cap.
    """

    required = {"origin", "feature", "loss_function_change"}
    missing = sorted(required - set(importance.columns))
    if missing:
        raise ValueError(f"Feature-importance table is missing columns {missing}")
    if expected_folds <= 0:
        raise ValueError("expected_folds must be positive")
    if not 1 <= minimum_positive_folds <= expected_folds:
        raise ValueError("minimum_positive_folds must lie inside [1, expected_folds]")
    if maximum_features <= 0:
        raise ValueError("maximum_features must be positive")
    if importance.empty:
        raise ValueError("Feature-importance table must not be empty")
    if importance.duplicated(["origin", "feature"]).any():
        raise ValueError("Feature importance must be unique by origin and feature")
    values = importance["loss_function_change"].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("LossFunctionChange values must be finite")

    fold_counts = importance.groupby("feature")["origin"].nunique()
    incomplete = fold_counts.loc[fold_counts.ne(expected_folds)]
    if not incomplete.empty:
        raise ValueError(
            "Every feature must have exactly one value per screening fold; "
            f"incomplete={incomplete.to_dict()}"
        )

    ranked = importance.copy()
    ranked["fold_rank"] = ranked.groupby("origin")[
        "loss_function_change"
    ].rank(method="average", ascending=False)
    records: list[dict[str, float | int | str | bool]] = []
    for feature, group in ranked.groupby("feature", sort=False):
        feature_values = group["loss_function_change"].to_numpy(dtype=float)
        median = float(np.median(feature_values))
        positive_folds = int(np.sum(feature_values > 0.0))
        eligible = median > 0.0 and positive_folds >= minimum_positive_folds
        if eligible:
            reason = "eligible"
        elif median <= 0.0:
            reason = "non_positive_median"
        else:
            reason = "insufficient_positive_folds"
        records.append(
            {
                "feature": str(feature),
                "mean_loss_function_change": float(np.mean(feature_values)),
                "median_loss_function_change": median,
                "std_loss_function_change": float(np.std(feature_values, ddof=1)),
                "min_loss_function_change": float(np.min(feature_values)),
                "max_loss_function_change": float(np.max(feature_values)),
                "positive_folds": positive_folds,
                "nonzero_folds": int(np.sum(~np.isclose(feature_values, 0.0))),
                "mean_fold_rank": float(group["fold_rank"].mean()),
                "eligible": eligible,
                "selection_reason": reason,
            }
        )

    summary = pd.DataFrame(records).sort_values(
        ["eligible", "median_loss_function_change", "mean_fold_rank", "feature"],
        ascending=[False, False, True, True],
        ignore_index=True,
    )
    eligible_indices = summary.index[summary["eligible"]].tolist()
    selected_indices = eligible_indices[:maximum_features]
    summary["selected"] = False
    summary.loc[selected_indices, "selected"] = True
    capped = eligible_indices[maximum_features:]
    if capped:
        summary.loc[capped, "selection_reason"] = "feature_cap"
    summary["selection_rank"] = np.nan
    summary.loc[selected_indices, "selection_rank"] = np.arange(
        1, len(selected_indices) + 1
    )
    selected = tuple(summary.loc[summary["selected"], "feature"])
    if not selected:
        raise ValueError("Selection rule removed every candidate feature")
    return summary, selected
