"""Monthly pseudo-origin datasets for rolling forecasting experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .backtesting import MonthlyFold, data_for_fold
from .features import build_feature_matrix, categorical_feature_names


@dataclass(frozen=True)
class MonthlyModelingDataset:
    """Aligned features, labels, metadata, and fold-level construction details."""

    features: pd.DataFrame
    target: pd.Series
    metadata: pd.DataFrame
    manifest: pd.DataFrame
    categorical_features: tuple[str, ...]


def build_monthly_modeling_dataset(
    load_frame: pd.DataFrame,
    folds: Sequence[MonthlyFold],
    feature_groups: Mapping[str, Mapping[str, Any] | None],
    *,
    weather_frame: pd.DataFrame | None = None,
    require_complete_features: bool = True,
) -> MonthlyModelingDataset:
    """Construct labeled monthly pseudo-forecasts with frozen feature origins.

    Each fold is treated exactly like a real forecast: feature history ends
    strictly before the month's first hour, while the target month's load is
    attached only after feature construction. A separately supplied weather
    frame is sliced at the same origin and may begin earlier than load history.
    """

    selected_folds = tuple(folds)
    if not selected_folds:
        raise ValueError("At least one monthly fold is required")
    origins = [fold.origin for fold in selected_folds]
    if origins != sorted(origins) or len(origins) != len(set(origins)):
        raise ValueError(
            "Monthly folds must have unique origins in chronological order"
        )
    for previous, current in zip(selected_folds, selected_folds[1:]):
        if previous.end > current.origin:
            raise ValueError("Monthly folds must not overlap")

    feature_blocks: list[pd.DataFrame] = []
    target_blocks: list[pd.Series] = []
    metadata_blocks: list[pd.DataFrame] = []
    manifest_records: list[dict[str, object]] = []
    expected_columns: tuple[str, ...] | None = None

    for fold in selected_folds:
        history, target = data_for_fold(load_frame, fold)
        weather_history = None
        if weather_frame is not None:
            weather_history = weather_frame.loc[
                weather_frame["period_start"] < fold.origin
            ].copy()
            if weather_history.empty:
                raise ValueError(f"No weather history before origin {fold.origin}")

        # Target outcomes and realized temperatures are deliberately withheld
        # from the feature builder even though they exist in the label store.
        target_times = target.loc[:, ["zone_id", "period_start"]]
        features = build_feature_matrix(
            history,
            target_times,
            fold.origin,
            feature_groups,
            weather_history=weather_history,
        )
        columns = tuple(features.columns)
        if expected_columns is None:
            expected_columns = columns
        elif columns != expected_columns:
            raise ValueError(f"Feature schema changed at origin {fold.origin}")

        numeric = features.select_dtypes(include=np.number).to_numpy(dtype=float)
        if np.isinf(numeric).any():
            raise ValueError(
                f"Feature matrix contains infinite values at {fold.origin}"
            )
        missing_count = int(features.isna().sum().sum())
        if require_complete_features and missing_count:
            missing = features.isna().sum()
            examples = missing.loc[missing.gt(0)].head(5).to_dict()
            raise ValueError(
                f"Feature matrix contains {missing_count} missing values at "
                f"{fold.origin}; columns={examples}"
            )

        feature_blocks.append(features.reset_index(drop=True))
        labels = target["load"].to_numpy(dtype=float)
        if not np.isfinite(labels).all():
            raise ValueError(f"Target loads must be finite at origin {fold.origin}")
        target_blocks.append(pd.Series(labels, name="load"))
        metadata_blocks.append(
            pd.DataFrame(
                {
                    "zone_id": target["zone_id"].to_numpy(),
                    "period_start": target["period_start"].to_numpy(),
                    "origin": fold.origin,
                    "forecast_end": fold.end,
                }
            )
        )
        manifest_records.append(
            {
                "origin": fold.origin,
                "forecast_end": fold.end,
                "load_history_start": history["period_start"].min(),
                "load_history_end": history["period_start"].max(),
                "weather_history_start": (
                    weather_history["period_start"].min()
                    if weather_history is not None
                    else pd.NaT
                ),
                "weather_history_end": (
                    weather_history["period_start"].max()
                    if weather_history is not None
                    else pd.NaT
                ),
                "target_rows": len(target),
                "feature_count": len(columns),
                "missing_feature_values": missing_count,
            }
        )

    features = pd.concat(feature_blocks, ignore_index=True)
    labels = pd.concat(target_blocks, ignore_index=True)
    metadata = pd.concat(metadata_blocks, ignore_index=True)
    if not (len(features) == len(labels) == len(metadata)):
        raise AssertionError("Modeling dataset components lost row alignment")

    categorical = categorical_feature_names(feature_groups)
    if not set(categorical).issubset(features.columns):
        raise AssertionError(
            "Configured categorical features are absent from the matrix"
        )
    return MonthlyModelingDataset(
        features=features,
        target=labels,
        metadata=metadata,
        manifest=pd.DataFrame(manifest_records),
        categorical_features=categorical,
    )


def rolling_origin_masks(
    metadata: pd.DataFrame,
    evaluation_origin: str | pd.Timestamp,
) -> tuple[np.ndarray, np.ndarray]:
    """Return rows available for fitting and rows evaluated at one origin.

    A pseudo-forecast label becomes available at its ``forecast_end``. Thus a
    fold ending exactly at the new evaluation origin is legal training data,
    while the evaluation month's own rows remain excluded.
    """

    required = {"origin", "forecast_end"}
    missing = sorted(required - set(metadata.columns))
    if missing:
        raise ValueError(f"Modeling metadata is missing columns {missing}")
    origin = pd.Timestamp(evaluation_origin)
    if origin.tz is not None:
        raise ValueError("evaluation_origin must be timezone-naive")

    training = metadata["forecast_end"].le(origin).to_numpy(dtype=bool)
    evaluation = metadata["origin"].eq(origin).to_numpy(dtype=bool)
    if not evaluation.any():
        raise ValueError(f"No evaluation rows for origin {origin}")
    if np.logical_and(training, evaluation).any():
        raise AssertionError("Evaluation labels entered their own training rows")
    return training, evaluation
