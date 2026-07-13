"""Historical analogue selection shared by baselines and model features."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype


def load_values_at_periods(
    training: pd.DataFrame,
    target: pd.DataFrame,
    reference_periods: pd.Series | pd.DatetimeIndex,
) -> np.ndarray:
    """Look up historical load by target zone and reference timestamp.

    Values are returned in target-row order. A missing historical key is
    represented by ``NaN`` so callers can choose whether absence is a fatal
    error (a baseline) or an explicit missing feature (a model input).
    """

    required_training = {"zone_id", "period_start", "load"}
    missing_training = sorted(required_training - set(training.columns))
    if missing_training:
        raise ValueError(f"Training frame is missing columns {missing_training}")
    if "zone_id" not in target:
        raise ValueError("Target frame is missing column 'zone_id'")
    if not is_datetime64_any_dtype(training["period_start"]):
        raise TypeError("Training period_start must have datetime64 dtype")
    if training["period_start"].isna().any() or training["load"].isna().any():
        raise ValueError("Training timestamps and load must not be missing")
    if not np.isfinite(training["load"].to_numpy(dtype=float)).all():
        raise ValueError("Training load must contain only finite values")
    if training.duplicated(["zone_id", "period_start"]).any():
        raise ValueError("Training rows must be unique by zone and operating hour")

    references = pd.DatetimeIndex(reference_periods)
    if len(references) != len(target):
        raise ValueError("reference_periods must contain one timestamp per target row")
    if references.isna().any():
        raise ValueError("Reference timestamps must not be missing")

    lookup = training.set_index(["zone_id", "period_start"])["load"]
    keys = pd.MultiIndex.from_arrays(
        [target["zone_id"].to_numpy(), references.to_numpy()],
        names=["zone_id", "period_start"],
    )
    return lookup.reindex(keys).to_numpy(dtype=float)


def previous_seasonal_load_samples(
    training: pd.DataFrame,
    target: pd.DataFrame,
    window_days: int,
    day_match: Literal["day_type", "day_of_week"],
) -> list[np.ndarray]:
    """Return load samples from windows centred in completed seasonal cycles.

    Every candidate matches target zone and operating hour. ``day_type`` keeps
    weekday/weekend status, while ``day_of_week`` requires the exact weekday.
    Windows are anchored on the target month/day in earlier years, so recent
    observations from the current seasonal cycle cannot enter accidentally.
    """

    samples = previous_seasonal_samples(
        training,
        target,
        value_columns=("load",),
        window_days=window_days,
        day_match=day_match,
    )
    return [values[:, 0] for values in samples]


def previous_seasonal_samples(
    training: pd.DataFrame,
    target: pd.DataFrame,
    value_columns: Iterable[str],
    window_days: int,
    day_match: Literal["none", "day_type", "day_of_week"],
) -> list[np.ndarray]:
    """Return multivariate samples from completed seasonal cycles.

    Candidate rows match target zone and operating hour. Calendar windows are
    anchored in earlier years, and ``day_match`` optionally filters them by
    weekday/weekend status or exact weekday. Each returned array has one column
    per requested value and one row per available historical analogue.
    """

    columns = tuple(value_columns)
    if not columns:
        raise ValueError("value_columns must contain at least one column")
    if len(columns) != len(set(columns)):
        raise ValueError("value_columns must not contain duplicates")
    if not isinstance(window_days, int) or window_days < 0:
        raise ValueError("window_days must be a non-negative integer")
    if day_match not in {"none", "day_type", "day_of_week"}:
        raise ValueError(
            "day_match must be 'none', 'day_type', or 'day_of_week'"
        )
    required_training = {"zone_id", "period_start", *columns}
    required_target = {"zone_id", "period_start"}
    missing_training = sorted(required_training - set(training.columns))
    missing_target = sorted(required_target - set(target.columns))
    if missing_training:
        raise ValueError(f"Training frame is missing columns {missing_training}")
    if missing_target:
        raise ValueError(f"Target frame is missing columns {missing_target}")
    if target.empty:
        raise ValueError("Target frame must contain at least one row")
    training_is_datetime = is_datetime64_any_dtype(training["period_start"])
    target_is_datetime = is_datetime64_any_dtype(target["period_start"])
    if not training_is_datetime or not target_is_datetime:
        raise TypeError("Training and target period_start must have datetime64 dtype")
    if training["period_start"].isna().any() or target["period_start"].isna().any():
        raise ValueError("Training and target timestamps must not be missing")
    if training[list(columns)].isna().any().any() or not np.isfinite(
        training[list(columns)].to_numpy(dtype=float)
    ).all():
        raise ValueError("Training sample values must contain only finite values")
    if training.duplicated(["zone_id", "period_start"]).any():
        raise ValueError("Training rows must be unique by zone and operating hour")
    if (
        not training.empty
        and training["period_start"].max() >= target["period_start"].min()
    ):
        raise ValueError("Training observations must be strictly earlier than targets")

    training_values = training.loc[:, columns].to_numpy(dtype=float)
    lookup = {
        (zone_id, period_start): position
        for position, (zone_id, period_start) in enumerate(
            zip(training["zone_id"], training["period_start"])
        )
    }
    first_year_by_zone = (
        training.groupby("zone_id")["period_start"].min().dt.year.to_dict()
    )

    samples_by_target: list[np.ndarray] = []
    for row in target.itertuples(index=False):
        target_day = row.period_start.normalize()
        target_day_of_week = target_day.dayofweek
        first_year = first_year_by_zone.get(row.zone_id)
        sample_positions: list[int] = []
        if first_year is not None:
            # The preceding anchor year retains a partial window that overlaps
            # the first available history year at the December/January boundary.
            for anchor_year in range(int(first_year) - 1, target_day.year):
                try:
                    anchor = target_day.replace(year=anchor_year)
                except ValueError:
                    anchor = target_day.replace(year=anchor_year, day=28)

                for offset in range(-window_days, window_days + 1):
                    candidate_day = anchor + pd.Timedelta(days=offset)
                    if day_match == "none":
                        matches_day = True
                    elif day_match == "day_type":
                        matches_day = (candidate_day.dayofweek >= 5) == (
                            target_day_of_week >= 5
                        )
                    else:
                        matches_day = candidate_day.dayofweek == target_day_of_week
                    if not matches_day:
                        continue
                    candidate_time = candidate_day + pd.Timedelta(
                        hours=row.period_start.hour
                    )
                    position = lookup.get((row.zone_id, candidate_time))
                    if position is not None:
                        sample_positions.append(position)
        samples_by_target.append(
            training_values[sample_positions]
            if sample_positions
            else np.empty((0, len(columns)), dtype=float)
        )
    return samples_by_target
