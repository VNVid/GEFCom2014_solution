"""Historical seasonal load profiles for target hours."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from ..analogues import previous_seasonal_load_samples
from ._shared import validate_pre_origin_history


def build_seasonal_load_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    day_type_window_days: int = 8,
    day_type_quantiles: Iterable[float] = (0.10, 0.50, 0.90),
    include_day_type_count: bool = True,
    hour_of_week_window_days: int = 15,
    hour_of_week_statistics: Iterable[str] = ("mean", "std", "count"),
    quantile_method: str = "linear",
) -> pd.DataFrame:
    """Build robust day-type and weekday-specific seasonal load summaries.

    Both analogue pools are centred on the target calendar date in completed
    prior seasonal cycles. The day-type pool reuses the empirical baseline's
    weekday/weekend rule; the narrower hour-of-week pool requires the exact
    target weekday. Current-cycle observations cannot enter either pool.
    """

    validate_pre_origin_history(history, target, origin, value_columns=("load",))
    levels = np.asarray(tuple(day_type_quantiles), dtype=float)
    if levels.ndim != 1 or not np.isfinite(levels).all():
        raise ValueError("day_type_quantiles must be a finite one-dimensional sequence")
    if levels.size and (
        not np.all((levels > 0) & (levels < 1))
        or not np.all(np.diff(levels) > 0)
    ):
        raise ValueError("day_type_quantiles must be strictly increasing inside (0, 1)")
    percentile_numbers = np.rint(100 * levels).astype(int)
    if levels.size and not np.allclose(100 * levels, percentile_numbers):
        raise ValueError("day_type_quantiles must be whole percentiles")
    if len(percentile_numbers) != len(set(percentile_numbers)):
        raise ValueError("day_type_quantiles produce duplicate feature names")

    how_statistics = tuple(str(value) for value in hour_of_week_statistics)
    allowed_statistics = {"mean", "std", "count"}
    unknown_statistics = sorted(set(how_statistics) - allowed_statistics)
    if unknown_statistics:
        raise ValueError(f"Unknown hour_of_week_statistics {unknown_statistics}")
    if len(how_statistics) != len(set(how_statistics)):
        raise ValueError("hour_of_week_statistics must not contain duplicates")
    if levels.size == 0 and not include_day_type_count and not how_statistics:
        raise ValueError("seasonal_load_profile configuration must produce a feature")

    day_type_samples = previous_seasonal_load_samples(
        history,
        target,
        window_days=int(day_type_window_days),
        day_match="day_type",
    )
    hour_of_week_samples = previous_seasonal_load_samples(
        history,
        target,
        window_days=int(hour_of_week_window_days),
        day_match="day_of_week",
    )

    features = pd.DataFrame(index=target.index)
    for level, percentile in zip(levels, percentile_numbers):
        name = f"load_seasonal_daytype_{day_type_window_days}d_q{percentile:02d}"
        features[name] = [
            float(np.quantile(samples, level, method=quantile_method))
            if samples.size
            else np.nan
            for samples in day_type_samples
        ]
    if include_day_type_count:
        features[f"load_seasonal_daytype_{day_type_window_days}d_count"] = np.asarray(
            [samples.size for samples in day_type_samples], dtype=np.int16
        )

    for statistic in how_statistics:
        name = f"load_seasonal_how_{hour_of_week_window_days}d_{statistic}"
        if statistic == "count":
            values = np.asarray(
                [samples.size for samples in hour_of_week_samples], dtype=np.int16
            )
        elif statistic == "mean":
            values = np.asarray(
                [
                    float(np.mean(samples)) if samples.size else np.nan
                    for samples in hour_of_week_samples
                ]
            )
        else:
            values = np.asarray(
                [
                    float(np.std(samples, ddof=0)) if samples.size else np.nan
                    for samples in hour_of_week_samples
                ]
            )
        features[name] = values
    return features
