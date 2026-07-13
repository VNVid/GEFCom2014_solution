"""Recent-level adjustments to historical seasonal load anchors."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype

from ..analogues import previous_seasonal_load_samples
from ._shared import complete_hourly_window, validate_pre_origin_history


def build_seasonal_load_adjustment_features(
    history: pd.DataFrame,
    target: pd.DataFrame,
    origin: str | pd.Timestamp,
    *,
    recent_window_days: int = 28,
    seasonal_window_days: int = 8,
    seasonal_quantile: float = 0.50,
    quantile_method: str = "linear",
    adjustment_types: Iterable[str] = ("multiplicative",),
) -> pd.DataFrame:
    """Adjust a seasonal anchor to the load level observed before ``origin``.

    For every hour in the complete recent window, a seasonal day-type estimate
    is constructed from earlier annual cycles. The level ratio is the recent
    observed mean divided by the mean of those estimates. Multiplying the
    target's seasonal estimate by that ratio produces an anchor which retains
    historical shape while reflecting the current system level.

    The optional additive form shifts the target seasonal anchor by the recent
    mean residual instead. The recent window ends strictly at ``origin``. If it
    or any corresponding seasonal estimate is incomplete, every requested
    adjustment is ``NaN`` for that zone.
    """

    forecast_origin = validate_pre_origin_history(
        history, target, origin, value_columns=("load",)
    )
    if "period_start" not in target:
        raise ValueError("Target frame is missing column 'period_start'")
    if not is_datetime64_any_dtype(target["period_start"]):
        raise TypeError("Target period_start must have a pandas datetime64 dtype")
    if target["period_start"].isna().any():
        raise ValueError("Target timestamps must not be missing")
    if target["period_start"].dt.tz is not None:
        raise ValueError("Target period_start must be timezone-naive")
    if not isinstance(recent_window_days, int) or recent_window_days <= 0:
        raise ValueError("recent_window_days must be a positive integer")
    if not isinstance(seasonal_window_days, int) or seasonal_window_days < 0:
        raise ValueError("seasonal_window_days must be a non-negative integer")
    if not np.isfinite(seasonal_quantile) or not 0 < seasonal_quantile < 1:
        raise ValueError("seasonal_quantile must be finite and inside (0, 1)")
    percentile = int(round(100 * seasonal_quantile))
    if not np.isclose(100 * seasonal_quantile, percentile):
        raise ValueError("seasonal_quantile must be a whole percentile")
    requested = tuple(str(value) for value in adjustment_types)
    unknown = sorted(set(requested) - {"multiplicative", "additive"})
    if unknown:
        raise ValueError(f"Unknown seasonal adjustment types {unknown}")
    if not requested or len(requested) != len(set(requested)):
        raise ValueError("adjustment_types must be non-empty and unique")

    target_samples = previous_seasonal_load_samples(
        history,
        target,
        window_days=seasonal_window_days,
        day_match="day_type",
    )
    target_seasonal = np.asarray(
        [
            float(np.quantile(samples, seasonal_quantile, method=quantile_method))
            if samples.size
            else np.nan
            for samples in target_samples
        ]
    )

    recent_start = forecast_origin - pd.Timedelta(days=recent_window_days)
    adjustment_by_zone: dict[object, tuple[float, float]] = {}
    for zone_id in target["zone_id"].unique():
        zone_history = history.loc[history["zone_id"] == zone_id].sort_values(
            "period_start"
        )
        recent = complete_hourly_window(zone_history, recent_start, forecast_origin)
        if recent is None:
            adjustment_by_zone[zone_id] = (np.nan, np.nan)
            continue

        earlier_cycles = zone_history.loc[
            zone_history["period_start"] < recent_start
        ]
        recent_samples = previous_seasonal_load_samples(
            earlier_cycles,
            recent.loc[:, ["zone_id", "period_start"]],
            window_days=seasonal_window_days,
            day_match="day_type",
        )
        recent_seasonal = np.asarray(
            [
                float(
                    np.quantile(
                        samples, seasonal_quantile, method=quantile_method
                    )
                )
                if samples.size
                else np.nan
                for samples in recent_samples
            ]
        )
        if not np.isfinite(recent_seasonal).all():
            adjustment_by_zone[zone_id] = (np.nan, np.nan)
            continue
        seasonal_mean = float(np.mean(recent_seasonal))
        recent_mean = float(recent["load"].mean())
        ratio = recent_mean / seasonal_mean if seasonal_mean != 0.0 else np.nan
        adjustment_by_zone[zone_id] = (ratio, recent_mean - seasonal_mean)

    ratio = target["zone_id"].map(
        {zone_id: values[0] for zone_id, values in adjustment_by_zone.items()}
    ).to_numpy(dtype=float)
    delta = target["zone_id"].map(
        {zone_id: values[1] for zone_id, values in adjustment_by_zone.items()}
    ).to_numpy(dtype=float)
    prefix = f"load_seasonal_daytype_{seasonal_window_days}d_q{percentile:02d}"
    output: dict[str, np.ndarray] = {}
    for adjustment in requested:
        if adjustment == "multiplicative":
            output[f"load_seasonal_level_ratio_{recent_window_days}d"] = ratio
            output[f"{prefix}_scaled_{recent_window_days}d"] = target_seasonal * ratio
        else:
            output[f"load_seasonal_level_delta_{recent_window_days}d"] = delta
            output[f"{prefix}_shifted_{recent_window_days}d"] = target_seasonal + delta
    return pd.DataFrame(output, index=target.index)
