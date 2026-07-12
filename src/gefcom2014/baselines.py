"""Simple forecast-time-safe baselines for monthly load forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .analogues import load_values_at_periods, previous_seasonal_load_samples


def seasonal_naive_forecast(
    training: pd.DataFrame,
    target: pd.DataFrame,
    quantiles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Repeat the same calendar hour one year earlier at every quantile.

    This reproduces the naive benchmark supplied with the GEFCom2014 load
    track.  February 29 maps to February 28 in a non-leap reference year,
    following pandas' calendar-year offset convention.
    """

    prior_periods = target["period_start"] - pd.DateOffset(years=1)
    point_forecast = load_values_at_periods(training, target, prior_periods)
    missing_mask = np.isnan(point_forecast)
    if missing_mask.any():
        missing = list(
            zip(
                target.loc[missing_mask, "zone_id"].to_numpy(),
                prior_periods.loc[missing_mask].to_numpy(),
            )
        )[:3]
        raise ValueError(f"Missing previous-year loads for target rows; examples={missing}")

    predictions = np.repeat(
        point_forecast[:, None], len(quantiles), axis=1
    )
    return predictions, np.ones(len(target), dtype=np.int16)


def seasonal_empirical_forecast(
    training: pd.DataFrame,
    target: pd.DataFrame,
    quantiles: np.ndarray,
    window_days: int = 8,
    quantile_method: str = "linear",
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast empirical quantiles from completed prior seasonal cycles.

    For each target hour, candidate windows are centred on the same calendar
    date in earlier years.  The inclusive calendar window is then filtered to
    the same zone, operating hour, and weekday/weekend type.  Because the
    lookup contains only the fold's pre-origin training snapshot, every
    selected observation is also strictly available at forecast time.
    """

    quantile_levels = np.asarray(quantiles, dtype=float)
    predictions = np.empty((len(target), len(quantile_levels)), dtype=float)
    sample_sizes = np.empty(len(target), dtype=np.int16)
    samples_by_target = previous_seasonal_load_samples(
        training, target, window_days=window_days, day_match="day_type"
    )
    for row_number, samples in enumerate(samples_by_target):
        if samples.size == 0:
            row = target.iloc[row_number]
            raise ValueError(
                "No seasonal empirical candidates for "
                f"zone={row['zone_id']}, period_start={row['period_start']}"
            )
        predictions[row_number] = np.quantile(
            samples, quantile_levels, method=quantile_method
        )
        sample_sizes[row_number] = len(samples)

    return predictions, sample_sizes
