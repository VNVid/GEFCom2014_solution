"""Simple forecast-time-safe baselines for monthly load forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


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

    lookup = training.set_index(["zone_id", "period_start"])["load"]
    if not lookup.index.is_unique:
        raise ValueError("Training rows must be unique by zone and operating hour")

    prior_periods = target["period_start"] - pd.DateOffset(years=1)
    prior_keys = pd.MultiIndex.from_arrays(
        [target["zone_id"].to_numpy(), prior_periods.to_numpy()],
        names=["zone_id", "period_start"],
    )
    point_forecast = lookup.reindex(prior_keys)
    if point_forecast.isna().any():
        missing = prior_keys[point_forecast.isna().to_numpy()][:3].tolist()
        raise ValueError(f"Missing previous-year loads for target rows; examples={missing}")

    predictions = np.repeat(
        point_forecast.to_numpy(dtype=float)[:, None], len(quantiles), axis=1
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

    if not isinstance(window_days, int) or window_days < 0:
        raise ValueError("window_days must be a non-negative integer")
    lookup_series = training.set_index(["zone_id", "period_start"])["load"]
    if not lookup_series.index.is_unique:
        raise ValueError("Training rows must be unique by zone and operating hour")
    lookup = lookup_series.to_dict()

    quantile_levels = np.asarray(quantiles, dtype=float)
    predictions = np.empty((len(target), len(quantile_levels)), dtype=float)
    sample_sizes = np.empty(len(target), dtype=np.int16)
    first_anchor_year = int(training["period_start"].dt.year.min()) - 1

    for row_number, row in enumerate(target.itertuples(index=False)):
        target_day = row.period_start.normalize()
        target_is_weekend = target_day.dayofweek >= 5
        samples: list[float] = []

        # Starting one year before the first observed year retains any partial
        # year-end window that overlaps the beginning of the load history.
        for anchor_year in range(first_anchor_year, target_day.year):
            try:
                anchor = target_day.replace(year=anchor_year)
            except ValueError:
                # A target on February 29 has no exact non-leap analogue.
                anchor = target_day.replace(year=anchor_year, day=28)

            for offset in range(-window_days, window_days + 1):
                candidate_day = anchor + pd.Timedelta(days=offset)
                if (candidate_day.dayofweek >= 5) != target_is_weekend:
                    continue
                candidate_time = candidate_day + pd.Timedelta(
                    hours=row.period_start.hour
                )
                value = lookup.get((row.zone_id, candidate_time))
                if value is not None:
                    samples.append(float(value))

        if not samples:
            raise ValueError(
                "No seasonal empirical candidates for "
                f"zone={row.zone_id}, period_start={row.period_start}"
            )
        predictions[row_number] = np.quantile(
            np.asarray(samples), quantile_levels, method=quantile_method
        )
        sample_sizes[row_number] = len(samples)

    return predictions, sample_sizes
