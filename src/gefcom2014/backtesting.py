"""Monthly rolling-origin folds with explicit leakage boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .data import interval_start


@dataclass(frozen=True)
class MonthlyFold:
    """One calendar-month forecast made at ``origin``."""

    origin: pd.Timestamp
    end: pd.Timestamp


def monthly_folds(start: str | pd.Timestamp, end: str | pd.Timestamp) -> list[MonthlyFold]:
    """Create consecutive half-open calendar-month folds in ``[start, end)``."""

    first = pd.Timestamp(start)
    stop = pd.Timestamp(end)
    for name, value in (("start", first), ("end", stop)):
        if value != value.normalize() or value.day != 1:
            raise ValueError(f"{name} must be midnight on the first day of a month")
        if value.tz is not None:
            raise ValueError(f"{name} must be timezone-naive")
    if first >= stop:
        raise ValueError("start must be earlier than end")

    boundaries = pd.date_range(first, stop, freq="MS")
    if boundaries[-1] != stop:
        raise ValueError("end must be an exact monthly boundary")
    return [
        MonthlyFold(origin=boundaries[i], end=boundaries[i + 1])
        for i in range(len(boundaries) - 1)
    ]


def prepare_backtest_frame(actuals: pd.DataFrame) -> pd.DataFrame:
    """Add canonical operating time while preserving model-relevant columns."""

    required = {"zone_id", "timestamp", "load"}
    missing = sorted(required - set(actuals.columns))
    if missing:
        raise ValueError(f"Backtest actuals are missing columns {missing}")

    frame = actuals.copy()
    if frame["load"].isna().any():
        raise ValueError("Backtest actuals must not contain missing loads")
    frame["period_start"] = interval_start(frame["timestamp"])
    frame = frame.sort_values(["period_start", "zone_id"], ignore_index=True)
    if frame.duplicated(["zone_id", "period_start"]).any():
        raise ValueError("Backtest actuals contain duplicate zone/hour rows")
    return frame


def data_for_fold(
    frame: pd.DataFrame, fold: MonthlyFold
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a pre-origin training snapshot and the fold's withheld actuals.

    The target frame may contain realized outcomes and weather for evaluation.
    Experiment runners must select forecast-time model inputs explicitly rather
    than passing the complete target frame into a forecasting function.
    """

    training = frame.loc[frame["period_start"] < fold.origin].copy()
    target = frame.loc[
        frame["period_start"].ge(fold.origin) & frame["period_start"].lt(fold.end)
    ].copy()
    if training.empty:
        raise ValueError(f"No training observations before {fold.origin}")
    if target.empty:
        raise ValueError(f"No target observations in [{fold.origin}, {fold.end})")
    if training["period_start"].max() >= fold.origin:
        raise AssertionError("Training data reached or crossed the forecast origin")

    expected = pd.date_range(fold.origin, fold.end, freq="h", inclusive="left")
    for zone_id, group in target.groupby("zone_id", sort=False):
        actual = pd.DatetimeIndex(group["period_start"])
        if not actual.equals(expected):
            raise ValueError(
                f"Target hours for zone {zone_id} do not exactly cover "
                f"[{fold.origin}, {fold.end})"
            )
    return training, target
