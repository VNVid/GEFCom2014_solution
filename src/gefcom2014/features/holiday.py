"""Deterministic US federal-holiday features for target timestamps."""

from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar

from .target import build_target_time_features


HOLIDAY_FEATURES = (
    "holiday_name",
    "is_holiday",
    "is_working_day",
    "is_day_before_holiday",
    "is_day_after_holiday",
    "is_year_end_holiday_period",
)
HOLIDAY_CATEGORICAL_FEATURES = ("holiday_name",)


def build_holiday_features(
    target: pd.DataFrame, origin: str | pd.Timestamp
) -> pd.DataFrame:
    """Describe target-day US federal-holiday context known in advance.

    The competition load represents a US utility and explicitly permits the
    federal holiday calendar. Pandas' calendar applies the official observed
    weekday when a fixed-date holiday falls on a weekend. The broader year-end
    indicator also covers the actual Christmas/New Year transition period.
    """

    target_time = build_target_time_features(target, origin)
    dates = target["period_start"].dt.normalize()
    calendar = USFederalHolidayCalendar()
    holiday_names = calendar.holidays(
        start=dates.min() - pd.Timedelta(days=1),
        end=dates.max() + pd.Timedelta(days=1),
        return_name=True,
    )
    holiday_by_date = holiday_names.to_dict()
    holiday_dates = set(holiday_names.index)

    is_holiday = dates.isin(holiday_dates).to_numpy(dtype=bool)
    is_weekday = target_time["day_of_week"].to_numpy(dtype=np.int8) < 5
    month = dates.dt.month.to_numpy(dtype=np.int8)
    day = dates.dt.day.to_numpy(dtype=np.int8)

    features = pd.DataFrame(index=target.index)
    features["holiday_name"] = dates.map(holiday_by_date).fillna("none").to_numpy()
    features["is_holiday"] = is_holiday.astype(np.int8)
    features["is_working_day"] = (is_weekday & ~is_holiday).astype(np.int8)
    features["is_day_before_holiday"] = (
        dates.add(pd.Timedelta(days=1)).isin(holiday_dates).to_numpy(dtype=np.int8)
    )
    features["is_day_after_holiday"] = (
        dates.sub(pd.Timedelta(days=1)).isin(holiday_dates).to_numpy(dtype=np.int8)
    )
    features["is_year_end_holiday_period"] = (
        ((month == 12) & (day >= 24)) | ((month == 1) & (day <= 2))
    ).astype(np.int8)
    return features.loc[:, list(HOLIDAY_FEATURES)]
