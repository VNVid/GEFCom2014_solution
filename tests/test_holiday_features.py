from __future__ import annotations

import pandas as pd

from gefcom2014.features import build_holiday_features


def _target(periods: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": pd.to_datetime(periods),
            "load": -9999.0,
            "w1": 9999.0,
        }
    )


def test_holiday_features_use_observed_federal_dates_and_adjacent_days() -> None:
    features = build_holiday_features(
        _target(
            [
                "2010-07-04 12:00",  # Sunday before the observed holiday
                "2010-07-05 12:00",  # observed Independence Day
                "2010-07-06 12:00",  # day after the observed holiday
            ]
        ),
        origin="2010-07-01",
    )

    assert features["holiday_name"].tolist() == [
        "none",
        "Independence Day",
        "none",
    ]
    assert features["is_holiday"].tolist() == [0, 1, 0]
    assert features["is_day_before_holiday"].tolist() == [1, 0, 0]
    assert features["is_day_after_holiday"].tolist() == [0, 0, 1]
    assert features["is_working_day"].tolist() == [0, 0, 1]


def test_year_end_period_is_deterministic_and_ignores_target_outcomes() -> None:
    first_target = _target(["2010-12-23 12:00", "2010-12-24 12:00"])
    second_target = first_target.copy()
    second_target["load"] = 123456.0
    second_target["w1"] = -123456.0

    first = build_holiday_features(first_target, origin="2010-12-01")
    second = build_holiday_features(second_target, origin="2010-12-01")

    pd.testing.assert_frame_equal(first, second)
    assert first["is_year_end_holiday_period"].tolist() == [0, 1]
