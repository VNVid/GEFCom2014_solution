from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import (
    TARGET_CATEGORICAL_FEATURES,
    TARGET_TIME_FEATURES,
    build_target_time_features,
)


def _target(periods: list[str], index: list[int] | None = None) -> pd.DataFrame:
    timestamps = pd.to_datetime(periods)
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": timestamps,
            # These evaluation-only columns must not affect deterministic
            # target-time features.
            "load": np.arange(len(timestamps), dtype=float) + 100.0,
            "w1": np.arange(len(timestamps), dtype=float) + 20.0,
        },
        index=index,
    )


def test_target_time_features_have_expected_calendar_and_horizon_values() -> None:
    target = _target(
        [
            "2011-01-01 00:00",
            "2011-01-01 06:00",
            "2011-01-03 05:00",
            "2011-01-08 00:00",
        ],
        index=[10, 20, 30, 40],
    )
    features = build_target_time_features(target, origin="2011-01-01")

    assert list(features.columns) == list(TARGET_TIME_FEATURES)
    assert features.index.tolist() == [10, 20, 30, 40]
    np.testing.assert_array_equal(features["hour"], [0, 6, 5, 0])
    np.testing.assert_array_equal(features["day_of_week"], [5, 5, 0, 5])
    np.testing.assert_array_equal(features["hour_of_week"], [120, 126, 5, 120])
    np.testing.assert_array_equal(features["month"], [1, 1, 1, 1])
    np.testing.assert_array_equal(features["is_weekend"], [1, 1, 0, 1])
    np.testing.assert_array_equal(features["horizon_hours"], [0, 6, 53, 168])
    np.testing.assert_array_equal(features["forecast_week"], [0, 0, 0, 1])
    assert features.loc[10, "hour_sin"] == pytest.approx(0.0)
    assert features.loc[10, "hour_cos"] == pytest.approx(1.0)
    assert features.loc[20, "hour_sin"] == pytest.approx(1.0)
    assert features.loc[20, "hour_cos"] == pytest.approx(0.0, abs=1e-12)
    assert set(TARGET_CATEGORICAL_FEATURES).issubset(features.columns)


def test_seasonal_encoding_aligns_dates_across_leap_and_non_leap_years() -> None:
    leap = build_target_time_features(
        _target(["2008-03-01 00:00"]), origin="2008-03-01"
    )
    non_leap = build_target_time_features(
        _target(["2009-03-01 00:00"]), origin="2009-03-01"
    )
    np.testing.assert_allclose(
        leap[["seasonal_day_sin", "seasonal_day_cos"]],
        non_leap[["seasonal_day_sin", "seasonal_day_cos"]],
    )


def test_seasonal_encoding_is_continuous_across_year_end() -> None:
    december = build_target_time_features(
        _target(["2010-12-31 00:00"]), origin="2010-12-01"
    ).loc[0, ["seasonal_day_sin", "seasonal_day_cos"]]
    january = build_target_time_features(
        _target(["2011-01-01 00:00"]), origin="2011-01-01"
    ).loc[0, ["seasonal_day_sin", "seasonal_day_cos"]]

    assert np.linalg.norm(december.to_numpy() - january.to_numpy()) < 0.02


def test_target_time_features_ignore_realized_load_and_weather() -> None:
    first = _target(["2010-06-15 12:00"])
    second = first.copy()
    second["load"] = 9999.0
    second["w1"] = -9999.0

    pd.testing.assert_frame_equal(
        build_target_time_features(first, origin="2010-06-01"),
        build_target_time_features(second, origin="2010-06-01"),
    )


@pytest.mark.parametrize(
    ("period", "origin", "message"),
    [
        ("2010-05-31 23:00", "2010-06-01", "inside the origin month"),
        ("2010-07-01 00:00", "2010-06-01", "inside the origin month"),
        ("2010-06-01 00:30", "2010-06-01", "hourly boundaries"),
    ],
)
def test_target_time_features_reject_invalid_target_periods(
    period: str, origin: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_target_time_features(_target([period]), origin=origin)


def test_target_time_features_require_month_start_origin() -> None:
    with pytest.raises(ValueError, match="first day"):
        build_target_time_features(
            _target(["2010-06-02 00:00"]), origin="2010-06-02"
        )
