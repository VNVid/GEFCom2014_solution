from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import build_recent_temperature_features


def _weather_history(*, complete_28_days: bool = True) -> pd.DataFrame:
    historical_periods = pd.date_range(
        "2010-01-25", "2010-02-01", freq="h", inclusive="left"
    )
    recent_periods = pd.date_range(
        "2011-01-04", "2011-02-01", freq="h", inclusive="left"
    )
    if not complete_28_days:
        recent_periods = recent_periods[1:]

    recent_position = np.arange(28 * 24)
    recent_hourly_mean = np.where(
        recent_position < 21 * 24,
        10.0,
        np.where(recent_position % 2 == 0, 18.0, 22.0),
    )
    if not complete_28_days:
        recent_hourly_mean = recent_hourly_mean[1:]

    frame = pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": historical_periods.append(recent_periods),
            "w1": np.concatenate(
                [
                    np.full(len(historical_periods), 14.0),
                    recent_hourly_mean - 1.0,
                ]
            ),
            "w2": np.concatenate(
                [
                    np.full(len(historical_periods), 16.0),
                    recent_hourly_mean + 1.0,
                ]
            ),
        }
    )
    return frame.sort_values("period_start", ignore_index=True)


def _target(w1: float = 9999.0, w2: float = 9999.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [pd.Timestamp("2011-02-01")],
            "w1": [w1],
            "w2": [w2],
        },
        index=[5],
    )


def _features(
    history: pd.DataFrame, target: pd.DataFrame | None = None
) -> pd.DataFrame:
    return build_recent_temperature_features(
        history,
        _target() if target is None else target,
        origin="2011-02-01",
        anomaly_seasonal_window_days=0,
        station_columns=("w1", "w2"),
    )


def test_recent_temperature_features_capture_level_volatility_and_anomaly() -> None:
    features = _features(_weather_history())

    assert features.columns.tolist() == [
        "temperature_recent_mean_1d",
        "temperature_recent_mean_7d",
        "temperature_recent_mean_28d",
        "temperature_recent_std_7d",
        "temperature_recent_anomaly_7d",
    ]
    assert features.index.tolist() == [5]
    assert features.loc[5, "temperature_recent_mean_1d"] == pytest.approx(20.0)
    assert features.loc[5, "temperature_recent_mean_7d"] == pytest.approx(20.0)
    assert features.loc[5, "temperature_recent_mean_28d"] == pytest.approx(12.5)
    assert features.loc[5, "temperature_recent_std_7d"] == pytest.approx(2.0)
    assert features.loc[5, "temperature_recent_anomaly_7d"] == pytest.approx(5.0)


def test_recent_temperature_windows_fail_independently_when_incomplete() -> None:
    features = _features(_weather_history(complete_28_days=False))

    assert np.isnan(features.loc[5, "temperature_recent_mean_28d"])
    assert features.loc[5, "temperature_recent_mean_1d"] == pytest.approx(20.0)
    assert features.loc[5, "temperature_recent_mean_7d"] == pytest.approx(20.0)
    assert features.loc[5, "temperature_recent_std_7d"] == pytest.approx(2.0)
    assert features.loc[5, "temperature_recent_anomaly_7d"] == pytest.approx(5.0)


def test_recent_temperature_features_ignore_realized_target_weather() -> None:
    history = _weather_history()
    first = _features(history, _target(w1=0.0, w2=0.0))
    second = _features(history, _target(w1=9999.0, w2=-9999.0))

    pd.testing.assert_frame_equal(first, second)


def test_recent_temperature_features_reject_weather_at_origin() -> None:
    future = pd.concat(
        [
            _weather_history(),
            pd.DataFrame(
                {
                    "zone_id": [1],
                    "period_start": [pd.Timestamp("2011-02-01")],
                    "w1": [10.0],
                    "w2": [20.0],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="strictly before origin"):
        _features(future)
