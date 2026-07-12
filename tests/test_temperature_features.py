from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import (
    build_feature_matrix,
    build_temperature_climatology_features,
)


def _history(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["period_start", "w1", "w2"])
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["zone_id"] = 1
    return frame.sort_values("period_start", ignore_index=True)


def _target(timestamp: str, w1: float = 9999.0, w2: float = 9999.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [pd.Timestamp(timestamp)],
            "w1": [w1],
            "w2": [w2],
        },
        index=[6],
    )


def test_temperature_climatology_aggregates_spatial_and_temporal_structure() -> None:
    history = _history(
        [
            ("2009-06-15 08:00", 10.0, 20.0),
            ("2010-06-15 08:00", 30.0, 40.0),
            ("2010-06-15 09:00", 900.0, 900.0),  # wrong hour
        ]
    )
    features = build_temperature_climatology_features(
        history,
        _target("2011-06-15 08:00"),
        origin="2011-06-01",
        seasonal_window_days=0,
        station_columns=("w1", "w2"),
    )

    assert features.columns.tolist() == [
        "temperature_clim_0d_mean",
        "temperature_clim_0d_min_station",
        "temperature_clim_0d_max_station",
        "temperature_clim_0d_station_std",
        "temperature_clim_0d_temporal_std",
    ]
    assert features.index.tolist() == [6]
    assert features.loc[6, "temperature_clim_0d_mean"] == pytest.approx(25.0)
    assert features.loc[6, "temperature_clim_0d_min_station"] == pytest.approx(20.0)
    assert features.loc[6, "temperature_clim_0d_max_station"] == pytest.approx(30.0)
    assert features.loc[6, "temperature_clim_0d_station_std"] == pytest.approx(5.0)
    assert features.loc[6, "temperature_clim_0d_temporal_std"] == pytest.approx(10.0)


def test_temperature_climatology_wraps_year_end_but_excludes_current_cycle() -> None:
    features = build_temperature_climatology_features(
        _history(
            [
                ("2009-12-30 08:00", 10.0, 20.0),
                ("2010-12-30 08:00", 900.0, 900.0),
            ]
        ),
        _target("2011-01-02 08:00"),
        origin="2011-01-01",
        seasonal_window_days=3,
        statistics=("mean",),
        station_columns=("w1", "w2"),
    )

    assert features.loc[6, "temperature_clim_3d_mean"] == pytest.approx(15.0)


def test_temperature_climatology_ignores_realized_target_weather() -> None:
    history = _history(
        [
            ("2009-06-15 08:00", 10.0, 20.0),
            ("2010-06-15 08:00", 30.0, 40.0),
        ]
    )
    first = build_temperature_climatology_features(
        history,
        _target("2011-06-15 08:00", w1=0.0, w2=0.0),
        origin="2011-06-01",
        seasonal_window_days=0,
        station_columns=("w1", "w2"),
    )
    second = build_temperature_climatology_features(
        history,
        _target("2011-06-15 08:00", w1=9999.0, w2=-9999.0),
        origin="2011-06-01",
        seasonal_window_days=0,
        station_columns=("w1", "w2"),
    )

    pd.testing.assert_frame_equal(first, second)


def test_feature_builder_accepts_separate_longer_weather_history() -> None:
    load_history = pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [pd.Timestamp("2010-12-31 23:00")],
            "load": [100.0],
        }
    )
    weather_history = _history(
        [
            ("2009-01-02 08:00", 10.0, 20.0),
            ("2010-01-02 08:00", 30.0, 40.0),
        ]
    )
    features = build_feature_matrix(
        load_history,
        _target("2011-01-02 08:00"),
        origin="2011-01-01",
        feature_groups={
            "temperature_climatology": {
                "seasonal_window_days": 0,
                "statistics": ["mean"],
                "station_columns": ["w1", "w2"],
            }
        },
        weather_history=weather_history,
    )

    assert features.loc[6, "temperature_clim_0d_mean"] == pytest.approx(25.0)


def test_temperature_climatology_rejects_missing_or_future_weather() -> None:
    missing = _history([("2010-06-15 08:00", np.nan, 20.0)])
    with pytest.raises(ValueError, match="must not be missing"):
        build_temperature_climatology_features(
            missing,
            _target("2011-06-15 08:00"),
            origin="2011-06-01",
            station_columns=("w1", "w2"),
        )

    future = _history([("2011-06-01 00:00", 10.0, 20.0)])
    with pytest.raises(ValueError, match="strictly before origin"):
        build_temperature_climatology_features(
            future,
            _target("2011-06-15 08:00"),
            origin="2011-06-01",
            station_columns=("w1", "w2"),
        )
