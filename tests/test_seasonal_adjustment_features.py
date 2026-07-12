from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import build_seasonal_load_adjustment_features


def _history_with_recent_level(
    *, recent_level: float = 200.0, complete_recent_day: bool = True
) -> pd.DataFrame:
    seasonal_periods = pd.date_range(
        "2010-01-20", "2010-02-10", freq="h", inclusive="left"
    )
    recent_periods = pd.date_range(
        "2011-01-31", "2011-02-01", freq="h", inclusive="left"
    )
    if not complete_recent_day:
        recent_periods = recent_periods[:-1]
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": seasonal_periods.append(recent_periods),
            "load": np.concatenate(
                [
                    np.full(len(seasonal_periods), 100.0),
                    np.full(len(recent_periods), recent_level),
                ]
            ),
        }
    )


def _target(load: float = 9999.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [pd.Timestamp("2011-02-01 05:00")],
            "load": [load],
        },
        index=[4],
    )


def test_recent_level_ratio_scales_the_historical_seasonal_anchor() -> None:
    features = build_seasonal_load_adjustment_features(
        _history_with_recent_level(),
        _target(),
        origin="2011-02-01",
        recent_window_days=1,
        seasonal_window_days=8,
    )

    assert features.columns.tolist() == [
        "load_seasonal_level_ratio_1d",
        "load_seasonal_daytype_8d_q50_scaled_1d",
    ]
    assert features.index.tolist() == [4]
    assert features.loc[4, "load_seasonal_level_ratio_1d"] == pytest.approx(2.0)
    assert features.loc[
        4, "load_seasonal_daytype_8d_q50_scaled_1d"
    ] == pytest.approx(200.0)


def test_additive_recent_level_shift_is_available_alongside_ratio() -> None:
    features = build_seasonal_load_adjustment_features(
        _history_with_recent_level(),
        _target(),
        origin="2011-02-01",
        recent_window_days=1,
        seasonal_window_days=8,
        adjustment_types=("multiplicative", "additive"),
    )

    assert features.columns.tolist() == [
        "load_seasonal_level_ratio_1d",
        "load_seasonal_daytype_8d_q50_scaled_1d",
        "load_seasonal_level_delta_1d",
        "load_seasonal_daytype_8d_q50_shifted_1d",
    ]
    assert features.loc[4, "load_seasonal_level_delta_1d"] == pytest.approx(100.0)
    assert features.loc[
        4, "load_seasonal_daytype_8d_q50_shifted_1d"
    ] == pytest.approx(200.0)


def test_incomplete_recent_window_produces_explicit_missing_features() -> None:
    features = build_seasonal_load_adjustment_features(
        _history_with_recent_level(complete_recent_day=False),
        _target(),
        origin="2011-02-01",
        recent_window_days=1,
    )

    assert features.isna().all().all()


def test_seasonal_adjustment_ignores_realized_target_load() -> None:
    history = _history_with_recent_level()
    first = build_seasonal_load_adjustment_features(
        history,
        _target(load=100.0),
        origin="2011-02-01",
        recent_window_days=1,
    )
    second = build_seasonal_load_adjustment_features(
        history,
        _target(load=-9999.0),
        origin="2011-02-01",
        recent_window_days=1,
    )

    pd.testing.assert_frame_equal(first, second)


def test_seasonal_adjustment_rejects_history_at_origin() -> None:
    history = pd.concat(
        [
            _history_with_recent_level(),
            pd.DataFrame(
                {
                    "zone_id": [1],
                    "period_start": [pd.Timestamp("2011-02-01")],
                    "load": [500.0],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="strictly before origin"):
        build_seasonal_load_adjustment_features(
            history,
            _target(),
            origin="2011-02-01",
            recent_window_days=1,
        )
