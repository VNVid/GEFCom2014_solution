from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import build_recent_load_features


def _history(periods: pd.DatetimeIndex, load: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": periods,
            "load": load,
        }
    )


def _target(periods: list[str], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": pd.to_datetime(periods),
            "load": 9999.0,
        },
        index=index,
    )


def test_recent_load_features_compute_configured_levels_and_yoy_ratios() -> None:
    origin = pd.Timestamp("2010-01-01")
    periods = pd.date_range("2008-01-01", origin, freq="h", inclusive="left")
    loads = np.where(periods.year == 2009, 200.0, 100.0)
    target = _target(["2010-01-01 00:00", "2010-01-15 12:00"], index=[4, 9])

    features = build_recent_load_features(
        _history(periods, loads), target, origin
    )

    assert features.index.tolist() == [4, 9]
    assert list(features.columns) == [
        "load_mean_7d",
        "load_mean_28d",
        "load_mean_365d",
        "load_std_28d",
        "load_mean_7d_minus_28d",
        "load_daily_slope_28d",
        "load_yoy_ratio_28d",
        "load_yoy_ratio_365d",
    ]
    np.testing.assert_allclose(features["load_mean_7d"], 200.0)
    np.testing.assert_allclose(features["load_mean_28d"], 200.0)
    np.testing.assert_allclose(features["load_mean_365d"], 200.0)
    np.testing.assert_allclose(features["load_std_28d"], 0.0)
    np.testing.assert_allclose(features["load_mean_7d_minus_28d"], 0.0)
    np.testing.assert_allclose(features["load_daily_slope_28d"], 0.0, atol=1e-12)
    np.testing.assert_allclose(features["load_yoy_ratio_28d"], 2.0)
    np.testing.assert_allclose(features["load_yoy_ratio_365d"], 2.0)


def test_recent_load_trend_uses_daily_means() -> None:
    origin = pd.Timestamp("2010-03-01")
    periods = pd.date_range(
        origin - pd.Timedelta(days=28), origin, freq="h", inclusive="left"
    )
    daily_level = 10.0 + np.repeat(np.arange(28, dtype=float), 24)
    features = build_recent_load_features(
        _history(periods, daily_level),
        _target(["2010-03-01 00:00"]),
        origin,
        mean_windows_days=(7, 28),
        std_windows_days=(),
        difference_windows_days=((7, 28),),
        trend_windows_days=(28,),
        yoy_windows_days=(),
    )

    assert features.loc[0, "load_mean_7d"] == pytest.approx(34.0)
    assert features.loc[0, "load_mean_28d"] == pytest.approx(23.5)
    assert features.loc[0, "load_mean_7d_minus_28d"] == pytest.approx(10.5)
    assert features.loc[0, "load_daily_slope_28d"] == pytest.approx(1.0)


def test_medium_term_window_can_bridge_recent_and_annual_levels() -> None:
    origin = pd.Timestamp("2010-04-01")
    periods = pd.date_range(
        origin - pd.Timedelta(days=90), origin, freq="h", inclusive="left"
    )
    loads = np.where(periods >= origin - pd.Timedelta(days=28), 200.0, 100.0)
    features = build_recent_load_features(
        _history(periods, loads),
        _target(["2010-04-01 00:00"]),
        origin,
        mean_windows_days=(28, 90),
        std_windows_days=(),
        difference_windows_days=((28, 90),),
        trend_windows_days=(),
        yoy_windows_days=(),
    )

    expected_90d = (28 * 200.0 + 62 * 100.0) / 90
    assert features.loc[0, "load_mean_90d"] == pytest.approx(expected_90d)
    assert features.loc[0, "load_mean_28d_minus_90d"] == pytest.approx(
        200.0 - expected_90d
    )


def test_incomplete_recent_window_produces_nan_instead_of_partial_mean() -> None:
    origin = pd.Timestamp("2010-03-01")
    periods = pd.date_range(
        origin - pd.Timedelta(days=6), origin, freq="h", inclusive="left"
    )
    features = build_recent_load_features(
        _history(periods, np.full(len(periods), 100.0)),
        _target(["2010-03-01 00:00"]),
        origin,
        mean_windows_days=(7,),
        std_windows_days=(),
        difference_windows_days=(),
        trend_windows_days=(),
        yoy_windows_days=(),
    )
    assert np.isnan(features.loc[0, "load_mean_7d"])


def test_recent_load_features_reject_history_at_or_after_origin() -> None:
    origin = pd.Timestamp("2010-03-01")
    history = _history(pd.DatetimeIndex([origin]), np.array([100.0]))
    with pytest.raises(ValueError, match="strictly before origin"):
        build_recent_load_features(
            history,
            _target(["2010-03-01 00:00"]),
            origin,
            mean_windows_days=(7,),
            std_windows_days=(),
            difference_windows_days=(),
            trend_windows_days=(),
            yoy_windows_days=(),
        )
