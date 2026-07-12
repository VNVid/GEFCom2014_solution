from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.baselines import seasonal_empirical_forecast
from gefcom2014.features import build_seasonal_load_features


def _history(rows: list[tuple[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["period_start", "load"])
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["zone_id"] = 1
    return frame.sort_values("period_start", ignore_index=True)


def _target(timestamp: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [pd.Timestamp(timestamp)],
            "load": [9999.0],
        }
    )


def test_seasonal_features_match_baseline_pool_and_exact_weekday_profile() -> None:
    history = _history(
        [
            ("2010-02-01 08:00", 5.0),
            ("2010-02-02 08:00", 10.0),
            ("2010-02-06 08:00", 100.0),  # weekend: excluded from day type
            ("2010-02-09 08:00", 20.0),
            ("2010-02-16 08:00", 30.0),  # outside ±8, inside exact-DOW ±15
            ("2010-02-02 09:00", 500.0),  # wrong operating hour
            ("2011-01-25 08:00", 999.0),  # current cycle: always excluded
        ]
    )
    target = _target("2011-02-01 08:00")  # Tuesday
    features = build_seasonal_load_features(
        history, target, origin="2011-02-01"
    )

    assert features.columns.tolist() == [
        "load_seasonal_daytype_8d_q10",
        "load_seasonal_daytype_8d_q50",
        "load_seasonal_daytype_8d_q90",
        "load_seasonal_daytype_8d_count",
        "load_seasonal_how_15d_mean",
        "load_seasonal_how_15d_std",
        "load_seasonal_how_15d_count",
    ]
    assert features.loc[0, "load_seasonal_daytype_8d_q10"] == pytest.approx(6.0)
    assert features.loc[0, "load_seasonal_daytype_8d_q50"] == pytest.approx(10.0)
    assert features.loc[0, "load_seasonal_daytype_8d_q90"] == pytest.approx(18.0)
    assert features.loc[0, "load_seasonal_daytype_8d_count"] == 3
    assert features.loc[0, "load_seasonal_how_15d_mean"] == pytest.approx(20.0)
    assert features.loc[0, "load_seasonal_how_15d_std"] == pytest.approx(
        np.std([10.0, 20.0, 30.0], ddof=0)
    )
    assert features.loc[0, "load_seasonal_how_15d_count"] == 3

    baseline, sample_sizes = seasonal_empirical_forecast(
        history, target, np.array([0.10, 0.50, 0.90]), window_days=8
    )
    np.testing.assert_allclose(
        features.loc[
            0,
            [
                "load_seasonal_daytype_8d_q10",
                "load_seasonal_daytype_8d_q50",
                "load_seasonal_daytype_8d_q90",
            ],
        ].to_numpy(dtype=float),
        baseline[0],
    )
    assert features.loc[0, "load_seasonal_daytype_8d_count"] == sample_sizes[0]


def test_seasonal_hour_of_week_profile_wraps_across_year_end() -> None:
    features = build_seasonal_load_features(
        _history([("2011-01-07 08:00", 42.0)]),
        _target("2011-12-30 08:00"),
        origin="2011-12-01",
    )

    assert features.loc[0, "load_seasonal_how_15d_mean"] == pytest.approx(42.0)
    assert features.loc[0, "load_seasonal_how_15d_count"] == 1


def test_empty_seasonal_pool_is_explicit_nan_with_zero_count() -> None:
    features = build_seasonal_load_features(
        _history([("2010-06-01 09:00", 100.0)]),
        _target("2011-06-01 08:00"),
        origin="2011-06-01",
    )

    assert np.isnan(features.loc[0, "load_seasonal_daytype_8d_q50"])
    assert features.loc[0, "load_seasonal_daytype_8d_count"] == 0
    assert np.isnan(features.loc[0, "load_seasonal_how_15d_mean"])
    assert features.loc[0, "load_seasonal_how_15d_count"] == 0
