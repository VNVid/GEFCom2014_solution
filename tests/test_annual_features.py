from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.baselines import seasonal_naive_forecast
from gefcom2014.features import build_annual_load_anchor_features


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
        },
        index=[7],
    )


def test_annual_anchors_distinguish_calendar_and_weekday_alignment() -> None:
    target = _target("2010-03-01 05:00")  # Monday
    history = _history(
        [
            ("2009-03-01 05:00", 100.0),  # same date, Sunday
            ("2009-03-02 05:00", 200.0),  # 364 days earlier, Monday
        ]
    )

    features = build_annual_load_anchor_features(
        history, target, origin="2010-03-01"
    )

    assert features.columns.tolist() == [
        "load_lag_calendar_1y",
        "load_lag_364d",
    ]
    assert features.index.tolist() == [7]
    assert features.loc[7, "load_lag_calendar_1y"] == pytest.approx(100.0)
    assert features.loc[7, "load_lag_364d"] == pytest.approx(200.0)


def test_calendar_year_anchor_exactly_matches_naive_baseline() -> None:
    history = _history(
        [
            ("2009-03-01 05:00", 100.0),
            ("2009-03-02 05:00", 200.0),
        ]
    )
    target = _target("2010-03-01 05:00")
    features = build_annual_load_anchor_features(
        history, target, origin="2010-03-01"
    )
    baseline, _ = seasonal_naive_forecast(history, target, np.array([0.50]))

    assert features.loc[7, "load_lag_calendar_1y"] == pytest.approx(baseline[0, 0])


def test_calendar_year_anchor_uses_explicit_february_29_mapping() -> None:
    target = _target("2008-02-29 08:00")
    fixed_day_reference = target.loc[7, "period_start"] - pd.Timedelta(days=364)
    history = _history(
        [
            ("2007-02-28 08:00", 100.0),
            (fixed_day_reference.strftime("%Y-%m-%d %H:%M"), 200.0),
        ]
    )

    features = build_annual_load_anchor_features(
        history, target, origin="2008-02-01"
    )

    assert features.loc[7, "load_lag_calendar_1y"] == pytest.approx(100.0)
    assert features.loc[7, "load_lag_364d"] == pytest.approx(200.0)


def test_missing_annual_reference_is_preserved_as_nan() -> None:
    features = build_annual_load_anchor_features(
        _history([("2009-01-01 00:00", 100.0)]),
        _target("2010-03-01 05:00"),
        origin="2010-03-01",
    )

    assert features.isna().all().all()


def test_annual_anchors_reject_history_at_or_after_origin() -> None:
    with pytest.raises(ValueError, match="strictly before origin"):
        build_annual_load_anchor_features(
            _history([("2010-03-01 00:00", 100.0)]),
            _target("2010-03-01 05:00"),
            origin="2010-03-01",
        )
