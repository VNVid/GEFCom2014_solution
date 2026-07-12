from __future__ import annotations

import numpy as np
import pandas as pd

from gefcom2014.features import build_recent_load_profile_features


def _target(periods: list[str], index: list[int] | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "zone_id": 1,
            "period_start": pd.to_datetime(periods),
            "load": 9999.0,
        },
        index=index,
    )


def test_recent_profiles_match_hour_and_hour_of_week() -> None:
    origin = pd.Timestamp("2010-03-01")  # Monday 00:00
    start = origin - pd.Timedelta(weeks=12)
    periods = pd.date_range(start, origin, freq="h", inclusive="left")
    week_index = ((periods - start).days // 7).to_numpy()
    hour_of_week = 24 * periods.dayofweek + periods.hour
    loads = 10.0 * week_index + hour_of_week
    history = pd.DataFrame(
        {"zone_id": 1, "period_start": periods, "load": loads}
    )
    target = _target(
        ["2010-03-01 05:00", "2010-03-07 05:00"], index=[3, 8]
    )

    features = build_recent_load_profile_features(history, target, origin)

    assert features.index.tolist() == [3, 8]
    assert features.columns.tolist() == [
        "load_last_day_same_hour",
        "load_last_same_hour_of_week",
        "load_how_mean_4w",
        "load_how_mean_12w",
        "load_how_std_12w",
    ]
    # The final complete day is Sunday, so its hour-05 value is used for both
    # targets regardless of their target weekday.
    np.testing.assert_allclose(features["load_last_day_same_hour"], [259.0, 259.0])
    np.testing.assert_allclose(
        features["load_last_same_hour_of_week"], [115.0, 259.0]
    )
    np.testing.assert_allclose(features["load_how_mean_4w"], [100.0, 244.0])
    np.testing.assert_allclose(features["load_how_mean_12w"], [60.0, 204.0])
    expected_std = np.std(10.0 * np.arange(12), ddof=0)
    np.testing.assert_allclose(features["load_how_std_12w"], expected_std)


def test_incomplete_profile_window_produces_nan() -> None:
    origin = pd.Timestamp("2010-03-01")
    periods = pd.date_range(
        origin - pd.Timedelta(weeks=4), origin, freq="h", inclusive="left"
    ).delete(100)
    history = pd.DataFrame(
        {"zone_id": 1, "period_start": periods, "load": 100.0}
    )
    features = build_recent_load_profile_features(
        history,
        _target(["2010-03-01 05:00"]),
        origin,
        mean_windows_weeks=(4,),
        std_windows_weeks=(),
        include_last_day=False,
        include_last_hour_of_week=False,
    )

    assert np.isnan(features.loc[0, "load_how_mean_4w"])


def test_recent_profiles_ignore_target_load_values() -> None:
    origin = pd.Timestamp("2010-03-01")
    periods = pd.date_range(
        origin - pd.Timedelta(weeks=1), origin, freq="h", inclusive="left"
    )
    history = pd.DataFrame(
        {"zone_id": 1, "period_start": periods, "load": np.arange(len(periods))}
    )
    first = _target(["2010-03-01 05:00"])
    second = first.copy()
    second["load"] = -9999.0

    pd.testing.assert_frame_equal(
        build_recent_load_profile_features(
            history,
            first,
            origin,
            mean_windows_weeks=(),
            std_windows_weeks=(),
        ),
        build_recent_load_profile_features(
            history,
            second,
            origin,
            mean_windows_weeks=(),
            std_windows_weeks=(),
        ),
    )
