from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gefcom2014.features import (
    TARGET_CATEGORICAL_FEATURES,
    build_feature_matrix,
    categorical_feature_names,
)


FEATURE_CONFIG = Path("configs/features/first_round.yaml")


def test_first_round_config_composes_all_feature_groups() -> None:
    with FEATURE_CONFIG.open("r", encoding="utf-8") as stream:
        groups = yaml.safe_load(stream)["feature_groups"]

    origin = pd.Timestamp("2010-01-01")
    periods = pd.date_range("2008-01-01", origin, freq="h", inclusive="left")
    history_data: dict[str, object] = {
        "zone_id": 1,
        "period_start": periods,
        "load": np.where(periods.year == 2009, 200.0, 100.0),
    }
    history_data.update({f"w{station}": 50.0 + station for station in range(1, 26)})
    history = pd.DataFrame(history_data)
    target = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "period_start": pd.to_datetime(
                ["2010-01-01 00:00", "2010-01-31 23:00"]
            ),
            "load": [9999.0, 9999.0],
            "w1": [-9999.0, -9999.0],
        },
        index=[2, 7],
    )
    features = build_feature_matrix(history, target, origin, groups)

    assert features.shape == (2, 45)
    assert features.index.tolist() == [2, 7]
    assert features.columns.is_unique
    assert features.loc[2, "horizon_hours"] == 0
    assert features.loc[7, "horizon_hours"] == 31 * 24 - 1
    assert features.loc[2, "load_yoy_ratio_365d"] == pytest.approx(2.0)
    assert features.loc[2, "load_last_day_same_hour"] == pytest.approx(200.0)
    assert features.loc[2, "load_how_std_12w"] == pytest.approx(0.0)
    assert features.loc[2, "load_seasonal_daytype_8d_count"] > 0
    assert features.loc[2, "load_seasonal_how_15d_count"] > 0
    assert features.loc[2, "load_seasonal_level_ratio_28d"] == pytest.approx(2.0)
    assert features.loc[
        2, "load_seasonal_daytype_8d_q50_scaled_28d"
    ] == pytest.approx(
        2.0 * features.loc[2, "load_seasonal_daytype_8d_q50"]
    )
    assert features.loc[2, "load_lag_calendar_1y"] == pytest.approx(200.0)
    assert features.loc[2, "load_lag_364d"] == pytest.approx(200.0)
    assert features.loc[2, "temperature_clim_15d_mean"] == pytest.approx(63.0)
    assert features.loc[2, "temperature_clim_15d_min_station"] == pytest.approx(
        51.0
    )
    assert features.loc[2, "temperature_clim_15d_max_station"] == pytest.approx(
        75.0
    )
    assert features.loc[2, "temperature_clim_15d_temporal_std"] == pytest.approx(
        0.0
    )
    assert features.loc[2, "temperature_recent_mean_1d"] == pytest.approx(63.0)
    assert features.loc[2, "temperature_recent_mean_7d"] == pytest.approx(63.0)
    assert features.loc[2, "temperature_recent_mean_28d"] == pytest.approx(63.0)
    assert features.loc[2, "temperature_recent_std_7d"] == pytest.approx(0.0)
    assert features.loc[2, "temperature_recent_anomaly_7d"] == pytest.approx(0.0)
    assert categorical_feature_names(groups) == TARGET_CATEGORICAL_FEATURES


def test_feature_groups_can_be_selected_independently() -> None:
    origin = pd.Timestamp("2010-01-01")
    periods = pd.date_range(
        origin - pd.Timedelta(days=7), origin, freq="h", inclusive="left"
    )
    history = pd.DataFrame(
        {"zone_id": 1, "period_start": periods, "load": 100.0}
    )
    target = pd.DataFrame(
        {"zone_id": [1], "period_start": [origin]}, index=[5]
    )

    target_only = build_feature_matrix(history, target, origin, {"target_time": {}})
    recent_only = build_feature_matrix(
        history,
        target,
        origin,
        {
            "recent_load": {
                "mean_windows_days": [7],
                "std_windows_days": [],
                "difference_windows_days": [],
                "trend_windows_days": [],
                "yoy_windows_days": [],
            }
        },
    )

    assert target_only.shape == (1, 11)
    assert recent_only.columns.tolist() == ["load_mean_7d"]
    assert categorical_feature_names({"recent_load": {}}) == ()


def test_feature_builder_rejects_unknown_group() -> None:
    with pytest.raises(ValueError, match="Unknown feature groups"):
        build_feature_matrix(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.Timestamp("2010-01-01"),
            {"future_load": {}},
        )
