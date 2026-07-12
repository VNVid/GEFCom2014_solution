from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gefcom2014.features import build_horizon_decay_features


def test_horizon_decay_uses_configured_exponential_scales() -> None:
    source = pd.DataFrame(
        {
            "temperature_recent_anomaly_7d": [8.0, 8.0],
            "load_last_day_same_hour": [100.0, 100.0],
        },
        index=[4, 9],
    )
    horizon = pd.Series([0, 48], index=[4, 9])

    features = build_horizon_decay_features(
        source,
        horizon,
        source_scales_days={
            "temperature_recent_anomaly_7d": 14,
            "load_last_day_same_hour": 2,
        },
    )

    assert features.columns.tolist() == [
        "temperature_recent_anomaly_7d_decay_14d",
        "load_last_day_same_hour_decay_2d",
    ]
    assert features.loc[4, "temperature_recent_anomaly_7d_decay_14d"] == 8.0
    assert features.loc[9, "temperature_recent_anomaly_7d_decay_14d"] == pytest.approx(
        8.0 * np.exp(-2.0 / 14.0)
    )
    assert features.loc[9, "load_last_day_same_hour_decay_2d"] == pytest.approx(
        100.0 * np.exp(-1.0)
    )


def test_horizon_decay_requires_an_available_numeric_source() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        build_horizon_decay_features(
            pd.DataFrame({"other": [1.0]}),
            pd.Series([0]),
            source_scales_days={"missing": 7},
        )
