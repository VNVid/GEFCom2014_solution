import numpy as np
import pandas as pd

from gefcom2014.backtesting import prepare_backtest_frame
from gefcom2014.baselines import seasonal_empirical_forecast, seasonal_naive_forecast
from gefcom2014.data import (
    discover_round_files,
    interval_start,
    load_training_history,
    read_benchmark_file,
)


LOAD_DIR = "data/GEFCom2014-L_V2/Load"


def _training(rows: list[tuple[str, float]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["period_start", "load"])
    frame["period_start"] = pd.to_datetime(frame["period_start"])
    frame["zone_id"] = 1
    return frame


def _target(timestamp: str) -> pd.DataFrame:
    period_start = pd.Timestamp(timestamp)
    return pd.DataFrame(
        {
            "zone_id": [1],
            "period_start": [period_start],
        }
    )


def test_seasonal_naive_repeats_previous_calendar_year_at_all_quantiles() -> None:
    training = _training([("2010-06-15 08:00", 123.4)])
    predictions, sample_sizes = seasonal_naive_forecast(
        training, _target("2011-06-15 08:00"), np.array([0.1, 0.5, 0.9])
    )
    np.testing.assert_allclose(predictions, [[123.4, 123.4, 123.4]])
    np.testing.assert_array_equal(sample_sizes, [1])


def test_seasonal_naive_reproduces_official_task_1_benchmark() -> None:
    rounds = discover_round_files(LOAD_DIR)
    training = load_training_history(LOAD_DIR, through_task=1).dropna(subset=["load"])
    training = prepare_backtest_frame(training)
    benchmark = read_benchmark_file(rounds[0].benchmark_path, task_id=1)
    target = benchmark.loc[:, ["zone_id", "timestamp"]].copy()
    target["period_start"] = interval_start(target["timestamp"])

    predictions, _ = seasonal_naive_forecast(
        training, target, np.array([0.01, 0.5, 0.99])
    )
    expected = benchmark["q0.01"].to_numpy(dtype=float)
    np.testing.assert_allclose(predictions, np.repeat(expected[:, None], 3, axis=1))


def test_empirical_baseline_excludes_current_season_and_filters_day_type_and_hour() -> None:
    training = _training(
        [
            ("2010-02-01 08:00", 10.0),  # prior-cycle weekday
            ("2010-02-02 08:00", 20.0),  # prior-cycle weekday
            ("2010-02-06 08:00", 500.0),  # weekend: excluded
            ("2010-02-03 09:00", 600.0),  # wrong hour: excluded
            ("2011-01-31 08:00", 999.0),  # recent current cycle: excluded
        ]
    )
    predictions, sample_sizes = seasonal_empirical_forecast(
        training,
        _target("2011-02-01 08:00"),
        np.array([0.25, 0.5, 0.75]),
        window_days=8,
    )
    np.testing.assert_allclose(predictions, [[12.5, 15.0, 17.5]])
    np.testing.assert_array_equal(sample_sizes, [2])


def test_empirical_baseline_uses_year_end_part_of_previous_seasonal_cycle() -> None:
    training = _training([("2011-01-03 08:00", 42.0)])
    predictions, sample_sizes = seasonal_empirical_forecast(
        training,
        _target("2011-12-30 08:00"),
        np.array([0.5]),
        window_days=8,
    )
    np.testing.assert_allclose(predictions, [[42.0]])
    np.testing.assert_array_equal(sample_sizes, [1])


def test_empirical_baseline_does_not_treat_recent_december_as_previous_january_cycle() -> None:
    training = _training(
        [("2010-01-02 08:00", 20.0), ("2010-12-26 08:00", 999.0)]
    )
    predictions, sample_sizes = seasonal_empirical_forecast(
        training,
        _target("2011-01-02 08:00"),
        np.array([0.5]),
        window_days=8,
    )
    np.testing.assert_allclose(predictions, [[20.0]])
    np.testing.assert_array_equal(sample_sizes, [1])
