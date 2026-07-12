from pathlib import Path

import pandas as pd
import pytest

from gefcom2014.data import (
    discover_round_files,
    interval_start,
    load_training_history,
    parse_competition_timestamps,
    read_benchmark_file,
)


LOAD_DIR = Path("data/GEFCom2014-L_V2/Load")


def test_compact_timestamp_parsing_handles_ambiguous_lengths_and_midnight() -> None:
    raw = pd.Series(["112001 1:00", "1112001 0:00", "1012010 1:00", "112011 1:00"])
    actual = parse_competition_timestamps(raw)
    expected = pd.Series(
        pd.to_datetime(
            [
                "2001-01-01 01:00",
                "2001-11-01 00:00",
                "2010-10-01 01:00",
                "2011-01-01 01:00",
            ]
        ),
        name="timestamp",
    )
    pd.testing.assert_series_equal(actual, expected)


def test_compact_timestamp_parser_uses_hourly_continuity_for_ambiguous_days() -> None:
    raw = pd.Series(["1102001 23:00", "1112001 0:00", "1112001 1:00", "1112001 2:00"])
    actual = parse_competition_timestamps(raw)
    expected = pd.Series(
        pd.date_range("2001-01-10 23:00", periods=4, freq="h"), name="timestamp"
    )
    pd.testing.assert_series_equal(actual, expected)


def test_hour_ending_timestamp_maps_to_prior_operating_interval() -> None:
    hour_ending = pd.Series(pd.to_datetime(["2011-12-31 01:00", "2012-01-01 00:00"]))
    expected = pd.Series(pd.to_datetime(["2011-12-31 00:00", "2011-12-31 23:00"]))
    pd.testing.assert_series_equal(interval_start(hour_ending), expected)


@pytest.mark.parametrize("task_id", range(1, 16))
def test_training_history_stops_before_each_forecast_origin(task_id: int) -> None:
    rounds = discover_round_files(LOAD_DIR)
    history = load_training_history(LOAD_DIR, through_task=task_id)
    target = read_benchmark_file(rounds[task_id - 1].benchmark_path, task_id)

    assert history["source_task"].max() == task_id
    assert history["timestamp"].max() < target["timestamp"].min()
    assert history["timestamp"].is_unique


def test_round_discovery_is_numeric_not_lexicographic() -> None:
    assert [item.task_id for item in discover_round_files(LOAD_DIR)] == list(range(1, 16))
