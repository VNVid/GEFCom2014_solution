"""Load and validate the sequential GEFCom2014 load-track releases.

The competition files are easy to misuse: Task 1 contains the long history,
whereas Tasks 2--15 contain only the newly revealed preceding month.  This
module makes that release structure explicit and provides a training-history
loader that never reads a later task.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd


WEATHER_COLUMNS = tuple(f"w{i}" for i in range(1, 26))
QUANTILES = np.round(np.arange(0.01, 1.0, 0.01), 2)
REQUIRED_TRAIN_COLUMNS = ("ZONEID", "TIMESTAMP", "LOAD", *WEATHER_COLUMNS)
TASK_DIRECTORY_PATTERN = re.compile(r"Task (\d+)$")


@dataclass(frozen=True)
class RoundFiles:
    """Paths belonging to one sequential forecasting round."""

    task_id: int
    train_path: Path
    benchmark_path: Path


def parse_competition_timestamps(values: pd.Series) -> pd.Series:
    """Parse the compact ``MMDDYYYY H:MM`` competition timestamps.

    Month and day are not zero padded, so their boundary can be ambiguous in
    isolation. Exact hourly continuity resolves that boundary. The dataset's
    timestamps are timezone-naive and are deliberately kept that way.
    """

    raw = values.astype("string").str.strip()
    parsed: list[datetime] = []
    previous: datetime | None = None
    for value in raw:
        candidates = _timestamp_candidates(value)
        expected = previous + pd.Timedelta(hours=1) if previous is not None else None
        continuous = [candidate for candidate in candidates if candidate == expected]
        if len(continuous) == 1:
            chosen = continuous[0]
        else:
            # The compact representation is intrinsically ambiguous in
            # isolation (``1112001`` could be Jan 11 or Nov 1).  Python's
            # strptime supplies the correct competition convention at the
            # first row or a deliberate discontinuity; thereafter exact
            # hourly continuity resolves every ambiguous date.
            chosen = datetime.strptime(value, "%m%d%Y %H:%M")
            if chosen not in candidates:  # defensive guard against parser drift
                raise ValueError(f"Could not disambiguate timestamp {value!r}")
        parsed.append(chosen)
        previous = chosen
    return pd.Series(pd.to_datetime(parsed), index=values.index, name="timestamp")


def _timestamp_candidates(value: str) -> list[datetime]:
    """Enumerate valid month/day splits for one compact timestamp."""

    try:
        date_token, time_token = value.split()
        hour_text, minute_text = time_token.split(":")
        year = int(date_token[-4:])
        month_day = date_token[:-4]
        hour = int(hour_text)
        minute = int(minute_text)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid competition timestamp {value!r}") from exc

    candidates: list[datetime] = []
    for month_width in (1, 2):
        month_text = month_day[:month_width]
        day_text = month_day[month_width:]
        if not month_text or not 1 <= len(day_text) <= 2:
            continue
        # The source format omits padding; accepting 01 would introduce extra
        # interpretations that the producer could not have emitted.
        if (len(month_text) > 1 and month_text.startswith("0")) or (
            len(day_text) > 1 and day_text.startswith("0")
        ):
            continue
        try:
            candidate = datetime(
                year, int(month_text), int(day_text), hour=hour, minute=minute
            )
        except ValueError:
            continue
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"Invalid competition timestamp {value!r}")
    return candidates


def interval_start(timestamps: pd.Series) -> pd.Series:
    """Map official hour-ending timestamps to the modeled interval start.

    The final Task 15 temperature file labels the last observation as hour 24
    on 2011-12-31, while the load solution labels the same value as
    2012-01-01 00:00. Subtracting one hour gives the correct operating date
    and a conventional 0--23 hour feature.
    """

    return timestamps - pd.Timedelta(hours=1)


def discover_round_files(load_dir: str | Path) -> list[RoundFiles]:
    """Discover and numerically order all Task directories."""

    root = Path(load_dir)
    rounds: list[RoundFiles] = []
    for directory in root.glob("Task *"):
        match = TASK_DIRECTORY_PATTERN.fullmatch(directory.name)
        if not match:
            continue
        task_id = int(match.group(1))
        train_path = directory / f"L{task_id}-train.csv"
        benchmark_path = directory / f"L{task_id}-benchmark.csv"
        if not train_path.is_file() or not benchmark_path.is_file():
            raise FileNotFoundError(f"Incomplete files for Task {task_id}: {directory}")
        rounds.append(RoundFiles(task_id, train_path, benchmark_path))

    rounds.sort(key=lambda item: item.task_id)
    ids = [item.task_id for item in rounds]
    if ids != list(range(1, 16)):
        raise ValueError(f"Expected Task 1 through Task 15, found {ids}")
    return rounds


def _coerce_numeric(frame: pd.DataFrame, columns: Iterable[str], path: Path) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    weather_missing = int(frame[list(WEATHER_COLUMNS)].isna().sum().sum())
    if weather_missing:
        raise ValueError(f"Found {weather_missing} invalid/missing weather values in {path}")


def read_train_file(path: str | Path, task_id: int) -> pd.DataFrame:
    """Read one raw train release into the canonical hourly schema."""

    csv_path = Path(path)
    raw = pd.read_csv(csv_path)
    missing_columns = sorted(set(REQUIRED_TRAIN_COLUMNS) - set(raw.columns))
    if missing_columns:
        raise ValueError(f"Missing columns {missing_columns} in {csv_path}")

    frame = raw.loc[:, REQUIRED_TRAIN_COLUMNS].rename(
        columns={"ZONEID": "zone_id", "TIMESTAMP": "timestamp", "LOAD": "load"}
    )
    frame["timestamp"] = parse_competition_timestamps(frame["timestamp"])
    frame["zone_id"] = pd.to_numeric(frame["zone_id"], errors="raise").astype("int16")
    _coerce_numeric(frame, ("load", *WEATHER_COLUMNS), csv_path)
    frame["source_task"] = np.int16(task_id)
    frame["source_kind"] = "train"
    return frame.sort_values("timestamp", ignore_index=True)


def read_benchmark_file(path: str | Path, task_id: int) -> pd.DataFrame:
    """Read a 99-quantile benchmark file with normalized quantile names."""

    csv_path = Path(path)
    raw = pd.read_csv(csv_path)
    required = {"ZONEID", "TIMESTAMP"}
    if not required.issubset(raw.columns):
        raise ValueError(f"Benchmark is missing {sorted(required - set(raw.columns))}: {csv_path}")

    quantile_lookup: dict[str, str] = {}
    for column in raw.columns[2:]:
        try:
            quantile = float(column)
        except ValueError as exc:
            raise ValueError(f"Invalid quantile column {column!r} in {csv_path}") from exc
        quantile_lookup[column] = f"q{quantile:.2f}"

    expected = [f"q{q:.2f}" for q in QUANTILES]
    if sorted(quantile_lookup.values()) != sorted(expected):
        raise ValueError(f"Expected quantiles 0.01--0.99 in {csv_path}")

    frame = raw.rename(
        columns={"ZONEID": "zone_id", "TIMESTAMP": "timestamp", **quantile_lookup}
    )
    frame["timestamp"] = parse_competition_timestamps(frame["timestamp"])
    frame["zone_id"] = pd.to_numeric(frame["zone_id"], errors="raise").astype("int16")
    frame[expected] = frame[expected].apply(pd.to_numeric, errors="raise")
    frame = frame.copy()  # defragment after assigning 99 quantile columns
    frame["source_task"] = np.int16(task_id)
    return frame[["zone_id", "timestamp", *expected, "source_task"]].sort_values(
        "timestamp", ignore_index=True
    )


def read_task15_solution(load_dir: str | Path) -> pd.DataFrame:
    """Read the released December 2011 load and station temperatures."""

    root = Path(load_dir)
    temperature_path = root / "Solution to Task 15" / "solution15_L_temperature.csv"
    load_path = root / "Solution to Task 15" / "solution15_L.csv"

    raw = pd.read_csv(temperature_path)
    required = {"date", "hour", "LOAD", *WEATHER_COLUMNS}
    if not required.issubset(raw.columns):
        raise ValueError(f"Unexpected Task 15 temperature solution schema: {temperature_path}")
    hours = pd.to_numeric(raw["hour"], errors="raise")
    if not hours.between(1, 24).all():
        raise ValueError("Task 15 solution hours must be in [1, 24]")

    frame = raw.rename(columns={"LOAD": "load"}).copy()
    frame["timestamp"] = pd.to_datetime(frame["date"], format="%m/%d/%Y", errors="raise")
    frame["timestamp"] += pd.to_timedelta(hours, unit="h")
    frame["zone_id"] = np.int16(1)
    _coerce_numeric(frame, ("load", *WEATHER_COLUMNS), temperature_path)
    frame["source_task"] = np.int16(16)
    frame["source_kind"] = "task15_solution"
    frame = frame[["zone_id", "timestamp", "load", *WEATHER_COLUMNS, "source_task", "source_kind"]]

    # The separately supplied load solution uses the canonical competition
    # timestamp convention.  Check the two official files against each other.
    load_only = pd.read_csv(load_path).rename(columns={"ZONEID": "zone_id", "LOAD": "load"})
    load_only["timestamp"] = parse_competition_timestamps(load_only["TIMESTAMP"])
    comparison = frame[["timestamp", "load"]].merge(
        load_only[["timestamp", "load"]], on="timestamp", suffixes=("_temperature", "_load")
    )
    if len(comparison) != len(frame) or not np.allclose(
        comparison["load_temperature"], comparison["load_load"]
    ):
        raise ValueError("The two official Task 15 solution files disagree")
    return frame.sort_values("timestamp", ignore_index=True)


def load_training_history(load_dir: str | Path, through_task: int) -> pd.DataFrame:
    """Return only information released by the start of ``through_task``.

    For example, the Task 3 history contains the Task 1 long history plus the
    newly revealed October and November 2010 loads from the Task 2 and Task 3
    train files.  No later task or final solution file is opened.
    """

    if not 1 <= through_task <= 15:
        raise ValueError("through_task must be between 1 and 15")
    rounds = discover_round_files(load_dir)
    pieces = [read_train_file(item.train_path, item.task_id) for item in rounds[:through_task]]
    history = pd.concat(pieces, ignore_index=True).sort_values("timestamp", ignore_index=True)
    _assert_unique_timestamps(history, context=f"training history through Task {through_task}")

    forecast_start = read_benchmark_file(
        rounds[through_task - 1].benchmark_path, through_task
    )["timestamp"].min()
    if history["timestamp"].max() >= forecast_start:
        raise ValueError(
            f"Training data reaches {history['timestamp'].max()}, "
            f"not before forecast start {forecast_start}"
        )
    return history


def load_complete_history(load_dir: str | Path, include_solution: bool = True) -> pd.DataFrame:
    """Assemble all sequential releases for retrospective analysis.

    Use this directly for whole-period descriptive analysis. Official-round
    forecasting should call :func:`load_training_history` at each release.
    Arbitrary-origin retrospective evaluation should use
    :func:`load_backtest_actuals`, followed by an explicit per-origin slice in
    the backtesting module. The complete frame must never be passed to a model.
    """

    rounds = discover_round_files(load_dir)
    pieces = [read_train_file(item.train_path, item.task_id) for item in rounds]
    if include_solution:
        pieces.append(read_task15_solution(load_dir))
    history = pd.concat(pieces, ignore_index=True).sort_values("timestamp", ignore_index=True)
    _assert_unique_timestamps(history, context="complete retrospective history")
    return history


def load_backtest_actuals(load_dir: str | Path) -> pd.DataFrame:
    """Return the complete observed-load timeline used as a backtest label store.

    This function deliberately exposes outcomes from every release so arbitrary
    historical folds can be evaluated, including folds before Task 1.  It does
    not define forecast-time availability: backtesting code must take a fresh
    ``timestamp < origin`` slice for every fold before fitting or forecasting.
    """

    history = load_complete_history(load_dir, include_solution=True)
    observed = history.dropna(subset=["load"]).reset_index(drop=True)
    _assert_unique_timestamps(observed, context="backtest actuals")
    return observed


def actuals_for_round(load_dir: str | Path, task_id: int) -> pd.DataFrame:
    """Load the labels that were revealed after a forecast round ended."""

    rounds = discover_round_files(load_dir)
    if not 1 <= task_id <= len(rounds):
        raise ValueError(f"Unknown task_id {task_id}")
    benchmark = read_benchmark_file(rounds[task_id - 1].benchmark_path, task_id)
    if task_id < 15:
        actual = read_train_file(rounds[task_id].train_path, task_id + 1)
    else:
        actual = read_task15_solution(load_dir)

    expected = pd.Index(benchmark["timestamp"])
    observed = pd.Index(actual["timestamp"])
    if not expected.equals(observed):
        missing = expected.difference(observed)
        extra = observed.difference(expected)
        raise ValueError(
            f"Task {task_id} target/actual timestamps differ; "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if actual["load"].isna().any():
        raise ValueError(f"Task {task_id} contains missing actual loads")
    return actual


def _assert_unique_timestamps(frame: pd.DataFrame, context: str) -> None:
    duplicates = frame.loc[frame["timestamp"].duplicated(keep=False), "timestamp"]
    if not duplicates.empty:
        examples = duplicates.astype(str).head(3).tolist()
        raise ValueError(f"Duplicate timestamps in {context}: {examples}")
