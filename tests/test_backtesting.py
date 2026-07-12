import pandas as pd
import pytest

from gefcom2014.backtesting import (
    MonthlyFold,
    data_for_fold,
    monthly_folds,
    prepare_backtest_frame,
)


def test_monthly_folds_are_consecutive_and_half_open() -> None:
    folds = monthly_folds("2009-01-01", "2009-04-01")
    assert folds == [
        MonthlyFold(pd.Timestamp("2009-01-01"), pd.Timestamp("2009-02-01")),
        MonthlyFold(pd.Timestamp("2009-02-01"), pd.Timestamp("2009-03-01")),
        MonthlyFold(pd.Timestamp("2009-03-01"), pd.Timestamp("2009-04-01")),
    ]


def test_monthly_folds_reject_partial_month_boundaries() -> None:
    with pytest.raises(ValueError, match="first day"):
        monthly_folds("2009-01-02", "2009-04-01")


def test_fold_training_stops_before_origin_and_target_covers_whole_month() -> None:
    period_starts = pd.date_range(
        "2008-12-31", "2009-02-01", freq="h", inclusive="left"
    )
    actuals = pd.DataFrame(
        {
            "zone_id": 1,
            "timestamp": period_starts + pd.Timedelta(hours=1),
            "load": range(len(period_starts)),
        }
    )
    frame = prepare_backtest_frame(actuals)
    fold = MonthlyFold(pd.Timestamp("2009-01-01"), pd.Timestamp("2009-02-01"))
    training, target = data_for_fold(frame, fold)

    assert training["period_start"].max() == pd.Timestamp("2008-12-31 23:00")
    assert target["period_start"].min() == fold.origin
    assert target["period_start"].max() == pd.Timestamp("2009-01-31 23:00")
    assert len(target) == 31 * 24


def test_backtest_frame_preserves_columns_for_future_models() -> None:
    actuals = pd.DataFrame(
        {
            "zone_id": [1],
            "timestamp": [pd.Timestamp("2009-01-01 01:00")],
            "load": [100.0],
            "w1": [42.0],
        }
    )
    frame = prepare_backtest_frame(actuals)

    assert frame.loc[0, "w1"] == 42.0
    assert frame.loc[0, "period_start"] == pd.Timestamp("2009-01-01 00:00")
