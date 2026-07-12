import numpy as np
import pytest

from gefcom2014.metrics import (
    central_interval_coverage,
    pinball_loss,
    quantile_coverage,
    quantile_crossing_count,
)


def test_pinball_loss_matches_hand_calculation() -> None:
    truth = np.array([0.0, 2.0])
    predictions = np.array([[1.0, 1.0], [1.0, 1.0]])
    quantiles = np.array([0.25, 0.75])

    # Row losses are (0.75, 0.25) and (0.25, 0.75), respectively.
    assert pinball_loss(truth, predictions, quantiles) == pytest.approx(0.5)


def test_pinball_rejects_transposed_predictions() -> None:
    with pytest.raises(ValueError, match="prediction shape"):
        pinball_loss(np.ones(3), np.ones((2, 3)), np.array([0.25, 0.75]))


def test_central_interval_coverage_is_inclusive() -> None:
    truth = np.array([0.0, 1.0, 2.0, 3.0])
    lower = np.array([0.0, 0.0, 2.0, 4.0])
    upper = np.array([0.0, 1.0, 2.5, 5.0])
    assert central_interval_coverage(truth, lower, upper) == pytest.approx(0.75)


def test_quantile_coverage_and_crossing_diagnostics() -> None:
    truth = np.array([1.0, 3.0])
    predictions = np.array([[0.0, 1.0, 2.0], [2.0, 4.0, 5.0]])
    quantiles = np.array([0.25, 0.5, 0.75])

    np.testing.assert_allclose(
        quantile_coverage(truth, predictions, quantiles), [0.0, 1.0, 1.0]
    )
    assert quantile_crossing_count(predictions, quantiles) == 0
    crossed = predictions.copy()
    crossed[0] = [0.0, 2.0, 1.0]
    assert quantile_crossing_count(crossed, quantiles) == 1


def test_pinball_rejects_unsorted_quantile_levels() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        pinball_loss(np.array([1.0]), np.array([[1.0, 1.0]]), np.array([0.9, 0.1]))
