from __future__ import annotations

import pandas as pd
import pytest

from gefcom2014.models import build_catboost_candidates, summarize_search_results


def test_search_expands_joint_depth_and_learning_rate_families() -> None:
    candidates = build_catboost_candidates(
        {
            "depths": [4, 6, 8],
            "schedules": [
                {
                    "learning_rate": 0.04,
                    "max_iterations": 500,
                    "iteration_counts": [250, 375, 500],
                },
                {
                    "learning_rate": 0.08,
                    "max_iterations": 250,
                    "iteration_counts": [125, 188, 250],
                },
            ],
        }
    )

    assert len(candidates) == 6
    assert candidates[0].name == "depth4_lr0p04"
    assert candidates[0].iteration_counts == (250, 375, 500)
    assert candidates[-1].name == "depth8_lr0p08"


def test_candidate_summary_uses_hour_weighted_pinball_loss() -> None:
    results = pd.DataFrame(
        {
            "candidate": ["candidate_a", "candidate_a"],
            "depth": [4, 4],
            "learning_rate": [0.04, 0.04],
            "l2_leaf_reg": [5.0, 5.0],
            "iterations": [250, 250],
            "evaluation_rows": [1, 3],
            "pinball_loss": [2.0, 4.0],
            "baseline_pinball_loss": [5.0, 5.0],
            "median_mae": [1.0, 3.0],
            "median_bias": [1.0, -1.0],
            "coverage_90": [0.8, 1.0],
            "mean_width_90": [10.0, 20.0],
            "invalid_90_intervals": [0, 2],
            "mean_absolute_calibration_error": [0.2, 0.1],
            "quantile_crossings": [2, 3],
            "fit_seconds": [10.0, 20.0],
        }
    )

    summary = summarize_search_results(results).iloc[0]

    assert summary["pinball_loss"] == pytest.approx(3.5)
    assert summary["baseline_pinball_loss"] == pytest.approx(5.0)
    assert summary["relative_improvement"] == pytest.approx(0.3)
    assert summary["invalid_90_intervals"] == 2
    assert summary["quantile_crossings"] == 5
