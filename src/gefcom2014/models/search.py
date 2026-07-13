"""CatBoost search-space expansion and fold-result aggregation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CatBoostBaseCandidate:
    """One fitted model whose tree prefixes form effective candidates."""

    name: str
    depth: int
    learning_rate: float
    max_iterations: int
    iteration_counts: tuple[int, ...]


def build_catboost_candidates(
    search: dict[str, Any],
) -> tuple[CatBoostBaseCandidate, ...]:
    """Expand configured depths and learning-rate schedules into actual fits."""

    candidates: list[CatBoostBaseCandidate] = []
    for depth_value in search["depths"]:
        depth = int(depth_value)
        for schedule in search["schedules"]:
            learning_rate = float(schedule["learning_rate"])
            maximum = int(schedule["max_iterations"])
            counts = tuple(int(value) for value in schedule["iteration_counts"])
            if depth <= 0 or learning_rate <= 0 or maximum <= 0:
                raise ValueError(
                    "Depth, learning rate, and iterations must be positive"
                )
            if not counts or counts != tuple(sorted(set(counts))):
                raise ValueError("iteration_counts must be unique and increasing")
            if counts[-1] != maximum or counts[0] <= 0:
                raise ValueError(
                    "iteration_counts must be positive and end at max_iterations"
                )
            rate = f"{learning_rate:g}".replace(".", "p")
            candidates.append(
                CatBoostBaseCandidate(
                    name=f"depth{depth}_lr{rate}",
                    depth=depth,
                    learning_rate=learning_rate,
                    max_iterations=maximum,
                    iteration_counts=counts,
                )
            )
    names = [candidate.name for candidate in candidates]
    if len(names) != len(set(names)):
        raise ValueError("Search configuration produced duplicate candidates")
    return tuple(candidates)


def resolve_l2_leaf_regs(search: dict[str, Any]) -> tuple[float, ...]:
    """Validate scalar or grid-form L2 regularization configuration."""

    has_scalar = "l2_leaf_reg" in search
    has_grid = "l2_leaf_regs" in search
    if has_scalar == has_grid:
        raise ValueError("Configure exactly one of l2_leaf_reg or l2_leaf_regs")
    raw_values = (
        [search["l2_leaf_reg"]]
        if has_scalar
        else list(search["l2_leaf_regs"])
    )
    values = tuple(float(value) for value in raw_values)
    if not values or any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("L2 regularization values must be finite and positive")
    if len(values) != len(set(values)):
        raise ValueError("L2 regularization values must be unique")
    return values


def effective_candidate_name(
    base: CatBoostBaseCandidate, iterations: int, l2: float
) -> str:
    """Name one tree-count prefix of a fitted base candidate."""

    regularization = f"{l2:g}".replace(".", "p")
    return f"{base.name}_l2{regularization}_trees{iterations}"


def summarize_search_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold diagnostics with hours as the primary loss weights."""

    records: list[dict[str, float | int | str]] = []
    for candidate, group in results.groupby("candidate", sort=False):
        weights = group["evaluation_rows"].to_numpy(dtype=float)
        loss = np.average(group["pinball_loss"], weights=weights)
        baseline = np.average(group["baseline_pinball_loss"], weights=weights)
        records.append(
            {
                "candidate": candidate,
                "depth": int(group["depth"].iloc[0]),
                "learning_rate": float(group["learning_rate"].iloc[0]),
                "l2_leaf_reg": float(group["l2_leaf_reg"].iloc[0]),
                "iterations": int(group["iterations"].iloc[0]),
                "folds": len(group),
                "forecast_hours": int(weights.sum()),
                "pinball_loss": loss,
                "baseline_pinball_loss": baseline,
                "pinball_difference": loss - baseline,
                "relative_improvement": 1.0 - loss / baseline,
                "fold_pinball_mean": float(group["pinball_loss"].mean()),
                "fold_pinball_std": float(group["pinball_loss"].std(ddof=1)),
                "median_mae": np.average(group["median_mae"], weights=weights),
                "median_bias": np.average(group["median_bias"], weights=weights),
                "coverage_90": np.average(group["coverage_90"], weights=weights),
                "mean_width_90": np.average(group["mean_width_90"], weights=weights),
                "invalid_90_intervals": int(
                    group["invalid_90_intervals"].sum()
                    if "invalid_90_intervals" in group
                    else 0
                ),
                "mean_absolute_calibration_error": np.average(
                    group["mean_absolute_calibration_error"], weights=weights
                ),
                "quantile_crossings": int(group["quantile_crossings"].sum()),
                "fit_seconds": float(group["fit_seconds"].sum()),
            }
        )
    return pd.DataFrame(records).sort_values(
        ["folds", "pinball_loss"], ascending=[False, True], ignore_index=True
    )
