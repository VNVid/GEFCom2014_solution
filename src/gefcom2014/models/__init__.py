"""Forecasting model wrappers with stable project-level contracts."""

from .catboost import (
    build_multi_quantile_loss,
    fit_catboost_quantiles,
    predict_catboost_quantiles,
)
from .search import (
    CatBoostBaseCandidate,
    build_catboost_candidates,
    effective_candidate_name,
    summarize_search_results,
)

__all__ = [
    "build_multi_quantile_loss",
    "CatBoostBaseCandidate",
    "build_catboost_candidates",
    "effective_candidate_name",
    "fit_catboost_quantiles",
    "predict_catboost_quantiles",
    "summarize_search_results",
]
