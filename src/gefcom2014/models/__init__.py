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
    resolve_l2_leaf_regs,
    summarize_search_results,
)
from .selection import select_features_from_loss_change
from .selected import resolve_candidate_parameters, resolve_selected_features

__all__ = [
    "build_multi_quantile_loss",
    "CatBoostBaseCandidate",
    "build_catboost_candidates",
    "effective_candidate_name",
    "fit_catboost_quantiles",
    "predict_catboost_quantiles",
    "resolve_l2_leaf_regs",
    "resolve_candidate_parameters",
    "resolve_selected_features",
    "summarize_search_results",
    "select_features_from_loss_change",
]
