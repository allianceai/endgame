from __future__ import annotations

"""Search strategies for AutoML pipeline configuration.

This module provides various search strategies for finding optimal
ML pipeline configurations:

- PortfolioSearch: Train diverse model portfolio in parallel (default)
- HeuristicSearch: Data-driven rules based on meta-features
- GeneticSearch: Evolutionary optimization of pipelines
- RandomSearch: Random valid pipeline sampling
- BayesianSearch: Optuna-based Bayesian optimization
"""

from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)
from endgame.automl.search.portfolio import PortfolioSearch

__all__ = [
    "BaseSearchStrategy",
    "PipelineConfig",
    "SearchResult",
    "PortfolioSearch",
]

# Lazy imports for other search strategies
def __getattr__(name: str):
    if name == "HeuristicSearch":
        from endgame.automl.search.heuristic import HeuristicSearch
        return HeuristicSearch
    elif name == "GeneticSearch":
        from endgame.automl.search.genetic import GeneticSearch
        return GeneticSearch
    elif name == "RandomSearch":
        from endgame.automl.search.random import RandomSearch
        return RandomSearch
    elif name == "BayesianSearch":
        from endgame.automl.search.bayesian import BayesianSearch
        return BayesianSearch
    elif name == "BanditSearch":
        from endgame.automl.search.bandit import BanditSearch
        return BanditSearch
    elif name == "AdaptiveSearch":
        from endgame.automl.search.adaptive import AdaptiveSearch
        return AdaptiveSearch
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
