from __future__ import annotations

"""Endgame AutoML Framework.

This module provides automated machine learning capabilities that match
AutoGluon's 3-line simplicity while leveraging Endgame's 100+ models,
meta-learning infrastructure, and competition-winning techniques.

Example
-------
>>> from endgame.automl import AutoMLPredictor
>>> predictor = AutoMLPredictor(label="target").fit("train.csv")
>>> predictions = predictor.predict("test.csv")

Or with more control:

>>> from endgame.automl import TabularPredictor
>>> predictor = TabularPredictor(
...     label="price",
...     presets="best_quality",
...     time_limit=3600,
... )
>>> predictor.fit(train_df)
>>> predictions = predictor.predict(test_df)

Available Presets
-----------------
- best_quality: Maximum quality, no time limit, uses all models
- high_quality: High quality with 4-hour default limit
- good_quality: Good quality with 1-hour default limit
- medium_quality: Balanced speed/quality (default), 15-minute limit
- fast: Fast training, 5-minute limit, single LGBM
- interpretable: Interpretable models only (EBM, linear, trees)

Search Strategies
-----------------
- portfolio: Train diverse model portfolio in parallel (default)
- heuristic: Data-driven rules based on meta-features
- genetic: Evolutionary optimization of pipelines
- random: Random valid pipeline sampling
- bayesian: Optuna-based Bayesian optimization
"""

from endgame.automl.base import BasePredictor, FitSummary
from endgame.automl.display import display_model, display_models
from endgame.automl.executors.constraint_check import DeploymentConstraints
from endgame.automl.guardrails import (
    DataQualityWarning,
    GuardrailsReport,
    QualityGuardrailsExecutor,
)
from endgame.automl.model_registry import (
    MODEL_FAMILIES,
    MODEL_REGISTRY,
    ModelInfo,
    get_default_portfolio,
    get_model_class,
    get_model_info,
    get_models_by_family,
    list_models,
    register_model,
    unregister_model,
)
from endgame.automl.orchestrator import (
    PipelineOrchestrator,
    PipelineResult,
    StageResult,
)
from endgame.automl.predictor import AutoMLPredictor
from endgame.automl.presets import (
    PRESETS,
    PresetConfig,
    get_preset,
    list_presets,
)
from endgame.automl.report import AutoMLReport, ReportGenerator
from endgame.automl.search import (
    BaseSearchStrategy,
    PipelineConfig,
    PortfolioSearch,
    SearchResult,
)
from endgame.automl.tabular import TabularPredictor
from endgame.automl.time_manager import TimeBudgetManager
from endgame.automl.utils import (
    DataLoader,
    infer_task_type,
    load_data,
)

__all__ = [
    # Main entry points
    "AutoMLPredictor",
    "TabularPredictor",
    "TextPredictor",
    "VisionPredictor",
    "TimeSeriesPredictor",
    "AudioPredictor",
    "MultiModalPredictor",
    # Base classes
    "BasePredictor",
    "FitSummary",
    # Display
    "display_model",
    "display_models",
    # Presets
    "PresetConfig",
    "PRESETS",
    "get_preset",
    "list_presets",
    # Search strategies
    "BaseSearchStrategy",
    "PipelineConfig",
    "SearchResult",
    "PortfolioSearch",
    # Orchestration
    "PipelineOrchestrator",
    "PipelineResult",
    "StageResult",
    # Time management
    "TimeBudgetManager",
    # Model registry
    "MODEL_REGISTRY",
    "MODEL_FAMILIES",
    "ModelInfo",
    "register_model",
    "unregister_model",
    "get_model_info",
    "get_model_class",
    "get_default_portfolio",
    "get_models_by_family",
    "list_models",
    # Data utilities
    "DataLoader",
    "load_data",
    "infer_task_type",
    # Guardrails
    "QualityGuardrailsExecutor",
    "DataQualityWarning",
    "GuardrailsReport",
    # Deployment constraints
    "DeploymentConstraints",
    # Report
    "AutoMLReport",
    "ReportGenerator",
]

# Lazy imports for optional domain predictors
def __getattr__(name: str):
    if name == "TextPredictor":
        from endgame.automl.text import TextPredictor
        return TextPredictor
    elif name == "VisionPredictor":
        from endgame.automl.vision import VisionPredictor
        return VisionPredictor
    elif name == "TimeSeriesPredictor":
        from endgame.automl.timeseries import TimeSeriesPredictor
        return TimeSeriesPredictor
    elif name == "AudioPredictor":
        from endgame.automl.audio import AudioPredictor
        return AudioPredictor
    elif name == "MultiModalPredictor":
        from endgame.automl.multimodal import MultiModalPredictor
        return MultiModalPredictor
    # Search strategies (lazy loaded)
    elif name == "HeuristicSearch":
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
