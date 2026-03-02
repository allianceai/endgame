from __future__ import annotations

"""Base classes for search strategies.

This module defines the interfaces and data structures used by
all search strategy implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class PipelineConfig:
    """Represents a complete ML pipeline configuration.

    A pipeline config specifies everything needed to train a model:
    preprocessing steps, model choice, and hyperparameters.

    Attributes
    ----------
    model_name : str
        Name of the model (key in model registry).
    model_params : dict
        Hyperparameters for the model.
    preprocessing : list of tuple
        Preprocessing steps as (name, params) tuples.
    feature_engineering : list of tuple
        Feature engineering steps as (name, params) tuples.
    ensemble_weight : float
        Initial weight for ensembling (may be adjusted later).
    config_id : str, optional
        Unique identifier for this configuration.
    metadata : dict
        Additional metadata about this configuration.

    Examples
    --------
    >>> config = PipelineConfig(
    ...     model_name="lgbm",
    ...     model_params={"n_estimators": 1000, "learning_rate": 0.05},
    ...     preprocessing=[
    ...         ("imputer", {"strategy": "median"}),
    ...         ("encoder", {"method": "target"}),
    ...     ],
    ... )
    """

    model_name: str
    model_params: dict[str, Any] = field(default_factory=dict)
    preprocessing: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    feature_engineering: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    ensemble_weight: float = 1.0
    config_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate config_id if not provided."""
        if self.config_id is None:
            import hashlib
            import json

            # Create a deterministic hash of the config
            config_str = json.dumps(
                {
                    "model": self.model_name,
                    "params": self.model_params,
                    "preproc": self.preprocessing,
                    "fe": self.feature_engineering,
                },
                sort_keys=True,
                default=str,
            )
            self.config_id = hashlib.md5(config_str.encode()).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "model_name": self.model_name,
            "model_params": self.model_params,
            "preprocessing": self.preprocessing,
            "feature_engineering": self.feature_engineering,
            "ensemble_weight": self.ensemble_weight,
            "config_id": self.config_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PipelineConfig:
        """Create from dictionary representation."""
        return cls(
            model_name=d["model_name"],
            model_params=d.get("model_params", {}),
            preprocessing=d.get("preprocessing", []),
            feature_engineering=d.get("feature_engineering", []),
            ensemble_weight=d.get("ensemble_weight", 1.0),
            config_id=d.get("config_id"),
            metadata=d.get("metadata", {}),
        )


@dataclass
class SearchResult:
    """Result from evaluating a pipeline configuration.

    Attributes
    ----------
    config : PipelineConfig
        The configuration that was evaluated.
    score : float
        Primary evaluation metric score.
    scores : dict
        All evaluation metric scores.
    fit_time : float
        Time taken to fit the model in seconds.
    predict_time : float
        Time taken for predictions in seconds.
    oof_predictions : np.ndarray, optional
        Out-of-fold predictions.
    feature_importances : dict, optional
        Feature importance scores.
    success : bool
        Whether the evaluation completed successfully.
    error : str, optional
        Error message if evaluation failed.
    metadata : dict
        Additional metadata about this result.
    """

    config: PipelineConfig
    score: float
    scores: dict[str, float] = field(default_factory=dict)
    fit_time: float = 0.0
    predict_time: float = 0.0
    oof_predictions: np.ndarray | None = None
    feature_importances: dict[str, float] | None = None
    success: bool = True
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSearchStrategy(ABC):
    """Base class for pipeline search strategies.

    A search strategy is responsible for suggesting pipeline configurations
    to try and updating its internal state based on the results.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    eval_metric : str or callable
        Evaluation metric to optimize.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    results_ : list of SearchResult
        Results from all evaluated configurations.
    best_result_ : SearchResult or None
        Best result found so far.
    n_evaluated_ : int
        Number of configurations evaluated.
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
        excluded_models: set[str] | None = None,
    ):
        self.task_type = task_type
        self.eval_metric = eval_metric
        self.random_state = random_state
        self.verbose = verbose
        self.excluded_models = excluded_models or set()

        # State
        self.results_: list[SearchResult] = []
        self.best_result_: SearchResult | None = None
        self.n_evaluated_: int = 0

        # Set random state
        if random_state is not None:
            np.random.seed(random_state)

    @abstractmethod
    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest pipeline configurations to try.

        Parameters
        ----------
        meta_features : dict, optional
            Dataset meta-features for informed suggestions.
        n_suggestions : int, default=1
            Number of configurations to suggest.

        Returns
        -------
        list of PipelineConfig
            Suggested configurations.
        """
        pass

    def update(self, result: SearchResult) -> None:
        """Update the search strategy with a new result.

        Parameters
        ----------
        result : SearchResult
            Result from evaluating a configuration.
        """
        self.results_.append(result)
        self.n_evaluated_ += 1

        # Update best result
        if result.success:
            if self.best_result_ is None or result.score > self.best_result_.score:
                self.best_result_ = result

                if self.verbose > 0:
                    print(
                        f"New best: {result.config.model_name} "
                        f"score={result.score:.4f}"
                    )

    def get_best(self, n: int = 1) -> list[SearchResult]:
        """Get the best results found so far.

        Parameters
        ----------
        n : int, default=1
            Number of best results to return.

        Returns
        -------
        list of SearchResult
            Top n results sorted by score (descending).
        """
        successful = [r for r in self.results_ if r.success]
        sorted_results = sorted(successful, key=lambda r: r.score, reverse=True)
        return sorted_results[:n]

    def get_results_summary(self) -> dict[str, Any]:
        """Get a summary of search results.

        Returns
        -------
        dict
            Summary statistics.
        """
        successful = [r for r in self.results_ if r.success]
        failed = [r for r in self.results_ if not r.success]

        if not successful:
            return {
                "n_evaluated": self.n_evaluated_,
                "n_successful": 0,
                "n_failed": len(failed),
                "best_score": None,
                "best_model": None,
            }

        scores = [r.score for r in successful]
        return {
            "n_evaluated": self.n_evaluated_,
            "n_successful": len(successful),
            "n_failed": len(failed),
            "best_score": max(scores),
            "mean_score": np.mean(scores),
            "std_score": np.std(scores),
            "best_model": self.best_result_.config.model_name if self.best_result_ else None,
            "total_fit_time": sum(r.fit_time for r in successful),
        }

    def should_stop(self, max_iterations: int | None = None) -> bool:
        """Check if search should stop.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum number of iterations.

        Returns
        -------
        bool
            Whether to stop searching.
        """
        if max_iterations and self.n_evaluated_ >= max_iterations:
            return True
        return False

    def reset(self) -> None:
        """Reset the search strategy state."""
        self.results_ = []
        self.best_result_ = None
        self.n_evaluated_ = 0


class SearchCallback:
    """Callback interface for search progress monitoring.

    Implement this interface to receive updates during search.
    """

    def on_search_start(self, strategy: BaseSearchStrategy) -> None:
        """Called when search begins."""
        pass

    def on_config_evaluated(
        self,
        strategy: BaseSearchStrategy,
        result: SearchResult,
    ) -> None:
        """Called after each configuration is evaluated."""
        pass

    def on_search_end(self, strategy: BaseSearchStrategy) -> None:
        """Called when search completes."""
        pass


class ProgressCallback(SearchCallback):
    """Simple progress callback that prints updates."""

    def __init__(self, total_configs: int | None = None):
        self.total_configs = total_configs

    def on_config_evaluated(
        self,
        strategy: BaseSearchStrategy,
        result: SearchResult,
    ) -> None:
        """Print progress after each evaluation."""
        n = strategy.n_evaluated_
        status = "OK" if result.success else "FAIL"
        score_str = f"{result.score:.4f}" if result.success else "N/A"

        if self.total_configs:
            print(
                f"[{n}/{self.total_configs}] {result.config.model_name}: "
                f"{status} score={score_str} time={result.fit_time:.1f}s"
            )
        else:
            print(
                f"[{n}] {result.config.model_name}: "
                f"{status} score={score_str} time={result.fit_time:.1f}s"
            )
