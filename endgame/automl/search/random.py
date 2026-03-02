from __future__ import annotations

"""Random search strategy.

This module provides a simple random search strategy that samples
valid pipeline configurations from the search space.
"""

import logging
import random
from typing import Any

import numpy as np

from endgame.automl.model_registry import MODEL_REGISTRY
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)


class RandomSearch(BaseSearchStrategy):
    """Random search strategy for pipeline optimization.

    This strategy randomly samples valid pipeline configurations from
    the search space. It's useful for exploration and as a baseline.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    eval_metric : str
        Evaluation metric to optimize.
    model_pool : list of str, optional
        Explicit list of models to consider.
    max_configs : int, optional
        Maximum total configurations to generate.
    include_preprocessing : bool, default=True
        Whether to include preprocessing in search.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> strategy = RandomSearch(
    ...     task_type="classification",
    ...     max_configs=100,
    ... )
    >>> configs = strategy.suggest(meta_features=meta_features, n_suggestions=10)
    """

    # Hyperparameter search spaces
    HP_SPACES = {
        # GBDT parameters
        "n_estimators": {"type": "int", "low": 100, "high": 3000, "log": False},
        "learning_rate": {"type": "float", "low": 0.005, "high": 0.3, "log": True},
        "max_depth": {"type": "int", "low": 3, "high": 10, "log": False},
        "num_leaves": {"type": "int", "low": 15, "high": 127, "log": False},
        "min_child_samples": {"type": "int", "low": 5, "high": 100, "log": False},
        "subsample": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0, "log": False},
        "reg_alpha": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        "reg_lambda": {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
        # Neural network parameters
        "n_epochs": {"type": "int", "low": 20, "high": 200, "log": False},
        "batch_size": {"type": "int", "low": 16, "high": 256, "log": False},
        "hidden_dim": {"type": "int", "low": 32, "high": 512, "log": False},
        "n_layers": {"type": "int", "low": 1, "high": 6, "log": False},
        "dropout": {"type": "float", "low": 0.0, "high": 0.5, "log": False},
        # Linear model parameters
        "C": {"type": "float", "low": 1e-4, "high": 100.0, "log": True},
        "alpha": {"type": "float", "low": 1e-6, "high": 10.0, "log": True},
    }

    # Preprocessing options
    PREPROCESSING_CHOICES = {
        "imputer": [
            None,
            {"strategy": "mean"},
            {"strategy": "median"},
            {"strategy": "most_frequent"},
        ],
        "encoder": [
            None,
            {"method": "onehot"},
            {"method": "target"},
            {"method": "ordinal"},
            {"method": "count"},
        ],
        "scaler": [
            None,
            {"method": "standard"},
            {"method": "minmax"},
            {"method": "robust"},
        ],
        "feature_selector": [
            None,
            {"method": "variance", "threshold": 0.01},
            {"method": "correlation", "threshold": 0.95},
        ],
    }

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        max_configs: int | None = None,
        include_preprocessing: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
        )

        self.max_configs = max_configs
        self.include_preprocessing = include_preprocessing

        # Set model pool
        if model_pool is not None:
            self.model_pool = model_pool
        else:
            self.model_pool = list(MODEL_REGISTRY.keys())

        # Track generated configs to avoid duplicates
        self._generated_ids: set[str] = set()

        # Set random seed
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest random pipeline configurations.

        Parameters
        ----------
        meta_features : dict, optional
            Dataset meta-features (used for filtering).
        n_suggestions : int, default=1
            Number of configurations to suggest.

        Returns
        -------
        list of PipelineConfig
            Suggested configurations.
        """
        if meta_features is None:
            meta_features = {}

        # Check if we've reached max configs
        if self.max_configs and len(self._generated_ids) >= self.max_configs:
            if self.verbose > 0:
                print(f"Reached max_configs limit ({self.max_configs})")
            return []

        # Get available models
        available = self._get_available_models(meta_features)

        configs = []
        attempts = 0
        max_attempts = n_suggestions * 10  # Avoid infinite loops

        while len(configs) < n_suggestions and attempts < max_attempts:
            attempts += 1

            # Generate random config
            config = self._generate_random_config(available, meta_features)

            # Check for duplicates
            if config.config_id not in self._generated_ids:
                self._generated_ids.add(config.config_id)
                configs.append(config)

                if self.max_configs and len(self._generated_ids) >= self.max_configs:
                    break

        if self.verbose > 0:
            print(f"Random search: generated {len(configs)} configs")

        return configs

    def _get_available_models(
        self,
        meta_features: dict[str, float],
    ) -> list[str]:
        """Get available models for this task.

        Parameters
        ----------
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        list of str
            Available model names.
        """
        available = []
        n_samples = meta_features.get("nr_inst", 10000)

        for model_name in self.model_pool:
            if model_name not in MODEL_REGISTRY:
                continue

            info = MODEL_REGISTRY[model_name]

            # Check task type
            if self.task_type == "classification":
                if "classification" not in info.task_types and "both" not in info.task_types:
                    continue
            elif self.task_type == "regression":
                if "regression" not in info.task_types and "both" not in info.task_types:
                    continue

            # Check sample size limits
            if info.max_samples and n_samples > info.max_samples:
                continue
            if info.min_samples and n_samples < info.min_samples:
                continue

            available.append(model_name)

        return available if available else ["lgbm"]

    def _generate_random_config(
        self,
        available_models: list[str],
        meta_features: dict[str, float],
    ) -> PipelineConfig:
        """Generate a random pipeline configuration.

        Parameters
        ----------
        available_models : list of str
            Available model names.
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        PipelineConfig
            Random configuration.
        """
        # Select random model
        model_name = random.choice(available_models)
        info = MODEL_REGISTRY.get(model_name)

        # Generate random parameters
        params = self._sample_params(model_name)

        # Generate random preprocessing
        preprocessing = []
        if self.include_preprocessing:
            preprocessing = self._sample_preprocessing(meta_features)

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={"source": "random_search"},
        )

    def _sample_params(self, model_name: str) -> dict[str, Any]:
        """Sample random hyperparameters for a model.

        Parameters
        ----------
        model_name : str
            Model name.

        Returns
        -------
        dict
            Sampled hyperparameters.
        """
        info = MODEL_REGISTRY.get(model_name)
        params = info.default_params.copy() if info else {}

        # Sample values for parameters in our search space
        for param_name, space in self.HP_SPACES.items():
            if param_name in params:
                value = self._sample_value(space)
                params[param_name] = value

        return params

    def _sample_value(self, space: dict[str, Any]) -> Any:
        """Sample a value from a parameter space.

        Parameters
        ----------
        space : dict
            Parameter space specification.

        Returns
        -------
        value
            Sampled value.
        """
        low = space["low"]
        high = space["high"]
        log_scale = space.get("log", False)

        if space["type"] == "int":
            if log_scale:
                value = int(np.exp(random.uniform(np.log(low), np.log(high))))
            else:
                value = random.randint(low, high)
        elif space["type"] == "float":
            if log_scale:
                value = np.exp(random.uniform(np.log(low), np.log(high)))
            else:
                value = random.uniform(low, high)
        else:
            value = random.choice([low, high])

        return value

    def _sample_preprocessing(
        self,
        meta_features: dict[str, float],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Sample random preprocessing steps.

        Parameters
        ----------
        meta_features : dict
            Dataset meta-features.

        Returns
        -------
        list of tuple
            Preprocessing steps.
        """
        steps = []

        for step_name, choices in self.PREPROCESSING_CHOICES.items():
            choice = random.choice(choices)
            if choice is not None:
                steps.append((step_name, choice))

        return steps

    def update(self, result: SearchResult) -> None:
        """Update strategy with evaluation result.

        Parameters
        ----------
        result : SearchResult
            Result from evaluating a configuration.
        """
        super().update(result)

        if self.verbose > 1:
            logger.debug(
                f"Evaluated {result.config.model_name}: "
                f"score={result.score:.4f}"
            )

    def get_search_space_size(self) -> int:
        """Estimate the size of the search space.

        Returns
        -------
        int
            Estimated number of possible configurations.
        """
        # This is a rough estimate
        n_models = len(self.model_pool)
        n_hp_combinations = 10 ** len(self.HP_SPACES)  # Very rough
        n_preprocessing = np.prod([len(v) for v in self.PREPROCESSING_CHOICES.values()])

        return int(n_models * n_hp_combinations * n_preprocessing)
