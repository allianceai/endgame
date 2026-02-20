"""Bayesian optimization search strategy.

This module provides a search strategy using Optuna for Bayesian
optimization of pipeline configurations.
"""

import logging
from typing import Any

from endgame.automl.model_registry import MODEL_REGISTRY
from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)


class BayesianSearch(BaseSearchStrategy):
    """Bayesian optimization search using Optuna.

    This strategy uses Optuna's TPE (Tree-structured Parzen Estimator)
    sampler for efficient hyperparameter optimization.

    Parameters
    ----------
    task_type : str
        Task type ("classification" or "regression").
    eval_metric : str
        Evaluation metric to optimize.
    model_pool : list of str, optional
        Models to optimize. If None, uses a default set.
    n_trials : int, default=100
        Maximum number of trials.
    timeout : int, optional
        Timeout in seconds.
    n_startup_trials : int, default=10
        Number of random trials before using TPE.
    pruner : str, default="hyperband"
        Pruning strategy: "hyperband", "median", or None.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> strategy = BayesianSearch(
    ...     task_type="classification",
    ...     n_trials=50,
    ... )
    >>> configs = strategy.suggest(meta_features=meta_features, n_suggestions=1)
    >>> # Train model and get score...
    >>> strategy.update(result)
    """

    def __init__(
        self,
        task_type: str = "classification",
        eval_metric: str = "auto",
        model_pool: list[str] | None = None,
        n_trials: int = 100,
        timeout: int | None = None,
        n_startup_trials: int = 10,
        pruner: str | None = "hyperband",
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            task_type=task_type,
            eval_metric=eval_metric,
            random_state=random_state,
            verbose=verbose,
        )

        self.n_trials = n_trials
        self.timeout = timeout
        self.n_startup_trials = n_startup_trials
        self.pruner_type = pruner

        # Set model pool
        if model_pool is not None:
            self.model_pool = model_pool
        else:
            # Default to fast models for Bayesian optimization
            self.model_pool = ["lgbm", "xgb", "catboost", "mlp", "linear"]

        # State
        self._study = None
        self._pending_trials: dict[str, Any] = {}  # config_id -> trial
        self._meta_features: dict[str, float] = {}
        self._available_models: list[str] = []

        # Lazy import optuna
        self._optuna = None

    def _import_optuna(self):
        """Lazily import optuna."""
        if self._optuna is None:
            try:
                import optuna
                self._optuna = optuna

                # Suppress optuna logging if not verbose
                if self.verbose < 2:
                    optuna.logging.set_verbosity(optuna.logging.WARNING)
            except ImportError:
                raise ImportError(
                    "Optuna is required for BayesianSearch. "
                    "Install it with: pip install optuna"
                )
        return self._optuna

    def _create_study(self) -> None:
        """Create the Optuna study."""
        optuna = self._import_optuna()

        # Create sampler
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=self.n_startup_trials,
            seed=self.random_state,
        )

        # Create pruner
        if self.pruner_type == "hyperband":
            pruner = optuna.pruners.HyperbandPruner()
        elif self.pruner_type == "median":
            pruner = optuna.pruners.MedianPruner()
        else:
            pruner = optuna.pruners.NopPruner()

        # Create study
        self._study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest pipeline configurations using Bayesian optimization.

        Parameters
        ----------
        meta_features : dict, optional
            Dataset meta-features.
        n_suggestions : int, default=1
            Number of configurations to suggest.

        Returns
        -------
        list of PipelineConfig
            Suggested configurations.
        """
        if meta_features is not None:
            self._meta_features = meta_features

        # Get available models
        self._available_models = self._get_available_models()

        # Create study if needed
        if self._study is None:
            self._create_study()

        optuna = self._import_optuna()

        configs = []
        for _ in range(n_suggestions):
            # Check if we've reached max trials
            if len(self._study.trials) >= self.n_trials:
                if self.verbose > 0:
                    print(f"Reached max trials ({self.n_trials})")
                break

            # Ask for a new trial
            trial = self._study.ask()

            # Sample configuration from trial
            config = self._sample_config(trial)

            # Store trial for later update
            self._pending_trials[config.config_id] = trial

            configs.append(config)

        if self.verbose > 0:
            print(f"Bayesian search: suggested {len(configs)} configs")

        return configs

    def _get_available_models(self) -> list[str]:
        """Get available models for this task.

        Returns
        -------
        list of str
            Available model names.
        """
        available = []
        n_samples = self._meta_features.get("nr_inst", 10000)

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

    def _sample_config(self, trial) -> PipelineConfig:
        """Sample a configuration from an Optuna trial.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial object.

        Returns
        -------
        PipelineConfig
            Sampled configuration.
        """
        # Sample model
        model_name = trial.suggest_categorical("model", self._available_models)
        info = MODEL_REGISTRY.get(model_name)

        # Sample hyperparameters based on model
        params = self._sample_model_params(trial, model_name)

        # Sample preprocessing
        preprocessing = self._sample_preprocessing(trial)

        return PipelineConfig(
            model_name=model_name,
            model_params=params,
            preprocessing=preprocessing,
            metadata={
                "source": "bayesian_search",
                "trial_number": trial.number,
            },
        )

    def _sample_model_params(
        self,
        trial,
        model_name: str,
    ) -> dict[str, Any]:
        """Sample hyperparameters for a specific model.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.
        model_name : str
            Model name.

        Returns
        -------
        dict
            Sampled hyperparameters.
        """
        info = MODEL_REGISTRY.get(model_name)
        params = info.default_params.copy() if info else {}

        if model_name in ("lgbm", "xgb", "catboost"):
            params.update(self._sample_gbdt_params(trial, model_name))
        elif model_name in ("ft_transformer", "saint", "tabnet", "mlp"):
            params.update(self._sample_neural_params(trial, model_name))
        elif model_name == "linear":
            params.update(self._sample_linear_params(trial))
        elif model_name == "svm":
            params.update(self._sample_svm_params(trial))

        return params

    def _sample_gbdt_params(self, trial, model_name: str) -> dict[str, Any]:
        """Sample GBDT hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.
        model_name : str
            GBDT model name.

        Returns
        -------
        dict
            Sampled parameters.
        """
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 3000),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        if model_name == "lgbm":
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)

        return params

    def _sample_neural_params(self, trial, model_name: str) -> dict[str, Any]:
        """Sample neural network hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.
        model_name : str
            Neural model name.

        Returns
        -------
        dict
            Sampled parameters.
        """
        return {
            "n_epochs": trial.suggest_int("n_epochs", 20, 200),
            "batch_size": trial.suggest_int("batch_size", 16, 256, log=True),
            "learning_rate": trial.suggest_float("nn_learning_rate", 1e-5, 1e-2, log=True),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True),
        }

    def _sample_linear_params(self, trial) -> dict[str, Any]:
        """Sample linear model hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.

        Returns
        -------
        dict
            Sampled parameters.
        """
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])
        params = {
            "penalty": penalty,
            "C": trial.suggest_float("C", 1e-4, 100.0, log=True),
        }

        if penalty == "elasticnet":
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)

        return params

    def _sample_svm_params(self, trial) -> dict[str, Any]:
        """Sample SVM hyperparameters.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.

        Returns
        -------
        dict
            Sampled parameters.
        """
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear", "poly"])
        params = {
            "kernel": kernel,
            "C": trial.suggest_float("svm_C", 1e-3, 100.0, log=True),
        }

        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1.0, log=True)
        elif kernel == "poly":
            params["degree"] = trial.suggest_int("degree", 2, 5)

        return params

    def _sample_preprocessing(self, trial) -> list[tuple[str, dict[str, Any]]]:
        """Sample preprocessing steps.

        Parameters
        ----------
        trial : optuna.Trial
            Optuna trial.

        Returns
        -------
        list of tuple
            Preprocessing steps.
        """
        steps = []

        # Imputation strategy
        imputer_strategy = trial.suggest_categorical(
            "imputer_strategy",
            ["none", "mean", "median", "most_frequent"]
        )
        if imputer_strategy != "none":
            steps.append(("imputer", {"strategy": imputer_strategy}))

        # Encoding method
        encoder_method = trial.suggest_categorical(
            "encoder_method",
            ["none", "onehot", "target", "ordinal"]
        )
        if encoder_method != "none":
            steps.append(("encoder", {"method": encoder_method}))

        # Scaling method
        scaler_method = trial.suggest_categorical(
            "scaler_method",
            ["none", "standard", "minmax", "robust"]
        )
        if scaler_method != "none":
            steps.append(("scaler", {"method": scaler_method}))

        return steps

    def update(self, result: SearchResult) -> None:
        """Update strategy with evaluation result.

        Parameters
        ----------
        result : SearchResult
            Result from evaluating a configuration.
        """
        super().update(result)

        # Find and complete the pending trial
        config_id = result.config.config_id
        if config_id in self._pending_trials:
            trial = self._pending_trials.pop(config_id)

            # Report result
            if result.success:
                self._study.tell(trial, result.score)
            else:
                # Mark as failed
                self._study.tell(trial, float("-inf"))

            if self.verbose > 1:
                logger.debug(
                    f"Completed trial {trial.number}: "
                    f"{result.config.model_name} -> {result.score:.4f}"
                )

    def get_best_params(self) -> dict[str, Any]:
        """Get the best hyperparameters found.

        Returns
        -------
        dict
            Best parameters.
        """
        if self._study is None or len(self._study.trials) == 0:
            return {}

        return self._study.best_params

    def get_optimization_history(self) -> list[dict[str, Any]]:
        """Get the optimization history.

        Returns
        -------
        list of dict
            Trial history.
        """
        if self._study is None:
            return []

        history = []
        for trial in self._study.trials:
            history.append({
                "number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": trial.state.name,
            })

        return history

    def get_param_importances(self) -> dict[str, float]:
        """Get hyperparameter importances.

        Returns
        -------
        dict
            Parameter importance scores.
        """
        if self._study is None or len(self._study.trials) < 10:
            return {}

        optuna = self._import_optuna()

        try:
            importances = optuna.importance.get_param_importances(self._study)
            return dict(importances)
        except Exception as e:
            logger.warning(f"Could not compute param importances: {e}")
            return {}

    def should_stop(self, max_iterations: int | None = None) -> bool:
        """Check if optimization should stop.

        Parameters
        ----------
        max_iterations : int, optional
            Maximum iterations.

        Returns
        -------
        bool
            Whether to stop.
        """
        if max_iterations and self.n_evaluated_ >= max_iterations:
            return True
        if self._study and len(self._study.trials) >= self.n_trials:
            return True
        return False
