from __future__ import annotations

"""Optuna-based hyperparameter optimization."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score

from endgame.core.base import EndgameEstimator
from endgame.core.types import OptimizationResult
from endgame.tune.spaces import get_space


class OptunaOptimizer(EndgameEstimator):
    """Automated hyperparameter optimization with Optuna.

    Provides Bayesian optimization of model hyperparameters with
    competition-specific search spaces and pruning.

    Parameters
    ----------
    estimator : endgame or sklearn estimator
        Model to optimize.
    param_space : Dict or str
        Parameter search space. If str, uses preset:
        - 'lgbm_standard', 'lgbm_large'
        - 'xgb_standard', 'xgb_large'
        - 'catboost_standard'
    metric : str or callable
        Optimization target: 'roc_auc', 'log_loss', 'rmse', etc.
    cv : int or CV splitter, default=5
        Cross-validation strategy.
    n_trials : int, default=100
        Number of optimization trials.
    timeout : int, optional
        Maximum optimization time in seconds.
    sampler : str, default='tpe'
        Optuna sampler: 'tpe', 'cmaes', 'random'.
    pruner : str, default='median'
        Early stopping: 'median', 'hyperband', 'none'.
    direction : str, default='maximize'
        Optimization direction: 'maximize' or 'minimize'.
    n_jobs : int, default=1
        Parallel jobs for cross-validation.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    best_params_ : Dict[str, Any]
        Best hyperparameters found.
    best_score_ : float
        Best validation score.
    study_ : optuna.Study
        Optuna study for further analysis.
    best_estimator_ : estimator
        Model fitted with best parameters.

    Examples
    --------
    >>> from endgame.tune import OptunaOptimizer
    >>> from endgame.models import LGBMWrapper
    >>> optimizer = OptunaOptimizer(
    ...     estimator=LGBMWrapper(),
    ...     param_space='lgbm_standard',
    ...     metric='roc_auc',
    ...     n_trials=100
    ... )
    >>> result = optimizer.optimize(X, y)
    >>> print(f"Best params: {result.best_params}")
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        param_space: dict | str | None = None,
        metric: str | Callable = "roc_auc",
        cv: int | Any = 5,
        n_trials: int = 100,
        timeout: int | None = None,
        sampler: str = "tpe",
        pruner: str = "median",
        direction: str = "maximize",
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.estimator = estimator
        self.param_space = param_space
        self.metric = metric
        self.cv = cv
        self.n_trials = n_trials
        self.timeout = timeout
        self.sampler = sampler
        self.pruner = pruner
        self.direction = direction
        self.n_jobs = n_jobs

        self.best_params_: dict[str, Any] = {}
        self.best_score_: float = 0.0
        self.study_: Any | None = None
        self.best_estimator_: BaseEstimator | None = None
        self._all_trials: list[dict] = []

    def _get_sampler(self):
        """Get Optuna sampler."""
        import optuna

        if self.sampler == "tpe":
            return optuna.samplers.TPESampler(seed=self.random_state)
        elif self.sampler == "cmaes":
            return optuna.samplers.CmaEsSampler(seed=self.random_state)
        elif self.sampler == "random":
            return optuna.samplers.RandomSampler(seed=self.random_state)
        else:
            raise ValueError(f"Unknown sampler: {self.sampler}")

    def _get_pruner(self):
        """Get Optuna pruner."""
        import optuna

        if self.pruner == "median":
            return optuna.pruners.MedianPruner()
        elif self.pruner == "hyperband":
            return optuna.pruners.HyperbandPruner()
        elif self.pruner == "none" or self.pruner is None:
            return optuna.pruners.NopPruner()
        else:
            raise ValueError(f"Unknown pruner: {self.pruner}")

    def _get_cv_splitter(self, y: np.ndarray) -> Any:
        """Get cross-validation splitter."""
        if isinstance(self.cv, int):
            if is_classifier(self.estimator):
                return StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=self.random_state,
                )
            return KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        return self.cv

    def _sample_params(self, trial) -> dict[str, Any]:
        """Sample hyperparameters from search space."""

        if isinstance(self.param_space, str):
            space = get_space(self.param_space)
        else:
            space = self.param_space

        params = {}
        for name, config in space.items():
            param_type = config["type"]

            if param_type == "int":
                params[name] = trial.suggest_int(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "float":
                params[name] = trial.suggest_float(
                    name,
                    config["low"],
                    config["high"],
                    log=config.get("log", False),
                )
            elif param_type == "categorical":
                params[name] = trial.suggest_categorical(
                    name,
                    config["choices"],
                )
            elif param_type == "loguniform":
                params[name] = trial.suggest_float(
                    name,
                    config["low"],
                    config["high"],
                    log=True,
                )
            else:
                raise ValueError(f"Unknown param type: {param_type}")

        return params

    def optimize(
        self,
        X,
        y,
        groups: np.ndarray | None = None,
        fit_params: dict | None = None,
    ) -> OptimizationResult:
        """Run hyperparameter optimization.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target values.
        groups : array-like, optional
            Group labels for group-aware CV.
        fit_params : dict, optional
            Additional parameters for estimator.fit().

        Returns
        -------
        OptimizationResult
            - best_params: Dict
            - best_score: float
            - study: optuna.Study
            - all_trials: List[Dict]
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for optimization. "
                "Install with: pip install optuna"
            )

        X = np.asarray(X)
        y = np.asarray(y)
        fit_params = fit_params or {}

        cv = self._get_cv_splitter(y)

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            # Sample hyperparameters
            params = self._sample_params(trial)

            # Create model with sampled params
            model = clone(self.estimator)
            model.set_params(**params)

            # Cross-validate
            try:
                scores = cross_val_score(
                    model,
                    X,
                    y,
                    cv=cv,
                    scoring=self.metric if isinstance(self.metric, str) else None,
                    n_jobs=self.n_jobs,
                    groups=groups,
                )
                score = scores.mean()
            except Exception as e:
                if self.verbose:
                    self._log(f"Trial failed: {e}", level="warn")
                raise optuna.TrialPruned()

            return score

        # Create study
        self.study_ = optuna.create_study(
            direction=self.direction,
            sampler=self._get_sampler(),
            pruner=self._get_pruner(),
        )

        # Run optimization
        self._log(f"Starting optimization with {self.n_trials} trials...")

        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            timeout=self.timeout,
            show_progress_bar=self.verbose,
        )

        # Extract results
        self.best_params_ = self.study_.best_params
        self.best_score_ = self.study_.best_value

        # Collect all trials
        self._all_trials = [
            {
                "number": t.number,
                "params": t.params,
                "value": t.value,
                "state": str(t.state),
            }
            for t in self.study_.trials
        ]

        self._log(f"Best score: {self.best_score_:.4f}")
        self._log(f"Best params: {self.best_params_}")

        # Fit best model on full data
        self.best_estimator_ = clone(self.estimator)
        self.best_estimator_.set_params(**self.best_params_)
        self.best_estimator_.fit(X, y, **fit_params)

        self._is_fitted = True

        return OptimizationResult(
            best_params=self.best_params_,
            best_score=self.best_score_,
            study=self.study_,
            all_trials=self._all_trials,
            n_trials=len(self._all_trials),
        )

    def get_param_importances(self) -> dict[str, float]:
        """Get hyperparameter importances.

        Returns
        -------
        Dict[str, float]
            Importance score for each hyperparameter.
        """
        if self.study_ is None:
            raise RuntimeError("Must run optimize() first")

        try:
            import optuna
            importances = optuna.importance.get_param_importances(self.study_)
            return dict(importances)
        except Exception:
            return {}

    def plot_optimization_history(self):
        """Plot optimization history."""
        if self.study_ is None:
            raise RuntimeError("Must run optimize() first")

        import optuna
        return optuna.visualization.plot_optimization_history(self.study_)

    def plot_param_importances(self):
        """Plot hyperparameter importances."""
        if self.study_ is None:
            raise RuntimeError("Must run optimize() first")

        import optuna
        return optuna.visualization.plot_param_importances(self.study_)
