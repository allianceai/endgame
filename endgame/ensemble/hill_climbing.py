from __future__ import annotations

"""Hill Climbing Ensemble: Forward selection with replacement."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)

from endgame.core.base import BaseEnsemble
from endgame.core.types import EnsembleResult


class HillClimbingEnsemble(BaseEnsemble):
    """Forward ensemble selection with replacement.

    Iteratively adds models that maximize validation metric.
    Key technique for non-differentiable metrics (F1, MAP@K).

    Algorithm:
    1. Start with empty ensemble
    2. For each iteration:
       a. For each model in pool:
          - Compute metric if added to current ensemble
       b. Add model that provides best improvement
    3. Allow repeating models (weighted averaging)

    Parameters
    ----------
    metric : str or callable, default='roc_auc'
        Metric to optimize: 'roc_auc', 'log_loss', 'f1', 'accuracy',
        'rmse', 'mae', 'r2', or custom callable(y_true, y_pred).
    n_iterations : int, default=100
        Number of hill climbing iterations.
    early_stopping : int, default=20
        Stop if no improvement for this many iterations.
    maximize : bool, default=True
        Whether to maximize or minimize the metric.
    init_weights : str, default='best_single'
        Initial weight strategy: 'best_single', 'uniform', 'none'.
    random_state : int, optional
        Random seed for tie-breaking.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    weights_ : Dict[int, float]
        Optimized model weights (by model index).
    best_score_ : float
        Best ensemble score achieved.
    selection_history_ : List[int]
        Order in which models were selected.

    Examples
    --------
    >>> from endgame.ensemble import HillClimbingEnsemble
    >>> ensemble = HillClimbingEnsemble(metric='roc_auc', n_iterations=100)
    >>> ensemble.fit(oof_predictions, y_train)
    >>> print(f"Weights: {ensemble.weights_}")
    >>> test_pred = ensemble.predict(test_predictions)
    """

    def __init__(
        self,
        metric: str | Callable = "roc_auc",
        n_iterations: int = 100,
        early_stopping: int = 20,
        maximize: bool = True,
        init_weights: str = "best_single",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.metric = metric
        self.n_iterations = n_iterations
        self.early_stopping = early_stopping
        self.maximize = maximize
        self.init_weights = init_weights

        self.weights_: dict[int, float] = {}
        self.best_score_: float = 0.0
        self.selection_history_: list[int] = []
        self._metric_func: Callable | None = None

    def _get_metric_func(self) -> Callable:
        """Get the metric function."""
        if callable(self.metric):
            return self.metric

        metric_map = {
            "roc_auc": lambda y, p: roc_auc_score(y, p),
            "auc": lambda y, p: roc_auc_score(y, p),
            "log_loss": lambda y, p: -log_loss(y, p),  # Negate for maximization
            "logloss": lambda y, p: -log_loss(y, p),
            "f1": lambda y, p: f1_score(y, (p >= 0.5).astype(int)),
            "f1_macro": lambda y, p: f1_score(y, (p >= 0.5).astype(int), average="macro"),
            "accuracy": lambda y, p: accuracy_score(y, (p >= 0.5).astype(int)),
            "rmse": lambda y, p: -np.sqrt(mean_squared_error(y, p)),  # Negate
            "mse": lambda y, p: -mean_squared_error(y, p),
            "mae": lambda y, p: -mean_absolute_error(y, p),
            "r2": lambda y, p: r2_score(y, p),
        }

        if self.metric.lower() not in metric_map:
            raise ValueError(
                f"Unknown metric '{self.metric}'. "
                f"Available: {list(metric_map.keys())}"
            )

        return metric_map[self.metric.lower()]

    def _compute_ensemble_prediction(
        self,
        predictions: list[np.ndarray],
        weights: dict[int, float],
    ) -> np.ndarray:
        """Compute weighted ensemble prediction."""
        if not weights:
            return np.zeros_like(predictions[0])

        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.zeros_like(predictions[0])

        result = np.zeros_like(predictions[0], dtype=np.float64)
        for idx, weight in weights.items():
            result += weight * predictions[idx]
        result /= total_weight

        return result

    def fit(
        self,
        predictions: list[np.ndarray],
        y_true: np.ndarray,
    ) -> HillClimbingEnsemble:
        """Find optimal ensemble weights via hill climbing.

        Parameters
        ----------
        predictions : List of shape (n_models, n_samples, ...)
            Out-of-fold predictions from each model.
        y_true : array-like
            True target values.

        Returns
        -------
        self
        """
        predictions = self._validate_predictions(predictions, y_true)
        y_true = np.asarray(y_true)
        n_models = len(predictions)

        self._metric_func = self._get_metric_func()

        # Evaluate individual model scores
        individual_scores = []
        for i, pred in enumerate(predictions):
            try:
                score = self._metric_func(y_true, pred)
            except Exception:
                score = -np.inf if self.maximize else np.inf
            individual_scores.append(score)

        self._log(f"Individual model scores: {[f'{s:.4f}' for s in individual_scores]}")

        # Initialize weights
        if self.init_weights == "best_single":
            best_idx = np.argmax(individual_scores) if self.maximize else np.argmin(individual_scores)
            self.weights_ = {best_idx: 1.0}
            current_score = individual_scores[best_idx]
        elif self.init_weights == "uniform":
            self.weights_ = {i: 1.0 / n_models for i in range(n_models)}
            current_pred = self._compute_ensemble_prediction(predictions, self.weights_)
            current_score = self._metric_func(y_true, current_pred)
        else:
            self.weights_ = {}
            current_score = -np.inf if self.maximize else np.inf

        self.best_score_ = current_score
        best_weights = self.weights_.copy()
        self.selection_history_ = list(self.weights_.keys())

        self._log(f"Initial score: {current_score:.4f}")

        # Hill climbing iterations
        no_improvement_count = 0
        rng = np.random.RandomState(self.random_state)

        for iteration in range(self.n_iterations):
            best_candidate_idx = -1
            best_candidate_score = current_score

            # Try adding each model
            for model_idx in range(n_models):
                # Create candidate weights
                candidate_weights = self.weights_.copy()
                candidate_weights[model_idx] = candidate_weights.get(model_idx, 0) + 1.0

                # Compute ensemble prediction
                candidate_pred = self._compute_ensemble_prediction(predictions, candidate_weights)

                # Evaluate
                try:
                    candidate_score = self._metric_func(y_true, candidate_pred)
                except Exception:
                    continue

                # Check if better
                is_better = (
                    (self.maximize and candidate_score > best_candidate_score) or
                    (not self.maximize and candidate_score < best_candidate_score)
                )

                if is_better:
                    best_candidate_idx = model_idx
                    best_candidate_score = candidate_score

            # Update if improvement found
            if best_candidate_idx >= 0:
                is_improvement = (
                    (self.maximize and best_candidate_score > current_score) or
                    (not self.maximize and best_candidate_score < current_score)
                )

                if is_improvement:
                    self.weights_[best_candidate_idx] = self.weights_.get(best_candidate_idx, 0) + 1.0
                    current_score = best_candidate_score
                    self.selection_history_.append(best_candidate_idx)
                    no_improvement_count = 0

                    if self.verbose and (iteration + 1) % 10 == 0:
                        self._log(f"Iteration {iteration + 1}: score={current_score:.4f}")

                    # Update best
                    is_new_best = (
                        (self.maximize and current_score > self.best_score_) or
                        (not self.maximize and current_score < self.best_score_)
                    )
                    if is_new_best:
                        self.best_score_ = current_score
                        best_weights = self.weights_.copy()
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1

            # Early stopping
            if no_improvement_count >= self.early_stopping:
                self._log(f"Early stopping at iteration {iteration + 1}")
                break

        # Use best weights found
        self.weights_ = best_weights

        # Normalize weights
        total = sum(self.weights_.values())
        if total > 0:
            self.weights_ = {k: v / total for k, v in self.weights_.items()}

        self._log(f"Final score: {self.best_score_:.4f}")
        self._log(f"Final weights: {self.weights_}")

        self._is_fitted = True
        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply learned weights to generate ensemble prediction.

        Parameters
        ----------
        predictions : List of shape (n_models, n_samples, ...)
            Predictions from each model.

        Returns
        -------
        ndarray
            Weighted ensemble prediction.
        """
        self._check_is_fitted()
        predictions = self._validate_predictions(predictions)
        return self._compute_ensemble_prediction(predictions, self.weights_)

    def get_result(self) -> EnsembleResult:
        """Get ensemble result summary.

        Returns
        -------
        EnsembleResult
            Result containing weights, score, and selected models.
        """
        self._check_is_fitted()

        # Compute improvement over best single model
        # (Would need to store individual scores for this)
        improvement = 0.0

        return EnsembleResult(
            weights=self.weights_,
            best_score=self.best_score_,
            improvement=improvement,
            selected_models=list(self.weights_.keys()),
        )
