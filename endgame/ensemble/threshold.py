"""Threshold optimization for classification."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from endgame.core.base import EndgameEstimator


class ThresholdOptimizer(EndgameEstimator):
    """Optimizes classification thresholds for target metrics.

    Standard 0.5 threshold is often suboptimal. This optimizer
    finds per-class thresholds that maximize the target metric.

    Parameters
    ----------
    metric : str or callable, default='f1'
        Metric to optimize: 'f1', 'f1_macro', 'f1_weighted',
        'accuracy', 'balanced_accuracy', or custom callable.
    search_method : str, default='grid'
        Search method: 'grid', 'optuna', 'hill_climb'.
    n_thresholds : int, default=100
        Number of thresholds to search (for grid search).
    threshold_range : Tuple[float, float], default=(0.1, 0.9)
        Range of thresholds to search.
    multiclass : bool, default=False
        Whether to optimize per-class thresholds.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    threshold_ : float or Dict[int, float]
        Optimized threshold(s).
    best_score_ : float
        Best score achieved.

    Examples
    --------
    >>> optimizer = ThresholdOptimizer(metric='f1')
    >>> optimizer.fit(y_true, y_proba)
    >>> print(f"Optimal threshold: {optimizer.threshold_}")
    >>> y_pred = optimizer.predict(y_proba)
    """

    def __init__(
        self,
        metric: str | Callable = "f1",
        search_method: str = "grid",
        n_thresholds: int = 100,
        threshold_range: tuple[float, float] = (0.1, 0.9),
        multiclass: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.metric = metric
        self.search_method = search_method
        self.n_thresholds = n_thresholds
        self.threshold_range = threshold_range
        self.multiclass = multiclass

        self.threshold_: float | dict[int, float] = 0.5
        self.best_score_: float = 0.0
        self._n_classes: int = 2

    def _get_metric_func(self, average: str | None = None) -> Callable:
        """Get the metric function."""
        if callable(self.metric):
            return self.metric

        metric_map = {
            "f1": lambda y, p: f1_score(y, p, average="binary" if self._n_classes == 2 else "macro"),
            "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
            "f1_weighted": lambda y, p: f1_score(y, p, average="weighted"),
            "f1_micro": lambda y, p: f1_score(y, p, average="micro"),
            "accuracy": accuracy_score,
            "balanced_accuracy": balanced_accuracy_score,
        }

        if self.metric.lower() not in metric_map:
            raise ValueError(f"Unknown metric: {self.metric}")

        return metric_map[self.metric.lower()]

    def _grid_search_binary(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[float, float]:
        """Grid search for binary classification threshold."""
        metric_func = self._get_metric_func()
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.n_thresholds,
        )

        best_threshold = 0.5
        best_score = -np.inf

        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)

            try:
                score = metric_func(y_true, y_pred)
            except Exception:
                continue

            if score > best_score:
                best_score = score
                best_threshold = threshold

        return best_threshold, best_score

    def _grid_search_multiclass(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[dict[int, float], float]:
        """Grid search for multiclass per-class thresholds."""
        n_classes = y_proba.shape[1]
        thresholds = np.linspace(
            self.threshold_range[0],
            self.threshold_range[1],
            self.n_thresholds,
        )

        # Start with equal thresholds
        best_thresholds = {i: 0.5 for i in range(n_classes)}
        metric_func = self._get_metric_func()

        # One-vs-rest threshold optimization
        for class_idx in range(n_classes):
            best_class_threshold = 0.5
            best_class_score = -np.inf

            for threshold in thresholds:
                # Adjust threshold for this class
                test_thresholds = best_thresholds.copy()
                test_thresholds[class_idx] = threshold

                # Apply thresholds
                y_pred = self._apply_multiclass_thresholds(y_proba, test_thresholds)

                try:
                    score = metric_func(y_true, y_pred)
                except Exception:
                    continue

                if score > best_class_score:
                    best_class_score = score
                    best_class_threshold = threshold

            best_thresholds[class_idx] = best_class_threshold

        # Final score
        y_pred = self._apply_multiclass_thresholds(y_proba, best_thresholds)
        final_score = metric_func(y_true, y_pred)

        return best_thresholds, final_score

    def _apply_multiclass_thresholds(
        self,
        y_proba: np.ndarray,
        thresholds: dict[int, float],
    ) -> np.ndarray:
        """Apply per-class thresholds and return predictions."""
        # Scale probabilities by threshold
        scaled_proba = y_proba.copy()
        for class_idx, threshold in thresholds.items():
            # Adjust probability relative to threshold
            scaled_proba[:, class_idx] = y_proba[:, class_idx] / (threshold + 1e-10)

        return np.argmax(scaled_proba, axis=1)

    def _hill_climb_binary(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> tuple[float, float]:
        """Hill climbing for binary threshold."""
        metric_func = self._get_metric_func()

        # Start at 0.5
        current_threshold = 0.5
        step_size = 0.1
        min_step = 0.001

        y_pred = (y_proba >= current_threshold).astype(int)
        current_score = metric_func(y_true, y_pred)

        while step_size >= min_step:
            improved = False

            for delta in [-step_size, step_size]:
                new_threshold = np.clip(
                    current_threshold + delta,
                    self.threshold_range[0],
                    self.threshold_range[1],
                )

                y_pred = (y_proba >= new_threshold).astype(int)

                try:
                    score = metric_func(y_true, y_pred)
                except Exception:
                    continue

                if score > current_score:
                    current_score = score
                    current_threshold = new_threshold
                    improved = True
                    break

            if not improved:
                step_size /= 2

        return current_threshold, current_score

    def fit(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
    ) -> "ThresholdOptimizer":
        """Find optimal threshold(s).

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_proba : array-like
            Predicted probabilities.
            Shape (n_samples,) for binary, (n_samples, n_classes) for multiclass.

        Returns
        -------
        self
        """
        y_true = np.asarray(y_true)
        y_proba = np.asarray(y_proba)

        # Determine binary vs multiclass
        if y_proba.ndim == 1:
            is_binary = True
            self._n_classes = 2
        elif y_proba.shape[1] == 2:
            is_binary = True
            self._n_classes = 2
            y_proba = y_proba[:, 1]  # Use positive class probability
        else:
            is_binary = not self.multiclass
            self._n_classes = y_proba.shape[1]

        if is_binary or not self.multiclass:
            # Binary threshold optimization
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]

            if self.search_method == "grid":
                self.threshold_, self.best_score_ = self._grid_search_binary(y_true, y_proba)
            elif self.search_method == "hill_climb":
                self.threshold_, self.best_score_ = self._hill_climb_binary(y_true, y_proba)
            else:
                raise ValueError(f"Unknown search method: {self.search_method}")
        else:
            # Multiclass per-class thresholds
            self.threshold_, self.best_score_ = self._grid_search_multiclass(y_true, y_proba)

        self._log(f"Optimal threshold: {self.threshold_}")
        self._log(f"Best score: {self.best_score_:.4f}")

        self._is_fitted = True
        return self

    def predict(self, y_proba: np.ndarray) -> np.ndarray:
        """Apply optimized threshold(s) to predictions.

        Parameters
        ----------
        y_proba : array-like
            Predicted probabilities.

        Returns
        -------
        ndarray
            Predicted labels.
        """
        self._check_is_fitted()
        y_proba = np.asarray(y_proba)

        if isinstance(self.threshold_, dict):
            # Multiclass
            return self._apply_multiclass_thresholds(y_proba, self.threshold_)
        else:
            # Binary
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            return (y_proba >= self.threshold_).astype(int)

    def transform(self, y_proba: np.ndarray) -> np.ndarray:
        """Alias for predict()."""
        return self.predict(y_proba)
