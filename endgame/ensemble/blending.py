from __future__ import annotations

"""Blending methods for ensemble combination."""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

from endgame.core.base import BaseEnsemble


class BlendingEnsemble(BaseEnsemble, ClassifierMixin):
    """Blending Ensemble using hold-out set for meta-learner training.

    Unlike stacking which uses cross-validation, blending uses a hold-out
    portion of the training data to generate meta-features for the
    second-level learner.

    Parameters
    ----------
    base_estimators : List[estimator]
        Level 1 models.
    meta_estimator : estimator, optional
        Level 2 model. Default: LogisticRegression for classification.
    blend_fraction : float, default=0.2
        Fraction of training data to use for blending (meta-learner training).
    use_proba : bool, default=True
        Use predict_proba for classification (if available).
    passthrough : bool, default=False
        Whether to include original features in Level 2.
    cv : int, optional
        Ignored. For API compatibility with StackingEnsemble.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimators_ : List[estimator]
        Fitted Level 1 models.
    meta_estimator_ : estimator
        Fitted Level 2 model.
    classes_ : ndarray
        Unique class labels (for classification).

    Examples
    --------
    >>> from endgame.ensemble import BlendingEnsemble
    >>> base_models = [RandomForestClassifier(), GradientBoostingClassifier()]
    >>> blender = BlendingEnsemble(base_estimators=base_models)
    >>> blender.fit(X_train, y_train)
    >>> predictions = blender.predict(X_test)
    """

    def __init__(
        self,
        base_estimators: list[BaseEstimator] | None = None,
        meta_estimator: BaseEstimator | None = None,
        blend_fraction: float = 0.2,
        use_proba: bool = True,
        passthrough: bool = False,
        cv: int | None = None,  # Ignored, for compatibility
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            estimators=base_estimators,
            random_state=random_state,
            verbose=verbose,
        )
        self.base_estimators = base_estimators or []
        self.meta_estimator = meta_estimator
        self.blend_fraction = blend_fraction
        self.use_proba = use_proba
        self.passthrough = passthrough
        self.cv = cv  # Ignored

        self.base_estimators_: list[BaseEstimator] = []
        self.meta_estimator_: BaseEstimator | None = None
        self._is_classifier: bool = True
        self._n_features_in: int = 0

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray | None = None,
        **fit_params,
    ) -> BlendingEnsemble:
        """Fit the blending ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self._n_features_in = X.shape[1]
        self._is_classifier = len(np.unique(y)) <= 20

        if self._is_classifier:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        # Split into train and blend sets
        if sample_weight is not None:
            X_train, X_blend, y_train, y_blend, w_train, w_blend = train_test_split(
                X, y, sample_weight,
                test_size=self.blend_fraction,
                random_state=self.random_state,
                stratify=y if self._is_classifier else None,
            )
        else:
            X_train, X_blend, y_train, y_blend = train_test_split(
                X, y,
                test_size=self.blend_fraction,
                random_state=self.random_state,
                stratify=y if self._is_classifier else None,
            )
            w_train = w_blend = None

        self._log(f"Training {len(self.base_estimators)} base estimators...")
        self._log(f"Train set: {len(X_train)}, Blend set: {len(X_blend)}")

        # Fit base estimators on training portion
        self.base_estimators_ = []
        blend_predictions = []

        for i, estimator in enumerate(self.base_estimators):
            self._log(f"  Fitting base estimator {i + 1}/{len(self.base_estimators)}")
            fitted = clone(estimator)

            if w_train is not None:
                fitted.fit(X_train, y_train, sample_weight=w_train)
            else:
                fitted.fit(X_train, y_train)

            self.base_estimators_.append(fitted)

            # Get predictions on blend set
            if self._is_classifier and self.use_proba and hasattr(fitted, "predict_proba"):
                pred = fitted.predict_proba(X_blend)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1:2]
            else:
                pred = fitted.predict(X_blend)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

            blend_predictions.append(pred)

        # Stack blend predictions
        meta_features = np.hstack(blend_predictions)

        if self.passthrough:
            meta_features = np.hstack([meta_features, X_blend])

        # Fit meta-estimator
        self._log("Fitting meta-estimator...")
        if self.meta_estimator is None:
            if self._is_classifier:
                self.meta_estimator_ = LogisticRegression(
                    C=1.0, max_iter=1000, random_state=self.random_state
                )
            else:
                self.meta_estimator_ = Ridge(alpha=1.0, random_state=self.random_state)
        else:
            self.meta_estimator_ = clone(self.meta_estimator)

        if w_blend is not None:
            self.meta_estimator_.fit(meta_features, y_blend, sample_weight=w_blend)
        else:
            self.meta_estimator_.fit(meta_features, y_blend)

        # Refit base estimators on full data for better test predictions
        self._log("Refitting base estimators on full data...")
        self.base_estimators_ = []
        for i, estimator in enumerate(self.base_estimators):
            fitted = clone(estimator)
            if sample_weight is not None:
                fitted.fit(X, y, sample_weight=sample_weight)
            else:
                fitted.fit(X, y)
            self.base_estimators_.append(fitted)

        self._is_fitted = True
        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Get meta-features from base estimator predictions."""
        predictions = []

        for fitted in self.base_estimators_:
            if self._is_classifier and self.use_proba and hasattr(fitted, "predict_proba"):
                pred = fitted.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1:2]
            else:
                pred = fitted.predict(X)
                if pred.ndim == 1:
                    pred = pred.reshape(-1, 1)

            predictions.append(pred)

        meta_features = np.hstack(predictions)

        if self.passthrough:
            meta_features = np.hstack([meta_features, X])

        return meta_features

    def predict(self, X) -> np.ndarray:
        """Predict using the blending ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()
        X = np.asarray(X)
        meta_features = self._get_meta_features(X)
        return self.meta_estimator_.predict(meta_features)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()

        if not self._is_classifier:
            raise ValueError("predict_proba only available for classification")

        X = np.asarray(X)
        meta_features = self._get_meta_features(X)

        if hasattr(self.meta_estimator_, "predict_proba"):
            return self.meta_estimator_.predict_proba(meta_features)

        if hasattr(self.meta_estimator_, "decision_function"):
            decision = self.meta_estimator_.decision_function(meta_features)
            proba = 1 / (1 + np.exp(-decision))
            if proba.ndim == 1:
                return np.vstack([1 - proba, proba]).T
            return proba

        raise ValueError("Meta-estimator doesn't support probability predictions")

    def score(self, X, y, sample_weight=None) -> float:
        """Return accuracy score on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Accuracy score.
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)


class OptimizedBlender(BaseEnsemble):
    """Optuna-powered blend weight optimization.

    Uses Bayesian optimization to find optimal weights for
    combining model predictions.

    Parameters
    ----------
    metric : str or callable
        Metric to optimize: 'roc_auc', 'rmse', 'mae', etc.
    n_trials : int, default=100
        Number of optimization trials.
    weight_bounds : Tuple[float, float], default=(0, 1)
        Bounds for individual model weights.
    normalize : bool, default=True
        Whether weights must sum to 1.
    maximize : bool, default=True
        Whether to maximize or minimize the metric.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    weights_ : Dict[int, float]
        Optimized model weights.
    best_score_ : float
        Best score achieved.
    study_ : optuna.Study
        Optuna study object for further analysis.

    Examples
    --------
    >>> blender = OptimizedBlender(metric='roc_auc', n_trials=100)
    >>> blender.fit(oof_predictions, y_train)
    >>> final_pred = blender.predict(test_predictions)
    """

    def __init__(
        self,
        metric: str | Callable = "roc_auc",
        n_trials: int = 100,
        weight_bounds: tuple[float, float] = (0, 1),
        normalize: bool = True,
        maximize: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.metric = metric
        self.n_trials = n_trials
        self.weight_bounds = weight_bounds
        self.normalize = normalize
        self.maximize = maximize

        self.best_score_: float = 0.0
        self.study_: Any | None = None

    def _get_metric_func(self) -> Callable:
        """Get the metric function."""
        if callable(self.metric):
            return self.metric

        from sklearn.metrics import (
            log_loss,
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            roc_auc_score,
        )

        metric_map = {
            "roc_auc": roc_auc_score,
            "log_loss": log_loss,
            "rmse": lambda y, p: np.sqrt(mean_squared_error(y, p)),
            "mse": mean_squared_error,
            "mae": mean_absolute_error,
            "r2": r2_score,
        }

        if self.metric.lower() not in metric_map:
            raise ValueError(f"Unknown metric: {self.metric}")

        return metric_map[self.metric.lower()]

    def fit(
        self,
        predictions: list[np.ndarray],
        y_true: np.ndarray,
    ) -> OptimizedBlender:
        """Optimize blend weights using Optuna.

        Parameters
        ----------
        predictions : List of arrays
            Out-of-fold predictions from each model.
        y_true : array-like
            True target values.

        Returns
        -------
        self
        """
        try:
            import optuna
        except ImportError:
            raise ImportError(
                "Optuna is required for OptimizedBlender. "
                "Install with: pip install optuna"
            )

        predictions = self._validate_predictions(predictions, y_true)
        y_true = np.asarray(y_true)
        n_models = len(predictions)

        metric_func = self._get_metric_func()

        # Determine optimization direction
        direction = "maximize" if self.maximize else "minimize"

        # Suppress Optuna logging if not verbose
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial: optuna.Trial) -> float:
            # Sample weights
            weights = []
            for i in range(n_models):
                w = trial.suggest_float(
                    f"weight_{i}",
                    self.weight_bounds[0],
                    self.weight_bounds[1],
                )
                weights.append(w)

            # Normalize if requested
            if self.normalize:
                total = sum(weights)
                if total > 0:
                    weights = [w / total for w in weights]
                else:
                    weights = [1.0 / n_models] * n_models

            # Compute blended prediction
            blended = np.zeros_like(predictions[0], dtype=np.float64)
            for w, pred in zip(weights, predictions):
                blended += w * pred

            # Evaluate
            try:
                score = metric_func(y_true, blended)
            except Exception:
                return float("-inf") if self.maximize else float("inf")

            return score

        # Create and run study
        self.study_ = optuna.create_study(
            direction=direction,
            sampler=optuna.samplers.TPESampler(seed=self.random_state),
        )

        self.study_.optimize(
            objective,
            n_trials=self.n_trials,
            show_progress_bar=self.verbose,
        )

        # Extract best weights
        best_params = self.study_.best_params
        weights = [best_params[f"weight_{i}"] for i in range(n_models)]

        if self.normalize:
            total = sum(weights)
            if total > 0:
                weights = [w / total for w in weights]

        self.weights_ = {i: w for i, w in enumerate(weights)}
        self.best_score_ = self.study_.best_value

        self._log(f"Best score: {self.best_score_:.4f}")
        self._log(f"Best weights: {self.weights_}")

        self._is_fitted = True
        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply optimized weights.

        Parameters
        ----------
        predictions : List of arrays
            Predictions from each model.

        Returns
        -------
        ndarray
            Blended prediction.
        """
        self._check_is_fitted()
        predictions = self._validate_predictions(predictions)
        return self._weighted_average(predictions, self.weights_)


class RankAverageBlender(BaseEnsemble):
    """Rank-based blending for submissions.

    Converts predictions to ranks before averaging.
    Robust to different prediction scales across models.

    Parameters
    ----------
    method : str, default='average'
        Rank method: 'average', 'min', 'max', 'dense', 'ordinal'.
    normalize : bool, default=True
        Whether to normalize ranks to [0, 1].
    weights : Dict[int, float], optional
        Optional model weights. If None, uniform weights.

    Examples
    --------
    >>> blender = RankAverageBlender()
    >>> final_pred = blender.blend(test_predictions)
    """

    def __init__(
        self,
        method: str = "average",
        normalize: bool = True,
        weights: dict[int, float] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.method = method
        self.normalize = normalize
        self.weights = weights

    def fit(
        self,
        predictions: list[np.ndarray] | None = None,
        y_true: np.ndarray | None = None,
    ) -> RankAverageBlender:
        """Fit the blender (stores weights if provided).

        Parameters
        ----------
        predictions : ignored
        y_true : ignored

        Returns
        -------
        self
        """
        if self.weights is not None:
            self.weights_ = self.weights.copy()
        else:
            self.weights_ = None

        self._is_fitted = True
        return self

    def blend(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Blend predictions using rank averaging.

        Parameters
        ----------
        predictions : List of arrays
            Predictions from each model.

        Returns
        -------
        ndarray
            Rank-averaged prediction.
        """
        predictions = self._validate_predictions(predictions)
        n_models = len(predictions)

        # Convert to ranks
        ranked_predictions = []
        for pred in predictions:
            ranks = stats.rankdata(pred, method=self.method)
            if self.normalize:
                ranks = ranks / len(ranks)
            ranked_predictions.append(ranks)

        # Weighted average of ranks
        if self.weights_ is None:
            weights = {i: 1.0 / n_models for i in range(n_models)}
        else:
            weights = self.weights_

        return self._weighted_average(ranked_predictions, weights)

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Alias for blend()."""
        return self.blend(predictions)


class PowerBlender(BaseEnsemble):
    """Power-weighted blending based on individual scores.

    Weights models by their validation scores raised to a power.
    Higher power = more weight to best models.

    Parameters
    ----------
    scores : List[float]
        Validation scores for each model.
    power : float, default=2.0
        Power to raise scores to (higher = more aggressive weighting).
    higher_is_better : bool, default=True
        Whether higher scores are better.

    Examples
    --------
    >>> scores = [0.85, 0.87, 0.86]
    >>> blender = PowerBlender(scores=scores, power=3.0)
    >>> final_pred = blender.predict(test_predictions)
    """

    def __init__(
        self,
        scores: list[float] | None = None,
        power: float = 2.0,
        higher_is_better: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.scores = scores
        self.power = power
        self.higher_is_better = higher_is_better

    def fit(
        self,
        predictions: list[np.ndarray] | None = None,
        y_true: np.ndarray | None = None,
        scores: list[float] | None = None,
    ) -> PowerBlender:
        """Compute power-weighted blending weights.

        Parameters
        ----------
        predictions : ignored
        y_true : ignored
        scores : List[float], optional
            Model scores (overrides constructor scores).

        Returns
        -------
        self
        """
        if scores is not None:
            self.scores = scores

        if self.scores is None:
            raise ValueError("scores must be provided")

        scores_arr = np.array(self.scores)

        # Invert if lower is better
        if not self.higher_is_better:
            scores_arr = 1.0 / (scores_arr + 1e-10)

        # Apply power
        weights = scores_arr ** self.power

        # Normalize
        weights = weights / weights.sum()

        self.weights_ = {i: w for i, w in enumerate(weights)}

        self._log(f"Power weights: {self.weights_}")

        self._is_fitted = True
        return self

    def predict(self, predictions: list[np.ndarray]) -> np.ndarray:
        """Apply power weights.

        Parameters
        ----------
        predictions : List of arrays
            Predictions from each model.

        Returns
        -------
        ndarray
            Power-weighted prediction.
        """
        self._check_is_fitted()
        predictions = self._validate_predictions(predictions)
        return self._weighted_average(predictions, self.weights_)
