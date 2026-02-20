"""Nested Cross-Validation for unbiased model evaluation.

Provides proper nested CV where the inner loop handles model selection
or hyperparameter tuning, and the outer loop provides unbiased performance
estimates.

Example
-------
>>> from endgame.validation import NestedCV
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.model_selection import GridSearchCV
>>>
>>> ncv = NestedCV(
...     estimator=RandomForestClassifier(),
...     search=GridSearchCV(
...         RandomForestClassifier(),
...         param_grid={'n_estimators': [50, 100], 'max_depth': [3, 5]},
...         cv=3, scoring='accuracy'
...     ),
...     outer_cv=5,
...     scoring='accuracy'
... )
>>> results = ncv.evaluate(X, y)
>>> print(f"Score: {results.mean_score:.4f} +/- {results.std_score:.4f}")
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import clone, is_classifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    get_scorer,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)


@dataclass
class NestedCVResult:
    """Results from nested cross-validation.

    Attributes
    ----------
    outer_scores : list of float
        Score for each outer fold.
    mean_score : float
        Mean of outer fold scores.
    std_score : float
        Standard deviation of outer fold scores.
    best_params : list of dict
        Best parameters found in each outer fold's inner search.
    oof_predictions : ndarray or None
        Out-of-fold predictions (if return_oof=True).
    inner_scores : list of float
        Best inner CV score for each outer fold.
    scoring : str
        Metric name used.
    """
    outer_scores: list[float] = field(default_factory=list)
    mean_score: float = 0.0
    std_score: float = 0.0
    best_params: list[dict[str, Any]] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    inner_scores: list[float] = field(default_factory=list)
    scoring: str = "accuracy"

    def __repr__(self) -> str:
        return (
            f"NestedCVResult(score={self.mean_score:.4f} +/- {self.std_score:.4f}, "
            f"n_folds={len(self.outer_scores)}, metric='{self.scoring}')"
        )


_METRIC_MAP = {
    "accuracy": accuracy_score,
    "balanced_accuracy": balanced_accuracy_score,
    "f1": lambda y, p: f1_score(y, p, average="weighted"),
    "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
    "r2": r2_score,
    "mse": mean_squared_error,
    "neg_mean_squared_error": lambda y, p: -mean_squared_error(y, p),
    "mae": mean_absolute_error,
    "neg_mean_absolute_error": lambda y, p: -mean_absolute_error(y, p),
}


class NestedCV:
    """Nested cross-validation for unbiased model evaluation.

    The inner loop performs model selection (hyperparameter tuning or
    algorithm comparison) and the outer loop estimates generalization
    performance using the best model from each inner fold.

    Parameters
    ----------
    estimator : estimator or None
        Base estimator to evaluate. If `search` is provided, this is
        ignored (the search object contains the estimator).
    search : estimator with fit/predict or None
        A search object (e.g., GridSearchCV, RandomizedSearchCV,
        OptunaOptimizer) that performs inner-loop model selection.
        Must have `best_estimator_` and `best_params_` after fitting.
        If None, `estimator` is used directly without inner tuning.
    outer_cv : int or CV splitter, default=5
        Number of outer folds or a CV splitter object.
    scoring : str or callable, default='auto'
        Scoring metric. 'auto' uses accuracy for classifiers, r2 for
        regressors. Can be a string key or a callable(y_true, y_pred).
    return_oof : bool, default=True
        Whether to return out-of-fold predictions.
    random_state : int or None, default=None
        Random state for reproducibility.
    verbose : int, default=0
        Verbosity level. 0=silent, 1=progress, 2=detailed.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.model_selection import GridSearchCV
    >>>
    >>> # With hyperparameter search
    >>> search = GridSearchCV(
    ...     RandomForestClassifier(random_state=42),
    ...     param_grid={'n_estimators': [50, 100, 200]},
    ...     cv=3, scoring='accuracy', refit=True
    ... )
    >>> ncv = NestedCV(search=search, outer_cv=5)
    >>> result = ncv.evaluate(X, y)
    >>>
    >>> # Without search (just evaluate a fixed model)
    >>> ncv = NestedCV(estimator=RandomForestClassifier(n_estimators=100))
    >>> result = ncv.evaluate(X, y)
    """

    def __init__(
        self,
        estimator=None,
        search=None,
        outer_cv: int | Any = 5,
        scoring: str | Callable = "auto",
        return_oof: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        if estimator is None and search is None:
            raise ValueError("Either 'estimator' or 'search' must be provided.")

        self.estimator = estimator
        self.search = search
        self.outer_cv = outer_cv
        self.scoring = scoring
        self.return_oof = return_oof
        self.random_state = random_state
        self.verbose = verbose

    def evaluate(self, X, y, groups=None) -> NestedCVResult:
        """Run nested cross-validation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        groups : array-like of shape (n_samples,), optional
            Group labels for GroupKFold-style splitting.

        Returns
        -------
        NestedCVResult
            Results containing scores, best params, and OOF predictions.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Determine if classifier
        if self.search is not None:
            base = self.search.estimator if hasattr(self.search, 'estimator') else self.search
        else:
            base = self.estimator
        is_clf = is_classifier(base)

        # Set up scoring
        scoring_name, score_fn = self._resolve_scoring(is_clf)

        # Set up outer CV
        outer_cv = self._resolve_cv(is_clf, y)

        # Initialize results
        outer_scores = []
        best_params = []
        inner_scores = []
        oof_preds = np.zeros(len(y)) if self.return_oof and not is_clf else None
        oof_proba = None

        if self.return_oof and is_clf:
            oof_preds = np.zeros(len(y), dtype=y.dtype)

        for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            if self.verbose >= 1:
                print(f"Outer fold {fold_idx + 1}/{outer_cv.get_n_splits()}: "
                      f"train={len(train_idx)}, test={len(test_idx)}")

            # Inner loop: model selection
            if self.search is not None:
                search = clone(self.search)
                search.fit(X_train, y_train)
                model = search.best_estimator_
                fold_params = search.best_params_
                fold_inner_score = search.best_score_
            else:
                model = clone(self.estimator)
                model.fit(X_train, y_train)
                fold_params = model.get_params()
                fold_inner_score = float('nan')

            # Outer evaluation
            y_pred = model.predict(X_test)
            fold_score = score_fn(y_test, y_pred)

            outer_scores.append(fold_score)
            best_params.append(fold_params)
            inner_scores.append(fold_inner_score)

            if self.return_oof and oof_preds is not None:
                oof_preds[test_idx] = y_pred

            if self.verbose >= 2:
                print(f"  Score: {fold_score:.4f}, "
                      f"Inner best: {fold_inner_score:.4f}, "
                      f"Params: {fold_params}")

        result = NestedCVResult(
            outer_scores=outer_scores,
            mean_score=float(np.mean(outer_scores)),
            std_score=float(np.std(outer_scores)),
            best_params=best_params,
            oof_predictions=oof_preds,
            inner_scores=inner_scores,
            scoring=scoring_name,
        )

        if self.verbose >= 1:
            print(f"\nNested CV Result: {result.mean_score:.4f} +/- {result.std_score:.4f}")

        return result

    def _resolve_scoring(self, is_clf: bool):
        """Resolve scoring metric."""
        if callable(self.scoring):
            return "custom", self.scoring

        if self.scoring == "auto":
            name = "accuracy" if is_clf else "r2"
        else:
            name = self.scoring

        if name in _METRIC_MAP:
            return name, _METRIC_MAP[name]

        # Try sklearn scorer
        try:
            scorer = get_scorer(name)
            def score_fn(y_true, y_pred):
                # Simple wrapper — doesn't use estimator
                return scorer._score_func(y_true, y_pred, **scorer._kwargs)
            return name, score_fn
        except (ValueError, KeyError):
            raise ValueError(f"Unknown scoring metric: {name}")

    def _resolve_cv(self, is_clf: bool, y: np.ndarray):
        """Resolve outer CV splitter."""
        if isinstance(self.outer_cv, int):
            if is_clf:
                return StratifiedKFold(
                    n_splits=self.outer_cv,
                    shuffle=True,
                    random_state=self.random_state,
                )
            else:
                return KFold(
                    n_splits=self.outer_cv,
                    shuffle=True,
                    random_state=self.random_state,
                )
        return self.outer_cv
