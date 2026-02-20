"""Voting Ensemble: Hard and soft voting for classification and regression.

Combines predictions from multiple heterogeneous estimators via majority
vote (classification) or averaging (regression). Supports sample weights,
probability averaging, and named estimator access.

Example
-------
>>> from endgame.ensemble import VotingClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> from sklearn.tree import DecisionTreeClassifier
>>> from sklearn.svm import SVC
>>>
>>> clf = VotingClassifier(
...     estimators=[
...         ("lr", LogisticRegression()),
...         ("dt", DecisionTreeClassifier()),
...         ("svm", SVC(probability=True)),
...     ],
...     voting="soft",
...     weights=[2, 1, 1],
... )
>>> clf.fit(X_train, y_train)
>>> clf.predict(X_test)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore[assignment,misc]
    delayed = None


def _fit_one(estimator, X, y, sample_weight=None, fit_params=None):
    """Fit a single estimator (joblib-friendly)."""
    fit_params = fit_params or {}
    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight, **fit_params)
        except TypeError:
            estimator.fit(X, y, **fit_params)
    else:
        estimator.fit(X, y, **fit_params)
    return estimator


class VotingClassifier(BaseEstimator, ClassifierMixin):
    """Soft / hard voting meta-classifier.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Named base classifiers.
    voting : {'hard', 'soft'}, default='soft'
        - ``'hard'``: majority-vote on predicted labels.
        - ``'soft'``: average predicted probabilities, then argmax.
    weights : array-like of shape (n_estimators,), optional
        Per-estimator weights. ``None`` means uniform.
    flatten_transform : bool, default=True
        If ``True``, ``transform`` returns shape
        ``(n_samples, n_classifiers * n_classes)`` instead of 3-D.
    n_jobs : int or None, default=None
        Parallel fitting jobs. ``-1`` uses all CPUs.
    verbose : bool, default=False
        Print progress during fit.

    Attributes
    ----------
    estimators_ : list of estimator
        Fitted clones in the same order as *estimators*.
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    le_ : dict
        Mapping from label to integer index (for hard voting).

    Examples
    --------
    >>> vc = VotingClassifier(
    ...     estimators=[("rf", RandomForest()), ("lr", LogisticRegression())],
    ...     voting="soft",
    ... )
    >>> vc.fit(X_train, y_train).predict(X_test)
    """

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        voting: str = "soft",
        weights: Sequence[float] | None = None,
        flatten_transform: bool = True,
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights
        self.flatten_transform = flatten_transform
        self.n_jobs = n_jobs
        self.verbose = verbose

    # ------------------------------------------------------------------ fit
    def fit(self, X, y, sample_weight=None, **fit_params):
        """Fit all base estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like, optional
        """
        X = np.asarray(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.le_ = {c: i for i, c in enumerate(self.classes_)}

        clones = [clone(est) for _, est in self.estimators]

        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one)(c, X, y, sample_weight, fit_params)
                for c in clones
            )
        else:
            self.estimators_ = [
                _fit_one(c, X, y, sample_weight, fit_params) for c in clones
            ]
        return self

    # --------------------------------------------------------------- predict
    def predict(self, X):
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if self.voting == "soft":
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # Hard voting — majority rule
        preds = self._collect_predictions(X)
        # preds shape: (n_estimators, n_samples)
        w = np.asarray(self.weights) if self.weights is not None else None
        n = preds.shape[1]
        out = np.empty(n, dtype=self.classes_.dtype)
        for i in range(n):
            votes = preds[:, i]
            if w is not None:
                counts: dict[Any, float] = {}
                for v, wt in zip(votes, w):
                    counts[v] = counts.get(v, 0.0) + wt
                out[i] = max(counts, key=counts.get)  # type: ignore[arg-type]
            else:
                out[i] = stats.mode(votes, keepdims=True).mode[0]
        return out

    # -------------------------------------------------------- predict_proba
    def predict_proba(self, X):
        """Average predicted probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        X = np.asarray(X)
        probas = np.asarray([est.predict_proba(X) for est in self.estimators_])
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            w = w / w.sum()
            avg = np.tensordot(w, probas, axes=([0], [0]))
        else:
            avg = probas.mean(axis=0)
        return avg

    # -------------------------------------------------------------- transform
    def transform(self, X):
        """Return per-estimator predictions or probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray
        """
        if self.voting == "soft":
            probas = np.asarray([est.predict_proba(X) for est in self.estimators_])
            if self.flatten_transform:
                return probas.reshape(probas.shape[1], -1)
            return probas.transpose(1, 0, 2)
        return self._collect_predictions(X).T

    # ----------------------------------------------------------------- helpers
    def _collect_predictions(self, X):
        X = np.asarray(X)
        return np.array([est.predict(X) for est in self.estimators_])

    def get_params(self, deep=True):
        out = super().get_params(deep=deep)
        if not deep:
            return out
        for name, est in self.estimators:
            for key, val in est.get_params(deep=True).items():
                out[f"{name}__{key}"] = val
        return out

    @property
    def named_estimators(self) -> dict[str, BaseEstimator]:
        """Access fitted estimators by name."""
        return {name: est for (name, _), est in zip(self.estimators, self.estimators_)}


class VotingRegressor(BaseEstimator, RegressorMixin):
    """Voting meta-regressor: averages predictions from multiple regressors.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Named base regressors.
    weights : array-like of shape (n_estimators,), optional
        Per-estimator weights. ``None`` means uniform.
    n_jobs : int or None, default=None
        Parallel fitting jobs.
    verbose : bool, default=False
        Print progress during fit.

    Attributes
    ----------
    estimators_ : list of estimator
        Fitted clones.

    Examples
    --------
    >>> vr = VotingRegressor(
    ...     estimators=[("ridge", Ridge()), ("rf", RandomForestRegressor())],
    ...     weights=[1, 2],
    ... )
    >>> vr.fit(X_train, y_train).predict(X_test)
    """

    def __init__(
        self,
        estimators: list[tuple[str, BaseEstimator]],
        weights: Sequence[float] | None = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.estimators = estimators
        self.weights = weights
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None, **fit_params):
        X = np.asarray(X)
        y = np.asarray(y)
        clones = [clone(est) for _, est in self.estimators]
        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_one)(c, X, y, sample_weight, fit_params)
                for c in clones
            )
        else:
            self.estimators_ = [
                _fit_one(c, X, y, sample_weight, fit_params) for c in clones
            ]
        return self

    def predict(self, X):
        X = np.asarray(X)
        preds = np.array([est.predict(X) for est in self.estimators_])
        if self.weights is not None:
            w = np.asarray(self.weights, dtype=float)
            w = w / w.sum()
            return (w[:, None] * preds).sum(axis=0)
        return preds.mean(axis=0)

    def transform(self, X):
        X = np.asarray(X)
        return np.array([est.predict(X) for est in self.estimators_]).T

    @property
    def named_estimators(self) -> dict[str, BaseEstimator]:
        return {name: est for (name, _), est in zip(self.estimators, self.estimators_)}
