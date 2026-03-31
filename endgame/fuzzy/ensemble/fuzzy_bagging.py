"""Membership-weighted bagging via Fuzzy C-Means.

Instead of uniform bootstrap sampling, each bag weights samples
according to their fuzzy membership degree to a randomly chosen
cluster centre.  This produces diverse, overlapping training
distributions that improve ensemble diversity while preserving the
density structure of the data.

Example
-------
>>> from endgame.fuzzy.ensemble import FuzzyBaggingClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>> bag = FuzzyBaggingClassifier(
...     base_estimator=DecisionTreeClassifier(max_depth=5),
...     n_estimators=10,
... )
>>> bag.fit(X_train, y_train)
>>> bag.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Fuzzy C-Means helpers
# ---------------------------------------------------------------------------


def _fuzzy_c_means_memberships(
    X: np.ndarray,
    n_clusters: int,
    fuzziness: float,
    rng: np.random.Generator,
    max_iter: int = 30,
) -> np.ndarray:
    """Run Fuzzy C-Means and return membership matrix.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    n_clusters : int
    fuzziness : float
        FCM fuzziness parameter (m > 1).
    rng : Generator
    max_iter : int

    Returns
    -------
    U : ndarray of shape (n_samples, n_clusters)
        Membership degrees (rows sum to 1).
    """
    n_samples, n_features = X.shape
    m = fuzziness

    # Initialise random membership matrix
    U = rng.random((n_samples, n_clusters)).astype(np.float64)
    U /= U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        # Update centres
        Um = U ** m
        denom = Um.sum(axis=0)  # (n_clusters,)
        denom = np.where(denom > 0, denom, 1.0)
        centers = (Um.T @ X) / denom[:, None]  # (n_clusters, n_features)

        # Update memberships
        dist = np.zeros((n_samples, n_clusters), dtype=np.float64)
        for k in range(n_clusters):
            diff = X - centers[k]
            dist[:, k] = np.sqrt((diff ** 2).sum(axis=1) + 1e-12)

        power = 2.0 / (m - 1.0)
        U_new = np.zeros_like(U)
        for k in range(n_clusters):
            ratio_sum = np.zeros(n_samples, dtype=np.float64)
            for j in range(n_clusters):
                ratio_sum += (dist[:, k] / (dist[:, j] + 1e-12)) ** power
            U_new[:, k] = 1.0 / (ratio_sum + 1e-12)
        U_new /= U_new.sum(axis=1, keepdims=True)
        U = U_new

    return U


# ---------------------------------------------------------------------------
# FuzzyBaggingClassifier
# ---------------------------------------------------------------------------


class FuzzyBaggingClassifier(BaseEstimator, ClassifierMixin):
    """Membership-weighted bagging classifier.

    For each bag, Fuzzy C-Means assigns a membership degree to each
    sample with respect to a randomly selected cluster centre.  Those
    memberships serve as ``sample_weight`` for the base estimator,
    producing diverse models that respect data density.

    Parameters
    ----------
    base_estimator : estimator or None, default=None
        Base learner (must support ``sample_weight`` in ``fit``).
        Defaults to ``DecisionTreeClassifier(max_depth=5)``.
    n_estimators : int, default=10
        Number of bags.
    fuzziness : float, default=2.0
        FCM fuzziness parameter (m).  Must be > 1.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyBaggingClassifier
    >>> bag = FuzzyBaggingClassifier(n_estimators=10)
    >>> bag.fit(X_train, y_train)
    >>> bag.predict(X_test)
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 10,
        fuzziness: float = 2.0,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.fuzziness = fuzziness
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> FuzzyBaggingClassifier:
        """Fit the fuzzy bagging classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        self.n_classes_ = len(self.classes_)
        self.label_encoder_ = le

        rng = np.random.default_rng(self.random_state)
        base = self.base_estimator or DecisionTreeClassifier(max_depth=5)

        # Compute FCM memberships (use n_estimators clusters so each bag
        # corresponds to a different cluster weighting)
        n_clusters = min(self.n_estimators, max(2, X.shape[0] // 10))
        U = _fuzzy_c_means_memberships(
            X, n_clusters=n_clusters, fuzziness=self.fuzziness, rng=rng,
        )

        self.estimators_: list[Any] = []
        for i in range(self.n_estimators):
            cluster_idx = i % n_clusters
            weights = U[:, cluster_idx].copy()
            # Add small uniform noise for diversity
            weights += rng.uniform(0, 0.01, size=weights.shape)
            weights /= weights.sum()

            est = clone(base)
            est.fit(X, y, sample_weight=weights)
            self.estimators_.append(est)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)
        probas = np.zeros((X.shape[0], self.n_classes_), dtype=np.float64)
        for est in self.estimators_:
            p = est.predict_proba(X)
            # Handle estimators that may have fewer classes
            if p.shape[1] == self.n_classes_:
                probas += p
            else:
                for j, cls in enumerate(est.classes_):
                    col = np.where(self.classes_ == cls)[0]
                    if len(col) > 0:
                        probas[:, col[0]] += p[:, j]
        probas /= len(self.estimators_)
        return probas

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]


# ---------------------------------------------------------------------------
# FuzzyBaggingRegressor
# ---------------------------------------------------------------------------


class FuzzyBaggingRegressor(BaseEstimator, RegressorMixin):
    """Membership-weighted bagging regressor.

    Same FCM-based sample weighting as the classifier variant but
    for regression tasks.

    Parameters
    ----------
    base_estimator : estimator or None, default=None
        Base learner (must support ``sample_weight``).
        Defaults to ``DecisionTreeRegressor(max_depth=5)``.
    n_estimators : int, default=10
        Number of bags.
    fuzziness : float, default=2.0
        FCM fuzziness parameter (m > 1).
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyBaggingRegressor
    >>> bag = FuzzyBaggingRegressor(n_estimators=10)
    >>> bag.fit(X_train, y_train)
    >>> bag.predict(X_test)
    """

    def __init__(
        self,
        base_estimator: Any = None,
        n_estimators: int = 10,
        fuzziness: float = 2.0,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.fuzziness = fuzziness
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> FuzzyBaggingRegressor:
        """Fit the fuzzy bagging regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        y = y.astype(np.float64)

        rng = np.random.default_rng(self.random_state)
        base = self.base_estimator or DecisionTreeRegressor(max_depth=5)

        n_clusters = min(self.n_estimators, max(2, X.shape[0] // 10))
        U = _fuzzy_c_means_memberships(
            X, n_clusters=n_clusters, fuzziness=self.fuzziness, rng=rng,
        )

        self.estimators_: list[Any] = []
        for i in range(self.n_estimators):
            cluster_idx = i % n_clusters
            weights = U[:, cluster_idx].copy()
            weights += rng.uniform(0, 0.01, size=weights.shape)
            weights /= weights.sum()

            est = clone(base)
            est.fit(X, y, sample_weight=weights)
            self.estimators_.append(est)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)
        preds = np.zeros(X.shape[0], dtype=np.float64)
        for est in self.estimators_:
            preds += est.predict(X)
        return preds / len(self.estimators_)
