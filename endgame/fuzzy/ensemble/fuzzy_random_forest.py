"""Fuzzy Random Forest ensemble.

Implements bagging over fuzzy decision trees where each split uses
Gaussian membership functions for soft routing. Every sample traverses
both branches weighted by its membership degree, producing smooth
decision boundaries.

Example
-------
>>> from endgame.fuzzy.ensemble import FuzzyRandomForestClassifier
>>> clf = FuzzyRandomForestClassifier(n_estimators=50, max_depth=4)
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.membership import GaussianMF


# ---------------------------------------------------------------------------
# Internal fuzzy tree used as the base estimator
# ---------------------------------------------------------------------------


class _FuzzyTreeNode:
    """A node in the internal fuzzy decision tree."""

    __slots__ = (
        "feature", "threshold", "sigma",
        "left", "right",
        "value", "is_leaf",
    )

    def __init__(self):
        self.feature: int = -1
        self.threshold: float = 0.0
        self.sigma: float = 1.0
        self.left: _FuzzyTreeNode | None = None
        self.right: _FuzzyTreeNode | None = None
        self.value: np.ndarray | None = None  # class dist or mean
        self.is_leaf: bool = False


class _FuzzyTree:
    """Simplified fuzzy decision tree for ensemble use.

    At each internal node a Gaussian membership function splits the data
    softly: ``mu_low = exp(-0.5*((x-threshold)/sigma)^2)`` routes left,
    ``1 - mu_low`` routes right.  Every sample therefore reaches *all*
    leaves with some (possibly tiny) weight, and the final prediction is
    the weight-normalised average across leaves.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum weighted sample count to allow a split.
    n_mfs : int
        Number of membership functions per split (kept at 2: low/high).
    task : str
        ``'classification'`` or ``'regression'``.
    n_classes : int
        Number of classes (ignored for regression).
    rng : np.random.Generator
        Random generator for feature sub-sampling.
    max_features : int
        Number of features to consider at each split.
    """

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        n_mfs: int,
        task: str,
        n_classes: int,
        rng: np.random.Generator,
        max_features: int,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_mfs = n_mfs
        self.task = task
        self.n_classes = n_classes
        self.rng = rng
        self.max_features = max_features
        self.root: _FuzzyTreeNode | None = None
        self.feature_importances: np.ndarray | None = None

    # -- fitting -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray) -> None:
        n_features = X.shape[1]
        self.feature_importances = np.zeros(n_features, dtype=np.float64)
        self.root = self._build(X, y, sample_weight, depth=0)

    def _build(
        self,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        depth: int,
    ) -> _FuzzyTreeNode:
        node = _FuzzyTreeNode()
        total_w = w.sum()

        # Leaf conditions
        if (
            depth >= self.max_depth
            or total_w < 2.0 * self.min_samples_leaf
            or (self.task == "classification" and len(np.unique(y)) <= 1)
        ):
            node.is_leaf = True
            node.value = self._leaf_value(y, w)
            return node

        best_gain = -np.inf
        best_feat = -1
        best_thresh = 0.0
        best_sigma = 1.0

        n_features = X.shape[1]
        candidates = self.rng.choice(
            n_features,
            size=min(self.max_features, n_features),
            replace=False,
        )

        parent_impurity = self._impurity(y, w)

        for feat in candidates:
            col = X[:, feat]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min < 1e-12:
                continue
            sigma = (col_max - col_min) / 4.0
            # Try several threshold candidates (quantiles)
            quantiles = np.quantile(col, [0.25, 0.5, 0.75])
            for thresh in quantiles:
                mu_left = np.exp(-0.5 * ((col - thresh) / max(sigma, 1e-12)) ** 2)
                mu_right = 1.0 - mu_left

                w_left = w * mu_left
                w_right = w * mu_right
                sw_left = w_left.sum()
                sw_right = w_right.sum()

                if sw_left < self.min_samples_leaf or sw_right < self.min_samples_leaf:
                    continue

                imp_left = self._impurity(y, w_left)
                imp_right = self._impurity(y, w_right)
                gain = parent_impurity - (
                    sw_left / total_w * imp_left + sw_right / total_w * imp_right
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    best_sigma = sigma

        if best_feat < 0 or best_gain <= 0.0:
            node.is_leaf = True
            node.value = self._leaf_value(y, w)
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.sigma = best_sigma

        # Record feature importance
        self.feature_importances[best_feat] += best_gain * total_w  # type: ignore[index]

        col = X[:, best_feat]
        mu_left = np.exp(
            -0.5 * ((col - best_thresh) / max(best_sigma, 1e-12)) ** 2
        )
        mu_right = 1.0 - mu_left

        node.left = self._build(X, y, w * mu_left, depth + 1)
        node.right = self._build(X, y, w * mu_right, depth + 1)
        return node

    # -- prediction --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return leaf values averaged by membership weight."""
        n = X.shape[0]
        if self.task == "classification":
            accum = np.zeros((n, self.n_classes), dtype=np.float64)
        else:
            accum = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        self._predict_node(X, self.root, np.ones(n), accum, weights)
        # normalise
        safe = np.where(weights > 0, weights, 1.0)
        if self.task == "classification":
            return accum / safe[:, None]
        return accum / safe

    def _predict_node(
        self,
        X: np.ndarray,
        node: _FuzzyTreeNode,
        w: np.ndarray,
        accum: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        if node.is_leaf:
            if self.task == "classification":
                accum += w[:, None] * node.value[None, :]
            else:
                accum += w * node.value
            weights += w
            return

        col = X[:, node.feature]
        mu_left = np.exp(
            -0.5 * ((col - node.threshold) / max(node.sigma, 1e-12)) ** 2
        )
        mu_right = 1.0 - mu_left

        self._predict_node(X, node.left, w * mu_left, accum, weights)
        self._predict_node(X, node.right, w * mu_right, accum, weights)

    # -- helpers -----------------------------------------------------------

    def _impurity(self, y: np.ndarray, w: np.ndarray) -> float:
        total = w.sum()
        if total < 1e-12:
            return 0.0
        if self.task == "classification":
            # Weighted Gini
            probs = np.zeros(self.n_classes)
            np.add.at(probs, y, w)
            probs /= total
            return float(1.0 - np.sum(probs ** 2))
        else:
            # Weighted MSE
            mean = np.average(y, weights=w)
            return float(np.average((y - mean) ** 2, weights=w))

    def _leaf_value(self, y: np.ndarray, w: np.ndarray) -> np.ndarray:
        if self.task == "classification":
            probs = np.zeros(self.n_classes, dtype=np.float64)
            np.add.at(probs, y, w)
            total = probs.sum()
            if total > 0:
                probs /= total
            return probs
        else:
            total = w.sum()
            if total < 1e-12:
                return np.array(0.0)
            return np.array(np.average(y, weights=w))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FuzzyRandomForestClassifier(BaseEstimator, ClassifierMixin):
    """Fuzzy Random Forest for classification.

    An ensemble of fuzzy decision trees trained on bootstrap samples.
    Each tree uses Gaussian membership functions for soft splits so that
    every sample is routed through both children with degree proportional
    to its membership.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of fuzzy trees in the forest.
    max_depth : int, default=5
        Maximum depth of each tree.
    max_features : str or int, default='sqrt'
        Number of features considered at each split.
        ``'sqrt'`` uses ``sqrt(n_features)``, ``'log2'`` uses
        ``log2(n_features)``, an int uses that number directly.
    min_samples_leaf : int, default=5
        Minimum weighted sample count in a leaf.
    n_mfs : int, default=2
        Number of membership functions per split node (low / high).
    random_state : int or None, default=None
        Random seed for reproducibility.
    n_jobs : int or None, default=None
        Not yet used (reserved for parallel fitting).
    verbose : bool, default=False
        Print progress during fitting.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_importances_ : ndarray of shape (n_features,)
        Normalised sum of fuzzy information gains per feature.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyRandomForestClassifier
    >>> clf = FuzzyRandomForestClassifier(n_estimators=20, max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        max_features: str | int = "sqrt",
        min_samples_leaf: int = 5,
        n_mfs: int = 2,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_mfs = n_mfs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    # -- sklearn API -------------------------------------------------------

    def fit(self, X: Any, y: Any) -> FuzzyRandomForestClassifier:
        """Fit the fuzzy random forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.

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

        max_feat = self._resolve_max_features(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        self.estimators_: list[_FuzzyTree] = []
        raw_importances = np.zeros(X.shape[1], dtype=np.float64)

        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"[FuzzyRandomForest] fitting tree {i + 1}/{self.n_estimators}")
            # Bootstrap sample
            idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot, y_boot = X[idx], y_enc[idx]
            w_boot = np.ones(X_boot.shape[0], dtype=np.float64)

            tree = _FuzzyTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_mfs=self.n_mfs,
                task="classification",
                n_classes=self.n_classes_,
                rng=rng,
                max_features=max_feat,
            )
            tree.fit(X_boot, y_boot, w_boot)
            self.estimators_.append(tree)
            raw_importances += tree.feature_importances  # type: ignore[operator]

        total = raw_importances.sum()
        self.feature_importances_ = (
            raw_importances / total if total > 0 else raw_importances
        )
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
        for tree in self.estimators_:
            probas += tree.predict(X)
        probas /= len(self.estimators_)
        # Ensure rows sum to 1
        row_sums = probas.sum(axis=1, keepdims=True)
        probas = np.where(row_sums > 0, probas / row_sums, 1.0 / self.n_classes_)
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

    # -- helpers -----------------------------------------------------------

    def _resolve_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        return n_features


class FuzzyRandomForestRegressor(BaseEstimator, RegressorMixin):
    """Fuzzy Random Forest for regression.

    An ensemble of fuzzy decision trees trained on bootstrap samples.
    Each tree uses Gaussian membership functions for soft splits and
    predictions are averaged across all trees.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of fuzzy trees in the forest.
    max_depth : int, default=5
        Maximum depth of each tree.
    max_features : str or int, default='sqrt'
        Features considered per split (``'sqrt'``, ``'log2'``, or int).
    min_samples_leaf : int, default=5
        Minimum weighted sample count in a leaf.
    n_mfs : int, default=2
        Number of membership functions per split node.
    random_state : int or None, default=None
        Random seed.
    n_jobs : int or None, default=None
        Reserved for parallel fitting.
    verbose : bool, default=False
        Print progress.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.
    feature_importances_ : ndarray of shape (n_features,)
        Normalised sum of fuzzy information gains per feature.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyRandomForestRegressor
    >>> reg = FuzzyRandomForestRegressor(n_estimators=20, max_depth=4)
    >>> reg.fit(X_train, y_train)
    >>> reg.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        max_features: str | int = "sqrt",
        min_samples_leaf: int = 5,
        n_mfs: int = 2,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.n_mfs = n_mfs
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FuzzyRandomForestRegressor:
        """Fit the fuzzy random forest regressor.

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

        max_feat = self._resolve_max_features(X.shape[1])
        rng = np.random.default_rng(self.random_state)

        self.estimators_: list[_FuzzyTree] = []
        raw_importances = np.zeros(X.shape[1], dtype=np.float64)

        for i in range(self.n_estimators):
            if self.verbose and (i + 1) % 10 == 0:
                print(f"[FuzzyRandomForest] fitting tree {i + 1}/{self.n_estimators}")
            idx = rng.choice(X.shape[0], size=X.shape[0], replace=True)
            X_boot, y_boot = X[idx], y[idx]
            w_boot = np.ones(X_boot.shape[0], dtype=np.float64)

            tree = _FuzzyTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                n_mfs=self.n_mfs,
                task="regression",
                n_classes=0,
                rng=rng,
                max_features=max_feat,
            )
            tree.fit(X_boot, y_boot, w_boot)
            self.estimators_.append(tree)
            raw_importances += tree.feature_importances  # type: ignore[operator]

        total = raw_importances.sum()
        self.feature_importances_ = (
            raw_importances / total if total > 0 else raw_importances
        )
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
        for tree in self.estimators_:
            preds += tree.predict(X)
        return preds / len(self.estimators_)

    def _resolve_max_features(self, n_features: int) -> int:
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if self.max_features == "sqrt":
            return max(1, int(math.sqrt(n_features)))
        if self.max_features == "log2":
            return max(1, int(math.log2(n_features)))
        return n_features
