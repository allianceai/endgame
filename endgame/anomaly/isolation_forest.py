"""Isolation Forest detectors with competition-tuned defaults.

This module provides sklearn-compatible Isolation Forest implementations:
- IsolationForestDetector: Standard Isolation Forest with optimized defaults
- ExtendedIsolationForest: Extended IF using random hyperplanes (better for clustered anomalies)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.ensemble import IsolationForest
from sklearn.utils.validation import check_array, check_is_fitted


class IsolationForestDetector(BaseEstimator, OutlierMixin):
    """Isolation Forest with competition-tuned defaults.

    This wrapper provides sensible defaults optimized for competition performance:
    - Higher n_estimators (200 vs sklearn's 100)
    - Bootstrap sampling enabled
    - max_features tuned for high-dimensional data
    - Consistent scoring convention (higher = more anomalous)

    Parameters
    ----------
    n_estimators : int, default=200
        Number of isolation trees. More trees = more stable anomaly scores.
    contamination : float or 'auto', default='auto'
        Expected proportion of anomalies. 'auto' uses heuristic based on
        training data distribution.
    max_samples : int or float or 'auto', default='auto'
        Number of samples to draw for each tree.
        - 'auto': min(256, n_samples)
        - int: exact number of samples
        - float: fraction of samples
    max_features : float or int, default=1.0
        Features to draw for each tree.
        - float: fraction of features
        - int: exact number of features
    bootstrap : bool, default=True
        Whether to bootstrap samples. True improves diversity.
    n_jobs : int, default=-1
        Parallel jobs for fitting trees. -1 uses all cores.
    random_state : int or None, default=None
        Random seed for reproducibility.
    warm_start : bool, default=False
        Reuse trees from previous fit and add more.

    Attributes
    ----------
    model_ : IsolationForest
        Fitted sklearn IsolationForest instance.
    threshold_ : float
        Decision threshold for binary anomaly classification.

    Examples
    --------
    >>> from endgame.anomaly import IsolationForestDetector
    >>> detector = IsolationForestDetector(contamination=0.1)
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)  # Higher = more anomalous
    >>> labels = detector.predict(X_test)  # 1 = anomaly, 0 = normal
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float | str = "auto",
        max_samples: int | float | str = "auto",
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        n_jobs: int = -1,
        random_state: int | None = None,
        warm_start: bool = False,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.warm_start = warm_start

    def fit(self, X: ArrayLike, y=None) -> IsolationForestDetector:
        """Fit the Isolation Forest on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : IsolationForestDetector
            Fitted detector.
        """
        X = check_array(X, accept_sparse=False)

        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            warm_start=self.warm_start,
        )
        self.model_.fit(X)

        # Store threshold for predict
        self.threshold_ = self.model_.offset_
        self.n_features_in_ = X.shape[1]

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples (opposite of sklearn convention).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher = more anomalous.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False)

        # Negate sklearn's scores so higher = more anomalous
        return -self.model_.decision_function(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for anomalies, 0 for normal samples.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False)

        # Convert sklearn's {-1, 1} to {1, 0}
        sklearn_labels = self.model_.predict(X)
        return (sklearn_labels == -1).astype(int)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training samples.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for anomalies, 0 for normal samples.
        """
        self.fit(X)
        return self.predict(X)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Return raw anomaly scores (average path length).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Average path lengths (lower = more anomalous).
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False)
        return self.model_.score_samples(X)


class ExtendedIsolationForest(BaseEstimator, OutlierMixin):
    """Extended Isolation Forest using random hyperplanes.

    Standard Isolation Forest uses axis-parallel splits, which can miss
    anomalies in clustered or elongated distributions. Extended IF uses
    random hyperplane splits, providing better detection for:
    - Clustered anomalies
    - Anomalies along linear subspaces
    - High-dimensional data with correlations

    Parameters
    ----------
    n_estimators : int, default=200
        Number of isolation trees.
    contamination : float, default=0.1
        Expected proportion of anomalies for threshold setting.
    max_samples : int or float or 'auto', default='auto'
        Number of samples to draw for each tree.
    extension_level : int or None, default=None
        Dimensionality of random hyperplanes.
        - None: full dimensionality (n_features - 1)
        - int: specific dimensionality (0 = standard IF)
    random_state : int or None, default=None
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Parallel jobs for evaluation. -1 uses all cores.

    Attributes
    ----------
    trees_ : list
        Fitted extended isolation trees.
    threshold_ : float
        Decision threshold for binary classification.

    References
    ----------
    Hariri, S., Kind, M. C., & Brunner, R. J. (2019).
    Extended Isolation Forest. IEEE TKDE.

    Examples
    --------
    >>> from endgame.anomaly import ExtendedIsolationForest
    >>> detector = ExtendedIsolationForest(contamination=0.1)
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.1,
        max_samples: int | float | str = "auto",
        extension_level: int | None = None,
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.extension_level = extension_level
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> ExtendedIsolationForest:
        """Fit Extended Isolation Forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : ExtendedIsolationForest
            Fitted detector.
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Determine sample size
        if self.max_samples == "auto":
            max_samples = min(256, n_samples)
        elif isinstance(self.max_samples, float):
            max_samples = int(self.max_samples * n_samples)
        else:
            max_samples = self.max_samples

        # Determine extension level
        if self.extension_level is None:
            self.extension_level_ = n_features - 1
        else:
            self.extension_level_ = min(self.extension_level, n_features - 1)

        # Height limit
        self.height_limit_ = int(np.ceil(np.log2(max_samples)))

        rng = np.random.default_rng(self.random_state)

        # Build trees
        self.trees_ = []
        for i in range(self.n_estimators):
            # Sample data
            indices = rng.choice(n_samples, size=max_samples, replace=False)
            X_sample = X[indices]

            # Build tree
            tree = self._build_tree(X_sample, 0, rng)
            self.trees_.append(tree)

        # Compute threshold from training scores
        train_scores = self.decision_function(X)
        self.threshold_ = np.percentile(train_scores, 100 * (1 - self.contamination))

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        depth: int,
        rng: np.random.Generator
    ) -> dict:
        """Recursively build an extended isolation tree."""
        n_samples, n_features = X.shape

        # Termination conditions
        if depth >= self.height_limit_ or n_samples <= 1:
            return {"type": "leaf", "size": n_samples}

        # Random hyperplane: n = random normal vector, p = random intercept
        # For extension_level = 0, this reduces to standard IF (single coordinate)
        if self.extension_level_ == 0:
            # Standard axis-parallel split
            split_dim = rng.integers(n_features)
            n = np.zeros(n_features)
            n[split_dim] = 1.0
        else:
            # Random hyperplane in extension_level + 1 dimensions
            dims = rng.choice(n_features, size=self.extension_level_ + 1, replace=False)
            n = np.zeros(n_features)
            n[dims] = rng.standard_normal(len(dims))
            n = n / (np.linalg.norm(n) + 1e-10)

        # Project data onto normal
        projections = X @ n

        # Random split point between min and max
        p_min, p_max = projections.min(), projections.max()
        if p_min == p_max:
            return {"type": "leaf", "size": n_samples}

        p = rng.uniform(p_min, p_max)

        # Split
        left_mask = projections < p
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            return {"type": "leaf", "size": n_samples}

        return {
            "type": "node",
            "n": n,
            "p": p,
            "left": self._build_tree(X[left_mask], depth + 1, rng),
            "right": self._build_tree(X[right_mask], depth + 1, rng),
        }

    def _path_length(self, x: np.ndarray, tree: dict, depth: int = 0) -> float:
        """Compute path length for a single sample."""
        if tree["type"] == "leaf":
            # Add expected path length for remaining elements
            n = tree["size"]
            if n <= 1:
                return depth
            return depth + self._c(n)

        projection = np.dot(x, tree["n"])
        if projection < tree["p"]:
            return self._path_length(x, tree["left"], depth + 1)
        else:
            return self._path_length(x, tree["right"], depth + 1)

    @staticmethod
    def _c(n: int) -> float:
        """Average path length of unsuccessful search in BST."""
        if n <= 1:
            return 0
        return 2.0 * (np.log(n - 1) + 0.5772156649) - 2.0 * (n - 1) / n

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute anomaly scores.

        Higher scores indicate more anomalous samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher = more anomalous.
        """
        check_is_fitted(self, ["trees_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        n_samples = X.shape[0]

        # Compute average path length for each sample
        path_lengths = np.zeros(n_samples)
        for tree in self.trees_:
            for i in range(n_samples):
                path_lengths[i] += self._path_length(X[i], tree)
        path_lengths /= self.n_estimators

        # Convert to anomaly score (shorter paths = higher score)
        # Score = 2^(-E[h(x)] / c(n))
        c_n = self._c(256)  # Use 256 as reference
        scores = np.power(2, -path_lengths / c_n)

        return scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for anomalies, 0 for normal samples.
        """
        check_is_fitted(self, ["trees_", "threshold_"])
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and predict anomaly labels."""
        self.fit(X)
        return self.predict(X)
