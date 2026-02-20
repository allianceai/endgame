"""Local Outlier Factor detector with competition-tuned defaults.

LOF measures local density deviation to identify samples that are substantially
less dense than their neighbors - making it effective for detecting local anomalies
that exist in low-density regions relative to their k-nearest neighbors.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import LocalOutlierFactor
from sklearn.utils.validation import check_array, check_is_fitted


class LocalOutlierFactorDetector(BaseEstimator, OutlierMixin):
    """Local Outlier Factor with competition-tuned defaults.

    LOF compares the local density of a point with that of its neighbors.
    Points with substantially lower density are considered outliers.
    Effective for detecting local anomalies in non-uniform distributions.

    Parameters
    ----------
    n_neighbors : int, default=20
        Number of neighbors for density estimation. Higher values make
        the detector more robust but may miss small local anomalies.
    contamination : float or 'auto', default='auto'
        Expected proportion of anomalies. Used for threshold setting.
    algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, default='auto'
        Algorithm for nearest neighbor queries.
    leaf_size : int, default=30
        Leaf size for tree algorithms.
    metric : str or callable, default='minkowski'
        Distance metric for neighbor queries.
    p : int, default=2
        Power parameter for Minkowski metric (2 = Euclidean).
    novelty : bool, default=True
        Whether to use LOF for novelty detection (scoring new samples).
        True enables predict() and decision_function() on unseen data.
    n_jobs : int, default=-1
        Parallel jobs for neighbor queries. -1 uses all cores.

    Attributes
    ----------
    model_ : LocalOutlierFactor
        Fitted sklearn LOF instance.
    threshold_ : float
        Decision threshold for binary classification.

    Examples
    --------
    >>> from endgame.anomaly import LocalOutlierFactorDetector
    >>> detector = LocalOutlierFactorDetector(contamination=0.1)
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)  # Higher = more anomalous
    >>> labels = detector.predict(X_test)  # 1 = anomaly, 0 = normal
    """

    def __init__(
        self,
        n_neighbors: int = 20,
        contamination: float | str = "auto",
        algorithm: str = "auto",
        leaf_size: int = 30,
        metric: str = "minkowski",
        p: int = 2,
        novelty: bool = True,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.p = p
        self.novelty = novelty
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> LocalOutlierFactorDetector:
        """Fit the LOF model on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (assumed to be mostly normal).
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : LocalOutlierFactorDetector
            Fitted detector.
        """
        X = check_array(X, accept_sparse=False)

        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            novelty=self.novelty,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)

        self.n_features_in_ = X.shape[1]
        self.threshold_ = self.model_.offset_

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute anomaly scores for samples.

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
        """Fit and predict anomaly labels on training data.

        Note: For LOF, this uses the transductive scores computed during
        fit, not the inductive scores from predict().

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
        X = check_array(X, accept_sparse=False)

        # Use non-novelty LOF for fit_predict (transductive)
        lof_transductive = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            novelty=False,
            n_jobs=self.n_jobs,
        )
        sklearn_labels = lof_transductive.fit_predict(X)

        # Also fit the novelty model for future predictions
        self.model_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            p=self.p,
            novelty=True,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self.n_features_in_ = X.shape[1]
        self.threshold_ = self.model_.offset_

        return (sklearn_labels == -1).astype(int)

    def score_samples(self, X: ArrayLike) -> np.ndarray:
        """Return negative LOF scores (sklearn convention).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Negative LOF scores (higher = more normal).
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False)
        return self.model_.score_samples(X)
