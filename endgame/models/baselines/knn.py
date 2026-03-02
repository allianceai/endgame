from __future__ import annotations

"""K-Nearest Neighbors classifiers and regressors.

KNN models provide a fundamentally different inductive bias from
tree-based and neural network models:
- Instance-based (lazy) learning - no explicit model training
- Local decision boundaries based on neighbor voting/averaging
- Sensitive to feature scaling and distance metric choice

These characteristics make KNN valuable for ensemble diversity.

References
----------
- Cover & Hart, "Nearest Neighbor Pattern Classification" (1967)
- sklearn.neighbors documentation
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler


class KNNClassifier(ClassifierMixin, BaseEstimator):
    """K-Nearest Neighbors Classifier with competition-tuned defaults.

    A wrapper around sklearn's KNeighborsClassifier with automatic
    feature scaling and sensible defaults for competitive ML.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : str, default='distance'
        Weight function: 'uniform' or 'distance'.
        'distance' often works better in practice.
    metric : str, default='minkowski'
        Distance metric: 'minkowski', 'euclidean', 'manhattan', 'cosine', etc.
    p : int, default=2
        Power parameter for Minkowski metric. p=2 is Euclidean, p=1 is Manhattan.
    leaf_size : int, default=30
        Leaf size for BallTree or KDTree.
    algorithm : str, default='auto'
        Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'.
    scale_features : bool, default=True
        Whether to standardize features before fitting.
        Highly recommended for distance-based methods.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.models.baselines import KNNClassifier
    >>> clf = KNNClassifier(n_neighbors=5, weights='distance')
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    KNN is different from other models because:
    1. Instance-based - stores training data, no explicit model
    2. Non-parametric - makes no assumptions about data distribution
    3. Local decision boundaries - can capture complex patterns
    4. Sensitive to curse of dimensionality in high dimensions

    The scale_features=True default is important because KNN
    relies on distance calculations that can be dominated by
    features with larger scales.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "distance",
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 30,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        scale_features: bool = True,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.leaf_size = leaf_size
        self.algorithm = algorithm
        self.scale_features = scale_features
        self.n_jobs = n_jobs

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: KNeighborsClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> KNNClassifier:
        """Fit the KNN classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Scale features
        if self.scale_features:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean

        # Adjust n_neighbors if necessary
        n_neighbors = min(self.n_neighbors, len(y) - 1)
        if n_neighbors < 1:
            n_neighbors = 1

        # Create and fit model
        self.model_ = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            leaf_size=self.leaf_size,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )

        self.model_.fit(X_scaled, y_encoded)
        self._is_fitted = True

        return self

    def _preprocess(self, X) -> np.ndarray:
        """Preprocess features for prediction."""
        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.scale_features and self._scaler is not None:
            return self._scaler.transform(X_clean)
        return X_clean

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("KNNClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        y_pred = self.model_.predict(X_proc)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("KNNClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.predict_proba(X_proc)

    def kneighbors(
        self, X=None, n_neighbors: int | None = None, return_distance: bool = True
    ):
        """Find the K-neighbors of a point.

        Parameters
        ----------
        X : array-like, optional
            Query points. If None, returns neighbors of training data.
        n_neighbors : int, optional
            Number of neighbors. If None, uses n_neighbors from init.
        return_distance : bool, default=True
            Whether to return distances.

        Returns
        -------
        neigh_dist : ndarray (if return_distance=True)
            Distances to neighbors.
        neigh_ind : ndarray
            Indices of neighbors.
        """
        if not self._is_fitted:
            raise RuntimeError("KNNClassifier has not been fitted.")

        if X is not None:
            X = self._preprocess(X)

        return self.model_.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=return_distance
        )


class KNNRegressor(RegressorMixin, BaseEstimator):
    """K-Nearest Neighbors Regressor with competition-tuned defaults.

    A wrapper around sklearn's KNeighborsRegressor with automatic
    feature scaling and sensible defaults for competitive ML.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of neighbors to use.
    weights : str, default='distance'
        Weight function: 'uniform' or 'distance'.
    metric : str, default='minkowski'
        Distance metric.
    p : int, default=2
        Power parameter for Minkowski metric.
    leaf_size : int, default=30
        Leaf size for BallTree or KDTree.
    algorithm : str, default='auto'
        Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'.
    scale_features : bool, default=True
        Whether to standardize features before fitting.
    n_jobs : int, default=-1
        Number of parallel jobs.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.models.baselines import KNNRegressor
    >>> reg = KNNRegressor(n_neighbors=10, weights='distance')
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)

    Notes
    -----
    KNN regression averages (or weighted-averages) the target values
    of the k nearest neighbors. This provides a local, non-parametric
    estimate that can capture complex patterns but may suffer from
    the curse of dimensionality.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: Literal["uniform", "distance"] = "distance",
        metric: str = "minkowski",
        p: int = 2,
        leaf_size: int = 30,
        algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto",
        scale_features: bool = True,
        n_jobs: int = -1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.p = p
        self.leaf_size = leaf_size
        self.algorithm = algorithm
        self.scale_features = scale_features
        self.n_jobs = n_jobs

        self.n_features_in_: int = 0
        self.model_: KNeighborsRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> KNNRegressor:
        """Fit the KNN regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)
        y_clean = np.nan_to_num(y, nan=0.0)

        # Scale features
        if self.scale_features:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean

        # Adjust n_neighbors if necessary
        n_neighbors = min(self.n_neighbors, len(y) - 1)
        if n_neighbors < 1:
            n_neighbors = 1

        # Create and fit model
        self.model_ = KNeighborsRegressor(
            n_neighbors=n_neighbors,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            leaf_size=self.leaf_size,
            algorithm=self.algorithm,
            n_jobs=self.n_jobs,
        )

        self.model_.fit(X_scaled, y_clean)
        self._is_fitted = True

        return self

    def _preprocess(self, X) -> np.ndarray:
        """Preprocess features for prediction."""
        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.scale_features and self._scaler is not None:
            return self._scaler.transform(X_clean)
        return X_clean

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("KNNRegressor has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.predict(X_proc)

    def kneighbors(
        self, X=None, n_neighbors: int | None = None, return_distance: bool = True
    ):
        """Find the K-neighbors of a point.

        Parameters
        ----------
        X : array-like, optional
            Query points. If None, returns neighbors of training data.
        n_neighbors : int, optional
            Number of neighbors. If None, uses n_neighbors from init.
        return_distance : bool, default=True
            Whether to return distances.

        Returns
        -------
        neigh_dist : ndarray (if return_distance=True)
            Distances to neighbors.
        neigh_ind : ndarray
            Indices of neighbors.
        """
        if not self._is_fitted:
            raise RuntimeError("KNNRegressor has not been fitted.")

        if X is not None:
            X = self._preprocess(X)

        return self.model_.kneighbors(
            X, n_neighbors=n_neighbors, return_distance=return_distance
        )
