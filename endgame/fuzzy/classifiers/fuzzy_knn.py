"""Fuzzy K-Nearest Neighbors classifier (Keller et al., 1985).

Assigns fuzzy class memberships based on nearest neighbor distances,
producing soft predictions that reflect neighborhood composition.

Example
-------
>>> from endgame.fuzzy.classifiers import FuzzyKNNClassifier
>>> clf = FuzzyKNNClassifier(n_neighbors=5, m=2.0)
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> labels = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FuzzyKNNClassifier(BaseEstimator, ClassifierMixin):
    """Fuzzy K-Nearest Neighbors classifier.

    Instance-based fuzzy classifier that computes class membership degrees
    for test samples using distance-weighted contributions from neighbors.
    Each training sample carries fuzzy memberships to all classes, initialized
    from its local neighborhood.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    m : float, default=2.0
        Fuzziness parameter (must be > 1). Higher values produce softer
        memberships. When m=2, the weighting is inversely proportional
        to squared distance.
    metric : str, default='euclidean'
        Distance metric. Supports 'euclidean' and 'manhattan'.
    weights : str, default='distance'
        Weight function. 'distance' uses inverse distance weighting,
        'uniform' uses equal weights for all neighbors.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    X_train_ : ndarray of shape (n_samples, n_features)
        Training data stored for prediction.
    memberships_ : ndarray of shape (n_samples, n_classes)
        Fuzzy class memberships for each training sample.

    References
    ----------
    J.M. Keller, M.R. Gray, J.A. Givens, "A Fuzzy K-Nearest Neighbor
    Algorithm", IEEE Transactions on Systems, Man, and Cybernetics, 1985.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.classifiers import FuzzyKNNClassifier
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    >>> y = np.array([0, 0, 0, 1, 1, 1])
    >>> clf = FuzzyKNNClassifier(n_neighbors=3, m=2.0)
    >>> clf.fit(X, y)
    FuzzyKNNClassifier(n_neighbors=3)
    >>> clf.predict(np.array([[2.5, 2.5]]))
    array([...])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        m: float = 2.0,
        metric: str = "euclidean",
        weights: str = "distance",
    ):
        self.n_neighbors = n_neighbors
        self.m = m
        self.metric = metric
        self.weights = weights

    def fit(self, X: Any, y: Any) -> FuzzyKNNClassifier:
        """Fit the fuzzy KNN model.

        Stores training data and computes fuzzy class memberships for
        each training sample based on its k-nearest neighbors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.m <= 1.0:
            raise ValueError(f"Fuzziness parameter m must be > 1, got {self.m}")
        if self.n_neighbors < 1:
            raise ValueError(
                f"n_neighbors must be >= 1, got {self.n_neighbors}"
            )

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        self.X_train_ = X.copy()
        self.y_train_ = y_encoded.copy()

        # Compute fuzzy memberships for training samples
        n_samples = X.shape[0]
        self.memberships_ = np.zeros((n_samples, self.n_classes_))

        # Use k neighbors (excluding self) for membership initialization
        k_init = min(self.n_neighbors, n_samples - 1)

        if k_init == 0:
            # Only one sample: assign full membership to its class
            for i in range(n_samples):
                self.memberships_[i, y_encoded[i]] = 1.0
            return self

        # Compute pairwise distances
        distances = self._compute_distances(X, X)

        for i in range(n_samples):
            # Sort distances, skip self (index 0 with distance 0)
            sorted_idx = np.argsort(distances[i])
            # Remove self from neighbor list
            neighbor_idx = sorted_idx[sorted_idx != i][:k_init]

            # Count class occurrences among neighbors
            neighbor_classes = y_encoded[neighbor_idx]

            for c in range(self.n_classes_):
                n_c = np.sum(neighbor_classes == c)
                if y_encoded[i] == c:
                    # Crisp initialization: 0.51 + 0.49 * (n_c / k_init)
                    self.memberships_[i, c] = 0.51 + 0.49 * (n_c / k_init)
                else:
                    # Non-member class
                    self.memberships_[i, c] = 0.49 * (n_c / k_init)

            # Ensure memberships sum to 1
            mem_sum = np.sum(self.memberships_[i])
            if mem_sum > 0:
                self.memberships_[i] /= mem_sum

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict fuzzy class membership degrees for samples in X.

        For each test sample, finds k nearest training neighbors and
        computes distance-weighted fuzzy memberships.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Fuzzy membership degrees (normalized to sum to 1).
        """
        check_is_fitted(self, ["X_train_", "memberships_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        n_test = X.shape[0]
        n_train = self.X_train_.shape[0]
        k = min(self.n_neighbors, n_train)
        exponent = 2.0 / (self.m - 1.0)

        # Compute distances from test to training samples
        distances = self._compute_distances(X, self.X_train_)

        proba = np.zeros((n_test, self.n_classes_))

        for i in range(n_test):
            # Find k nearest neighbors
            neighbor_idx = np.argsort(distances[i])[:k]
            neighbor_dists = distances[i, neighbor_idx]

            # Check for zero distances (identical points)
            zero_mask = neighbor_dists == 0.0
            if np.any(zero_mask):
                # Average memberships of all identical neighbors
                identical_idx = neighbor_idx[zero_mask]
                proba[i] = np.mean(self.memberships_[identical_idx], axis=0)
            else:
                if self.weights == "uniform":
                    # Equal weights for all neighbors
                    proba[i] = np.mean(self.memberships_[neighbor_idx], axis=0)
                else:
                    # Distance-weighted: w_j = 1 / d_j^(2/(m-1))
                    inv_dist_weights = 1.0 / (neighbor_dists ** exponent)
                    weight_sum = np.sum(inv_dist_weights)

                    for c in range(self.n_classes_):
                        proba[i, c] = np.sum(
                            self.memberships_[neighbor_idx, c] * inv_dist_weights
                        ) / weight_sum

            # Normalize to ensure probabilities sum to 1
            row_sum = np.sum(proba[i])
            if row_sum > 0:
                proba[i] /= row_sum
            else:
                # Fallback: uniform distribution
                proba[i] = 1.0 / self.n_classes_

        return proba

    def _compute_distances(
        self, X1: np.ndarray, X2: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise distances between X1 and X2.

        Parameters
        ----------
        X1 : ndarray of shape (n1, n_features)
        X2 : ndarray of shape (n2, n_features)

        Returns
        -------
        ndarray of shape (n1, n2)
            Pairwise distances.
        """
        if self.metric == "euclidean":
            # Efficient computation: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*a.b
            sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
            sq2 = np.sum(X2 ** 2, axis=1, keepdims=True)
            dists_sq = sq1 + sq2.T - 2.0 * X1 @ X2.T
            # Clamp to avoid negative values from numerical errors
            dists_sq = np.maximum(dists_sq, 0.0)
            return np.sqrt(dists_sq)
        elif self.metric == "manhattan":
            # Use broadcasting
            return np.sum(
                np.abs(X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2
            )
        else:
            raise ValueError(
                f"Unknown metric: {self.metric}. "
                f"Supported: 'euclidean', 'manhattan'"
            )
