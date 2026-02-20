"""ReliefF and MultiSURF feature selection algorithms."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class ReliefFSelector(TransformerMixin, BaseEstimator):
    """ReliefF feature selection algorithm.

    ReliefF is an instance-based feature weighting algorithm that naturally
    handles feature interactions. It evaluates features by how well they
    distinguish between near-miss instances.

    Parameters
    ----------
    n_features : int, default=10
        Number of features to select.

    n_neighbors : int, default=10
        Number of neighbors to consider for each instance.

    n_samples : int or float, default=1.0
        Number of samples to use for estimation.
        - If int, uses that many samples.
        - If float (0-1), uses that fraction of samples.

    algorithm : str, default='relieff'
        Algorithm variant:
        - 'relieff': Standard ReliefF
        - 'multisurf': MultiSURF (adaptive radius)

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    feature_importances_ : ndarray
        Feature importance scores.

    selected_features_ : ndarray
        Indices of selected features.

    ranking_ : ndarray
        Feature ranking by importance.

    Example
    -------
    >>> from endgame.feature_selection import ReliefFSelector
    >>> selector = ReliefFSelector(n_features=20, n_neighbors=10)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        n_features: int = 10,
        n_neighbors: int = 10,
        n_samples: float = 1.0,
        algorithm: Literal["relieff", "multisurf"] = "relieff",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_features = n_features
        self.n_neighbors = n_neighbors
        self.n_samples = n_samples
        self.algorithm = algorithm
        self.random_state = random_state
        self.verbose = verbose

    def _diff(self, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
        """Compute feature-wise difference between instances."""
        return np.abs(x1 - x2)

    def fit(self, X, y):
        """Fit the ReliefF selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels (must be discrete for classification).

        Returns
        -------
        self : ReliefFSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Encode labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        self.classes_ = le.classes_
        n_classes = len(self.classes_)

        # Scale features for distance computation
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute feature ranges for normalization
        feature_ranges = X_scaled.max(axis=0) - X_scaled.min(axis=0)
        feature_ranges[feature_ranges == 0] = 1.0

        # Determine number of samples to use
        if isinstance(self.n_samples, float):
            m = int(n_samples * self.n_samples)
        else:
            m = min(self.n_samples, n_samples)

        # Random sample selection
        rng = np.random.RandomState(self.random_state)
        sample_indices = rng.choice(n_samples, size=m, replace=False)

        # Initialize feature weights
        weights = np.zeros(n_features)

        # Class priors
        class_counts = np.bincount(y_encoded)
        class_priors = class_counts / n_samples

        if self.algorithm == "relieff":
            weights = self._relieff(
                X_scaled, y_encoded, sample_indices, n_classes,
                class_priors, feature_ranges
            )
        else:  # multisurf
            weights = self._multisurf(
                X_scaled, y_encoded, sample_indices, n_classes,
                class_priors, feature_ranges
            )

        self.feature_importances_ = weights

        # Select top features
        self.ranking_ = np.argsort(weights)[::-1]
        self.selected_features_ = self.ranking_[:self.n_features]
        self._support_mask = np.isin(np.arange(n_features), self.selected_features_)

        return self

    def _relieff(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        n_classes: int,
        class_priors: np.ndarray,
        feature_ranges: np.ndarray,
    ) -> np.ndarray:
        """Standard ReliefF algorithm."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        m = len(sample_indices)

        # Fit nearest neighbors for each class
        class_nns = {}
        for c in range(n_classes):
            class_mask = y == c
            if class_mask.sum() > self.n_neighbors:
                nn = NearestNeighbors(n_neighbors=self.n_neighbors + 1)
                nn.fit(X[class_mask])
                class_nns[c] = (nn, np.where(class_mask)[0])

        for i, idx in enumerate(sample_indices):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Processing sample {i + 1}/{m}")

            x_i = X[idx]
            y_i = y[idx]

            # Find nearest hits (same class)
            if y_i in class_nns:
                nn, class_indices = class_nns[y_i]
                distances, neighbors = nn.kneighbors([x_i])
                # Exclude self if present
                hit_indices = [
                    class_indices[n] for n in neighbors[0]
                    if class_indices[n] != idx
                ][:self.n_neighbors]

                for hit_idx in hit_indices:
                    diff = self._diff(x_i, X[hit_idx]) / feature_ranges
                    weights -= diff / (m * self.n_neighbors)

            # Find nearest misses (different classes)
            for c in range(n_classes):
                if c == y_i or c not in class_nns:
                    continue

                nn, class_indices = class_nns[c]
                distances, neighbors = nn.kneighbors([x_i])
                miss_indices = [class_indices[n] for n in neighbors[0]][:self.n_neighbors]

                prior_weight = class_priors[c] / (1 - class_priors[y_i] + 1e-10)

                for miss_idx in miss_indices:
                    diff = self._diff(x_i, X[miss_idx]) / feature_ranges
                    weights += prior_weight * diff / (m * self.n_neighbors)

        return weights

    def _multisurf(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_indices: np.ndarray,
        n_classes: int,
        class_priors: np.ndarray,
        feature_ranges: np.ndarray,
    ) -> np.ndarray:
        """MultiSURF algorithm with adaptive radius."""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        m = len(sample_indices)

        # Compute pairwise distances
        from sklearn.metrics import pairwise_distances
        distances = pairwise_distances(X)

        for i, idx in enumerate(sample_indices):
            if self.verbose and (i + 1) % 100 == 0:
                print(f"Processing sample {i + 1}/{m}")

            x_i = X[idx]
            y_i = y[idx]

            # Adaptive radius: mean distance - std
            d_i = distances[idx]
            threshold = np.mean(d_i) - np.std(d_i)

            # Find neighbors within threshold
            near_mask = d_i < threshold
            near_mask[idx] = False  # Exclude self

            near_indices = np.where(near_mask)[0]
            if len(near_indices) == 0:
                continue

            # Separate hits and misses
            hit_mask = y[near_indices] == y_i
            hits = near_indices[hit_mask]
            misses = near_indices[~hit_mask]

            # Update weights for hits
            for hit_idx in hits:
                diff = self._diff(x_i, X[hit_idx]) / feature_ranges
                weights -= diff / (m * max(len(hits), 1))

            # Update weights for misses
            for miss_idx in misses:
                y_miss = y[miss_idx]
                prior_weight = class_priors[y_miss] / (1 - class_priors[y_i] + 1e-10)
                diff = self._diff(x_i, X[miss_idx]) / feature_ranges
                weights += prior_weight * diff / (m * max(len(misses), 1))

        return weights

    def transform(self, X) -> np.ndarray:
        """Select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with selected features.
        """
        check_is_fitted(self, "selected_features_")
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, "_support_mask")
        if indices:
            return self.selected_features_
        return self._support_mask

    def get_feature_ranking(self) -> np.ndarray:
        """Get complete feature ranking."""
        check_is_fitted(self, "ranking_")
        return self.ranking_
