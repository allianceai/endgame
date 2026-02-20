"""Correlation-based feature selection."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class CorrelationSelector(TransformerMixin, BaseEstimator):
    """Remove highly correlated features.

    Identifies and removes features that are highly correlated with other
    features, keeping only one from each correlated group.

    Parameters
    ----------
    threshold : float, default=0.95
        Correlation threshold. Features with correlation above this
        are considered redundant.

    method : str, default='pearson'
        Correlation method:
        - 'pearson': Pearson correlation (linear)
        - 'spearman': Spearman rank correlation (monotonic)
        - 'kendall': Kendall tau correlation (ordinal)

    keep : str, default='first'
        Which feature to keep from correlated pairs:
        - 'first': Keep the first feature encountered
        - 'variance': Keep the feature with higher variance
        - 'target_corr': Keep the feature with higher target correlation

    Attributes
    ----------
    features_to_drop_ : list
        Indices of features to drop.

    selected_features_ : ndarray
        Indices of selected features.

    correlation_matrix_ : ndarray
        Computed correlation matrix.

    Example
    -------
    >>> from endgame.feature_selection import CorrelationSelector
    >>> selector = CorrelationSelector(threshold=0.90)
    >>> X_reduced = selector.fit_transform(X)
    """

    def __init__(
        self,
        threshold: float = 0.95,
        method: Literal["pearson", "spearman", "kendall"] = "pearson",
        keep: Literal["first", "variance", "target_corr"] = "first",
    ):
        self.threshold = threshold
        self.method = method
        self.keep = keep

    def _compute_correlation(self, X: np.ndarray) -> np.ndarray:
        """Compute correlation matrix."""
        if self.method == "pearson":
            return np.corrcoef(X, rowvar=False)
        elif self.method == "spearman":
            from scipy.stats import spearmanr
            corr, _ = spearmanr(X)
            return corr if corr.ndim == 2 else np.array([[1.0]])
        elif self.method == "kendall":
            from scipy.stats import kendalltau
            n_features = X.shape[1]
            corr = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    tau, _ = kendalltau(X[:, i], X[:, j])
                    corr[i, j] = tau
                    corr[j, i] = tau
            return corr
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def fit(self, X, y=None):
        """Fit the correlation selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target values (required if keep='target_corr').

        Returns
        -------
        self : CorrelationSelector
        """
        X = check_array(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Compute correlation matrix
        self.correlation_matrix_ = self._compute_correlation(X)
        corr = np.abs(self.correlation_matrix_)

        # Handle NaN correlations
        corr = np.nan_to_num(corr, nan=0.0)

        # Compute feature scores for tie-breaking
        if self.keep == "variance":
            feature_scores = np.var(X, axis=0)
        elif self.keep == "target_corr" and y is not None:
            y = np.asarray(y).ravel()
            feature_scores = np.abs([np.corrcoef(X[:, i], y)[0, 1] for i in range(n_features)])
            feature_scores = np.nan_to_num(feature_scores, nan=0.0)
        else:
            feature_scores = np.arange(n_features)[::-1]  # Prefer earlier features

        # Find features to drop
        features_to_drop = set()

        for i in range(n_features):
            if i in features_to_drop:
                continue

            for j in range(i + 1, n_features):
                if j in features_to_drop:
                    continue

                if corr[i, j] > self.threshold:
                    # Drop the feature with lower score
                    if feature_scores[i] >= feature_scores[j]:
                        features_to_drop.add(j)
                    else:
                        features_to_drop.add(i)
                        break  # Move to next i

        self.features_to_drop_ = sorted(features_to_drop)
        self.selected_features_ = np.array([
            i for i in range(n_features) if i not in features_to_drop
        ])
        self._support_mask = np.isin(np.arange(n_features), self.selected_features_)

        return self

    def transform(self, X) -> np.ndarray:
        """Remove correlated features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with correlated features removed.
        """
        check_is_fitted(self, "selected_features_")
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, "_support_mask")
        if indices:
            return self.selected_features_
        return self._support_mask

    def get_correlated_pairs(self) -> list:
        """Get pairs of highly correlated features.

        Returns
        -------
        pairs : list of tuples
            Each tuple is (feature_i, feature_j, correlation).
        """
        check_is_fitted(self, "correlation_matrix_")
        corr = np.abs(self.correlation_matrix_)
        n = corr.shape[0]

        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                if corr[i, j] > self.threshold:
                    pairs.append((i, j, self.correlation_matrix_[i, j]))

        return sorted(pairs, key=lambda x: -abs(x[2]))
