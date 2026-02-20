"""Discretization utilities for Bayesian Network Classifiers.

Bayesian Networks with discrete CPTs require categorical features.
This module provides strategies for discretizing continuous features
while preserving information relevant for classification.

Strategies:
- MDLP: Minimum Description Length Principle (supervised, optimal for BNCs)
- Equal-width: Fixed-width bins
- Equal-frequency: Quantile-based bins
- K-means: Cluster-based discretization
"""

import numpy as np
from sklearn.base import TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.base import EndgameEstimator


class BayesianDiscretizer(EndgameEstimator, TransformerMixin):
    """
    Discretizes continuous features for Bayesian Network Classifier consumption.

    Supports multiple discretization strategies with automatic handling of
    already-discrete features.

    Parameters
    ----------
    strategy : {'mdlp', 'equal_width', 'equal_freq', 'kmeans'}, default='mdlp'
        Discretization strategy:
        - 'mdlp': Minimum Description Length Principle (supervised, requires y)
        - 'equal_width': Fixed-width bins
        - 'equal_freq': Equal-frequency bins (quantiles)
        - 'kmeans': Cluster-based discretization

    max_bins : int, default=10
        Maximum number of bins per feature.

    min_samples_bin : int, default=5
        Minimum samples per bin (affects MDLP stopping criterion).

    discrete_features : array-like of int | 'auto' | None, default='auto'
        Which features are already discrete:
        - 'auto': Detect based on dtype and unique values
        - list of int: Indices of discrete features
        - None: Treat all features as continuous

    max_unique_continuous : int, default=20
        If 'auto', features with <= this many unique values are considered discrete.

    random_state : int, optional
        Random seed for kmeans initialization.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    n_bins_ : np.ndarray
        Number of bins for each feature.

    bin_edges_ : list[np.ndarray]
        Bin edges for each continuous feature.

    discrete_features_ : np.ndarray
        Boolean mask of discrete features.

    feature_names_in_ : np.ndarray
        Feature names (if input was DataFrame).

    Examples
    --------
    >>> from endgame.preprocessing import BayesianDiscretizer
    >>> disc = BayesianDiscretizer(strategy='mdlp')
    >>> X_disc = disc.fit_transform(X_train, y_train)
    >>> X_test_disc = disc.transform(X_test)
    """

    def __init__(
        self,
        strategy: str = 'mdlp',
        max_bins: int = 10,
        min_samples_bin: int = 5,
        discrete_features: str | list[int] | None = 'auto',
        max_unique_continuous: int = 20,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.strategy = strategy
        self.max_bins = max_bins
        self.min_samples_bin = min_samples_bin
        self.discrete_features = discrete_features
        self.max_unique_continuous = max_unique_continuous

        # Set during fit
        self.n_bins_: np.ndarray | None = None
        self.bin_edges_: list[np.ndarray] | None = None
        self.discrete_features_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None

    def fit(self, X, y=None, **fit_params) -> 'BayesianDiscretizer':
        """
        Fit the discretizer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values. Required for 'mdlp' strategy.

        Returns
        -------
        self
        """
        # Validate input
        if self.strategy == 'mdlp' and y is None:
            raise ValueError("MDLP strategy requires y to be provided")

        if y is not None:
            X, y = check_X_y(X, y)
        else:
            X = check_array(X)

        self.n_features_in_ = X.shape[1]

        # Detect discrete features
        self.discrete_features_ = self._detect_discrete(X)

        # Initialize storage
        self.n_bins_ = np.zeros(self.n_features_in_, dtype=int)
        self.bin_edges_ = [None] * self.n_features_in_

        # Fit discretizer for each continuous feature
        for i in range(self.n_features_in_):
            if self.discrete_features_[i]:
                # Already discrete - just record cardinality
                self.n_bins_[i] = len(np.unique(X[:, i]))
                continue

            if self.strategy == 'mdlp':
                edges = self._fit_mdlp(X[:, i], y)
            elif self.strategy == 'equal_width':
                edges = self._fit_equal_width(X[:, i])
            elif self.strategy == 'equal_freq':
                edges = self._fit_equal_freq(X[:, i])
            elif self.strategy == 'kmeans':
                edges = self._fit_kmeans(X[:, i])
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            self.bin_edges_[i] = edges
            self.n_bins_[i] = len(edges) - 1

        self._log(f"Fitted discretizer: {sum(~self.discrete_features_)} continuous features, "
                 f"avg {np.mean(self.n_bins_):.1f} bins")

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """
        Transform continuous features to discrete.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        np.ndarray
            Discretized data with integer values.
        """
        check_is_fitted(self, ['bin_edges_', 'discrete_features_'])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        X_disc = np.zeros_like(X, dtype=int)

        for i in range(self.n_features_in_):
            if self.discrete_features_[i]:
                # Already discrete
                X_disc[:, i] = X[:, i].astype(int)
            else:
                # Digitize using bin edges
                edges = self.bin_edges_[i]
                # np.digitize returns 1-indexed bins, subtract 1
                X_disc[:, i] = np.digitize(X[:, i], edges[1:-1])
                # Clip to valid range
                X_disc[:, i] = np.clip(X_disc[:, i], 0, self.n_bins_[i] - 1)

        return X_disc

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def _detect_discrete(self, X: np.ndarray) -> np.ndarray:
        """Detect which features are already discrete."""
        n_features = X.shape[1]
        discrete = np.zeros(n_features, dtype=bool)

        if self.discrete_features is None:
            # All continuous
            return discrete

        if isinstance(self.discrete_features, str) and self.discrete_features == 'auto':
            for i in range(n_features):
                col = X[:, i]

                # Check if integer-valued
                is_integer = np.allclose(col, col.astype(int))

                # Check cardinality
                n_unique = len(np.unique(col))

                discrete[i] = is_integer and n_unique <= self.max_unique_continuous

        elif isinstance(self.discrete_features, (list, np.ndarray)):
            for idx in self.discrete_features:
                if 0 <= idx < n_features:
                    discrete[idx] = True

        return discrete

    def _fit_mdlp(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Fit using Minimum Description Length Principle.

        MDLP finds cut points that minimize the entropy of the resulting
        partition while penalizing model complexity.
        """
        # Sort data
        sort_idx = np.argsort(x)
        x_sorted = x[sort_idx]
        y_sorted = y[sort_idx]

        # Find candidate cut points (midpoints between different labels)
        candidates = []
        for i in range(len(x_sorted) - 1):
            if y_sorted[i] != y_sorted[i + 1] or x_sorted[i] != x_sorted[i + 1]:
                # Potential cut point
                if x_sorted[i] != x_sorted[i + 1]:
                    midpoint = (x_sorted[i] + x_sorted[i + 1]) / 2
                    candidates.append((i + 1, midpoint))  # (index, value)

        if not candidates:
            # No cut points found - return single bin
            return np.array([-np.inf, np.inf])

        # Recursive partitioning
        cuts = self._mdlp_recursive(x_sorted, y_sorted, candidates, 0, len(x_sorted))

        # Limit number of bins
        if len(cuts) > self.max_bins - 1:
            # Keep the most informative cuts
            cuts = sorted(cuts)[:self.max_bins - 1]

        # Build edges
        edges = [-np.inf] + sorted(cuts) + [np.inf]
        return np.array(edges)

    def _mdlp_recursive(
        self,
        x: np.ndarray,
        y: np.ndarray,
        candidates: list[tuple[int, float]],
        start: int,
        end: int,
    ) -> list[float]:
        """Recursive MDLP partitioning."""
        if end - start < 2 * self.min_samples_bin:
            return []

        n = end - start
        y_subset = y[start:end]

        # Current entropy
        current_entropy = self._entropy(y_subset)

        # Find best cut
        best_cut = None
        best_gain = 0
        best_idx = None

        for idx, cut_val in candidates:
            if idx <= start or idx >= end:
                continue

            # Split
            left_y = y[start:idx]
            right_y = y[idx:end]

            if len(left_y) < self.min_samples_bin or len(right_y) < self.min_samples_bin:
                continue

            # Weighted entropy after split
            p_left = len(left_y) / n
            p_right = len(right_y) / n
            new_entropy = p_left * self._entropy(left_y) + p_right * self._entropy(right_y)

            gain = current_entropy - new_entropy

            if gain > best_gain:
                best_gain = gain
                best_cut = cut_val
                best_idx = idx

        if best_cut is None:
            return []

        # MDLP stopping criterion
        if not self._mdlp_accept(y_subset, y[start:best_idx], y[best_idx:end], best_gain):
            return []

        # Recurse
        left_cuts = self._mdlp_recursive(x, y, candidates, start, best_idx)
        right_cuts = self._mdlp_recursive(x, y, candidates, best_idx, end)

        return left_cuts + [best_cut] + right_cuts

    def _mdlp_accept(
        self,
        y_all: np.ndarray,
        y_left: np.ndarray,
        y_right: np.ndarray,
        gain: float,
    ) -> bool:
        """MDLP stopping criterion (Fayyad & Irani, 1993)."""
        n = len(y_all)
        k = len(np.unique(y_all))
        k1 = len(np.unique(y_left))
        k2 = len(np.unique(y_right))

        ent = self._entropy(y_all)
        ent1 = self._entropy(y_left)
        ent2 = self._entropy(y_right)

        # MDL criterion
        delta = np.log2(3**k - 2) - (k * ent - k1 * ent1 - k2 * ent2)
        threshold = (np.log2(n - 1) + delta) / n

        return gain > threshold

    def _entropy(self, y: np.ndarray) -> float:
        """Compute entropy."""
        if len(y) == 0:
            return 0.0

        _, counts = np.unique(y, return_counts=True)
        probs = counts / len(y)
        probs = probs[probs > 0]

        return -np.sum(probs * np.log2(probs))

    def _fit_equal_width(self, x: np.ndarray) -> np.ndarray:
        """Fit equal-width bins."""
        x_min, x_max = x.min(), x.max()

        if x_min == x_max:
            return np.array([-np.inf, np.inf])

        n_bins = min(self.max_bins, len(np.unique(x)))
        edges = np.linspace(x_min, x_max, n_bins + 1)
        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges

    def _fit_equal_freq(self, x: np.ndarray) -> np.ndarray:
        """Fit equal-frequency (quantile) bins."""
        n_bins = min(self.max_bins, len(np.unique(x)))

        # Compute quantiles
        percentiles = np.linspace(0, 100, n_bins + 1)
        edges = np.percentile(x, percentiles)

        # Remove duplicates
        edges = np.unique(edges)

        if len(edges) < 2:
            return np.array([-np.inf, np.inf])

        edges[0] = -np.inf
        edges[-1] = np.inf

        return edges

    def _fit_kmeans(self, x: np.ndarray) -> np.ndarray:
        """Fit bins using k-means clustering."""
        try:
            from sklearn.cluster import KMeans
        except ImportError:
            raise ImportError("kmeans strategy requires sklearn")

        n_bins = min(self.max_bins, len(np.unique(x)))

        if n_bins <= 1:
            return np.array([-np.inf, np.inf])

        # Fit k-means
        x_reshaped = x.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_bins, random_state=self.random_state, n_init=10)
        kmeans.fit(x_reshaped)

        # Get cluster centers and sort
        centers = np.sort(kmeans.cluster_centers_.flatten())

        # Compute edges as midpoints between centers
        edges = [-np.inf]
        for i in range(len(centers) - 1):
            edges.append((centers[i] + centers[i + 1]) / 2)
        edges.append(np.inf)

        return np.array(edges)

    def get_feature_names_out(self, input_features: list[str] | None = None) -> list[str]:
        """Get output feature names."""
        check_is_fitted(self, ['n_features_in_'])

        if input_features is None:
            input_features = [f"x{i}" for i in range(self.n_features_in_)]

        # Output names are same as input (just discretized)
        return list(input_features)

    def inverse_transform(self, X_disc: np.ndarray) -> np.ndarray:
        """
        Approximate inverse transform (returns bin centers).

        Note: This is lossy - the original continuous values cannot be
        recovered exactly.

        Parameters
        ----------
        X_disc : np.ndarray
            Discretized data.

        Returns
        -------
        np.ndarray
            Approximate continuous values (bin centers).
        """
        check_is_fitted(self, ['bin_edges_', 'discrete_features_'])
        X_disc = check_array(X_disc)

        X_cont = np.zeros_like(X_disc, dtype=float)

        for i in range(self.n_features_in_):
            if self.discrete_features_[i]:
                # Already discrete - no change
                X_cont[:, i] = X_disc[:, i]
            else:
                edges = self.bin_edges_[i]
                # Replace -inf and inf with reasonable bounds for center calc
                finite_edges = edges.copy()
                finite_mask = ~np.isinf(finite_edges)

                if np.sum(finite_mask) >= 2:
                    data_range = finite_edges[finite_mask].max() - finite_edges[finite_mask].min()
                    if finite_edges[0] == -np.inf:
                        finite_edges[0] = finite_edges[1] - data_range * 0.1
                    if finite_edges[-1] == np.inf:
                        finite_edges[-1] = finite_edges[-2] + data_range * 0.1

                # Bin centers
                centers = (finite_edges[:-1] + finite_edges[1:]) / 2

                # Map bin indices to centers
                for j, bin_idx in enumerate(X_disc[:, i].astype(int)):
                    if 0 <= bin_idx < len(centers):
                        X_cont[j, i] = centers[bin_idx]
                    else:
                        X_cont[j, i] = centers[min(bin_idx, len(centers) - 1)]

        return X_cont
