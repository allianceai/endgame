"""AutoCluster: automatic clustering method selection based on data properties.

Selects the best clustering algorithm and parameters based on dataset
characteristics (n, d, expected k, noise detection needs).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


class AutoCluster(BaseEstimator, ClusterMixin):
    """Automatic clustering with method selection based on data properties.

    Selects the best clustering algorithm based on:
    - Dataset size (n)
    - Dimensionality (d)
    - Whether k is specified
    - Whether noise detection is needed

    Parameters
    ----------
    n_clusters : int or 'auto', default='auto'
        Number of clusters. 'auto' uses algorithms that determine k
        automatically (HDBSCAN, k*-Means, or GMM with BIC).
    detect_noise : bool, default=False
        Whether to detect noise/outlier points (label -1). If True,
        prefers density-based methods (HDBSCAN, DBSCAN).
    prefer : str or None, default=None
        Override automatic selection: 'centroid', 'density',
        'hierarchical', 'distribution', 'spectral'. If None, auto-selects.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    **kwargs
        Additional parameters passed to the selected clusterer.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    selected_method_ : str
        Name of the selected algorithm.
    clusterer_ : BaseEstimator
        The fitted clusterer instance.
    n_clusters_ : int
        Number of clusters found.

    Examples
    --------
    >>> from endgame.clustering import AutoCluster
    >>> ac = AutoCluster(n_clusters='auto', detect_noise=True)
    >>> labels = ac.fit_predict(X)
    >>> print(f"Selected: {ac.selected_method_}, k={ac.n_clusters_}")
    """

    def __init__(
        self,
        n_clusters: int | str = "auto",
        detect_noise: bool = False,
        prefer: str | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        self.n_clusters = n_clusters
        self.detect_noise = detect_noise
        self.prefer = prefer
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs

    def _log(self, msg: str):
        if self.verbose:
            print(f"[AutoCluster] {msg}")

    def _select_method(self, n: int, d: int) -> tuple[str, dict]:
        """Select clustering method based on data characteristics.

        Returns (method_name, params_dict).
        """
        k = self.n_clusters
        auto_k = (k == "auto")

        if self.prefer is not None:
            return self._select_from_preference(self.prefer, n, d, k, auto_k)

        # Noise detection requested -> density-based
        if self.detect_noise:
            if auto_k:
                return "hdbscan", {"min_cluster_size": max(15, n // 100)}
            else:
                return "dbscan", {"eps": 0.5, "min_samples": 5}

        # Auto-k requested
        if auto_k:
            if n > 50000:
                # Large dataset: use k*-means (fast, auto-k)
                return "kstar_means", {"k_max": min(50, n // 100)}
            elif n > 10000:
                return "hdbscan", {"min_cluster_size": max(15, n // 100)}
            else:
                # Small-medium: k*-means for auto-k
                return "kstar_means", {"k_max": min(30, n // 10)}

        # k is specified
        k = int(k)

        # Large dataset
        if n > 100000:
            return "minibatch_kmeans", {"n_clusters": k}

        # Small dataset with low d -> spectral can shine
        if n < 5000 and d < 50:
            return "kmeans", {"n_clusters": k}

        # Default: K-Means
        return "kmeans", {"n_clusters": k}

    def _select_from_preference(
        self, prefer: str, n: int, d: int, k: Any, auto_k: bool
    ) -> tuple[str, dict]:
        """Select based on user preference category."""
        if prefer == "centroid":
            if auto_k:
                return "kstar_means", {}
            return "kmeans", {"n_clusters": int(k)}

        if prefer == "density":
            if auto_k:
                return "hdbscan", {"min_cluster_size": max(15, n // 100)}
            return "dbscan", {"eps": 0.5, "min_samples": 5}

        if prefer == "hierarchical":
            if auto_k:
                return "agglomerative", {"n_clusters": None, "distance_threshold": 1.0}
            return "agglomerative", {"n_clusters": int(k)}

        if prefer == "distribution":
            if auto_k:
                return "gmm_auto", {}
            return "gmm", {"n_components": int(k)}

        if prefer == "spectral":
            k_val = max(2, int(k)) if not auto_k else 8
            return "spectral", {"n_clusters": k_val}

        raise ValueError(
            f"Unknown preference '{prefer}'. "
            "Expected: 'centroid', 'density', 'hierarchical', 'distribution', 'spectral'."
        )

    def _build_clusterer(self, method: str, params: dict) -> BaseEstimator:
        """Instantiate the selected clusterer."""
        # Merge user kwargs (overrides auto-selected params)
        merged = {**params}
        merged.update(self.kwargs)

        # Add random_state if the clusterer supports it
        if "random_state" not in merged and self.random_state is not None:
            merged["random_state"] = self.random_state

        if method == "kmeans":
            from endgame.clustering.centroid import KMeansClusterer
            return KMeansClusterer(**merged)

        if method == "minibatch_kmeans":
            from endgame.clustering.centroid import MiniBatchKMeansClusterer
            return MiniBatchKMeansClusterer(**merged)

        if method == "kstar_means":
            from endgame.clustering.centroid import KStarMeansClusterer
            return KStarMeansClusterer(**merged)

        if method == "dbscan":
            merged.pop("random_state", None)
            from endgame.clustering.density import DBSCANClusterer
            return DBSCANClusterer(**merged)

        if method == "hdbscan":
            merged.pop("random_state", None)
            from endgame.clustering.density import HDBSCANClusterer
            return HDBSCANClusterer(**merged)

        if method == "agglomerative":
            merged.pop("random_state", None)
            from endgame.clustering.hierarchical import AgglomerativeClusterer
            return AgglomerativeClusterer(**merged)

        if method == "gmm":
            from endgame.clustering.distribution import GaussianMixtureClusterer
            return GaussianMixtureClusterer(**merged)

        if method == "gmm_auto":
            # Will select n_components via BIC in fit()
            from endgame.clustering.distribution import GaussianMixtureClusterer
            return GaussianMixtureClusterer(**merged)

        if method == "spectral":
            from endgame.clustering.graph import SpectralClusterer
            return SpectralClusterer(**merged)

        raise ValueError(f"Unknown method: {method}")

    def fit(self, X: ArrayLike, y=None) -> AutoCluster:
        """Fit the auto-selected clusterer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : ignored

        Returns
        -------
        self
        """
        X = check_array(X)
        n, d = X.shape

        method, params = self._select_method(n, d)
        self.selected_method_ = method
        self._log(f"Selected method: {method} (n={n}, d={d})")

        # Special case: GMM with auto k selection
        if method == "gmm_auto":
            from endgame.clustering.distribution import GaussianMixtureClusterer
            gmm = GaussianMixtureClusterer(
                random_state=self.random_state, **self.kwargs
            )
            best_k = gmm.select_n_components(X)
            self._log(f"GMM BIC selected k={best_k}")
            self.clusterer_ = GaussianMixtureClusterer(
                n_components=best_k,
                random_state=self.random_state,
                **self.kwargs,
            )
        else:
            self.clusterer_ = self._build_clusterer(method, params)

        self.clusterer_.fit(X)
        self.labels_ = self.clusterer_.labels_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = d
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new data (if supported).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        if not hasattr(self, "clusterer_"):
            raise RuntimeError("AutoCluster has not been fitted.")
        if hasattr(self.clusterer_, "predict"):
            return self.clusterer_.predict(X)
        raise NotImplementedError(
            f"{self.selected_method_} does not support predict() on new data. "
            "Use fit_predict() instead."
        )

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
