"""Hierarchical clustering: Agglomerative and Genie.

AgglomerativeClusterer wraps sklearn with all linkage options.
GenieClusterer wraps genieclust (optional dependency).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array

try:
    import genieclust
    HAS_GENIECLUST = True
except ImportError:
    HAS_GENIECLUST = False


class AgglomerativeClusterer(BaseEstimator, ClusterMixin):
    """Agglomerative hierarchical clustering with multiple linkage options.

    Ward's linkage (default) is the strongest general-purpose option.
    Average linkage is robust. Single linkage is fast but chaining-sensitive.
    Complete linkage produces compact clusters.

    Parameters
    ----------
    n_clusters : int or None, default=2
        Number of clusters. If None, must provide ``distance_threshold``.
    linkage : str, default='ward'
        Linkage criterion: 'ward', 'average', 'complete', 'single'.
    metric : str, default='euclidean'
        Distance metric (only used with non-ward linkage).
    distance_threshold : float or None, default=None
        Distance threshold for stopping. If set, ``n_clusters`` must be None.
    connectivity : array-like or callable or None, default=None
        Connectivity constraints.
    compute_full_tree : bool or 'auto', default='auto'
        Whether to compute the full dendrogram.
    compute_distances : bool, default=False
        Whether to compute distances between clusters.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    n_clusters_ : int
        Number of clusters.
    n_leaves_ : int
        Number of leaves in the dendrogram.
    children_ : ndarray of shape (n_nodes-1, 2)
        Merge history.
    distances_ : ndarray or None
        Distances between merged clusters (if ``compute_distances=True``).
    """

    def __init__(
        self,
        n_clusters: int | None = 2,
        linkage: str = "ward",
        metric: str = "euclidean",
        distance_threshold: float | None = None,
        connectivity: Any = None,
        compute_full_tree: bool | str = "auto",
        compute_distances: bool = False,
    ):
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.metric = metric
        self.distance_threshold = distance_threshold
        self.connectivity = connectivity
        self.compute_full_tree = compute_full_tree
        self.compute_distances = compute_distances

    def fit(self, X: ArrayLike, y=None) -> AgglomerativeClusterer:
        """Fit agglomerative clustering."""
        from sklearn.cluster import AgglomerativeClustering

        X = check_array(X)
        self.model_ = AgglomerativeClustering(
            n_clusters=self.n_clusters,
            linkage=self.linkage,
            metric=self.metric,
            distance_threshold=self.distance_threshold,
            connectivity=self.connectivity,
            compute_full_tree=self.compute_full_tree,
            compute_distances=self.compute_distances,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.n_clusters_ = self.model_.n_clusters_
        self.n_leaves_ = self.model_.n_leaves_
        self.children_ = self.model_.children_
        self.distances_ = getattr(self.model_, "distances_", None)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class GenieClusterer(BaseEstimator, ClusterMixin):
    """Genie clustering: MST-based with Gini index threshold.

    Builds a minimum spanning tree and merges clusters using single linkage,
    but applies a Gini index threshold on cluster sizes to prevent the
    pathological chaining behavior. Consistently outperforms Ward and
    average linkage on standard benchmarks.

    Requires the ``genieclust`` package.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters.
    gini_threshold : float, default=0.3
        Gini index threshold for cluster size inequality. Lower values
        enforce more balanced clusters. 0 = single linkage, 1 = balanced.
    affinity : str, default='euclidean'
        Distance metric.
    exact : bool, default=True
        Use exact (True) or approximate (False) algorithm.
    compute_full_tree : bool, default=True
        Whether to compute the full hierarchy.
    M : int, default=1
        Smoothing factor for the mutual reachability distance.
        M=1 is standard MST; larger M approaches HDBSCAN*-like behavior.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    n_clusters_ : int
        Number of clusters.

    References
    ----------
    Gagolewski, M. (2016). "Genie: A new, fast, and outlier-resistant
    hierarchical clustering algorithm." Information Sciences.
    Gagolewski, M. (2025). Journal of Classification.
    """

    def __init__(
        self,
        n_clusters: int = 2,
        gini_threshold: float = 0.3,
        affinity: str = "euclidean",
        exact: bool = True,
        compute_full_tree: bool = True,
        M: int = 1,
    ):
        self.n_clusters = n_clusters
        self.gini_threshold = gini_threshold
        self.affinity = affinity
        self.exact = exact
        self.compute_full_tree = compute_full_tree
        self.M = M

    def fit(self, X: ArrayLike, y=None) -> GenieClusterer:
        """Fit Genie clustering."""
        if not HAS_GENIECLUST:
            raise ImportError(
                "genieclust is required for GenieClusterer. "
                "Install with: pip install genieclust"
            )

        X = check_array(X, dtype=np.float64)
        self.model_ = genieclust.Genie(
            n_clusters=self.n_clusters,
            gini_threshold=self.gini_threshold,
            affinity=self.affinity,
            exact=self.exact,
            compute_full_tree=self.compute_full_tree,
            M=self.M,
        )
        self.labels_ = self.model_.fit_predict(X)
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
