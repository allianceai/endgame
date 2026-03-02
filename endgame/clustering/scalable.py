"""Scalable and miscellaneous clustering: BIRCH, Mean Shift, FINCH.

BIRCH and Mean Shift wrap sklearn. FINCH wraps finch-clust (optional).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from finch import FINCH as _FINCH
    HAS_FINCH = True
except ImportError:
    HAS_FINCH = False


class BIRCHClusterer(BaseEstimator, ClusterMixin):
    """BIRCH incremental hierarchical clustering.

    Builds a CF-tree (Clustering Feature tree) for incremental clustering.
    Designed for very large datasets or streaming scenarios.

    Parameters
    ----------
    n_clusters : int or None, default=3
        Final number of clusters. If None, the subclusters from the
        CF-tree leaf nodes are returned directly.
    threshold : float, default=0.5
        CF-tree leaf radius threshold.
    branching_factor : int, default=50
        Maximum CF entries per node.
    compute_labels : bool, default=True
        Whether to compute labels for training data.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    subcluster_centers_ : ndarray
        CF-tree subcluster centres.
    n_clusters_ : int
        Number of clusters.
    """

    def __init__(
        self,
        n_clusters: int | None = 3,
        threshold: float = 0.5,
        branching_factor: int = 50,
        compute_labels: bool = True,
    ):
        self.n_clusters = n_clusters
        self.threshold = threshold
        self.branching_factor = branching_factor
        self.compute_labels = compute_labels

    def fit(self, X: ArrayLike, y=None) -> BIRCHClusterer:
        """Fit BIRCH."""
        from sklearn.cluster import Birch

        X = check_array(X)
        self.model_ = Birch(
            n_clusters=self.n_clusters,
            threshold=self.threshold,
            branching_factor=self.branching_factor,
            compute_labels=self.compute_labels,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.subcluster_centers_ = self.model_.subcluster_centers_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new data."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.predict(X)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def partial_fit(self, X: ArrayLike, y=None) -> BIRCHClusterer:
        """Incremental fit on a batch of data."""
        X = check_array(X)
        if not hasattr(self, "model_"):
            from sklearn.cluster import Birch
            self.model_ = Birch(
                n_clusters=self.n_clusters,
                threshold=self.threshold,
                branching_factor=self.branching_factor,
            )
        self.model_.partial_fit(X)
        if hasattr(self.model_, "labels_"):
            self.labels_ = self.model_.labels_
        return self


class MeanShiftClusterer(BaseEstimator, ClusterMixin):
    """Mean Shift mode-finding clustering.

    Non-parametric mode finding via kernel density gradient ascent.
    Automatically determines k by finding density modes.

    Parameters
    ----------
    bandwidth : float or None, default=None
        Kernel bandwidth. If None, estimated automatically.
    bin_seeding : bool, default=False
        Speed up by discretising seed points.
    min_bin_freq : int, default=1
        Minimum bin frequency for seeding.
    cluster_all : bool, default=True
        If False, orphan points get label -1.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Mode locations.
    n_clusters_ : int
        Number of clusters found.
    """

    def __init__(
        self,
        bandwidth: float | None = None,
        bin_seeding: bool = False,
        min_bin_freq: int = 1,
        cluster_all: bool = True,
        n_jobs: int = -1,
    ):
        self.bandwidth = bandwidth
        self.bin_seeding = bin_seeding
        self.min_bin_freq = min_bin_freq
        self.cluster_all = cluster_all
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> MeanShiftClusterer:
        """Fit Mean Shift."""
        from sklearn.cluster import MeanShift

        X = check_array(X)
        self.model_ = MeanShift(
            bandwidth=self.bandwidth,
            bin_seeding=self.bin_seeding,
            min_bin_freq=self.min_bin_freq,
            cluster_all=self.cluster_all,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.cluster_centers_ = self.model_.cluster_centers_
        self.n_clusters_ = len(self.cluster_centers_)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new data."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.predict(X)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class FINCHClusterer(BaseEstimator, ClusterMixin):
    """FINCH: First Integer Neighbour Clustering Hierarchy.

    Zero-parameter clustering that uses first-neighbour relations to
    recursively merge clusters in O(n log n) with O(n) memory. Produces
    a hierarchy of partitions in 4-10 steps.

    Requires the ``finch-clust`` package.

    Parameters
    ----------
    req_clust : int or None, default=None
        Requested number of clusters. If None, returns the partition at
        the first hierarchy level where all points are in the same cluster
        (i.e. the finest reasonable partition).
    distance : str, default='euclidean'
        Distance metric: 'euclidean' or 'cosine'.
    verbose : bool, default=False
        Print hierarchy information.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels at the selected partition level.
    all_partitions_ : ndarray of shape (n_samples, n_levels)
        All hierarchy levels.
    n_clusters_ : int
        Number of clusters at the selected level.
    n_levels_ : int
        Number of hierarchy levels found.

    References
    ----------
    Sarfraz et al., "Efficient Parameter-Free Clustering Using First
    Neighbor Relations", CVPR 2019.
    """

    def __init__(
        self,
        req_clust: int | None = None,
        distance: str = "euclidean",
        verbose: bool = False,
    ):
        self.req_clust = req_clust
        self.distance = distance
        self.verbose = verbose

    def fit(self, X: ArrayLike, y=None) -> FINCHClusterer:
        """Fit FINCH."""
        if not HAS_FINCH:
            raise ImportError(
                "finch-clust is required for FINCHClusterer. "
                "Install with: pip install finch-clust"
                " (not included in standard extras groups)"
            )

        X = check_array(X, dtype=np.float64)

        partitions, num_clust, _ = _FINCH(
            X,
            req_clust=self.req_clust,
            distance=self.distance,
            verbose=self.verbose,
        )

        self.all_partitions_ = partitions
        self.n_levels_ = partitions.shape[1]

        # Select the appropriate partition level
        if self.req_clust is not None:
            # The library returns partitions; last column is the requested k
            self.labels_ = partitions[:, -1]
        else:
            # Use the first (finest) partition
            self.labels_ = partitions[:, 0]

        self.n_clusters_ = len(set(self.labels_))
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
