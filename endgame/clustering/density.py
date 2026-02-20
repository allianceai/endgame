"""Density-based clustering: DBSCAN, HDBSCAN, OPTICS, DPC.

DBSCAN, HDBSCAN, and OPTICS wrap sklearn. DPC (Density Peaks Clustering)
is implemented from scratch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array


class DBSCANClusterer(BaseEstimator, ClusterMixin):
    """DBSCAN density-based clustering with competition defaults.

    Finds arbitrary-shaped clusters and labels noise points as -1.

    Parameters
    ----------
    eps : float, default=0.5
        Neighbourhood radius.
    min_samples : int, default=5
        Minimum samples in a neighbourhood for a core point.
    metric : str, default='euclidean'
        Distance metric.
    algorithm : str, default='auto'
        Nearest neighbours algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'.
    leaf_size : int, default=30
        Leaf size for tree-based algorithms.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels (-1 for noise).
    core_sample_indices_ : ndarray
        Indices of core samples.
    n_clusters_ : int
        Number of clusters found (excluding noise).
    """

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        algorithm: str = "auto",
        leaf_size: int = 30,
        n_jobs: int = -1,
    ):
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> DBSCANClusterer:
        """Fit DBSCAN."""
        from sklearn.cluster import DBSCAN

        X = check_array(X)
        self.model_ = DBSCAN(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.core_sample_indices_ = self.model_.core_sample_indices_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class HDBSCANClusterer(BaseEstimator, ClusterMixin):
    """HDBSCAN hierarchical density-based clustering.

    Runs DBSCAN across all eps values simultaneously via mutual reachability
    MST, extracting the most stable clusters. Only real param is
    ``min_cluster_size``. Handles variable-density clusters.

    Uses sklearn's HDBSCAN (>=1.3) with fallback to the ``hdbscan`` package.

    Parameters
    ----------
    min_cluster_size : int, default=15
        Minimum cluster size.
    min_samples : int or None, default=None
        Core distance samples. Defaults to ``min_cluster_size``.
    metric : str, default='euclidean'
        Distance metric.
    cluster_selection_method : str, default='eom'
        Cluster extraction: 'eom' (Excess of Mass) or 'leaf'.
    cluster_selection_epsilon : float, default=0.0
        Distance threshold for merging clusters.
    alpha : float, default=1.0
        Mutual reachability smoothing.
    allow_single_cluster : bool, default=False
        Whether to allow a single-cluster result.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels (-1 for noise).
    probabilities_ : ndarray of shape (n_samples,)
        Cluster membership probabilities.
    n_clusters_ : int
        Number of clusters found.
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int | None = None,
        metric: str = "euclidean",
        cluster_selection_method: str = "eom",
        cluster_selection_epsilon: float = 0.0,
        alpha: float = 1.0,
        allow_single_cluster: bool = False,
        n_jobs: int = -1,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.metric = metric
        self.cluster_selection_method = cluster_selection_method
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.alpha = alpha
        self.allow_single_cluster = allow_single_cluster
        self.n_jobs = n_jobs

    def _get_hdbscan(self):
        """Get HDBSCAN class, preferring sklearn >= 1.3."""
        try:
            from sklearn.cluster import HDBSCAN
            return HDBSCAN, "sklearn"
        except ImportError:
            pass

        try:
            import hdbscan
            return hdbscan.HDBSCAN, "hdbscan"
        except ImportError:
            raise ImportError(
                "HDBSCAN requires sklearn >= 1.3 or the hdbscan package. "
                "Install with: pip install hdbscan"
            )

    def fit(self, X: ArrayLike, y=None) -> HDBSCANClusterer:
        """Fit HDBSCAN."""
        X = check_array(X)
        HDBSCAN, backend = self._get_hdbscan()

        params = {
            "min_cluster_size": self.min_cluster_size,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "cluster_selection_method": self.cluster_selection_method,
            "alpha": self.alpha,
            "allow_single_cluster": self.allow_single_cluster,
        }

        if backend == "sklearn":
            params["cluster_selection_epsilon"] = self.cluster_selection_epsilon
            params["n_jobs"] = self.n_jobs
        else:
            # hdbscan package uses slightly different params
            params["cluster_selection_epsilon"] = self.cluster_selection_epsilon

        self.model_ = HDBSCAN(**params)
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.probabilities_ = getattr(self.model_, "probabilities_", np.ones(len(X)))
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class OPTICSClusterer(BaseEstimator, ClusterMixin):
    """OPTICS ordering-based clustering.

    Produces a reachability plot and extracts clusters, generalizing DBSCAN
    to handle varying density.

    Parameters
    ----------
    min_samples : int or float, default=5
        Core distance parameter.
    max_eps : float, default=inf
        Maximum neighbourhood radius.
    metric : str, default='minkowski'
        Distance metric.
    p : float, default=2
        Minkowski power (2 = Euclidean).
    cluster_method : str, default='xi'
        Extraction method: 'xi' or 'dbscan'.
    xi : float, default=0.05
        Steepness threshold for xi extraction.
    min_cluster_size : int or float or None, default=None
        Minimum cluster size for extraction.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels (-1 for noise).
    reachability_ : ndarray of shape (n_samples,)
        Reachability distances.
    ordering_ : ndarray of shape (n_samples,)
        OPTICS ordering.
    n_clusters_ : int
        Number of clusters found.
    """

    def __init__(
        self,
        min_samples: int | float = 5,
        max_eps: float = np.inf,
        metric: str = "minkowski",
        p: float = 2,
        cluster_method: str = "xi",
        xi: float = 0.05,
        min_cluster_size: int | float | None = None,
        n_jobs: int = -1,
    ):
        self.min_samples = min_samples
        self.max_eps = max_eps
        self.metric = metric
        self.p = p
        self.cluster_method = cluster_method
        self.xi = xi
        self.min_cluster_size = min_cluster_size
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> OPTICSClusterer:
        """Fit OPTICS."""
        from sklearn.cluster import OPTICS

        X = check_array(X)
        self.model_ = OPTICS(
            min_samples=self.min_samples,
            max_eps=self.max_eps,
            metric=self.metric,
            p=self.p,
            cluster_method=self.cluster_method,
            xi=self.xi,
            min_cluster_size=self.min_cluster_size,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.reachability_ = self.model_.reachability_
        self.ordering_ = self.model_.ordering_
        self.n_clusters_ = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class DensityPeaksClusterer(BaseEstimator, ClusterMixin):
    """Density Peaks Clustering (DPC).

    Cluster centres are points with simultaneously high local density (rho)
    and large distance to any denser point (delta). Points are assigned by
    following the chain to the nearest denser neighbour.

    Parameters
    ----------
    n_clusters : int or None, default=None
        Number of clusters. If None, auto-select from the decision graph
        using ``gamma_threshold``.
    percent : float, default=2.0
        Percentage of data to use as cutoff distance (d_c) for density
        estimation. E.g. 2.0 means d_c is the distance at the 2nd
        percentile of all pairwise distances.
    gamma_threshold : float or None, default=None
        If ``n_clusters`` is None, points with ``rho * delta`` above this
        threshold are chosen as centres. If None, uses Otsu-like
        thresholding on gamma values.
    metric : str, default='euclidean'
        Distance metric.
    random_state : int or None, default=None
        Random seed (for tie-breaking).

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    rho_ : ndarray of shape (n_samples,)
        Local densities.
    delta_ : ndarray of shape (n_samples,)
        Distance to nearest denser point.
    centers_ : ndarray of shape (n_centers,)
        Indices of cluster centres.
    n_clusters_ : int
        Number of clusters found.

    References
    ----------
    Rodriguez & Laio, "Clustering by fast search and find of density peaks",
    Science, 2014.
    """

    def __init__(
        self,
        n_clusters: int | None = None,
        percent: float = 2.0,
        gamma_threshold: float | None = None,
        metric: str = "euclidean",
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.percent = percent
        self.gamma_threshold = gamma_threshold
        self.metric = metric
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> DensityPeaksClusterer:
        """Fit DPC."""
        from sklearn.metrics import pairwise_distances

        X = check_array(X, dtype=np.float64)
        n = X.shape[0]

        # Compute pairwise distances
        dist = pairwise_distances(X, metric=self.metric)

        # Compute cutoff distance d_c
        tri_idx = np.triu_indices(n, k=1)
        all_dists = dist[tri_idx]
        d_c = np.percentile(all_dists, self.percent)
        d_c = max(d_c, 1e-10)

        # Compute local density rho (Gaussian kernel)
        rho = np.sum(np.exp(-(dist / d_c) ** 2), axis=1) - 1.0
        self.rho_ = rho

        # Compute delta: distance to nearest point with higher density
        delta = np.full(n, np.inf)
        nearest_denser = np.full(n, -1, dtype=int)

        # Sort by density descending
        order = np.argsort(-rho)
        for i_rank in range(1, n):
            idx = order[i_rank]
            # Among all points with higher density, find nearest
            higher_mask = rho > rho[idx]
            if not higher_mask.any():
                delta[idx] = dist[idx].max()
                continue
            higher_dists = dist[idx][higher_mask]
            nearest_pos = np.argmin(higher_dists)
            delta[idx] = higher_dists[nearest_pos]
            nearest_denser[idx] = np.where(higher_mask)[0][nearest_pos]

        # Point with highest density: delta = max distance
        top = order[0]
        delta[top] = dist[top].max()
        self.delta_ = delta

        # Select centres
        gamma = rho * delta

        if self.n_clusters is not None:
            # Pick top n_clusters by gamma
            center_indices = np.argsort(-gamma)[: self.n_clusters]
        else:
            # Auto-select via threshold
            if self.gamma_threshold is not None:
                center_indices = np.where(gamma > self.gamma_threshold)[0]
            else:
                # Otsu-like: threshold at mean + 2*std of gamma
                threshold = np.mean(gamma) + 2 * np.std(gamma)
                center_indices = np.where(gamma > threshold)[0]

            if len(center_indices) < 1:
                center_indices = np.array([order[0]])

        self.centers_ = center_indices
        self.n_clusters_ = len(center_indices)

        # Assign labels: follow nearest denser neighbour chain to a centre
        labels = np.full(n, -1, dtype=int)
        centre_set = set(center_indices.tolist())
        for i, idx in enumerate(center_indices):
            labels[idx] = i

        # Assign in order of decreasing density
        for idx in order:
            if labels[idx] >= 0:
                continue
            # Walk up the chain
            chain = [idx]
            current = idx
            while labels[current] < 0 and nearest_denser[current] >= 0:
                current = nearest_denser[current]
                chain.append(current)
            if labels[current] >= 0:
                for c in chain:
                    labels[c] = labels[current]
            else:
                # Assign to nearest centre
                centre_dists = dist[idx][center_indices]
                labels[idx] = np.argmin(centre_dists)

        self.labels_ = labels
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
