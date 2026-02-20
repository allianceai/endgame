"""Graph/spectral clustering: Spectral Clustering and Affinity Propagation.

Both wrap sklearn with competition-tuned defaults.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


class SpectralClusterer(BaseEstimator, ClusterMixin):
    """Spectral clustering via graph Laplacian eigenvectors.

    Constructs a similarity graph, computes eigenvectors of the graph
    Laplacian, then runs k-means in the spectral embedding. Excels at
    non-convex clusters (concentric circles, spirals).

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    affinity : str, default='rbf'
        Similarity measure: 'rbf', 'nearest_neighbors', 'precomputed'.
    gamma : float or None, default=None
        RBF kernel bandwidth. If None, uses 1/n_features.
    n_neighbors : int, default=10
        Number of neighbours for 'nearest_neighbors' affinity.
    n_init : int, default=10
        k-means initializations in spectral space.
    assign_labels : str, default='kmeans'
        Label assignment: 'kmeans' or 'discretize'.
    random_state : int or None, default=None
        Random seed.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Computed affinity matrix.
    n_clusters_ : int
        Number of clusters.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        affinity: str = "rbf",
        gamma: float | None = None,
        n_neighbors: int = 10,
        n_init: int = 10,
        assign_labels: str = "kmeans",
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma
        self.n_neighbors = n_neighbors
        self.n_init = n_init
        self.assign_labels = assign_labels
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> SpectralClusterer:
        """Fit spectral clustering."""
        from sklearn.cluster import SpectralClustering

        X = check_array(X)
        # sklearn requires gamma to be a positive float (validated for all affinities)
        gamma = self.gamma
        if gamma is None:
            gamma = 1.0 / X.shape[1]
        self.model_ = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity=self.affinity,
            gamma=gamma,
            n_neighbors=self.n_neighbors,
            n_init=self.n_init,
            assign_labels=self.assign_labels,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.labels_ = self.model_.fit_predict(X)
        self.affinity_matrix_ = self.model_.affinity_matrix_
        self.n_clusters_ = self.n_clusters
        self.n_features_in_ = X.shape[1]
        return self

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_


class AffinityPropagationClusterer(BaseEstimator, ClusterMixin):
    """Affinity Propagation clustering via message passing.

    Simultaneously chooses exemplars and assigns points via responsibility
    and availability messages. No k required.

    Parameters
    ----------
    damping : float, default=0.5
        Damping factor (0.5 to 1). Higher = more stable but slower.
    max_iter : int, default=200
        Maximum message-passing iterations.
    convergence_iter : int, default=15
        Iterations without change for convergence.
    preference : float or array-like or None, default=None
        Preference for each point to be an exemplar. Larger = more clusters.
        None uses the median of the similarity matrix.
    affinity : str, default='euclidean'
        Affinity type: 'euclidean' or 'precomputed'.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    cluster_centers_indices_ : ndarray
        Indices of exemplar points.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Exemplar coordinates.
    n_clusters_ : int
        Number of clusters found.
    n_iter_ : int
        Iterations run.
    """

    def __init__(
        self,
        damping: float = 0.5,
        max_iter: int = 200,
        convergence_iter: int = 15,
        preference: float | np.ndarray | None = None,
        affinity: str = "euclidean",
        random_state: int | None = None,
    ):
        self.damping = damping
        self.max_iter = max_iter
        self.convergence_iter = convergence_iter
        self.preference = preference
        self.affinity = affinity
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> AffinityPropagationClusterer:
        """Fit Affinity Propagation."""
        from sklearn.cluster import AffinityPropagation

        X = check_array(X)
        self.model_ = AffinityPropagation(
            damping=self.damping,
            max_iter=self.max_iter,
            convergence_iter=self.convergence_iter,
            preference=self.preference,
            affinity=self.affinity,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.cluster_centers_indices_ = self.model_.cluster_centers_indices_
        self.cluster_centers_ = self.model_.cluster_centers_
        self.n_clusters_ = len(self.cluster_centers_indices_)
        self.n_iter_ = self.model_.n_iter_
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
