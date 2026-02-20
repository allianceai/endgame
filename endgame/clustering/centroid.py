"""Centroid-based clustering: K-Means, Mini-Batch K-Means, k*-Means (auto-k).

K-Means and Mini-Batch K-Means wrap sklearn with competition-tuned defaults.
k*-Means implements MDL-based automatic k determination.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


class KMeansClusterer(BaseEstimator, ClusterMixin):
    """K-Means / K-Means++ clustering with competition-tuned defaults.

    Wraps sklearn's KMeans with higher ``n_init`` for stability and
    ``k-means++`` initialization by default.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    init : str or ndarray, default='k-means++'
        Initialization method: 'k-means++', 'random', or centroid array.
    n_init : int, default=20
        Number of initializations (higher than sklearn's 10 for stability).
    max_iter : int, default=300
        Maximum iterations per run.
    tol : float, default=1e-4
        Convergence tolerance.
    algorithm : str, default='lloyd'
        Algorithm: 'lloyd' or 'elkan'.
    random_state : int or None, default=None
        Random seed.
    n_jobs : int, default=-1
        Parallel jobs.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centroids.
    inertia_ : float
        Sum of squared distances to nearest centroid.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        init: str | np.ndarray = "k-means++",
        n_init: int = 20,
        max_iter: int = 300,
        tol: float = 1e-4,
        algorithm: str = "lloyd",
        random_state: int | None = None,
        n_jobs: int = -1,
    ):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.algorithm = algorithm
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X: ArrayLike, y=None) -> KMeansClusterer:
        """Fit K-Means."""
        from sklearn.cluster import KMeans

        X = check_array(X)
        self.model_ = KMeans(
            n_clusters=self.n_clusters,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            algorithm=self.algorithm,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.cluster_centers_ = self.model_.cluster_centers_
        self.inertia_ = self.model_.inertia_
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

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform X to cluster-distance space."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.transform(X)


class MiniBatchKMeansClusterer(BaseEstimator, ClusterMixin):
    """Mini-Batch K-Means for large-scale clustering.

    Trades small accuracy loss for massive speed gains on datasets >100K
    by using random mini-batches instead of full passes.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    batch_size : int, default=1024
        Mini-batch size.
    init : str, default='k-means++'
        Initialization method.
    n_init : int, default=10
        Number of initializations.
    max_iter : int, default=300
        Maximum iterations.
    max_no_improvement : int, default=10
        Early stopping patience.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Centroids.
    inertia_ : float
        Sum of squared distances.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        batch_size: int = 1024,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        max_no_improvement: int = 10,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.batch_size = batch_size
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.max_no_improvement = max_no_improvement
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> MiniBatchKMeansClusterer:
        """Fit Mini-Batch K-Means."""
        from sklearn.cluster import MiniBatchKMeans

        X = check_array(X)
        self.model_ = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=self.batch_size,
            init=self.init,
            n_init=self.n_init,
            max_iter=self.max_iter,
            max_no_improvement=self.max_no_improvement,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        self.labels_ = self.model_.labels_
        self.cluster_centers_ = self.model_.cluster_centers_
        self.inertia_ = self.model_.inertia_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.predict(X)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_

    def partial_fit(self, X: ArrayLike, y=None) -> MiniBatchKMeansClusterer:
        """Incremental fit on a batch of data."""
        X = check_array(X)
        if not hasattr(self, "model_"):
            from sklearn.cluster import MiniBatchKMeans

            self.model_ = MiniBatchKMeans(
                n_clusters=self.n_clusters,
                batch_size=self.batch_size,
                init=self.init,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        self.model_.partial_fit(X)
        self.labels_ = self.model_.labels_
        self.cluster_centers_ = self.model_.cluster_centers_
        return self


class KStarMeansClusterer(BaseEstimator, ClusterMixin):
    """k*-Means: automatic k determination via Minimum Description Length.

    Extends K-Means by splitting and merging clusters based on MDL cost.
    Starts with ``k_init`` clusters and iteratively splits clusters that
    reduce description length and merges clusters that increase it.

    Parameters
    ----------
    k_init : int, default=2
        Initial number of clusters.
    k_max : int, default=50
        Maximum number of clusters to consider.
    max_splits : int, default=20
        Maximum split/merge iterations.
    max_iter : int, default=300
        K-Means iterations per refinement step.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Cluster labels.
    cluster_centers_ : ndarray of shape (k_optimal, n_features)
        Centroids.
    n_clusters_ : int
        Optimal number of clusters found.
    mdl_history_ : list of float
        MDL cost at each iteration.

    References
    ----------
    k*-Means (2025): automatic k via MDL sub-cluster splitting.
    """

    def __init__(
        self,
        k_init: int = 2,
        k_max: int = 50,
        max_splits: int = 20,
        max_iter: int = 300,
        random_state: int | None = None,
    ):
        self.k_init = k_init
        self.k_max = k_max
        self.max_splits = max_splits
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _mdl_cost(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> float:
        """Compute two-part MDL cost of a clustering.

        MDL = model_cost + assignment_cost + residual_cost

        model_cost: bits to encode k centroids and per-cluster variances
        assignment_cost: bits to encode cluster assignments (n * log k)
        residual_cost: bits to encode residuals under spherical Gaussians
        """
        n, d = X.shape
        k = len(centers)
        if k == 0 or n == 0:
            return float("inf")

        # Model cost: encode k centroids (k*d) + k variances + (k-1) mixing
        n_params = k * (d + 1) + max(k - 1, 0)
        model_cost = 0.5 * n_params * np.log(n + 1)

        # Assignment cost: each point needs log(k) bits
        assignment_cost = n * np.log(max(k, 2))

        # Residual cost: log-likelihood under spherical Gaussian per cluster
        residual_cost = 0.0
        for j in range(k):
            mask = labels == j
            n_j = mask.sum()
            if n_j <= 1:
                continue
            X_j = X[mask]
            residuals = X_j - centers[j]
            ss = np.sum(residuals ** 2)
            variance = ss / (n_j * d) + 1e-10
            # -log p(data|model) = n_j*d/2*log(2*pi*var) + ss/(2*var)
            residual_cost += 0.5 * n_j * d * np.log(2 * np.pi * variance) + ss / (2 * variance)

        return model_cost + assignment_cost + residual_cost

    def fit(self, X: ArrayLike, y=None) -> KStarMeansClusterer:
        """Fit k*-Means with automatic k selection."""
        from sklearn.cluster import KMeans

        X = check_array(X, dtype=np.float64)
        n, d = X.shape
        rng = np.random.RandomState(self.random_state)

        # Start with k_init clusters
        k = min(self.k_init, n)
        km = KMeans(n_clusters=k, n_init=10, max_iter=self.max_iter,
                     random_state=rng.randint(2**31))
        km.fit(X)
        labels = km.labels_
        centers = km.cluster_centers_

        best_mdl = self._mdl_cost(X, labels, centers)
        self.mdl_history_ = [best_mdl]

        for iteration in range(self.max_splits):
            improved = False

            # Try splitting each cluster
            if k < self.k_max:
                for j in range(k):
                    mask = labels == j
                    n_j = mask.sum()
                    if n_j < 4:
                        continue

                    X_j = X[mask]
                    # Split cluster j into 2 via 2-means
                    km_split = KMeans(n_clusters=2, n_init=5, max_iter=100,
                                       random_state=rng.randint(2**31))
                    km_split.fit(X_j)

                    # Create candidate: replace cluster j with two sub-clusters
                    new_centers = list(centers)
                    new_centers[j] = km_split.cluster_centers_[0]
                    new_centers.append(km_split.cluster_centers_[1])
                    new_centers = np.array(new_centers)

                    # Reassign all points
                    dists = np.linalg.norm(
                        X[:, None, :] - new_centers[None, :, :], axis=2
                    )
                    new_labels = dists.argmin(axis=1)
                    new_mdl = self._mdl_cost(X, new_labels, new_centers)

                    if new_mdl < best_mdl:
                        best_mdl = new_mdl
                        labels = new_labels
                        centers = new_centers
                        k = len(centers)
                        improved = True
                        break  # Restart scan

            # Try merging closest cluster pairs
            if k > 2:
                best_merge_mdl = best_mdl
                best_merge = None
                for i in range(k):
                    for j in range(i + 1, k):
                        new_centers = np.delete(centers, j, axis=0)
                        # Merged center is weighted average
                        n_i = (labels == i).sum()
                        n_j_count = (labels == j).sum()
                        if n_i + n_j_count == 0:
                            continue
                        new_centers[i] = (
                            centers[i] * n_i + centers[j] * n_j_count
                        ) / (n_i + n_j_count)

                        dists = np.linalg.norm(
                            X[:, None, :] - new_centers[None, :, :], axis=2
                        )
                        new_labels = dists.argmin(axis=1)
                        new_mdl = self._mdl_cost(X, new_labels, new_centers)

                        if new_mdl < best_merge_mdl:
                            best_merge_mdl = new_mdl
                            best_merge = (new_labels, new_centers)

                if best_merge is not None and best_merge_mdl < best_mdl:
                    best_mdl = best_merge_mdl
                    labels, centers = best_merge
                    k = len(centers)
                    improved = True

            self.mdl_history_.append(best_mdl)

            if not improved:
                break

        # Final refinement
        if k >= 2:
            km_final = KMeans(n_clusters=k, init=centers, n_init=1,
                               max_iter=self.max_iter,
                               random_state=rng.randint(2**31))
            km_final.fit(X)
            labels = km_final.labels_
            centers = km_final.cluster_centers_

        self.labels_ = labels
        self.cluster_centers_ = centers
        self.n_clusters_ = len(centers)
        self.n_features_in_ = d
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict cluster labels for new data."""
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X, dtype=np.float64)
        dists = np.linalg.norm(
            X[:, None, :] - self.cluster_centers_[None, :, :], axis=2
        )
        return dists.argmin(axis=1)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return cluster labels."""
        self.fit(X)
        return self.labels_
