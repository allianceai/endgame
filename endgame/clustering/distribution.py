"""Distribution-based clustering: GMM and Fuzzy C-Means.

GaussianMixtureClusterer wraps sklearn's GaussianMixture.
FuzzyCMeansClusterer is implemented from scratch.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils.validation import check_array, check_is_fitted


class GaussianMixtureClusterer(BaseEstimator, ClusterMixin):
    """Gaussian Mixture Model clustering.

    Fits k Gaussians via EM. The probabilistic analog of K-Means — gives
    soft assignments and handles elliptical clusters. Supports BIC/AIC for
    model selection.

    Parameters
    ----------
    n_components : int, default=8
        Number of mixture components.
    covariance_type : str, default='full'
        Covariance type: 'full', 'tied', 'diag', 'spherical'.
    n_init : int, default=5
        Number of EM initializations.
    max_iter : int, default=200
        Maximum EM iterations.
    tol : float, default=1e-3
        Convergence tolerance.
    reg_covar : float, default=1e-6
        Covariance regularization.
    init_params : str, default='k-means++'
        Initialization: 'kmeans', 'k-means++', 'random', 'random_from_data'.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Hard cluster assignments (argmax of responsibilities).
    probabilities_ : ndarray of shape (n_samples, n_components)
        Soft assignment probabilities (responsibilities).
    means_ : ndarray of shape (n_components, n_features)
        Component means.
    covariances_ : ndarray
        Component covariances.
    weights_ : ndarray of shape (n_components,)
        Mixing weights.
    bic_ : float
        Bayesian Information Criterion of the fitted model.
    aic_ : float
        Akaike Information Criterion of the fitted model.
    """

    def __init__(
        self,
        n_components: int = 8,
        covariance_type: str = "full",
        n_init: int = 5,
        max_iter: int = 200,
        tol: float = 1e-3,
        reg_covar: float = 1e-6,
        init_params: str = "k-means++",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> GaussianMixtureClusterer:
        """Fit GMM."""
        from sklearn.mixture import GaussianMixture

        X = check_array(X)
        self.model_ = GaussianMixture(
            n_components=self.n_components,
            covariance_type=self.covariance_type,
            n_init=self.n_init,
            max_iter=self.max_iter,
            tol=self.tol,
            reg_covar=self.reg_covar,
            init_params=self.init_params,
            random_state=self.random_state,
        )
        self.model_.fit(X)
        self.probabilities_ = self.model_.predict_proba(X)
        self.labels_ = self.probabilities_.argmax(axis=1)
        self.means_ = self.model_.means_
        self.covariances_ = self.model_.covariances_
        self.weights_ = self.model_.weights_
        self.bic_ = self.model_.bic(X)
        self.aic_ = self.model_.aic(X)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict hard cluster labels."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict soft cluster probabilities."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.predict_proba(X)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return hard cluster labels."""
        self.fit(X)
        return self.labels_

    def score(self, X: ArrayLike) -> float:
        """Return average log-likelihood."""
        check_is_fitted(self, ["model_"])
        X = check_array(X)
        return self.model_.score(X)

    def select_n_components(
        self,
        X: ArrayLike,
        k_range: range | None = None,
        criterion: str = "bic",
    ) -> int:
        """Select optimal n_components via BIC or AIC.

        Parameters
        ----------
        X : array-like
            Data to evaluate.
        k_range : range or None, default=None
            Range of k values. Defaults to range(1, 21).
        criterion : str, default='bic'
            Selection criterion: 'bic' or 'aic'.

        Returns
        -------
        int
            Optimal number of components.
        """
        from sklearn.mixture import GaussianMixture

        X = check_array(X)
        if k_range is None:
            k_range = range(1, min(21, X.shape[0]))

        scores = {}
        for k in k_range:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type=self.covariance_type,
                n_init=self.n_init,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
            gmm.fit(X)
            scores[k] = gmm.bic(X) if criterion == "bic" else gmm.aic(X)

        return min(scores, key=scores.get)


class FuzzyCMeansClusterer(BaseEstimator, ClusterMixin):
    """Fuzzy C-Means clustering.

    Soft version of K-Means where each point has a degree of membership
    in each cluster. Useful when clusters genuinely overlap.

    Parameters
    ----------
    n_clusters : int, default=8
        Number of clusters.
    m : float, default=2.0
        Fuzziness coefficient (m > 1). Higher values = softer assignments.
        m = 1 approaches hard K-Means; m >> 1 approaches uniform membership.
    max_iter : int, default=300
        Maximum iterations.
    tol : float, default=1e-4
        Convergence tolerance on membership matrix change.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    labels_ : ndarray of shape (n_samples,)
        Hard cluster labels (argmax of membership).
    membership_ : ndarray of shape (n_samples, n_clusters)
        Fuzzy membership matrix.
    cluster_centers_ : ndarray of shape (n_clusters, n_features)
        Cluster centroids.
    n_iter_ : int
        Number of iterations run.
    """

    def __init__(
        self,
        n_clusters: int = 8,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X: ArrayLike, y=None) -> FuzzyCMeansClusterer:
        """Fit Fuzzy C-Means."""
        X = check_array(X, dtype=np.float64)
        n, d = X.shape
        k = self.n_clusters
        rng = np.random.RandomState(self.random_state)

        # Initialize membership matrix randomly
        U = rng.dirichlet(np.ones(k), size=n)  # (n, k)

        for iteration in range(self.max_iter):
            # Compute centres: c_j = sum(u_ij^m * x_i) / sum(u_ij^m)
            Um = U ** self.m  # (n, k)
            centers = (Um.T @ X) / Um.sum(axis=0)[:, None]  # (k, d)

            # Compute distances: d_ij = ||x_i - c_j||
            dists = np.zeros((n, k))
            for j in range(k):
                dists[:, j] = np.linalg.norm(X - centers[j], axis=1)

            # Avoid division by zero
            dists = np.maximum(dists, 1e-10)

            # Update membership
            power = 2.0 / (self.m - 1)
            U_new = np.zeros((n, k))
            for j in range(k):
                # u_ij = 1 / sum_l (d_ij/d_il)^(2/(m-1))
                ratios = (dists[:, j:j+1] / dists) ** power  # (n, k)
                U_new[:, j] = 1.0 / ratios.sum(axis=1)

            # Check convergence
            change = np.max(np.abs(U_new - U))
            U = U_new

            if change < self.tol:
                break

        self.membership_ = U
        self.labels_ = U.argmax(axis=1)
        self.cluster_centers_ = centers
        self.n_iter_ = iteration + 1
        self.n_features_in_ = d
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict hard cluster labels for new data."""
        return self.predict_memberships(X).argmax(axis=1)

    def predict_memberships(self, X: ArrayLike) -> np.ndarray:
        """Predict fuzzy membership for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_clusters)
            Membership matrix.
        """
        check_is_fitted(self, ["cluster_centers_"])
        X = check_array(X, dtype=np.float64)
        n = X.shape[0]
        k = self.n_clusters

        dists = np.zeros((n, k))
        for j in range(k):
            dists[:, j] = np.linalg.norm(X - self.cluster_centers_[j], axis=1)
        dists = np.maximum(dists, 1e-10)

        power = 2.0 / (self.m - 1)
        U = np.zeros((n, k))
        for j in range(k):
            ratios = (dists[:, j:j+1] / dists) ** power
            U[:, j] = 1.0 / ratios.sum(axis=1)

        return U

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return hard cluster labels."""
        self.fit(X)
        return self.labels_
