"""Linear Dimensionality Reduction Methods.

This module provides sklearn-compatible wrappers for linear dimensionality
reduction techniques including PCA variants, SVD, and ICA.

Classes
-------
PCAReducer : Standard Principal Component Analysis
RandomizedPCA : Randomized PCA using randomized SVD (faster for large data)
TruncatedSVDReducer : Truncated SVD / LSA (works with sparse matrices)
KernelPCAReducer : Kernel PCA for nonlinear projections
ICAReducer : Independent Component Analysis
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import (
    PCA,
    FastICA,
    KernelPCA,
    TruncatedSVD,
)
from sklearn.utils.validation import check_array, check_is_fitted


class PCAReducer(TransformerMixin, BaseEstimator):
    """Principal Component Analysis for dimensionality reduction.

    A thin wrapper around sklearn's PCA with additional utilities for
    variance analysis and automatic component selection.

    Parameters
    ----------
    n_components : int, float, or 'mle', default=None
        Number of components to keep.
        - If int, selects that many components.
        - If float (0-1), selects components to explain that fraction of variance.
        - If 'mle', uses Minka's MLE to guess the dimension.
        - If None, keeps all components.

    whiten : bool, default=False
        Whether to whiten the data (unit variance in each component).

    svd_solver : {'auto', 'full', 'arpack', 'randomized'}, default='auto'
        SVD solver to use.

    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    components_ : ndarray of shape (n_components, n_features)
        Principal axes in feature space.

    explained_variance_ratio_ : ndarray
        Percentage of variance explained by each component.

    n_components_ : int
        The estimated number of components.

    Example
    -------
    >>> from endgame.dimensionality_reduction import PCAReducer
    >>> pca = PCAReducer(n_components=0.95)  # Keep 95% variance
    >>> X_reduced = pca.fit_transform(X)
    >>> print(f"Reduced from {X.shape[1]} to {X_reduced.shape[1]} dimensions")
    """

    def __init__(
        self,
        n_components: int | float | Literal["mle"] | None = None,
        whiten: bool = False,
        svd_solver: Literal["auto", "full", "arpack", "randomized"] = "auto",
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the PCA model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PCAReducer
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._pca = PCA(
            n_components=self.n_components,
            whiten=self.whiten,
            svd_solver=self.svd_solver,
            random_state=self.random_state,
        )
        self._pca.fit(X)

        # Expose key attributes
        self.components_ = self._pca.components_
        self.explained_variance_ = self._pca.explained_variance_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.n_components_ = self._pca.n_components_
        self.mean_ = self._pca.mean_

        return self

    def transform(self, X) -> np.ndarray:
        """Apply dimensionality reduction to X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, "_pca")
        X = check_array(X)
        return self._pca.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        """Transform data back to original space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in reduced space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, "_pca")
        return self._pca.inverse_transform(X)

    def get_cumulative_variance(self) -> np.ndarray:
        """Get cumulative explained variance ratio.

        Returns
        -------
        cumulative : ndarray
            Cumulative sum of explained variance ratios.
        """
        check_is_fitted(self, "explained_variance_ratio_")
        return np.cumsum(self.explained_variance_ratio_)

    def get_n_components_for_variance(self, variance_threshold: float) -> int:
        """Get number of components needed to explain given variance.

        Parameters
        ----------
        variance_threshold : float
            Desired cumulative explained variance (0 to 1).

        Returns
        -------
        n_components : int
            Number of components needed.
        """
        check_is_fitted(self, "explained_variance_ratio_")
        cumulative = self.get_cumulative_variance()
        return int(np.searchsorted(cumulative, variance_threshold) + 1)


class RandomizedPCA(TransformerMixin, BaseEstimator):
    """Randomized PCA using randomized SVD.

    Faster than standard PCA for large datasets with many features.
    Uses the randomized SVD algorithm which is more efficient when
    n_components << min(n_samples, n_features).

    Parameters
    ----------
    n_components : int, default=50
        Number of components to keep.

    n_oversamples : int, default=10
        Additional samples for the randomized SVD solver.

    n_iter : int or 'auto', default='auto'
        Number of power iterations for the randomized SVD solver.

    whiten : bool, default=False
        Whether to whiten the data.

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from endgame.dimensionality_reduction import RandomizedPCA
    >>> rpca = RandomizedPCA(n_components=100)
    >>> X_reduced = rpca.fit_transform(X_large)  # Fast for large X
    """

    def __init__(
        self,
        n_components: int = 50,
        n_oversamples: int = 10,
        n_iter: int | Literal["auto"] = "auto",
        whiten: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.whiten = whiten
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the randomized PCA model."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._pca = PCA(
            n_components=self.n_components,
            svd_solver="randomized",
            whiten=self.whiten,
            random_state=self.random_state,
            n_oversamples=self.n_oversamples,
            iterated_power=self.n_iter,
        )
        self._pca.fit(X)

        self.components_ = self._pca.components_
        self.explained_variance_ratio_ = self._pca.explained_variance_ratio_
        self.n_components_ = self._pca.n_components_

        return self

    def transform(self, X) -> np.ndarray:
        """Apply dimensionality reduction."""
        check_is_fitted(self, "_pca")
        X = check_array(X)
        return self._pca.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        """Reconstruct data from reduced representation."""
        check_is_fitted(self, "_pca")
        return self._pca.inverse_transform(X)


class TruncatedSVDReducer(TransformerMixin, BaseEstimator):
    """Truncated SVD (LSA) for dimensionality reduction.

    Unlike PCA, this works directly with sparse matrices without
    centering, making it suitable for text data (TF-IDF).

    Parameters
    ----------
    n_components : int, default=50
        Number of components.

    algorithm : {'arpack', 'randomized'}, default='randomized'
        SVD solver to use.

    n_iter : int, default=5
        Number of iterations for randomized SVD.

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from endgame.dimensionality_reduction import TruncatedSVDReducer
    >>> from sklearn.feature_extraction.text import TfidfVectorizer
    >>> tfidf = TfidfVectorizer()
    >>> X_sparse = tfidf.fit_transform(texts)
    >>> svd = TruncatedSVDReducer(n_components=100)
    >>> X_dense = svd.fit_transform(X_sparse)  # Works with sparse input
    """

    def __init__(
        self,
        n_components: int = 50,
        algorithm: Literal["arpack", "randomized"] = "randomized",
        n_iter: int = 5,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the truncated SVD model."""
        self.n_features_in_ = X.shape[1]

        self._svd = TruncatedSVD(
            n_components=self.n_components,
            algorithm=self.algorithm,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self._svd.fit(X)

        self.components_ = self._svd.components_
        self.explained_variance_ratio_ = self._svd.explained_variance_ratio_
        self.singular_values_ = self._svd.singular_values_

        return self

    def transform(self, X) -> np.ndarray:
        """Apply dimensionality reduction."""
        check_is_fitted(self, "_svd")
        return self._svd.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        """Reconstruct from reduced representation."""
        check_is_fitted(self, "_svd")
        return self._svd.inverse_transform(X)


class KernelPCAReducer(TransformerMixin, BaseEstimator):
    """Kernel PCA for nonlinear dimensionality reduction.

    Applies PCA in a kernel-induced feature space, allowing for
    nonlinear projections while remaining computationally tractable.

    Parameters
    ----------
    n_components : int, default=50
        Number of components.

    kernel : str, default='rbf'
        Kernel type: 'linear', 'poly', 'rbf', 'sigmoid', 'cosine'.

    gamma : float, optional
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
        If None, defaults to 1/n_features.

    degree : int, default=3
        Degree for polynomial kernel.

    coef0 : float, default=1.0
        Independent term in 'poly' and 'sigmoid'.

    fit_inverse_transform : bool, default=False
        Whether to learn the inverse transform (expensive).

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from endgame.dimensionality_reduction import KernelPCAReducer
    >>> kpca = KernelPCAReducer(n_components=2, kernel='rbf', gamma=0.1)
    >>> X_nonlinear = kpca.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int = 50,
        kernel: Literal["linear", "poly", "rbf", "sigmoid", "cosine"] = "rbf",
        gamma: float | None = None,
        degree: int = 3,
        coef0: float = 1.0,
        fit_inverse_transform: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.fit_inverse_transform = fit_inverse_transform
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the Kernel PCA model."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            fit_inverse_transform=self.fit_inverse_transform,
            random_state=self.random_state,
        )
        self._kpca.fit(X)

        self.eigenvalues_ = self._kpca.eigenvalues_
        self.eigenvectors_ = self._kpca.eigenvectors_

        return self

    def transform(self, X) -> np.ndarray:
        """Apply dimensionality reduction."""
        check_is_fitted(self, "_kpca")
        X = check_array(X)
        return self._kpca.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._kpca = KernelPCA(
            n_components=self.n_components,
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0,
            fit_inverse_transform=self.fit_inverse_transform,
            random_state=self.random_state,
        )
        result = self._kpca.fit_transform(X)

        self.eigenvalues_ = self._kpca.eigenvalues_
        self.eigenvectors_ = self._kpca.eigenvectors_

        return result

    def inverse_transform(self, X) -> np.ndarray:
        """Reconstruct from reduced representation.

        Only available if fit_inverse_transform=True.
        """
        check_is_fitted(self, "_kpca")
        if not self.fit_inverse_transform:
            raise ValueError(
                "inverse_transform requires fit_inverse_transform=True"
            )
        return self._kpca.inverse_transform(X)


class ICAReducer(TransformerMixin, BaseEstimator):
    """Independent Component Analysis for dimensionality reduction.

    ICA separates a multivariate signal into additive, independent
    components. Useful when the underlying sources are non-Gaussian.

    Parameters
    ----------
    n_components : int, optional
        Number of components. If None, uses all features.

    algorithm : {'parallel', 'deflation'}, default='parallel'
        ICA algorithm to use.

    whiten : str, default='unit-variance'
        Whitening strategy. Use 'unit-variance' for sklearn >= 1.1.

    fun : {'logcosh', 'exp', 'cube'}, default='logcosh'
        Functional form of the G function for approximating negentropy.

    max_iter : int, default=200
        Maximum number of iterations.

    tol : float, default=1e-4
        Tolerance for convergence.

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from endgame.dimensionality_reduction import ICAReducer
    >>> ica = ICAReducer(n_components=10)
    >>> X_independent = ica.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int | None = None,
        algorithm: Literal["parallel", "deflation"] = "parallel",
        whiten: str = "unit-variance",
        fun: Literal["logcosh", "exp", "cube"] = "logcosh",
        max_iter: int = 200,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.algorithm = algorithm
        self.whiten = whiten
        self.fun = fun
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the ICA model."""
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._ica = FastICA(
            n_components=self.n_components,
            algorithm=self.algorithm,
            whiten=self.whiten,
            fun=self.fun,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
        )
        self._ica.fit(X)

        self.components_ = self._ica.components_
        self.mixing_ = self._ica.mixing_
        self.n_iter_ = self._ica.n_iter_

        return self

    def transform(self, X) -> np.ndarray:
        """Apply ICA transformation."""
        check_is_fitted(self, "_ica")
        X = check_array(X)
        return self._ica.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        """Reconstruct signals from independent components."""
        check_is_fitted(self, "_ica")
        return self._ica.inverse_transform(X)
