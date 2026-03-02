from __future__ import annotations

"""Spatial filtering and Riemannian geometry for multi-channel signals.

Provides sklearn-compatible spatial filtering methods:
- Common Spatial Patterns (CSP)
- Tangent Space projection
- Covariance estimation

These methods operate on multi-channel signals (e.g., EEG) and extract
features that capture spatial relationships between channels.

References
----------
- Koles et al. (1990): Common Spatial Patterns
- Barachant et al. (2012): Riemannian geometry for EEG
- pyriemann library: Riemannian geometry implementations
"""


import numpy as np
from scipy import linalg

from endgame.signal.base import (
    BaseFeatureExtractor,
    BaseSignalTransformer,
)


def _covariance(X: np.ndarray, estimator: str = "empirical") -> np.ndarray:
    """Compute covariance matrix.

    Parameters
    ----------
    X : np.ndarray
        Signal of shape (n_channels, n_samples).
    estimator : str, default='empirical'
        Covariance estimator: 'empirical', 'oas', 'lwf', 'scm'.

    Returns
    -------
    np.ndarray
        Covariance matrix of shape (n_channels, n_channels).
    """
    n_channels, n_samples = X.shape

    if estimator == "empirical":
        return np.cov(X)
    elif estimator == "scm":
        # Sample covariance matrix (not unbiased)
        return X @ X.T / n_samples
    elif estimator == "oas":
        # Oracle Approximating Shrinkage
        try:
            from sklearn.covariance import OAS
            oas = OAS().fit(X.T)
            return oas.covariance_
        except ImportError:
            return np.cov(X)
    elif estimator == "lwf":
        # Ledoit-Wolf shrinkage
        try:
            from sklearn.covariance import LedoitWolf
            lw = LedoitWolf().fit(X.T)
            return lw.covariance_
        except ImportError:
            return np.cov(X)
    else:
        raise ValueError(f"Unknown estimator: {estimator}")


def _matrix_power(A: np.ndarray, power: float) -> np.ndarray:
    """Compute matrix power using eigendecomposition.

    Parameters
    ----------
    A : np.ndarray
        Symmetric positive definite matrix.
    power : float
        Power to raise matrix to.

    Returns
    -------
    np.ndarray
        A^power.
    """
    eigenvalues, eigenvectors = linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
    return eigenvectors @ np.diag(eigenvalues**power) @ eigenvectors.T


def _logm(A: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm for SPD matrix.

    Parameters
    ----------
    A : np.ndarray
        Symmetric positive definite matrix.

    Returns
    -------
    np.ndarray
        Matrix logarithm.
    """
    eigenvalues, eigenvectors = linalg.eigh(A)
    eigenvalues = np.maximum(eigenvalues, 1e-10)
    return eigenvectors @ np.diag(np.log(eigenvalues)) @ eigenvectors.T


def _expm(A: np.ndarray) -> np.ndarray:
    """Compute matrix exponential for symmetric matrix.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix.

    Returns
    -------
    np.ndarray
        Matrix exponential.
    """
    eigenvalues, eigenvectors = linalg.eigh(A)
    return eigenvectors @ np.diag(np.exp(eigenvalues)) @ eigenvectors.T


def _mean_riemann(covmats: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Compute Riemannian mean of covariance matrices.

    Uses iterative geodesic descent to find the Frechet mean.

    Parameters
    ----------
    covmats : np.ndarray
        Covariance matrices of shape (n_matrices, n_channels, n_channels).
    tol : float, default=1e-8
        Convergence tolerance.
    max_iter : int, default=50
        Maximum iterations.

    Returns
    -------
    np.ndarray
        Riemannian mean of shape (n_channels, n_channels).
    """
    n_matrices = len(covmats)

    # Initialize with arithmetic mean
    mean = np.mean(covmats, axis=0)

    for _ in range(max_iter):
        # Compute tangent vectors at current mean
        mean_sqrt_inv = _matrix_power(mean, -0.5)

        tangent_sum = np.zeros_like(mean)
        for cov in covmats:
            # Project to tangent space
            tangent = _logm(mean_sqrt_inv @ cov @ mean_sqrt_inv)
            tangent_sum += tangent

        tangent_mean = tangent_sum / n_matrices

        # Check convergence
        if np.linalg.norm(tangent_mean, "fro") < tol:
            break

        # Update mean
        mean_sqrt = _matrix_power(mean, 0.5)
        mean = mean_sqrt @ _expm(tangent_mean) @ mean_sqrt

    return mean


class CovarianceEstimator(BaseSignalTransformer):
    """Estimate covariance matrices from multi-channel signals.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    estimator : str, default='empirical'
        Covariance estimator: 'empirical', 'oas', 'lwf', 'scm'.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    n_channels_ : int
        Number of channels.

    Examples
    --------
    >>> cov_est = CovarianceEstimator(estimator='oas')
    >>> covmats = cov_est.fit_transform(X)  # X: (n_trials, n_channels, n_samples)
    """

    def __init__(
        self,
        fs: float | None = None,
        estimator: str = "empirical",
        copy: bool = True,
    ):
        super().__init__(fs=fs, copy=copy)
        self.estimator = estimator

    def fit(self, X, y=None, **fit_params) -> CovarianceEstimator:
        """Fit the estimator.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).
        y : ignored

        Returns
        -------
        self
        """
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (trials, channels, samples), got shape {X.shape}")

        self.n_channels_ = X.shape[1]
        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Compute covariance matrices.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        np.ndarray
            Covariance matrices of shape (n_trials, n_channels, n_channels).
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if self.copy:
            X = X.copy()

        n_trials = X.shape[0]
        covmats = np.zeros((n_trials, self.n_channels_, self.n_channels_))

        for i in range(n_trials):
            covmats[i] = _covariance(X[i], self.estimator)

        return covmats


class CSP(BaseSignalTransformer):
    """Common Spatial Patterns for discriminative spatial filtering.

    CSP finds spatial filters that maximize variance for one class
    while minimizing variance for another class.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    n_components : int, default=4
        Number of CSP components (filters) per class.
    reg : float or str, optional
        Regularization parameter or method ('ledoit_wolf', 'oas').
    log : bool, default=True
        Apply log transform to variances (recommended for classification).
    cov_estimator : str, default='empirical'
        Covariance estimator.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    filters_ : np.ndarray
        Spatial filters of shape (n_filters, n_channels).
    patterns_ : np.ndarray
        Spatial patterns of shape (n_channels, n_filters).
    eigenvalues_ : np.ndarray
        Eigenvalues corresponding to each filter.

    References
    ----------
    Koles, Z. J., et al. (1990). Spatial patterns underlying population
    differences in the background EEG.

    Examples
    --------
    >>> csp = CSP(n_components=4)
    >>> features = csp.fit_transform(X, y)  # X: (n_trials, n_channels, n_samples)
    """

    def __init__(
        self,
        fs: float | None = None,
        n_components: int = 4,
        reg: float | str | None = None,
        log: bool = True,
        cov_estimator: str = "empirical",
        copy: bool = True,
    ):
        super().__init__(fs=fs, copy=copy)
        self.n_components = n_components
        self.reg = reg
        self.log = log
        self.cov_estimator = cov_estimator

    def fit(self, X, y, **fit_params) -> CSP:
        """Fit CSP spatial filters.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).
        y : np.ndarray
            Binary class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim != 3:
            raise ValueError(f"Expected 3D array (trials, channels, samples), got shape {X.shape}")

        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError(f"CSP requires exactly 2 classes, got {len(classes)}")

        n_trials, n_channels, n_samples = X.shape

        # Compute class covariances
        covmats = np.zeros((n_trials, n_channels, n_channels))
        for i in range(n_trials):
            covmats[i] = _covariance(X[i], self.cov_estimator)

        # Average covariance per class
        cov_class1 = np.mean(covmats[y == classes[0]], axis=0)
        cov_class2 = np.mean(covmats[y == classes[1]], axis=0)

        # Apply regularization
        if self.reg is not None:
            if isinstance(self.reg, (int, float)):
                cov_class1 = (1 - self.reg) * cov_class1 + self.reg * np.eye(n_channels)
                cov_class2 = (1 - self.reg) * cov_class2 + self.reg * np.eye(n_channels)

        # Solve generalized eigenvalue problem
        # cov_class1 * W = cov_composite * W * D
        cov_composite = cov_class1 + cov_class2

        # Whitening transform
        eigenvalues, eigenvectors = linalg.eigh(cov_composite)
        eigenvalues = np.maximum(eigenvalues, 1e-10)
        whitening = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

        # Apply whitening to class1 covariance
        cov_class1_white = whitening @ cov_class1 @ whitening.T

        # Eigenvectors of whitened class1 covariance
        eigenvalues, eigenvectors = linalg.eigh(cov_class1_white)

        # Sort by eigenvalue (most discriminative first and last)
        sorted_idx = np.argsort(eigenvalues)
        # Take first and last n_components
        n_comp = min(self.n_components, n_channels // 2)
        selected_idx = np.concatenate([sorted_idx[:n_comp], sorted_idx[-n_comp:]])

        self.eigenvalues_ = eigenvalues[selected_idx]

        # Spatial filters
        self.filters_ = (whitening.T @ eigenvectors[:, selected_idx]).T
        # Spatial patterns (for visualization)
        self.patterns_ = linalg.pinv(self.filters_).T

        self.n_channels_ = n_channels
        self._is_fitted = True

        return self

    def transform(self, X) -> np.ndarray:
        """Apply CSP spatial filters.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        np.ndarray
            CSP features of shape (n_trials, n_filters).
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if self.copy:
            X = X.copy()

        n_trials = X.shape[0]
        n_filters = len(self.filters_)

        features = np.zeros((n_trials, n_filters))

        for i in range(n_trials):
            # Apply spatial filters
            filtered = self.filters_ @ X[i]
            # Compute variance of each filtered signal
            variances = np.var(filtered, axis=1)

            if self.log:
                features[i] = np.log(variances / np.sum(variances))
            else:
                features[i] = variances

        return features


class TangentSpace(BaseFeatureExtractor):
    """Tangent space projection for Riemannian geometry.

    Projects covariance matrices onto the tangent space at the
    Riemannian mean, enabling use of Euclidean classifiers.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    metric : str, default='riemann'
        Metric for mean computation: 'riemann', 'euclid', 'logeuclid'.
    reference : np.ndarray, optional
        Reference point (mean). If None, computed from training data.
    cov_estimator : str, default='empirical'
        Covariance estimator for input signals.

    Attributes
    ----------
    reference_ : np.ndarray
        Reference covariance matrix.
    feature_names_ : list
        Names of tangent space features.

    References
    ----------
    Barachant, A., et al. (2012). Multiclass brain-computer interface
    classification by Riemannian geometry.

    Examples
    --------
    >>> ts = TangentSpace()
    >>> # From covariance matrices
    >>> features = ts.fit_transform(covmats)
    >>> # Or from raw signals (computes covariances internally)
    >>> features = ts.fit_transform(X, input_type='signals')
    """

    def __init__(
        self,
        fs: float | None = None,
        metric: str = "riemann",
        reference: np.ndarray | None = None,
        cov_estimator: str = "empirical",
    ):
        super().__init__(fs=fs)
        self.metric = metric
        self.reference = reference
        self.cov_estimator = cov_estimator

    def fit(self, X, y=None, input_type: str = "covariances", **fit_params) -> TangentSpace:
        """Fit the tangent space projector.

        Parameters
        ----------
        X : np.ndarray
            Either covariance matrices (n_trials, n_channels, n_channels)
            or raw signals (n_trials, n_channels, n_samples).
        y : ignored
        input_type : str, default='covariances'
            Type of input: 'covariances' or 'signals'.

        Returns
        -------
        self
        """
        X = np.asarray(X)

        # Convert signals to covariances if needed
        if input_type == "signals":
            if X.ndim != 3:
                raise ValueError(f"Expected 3D array for signals, got shape {X.shape}")
            covmats = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                covmats[i] = _covariance(X[i], self.cov_estimator)
        else:
            covmats = X

        if covmats.ndim != 3 or covmats.shape[1] != covmats.shape[2]:
            raise ValueError(
                f"Expected covariance matrices (n, ch, ch), got shape {covmats.shape}"
            )

        self.n_channels_ = covmats.shape[1]

        # Compute reference (mean)
        if self.reference is not None:
            self.reference_ = self.reference
        elif self.metric == "riemann":
            self.reference_ = _mean_riemann(covmats)
        elif self.metric == "euclid":
            self.reference_ = np.mean(covmats, axis=0)
        elif self.metric == "logeuclid":
            log_covmats = np.array([_logm(cov) for cov in covmats])
            self.reference_ = _expm(np.mean(log_covmats, axis=0))
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

        # Feature names (upper triangular elements)
        n_features = self.n_channels_ * (self.n_channels_ + 1) // 2
        self.feature_names_ = [f"ts_{i}" for i in range(n_features)]

        self._is_fitted = True
        return self

    def transform(self, X, input_type: str = "covariances") -> np.ndarray:
        """Project to tangent space.

        Parameters
        ----------
        X : np.ndarray
            Either covariance matrices or raw signals.
        input_type : str, default='covariances'
            Type of input: 'covariances' or 'signals'.

        Returns
        -------
        np.ndarray
            Tangent space features of shape (n_trials, n_features).
        """
        X = np.asarray(X)

        # Convert signals to covariances if needed
        if input_type == "signals":
            covmats = np.zeros((X.shape[0], X.shape[1], X.shape[1]))
            for i in range(X.shape[0]):
                covmats[i] = _covariance(X[i], self.cov_estimator)
        else:
            covmats = X

        n_trials = len(covmats)

        # Reference inverse square root
        ref_invsqrt = _matrix_power(self.reference_, -0.5)

        features = []
        for cov in covmats:
            # Project to tangent space at reference
            tangent = _logm(ref_invsqrt @ cov @ ref_invsqrt)

            # Extract upper triangular (including diagonal)
            # Scale off-diagonal by sqrt(2) for proper metric
            idx = np.triu_indices(self.n_channels_)
            vec = tangent[idx]

            # Scale off-diagonal elements
            diag_idx = np.arange(self.n_channels_)
            off_diag_mask = idx[0] != idx[1]
            vec[off_diag_mask] *= np.sqrt(2)

            features.append(vec)

        return np.array(features)


class FilterBankCSP(BaseSignalTransformer):
    """Filter Bank Common Spatial Patterns.

    Applies CSP to multiple frequency bands and concatenates features.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    bands : list of tuple
        Frequency bands as (low, high) tuples.
    n_components : int, default=4
        Number of CSP components per band.
    filter_order : int, default=5
        Order of bandpass filters.

    Examples
    --------
    >>> bands = [(4, 8), (8, 12), (12, 30)]
    >>> fbcsp = FilterBankCSP(fs=256, bands=bands)
    >>> features = fbcsp.fit_transform(X, y)
    """

    def __init__(
        self,
        fs: float,
        bands: list[tuple[float, float]],
        n_components: int = 4,
        filter_order: int = 5,
    ):
        super().__init__(fs=fs)
        self.bands = bands
        self.n_components = n_components
        self.filter_order = filter_order

    def fit(self, X, y, **fit_params) -> FilterBankCSP:
        """Fit FilterBank CSP.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).
        y : np.ndarray
            Binary class labels.

        Returns
        -------
        self
        """
        from scipy.signal import butter, sosfiltfilt

        X = np.asarray(X)
        y = np.asarray(y)

        self.csps_ = []
        self.filters_ = []

        for low, high in self.bands:
            # Design bandpass filter
            sos = butter(self.filter_order, [low, high], btype="bandpass", fs=self.fs, output="sos")
            self.filters_.append(sos)

            # Filter data
            X_filtered = np.zeros_like(X)
            for i in range(X.shape[0]):
                for ch in range(X.shape[1]):
                    X_filtered[i, ch] = sosfiltfilt(sos, X[i, ch])

            # Fit CSP for this band
            csp = CSP(n_components=self.n_components)
            csp.fit(X_filtered, y)
            self.csps_.append(csp)

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Apply FilterBank CSP.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        np.ndarray
            FBCSP features of shape (n_trials, n_bands * n_components * 2).
        """
        from scipy.signal import sosfiltfilt

        self._check_is_fitted()
        X = np.asarray(X)

        all_features = []

        for sos, csp in zip(self.filters_, self.csps_):
            # Filter data
            X_filtered = np.zeros_like(X)
            for i in range(X.shape[0]):
                for ch in range(X.shape[1]):
                    X_filtered[i, ch] = sosfiltfilt(sos, X[i, ch])

            # Apply CSP
            features = csp.transform(X_filtered)
            all_features.append(features)

        return np.hstack(all_features)
