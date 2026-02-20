"""Entropy measures for signal complexity analysis.

Provides sklearn-compatible entropy feature extractors:
- Permutation entropy
- Sample entropy
- Approximate entropy
- SVD entropy
- Spectral entropy

These measures quantify signal complexity and irregularity,
commonly used in EEG/biosignal analysis.

References
----------
- Bandt & Pompe (2002): Permutation entropy
- Richman & Moorman (2000): Sample entropy
- Pincus (1991): Approximate entropy
- antropy library: Algorithm implementations
"""


import numpy as np
from scipy import signal as scipy_signal
from scipy.special import comb

from endgame.signal.base import (
    BaseFeatureExtractor,
    ensure_2d_signals,
)


def _embed(x: np.ndarray, order: int, delay: int) -> np.ndarray:
    """Time-delay embedding of a signal.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    order : int
        Embedding dimension.
    delay : int
        Time delay (lag).

    Returns
    -------
    np.ndarray
        Embedded signal of shape (n_vectors, order).
    """
    n = len(x)
    n_vectors = n - (order - 1) * delay
    if n_vectors <= 0:
        raise ValueError(
            f"Signal too short for embedding: {n} samples, "
            f"order={order}, delay={delay}"
        )

    embedded = np.zeros((n_vectors, order))
    for i in range(order):
        embedded[:, i] = x[i * delay : i * delay + n_vectors]

    return embedded


def _count_neighbors(embedded: np.ndarray, r: float, metric: str = "chebyshev") -> int:
    """Count pairs of vectors within distance r.

    Parameters
    ----------
    embedded : np.ndarray
        Embedded signal of shape (n_vectors, dimension).
    r : float
        Tolerance threshold.
    metric : str
        Distance metric: 'chebyshev' (max norm) or 'euclidean'.

    Returns
    -------
    int
        Number of pairs within distance r.
    """
    n = len(embedded)
    count = 0

    for i in range(n - 1):
        for j in range(i + 1, n):
            if metric == "chebyshev":
                dist = np.max(np.abs(embedded[i] - embedded[j]))
            else:
                dist = np.sqrt(np.sum((embedded[i] - embedded[j]) ** 2))

            if dist < r:
                count += 1

    return count


def permutation_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute permutation entropy of a signal.

    Permutation entropy measures the complexity of a time series by
    analyzing the order relations between successive values.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    order : int, default=3
        Embedding dimension (typically 3-7).
    delay : int, default=1
        Time delay between elements.
    normalize : bool, default=True
        If True, normalize by log2(order!).

    Returns
    -------
    float
        Permutation entropy value.

    References
    ----------
    Bandt, C., & Pompe, B. (2002). Permutation entropy: a natural complexity
    measure for time series. Physical review letters, 88(17), 174102.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Create embedded vectors
    n_vectors = n - (order - 1) * delay
    if n_vectors <= 0:
        return np.nan

    # Get ordinal patterns
    from math import factorial
    n_patterns = factorial(order)
    pattern_counts = np.zeros(n_patterns)

    for i in range(n_vectors):
        # Extract embedding vector
        vec = x[i : i + order * delay : delay]
        # Get permutation pattern (argsort gives ranking)
        pattern = tuple(np.argsort(vec))
        # Convert pattern to index (simple hash)
        idx = sum(p * factorial(order - 1 - j) for j, p in enumerate(pattern))
        pattern_counts[int(idx % n_patterns)] += 1

    # Compute entropy from probability distribution
    probs = pattern_counts / n_vectors
    probs = probs[probs > 0]  # Remove zeros
    entropy = -np.sum(probs * np.log2(probs))

    if normalize:
        entropy /= np.log2(n_patterns)

    return entropy


def sample_entropy(
    x: np.ndarray,
    order: int = 2,
    r: float | None = None,
) -> float:
    """Compute sample entropy of a signal.

    Sample entropy measures the complexity of a time series based on
    approximate entropy but without counting self-matches.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    order : int, default=2
        Embedding dimension.
    r : float, optional
        Tolerance threshold. If None, uses 0.2 * std(x).

    Returns
    -------
    float
        Sample entropy value.

    References
    ----------
    Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis
    using approximate entropy and sample entropy.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if r is None:
        r = 0.2 * np.std(x, ddof=1)

    if r <= 0:
        return np.nan

    # Count matches for embedding dimension m
    embedded_m = _embed(x, order, 1)
    count_m = _count_neighbors(embedded_m, r)

    # Count matches for embedding dimension m+1
    embedded_m1 = _embed(x, order + 1, 1)
    count_m1 = _count_neighbors(embedded_m1, r)

    # Avoid log(0)
    if count_m == 0 or count_m1 == 0:
        return np.nan

    # Normalize by number of possible pairs
    n_m = len(embedded_m)
    n_m1 = len(embedded_m1)

    phi_m = count_m / comb(n_m, 2)
    phi_m1 = count_m1 / comb(n_m1, 2)

    return -np.log(phi_m1 / phi_m)


def approximate_entropy(
    x: np.ndarray,
    order: int = 2,
    r: float | None = None,
) -> float:
    """Compute approximate entropy of a signal.

    Approximate entropy measures the likelihood that similar patterns
    remain similar on the next comparison.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    order : int, default=2
        Embedding dimension.
    r : float, optional
        Tolerance threshold. If None, uses 0.2 * std(x).

    Returns
    -------
    float
        Approximate entropy value.

    References
    ----------
    Pincus, S. M. (1991). Approximate entropy as a measure of system complexity.
    Proceedings of the National Academy of Sciences, 88(6), 2297-2301.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if r is None:
        r = 0.2 * np.std(x, ddof=1)

    if r <= 0:
        return np.nan

    def phi(m):
        """Count template matches for dimension m."""
        embedded = _embed(x, m, 1)
        n_vec = len(embedded)

        # Count matches including self-matches
        counts = np.zeros(n_vec)
        for i in range(n_vec):
            for j in range(n_vec):
                if np.max(np.abs(embedded[i] - embedded[j])) < r:
                    counts[i] += 1

        # Normalize and compute mean log
        return np.mean(np.log(counts / n_vec))

    return phi(order) - phi(order + 1)


def svd_entropy(
    x: np.ndarray,
    order: int = 3,
    delay: int = 1,
    normalize: bool = True,
) -> float:
    """Compute SVD entropy of a signal.

    SVD entropy measures the complexity of a time series using singular
    value decomposition of the embedded signal.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    order : int, default=3
        Embedding dimension.
    delay : int, default=1
        Time delay.
    normalize : bool, default=True
        If True, normalize by log2(order).

    Returns
    -------
    float
        SVD entropy value.
    """
    x = np.asarray(x).flatten()

    # Create embedded matrix
    embedded = _embed(x, order, delay)

    # SVD
    _, s, _ = np.linalg.svd(embedded, full_matrices=False)

    # Normalize singular values
    s_norm = s / np.sum(s)
    s_norm = s_norm[s_norm > 0]

    # Compute entropy
    entropy = -np.sum(s_norm * np.log2(s_norm))

    if normalize:
        entropy /= np.log2(order)

    return entropy


def spectral_entropy(
    x: np.ndarray,
    fs: float,
    method: str = "welch",
    nperseg: int | None = None,
    normalize: bool = True,
) -> float:
    """Compute spectral entropy of a signal.

    Spectral entropy measures the flatness of the power spectrum,
    indicating how spread the spectral energy is across frequencies.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    fs : float
        Sample rate in Hz.
    method : str, default='welch'
        PSD estimation method: 'welch' or 'fft'.
    nperseg : int, optional
        Segment length for Welch's method.
    normalize : bool, default=True
        If True, normalize by log2(n_freqs).

    Returns
    -------
    float
        Spectral entropy value.
    """
    x = np.asarray(x).flatten()

    if method == "welch":
        nperseg = nperseg if nperseg is not None else min(256, len(x))
        freqs, psd = scipy_signal.welch(x, fs=fs, nperseg=nperseg)
    else:
        from scipy.fft import rfft, rfftfreq
        fft_vals = np.abs(rfft(x)) ** 2
        freqs = rfftfreq(len(x), 1 / fs)
        psd = fft_vals

    # Normalize to probability distribution
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]

    # Compute entropy
    entropy = -np.sum(psd_norm * np.log2(psd_norm))

    if normalize:
        entropy /= np.log2(len(psd_norm))

    return entropy


class PermutationEntropy(BaseFeatureExtractor):
    """Permutation entropy feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    order : int, default=3
        Embedding dimension.
    delay : int, default=1
        Time delay.
    normalize : bool, default=True
        Normalize entropy by maximum possible value.

    Examples
    --------
    >>> pe = PermutationEntropy(order=3)
    >>> features = pe.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        order: int = 3,
        delay: int = 1,
        normalize: bool = True,
    ):
        super().__init__(fs=fs)
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def fit(self, X, y=None, **fit_params) -> "PermutationEntropy":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["permutation_entropy"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            pe = permutation_entropy(sig, self.order, self.delay, self.normalize)
            features.append([pe])

        return np.array(features)


class SampleEntropy(BaseFeatureExtractor):
    """Sample entropy feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    order : int, default=2
        Embedding dimension.
    r : float, optional
        Tolerance threshold. If None, uses 0.2 * std(x).

    Examples
    --------
    >>> se = SampleEntropy(order=2)
    >>> features = se.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        order: int = 2,
        r: float | None = None,
    ):
        super().__init__(fs=fs)
        self.order = order
        self.r = r

    def fit(self, X, y=None, **fit_params) -> "SampleEntropy":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["sample_entropy"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            se = sample_entropy(sig, self.order, self.r)
            features.append([se])

        return np.array(features)


class ApproximateEntropy(BaseFeatureExtractor):
    """Approximate entropy feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    order : int, default=2
        Embedding dimension.
    r : float, optional
        Tolerance threshold. If None, uses 0.2 * std(x).

    Examples
    --------
    >>> ae = ApproximateEntropy(order=2)
    >>> features = ae.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        order: int = 2,
        r: float | None = None,
    ):
        super().__init__(fs=fs)
        self.order = order
        self.r = r

    def fit(self, X, y=None, **fit_params) -> "ApproximateEntropy":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["approximate_entropy"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            ae = approximate_entropy(sig, self.order, self.r)
            features.append([ae])

        return np.array(features)


class SpectralEntropy(BaseFeatureExtractor):
    """Spectral entropy feature extractor.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    method : str, default='welch'
        PSD estimation method.
    nperseg : int, optional
        Segment length for Welch.
    normalize : bool, default=True
        Normalize by maximum entropy.

    Examples
    --------
    >>> se = SpectralEntropy(fs=256)
    >>> features = se.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float,
        method: str = "welch",
        nperseg: int | None = None,
        normalize: bool = True,
    ):
        super().__init__(fs=fs)
        self.method = method
        self.nperseg = nperseg
        self.normalize = normalize

    def fit(self, X, y=None, **fit_params) -> "SpectralEntropy":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["spectral_entropy"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            se = spectral_entropy(sig, self.fs, self.method, self.nperseg, self.normalize)
            features.append([se])

        return np.array(features)


class SVDEntropy(BaseFeatureExtractor):
    """SVD entropy feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    order : int, default=3
        Embedding dimension.
    delay : int, default=1
        Time delay.
    normalize : bool, default=True
        Normalize by maximum entropy.

    Examples
    --------
    >>> svde = SVDEntropy(order=3)
    >>> features = svde.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        order: int = 3,
        delay: int = 1,
        normalize: bool = True,
    ):
        super().__init__(fs=fs)
        self.order = order
        self.delay = delay
        self.normalize = normalize

    def fit(self, X, y=None, **fit_params) -> "SVDEntropy":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["svd_entropy"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            se = svd_entropy(sig, self.order, self.delay, self.normalize)
            features.append([se])

        return np.array(features)


class EntropyFeatureExtractor(BaseFeatureExtractor):
    """Comprehensive entropy feature extractor.

    Extracts multiple entropy measures in a single transformer.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    include_permutation : bool, default=True
        Include permutation entropy.
    include_sample : bool, default=True
        Include sample entropy.
    include_approximate : bool, default=False
        Include approximate entropy (slow).
    include_svd : bool, default=True
        Include SVD entropy.
    include_spectral : bool, default=True
        Include spectral entropy.
    perm_order : int, default=3
        Permutation entropy order.
    sample_order : int, default=2
        Sample/approximate entropy order.
    svd_order : int, default=3
        SVD entropy order.

    Examples
    --------
    >>> extractor = EntropyFeatureExtractor(fs=256)
    >>> features = extractor.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float,
        include_permutation: bool = True,
        include_sample: bool = True,
        include_approximate: bool = False,
        include_svd: bool = True,
        include_spectral: bool = True,
        perm_order: int = 3,
        sample_order: int = 2,
        svd_order: int = 3,
    ):
        super().__init__(fs=fs)
        self.include_permutation = include_permutation
        self.include_sample = include_sample
        self.include_approximate = include_approximate
        self.include_svd = include_svd
        self.include_spectral = include_spectral
        self.perm_order = perm_order
        self.sample_order = sample_order
        self.svd_order = svd_order

    def fit(self, X, y=None, **fit_params) -> "EntropyFeatureExtractor":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = []
        if self.include_permutation:
            self.feature_names_.append("permutation_entropy")
        if self.include_sample:
            self.feature_names_.append("sample_entropy")
        if self.include_approximate:
            self.feature_names_.append("approximate_entropy")
        if self.include_svd:
            self.feature_names_.append("svd_entropy")
        if self.include_spectral:
            self.feature_names_.append("spectral_entropy")

        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        all_features = []
        for sig in X_2d:
            features = []

            if self.include_permutation:
                features.append(permutation_entropy(sig, self.perm_order, normalize=True))
            if self.include_sample:
                features.append(sample_entropy(sig, self.sample_order))
            if self.include_approximate:
                features.append(approximate_entropy(sig, self.sample_order))
            if self.include_svd:
                features.append(svd_entropy(sig, self.svd_order, normalize=True))
            if self.include_spectral:
                features.append(spectral_entropy(sig, self.fs, normalize=True))

            all_features.append(features)

        return np.array(all_features)
