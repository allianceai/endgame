"""Complexity and fractal dimension measures for signal analysis.

Provides sklearn-compatible feature extractors for:
- Fractal dimensions (Higuchi, Petrosian, Katz)
- Hurst exponent
- Detrended Fluctuation Analysis (DFA)
- Lempel-Ziv Complexity

These measures quantify the self-similarity and complexity of signals,
commonly used in EEG/biosignal analysis.

References
----------
- Higuchi (1988): Higuchi fractal dimension
- Petrosian (1995): Petrosian fractal dimension
- Katz (1988): Katz fractal dimension
- Hurst (1951): Hurst exponent
- Peng et al. (1994): Detrended fluctuation analysis
- Lempel & Ziv (1976): Lempel-Ziv complexity
"""


import numpy as np

from endgame.signal.base import (
    BaseFeatureExtractor,
    ensure_2d_signals,
)


def higuchi_fd(x: np.ndarray, kmax: int = 10) -> float:
    """Compute Higuchi fractal dimension of a signal.

    The Higuchi fractal dimension (HFD) estimates the fractal dimension
    of a time series by analyzing the length of the curve at different
    scales.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    kmax : int, default=10
        Maximum scale factor.

    Returns
    -------
    float
        Higuchi fractal dimension (typically 1-2).

    References
    ----------
    Higuchi, T. (1988). Approach to an irregular time series on the basis
    of the fractal theory. Physica D: Nonlinear Phenomena, 31(2), 277-283.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if kmax > n // 2:
        kmax = n // 2

    # Compute curve length for each scale k
    lk = np.zeros(kmax)

    for k in range(1, kmax + 1):
        # Average length over all starting points
        lm = np.zeros(k)
        for m in range(k):
            # Number of points in this subseries
            num_points = (n - m - 1) // k

            if num_points > 0:
                # Sum of absolute differences
                diff_sum = 0.0
                for i in range(1, num_points + 1):
                    idx1 = m + i * k
                    idx2 = m + (i - 1) * k
                    if idx1 < n:
                        diff_sum += abs(x[idx1] - x[idx2])

                # Normalized length
                lm[m] = (diff_sum * (n - 1)) / (k * num_points * k)

        # Average over starting points
        lk[k - 1] = np.mean(lm)

    # Linear regression of log(L(k)) vs log(1/k)
    k_vals = np.arange(1, kmax + 1)
    valid = lk > 0
    if np.sum(valid) < 2:
        return np.nan

    log_k = np.log(1.0 / k_vals[valid])
    log_lk = np.log(lk[valid])

    # Slope is the fractal dimension
    slope, _ = np.polyfit(log_k, log_lk, 1)

    return slope


def petrosian_fd(x: np.ndarray) -> float:
    """Compute Petrosian fractal dimension of a signal.

    The Petrosian fractal dimension (PFD) provides a fast estimate of
    the fractal dimension based on the number of sign changes in the
    first derivative.

    Parameters
    ----------
    x : np.ndarray
        1D signal.

    Returns
    -------
    float
        Petrosian fractal dimension.

    References
    ----------
    Petrosian, A. (1995). Kolmogorov complexity of finite sequences and
    recognition of different preictal EEG patterns.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # First derivative
    dx = np.diff(x)

    # Number of sign changes (zero crossings of derivative)
    n_delta = np.sum(dx[:-1] * dx[1:] < 0)

    # Petrosian formula
    if n_delta == 0:
        return np.log10(n)

    return np.log10(n) / (np.log10(n) + np.log10(n / (n + 0.4 * n_delta)))


def katz_fd(x: np.ndarray) -> float:
    """Compute Katz fractal dimension of a signal.

    The Katz fractal dimension estimates the complexity based on the
    ratio of the total path length to the maximum distance from the
    first point.

    Parameters
    ----------
    x : np.ndarray
        1D signal.

    Returns
    -------
    float
        Katz fractal dimension.

    References
    ----------
    Katz, M. J. (1988). Fractals and the analysis of waveforms.
    Computers in Biology and Medicine, 18(3), 145-156.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Total path length (sum of distances between consecutive points)
    # Using unit time steps
    dists = np.sqrt(1 + np.diff(x) ** 2)
    L = np.sum(dists)

    # Maximum distance from first point
    indices = np.arange(n)
    distances_from_start = np.sqrt(indices**2 + (x - x[0]) ** 2)
    d = np.max(distances_from_start)

    # Katz formula
    if d == 0:
        return np.nan

    return np.log10(n - 1) / (np.log10(n - 1) + np.log10(d / L))


def hurst_exponent(x: np.ndarray, max_lag: int | None = None) -> float:
    """Compute Hurst exponent of a signal using R/S analysis.

    The Hurst exponent (H) measures the long-term memory of a time series:
    - H < 0.5: Anti-persistent (mean-reverting)
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (trending)

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    max_lag : int, optional
        Maximum lag for R/S calculation. If None, uses n//2.

    Returns
    -------
    float
        Hurst exponent (typically 0-1).

    References
    ----------
    Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
    Transactions of the American Society of Civil Engineers, 116, 770-799.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if max_lag is None:
        max_lag = n // 2

    # Use powers of 2 for lags
    lags = []
    lag = 4
    while lag <= max_lag:
        lags.append(lag)
        lag *= 2

    if len(lags) < 2:
        return np.nan

    rs_values = []

    for lag in lags:
        # Split into non-overlapping segments
        n_segments = n // lag
        if n_segments == 0:
            continue

        rs_segment = []
        for i in range(n_segments):
            segment = x[i * lag : (i + 1) * lag]

            # Mean-adjusted cumulative sum
            mean_adj = segment - np.mean(segment)
            cum_sum = np.cumsum(mean_adj)

            # Range
            R = np.max(cum_sum) - np.min(cum_sum)

            # Standard deviation
            S = np.std(segment, ddof=1)

            if S > 0:
                rs_segment.append(R / S)

        if rs_segment:
            rs_values.append(np.mean(rs_segment))
        else:
            rs_values.append(np.nan)

    # Remove NaN values
    valid = ~np.isnan(rs_values)
    if np.sum(valid) < 2:
        return np.nan

    lags = np.array(lags)[valid]
    rs_values = np.array(rs_values)[valid]

    # Linear regression of log(R/S) vs log(lag)
    log_lag = np.log(lags)
    log_rs = np.log(rs_values)

    slope, _ = np.polyfit(log_lag, log_rs, 1)

    return slope


def detrended_fluctuation(
    x: np.ndarray,
    scale_min: int = 4,
    scale_max: int | None = None,
    n_scales: int = 10,
) -> float:
    """Compute Detrended Fluctuation Analysis (DFA) exponent.

    DFA measures the self-similarity in a non-stationary time series
    by analyzing the fluctuations around local linear trends.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    scale_min : int, default=4
        Minimum scale (window size).
    scale_max : int, optional
        Maximum scale. If None, uses n//4.
    n_scales : int, default=10
        Number of scales to evaluate.

    Returns
    -------
    float
        DFA exponent (alpha):
        - alpha < 0.5: Anti-correlated
        - alpha = 0.5: White noise
        - alpha = 1.0: 1/f noise (pink noise)
        - alpha = 1.5: Brownian noise
        - alpha > 1.0: Non-stationary, unbounded

    References
    ----------
    Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides.
    Physical Review E, 49(2), 1685.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    if scale_max is None:
        scale_max = n // 4

    if scale_max < scale_min:
        return np.nan

    # Generate log-spaced scales
    scales = np.unique(
        np.logspace(np.log10(scale_min), np.log10(scale_max), n_scales).astype(int)
    )

    # Integrate the signal (cumulative sum of mean-adjusted signal)
    y = np.cumsum(x - np.mean(x))

    fluctuations = []

    for scale in scales:
        # Number of segments
        n_segments = n // scale

        if n_segments < 1:
            fluctuations.append(np.nan)
            continue

        # Compute fluctuation for each segment
        f2_segments = []

        for i in range(n_segments):
            segment = y[i * scale : (i + 1) * scale]
            t = np.arange(len(segment))

            # Fit linear trend and compute residuals
            coeffs = np.polyfit(t, segment, 1)
            trend = np.polyval(coeffs, t)
            residuals = segment - trend

            # Mean square fluctuation
            f2_segments.append(np.mean(residuals**2))

        # RMS fluctuation for this scale
        fluctuations.append(np.sqrt(np.mean(f2_segments)))

    # Remove NaN values
    valid = ~np.isnan(fluctuations)
    if np.sum(valid) < 2:
        return np.nan

    scales = scales[valid]
    fluctuations = np.array(fluctuations)[valid]

    # Linear regression of log(F) vs log(scale)
    log_scale = np.log(scales)
    log_fluct = np.log(fluctuations)

    slope, _ = np.polyfit(log_scale, log_fluct, 1)

    return slope


def lempel_ziv_complexity(x: np.ndarray, threshold: float | None = None) -> float:
    """Compute Lempel-Ziv complexity of a signal.

    LZC measures the complexity of a signal by counting the number of
    distinct patterns found during sequential parsing.

    Parameters
    ----------
    x : np.ndarray
        1D signal.
    threshold : float, optional
        Threshold for binarization. If None, uses median.

    Returns
    -------
    float
        Normalized Lempel-Ziv complexity (0-1).

    References
    ----------
    Lempel, A., & Ziv, J. (1976). On the complexity of finite sequences.
    IEEE Transactions on Information Theory, 22(1), 75-81.
    """
    x = np.asarray(x).flatten()
    n = len(x)

    # Binarize the signal
    if threshold is None:
        threshold = np.median(x)

    binary = (x >= threshold).astype(int)

    # Convert to string for easier pattern matching
    s = "".join(map(str, binary))

    # Lempel-Ziv parsing
    complexity = 1
    i = 0
    prefix_len = 1

    while prefix_len + i < n:
        # Check if the next substring is in the prefix
        if s[i : i + prefix_len + 1] in s[:i + prefix_len]:
            prefix_len += 1
        else:
            complexity += 1
            i += prefix_len
            prefix_len = 1

    # Normalize by theoretical maximum (n / log2(n))
    if n > 1:
        max_complexity = n / np.log2(n)
        normalized = complexity / max_complexity
    else:
        normalized = 0.0

    return normalized


class HiguchiFD(BaseFeatureExtractor):
    """Higuchi fractal dimension feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    kmax : int, default=10
        Maximum scale factor.

    Examples
    --------
    >>> hfd = HiguchiFD(kmax=10)
    >>> features = hfd.fit_transform(signals)
    """

    def __init__(self, fs: float | None = None, kmax: int = 10):
        super().__init__(fs=fs)
        self.kmax = kmax

    def fit(self, X, y=None, **fit_params) -> "HiguchiFD":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["higuchi_fd"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append([higuchi_fd(sig, self.kmax)])

        return np.array(features)


class PetrosianFD(BaseFeatureExtractor):
    """Petrosian fractal dimension feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.

    Examples
    --------
    >>> pfd = PetrosianFD()
    >>> features = pfd.fit_transform(signals)
    """

    def __init__(self, fs: float | None = None):
        super().__init__(fs=fs)

    def fit(self, X, y=None, **fit_params) -> "PetrosianFD":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["petrosian_fd"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append([petrosian_fd(sig)])

        return np.array(features)


class KatzFD(BaseFeatureExtractor):
    """Katz fractal dimension feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.

    Examples
    --------
    >>> kfd = KatzFD()
    >>> features = kfd.fit_transform(signals)
    """

    def __init__(self, fs: float | None = None):
        super().__init__(fs=fs)

    def fit(self, X, y=None, **fit_params) -> "KatzFD":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["katz_fd"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append([katz_fd(sig)])

        return np.array(features)


class HurstExponent(BaseFeatureExtractor):
    """Hurst exponent feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    max_lag : int, optional
        Maximum lag for R/S calculation.

    Examples
    --------
    >>> he = HurstExponent()
    >>> features = he.fit_transform(signals)
    """

    def __init__(self, fs: float | None = None, max_lag: int | None = None):
        super().__init__(fs=fs)
        self.max_lag = max_lag

    def fit(self, X, y=None, **fit_params) -> "HurstExponent":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["hurst_exponent"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append([hurst_exponent(sig, self.max_lag)])

        return np.array(features)


class DFA(BaseFeatureExtractor):
    """Detrended Fluctuation Analysis feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    scale_min : int, default=4
        Minimum scale.
    scale_max : int, optional
        Maximum scale.
    n_scales : int, default=10
        Number of scales.

    Examples
    --------
    >>> dfa = DFA()
    >>> features = dfa.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        scale_min: int = 4,
        scale_max: int | None = None,
        n_scales: int = 10,
    ):
        super().__init__(fs=fs)
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.n_scales = n_scales

    def fit(self, X, y=None, **fit_params) -> "DFA":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["dfa_alpha"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append(
                [detrended_fluctuation(sig, self.scale_min, self.scale_max, self.n_scales)]
            )

        return np.array(features)


class LempelZivComplexity(BaseFeatureExtractor):
    """Lempel-Ziv complexity feature extractor.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    threshold : float, optional
        Binarization threshold.

    Examples
    --------
    >>> lzc = LempelZivComplexity()
    >>> features = lzc.fit_transform(signals)
    """

    def __init__(self, fs: float | None = None, threshold: float | None = None):
        super().__init__(fs=fs)
        self.threshold = threshold

    def fit(self, X, y=None, **fit_params) -> "LempelZivComplexity":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["lempel_ziv_complexity"]
        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            features.append([lempel_ziv_complexity(sig, self.threshold)])

        return np.array(features)


class ComplexityFeatureExtractor(BaseFeatureExtractor):
    """Comprehensive complexity feature extractor.

    Extracts multiple complexity measures in a single transformer.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    include_higuchi : bool, default=True
        Include Higuchi fractal dimension.
    include_petrosian : bool, default=True
        Include Petrosian fractal dimension.
    include_katz : bool, default=True
        Include Katz fractal dimension.
    include_hurst : bool, default=True
        Include Hurst exponent.
    include_dfa : bool, default=True
        Include DFA exponent.
    include_lzc : bool, default=True
        Include Lempel-Ziv complexity.
    higuchi_kmax : int, default=10
        kmax for Higuchi FD.

    Examples
    --------
    >>> extractor = ComplexityFeatureExtractor()
    >>> features = extractor.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        include_higuchi: bool = True,
        include_petrosian: bool = True,
        include_katz: bool = True,
        include_hurst: bool = True,
        include_dfa: bool = True,
        include_lzc: bool = True,
        higuchi_kmax: int = 10,
    ):
        super().__init__(fs=fs)
        self.include_higuchi = include_higuchi
        self.include_petrosian = include_petrosian
        self.include_katz = include_katz
        self.include_hurst = include_hurst
        self.include_dfa = include_dfa
        self.include_lzc = include_lzc
        self.higuchi_kmax = higuchi_kmax

    def fit(self, X, y=None, **fit_params) -> "ComplexityFeatureExtractor":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = []
        if self.include_higuchi:
            self.feature_names_.append("higuchi_fd")
        if self.include_petrosian:
            self.feature_names_.append("petrosian_fd")
        if self.include_katz:
            self.feature_names_.append("katz_fd")
        if self.include_hurst:
            self.feature_names_.append("hurst_exponent")
        if self.include_dfa:
            self.feature_names_.append("dfa_alpha")
        if self.include_lzc:
            self.feature_names_.append("lempel_ziv_complexity")

        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        all_features = []
        for sig in X_2d:
            features = []

            if self.include_higuchi:
                features.append(higuchi_fd(sig, self.higuchi_kmax))
            if self.include_petrosian:
                features.append(petrosian_fd(sig))
            if self.include_katz:
                features.append(katz_fd(sig))
            if self.include_hurst:
                features.append(hurst_exponent(sig))
            if self.include_dfa:
                features.append(detrended_fluctuation(sig))
            if self.include_lzc:
                features.append(lempel_ziv_complexity(sig))

            all_features.append(features)

        return np.array(all_features)
