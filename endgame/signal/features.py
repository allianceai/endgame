"""Time-domain feature extraction for signal processing.

Provides sklearn-compatible feature extractors for time-domain analysis:
- Statistical features (mean, std, skew, kurtosis, etc.)
- Hjorth parameters (activity, mobility, complexity)
- Zero-crossing features
- Peak detection features
- Comprehensive time-domain feature sets

References
----------
- neurokit2: Time-domain feature calculations
- tsfel: Feature extraction library
- scipy.stats: Statistical measures
"""

from typing import Any

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats

from endgame.signal.base import (
    BaseFeatureExtractor,
    ensure_2d_signals,
)


class StatisticalFeatures(BaseFeatureExtractor):
    """Extract statistical features from signals.

    Computes basic and higher-order statistical measures.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz (not used but kept for consistency).
    features : list of str, optional
        Features to compute. Default includes all.
        Options: 'mean', 'std', 'var', 'min', 'max', 'range',
        'median', 'mad', 'iqr', 'skew', 'kurtosis', 'rms',
        'energy', 'line_length', 'crest_factor'.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> extractor = StatisticalFeatures()
    >>> features = extractor.fit_transform(signals)
    """

    ALL_FEATURES = [
        "mean",
        "std",
        "var",
        "min",
        "max",
        "range",
        "median",
        "mad",
        "iqr",
        "skew",
        "kurtosis",
        "rms",
        "energy",
        "line_length",
        "crest_factor",
    ]

    def __init__(
        self,
        fs: float | None = None,
        features: list[str] | None = None,
    ):
        super().__init__(fs=fs)
        self.features = features if features is not None else self.ALL_FEATURES.copy()

        # Validate features
        for f in self.features:
            if f not in self.ALL_FEATURES:
                raise ValueError(f"Unknown feature: {f}")

    def fit(self, X, y=None, **fit_params) -> "StatisticalFeatures":
        """Fit the extractor.

        Parameters
        ----------
        X : array-like
            Input signals.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = [f"stat_{f}" for f in self.features]
        return self

    def transform(self, X) -> np.ndarray:
        """Extract statistical features.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            sig_features = self._compute_features(sig)
            features.append(sig_features)

        return np.array(features)

    def _compute_features(self, x: np.ndarray) -> list[float]:
        """Compute features for a single signal."""
        feats = []

        for f in self.features:
            if f == "mean":
                feats.append(np.mean(x))
            elif f == "std":
                feats.append(np.std(x, ddof=1))
            elif f == "var":
                feats.append(np.var(x, ddof=1))
            elif f == "min":
                feats.append(np.min(x))
            elif f == "max":
                feats.append(np.max(x))
            elif f == "range":
                feats.append(np.ptp(x))
            elif f == "median":
                feats.append(np.median(x))
            elif f == "mad":
                # Median absolute deviation
                feats.append(np.median(np.abs(x - np.median(x))))
            elif f == "iqr":
                feats.append(stats.iqr(x))
            elif f == "skew":
                feats.append(stats.skew(x))
            elif f == "kurtosis":
                feats.append(stats.kurtosis(x))
            elif f == "rms":
                feats.append(np.sqrt(np.mean(x**2)))
            elif f == "energy":
                feats.append(np.sum(x**2))
            elif f == "line_length":
                feats.append(np.sum(np.abs(np.diff(x))))
            elif f == "crest_factor":
                rms = np.sqrt(np.mean(x**2))
                feats.append(np.max(np.abs(x)) / (rms + 1e-10))

        return feats


class HjorthParameters(BaseFeatureExtractor):
    """Extract Hjorth parameters from signals.

    Computes the three Hjorth parameters:
    - Activity: Signal variance (power)
    - Mobility: Mean frequency (normalized first derivative variance)
    - Complexity: Change in frequency (bandwidth)

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    normalize : bool, default=False
        If True, normalize by signal variance.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    References
    ----------
    Hjorth, B. (1970). EEG analysis based on time domain properties.
    Electroencephalography and clinical neurophysiology, 29(3), 306-310.

    Examples
    --------
    >>> hjorth = HjorthParameters()
    >>> features = hjorth.fit_transform(eeg_signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        normalize: bool = False,
    ):
        super().__init__(fs=fs)
        self.normalize = normalize

    def fit(self, X, y=None, **fit_params) -> "HjorthParameters":
        """Fit the extractor.

        Parameters
        ----------
        X : array-like
            Input signals.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)
        self.feature_names_ = ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
        return self

    def transform(self, X) -> np.ndarray:
        """Extract Hjorth parameters.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, 3)
            Hjorth parameters [activity, mobility, complexity].
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            activity, mobility, complexity = self._compute_hjorth(sig)
            features.append([activity, mobility, complexity])

        return np.array(features)

    def _compute_hjorth(self, x: np.ndarray) -> tuple[float, float, float]:
        """Compute Hjorth parameters for a single signal."""
        # First derivative
        dx = np.diff(x)
        # Second derivative
        ddx = np.diff(dx)

        # Activity: variance of the signal
        activity = np.var(x)

        # Mobility: sqrt(var(dx) / var(x))
        var_dx = np.var(dx)
        mobility = np.sqrt(var_dx / (activity + 1e-10))

        # Complexity: mobility(dx) / mobility(x)
        var_ddx = np.var(ddx)
        mobility_dx = np.sqrt(var_ddx / (var_dx + 1e-10))
        complexity = mobility_dx / (mobility + 1e-10)

        if self.normalize:
            activity = activity / (activity + 1e-10)

        return activity, mobility, complexity


class ZeroCrossingFeatures(BaseFeatureExtractor):
    """Extract zero-crossing related features.

    Computes features based on signal zero crossings and
    threshold crossings.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    threshold : float, default=0.0
        Threshold for crossing detection.
    compute_rate : bool, default=True
        If True, compute crossing rate (requires fs).

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> zc = ZeroCrossingFeatures(fs=256)
    >>> features = zc.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        threshold: float = 0.0,
        compute_rate: bool = True,
    ):
        super().__init__(fs=fs)
        self.threshold = threshold
        self.compute_rate = compute_rate

    def fit(self, X, y=None, **fit_params) -> "ZeroCrossingFeatures":
        """Fit the extractor.

        Parameters
        ----------
        X : array-like
            Input signals.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = [
            "zero_crossing_count",
            "zero_crossing_mean_interval",
            "zero_crossing_std_interval",
        ]
        if self.compute_rate and self.fs is not None:
            self.feature_names_.append("zero_crossing_rate")

        return self

    def transform(self, X) -> np.ndarray:
        """Extract zero-crossing features.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            sig_features = self._compute_features(sig)
            features.append(sig_features)

        return np.array(features)

    def _compute_features(self, x: np.ndarray) -> list[float]:
        """Compute zero-crossing features for a single signal."""
        # Shift by threshold
        x_shifted = x - self.threshold

        # Find zero crossings
        sign_changes = np.diff(np.sign(x_shifted))
        crossings = np.where(sign_changes != 0)[0]

        # Count
        count = len(crossings)

        # Intervals between crossings
        if count > 1:
            intervals = np.diff(crossings)
            mean_interval = np.mean(intervals)
            std_interval = np.std(intervals)
        else:
            mean_interval = len(x) if count == 0 else len(x) / 2
            std_interval = 0.0

        feats = [count, mean_interval, std_interval]

        # Crossing rate
        if self.compute_rate and self.fs is not None:
            duration = len(x) / self.fs
            rate = count / duration
            feats.append(rate)

        return feats


class PeakFeatures(BaseFeatureExtractor):
    """Extract peak-related features from signals.

    Detects peaks and computes statistics about their
    amplitude, width, and distribution.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    height : float or tuple, optional
        Required height of peaks.
    threshold : float, optional
        Required threshold of peaks.
    distance : int, optional
        Required minimal horizontal distance between peaks.
    prominence : float or tuple, optional
        Required prominence of peaks.
    width : float or tuple, optional
        Required width of peaks in samples.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> peaks = PeakFeatures(fs=256, prominence=0.5)
    >>> features = peaks.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float | None = None,
        height: float | None = None,
        threshold: float | None = None,
        distance: int | None = None,
        prominence: float | None = None,
        width: float | None = None,
    ):
        super().__init__(fs=fs)
        self.height = height
        self.threshold = threshold
        self.distance = distance
        self.prominence = prominence
        self.width = width

    def fit(self, X, y=None, **fit_params) -> "PeakFeatures":
        """Fit the extractor.

        Parameters
        ----------
        X : array-like
            Input signals.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = [
            "peak_count",
            "peak_mean_height",
            "peak_std_height",
            "peak_max_height",
            "peak_mean_prominence",
            "peak_mean_width",
            "peak_mean_interval",
        ]
        if self.fs is not None:
            self.feature_names_.append("peak_rate")

        return self

    def transform(self, X) -> np.ndarray:
        """Extract peak features.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        features = []
        for sig in X_2d:
            sig_features = self._compute_features(sig)
            features.append(sig_features)

        return np.array(features)

    def _compute_features(self, x: np.ndarray) -> list[float]:
        """Compute peak features for a single signal."""
        # Find peaks
        peaks, properties = scipy_signal.find_peaks(
            x,
            height=self.height,
            threshold=self.threshold,
            distance=self.distance,
            prominence=self.prominence,
            width=self.width,
        )

        count = len(peaks)

        if count > 0:
            heights = x[peaks]
            mean_height = np.mean(heights)
            std_height = np.std(heights) if count > 1 else 0.0
            max_height = np.max(heights)

            # Get prominence and width
            if "prominences" in properties:
                mean_prominence = np.mean(properties["prominences"])
            else:
                proms, _, _ = scipy_signal.peak_prominences(x, peaks)
                mean_prominence = np.mean(proms) if len(proms) > 0 else 0.0

            if "widths" in properties:
                mean_width = np.mean(properties["widths"])
            else:
                widths, _, _, _ = scipy_signal.peak_widths(x, peaks)
                mean_width = np.mean(widths) if len(widths) > 0 else 0.0

            # Peak intervals
            if count > 1:
                intervals = np.diff(peaks)
                mean_interval = np.mean(intervals)
            else:
                mean_interval = len(x)
        else:
            mean_height = 0.0
            std_height = 0.0
            max_height = 0.0
            mean_prominence = 0.0
            mean_width = 0.0
            mean_interval = len(x)

        feats = [
            count,
            mean_height,
            std_height,
            max_height,
            mean_prominence,
            mean_width,
            mean_interval,
        ]

        # Peak rate
        if self.fs is not None:
            duration = len(x) / self.fs
            rate = count / duration
            feats.append(rate)

        return feats


class TimeDomainFeatures(BaseFeatureExtractor):
    """Comprehensive time-domain feature extraction.

    Combines multiple time-domain feature extractors into a single
    transformer for convenience.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    include_statistical : bool, default=True
        Include statistical features.
    include_hjorth : bool, default=True
        Include Hjorth parameters.
    include_zero_crossing : bool, default=True
        Include zero-crossing features.
    include_peaks : bool, default=True
        Include peak features.
    statistical_features : list of str, optional
        Specific statistical features to include.
    peak_params : dict, optional
        Parameters for peak detection.

    Attributes
    ----------
    feature_names_ : list of str
        Names of all extracted features.

    Examples
    --------
    >>> extractor = TimeDomainFeatures(fs=256)
    >>> features = extractor.fit_transform(signals)
    >>> print(extractor.feature_names_)
    """

    def __init__(
        self,
        fs: float | None = None,
        include_statistical: bool = True,
        include_hjorth: bool = True,
        include_zero_crossing: bool = True,
        include_peaks: bool = True,
        statistical_features: list[str] | None = None,
        peak_params: dict[str, Any] | None = None,
    ):
        super().__init__(fs=fs)
        self.include_statistical = include_statistical
        self.include_hjorth = include_hjorth
        self.include_zero_crossing = include_zero_crossing
        self.include_peaks = include_peaks
        self.statistical_features = statistical_features
        self.peak_params = peak_params or {}

    def fit(self, X, y=None, **fit_params) -> "TimeDomainFeatures":
        """Fit all feature extractors.

        Parameters
        ----------
        X : array-like
            Input signals.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self._extractors = []
        self.feature_names_ = []

        if self.include_statistical:
            ext = StatisticalFeatures(fs=self.fs, features=self.statistical_features)
            ext.fit(X)
            self._extractors.append(ext)
            self.feature_names_.extend(ext.feature_names_)

        if self.include_hjorth:
            ext = HjorthParameters(fs=self.fs)
            ext.fit(X)
            self._extractors.append(ext)
            self.feature_names_.extend(ext.feature_names_)

        if self.include_zero_crossing:
            ext = ZeroCrossingFeatures(fs=self.fs)
            ext.fit(X)
            self._extractors.append(ext)
            self.feature_names_.extend(ext.feature_names_)

        if self.include_peaks:
            ext = PeakFeatures(fs=self.fs, **self.peak_params)
            ext.fit(X)
            self._extractors.append(ext)
            self.feature_names_.extend(ext.feature_names_)

        return self

    def transform(self, X) -> np.ndarray:
        """Extract all time-domain features.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        X = self._validate_signal(X)

        feature_arrays = []
        for ext in self._extractors:
            feats = ext.transform(X)
            feature_arrays.append(feats)

        return np.hstack(feature_arrays)


def compute_rms(x: np.ndarray) -> float:
    """Compute root mean square of a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    Returns
    -------
    float
        RMS value.
    """
    return np.sqrt(np.mean(x**2))


def compute_energy(x: np.ndarray) -> float:
    """Compute signal energy (sum of squared values).

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    Returns
    -------
    float
        Signal energy.
    """
    return np.sum(x**2)


def compute_line_length(x: np.ndarray) -> float:
    """Compute line length (sum of absolute differences).

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    Returns
    -------
    float
        Line length.
    """
    return np.sum(np.abs(np.diff(x)))


def compute_hjorth(x: np.ndarray) -> tuple[float, float, float]:
    """Compute Hjorth parameters.

    Parameters
    ----------
    x : np.ndarray
        Input signal.

    Returns
    -------
    tuple of (activity, mobility, complexity)
    """
    ext = HjorthParameters()
    ext.fit(x.reshape(1, -1))
    result = ext.transform(x.reshape(1, -1))
    return tuple(result[0])


def count_zero_crossings(x: np.ndarray, threshold: float = 0.0) -> int:
    """Count zero crossings in a signal.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    threshold : float, default=0.0
        Threshold for crossing detection.

    Returns
    -------
    int
        Number of zero crossings.
    """
    x_shifted = x - threshold
    sign_changes = np.diff(np.sign(x_shifted))
    return int(np.sum(sign_changes != 0))
