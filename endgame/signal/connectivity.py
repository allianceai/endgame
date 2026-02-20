"""Connectivity and EEG-specific feature extraction.

Provides sklearn-compatible feature extractors for:
- Coherence between channels
- Phase-locking value (PLV)
- Cross-correlation
- Burst and suppression detection
- Spike detection

These measures quantify relationships between channels and
detect specific patterns in biosignals.

References
----------
- Nunez & Srinivasan (2006): EEG coherence
- Lachaux et al. (1999): Phase-locking value
- Steriade et al. (1994): Burst-suppression patterns
"""


import numpy as np
from scipy import signal as scipy_signal

from endgame.signal.base import (
    BaseFeatureExtractor,
    ensure_2d_signals,
)


def coherence(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute magnitude-squared coherence between two signals.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals.
    fs : float
        Sample rate in Hz.
    nperseg : int, optional
        Segment length.
    noverlap : int, optional
        Overlap between segments.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins.
    coh : np.ndarray
        Coherence values (0-1).
    """
    nperseg = nperseg if nperseg is not None else min(256, len(x))
    freqs, coh = scipy_signal.coherence(x, y, fs=fs, nperseg=nperseg, noverlap=noverlap)
    return freqs, coh


def phase_locking_value(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    band: tuple[float, float] | None = None,
) -> float:
    """Compute phase-locking value between two signals.

    PLV measures the consistency of phase difference between signals.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals.
    fs : float
        Sample rate in Hz.
    band : tuple, optional
        Frequency band (low, high) to filter before computing PLV.

    Returns
    -------
    float
        Phase-locking value (0-1).

    References
    ----------
    Lachaux, J. P., et al. (1999). Measuring phase synchrony in brain signals.
    Human brain mapping, 8(4), 194-208.
    """
    from scipy.signal import butter, hilbert, sosfiltfilt

    # Optionally bandpass filter
    if band is not None:
        sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
        x = sosfiltfilt(sos, x)
        y = sosfiltfilt(sos, y)

    # Compute instantaneous phase using Hilbert transform
    phase_x = np.angle(hilbert(x))
    phase_y = np.angle(hilbert(y))

    # Phase difference
    phase_diff = phase_x - phase_y

    # PLV is the magnitude of the mean phase difference vector
    plv = np.abs(np.mean(np.exp(1j * phase_diff)))

    return plv


def cross_correlation(
    x: np.ndarray,
    y: np.ndarray,
    max_lag: int | None = None,
    normalize: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute cross-correlation between two signals.

    Parameters
    ----------
    x, y : np.ndarray
        Input signals.
    max_lag : int, optional
        Maximum lag. If None, uses len(x) - 1.
    normalize : bool, default=True
        Normalize to correlation coefficient (-1 to 1).

    Returns
    -------
    lags : np.ndarray
        Lag values.
    corr : np.ndarray
        Cross-correlation values.
    """
    n = len(x)
    if max_lag is None:
        max_lag = n - 1

    # Full cross-correlation
    corr = np.correlate(x - np.mean(x), y - np.mean(y), mode="full")

    if normalize:
        corr = corr / (n * np.std(x) * np.std(y))

    # Extract relevant lags
    mid = len(corr) // 2
    lags = np.arange(-max_lag, max_lag + 1)
    corr = corr[mid - max_lag : mid + max_lag + 1]

    return lags, corr


def detect_bursts(
    x: np.ndarray,
    fs: float,
    threshold_std: float = 2.0,
    min_duration_ms: float = 100.0,
) -> list[tuple[int, int]]:
    """Detect burst periods in a signal.

    Bursts are periods where signal amplitude exceeds a threshold.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sample rate in Hz.
    threshold_std : float, default=2.0
        Threshold in standard deviations above mean.
    min_duration_ms : float, default=100.0
        Minimum burst duration in milliseconds.

    Returns
    -------
    list of (start, end) tuples
        Burst intervals as sample indices.
    """
    # Compute envelope using Hilbert transform
    envelope = np.abs(scipy_signal.hilbert(x))

    # Threshold
    threshold = np.mean(envelope) + threshold_std * np.std(envelope)

    # Find above-threshold regions
    above = envelope > threshold

    # Find burst boundaries
    diff = np.diff(above.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if above[0]:
        starts = np.concatenate([[0], starts])
    if above[-1]:
        ends = np.concatenate([ends, [len(x)]])

    # Filter by minimum duration
    min_samples = int(min_duration_ms * fs / 1000)
    bursts = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            bursts.append((start, end))

    return bursts


def detect_suppressions(
    x: np.ndarray,
    fs: float,
    threshold_uv: float = 10.0,
    min_duration_ms: float = 500.0,
) -> list[tuple[int, int]]:
    """Detect suppression periods in a signal.

    Suppressions are periods of low amplitude activity.

    Parameters
    ----------
    x : np.ndarray
        Input signal (assumed to be in microvolts).
    fs : float
        Sample rate in Hz.
    threshold_uv : float, default=10.0
        Amplitude threshold in microvolts.
    min_duration_ms : float, default=500.0
        Minimum suppression duration in milliseconds.

    Returns
    -------
    list of (start, end) tuples
        Suppression intervals as sample indices.
    """
    # Compute envelope
    envelope = np.abs(scipy_signal.hilbert(x))

    # Find below-threshold regions
    below = envelope < threshold_uv

    # Find suppression boundaries
    diff = np.diff(below.astype(int))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    # Handle edge cases
    if below[0]:
        starts = np.concatenate([[0], starts])
    if below[-1]:
        ends = np.concatenate([ends, [len(x)]])

    # Filter by minimum duration
    min_samples = int(min_duration_ms * fs / 1000)
    suppressions = []
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            suppressions.append((start, end))

    return suppressions


def detect_spikes(
    x: np.ndarray,
    fs: float,
    threshold_std: float = 3.0,
    max_duration_ms: float = 70.0,
    min_duration_ms: float = 20.0,
) -> list[int]:
    """Detect spikes in a signal.

    Spikes are brief, high-amplitude transients.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sample rate in Hz.
    threshold_std : float, default=3.0
        Threshold in standard deviations.
    max_duration_ms : float, default=70.0
        Maximum spike duration in milliseconds.
    min_duration_ms : float, default=20.0
        Minimum spike duration in milliseconds.

    Returns
    -------
    list of int
        Spike peak locations as sample indices.
    """
    # Threshold
    threshold = np.mean(x) + threshold_std * np.std(x)

    # Find peaks above threshold
    peaks, properties = scipy_signal.find_peaks(
        np.abs(x),
        height=threshold,
        distance=int(min_duration_ms * fs / 1000),
    )

    # Filter by width
    max_samples = int(max_duration_ms * fs / 1000)
    min_samples = int(min_duration_ms * fs / 1000)

    widths, _, _, _ = scipy_signal.peak_widths(np.abs(x), peaks)

    valid_spikes = []
    for peak, width in zip(peaks, widths):
        if min_samples <= width <= max_samples:
            valid_spikes.append(peak)

    return valid_spikes


class CoherenceFeatureExtractor(BaseFeatureExtractor):
    """Extract coherence features between all channel pairs.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    bands : dict, optional
        Frequency bands for band-averaged coherence.
        Default includes standard EEG bands.
    nperseg : int, optional
        Segment length for coherence estimation.

    Examples
    --------
    >>> coh = CoherenceFeatureExtractor(fs=256)
    >>> features = coh.fit_transform(X)  # X: (n_trials, n_channels, n_samples)
    """

    DEFAULT_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 50),
    }

    def __init__(
        self,
        fs: float,
        bands: dict[str, tuple[float, float]] | None = None,
        nperseg: int | None = None,
    ):
        super().__init__(fs=fs)
        self.bands = bands if bands is not None else self.DEFAULT_BANDS.copy()
        self.nperseg = nperseg

    def fit(self, X, y=None, **fit_params) -> "CoherenceFeatureExtractor":
        """Fit the extractor.

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
            raise ValueError(f"Expected 3D array, got shape {X.shape}")

        self.n_channels_ = X.shape[1]

        # Number of channel pairs
        n_pairs = self.n_channels_ * (self.n_channels_ - 1) // 2

        # Feature names
        self.feature_names_ = []
        self._channel_pairs = []
        for i in range(self.n_channels_):
            for j in range(i + 1, self.n_channels_):
                self._channel_pairs.append((i, j))
                for band_name in self.bands:
                    self.feature_names_.append(f"coh_{i}_{j}_{band_name}")

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Extract coherence features.

        Parameters
        ----------
        X : np.ndarray
            Multi-channel signals of shape (n_trials, n_channels, n_samples).

        Returns
        -------
        np.ndarray
            Coherence features of shape (n_trials, n_features).
        """
        X = np.asarray(X)
        n_trials = X.shape[0]

        all_features = []

        for trial in range(n_trials):
            features = []
            for i, j in self._channel_pairs:
                freqs, coh = coherence(
                    X[trial, i], X[trial, j], self.fs, self.nperseg
                )

                # Band-averaged coherence
                for band_name, (low, high) in self.bands.items():
                    mask = (freqs >= low) & (freqs <= high)
                    band_coh = np.mean(coh[mask]) if np.any(mask) else 0.0
                    features.append(band_coh)

            all_features.append(features)

        return np.array(all_features)


class PLVFeatureExtractor(BaseFeatureExtractor):
    """Extract phase-locking value features between channel pairs.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    bands : dict, optional
        Frequency bands for band-specific PLV.

    Examples
    --------
    >>> plv = PLVFeatureExtractor(fs=256)
    >>> features = plv.fit_transform(X)
    """

    DEFAULT_BANDS = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    def __init__(
        self,
        fs: float,
        bands: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(fs=fs)
        self.bands = bands if bands is not None else self.DEFAULT_BANDS.copy()

    def fit(self, X, y=None, **fit_params) -> "PLVFeatureExtractor":
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")

        self.n_channels_ = X.shape[1]

        self.feature_names_ = []
        self._channel_pairs = []
        for i in range(self.n_channels_):
            for j in range(i + 1, self.n_channels_):
                self._channel_pairs.append((i, j))
                for band_name in self.bands:
                    self.feature_names_.append(f"plv_{i}_{j}_{band_name}")

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)
        n_trials = X.shape[0]

        all_features = []

        for trial in range(n_trials):
            features = []
            for i, j in self._channel_pairs:
                for band_name, band in self.bands.items():
                    plv = phase_locking_value(
                        X[trial, i], X[trial, j], self.fs, band
                    )
                    features.append(plv)

            all_features.append(features)

        return np.array(all_features)


class BurstSuppressionFeatures(BaseFeatureExtractor):
    """Extract burst-suppression features from signals.

    Commonly used in EEG analysis for detecting anesthesia depth
    and certain pathological states.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    burst_threshold_std : float, default=2.0
        Threshold for burst detection.
    suppression_threshold : float, default=10.0
        Threshold for suppression detection.
    min_burst_ms : float, default=100.0
        Minimum burst duration.
    min_suppression_ms : float, default=500.0
        Minimum suppression duration.

    Examples
    --------
    >>> bsr = BurstSuppressionFeatures(fs=256)
    >>> features = bsr.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float,
        burst_threshold_std: float = 2.0,
        suppression_threshold: float = 10.0,
        min_burst_ms: float = 100.0,
        min_suppression_ms: float = 500.0,
    ):
        super().__init__(fs=fs)
        self.burst_threshold_std = burst_threshold_std
        self.suppression_threshold = suppression_threshold
        self.min_burst_ms = min_burst_ms
        self.min_suppression_ms = min_suppression_ms

    def fit(self, X, y=None, **fit_params) -> "BurstSuppressionFeatures":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = [
            "n_bursts",
            "burst_rate",
            "mean_burst_duration",
            "std_burst_duration",
            "n_suppressions",
            "suppression_rate",
            "mean_suppression_duration",
            "std_suppression_duration",
            "burst_suppression_ratio",
        ]

        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        all_features = []

        for sig in X_2d:
            # Detect bursts
            bursts = detect_bursts(
                sig, self.fs, self.burst_threshold_std, self.min_burst_ms
            )

            # Detect suppressions
            suppressions = detect_suppressions(
                sig, self.fs, self.suppression_threshold, self.min_suppression_ms
            )

            # Compute features
            n_samples = len(sig)
            duration_sec = n_samples / self.fs

            # Burst statistics
            n_bursts = len(bursts)
            burst_rate = n_bursts / duration_sec if duration_sec > 0 else 0

            if n_bursts > 0:
                burst_durations = [(end - start) / self.fs * 1000 for start, end in bursts]
                mean_burst_dur = np.mean(burst_durations)
                std_burst_dur = np.std(burst_durations) if n_bursts > 1 else 0
                total_burst_samples = sum(end - start for start, end in bursts)
            else:
                mean_burst_dur = 0
                std_burst_dur = 0
                total_burst_samples = 0

            # Suppression statistics
            n_suppressions = len(suppressions)
            suppression_rate = n_suppressions / duration_sec if duration_sec > 0 else 0

            if n_suppressions > 0:
                supp_durations = [(end - start) / self.fs * 1000 for start, end in suppressions]
                mean_supp_dur = np.mean(supp_durations)
                std_supp_dur = np.std(supp_durations) if n_suppressions > 1 else 0
                total_supp_samples = sum(end - start for start, end in suppressions)
            else:
                mean_supp_dur = 0
                std_supp_dur = 0
                total_supp_samples = 0

            # Burst-suppression ratio
            bsr = total_supp_samples / n_samples if n_samples > 0 else 0

            features = [
                n_bursts,
                burst_rate,
                mean_burst_dur,
                std_burst_dur,
                n_suppressions,
                suppression_rate,
                mean_supp_dur,
                std_supp_dur,
                bsr,
            ]

            all_features.append(features)

        return np.array(all_features)


class SpikeFeatures(BaseFeatureExtractor):
    """Extract spike-related features from signals.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    threshold_std : float, default=3.0
        Detection threshold in standard deviations.
    max_duration_ms : float, default=70.0
        Maximum spike duration.

    Examples
    --------
    >>> spike_feat = SpikeFeatures(fs=256)
    >>> features = spike_feat.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float,
        threshold_std: float = 3.0,
        max_duration_ms: float = 70.0,
    ):
        super().__init__(fs=fs)
        self.threshold_std = threshold_std
        self.max_duration_ms = max_duration_ms

    def fit(self, X, y=None, **fit_params) -> "SpikeFeatures":
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        self.feature_names_ = [
            "n_spikes",
            "spike_rate",
            "mean_spike_amplitude",
            "std_spike_amplitude",
            "mean_spike_interval",
        ]

        return self

    def transform(self, X) -> np.ndarray:
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        all_features = []

        for sig in X_2d:
            spikes = detect_spikes(sig, self.fs, self.threshold_std, self.max_duration_ms)

            n_samples = len(sig)
            duration_sec = n_samples / self.fs

            n_spikes = len(spikes)
            spike_rate = n_spikes / duration_sec if duration_sec > 0 else 0

            if n_spikes > 0:
                amplitudes = np.abs(sig[spikes])
                mean_amp = np.mean(amplitudes)
                std_amp = np.std(amplitudes) if n_spikes > 1 else 0

                if n_spikes > 1:
                    intervals = np.diff(spikes) / self.fs * 1000  # in ms
                    mean_interval = np.mean(intervals)
                else:
                    mean_interval = 0
            else:
                mean_amp = 0
                std_amp = 0
                mean_interval = 0

            features = [n_spikes, spike_rate, mean_amp, std_amp, mean_interval]
            all_features.append(features)

        return np.array(all_features)


class ConnectivityFeatureExtractor(BaseFeatureExtractor):
    """Comprehensive connectivity feature extractor.

    Combines coherence, PLV, and cross-correlation features.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    include_coherence : bool, default=True
        Include coherence features.
    include_plv : bool, default=True
        Include PLV features.
    bands : dict, optional
        Frequency bands for analysis.

    Examples
    --------
    >>> conn = ConnectivityFeatureExtractor(fs=256)
    >>> features = conn.fit_transform(X)
    """

    DEFAULT_BANDS = {
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
    }

    def __init__(
        self,
        fs: float,
        include_coherence: bool = True,
        include_plv: bool = True,
        bands: dict[str, tuple[float, float]] | None = None,
    ):
        super().__init__(fs=fs)
        self.include_coherence = include_coherence
        self.include_plv = include_plv
        self.bands = bands if bands is not None else self.DEFAULT_BANDS.copy()

    def fit(self, X, y=None, **fit_params) -> "ConnectivityFeatureExtractor":
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError(f"Expected 3D array, got shape {X.shape}")

        self._extractors = []
        self.feature_names_ = []

        if self.include_coherence:
            coh = CoherenceFeatureExtractor(fs=self.fs, bands=self.bands)
            coh.fit(X)
            self._extractors.append(coh)
            self.feature_names_.extend(coh.feature_names_)

        if self.include_plv:
            plv = PLVFeatureExtractor(fs=self.fs, bands=self.bands)
            plv.fit(X)
            self._extractors.append(plv)
            self.feature_names_.extend(plv.feature_names_)

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        X = np.asarray(X)

        feature_arrays = []
        for ext in self._extractors:
            feats = ext.transform(X)
            feature_arrays.append(feats)

        return np.hstack(feature_arrays)
