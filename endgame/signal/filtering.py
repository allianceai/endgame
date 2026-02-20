"""Digital filtering for signal processing.

Provides sklearn-compatible wrappers around scipy.signal filters
with sensible defaults for biosignal processing.

Supported filters:
- Butterworth (lowpass, highpass, bandpass, bandstop)
- FIR (window-based design)
- Savitzky-Golay (smoothing with polynomial fitting)
- Notch (powerline interference removal)
- Median (impulse noise removal)

Examples
--------
>>> from endgame.signal import ButterworthFilter, NotchFilter
>>>
>>> # Bandpass filter for EEG
>>> bp = ButterworthFilter(lowcut=0.5, highcut=50, fs=256, order=4)
>>> filtered = bp.fit_transform(raw_eeg)
>>>
>>> # Remove powerline noise
>>> notch = NotchFilter(freq=50, fs=256, Q=30)
>>> clean = notch.fit_transform(filtered)
"""

from typing import Literal

import numpy as np
from scipy import signal as scipy_signal

from endgame.signal.base import (
    BaseSignalTransformer,
    ensure_2d_signals,
    restore_shape,
)


class ButterworthFilter(BaseSignalTransformer):
    """Butterworth IIR filter with zero-phase filtering.

    Applies a Butterworth filter using scipy.signal.filtfilt for
    zero-phase distortion (forward-backward filtering).

    Parameters
    ----------
    lowcut : float, optional
        Low cutoff frequency in Hz. Required for highpass/bandpass.
    highcut : float, optional
        High cutoff frequency in Hz. Required for lowpass/bandpass.
    fs : float
        Sample rate in Hz.
    order : int, default=4
        Filter order. Higher = sharper rolloff but more ringing.
    btype : str, optional
        Filter type: 'lowpass', 'highpass', 'bandpass', 'bandstop'.
        Auto-detected from lowcut/highcut if not specified.
    padlen : int, optional
        Padding length for filtfilt. Default is 3 * max(len(a), len(b)).

    Examples
    --------
    >>> # Bandpass filter 0.5-50 Hz
    >>> filt = ButterworthFilter(lowcut=0.5, highcut=50, fs=256, order=4)
    >>> filtered = filt.fit_transform(signal)

    >>> # Highpass filter (remove DC and drift)
    >>> filt = ButterworthFilter(lowcut=0.1, fs=256)
    >>> filtered = filt.fit_transform(signal)

    Notes
    -----
    Uses second-order sections (SOS) format for numerical stability,
    especially at higher orders.
    """

    def __init__(
        self,
        lowcut: float | None = None,
        highcut: float | None = None,
        fs: float = None,
        order: int = 4,
        btype: Literal["lowpass", "highpass", "bandpass", "bandstop"] | None = None,
        padlen: int | None = None,
    ):
        super().__init__(fs=fs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.order = order
        self.btype = btype
        self.padlen = padlen

        self._sos = None

    def _design_filter(self) -> np.ndarray:
        """Design the Butterworth filter."""
        fs = self._check_fs()
        nyq = fs / 2

        # Determine filter type
        if self.btype is not None:
            btype = self.btype
        elif self.lowcut is not None and self.highcut is not None:
            btype = "bandpass"
        elif self.lowcut is not None:
            btype = "highpass"
        elif self.highcut is not None:
            btype = "lowpass"
        else:
            raise ValueError("Must specify lowcut and/or highcut")

        # Determine critical frequencies
        if btype == "lowpass":
            Wn = self.highcut / nyq
        elif btype == "highpass":
            Wn = self.lowcut / nyq
        elif btype in ("bandpass", "bandstop"):
            Wn = [self.lowcut / nyq, self.highcut / nyq]
        else:
            raise ValueError(f"Unknown btype: {btype}")

        # Validate frequencies
        Wn = np.asarray(Wn)
        if np.any(Wn <= 0) or np.any(Wn >= 1):
            raise ValueError(
                f"Critical frequencies must be between 0 and Nyquist ({nyq} Hz). "
                f"Got: lowcut={self.lowcut}, highcut={self.highcut}"
            )

        # Design filter using SOS for stability
        sos = scipy_signal.butter(self.order, Wn, btype=btype, output="sos")
        return sos

    def fit(self, X, y=None, **fit_params) -> "ButterworthFilter":
        """Fit the filter (designs the filter coefficients)."""
        super().fit(X, y, **fit_params)
        self._sos = self._design_filter()
        return self

    def transform(self, X) -> np.ndarray:
        """Apply zero-phase Butterworth filter.

        Parameters
        ----------
        X : array-like
            Input signal(s).

        Returns
        -------
        np.ndarray
            Filtered signal(s).
        """
        self._check_is_fitted()

        X = self._validate_signal(X, ensure_2d=False)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        if self.copy:
            X_2d = X_2d.copy()

        # Apply filter to each channel
        filtered = np.zeros_like(X_2d)
        for i in range(X_2d.shape[0]):
            filtered[i] = scipy_signal.sosfiltfilt(
                self._sos, X_2d[i], padlen=self.padlen
            )

        return restore_shape(filtered, was_1d, original_shape)

    def get_frequency_response(
        self,
        n_points: int = 512,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the filter's frequency response.

        Parameters
        ----------
        n_points : int, default=512
            Number of frequency points.

        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz.
        response : np.ndarray
            Magnitude response (absolute value).
        """
        self._check_is_fitted()
        fs = self._check_fs()

        w, h = scipy_signal.sosfreqz(self._sos, worN=n_points, fs=fs)
        return w, np.abs(h)


class FIRFilter(BaseSignalTransformer):
    """FIR filter using window-based design.

    Finite Impulse Response filter designed using the window method.
    Linear phase (no phase distortion) when using filtfilt.

    Parameters
    ----------
    lowcut : float, optional
        Low cutoff frequency in Hz.
    highcut : float, optional
        High cutoff frequency in Hz.
    fs : float
        Sample rate in Hz.
    numtaps : int, default=101
        Number of filter taps (filter length). Must be odd for Type I filter.
    window : str, default='hamming'
        Window function: 'hamming', 'hann', 'blackman', 'kaiser', etc.
    pass_zero : bool or str, optional
        If True, DC gain is 1. Auto-detected if not specified.

    Examples
    --------
    >>> # Bandpass FIR filter
    >>> filt = FIRFilter(lowcut=1, highcut=40, fs=256, numtaps=101)
    >>> filtered = filt.fit_transform(signal)

    Notes
    -----
    FIR filters are inherently stable and have linear phase, but require
    more taps than IIR filters for sharp transitions.
    """

    def __init__(
        self,
        lowcut: float | None = None,
        highcut: float | None = None,
        fs: float = None,
        numtaps: int = 101,
        window: str = "hamming",
        pass_zero: bool | str | None = None,
    ):
        super().__init__(fs=fs)
        self.lowcut = lowcut
        self.highcut = highcut
        self.numtaps = numtaps
        self.window = window
        self.pass_zero = pass_zero

        self._b = None

    def _design_filter(self) -> np.ndarray:
        """Design the FIR filter."""
        fs = self._check_fs()
        nyq = fs / 2

        # Ensure odd number of taps
        numtaps = self.numtaps
        if numtaps % 2 == 0:
            numtaps += 1

        # Determine cutoff and pass_zero
        if self.lowcut is not None and self.highcut is not None:
            cutoff = [self.lowcut, self.highcut]
            pass_zero = self.pass_zero if self.pass_zero is not None else False
        elif self.lowcut is not None:
            cutoff = self.lowcut
            pass_zero = self.pass_zero if self.pass_zero is not None else False
        elif self.highcut is not None:
            cutoff = self.highcut
            pass_zero = self.pass_zero if self.pass_zero is not None else True
        else:
            raise ValueError("Must specify lowcut and/or highcut")

        b = scipy_signal.firwin(
            numtaps,
            cutoff,
            fs=fs,
            window=self.window,
            pass_zero=pass_zero,
        )
        return b

    def fit(self, X, y=None, **fit_params) -> "FIRFilter":
        """Fit the filter."""
        super().fit(X, y, **fit_params)
        self._b = self._design_filter()
        return self

    def transform(self, X) -> np.ndarray:
        """Apply FIR filter."""
        self._check_is_fitted()

        X = self._validate_signal(X, ensure_2d=False)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        if self.copy:
            X_2d = X_2d.copy()

        # Apply filter using filtfilt for zero-phase
        filtered = np.zeros_like(X_2d)
        for i in range(X_2d.shape[0]):
            filtered[i] = scipy_signal.filtfilt(self._b, [1.0], X_2d[i])

        return restore_shape(filtered, was_1d, original_shape)


class SavgolFilter(BaseSignalTransformer):
    """Savitzky-Golay smoothing filter.

    Performs polynomial smoothing using local least-squares fitting.
    Preserves signal features better than simple moving average.

    Parameters
    ----------
    window_length : int, default=11
        Window size in samples. Must be odd and > polyorder.
    polyorder : int, default=3
        Polynomial order for fitting.
    deriv : int, default=0
        Derivative order (0 = smoothing, 1 = first derivative, etc.)
    delta : float, default=1.0
        Sample spacing (for derivative calculation).
    mode : str, default='interp'
        Edge handling: 'mirror', 'constant', 'nearest', 'wrap', 'interp'.

    Examples
    --------
    >>> # Smooth signal
    >>> filt = SavgolFilter(window_length=11, polyorder=3)
    >>> smoothed = filt.fit_transform(noisy_signal)

    >>> # Compute smooth derivative
    >>> filt = SavgolFilter(window_length=11, polyorder=3, deriv=1, delta=1/fs)
    >>> velocity = filt.fit_transform(position)

    Notes
    -----
    Savitzky-Golay filters are particularly good for preserving peaks
    and other high-frequency features while removing noise.
    """

    def __init__(
        self,
        window_length: int = 11,
        polyorder: int = 3,
        deriv: int = 0,
        delta: float = 1.0,
        mode: str = "interp",
        fs: float | None = None,
    ):
        super().__init__(fs=fs)
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.mode = mode

    def fit(self, X, y=None, **fit_params) -> "SavgolFilter":
        """Fit the filter."""
        super().fit(X, y, **fit_params)

        # Validate parameters
        if self.window_length % 2 == 0:
            raise ValueError("window_length must be odd")
        if self.window_length <= self.polyorder:
            raise ValueError("window_length must be > polyorder")

        return self

    def transform(self, X) -> np.ndarray:
        """Apply Savitzky-Golay filter."""
        self._check_is_fitted()

        X = self._validate_signal(X, ensure_2d=False)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        # Use delta based on fs if available
        delta = self.delta
        if self.fs is not None and self.deriv > 0:
            delta = 1.0 / self.fs

        # Apply filter
        filtered = scipy_signal.savgol_filter(
            X_2d,
            self.window_length,
            self.polyorder,
            deriv=self.deriv,
            delta=delta,
            axis=-1,
            mode=self.mode,
        )

        return restore_shape(filtered, was_1d, original_shape)


class NotchFilter(BaseSignalTransformer):
    """Notch filter for removing specific frequencies.

    Typically used to remove powerline interference (50/60 Hz)
    and its harmonics.

    Parameters
    ----------
    freq : float or List[float]
        Notch frequency/frequencies in Hz.
    fs : float
        Sample rate in Hz.
    Q : float, default=30
        Quality factor. Higher Q = narrower notch.
    harmonics : int, default=0
        Number of harmonics to also filter (0 = fundamental only).

    Examples
    --------
    >>> # Remove 60 Hz powerline noise
    >>> notch = NotchFilter(freq=60, fs=256, Q=30)
    >>> clean = notch.fit_transform(signal)

    >>> # Remove 50 Hz and first 3 harmonics
    >>> notch = NotchFilter(freq=50, fs=1000, Q=30, harmonics=3)
    >>> clean = notch.fit_transform(signal)
    """

    def __init__(
        self,
        freq: float | list[float],
        fs: float,
        Q: float = 30,
        harmonics: int = 0,
    ):
        super().__init__(fs=fs)
        self.freq = freq
        self.Q = Q
        self.harmonics = harmonics

        self._filters = []

    def fit(self, X, y=None, **fit_params) -> "NotchFilter":
        """Design notch filter(s)."""
        super().fit(X, y, **fit_params)

        fs = self._check_fs()
        nyq = fs / 2

        # Collect all frequencies to notch
        freqs = [self.freq] if isinstance(self.freq, (int, float)) else list(self.freq)

        # Add harmonics
        if self.harmonics > 0:
            base_freqs = freqs.copy()
            for f in base_freqs:
                for h in range(2, self.harmonics + 2):
                    harmonic = f * h
                    if harmonic < nyq:
                        freqs.append(harmonic)

        # Design filters
        self._filters = []
        for f in freqs:
            if f < nyq:
                b, a = scipy_signal.iirnotch(f, self.Q, fs)
                self._filters.append((b, a))

        return self

    def transform(self, X) -> np.ndarray:
        """Apply notch filter(s)."""
        self._check_is_fitted()

        X = self._validate_signal(X, ensure_2d=False)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        if self.copy:
            X_2d = X_2d.copy()

        # Apply each notch filter
        filtered = X_2d.copy()
        for b, a in self._filters:
            for i in range(filtered.shape[0]):
                filtered[i] = scipy_signal.filtfilt(b, a, filtered[i])

        return restore_shape(filtered, was_1d, original_shape)


class MedianFilter(BaseSignalTransformer):
    """Median filter for impulse noise removal.

    Non-linear filter that replaces each sample with the median
    of neighboring samples. Excellent for removing spikes/artifacts.

    Parameters
    ----------
    kernel_size : int, default=3
        Filter window size. Must be odd.

    Examples
    --------
    >>> # Remove spike artifacts
    >>> filt = MedianFilter(kernel_size=5)
    >>> clean = filt.fit_transform(signal_with_spikes)
    """

    def __init__(
        self,
        kernel_size: int = 3,
        fs: float | None = None,
    ):
        super().__init__(fs=fs)
        self.kernel_size = kernel_size

    def fit(self, X, y=None, **fit_params) -> "MedianFilter":
        """Fit the filter."""
        super().fit(X, y, **fit_params)

        if self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")

        return self

    def transform(self, X) -> np.ndarray:
        """Apply median filter."""
        self._check_is_fitted()

        X = self._validate_signal(X, ensure_2d=False)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        # Apply filter
        filtered = scipy_signal.medfilt(X_2d, kernel_size=(1, self.kernel_size))

        return restore_shape(filtered, was_1d, original_shape)


class FilterBank(BaseSignalTransformer):
    """Bank of parallel bandpass filters.

    Decomposes signal into multiple frequency bands simultaneously.
    Useful for spectral analysis and feature extraction.

    Parameters
    ----------
    bands : Dict[str, Tuple[float, float]]
        Dictionary mapping band names to (low, high) frequency tuples.
        Example: {'alpha': (8, 13), 'beta': (13, 30)}
    fs : float
        Sample rate in Hz.
    filter_type : str, default='butterworth'
        Filter type: 'butterworth' or 'fir'.
    order : int, default=4
        Filter order (for Butterworth).
    numtaps : int, default=101
        Number of taps (for FIR).
    output : str, default='dict'
        Output format: 'dict' (separate bands) or 'stack' (3D array).

    Examples
    --------
    >>> bands = {
    ...     'delta': (0.5, 4),
    ...     'theta': (4, 8),
    ...     'alpha': (8, 13),
    ...     'beta': (13, 30),
    ...     'gamma': (30, 100),
    ... }
    >>> fb = FilterBank(bands=bands, fs=256)
    >>> band_signals = fb.fit_transform(eeg_signal)
    >>> alpha_signal = band_signals['alpha']
    """

    def __init__(
        self,
        bands: dict,
        fs: float,
        filter_type: Literal["butterworth", "fir"] = "butterworth",
        order: int = 4,
        numtaps: int = 101,
        output: Literal["dict", "stack"] = "dict",
    ):
        super().__init__(fs=fs)
        self.bands = bands
        self.filter_type = filter_type
        self.order = order
        self.numtaps = numtaps
        self.output = output

        self._filters = {}

    def fit(self, X, y=None, **fit_params) -> "FilterBank":
        """Design all bandpass filters."""
        super().fit(X, y, **fit_params)

        for name, (low, high) in self.bands.items():
            if self.filter_type == "butterworth":
                filt = ButterworthFilter(
                    lowcut=low, highcut=high, fs=self.fs, order=self.order
                )
            else:
                filt = FIRFilter(
                    lowcut=low, highcut=high, fs=self.fs, numtaps=self.numtaps
                )
            filt.fit(X)
            self._filters[name] = filt

        return self

    def transform(self, X) -> dict | np.ndarray:
        """Apply filter bank.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        dict or np.ndarray
            If output='dict': {band_name: filtered_signal}
            If output='stack': (n_bands, ...) array
        """
        self._check_is_fitted()

        results = {}
        for name, filt in self._filters.items():
            results[name] = filt.transform(X)

        if self.output == "dict":
            return results
        else:
            # Stack into array
            return np.stack(list(results.values()), axis=0)

    @property
    def band_names(self) -> list[str]:
        """Get list of band names."""
        return list(self.bands.keys())
