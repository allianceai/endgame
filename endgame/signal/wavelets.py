from __future__ import annotations

"""Wavelet transform module for signal processing.

Provides sklearn-compatible wavelet transforms:
- Continuous Wavelet Transform (CWT)
- Discrete Wavelet Transform (DWT)
- Wavelet Packet Decomposition
- Wavelet-based feature extraction

Requires: PyWavelets (pywt)

References
----------
- PyWavelets: https://pywavelets.readthedocs.io/
- MNE-Python: Time-frequency analysis
"""


import numpy as np

from endgame.signal.base import (
    BaseFeatureExtractor,
    BaseSignalTransformer,
    ensure_2d_signals,
)

# Import pywt (optional dependency)
try:
    import pywt

    HAS_PYWT = True
except ImportError:
    HAS_PYWT = False


def _check_pywt():
    """Check if PyWavelets is available."""
    if not HAS_PYWT:
        raise ImportError(
            "PyWavelets is required for wavelet transforms. "
            "Install with: pip install PyWavelets"
        )


class CWTTransformer(BaseSignalTransformer):
    """Continuous Wavelet Transform for time-frequency analysis.

    Computes CWT using complex Morlet or other wavelets,
    producing a time-frequency representation.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    wavelet : str, default='morl'
        Wavelet to use. Options: 'morl' (Morlet), 'cmor' (complex Morlet),
        'cgau' (complex Gaussian), 'mexh' (Mexican hat), 'gaus'.
    freqs : array-like, optional
        Frequencies to analyze in Hz. If None, uses logarithmically
        spaced frequencies from 1 Hz to Nyquist.
    n_freqs : int, default=32
        Number of frequency bins if freqs is None.
    output : str, default='power'
        Output type: 'complex', 'magnitude', 'power', 'phase'.
    normalize : bool, default=True
        Whether to normalize coefficients.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    scales_ : np.ndarray
        Wavelet scales used.
    freqs_ : np.ndarray
        Corresponding frequencies in Hz.

    Examples
    --------
    >>> cwt = CWTTransformer(fs=256, freqs=np.arange(1, 50))
    >>> tfr = cwt.fit_transform(signal)  # (n_freqs, n_samples)
    """

    def __init__(
        self,
        fs: float,
        wavelet: str = "morl",
        freqs: np.ndarray | None = None,
        n_freqs: int = 32,
        output: str = "power",
        normalize: bool = True,
        copy: bool = True,
    ):
        _check_pywt()
        super().__init__(fs=fs, copy=copy)
        self.wavelet = wavelet
        self.freqs = freqs
        self.n_freqs = n_freqs
        self.output = output
        self.normalize = normalize

        if output not in ("complex", "magnitude", "power", "phase"):
            raise ValueError(
                f"output must be 'complex', 'magnitude', 'power', or 'phase', "
                f"got {output}"
            )

    def fit(self, X, y=None, **fit_params) -> CWTTransformer:
        """Fit the transformer (compute scales).

        Parameters
        ----------
        X : array-like
            Input signal.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        # Determine frequencies
        if self.freqs is not None:
            self.freqs_ = np.asarray(self.freqs)
        else:
            # Logarithmically spaced from 1 Hz to Nyquist/2
            nyquist = self.fs / 2
            self.freqs_ = np.logspace(0, np.log10(nyquist / 2), self.n_freqs)

        # Convert frequencies to scales
        # For Morlet wavelet: scale = (center_frequency * fs) / freq
        center_freq = pywt.central_frequency(self.wavelet)
        self.scales_ = center_freq * self.fs / self.freqs_

        return self

    def transform(self, X) -> np.ndarray:
        """Compute CWT of signal.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            CWT coefficients. Shape depends on input:
            - 1D input: (n_freqs, n_samples)
            - 2D input: (n_channels, n_freqs, n_samples)
            - 3D input: (n_trials, n_channels, n_freqs, n_samples)
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        results = []
        for sig in X_2d:
            coeffs, _ = pywt.cwt(sig, self.scales_, self.wavelet, 1 / self.fs)

            # Normalize
            if self.normalize:
                coeffs = coeffs / np.sqrt(self.scales_[:, np.newaxis])

            # Convert to requested output
            if self.output == "complex":
                result = coeffs
            elif self.output == "magnitude":
                result = np.abs(coeffs)
            elif self.output == "power":
                result = np.abs(coeffs) ** 2
            elif self.output == "phase":
                result = np.angle(coeffs)

            results.append(result)

        results = np.array(results)

        # Reshape output
        if was_1d:
            return results[0]  # (n_freqs, n_samples)
        elif len(original_shape) == 2:
            return results  # (n_channels, n_freqs, n_samples)
        else:
            # Reshape for 3D input
            n_trials, n_channels, _ = original_shape
            n_freqs, n_samples = results.shape[1], results.shape[2]
            return results.reshape(n_trials, n_channels, n_freqs, n_samples)

    def get_frequencies(self) -> np.ndarray:
        """Get the analysis frequencies."""
        self._check_is_fitted()
        return self.freqs_


class DWTTransformer(BaseSignalTransformer):
    """Discrete Wavelet Transform for multi-resolution analysis.

    Decomposes signal into approximation and detail coefficients
    at multiple scales.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    wavelet : str, default='db4'
        Wavelet to use. Options: 'db1'-'db20', 'sym2'-'sym20',
        'coif1'-'coif5', 'bior', 'rbio', etc.
    level : int, optional
        Decomposition level. If None, uses maximum level.
    mode : str, default='symmetric'
        Signal extension mode: 'symmetric', 'periodic', 'reflect', etc.
    output : str, default='coeffs'
        Output type: 'coeffs' (all coefficients), 'detail' (detail only),
        'approx' (approximation only).
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    level_ : int
        Actual decomposition level used.
    coeff_lengths_ : list of int
        Length of each coefficient array.

    Examples
    --------
    >>> dwt = DWTTransformer(wavelet='db4', level=4)
    >>> coeffs = dwt.fit_transform(signal)
    """

    def __init__(
        self,
        fs: float | None = None,
        wavelet: str = "db4",
        level: int | None = None,
        mode: str = "symmetric",
        output: str = "coeffs",
        copy: bool = True,
    ):
        _check_pywt()
        super().__init__(fs=fs, copy=copy)
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.output = output

        if output not in ("coeffs", "detail", "approx"):
            raise ValueError(
                f"output must be 'coeffs', 'detail', or 'approx', got {output}"
            )

    def fit(self, X, y=None, **fit_params) -> DWTTransformer:
        """Fit the transformer.

        Parameters
        ----------
        X : array-like
            Input signal.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        n_samples = self._get_n_samples(X)

        # Determine decomposition level
        if self.level is not None:
            self.level_ = self.level
        else:
            self.level_ = pywt.dwt_max_level(n_samples, self.wavelet)

        # Store coefficient structure by doing a trial decomposition
        trial_coeffs = pywt.wavedec(
            np.zeros(n_samples), self.wavelet, mode=self.mode, level=self.level_
        )
        self.coeff_lengths_ = [len(c) for c in trial_coeffs]

        return self

    def transform(self, X) -> np.ndarray:
        """Compute DWT of signal.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            DWT coefficients concatenated into a single array.
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        results = []
        for sig in X_2d:
            coeffs = pywt.wavedec(sig, self.wavelet, mode=self.mode, level=self.level_)

            if self.output == "coeffs":
                # Concatenate all coefficients
                result = np.concatenate(coeffs)
            elif self.output == "approx":
                # Only approximation coefficients
                result = coeffs[0]
            elif self.output == "detail":
                # Only detail coefficients
                result = np.concatenate(coeffs[1:])

            results.append(result)

        result = np.array(results)

        if was_1d:
            return result.flatten()
        elif len(original_shape) == 3:
            n_trials, n_channels, _ = original_shape
            n_coeffs = result.shape[-1]
            return result.reshape(n_trials, n_channels, n_coeffs)

        return result

    def inverse_transform(self, coeffs) -> np.ndarray:
        """Reconstruct signal from DWT coefficients.

        Parameters
        ----------
        coeffs : array-like
            DWT coefficients (concatenated).

        Returns
        -------
        np.ndarray
            Reconstructed signal.
        """
        self._check_is_fitted()

        if self.output != "coeffs":
            raise ValueError("inverse_transform only works with output='coeffs'")

        coeffs = np.asarray(coeffs)
        coeffs_2d, was_1d, original_shape = ensure_2d_signals(coeffs)

        results = []
        for coeff_vec in coeffs_2d:
            # Split concatenated coefficients
            coeff_list = self._split_coeffs(coeff_vec)
            # Reconstruct
            sig = pywt.waverec(coeff_list, self.wavelet, mode=self.mode)
            results.append(sig)

        result = np.array(results)

        if was_1d:
            return result.flatten()

        return result

    def _split_coeffs(self, coeff_vec: np.ndarray) -> list[np.ndarray]:
        """Split concatenated coefficients back into list."""
        # This is a simplified split - actual lengths depend on signal length
        # For proper reconstruction, we'd need to store coefficient lengths
        coeffs = pywt.wavedec(
            np.zeros(self.n_samples_seen_), self.wavelet, mode=self.mode, level=self.level_
        )

        result = []
        idx = 0
        for c in coeffs:
            length = len(c)
            result.append(coeff_vec[idx : idx + length])
            idx += length

        return result


class WaveletPacketTransformer(BaseSignalTransformer):
    """Wavelet Packet Decomposition for full frequency resolution.

    Unlike DWT which only decomposes approximation coefficients,
    wavelet packets decompose both approximation and detail
    coefficients at each level.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    wavelet : str, default='db4'
        Wavelet to use.
    level : int, optional
        Decomposition level. If None, uses maximum level.
    mode : str, default='symmetric'
        Signal extension mode.
    order : str, default='freq'
        Node ordering: 'freq' (frequency order) or 'natural'.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    level_ : int
        Actual decomposition level used.
    n_nodes_ : int
        Number of terminal nodes (2^level).

    Examples
    --------
    >>> wp = WaveletPacketTransformer(wavelet='db4', level=3)
    >>> coeffs = wp.fit_transform(signal)  # (8, n_coeffs)
    """

    def __init__(
        self,
        fs: float | None = None,
        wavelet: str = "db4",
        level: int | None = None,
        mode: str = "symmetric",
        order: str = "freq",
        copy: bool = True,
    ):
        _check_pywt()
        super().__init__(fs=fs, copy=copy)
        self.wavelet = wavelet
        self.level = level
        self.mode = mode
        self.order = order

    def fit(self, X, y=None, **fit_params) -> WaveletPacketTransformer:
        """Fit the transformer.

        Parameters
        ----------
        X : array-like
            Input signal.
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        super().fit(X, y, **fit_params)

        n_samples = self._get_n_samples(X)

        # Determine decomposition level
        if self.level is not None:
            self.level_ = self.level
        else:
            self.level_ = min(4, pywt.dwt_max_level(n_samples, self.wavelet))

        self.n_nodes_ = 2**self.level_

        return self

    def transform(self, X) -> np.ndarray:
        """Compute wavelet packet decomposition.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            Wavelet packet coefficients.
            Shape: (n_signals, n_nodes, n_coeffs_per_node)
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        results = []
        for sig in X_2d:
            wp = pywt.WaveletPacket(sig, self.wavelet, mode=self.mode)

            # Get terminal nodes at specified level
            nodes = [
                node.path for node in wp.get_level(self.level_, order=self.order)
            ]

            # Extract coefficients from each node
            node_coeffs = []
            for path in nodes:
                node_coeffs.append(wp[path].data)

            # Pad to same length
            max_len = max(len(c) for c in node_coeffs)
            padded = [np.pad(c, (0, max_len - len(c))) for c in node_coeffs]

            results.append(np.array(padded))

        result = np.array(results)

        if was_1d:
            return result[0]  # (n_nodes, n_coeffs)

        return result

    def get_frequency_bands(self) -> list[tuple[float, float]]:
        """Get frequency bands for each wavelet packet node.

        Returns
        -------
        list of (low, high) tuples
            Frequency bands in Hz for each node.
        """
        self._check_is_fitted()

        if self.fs is None:
            raise ValueError("fs must be specified to get frequency bands")

        nyquist = self.fs / 2
        band_width = nyquist / self.n_nodes_

        bands = []
        for i in range(self.n_nodes_):
            if self.order == "freq":
                # Gray code ordering for frequency order
                low = i * band_width
                high = (i + 1) * band_width
            else:
                low = i * band_width
                high = (i + 1) * band_width

            bands.append((low, high))

        return bands


class WaveletFeatureExtractor(BaseFeatureExtractor):
    """Extract features from wavelet coefficients.

    Computes statistical features from wavelet decomposition
    at multiple scales.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    wavelet : str, default='db4'
        Wavelet to use.
    level : int, optional
        Decomposition level.
    features : list of str, optional
        Features to extract. Default: ['energy', 'mean', 'std', 'entropy'].
    use_packet : bool, default=False
        If True, use wavelet packet decomposition.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> extractor = WaveletFeatureExtractor(wavelet='db4', level=4)
    >>> features = extractor.fit_transform(signals)
    """

    DEFAULT_FEATURES = ["energy", "mean", "std", "entropy"]

    def __init__(
        self,
        fs: float | None = None,
        wavelet: str = "db4",
        level: int | None = None,
        features: list[str] | None = None,
        use_packet: bool = False,
    ):
        _check_pywt()
        super().__init__(fs=fs)
        self.wavelet = wavelet
        self.level = level
        self.features = features if features is not None else self.DEFAULT_FEATURES.copy()
        self.use_packet = use_packet

    def fit(self, X, y=None, **fit_params) -> WaveletFeatureExtractor:
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

        n_samples = self._get_n_samples(X)

        # Determine level
        if self.level is not None:
            self.level_ = self.level
        else:
            self.level_ = min(4, pywt.dwt_max_level(n_samples, self.wavelet))

        # Number of coefficient sets
        if self.use_packet:
            self.n_coeff_sets_ = 2**self.level_
            self._band_names = [f"wp_{i}" for i in range(self.n_coeff_sets_)]
        else:
            self.n_coeff_sets_ = self.level_ + 1  # approx + details
            self._band_names = ["approx"] + [f"detail_{i}" for i in range(1, self.level_ + 1)]

        # Build feature names
        self.feature_names_ = []
        for band in self._band_names:
            for feat in self.features:
                self.feature_names_.append(f"wavelet_{band}_{feat}")

        return self

    def transform(self, X) -> np.ndarray:
        """Extract wavelet features.

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

        all_features = []
        for sig in X_2d:
            if self.use_packet:
                coeff_sets = self._get_packet_coeffs(sig)
            else:
                coeff_sets = pywt.wavedec(sig, self.wavelet, level=self.level_)

            sig_features = []
            for coeffs in coeff_sets:
                for feat in self.features:
                    sig_features.append(self._compute_feature(coeffs, feat))

            all_features.append(sig_features)

        return np.array(all_features)

    def _get_packet_coeffs(self, x: np.ndarray) -> list[np.ndarray]:
        """Get wavelet packet coefficients."""
        wp = pywt.WaveletPacket(x, self.wavelet, mode="symmetric")
        nodes = [node.path for node in wp.get_level(self.level_, order="freq")]
        return [wp[path].data for path in nodes]

    def _compute_feature(self, coeffs: np.ndarray, feature: str) -> float:
        """Compute a single feature from coefficients."""
        if feature == "energy":
            return np.sum(coeffs**2)
        elif feature == "mean":
            return np.mean(coeffs)
        elif feature == "std":
            return np.std(coeffs)
        elif feature == "entropy":
            # Shannon entropy of normalized squared coefficients
            squared = coeffs**2
            normalized = squared / (np.sum(squared) + 1e-10)
            return -np.sum(normalized * np.log2(normalized + 1e-10))
        elif feature == "max":
            return np.max(np.abs(coeffs))
        elif feature == "min":
            return np.min(np.abs(coeffs))
        elif feature == "range":
            return np.ptp(coeffs)
        else:
            raise ValueError(f"Unknown feature: {feature}")


def compute_cwt(
    x: np.ndarray,
    fs: float,
    freqs: np.ndarray | None = None,
    wavelet: str = "morl",
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function to compute CWT.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sample rate in Hz.
    freqs : np.ndarray, optional
        Frequencies to analyze.
    wavelet : str, default='morl'
        Wavelet to use.

    Returns
    -------
    freqs : np.ndarray
        Analysis frequencies.
    coeffs : np.ndarray
        CWT coefficients (power).

    Examples
    --------
    >>> freqs, power = compute_cwt(signal, fs=256)
    """
    _check_pywt()
    cwt = CWTTransformer(fs=fs, freqs=freqs, wavelet=wavelet, output="power")
    cwt.fit(x)
    coeffs = cwt.transform(x)
    return cwt.freqs_, coeffs


def compute_dwt(
    x: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> list[np.ndarray]:
    """Convenience function to compute DWT.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    wavelet : str, default='db4'
        Wavelet to use.
    level : int, optional
        Decomposition level.

    Returns
    -------
    list of np.ndarray
        Coefficients [approx, detail1, detail2, ...].

    Examples
    --------
    >>> coeffs = compute_dwt(signal, wavelet='db4', level=4)
    >>> approx, d1, d2, d3, d4 = coeffs
    """
    _check_pywt()
    return pywt.wavedec(x, wavelet, level=level)


def reconstruct_from_dwt(
    coeffs: list[np.ndarray],
    wavelet: str = "db4",
) -> np.ndarray:
    """Reconstruct signal from DWT coefficients.

    Parameters
    ----------
    coeffs : list of np.ndarray
        Coefficients [approx, detail1, detail2, ...].
    wavelet : str, default='db4'
        Wavelet used for decomposition.

    Returns
    -------
    np.ndarray
        Reconstructed signal.
    """
    _check_pywt()
    return pywt.waverec(coeffs, wavelet)
