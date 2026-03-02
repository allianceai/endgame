from __future__ import annotations

"""Spectral analysis transformers for signal processing.

Provides sklearn-compatible spectral analysis methods:
- FFT for frequency spectrum
- Welch's method for PSD estimation
- Multitaper PSD for high-resolution spectral analysis
- Band power extraction for frequency band features

References
----------
- scipy.signal: Welch PSD, periodogram
- MNE-Python: Multitaper implementation
- neurokit2: Band power calculations
"""

from typing import Any

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq

from endgame.signal.base import (
    BaseFeatureExtractor,
    BaseSignalTransformer,
    ensure_2d_signals,
)


class FFTTransformer(BaseSignalTransformer):
    """Fast Fourier Transform for frequency spectrum analysis.

    Computes the FFT of input signals, returning either the full
    complex spectrum, magnitude, power, or phase.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    output : str, default='magnitude'
        Output type: 'complex', 'magnitude', 'power', 'phase', 'db'.
    one_sided : bool, default=True
        If True, return only positive frequencies (real FFT).
    n_fft : int, optional
        FFT length. If None, uses signal length.
    normalize : bool, default=True
        If True, normalize FFT by signal length.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    freqs_ : np.ndarray
        Frequency bins after fit.

    Examples
    --------
    >>> fft_trans = FFTTransformer(fs=256, output='magnitude')
    >>> spectrum = fft_trans.fit_transform(signal)
    >>> freqs = fft_trans.freqs_
    """

    def __init__(
        self,
        fs: float,
        output: str = "magnitude",
        one_sided: bool = True,
        n_fft: int | None = None,
        normalize: bool = True,
        copy: bool = True,
    ):
        super().__init__(fs=fs, copy=copy)
        self.output = output
        self.one_sided = one_sided
        self.n_fft = n_fft
        self.normalize = normalize

        if output not in ("complex", "magnitude", "power", "phase", "db"):
            raise ValueError(
                f"output must be 'complex', 'magnitude', 'power', 'phase', or 'db', "
                f"got {output}"
            )

    def fit(self, X, y=None, **fit_params) -> FFTTransformer:
        """Fit the transformer (compute frequency bins).

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
        n_fft = self.n_fft if self.n_fft is not None else n_samples

        if self.one_sided:
            self.freqs_ = rfftfreq(n_fft, 1 / self.fs)
        else:
            self.freqs_ = fftfreq(n_fft, 1 / self.fs)

        return self

    def transform(self, X) -> np.ndarray:
        """Compute FFT of signal.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            FFT result based on output parameter.
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)
        n_signals, n_samples = X_2d.shape
        n_fft = self.n_fft if self.n_fft is not None else n_samples

        # Compute FFT
        if self.one_sided:
            fft_result = rfft(X_2d, n=n_fft, axis=-1)
        else:
            fft_result = fft(X_2d, n=n_fft, axis=-1)

        # Normalize
        if self.normalize:
            fft_result = fft_result / n_samples

        # Convert to requested output
        if self.output == "complex":
            result = fft_result
        elif self.output == "magnitude":
            result = np.abs(fft_result)
        elif self.output == "power":
            result = np.abs(fft_result) ** 2
        elif self.output == "phase":
            result = np.angle(fft_result)
        elif self.output == "db":
            result = 20 * np.log10(np.abs(fft_result) + 1e-10)

        # Restore shape for 3D input
        if len(original_shape) == 3:
            n_trials, n_channels, _ = original_shape
            n_freqs = result.shape[-1]
            result = result.reshape(n_trials, n_channels, n_freqs)
        elif was_1d:
            result = result.flatten()

        return result

    def get_frequency_resolution(self) -> float:
        """Get frequency resolution in Hz."""
        self._check_is_fitted()
        return self.freqs_[1] - self.freqs_[0]


class WelchPSD(BaseSignalTransformer):
    """Welch's method for Power Spectral Density estimation.

    Estimates PSD by averaging modified periodograms from
    overlapping segments.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    nperseg : int, optional
        Length of each segment. Default is 256 or signal length.
    noverlap : int, optional
        Number of points to overlap. Default is nperseg // 2.
    nfft : int, optional
        FFT length. Default is nperseg.
    window : str or tuple, default='hann'
        Window function to use.
    detrend : str or False, default='constant'
        Detrending method: 'constant', 'linear', or False.
    scaling : str, default='density'
        'density' for V^2/Hz, 'spectrum' for V^2.
    average : str, default='mean'
        Averaging method: 'mean' or 'median'.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    freqs_ : np.ndarray
        Frequency bins after fit.

    Examples
    --------
    >>> psd = WelchPSD(fs=256, nperseg=256)
    >>> power = psd.fit_transform(signal)
    >>> freqs = psd.freqs_
    """

    def __init__(
        self,
        fs: float,
        nperseg: int | None = None,
        noverlap: int | None = None,
        nfft: int | None = None,
        window: str = "hann",
        detrend: str | bool = "constant",
        scaling: str = "density",
        average: str = "mean",
        copy: bool = True,
    ):
        super().__init__(fs=fs, copy=copy)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.window = window
        self.detrend = detrend
        self.scaling = scaling
        self.average = average

    def fit(self, X, y=None, **fit_params) -> WelchPSD:
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
        nperseg = self.nperseg if self.nperseg is not None else min(256, n_samples)

        # Compute frequencies
        self.freqs_, _ = signal.welch(
            np.zeros(n_samples),
            fs=self.fs,
            nperseg=nperseg,
            noverlap=self.noverlap,
            nfft=self.nfft,
            window=self.window,
        )

        return self

    def transform(self, X) -> np.ndarray:
        """Compute PSD using Welch's method.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            Power spectral density.
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)
        n_samples = X_2d.shape[1]
        nperseg = self.nperseg if self.nperseg is not None else min(256, n_samples)

        results = []
        for sig in X_2d:
            freqs, psd = signal.welch(
                sig,
                fs=self.fs,
                nperseg=nperseg,
                noverlap=self.noverlap,
                nfft=self.nfft,
                window=self.window,
                detrend=self.detrend,
                scaling=self.scaling,
                average=self.average,
            )
            results.append(psd)

        result = np.array(results)

        # Restore shape for 3D input
        if len(original_shape) == 3:
            n_trials, n_channels, _ = original_shape
            n_freqs = result.shape[-1]
            result = result.reshape(n_trials, n_channels, n_freqs)
        elif was_1d:
            result = result.flatten()

        return result

    def get_frequency_bands(
        self,
        X: np.ndarray,
        bands: dict[str, tuple[float, float]],
    ) -> dict[str, np.ndarray]:
        """Get PSD values within specified frequency bands.

        Parameters
        ----------
        X : np.ndarray
            Input signal.
        bands : dict
            Dictionary mapping band names to (low, high) frequency tuples.

        Returns
        -------
        dict
            Dictionary mapping band names to PSD arrays within each band.
        """
        psd = self.transform(X)
        freqs = self.freqs_

        band_psds = {}
        for name, (low, high) in bands.items():
            mask = (freqs >= low) & (freqs <= high)
            if psd.ndim == 1:
                band_psds[name] = psd[mask]
            else:
                band_psds[name] = psd[..., mask]

        return band_psds


class MultitaperPSD(BaseSignalTransformer):
    """Multitaper Power Spectral Density estimation.

    Uses multiple orthogonal tapers (DPSS/Slepian sequences) to reduce
    variance in PSD estimation while maintaining frequency resolution.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    bandwidth : float, optional
        Frequency bandwidth of the tapers in Hz.
        Default is 4 * fs / n_samples (NW=4).
    n_tapers : int, optional
        Number of tapers to use. Default is 2 * bandwidth - 1.
    low_bias : bool, default=True
        Only use tapers with concentration ratio > 0.9.
    adaptive : bool, default=False
        Use adaptive weighting for combining tapers.
    normalization : str, default='full'
        PSD normalization: 'full', 'length'.
    copy : bool, default=True
        Whether to copy input data.

    Attributes
    ----------
    freqs_ : np.ndarray
        Frequency bins after fit.
    dpss_ : np.ndarray
        DPSS tapers used.
    eigenvalues_ : np.ndarray
        Eigenvalues of the tapers.

    References
    ----------
    Thomson, D.J. (1982). Spectrum estimation and harmonic analysis.
    Proceedings of the IEEE, 70(9), 1055-1096.

    Examples
    --------
    >>> mt_psd = MultitaperPSD(fs=256, bandwidth=4)
    >>> power = mt_psd.fit_transform(signal)
    """

    def __init__(
        self,
        fs: float,
        bandwidth: float | None = None,
        n_tapers: int | None = None,
        low_bias: bool = True,
        adaptive: bool = False,
        normalization: str = "full",
        copy: bool = True,
    ):
        super().__init__(fs=fs, copy=copy)
        self.bandwidth = bandwidth
        self.n_tapers = n_tapers
        self.low_bias = low_bias
        self.adaptive = adaptive
        self.normalization = normalization

    def fit(self, X, y=None, **fit_params) -> MultitaperPSD:
        """Fit the transformer (compute DPSS tapers).

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

        # Compute time-bandwidth product
        if self.bandwidth is not None:
            NW = self.bandwidth * n_samples / (2 * self.fs)
        else:
            NW = 4  # Default

        # Number of tapers
        if self.n_tapers is not None:
            K = self.n_tapers
        else:
            K = int(2 * NW - 1)

        K = max(1, K)

        # Compute DPSS tapers
        self.dpss_, self.eigenvalues_ = signal.windows.dpss(
            n_samples, NW, K, return_ratios=True
        )

        # Filter low-bias tapers
        if self.low_bias:
            keep = self.eigenvalues_ > 0.9
            if keep.sum() > 0:
                self.dpss_ = self.dpss_[keep]
                self.eigenvalues_ = self.eigenvalues_[keep]

        # Compute frequency bins
        self.freqs_ = rfftfreq(n_samples, 1 / self.fs)
        self._n_samples = n_samples

        return self

    def transform(self, X) -> np.ndarray:
        """Compute multitaper PSD.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            Power spectral density.
        """
        self._check_is_fitted()
        X = self._validate_signal(X)

        if self.copy:
            X = X.copy()

        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        results = []
        for sig in X_2d:
            psd = self._compute_multitaper_psd(sig)
            results.append(psd)

        result = np.array(results)

        # Restore shape for 3D input
        if len(original_shape) == 3:
            n_trials, n_channels, _ = original_shape
            n_freqs = result.shape[-1]
            result = result.reshape(n_trials, n_channels, n_freqs)
        elif was_1d:
            result = result.flatten()

        return result

    def _compute_multitaper_psd(self, x: np.ndarray) -> np.ndarray:
        """Compute multitaper PSD for a single signal."""
        n_tapers = len(self.dpss_)

        # Apply each taper and compute FFT
        tapered_spectra = []
        for taper in self.dpss_:
            tapered = x * taper
            spectrum = rfft(tapered)
            tapered_spectra.append(np.abs(spectrum) ** 2)

        tapered_spectra = np.array(tapered_spectra)

        if self.adaptive and n_tapers > 1:
            # Adaptive weighting
            psd = self._adaptive_weights(tapered_spectra)
        else:
            # Simple average
            weights = self.eigenvalues_ / self.eigenvalues_.sum()
            psd = np.average(tapered_spectra, axis=0, weights=weights)

        # Normalization
        if self.normalization == "full":
            psd = psd / self.fs
        elif self.normalization == "length":
            psd = psd / len(x)

        return psd

    def _adaptive_weights(self, tapered_spectra: np.ndarray) -> np.ndarray:
        """Compute adaptive weights for multitaper combination.

        Uses iterative method from Thomson (1982).
        """
        n_tapers, n_freqs = tapered_spectra.shape

        # Initial estimate (simple average)
        psd = np.mean(tapered_spectra, axis=0)

        # Iterate to find optimal weights
        for _ in range(5):
            weights = np.zeros((n_tapers, n_freqs))
            for k in range(n_tapers):
                weights[k] = (
                    self.eigenvalues_[k]
                    * psd
                    / (
                        self.eigenvalues_[k] * psd
                        + (1 - self.eigenvalues_[k]) * np.var(psd)
                    )
                )

            # Normalize weights
            weights = weights / weights.sum(axis=0, keepdims=True)

            # Update PSD estimate
            psd = np.sum(weights * tapered_spectra, axis=0)

        return psd


class BandPowerExtractor(BaseFeatureExtractor):
    """Extract power in specified frequency bands.

    Computes absolute and relative band powers from signals,
    commonly used for EEG analysis.

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    bands : dict
        Dictionary mapping band names to (low, high) frequency tuples.
        Default includes standard EEG bands.
    method : str, default='welch'
        PSD estimation method: 'welch', 'multitaper', 'fft'.
    relative : bool, default=True
        If True, also compute relative band powers.
    log_power : bool, default=False
        If True, return log10 of band powers.
    welch_params : dict, optional
        Additional parameters for Welch's method.
    multitaper_params : dict, optional
        Additional parameters for multitaper method.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> bands = {'alpha': (8, 13), 'beta': (13, 30)}
    >>> bp = BandPowerExtractor(fs=256, bands=bands)
    >>> features = bp.fit_transform(eeg_signals)
    """

    # Default EEG bands
    DEFAULT_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    def __init__(
        self,
        fs: float,
        bands: dict[str, tuple[float, float]] | None = None,
        method: str = "welch",
        relative: bool = True,
        log_power: bool = False,
        welch_params: dict[str, Any] | None = None,
        multitaper_params: dict[str, Any] | None = None,
    ):
        super().__init__(fs=fs)
        self.bands = bands if bands is not None else self.DEFAULT_BANDS.copy()
        self.method = method
        self.relative = relative
        self.log_power = log_power
        self.welch_params = welch_params or {}
        self.multitaper_params = multitaper_params or {}

        if method not in ("welch", "multitaper", "fft"):
            raise ValueError(f"method must be 'welch', 'multitaper', or 'fft', got {method}")

    def fit(self, X, y=None, **fit_params) -> BandPowerExtractor:
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

        # Create PSD estimator
        if self.method == "welch":
            self._psd = WelchPSD(fs=self.fs, **self.welch_params)
        elif self.method == "multitaper":
            self._psd = MultitaperPSD(fs=self.fs, **self.multitaper_params)
        else:
            self._psd = FFTTransformer(fs=self.fs, output="power")

        self._psd.fit(X)

        # Build feature names
        self.feature_names_ = []
        for band_name in self.bands:
            self.feature_names_.append(f"bp_{band_name}_abs")
        if self.relative:
            for band_name in self.bands:
                self.feature_names_.append(f"bp_{band_name}_rel")

        return self

    def transform(self, X) -> np.ndarray:
        """Extract band power features.

        Parameters
        ----------
        X : array-like
            Input signals of shape (n_samples, n_timepoints) or
            (n_samples, n_channels, n_timepoints).

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted band power features.
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        # Get PSD
        psd = self._psd.transform(X_2d)
        freqs = self._psd.freqs_

        # Frequency resolution for integration
        freq_res = freqs[1] - freqs[0]

        features = []
        for sig_psd in psd:
            sig_features = []

            # Total power for relative calculation
            total_power = np.trapz(sig_psd, dx=freq_res)

            # Absolute band powers
            abs_powers = {}
            for band_name, (low, high) in self.bands.items():
                mask = (freqs >= low) & (freqs <= high)
                band_power = np.trapz(sig_psd[mask], dx=freq_res)
                abs_powers[band_name] = band_power

                if self.log_power:
                    sig_features.append(np.log10(band_power + 1e-10))
                else:
                    sig_features.append(band_power)

            # Relative band powers
            if self.relative:
                for band_name in self.bands:
                    rel_power = abs_powers[band_name] / (total_power + 1e-10)
                    sig_features.append(rel_power)

            features.append(sig_features)

        return np.array(features)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_


class SpectralFeatureExtractor(BaseFeatureExtractor):
    """Extract comprehensive spectral features from signals.

    Computes a variety of frequency-domain features including:
    - Spectral centroid, spread, skewness, kurtosis
    - Spectral entropy
    - Spectral edge frequencies
    - Spectral flatness and rolloff
    - Peak frequency

    Parameters
    ----------
    fs : float
        Sample rate in Hz.
    method : str, default='welch'
        PSD estimation method: 'welch', 'multitaper', 'fft'.
    edge_percentiles : list of float, default=[0.5, 0.75, 0.9, 0.95]
        Percentiles for spectral edge frequency computation.
    welch_params : dict, optional
        Additional parameters for Welch's method.

    Attributes
    ----------
    feature_names_ : list of str
        Names of extracted features.

    Examples
    --------
    >>> extractor = SpectralFeatureExtractor(fs=256)
    >>> features = extractor.fit_transform(signals)
    """

    def __init__(
        self,
        fs: float,
        method: str = "welch",
        edge_percentiles: list[float] | None = None,
        welch_params: dict[str, Any] | None = None,
    ):
        super().__init__(fs=fs)
        self.method = method
        self.edge_percentiles = edge_percentiles or [0.5, 0.75, 0.9, 0.95]
        self.welch_params = welch_params or {}

    def fit(self, X, y=None, **fit_params) -> SpectralFeatureExtractor:
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

        # Create PSD estimator
        if self.method == "welch":
            self._psd = WelchPSD(fs=self.fs, **self.welch_params)
        elif self.method == "multitaper":
            self._psd = MultitaperPSD(fs=self.fs)
        else:
            self._psd = FFTTransformer(fs=self.fs, output="power")

        self._psd.fit(X)

        # Build feature names
        self.feature_names_ = [
            "spectral_centroid",
            "spectral_spread",
            "spectral_skewness",
            "spectral_kurtosis",
            "spectral_entropy",
            "spectral_flatness",
            "spectral_rolloff",
            "peak_frequency",
            "mean_frequency",
            "median_frequency",
        ]
        for pct in self.edge_percentiles:
            self.feature_names_.append(f"spectral_edge_{int(pct * 100)}")

        return self

    def transform(self, X) -> np.ndarray:
        """Extract spectral features.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted spectral features.
        """
        X = self._validate_signal(X)
        X_2d, was_1d, original_shape = ensure_2d_signals(X)

        # Get PSD
        psd = self._psd.transform(X_2d)
        freqs = self._psd.freqs_

        features = []
        for sig_psd in psd:
            sig_features = self._compute_spectral_features(sig_psd, freqs)
            features.append(sig_features)

        return np.array(features)

    def _compute_spectral_features(
        self, psd: np.ndarray, freqs: np.ndarray
    ) -> list[float]:
        """Compute spectral features for a single PSD."""
        # Normalize PSD to probability distribution
        psd_norm = psd / (psd.sum() + 1e-10)

        # Spectral centroid (mean frequency weighted by power)
        centroid = np.sum(freqs * psd_norm)

        # Spectral spread (std of frequency distribution)
        spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd_norm))

        # Spectral skewness
        skewness = np.sum(((freqs - centroid) ** 3) * psd_norm) / (spread**3 + 1e-10)

        # Spectral kurtosis
        kurtosis = np.sum(((freqs - centroid) ** 4) * psd_norm) / (spread**4 + 1e-10)

        # Spectral entropy
        psd_prob = psd_norm + 1e-10
        entropy = -np.sum(psd_prob * np.log2(psd_prob))

        # Spectral flatness (geometric mean / arithmetic mean)
        geom_mean = np.exp(np.mean(np.log(psd + 1e-10)))
        arith_mean = np.mean(psd)
        flatness = geom_mean / (arith_mean + 1e-10)

        # Spectral rolloff (frequency below which 85% of power lies)
        cumsum = np.cumsum(psd)
        rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
        rolloff = freqs[min(rolloff_idx, len(freqs) - 1)]

        # Peak frequency
        peak_freq = freqs[np.argmax(psd)]

        # Mean frequency
        mean_freq = centroid

        # Median frequency
        median_idx = np.searchsorted(cumsum, 0.5 * cumsum[-1])
        median_freq = freqs[min(median_idx, len(freqs) - 1)]

        features = [
            centroid,
            spread,
            skewness,
            kurtosis,
            entropy,
            flatness,
            rolloff,
            peak_freq,
            mean_freq,
            median_freq,
        ]

        # Spectral edge frequencies
        for pct in self.edge_percentiles:
            edge_idx = np.searchsorted(cumsum, pct * cumsum[-1])
            edge_freq = freqs[min(edge_idx, len(freqs) - 1)]
            features.append(edge_freq)

        return features

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        return self.feature_names_


def compute_psd(
    x: np.ndarray,
    fs: float,
    method: str = "welch",
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Convenience function to compute PSD.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sample rate in Hz.
    method : str, default='welch'
        PSD estimation method: 'welch', 'multitaper', 'fft'.
    **kwargs
        Additional parameters for the PSD method.

    Returns
    -------
    freqs : np.ndarray
        Frequency bins.
    psd : np.ndarray
        Power spectral density.

    Examples
    --------
    >>> freqs, psd = compute_psd(signal, fs=256, method='welch')
    """
    if method == "welch":
        estimator = WelchPSD(fs=fs, **kwargs)
    elif method == "multitaper":
        estimator = MultitaperPSD(fs=fs, **kwargs)
    elif method == "fft":
        estimator = FFTTransformer(fs=fs, output="power", **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")

    estimator.fit(x)
    psd = estimator.transform(x)

    return estimator.freqs_, psd


def compute_band_power(
    x: np.ndarray,
    fs: float,
    band: tuple[float, float],
    method: str = "welch",
    relative: bool = False,
) -> float:
    """Compute power in a frequency band.

    Parameters
    ----------
    x : np.ndarray
        Input signal.
    fs : float
        Sample rate in Hz.
    band : tuple of (low, high)
        Frequency band in Hz.
    method : str, default='welch'
        PSD estimation method.
    relative : bool, default=False
        If True, return relative band power.

    Returns
    -------
    float
        Band power.

    Examples
    --------
    >>> alpha_power = compute_band_power(eeg, fs=256, band=(8, 13))
    """
    freqs, psd = compute_psd(x, fs, method=method)

    # Flatten if needed
    if psd.ndim > 1:
        psd = psd.flatten()

    freq_res = freqs[1] - freqs[0]
    mask = (freqs >= band[0]) & (freqs <= band[1])
    band_power = np.trapz(psd[mask], dx=freq_res)

    if relative:
        total_power = np.trapz(psd, dx=freq_res)
        return band_power / (total_power + 1e-10)

    return band_power
