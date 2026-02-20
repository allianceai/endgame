"""Spectrogram conversion for audio classification."""


import numpy as np
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator


class SpectrogramTransformer(EndgameEstimator, TransformerMixin):
    """Converts audio waveforms to mel-spectrograms.

    Dominant strategy for audio competitions: Treat audio as images.

    Parameters
    ----------
    sample_rate : int, default=32000
        Audio sample rate.
    n_mels : int, default=128
        Number of mel bands.
    n_fft : int, default=2048
        FFT window size.
    hop_length : int, default=512
        Number of samples between frames.
    fmin : int, default=20
        Minimum frequency.
    fmax : int, default=16000
        Maximum frequency.
    power : float, default=2.0
        Exponent for magnitude spectrogram.
    normalize : bool, default=True
        Whether to normalize spectrograms.
    to_db : bool, default=True
        Convert to decibels.

    Examples
    --------
    >>> spec = SpectrogramTransformer(sample_rate=32000, n_mels=128)
    >>> spectrograms = spec.transform(audio_waveforms)
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: int = 20,
        fmax: int | None = None,
        power: float = 2.0,
        normalize: bool = True,
        to_db: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.power = power
        self.normalize = normalize
        self.to_db = to_db

        self._mel_basis = None

    def _check_requirements(self):
        """Check if librosa is installed."""
        try:
            import librosa
        except ImportError:
            raise ImportError(
                "librosa is required for SpectrogramTransformer. "
                "Install with: pip install librosa"
            )

    def fit(self, X, y=None) -> "SpectrogramTransformer":
        """Fit the transformer (precompute mel filterbank).

        Parameters
        ----------
        X : ignored
        y : ignored

        Returns
        -------
        self
        """
        self._check_requirements()
        import librosa

        self._mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert waveforms to mel-spectrograms.

        Parameters
        ----------
        X : ndarray
            Audio waveforms of shape (n_samples, n_time) or (n_time,).

        Returns
        -------
        ndarray
            Mel-spectrograms of shape (n_samples, n_mels, n_frames) or
            (n_mels, n_frames) for single input.
        """
        if not self._is_fitted:
            self.fit(X)

        import librosa

        single_input = X.ndim == 1
        if single_input:
            X = X[np.newaxis, :]

        spectrograms = []

        for waveform in X:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.power,
            )

            # Convert to dB
            if self.to_db:
                mel_spec = librosa.power_to_db(mel_spec, ref=np.max)

            # Normalize
            if self.normalize:
                mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)

            spectrograms.append(mel_spec)

        result = np.array(spectrograms)

        if single_input:
            return result[0]
        return result

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        return self.fit(X, y).transform(X)


class PCENTransformer(SpectrogramTransformer):
    """Per-Channel Energy Normalization for spectrograms.

    More robust than standard normalization for varying recording conditions.
    Common in bird audio competitions.

    Parameters
    ----------
    gain : float, default=0.98
        PCEN gain.
    bias : float, default=2.0
        PCEN bias.
    power : float, default=0.5
        PCEN power.
    time_constant : float, default=0.4
        Time constant for smoothing.
    eps : float, default=1e-6
        Small constant for numerical stability.

    Examples
    --------
    >>> pcen = PCENTransformer(sample_rate=32000)
    >>> spectrograms = pcen.transform(audio_waveforms)
    """

    def __init__(
        self,
        sample_rate: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: int = 20,
        fmax: int | None = None,
        gain: float = 0.98,
        bias: float = 2.0,
        pcen_power: float = 0.5,
        time_constant: float = 0.4,
        eps: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            sample_rate=sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            fmin=fmin,
            fmax=fmax,
            power=1.0,  # PCEN needs power spectrum, not mel
            normalize=False,
            to_db=False,
            random_state=random_state,
            verbose=verbose,
        )
        self.gain = gain
        self.bias = bias
        self.pcen_power = pcen_power
        self.time_constant = time_constant
        self.eps = eps

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Convert waveforms to PCEN spectrograms."""
        if not self._is_fitted:
            self.fit(X)

        import librosa

        single_input = X.ndim == 1
        if single_input:
            X = X[np.newaxis, :]

        spectrograms = []

        for waveform in X:
            # Compute mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=waveform,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=1.0,  # Magnitude, not power
            )

            # Apply PCEN
            pcen_spec = librosa.pcen(
                mel_spec * (2 ** 31),  # Scale for numerical stability
                sr=self.sample_rate,
                hop_length=self.hop_length,
                gain=self.gain,
                bias=self.bias,
                power=self.pcen_power,
                time_constant=self.time_constant,
                eps=self.eps,
            )

            spectrograms.append(pcen_spec)

        result = np.array(spectrograms)

        if single_input:
            return result[0]
        return result
