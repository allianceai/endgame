from __future__ import annotations

"""Audio feature extraction for classification tasks.

Extracts handcrafted audio features (MFCCs, spectral, temporal) from raw
waveforms. These features serve as strong baselines and complement deep
learning approaches.
"""


import numpy as np
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator


class AudioFeatureExtractor(EndgameEstimator, TransformerMixin):
    """Extract audio features from waveforms for tabular classifiers.

    Computes a comprehensive set of handcrafted audio features including
    MFCCs, spectral descriptors, temporal statistics, and rhythm features.
    Output is a 2D array suitable for any sklearn classifier.

    Parameters
    ----------
    sample_rate : int, default=22050
        Audio sample rate in Hz.
    n_mfcc : int, default=20
        Number of MFCC coefficients to extract.
    n_mels : int, default=128
        Number of mel bands for mel-spectrogram computation.
    n_fft : int, default=2048
        FFT window size.
    hop_length : int, default=512
        Number of samples between frames.
    features : list of str, optional
        Which feature groups to extract. Default is all available:
        - 'mfcc': Mel-frequency cepstral coefficients (mean, std, delta, delta2)
        - 'spectral': Spectral centroid, bandwidth, rolloff, contrast, flatness
        - 'temporal': RMS energy, zero-crossing rate, tempo
        - 'chroma': Chroma features (pitch class profiles)
        - 'tonnetz': Tonal centroid features
    include_deltas : bool, default=True
        Include delta and delta-delta MFCCs.

    Examples
    --------
    >>> extractor = AudioFeatureExtractor(sample_rate=22050, n_mfcc=13)
    >>> features = extractor.transform(waveforms)
    >>> features.shape  # (n_samples, n_features)
    """

    FEATURE_GROUPS = ["mfcc", "spectral", "temporal", "chroma", "tonnetz", "mel_spectrogram"]

    def __init__(
        self,
        sample_rate: int = 22050,
        n_mfcc: int = 20,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        features: list[str] | None = None,
        include_deltas: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.features = features or self.FEATURE_GROUPS
        self.include_deltas = include_deltas

        self.feature_names_: list[str] | None = None

    def _check_requirements(self):
        try:
            import librosa  # noqa: F401
        except ImportError:
            raise ImportError(
                "librosa is required for AudioFeatureExtractor. "
                "Install with: pip install librosa"
            )

    def fit(self, X, y=None) -> AudioFeatureExtractor:
        """Fit the extractor (computes feature names).

        Parameters
        ----------
        X : array-like
            Audio waveforms. Can be a list of 1D arrays (variable length)
            or a 2D array of shape (n_samples, n_time).
        y : ignored

        Returns
        -------
        self
        """
        self._check_requirements()
        # Extract one sample to determine feature names
        if isinstance(X, np.ndarray) and X.ndim == 2:
            sample = X[0]
        elif isinstance(X, (list, tuple)):
            sample = np.asarray(X[0], dtype=np.float32)
        else:
            sample = np.asarray(X, dtype=np.float32)

        feats, names = self._extract_single(sample, return_names=True)
        self.feature_names_ = names
        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Extract features from audio waveforms.

        Parameters
        ----------
        X : array-like
            Audio waveforms. Can be a list of 1D arrays (variable length)
            or a 2D array of shape (n_samples, n_time).

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Extracted feature matrix.
        """
        if not self._is_fitted:
            self.fit(X)

        if isinstance(X, np.ndarray) and X.ndim == 1:
            X = [X]
        elif isinstance(X, np.ndarray) and X.ndim == 2:
            X = list(X)

        all_features = []
        for waveform in X:
            waveform = np.asarray(waveform, dtype=np.float32)
            feats, _ = self._extract_single(waveform, return_names=False)
            all_features.append(feats)

        return np.vstack(all_features)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get feature names for output columns."""
        if self.feature_names_ is None:
            raise RuntimeError("Call fit() first to compute feature names.")
        return self.feature_names_

    def _extract_single(
        self,
        waveform: np.ndarray,
        return_names: bool = True,
    ) -> tuple[np.ndarray, list[str] | None]:
        """Extract features from a single waveform."""
        import librosa

        features = []
        names = [] if return_names else None

        sr = self.sample_rate

        # --- MFCC features ---
        if "mfcc" in self.features:
            mfcc = librosa.feature.mfcc(
                y=waveform, sr=sr, n_mfcc=self.n_mfcc,
                n_fft=self.n_fft, hop_length=self.hop_length,
                n_mels=self.n_mels,
            )
            # Statistics over time
            mfcc_mean = mfcc.mean(axis=1)
            mfcc_std = mfcc.std(axis=1)
            features.extend([mfcc_mean, mfcc_std])
            if return_names:
                names.extend([f"mfcc_{i}_mean" for i in range(self.n_mfcc)])
                names.extend([f"mfcc_{i}_std" for i in range(self.n_mfcc)])

            if self.include_deltas:
                delta = librosa.feature.delta(mfcc)
                delta2 = librosa.feature.delta(mfcc, order=2)
                delta_mean = delta.mean(axis=1)
                delta_std = delta.std(axis=1)
                delta2_mean = delta2.mean(axis=1)
                delta2_std = delta2.std(axis=1)
                features.extend([delta_mean, delta_std, delta2_mean, delta2_std])
                if return_names:
                    names.extend([f"mfcc_delta_{i}_mean" for i in range(self.n_mfcc)])
                    names.extend([f"mfcc_delta_{i}_std" for i in range(self.n_mfcc)])
                    names.extend([f"mfcc_delta2_{i}_mean" for i in range(self.n_mfcc)])
                    names.extend([f"mfcc_delta2_{i}_std" for i in range(self.n_mfcc)])

        # --- Spectral features ---
        if "spectral" in self.features:
            spec_centroid = librosa.feature.spectral_centroid(
                y=waveform, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            spec_bandwidth = librosa.feature.spectral_bandwidth(
                y=waveform, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            spec_rolloff = librosa.feature.spectral_rolloff(
                y=waveform, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            spec_flatness = librosa.feature.spectral_flatness(
                y=waveform, n_fft=self.n_fft, hop_length=self.hop_length
            )[0]
            spec_contrast = librosa.feature.spectral_contrast(
                y=waveform, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )

            for feat_arr, feat_name in [
                (spec_centroid, "spectral_centroid"),
                (spec_bandwidth, "spectral_bandwidth"),
                (spec_rolloff, "spectral_rolloff"),
                (spec_flatness, "spectral_flatness"),
            ]:
                features.append(np.array([feat_arr.mean(), feat_arr.std()]))
                if return_names:
                    names.extend([f"{feat_name}_mean", f"{feat_name}_std"])

            # Spectral contrast has 7 bands
            contrast_mean = spec_contrast.mean(axis=1)
            contrast_std = spec_contrast.std(axis=1)
            features.extend([contrast_mean, contrast_std])
            if return_names:
                names.extend([f"spectral_contrast_{i}_mean" for i in range(7)])
                names.extend([f"spectral_contrast_{i}_std" for i in range(7)])

        # --- Temporal features ---
        if "temporal" in self.features:
            rms = librosa.feature.rms(
                y=waveform, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]
            zcr = librosa.feature.zero_crossing_rate(
                waveform, frame_length=self.n_fft, hop_length=self.hop_length
            )[0]

            features.append(np.array([
                rms.mean(), rms.std(), rms.max(),
                zcr.mean(), zcr.std(),
            ]))
            if return_names:
                names.extend([
                    "rms_mean", "rms_std", "rms_max",
                    "zcr_mean", "zcr_std",
                ])

            # Tempo
            onset_env = librosa.onset.onset_strength(
                y=waveform, sr=sr, hop_length=self.hop_length
            )
            tempo = librosa.feature.tempo(
                onset_envelope=onset_env, sr=sr, hop_length=self.hop_length
            )
            features.append(np.array([tempo[0]]))
            if return_names:
                names.append("tempo")

        # --- Chroma features ---
        if "chroma" in self.features:
            chroma = librosa.feature.chroma_stft(
                y=waveform, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length
            )
            chroma_mean = chroma.mean(axis=1)
            chroma_std = chroma.std(axis=1)
            features.extend([chroma_mean, chroma_std])
            if return_names:
                names.extend([f"chroma_{i}_mean" for i in range(12)])
                names.extend([f"chroma_{i}_std" for i in range(12)])

        # --- Tonnetz features ---
        if "tonnetz" in self.features:
            harmonic = librosa.effects.harmonic(waveform)
            tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sr)
            tonnetz_mean = tonnetz.mean(axis=1)
            tonnetz_std = tonnetz.std(axis=1)
            features.extend([tonnetz_mean, tonnetz_std])
            if return_names:
                names.extend([f"tonnetz_{i}_mean" for i in range(6)])
                names.extend([f"tonnetz_{i}_std" for i in range(6)])

        # --- Mel-spectrogram band features ---
        if "mel_spectrogram" in self.features:
            mel_spec = librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_mels=self.n_mels,
                n_fft=self.n_fft, hop_length=self.hop_length,
            )
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_mean = mel_spec_db.mean(axis=1)
            mel_std = mel_spec_db.std(axis=1)
            features.extend([mel_mean, mel_std])
            if return_names:
                names.extend([f"mel_{i}_mean" for i in range(self.n_mels)])
                names.extend([f"mel_{i}_std" for i in range(self.n_mels)])

        feature_vector = np.concatenate(features).reshape(1, -1)
        return feature_vector, names
