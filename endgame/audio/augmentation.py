"""Audio-specific augmentations."""


import numpy as np

from endgame.core.base import EndgameEstimator


class AudioAugmentation(EndgameEstimator):
    """Audio-specific augmentations for training.

    Parameters
    ----------
    augmentations : List[str]
        Augmentations to apply:
        - 'noise': Add Gaussian noise
        - 'timeshift': Random time shift
        - 'pitchshift': Random pitch shift
        - 'timestretch': Random time stretch
        - 'specaugment': SpecAugment (frequency/time masking)
        - 'mixup': Mix two audio samples
        - 'gain': Random volume change
    noise_level : float, default=0.005
        Standard deviation of Gaussian noise.
    shift_range : float, default=0.1
        Maximum shift as fraction of audio length.
    pitch_range : int, default=2
        Maximum pitch shift in semitones.
    stretch_range : Tuple[float, float], default=(0.8, 1.2)
        Time stretch range.
    p : float, default=0.5
        Probability of applying each augmentation.
    sample_rate : int, default=32000
        Audio sample rate.

    Examples
    --------
    >>> aug = AudioAugmentation(
    ...     augmentations=['noise', 'timeshift', 'specaugment']
    ... )
    >>> augmented = aug.apply(audio)
    """

    def __init__(
        self,
        augmentations: list[str] | None = None,
        noise_level: float = 0.005,
        shift_range: float = 0.1,
        pitch_range: int = 2,
        stretch_range: tuple[float, float] = (0.8, 1.2),
        p: float = 0.5,
        sample_rate: int = 32000,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.augmentations = augmentations or ["noise", "timeshift"]
        self.noise_level = noise_level
        self.shift_range = shift_range
        self.pitch_range = pitch_range
        self.stretch_range = stretch_range
        self.p = p
        self.sample_rate = sample_rate

        self._rng = np.random.RandomState(random_state)

    def apply(
        self,
        audio: np.ndarray,
        target: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply augmentations to audio.

        Parameters
        ----------
        audio : ndarray
            Audio waveform.
        target : ndarray, optional
            Target labels (for mixup).

        Returns
        -------
        audio : ndarray
            Augmented audio.
        target : ndarray, optional
            Modified target (if mixup applied).
        """
        audio = audio.copy()

        for aug_name in self.augmentations:
            if self._rng.random() > self.p:
                continue

            if aug_name == "noise":
                audio = self._add_noise(audio)
            elif aug_name == "timeshift":
                audio = self._time_shift(audio)
            elif aug_name == "pitchshift":
                audio = self._pitch_shift(audio)
            elif aug_name == "timestretch":
                audio = self._time_stretch(audio)
            elif aug_name == "gain":
                audio = self._random_gain(audio)

        return audio, target

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise."""
        noise = self._rng.normal(0, self.noise_level, len(audio))
        return audio + noise

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Random time shift."""
        shift = int(len(audio) * self._rng.uniform(-self.shift_range, self.shift_range))
        if shift > 0:
            return np.pad(audio[shift:], (0, shift), mode='constant')
        elif shift < 0:
            return np.pad(audio[:shift], (-shift, 0), mode='constant')
        return audio

    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Random pitch shift."""
        try:
            import librosa
        except ImportError:
            return audio

        n_steps = self._rng.randint(-self.pitch_range, self.pitch_range + 1)
        return librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps,
        )

    def _time_stretch(self, audio: np.ndarray) -> np.ndarray:
        """Random time stretch."""
        try:
            import librosa
        except ImportError:
            return audio

        rate = self._rng.uniform(*self.stretch_range)
        stretched = librosa.effects.time_stretch(audio, rate=rate)

        # Pad or trim to original length
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        else:
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))

        return stretched

    def _random_gain(self, audio: np.ndarray) -> np.ndarray:
        """Random volume change."""
        gain = self._rng.uniform(0.5, 2.0)
        return audio * gain

    def apply_specaugment(
        self,
        spectrogram: np.ndarray,
        freq_mask_param: int = 10,
        time_mask_param: int = 20,
        n_freq_masks: int = 2,
        n_time_masks: int = 2,
    ) -> np.ndarray:
        """Apply SpecAugment to spectrogram.

        Parameters
        ----------
        spectrogram : ndarray
            Mel-spectrogram of shape (n_mels, n_frames).
        freq_mask_param : int
            Maximum frequency mask width.
        time_mask_param : int
            Maximum time mask width.
        n_freq_masks : int
            Number of frequency masks.
        n_time_masks : int
            Number of time masks.

        Returns
        -------
        ndarray
            Augmented spectrogram.
        """
        spec = spectrogram.copy()
        n_mels, n_frames = spec.shape

        # Frequency masking
        for _ in range(n_freq_masks):
            f = self._rng.randint(0, freq_mask_param + 1)
            f0 = self._rng.randint(0, max(1, n_mels - f))
            spec[f0:f0 + f, :] = spec.mean()

        # Time masking
        for _ in range(n_time_masks):
            t = self._rng.randint(0, time_mask_param + 1)
            t0 = self._rng.randint(0, max(1, n_frames - t))
            spec[:, t0:t0 + t] = spec.mean()

        return spec

    def mixup(
        self,
        audio1: np.ndarray,
        audio2: np.ndarray,
        target1: np.ndarray,
        target2: np.ndarray,
        alpha: float = 0.4,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mix two audio samples with their labels.

        Parameters
        ----------
        audio1, audio2 : ndarray
            Audio waveforms.
        target1, target2 : ndarray
            One-hot encoded labels.
        alpha : float
            Beta distribution parameter.

        Returns
        -------
        mixed_audio : ndarray
            Mixed audio.
        mixed_target : ndarray
            Mixed labels.
        """
        lam = self._rng.beta(alpha, alpha)

        # Match lengths
        min_len = min(len(audio1), len(audio2))
        audio1 = audio1[:min_len]
        audio2 = audio2[:min_len]

        mixed_audio = lam * audio1 + (1 - lam) * audio2
        mixed_target = lam * target1 + (1 - lam) * target2

        return mixed_audio, mixed_target
