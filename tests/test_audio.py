"""Tests for audio module."""

import numpy as np
import pytest


class TestSpectrogramTransformer:
    """Tests for SpectrogramTransformer."""

    def test_mel_spectrogram(self):
        """Test mel spectrogram extraction."""
        librosa = pytest.importorskip("librosa")
        from endgame.audio import SpectrogramTransformer

        # Generate dummy audio
        sr = 32000
        duration = 1.0
        audio = np.random.randn(int(sr * duration))

        transformer = SpectrogramTransformer(
            sample_rate=sr,
            n_mels=128,
            n_fft=2048,
            hop_length=512,
        )
        spec = transformer.transform(audio)

        assert spec.shape[0] == 128  # n_mels

    def test_power_to_db(self):
        """Test power to decibel conversion."""
        librosa = pytest.importorskip("librosa")
        from endgame.audio import SpectrogramTransformer

        sr = 32000
        audio = np.random.randn(sr)

        # Test dB conversion without normalization so raw dB values are preserved
        transformer_raw = SpectrogramTransformer(
            sample_rate=sr,
            n_mels=64,
            to_db=True,
            normalize=False,
        )
        spec_raw = transformer_raw.transform(audio)

        # Raw dB values (ref=np.max) are typically negative
        assert spec_raw.mean() < 0

        # With normalization (default), mean is approximately zero
        transformer_norm = SpectrogramTransformer(
            sample_rate=sr,
            n_mels=64,
            to_db=True,
        )
        spec_norm = transformer_norm.transform(audio)
        assert abs(spec_norm.mean()) < 1e-6


class TestPCENTransformer:
    """Tests for PCEN transformer."""

    def test_pcen(self):
        """Test PCEN normalization."""
        librosa = pytest.importorskip("librosa")
        from endgame.audio import PCENTransformer

        sr = 32000
        audio = np.random.randn(sr)

        transformer = PCENTransformer(
            sample_rate=sr,
            n_mels=64,
        )
        spec = transformer.transform(audio)

        assert spec.shape[0] == 64
        # PCEN output should be bounded
        assert spec.min() >= 0


class TestAudioAugmentation:
    """Tests for audio augmentation."""

    def test_add_noise(self):
        """Test noise augmentation."""
        from endgame.audio import AudioAugmentation

        audio = np.zeros(1000)

        aug = AudioAugmentation(
            augmentations=["noise"],
            noise_level=0.01,
            p=1.0,
            random_state=42,
        )
        augmented, _ = aug.apply(audio)

        # Should have added noise
        assert augmented.std() > 0

    def test_time_shift(self):
        """Test time shift augmentation."""
        from endgame.audio import AudioAugmentation

        # Create audio with clear structure
        audio = np.concatenate([np.ones(500), np.zeros(500)])

        aug = AudioAugmentation(
            augmentations=["timeshift"],
            shift_range=0.2,
            p=1.0,
            random_state=42,
        )
        augmented, _ = aug.apply(audio)

        # Shape should be preserved
        assert len(augmented) == len(audio)

    def test_random_gain(self):
        """Test random gain augmentation."""
        from endgame.audio import AudioAugmentation

        audio = np.ones(1000)

        aug = AudioAugmentation(
            augmentations=["gain"],
            p=1.0,
            random_state=42,
        )
        augmented, _ = aug.apply(audio)

        # Gain should change amplitude
        assert augmented.mean() != 1.0

    def test_specaugment(self):
        """Test SpecAugment."""
        from endgame.audio import AudioAugmentation

        # Create dummy spectrogram
        spec = np.ones((64, 100))

        aug = AudioAugmentation(random_state=42)
        augmented = aug.apply_specaugment(
            spec,
            freq_mask_param=10,
            time_mask_param=10,
            n_freq_masks=2,
            n_time_masks=2,
        )

        # Should have masked regions
        assert augmented.shape == spec.shape

    def test_mixup(self):
        """Test mixup augmentation."""
        from endgame.audio import AudioAugmentation

        audio1 = np.ones(1000)
        audio2 = np.zeros(1000)
        target1 = np.array([1, 0])
        target2 = np.array([0, 1])

        aug = AudioAugmentation(random_state=42)
        mixed_audio, mixed_target = aug.mixup(
            audio1, audio2, target1, target2, alpha=0.4
        )

        # Mixed values should be between originals
        assert 0 < mixed_audio.mean() < 1
        assert mixed_target.sum() == 1.0  # Still sums to 1

    def test_multiple_augmentations(self):
        """Test applying multiple augmentations."""
        from endgame.audio import AudioAugmentation

        audio = np.random.randn(1000)

        aug = AudioAugmentation(
            augmentations=["noise", "timeshift", "gain"],
            p=1.0,
            random_state=42,
        )
        augmented, _ = aug.apply(audio)

        # Should be modified
        assert not np.allclose(augmented, audio)
