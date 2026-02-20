"""Audio module: Spectrogram processing, SED models, augmentation, features."""

from endgame.audio.augmentation import AudioAugmentation
from endgame.audio.features import AudioFeatureExtractor
from endgame.audio.pretrained import PretrainedAudioClassifier
from endgame.audio.sed import SEDModel
from endgame.audio.spectrogram import PCENTransformer, SpectrogramTransformer

__all__ = [
    "SpectrogramTransformer",
    "PCENTransformer",
    "AudioAugmentation",
    "SEDModel",
    "AudioFeatureExtractor",
    "PretrainedAudioClassifier",
]
