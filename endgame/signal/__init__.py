from __future__ import annotations

"""Signal processing module for time series and biosignal analysis.

This module provides sklearn-compatible signal processing transformers
and feature extractors designed to integrate with the time series module.

Submodules
----------
filtering : Digital filters (Butterworth, FIR, Savgol, notch)
spectral : Frequency analysis (FFT, Welch PSD, multitaper, band powers)
wavelets : Wavelet transforms (CWT, DWT, scattering)
features : Time-domain and nonlinear feature extraction
entropy : Entropy measures (sample, permutation, spectral)
complexity : Fractal dimensions, Hurst exponent, DFA, LZC
spatial : CSP, tangent space, Riemannian geometry
connectivity : Coherence, PLV, burst/suppression detection

Integration with Time Series
----------------------------
Signal processing transformers can be used in sklearn pipelines with
time series forecasters and classifiers:

>>> from endgame.signal import ButterworthFilter, WelchPSD
>>> from endgame.timeseries import MiniRocketClassifier
>>> from sklearn.pipeline import make_pipeline
>>>
>>> pipe = make_pipeline(
...     ButterworthFilter(lowcut=1, highcut=50, fs=256),
...     WelchPSD(fs=256, nperseg=256),
...     MiniRocketClassifier()
... )

Examples
--------
>>> from endgame.signal import ButterworthFilter, BandPowerExtractor
>>>
>>> # Filter signal
>>> filt = ButterworthFilter(lowcut=0.5, highcut=40, fs=256)
>>> filtered = filt.fit_transform(raw_signal)
>>>
>>> # Extract band powers
>>> bp = BandPowerExtractor(fs=256, bands={'alpha': (8, 13), 'beta': (13, 30)})
>>> powers = bp.fit_transform(filtered)

>>> # Entropy and complexity features
>>> from endgame.signal import EntropyFeatureExtractor, ComplexityFeatureExtractor
>>> entropy = EntropyFeatureExtractor(fs=256)
>>> complexity = ComplexityFeatureExtractor()
"""

# Base classes
from endgame.signal.base import (
    BaseFeatureExtractor,
    BaseSignalTransformer,
    SignalMixin,
)

# Complexity and fractal measures
from endgame.signal.complexity import (
    DFA,
    ComplexityFeatureExtractor,
    HiguchiFD,
    HurstExponent,
    KatzFD,
    LempelZivComplexity,
    PetrosianFD,
    detrended_fluctuation,
    higuchi_fd,
    hurst_exponent,
    katz_fd,
    lempel_ziv_complexity,
    petrosian_fd,
)

# Connectivity and EEG-specific features
from endgame.signal.connectivity import (
    BurstSuppressionFeatures,
    CoherenceFeatureExtractor,
    ConnectivityFeatureExtractor,
    PLVFeatureExtractor,
    SpikeFeatures,
    coherence,
    cross_correlation,
    detect_bursts,
    detect_spikes,
    detect_suppressions,
    phase_locking_value,
)

# Entropy measures
from endgame.signal.entropy import (
    ApproximateEntropy,
    EntropyFeatureExtractor,
    PermutationEntropy,
    SampleEntropy,
    SpectralEntropy,
    SVDEntropy,
    approximate_entropy,
    permutation_entropy,
    sample_entropy,
    spectral_entropy,
    svd_entropy,
)

# Time-domain features
from endgame.signal.features import (
    HjorthParameters,
    PeakFeatures,
    StatisticalFeatures,
    TimeDomainFeatures,
    ZeroCrossingFeatures,
    compute_energy,
    compute_hjorth,
    compute_line_length,
    compute_rms,
    count_zero_crossings,
)

# Filtering
from endgame.signal.filtering import (
    ButterworthFilter,
    FilterBank,
    FIRFilter,
    MedianFilter,
    NotchFilter,
    SavgolFilter,
)

# Spatial filtering (CSP, Tangent Space)
from endgame.signal.spatial import (
    CSP,
    CovarianceEstimator,
    FilterBankCSP,
    TangentSpace,
)

# Spectral analysis
from endgame.signal.spectral import (
    BandPowerExtractor,
    FFTTransformer,
    MultitaperPSD,
    SpectralFeatureExtractor,
    WelchPSD,
    compute_band_power,
    compute_psd,
)

__all__ = [
    # Base
    "BaseFeatureExtractor",
    "BaseSignalTransformer",
    "SignalMixin",
    # Filtering
    "ButterworthFilter",
    "FIRFilter",
    "SavgolFilter",
    "NotchFilter",
    "MedianFilter",
    "FilterBank",
    # Spectral
    "FFTTransformer",
    "WelchPSD",
    "MultitaperPSD",
    "BandPowerExtractor",
    "SpectralFeatureExtractor",
    "compute_psd",
    "compute_band_power",
    # Features
    "TimeDomainFeatures",
    "StatisticalFeatures",
    "HjorthParameters",
    "ZeroCrossingFeatures",
    "PeakFeatures",
    "compute_hjorth",
    "compute_rms",
    "compute_energy",
    "compute_line_length",
    "count_zero_crossings",
    # Entropy
    "PermutationEntropy",
    "SampleEntropy",
    "ApproximateEntropy",
    "SpectralEntropy",
    "SVDEntropy",
    "EntropyFeatureExtractor",
    "permutation_entropy",
    "sample_entropy",
    "approximate_entropy",
    "spectral_entropy",
    "svd_entropy",
    # Complexity
    "HiguchiFD",
    "PetrosianFD",
    "KatzFD",
    "HurstExponent",
    "DFA",
    "LempelZivComplexity",
    "ComplexityFeatureExtractor",
    "higuchi_fd",
    "petrosian_fd",
    "katz_fd",
    "hurst_exponent",
    "detrended_fluctuation",
    "lempel_ziv_complexity",
    # Spatial
    "CovarianceEstimator",
    "CSP",
    "TangentSpace",
    "FilterBankCSP",
    # Connectivity
    "CoherenceFeatureExtractor",
    "PLVFeatureExtractor",
    "BurstSuppressionFeatures",
    "SpikeFeatures",
    "ConnectivityFeatureExtractor",
    "coherence",
    "phase_locking_value",
    "cross_correlation",
    "detect_bursts",
    "detect_suppressions",
    "detect_spikes",
]

# Wavelet transforms (optional: pywavelets)
try:
    from endgame.signal.wavelets import (
        CWTTransformer,
        DWTTransformer,
        WaveletFeatureExtractor,
        WaveletPacketTransformer,
        compute_cwt,
        compute_dwt,
        reconstruct_from_dwt,
    )

    __all__.extend([
        "CWTTransformer",
        "DWTTransformer",
        "WaveletPacketTransformer",
        "WaveletFeatureExtractor",
        "compute_cwt",
        "compute_dwt",
        "reconstruct_from_dwt",
    ])
except ImportError:
    pass
