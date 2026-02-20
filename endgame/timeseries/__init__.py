"""Time series forecasting and classification module.

This module provides sklearn-compatible time series models for both
forecasting and classification, plus feature extraction and validation.

Architecture designed to integrate with future signal processing module.

Submodules
----------
baselines : Simple forecasting baselines (Naive, SeasonalNaive, MA, EMA)
statistical : Statistical models via statsforecast (ARIMA, ETS, Theta)
neural : Deep learning models via Darts (N-BEATS, TFT, PatchTST)
features : Time series feature extraction (tsfresh integration)
validation : Time series cross-validation strategies
rocket : ROCKET, MiniROCKET, and HYDRA time series classification

Examples
--------
>>> # Forecasting
>>> from endgame.timeseries import NaiveForecaster, AutoARIMAForecaster
>>> model = NaiveForecaster(strategy='last')
>>> model.fit(y_train)
>>> forecast = model.predict(horizon=7)

>>> # Classification with ROCKET
>>> from endgame.timeseries import MiniRocketClassifier
>>> clf = MiniRocketClassifier()
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

from endgame.timeseries.base import (
    BaseForecaster,
    ForecasterMixin,
    MultivariateForecasterMixin,
    UnivariateForecasterMixin,
)
from endgame.timeseries.baselines import (
    DriftForecaster,
    ExponentialSmoothingForecaster,
    MovingAverageForecaster,
    NaiveForecaster,
    SeasonalNaiveForecaster,
    ThetaForecaster,
)
from endgame.timeseries.validation import (
    BlockedTimeSeriesSplit,
    ExpandingWindowCV,
    SlidingWindowCV,
    coverage,
    interval_width,
    mape,
    mase,
    rmsse,
    smape,
    wape,
    winkler_score,
)

__all__ = [
    # Base classes
    "BaseForecaster",
    "ForecasterMixin",
    "UnivariateForecasterMixin",
    "MultivariateForecasterMixin",
    # Simple baselines
    "NaiveForecaster",
    "SeasonalNaiveForecaster",
    "MovingAverageForecaster",
    "ExponentialSmoothingForecaster",
    "DriftForecaster",
    "ThetaForecaster",
    # Validation
    "ExpandingWindowCV",
    "SlidingWindowCV",
    "BlockedTimeSeriesSplit",
    # Metrics
    "mase",
    "smape",
    "mape",
    "rmsse",
    "wape",
    "coverage",
    "interval_width",
    "winkler_score",
]

# statsforecast wrappers (optional dependency)
try:
    from endgame.timeseries.statistical import (
        AutoARIMAForecaster,
        AutoETSForecaster,
        AutoThetaForecaster,
        CESForecaster,
        MSTLForecaster,
    )
    __all__.extend([
        "AutoARIMAForecaster",
        "AutoETSForecaster",
        "AutoThetaForecaster",
        "MSTLForecaster",
        "CESForecaster",
    ])
except ImportError:
    pass

# tsfresh feature extraction (optional dependency)
try:
    from endgame.timeseries.features import (
        TimeSeriesFeatureExtractor,
        TSFreshFeatureExtractor,
    )
    __all__.extend([
        "TSFreshFeatureExtractor",
        "TimeSeriesFeatureExtractor",
    ])
except ImportError:
    pass

# Darts neural models (optional dependency)
try:
    from endgame.timeseries.neural import (
        DLinearForecaster,
        NBEATSForecaster,
        NHITSForecaster,
        PatchTSTForecaster,
        TFTForecaster,
        TimesNetForecaster,
    )
    __all__.extend([
        "NBEATSForecaster",
        "NHITSForecaster",
        "TFTForecaster",
        "PatchTSTForecaster",
        "DLinearForecaster",
        "TimesNetForecaster",
    ])
except ImportError:
    pass

# ROCKET family - time series classification (optional: sktime or native)
from endgame.timeseries.rocket import (
    HydraClassifier,
    HydraMiniRocketClassifier,
    HydraTransformer,
    MiniRocketClassifier,
    MiniRocketTransformer,
    MultiRocketClassifier,
    MultiRocketTransformer,
    RocketClassifier,
    RocketTransformer,
)

__all__.extend([
    # Transformers
    "RocketTransformer",
    "MiniRocketTransformer",
    "MultiRocketTransformer",
    "HydraTransformer",
    # Classifiers
    "RocketClassifier",
    "MiniRocketClassifier",
    "MultiRocketClassifier",
    "HydraClassifier",
    "HydraMiniRocketClassifier",
])
