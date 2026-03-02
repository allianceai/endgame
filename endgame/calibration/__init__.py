from __future__ import annotations

"""Calibration module: Conformal prediction and probability calibration.

This module provides tools for:
- Conformal prediction (prediction sets/intervals with coverage guarantees)
- Probability calibration (Temperature scaling, Platt scaling, etc.)
- Calibration analysis and diagnostics
"""

from endgame.calibration.analysis import (
    CalibrationAnalyzer,
    CalibrationReport,
    brier_score_decomposition,
    expected_calibration_error,
    maximum_calibration_error,
)
from endgame.calibration.conformal import (
    ConformalClassifier,
    ConformalRegressor,
    ConformizedQuantileRegressor,
)
from endgame.calibration.scaling import (
    BetaCalibration,
    HistogramBinning,
    IsotonicCalibration,
    PlattScaling,
    TemperatureScaling,
)
from endgame.calibration.venn_abers import VennABERS

__all__ = [
    # Conformal prediction
    "ConformalClassifier",
    "ConformalRegressor",
    "ConformizedQuantileRegressor",
    # Probability calibration
    "TemperatureScaling",
    "PlattScaling",
    "BetaCalibration",
    "IsotonicCalibration",
    "HistogramBinning",
    "VennABERS",
    # Analysis
    "CalibrationAnalyzer",
    "CalibrationReport",
    "expected_calibration_error",
    "maximum_calibration_error",
    "brier_score_decomposition",
]
