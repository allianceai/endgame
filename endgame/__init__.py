from __future__ import annotations

"""
Endgame: The Competitive Machine Learning Framework

Sklearn-Native, Polars-Powered, Victory-Driven.

Usage:
    import endgame as eg

    # Check for train/test drift
    av = eg.validation.AdversarialValidator()
    result = av.check_drift(X_train, X_test)

    # Feature engineering
    encoder = eg.preprocessing.SafeTargetEncoder()
    X_encoded = encoder.fit_transform(X, y)

    # Train models
    model = eg.models.LGBMWrapper(preset='endgame')
    model.fit(X_train, y_train)

    # Ensemble
    ensemble = eg.ensemble.HillClimbingEnsemble(metric='roc_auc')
    ensemble.fit(oof_predictions, y_train)
"""

__version__ = "1.0.1"

# Note: models imported lazily for heavyweight optional dependencies
# from endgame import models
from endgame import (
    anomaly,
    calibration,
    clustering,
    ensemble,
    fairness,
    preprocessing,
    semi_supervised,
    tune,
    utils,
    validation,
)


# Lazy imports for heavy modules
def __getattr__(name: str):
    import importlib

    # Models and other heavy modules imported lazily for optional dependencies
    if name in ("models", "vision", "nlp", "audio", "benchmark", "kaggle", "quick", "visualization", "persistence", "explain", "tracking", "timeseries", "signal", "automl", "dimensionality_reduction", "feature_selection"):
        module = importlib.import_module(f"endgame.{name}")
        globals()[name] = module
        return module
    if name == "save":
        from endgame.persistence import save
        return save
    if name == "load":
        from endgame.persistence import load
        return load
    if name == "export_onnx":
        from endgame.persistence import export_onnx
        return export_onnx
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "validation",
    "preprocessing",
    "models",
    "ensemble",
    "tune",
    "calibration",
    "semi_supervised",
    "anomaly",
    "fairness",
    "clustering",
    "benchmark",
    "vision",
    "nlp",
    "audio",
    "kaggle",
    "utils",
    "quick",
    "visualization",
    "persistence",
    "tracking",
    "timeseries",
    "signal",
    "automl",
    "dimensionality_reduction",
    "feature_selection",
    "save",
    "load",
    "export_onnx",
    "explain",
]
