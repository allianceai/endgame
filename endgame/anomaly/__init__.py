from __future__ import annotations

"""Anomaly detection module: Isolation Forest, LOF, GritBot, PyOD integration, and more.

This module provides sklearn-compatible anomaly detectors with competition-tuned
defaults, including wrappers for popular algorithms and PyOD integration.

Example
-------
>>> from endgame.anomaly import IsolationForestDetector, PyODDetector, GritBotDetector
>>>
>>> # Simple Isolation Forest
>>> detector = IsolationForestDetector(contamination=0.1)
>>> detector.fit(X_train)
>>> scores = detector.decision_function(X_test)  # Higher = more anomalous
>>> labels = detector.predict(X_test)  # 1 = anomaly, 0 = normal
>>>
>>> # PyOD wrapper with any algorithm
>>> detector = PyODDetector(algorithm='ECOD', contamination=0.05)
>>> detector.fit(X_train)
>>> scores = detector.decision_function(X_test)
>>>
>>> # GritBot: Rule-based with interpretable context
>>> detector = GritBotDetector(filtering_level=50)
>>> detector.fit(X_train)
>>> print(detector.get_anomaly_report())  # Human-readable explanations
"""

from endgame.anomaly.gritbot import (
    Anomaly,
    AnomalyContext,
    GritBotDetector,
)
from endgame.anomaly.isolation_forest import IsolationForestDetector

try:
    from treeple import ExtendedIsolationForest
except ImportError:
    from endgame.anomaly.isolation_forest import ExtendedIsolationForest
from endgame.anomaly.lof import LocalOutlierFactorDetector
from endgame.anomaly.pyod_wrapper import (
    PYOD_ALGORITHMS,
    PyODDetector,
    create_detector_ensemble,
)

__all__ = [
    # Core detectors
    "IsolationForestDetector",
    "ExtendedIsolationForest",
    "LocalOutlierFactorDetector",
    # GritBot (rule-based, interpretable)
    "GritBotDetector",
    "Anomaly",
    "AnomalyContext",
    # PyOD integration
    "PyODDetector",
    "PYOD_ALGORITHMS",
    "create_detector_ensemble",
]
