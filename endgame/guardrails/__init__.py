from __future__ import annotations

"""Standalone Guardrails Module for Data Leakage Detection & Removal.

Provides an sklearn-compatible ``LeakageDetector`` transformer and a
``check_data_quality`` convenience function for detecting data health
issues, target leakage, feature redundancy, and train/test problems.

Classes
-------
LeakageDetector : sklearn TransformerMixin
    Configurable data quality and leakage detection transformer.
GuardrailsReport : dataclass
    Aggregated report of all detected issues.
DataQualityWarning : dataclass
    Individual data quality issue.

Functions
---------
check_data_quality(X, y, ...) -> GuardrailsReport
    One-line data quality check.

Example
-------
>>> import endgame as eg
>>> report = eg.check_data_quality(X_train, y_train)
>>> print(report)

>>> from endgame.guardrails import LeakageDetector
>>> detector = LeakageDetector(mode="fix")
>>> X_clean = detector.fit_transform(X_train, y_train)
"""

from typing import Any

from endgame.guardrails.detector import LeakageDetector
from endgame.guardrails.report import DataQualityWarning, GuardrailsReport

__all__ = [
    "LeakageDetector",
    "GuardrailsReport",
    "DataQualityWarning",
    "check_data_quality",
]


def check_data_quality(
    X: Any,
    y: Any = None,
    *,
    X_test: Any = None,
    checks: str | list[str] = "default",
    leakage_threshold: float = 0.95,
    mi_threshold: float = 0.95,
    redundancy_threshold: float = 0.98,
    suspect_patterns: list[str] | None = None,
    future_features: list[str] | None = None,
    time_budget: float | None = None,
    verbose: bool = False,
) -> GuardrailsReport:
    """Run data quality checks and return a report.

    This is a convenience function that creates a ``LeakageDetector``
    in detect mode and returns its report.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like, optional
        Target variable.
    X_test : array-like, optional
        Test features for train/test checks.
    checks : str or list of str, default="default"
        Which checks to run.
    leakage_threshold : float, default=0.95
        Correlation threshold for leakage detection.
    mi_threshold : float, default=0.95
        Mutual information threshold for leakage detection.
    redundancy_threshold : float, default=0.98
        Correlation threshold for redundancy detection.
    suspect_patterns : list of str, optional
        Regex patterns for suspect features.
    future_features : list of str, optional
        Feature names with future information.
    time_budget : float, optional
        Time budget in seconds.
    verbose : bool, default=False
        Print progress.

    Returns
    -------
    GuardrailsReport
        Report of detected issues.

    Examples
    --------
    >>> from endgame.guardrails import check_data_quality
    >>> report = check_data_quality(X_train, y_train)
    >>> print(report.summary())
    >>> report.features_to_drop
    """
    detector = LeakageDetector(
        mode="detect",
        checks=checks,
        leakage_threshold=leakage_threshold,
        mi_threshold=mi_threshold,
        redundancy_threshold=redundancy_threshold,
        suspect_patterns=suspect_patterns,
        future_features=future_features,
        time_budget=time_budget,
        verbose=verbose,
    )
    detector.fit(X, y, X_test=X_test)
    return detector.report_
