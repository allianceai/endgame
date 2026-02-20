"""Fairness module: Bias metrics, mitigation methods, and reporting.

This module provides tools for measuring and mitigating algorithmic bias
across sensitive attributes in binary classification:

- **Metrics**: Demographic parity, equalized odds, disparate impact,
  calibration by group.
- **Pre-processing**: ``ReweighingPreprocessor`` computes sample weights
  to achieve statistical parity.
- **In-processing**: ``ExponentiatedGradient`` trains classifiers under
  fairness constraints (requires fairlearn).
- **Post-processing**: ``CalibratedEqOdds`` adjusts per-group thresholds
  to equalize odds.
- **Reporting**: ``FairnessReport`` generates standalone HTML reports.

Example
-------
>>> import numpy as np
>>> from endgame.fairness import demographic_parity, equalized_odds, FairnessReport
>>>
>>> # Compute metrics
>>> dp = demographic_parity(y_true, y_pred, sensitive_attr)
>>> print(dp["max_disparity"])
>>>
>>> # Generate a full HTML report
>>> report = FairnessReport(model, X_test, y_test, sensitive_attr)
>>> report.save("fairness_report.html")
"""

from endgame.fairness.metrics import (
    calibration_by_group,
    demographic_parity,
    disparate_impact,
    equalized_odds,
)
from endgame.fairness.mitigation import (
    CalibratedEqOdds,
    ExponentiatedGradient,
    ReweighingPreprocessor,
)
from endgame.fairness.report import FairnessReport

__all__ = [
    # Metrics
    "demographic_parity",
    "equalized_odds",
    "disparate_impact",
    "calibration_by_group",
    # Mitigation - pre-processing
    "ReweighingPreprocessor",
    # Mitigation - in-processing
    "ExponentiatedGradient",
    # Mitigation - post-processing
    "CalibratedEqOdds",
    # Reporting
    "FairnessReport",
]
