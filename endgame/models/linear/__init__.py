from __future__ import annotations

"""Linear models including MARS (Multivariate Adaptive Regression Splines)."""

from endgame.models.linear.mars import MARSClassifier, MARSRegressor

__all__ = [
    "MARSRegressor",
    "MARSClassifier",
]
