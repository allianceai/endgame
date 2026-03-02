from __future__ import annotations

"""Symbolic Regression models for interpretable equation discovery.

Pure-Python GP engine — no Julia or PySR dependency required.
"""

from endgame.models.symbolic._operators import OPERATOR_SETS
from endgame.models.symbolic.symbolic_classifier import SymbolicClassifier
from endgame.models.symbolic.symbolic_regressor import (
    PRESETS,
    SymbolicRegressor,
)

__all__ = [
    "SymbolicRegressor",
    "SymbolicClassifier",
    "PRESETS",
    "OPERATOR_SETS",
]
