from __future__ import annotations

"""Quick API for rapid prototyping and model comparison.

This module provides high-level functions for quickly training and
comparing models without manual configuration. Inspired by AutoGluon's
presets system but focused on competitive ML workflows.

Usage
-----
>>> import endgame as eg

>>> # Quick classification with default preset
>>> result = eg.quick.classify(X, y)

>>> # Quick regression with fast preset
>>> result = eg.quick.regress(X, y, preset='fast')

>>> # Compare multiple models
>>> comparison = eg.quick.compare(X, y, task='classification')

Presets
-------
- 'fast': Quick iteration, minimal tuning (~1 min)
- 'default': Balanced speed/accuracy (~5 min)
- 'competition': Full pipeline for competitions (~30 min)
- 'interpretable': Only interpretable models
"""

from endgame.quick.api import (
    PRESETS,
    ComparisonResult,
    QuickResult,
    classify,
    compare,
    regress,
)

__all__ = [
    "classify",
    "regress",
    "compare",
    "QuickResult",
    "ComparisonResult",
    "PRESETS",
]
