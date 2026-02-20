"""Subgroup discovery and rule induction methods.

This module provides algorithms for finding interpretable subgroups
in data that have unusual or interesting properties:

- PRIM: Patient Rule Induction Method for bump hunting
"""

from endgame.models.subgroup.prim import (
    Box,
    PRIMClassifier,
    PRIMRegressor,
    PRIMResult,
)

__all__ = [
    "PRIMClassifier",
    "PRIMRegressor",
    "Box",
    "PRIMResult",
]
