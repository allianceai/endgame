"""Fuzzy-rough set methods for classification and feature selection.

Combines fuzzy set theory with rough set theory for handling
uncertainty in both feature values and class boundaries.
"""

from endgame.fuzzy.rough.frnn import FuzzyRoughNNClassifier
from endgame.fuzzy.rough.frfs import FuzzyRoughFeatureSelector

__all__ = [
    "FuzzyRoughNNClassifier",
    "FuzzyRoughFeatureSelector",
]
