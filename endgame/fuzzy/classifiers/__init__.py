"""Fuzzy classifiers: FuzzyKNN, FuzzyDecisionTree, NEFCLASS.

Provides sklearn-compatible fuzzy classification algorithms including
instance-based, tree-based, and neuro-fuzzy approaches.
"""

from endgame.fuzzy.classifiers.fuzzy_knn import FuzzyKNNClassifier
from endgame.fuzzy.classifiers.fuzzy_decision_tree import (
    FuzzyDecisionTreeClassifier,
    FuzzyDecisionTreeRegressor,
)
from endgame.fuzzy.classifiers.nefclass import NEFCLASSClassifier

__all__ = [
    "FuzzyKNNClassifier",
    "FuzzyDecisionTreeClassifier",
    "FuzzyDecisionTreeRegressor",
    "NEFCLASSClassifier",
]
