"""Fuzzy ensemble methods.

Provides fuzzy extensions of bagging, boosting, and stacking
that leverage soft splits and membership-weighted aggregation.

Classes
-------
FuzzyRandomForestClassifier
    Ensemble of fuzzy decision trees with bootstrap aggregation.
FuzzyRandomForestRegressor
    Regression variant of fuzzy random forest.
FuzzyBoostedTreesClassifier
    Gradient boosting with fuzzy (soft) split nodes.
FuzzyBoostedTreesRegressor
    Regression variant of fuzzy boosted trees.
FuzzyBaggingClassifier
    Membership-weighted bagging with fuzzy c-means sample weighting.
FuzzyBaggingRegressor
    Regression variant of fuzzy bagging.
StackedFuzzySystem
    Meta-learning with fuzzy TSK combiner over base estimator outputs.
"""

from endgame.fuzzy.ensemble.fuzzy_random_forest import (
    FuzzyRandomForestClassifier,
    FuzzyRandomForestRegressor,
)
from endgame.fuzzy.ensemble.fuzzy_boosted_trees import (
    FuzzyBoostedTreesClassifier,
    FuzzyBoostedTreesRegressor,
)
from endgame.fuzzy.ensemble.fuzzy_bagging import (
    FuzzyBaggingClassifier,
    FuzzyBaggingRegressor,
)
from endgame.fuzzy.ensemble.stacked_fuzzy import StackedFuzzySystem

__all__ = [
    "FuzzyRandomForestClassifier",
    "FuzzyRandomForestRegressor",
    "FuzzyBoostedTreesClassifier",
    "FuzzyBoostedTreesRegressor",
    "FuzzyBaggingClassifier",
    "FuzzyBaggingRegressor",
    "StackedFuzzySystem",
]
