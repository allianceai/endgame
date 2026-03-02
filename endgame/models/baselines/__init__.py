from __future__ import annotations

"""Simple baseline models for ensemble diversity.

This module provides simple yet effective models that have different
inductive biases from tree-based and neural network models:

- Extreme Learning Machine (random projection neural network)
- Naive Bayes (feature independence assumption)
- Discriminant Analysis (LDA, QDA, RDA)
- K-Nearest Neighbors (instance-based learning)
- Linear Models (global linear decision boundaries)

These models are valuable for ensemble diversity because they make
fundamentally different assumptions about the data.
"""

from endgame.models.baselines.discriminant import (
    LDAClassifier,
    QDAClassifier,
    RDAClassifier,
)
from endgame.models.baselines.elm import (
    ELMClassifier,
    ELMRegressor,
)
from endgame.models.baselines.knn import (
    KNNClassifier,
    KNNRegressor,
)
from endgame.models.baselines.linear import (
    LinearClassifier,
    LinearRegressor,
)
from endgame.models.baselines.naive_bayes import (
    NaiveBayesClassifier,
)

__all__ = [
    "ELMClassifier",
    "ELMRegressor",
    "NaiveBayesClassifier",
    "LDAClassifier",
    "QDAClassifier",
    "RDAClassifier",
    "KNNClassifier",
    "KNNRegressor",
    "LinearClassifier",
    "LinearRegressor",
]
