from __future__ import annotations

"""Semi-supervised learning methods."""

from endgame.semi_supervised.self_training import (
    SelfTrainingClassifier,
    SelfTrainingRegressor,
)

__all__ = [
    "SelfTrainingClassifier",
    "SelfTrainingRegressor",
]
