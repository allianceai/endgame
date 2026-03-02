from __future__ import annotations

"""Ensemble module: Classic and SOTA ensemble methods.

Classic: Voting, Bagging, Boosting, Stacking, Blending, Hill Climbing.
SOTA:    Super Learner, Bayesian Model Averaging, Negative Correlation
         Learning, Snapshot Ensemble, Cascade Ensemble.
Also:    Threshold optimization, knowledge distillation, multi-output.
"""

# Classic ensembles
from endgame.ensemble.bagging import BaggingClassifier, BaggingRegressor
from endgame.ensemble.bayesian_averaging import BayesianModelAveraging
from endgame.ensemble.blending import (
    BlendingEnsemble,
    OptimizedBlender,
    PowerBlender,
    RankAverageBlender,
)
from endgame.ensemble.boosting import AdaBoostClassifier, AdaBoostRegressor
from endgame.ensemble.cascade import CascadeEnsemble
from endgame.ensemble.distillation import KnowledgeDistiller
from endgame.ensemble.hill_climbing import HillClimbingEnsemble
from endgame.ensemble.multi_output import (
    ClassifierChain,
    MultiOutputClassifier,
    MultiOutputRegressor,
    RegressorChain,
)
from endgame.ensemble.negative_correlation import NegativeCorrelationEnsemble
from endgame.ensemble.snapshot import SnapshotEnsemble

# Stacking & blending
from endgame.ensemble.stacking import StackingEnsemble

# SOTA ensembles
from endgame.ensemble.super_learner import SuperLearner

# Utilities
from endgame.ensemble.threshold import ThresholdOptimizer
from endgame.ensemble.voting import VotingClassifier, VotingRegressor

__all__ = [
    # Classic
    "VotingClassifier",
    "VotingRegressor",
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    # Stacking & blending
    "StackingEnsemble",
    "BlendingEnsemble",
    "OptimizedBlender",
    "RankAverageBlender",
    "PowerBlender",
    "HillClimbingEnsemble",
    # SOTA
    "SuperLearner",
    "BayesianModelAveraging",
    "NegativeCorrelationEnsemble",
    "SnapshotEnsemble",
    "CascadeEnsemble",
    # Utilities
    "ThresholdOptimizer",
    "KnowledgeDistiller",
    "MultiOutputClassifier",
    "MultiOutputRegressor",
    "ClassifierChain",
    "RegressorChain",
]
