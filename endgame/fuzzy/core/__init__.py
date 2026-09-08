"""Fuzzy logic core utilities: membership functions, operators, defuzzification."""

from endgame.fuzzy.core.base import (
    BaseFuzzyClassifier,
    BaseFuzzyRegressor,
    BaseFuzzySystem,
    BaseFuzzyTransformer,
)
from endgame.fuzzy.core.defuzzification import (
    bisector,
    centroid,
    defuzzify,
    height_method,
    mean_of_maxima,
    weighted_average,
)
from endgame.fuzzy.core.membership import (
    BaseMembershipFunction,
    DifferenceSigmoidalMF,
    GaussianMF,
    GeneralizedBellMF,
    IntervalType2GaussianMF,
    IntervalType2TriangularMF,
    PiMF,
    SigmoidalMF,
    TrapezoidalMF,
    TriangularMF,
)
from endgame.fuzzy.core.operators import (
    HamacherTConorm,
    HamacherTNorm,
    LukasiewiczTConorm,
    LukasiewiczTNorm,
    MaxTConorm,
    MinTNorm,
    ProbabilisticSumTConorm,
    ProductTNorm,
    t_conorm,
    t_norm,
)

__all__ = [
    # Membership functions
    "BaseMembershipFunction",
    "TriangularMF",
    "TrapezoidalMF",
    "GaussianMF",
    "GeneralizedBellMF",
    "SigmoidalMF",
    "DifferenceSigmoidalMF",
    "PiMF",
    "IntervalType2GaussianMF",
    "IntervalType2TriangularMF",
    # Operators
    "t_norm",
    "t_conorm",
    "MinTNorm",
    "ProductTNorm",
    "LukasiewiczTNorm",
    "HamacherTNorm",
    "MaxTConorm",
    "ProbabilisticSumTConorm",
    "LukasiewiczTConorm",
    "HamacherTConorm",
    # Defuzzification
    "defuzzify",
    "centroid",
    "bisector",
    "mean_of_maxima",
    "weighted_average",
    "height_method",
    # Base classes
    "BaseFuzzySystem",
    "BaseFuzzyClassifier",
    "BaseFuzzyRegressor",
    "BaseFuzzyTransformer",
]
