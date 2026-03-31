"""Fuzzy logic core utilities: membership functions, operators, defuzzification."""

from endgame.fuzzy.core.membership import (
    BaseMembershipFunction,
    TriangularMF,
    TrapezoidalMF,
    GaussianMF,
    GeneralizedBellMF,
    SigmoidalMF,
    DifferenceSigmoidalMF,
    PiMF,
    IntervalType2GaussianMF,
    IntervalType2TriangularMF,
)
from endgame.fuzzy.core.operators import (
    t_norm,
    t_conorm,
    MinTNorm,
    ProductTNorm,
    LukasiewiczTNorm,
    HamacherTNorm,
    MaxTConorm,
    ProbabilisticSumTConorm,
    LukasiewiczTConorm,
    HamacherTConorm,
)
from endgame.fuzzy.core.defuzzification import (
    defuzzify,
    centroid,
    bisector,
    mean_of_maxima,
    weighted_average,
    height_method,
)
from endgame.fuzzy.core.base import (
    BaseFuzzySystem,
    BaseFuzzyClassifier,
    BaseFuzzyRegressor,
    BaseFuzzyTransformer,
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
