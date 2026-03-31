"""Interval and General Type-2 Fuzzy Logic Systems.

Provides IT2 FLS (Mamdani), IT2 TSK, IT2 ANFIS, and General Type-2 FLS
implementations with Karnik-Mendel and Nie-Tan type reduction.
"""

from endgame.fuzzy.type2.it2_fls import (
    IT2FLSClassifier,
    IT2FLSRegressor,
    karnik_mendel,
    nie_tan,
)
from endgame.fuzzy.type2.it2_tsk import IT2TSKRegressor
from endgame.fuzzy.type2.it2_anfis import IT2ANFISRegressor
from endgame.fuzzy.type2.general_t2 import GeneralType2FLS

__all__ = [
    "IT2FLSRegressor",
    "IT2FLSClassifier",
    "IT2TSKRegressor",
    "IT2ANFISRegressor",
    "GeneralType2FLS",
    "karnik_mendel",
    "nie_tan",
]
