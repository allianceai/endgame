"""Neuro-fuzzy and hybrid fuzzy-neural architectures."""

from endgame.fuzzy.neurofuzzy.falcon import FALCONClassifier, FALCONRegressor
from endgame.fuzzy.neurofuzzy.sofnn import SOFNNRegressor
from endgame.fuzzy.neurofuzzy.denfis import DENFISRegressor, DENFISClassifier
from endgame.fuzzy.neurofuzzy.fnn_tsk import FNNTSKRegressor, FNNTSKClassifier

__all__ = [
    "FALCONClassifier",
    "FALCONRegressor",
    "SOFNNRegressor",
    "DENFISRegressor",
    "DENFISClassifier",
    "FNNTSKRegressor",
    "FNNTSKClassifier",
]
