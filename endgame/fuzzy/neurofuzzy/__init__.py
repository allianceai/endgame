"""Neuro-fuzzy and hybrid fuzzy-neural architectures."""

from endgame.fuzzy.neurofuzzy.denfis import DENFISClassifier, DENFISRegressor
from endgame.fuzzy.neurofuzzy.falcon import FALCONClassifier, FALCONRegressor
from endgame.fuzzy.neurofuzzy.fnn_tsk import FNNTSKClassifier, FNNTSKRegressor
from endgame.fuzzy.neurofuzzy.sofnn import SOFNNRegressor

__all__ = [
    "FALCONClassifier",
    "FALCONRegressor",
    "SOFNNRegressor",
    "DENFISRegressor",
    "DENFISClassifier",
    "FNNTSKRegressor",
    "FNNTSKClassifier",
]
