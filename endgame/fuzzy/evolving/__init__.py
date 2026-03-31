"""Evolving (online) fuzzy inference systems.

Provides incremental fuzzy systems that adapt their structure and parameters
from streaming data without retraining from scratch.

Classes
-------
EvolvingTSK
    Evolving Takagi-Sugeno-Kang fuzzy system (eTS) by Angelov & Filev.
EvolvingTSKPlus
    Improved eTS+ with rule merging and local learning.
PANFISRegressor
    Parsimonious ANFIS regressor with online rule growing/pruning.
PANFISClassifier
    Parsimonious ANFIS classifier with online rule growing/pruning.
AutoCloudClassifier
    Data cloud-based evolving classifier using recursive density estimation.
FLEXFISRegressor
    Flexible Fuzzy Inference System with incremental eClustering.
"""

from endgame.fuzzy.evolving.ets import EvolvingTSK, EvolvingTSKPlus
from endgame.fuzzy.evolving.panfis import PANFISClassifier, PANFISRegressor
from endgame.fuzzy.evolving.autocloud import AutoCloudClassifier
from endgame.fuzzy.evolving.flexfis import FLEXFISRegressor

__all__ = [
    "EvolvingTSK",
    "EvolvingTSKPlus",
    "PANFISRegressor",
    "PANFISClassifier",
    "AutoCloudClassifier",
    "FLEXFISRegressor",
]
