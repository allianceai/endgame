"""Interpretable Machine Learning Models.

This module provides wrappers and implementations for interpretable ML models
that produce human-readable predictions (rules, GAMs, sparse models).

Models included:
- CORELS: Certifiably Optimal Rule Lists
- NODE-GAM: Neural Oblivious Decision Ensembles for GAMs
- GAMI-Net: Generalized Additive Models with Structured Interactions
- SLIM/FasterRisk: Sparse Integer Linear Models for risk scores
- pyGAM: Generalized Additive Models with various splines
- GOSDT: Globally Optimal Decision Trees with Sparsity

Example
-------
>>> from endgame.models.interpretable import CORELSClassifier
>>> clf = CORELSClassifier(max_card=2, c=0.001)
>>> clf.fit(X_train, y_train)
>>> print(clf.get_rules())

>>> from endgame.models.interpretable import GAMClassifier
>>> gam = GAMClassifier(n_splines=25)
>>> gam.fit(X_train, y_train)
>>> gam.summary()
"""

from endgame.models.interpretable.corels import CORELSClassifier
from endgame.models.interpretable.gam import GAMClassifier, GAMRegressor
from endgame.models.interpretable.gami_net import GAMINetClassifier, GAMINetRegressor
from endgame.models.interpretable.gosdt import GOSDTClassifier
from endgame.models.interpretable.node_gam import NodeGAMClassifier, NodeGAMRegressor
from endgame.models.interpretable.slim import FasterRiskClassifier, SLIMClassifier

__all__ = [
    "CORELSClassifier",
    "NodeGAMClassifier",
    "NodeGAMRegressor",
    "GAMINetClassifier",
    "GAMINetRegressor",
    "SLIMClassifier",
    "FasterRiskClassifier",
    "GAMClassifier",
    "GAMRegressor",
    "GOSDTClassifier",
]
