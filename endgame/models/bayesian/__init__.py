from __future__ import annotations

"""Bayesian Network Classifiers for Endgame.

This module provides a unified API for Bayesian Network Classification:

Classic Solvers (CPU-optimized):
- TANClassifier: Tree Augmented Naive Bayes
- EBMCClassifier: Efficient Bayesian Multivariate Classifier
- ESKDBClassifier: Ensemble of Selective K-Dependence Bayes
- KDBClassifier: K-Dependence Bayes (base for ESKDB)

Neural Solvers (GPU-accelerated):
- NeuralKDBClassifier: Neural conditional estimators

Structure Learning:
- AutoSLE: Scalable structure learning ensemble

All classifiers implement the sklearn interface (fit, predict, predict_proba).

Examples
--------
>>> from endgame.models.bayesian import TANClassifier, BayesianDiscretizer
>>>
>>> # Discretize continuous features
>>> disc = BayesianDiscretizer(strategy='mdlp')
>>> X_disc = disc.fit_transform(X_train, y_train)
>>>
>>> # Train TAN classifier
>>> clf = TANClassifier(smoothing=1.0)
>>> clf.fit(X_disc, y_train)
>>>
>>> # Predict
>>> proba = clf.predict_proba(disc.transform(X_test))
"""

# Base classes and exceptions
from endgame.models.bayesian.base import (
    BaseBayesianClassifier,
    BayesianClassifierMixin,
    BayesianNetworkError,
    BayesianSerializationMixin,
    CardinalityError,
    ConvergenceWarning,
    CyclicGraphError,
    MissingValueStrategy,
    NotFittedError,
    StructureLearningError,
)

# Classic classifiers (always available)
from endgame.models.bayesian.classic import (
    EBMCClassifier,
    ESKDBClassifier,
    KDBClassifier,
    TANClassifier,
)

# Structure learning
from endgame.models.bayesian.structure import (
    bdeu_score,
    bic_score,
    build_kdb_structure,
    chow_liu_tree,
    compute_cmi_matrix,
    compute_conditional_mutual_information,
    compute_mutual_information,
    compute_structure_score,
    greedy_hill_climbing,
    k2_score,
)
from endgame.models.bayesian.structure.autosle import AutoSLE

__all__ = [
    # Base classes
    "BayesianClassifierMixin",
    "BayesianSerializationMixin",
    "BaseBayesianClassifier",
    # Exceptions
    "BayesianNetworkError",
    "StructureLearningError",
    "CyclicGraphError",
    "CardinalityError",
    "ConvergenceWarning",
    "NotFittedError",
    "MissingValueStrategy",
    # Classic classifiers
    "TANClassifier",
    "EBMCClassifier",
    "ESKDBClassifier",
    "KDBClassifier",
    # Structure learning
    "AutoSLE",
    "bdeu_score",
    "bic_score",
    "k2_score",
    "compute_structure_score",
    "compute_mutual_information",
    "compute_conditional_mutual_information",
    "compute_cmi_matrix",
    "chow_liu_tree",
    "build_kdb_structure",
    "greedy_hill_climbing",
]

# Neural classifiers require PyTorch
try:
    from endgame.models.bayesian.neural import (
        ConditionalEmbeddingNet,
        NeuralKDBClassifier,
    )
    __all__.extend([
        "NeuralKDBClassifier",
        "ConditionalEmbeddingNet",
    ])
except ImportError:
    # PyTorch not available
    pass
