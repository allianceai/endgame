from __future__ import annotations

"""Neural Bayesian Network Classifiers using learned embeddings."""

# Neural components require PyTorch
try:
    from endgame.models.bayesian.neural.embeddings import ConditionalEmbeddingNet
    from endgame.models.bayesian.neural.neural_kdb import NeuralKDBClassifier

    __all__ = [
        "ConditionalEmbeddingNet",
        "NeuralKDBClassifier",
    ]
except ImportError:
    # PyTorch not installed
    __all__ = []
