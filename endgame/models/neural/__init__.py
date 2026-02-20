"""Neural network models for tabular data.

This module provides PyTorch-based neural network implementations optimized
for tabular data, including:

- MLPClassifier/MLPRegressor: Standard multi-layer perceptrons with modern techniques
- TabNetClassifier/TabNetRegressor: Attention-based architecture with interpretability
- EmbeddingMLPClassifier/EmbeddingMLPRegressor: MLPs with entity embeddings for categoricals

All models follow the sklearn interface and support GPU acceleration.

Note: These models require PyTorch. Install with: pip install torch
"""

__all__ = []

# MLP requires PyTorch
try:
    from endgame.models.neural.mlp import MLPClassifier, MLPRegressor
    __all__.extend(["MLPClassifier", "MLPRegressor"])
except (ImportError, NameError):
    pass

# Embedding MLP requires PyTorch
try:
    from endgame.models.neural.embedding_mlp import (
        EmbeddingMLPClassifier,
        EmbeddingMLPRegressor,
    )
    __all__.extend(["EmbeddingMLPClassifier", "EmbeddingMLPRegressor"])
except (ImportError, NameError):
    pass

# TabNet requires optional dependency
try:
    from endgame.models.neural.tabnet import TabNetClassifier, TabNetRegressor
    __all__.extend(["TabNetClassifier", "TabNetRegressor"])
except (ImportError, NameError):
    pass
