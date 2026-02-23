"""Neuroevolution models: NEAT and TensorNEAT sklearn-compatible classifiers and regressors."""

from endgame.models.neuroevolution.neat import NEATClassifier, NEATRegressor

__all__ = ["NEATClassifier", "NEATRegressor"]

try:
    from endgame.models.neuroevolution.tensor_neat import (
        TensorNEATClassifier,
        TensorNEATRegressor,
    )
    __all__.extend(["TensorNEATClassifier", "TensorNEATRegressor"])
except ImportError:
    pass
