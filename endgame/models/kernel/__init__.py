from __future__ import annotations

"""Kernel-based models for ensemble diversity.

This module provides kernel method wrappers with competition-tuned defaults:
- Gaussian Process (Bayesian kernel method with uncertainty)
- Support Vector Machines (max-margin kernel method)

These models have fundamentally different inductive biases from tree-based
and neural network models, making them valuable for ensemble diversity.
"""

from endgame.models.kernel.gaussian_process import (
    GPClassifier,
    GPRegressor,
)
from endgame.models.kernel.svm import (
    SVMClassifier,
    SVMRegressor,
)

__all__ = [
    "GPClassifier",
    "GPRegressor",
    "SVMClassifier",
    "SVMRegressor",
]
