"""Dimensionality Reduction Module.

This module provides unified interfaces for various dimensionality reduction
techniques, from classical linear methods to modern manifold learning.

Classes
-------
Linear Methods:
    PCAReducer : Principal Component Analysis
    RandomizedPCA : Randomized PCA for large datasets
    TruncatedSVDReducer : Truncated SVD (LSA)
    KernelPCAReducer : Kernel PCA for nonlinear projections
    ICAReducer : Independent Component Analysis

Manifold Learning:
    UMAPReducer : Uniform Manifold Approximation and Projection
    ParametricUMAP : Neural network-based UMAP
    TriMAPReducer : TriMAP for preserving global structure
    PHATEReducer : PHATE for trajectory visualization
    PaCMAPReducer : Pairwise Controlled Manifold Approximation

Deep Learning:
    VAEReducer : Variational Autoencoder for dimensionality reduction

Example
-------
>>> from endgame.dimensionality_reduction import UMAPReducer, PCAReducer
>>> # UMAP for visualization
>>> umap = UMAPReducer(n_components=2)
>>> X_2d = umap.fit_transform(X)
>>> # PCA for preprocessing
>>> pca = PCAReducer(n_components=50)
>>> X_reduced = pca.fit_transform(X_train)
>>> X_test_reduced = pca.transform(X_test)
"""

from endgame.dimensionality_reduction.linear import (
    ICAReducer,
    KernelPCAReducer,
    PCAReducer,
    RandomizedPCA,
    TruncatedSVDReducer,
)
from endgame.dimensionality_reduction.manifold import (
    PaCMAPReducer,
    ParametricUMAP,
    PHATEReducer,
    TriMAPReducer,
    UMAPReducer,
)
from endgame.dimensionality_reduction.vae import VAEReducer

__all__ = [
    # Linear
    "PCAReducer",
    "RandomizedPCA",
    "TruncatedSVDReducer",
    "KernelPCAReducer",
    "ICAReducer",
    # Manifold
    "UMAPReducer",
    "ParametricUMAP",
    "TriMAPReducer",
    "PHATEReducer",
    "PaCMAPReducer",
    # Deep Learning
    "VAEReducer",
]
