from __future__ import annotations

"""Manifold Learning Dimensionality Reduction Methods.

This module provides sklearn-compatible wrappers for modern manifold learning
techniques including UMAP, TriMAP, PHATE, and PaCMAP.

Classes
-------
UMAPReducer : Uniform Manifold Approximation and Projection
ParametricUMAP : Neural network-based UMAP with transform capability
TriMAPReducer : TriMAP for global + local structure preservation
PHATEReducer : PHATE for trajectory/progression visualization
PaCMAPReducer : Pairwise Controlled Manifold Approximation
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Check for optional dependencies
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    umap = None

try:
    import trimap
    HAS_TRIMAP = True
except ImportError:
    HAS_TRIMAP = False
    trimap = None

try:
    import phate
    HAS_PHATE = True
except ImportError:
    HAS_PHATE = False
    phate = None

try:
    import pacmap
    HAS_PACMAP = True
except ImportError:
    HAS_PACMAP = False
    pacmap = None


def _check_umap_installed():
    """Raise ImportError if umap-learn is not installed."""
    if not HAS_UMAP:
        raise ImportError(
            "The 'umap-learn' package is required. "
            "Install it with: pip install umap-learn"
        )


def _check_trimap_installed():
    """Raise ImportError if trimap is not installed."""
    if not HAS_TRIMAP:
        raise ImportError(
            "The 'trimap' package is required. "
            "Install it with: pip install trimap"
        )


def _check_phate_installed():
    """Raise ImportError if phate is not installed."""
    if not HAS_PHATE:
        raise ImportError(
            "The 'phate' package is required. "
            "Install it with: pip install phate"
        )


def _check_pacmap_installed():
    """Raise ImportError if pacmap is not installed."""
    if not HAS_PACMAP:
        raise ImportError(
            "The 'pacmap' package is required. "
            "Install it with: pip install pacmap"
        )


class UMAPReducer(TransformerMixin, BaseEstimator):
    """Uniform Manifold Approximation and Projection (UMAP).

    UMAP is a manifold learning technique that preserves both local and
    global structure better than t-SNE while being significantly faster.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    n_neighbors : int, default=15
        Number of neighbors for constructing the local manifold.
        Larger values capture more global structure.

    min_dist : float, default=0.1
        Minimum distance between embedded points.
        Smaller values create tighter clusters.

    metric : str, default='euclidean'
        Distance metric: 'euclidean', 'manhattan', 'cosine', 'correlation', etc.

    spread : float, default=1.0
        Effective scale of embedded points.

    learning_rate : float, default=1.0
        Learning rate for the embedding optimization.

    n_epochs : int, optional
        Number of training epochs. If None, auto-determined.

    init : str, default='spectral'
        Initialization: 'spectral', 'random', or array.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Embedding of the training data.

    Example
    -------
    >>> from endgame.dimensionality_reduction import UMAPReducer
    >>> umap = UMAPReducer(n_components=2, n_neighbors=30)
    >>> X_2d = umap.fit_transform(X)
    >>> # For new data (uses approximate transform)
    >>> X_new_2d = umap.transform(X_new)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        spread: float = 1.0,
        learning_rate: float = 1.0,
        n_epochs: int | None = None,
        init: str = "spectral",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.spread = spread
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.init = init
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the UMAP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target labels for semi-supervised mode.

        Returns
        -------
        self : UMAPReducer
        """
        _check_umap_installed()
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            spread=self.spread,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._umap.fit(X, y)
        self.embedding_ = self._umap.embedding_

        return self

    def transform(self, X) -> np.ndarray:
        """Transform new data to the embedding space.

        Uses the learned transform to embed new points. Note that this
        is an approximation based on the nearest neighbors in the
        training set.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, "_umap")
        X = check_array(X)
        return self._umap.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step.

        This is more efficient than calling fit then transform.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like, optional
            Target labels for semi-supervised mode.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Embedding of training data.
        """
        _check_umap_installed()
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._umap = umap.UMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            spread=self.spread,
            learning_rate=self.learning_rate,
            n_epochs=self.n_epochs,
            init=self.init,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.embedding_ = self._umap.fit_transform(X, y)
        return self.embedding_

    def inverse_transform(self, X) -> np.ndarray:
        """Transform from embedding space back to data space.

        Note: This is an approximation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in embedding space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Approximate reconstruction.
        """
        check_is_fitted(self, "_umap")
        return self._umap.inverse_transform(X)


class ParametricUMAP(TransformerMixin, BaseEstimator):
    """Parametric UMAP using neural networks.

    Unlike standard UMAP, Parametric UMAP learns an explicit mapping
    function using a neural network, enabling faster transforms on
    new data and the ability to train on mini-batches.

    Requires TensorFlow/Keras to be installed.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    n_neighbors : int, default=15
        Number of neighbors for constructing the local manifold.

    min_dist : float, default=0.1
        Minimum distance between embedded points.

    metric : str, default='euclidean'
        Distance metric.

    encoder_layers : list of int, optional
        Sizes of encoder hidden layers. Default [256, 256].

    decoder_layers : list of int, optional
        Sizes of decoder hidden layers for reconstruction. Default [256, 256].

    n_training_epochs : int, default=100
        Number of training epochs for the neural network.

    batch_size : int, default=256
        Mini-batch size for training.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Whether to print progress.

    Example
    -------
    >>> from endgame.dimensionality_reduction import ParametricUMAP
    >>> pumap = ParametricUMAP(n_components=2)
    >>> pumap.fit(X_train)
    >>> # Fast transform on new data
    >>> X_new_2d = pumap.transform(X_test)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
        encoder_layers: list | None = None,
        decoder_layers: list | None = None,
        n_training_epochs: int = 100,
        batch_size: int = 256,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.metric = metric
        self.encoder_layers = encoder_layers or [256, 256]
        self.decoder_layers = decoder_layers or [256, 256]
        self.n_training_epochs = n_training_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def _build_encoder(self, input_dim: int):
        """Build the encoder network."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is required for ParametricUMAP. "
                "Install with: pip install tensorflow"
            )

        layers = [keras.layers.InputLayer(input_shape=(input_dim,))]

        for units in self.encoder_layers:
            layers.append(keras.layers.Dense(units, activation="relu"))
            layers.append(keras.layers.BatchNormalization())

        layers.append(keras.layers.Dense(self.n_components))

        return keras.Sequential(layers)

    def _build_decoder(self, output_dim: int):
        """Build the decoder network."""
        try:
            from tensorflow import keras
        except ImportError:
            raise ImportError(
                "TensorFlow is required for ParametricUMAP. "
                "Install with: pip install tensorflow"
            )

        layers = [keras.layers.InputLayer(input_shape=(self.n_components,))]

        for units in self.decoder_layers:
            layers.append(keras.layers.Dense(units, activation="relu"))
            layers.append(keras.layers.BatchNormalization())

        layers.append(keras.layers.Dense(output_dim))

        return keras.Sequential(layers)

    def fit(self, X, y=None):
        """Fit the Parametric UMAP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : ParametricUMAP
        """
        _check_umap_installed()

        try:
            from umap.parametric_umap import ParametricUMAP as _ParametricUMAP
        except ImportError:
            raise ImportError(
                "Parametric UMAP requires TensorFlow. "
                "Install with: pip install tensorflow"
            )

        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Build encoder architecture
        encoder = self._build_encoder(self.n_features_in_)
        decoder = self._build_decoder(self.n_features_in_)

        self._pumap = _ParametricUMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            min_dist=self.min_dist,
            metric=self.metric,
            encoder=encoder,
            decoder=decoder,
            n_training_epochs=self.n_training_epochs,
            batch_size=self.batch_size,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._pumap.fit(X)
        self.embedding_ = self._pumap.embedding_

        return self

    def transform(self, X) -> np.ndarray:
        """Transform new data using the learned encoder.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, "_pumap")
        X = check_array(X)
        return self._pumap.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.embedding_

    def inverse_transform(self, X) -> np.ndarray:
        """Transform from embedding space back to data space.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_components)
            Data in embedding space.

        Returns
        -------
        X_original : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, "_pumap")
        return self._pumap.inverse_transform(X)


class TriMAPReducer(TransformerMixin, BaseEstimator):
    """TriMAP: Dimensionality Reduction Using Triplet Constraints.

    TriMAP uses triplet constraints to capture both local and global
    structure better than t-SNE or UMAP, particularly for hierarchical
    data structures.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    n_inliers : int, default=12
        Number of nearest neighbor inliers per point.

    n_outliers : int, default=4
        Number of random outliers per point.

    n_random : int, default=3
        Number of random triplets per point.

    weight_adj : float, optional
        Weight adjustment factor for triplet loss.

    n_iters : int, default=400
        Number of optimization iterations.

    apply_pca : bool, default=True
        Whether to apply PCA for initialization.

    verbose : bool, default=False
        Whether to print progress.

    Example
    -------
    >>> from endgame.dimensionality_reduction import TriMAPReducer
    >>> trimap = TriMAPReducer(n_components=2, n_inliers=15)
    >>> X_2d = trimap.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_inliers: int = 12,
        n_outliers: int = 4,
        n_random: int = 3,
        weight_adj: float | None = None,
        n_iters: int = 400,
        apply_pca: bool = True,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.n_inliers = n_inliers
        self.n_outliers = n_outliers
        self.n_random = n_random
        self.weight_adj = weight_adj
        self.n_iters = n_iters
        self.apply_pca = apply_pca
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the TriMAP model.

        Note: TriMAP is a transductive method, so fit stores the
        embedding but doesn't create a general transform function.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : TriMAPReducer
        """
        _check_trimap_installed()
        X = check_array(X, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        kwargs = {
            "n_dims": self.n_components,
            "n_inliers": self.n_inliers,
            "n_outliers": self.n_outliers,
            "n_random": self.n_random,
            "n_iters": self.n_iters,
            "apply_pca": self.apply_pca,
            "verbose": self.verbose,
        }
        if self.weight_adj is not None:
            kwargs["weight_adj"] = self.weight_adj

        self.embedding_ = trimap.TRIMAP(**kwargs).fit_transform(X)

        self._X_fit = X.copy()
        return self

    def transform(self, X) -> np.ndarray:
        """Transform new data.

        Note: TriMAP doesn't have native out-of-sample support.
        This uses a nearest-neighbor approximation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Approximate embedding.
        """
        check_is_fitted(self, "embedding_")
        X = check_array(X, dtype=np.float64)

        # Use nearest neighbor interpolation
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=5)
        nn.fit(self._X_fit)
        distances, indices = nn.kneighbors(X)

        # Weight by inverse distance
        weights = 1.0 / (distances + 1e-10)
        weights = weights / weights.sum(axis=1, keepdims=True)

        # Weighted average of neighbor embeddings
        X_transformed = np.zeros((X.shape[0], self.n_components))
        for i in range(X.shape[0]):
            X_transformed[i] = np.average(
                self.embedding_[indices[i]], weights=weights[i], axis=0
            )

        return X_transformed

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return the embedding."""
        self.fit(X)
        return self.embedding_


class PHATEReducer(TransformerMixin, BaseEstimator):
    """PHATE: Potential of Heat-diffusion for Affinity-based Transition Embedding.

    PHATE is designed for visualizing trajectories and progressions in
    high-dimensional biological data. It preserves both local and global
    structures through diffusion-based distances.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    knn : int, default=5
        Number of nearest neighbors for graph construction.

    decay : int, default=40
        Decay rate of the kernel tails.

    t : int or 'auto', default='auto'
        Power of the diffusion operator.

    gamma : float, default=1.0
        Informational distance constant between -1 and 1.

    n_pca : int, default=100
        Number of principal components for initial reduction.

    knn_dist : str, default='euclidean'
        Distance metric for KNN graph.

    mds_solver : str, default='sgd'
        MDS solver: 'sgd' or 'smacof'.

    random_state : int, optional
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Example
    -------
    >>> from endgame.dimensionality_reduction import PHATEReducer
    >>> phate = PHATEReducer(n_components=2, knn=10)
    >>> X_2d = phate.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int = 2,
        knn: int = 5,
        decay: int = 40,
        t: int | Literal["auto"] = "auto",
        gamma: float = 1.0,
        n_pca: int = 100,
        knn_dist: str = "euclidean",
        mds_solver: str = "sgd",
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.n_components = n_components
        self.knn = knn
        self.decay = decay
        self.t = t
        self.gamma = gamma
        self.n_pca = n_pca
        self.knn_dist = knn_dist
        self.mds_solver = mds_solver
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y=None):
        """Fit the PHATE model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : PHATEReducer
        """
        _check_phate_installed()
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._phate = phate.PHATE(
            n_components=self.n_components,
            knn=self.knn,
            decay=self.decay,
            t=self.t,
            gamma=self.gamma,
            n_pca=self.n_pca,
            knn_dist=self.knn_dist,
            mds_solver=self.mds_solver,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self._phate.fit(X)
        self.embedding_ = self._phate.embedding

        return self

    def transform(self, X) -> np.ndarray:
        """Transform new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, "_phate")
        X = check_array(X)
        return self._phate.transform(X)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return the embedding."""
        _check_phate_installed()
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        self._phate = phate.PHATE(
            n_components=self.n_components,
            knn=self.knn,
            decay=self.decay,
            t=self.t,
            gamma=self.gamma,
            n_pca=self.n_pca,
            knn_dist=self.knn_dist,
            mds_solver=self.mds_solver,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.embedding_ = self._phate.fit_transform(X)
        return self.embedding_


class PaCMAPReducer(TransformerMixin, BaseEstimator):
    """PaCMAP: Pairwise Controlled Manifold Approximation.

    PaCMAP preserves both local and global structure by considering
    pairs (neighbors), mid-near pairs, and far pairs during optimization.
    It's faster than t-SNE and UMAP with competitive quality.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the embedded space.

    n_neighbors : int, default=10
        Number of neighbors for local structure.

    MN_ratio : float, default=0.5
        Ratio of mid-near pairs to neighbor pairs.

    FP_ratio : float, default=2.0
        Ratio of further pairs to neighbor pairs.

    num_iters : int, default=450
        Number of iterations for optimization.

    lr : float, default=1.0
        Learning rate.

    apply_pca : bool, default=True
        Whether to apply PCA for initialization.

    verbose : bool, default=False
        Whether to print progress.

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> from endgame.dimensionality_reduction import PaCMAPReducer
    >>> pacmap = PaCMAPReducer(n_components=2, n_neighbors=15)
    >>> X_2d = pacmap.fit_transform(X)
    """

    def __init__(
        self,
        n_components: int = 2,
        n_neighbors: int = 10,
        MN_ratio: float = 0.5,
        FP_ratio: float = 2.0,
        num_iters: int = 450,
        lr: float = 1.0,
        apply_pca: bool = True,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.MN_ratio = MN_ratio
        self.FP_ratio = FP_ratio
        self.num_iters = num_iters
        self.lr = lr
        self.apply_pca = apply_pca
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the PaCMAP model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : PaCMAPReducer
        """
        _check_pacmap_installed()
        X = check_array(X)
        self.n_features_in_ = X.shape[1]

        # Store training data for transform (PaCMAP needs it)
        self._X_fit = X.copy()

        self._pacmap = pacmap.PaCMAP(
            n_components=self.n_components,
            n_neighbors=self.n_neighbors,
            MN_ratio=self.MN_ratio,
            FP_ratio=self.FP_ratio,
            num_iters=self.num_iters,
            lr=self.lr,
            apply_pca=self.apply_pca,
            verbose=self.verbose,
            random_state=self.random_state,
        )
        self.embedding_ = self._pacmap.fit_transform(X)

        return self

    def transform(self, X) -> np.ndarray:
        """Transform new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_new : ndarray of shape (n_samples, n_components)
            Transformed data.
        """
        check_is_fitted(self, "_pacmap")
        X = check_array(X)
        # PaCMAP requires the original basis data for transform
        return self._pacmap.transform(X, basis=self._X_fit)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and return the embedding."""
        self.fit(X)
        return self.embedding_
