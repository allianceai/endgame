from __future__ import annotations

"""ROCKET, MiniROCKET, and HYDRA time series transformers and classifiers.

This module provides sklearn-compatible wrappers for the ROCKET family of
time series classification methods:

- ROCKET: Random Convolutional Kernel Transform (Dempster et al., 2020)
- MiniROCKET: Faster, almost deterministic variant (Dempster et al., 2021)
- MultiROCKET: Multivariate extension with additional pooling
- HYDRA: Hybrid Dictionary-ROCKET approach (Dempster et al., 2023)

These methods transform time series into features using random/fixed
convolutional kernels, then use a linear classifier (typically Ridge).

Installation
------------
For native implementations:
    pip install rocket-learn  # Original ROCKET/MiniROCKET
    pip install hydra-ts      # HYDRA

For sktime wrappers (recommended):
    pip install sktime

Examples
--------
>>> from endgame.timeseries import RocketClassifier, MiniRocketClassifier
>>> model = MiniRocketClassifier()
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)

References
----------
- Dempster, A., et al. (2020). "ROCKET: Exceptionally fast and accurate time
  series classification using random convolutional kernels." Data Mining and
  Knowledge Discovery, 34(5), 1454-1495.
- Dempster, A., et al. (2021). "MINIROCKET: A very fast (almost) deterministic
  transform for time series classification." KDD 2021.
- Dempster, A., et al. (2023). "HYDRA: Competing convolutional kernels for fast
  and accurate time series classification." Data Mining and Knowledge Discovery.
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

# Check for different backends
try:
    from sktime.transformations.panel.rocket import MiniRocket as SktimeMiniRocket
    from sktime.transformations.panel.rocket import MiniRocketMultivariate as SktimeMiniRocketMV
    from sktime.transformations.panel.rocket import Rocket as SktimeRocket
    HAS_SKTIME_ROCKET = True
except ImportError:
    HAS_SKTIME_ROCKET = False

try:
    from sktime.transformations.panel.rocket import MultiRocket as SktimeMultiRocket
    HAS_SKTIME_MULTIROCKET = True
except ImportError:
    HAS_SKTIME_MULTIROCKET = False

try:
    from sktime.transformations.panel.hydra import Hydra as SktimeHydra
    HAS_SKTIME_HYDRA = True
except ImportError:
    HAS_SKTIME_HYDRA = False

# Check for native implementations
try:
    from rocket_functions import apply_kernels, generate_kernels
    HAS_NATIVE_ROCKET = True
except ImportError:
    HAS_NATIVE_ROCKET = False

try:
    from minirocket import fit as minirocket_fit
    from minirocket import transform as minirocket_transform
    HAS_NATIVE_MINIROCKET = True
except ImportError:
    HAS_NATIVE_MINIROCKET = False

try:
    from hydra import Hydra as NativeHydra
    HAS_NATIVE_HYDRA = True
except ImportError:
    HAS_NATIVE_HYDRA = False


def _check_rocket_available():
    """Check if any ROCKET implementation is available."""
    if not (HAS_SKTIME_ROCKET or HAS_NATIVE_ROCKET):
        raise ImportError(
            "ROCKET requires either sktime or the native rocket implementation. "
            "Install with: pip install sktime  OR  pip install rocket-learn"
        )


def _check_minirocket_available():
    """Check if any MiniROCKET implementation is available."""
    if not (HAS_SKTIME_ROCKET or HAS_NATIVE_MINIROCKET):
        raise ImportError(
            "MiniROCKET requires either sktime or the native minirocket implementation. "
            "Install with: pip install sktime  OR  pip install rocket-learn"
        )


def _check_hydra_available():
    """Check if any HYDRA implementation is available."""
    if not (HAS_SKTIME_HYDRA or HAS_NATIVE_HYDRA):
        raise ImportError(
            "HYDRA requires either sktime or the native hydra implementation. "
            "Install with: pip install sktime  OR  pip install hydra-ts"
        )


def _to_sktime_format(X: np.ndarray) -> np.ndarray:
    """Convert array to sktime's expected 3D format.

    sktime expects shape (n_instances, n_channels, n_timepoints) or
    (n_instances, n_timepoints) for univariate.
    """
    X = np.asarray(X)

    if X.ndim == 1:
        # Single time series
        X = X.reshape(1, 1, -1)
    elif X.ndim == 2:
        # (n_instances, n_timepoints) -> (n_instances, 1, n_timepoints)
        X = X.reshape(X.shape[0], 1, X.shape[1])
    elif X.ndim == 3:
        # Already in correct format
        pass
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {X.shape}")

    return X.astype(np.float32)


class RocketTransformer(BaseEstimator, TransformerMixin):
    """ROCKET (Random Convolutional Kernel Transform) for time series.

    Transforms time series using random convolutional kernels with random
    length, weights, bias, dilation, and padding. Extracts max value and
    proportion of positive values (PPV) from each convolution.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of random kernels to generate.
    normalize : bool, default=True
        Whether to normalize input time series.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_out_ : int
        Number of output features (2 * n_kernels).

    Examples
    --------
    >>> transformer = RocketTransformer(n_kernels=10000)
    >>> X_transformed = transformer.fit_transform(X_train)
    >>> print(X_transformed.shape)  # (n_samples, 20000)

    Notes
    -----
    ROCKET generates 2 features per kernel:
    - Maximum value of the convolution
    - Proportion of positive values (PPV)

    For 10,000 kernels (default), this produces 20,000 features.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        normalize: bool = True,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.normalize = normalize
        self.random_state = random_state

        self._transformer = None
        self._backend = None
        self.n_features_out_ = 2 * n_kernels

    def fit(self, X, y=None):
        """Fit the ROCKET transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Training time series.
        y : ignored

        Returns
        -------
        self
        """
        _check_rocket_available()

        X = _to_sktime_format(X)

        if HAS_SKTIME_ROCKET:
            self._backend = "sktime"
            self._transformer = SktimeRocket(
                num_kernels=self.n_kernels,
                normalise=self.normalize,
                random_state=self.random_state,
            )
            self._transformer.fit(X)
        elif HAS_NATIVE_ROCKET:
            self._backend = "native"
            n_timepoints = X.shape[2]
            self._kernels = generate_kernels(n_timepoints, self.n_kernels)

        return self

    def transform(self, X) -> np.ndarray:
        """Transform time series using ROCKET kernels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints) or (n_samples, n_channels, n_timepoints)
            Time series to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Transformed features.
        """
        X = _to_sktime_format(X)

        if self._backend == "sktime":
            return self._transformer.transform(X)
        elif self._backend == "native":
            # Normalize if requested
            if self.normalize:
                X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-8)
            # Apply kernels (expects 2D: n_samples x n_timepoints)
            X_2d = X.squeeze(axis=1) if X.shape[1] == 1 else X.reshape(X.shape[0], -1)
            return apply_kernels(X_2d, self._kernels)
        else:
            raise RuntimeError("Transformer not fitted")


class MiniRocketTransformer(BaseEstimator, TransformerMixin):
    """MiniROCKET transform for time series.

    A faster, almost deterministic variant of ROCKET using fixed kernels
    with limited weight values. Up to 75x faster than ROCKET on large datasets.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels (approximately - actual number depends on implementation).
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_out_ : int
        Number of output features.

    Examples
    --------
    >>> transformer = MiniRocketTransformer()
    >>> X_transformed = transformer.fit_transform(X_train)

    Notes
    -----
    Key differences from ROCKET:
    - Uses fixed kernels of length 9
    - Weights restricted to two values (-1 and 2)
    - Uses 84 fixed convolutions as seeds for dilations
    - Does NOT require normalized input
    - Almost deterministic (small random component in dilation selection)

    Minimum time series length is 9.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.random_state = random_state

        self._transformer = None
        self._backend = None
        self.n_features_out_ = None

    def fit(self, X, y=None):
        """Fit the MiniROCKET transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints)
            Training time series. Must have length >= 9.
        y : ignored

        Returns
        -------
        self
        """
        _check_minirocket_available()

        X = _to_sktime_format(X)
        n_timepoints = X.shape[2]

        if n_timepoints < 9:
            raise ValueError(
                f"MiniROCKET requires time series of length >= 9, got {n_timepoints}"
            )

        if HAS_SKTIME_ROCKET:
            self._backend = "sktime"
            # Use multivariate version if needed
            if X.shape[1] > 1:
                self._transformer = SktimeMiniRocketMV(
                    num_kernels=self.n_kernels,
                    random_state=self.random_state,
                )
            else:
                self._transformer = SktimeMiniRocket(
                    num_kernels=self.n_kernels,
                    random_state=self.random_state,
                )
            self._transformer.fit(X)
            # Get actual number of features
            sample_transform = self._transformer.transform(X[:1])
            self.n_features_out_ = sample_transform.shape[1]
        elif HAS_NATIVE_MINIROCKET:
            self._backend = "native"
            X_2d = X.squeeze(axis=1).astype(np.float32)
            self._parameters = minirocket_fit(X_2d)
            self.n_features_out_ = 9996  # Default for MiniROCKET

        return self

    def transform(self, X) -> np.ndarray:
        """Transform time series using MiniROCKET.

        Parameters
        ----------
        X : array-like
            Time series to transform.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Transformed features.
        """
        X = _to_sktime_format(X)

        if self._backend == "sktime":
            return self._transformer.transform(X)
        elif self._backend == "native":
            X_2d = X.squeeze(axis=1).astype(np.float32)
            return minirocket_transform(X_2d, self._parameters)
        else:
            raise RuntimeError("Transformer not fitted")


class MultiRocketTransformer(BaseEstimator, TransformerMixin):
    """MultiROCKET transform for time series.

    Extension of MiniROCKET for multivariate time series with additional
    pooling operations (mean, standard deviation).

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels.
    n_features_per_kernel : int, default=4
        Number of features per kernel (PPV + additional pooling).
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> transformer = MultiRocketTransformer()
    >>> X_transformed = transformer.fit_transform(X_train)  # Multivariate OK
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        n_features_per_kernel: int = 4,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.n_features_per_kernel = n_features_per_kernel
        self.random_state = random_state

        self._transformer = None
        self.n_features_out_ = None

    def fit(self, X, y=None):
        """Fit MultiROCKET transformer."""
        if not HAS_SKTIME_MULTIROCKET:
            raise ImportError(
                "MultiROCKET requires sktime with MultiRocket support. "
                "Install with: pip install sktime"
            )

        X = _to_sktime_format(X)

        self._transformer = SktimeMultiRocket(
            num_kernels=self.n_kernels,
            n_features_per_kernel=self.n_features_per_kernel,
            random_state=self.random_state,
        )
        self._transformer.fit(X)

        sample_transform = self._transformer.transform(X[:1])
        self.n_features_out_ = sample_transform.shape[1]

        return self

    def transform(self, X) -> np.ndarray:
        """Transform time series using MultiROCKET."""
        X = _to_sktime_format(X)
        return self._transformer.transform(X)


class HydraTransformer(BaseEstimator, TransformerMixin):
    """HYDRA (Hybrid Dictionary-ROCKET) transform for time series.

    Combines dictionary methods with convolutional kernels. Uses competing
    kernels that extract and count symbolic patterns while leveraging
    kernel transformations.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of kernel groups.
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    n_features_out_ : int
        Number of output features.

    Examples
    --------
    >>> transformer = HydraTransformer(n_kernels=8, n_groups=64)
    >>> X_transformed = transformer.fit_transform(X_train)

    Notes
    -----
    HYDRA can be combined with ROCKET/MiniROCKET for improved accuracy.
    A single hyperparameter controls the trade-off between dictionary-like
    and ROCKET-like behavior.

    References
    ----------
    Dempster, A., et al. (2023). "HYDRA: Competing convolutional kernels for
    fast and accurate time series classification."
    """

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.random_state = random_state

        self._transformer = None
        self._backend = None
        self.n_features_out_ = None

    def fit(self, X, y=None):
        """Fit HYDRA transformer."""
        _check_hydra_available()

        X = _to_sktime_format(X)

        if HAS_SKTIME_HYDRA:
            self._backend = "sktime"
            self._transformer = SktimeHydra(
                n_kernels=self.n_kernels,
                n_groups=self.n_groups,
                random_state=self.random_state,
            )
            self._transformer.fit(X)
            sample_transform = self._transformer.transform(X[:1])
            self.n_features_out_ = sample_transform.shape[1]
        elif HAS_NATIVE_HYDRA:
            self._backend = "native"
            X_2d = X.squeeze(axis=1).astype(np.float32)
            self._hydra = NativeHydra(
                k=self.n_kernels,
                g=self.n_groups,
            )
            self._hydra.fit(X_2d)
            self.n_features_out_ = self.n_kernels * self.n_groups * 2

        return self

    def transform(self, X) -> np.ndarray:
        """Transform time series using HYDRA."""
        X = _to_sktime_format(X)

        if self._backend == "sktime":
            return self._transformer.transform(X)
        elif self._backend == "native":
            X_2d = X.squeeze(axis=1).astype(np.float32)
            return self._hydra.transform(X_2d)
        else:
            raise RuntimeError("Transformer not fitted")


class RocketClassifier(BaseEstimator, ClassifierMixin):
    """ROCKET classifier for time series classification.

    Combines ROCKET transform with RidgeClassifierCV for fast and accurate
    time series classification.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of random kernels.
    normalize : bool, default=True
        Whether to normalize input.
    alphas : array-like, optional
        Regularization values to try in RidgeClassifierCV.
        Default: np.logspace(-3, 3, 10).
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    transformer_ : RocketTransformer
        Fitted ROCKET transformer.
    classifier_ : RidgeClassifierCV
        Fitted classifier.

    Examples
    --------
    >>> clf = RocketClassifier(n_kernels=10000)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> accuracy = clf.score(X_test, y_test)

    Notes
    -----
    This achieves state-of-the-art accuracy on many UCR datasets while
    being orders of magnitude faster than deep learning approaches.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        normalize: bool = True,
        alphas: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.normalize = normalize
        self.alphas = alphas
        self.random_state = random_state

        self.classes_ = None
        self.transformer_ = None
        self.classifier_ = None
        self._scaler = None

    def fit(self, X, y):
        """Fit the ROCKET classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_timepoints)
            Training time series.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Create and fit transformer
        self.transformer_ = RocketTransformer(
            n_kernels=self.n_kernels,
            normalize=self.normalize,
            random_state=self.random_state,
        )
        X_transformed = self.transformer_.fit_transform(X)

        # Scale features
        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_transformed)

        # Train classifier
        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like
            Time series to classify.

        Returns
        -------
        np.ndarray
            Predicted labels.
        """
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Note: RidgeClassifier doesn't have native predict_proba.
        This uses decision function with softmax normalization.

        Parameters
        ----------
        X : array-like
            Time series to classify.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)

        # Use decision function
        decision = self.classifier_.decision_function(X_scaled)

        # Convert to probabilities via softmax
        if decision.ndim == 1:
            # Binary case
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            # Multiclass case
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


class MiniRocketClassifier(BaseEstimator, ClassifierMixin):
    """MiniROCKET classifier for time series classification.

    Faster variant of ROCKET using fixed kernels. Up to 75x faster
    on large datasets while maintaining similar accuracy.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels.
    alphas : array-like, optional
        Regularization values for RidgeClassifierCV.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> clf = MiniRocketClassifier()
    >>> clf.fit(X_train, y_train)
    >>> accuracy = clf.score(X_test, y_test)

    Notes
    -----
    Recommended as the default ROCKET variant due to speed and
    near-deterministic behavior.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        alphas: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.alphas = alphas
        self.random_state = random_state

        self.classes_ = None
        self.transformer_ = None
        self.classifier_ = None
        self._scaler = None

    def fit(self, X, y):
        """Fit the MiniROCKET classifier."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.transformer_ = MiniRocketTransformer(
            n_kernels=self.n_kernels,
            random_state=self.random_state,
        )
        X_transformed = self.transformer_.fit_transform(X)

        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_transformed)

        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)

        decision = self.classifier_.decision_function(X_scaled)

        if decision.ndim == 1:
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


class MultiRocketClassifier(BaseEstimator, ClassifierMixin):
    """MultiROCKET classifier for multivariate time series.

    Extension of MiniROCKET with additional pooling operations.

    Parameters
    ----------
    n_kernels : int, default=10000
        Number of kernels.
    n_features_per_kernel : int, default=4
        Features per kernel.
    alphas : array-like, optional
        Regularization values.
    random_state : int, optional
        Random seed.
    """

    def __init__(
        self,
        n_kernels: int = 10000,
        n_features_per_kernel: int = 4,
        alphas: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.n_features_per_kernel = n_features_per_kernel
        self.alphas = alphas
        self.random_state = random_state

        self.classes_ = None
        self.transformer_ = None
        self.classifier_ = None
        self._scaler = None

    def fit(self, X, y):
        """Fit the MultiROCKET classifier."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.transformer_ = MultiRocketTransformer(
            n_kernels=self.n_kernels,
            n_features_per_kernel=self.n_features_per_kernel,
            random_state=self.random_state,
        )
        X_transformed = self.transformer_.fit_transform(X)

        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_transformed)

        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)

        decision = self.classifier_.decision_function(X_scaled)

        if decision.ndim == 1:
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


class HydraClassifier(BaseEstimator, ClassifierMixin):
    """HYDRA classifier for time series classification.

    Combines dictionary methods with ROCKET using competing kernels.
    Often more accurate than ROCKET alone.

    Parameters
    ----------
    n_kernels : int, default=8
        Number of kernels per group.
    n_groups : int, default=64
        Number of kernel groups.
    alphas : array-like, optional
        Regularization values.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> clf = HydraClassifier()
    >>> clf.fit(X_train, y_train)
    >>> accuracy = clf.score(X_test, y_test)
    """

    def __init__(
        self,
        n_kernels: int = 8,
        n_groups: int = 64,
        alphas: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.n_kernels = n_kernels
        self.n_groups = n_groups
        self.alphas = alphas
        self.random_state = random_state

        self.classes_ = None
        self.transformer_ = None
        self.classifier_ = None
        self._scaler = None

    def fit(self, X, y):
        """Fit the HYDRA classifier."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        self.transformer_ = HydraTransformer(
            n_kernels=self.n_kernels,
            n_groups=self.n_groups,
            random_state=self.random_state,
        )
        X_transformed = self.transformer_.fit_transform(X)

        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_transformed)

        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        X_transformed = self.transformer_.transform(X)
        X_scaled = self._scaler.transform(X_transformed)

        decision = self.classifier_.decision_function(X_scaled)

        if decision.ndim == 1:
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)


class HydraMiniRocketClassifier(BaseEstimator, ClassifierMixin):
    """Combined HYDRA + MiniROCKET classifier.

    Concatenates features from both HYDRA and MiniROCKET for improved
    accuracy. This combination often achieves the best results.

    Parameters
    ----------
    hydra_kernels : int, default=8
        HYDRA kernels per group.
    hydra_groups : int, default=64
        HYDRA kernel groups.
    minirocket_kernels : int, default=10000
        MiniROCKET kernels.
    alphas : array-like, optional
        Regularization values.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> clf = HydraMiniRocketClassifier()
    >>> clf.fit(X_train, y_train)
    >>> accuracy = clf.score(X_test, y_test)

    Notes
    -----
    This combination was shown to significantly improve accuracy over
    either method alone in the HYDRA paper.
    """

    def __init__(
        self,
        hydra_kernels: int = 8,
        hydra_groups: int = 64,
        minirocket_kernels: int = 10000,
        alphas: np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.hydra_kernels = hydra_kernels
        self.hydra_groups = hydra_groups
        self.minirocket_kernels = minirocket_kernels
        self.alphas = alphas
        self.random_state = random_state

        self.classes_ = None
        self.hydra_transformer_ = None
        self.minirocket_transformer_ = None
        self.classifier_ = None
        self._scaler = None

    def fit(self, X, y):
        """Fit the combined classifier."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Fit HYDRA
        self.hydra_transformer_ = HydraTransformer(
            n_kernels=self.hydra_kernels,
            n_groups=self.hydra_groups,
            random_state=self.random_state,
        )
        X_hydra = self.hydra_transformer_.fit_transform(X)

        # Fit MiniROCKET
        self.minirocket_transformer_ = MiniRocketTransformer(
            n_kernels=self.minirocket_kernels,
            random_state=self.random_state,
        )
        X_minirocket = self.minirocket_transformer_.fit_transform(X)

        # Concatenate features
        X_combined = np.hstack([X_hydra, X_minirocket])

        self._scaler = StandardScaler(with_mean=False)
        X_scaled = self._scaler.fit_transform(X_combined)

        alphas = self.alphas if self.alphas is not None else np.logspace(-3, 3, 10)
        self.classifier_ = RidgeClassifierCV(alphas=alphas)
        self.classifier_.fit(X_scaled, y)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        X_hydra = self.hydra_transformer_.transform(X)
        X_minirocket = self.minirocket_transformer_.transform(X)
        X_combined = np.hstack([X_hydra, X_minirocket])
        X_scaled = self._scaler.transform(X_combined)
        return self.classifier_.predict(X_scaled)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        X_hydra = self.hydra_transformer_.transform(X)
        X_minirocket = self.minirocket_transformer_.transform(X)
        X_combined = np.hstack([X_hydra, X_minirocket])
        X_scaled = self._scaler.transform(X_combined)

        decision = self.classifier_.decision_function(X_scaled)

        if decision.ndim == 1:
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        else:
            exp_decision = np.exp(decision - decision.max(axis=1, keepdims=True))
            return exp_decision / exp_decision.sum(axis=1, keepdims=True)

    def score(self, X, y) -> float:
        """Return accuracy score."""
        return np.mean(self.predict(X) == y)
