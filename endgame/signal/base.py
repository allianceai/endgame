from __future__ import annotations

"""Base classes for signal processing transformers.

Provides sklearn-compatible base classes that handle:
- Flexible input formats (1D, 2D, 3D arrays)
- Sample rate tracking
- Channel-wise operations
- Integration with time series module
"""

from abc import abstractmethod
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class SignalMixin:
    """Mixin providing common signal processing functionality.

    Handles input validation, reshaping, and sample rate management
    for all signal processing transformers.
    """

    _estimator_type = "transformer"

    def _validate_signal(
        self,
        X: Any,
        ensure_2d: bool = True,
    ) -> np.ndarray:
        """Validate and convert input signal.

        Parameters
        ----------
        X : array-like
            Input signal. Can be:
            - 1D: (n_samples,) single channel
            - 2D: (n_samples, n_channels) or (n_trials, n_samples)
            - 3D: (n_trials, n_channels, n_samples)
        ensure_2d : bool, default=True
            If True, ensures output is at least 2D.

        Returns
        -------
        np.ndarray
            Validated signal array.
        """
        X = np.asarray(X, dtype=np.float64)

        if X.ndim == 1:
            if ensure_2d:
                X = X.reshape(1, -1)  # (1, n_samples)
        elif X.ndim == 2:
            pass  # Already 2D
        elif X.ndim == 3:
            pass  # 3D is fine
        else:
            raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {X.shape}")

        return X

    def _get_n_samples(self, X: np.ndarray) -> int:
        """Get number of time samples from array."""
        if X.ndim == 1:
            return X.shape[0]
        elif X.ndim == 2:
            return X.shape[1]  # (n_channels, n_samples)
        else:
            return X.shape[2]  # (n_trials, n_channels, n_samples)

    def _get_n_channels(self, X: np.ndarray) -> int:
        """Get number of channels from array."""
        if X.ndim == 1:
            return 1
        elif X.ndim == 2:
            return X.shape[0]
        else:
            return X.shape[1]

    def _apply_along_axis(
        self,
        func,
        X: np.ndarray,
        axis: int = -1,
        **kwargs,
    ) -> np.ndarray:
        """Apply function along time axis.

        Parameters
        ----------
        func : callable
            Function to apply.
        X : np.ndarray
            Input signal.
        axis : int, default=-1
            Axis to apply along (typically time axis).
        **kwargs
            Additional arguments to pass to func.

        Returns
        -------
        np.ndarray
            Transformed signal.
        """
        return np.apply_along_axis(func, axis, X, **kwargs)

    def _check_fs(self, fs: float | None = None) -> float:
        """Check and return sample rate."""
        if fs is not None:
            return float(fs)
        if hasattr(self, 'fs') and self.fs is not None:
            return float(self.fs)
        raise ValueError(
            "Sample rate (fs) must be specified either in constructor or method call"
        )


class BaseSignalTransformer(BaseEstimator, TransformerMixin, SignalMixin):
    """Base class for all signal processing transformers.

    Provides sklearn-compatible interface with signal-specific extensions.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz. Required for frequency-dependent operations.
    copy : bool, default=True
        Whether to copy input data before processing.

    Attributes
    ----------
    n_samples_seen_ : int
        Number of samples processed during fit.
    n_channels_seen_ : int
        Number of channels seen during fit.
    """

    def __init__(
        self,
        fs: float | None = None,
        copy: bool = True,
    ):
        self.fs = fs
        self.copy = copy

        self._is_fitted = False
        self.n_samples_seen_: int | None = None
        self.n_channels_seen_: int | None = None

    def _check_is_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this transformer."
            )

    def fit(self, X, y=None, **fit_params) -> BaseSignalTransformer:
        """Fit the transformer.

        Most signal transformers don't need fitting, but this provides
        a consistent sklearn interface.

        Parameters
        ----------
        X : array-like
            Input signal.
        y : ignored
        **fit_params : dict
            Additional parameters.

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        self.n_samples_seen_ = self._get_n_samples(X)
        self.n_channels_seen_ = self._get_n_channels(X)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Transform the signal.

        Parameters
        ----------
        X : array-like
            Input signal.

        Returns
        -------
        np.ndarray
            Transformed signal.
        """
        pass

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def inverse_transform(self, X) -> np.ndarray:
        """Inverse transform (if applicable).

        Parameters
        ----------
        X : array-like
            Transformed signal.

        Returns
        -------
        np.ndarray
            Reconstructed signal.

        Raises
        ------
        NotImplementedError
            If inverse transform is not supported.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support inverse_transform"
        )


class BaseFeatureExtractor(BaseEstimator, TransformerMixin, SignalMixin):
    """Base class for feature extraction from signals.

    Unlike transformers that output signals, feature extractors output
    feature vectors suitable for machine learning.

    Parameters
    ----------
    fs : float, optional
        Sample rate in Hz.
    """

    def __init__(self, fs: float | None = None):
        self.fs = fs
        self._is_fitted = False
        self.feature_names_: list[str] | None = None

    def fit(self, X, y=None, **fit_params) -> BaseFeatureExtractor:
        """Fit the feature extractor.

        Parameters
        ----------
        X : array-like
            Input signals of shape (n_samples, n_timepoints) or
            (n_samples, n_channels, n_timepoints).
        y : ignored

        Returns
        -------
        self
        """
        X = self._validate_signal(X)
        self._is_fitted = True
        return self

    @abstractmethod
    def transform(self, X) -> np.ndarray:
        """Extract features from signals.

        Parameters
        ----------
        X : array-like
            Input signals.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Extracted features.
        """
        pass

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.

        Returns
        -------
        List[str]
            Feature names.
        """
        if self.feature_names_ is None:
            raise RuntimeError("Feature names not available. Call fit first.")
        return self.feature_names_


def ensure_2d_signals(X: np.ndarray) -> tuple[np.ndarray, bool, tuple]:
    """Ensure signals are 2D for processing.

    Parameters
    ----------
    X : np.ndarray
        Input of shape (n_samples,), (n_channels, n_samples), or
        (n_trials, n_channels, n_samples).

    Returns
    -------
    X_2d : np.ndarray
        2D array of shape (n_signals, n_samples).
    was_1d : bool
        Whether input was 1D.
    original_shape : tuple
        Original shape for reshaping back.
    """
    original_shape = X.shape
    was_1d = X.ndim == 1

    if X.ndim == 1:
        X_2d = X.reshape(1, -1)
    elif X.ndim == 2:
        X_2d = X
    elif X.ndim == 3:
        # Flatten trials and channels
        n_trials, n_channels, n_samples = X.shape
        X_2d = X.reshape(n_trials * n_channels, n_samples)
    else:
        raise ValueError(f"Expected 1D, 2D, or 3D array, got shape {X.shape}")

    return X_2d, was_1d, original_shape


def restore_shape(X: np.ndarray, was_1d: bool, original_shape: tuple) -> np.ndarray:
    """Restore array to original shape after processing.

    Parameters
    ----------
    X : np.ndarray
        Processed 2D array.
    was_1d : bool
        Whether original was 1D.
    original_shape : tuple
        Original shape.

    Returns
    -------
    np.ndarray
        Array reshaped to match original dimensionality.
    """
    if was_1d:
        return X.flatten()
    elif len(original_shape) == 2:
        return X
    elif len(original_shape) == 3:
        n_trials, n_channels, _ = original_shape
        n_samples_out = X.shape[1]
        return X.reshape(n_trials, n_channels, n_samples_out)
    return X
