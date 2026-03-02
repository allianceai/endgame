from __future__ import annotations

"""Gaussian Process models with competition-tuned defaults.

Gaussian Processes provide Bayesian inference with kernel methods,
offering principled uncertainty quantification and different error
patterns from tree-based and neural network models.

References
----------
- Rasmussen & Williams, "Gaussian Processes for Machine Learning" (2006)
- sklearn.gaussian_process documentation
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.gaussian_process import GaussianProcessClassifier as _GPClassifier
from sklearn.gaussian_process import GaussianProcessRegressor as _GPRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Kernel presets for different problem types
KERNEL_PRESETS = {
    "rbf": lambda length_scale: ConstantKernel(1.0) * RBF(length_scale=length_scale),
    "matern": lambda length_scale: ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5),
    "matern12": lambda length_scale: ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=0.5),
    "matern32": lambda length_scale: ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=1.5),
    "matern52": lambda length_scale: ConstantKernel(1.0) * Matern(length_scale=length_scale, nu=2.5),
    "rq": lambda length_scale: ConstantKernel(1.0) * RationalQuadratic(length_scale=length_scale),
    "linear": lambda length_scale: ConstantKernel(1.0) * DotProduct(sigma_0=1.0),
}


class GPClassifier(ClassifierMixin, BaseEstimator):
    """Gaussian Process Classifier with competition-tuned defaults.

    A Bayesian kernel method that provides probabilistic predictions with
    principled uncertainty estimates. Different inductive bias from trees
    and neural networks, making it valuable for ensemble diversity.

    Parameters
    ----------
    kernel : str or sklearn kernel, default='rbf'
        Kernel type. Options: 'rbf', 'matern', 'matern12', 'matern32',
        'matern52', 'rq', 'linear', or a sklearn kernel object.
    length_scale : float, default=1.0
        Length scale parameter for the kernel.
    n_restarts_optimizer : int, default=3
        Number of restarts for the optimizer.
    max_iter_predict : int, default=100
        Maximum iterations for prediction.
    warm_start : bool, default=False
        Use previous fit as initialization.
    multi_class : str, default='one_vs_rest'
        Multi-class strategy: 'one_vs_rest' or 'one_vs_one'.
    auto_scale : bool, default=True
        Automatically scale features before fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    model_ : GaussianProcessClassifier
        Fitted sklearn GP classifier.

    Examples
    --------
    >>> from endgame.models.kernel import GPClassifier
    >>> clf = GPClassifier(kernel='rbf', random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> # Get uncertainty
    >>> proba, std = clf.predict_proba(X_test, return_std=True)

    Notes
    -----
    Gaussian Processes excel on small-medium datasets where uncertainty
    matters. They scale O(n^3) with training size, so not suitable for
    large datasets (>10k samples) without approximations.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        kernel: str | Any = "rbf",
        length_scale: float = 1.0,
        n_restarts_optimizer: int = 3,
        max_iter_predict: int = 100,
        warm_start: bool = False,
        multi_class: str = "one_vs_rest",
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.n_restarts_optimizer = n_restarts_optimizer
        self.max_iter_predict = max_iter_predict
        self.warm_start = warm_start
        self.multi_class = multi_class
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: _GPClassifier | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def _get_kernel(self):
        """Get kernel object from string or return as-is."""
        if isinstance(self.kernel, str):
            if self.kernel not in KERNEL_PRESETS:
                raise ValueError(f"Unknown kernel: {self.kernel}. "
                               f"Options: {list(KERNEL_PRESETS.keys())}")
            return KERNEL_PRESETS[self.kernel](self.length_scale)
        return self.kernel

    def fit(self, X, y, **fit_params) -> GPClassifier:
        """Fit the Gaussian Process classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Create and fit model
        kernel = self._get_kernel()

        self.model_ = _GPClassifier(
            kernel=kernel,
            n_restarts_optimizer=self.n_restarts_optimizer,
            max_iter_predict=self.max_iter_predict,
            warm_start=self.warm_start,
            multi_class=self.multi_class,
            random_state=self.random_state,
        )

        self.model_.fit(X_scaled, y_encoded)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("GPClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        y_pred = self.model_.predict(X_scaled)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X, return_std: bool = False) -> np.ndarray | tuple:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        return_std : bool, default=False
            If True, also return uncertainty estimates.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        std : ndarray of shape (n_samples,), optional
            Uncertainty estimates (if return_std=True).
        """
        if not self._is_fitted:
            raise RuntimeError("GPClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        proba = self.model_.predict_proba(X_scaled)

        if return_std:
            # Estimate uncertainty from entropy of predictions
            entropy = -np.sum(proba * np.log(proba + 1e-10), axis=1)
            max_entropy = np.log(self.n_classes_)
            std = entropy / max_entropy  # Normalized uncertainty
            return proba, std

        return proba


class GPRegressor(RegressorMixin, BaseEstimator):
    """Gaussian Process Regressor with competition-tuned defaults.

    A Bayesian kernel method that provides predictions with principled
    uncertainty estimates through the posterior predictive distribution.

    Parameters
    ----------
    kernel : str or sklearn kernel, default='rbf'
        Kernel type. Options: 'rbf', 'matern', 'matern12', 'matern32',
        'matern52', 'rq', 'linear', or a sklearn kernel object.
    length_scale : float, default=1.0
        Length scale parameter for the kernel.
    alpha : float, default=1e-10
        Value added to diagonal for numerical stability.
    n_restarts_optimizer : int, default=3
        Number of restarts for the optimizer.
    normalize_y : bool, default=True
        Normalize target values.
    auto_scale : bool, default=True
        Automatically scale features before fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    model_ : GaussianProcessRegressor
        Fitted sklearn GP regressor.

    Examples
    --------
    >>> from endgame.models.kernel import GPRegressor
    >>> reg = GPRegressor(kernel='matern', random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred, y_std = reg.predict(X_test, return_std=True)
    >>> # Prediction intervals
    >>> lower = y_pred - 1.96 * y_std
    >>> upper = y_pred + 1.96 * y_std
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        kernel: str | Any = "rbf",
        length_scale: float = 1.0,
        alpha: float = 1e-10,
        n_restarts_optimizer: int = 3,
        normalize_y: bool = True,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.kernel = kernel
        self.length_scale = length_scale
        self.alpha = alpha
        self.n_restarts_optimizer = n_restarts_optimizer
        self.normalize_y = normalize_y
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.n_features_in_: int = 0
        self.model_: _GPRegressor | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted: bool = False

    def _get_kernel(self):
        """Get kernel object from string or return as-is."""
        if isinstance(self.kernel, str):
            if self.kernel not in KERNEL_PRESETS:
                raise ValueError(f"Unknown kernel: {self.kernel}. "
                               f"Options: {list(KERNEL_PRESETS.keys())}")
            # Add white noise kernel for regression
            base_kernel = KERNEL_PRESETS[self.kernel](self.length_scale)
            return base_kernel + WhiteKernel(noise_level=0.1)
        return self.kernel

    def fit(self, X, y, **fit_params) -> GPRegressor:
        """Fit the Gaussian Process regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Create and fit model
        kernel = self._get_kernel()

        self.model_ = _GPRegressor(
            kernel=kernel,
            alpha=self.alpha,
            n_restarts_optimizer=self.n_restarts_optimizer,
            normalize_y=self.normalize_y,
            random_state=self.random_state,
        )

        self.model_.fit(X_scaled, y)
        self._is_fitted = True

        return self

    def predict(self, X, return_std: bool = False, return_cov: bool = False):
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        return_std : bool, default=False
            If True, return standard deviation of predictions.
        return_cov : bool, default=False
            If True, return covariance of predictions.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        y_std : ndarray of shape (n_samples,), optional
            Standard deviation (if return_std=True).
        y_cov : ndarray of shape (n_samples, n_samples), optional
            Covariance matrix (if return_cov=True).
        """
        if not self._is_fitted:
            raise RuntimeError("GPRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model_.predict(X_scaled, return_std=return_std, return_cov=return_cov)

    def predict_interval(self, X, alpha: float = 0.05) -> tuple:
        """Predict with prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        alpha : float, default=0.05
            Significance level (0.05 = 95% interval).

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Point predictions.
        lower : ndarray of shape (n_samples,)
            Lower bound of prediction interval.
        upper : ndarray of shape (n_samples,)
            Upper bound of prediction interval.
        """
        from scipy import stats

        y_pred, y_std = self.predict(X, return_std=True)
        z = stats.norm.ppf(1 - alpha / 2)

        lower = y_pred - z * y_std
        upper = y_pred + z * y_std

        return y_pred, lower, upper

    def sample_y(self, X, n_samples: int = 1, random_state: int | None = None) -> np.ndarray:
        """Sample from the posterior predictive distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Query points.
        n_samples : int, default=1
            Number of samples to draw.
        random_state : int, optional
            Random seed.

        Returns
        -------
        samples : ndarray of shape (n_query, n_samples)
            Samples from posterior predictive.
        """
        if not self._is_fitted:
            raise RuntimeError("GPRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model_.sample_y(X_scaled, n_samples=n_samples, random_state=random_state)
