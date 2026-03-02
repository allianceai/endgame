from __future__ import annotations

"""Extreme Learning Machine implementation.

ELM is a single-layer feedforward neural network where input weights are
randomly assigned and never updated. Only the output weights are learned
via a closed-form solution (pseudoinverse), making training extremely fast.

This fundamentally different optimization (no backpropagation) provides
unique predictions that enhance ensemble diversity.

References
----------
- Huang et al., "Extreme Learning Machine: Theory and Applications" (2006)
- Huang et al., "Extreme Learning Machine for Regression and Multiclass Classification" (2012)
"""

from collections.abc import Callable

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler


def _sigmoid(x):
    """Sigmoid activation function."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x):
    """Tanh activation function."""
    return np.tanh(x)


def _relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def _leaky_relu(x, alpha=0.01):
    """Leaky ReLU activation function."""
    return np.where(x > 0, x, alpha * x)


def _sin(x):
    """Sinusoidal activation function."""
    return np.sin(x)


def _hardlim(x):
    """Hard limit activation function."""
    return (x >= 0).astype(float)


ACTIVATION_FUNCTIONS = {
    "sigmoid": _sigmoid,
    "tanh": _tanh,
    "relu": _relu,
    "leaky_relu": _leaky_relu,
    "sin": _sin,
    "hardlim": _hardlim,
}


class ELMClassifier(ClassifierMixin, BaseEstimator):
    """Extreme Learning Machine Classifier.

    A single-layer neural network with random input weights and
    analytically computed output weights. Training is extremely fast
    (milliseconds) because there's no iterative optimization.

    Parameters
    ----------
    n_hidden : int, default=500
        Number of hidden neurons.
    activation : str or callable, default='sigmoid'
        Activation function: 'sigmoid', 'tanh', 'relu', 'leaky_relu',
        'sin', 'hardlim', or a callable.
    alpha : float, default=1e-6
        Regularization parameter for ridge regression.
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
    input_weights_ : ndarray
        Random input-to-hidden weights.
    biases_ : ndarray
        Random hidden layer biases.
    output_weights_ : ndarray
        Learned hidden-to-output weights.

    Examples
    --------
    >>> from endgame.models.baselines import ELMClassifier
    >>> clf = ELMClassifier(n_hidden=500, random_state=42)
    >>> clf.fit(X_train, y_train)  # Milliseconds!
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    ELM is valuable for ensemble diversity because:
    1. No backpropagation - fundamentally different optimization
    2. Random projections explore different feature spaces
    3. Extremely fast - can train many models for ensemble selection
    4. Often surprisingly competitive with slower methods

    The analytical solution is: beta = pinv(H) @ T
    where H is the hidden layer output and T is the target.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_hidden: int = 500,
        activation: str | Callable = "sigmoid",
        alpha: float = 1e-6,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.input_weights_: np.ndarray | None = None
        self.biases_: np.ndarray | None = None
        self.output_weights_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def _get_activation(self) -> Callable:
        """Get activation function."""
        if callable(self.activation):
            return self.activation
        if self.activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation: {self.activation}. "
                           f"Options: {list(ACTIVATION_FUNCTIONS.keys())}")
        return ACTIVATION_FUNCTIONS[self.activation]

    def _compute_hidden_output(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden layer output H."""
        activation = self._get_activation()
        # H = activation(X @ W + b)
        H = activation(X @ self.input_weights_ + self.biases_)
        return H

    def fit(self, X, y, **fit_params) -> ELMClassifier:
        """Fit the ELM classifier.

        Training is O(n * m * h) where n=samples, m=features, h=hidden.
        The closed-form solution makes this extremely fast.

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
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # One-hot encode targets for multi-class
        if self.n_classes_ > 2:
            T = np.eye(self.n_classes_)[y_encoded]
        else:
            T = y_encoded.reshape(-1, 1)

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Initialize random input weights and biases
        # Using uniform distribution in [-1, 1]
        self.input_weights_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.biases_ = rng.uniform(-1, 1, (1, self.n_hidden))

        # Compute hidden layer output
        H = self._compute_hidden_output(X_scaled)

        # Compute output weights using regularized pseudoinverse
        # beta = (H^T H + alpha*I)^(-1) H^T T
        HtH = H.T @ H
        regularized = HtH + self.alpha * np.eye(self.n_hidden)
        HtT = H.T @ T

        try:
            self.output_weights_ = np.linalg.solve(regularized, HtT)
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse if singular
            self.output_weights_ = np.linalg.pinv(regularized) @ HtT

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
            raise RuntimeError("ELMClassifier has not been fitted.")

        proba = self.predict_proba(X)
        y_pred_encoded = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(y_pred_encoded)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities (softmax normalized).
        """
        if not self._is_fitted:
            raise RuntimeError("ELMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Compute hidden layer and output
        H = self._compute_hidden_output(X_scaled)
        output = H @ self.output_weights_

        # For binary classification
        if self.n_classes_ == 2:
            proba_pos = _sigmoid(output).ravel()
            proba = np.column_stack([1 - proba_pos, proba_pos])
        else:
            # Softmax for multi-class
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            proba = exp_output / np.sum(exp_output, axis=1, keepdims=True)

        return proba


class ELMRegressor(RegressorMixin, BaseEstimator):
    """Extreme Learning Machine Regressor.

    A single-layer neural network with random input weights and
    analytically computed output weights for regression.

    Parameters
    ----------
    n_hidden : int, default=500
        Number of hidden neurons.
    activation : str or callable, default='tanh'
        Activation function. 'tanh' is preferred for regression
        (unbounded, symmetric). 'sigmoid' compresses to [0,1].
    alpha : float, default=0.01
        Regularization parameter for ridge regression on output weights.
    auto_scale : bool, default=True
        Automatically scale features before fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    input_weights_ : ndarray
        Random input-to-hidden weights.
    output_weights_ : ndarray
        Learned hidden-to-output weights.

    Examples
    --------
    >>> from endgame.models.baselines import ELMRegressor
    >>> reg = ELMRegressor(n_hidden=500, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_hidden: int = 500,
        activation: str | Callable = "tanh",
        alpha: float = 0.01,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.n_hidden = n_hidden
        self.activation = activation
        self.alpha = alpha
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.n_features_in_: int = 0
        self.input_weights_: np.ndarray | None = None
        self.biases_: np.ndarray | None = None
        self.output_weights_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._is_fitted: bool = False

    def _get_activation(self) -> Callable:
        """Get activation function."""
        if callable(self.activation):
            return self.activation
        if self.activation not in ACTIVATION_FUNCTIONS:
            raise ValueError(f"Unknown activation: {self.activation}. "
                           f"Options: {list(ACTIVATION_FUNCTIONS.keys())}")
        return ACTIVATION_FUNCTIONS[self.activation]

    def _compute_hidden_output(self, X: np.ndarray) -> np.ndarray:
        """Compute hidden layer output H."""
        activation = self._get_activation()
        H = activation(X @ self.input_weights_ + self.biases_)
        return H

    def fit(self, X, y, **fit_params) -> ELMRegressor:
        """Fit the ELM regressor.

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
        rng = np.random.RandomState(self.random_state)

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            # Also scale target
            self._y_mean = np.mean(y)
            self._y_std = np.std(y) + 1e-8
            y_scaled = (y - self._y_mean) / self._y_std
        else:
            X_scaled = X
            y_scaled = y

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0)

        T = y_scaled.reshape(-1, 1)

        # Initialize random input weights and biases
        self.input_weights_ = rng.uniform(-1, 1, (n_features, self.n_hidden))
        self.biases_ = rng.uniform(-1, 1, (1, self.n_hidden))

        # Compute hidden layer output
        H = self._compute_hidden_output(X_scaled)

        # Compute output weights using regularized pseudoinverse
        HtH = H.T @ H
        regularized = HtH + self.alpha * np.eye(self.n_hidden)
        HtT = H.T @ T

        try:
            self.output_weights_ = np.linalg.solve(regularized, HtT)
        except np.linalg.LinAlgError:
            self.output_weights_ = np.linalg.pinv(regularized) @ HtT

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("ELMRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Compute prediction
        H = self._compute_hidden_output(X_scaled)
        y_pred = (H @ self.output_weights_).ravel()

        # Inverse scale
        if self.auto_scale:
            y_pred = y_pred * self._y_std + self._y_mean

        return y_pred
