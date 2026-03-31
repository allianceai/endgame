"""Hierarchical TSK (HTSK) fuzzy systems.

Stacks multiple TSK layers to avoid the curse of dimensionality,
with each layer reducing the effective input dimension through
fuzzy rule aggregation.

References
----------
Wu, D., & Tan, W. W. (2006). A simplified type-2 fuzzy logic controller
for real-time control. ISA Transactions, 45(4), 503-516.

Example
-------
>>> from endgame.fuzzy.modern.htsk import HTSKRegressor
>>> reg = HTSKRegressor(n_layers=3, n_rules_per_layer=5)
>>> reg.fit(X_train, y_train)
>>> predictions = reg.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _HTSKLayer:
    """Single TSK layer with Gaussian antecedents and linear consequents.

    Parameters
    ----------
    n_rules : int
        Number of fuzzy rules in this layer.
    order : int
        TSK order (0 or 1).
    """

    def __init__(self, n_rules: int, order: int = 1):
        self.n_rules = n_rules
        self.order = order
        self.centers_: np.ndarray | None = None
        self.sigmas_: np.ndarray | None = None
        self.consequent_params_: np.ndarray | None = None

    def init_params(self, X: np.ndarray, rng: np.random.RandomState) -> None:
        """Initialize rule parameters from data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        rng : RandomState
            Random number generator.
        """
        n_samples, n_features = X.shape
        n_rules = min(self.n_rules, n_samples)

        # Initialize centers via random selection from data
        indices = rng.choice(n_samples, size=n_rules, replace=False)
        self.centers_ = X[indices].copy()

        # Initialize sigmas from feature-wise standard deviations
        stds = np.std(X, axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        self.sigmas_ = np.tile(stds, (n_rules, 1))

        # Initialize consequent parameters
        if self.order == 0:
            self.consequent_params_ = rng.randn(n_rules, 1) * 0.01
        else:
            # order=1: one weight per feature + bias
            self.consequent_params_ = rng.randn(n_rules, n_features + 1) * 0.01

    def forward(self, X: np.ndarray) -> np.ndarray:
        """Compute layer output via TSK inference.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
            Normalized rule outputs (serves as input to next layer).
        """
        firing = self._compute_firing(X)
        consequents = self._compute_consequents(X)

        # Normalize firing strengths
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum

        # Weighted rule outputs
        return normalized * consequents

    def _compute_firing(self, X: np.ndarray) -> np.ndarray:
        """Compute Gaussian firing strengths.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_rules = self.centers_.shape[0]
        firing = np.ones((n_samples, n_rules))

        for r in range(n_rules):
            diff = X - self.centers_[r]
            sigma_sq = self.sigmas_[r] ** 2 + 1e-10
            exponent = -0.5 * np.sum(diff ** 2 / sigma_sq, axis=1)
            firing[:, r] = np.exp(exponent)

        return firing

    def _compute_consequents(self, X: np.ndarray) -> np.ndarray:
        """Compute consequent values for each rule.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_rules = self.centers_.shape[0]
        consequents = np.zeros((n_samples, n_rules))

        if self.order == 0:
            for r in range(n_rules):
                consequents[:, r] = self.consequent_params_[r, 0]
        else:
            X_aug = np.column_stack([X, np.ones(n_samples)])
            for r in range(n_rules):
                consequents[:, r] = X_aug @ self.consequent_params_[r]

        return consequents

    def update_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float,
    ) -> float:
        """Update parameters via gradient descent.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
        lr : float
            Learning rate.

        Returns
        -------
        float
            Mean squared error loss.
        """
        firing = self._compute_firing(X)
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum  # (n_samples, n_rules)

        consequents = self._compute_consequents(X)  # (n_samples, n_rules)
        output = np.sum(normalized * consequents, axis=1)  # (n_samples,)

        if y.ndim > 1:
            y_flat = y.ravel()[:len(output)]
        else:
            y_flat = y

        error = output - y_flat
        loss = float(np.mean(error ** 2))

        n_rules = self.centers_.shape[0]
        n_samples = X.shape[0]

        # Gradient for consequent params
        X_aug = np.column_stack([X, np.ones(n_samples)])
        for r in range(n_rules):
            grad = (error * normalized[:, r])[:, None] * X_aug
            grad_mean = grad.mean(axis=0)
            if self.order == 0:
                self.consequent_params_[r, 0] -= lr * grad_mean[-1]
            else:
                self.consequent_params_[r] -= lr * grad_mean

        # Gradient for centers and sigmas
        for r in range(n_rules):
            diff = X - self.centers_[r]
            sigma_sq = self.sigmas_[r] ** 2 + 1e-10

            # Partial derivative of firing w.r.t. center
            d_fire_center = firing[:, r:r+1] * diff / sigma_sq
            # Chain through normalized output
            rule_output = consequents[:, r]
            other_output = np.sum(normalized * consequents, axis=1) - normalized[:, r] * rule_output
            d_norm_fire = (rule_output[:, None] * fire_sum.ravel()[:, None] -
                          np.sum(normalized * consequents, axis=1)[:, None]) / (fire_sum.ravel()[:, None] ** 2 + 1e-10)
            grad_center = (error[:, None] * d_norm_fire * d_fire_center).mean(axis=0)
            self.centers_[r] -= lr * np.clip(grad_center, -1.0, 1.0)

            # Partial derivative of firing w.r.t. sigma
            d_fire_sigma = firing[:, r:r+1] * diff ** 2 / (self.sigmas_[r] ** 3 + 1e-10)
            grad_sigma = (error[:, None] * d_norm_fire * d_fire_sigma).mean(axis=0)
            self.sigmas_[r] -= lr * np.clip(grad_sigma, -1.0, 1.0)
            self.sigmas_[r] = np.maximum(self.sigmas_[r], 1e-4)

        return loss


class HTSKRegressor(BaseEstimator, RegressorMixin):
    """Hierarchical TSK fuzzy system for regression.

    Stacks multiple TSK layers where each layer reduces the
    dimensionality through fuzzy rule aggregation, avoiding
    the exponential growth of rules with input dimension.

    Parameters
    ----------
    n_layers : int, default=3
        Number of stacked TSK layers.
    n_rules_per_layer : int, default=5
        Number of fuzzy rules in each layer.
    order : int, default=1
        TSK order (0=constant, 1=linear consequents).
    n_epochs : int, default=50
        Number of training epochs per layer.
    lr : float, default=0.01
        Learning rate for gradient descent.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    layers_ : list of _HTSKLayer
        Trained TSK layers.
    n_features_in_ : int
        Number of features seen during fit.
    output_weights_ : ndarray
        Final output combination weights.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.htsk import HTSKRegressor
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] + np.random.randn(200) * 0.1
    >>> reg = HTSKRegressor(n_layers=2, n_rules_per_layer=5, n_epochs=30)
    >>> reg.fit(X, y)
    HTSKRegressor(n_epochs=30, n_layers=2)
    >>> preds = reg.predict(X[:5])
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_rules_per_layer: int = 5,
        order: int = 1,
        n_epochs: int = 50,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_rules_per_layer = n_rules_per_layer
        self.order = order
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> HTSKRegressor:
        """Fit the hierarchical TSK model.

        Each layer is trained sequentially using the previous layer's
        output as input to the next layer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        self.layers_ = []
        current_input = X.copy()

        for layer_idx in range(self.n_layers):
            layer = _HTSKLayer(
                n_rules=self.n_rules_per_layer,
                order=self.order,
            )
            layer.init_params(current_input, rng)

            for epoch in range(self.n_epochs):
                loss = layer.update_params(current_input, y, self.lr)
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"[HTSK] Layer {layer_idx+1}/{self.n_layers}, "
                        f"Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.6f}"
                    )

            self.layers_.append(layer)
            current_input = layer.forward(current_input)

        # Final output weights via least squares
        n_rules = current_input.shape[1]
        current_aug = np.column_stack([current_input, np.ones(len(current_input))])
        self.output_weights_, _, _, _ = np.linalg.lstsq(current_aug, y, rcond=None)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["layers_", "output_weights_"])
        X = check_array(X)

        current = X
        for layer in self.layers_:
            current = layer.forward(current)

        current_aug = np.column_stack([current, np.ones(len(current))])
        return current_aug @ self.output_weights_


class HTSKClassifier(BaseEstimator, ClassifierMixin):
    """Hierarchical TSK fuzzy system for classification.

    Uses one-vs-rest decomposition with an HTSK regressor per class
    to produce class probabilities via softmax normalization.

    Parameters
    ----------
    n_layers : int, default=3
        Number of stacked TSK layers.
    n_rules_per_layer : int, default=5
        Number of fuzzy rules in each layer.
    order : int, default=1
        TSK order (0=constant, 1=linear consequents).
    n_epochs : int, default=50
        Number of training epochs per layer.
    lr : float, default=0.01
        Learning rate.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.htsk import HTSKClassifier
    >>> X = np.random.randn(100, 5)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = HTSKClassifier(n_layers=2, n_rules_per_layer=3, n_epochs=20)
    >>> clf.fit(X, y)
    HTSKClassifier(n_epochs=20, n_layers=2, n_rules_per_layer=3)
    >>> clf.predict(X[:3])
    array([...])
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_rules_per_layer: int = 5,
        order: int = 1,
        n_epochs: int = 50,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_rules_per_layer = n_rules_per_layer
        self.order = order
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> HTSKClassifier:
        """Fit the classifier using one HTSK regressor per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        self.regressors_ = []
        for c in range(n_classes):
            y_binary = (y_enc == c).astype(np.float64)
            reg = HTSKRegressor(
                n_layers=self.n_layers,
                n_rules_per_layer=self.n_rules_per_layer,
                order=self.order,
                n_epochs=self.n_epochs,
                lr=self.lr,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary)
            self.regressors_.append(reg)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via softmax over regressor outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["regressors_"])
        X = check_array(X)

        raw = np.column_stack([reg.predict(X) for reg in self.regressors_])

        # Softmax normalization
        raw_shifted = raw - raw.max(axis=1, keepdims=True)
        exp_raw = np.exp(raw_shifted)
        return exp_raw / exp_raw.sum(axis=1, keepdims=True)
