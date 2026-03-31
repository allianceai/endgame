"""SOFNN (Self-Organizing Fuzzy Neural Network).

Dynamically adds and prunes neurons (rules) based on data complexity.
Supports both batch and online learning modes.

References
----------
- Leng et al., "An approach for on-line extraction of fuzzy rules using a
  self-organising fuzzy neural network" (2005)

Example
-------
>>> from endgame.fuzzy.neurofuzzy import SOFNNRegressor
>>> model = SOFNNRegressor(threshold_add=0.5, max_rules=50)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SOFNNRegressor(BaseEstimator, RegressorMixin):
    """Self-Organizing Fuzzy Neural Network for regression.

    Automatically determines the number of rules by adding neurons when
    approximation error is high and pruning neurons with low contribution.

    Parameters
    ----------
    threshold_add : float, default=0.1
        Error threshold for adding a new rule. If the local error exceeds
        this, a new rule is created.
    threshold_prune : float, default=0.01
        Contribution threshold for pruning. Rules with firing strength
        contribution below this are removed.
    max_rules : int, default=50
        Maximum number of rules allowed.
    sigma_init : float, default=0.5
        Initial width of Gaussian membership functions.
    lr : float, default=0.01
        Learning rate for parameter updates.
    n_epochs : int, default=50
        Number of training epochs (batch mode).
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.
    """

    def __init__(
        self,
        threshold_add: float = 0.1,
        threshold_prune: float = 0.01,
        max_rules: int = 50,
        sigma_init: float = 0.5,
        lr: float = 0.01,
        n_epochs: int = 50,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.threshold_add = threshold_add
        self.threshold_prune = threshold_prune
        self.max_rules = max_rules
        self.sigma_init = sigma_init
        self.lr = lr
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> SOFNNRegressor:
        """Fit the SOFNN by growing and pruning rules.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Initialize with first data point as a rule
        self.centers_ = [X[0].copy()]
        self.sigmas_ = [np.full(n_features, self.sigma_init)]
        self.consequent_weights_ = [np.zeros(n_features + 1)]  # Linear consequent

        # Normalize data ranges for sigma initialization
        x_range = np.ptp(X, axis=0)
        x_range[x_range == 0] = 1.0

        for epoch in range(self.n_epochs):
            perm = rng.permutation(n_samples)
            total_error = 0.0

            for idx in perm:
                xi = X[idx]
                yi = y[idx]

                # Compute firing strengths
                strengths = self._firing_strengths(xi.reshape(1, -1)).ravel()
                y_pred = self._predict_single(xi, strengths)
                error = yi - y_pred
                total_error += error ** 2

                # Structure learning: add rule if error is large
                if (abs(error) > self.threshold_add
                        and len(self.centers_) < self.max_rules):
                    # Check if this point is far from existing centers
                    min_dist = float("inf")
                    for c in self.centers_:
                        dist = np.linalg.norm((xi - c) / x_range)
                        min_dist = min(min_dist, dist)

                    if min_dist > self.sigma_init * 0.5:
                        self.centers_.append(xi.copy())
                        self.sigmas_.append(np.full(n_features, self.sigma_init))
                        self.consequent_weights_.append(np.zeros(n_features + 1))
                        strengths = self._firing_strengths(xi.reshape(1, -1)).ravel()

                # Parameter learning: update consequent weights
                norm_strengths = strengths / (strengths.sum() + 1e-10)
                x_aug = np.append(xi, 1.0)
                for r in range(len(self.centers_)):
                    self.consequent_weights_[r] += (
                        self.lr * error * norm_strengths[r] * x_aug
                    )

                # Update centers and sigmas (gradient descent on error)
                for r in range(len(self.centers_)):
                    if strengths[r] < 1e-10:
                        continue
                    cons_out = np.dot(self.consequent_weights_[r],  x_aug)
                    diff = xi - self.centers_[r]
                    sigma_sq = self.sigmas_[r] ** 2 + 1e-10

                    # Gradient of firing strength w.r.t. center
                    d_center = strengths[r] * diff / sigma_sq
                    # Gradient of firing strength w.r.t. sigma
                    d_sigma = strengths[r] * (diff ** 2) / (self.sigmas_[r] ** 3 + 1e-10)

                    weight = error * norm_strengths[r]
                    self.centers_[r] += self.lr * weight * d_center
                    self.sigmas_[r] += self.lr * weight * d_sigma
                    self.sigmas_[r] = np.maximum(self.sigmas_[r], 0.01)

            # Pruning: remove rules with consistently low contribution
            if len(self.centers_) > 1:
                all_strengths = self._firing_strengths(X)  # (n_samples, n_rules)
                avg_contribution = all_strengths.mean(axis=0) / (
                    all_strengths.sum(axis=1, keepdims=True).mean() + 1e-10
                )
                keep = avg_contribution > self.threshold_prune
                if keep.sum() < 1:
                    keep[np.argmax(avg_contribution)] = True
                if not keep.all():
                    keep_idx = np.where(keep)[0]
                    self.centers_ = [self.centers_[i] for i in keep_idx]
                    self.sigmas_ = [self.sigmas_[i] for i in keep_idx]
                    self.consequent_weights_ = [
                        self.consequent_weights_[i] for i in keep_idx
                    ]

            rmse = np.sqrt(total_error / n_samples)
            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1}/{self.n_epochs}, "
                    f"RMSE: {rmse:.6f}, Rules: {len(self.centers_)}"
                )

        self.n_rules_ = len(self.centers_)
        return self

    def _firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Compute Gaussian firing strengths."""
        n_samples = X.shape[0]
        n_rules = len(self.centers_)
        strengths = np.zeros((n_samples, n_rules))
        for r in range(n_rules):
            diff = X - self.centers_[r]
            sigma_sq = self.sigmas_[r] ** 2 + 1e-10
            strengths[:, r] = np.exp(-0.5 * np.sum(diff ** 2 / sigma_sq, axis=1))
        return strengths

    def _predict_single(
        self, x: np.ndarray, strengths: np.ndarray
    ) -> float:
        """Predict for a single sample given firing strengths."""
        norm = strengths / (strengths.sum() + 1e-10)
        x_aug = np.append(x, 1.0)
        output = 0.0
        for r in range(len(self.centers_)):
            output += norm[r] * np.dot(self.consequent_weights_[r], x_aug)
        return output

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["centers_"])
        X = check_array(X, dtype=np.float64)
        strengths = self._firing_strengths(X)
        norm = strengths / (strengths.sum(axis=1, keepdims=True) + 1e-10)

        n_samples = X.shape[0]
        X_aug = np.column_stack([X, np.ones(n_samples)])
        weights = np.array(self.consequent_weights_)  # (n_rules, n_features+1)
        consequents = X_aug @ weights.T  # (n_samples, n_rules)
        return np.sum(norm * consequents, axis=1)

    def partial_fit(self, X: Any, y: Any) -> SOFNNRegressor:
        """Online update with new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        if not hasattr(self, "centers_"):
            return self.fit(X, y)

        for i in range(len(X)):
            xi = X[i]
            yi = y[i]
            strengths = self._firing_strengths(xi.reshape(1, -1)).ravel()
            y_pred = self._predict_single(xi, strengths)
            error = yi - y_pred

            if abs(error) > self.threshold_add and len(self.centers_) < self.max_rules:
                self.centers_.append(xi.copy())
                self.sigmas_.append(np.full(self.n_features_in_, self.sigma_init))
                self.consequent_weights_.append(np.zeros(self.n_features_in_ + 1))
                strengths = self._firing_strengths(xi.reshape(1, -1)).ravel()

            norm_strengths = strengths / (strengths.sum() + 1e-10)
            x_aug = np.append(xi, 1.0)
            for r in range(len(self.centers_)):
                self.consequent_weights_[r] += (
                    self.lr * error * norm_strengths[r] * x_aug
                )

        self.n_rules_ = len(self.centers_)
        return self
