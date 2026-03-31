"""DENFIS (Dynamic Evolving Neuro-Fuzzy Inference System).

Kasabov's online evolving system that creates rules incrementally
from streaming data using the Evolving Clustering Method (ECM).

References
----------
- Kasabov & Song, "DENFIS: Dynamic Evolving Neural-Fuzzy Inference System
  and its Application for Time-Series Prediction" (2002)

Example
-------
>>> from endgame.fuzzy.neurofuzzy import DENFISRegressor
>>> model = DENFISRegressor(distance_threshold=0.3)
>>> model.fit(X_train, y_train)
>>> model.partial_fit(X_new, y_new)  # Online update
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _ECMClusterer:
    """Evolving Clustering Method for online rule creation.

    Creates and updates clusters incrementally. Each cluster becomes
    the center of a fuzzy rule's antecedent.
    """

    def __init__(self, distance_threshold: float = 0.3, max_clusters: int = 100):
        self.distance_threshold = distance_threshold
        self.max_clusters = max_clusters
        self.centers: list[np.ndarray] = []
        self.radii: list[float] = []
        self.counts: list[int] = []

    def update(self, x: np.ndarray) -> int:
        """Add a data point, creating or updating a cluster.

        Returns the index of the assigned cluster.
        """
        if len(self.centers) == 0:
            self.centers.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return 0

        # Find nearest cluster
        dists = [np.linalg.norm(x - c) for c in self.centers]
        nearest = int(np.argmin(dists))
        min_dist = dists[nearest]

        if min_dist <= self.distance_threshold:
            # Update existing cluster
            n = self.counts[nearest]
            self.centers[nearest] = (
                self.centers[nearest] * n + x
            ) / (n + 1)
            self.radii[nearest] = max(self.radii[nearest], min_dist)
            self.counts[nearest] += 1
            return nearest
        elif len(self.centers) < self.max_clusters:
            # Create new cluster
            self.centers.append(x.copy())
            self.radii.append(0.0)
            self.counts.append(1)
            return len(self.centers) - 1
        else:
            # Merge with nearest if at capacity
            n = self.counts[nearest]
            self.centers[nearest] = (
                self.centers[nearest] * n + x
            ) / (n + 1)
            self.radii[nearest] = max(self.radii[nearest], min_dist)
            self.counts[nearest] += 1
            return nearest


class DENFISRegressor(BaseEstimator, RegressorMixin):
    """Dynamic Evolving Neuro-Fuzzy Inference System for regression.

    Creates fuzzy rules incrementally from data using ECM clustering.
    Each cluster defines a rule's antecedent (Gaussian MFs), and
    consequent parameters are learned via local weighted least squares.

    Parameters
    ----------
    distance_threshold : float, default=0.3
        Threshold for creating new clusters/rules.
    max_rules : int, default=100
        Maximum number of rules.
    sigma_scale : float, default=1.5
        Scaling factor for Gaussian widths (relative to cluster radius).
    order : int, default=1
        Consequent order: 0 for constant, 1 for linear.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print progress.
    """

    def __init__(
        self,
        distance_threshold: float = 0.3,
        max_rules: int = 100,
        sigma_scale: float = 1.5,
        order: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.distance_threshold = distance_threshold
        self.max_rules = max_rules
        self.sigma_scale = sigma_scale
        self.order = order
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> DENFISRegressor:
        """Fit DENFIS by processing all training data.

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

        # Scale data for distance computations
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Phase 1: Create rules via ECM
        self.ecm_ = _ECMClusterer(
            distance_threshold=self.distance_threshold,
            max_clusters=self.max_rules,
        )
        assignments = np.zeros(len(X), dtype=int)
        for i in range(len(X_scaled)):
            assignments[i] = self.ecm_.update(X_scaled[i])

        # Phase 2: Compute consequent parameters via local weighted LSE
        n_rules = len(self.ecm_.centers)
        n_features = X.shape[1]

        if self.order == 0:
            self.consequent_params_ = np.zeros(n_rules)
        else:
            self.consequent_params_ = np.zeros((n_rules, n_features + 1))

        # Compute sigmas from cluster data
        self.sigmas_ = np.ones((n_rules, n_features))
        for r in range(n_rules):
            mask = assignments == r
            if mask.sum() > 0:
                cluster_data = X_scaled[mask]
                std = np.std(cluster_data, axis=0)
                std[std < 0.01] = 0.01
                self.sigmas_[r] = std * self.sigma_scale

        # Solve for consequent params per rule
        strengths = self._firing_strengths(X_scaled)
        for r in range(n_rules):
            w = strengths[:, r]
            w_sum = w.sum()
            if w_sum < 1e-10:
                continue

            if self.order == 0:
                self.consequent_params_[r] = np.average(y, weights=w)
            else:
                X_aug = np.column_stack([X, np.ones(len(X))])
                W = np.diag(w)
                try:
                    # Weighted least squares: (X'WX)^-1 X'Wy
                    XtW = X_aug.T @ W
                    self.consequent_params_[r] = np.linalg.lstsq(
                        XtW @ X_aug, XtW @ y, rcond=None
                    )[0]
                except np.linalg.LinAlgError:
                    self.consequent_params_[r, -1] = np.average(y, weights=w)

        self.n_rules_ = n_rules
        return self

    def _firing_strengths(self, X_scaled: np.ndarray) -> np.ndarray:
        """Compute Gaussian firing strengths in scaled space."""
        n_samples = X_scaled.shape[0]
        n_rules = len(self.ecm_.centers)
        strengths = np.zeros((n_samples, n_rules))
        for r in range(n_rules):
            diff = X_scaled - self.ecm_.centers[r]
            sigma_sq = self.sigmas_[r] ** 2
            strengths[:, r] = np.exp(-0.5 * np.sum(diff ** 2 / sigma_sq, axis=1))
        return strengths

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["ecm_", "consequent_params_"])
        X = check_array(X, dtype=np.float64)
        X_scaled = self.scaler_.transform(X)
        strengths = self._firing_strengths(X_scaled)
        norm = strengths / (strengths.sum(axis=1, keepdims=True) + 1e-10)

        if self.order == 0:
            return norm @ self.consequent_params_
        else:
            X_aug = np.column_stack([X, np.ones(len(X))])
            consequents = X_aug @ self.consequent_params_.T  # (n_samples, n_rules)
            return np.sum(norm * consequents, axis=1)

    def partial_fit(self, X: Any, y: Any) -> DENFISRegressor:
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
        if not hasattr(self, "ecm_"):
            return self.fit(X, y)

        X_scaled = self.scaler_.transform(X)
        n_features = X.shape[1]

        for i in range(len(X)):
            old_n_rules = len(self.ecm_.centers)
            cluster_idx = self.ecm_.update(X_scaled[i])

            # If new rule was created, expand parameters
            if len(self.ecm_.centers) > old_n_rules:
                if self.order == 0:
                    self.consequent_params_ = np.append(
                        self.consequent_params_, y[i]
                    )
                else:
                    new_params = np.zeros(n_features + 1)
                    new_params[-1] = y[i]
                    self.consequent_params_ = np.vstack([
                        self.consequent_params_, new_params
                    ])
                new_sigma = np.full(n_features, 0.5 * self.sigma_scale)
                self.sigmas_ = np.vstack([self.sigmas_, new_sigma])
            else:
                # Simple online update of consequent
                strengths = self._firing_strengths(X_scaled[i:i+1]).ravel()
                norm = strengths / (strengths.sum() + 1e-10)
                y_pred = self._predict_single(X[i], norm)
                error = y[i] - y_pred
                lr = 0.01
                if self.order == 0:
                    self.consequent_params_[cluster_idx] += lr * error
                else:
                    x_aug = np.append(X[i], 1.0)
                    self.consequent_params_[cluster_idx] += (
                        lr * error * norm[cluster_idx] * x_aug
                    )

        self.n_rules_ = len(self.ecm_.centers)
        return self

    def _predict_single(self, x: np.ndarray, norm_strengths: np.ndarray) -> float:
        if self.order == 0:
            return float(np.dot(norm_strengths, self.consequent_params_))
        else:
            x_aug = np.append(x, 1.0)
            consequents = self.consequent_params_ @ x_aug
            return float(np.dot(norm_strengths, consequents))


class DENFISClassifier(BaseEstimator, ClassifierMixin):
    """DENFIS for classification.

    Creates one DENFIS regressor per class (one-vs-rest) and predicts
    based on highest output.

    Parameters
    ----------
    distance_threshold : float, default=0.3
        Threshold for creating new clusters/rules.
    max_rules : int, default=100
        Maximum rules per class.
    sigma_scale : float, default=1.5
        Scaling for Gaussian widths.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print progress.
    """

    def __init__(
        self,
        distance_threshold: float = 0.3,
        max_rules: int = 100,
        sigma_scale: float = 1.5,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.distance_threshold = distance_threshold
        self.max_rules = max_rules
        self.sigma_scale = sigma_scale
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> DENFISClassifier:
        """Fit one DENFIS per class.

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
        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        self.regressors_ = []
        for c in range(self.n_classes_):
            y_binary = (y_enc == c).astype(np.float64)
            reg = DENFISRegressor(
                distance_threshold=self.distance_threshold,
                max_rules=self.max_rules,
                sigma_scale=self.sigma_scale,
                order=0,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary)
            self.regressors_.append(reg)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["regressors_"])
        X = check_array(X, dtype=np.float64)
        outputs = np.column_stack([r.predict(X) for r in self.regressors_])
        # Softmax normalization
        exp_out = np.exp(outputs - outputs.max(axis=1, keepdims=True))
        return exp_out / (exp_out.sum(axis=1, keepdims=True) + 1e-10)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
