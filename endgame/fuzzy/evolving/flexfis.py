"""FLEXFIS — Flexible Fuzzy Inference System.

Implements an incremental fuzzy regression system with generalized smart
rule evolution based on an extended version of eClustering for structure
identification and local learning for consequent adaptation.

References
----------
- Lughofer, E. (2008). FLEXFIS: A robust incremental learning approach
  for evolving Takagi-Sugeno fuzzy models. IEEE Trans. Fuzzy Systems,
  16(6), 1393-1410.
- Lughofer, E. (2011). Evolving Fuzzy Systems - Methodologies, Advanced
  Concepts and Applications. Springer.

Example
-------
>>> from endgame.fuzzy.evolving.flexfis import FLEXFISRegressor
>>> import numpy as np
>>> X = np.random.randn(200, 3)
>>> y = X @ np.array([1.0, -2.0, 0.5]) + 1.0
>>> model = FLEXFISRegressor(max_rules=30, vigilance=0.5)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FLEXFISRegressor(BaseEstimator, RegressorMixin):
    """Flexible Fuzzy Inference System for incremental regression.

    Uses an extended eClustering algorithm for online structure
    identification (adding/merging/splitting rules) and local weighted
    least squares for consequent parameter adaptation.

    Parameters
    ----------
    max_rules : int, default=50
        Maximum number of fuzzy rules.
    vigilance : float, default=0.5
        Vigilance parameter controlling cluster granularity. Lower values
        create fewer, broader rules; higher values create more, tighter
        rules. Range: (0, 1).
    lr : float, default=0.01
        Learning rate for consequent parameter updates.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Cluster centers (rule antecedent focal points).
    radii_ : ndarray of shape (n_rules,)
        Cluster radii for each rule.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        TSK consequent parameters (linear weights + bias) per rule.
    n_features_in_ : int
        Number of features seen during fit.
    n_rules_ : int
        Current number of rules.
    n_samples_seen_ : int
        Total samples processed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.flexfis import FLEXFISRegressor
    >>> X = np.random.randn(300, 4)
    >>> y = np.sin(X[:, 0]) + X[:, 1] ** 2
    >>> model = FLEXFISRegressor(max_rules=20, vigilance=0.4)
    >>> model.fit(X, y)
    FLEXFISRegressor(max_rules=20, vigilance=0.4)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        max_rules: int = 50,
        vigilance: float = 0.5,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.max_rules = max_rules
        self.vigilance = vigilance
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def _init_state(self, n_features: int) -> None:
        """Initialize internal state.

        Parameters
        ----------
        n_features : int
            Number of input features.
        """
        self.n_features_in_ = n_features
        self.centers_ = None  # (n_rules, n_features)
        self.radii_ = None  # (n_rules,)
        self.consequent_params_ = None  # (n_rules, n_features + 1)
        self._P_matrices_ = []  # RLS covariance matrices
        self._cluster_counts_ = []  # sample counts per cluster
        self.n_samples_seen_ = 0
        self._global_mean_ = np.zeros(n_features)
        self._global_var_ = np.zeros(n_features)

    def _compute_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized firing strengths using Gaussian antecedents.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
            Normalized firing strengths.
        """
        n_samples = X.shape[0]
        n_rules = self.centers_.shape[0]
        strengths = np.zeros((n_samples, n_rules))

        for i in range(n_rules):
            diff = X - self.centers_[i]
            r_sq = self.radii_[i] ** 2 + 1e-10
            dist_sq = np.sum(diff ** 2, axis=1)
            strengths[:, i] = np.exp(-dist_sq / (2.0 * r_sq))

        total = np.sum(strengths, axis=1, keepdims=True)
        total = np.maximum(total, 1e-10)
        return strengths / total

    def _eclustering_potential(self, x: np.ndarray) -> float:
        """Compute eClustering potential (Cauchy-type density).

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input point.

        Returns
        -------
        float
            Potential value.
        """
        if self.n_samples_seen_ < 2:
            return 1.0

        diff = x - self._global_mean_
        var_sum = np.sum(self._global_var_) + 1e-10
        return 1.0 / (1.0 + np.sum(diff ** 2) / var_sum)

    def _find_winner(self, x: np.ndarray) -> tuple[int, float]:
        """Find the winning (closest) cluster for a data point.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input point.

        Returns
        -------
        tuple of (int, float)
            Index of the winning cluster and the distance to it.
        """
        dists = np.sqrt(np.sum((self.centers_ - x) ** 2, axis=1))
        winner = np.argmin(dists)
        return int(winner), float(dists[winner])

    def _add_rule(self, x: np.ndarray, y: float) -> None:
        """Add a new rule centered at the given data point.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Center for the new rule.
        y : float
            Target value for consequent initialization.
        """
        n_ext = self.n_features_in_ + 1

        # Default radius based on vigilance and global variance
        default_radius = (1.0 - self.vigilance) * (
            np.sqrt(np.sum(self._global_var_)) + 0.1
        )
        default_radius = max(default_radius, 0.01)

        if self.centers_ is None:
            self.centers_ = x.reshape(1, -1).copy()
            self.radii_ = np.array([default_radius])
            params = np.zeros((1, n_ext))
            params[0, -1] = y  # bias = target
            self.consequent_params_ = params
        else:
            self.centers_ = np.vstack([self.centers_, x.reshape(1, -1)])
            self.radii_ = np.append(self.radii_, default_radius)
            new_params = np.zeros(n_ext)
            new_params[-1] = y
            self.consequent_params_ = np.vstack(
                [self.consequent_params_, new_params]
            )

        self._P_matrices_.append(np.eye(n_ext) * 1000.0)
        self._cluster_counts_.append(1)

        if self.verbose:
            n = self.centers_.shape[0]
            print(f"[FLEXFIS] Added rule {n}. Total: {n}")

    def _merge_closest_rules(self) -> None:
        """Merge the two closest rules if the rule base exceeds max_rules."""
        if self.centers_ is None or self.centers_.shape[0] <= self.max_rules:
            return

        n = self.centers_.shape[0]
        min_dist = np.inf
        merge_i, merge_j = 0, 1

        for i in range(n):
            for j in range(i + 1, n):
                d = np.sqrt(np.sum((self.centers_[i] - self.centers_[j]) ** 2))
                if d < min_dist:
                    min_dist = d
                    merge_i, merge_j = i, j

        # Weighted merge
        ci = self._cluster_counts_[merge_i]
        cj = self._cluster_counts_[merge_j]
        w_total = ci + cj

        self.centers_[merge_i] = (
            ci * self.centers_[merge_i] + cj * self.centers_[merge_j]
        ) / w_total
        self.consequent_params_[merge_i] = (
            ci * self.consequent_params_[merge_i]
            + cj * self.consequent_params_[merge_j]
        ) / w_total
        self.radii_[merge_i] = max(self.radii_[merge_i], self.radii_[merge_j])
        self._cluster_counts_[merge_i] = w_total
        self._P_matrices_[merge_i] = (
            self._P_matrices_[merge_i] + self._P_matrices_[merge_j]
        ) / 2.0

        # Remove rule merge_j
        self.centers_ = np.delete(self.centers_, merge_j, axis=0)
        self.consequent_params_ = np.delete(
            self.consequent_params_, merge_j, axis=0
        )
        self.radii_ = np.delete(self.radii_, merge_j)
        del self._P_matrices_[merge_j]
        del self._cluster_counts_[merge_j]

        if self.verbose:
            print(
                f"[FLEXFIS] Merged rules {merge_i} and {merge_j}. "
                f"Remaining: {self.centers_.shape[0]}"
            )

    def _update_consequent_rls(
        self, rule_idx: int, x_ext: np.ndarray, y: float, firing: float
    ) -> None:
        """Update consequent parameters using weighted RLS.

        Parameters
        ----------
        rule_idx : int
            Index of the rule.
        x_ext : ndarray of shape (n_features + 1,)
            Extended input [x, 1].
        y : float
            Target value.
        firing : float
            Normalized firing strength.
        """
        if firing < 1e-10:
            return

        P = self._P_matrices_[rule_idx]
        theta = self.consequent_params_[rule_idx]

        x_w = x_ext * firing
        error = y - x_ext @ theta

        Px = P @ x_w
        denom = 1.0 + x_w @ Px
        K = Px / denom

        self.consequent_params_[rule_idx] = theta + K * error
        self._P_matrices_[rule_idx] = P - np.outer(K, x_w @ P)

    def _process_sample(self, x: np.ndarray, y: float) -> None:
        """Process a single sample through eClustering and RLS update.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features.
        y : float
            Target value.
        """
        self.n_samples_seen_ += 1

        # Update global statistics
        if self.n_samples_seen_ == 1:
            self._global_mean_ = x.copy()
            self._global_var_ = np.zeros_like(x)
        else:
            old_mean = self._global_mean_.copy()
            n = self.n_samples_seen_
            self._global_mean_ = old_mean + (x - old_mean) / n
            if n > 2:
                self._global_var_ = (
                    self._global_var_ * (n - 2) / (n - 1)
                    + (x - old_mean) * (x - self._global_mean_) / (n - 1)
                )
            else:
                self._global_var_ = (x - old_mean) * (x - self._global_mean_)

        if self.centers_ is None:
            self._add_rule(x, y)
            return

        # eClustering: find winner and check vigilance
        winner, win_dist = self._find_winner(x)
        potential = self._eclustering_potential(x)

        # Create new rule if point is far from winner relative to its radius
        # and the potential is significant
        vigilance_dist = self.radii_[winner] * (1.0 / (self.vigilance + 1e-10))
        if win_dist > vigilance_dist and potential > 0.1:
            self._add_rule(x, y)
            self._merge_closest_rules()  # enforce max_rules
        else:
            # Update winner cluster
            count = self._cluster_counts_[winner]
            new_count = count + 1
            self.centers_[winner] = (
                self.centers_[winner] * count + x
            ) / new_count
            # Adaptive radius
            self.radii_[winner] = max(
                self.radii_[winner],
                win_dist * 0.5,
            )
            self._cluster_counts_[winner] = new_count

        # Update consequent parameters for all active rules
        x_ext = np.append(x, 1.0)
        strengths = self._compute_firing_strengths(x.reshape(1, -1))[0]

        for i in range(self.centers_.shape[0]):
            self._update_consequent_rls(i, x_ext, y, strengths[i])

    def fit(self, X: Any, y: Any) -> FLEXFISRegressor:
        """Fit the FLEXFIS model by processing all samples sequentially.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self._init_state(X.shape[1])

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X.shape[0])

        for idx in indices:
            self._process_sample(X[idx], y[idx])

        return self

    def partial_fit(self, X: Any, y: Any) -> FLEXFISRegressor:
        """Incrementally update the model with new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New input data.
        y : array-like of shape (n_samples,)
            New target values.

        Returns
        -------
        self
            Updated estimator.
        """
        X, y = check_X_y(X, y)

        if not hasattr(self, "n_features_in_"):
            self._init_state(X.shape[1])

        for i in range(X.shape[0]):
            self._process_sample(X[i], y[i])

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using TSK inference.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["centers_", "consequent_params_"])
        X = check_array(X)

        strengths = self._compute_firing_strengths(X)
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])

        # TSK output: weighted average of linear consequents
        rule_outputs = X_ext @ self.consequent_params_.T  # (n_samples, n_rules)
        predictions = np.sum(strengths * rule_outputs, axis=1)

        return predictions

    @property
    def n_rules_(self) -> int:
        """Number of rules in the current rule base."""
        check_is_fitted(self, ["centers_"])
        return self.centers_.shape[0]
