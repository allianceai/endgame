"""Evolving Takagi-Sugeno-Kang (eTS / eTS+) fuzzy systems.

Implements the eTS algorithm by Angelov & Filev (2004) and the improved
eTS+ with rule merging. Both systems evolve their rule base online using
the data cloud concept for structure identification and Recursive Least
Squares (RLS) for consequent parameter adaptation.

References
----------
- Angelov, P. & Filev, D. (2004). An approach to online identification of
  Takagi-Sugeno fuzzy models. IEEE Trans. SMC-B, 34(1), 484-498.
- Angelov, P. & Filev, D. (2005). Simpl_eTS: A simplified method for
  learning evolving Takagi-Sugeno fuzzy models. FUZZ-IEEE.

Example
-------
>>> from endgame.fuzzy.evolving.ets import EvolvingTSK
>>> import numpy as np
>>> X = np.random.randn(100, 3)
>>> y = X @ np.array([1.0, 2.0, 3.0]) + 0.5
>>> model = EvolvingTSK(radius=0.5)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class EvolvingTSK(BaseEstimator, RegressorMixin):
    """Evolving Takagi-Sugeno-Kang fuzzy system (eTS).

    Online TSK system that evolves its rule base using the data cloud
    concept. New rules are created when incoming data has higher
    potential (density) than existing rule centers. Consequent parameters
    are updated via Recursive Least Squares (RLS).

    Parameters
    ----------
    radius : float, default=0.5
        Zone of influence radius for rule creation. Smaller values create
        more rules; larger values create fewer, broader rules.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during fitting.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Centers (focal points) of evolved fuzzy rules.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        TSK consequent parameters for each rule (weights + bias).
    n_rules_ : int
        Number of rules in the current rule base.
    n_features_in_ : int
        Number of features seen during fit.
    n_samples_seen_ : int
        Total number of samples processed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.ets import EvolvingTSK
    >>> X = np.random.randn(200, 2)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    >>> model = EvolvingTSK(radius=0.4)
    >>> model.fit(X, y)
    EvolvingTSK(radius=0.4)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        radius: float = 0.5,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.radius = radius
        self.random_state = random_state
        self.verbose = verbose

    def _compute_potential(self, x: np.ndarray) -> float:
        """Compute Cauchy-type potential of a data point.

        The potential measures density of data around the point using
        all previously seen data (computed recursively).

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input data point.

        Returns
        -------
        float
            Potential value in (0, 1].
        """
        # Recursive potential calculation based on distances
        if self.n_samples_seen_ == 1:
            return 1.0

        # Update cumulative sums for recursive computation
        dist_sum = np.sum((x - self._data_mean_) ** 2)
        var_sum = np.sum(self._data_var_) + 1e-10
        potential = 1.0 / (1.0 + dist_sum / var_sum)
        return float(potential)

    def _compute_rule_potential(self, rule_idx: int) -> float:
        """Compute potential of an existing rule center.

        Parameters
        ----------
        rule_idx : int
            Index of the rule.

        Returns
        -------
        float
            Updated potential of the rule center.
        """
        center = self.centers_[rule_idx]
        dist_sum = np.sum((center - self._data_mean_) ** 2)
        var_sum = np.sum(self._data_var_) + 1e-10
        potential = 1.0 / (1.0 + dist_sum / var_sum)
        return float(potential)

    def _compute_firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Compute normalized firing strengths for all rules.

        Uses Gaussian-based antecedent with the rule centers and radius.

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

        r_sq = self.radius ** 2

        for i in range(n_rules):
            diff = X - self.centers_[i]
            dist_sq = np.sum(diff ** 2, axis=1)
            strengths[:, i] = np.exp(-dist_sq / (2.0 * r_sq + 1e-10))

        # Normalize
        total = np.sum(strengths, axis=1, keepdims=True)
        total = np.maximum(total, 1e-10)
        return strengths / total

    def _update_consequent_rls(
        self, rule_idx: int, x_ext: np.ndarray, y: float, firing: float
    ) -> None:
        """Update consequent parameters for a rule using weighted RLS.

        Parameters
        ----------
        rule_idx : int
            Index of the rule to update.
        x_ext : ndarray of shape (n_features + 1,)
            Extended input vector [x, 1].
        y : float
            Target value.
        firing : float
            Normalized firing strength for this rule.
        """
        if firing < 1e-10:
            return

        P = self._P_matrices_[rule_idx]
        theta = self.consequent_params_[rule_idx]

        # Weighted RLS update
        x_w = x_ext * firing
        y_hat = x_ext @ theta
        error = y - y_hat

        Px = P @ x_w
        denom = 1.0 + x_w @ Px
        K = Px / denom

        self.consequent_params_[rule_idx] = theta + K * error
        self._P_matrices_[rule_idx] = P - np.outer(K, x_w @ P)

    def _add_rule(self, x: np.ndarray, y: float) -> None:
        """Add a new rule centered at the given data point.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Center for the new rule.
        y : float
            Target value for initializing consequent.
        """
        n_ext = self.n_features_in_ + 1

        if self.centers_ is None:
            self.centers_ = x.reshape(1, -1).copy()
            self.consequent_params_ = np.zeros((1, n_ext))
            self.consequent_params_[0, -1] = y  # bias = y
            self._P_matrices_ = [np.eye(n_ext) * 1000.0]
            self._rule_potentials_ = [1.0]
        else:
            self.centers_ = np.vstack([self.centers_, x.reshape(1, -1)])
            new_params = np.zeros(n_ext)
            new_params[-1] = y
            self.consequent_params_ = np.vstack(
                [self.consequent_params_, new_params]
            )
            self._P_matrices_.append(np.eye(n_ext) * 1000.0)
            self._rule_potentials_.append(1.0)

        if self.verbose:
            print(
                f"[EvolvingTSK] Added rule {len(self._rule_potentials_)}. "
                f"Total rules: {len(self._rule_potentials_)}"
            )

    def _process_sample(self, x: np.ndarray, y: float) -> None:
        """Process a single data sample for online learning.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features.
        y : float
            Target value.
        """
        self.n_samples_seen_ += 1

        # Update running statistics for potential computation
        if self.n_samples_seen_ == 1:
            self._data_mean_ = x.copy()
            self._data_var_ = np.zeros_like(x)
        else:
            old_mean = self._data_mean_.copy()
            self._data_mean_ = (
                old_mean * (self.n_samples_seen_ - 1) + x
            ) / self.n_samples_seen_
            self._data_var_ = (
                self._data_var_ * (self.n_samples_seen_ - 2)
                / max(self.n_samples_seen_ - 1, 1)
                + (x - old_mean) * (x - self._data_mean_)
            ) / max(self.n_samples_seen_ - 1, 1) if self.n_samples_seen_ > 2 else (
                (x - old_mean) * (x - self._data_mean_)
            )

        # Compute potential of the new sample
        new_potential = self._compute_potential(x)

        if self.centers_ is None:
            # First sample always creates a rule
            self._add_rule(x, y)
            return

        # Update potentials of existing rule centers
        max_existing_potential = -np.inf
        for i in range(len(self._rule_potentials_)):
            self._rule_potentials_[i] = self._compute_rule_potential(i)
            max_existing_potential = max(
                max_existing_potential, self._rule_potentials_[i]
            )

        # Check if the new sample should create a new rule
        # Condition: new sample potential is higher than all existing
        # OR the new sample is far from all existing centers
        min_dist = np.min(
            np.sqrt(np.sum((self.centers_ - x) ** 2, axis=1))
        )

        if new_potential > max_existing_potential or min_dist > self.radius:
            self._add_rule(x, y)
        else:
            # Update the closest rule center (optional: shift toward new data)
            closest = np.argmin(
                np.sum((self.centers_ - x) ** 2, axis=1)
            )
            # Gradual center adaptation
            lr = 1.0 / self.n_samples_seen_
            self.centers_[closest] += lr * (x - self.centers_[closest])

        # Update consequent parameters via RLS for all active rules
        x_ext = np.append(x, 1.0)
        strengths = self._compute_firing_strengths(x.reshape(1, -1))[0]

        for i in range(len(self._rule_potentials_)):
            self._update_consequent_rls(i, x_ext, y, strengths[i])

    def fit(self, X: Any, y: Any) -> EvolvingTSK:
        """Fit the eTS model by processing all samples sequentially.

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
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = 0
        self.centers_ = None
        self.consequent_params_ = None
        self._P_matrices_ = []
        self._rule_potentials_ = []
        self._data_mean_ = None
        self._data_var_ = None

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X.shape[0])

        for idx in indices:
            self._process_sample(X[idx], y[idx])

        return self

    def partial_fit(self, X: Any, y: Any) -> EvolvingTSK:
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
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = 0
            self.centers_ = None
            self.consequent_params_ = None
            self._P_matrices_ = []
            self._rule_potentials_ = []
            self._data_mean_ = None
            self._data_var_ = None

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


class EvolvingTSKPlus(EvolvingTSK):
    """Improved evolving TSK fuzzy system (eTS+).

    Extends EvolvingTSK with rule merging when rule centers become
    similar, and improved antecedent management through local learning.

    Parameters
    ----------
    radius : float, default=0.5
        Zone of influence radius for rule creation.
    merge_threshold : float, default=0.5
        Distance threshold below which two rules are merged.
        Expressed as a fraction of the radius.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Centers of evolved fuzzy rules.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        TSK consequent parameters for each rule.
    n_rules_ : int
        Number of rules after merging.
    n_features_in_ : int
        Number of features seen during fit.
    n_merges_ : int
        Total number of rule merges performed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.ets import EvolvingTSKPlus
    >>> X = np.random.randn(200, 2)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    >>> model = EvolvingTSKPlus(radius=0.4, merge_threshold=0.3)
    >>> model.fit(X, y)
    EvolvingTSKPlus(merge_threshold=0.3, radius=0.4)
    >>> model.n_rules_
    5
    """

    def __init__(
        self,
        radius: float = 0.5,
        merge_threshold: float = 0.5,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(radius=radius, random_state=random_state, verbose=verbose)
        self.merge_threshold = merge_threshold

    def _merge_rules(self) -> None:
        """Merge rules whose centers are closer than merge_threshold * radius.

        The merged rule takes the weighted average of both centers and
        consequent parameters, weighted by their potentials.
        """
        if self.centers_ is None or self.centers_.shape[0] < 2:
            return

        threshold_dist = self.merge_threshold * self.radius
        merged = True

        while merged:
            merged = False
            n_rules = self.centers_.shape[0]

            for i in range(n_rules):
                for j in range(i + 1, n_rules):
                    dist = np.sqrt(
                        np.sum((self.centers_[i] - self.centers_[j]) ** 2)
                    )
                    if dist < threshold_dist:
                        # Merge j into i using potential-weighted average
                        pi = self._rule_potentials_[i]
                        pj = self._rule_potentials_[j]
                        w_total = pi + pj + 1e-10

                        self.centers_[i] = (
                            pi * self.centers_[i] + pj * self.centers_[j]
                        ) / w_total
                        self.consequent_params_[i] = (
                            pi * self.consequent_params_[i]
                            + pj * self.consequent_params_[j]
                        ) / w_total
                        self._rule_potentials_[i] = max(pi, pj)

                        # Average covariance matrices
                        self._P_matrices_[i] = (
                            self._P_matrices_[i] + self._P_matrices_[j]
                        ) / 2.0

                        # Remove rule j
                        self.centers_ = np.delete(self.centers_, j, axis=0)
                        self.consequent_params_ = np.delete(
                            self.consequent_params_, j, axis=0
                        )
                        del self._P_matrices_[j]
                        del self._rule_potentials_[j]

                        self.n_merges_ += 1
                        merged = True

                        if self.verbose:
                            print(
                                f"[eTS+] Merged rules {i} and {j}. "
                                f"Remaining: {self.centers_.shape[0]}"
                            )
                        break
                if merged:
                    break

    def _process_sample(self, x: np.ndarray, y: float) -> None:
        """Process sample with post-merge step.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features.
        y : float
            Target value.
        """
        super()._process_sample(x, y)
        self._merge_rules()

    def fit(self, X: Any, y: Any) -> EvolvingTSKPlus:
        """Fit the eTS+ model with rule merging.

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
        self.n_merges_ = 0
        super().fit(X, y)
        return self

    def partial_fit(self, X: Any, y: Any) -> EvolvingTSKPlus:
        """Incrementally update with rule merging.

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
        if not hasattr(self, "n_merges_"):
            self.n_merges_ = 0
        super().partial_fit(X, y)
        return self
