"""Parsimonious Network-based Fuzzy Inference System (PANFIS).

Implements online fuzzy systems with automatic rule growing and pruning.
Rules are added when new data is distant from all existing rules and
pruned when their statistical contribution becomes insignificant.

References
----------
- Pratama, M., Anavatti, S. G., Angelov, P. P., & Lughofer, E. (2014).
  PANFIS: A novel incremental learning machine. IEEE Trans. Neural
  Networks and Learning Systems, 25(1), 55-68.

Example
-------
>>> from endgame.fuzzy.evolving.panfis import PANFISRegressor
>>> import numpy as np
>>> X = np.random.randn(100, 3)
>>> y = X @ np.array([1.0, -0.5, 2.0])
>>> model = PANFISRegressor(max_rules=20)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _PANFISBase(BaseEstimator):
    """Base class for PANFIS regressor and classifier.

    Parameters
    ----------
    max_rules : int, default=50
        Maximum number of fuzzy rules allowed.
    grow_threshold : float, default=0.3
        Minimum distance (normalized) to all existing rule centers
        required to create a new rule.
    prune_threshold : float, default=0.01
        Minimum contribution a rule must have to avoid pruning.
    lr : float, default=0.01
        Learning rate for consequent parameter gradient update.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        max_rules: int = 50,
        grow_threshold: float = 0.3,
        prune_threshold: float = 0.01,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.max_rules = max_rules
        self.grow_threshold = grow_threshold
        self.prune_threshold = prune_threshold
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def _init_state(self, n_features: int, n_outputs: int) -> None:
        """Initialize internal state for online learning.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_outputs : int
            Number of output dimensions.
        """
        self.n_features_in_ = n_features
        self._n_outputs = n_outputs
        self.centers_ = None  # (n_rules, n_features)
        self.spreads_ = None  # (n_rules, n_features)
        self.consequent_params_ = None  # (n_rules, n_features + 1, n_outputs)
        self._rule_ages_ = []
        self._rule_contributions_ = []
        self.n_samples_seen_ = 0
        self._data_range_ = None

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
            spread_sq = self.spreads_[i] ** 2 + 1e-10
            exponent = -0.5 * np.sum(diff ** 2 / spread_sq, axis=1)
            strengths[:, i] = np.exp(exponent)

        total = np.sum(strengths, axis=1, keepdims=True)
        total = np.maximum(total, 1e-10)
        return strengths / total

    def _compute_output(self, X: np.ndarray) -> np.ndarray:
        """Compute raw TSK output.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Raw output values.
        """
        strengths = self._compute_firing_strengths(X)
        X_ext = np.hstack([X, np.ones((X.shape[0], 1))])

        # Rule outputs: (n_samples, n_rules, n_outputs)
        rule_out = np.einsum("sf,rfo->sro", X_ext, self.consequent_params_)

        # Weighted sum
        output = np.einsum("sr,sro->so", strengths, rule_out)
        return output

    def _grow_check(self, x: np.ndarray) -> bool:
        """Check if a new rule should be created for this sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input sample.

        Returns
        -------
        bool
            True if a new rule should be added.
        """
        if self.centers_ is None:
            return True

        if self.centers_.shape[0] >= self.max_rules:
            return False

        # Normalized distance to closest center
        diffs = self.centers_ - x
        if self._data_range_ is not None:
            diffs = diffs / (self._data_range_ + 1e-10)
        dists = np.sqrt(np.sum(diffs ** 2, axis=1))
        min_dist = np.min(dists)

        return min_dist > self.grow_threshold

    def _add_rule(self, x: np.ndarray, y: np.ndarray) -> None:
        """Add a new rule centered at the given point.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Center for the new rule.
        y : ndarray of shape (n_outputs,)
            Target value for initializing the consequent.
        """
        n_ext = self.n_features_in_ + 1
        default_spread = self.grow_threshold * (
            self._data_range_ if self._data_range_ is not None
            else np.ones(self.n_features_in_)
        )
        default_spread = np.maximum(default_spread, 1e-4)

        if self.centers_ is None:
            self.centers_ = x.reshape(1, -1).copy()
            self.spreads_ = default_spread.reshape(1, -1)
            params = np.zeros((1, n_ext, self._n_outputs))
            # Initialize bias to target
            for o in range(self._n_outputs):
                params[0, -1, o] = y[o]
            self.consequent_params_ = params
        else:
            self.centers_ = np.vstack([self.centers_, x.reshape(1, -1)])
            self.spreads_ = np.vstack(
                [self.spreads_, default_spread.reshape(1, -1)]
            )
            new_params = np.zeros((1, n_ext, self._n_outputs))
            for o in range(self._n_outputs):
                new_params[0, -1, o] = y[o]
            self.consequent_params_ = np.concatenate(
                [self.consequent_params_, new_params], axis=0
            )

        self._rule_ages_.append(0)
        self._rule_contributions_.append(1.0)

        if self.verbose:
            print(
                f"[PANFIS] Added rule. Total: {self.centers_.shape[0]}"
            )

    def _prune_rules(self) -> None:
        """Remove rules with insignificant contribution."""
        if self.centers_ is None or self.centers_.shape[0] <= 1:
            return

        contributions = np.array(self._rule_contributions_)
        # Normalize contributions
        total = np.sum(contributions)
        if total > 0:
            norm_contrib = contributions / total
        else:
            return

        keep = norm_contrib >= self.prune_threshold
        # Always keep at least one rule
        if np.sum(keep) < 1:
            keep[np.argmax(norm_contrib)] = True

        if np.all(keep):
            return

        n_removed = np.sum(~keep)
        keep_idx = np.where(keep)[0]

        self.centers_ = self.centers_[keep_idx]
        self.spreads_ = self.spreads_[keep_idx]
        self.consequent_params_ = self.consequent_params_[keep_idx]
        self._rule_ages_ = [self._rule_ages_[i] for i in keep_idx]
        self._rule_contributions_ = [
            self._rule_contributions_[i] for i in keep_idx
        ]

        if self.verbose:
            print(
                f"[PANFIS] Pruned {n_removed} rules. "
                f"Remaining: {self.centers_.shape[0]}"
            )

    def _update_sample(self, x: np.ndarray, y: np.ndarray) -> None:
        """Process a single sample: grow, update, prune.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features.
        y : ndarray of shape (n_outputs,)
            Target output.
        """
        self.n_samples_seen_ += 1

        # Update data range
        if self._data_range_ is None:
            self._data_min_ = x.copy()
            self._data_max_ = x.copy()
            self._data_range_ = np.ones_like(x)
        else:
            self._data_min_ = np.minimum(self._data_min_, x)
            self._data_max_ = np.maximum(self._data_max_, x)
            self._data_range_ = np.maximum(
                self._data_max_ - self._data_min_, 1e-10
            )

        # Growing phase
        if self._grow_check(x):
            self._add_rule(x, y)

        # Update consequent parameters via gradient descent
        x_ext = np.append(x, 1.0)
        strengths = self._compute_firing_strengths(x.reshape(1, -1))[0]

        # Compute prediction error
        pred = self._compute_output(x.reshape(1, -1))[0]
        error = y - pred  # (n_outputs,)

        for i in range(self.centers_.shape[0]):
            w = strengths[i]
            if w < 1e-10:
                continue

            # Update consequent: gradient step
            # d(loss)/d(theta_i) = -2 * w * x_ext * error
            grad = np.outer(x_ext, error) * w
            self.consequent_params_[i] += self.lr * grad

            # Track contribution as exponential moving average of firing
            self._rule_contributions_[i] = (
                0.99 * self._rule_contributions_[i] + 0.01 * w
            )
            self._rule_ages_[i] += 1

            # Adapt center and spread (local learning)
            lr_struct = self.lr * 0.1
            diff = x - self.centers_[i]
            self.centers_[i] += lr_struct * w * diff
            self.spreads_[i] += lr_struct * w * (
                diff ** 2 / (self.spreads_[i] + 1e-10) - self.spreads_[i]
            )
            self.spreads_[i] = np.maximum(self.spreads_[i], 1e-4)

        # Pruning phase (every 50 samples to avoid excessive checking)
        if self.n_samples_seen_ % 50 == 0:
            self._prune_rules()


class PANFISRegressor(_PANFISBase, RegressorMixin):
    """Parsimonious ANFIS Regressor with online rule growing and pruning.

    Evolving fuzzy system that automatically adds rules when data falls
    outside existing rule coverage and removes rules that contribute
    insignificantly to the output.

    Parameters
    ----------
    max_rules : int, default=50
        Maximum number of fuzzy rules.
    grow_threshold : float, default=0.3
        Distance threshold for creating new rules.
    prune_threshold : float, default=0.01
        Minimum contribution threshold to keep a rule.
    lr : float, default=0.01
        Learning rate for consequent parameter updates.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Rule centers.
    spreads_ : ndarray of shape (n_rules, n_features)
        Gaussian spread per feature per rule.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1, 1)
        TSK consequent parameters.
    n_features_in_ : int
        Number of features.
    n_rules_ : int
        Current number of rules.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.panfis import PANFISRegressor
    >>> X = np.random.randn(200, 3)
    >>> y = X[:, 0] * 2 - X[:, 1] + 0.5
    >>> model = PANFISRegressor(max_rules=20, grow_threshold=0.3)
    >>> model.fit(X, y)
    PANFISRegressor(max_rules=20)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def fit(self, X: Any, y: Any) -> PANFISRegressor:
        """Fit the PANFIS regressor on training data.

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
        self._init_state(X.shape[1], n_outputs=1)

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X.shape[0])

        for idx in indices:
            self._update_sample(X[idx], y[idx : idx + 1])

        return self

    def partial_fit(self, X: Any, y: Any) -> PANFISRegressor:
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
            self._init_state(X.shape[1], n_outputs=1)

        for i in range(X.shape[0]):
            self._update_sample(X[i], y[i : i + 1])

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
        check_is_fitted(self, ["centers_", "consequent_params_"])
        X = check_array(X)
        output = self._compute_output(X)
        return output[:, 0]

    @property
    def n_rules_(self) -> int:
        """Number of rules in the current rule base."""
        check_is_fitted(self, ["centers_"])
        return self.centers_.shape[0]


class PANFISClassifier(_PANFISBase, ClassifierMixin):
    """Parsimonious ANFIS Classifier with online rule growing and pruning.

    Evolving fuzzy classifier that maintains a separate fuzzy output
    per class, with automatic structure adaptation.

    Parameters
    ----------
    max_rules : int, default=50
        Maximum number of fuzzy rules.
    grow_threshold : float, default=0.3
        Distance threshold for creating new rules.
    prune_threshold : float, default=0.01
        Minimum contribution threshold to keep a rule.
    lr : float, default=0.01
        Learning rate for consequent parameter updates.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Rule centers.
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    n_rules_ : int
        Current number of rules.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.panfis import PANFISClassifier
    >>> X = np.random.randn(200, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> model = PANFISClassifier(max_rules=20)
    >>> model.fit(X, y)
    PANFISClassifier(max_rules=20)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def fit(self, X: Any, y: Any) -> PANFISClassifier:
        """Fit the PANFIS classifier on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)

        self._label_encoder_ = LabelEncoder()
        y_enc = self._label_encoder_.fit_transform(y)
        self.classes_ = self._label_encoder_.classes_
        n_classes = len(self.classes_)

        self._init_state(X.shape[1], n_outputs=n_classes)

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X.shape[0])

        for idx in indices:
            # One-hot encode target
            target = np.zeros(n_classes)
            target[y_enc[idx]] = 1.0
            self._update_sample(X[idx], target)

        return self

    def partial_fit(
        self, X: Any, y: Any, classes: Any = None
    ) -> PANFISClassifier:
        """Incrementally update the classifier with new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New input data.
        y : array-like of shape (n_samples,)
            New class labels.
        classes : array-like, optional
            All possible class labels. Required on first call if not
            all classes are present in the first batch.

        Returns
        -------
        self
            Updated estimator.
        """
        X, y = check_X_y(X, y)

        if not hasattr(self, "classes_"):
            if classes is not None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = np.unique(y)
            self._label_encoder_ = LabelEncoder()
            self._label_encoder_.fit(self.classes_)
            self._init_state(X.shape[1], n_outputs=len(self.classes_))

        y_enc = self._label_encoder_.transform(y)
        n_classes = len(self.classes_)

        for i in range(X.shape[0]):
            target = np.zeros(n_classes)
            target[y_enc[i]] = 1.0
            self._update_sample(X[i], target)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities (softmax of raw outputs).
        """
        check_is_fitted(self, ["centers_", "consequent_params_"])
        X = check_array(X)
        raw = self._compute_output(X)

        # Softmax normalization
        exp_raw = np.exp(raw - np.max(raw, axis=1, keepdims=True))
        proba = exp_raw / np.sum(exp_raw, axis=1, keepdims=True)
        return proba

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    @property
    def n_rules_(self) -> int:
        """Number of rules in the current rule base."""
        check_is_fitted(self, ["centers_"])
        return self.centers_.shape[0]
