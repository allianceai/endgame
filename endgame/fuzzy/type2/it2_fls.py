"""Interval Type-2 Fuzzy Logic System (IT2 FLS).

Implements Mamdani-style IT2 FLS with Footprint of Uncertainty (FOU)
and Karnik-Mendel type reduction algorithm.

Key differences from Type-1 FLS:
- Each membership function has upper and lower bounds (FOU)
- Dual firing strengths (upper and lower) for each rule
- Type reduction via Karnik-Mendel or Nie-Tan to collapse IT2 to Type-1
- Final crisp output = (y_l + y_r) / 2

Example
-------
>>> from endgame.fuzzy.type2.it2_fls import IT2FLSRegressor
>>> import numpy as np
>>> X = np.random.randn(100, 3)
>>> y = X @ [1, 2, 3] + 0.1 * np.random.randn(100)
>>> model = IT2FLSRegressor(n_rules=5, n_mfs=3)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.membership import (
    IntervalType2GaussianMF,
    IntervalType2TriangularMF,
    create_uniform_mfs,
)
from endgame.fuzzy.core.operators import ProductTNorm


def karnik_mendel(
    firing_upper: np.ndarray,
    firing_lower: np.ndarray,
    consequents: np.ndarray,
    max_iter: int = 100,
) -> tuple[float, float]:
    """Karnik-Mendel algorithm for type reduction.

    Computes the interval [y_l, y_r] by finding switch points in the
    sorted consequent space where upper/lower firing strengths switch.

    Parameters
    ----------
    firing_upper : ndarray of shape (n_rules,)
        Upper firing strengths for each rule.
    firing_lower : ndarray of shape (n_rules,)
        Lower firing strengths for each rule.
    consequents : ndarray of shape (n_rules,)
        Consequent (output) values for each rule.
    max_iter : int, default=100
        Maximum iterations for convergence.

    Returns
    -------
    tuple of (float, float)
        (y_l, y_r) interval endpoints.
    """
    n = len(consequents)
    if n == 0:
        return 0.0, 0.0

    # Sort by consequent values
    sort_idx = np.argsort(consequents)
    c = consequents[sort_idx]
    f_upper = firing_upper[sort_idx]
    f_lower = firing_lower[sort_idx]

    # --- Compute y_l ---
    # Initialize with all upper firings
    weights = f_upper.copy()
    total_w = np.sum(weights)
    if total_w < 1e-12:
        return float(np.mean(c)), float(np.mean(c))

    y_l = float(np.dot(weights, c) / total_w)

    for _ in range(max_iter):
        # Find switch point L: largest index where c[i] <= y_l
        L = np.searchsorted(c, y_l, side="right") - 1
        L = np.clip(L, 0, n - 1)

        # Use upper firings for i <= L, lower firings for i > L
        new_weights = np.empty(n)
        new_weights[: L + 1] = f_upper[: L + 1]
        new_weights[L + 1 :] = f_lower[L + 1 :]

        total_new = np.sum(new_weights)
        if total_new < 1e-12:
            break
        y_l_new = float(np.dot(new_weights, c) / total_new)

        if abs(y_l_new - y_l) < 1e-10:
            y_l = y_l_new
            break
        y_l = y_l_new

    # --- Compute y_r ---
    weights = f_upper.copy()
    total_w = np.sum(weights)
    y_r = float(np.dot(weights, c) / total_w) if total_w > 1e-12 else float(np.mean(c))

    for _ in range(max_iter):
        # Find switch point R: smallest index where c[i] >= y_r
        R = np.searchsorted(c, y_r, side="left")
        R = np.clip(R, 0, n - 1)

        # Use lower firings for i <= R, upper firings for i > R
        new_weights = np.empty(n)
        new_weights[: R + 1] = f_lower[: R + 1]
        new_weights[R + 1 :] = f_upper[R + 1 :]

        total_new = np.sum(new_weights)
        if total_new < 1e-12:
            break
        y_r_new = float(np.dot(new_weights, c) / total_new)

        if abs(y_r_new - y_r) < 1e-10:
            y_r = y_r_new
            break
        y_r = y_r_new

    return y_l, y_r


def nie_tan(
    firing_upper: np.ndarray,
    firing_lower: np.ndarray,
    consequents: np.ndarray,
) -> float:
    """Nie-Tan type reduction method.

    A simpler, direct computation alternative to Karnik-Mendel.
    Output = weighted average using (upper + lower) / 2 as weights.

    Parameters
    ----------
    firing_upper : ndarray of shape (n_rules,)
        Upper firing strengths.
    firing_lower : ndarray of shape (n_rules,)
        Lower firing strengths.
    consequents : ndarray of shape (n_rules,)
        Consequent values.

    Returns
    -------
    float
        Crisp output value.
    """
    avg_firing = (firing_upper + firing_lower) / 2.0
    total = np.sum(avg_firing)
    if total < 1e-12:
        return float(np.mean(consequents)) if len(consequents) > 0 else 0.0
    return float(np.dot(avg_firing, consequents) / total)


def _create_it2_mfs(
    X: np.ndarray,
    n_mfs: int,
    mf_type: str = "gaussian",
    fou_factor: float = 0.3,
) -> list[list]:
    """Create IT2 membership functions from data statistics.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    n_mfs : int
        Number of MFs per feature.
    mf_type : str
        'gaussian' or 'triangular'.
    fou_factor : float
        Controls the width of the FOU (0 = Type-1, 1 = very uncertain).

    Returns
    -------
    list of list
        mfs[feature][mf_index] is an IT2 MF.
    """
    n_features = X.shape[1]
    all_mfs = []

    for j in range(n_features):
        x_min = float(np.min(X[:, j]))
        x_max = float(np.max(X[:, j]))
        padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
        x_min -= padding
        x_max += padding

        centers = np.linspace(x_min, x_max, n_mfs)
        step = (x_max - x_min) / max(n_mfs - 1, 1)
        feature_mfs = []

        if mf_type == "gaussian":
            sigma_base = step / 2.0
            sigma_base = max(sigma_base, 1e-6)
            for c in centers:
                sigma_lower = sigma_base * (1.0 - fou_factor * 0.5)
                sigma_upper = sigma_base * (1.0 + fou_factor * 0.5)
                sigma_lower = max(sigma_lower, 1e-6)
                feature_mfs.append(
                    IntervalType2GaussianMF(
                        center=float(c),
                        sigma_lower=sigma_lower,
                        sigma_upper=sigma_upper,
                    )
                )
        elif mf_type == "triangular":
            for i, c in enumerate(centers):
                a = float(c - step) if i > 0 else float(x_min - step)
                b = float(c)
                cc = float(c + step) if i < n_mfs - 1 else float(x_max + step)
                # Inner (lower) triangle is narrower
                shrink = step * fou_factor * 0.3
                a_lower = a + shrink
                c_lower = cc - shrink
                # Ensure validity
                a_lower = min(a_lower, b)
                c_lower = max(c_lower, b)
                feature_mfs.append(
                    IntervalType2TriangularMF(
                        a_lower=a_lower,
                        b_lower=b,
                        c_lower=c_lower,
                        a_upper=a,
                        b_upper=b,
                        c_upper=cc,
                    )
                )
        else:
            raise ValueError(f"Unknown mf_type for IT2: {mf_type}")

        all_mfs.append(feature_mfs)

    return all_mfs


class IT2FLSRegressor(BaseEstimator, RegressorMixin):
    """Interval Type-2 Fuzzy Logic System for regression.

    Uses IT2 membership functions with Footprint of Uncertainty (FOU).
    Type reduction via Karnik-Mendel or Nie-Tan algorithm produces
    an interval [y_l, y_r], and the final output is (y_l + y_r) / 2.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input variable.
    type_reduction : str, default='km'
        Type reduction method: 'km' (Karnik-Mendel) or 'nt' (Nie-Tan).
    mf_type : str, default='gaussian'
        MF type: 'gaussian' or 'triangular'.
    fou_factor : float, default=0.3
        Footprint of Uncertainty size (0=Type-1, higher=more uncertain).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    mfs_ : list of list
        IT2 membership functions per feature.
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        MF index assignments for each rule and feature.
    consequent_values_ : ndarray of shape (n_rules,)
        Consequent output values for each rule.

    Examples
    --------
    >>> from endgame.fuzzy.type2.it2_fls import IT2FLSRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0] + 2 * X[:, 1]
    >>> reg = IT2FLSRegressor(n_rules=5, n_mfs=3)
    >>> reg.fit(X, y)
    IT2FLSRegressor(n_mfs=3, n_rules=5)
    >>> reg.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        type_reduction: str = "km",
        mf_type: str = "gaussian",
        fou_factor: float = 0.3,
        random_state: int | None = None,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.type_reduction = type_reduction
        self.mf_type = mf_type
        self.fou_factor = fou_factor
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> IT2FLSRegressor:
        """Fit the IT2 FLS regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Initialize IT2 membership functions
        self.mfs_ = _create_it2_mfs(
            X, self.n_mfs, mf_type=self.mf_type, fou_factor=self.fou_factor
        )

        # Assign antecedent MF indices for each rule
        self.antecedent_indices_ = rng.randint(
            0, self.n_mfs, size=(self.n_rules, self.n_features_in_)
        )

        # Compute upper and lower firing strengths
        firing_upper, firing_lower = self._compute_dual_firing(X)

        # Solve for consequent values using weighted least squares
        avg_firing = (firing_upper + firing_lower) / 2.0
        total_firing = avg_firing.sum(axis=0)
        mask = total_firing > 1e-12
        self.consequent_values_ = np.zeros(self.n_rules)

        if mask.any():
            # Weighted mean of y for each rule
            for r in range(self.n_rules):
                w = avg_firing[:, r]
                w_sum = np.sum(w)
                if w_sum > 1e-12:
                    self.consequent_values_[r] = np.dot(w, y) / w_sum
                else:
                    self.consequent_values_[r] = np.mean(y)
        else:
            self.consequent_values_[:] = np.mean(y)

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
        check_is_fitted(self, ["mfs_", "antecedent_indices_", "consequent_values_"])
        X = check_array(X, dtype=np.float64)

        firing_upper, firing_lower = self._compute_dual_firing(X)
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            f_upper = firing_upper[i]
            f_lower = firing_lower[i]

            if self.type_reduction == "km":
                y_l, y_r = karnik_mendel(
                    f_upper, f_lower, self.consequent_values_
                )
                predictions[i] = (y_l + y_r) / 2.0
            elif self.type_reduction == "nt":
                predictions[i] = nie_tan(
                    f_upper, f_lower, self.consequent_values_
                )
            else:
                raise ValueError(
                    f"Unknown type_reduction: {self.type_reduction}. "
                    "Choose 'km' or 'nt'."
                )

        return predictions

    def _compute_dual_firing(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute upper and lower firing strengths for all rules.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        tuple of (ndarray, ndarray), each of shape (n_samples, n_rules)
            Upper and lower firing strengths.
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        firing_upper = np.ones((n_samples, self.n_rules))
        firing_lower = np.ones((n_samples, self.n_rules))

        for r in range(self.n_rules):
            for j in range(n_features):
                mf_idx = self.antecedent_indices_[r, j]
                mf = self.mfs_[j][mf_idx]
                upper_vals = mf.upper(X[:, j])
                lower_vals = mf.lower(X[:, j])
                firing_upper[:, r] *= upper_vals
                firing_lower[:, r] *= lower_vals

        return firing_upper, firing_lower


class IT2FLSClassifier(BaseEstimator, ClassifierMixin):
    """Interval Type-2 Fuzzy Logic System for classification.

    Uses IT2 membership functions with FOU and type reduction for
    multi-class classification.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules per class.
    n_mfs : int, default=3
        Number of membership functions per input variable.
    type_reduction : str, default='km'
        Type reduction method: 'km' (Karnik-Mendel) or 'nt' (Nie-Tan).
    mf_type : str, default='gaussian'
        MF type: 'gaussian' or 'triangular'.
    fou_factor : float, default=0.3
        Footprint of Uncertainty size.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : ndarray
        Unique class labels.
    regressors_ : dict
        One IT2FLSRegressor per class (one-vs-rest).

    Examples
    --------
    >>> from endgame.fuzzy.type2.it2_fls import IT2FLSClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = IT2FLSClassifier(n_rules=5, n_mfs=3)
    >>> clf.fit(X, y)
    IT2FLSClassifier(n_mfs=3, n_rules=5)
    >>> clf.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        type_reduction: str = "km",
        mf_type: str = "gaussian",
        fou_factor: float = 0.3,
        random_state: int | None = None,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.type_reduction = type_reduction
        self.mf_type = mf_type
        self.fou_factor = fou_factor
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> IT2FLSClassifier:
        """Fit the IT2 FLS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_encoded = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # One-vs-rest: train one regressor per class
        self.regressors_ = {}
        for c in range(self.n_classes_):
            y_binary = (y_encoded == c).astype(np.float64)
            reg = IT2FLSRegressor(
                n_rules=self.n_rules,
                n_mfs=self.n_mfs,
                type_reduction=self.type_reduction,
                mf_type=self.mf_type,
                fou_factor=self.fou_factor,
                random_state=self.random_state,
            )
            reg.fit(X, y_binary)
            self.regressors_[c] = reg

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
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities (softmax-normalized).
        """
        check_is_fitted(self, ["regressors_", "classes_"])
        X = check_array(X, dtype=np.float64)

        n_samples = X.shape[0]
        raw_scores = np.zeros((n_samples, self.n_classes_))

        for c in range(self.n_classes_):
            raw_scores[:, c] = self.regressors_[c].predict(X)

        # Softmax normalization
        exp_scores = np.exp(raw_scores - np.max(raw_scores, axis=1, keepdims=True))
        proba = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return proba
