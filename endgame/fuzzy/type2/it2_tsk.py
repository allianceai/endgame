"""Interval Type-2 TSK (Takagi-Sugeno-Kang) Fuzzy System.

Antecedents are IT2 membership functions; consequents are crisp linear
functions of the inputs. Type reduction via Karnik-Mendel or Nie-Tan.

Example
-------
>>> from endgame.fuzzy.type2.it2_tsk import IT2TSKRegressor
>>> import numpy as np
>>> X = np.random.randn(200, 3)
>>> y = X @ [1, -0.5, 2] + 0.1 * np.random.randn(200)
>>> model = IT2TSKRegressor(n_rules=5, n_mfs=3, order=1)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.type2.it2_fls import (
    _create_it2_mfs,
    karnik_mendel,
    nie_tan,
)


class IT2TSKRegressor(BaseEstimator, RegressorMixin):
    """Interval Type-2 TSK Fuzzy System for regression.

    Antecedents use IT2 membership functions with Footprint of Uncertainty.
    Consequents are crisp linear functions: y_r = a_r^T x + b_r (order=1)
    or constants y_r = b_r (order=0).

    For each rule, upper and lower firing strengths are computed from the
    product of upper/lower MF values. Type reduction (KM or NT) combines
    the dual firing strengths and consequent outputs into a crisp value.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    order : int, default=1
        TSK order: 0 for constant consequents, 1 for linear consequents.
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
    mfs_ : list of list
        IT2 membership functions per feature.
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        MF index assignments for each rule.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1) or (n_rules, 1)
        Linear (or constant) consequent parameters per rule.

    Examples
    --------
    >>> from endgame.fuzzy.type2.it2_tsk import IT2TSKRegressor
    >>> import numpy as np
    >>> X = np.random.randn(150, 2)
    >>> y = 3 * X[:, 0] - X[:, 1] + 1
    >>> reg = IT2TSKRegressor(n_rules=5, order=1, n_mfs=3)
    >>> reg.fit(X, y)
    IT2TSKRegressor(n_mfs=3, n_rules=5)
    >>> reg.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_rules: int = 10,
        order: int = 1,
        n_mfs: int = 3,
        type_reduction: str = "km",
        mf_type: str = "gaussian",
        fou_factor: float = 0.3,
        random_state: int | None = None,
    ):
        self.n_rules = n_rules
        self.order = order
        self.n_mfs = n_mfs
        self.type_reduction = type_reduction
        self.mf_type = mf_type
        self.fou_factor = fou_factor
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> IT2TSKRegressor:
        """Fit the IT2 TSK system.

        Initializes IT2 membership functions from data, assigns antecedent
        indices, and solves for consequent parameters via weighted LSE.

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
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        rng = np.random.RandomState(self.random_state)

        # Initialize IT2 MFs
        self.mfs_ = _create_it2_mfs(
            X, self.n_mfs, mf_type=self.mf_type, fou_factor=self.fou_factor
        )

        # Assign antecedent indices
        self.antecedent_indices_ = rng.randint(
            0, self.n_mfs, size=(self.n_rules, n_features)
        )

        # Compute dual firing strengths
        firing_upper, firing_lower = self._compute_dual_firing(X)
        avg_firing = (firing_upper + firing_lower) / 2.0  # (n_samples, n_rules)

        # Solve consequent parameters via weighted LSE
        if self.order == 0:
            # Constant consequents: one parameter per rule
            self.consequent_params_ = np.zeros((self.n_rules, 1))
            for r in range(self.n_rules):
                w = avg_firing[:, r]
                w_sum = np.sum(w)
                if w_sum > 1e-12:
                    self.consequent_params_[r, 0] = np.dot(w, y) / w_sum
                else:
                    self.consequent_params_[r, 0] = np.mean(y)
        elif self.order == 1:
            # Linear consequents: [a1, a2, ..., an, b] per rule
            self.consequent_params_ = np.zeros((self.n_rules, n_features + 1))
            X_ext = np.column_stack([X, np.ones(n_samples)])  # augmented

            for r in range(self.n_rules):
                w = avg_firing[:, r]
                w_sum = np.sum(w)
                if w_sum > 1e-12:
                    # Weighted least squares: (X^T W X)^-1 X^T W y
                    W = np.diag(np.sqrt(w + 1e-12))
                    Xw = W @ X_ext
                    yw = W @ y
                    try:
                        params, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
                        self.consequent_params_[r] = params
                    except np.linalg.LinAlgError:
                        self.consequent_params_[r, -1] = np.dot(w, y) / w_sum
                else:
                    self.consequent_params_[r, -1] = np.mean(y)
        else:
            raise ValueError(f"order must be 0 or 1, got {self.order}")

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
        check_is_fitted(self, ["mfs_", "antecedent_indices_", "consequent_params_"])
        X = check_array(X, dtype=np.float64)
        n_samples = X.shape[0]

        firing_upper, firing_lower = self._compute_dual_firing(X)

        # Compute consequent outputs for each rule and sample
        consequent_outputs = self._compute_consequents(X)  # (n_samples, n_rules)

        predictions = np.zeros(n_samples)
        for i in range(n_samples):
            f_upper = firing_upper[i]
            f_lower = firing_lower[i]
            c_vals = consequent_outputs[i]

            if self.type_reduction == "km":
                y_l, y_r = karnik_mendel(f_upper, f_lower, c_vals)
                predictions[i] = (y_l + y_r) / 2.0
            elif self.type_reduction == "nt":
                predictions[i] = nie_tan(f_upper, f_lower, c_vals)
            else:
                raise ValueError(
                    f"Unknown type_reduction: {self.type_reduction}. "
                    "Choose 'km' or 'nt'."
                )

        return predictions

    def _compute_dual_firing(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute upper and lower firing strengths.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        tuple of (ndarray, ndarray), each of shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_features = X.shape[1]
        firing_upper = np.ones((n_samples, self.n_rules))
        firing_lower = np.ones((n_samples, self.n_rules))

        for r in range(self.n_rules):
            for j in range(n_features):
                mf_idx = self.antecedent_indices_[r, j]
                mf = self.mfs_[j][mf_idx]
                firing_upper[:, r] *= mf.upper(X[:, j])
                firing_lower[:, r] *= mf.lower(X[:, j])

        return firing_upper, firing_lower

    def _compute_consequents(self, X: np.ndarray) -> np.ndarray:
        """Compute consequent outputs for each rule and sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        outputs = np.zeros((n_samples, self.n_rules))

        if self.order == 0:
            for r in range(self.n_rules):
                outputs[:, r] = self.consequent_params_[r, 0]
        else:
            X_ext = np.column_stack([X, np.ones(n_samples)])
            for r in range(self.n_rules):
                outputs[:, r] = X_ext @ self.consequent_params_[r]

        return outputs
