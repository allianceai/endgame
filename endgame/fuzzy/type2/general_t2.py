"""General Type-2 Fuzzy Logic System via alpha-plane representation.

A General Type-2 FLS discretizes the third dimension (secondary membership)
into alpha levels. At each alpha level, an IT2 FLS operation is performed
with the corresponding alpha-cut. Results are aggregated across all alpha
planes to produce the final output.

This is computationally more expensive than IT2 FLS but captures richer
uncertainty through the secondary membership function.

Example
-------
>>> from endgame.fuzzy.type2.general_t2 import GeneralType2FLS
>>> import numpy as np
>>> X = np.random.randn(100, 3)
>>> y = X @ [1, 2, 3] + 0.1 * np.random.randn(100)
>>> model = GeneralType2FLS(n_rules=5, n_mfs=3, n_alpha_planes=11)
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
from endgame.fuzzy.core.membership import (
    IntervalType2GaussianMF,
    IntervalType2TriangularMF,
)


def _alpha_cut_it2_mf(
    mf: Any,
    x: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute alpha-cut of an IT2 membership function.

    At alpha level, the upper MF is kept as-is and the lower MF is
    interpolated between the lower MF and upper MF based on alpha:
        effective_lower = lower + alpha * (upper - lower)

    This shrinks the FOU at higher alpha levels, so that alpha=0 gives
    the full IT2 FOU and alpha=1 collapses to the upper MF (Type-1).

    Parameters
    ----------
    mf : IT2 membership function
        Must have .upper(x) and .lower(x) methods.
    x : ndarray of shape (n_samples,)
        Input values.
    alpha : float
        Alpha level in [0, 1].

    Returns
    -------
    tuple of (ndarray, ndarray)
        (upper_cut, lower_cut) membership values at this alpha level.
    """
    upper = mf.upper(x)
    lower = mf.lower(x)

    # At alpha=0: full FOU (upper, lower as-is)
    # At alpha=1: collapsed to upper MF only
    # The alpha-cut shrinks the lower bound upward
    lower_cut = lower + alpha * (upper - lower)
    upper_cut = upper.copy()

    return upper_cut, lower_cut


class GeneralType2FLS(BaseEstimator, RegressorMixin):
    """General Type-2 Fuzzy Logic System via alpha-plane representation.

    Discretizes the secondary membership dimension into alpha levels.
    At each alpha level, performs IT2 FLS operations (firing strength
    computation and type reduction). Final output is the weighted
    average across all alpha planes.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_alpha_planes : int, default=11
        Number of alpha levels (evenly spaced in [0, 1]).
        Higher values give more accurate but slower computation.
    type_reduction : str, default='km'
        Type reduction at each alpha level: 'km' or 'nt'.
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
    consequent_values_ : ndarray of shape (n_rules,)
        Consequent output values for each rule.
    alpha_levels_ : ndarray of shape (n_alpha_planes,)
        Alpha levels used for computation.

    Examples
    --------
    >>> from endgame.fuzzy.type2.general_t2 import GeneralType2FLS
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0] + 2 * X[:, 1]
    >>> model = GeneralType2FLS(n_rules=5, n_mfs=3, n_alpha_planes=11)
    >>> model.fit(X, y)
    GeneralType2FLS(n_mfs=3, n_rules=5)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        n_alpha_planes: int = 11,
        type_reduction: str = "km",
        mf_type: str = "gaussian",
        fou_factor: float = 0.3,
        random_state: int | None = None,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.n_alpha_planes = n_alpha_planes
        self.type_reduction = type_reduction
        self.mf_type = mf_type
        self.fou_factor = fou_factor
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> GeneralType2FLS:
        """Fit the General Type-2 FLS.

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

        # Alpha levels
        self.alpha_levels_ = np.linspace(0, 1, self.n_alpha_planes)

        # Initialize IT2 MFs
        self.mfs_ = _create_it2_mfs(
            X, self.n_mfs, mf_type=self.mf_type, fou_factor=self.fou_factor
        )

        # Assign antecedent indices
        self.antecedent_indices_ = rng.randint(
            0, self.n_mfs, size=(self.n_rules, self.n_features_in_)
        )

        # Compute average firing across alpha planes for consequent estimation
        avg_firing_all = np.zeros((X.shape[0], self.n_rules))
        for alpha in self.alpha_levels_:
            f_upper, f_lower = self._compute_alpha_firing(X, alpha)
            avg_firing_all += (f_upper + f_lower) / 2.0
        avg_firing_all /= self.n_alpha_planes

        # Solve consequent values
        self.consequent_values_ = np.zeros(self.n_rules)
        for r in range(self.n_rules):
            w = avg_firing_all[:, r]
            w_sum = np.sum(w)
            if w_sum > 1e-12:
                self.consequent_values_[r] = np.dot(w, y) / w_sum
            else:
                self.consequent_values_[r] = np.mean(y)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        At each alpha level, computes dual firing strengths and performs
        type reduction. The final output is the average across alpha planes.

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
        n_samples = X.shape[0]

        # Accumulate predictions across alpha planes
        predictions_sum = np.zeros(n_samples)

        for alpha in self.alpha_levels_:
            firing_upper, firing_lower = self._compute_alpha_firing(X, alpha)

            for i in range(n_samples):
                f_upper = firing_upper[i]
                f_lower = firing_lower[i]

                if self.type_reduction == "km":
                    y_l, y_r = karnik_mendel(
                        f_upper, f_lower, self.consequent_values_
                    )
                    predictions_sum[i] += (y_l + y_r) / 2.0
                elif self.type_reduction == "nt":
                    predictions_sum[i] += nie_tan(
                        f_upper, f_lower, self.consequent_values_
                    )
                else:
                    raise ValueError(
                        f"Unknown type_reduction: {self.type_reduction}. "
                        "Choose 'km' or 'nt'."
                    )

        # Average across alpha planes
        predictions = predictions_sum / self.n_alpha_planes
        return predictions

    def _compute_alpha_firing(
        self, X: np.ndarray, alpha: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute dual firing strengths at a given alpha level.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        alpha : float
            Alpha level in [0, 1].

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
                upper_cut, lower_cut = _alpha_cut_it2_mf(mf, X[:, j], alpha)
                firing_upper[:, r] *= upper_cut
                firing_lower[:, r] *= lower_cut

        return firing_upper, firing_lower
