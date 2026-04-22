"""Multivariate Adaptive Regression Splines (MARS) Implementation.

MARS builds piecewise linear regression models by automatically discovering knots
(thresholds) in the data where the relationship between features and target changes
slope. It was invented by Jerome Friedman in 1991.

This module provides:
- MARSRegressor: MARS for regression tasks
- MARSClassifier: MARS for classification via logistic regression on basis functions

References
----------
Friedman, J. (1991). Multivariate adaptive regression splines.
The Annals of Statistics, 19(1), 1-67.

Friedman, J. (1993). Fast MARS. Stanford University Technical Report 110.

Milborrow, S. Earth package vignette (R implementation reference).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from endgame.models.linear.basis import BasisFunction, HingeSpec, LinearBasisFunction
from typing import Any


class MARSRegressor(GlassboxMixin, BaseEstimator, RegressorMixin):
    """Multivariate Adaptive Regression Splines for regression.

    MARS builds a piecewise linear model by discovering knots (thresholds)
    where the relationship between features and target changes. The model
    is an additive combination of hinge functions: max(0, x - knot) and
    max(0, knot - x).

    Parameters
    ----------
    max_terms : int, default=None
        Maximum number of basis functions (including intercept).
        If None, defaults to min(100, max(20, 2 * n_features)) + 1.

    max_degree : int, default=1
        Maximum degree of interactions.
        1 = additive model (no interactions)
        2 = pairwise interactions allowed
        3 = three-way interactions allowed

    penalty : float, default=3.0
        Generalized Cross-Validation (GCV) penalty per knot.
        Higher values produce simpler models. Typical range: 2-4.

    thresh : float, default=0.001
        Forward pass stopping threshold. Stops adding terms when
        R^2 improvement falls below this value.

    min_span : int, default=None
        Minimum number of observations between knots.
        If None, automatically calculated based on data size.

    endspan : int, default=None
        Minimum observations before first knot and after last knot.
        If None, automatically calculated based on data size.

    fast_k : int, default=20
        In the forward pass, only consider the best fast_k parent terms
        when searching for new basis functions. Set to 0 to consider all
        parents (slower but potentially better). This is "Fast MARS."

    feature_names : list of str, default=None
        Names for features (used in summary output).

    allow_linear : bool, default=True
        If True, allows linear terms (no hinge) for features that
        appear to have purely linear relationships.

    Attributes
    ----------
    basis_functions_ : list of BasisFunction
        The selected basis functions after pruning.

    coef_ : ndarray of shape (n_basis_functions,)
        Coefficients for each basis function.

    intercept_ : float
        The intercept (coefficient of the constant basis function).

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features seen during fit.

    gcv_ : float
        Generalized Cross-Validation score of the final model.

    rsq_ : float
        R-squared of the final model on training data.

    forward_pass_record_ : list
        Record of terms added during forward pass (for diagnostics).

    pruning_record_ : list
        Record of pruning decisions (for diagnostics).

    Examples
    --------
    >>> from endgame.models import MARSRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = np.maximum(0, X[:, 0] - 0.5) + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>> model = MARSRegressor(max_degree=1)
    >>> model.fit(X, y)
    MARSRegressor(max_degree=1)
    >>> print(model.summary())  # doctest: +SKIP
    >>> predictions = model.predict(X)

    References
    ----------
    Friedman, J. (1991). Multivariate adaptive regression splines.
    The Annals of Statistics, 19(1), 1-67.

    Friedman, J. (1993). Fast MARS. Stanford University Technical Report 110.

    Milborrow, S. Earth package vignette (R implementation reference).
    """

    def __init__(
        self,
        max_terms: int | None = None,
        max_degree: int = 1,
        penalty: float = 3.0,
        thresh: float = 0.001,
        min_span: int | None = None,
        endspan: int | None = None,
        fast_k: int = 20,
        feature_names: list[str] | None = None,
        allow_linear: bool = True,
    ):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.thresh = thresh
        self.min_span = min_span
        self.endspan = endspan
        self.fast_k = fast_k
        self.feature_names = feature_names
        self.allow_linear = allow_linear

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> MARSRegressor:
        """Fit the MARS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.feature_names is None and hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        # Validate inputs
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        n_samples, n_features = X.shape

        # Store feature names
        self.n_features_in_ = n_features
        if self.feature_names is not None:
            self.feature_names_in_ = np.array(self.feature_names)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(n_features)])

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)
            if sample_weight.shape[0] != n_samples:
                raise ValueError(
                    f"sample_weight has {sample_weight.shape[0]} samples, "
                    f"expected {n_samples}"
                )
            # Normalize weights to sum to n_samples
            sample_weight = sample_weight * (n_samples / np.sum(sample_weight))
        else:
            sample_weight = np.ones(n_samples, dtype=np.float64)

        # Store references for variable importance (no copy needed — fit
        # owns these arrays and they're not mutated after this point).
        self._X_train = X
        self._y_train = y
        self._sample_weight = sample_weight

        # Calculate default parameters
        self._max_terms = self.max_terms
        if self._max_terms is None:
            self._max_terms = min(100, max(20, 2 * n_features)) + 1

        self._min_span = self.min_span
        if self._min_span is None:
            # Friedman's recommendation: aim for ~20-50 knots per feature
            self._min_span = max(1, int(n_samples / 50))

        self._endspan = self.endspan
        if self._endspan is None:
            self._endspan = self._min_span

        # Check for constant features
        self._feature_variance = np.var(X, axis=0)
        self._valid_features = self._feature_variance > 1e-10

        # Forward pass: build model greedily
        self.forward_pass_record_ = []
        forward_basis = self._forward_pass(X, y, sample_weight)

        # Backward pass: prune model using GCV
        self.pruning_record_ = []
        self.basis_functions_ = self._backward_pass(forward_basis, X, y, sample_weight)

        # Fit final coefficients
        B = self.get_basis_matrix(X)
        self.coef_, self._rss = self._fit_coefficients(B, y, sample_weight)

        # Store intercept separately (coefficient of first basis function)
        self.intercept_ = self.coef_[0]

        # Compute final statistics
        self.gcv_ = self._compute_gcv(self.basis_functions_, X, y, sample_weight)
        total_ss = np.sum(sample_weight * (y - np.average(y, weights=sample_weight)) ** 2)
        self.rsq_ = 1.0 - self._rss / total_ss if total_ss > 0 else 0.0

        return self

    def predict(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict using the fitted MARS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        B = self.get_basis_matrix(X)
        return B @ self.coef_

    def get_basis_matrix(self, X: ArrayLike) -> NDArray[np.floating]:
        """Compute the basis function matrix for given X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        B : ndarray of shape (n_samples, n_basis_functions)
            Matrix where B[i, j] is the value of basis function j
            evaluated at sample i.
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_basis = len(self.basis_functions_)

        B = np.empty((n_samples, n_basis), dtype=np.float64)
        for j, bf in enumerate(self.basis_functions_):
            B[:, j] = bf.evaluate(X)

        return B

    def summary(self) -> str:
        """Return a human-readable summary of the model.

        Returns a string showing:
        - Model equation with all basis functions
        - R^2 and GCV statistics
        - Variable importance

        Returns
        -------
        summary : str
            Formatted model summary.
        """
        check_is_fitted(self)

        lines = []
        lines.append("MARS Model Summary")
        lines.append("=" * 50)
        lines.append("")

        # Call info
        params = []
        if self.max_terms is not None:
            params.append(f"max_terms={self._max_terms}")
        params.append(f"max_degree={self.max_degree}")
        params.append(f"penalty={self.penalty}")
        lines.append(f"Call: MARSRegressor({', '.join(params)})")
        lines.append("")

        # Basis functions
        lines.append(f"Basis Functions ({len(self.basis_functions_)} terms):")
        lines.append("-" * 40)

        for i, bf in enumerate(self.basis_functions_):
            if bf.is_intercept:
                name = "Intercept"
            else:
                name = bf.to_str_with_names(list(self.feature_names_in_))
            coef = self.coef_[i]
            lines.append(f"  {name:40s} {coef:10.4f}")

        lines.append("")

        # Model statistics
        lines.append("Model Statistics:")
        lines.append("-" * 40)
        lines.append(f"  GCV:          {self.gcv_:.4f}")
        lines.append(f"  RSS:          {self._rss:.4f}")
        lines.append(f"  R-squared:    {self.rsq_:.4f}")

        # Adjusted R-squared
        n = len(self._y_train)
        p = len(self.basis_functions_)
        if n > p:
            adj_rsq = 1.0 - (1.0 - self.rsq_) * (n - 1) / (n - p)
            lines.append(f"  Adj R-sq:     {adj_rsq:.4f}")

        lines.append("")

        # Variable importance
        importance = self.compute_variable_importance()
        if importance:
            lines.append("Variable Importance:")
            lines.append("-" * 40)
            sorted_imp = sorted(importance.items(), key=lambda x: -x[1])
            for name, imp in sorted_imp:
                if imp > 0:
                    lines.append(f"  {name:20s} {imp:6.1f}")

        lines.append("")

        # Equation
        lines.append("Equation:")
        lines.append("-" * 40)
        eq_parts = []
        for i, bf in enumerate(self.basis_functions_):
            coef = self.coef_[i]
            if bf.is_intercept:
                eq_parts.append(f"{coef:.4g}")
            else:
                bf_str = bf.to_str_with_names(list(self.feature_names_in_))
                if coef >= 0:
                    eq_parts.append(f" + {coef:.4g}*{bf_str}")
                else:
                    eq_parts.append(f" - {abs(coef):.4g}*{bf_str}")

        equation = "y = " + "".join(eq_parts)
        # Wrap long equations
        if len(equation) > 70:
            wrapped = ["y = " + eq_parts[0]]
            current_line = "    "
            for part in eq_parts[1:]:
                if len(current_line) + len(part) > 66:
                    wrapped.append(current_line)
                    current_line = "    " + part
                else:
                    current_line += part
            if current_line.strip():
                wrapped.append(current_line)
            equation = "\n".join(wrapped)

        lines.append(equation)

        return "\n".join(lines)

    def compute_variable_importance(self) -> dict[str, float]:
        """Compute variable importance based on GCV decrease.

        For each variable, compute how much GCV would increase
        if all basis functions involving that variable were removed.

        Returns
        -------
        importance : dict
            {feature_name: importance_score}
            Scores are normalized so max = 100.
        """
        check_is_fitted(self)

        importances = {}

        for feature_idx in range(self.n_features_in_):
            # Find all basis functions using this feature
            bf_using_feature = [
                i for i, bf in enumerate(self.basis_functions_)
                if feature_idx in bf.feature_indices
            ]

            if not bf_using_feature:
                importances[feature_idx] = 0.0
                continue

            # Compute GCV without these basis functions
            subset = [
                bf for i, bf in enumerate(self.basis_functions_)
                if i not in bf_using_feature or bf.is_intercept
            ]

            # Need at least intercept
            if not subset:
                subset = [BasisFunction()]

            gcv_without = self._compute_gcv(
                subset, self._X_train, self._y_train, self._sample_weight
            )

            # Importance = increase in GCV when variable is removed
            importances[feature_idx] = max(0.0, gcv_without - self.gcv_)

        # Normalize to 0-100 scale
        max_imp = max(importances.values()) if importances else 1
        if max_imp > 0:
            importances = {k: 100 * v / max_imp for k, v in importances.items()}

        # Convert to feature names
        named_importances = {
            self.feature_names_in_[k]: v
            for k, v in importances.items()
        }

        return named_importances

    _MAX_FORWARD_SAMPLES = 2000

    def _forward_pass(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating],
    ) -> list[BasisFunction | LinearBasisFunction]:
        """Greedy forward pass to add basis functions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        basis_functions : list
            List of basis functions added during forward pass.
        """
        n_samples, n_features = X.shape

        if n_samples > int(self._MAX_FORWARD_SAMPLES * 1.5):
            rng = np.random.RandomState(42)
            _idx = rng.choice(n_samples, self._MAX_FORWARD_SAMPLES, replace=False)
            X = X[_idx]
            y = y[_idx]
            sample_weight = sample_weight[_idx]
            sample_weight = sample_weight * (self._MAX_FORWARD_SAMPLES / np.sum(sample_weight))
            n_samples = self._MAX_FORWARD_SAMPLES

        # Start with intercept only
        basis_functions: list[BasisFunction | LinearBasisFunction] = [
            BasisFunction()
        ]

        # Build initial basis matrix with QR decomposition for fast updates
        B = np.ones((n_samples, 1), dtype=np.float64)
        # Weight the basis matrix for weighted least squares
        sqrt_w = np.sqrt(sample_weight)
        B_weighted = B * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        Q, R = np.linalg.qr(B_weighted)

        # Compute initial RSS
        residuals = y_weighted - Q @ (Q.T @ y_weighted)
        current_rss = np.sum(residuals ** 2)
        total_ss = np.sum(sample_weight * (y - np.average(y, weights=sample_weight)) ** 2)

        # Pre-compute candidate knots for each feature
        knot_candidates = {}
        for j in range(n_features):
            if self._valid_features[j]:
                knot_candidates[j] = self._get_candidate_knots(X[:, j])
            else:
                knot_candidates[j] = np.array([])

        while len(basis_functions) < self._max_terms:
            best_decrease = 0.0
            best_pair: tuple | None = None
            best_linear: LinearBasisFunction | None = None

            # Determine which parents to consider (Fast MARS)
            if self.fast_k > 0 and len(basis_functions) > self.fast_k:
                # Select top fast_k parents based on their contribution
                # Use absolute coefficient as proxy for importance
                parent_indices = self._select_top_parents(
                    basis_functions, B, y, sample_weight, self.fast_k
                )
            else:
                parent_indices = list(range(len(basis_functions)))

            # Consider each existing basis function as a parent
            for parent_idx in parent_indices:
                parent = basis_functions[parent_idx]

                if parent.degree >= self.max_degree:
                    continue

                parent_features = set(parent.feature_indices)

                # Pre-evaluate parent once (intercept → all ones)
                if parent.is_intercept:
                    parent_vals = None  # signals "ones"
                else:
                    parent_vals = parent.evaluate(X)

                for feature_j in range(n_features):
                    if not self._valid_features[feature_j]:
                        continue
                    if feature_j in parent_features:
                        continue

                    knots = knot_candidates[feature_j]
                    if len(knots) == 0:
                        continue

                    x_col = X[:, feature_j]

                    # Vectorised hinge: compute for ALL knots at once
                    # diff shape: (n_samples, n_knots)
                    diff = x_col[:, np.newaxis] - knots[np.newaxis, :]
                    h_plus_all = np.maximum(0.0, diff)   # max(0, x - t)
                    h_minus_all = np.maximum(0.0, -diff)  # max(0, t - x)

                    if parent_vals is not None:
                        h_plus_all = h_plus_all * parent_vals[:, np.newaxis]
                        h_minus_all = h_minus_all * parent_vals[:, np.newaxis]

                    # Weight columns
                    h_plus_w = h_plus_all * sqrt_w[:, np.newaxis]
                    h_minus_w = h_minus_all * sqrt_w[:, np.newaxis]

                    # Check for all-zeros columns (energy check)
                    energy_plus = np.einsum("ij,ij->j", h_plus_w, h_plus_w)
                    energy_minus = np.einsum("ij,ij->j", h_minus_w, h_minus_w)
                    valid = (energy_plus > 1e-10) & (energy_minus > 1e-10)

                    if not np.any(valid):
                        # Try linear term if applicable
                        if self.allow_linear and parent.is_intercept:
                            col_linear = x_col * sqrt_w
                            if np.dot(col_linear, col_linear) > 1e-10:
                                rss_dec = self._compute_rss_decrease_single(
                                    Q, y_weighted, col_linear,
                                )
                                if rss_dec > best_decrease:
                                    best_decrease = rss_dec
                                    best_pair = None
                                    best_linear = (LinearBasisFunction(feature_j), col_linear)
                        continue

                    # Project all valid knot columns at once: Q.T @ col
                    QtHp = Q.T @ h_plus_w[:, valid]   # (k, n_valid)
                    QtHm = Q.T @ h_minus_w[:, valid]

                    # Orthogonal component for each knot
                    hp_orth = h_plus_w[:, valid] - Q @ QtHp
                    hm_orth = h_minus_w[:, valid] - Q @ QtHm

                    # Gram-Schmidt: orthogonalise minus against plus
                    norms_p = np.sqrt(np.einsum("ij,ij->j", hp_orth, hp_orth))
                    safe_p = norms_p > 1e-10

                    var1 = np.zeros(safe_p.shape)
                    if np.any(safe_p):
                        hp_unit = hp_orth[:, safe_p] / norms_p[np.newaxis, safe_p]
                        var1[safe_p] = np.einsum("ij,i->j", hp_unit, y_weighted) ** 2

                        # Project hm_orth against hp_unit
                        dot_pm = np.einsum("ij,ij->j", hp_unit, hm_orth[:, safe_p])
                        hm_orth2 = hm_orth[:, safe_p] - hp_unit * dot_pm[np.newaxis, :]
                    else:
                        hm_orth2 = hm_orth

                    # Compute var2 for all valid knots
                    var2 = np.zeros(safe_p.shape)
                    if np.any(safe_p):
                        norms_m = np.sqrt(np.einsum("ij,ij->j", hm_orth2, hm_orth2))
                        safe_m = norms_m > 1e-10
                        if np.any(safe_m):
                            hm_unit = hm_orth2[:, safe_m] / norms_m[np.newaxis, safe_m]
                            v2_vals = np.einsum("ij,i->j", hm_unit, y_weighted) ** 2
                            idx_m = np.where(safe_p)[0][safe_m]
                            var2[idx_m] = v2_vals

                    rss_dec_all = var1 + var2
                    best_k = int(np.argmax(rss_dec_all))
                    if rss_dec_all[best_k] > best_decrease:
                        best_decrease = rss_dec_all[best_k]
                        # Map back to original knot index
                        orig_idx = np.where(valid)[0][best_k]
                        knot_t = knots[orig_idx]
                        h_plus = parent.extend(HingeSpec(feature_j, knot_t, +1))
                        h_minus = parent.extend(HingeSpec(feature_j, knot_t, -1))
                        best_pair = (
                            h_plus, h_minus,
                            h_plus_w[:, orig_idx], h_minus_w[:, orig_idx],
                        )
                        best_linear = None

                    # Also try linear term if allowed and parent is intercept
                    if self.allow_linear and parent.is_intercept:
                        col_linear = x_col * sqrt_w
                        if np.dot(col_linear, col_linear) > 1e-10:
                            rss_dec = self._compute_rss_decrease_single(
                                Q, y_weighted, col_linear,
                            )
                            if rss_dec > best_decrease:
                                best_decrease = rss_dec
                                best_pair = None
                                best_linear = (LinearBasisFunction(feature_j), col_linear)

            # Stopping criterion
            if total_ss > 0:
                r2_improvement = best_decrease / total_ss
            else:
                r2_improvement = 0.0

            if r2_improvement < self.thresh:
                break

            if best_decrease <= 0:
                break

            # Add the best term(s)
            if best_linear is not None:
                linear, col_linear = best_linear
                basis_functions.append(linear)
                B = np.column_stack([B, col_linear / sqrt_w[:, np.newaxis]])
                B_weighted = np.column_stack([B_weighted[:, :-1] if B_weighted.shape[1] > 1 else B_weighted, col_linear])
                # Actually append properly
                B_weighted = B * sqrt_w[:, np.newaxis]
                Q, R = np.linalg.qr(B_weighted)

                self.forward_pass_record_.append({
                    "type": "linear",
                    "feature": feature_j,
                    "rss_decrease": best_decrease,
                    "r2_improvement": r2_improvement,
                })
            elif best_pair is not None:
                h_plus, h_minus, col_plus, col_minus = best_pair
                basis_functions.append(h_plus)
                basis_functions.append(h_minus)

                # Update basis matrix
                B = np.column_stack([
                    B,
                    col_plus / sqrt_w,
                    col_minus / sqrt_w
                ])
                B_weighted = B * sqrt_w[:, np.newaxis]
                Q, R = np.linalg.qr(B_weighted)

                self.forward_pass_record_.append({
                    "type": "hinge_pair",
                    "feature": h_plus.hinges[-1].feature_idx,
                    "knot": h_plus.hinges[-1].knot,
                    "parent_degree": h_plus.degree - 1,
                    "rss_decrease": best_decrease,
                    "r2_improvement": r2_improvement,
                })

            # Update RSS
            residuals = y_weighted - Q @ (Q.T @ y_weighted)
            current_rss = np.sum(residuals ** 2)

        return basis_functions

    def _backward_pass(
        self,
        basis_functions: list[BasisFunction | LinearBasisFunction],
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating],
    ) -> list[BasisFunction | LinearBasisFunction]:
        """Backward pass to prune basis functions using GCV.

        Uses Gram matrix (normal equations) to avoid O(n) lstsq per
        candidate removal.  Pre-computes G = B_w^T B_w and h = B_w^T y_w
        once, then each candidate evaluation is O(k^3) instead of
        O(n * k^2).
        """
        n = len(y)
        sqrt_w = np.sqrt(sample_weight)
        y_weighted = y * sqrt_w

        current = list(basis_functions)
        active = list(range(len(current)))

        B_full = np.column_stack([bf.evaluate(X) for bf in current])
        B_full_w = B_full * sqrt_w[:, np.newaxis]

        G = B_full_w.T @ B_full_w
        h = B_full_w.T @ y_weighted
        yWy = float(y_weighted @ y_weighted)

        def _gcv_subset(idx_list):
            idx = np.array(idx_list)
            G_sub = G[np.ix_(idx, idx)].copy()
            h_sub = h[idx]
            reg = 1e-12 * max(np.mean(np.diag(G_sub)), 1e-30)
            G_sub[np.diag_indices_from(G_sub)] += reg
            try:
                coef = np.linalg.solve(G_sub, h_sub)
            except np.linalg.LinAlgError:
                return np.inf
            rss = max(0.0, yWy - float(h_sub @ coef))
            n_coef = len(idx_list)
            n_knots = sum(current[i].degree for i in idx_list)
            eff = n_coef + self.penalty * n_knots
            denom = n * (1.0 - eff / n) ** 2
            return rss / denom if denom > 0 else np.inf

        best_gcv = _gcv_subset(active)
        best_model_indices = list(active)

        while len(active) > 1:
            best_removal_gcv = np.inf
            col_to_remove = None

            for i, a in enumerate(active):
                if current[a].is_intercept:
                    continue
                subset = active[:i] + active[i + 1:]
                gcv = _gcv_subset(subset)
                if gcv < best_removal_gcv:
                    best_removal_gcv = gcv
                    col_to_remove = a

            if col_to_remove is None:
                break

            self.pruning_record_.append({
                "removed": str(current[col_to_remove]),
                "gcv_after": best_removal_gcv,
                "n_terms_after": len(active) - 1,
            })

            active.remove(col_to_remove)

            if best_removal_gcv < best_gcv:
                best_gcv = best_removal_gcv
                best_model_indices = list(active)

        return [current[i] for i in best_model_indices]

    def _compute_gcv(
        self,
        basis_functions: list[BasisFunction | LinearBasisFunction],
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating],
    ) -> float:
        """Compute Generalized Cross-Validation score.

        GCV = RSS / (n * (1 - effective_params / n)^2)

        Where effective_params accounts for both the number of
        coefficients and the knot selection process.

        Parameters
        ----------
        basis_functions : list
            Current model's basis functions.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        gcv : float
            GCV score (lower is better).
        """
        n = len(y)

        # Build basis matrix
        B = np.column_stack([bf.evaluate(X) for bf in basis_functions])

        # Fit coefficients with weighted least squares
        sqrt_w = np.sqrt(sample_weight)
        B_weighted = B * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        try:
            coef, residuals, rank, s = np.linalg.lstsq(B_weighted, y_weighted, rcond=None)
        except np.linalg.LinAlgError:
            return np.inf

        # Compute RSS
        y_pred = B @ coef
        rss = np.sum(sample_weight * (y - y_pred) ** 2)

        # Count effective parameters
        # Number of coefficients (columns in B)
        n_coefficients = B.shape[1]

        # Number of knots (hinges) - counts the knot selection cost
        n_knots = sum(bf.degree for bf in basis_functions)

        # Effective parameters: coefficients + penalty * knots
        effective_params = n_coefficients + self.penalty * n_knots

        # GCV formula
        denominator = n * (1 - effective_params / n) ** 2

        if denominator <= 0:
            return np.inf

        gcv = rss / denominator

        return gcv

    def _fit_coefficients(
        self,
        B: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], float]:
        """Fit coefficients using weighted least squares.

        Parameters
        ----------
        B : ndarray of shape (n_samples, n_basis_functions)
            Basis matrix.
        y : ndarray of shape (n_samples,)
            Target values.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        coef : ndarray of shape (n_basis_functions,)
            Fitted coefficients.
        rss : float
            Residual sum of squares.
        """
        sqrt_w = np.sqrt(sample_weight)
        B_weighted = B * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        coef, residuals, rank, s = np.linalg.lstsq(B_weighted, y_weighted, rcond=None)

        y_pred = B @ coef
        rss = np.sum(sample_weight * (y - y_pred) ** 2)

        return coef, rss

    def _get_candidate_knots(
        self,
        x: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Get candidate knot locations for a feature.

        Parameters
        ----------
        x : ndarray of shape (n_samples,)
            Feature values.

        Returns
        -------
        knots : ndarray
            Candidate knot values.
        """
        n = len(x)

        # Sort and get unique values
        sorted_unique = np.sort(np.unique(x))

        if len(sorted_unique) <= 1:
            return np.array([])

        # Only keep values that have enough observations on each side
        if len(sorted_unique) <= 2 * self._endspan:
            # For small number of unique values, use middle values
            if len(sorted_unique) > 2:
                return sorted_unique[1:-1]
            return np.array([])

        # Select knots with proper spacing
        # Exclude endpoints based on endspan
        candidates = sorted_unique[self._endspan:-self._endspan]

        if len(candidates) > 25:
            indices = np.linspace(0, len(candidates) - 1, 25, dtype=int)
            candidates = candidates[indices]

        return candidates

    def _compute_rss_decrease_fast(
        self,
        Q: NDArray[np.floating],
        y_weighted: NDArray[np.floating],
        new_col1: NDArray[np.floating],
        new_col2: NDArray[np.floating],
    ) -> float:
        """Compute RSS decrease from adding two columns using QR update.

        This is O(n * k) instead of O(n * k^2) for full refitting.

        Parameters
        ----------
        Q : ndarray
            Q matrix from QR decomposition of current basis.
        y_weighted : ndarray
            Weighted target values.
        new_col1 : ndarray
            First new basis function evaluated (weighted).
        new_col2 : ndarray
            Second new basis function evaluated (weighted).

        Returns
        -------
        rss_decrease : float
            Decrease in RSS from adding these columns.
        """
        # Project new columns onto orthogonal complement of current space
        new_col1_orth = new_col1 - Q @ (Q.T @ new_col1)
        new_col2_orth = new_col2 - Q @ (Q.T @ new_col2)

        # Gram-Schmidt on the two new columns
        norm1 = np.linalg.norm(new_col1_orth)
        if norm1 < 1e-10:
            new_col1_orth = np.zeros_like(new_col1_orth)
            var1 = 0.0
        else:
            new_col1_orth = new_col1_orth / norm1
            var1 = (new_col1_orth @ y_weighted) ** 2

        new_col2_orth = new_col2_orth - new_col1_orth * (new_col1_orth @ new_col2_orth)
        norm2 = np.linalg.norm(new_col2_orth)
        if norm2 < 1e-10:
            var2 = 0.0
        else:
            new_col2_orth = new_col2_orth / norm2
            var2 = (new_col2_orth @ y_weighted) ** 2

        # Additional variance explained
        return var1 + var2

    def _compute_rss_decrease_single(
        self,
        Q: NDArray[np.floating],
        y_weighted: NDArray[np.floating],
        new_col: NDArray[np.floating],
    ) -> float:
        """Compute RSS decrease from adding a single column.

        Parameters
        ----------
        Q : ndarray
            Q matrix from QR decomposition of current basis.
        y_weighted : ndarray
            Weighted target values.
        new_col : ndarray
            New basis function evaluated (weighted).

        Returns
        -------
        rss_decrease : float
            Decrease in RSS from adding this column.
        """
        # Project new column onto orthogonal complement
        new_col_orth = new_col - Q @ (Q.T @ new_col)

        norm = np.linalg.norm(new_col_orth)
        if norm < 1e-10:
            return 0.0

        new_col_orth = new_col_orth / norm
        return (new_col_orth @ y_weighted) ** 2

    def _select_top_parents(
        self,
        basis_functions: list[BasisFunction | LinearBasisFunction],
        B: NDArray[np.floating],
        y: NDArray[np.floating],
        sample_weight: NDArray[np.floating],
        k: int,
    ) -> list[int]:
        """Select top k parent basis functions for Fast MARS.

        Parameters
        ----------
        basis_functions : list
            Current basis functions.
        B : ndarray
            Current basis matrix.
        y : ndarray
            Target values.
        sample_weight : ndarray
            Sample weights.
        k : int
            Number of parents to select.

        Returns
        -------
        indices : list of int
            Indices of top k parents.
        """
        # Always include intercept
        selected = [0]

        # Compute residual contribution for each basis function
        sqrt_w = np.sqrt(sample_weight)
        B_weighted = B * sqrt_w[:, np.newaxis]
        y_weighted = y * sqrt_w

        coef, _, _, _ = np.linalg.lstsq(B_weighted, y_weighted, rcond=None)

        # Compute contribution as |coef| * std(basis)
        contributions = []
        for i in range(1, len(basis_functions)):
            if i < B.shape[1]:
                contrib = abs(coef[i]) * np.std(B[:, i])
                contributions.append((i, contrib))

        # Sort by contribution and take top k-1 (intercept is already included)
        contributions.sort(key=lambda x: -x[1])
        for i, _ in contributions[:k - 1]:
            selected.append(i)

        return selected


class MARSClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """MARS for classification via logistic regression on basis functions.

    Fits a MARS model to discover basis functions, then uses logistic
    regression on those basis functions for classification.

    Parameters
    ----------
    max_terms : int, default=None
        Maximum number of basis functions (including intercept).
        If None, defaults to min(100, max(20, 2 * n_features)) + 1.

    max_degree : int, default=1
        Maximum degree of interactions.
        1 = additive model (no interactions)
        2 = pairwise interactions allowed
        3 = three-way interactions allowed

    penalty : float, default=3.0
        Generalized Cross-Validation (GCV) penalty per knot.
        Higher values produce simpler models.

    thresh : float, default=0.001
        Forward pass stopping threshold.

    min_span : int, default=None
        Minimum number of observations between knots.

    endspan : int, default=None
        Minimum observations before first knot and after last knot.

    fast_k : int, default=20
        Fast MARS parameter (see MARSRegressor).

    feature_names : list of str, default=None
        Names for features.

    allow_linear : bool, default=True
        If True, allows linear terms.

    method : str, default='logistic'
        Classification method:
        - 'logistic': Logistic regression on MARS basis functions
        - 'threshold': Threshold regression predictions at 0.5

    logistic_C : float, default=1.0
        Regularization parameter for logistic regression.
        Only used when method='logistic'.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    mars_regressor_ : MARSRegressor
        Underlying MARS model for basis function discovery.

    logistic_ : LogisticRegression
        Fitted logistic regression model (when method='logistic').

    Examples
    --------
    >>> from endgame.models import MARSClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 3)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> model = MARSClassifier(max_degree=1)
    >>> model.fit(X, y)
    MARSClassifier(max_degree=1)
    >>> predictions = model.predict(X)
    >>> probas = model.predict_proba(X)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        max_terms: int | None = None,
        max_degree: int = 1,
        penalty: float = 3.0,
        thresh: float = 0.001,
        min_span: int | None = None,
        endspan: int | None = None,
        fast_k: int = 20,
        feature_names: list[str] | None = None,
        allow_linear: bool = True,
        method: str = "logistic",
        logistic_C: float = 1.0,
    ):
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.thresh = thresh
        self.min_span = min_span
        self.endspan = endspan
        self.fast_k = fast_k
        self.feature_names = feature_names
        self.allow_linear = allow_linear
        self.method = method
        self.logistic_C = logistic_C

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> MARSClassifier:
        """Fit the MARS classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Individual weights for each sample.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if self.feature_names is None and hasattr(X, "columns"):
            self.feature_names = list(X.columns)

        X, y = check_X_y(X, y, dtype=np.float64)

        # Store classes
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)

        if n_classes < 2:
            raise ValueError("Need at least 2 classes for classification")

        # Store feature count
        self.n_features_in_ = X.shape[1]
        if self.feature_names is not None:
            self.feature_names_in_ = np.array(self.feature_names)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(self.n_features_in_)])

        # Create and fit the underlying MARS regressor
        # For binary classification, use y directly
        # For multiclass, we fit on the first class indicator
        if n_classes == 2:
            y_reg = (y == self.classes_[1]).astype(np.float64)
        else:
            y_reg = y.astype(np.float64)

        self.mars_regressor_ = MARSRegressor(
            max_terms=self.max_terms,
            max_degree=self.max_degree,
            penalty=self.penalty,
            thresh=self.thresh,
            min_span=self.min_span,
            endspan=self.endspan,
            fast_k=self.fast_k,
            feature_names=self.feature_names,
            allow_linear=self.allow_linear,
        )
        self.mars_regressor_.fit(X, y_reg, sample_weight=sample_weight)

        # Get basis matrix
        B = self.mars_regressor_.get_basis_matrix(X)

        if self.method == "logistic":
            # Fit logistic regression on basis functions
            self.logistic_ = LogisticRegression(
                C=self.logistic_C,
                solver="lbfgs",
                max_iter=1000,
            )
            self.logistic_.fit(B, y, sample_weight=sample_weight)
        elif self.method == "threshold":
            # Just use the regression predictions
            pass
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def predict(self, X: ArrayLike) -> NDArray:
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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if self.method == "logistic":
            B = self.mars_regressor_.get_basis_matrix(X)
            return self.logistic_.predict(B)
        else:
            # Threshold method
            y_pred_reg = self.mars_regressor_.predict(X)
            if len(self.classes_) == 2:
                return np.where(y_pred_reg >= 0.5, self.classes_[1], self.classes_[0])
            else:
                return self.classes_[np.clip(np.round(y_pred_reg).astype(int), 0, len(self.classes_) - 1)]

    def predict_proba(self, X: ArrayLike) -> NDArray[np.floating]:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if self.method == "logistic":
            B = self.mars_regressor_.get_basis_matrix(X)
            return self.logistic_.predict_proba(B)
        else:
            # Threshold method - use sigmoid-like transformation
            y_pred_reg = self.mars_regressor_.predict(X)
            y_pred_reg = np.clip(y_pred_reg, 0, 1)
            if len(self.classes_) == 2:
                return np.column_stack([1 - y_pred_reg, y_pred_reg])
            else:
                # Simple softmax-like for multiclass
                n_samples = len(y_pred_reg)
                proba = np.zeros((n_samples, len(self.classes_)))
                for i, c in enumerate(self.classes_):
                    proba[:, i] = np.exp(-np.abs(y_pred_reg - c))
                proba /= proba.sum(axis=1, keepdims=True)
                return proba

    def summary(self) -> str:
        """Return a human-readable summary of the model.

        Returns
        -------
        summary : str
            Formatted model summary.
        """
        check_is_fitted(self)

        lines = [
            "MARS Classifier Summary",
            "=" * 50,
            "",
            f"Classes: {self.classes_}",
            f"Method: {self.method}",
            "",
            "Underlying MARS Model:",
            "-" * 40,
        ]

        # Add regressor summary (indented)
        mars_summary = self.mars_regressor_.summary()
        for line in mars_summary.split("\n"):
            lines.append("  " + line)

        return "\n".join(lines)

    @property
    def basis_functions_(self):
        """Return basis functions from underlying MARS regressor."""
        check_is_fitted(self)
        return self.mars_regressor_.basis_functions_

    def get_basis_matrix(self, X: ArrayLike) -> NDArray[np.floating]:
        """Compute the basis function matrix for given X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        B : ndarray of shape (n_samples, n_basis_functions)
            Basis matrix.
        """
        check_is_fitted(self)
        return self.mars_regressor_.get_basis_matrix(X)

    def compute_variable_importance(self) -> dict[str, float]:
        """Compute variable importance.

        Returns
        -------
        importance : dict
            {feature_name: importance_score}
        """
        check_is_fitted(self)
        return self.mars_regressor_.compute_variable_importance()


def _mars_structure(self) -> dict[str, Any]:
    check_is_fitted(self)
    feature_names = list(self.feature_names_in_) if getattr(self, "feature_names_in_", None) is not None else [f"x{i}" for i in range(self.n_features_in_)]
    basis_dicts = []
    for bf, coef in zip(self.basis_functions_, self.coef_):
        hinges = [
            {
                "feature_index": int(h.feature_idx),
                "feature": feature_names[h.feature_idx] if h.feature_idx < len(feature_names) else f"x{h.feature_idx}",
                "knot": float(h.knot),
                "direction": int(h.direction),
            }
            for h in getattr(bf, "hinges", [])
        ]
        basis_dicts.append({
            "coefficient": float(coef),
            "is_intercept": bool(bf.is_intercept) if hasattr(bf, "is_intercept") else (len(hinges) == 0),
            "degree": len(hinges),
            "hinges": hinges,
            "text": str(bf),
        })
    return {
        "intercept": float(self.intercept_),
        "basis_functions": basis_dicts,
        "n_terms": len(basis_dicts),
    }


MARSRegressor._structure_type = "additive"
MARSRegressor._structure_content = _mars_structure


def _mars_classifier_structure(self) -> dict[str, Any]:
    check_is_fitted(self)
    # MARSClassifier wraps a MARSRegressor internally plus logistic regression
    reg = self.mars_regressor_
    base = _mars_structure(reg)
    base["link"] = "logit"
    base["logistic_coef"] = [float(c) for c in np.asarray(self.logistic_.coef_).ravel()]
    base["logistic_intercept"] = float(np.asarray(self.logistic_.intercept_).ravel()[0])
    return base


MARSClassifier._structure_type = "additive"
MARSClassifier._structure_content = _mars_classifier_structure
