"""Takagi-Sugeno-Kang (TSK) Fuzzy Inference System.

Implements order-0 (constant) and order-1 (linear) TSK systems
for regression and classification.

Example
-------
>>> from endgame.fuzzy.inference.tsk import TSKRegressor
>>> model = TSKRegressor(n_mfs=3, order=1)
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
"""

from __future__ import annotations

from itertools import product as iterproduct
from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyClassifier, BaseFuzzyRegressor


class TSKRegressor(BaseFuzzyRegressor):
    """Takagi-Sugeno-Kang Fuzzy Inference System for regression.

    Each rule has the form:
        IF x1 IS A1_r AND x2 IS A2_r AND ... THEN y_r = f(x)

    where f(x) is a constant (order=0) or linear function (order=1).

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_rules : int or None, default=None
        Number of rules. If None, uses grid enumeration
        (n_mfs ** n_features rules, capped at 200).
    order : int, default=1
        TSK order. 0 = constant consequents, 1 = linear consequents.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination.
    optimizer : str, default='lse'
        Method for fitting consequent parameters:
        'lse' = weighted least squares estimation.
    reg_lambda : float, default=1e-6
        Regularization parameter for least squares.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    mfs_ : list of list of BaseMembershipFunction
        Antecedent membership functions, indexed [feature][mf_index].
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        For each rule and feature, the index of the antecedent MF.
    consequent_params_ : ndarray of shape (n_rules, n_consequent_params)
        Consequent parameters. For order=0: shape (n_rules, 1).
        For order=1: shape (n_rules, n_features + 1) where last column
        is the bias term.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.tsk import TSKRegressor
    >>> X = np.random.rand(200, 2)
    >>> y = 3 * X[:, 0] - 2 * X[:, 1] + 1 + np.random.randn(200) * 0.1
    >>> reg = TSKRegressor(n_mfs=3, order=1)
    >>> reg.fit(X, y)
    TSKRegressor(...)
    >>> preds = reg.predict(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        n_rules: int | None = None,
        order: int = 1,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        optimizer: str = "lse",
        reg_lambda: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=n_rules or 0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.order = order
        self.optimizer = optimizer
        self.reg_lambda = reg_lambda
        # Store the user-provided n_rules separately to distinguish None
        self._n_rules_user = n_rules

    def _build_antecedent_indices(self, n_features: int) -> np.ndarray:
        """Build antecedent indices via grid enumeration or random assignment.

        Parameters
        ----------
        n_features : int
            Number of input features.

        Returns
        -------
        ndarray of shape (n_rules, n_features)
        """
        if self._n_rules_user is None:
            # Grid enumeration, capped
            total_grid = self.n_mfs ** n_features
            if total_grid <= 200:
                indices = np.array(
                    list(iterproduct(range(self.n_mfs), repeat=n_features)),
                    dtype=int,
                )
            else:
                # Too many combinations, sample randomly
                rng = np.random.RandomState(self.random_state)
                indices = rng.randint(0, self.n_mfs, size=(200, n_features))
            return indices
        else:
            # Random assignment with specified number of rules
            rng = np.random.RandomState(self.random_state)
            return rng.randint(0, self.n_mfs, size=(self._n_rules_user, n_features))

    def fit(self, X: Any, y: Any) -> TSKRegressor:
        """Fit the TSK fuzzy inference system.

        Steps:
        1. Initialize antecedent membership functions.
        2. Determine antecedent indices (grid or random).
        3. Compute firing strengths for all training samples.
        4. Solve for consequent parameters using weighted least squares.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_ = [f"x{j}" for j in range(n_features)]

        # Initialize MFs from data
        self.mfs_ = self._init_membership_functions(X)

        # Build antecedent assignments
        self.antecedent_indices_ = self._build_antecedent_indices(n_features)
        actual_n_rules = self.antecedent_indices_.shape[0]
        self.n_rules = actual_n_rules

        # Compute firing strengths: (n_samples, n_rules)
        strengths = self._compute_firing_strengths(X, self.antecedent_indices_)

        # Normalize firing strengths
        strength_sums = np.sum(strengths, axis=1, keepdims=True)
        strength_sums = np.maximum(strength_sums, 1e-12)
        norm_strengths = strengths / strength_sums

        # Solve for consequent parameters using weighted LSE
        if self.order == 0:
            # Each rule has one constant parameter
            # y ≈ sum_r(w_r * c_r) → solve for c_r
            # This is a weighted least squares: y = norm_strengths @ c
            self.consequent_params_ = self._solve_lse(
                norm_strengths, y, None, actual_n_rules, 1
            )
        elif self.order == 1:
            # Each rule has (n_features + 1) parameters: [w0, w1, ..., wn, bias]
            # y ≈ sum_r(w_r * (p_r0*x0 + p_r1*x1 + ... + p_rn + bias_r))
            # Build the design matrix
            self.consequent_params_ = self._solve_lse(
                norm_strengths, y, X, actual_n_rules, n_features + 1
            )
        else:
            raise ValueError(f"order must be 0 or 1, got {self.order}")

        self._log(f"Fitted TSK with {actual_n_rules} rules, order={self.order}")
        return self

    def _solve_lse(
        self,
        norm_strengths: np.ndarray,
        y: np.ndarray,
        X: np.ndarray | None,
        n_rules: int,
        n_params: int,
    ) -> np.ndarray:
        """Solve for consequent parameters using weighted least squares.

        Parameters
        ----------
        norm_strengths : ndarray of shape (n_samples, n_rules)
            Normalized firing strengths.
        y : ndarray of shape (n_samples,)
            Target values.
        X : ndarray of shape (n_samples, n_features) or None
            Input data (None for order-0).
        n_rules : int
            Number of rules.
        n_params : int
            Number of consequent parameters per rule.

        Returns
        -------
        ndarray of shape (n_rules, n_params)
            Consequent parameters.
        """
        n_samples = norm_strengths.shape[0]

        if n_params == 1:
            # Order 0: y = W @ c where W = norm_strengths, c = (n_rules,)
            # Design matrix is just the normalized strengths
            A = norm_strengths
        else:
            # Order 1: build expanded design matrix
            # For each rule r, columns are: w_r*x1, w_r*x2, ..., w_r*xn, w_r*1
            n_features = X.shape[1]
            A = np.zeros((n_samples, n_rules * n_params))
            for r in range(n_rules):
                w_r = norm_strengths[:, r]
                for j in range(n_features):
                    A[:, r * n_params + j] = w_r * X[:, j]
                # Bias term
                A[:, r * n_params + n_features] = w_r

        # Regularized least squares: (A^T A + lambda I) theta = A^T y
        ATA = A.T @ A
        ATA += self.reg_lambda * np.eye(ATA.shape[0])
        ATy = A.T @ y

        try:
            theta = np.linalg.solve(ATA, ATy)
        except np.linalg.LinAlgError:
            theta = np.linalg.lstsq(A, y, rcond=None)[0]

        return theta.reshape(n_rules, n_params)

    def _compute_rule_outputs(self, X: np.ndarray) -> np.ndarray:
        """Compute the consequent output for each rule and sample.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
        """
        n_samples = X.shape[0]
        n_rules = self.consequent_params_.shape[0]

        if self.order == 0:
            # Broadcast constants
            return np.tile(self.consequent_params_.ravel(), (n_samples, 1))
        else:
            # Linear: y_r = p0*x0 + p1*x1 + ... + bias
            outputs = np.zeros((n_samples, n_rules))
            for r in range(n_rules):
                params = self.consequent_params_[r]
                # params[:n_features] are coefficients, params[-1] is bias
                outputs[:, r] = X @ params[:-1] + params[-1]
            return outputs

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Computes firing strengths, normalizes them, multiplies by
        rule consequent outputs, and sums (weighted average).

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

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        strengths = self._compute_firing_strengths(X, self.antecedent_indices_)

        # Normalize firing strengths
        strength_sums = np.sum(strengths, axis=1, keepdims=True)
        strength_sums = np.maximum(strength_sums, 1e-12)
        norm_strengths = strengths / strength_sums

        # Compute rule consequent outputs
        rule_outputs = self._compute_rule_outputs(X)

        # Weighted sum
        predictions = np.sum(norm_strengths * rule_outputs, axis=1)
        return predictions


class TSKClassifier(BaseFuzzyClassifier):
    """Takagi-Sugeno-Kang Fuzzy Inference System for classification.

    Uses one TSK regressor per class (one-vs-rest) with softmax
    normalization to produce probability estimates.

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_rules : int or None, default=None
        Number of rules per per-class TSK regressor.
    order : int, default=1
        TSK order (0 or 1).
    mf_type : str, default='gaussian'
        Type of membership function.
    t_norm : str, default='product'
        T-norm for antecedent combination.
    optimizer : str, default='lse'
        Consequent parameter estimation method.
    reg_lambda : float, default=1e-6
        Regularization parameter.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    estimators_ : list of TSKRegressor
        One regressor per class.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.tsk import TSKClassifier
    >>> X = np.random.rand(200, 3)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> clf = TSKClassifier(n_mfs=3, order=1)
    >>> clf.fit(X, y)
    TSKClassifier(...)
    >>> proba = clf.predict_proba(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        n_rules: int | None = None,
        order: int = 1,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        optimizer: str = "lse",
        reg_lambda: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=n_rules or 0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.order = order
        self.optimizer = optimizer
        self.reg_lambda = reg_lambda
        self._n_rules_user = n_rules

    def fit(self, X: Any, y: Any) -> TSKClassifier:
        """Fit a one-vs-rest ensemble of TSK regressors.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        y_encoded = self._encode_labels(y)

        self.estimators_ = []
        for c in range(self.n_classes_):
            y_binary = (y_encoded == c).astype(np.float64)
            reg = TSKRegressor(
                n_mfs=self.n_mfs,
                n_rules=self._n_rules_user,
                order=self.order,
                mf_type=self.mf_type,
                t_norm=self.t_norm,
                optimizer=self.optimizer,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary)
            self.estimators_.append(reg)

        # Store for base class compatibility
        self.antecedent_indices_ = self.estimators_[0].antecedent_indices_
        self.mfs_ = self.estimators_[0].mfs_
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via softmax over per-class TSK outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X, dtype=np.float64)

        raw_outputs = np.column_stack([
            reg.predict(X) for reg in self.estimators_
        ])

        # Softmax
        exp_vals = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
        proba = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return proba
