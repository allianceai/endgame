"""Target transformation wrappers for regression.

Applies invertible transformations to the target variable during training,
then inverse-transforms predictions at inference time. This can improve
regression performance when the target distribution is skewed, heavy-tailed,
or otherwise non-normal.

Similar to sklearn's TransformedTargetRegressor but with more flexible
transform selection including automatic normality-based selection.

Supported transforms:
- log, log1p, sqrt: Simple monotonic transforms
- box_cox, yeo_johnson: Power transforms (scipy)
- quantile: QuantileTransformer-based normalization
- rank: Rank-based (ordinal) normalization
- auto: Automatically selects the best transform via Shapiro-Wilk normality test
- none: No transformation (passthrough)
"""

from typing import Any

import numpy as np
from sklearn.base import RegressorMixin, clone
from sklearn.preprocessing import QuantileTransformer as _SklearnQuantileTransformer

from endgame.core.base import EndgameEstimator

# Methods that require strictly positive targets
_POSITIVE_METHODS = {"log", "box_cox"}
# Methods that require non-negative targets
_NONNEG_METHODS = {"sqrt", "log1p"}
# All valid method strings
_VALID_METHODS = {
    "auto", "log", "log1p", "sqrt", "box_cox",
    "yeo_johnson", "quantile", "rank", "none",
}


def _check_positive(y: np.ndarray, method: str) -> None:
    """Raise ValueError if y contains non-positive values for a method that requires them."""
    if np.any(y <= 0):
        raise ValueError(
            f"Target contains non-positive values. Method '{method}' requires "
            f"all positive targets. Consider 'log1p', 'yeo_johnson', 'quantile', "
            f"or 'auto' instead."
        )


def _check_nonneg(y: np.ndarray, method: str) -> None:
    """Raise ValueError if y contains negative values for a method that requires non-negative."""
    if np.any(y < 0):
        raise ValueError(
            f"Target contains negative values. Method '{method}' requires "
            f"all non-negative targets. Consider 'yeo_johnson', 'quantile', "
            f"or 'auto' instead."
        )


class TargetTransformer(EndgameEstimator, RegressorMixin):
    """Wrapper that applies target transformations for regression.

    Transforms the target variable y during ``fit``, trains the wrapped
    regressor on the transformed targets, and inverse-transforms predictions
    at inference time.

    Parameters
    ----------
    regressor : estimator
        Any sklearn-compatible regressor. This is required.
    method : str, default='auto'
        Transformation method. One of:

        - ``'auto'``: Test normality via Shapiro-Wilk; try Box-Cox and
          Yeo-Johnson and pick whichever produces the most normal
          transformed y. Falls back to ``'yeo_johnson'`` when Box-Cox is
          not applicable (non-positive targets).
        - ``'log'``: Natural log. Requires strictly positive targets.
        - ``'log1p'``: ``log(1 + y)``. Requires non-negative targets.
        - ``'sqrt'``: Square root. Requires non-negative targets.
        - ``'box_cox'``: Box-Cox power transform (scipy). Requires
          strictly positive targets.
        - ``'yeo_johnson'``: Yeo-Johnson power transform (scipy). Works
          with any real-valued targets.
        - ``'quantile'``: Sklearn QuantileTransformer mapping to normal.
        - ``'rank'``: Rank-based (ordinal) normalization.
        - ``'none'``: No transformation (passthrough).
    random_state : int, optional
        Random seed for reproducibility (passed to quantile transform
        and the wrapped regressor if it supports it).
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    regressor_ : estimator
        The fitted regressor (clone of ``regressor``).
    method_ : str
        The method actually used (relevant when ``method='auto'``).
    lambda_ : float or None
        The fitted lambda parameter for Box-Cox / Yeo-Johnson transforms.
    qt_ : QuantileTransformer or None
        Fitted QuantileTransformer instance (for ``method='quantile'``).
    y_train_sorted_ : ndarray or None
        Sorted training targets for rank inverse transform.
    feature_importances_ : ndarray
        Delegated from the wrapped regressor, if available.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from endgame.preprocessing import TargetTransformer
    >>> model = TargetTransformer(
    ...     regressor=RandomForestRegressor(n_estimators=100, random_state=42),
    ...     method='auto',
    ... )
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        regressor: Any = None,
        method: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        if regressor is None:
            raise TypeError(
                "TargetTransformer requires a regressor. Pass a sklearn-compatible "
                "regressor via the 'regressor' parameter."
            )
        self.regressor = regressor
        self.method = method

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X, y, **fit_params) -> "TargetTransformer":
        """Fit the wrapped regressor on transformed targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        **fit_params : dict
            Additional parameters forwarded to the wrapped regressor's
            ``fit`` method (e.g. ``sample_weight``).

        Returns
        -------
        self
            Fitted TargetTransformer.
        """
        X, y = self._validate_data(X, y, reset=True)
        y = y.astype(np.float64)

        if self.method not in _VALID_METHODS:
            raise ValueError(
                f"Unknown method '{self.method}'. Must be one of {sorted(_VALID_METHODS)}."
            )

        # Handle constant target edge case
        if np.all(y == y[0]):
            self._log("Target is constant; using 'none' transform.", level="warn")
            self.method_ = "none"
            self.lambda_ = None
            self.qt_ = None
            self.y_train_sorted_ = None
            self.regressor_ = clone(self.regressor)
            self.regressor_.fit(X, y, **fit_params)
            self._is_fitted = True
            return self

        # Resolve 'auto'
        if self.method == "auto":
            self.method_ = self._select_auto(y)
            self._log(f"Auto-selected method: '{self.method_}'")
        else:
            self.method_ = self.method

        # Initialize transform state
        self.lambda_ = None
        self.qt_ = None
        self.y_train_sorted_ = None

        # Apply forward transform
        y_transformed = self._forward(y, fit=True)

        # Fit the regressor on transformed targets
        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_transformed, **fit_params)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict(self, X) -> np.ndarray:
        """Predict target values, inverse-transforming the regressor's output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values in the original scale.
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        y_pred_transformed = self.regressor_.predict(X)
        return self._inverse(y_pred_transformed)

    def predict_proba(self, X) -> np.ndarray:
        """Pass through to the wrapped regressor's predict_proba, if available.

        Some regressors (e.g. NGBoost) support probabilistic predictions.
        This method delegates directly without inverse-transforming, as the
        semantics are regressor-specific.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray
            Whatever the wrapped regressor returns from predict_proba.

        Raises
        ------
        AttributeError
            If the wrapped regressor does not support predict_proba.
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        if not hasattr(self.regressor_, "predict_proba"):
            raise AttributeError(
                f"The wrapped regressor {type(self.regressor_).__name__} "
                f"does not support predict_proba."
            )
        return self.regressor_.predict_proba(X)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the wrapped regressor.

        Returns
        -------
        ndarray of shape (n_features,)
            Feature importances.

        Raises
        ------
        AttributeError
            If the wrapped regressor does not expose feature_importances_.
        """
        self._check_is_fitted()
        if hasattr(self.regressor_, "feature_importances_"):
            return self.regressor_.feature_importances_
        raise AttributeError(
            f"The wrapped regressor {type(self.regressor_).__name__} "
            f"does not expose feature_importances_."
        )

    # ------------------------------------------------------------------
    # Auto selection
    # ------------------------------------------------------------------

    def _select_auto(self, y: np.ndarray) -> str:
        """Select the best transform automatically based on normality.

        Strategy:
        1. Compute Shapiro-Wilk statistic on raw y.
        2. If y is already normal (p > 0.05), use 'none'.
        3. Otherwise, try Box-Cox (if applicable) and Yeo-Johnson.
        4. Return whichever produces the highest Shapiro-Wilk p-value.

        Parameters
        ----------
        y : ndarray
            Target array.

        Returns
        -------
        str
            Selected method name.
        """
        from scipy import stats

        # Shapiro-Wilk has a sample size limit; subsample if needed
        y_test = y
        if len(y) > 5000:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(y), 5000, replace=False)
            y_test = y[idx]

        _, p_raw = stats.shapiro(y_test)
        self._log(f"Raw target Shapiro-Wilk p={p_raw:.4f}")

        if p_raw > 0.05:
            return "none"

        candidates: dict[str, float] = {}

        # Try Yeo-Johnson (always applicable)
        try:
            y_yj, lam_yj = stats.yeojohnson(y)
            y_yj_test = y_yj if len(y) <= 5000 else y_yj[idx]
            _, p_yj = stats.shapiro(y_yj_test)
            candidates["yeo_johnson"] = p_yj
            self._log(f"Yeo-Johnson Shapiro-Wilk p={p_yj:.4f} (lambda={lam_yj:.4f})")
        except Exception:
            pass

        # Try Box-Cox (requires strictly positive)
        if np.all(y > 0):
            try:
                y_bc, lam_bc = stats.boxcox(y)
                y_bc_test = y_bc if len(y) <= 5000 else y_bc[idx]
                _, p_bc = stats.shapiro(y_bc_test)
                candidates["box_cox"] = p_bc
                self._log(f"Box-Cox Shapiro-Wilk p={p_bc:.4f} (lambda={lam_bc:.4f})")
            except Exception:
                pass

        if not candidates:
            return "yeo_johnson"

        best = max(candidates, key=candidates.get)
        return best

    # ------------------------------------------------------------------
    # Forward / inverse transforms
    # ------------------------------------------------------------------

    def _forward(self, y: np.ndarray, fit: bool = False) -> np.ndarray:
        """Apply the forward transform to target values.

        Parameters
        ----------
        y : ndarray
            Target values.
        fit : bool
            Whether this is the fitting step (fit transform parameters).

        Returns
        -------
        ndarray
            Transformed target values.
        """
        method = self.method_

        if method == "none":
            return y.copy()

        elif method == "log":
            _check_positive(y, method)
            return np.log(y)

        elif method == "log1p":
            _check_nonneg(y, method)
            return np.log1p(y)

        elif method == "sqrt":
            _check_nonneg(y, method)
            return np.sqrt(y)

        elif method == "box_cox":
            from scipy import stats
            _check_positive(y, method)
            if fit:
                y_t, self.lambda_ = stats.boxcox(y)
                return y_t
            else:
                return stats.boxcox(y, lmbda=self.lambda_)

        elif method == "yeo_johnson":
            from scipy import stats
            if fit:
                y_t, self.lambda_ = stats.yeojohnson(y)
                return y_t
            else:
                return stats.yeojohnson(y, lmbda=self.lambda_)

        elif method == "quantile":
            if fit:
                self.qt_ = _SklearnQuantileTransformer(
                    output_distribution="normal",
                    random_state=self.random_state,
                )
                return self.qt_.fit_transform(y.reshape(-1, 1)).ravel()
            else:
                return self.qt_.transform(y.reshape(-1, 1)).ravel()

        elif method == "rank":
            if fit:
                self.y_train_sorted_ = np.sort(y)
            n = len(y)
            # Rank transform: map to [0, 1] via ranks then to normal quantiles
            ranks = np.searchsorted(self.y_train_sorted_, y, side="right")
            # Clip to valid quantile range
            quantiles = np.clip(ranks / len(self.y_train_sorted_), 1e-6, 1 - 1e-6)
            from scipy import stats
            return stats.norm.ppf(quantiles)

        else:
            raise ValueError(f"Unknown method '{method}'.")

    def _inverse(self, y: np.ndarray) -> np.ndarray:
        """Apply the inverse transform to predicted values.

        Parameters
        ----------
        y : ndarray
            Transformed predicted values.

        Returns
        -------
        ndarray
            Predictions in the original target scale.
        """
        method = self.method_

        if method == "none":
            return y.copy()

        elif method == "log":
            return np.exp(y)

        elif method == "log1p":
            return np.expm1(y)

        elif method == "sqrt":
            # Clip to avoid negative values from numeric noise
            return np.square(np.clip(y, 0.0, None))

        elif method == "box_cox":
            from scipy.special import inv_boxcox
            return inv_boxcox(y, self.lambda_)

        elif method == "yeo_johnson":
            from scipy.special import inv_boxcox
            # Yeo-Johnson inverse is not provided by scipy directly.
            # Implement manually following the Yeo-Johnson definition.
            return _inv_yeojohnson(y, self.lambda_)

        elif method == "quantile":
            return self.qt_.inverse_transform(y.reshape(-1, 1)).ravel()

        elif method == "rank":
            from scipy import stats
            # Map normal quantiles back to [0, 1], then interpolate
            quantiles = stats.norm.cdf(y)
            # Interpolate back to original scale using stored sorted targets
            n = len(self.y_train_sorted_)
            indices = quantiles * (n - 1)
            idx_low = np.clip(np.floor(indices).astype(int), 0, n - 1)
            idx_high = np.clip(np.ceil(indices).astype(int), 0, n - 1)
            frac = indices - idx_low
            return (
                self.y_train_sorted_[idx_low] * (1 - frac)
                + self.y_train_sorted_[idx_high] * frac
            )

        else:
            raise ValueError(f"Unknown method '{method}'.")


def _inv_yeojohnson(y: np.ndarray, lam: float) -> np.ndarray:
    """Inverse Yeo-Johnson transform.

    The Yeo-Johnson transform is defined piecewise:

    For x >= 0:
        if lam != 0: y = ((x + 1)^lam - 1) / lam
        if lam == 0: y = log(x + 1)

    For x < 0:
        if lam != 2: y = -((-x + 1)^(2 - lam) - 1) / (2 - lam)
        if lam == 2: y = -log(-x + 1)

    This function inverts those definitions.

    Parameters
    ----------
    y : ndarray
        Transformed values.
    lam : float
        Lambda parameter.

    Returns
    -------
    ndarray
        Original-scale values.
    """
    x = np.zeros_like(y, dtype=np.float64)

    pos = y >= 0
    neg = ~pos

    # Invert positive branch
    if np.any(pos):
        if np.abs(lam) < 1e-12:
            # lam ~ 0: y = log(x + 1) => x = exp(y) - 1
            x[pos] = np.exp(y[pos]) - 1
        else:
            # y = ((x+1)^lam - 1) / lam => x = (y*lam + 1)^(1/lam) - 1
            x[pos] = np.power(y[pos] * lam + 1, 1.0 / lam) - 1

    # Invert negative branch
    if np.any(neg):
        if np.abs(lam - 2) < 1e-12:
            # lam ~ 2: y = -log(-x + 1) => x = 1 - exp(-y)
            x[neg] = 1 - np.exp(-y[neg])
        else:
            # y = -((-x+1)^(2-lam) - 1) / (2-lam)
            # => -x + 1 = (-y*(2-lam) + 1)^(1/(2-lam))
            # => x = 1 - (-y*(2-lam) + 1)^(1/(2-lam))
            x[neg] = 1 - np.power(-y[neg] * (2 - lam) + 1, 1.0 / (2 - lam))

    return x


class TargetQuantileTransformer(EndgameEstimator, RegressorMixin):
    """Convenience wrapper applying QuantileTransformer to the target.

    This is a specialized shortcut for ``TargetTransformer(method='quantile')``.
    It wraps a regressor and normalizes the target via sklearn's
    QuantileTransformer before fitting.

    Parameters
    ----------
    regressor : estimator
        Any sklearn-compatible regressor.
    n_quantiles : int, default=1000
        Number of quantiles for the QuantileTransformer.
    output_distribution : str, default='normal'
        Output distribution: 'normal' or 'uniform'.
    subsample : int, default=100000
        Subsample size for quantile estimation.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    regressor_ : estimator
        The fitted regressor.
    qt_ : QuantileTransformer
        The fitted target QuantileTransformer.
    feature_importances_ : ndarray
        Delegated from the wrapped regressor, if available.

    Examples
    --------
    >>> from sklearn.linear_model import Ridge
    >>> from endgame.preprocessing.target_transform import TargetQuantileTransformer
    >>> model = TargetQuantileTransformer(
    ...     regressor=Ridge(),
    ...     n_quantiles=500,
    ...     output_distribution='normal',
    ... )
    >>> model.fit(X_train, y_train)
    >>> preds = model.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        regressor: Any = None,
        n_quantiles: int = 1000,
        output_distribution: str = "normal",
        subsample: int = 100_000,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        if regressor is None:
            raise TypeError(
                "TargetQuantileTransformer requires a regressor. Pass a "
                "sklearn-compatible regressor via the 'regressor' parameter."
            )
        self.regressor = regressor
        self.n_quantiles = n_quantiles
        self.output_distribution = output_distribution
        self.subsample = subsample

    def fit(self, X, y, **fit_params) -> "TargetQuantileTransformer":
        """Fit the wrapped regressor on quantile-transformed targets.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        **fit_params : dict
            Additional parameters forwarded to the regressor.

        Returns
        -------
        self
        """
        X, y = self._validate_data(X, y, reset=True)
        y = y.astype(np.float64)

        self.qt_ = _SklearnQuantileTransformer(
            n_quantiles=min(self.n_quantiles, len(y)),
            output_distribution=self.output_distribution,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        y_transformed = self.qt_.fit_transform(y.reshape(-1, 1)).ravel()

        self.regressor_ = clone(self.regressor)
        self.regressor_.fit(X, y_transformed, **fit_params)
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values, inverse-transforming the output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values in the original scale.
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        y_pred_transformed = self.regressor_.predict(X)
        return self.qt_.inverse_transform(
            y_pred_transformed.reshape(-1, 1)
        ).ravel()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the wrapped regressor."""
        self._check_is_fitted()
        if hasattr(self.regressor_, "feature_importances_"):
            return self.regressor_.feature_importances_
        raise AttributeError(
            f"The wrapped regressor {type(self.regressor_).__name__} "
            f"does not expose feature_importances_."
        )
