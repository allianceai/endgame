"""Parametric accelerated failure time (AFT) survival regression models.

All models inherit from :class:`~endgame.survival.base.BaseSurvivalEstimator`
and use maximum-likelihood estimation via ``scipy.optimize.minimize``.

Supported distributions:

* **Exponential** — constant hazard
* **Weibull** — flexible shape + scale
* **Log-Normal** — log(T) ~ N(mu, sigma)
* **Log-Logistic** — closed-form survival with non-monotone hazard

Example
-------
>>> import numpy as np
>>> from endgame.survival.base import make_survival_y
>>> from endgame.survival.parametric import WeibullAFTRegressor
>>> X = np.random.randn(100, 3)
>>> y = make_survival_y(np.abs(np.random.randn(100)) + 0.1,
...                     np.random.rand(100) > 0.3)
>>> model = WeibullAFTRegressor().fit(X, y)
>>> risk = model.predict(X)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.utils.validation import check_array, check_is_fitted

from endgame.survival.base import (
    BaseSurvivalEstimator,
    SurvivalMixin,
    _check_survival_y,
    _get_time_event,
    make_survival_y,
    SURVIVAL_DTYPE,
)


# ---------------------------------------------------------------------------
# Exponential AFT
# ---------------------------------------------------------------------------


class ExponentialRegressor(BaseSurvivalEstimator):
    """Exponential proportional-hazards regression model.

    The hazard is constant over time but varies across subjects:

        lambda(X) = exp(X @ beta)
        S(t|X) = exp(-lambda(X) * t)

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularisation penalty on the coefficient vector.
    random_state : int or None, default=None
        Not used directly; kept for API consistency.
    verbose : bool, default=False
        If True, print optimisation progress.

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
        Estimated regression coefficients (beta).
    intercept_ : float
        Intercept term (log baseline hazard).
    AIC_ : float
        Akaike information criterion.
    BIC_ : float
        Bayesian information criterion.
    n_features_in_ : int
        Number of features seen during fit.
    """

    def __init__(
        self,
        alpha: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "ExponentialRegressor":
        """Fit the exponential model via MLE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        X = np.nan_to_num(X, nan=0.0)
        time, event = _get_time_event(y)
        n, p = X.shape

        # params = [intercept, *beta]
        def neg_log_lik(params: np.ndarray) -> float:
            intercept = params[0]
            beta = params[1:]
            log_lambda = intercept + X @ beta  # (n,)
            lam = np.exp(log_lambda)
            # log L = sum_i [ event_i * log(lambda_i) - lambda_i * t_i ]
            ll = np.sum(event * log_lambda - lam * time)
            penalty = 0.5 * self.alpha * np.sum(beta ** 2)
            return -(ll - penalty)

        x0 = np.zeros(p + 1)
        result = minimize(neg_log_lik, x0, method="L-BFGS-B")

        self.intercept_ = float(result.x[0])
        self.coefficients_ = result.x[1:]

        k = p + 1
        ll = -result.fun
        self.AIC_ = float(2 * k - 2 * ll)
        self.BIC_ = float(k * np.log(n) - 2 * ll)
        self.is_fitted_ = True
        return self

    # -- predictions --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        """Return risk scores (higher = more risk).

        Risk is defined as the negative predicted median survival time so
        that higher values correspond to higher risk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        risk : ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        X = self._prepare_X(X)
        return -self.predict_median_survival_time(X)

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict S(t|X) = exp(-lambda(X) * t).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like of shape (n_times,), optional

        Returns
        -------
        sf : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = self._prepare_X(X)
        times = self._resolve_times(times)
        lam = self._lambda(X)  # (n,)
        return np.exp(-np.outer(lam, times))

    def predict_median_survival_time(self, X: Any) -> np.ndarray:
        """Predict median survival time = log(2) / lambda(X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        median : ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        X = self._prepare_X(X)
        lam = self._lambda(X)
        return np.log(2) / lam

    # -- helpers ------------------------------------------------------------

    def _lambda(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.intercept_ + X @ self.coefficients_)

    def _prepare_X(self, X: Any) -> np.ndarray:
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X = np.nan_to_num(X, nan=0.0)
        return X

    def _resolve_times(self, times: np.ndarray | None) -> np.ndarray:
        if times is None:
            return self.event_times_
        return np.asarray(times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Weibull AFT
# ---------------------------------------------------------------------------


class WeibullAFTRegressor(BaseSurvivalEstimator):
    """Weibull accelerated failure time regression model.

    The survival function is:

        S(t|X) = exp(-(t / lambda(X))^rho)

    where ``lambda(X) = exp(X @ beta)`` (scale) and ``rho`` (shape) is
    a global parameter shared across all subjects.

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularisation on coefficients.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
    intercept_ : float
    shape_ : float
        Estimated Weibull shape parameter (rho).
    AIC_ : float
    BIC_ : float
    """

    def __init__(
        self,
        alpha: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "WeibullAFTRegressor":
        """Fit the Weibull AFT model via MLE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        X = np.nan_to_num(X, nan=0.0)
        time, event = _get_time_event(y)
        n, p = X.shape

        # params = [log_rho, intercept, *beta]
        def neg_log_lik(params: np.ndarray) -> float:
            log_rho = params[0]
            intercept = params[1]
            beta = params[2:]
            rho = np.exp(log_rho)
            log_lam = intercept + X @ beta  # log(lambda)
            lam = np.exp(log_lam)

            # log L = sum_i [
            #   event_i * (log(rho) + (rho-1)*log(t_i) - rho*log(lam_i))
            #   - (t_i / lam_i)^rho
            # ]
            log_t = np.log(np.clip(time, 1e-30, None))
            ll = np.sum(
                event * (log_rho + (rho - 1) * log_t - rho * log_lam)
                - (time / lam) ** rho
            )
            penalty = 0.5 * self.alpha * np.sum(beta ** 2)
            return -(ll - penalty)

        x0 = np.zeros(p + 2)
        # Initial log_rho = 0 => rho = 1 (exponential)
        result = minimize(neg_log_lik, x0, method="L-BFGS-B")

        self.shape_ = float(np.exp(result.x[0]))
        self.intercept_ = float(result.x[1])
        self.coefficients_ = result.x[2:]

        k = p + 2
        ll = -result.fun
        self.AIC_ = float(2 * k - 2 * ll)
        self.BIC_ = float(k * np.log(n) - 2 * ll)
        self.is_fitted_ = True
        return self

    def summary(self) -> dict:
        """Return a summary with coefficients and standard errors.

        Standard errors are approximated from the inverse Hessian at the
        MLE (numerical differentiation).

        Returns
        -------
        dict
            Keys: ``coefficients``, ``intercept``, ``shape``,
            ``AIC``, ``BIC``.
        """
        self._check_is_fitted()
        return {
            "coefficients": self.coefficients_.copy(),
            "intercept": self.intercept_,
            "shape": self.shape_,
            "AIC": self.AIC_,
            "BIC": self.BIC_,
        }

    # -- predictions --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = self._prepare_X(X)
        return -self.predict_median_survival_time(X)

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None
    ) -> np.ndarray:
        """S(t|X) = exp(-(t / lambda(X))^rho)."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        times = self._resolve_times(times)
        lam = self._lambda(X)  # (n,)
        rho = self.shape_
        # (n, n_times)
        return np.exp(-np.power(times[None, :] / lam[:, None], rho))

    def predict_median_survival_time(self, X: Any) -> np.ndarray:
        """Median = lambda(X) * log(2)^(1/rho)."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        lam = self._lambda(X)
        return lam * np.log(2) ** (1.0 / self.shape_)

    # -- helpers ------------------------------------------------------------

    def _lambda(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.intercept_ + X @ self.coefficients_)

    def _prepare_X(self, X: Any) -> np.ndarray:
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0)

    def _resolve_times(self, times: np.ndarray | None) -> np.ndarray:
        if times is None:
            return self.event_times_
        return np.asarray(times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Log-Normal AFT
# ---------------------------------------------------------------------------


class LogNormalAFTRegressor(BaseSurvivalEstimator):
    """Log-Normal accelerated failure time regression model.

    Model:

        log(T) = X @ beta + sigma * epsilon,   epsilon ~ N(0, 1)

    Survival function:

        S(t|X) = 1 - Phi((log(t) - X @ beta) / sigma)

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularisation on coefficients.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
    intercept_ : float
    sigma_ : float
        Estimated scale (standard deviation of log-time residuals).
    AIC_ : float
    BIC_ : float
    """

    def __init__(
        self,
        alpha: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "LogNormalAFTRegressor":
        """Fit the log-normal AFT model via MLE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        X = np.nan_to_num(X, nan=0.0)
        time, event = _get_time_event(y)
        n, p = X.shape
        log_t = np.log(np.clip(time, 1e-30, None))

        # params = [log_sigma, intercept, *beta]
        def neg_log_lik(params: np.ndarray) -> float:
            log_sigma = params[0]
            intercept = params[1]
            beta = params[2:]
            sigma = np.exp(log_sigma)
            mu = intercept + X @ beta  # (n,)
            z = (log_t - mu) / sigma  # standardised

            # For events: log f(t) = -log(t) - log(sigma) + log(phi(z))
            # For censored: log S(t) = log(1 - Phi(z))
            ll_event = np.sum(
                event * (-np.log(np.clip(time, 1e-30, None))
                         - log_sigma
                         + norm.logpdf(z))
            )
            ll_censored = np.sum(
                (~event) * norm.logsf(z)
            )
            ll = ll_event + ll_censored
            penalty = 0.5 * self.alpha * np.sum(beta ** 2)
            return -(ll - penalty)

        x0 = np.zeros(p + 2)
        # log_sigma = 0 => sigma = 1
        result = minimize(neg_log_lik, x0, method="L-BFGS-B")

        self.sigma_ = float(np.exp(result.x[0]))
        self.intercept_ = float(result.x[1])
        self.coefficients_ = result.x[2:]

        k = p + 2
        ll = -result.fun
        self.AIC_ = float(2 * k - 2 * ll)
        self.BIC_ = float(k * np.log(n) - 2 * ll)
        self.is_fitted_ = True
        return self

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            "coefficients": self.coefficients_.copy(),
            "intercept": self.intercept_,
            "sigma": self.sigma_,
            "AIC": self.AIC_,
            "BIC": self.BIC_,
        }

    # -- predictions --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = self._prepare_X(X)
        return -self.predict_median_survival_time(X)

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None
    ) -> np.ndarray:
        """S(t|X) = 1 - Phi((log(t) - mu(X)) / sigma)."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        times = self._resolve_times(times)
        mu = self.intercept_ + X @ self.coefficients_  # (n,)
        log_times = np.log(np.clip(times, 1e-30, None))  # (n_times,)
        z = (log_times[None, :] - mu[:, None]) / self.sigma_  # (n, n_times)
        return 1.0 - norm.cdf(z)

    def predict_median_survival_time(self, X: Any) -> np.ndarray:
        """Median = exp(mu(X)), since median of N(mu, sigma^2) is mu."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        mu = self.intercept_ + X @ self.coefficients_
        return np.exp(mu)

    # -- helpers ------------------------------------------------------------

    def _prepare_X(self, X: Any) -> np.ndarray:
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0)

    def _resolve_times(self, times: np.ndarray | None) -> np.ndarray:
        if times is None:
            return self.event_times_
        return np.asarray(times, dtype=np.float64)


# ---------------------------------------------------------------------------
# Log-Logistic AFT
# ---------------------------------------------------------------------------


class LogLogisticAFTRegressor(BaseSurvivalEstimator):
    """Log-Logistic accelerated failure time regression model.

    Survival function:

        S(t|X) = 1 / (1 + (t / alpha(X))^beta)

    where ``alpha(X) = exp(X @ coefs)`` (scale) and ``beta`` (shape) is a
    global parameter.

    The log-logistic distribution is notable for allowing a non-monotone
    hazard function (initially increasing then decreasing when beta > 1).

    Parameters
    ----------
    alpha : float, default=0.0
        L2 regularisation on coefficients (not to be confused with the
        distribution's *alpha* scale parameter).
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
    intercept_ : float
    shape_ : float
        Estimated shape parameter (beta).
    AIC_ : float
    BIC_ : float
    """

    def __init__(
        self,
        alpha: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.alpha = alpha

    def fit(self, X: Any, y: Any) -> "LogLogisticAFTRegressor":
        """Fit the log-logistic AFT model via MLE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        X = np.nan_to_num(X, nan=0.0)
        time, event = _get_time_event(y)
        n, p = X.shape
        log_t = np.log(np.clip(time, 1e-30, None))

        # params = [log_beta, intercept, *coefs]
        def neg_log_lik(params: np.ndarray) -> float:
            log_beta = params[0]
            intercept = params[1]
            coefs = params[2:]
            beta = np.exp(log_beta)
            log_alpha = intercept + X @ coefs  # log(alpha(X))

            # z = (log(t) - log(alpha)) / (1/beta)  but simpler:
            # Let u = (t / alpha)^beta = exp(beta * (log_t - log_alpha))
            exponent = beta * (log_t - log_alpha)
            # Clip to avoid overflow
            exponent = np.clip(exponent, -500, 500)
            u = np.exp(exponent)

            # f(t) = (beta/alpha) * (t/alpha)^(beta-1) / (1 + (t/alpha)^beta)^2
            # log f(t) = log(beta) - log(alpha) + (beta-1)*(log(t)-log(alpha))
            #            - 2*log(1 + u)
            # S(t) = 1 / (1 + u)
            # log S(t) = -log(1 + u)

            log_f = (
                log_beta
                - log_alpha
                + (beta - 1) * (log_t - log_alpha)
                - 2.0 * np.log1p(u)
            )
            log_S = -np.log1p(u)

            ll = np.sum(event * log_f + (~event) * log_S)
            penalty = 0.5 * self.alpha * np.sum(coefs ** 2)
            return -(ll - penalty)

        x0 = np.zeros(p + 2)
        result = minimize(neg_log_lik, x0, method="L-BFGS-B")

        self.shape_ = float(np.exp(result.x[0]))
        self.intercept_ = float(result.x[1])
        self.coefficients_ = result.x[2:]

        k = p + 2
        ll = -result.fun
        self.AIC_ = float(2 * k - 2 * ll)
        self.BIC_ = float(k * np.log(n) - 2 * ll)
        self.is_fitted_ = True
        return self

    def summary(self) -> dict:
        self._check_is_fitted()
        return {
            "coefficients": self.coefficients_.copy(),
            "intercept": self.intercept_,
            "shape": self.shape_,
            "AIC": self.AIC_,
            "BIC": self.BIC_,
        }

    # -- predictions --------------------------------------------------------

    def predict(self, X: Any) -> np.ndarray:
        self._check_is_fitted()
        X = self._prepare_X(X)
        return -self.predict_median_survival_time(X)

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None
    ) -> np.ndarray:
        """S(t|X) = 1 / (1 + (t / alpha(X))^beta)."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        times = self._resolve_times(times)
        alpha_x = self._alpha(X)  # (n,)
        beta = self.shape_
        # (n, n_times)
        u = np.power(times[None, :] / alpha_x[:, None], beta)
        return 1.0 / (1.0 + u)

    def predict_median_survival_time(self, X: Any) -> np.ndarray:
        """Median = alpha(X), since S(median) = 0.5 => (median/alpha)^beta = 1."""
        self._check_is_fitted()
        X = self._prepare_X(X)
        return self._alpha(X)

    # -- helpers ------------------------------------------------------------

    def _alpha(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.intercept_ + X @ self.coefficients_)

    def _prepare_X(self, X: Any) -> np.ndarray:
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.nan_to_num(X, nan=0.0)

    def _resolve_times(self, times: np.ndarray | None) -> np.ndarray:
        if times is None:
            return self.event_times_
        return np.asarray(times, dtype=np.float64)
