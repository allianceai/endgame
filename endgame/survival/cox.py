"""Cox Proportional Hazards models.

Provides a pure-NumPy Cox PH implementation and an elastic-net regularised
variant that optionally delegates to *scikit-survival*.

Example
-------
>>> import numpy as np
>>> from endgame.survival.cox import CoxPHRegressor
>>> from endgame.survival.base import make_survival_y
>>> X = np.random.randn(200, 5)
>>> y = make_survival_y(np.abs(np.random.randn(200)) + 0.1,
...                     np.random.rand(200) > 0.3)
>>> model = CoxPHRegressor(penalizer=0.1)
>>> model.fit(X, y)  # doctest: +SKIP
>>> risk = model.predict(X)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import optimize, stats

from endgame.survival.base import (
    BaseSurvivalEstimator,
    _check_survival_y,
    _get_time_event,
    SURVIVAL_DTYPE,
)

try:
    from sksurv.linear_model import CoxnetSurvivalAnalysis as _CoxNet

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _breslow_baseline(
    times: np.ndarray,
    events: np.ndarray,
    X: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Breslow baseline cumulative hazard.

    Parameters
    ----------
    times : (n,) observed times
    events : (n,) boolean event indicator
    X : (n, p) features
    beta : (p,) coefficients

    Returns
    -------
    unique_times : sorted unique event times
    baseline_cumulative_hazard : H_0(t) at each unique event time
    """
    risk_scores = np.exp(X @ beta)
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_events = events[order]
    sorted_risk = risk_scores[order]

    # Reverse cumulative sum of risk scores (risk set sums)
    risk_set_sum = np.cumsum(sorted_risk[::-1])[::-1]

    # Only consider event times
    event_mask = sorted_events.astype(bool)
    event_times = sorted_times[event_mask]
    event_risk_set = risk_set_sum[event_mask]

    # Handle tied event times: for ties, take the largest risk set sum
    unique_times, first_idx = np.unique(event_times, return_index=True)
    # For each unique time, count events and use the risk set at the first
    # occurrence (which has the largest risk set since sorted ascending)
    increments = np.zeros(len(unique_times))
    for i, ut in enumerate(unique_times):
        mask = event_times == ut
        n_events_at_t = mask.sum()
        increments[i] = n_events_at_t / event_risk_set[first_idx[i]]

    baseline_cumhaz = np.cumsum(increments)
    return unique_times, baseline_cumhaz


def _concordance_index_fast(
    times: np.ndarray, events: np.ndarray, risk: np.ndarray
) -> float:
    """Harrell's concordance index (quick O(n^2) version)."""
    concordant = 0
    discordant = 0
    tied_risk = 0
    n = len(times)
    for i in range(n):
        if not events[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            if times[j] < times[i]:
                continue
            if times[j] == times[i] and not events[j]:
                continue
            if risk[i] > risk[j]:
                concordant += 1
            elif risk[i] < risk[j]:
                discordant += 1
            else:
                tied_risk += 1
    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


# ---------------------------------------------------------------------------
# CoxPHRegressor
# ---------------------------------------------------------------------------


class CoxPHRegressor(BaseSurvivalEstimator):
    """Cox Proportional Hazards model (pure NumPy).

    Maximises the partial likelihood via L-BFGS-B with optional elastic-net
    regularisation.

    Parameters
    ----------
    penalizer : float, default=0.0
        Regularisation strength.  When ``l1_ratio=0`` this is pure L2
        (ridge); when ``l1_ratio=1`` it is pure L1 (lasso).
    l1_ratio : float, default=0.0
        Elastic-net mixing: 0 = L2, 1 = L1.
    max_iter : int, default=100
        Maximum L-BFGS-B iterations.
    tol : float, default=1e-9
        Convergence tolerance for the optimiser.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coefficients_ : ndarray of shape (n_features,)
    hazard_ratios_ : ndarray of shape (n_features,)
        ``exp(coefficients_)``
    baseline_survival_ : ndarray
    baseline_cumulative_hazard_ : ndarray
    concordance_index_ : float
    log_likelihood_ : float

    Example
    -------
    >>> model = CoxPHRegressor(penalizer=0.01)
    """

    def __init__(
        self,
        penalizer: float = 0.0,
        l1_ratio: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-9,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.tol = tol

    # -- internal -----------------------------------------------------------

    @staticmethod
    def _neg_log_partial_likelihood(
        beta: np.ndarray,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        penalizer: float,
        l1_ratio: float,
    ) -> tuple[float, np.ndarray]:
        """Negative log partial likelihood and its gradient.

        Uses the Breslow approximation for ties.
        """
        n, p = X.shape
        xb = X @ beta  # (n,)
        # For numerical stability subtract max within risk sets.
        # We compute per-event contribution.
        order = np.argsort(-times)  # descending time
        sorted_times = times[order]
        sorted_events = events[order]
        sorted_xb = xb[order]
        sorted_X = X[order]

        # Cumulative sums in ascending risk-set order (descending time = we
        # walk from largest time to smallest, accumulating the risk set).
        exp_xb = np.exp(sorted_xb - sorted_xb.max())  # stabilise
        # Re-scale: we need true exp(xb) for the ratio, but the shift cancels
        # in the log-sum-exp formulation below. Let's be precise:
        exp_xb_true = np.exp(sorted_xb)

        cum_exp = np.cumsum(exp_xb_true)  # cumulative risk set sum
        cum_X_exp = np.cumsum(sorted_X * exp_xb_true[:, None], axis=0)

        # Log partial likelihood
        event_mask = sorted_events.astype(bool)
        log_risk_set = np.log(cum_exp)

        ll = np.sum(sorted_xb[event_mask] - log_risk_set[event_mask])

        # Gradient
        weighted_mean_X = cum_X_exp[event_mask] / cum_exp[event_mask, None]
        grad = np.sum(sorted_X[event_mask] - weighted_mean_X, axis=0)

        # Regularisation (penalise the *positive* ll so negate for minimiser)
        l2_pen = 0.5 * penalizer * (1 - l1_ratio) * np.dot(beta, beta)
        l1_pen = penalizer * l1_ratio * np.sum(np.abs(beta))
        ll -= l2_pen + l1_pen

        grad -= penalizer * (1 - l1_ratio) * beta
        # L1 sub-gradient
        grad -= penalizer * l1_ratio * np.sign(beta)

        # We minimise negative log-likelihood
        return -ll, -grad

    # -- public API ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: Any) -> CoxPHRegressor:
        """Fit Cox PH model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target (structured array, tuple, or DataFrame)

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        times, events = _get_time_event(y)

        n, p = X.shape
        beta0 = np.zeros(p)

        self._log(f"Fitting CoxPH on {n} samples, {p} features, "
                  f"{events.sum()} events")

        result = optimize.minimize(
            fun=self._neg_log_partial_likelihood,
            x0=beta0,
            args=(X, times, events, self.penalizer, self.l1_ratio),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.max_iter, "ftol": self.tol, "gtol": self.tol},
        )

        self.coefficients_ = result.x
        self.hazard_ratios_ = np.exp(self.coefficients_)
        self.log_likelihood_ = -result.fun

        # Breslow baseline
        self._event_times_breslow, self.baseline_cumulative_hazard_ = (
            _breslow_baseline(times, events, X, self.coefficients_)
        )
        self.baseline_survival_ = np.exp(-self.baseline_cumulative_hazard_)

        # Store training data summary for Schoenfeld residuals
        self._train_X = X
        self._train_times = times
        self._train_events = events

        self.is_fitted_ = True

        # Training concordance
        risk = self.predict(X)
        self.concordance_index_ = _concordance_index_fast(times, events, risk)

        self._log(f"Converged: {result.success}, C-index: "
                  f"{self.concordance_index_:.4f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores ``exp(X @ beta)``.

        Higher values indicate higher risk.
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.exp(X @ self.coefficients_)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function S(t|X) = S_0(t)^exp(X@beta).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional
            Evaluation times.  If *None*, uses training event times.

        Returns
        -------
        S : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if times is None:
            times = self._event_times_breslow
        else:
            times = np.asarray(times, dtype=np.float64)

        baseline_surv = self._interpolate_baseline(times, survival=True)
        risk = np.exp(X @ self.coefficients_)  # (n,)
        # S(t|X) = S_0(t)^exp(X@beta)
        # log S = exp(X@beta) * log S_0(t)
        log_baseline = np.log(np.clip(baseline_surv, 1e-300, None))
        S = np.exp(risk[:, None] * log_baseline[None, :])
        return S

    def predict_cumulative_hazard(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict cumulative hazard H(t|X) = H_0(t) * exp(X@beta).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional

        Returns
        -------
        H : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if times is None:
            times = self._event_times_breslow
        else:
            times = np.asarray(times, dtype=np.float64)

        baseline_cumhaz = self._interpolate_baseline(times, survival=False)
        risk = np.exp(X @ self.coefficients_)
        return risk[:, None] * baseline_cumhaz[None, :]

    def predict_median_survival_time(self, X: np.ndarray) -> np.ndarray:
        """Find t where S(t|X) = 0.5."""
        self._check_is_fitted()
        S = self.predict_survival_function(X)
        times = self._event_times_breslow
        n = S.shape[0]
        medians = np.full(n, np.inf)
        for i in range(n):
            idx = np.where(S[i] <= 0.5)[0]
            if len(idx) > 0:
                medians[i] = times[idx[0]]
        return medians

    def summary(self) -> dict[str, Any]:
        """Return summary statistics for each coefficient.

        Returns
        -------
        dict with keys: 'coef', 'exp(coef)', 'se', 'z', 'p' per feature.
        """
        self._check_is_fitted()
        beta = self.coefficients_
        X = self._train_X
        times = self._train_times
        events = self._train_events

        # Approximate standard errors via the Hessian (observed information)
        n, p = X.shape
        order = np.argsort(-times)
        sorted_X = X[order]
        sorted_events = events[order]
        exp_xb = np.exp(sorted_X @ beta)

        cum_exp = np.cumsum(exp_xb)
        cum_X_exp = np.cumsum(sorted_X * exp_xb[:, None], axis=0)
        cum_X2_exp = np.cumsum(
            (sorted_X[:, :, None] * sorted_X[:, None, :])
            * exp_xb[:, None, None],
            axis=0,
        )

        event_mask = sorted_events.astype(bool)
        # Information matrix
        I = np.zeros((p, p))
        for idx in np.where(event_mask)[0]:
            w = cum_X_exp[idx] / cum_exp[idx]
            V = cum_X2_exp[idx] / cum_exp[idx] - np.outer(w, w)
            I += V

        try:
            se = np.sqrt(np.diag(np.linalg.inv(I)))
        except np.linalg.LinAlgError:
            se = np.full(p, np.nan)

        z = beta / se
        p_values = 2 * stats.norm.sf(np.abs(z))

        return {
            "coef": beta,
            "exp(coef)": np.exp(beta),
            "se": se,
            "z": z,
            "p": p_values,
        }

    def check_proportional_hazards(
        self, X: np.ndarray, y: Any
    ) -> dict[str, float]:
        """Schoenfeld residuals test for proportional hazards assumption.

        Returns a dict mapping feature index to p-value.  Small p-values
        indicate violation of proportional hazards for that feature.
        """
        self._check_is_fitted()
        X, y = self._validate_survival_data(X, y, reset=False)
        times, events = _get_time_event(y)
        n, p = X.shape

        beta = self.coefficients_
        order = np.argsort(times)
        sorted_times = times[order]
        sorted_events = events[order]
        sorted_X = X[order]

        exp_xb = np.exp(sorted_X @ beta)

        # Compute Schoenfeld residuals for each event
        schoenfeld = []
        event_times_list = []
        for i in range(n):
            if not sorted_events[i]:
                continue
            # Risk set: all j with time >= time_i
            risk_mask = sorted_times >= sorted_times[i]
            risk_exp = exp_xb[risk_mask]
            risk_X = sorted_X[risk_mask]
            weighted_mean = (risk_X * risk_exp[:, None]).sum(axis=0) / risk_exp.sum()
            schoenfeld.append(sorted_X[i] - weighted_mean)
            event_times_list.append(sorted_times[i])

        if len(schoenfeld) == 0:
            return {j: 1.0 for j in range(p)}

        schoenfeld = np.array(schoenfeld)  # (n_events, p)
        event_times_arr = np.array(event_times_list)

        # Correlation test with time
        p_values = {}
        for j in range(p):
            if np.std(schoenfeld[:, j]) < 1e-15:
                p_values[j] = 1.0
                continue
            _, pval = stats.spearmanr(event_times_arr, schoenfeld[:, j])
            p_values[j] = float(pval)

        return p_values

    # -- internal helpers ---------------------------------------------------

    def _interpolate_baseline(
        self, times: np.ndarray, survival: bool = True
    ) -> np.ndarray:
        """Interpolate baseline cumulative hazard / survival at given times.

        Uses step-function (last value before or at t).
        """
        bt = self._event_times_breslow
        bh = self.baseline_cumulative_hazard_

        indices = np.searchsorted(bt, times, side="right") - 1
        cumhaz = np.where(indices < 0, 0.0, bh[np.clip(indices, 0, len(bh) - 1)])

        if survival:
            return np.exp(-cumhaz)
        return cumhaz


# ---------------------------------------------------------------------------
# CoxNetRegressor
# ---------------------------------------------------------------------------


class CoxNetRegressor(BaseSurvivalEstimator):
    """Elastic-net regularised Cox model.

    Delegates to ``sksurv.linear_model.CoxnetSurvivalAnalysis`` when
    available; otherwise falls back to :class:`CoxPHRegressor` with
    elastic-net penalty.

    Parameters
    ----------
    n_alphas : int, default=100
        Number of alphas along the regularisation path (sksurv only).
    l1_ratio : float, default=0.5
        Elastic-net mixing parameter.
    alpha_min_ratio : float, default=0.01
        Ratio of smallest to largest alpha (sksurv only).
    max_iter : int, default=100000
        Maximum iterations for the optimiser.
    tol : float, default=1e-7
        Convergence tolerance.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    alphas_ : ndarray
        Regularisation path (sksurv) or single alpha.
    coef_path_ : ndarray
        Coefficient values along the path (sksurv) or single set.
    """

    def __init__(
        self,
        n_alphas: int = 100,
        l1_ratio: float = 0.5,
        alpha_min_ratio: float = 0.01,
        max_iter: int = 100_000,
        tol: float = 1e-7,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_alphas = n_alphas
        self.l1_ratio = l1_ratio
        self.alpha_min_ratio = alpha_min_ratio
        self.max_iter = max_iter
        self.tol = tol

    # -- public API ---------------------------------------------------------

    def fit(self, X: np.ndarray, y: Any) -> CoxNetRegressor:
        """Fit elastic-net Cox model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        times, events = _get_time_event(y)

        if HAS_SKSURV:
            self._log("Using scikit-survival CoxnetSurvivalAnalysis")
            sksurv_y = np.array(
                [(bool(e), float(t)) for e, t in zip(events, times)],
                dtype=[("event", bool), ("time", np.float64)],
            )
            self._backend = "sksurv"
            self._model = _CoxNet(
                n_alphas=self.n_alphas,
                l1_ratio=self.l1_ratio,
                alpha_min_ratio=self.alpha_min_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self._model.fit(X, sksurv_y)
            self.alphas_ = self._model.alphas_
            self.coef_path_ = self._model.coef_
            self.coefficients_ = self._model.coef_[:, -1]
        else:
            self._log("sksurv not available; falling back to CoxPHRegressor")
            self._backend = "fallback"
            # Approximate: single penalizer = median-ish alpha
            pen = 0.1  # reasonable default
            self._model = CoxPHRegressor(
                penalizer=pen,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                tol=self.tol,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self._model.fit(X, y)
            self.coefficients_ = self._model.coefficients_
            self.alphas_ = np.array([pen])
            self.coef_path_ = self.coefficients_[:, None]

        # Breslow baseline (shared logic)
        self._event_times_breslow, self._baseline_cumhaz = _breslow_baseline(
            times, events, X, self.coefficients_
        )
        self._baseline_surv = np.exp(-self._baseline_cumhaz)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores ``exp(X @ beta)``."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.exp(X @ self.coefficients_)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function S(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if times is None:
            times = self._event_times_breslow
        else:
            times = np.asarray(times, dtype=np.float64)

        bt = self._event_times_breslow
        bh = self._baseline_cumhaz
        indices = np.searchsorted(bt, times, side="right") - 1
        cumhaz = np.where(indices < 0, 0.0, bh[np.clip(indices, 0, len(bh) - 1)])
        baseline_surv = np.exp(-cumhaz)

        risk = np.exp(X @ self.coefficients_)
        log_baseline = np.log(np.clip(baseline_surv, 1e-300, None))
        return np.exp(risk[:, None] * log_baseline[None, :])
