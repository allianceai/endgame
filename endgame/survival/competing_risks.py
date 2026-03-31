"""Competing risks survival models.

Provides models for settings where multiple event types compete to be
the first to occur.

Example
-------
>>> from endgame.survival.competing_risks import AalenJohansenEstimator
>>> import numpy as np
>>> time = np.array([1, 2, 3, 4, 5, 6, 7, 8])
>>> event_type = np.array([1, 0, 2, 1, 0, 2, 1, 0])  # 0=censored
>>> aj = AalenJohansenEstimator()
>>> aj.fit(time, event_type)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted

from endgame.survival.base import (
    BaseSurvivalEstimator,
    _check_survival_y,
    _get_time_event,
    make_survival_y,
)


# ---------------------------------------------------------------------------
# Aalen-Johansen Estimator
# ---------------------------------------------------------------------------


class AalenJohansenEstimator(BaseEstimator):
    """Aalen-Johansen non-parametric cumulative incidence estimator.

    Estimates cause-specific cumulative incidence functions (CIF) in the
    presence of competing risks.

    CIF_k(t) = sum_{t_i <= t} [d_ik / n_i] * S(t_i^-)

    where S(t^-) is the overall Kaplan-Meier survival function just
    before time t, d_ik is the number of type-k events at t_i, and
    n_i is the number at risk.

    Parameters
    ----------
    verbose : bool, default=False

    Attributes
    ----------
    cumulative_incidence_ : dict
        Maps event type → CIF array.
    timeline_ : ndarray
        Unique event times.
    event_types_ : ndarray
        Unique non-zero event types.
    n_events_per_type_ : dict
        Number of events per type.

    Example
    -------
    >>> aj = AalenJohansenEstimator()
    >>> aj.fit(time, event_type)
    >>> cif = aj.predict_cumulative_incidence()
    """

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(
        self,
        time: Any,
        event_type: np.ndarray | None = None,
    ) -> AalenJohansenEstimator:
        """Fit Aalen-Johansen estimator.

        Parameters
        ----------
        time : array-like of shape (n_samples,) or structured survival array
            Observed times. If a structured survival array (with 'time'
            and 'event' fields), the time is extracted automatically.
        event_type : array-like of shape (n_samples,), optional
            Event type indicator: 0 = censored, 1..K = competing events.
            Required unless time is a plain array and event_type is given.

        Returns
        -------
        self
        """
        # Accept structured survival array
        if isinstance(time, np.ndarray) and time.dtype.names is not None:
            y = _check_survival_y(time)
            t, e = _get_time_event(y)
            time = t
            if event_type is None:
                event_type = e.astype(int)
        else:
            time = np.asarray(time, dtype=np.float64)

        event_type = np.asarray(event_type, dtype=int)

        if len(time) != len(event_type):
            raise ValueError("time and event_type must have the same length")

        self.event_types_ = np.sort(np.unique(event_type[event_type > 0]))
        n_types = len(self.event_types_)

        # Count events per type
        self.n_events_per_type_ = {}
        for k in self.event_types_:
            self.n_events_per_type_[k] = int((event_type == k).sum())

        # Sort by time
        order = np.argsort(time)
        time_sorted = time[order]
        etype_sorted = event_type[order]

        unique_times = np.sort(np.unique(time_sorted[etype_sorted > 0]))
        self.timeline_ = unique_times

        n_total = len(time)

        # Compute overall Kaplan-Meier S(t^-)
        # and cause-specific cumulative incidence
        cif = {k: np.zeros(len(unique_times)) for k in self.event_types_}
        surv = 1.0  # S(t^-)

        for idx, t in enumerate(unique_times):
            # Number at risk at time t
            n_at_risk = np.sum(time_sorted >= t)
            if n_at_risk == 0:
                break

            # Events of each type at time t
            at_t = time_sorted == t
            for k in self.event_types_:
                d_k = np.sum(at_t & (etype_sorted == k))
                cif_increment = (d_k / n_at_risk) * surv
                if idx == 0:
                    cif[k][idx] = cif_increment
                else:
                    cif[k][idx] = cif[k][idx - 1] + cif_increment

            # Update overall survival (all event types reduce survival)
            d_total = np.sum(at_t & (etype_sorted > 0))
            surv *= 1.0 - d_total / n_at_risk

            # Carry forward CIF for types with no events at this time
            for k in self.event_types_:
                if idx > 0 and np.sum(at_t & (etype_sorted == k)) == 0:
                    cif[k][idx] = cif[k][idx - 1]

        self.cumulative_incidence_ = cif
        self.n_features_in_ = 0
        self.is_fitted_ = True
        return self

    def predict_cumulative_incidence(
        self,
        times: np.ndarray | None = None,
    ) -> dict[int, np.ndarray]:
        """Predict cumulative incidence functions.

        Parameters
        ----------
        times : array-like, optional
            Time points. If None, uses the event times from training.

        Returns
        -------
        cif : dict
            Maps event type → CIF values at requested times.
        """
        check_is_fitted(self, ["cumulative_incidence_"])

        if times is None:
            return {k: v.copy() for k, v in self.cumulative_incidence_.items()}

        times = np.asarray(times, dtype=np.float64)
        result = {}
        for k, cif_vals in self.cumulative_incidence_.items():
            # Step function interpolation
            result[k] = np.interp(
                times,
                self.timeline_,
                cif_vals,
                left=0.0,
                right=cif_vals[-1] if len(cif_vals) > 0 else 0.0,
            )
        return result


# ---------------------------------------------------------------------------
# Cause-Specific Cox
# ---------------------------------------------------------------------------


class CauseSpecificCoxRegressor(BaseSurvivalEstimator):
    """Cause-specific Cox regression for competing risks.

    Fits a separate Cox proportional hazards model for each event type.
    For event type k, treats type-k events as "events" and everything
    else (other event types + censoring) as "censored".

    Parameters
    ----------
    penalizer : float, default=0.0
        L2 penalty for each cause-specific Cox model.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    cause_models_ : dict
        Maps event type → fitted CoxPHRegressor.
    event_types_ : ndarray
        Unique non-zero event types.
    """

    def __init__(
        self,
        penalizer: float = 0.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.penalizer = penalizer

    def fit(
        self,
        X: Any,
        y: Any,
        event_type: np.ndarray | None = None,
    ) -> CauseSpecificCoxRegressor:
        """Fit cause-specific Cox models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : structured survival array
            Must contain 'time' field. The 'event' field is overridden
            by event_type if provided.
        event_type : array-like of shape (n_samples,), optional
            Event type indicator: 0 = censored, 1..K = competing events.
            If None, uses y['event'] as binary (single event type).

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        if event_type is None:
            event_type = event.astype(int)

        event_type = np.asarray(event_type, dtype=int)
        self.event_types_ = np.sort(np.unique(event_type[event_type > 0]))

        from endgame.survival.cox import CoxPHRegressor

        self.cause_models_ = {}
        for k in self.event_types_:
            # For cause k: event=1 if type==k, else 0
            cause_event = (event_type == k).astype(bool)
            cause_y = make_survival_y(time, cause_event)

            model = CoxPHRegressor(
                penalizer=self.penalizer,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            model.fit(X, cause_y)
            self.cause_models_[k] = model
            self._log(f"Fitted Cox for cause {k}: C-index={model.concordance_index_:.3f}")

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Overall risk score (sum of cause-specific risks)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        total_risk = np.zeros(len(X))
        for k, model in self.cause_models_.items():
            total_risk += model.predict(X)
        return total_risk

    def predict_cumulative_incidence(
        self,
        X: Any,
        times: np.ndarray | None = None,
        cause: int = 1,
    ) -> np.ndarray:
        """Predict cumulative incidence for a specific cause.

        Uses the cause-specific hazard and overall survival:
        CIF_k(t|X) = integral_0^t h_k(s|X) * S(s^-|X) ds

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional
        cause : int, default=1
            Which cause to compute CIF for.

        Returns
        -------
        cif : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = self._to_numpy(X)

        if cause not in self.cause_models_:
            raise ValueError(
                f"Cause {cause} not found. Available: {list(self.cause_models_.keys())}"
            )

        if times is None:
            times = self.event_times_

        times = np.asarray(times, dtype=np.float64)
        n = len(X)
        n_times = len(times)

        # Get cause-specific cumulative hazard
        cause_model = self.cause_models_[cause]
        cause_cumhaz = cause_model.predict_cumulative_hazard(X, times)

        # Get overall survival from all causes
        overall_cumhaz = np.zeros((n, n_times))
        for k, model in self.cause_models_.items():
            overall_cumhaz += model.predict_cumulative_hazard(X, times)
        overall_surv = np.exp(-overall_cumhaz)

        # CIF = cumulative sum of (cause-specific hazard increment * overall survival)
        cif = np.zeros((n, n_times))
        for j in range(n_times):
            if j == 0:
                dH = cause_cumhaz[:, 0]
                S_prev = np.ones(n)
            else:
                dH = cause_cumhaz[:, j] - cause_cumhaz[:, j - 1]
                S_prev = overall_surv[:, j - 1]
            cif[:, j] = cif[:, j - 1] + dH * S_prev if j > 0 else dH * S_prev

        return cif


# ---------------------------------------------------------------------------
# Fine-Gray
# ---------------------------------------------------------------------------


class FineGrayRegressor(BaseSurvivalEstimator):
    """Fine-Gray subdistribution hazard model for competing risks.

    Models the subdistribution hazard directly, allowing direct
    estimation of the cumulative incidence function for a specific cause.

    The key idea: subjects who experience a competing event remain in
    the risk set (with decreasing weights from IPCW).

    Parameters
    ----------
    cause_of_interest : int, default=1
        The event type to model.
    penalizer : float, default=0.0
        L2 regularization.
    max_iter : int, default=100
    tol : float, default=1e-9
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coefficients_ : ndarray
        Regression coefficients for subdistribution hazard.
    baseline_cif_ : ndarray
        Baseline cumulative incidence function.
    """

    def __init__(
        self,
        cause_of_interest: int = 1,
        penalizer: float = 0.0,
        max_iter: int = 100,
        tol: float = 1e-9,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.cause_of_interest = cause_of_interest
        self.penalizer = penalizer
        self.max_iter = max_iter
        self.tol = tol

    def fit(
        self,
        X: Any,
        y: Any,
        event_type: np.ndarray | None = None,
    ) -> FineGrayRegressor:
        """Fit Fine-Gray model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : structured survival array
        event_type : array-like of shape (n_samples,), optional
            0 = censored, 1..K = event types.
        """
        from scipy.optimize import minimize

        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        if event_type is None:
            event_type = event.astype(int)
        event_type = np.asarray(event_type, dtype=int)

        n, p = X.shape
        k = self.cause_of_interest

        # Indicator: event of interest
        is_event_k = event_type == k
        # Indicator: competing event (different event type)
        is_competing = (event_type > 0) & (event_type != k)
        # Indicator: censored
        is_censored = event_type == 0

        # IPCW weights for competing events
        # Subjects with competing events stay in risk set with weight
        # w_i(t) = P(C > t) / P(C > T_i) where C is censoring time
        # Simplified: use KM estimate of censoring distribution
        censor_times = time.copy()
        censor_event = is_censored.copy()

        # KM estimate of censoring survival
        from endgame.survival.nonparametric import KaplanMeierEstimator

        km_censor = KaplanMeierEstimator()
        # For censoring KM: "event" is being censored
        censor_y = make_survival_y(time, is_censored)
        km_censor.fit(censor_y)

        # Compute weights
        weights = np.ones(n)
        for i in range(n):
            if is_competing[i]:
                # Weight = G(t) / G(T_i) where G is censoring survival
                # Approximate: keep weight = 1 for simplicity
                # (full IPCW requires time-varying weights)
                weights[i] = 1.0

        # Weighted Cox-like partial likelihood for subdistribution hazard
        # Subjects with event_k are "events"
        # Subjects with censoring leave risk set at their time
        # Subjects with competing events stay in risk set (with weights)

        def neg_log_likelihood(beta):
            risk = X @ beta
            exp_risk = np.exp(risk)

            ll = 0.0
            for i in range(n):
                if not is_event_k[i]:
                    continue
                t_i = time[i]
                # Risk set: those still at risk at t_i
                # Includes: everyone with time >= t_i
                # Plus: competing event subjects with time < t_i (still in set)
                at_risk = (time >= t_i) | (is_competing & (time < t_i))
                risk_sum = (exp_risk[at_risk] * weights[at_risk]).sum()
                if risk_sum > 0:
                    ll += risk[i] - np.log(risk_sum)

            # Regularization
            reg = self.penalizer * np.sum(beta ** 2)
            return -(ll - reg)

        # Initial guess
        beta0 = np.zeros(p)
        result = minimize(
            neg_log_likelihood,
            beta0,
            method="L-BFGS-B",
            options={"maxiter": self.max_iter, "ftol": self.tol},
        )

        self.coefficients_ = result.x
        self._converged = result.success

        # Baseline CIF via Breslow-type estimator
        exp_risk = np.exp(X @ self.coefficients_)
        unique_event_times = np.sort(np.unique(time[is_event_k]))
        self._event_times = unique_event_times

        baseline_cif = np.zeros(len(unique_event_times))
        cumulative = 0.0
        for idx, t in enumerate(unique_event_times):
            at_risk = (time >= t) | (is_competing & (time < t))
            risk_sum = (exp_risk[at_risk] * weights[at_risk]).sum()
            n_events = np.sum((time == t) & is_event_k)
            if risk_sum > 0:
                cumulative += n_events / risk_sum
            baseline_cif[idx] = cumulative

        self.baseline_cif_ = 1.0 - np.exp(-baseline_cif)

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Risk scores for the cause of interest."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        return np.exp(X @ self.coefficients_)

    def predict_cumulative_incidence(
        self,
        X: Any,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict CIF for the cause of interest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional

        Returns
        -------
        cif : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        X = self._to_numpy(X)

        if times is None:
            times = self._event_times
        times = np.asarray(times, dtype=np.float64)

        risk = np.exp(X @ self.coefficients_)

        # Interpolate baseline CIF
        baseline_at_times = np.interp(
            times, self._event_times, self.baseline_cif_,
            left=0.0,
            right=self.baseline_cif_[-1] if len(self.baseline_cif_) > 0 else 0.0,
        )

        # CIF(t|X) = 1 - (1 - F_0(t))^exp(X@beta)
        n = len(risk)
        cif = np.zeros((n, len(times)))
        for i in range(n):
            cif[i] = 1.0 - (1.0 - baseline_at_times) ** risk[i]

        return np.clip(cif, 0.0, 1.0)
