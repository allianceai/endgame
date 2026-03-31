"""Non-parametric survival estimators (pure NumPy, zero external dependencies).

Provides the Kaplan-Meier and Nelson-Aalen estimators for univariate
survival analysis.  These estimators do **not** use covariates — they
estimate a single survival or cumulative-hazard curve from time-to-event
data.

Example
-------
>>> import numpy as np
>>> from endgame.survival.base import make_survival_y
>>> from endgame.survival.nonparametric import KaplanMeierEstimator
>>> y = make_survival_y([1, 2, 3, 4, 5], [True, False, True, True, False])
>>> km = KaplanMeierEstimator().fit(y)
>>> km.median_survival_time_
3.0
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import chi2 as chi2_dist
from sklearn.base import BaseEstimator

from endgame.survival.base import (
    _check_survival_y,
    _get_time_event,
)


# ---------------------------------------------------------------------------
# Kaplan-Meier
# ---------------------------------------------------------------------------


class KaplanMeierEstimator(BaseEstimator):
    """Kaplan-Meier (product-limit) estimator of the survival function.

    The estimator computes:

        S(t) = prod_{t_i <= t} (1 - d_i / n_i)

    where *d_i* is the number of events at time *t_i* and *n_i* is the
    number of subjects at risk just before *t_i*.

    Parameters
    ----------
    confidence_level : float, default=0.95
        Confidence level for the pointwise confidence intervals computed
        via Greenwood's formula.

    Attributes
    ----------
    timeline_ : ndarray of shape (n_unique_event_times,)
        Sorted unique times at which events occurred.
    survival_function_ : ndarray of shape (n_unique_event_times,)
        Estimated S(t) at each time in ``timeline_``.
    confidence_intervals_ : ndarray of shape (n_unique_event_times, 2)
        Lower and upper confidence bounds for S(t).
    median_survival_time_ : float
        Smallest time *t* such that S(t) <= 0.5.  ``np.inf`` if S never
        drops to 0.5.
    n_events_ : int
        Total number of observed events.
    n_censored_ : int
        Total number of censored observations.

    Example
    -------
    >>> import numpy as np
    >>> from endgame.survival.base import make_survival_y
    >>> km = KaplanMeierEstimator()
    >>> y = make_survival_y([1, 2, 3, 4, 5], [True, True, True, False, True])
    >>> km.fit(y)
    KaplanMeierEstimator()
    >>> km.survival_function_
    array([0.8 , 0.6 , 0.4 , 0.4 , 0.2 ])
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def fit(self, y: Any) -> "KaplanMeierEstimator":
        """Fit the Kaplan-Meier estimator.

        Parameters
        ----------
        y : structured array with 'event' and 'time' fields, tuple, or
            DataFrame.  See :func:`~endgame.survival.base._check_survival_y`.

        Returns
        -------
        self
        """
        y = _check_survival_y(y)
        time, event = _get_time_event(y)

        self.n_events_ = int(event.sum())
        self.n_censored_ = int((~event).sum())

        # Unique event times (sorted)
        unique_times = np.sort(np.unique(time[event]))
        self.timeline_ = unique_times

        n_subjects = len(time)
        survival = np.ones(len(unique_times), dtype=np.float64)
        greenwood_sum = np.zeros(len(unique_times), dtype=np.float64)

        for i, t in enumerate(unique_times):
            n_i = np.sum(time >= t)  # at-risk just before t
            d_i = np.sum((time == t) & event)  # events at t

            if n_i > 0:
                ratio = 1.0 - d_i / n_i
            else:
                ratio = 1.0

            if i == 0:
                survival[i] = ratio
            else:
                survival[i] = survival[i - 1] * ratio

            # Greenwood increment: d_i / (n_i * (n_i - d_i))
            if n_i > 0 and (n_i - d_i) > 0:
                greenwood_inc = d_i / (n_i * (n_i - d_i))
            else:
                greenwood_inc = 0.0

            if i == 0:
                greenwood_sum[i] = greenwood_inc
            else:
                greenwood_sum[i] = greenwood_sum[i - 1] + greenwood_inc

        self.survival_function_ = survival

        # Confidence intervals via Greenwood's formula
        z = _z_value(self.confidence_level)
        variance = survival ** 2 * greenwood_sum
        se = np.sqrt(variance)
        lower = np.clip(survival - z * se, 0.0, 1.0)
        upper = np.clip(survival + z * se, 0.0, 1.0)
        self.confidence_intervals_ = np.column_stack([lower, upper])

        # Median survival time
        idx = np.where(survival <= 0.5)[0]
        if len(idx) > 0:
            self.median_survival_time_ = float(unique_times[idx[0]])
        else:
            self.median_survival_time_ = np.inf

        return self

    def predict_survival_function(
        self, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate the survival function at arbitrary times.

        Uses a right-continuous step function (carry-forward interpolation):
        S(t) = S(t_k) where t_k is the largest event time <= t.

        Parameters
        ----------
        times : array-like of shape (n_times,), optional
            Query time points.  If *None*, returns S at ``timeline_``.

        Returns
        -------
        survival : ndarray of shape (n_times,)
        """
        self._check_is_fitted()
        if times is None:
            return self.survival_function_.copy()

        times = np.asarray(times, dtype=np.float64)
        return _step_function(self.timeline_, self.survival_function_, times)

    def summary(self) -> dict:
        """Return a summary dictionary.

        Returns
        -------
        dict
            Keys: ``timeline``, ``survival``, ``ci_lower``, ``ci_upper``,
            ``n_events``, ``n_censored``, ``median_survival_time``.
        """
        self._check_is_fitted()
        return {
            "timeline": self.timeline_.copy(),
            "survival": self.survival_function_.copy(),
            "ci_lower": self.confidence_intervals_[:, 0].copy(),
            "ci_upper": self.confidence_intervals_[:, 1].copy(),
            "n_events": self.n_events_,
            "n_censored": self.n_censored_,
            "median_survival_time": self.median_survival_time_,
        }

    # -- static helpers -----------------------------------------------------

    @staticmethod
    def log_rank_test(y1: Any, y2: Any) -> dict:
        """Two-sample log-rank test.

        Tests the null hypothesis that two groups have identical survival
        functions.

        Parameters
        ----------
        y1, y2 : survival targets (structured array, tuple, or DataFrame)

        Returns
        -------
        dict
            ``chi2`` : float — test statistic,
            ``p_value`` : float — p-value from chi-squared(1) distribution.

        Example
        -------
        >>> import numpy as np
        >>> from endgame.survival.base import make_survival_y
        >>> y1 = make_survival_y([1, 2, 3], [True, True, True])
        >>> y2 = make_survival_y([4, 5, 6], [True, True, True])
        >>> result = KaplanMeierEstimator.log_rank_test(y1, y2)
        >>> result['chi2'] > 0
        True
        """
        y1 = _check_survival_y(y1)
        y2 = _check_survival_y(y2)

        time1, event1 = _get_time_event(y1)
        time2, event2 = _get_time_event(y2)

        # Pool all unique event times
        all_times = np.sort(
            np.unique(
                np.concatenate([time1[event1], time2[event2]])
            )
        )

        observed_1 = 0.0
        expected_1 = 0.0
        variance = 0.0

        for t in all_times:
            # At-risk counts
            n1 = np.sum(time1 >= t)
            n2 = np.sum(time2 >= t)
            n = n1 + n2

            # Event counts
            d1 = np.sum((time1 == t) & event1)
            d2 = np.sum((time2 == t) & event2)
            d = d1 + d2

            if n == 0:
                continue

            e1 = n1 * d / n  # expected events in group 1
            observed_1 += d1
            expected_1 += e1

            if n > 1:
                variance += (n1 * n2 * d * (n - d)) / (n ** 2 * (n - 1))

        if variance == 0:
            return {"chi2": 0.0, "p_value": 1.0}

        chi2_stat = (observed_1 - expected_1) ** 2 / variance
        p_value = float(1.0 - chi2_dist.cdf(chi2_stat, df=1))
        return {"chi2": float(chi2_stat), "p_value": p_value}

    # -- private ------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "survival_function_"):
            raise RuntimeError(
                "KaplanMeierEstimator has not been fitted. Call 'fit' first."
            )


# ---------------------------------------------------------------------------
# Nelson-Aalen
# ---------------------------------------------------------------------------


class NelsonAalenEstimator(BaseEstimator):
    """Nelson-Aalen estimator of the cumulative hazard function.

    Computes:

        H(t) = sum_{t_i <= t} d_i / n_i

    with variance estimated as:

        Var(H(t)) = sum_{t_i <= t} d_i / n_i^2

    Parameters
    ----------
    confidence_level : float, default=0.95
        Confidence level for pointwise confidence intervals.

    Attributes
    ----------
    timeline_ : ndarray of shape (n_unique_event_times,)
        Sorted unique event times.
    cumulative_hazard_ : ndarray of shape (n_unique_event_times,)
        Estimated H(t) at each time in ``timeline_``.
    confidence_intervals_ : ndarray of shape (n_unique_event_times, 2)
        Lower and upper confidence bounds for H(t).
    n_events_ : int
        Total number of observed events.
    n_censored_ : int
        Total number of censored observations.
    """

    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level

    def fit(self, y: Any) -> "NelsonAalenEstimator":
        """Fit the Nelson-Aalen estimator.

        Parameters
        ----------
        y : structured survival array, tuple, or DataFrame.

        Returns
        -------
        self
        """
        y = _check_survival_y(y)
        time, event = _get_time_event(y)

        self.n_events_ = int(event.sum())
        self.n_censored_ = int((~event).sum())

        unique_times = np.sort(np.unique(time[event]))
        self.timeline_ = unique_times

        cum_hazard = np.zeros(len(unique_times), dtype=np.float64)
        cum_var = np.zeros(len(unique_times), dtype=np.float64)

        for i, t in enumerate(unique_times):
            n_i = np.sum(time >= t)
            d_i = np.sum((time == t) & event)

            inc = d_i / n_i if n_i > 0 else 0.0
            var_inc = d_i / (n_i ** 2) if n_i > 0 else 0.0

            if i == 0:
                cum_hazard[i] = inc
                cum_var[i] = var_inc
            else:
                cum_hazard[i] = cum_hazard[i - 1] + inc
                cum_var[i] = cum_var[i - 1] + var_inc

        self.cumulative_hazard_ = cum_hazard

        # Confidence intervals
        z = _z_value(self.confidence_level)
        se = np.sqrt(cum_var)
        lower = np.clip(cum_hazard - z * se, 0.0, None)
        upper = cum_hazard + z * se
        self.confidence_intervals_ = np.column_stack([lower, upper])

        return self

    def predict_cumulative_hazard(
        self, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Evaluate the cumulative hazard at arbitrary times.

        Parameters
        ----------
        times : array-like of shape (n_times,), optional
            Query time points.  If *None*, returns H at ``timeline_``.

        Returns
        -------
        cumulative_hazard : ndarray of shape (n_times,)
        """
        self._check_is_fitted()
        if times is None:
            return self.cumulative_hazard_.copy()

        times = np.asarray(times, dtype=np.float64)
        return _step_function(
            self.timeline_, self.cumulative_hazard_, times, fill_before=0.0
        )

    def summary(self) -> dict:
        """Return a summary dictionary.

        Returns
        -------
        dict
            Keys: ``timeline``, ``cumulative_hazard``, ``ci_lower``,
            ``ci_upper``, ``n_events``, ``n_censored``.
        """
        self._check_is_fitted()
        return {
            "timeline": self.timeline_.copy(),
            "cumulative_hazard": self.cumulative_hazard_.copy(),
            "ci_lower": self.confidence_intervals_[:, 0].copy(),
            "ci_upper": self.confidence_intervals_[:, 1].copy(),
            "n_events": self.n_events_,
            "n_censored": self.n_censored_,
        }

    # -- private ------------------------------------------------------------

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "cumulative_hazard_"):
            raise RuntimeError(
                "NelsonAalenEstimator has not been fitted. Call 'fit' first."
            )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _z_value(confidence_level: float) -> float:
    """Return the z critical value for a two-sided confidence interval."""
    from scipy.stats import norm

    alpha = 1.0 - confidence_level
    return float(norm.ppf(1.0 - alpha / 2.0))


def _step_function(
    knots: np.ndarray,
    values: np.ndarray,
    query: np.ndarray,
    fill_before: float | None = None,
) -> np.ndarray:
    """Evaluate a right-continuous step function via carry-forward.

    For query times before the first knot, returns 1.0 (survival) or
    ``fill_before`` if specified.

    Parameters
    ----------
    knots : sorted array of step locations
    values : function values at knots
    query : times at which to evaluate
    fill_before : value to use for times before the first knot.
        If *None*, defaults to 1.0 (appropriate for survival functions).

    Returns
    -------
    result : ndarray same shape as *query*
    """
    if fill_before is None:
        fill_before = 1.0

    idx = np.searchsorted(knots, query, side="right") - 1
    result = np.where(idx < 0, fill_before, values[np.clip(idx, 0, len(values) - 1)])
    return result
