"""Base classes for survival analysis.

This module provides the foundational abstractions for all survival estimators
in the survival module, designed with sklearn compatibility.

The standard input format for survival data is a structured NumPy array with
fields ``('event', bool)`` and ``('time', float)``, following scikit-survival's
convention. Use :func:`make_survival_y` to create this from separate arrays.

Example
-------
>>> from endgame.survival.base import make_survival_y
>>> import numpy as np
>>> time = np.array([5.0, 3.0, 8.0, 1.0])
>>> event = np.array([True, False, True, True])
>>> y = make_survival_y(time, event)
>>> y.dtype.names
('event', 'time')
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl

    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


# ---------------------------------------------------------------------------
# Structured array helpers
# ---------------------------------------------------------------------------

SURVIVAL_DTYPE = np.dtype([("event", bool), ("time", np.float64)])


def make_survival_y(
    time: np.ndarray,
    event: np.ndarray,
) -> np.ndarray:
    """Create a structured survival array from time and event arrays.

    Parameters
    ----------
    time : array-like of shape (n_samples,)
        Observed times (duration until event or censoring).
    event : array-like of shape (n_samples,)
        Event indicator (True/1 = event observed, False/0 = censored).

    Returns
    -------
    y : ndarray of shape (n_samples,)
        Structured array with fields ``('event', bool)`` and ``('time', float64)``.

    Example
    -------
    >>> y = make_survival_y([5, 3, 8], [True, False, True])
    >>> y['time']
    array([5., 3., 8.])
    >>> y['event']
    array([ True, False,  True])
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    if time.shape != event.shape:
        raise ValueError(
            f"time and event must have the same shape, "
            f"got {time.shape} and {event.shape}"
        )
    y = np.empty(len(time), dtype=SURVIVAL_DTYPE)
    y["event"] = event
    y["time"] = time
    return y


def _check_survival_y(y: Any) -> np.ndarray:
    """Validate and convert survival target to structured array.

    Accepts:
    - Structured ndarray with 'event' and 'time' fields
    - Tuple of (time, event) arrays
    - pandas DataFrame with 'time' and 'event' columns

    Returns
    -------
    y : ndarray with SURVIVAL_DTYPE
    """
    if isinstance(y, np.ndarray) and y.dtype.names is not None:
        if "event" in y.dtype.names and "time" in y.dtype.names:
            # Already structured
            out = np.empty(len(y), dtype=SURVIVAL_DTYPE)
            out["event"] = y["event"].astype(bool)
            out["time"] = y["time"].astype(np.float64)
            return out

    if isinstance(y, tuple) and len(y) == 2:
        return make_survival_y(y[0], y[1])

    if HAS_PANDAS and isinstance(y, pd.DataFrame):
        if "time" in y.columns and "event" in y.columns:
            return make_survival_y(y["time"].values, y["event"].values)

    if HAS_POLARS and isinstance(y, (pl.DataFrame, pl.LazyFrame)):
        if isinstance(y, pl.LazyFrame):
            y = y.collect()
        if "time" in y.columns and "event" in y.columns:
            return make_survival_y(
                y["time"].to_numpy(), y["event"].to_numpy()
            )

    raise ValueError(
        "y must be a structured array with 'event' and 'time' fields, "
        "a (time, event) tuple, or a DataFrame with 'time' and 'event' columns. "
        f"Got {type(y)}"
    )


def _get_time_event(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract time and event arrays from structured survival array."""
    return y["time"].astype(np.float64), y["event"].astype(bool)


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurvivalPrediction:
    """Container for survival model predictions.

    Attributes
    ----------
    risk_scores : ndarray of shape (n_samples,)
        Risk scores (higher = more risk). Used for concordance index.
    survival_function : ndarray of shape (n_samples, n_times), optional
        S(t|X) survival probability at each time point.
    cumulative_hazard : ndarray of shape (n_samples, n_times), optional
        H(t|X) cumulative hazard at each time point.
    times : ndarray of shape (n_times,), optional
        Time points corresponding to columns of survival_function / cumulative_hazard.
    median_survival_time : ndarray of shape (n_samples,), optional
        Predicted median survival time for each sample.
    """

    risk_scores: np.ndarray
    survival_function: np.ndarray | None = None
    cumulative_hazard: np.ndarray | None = None
    times: np.ndarray | None = None
    median_survival_time: np.ndarray | None = None


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class SurvivalMixin:
    """Mixin providing common survival estimator functionality.

    Defines the interface that all survival estimators must implement.
    """

    _estimator_type = "survival"

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores.

        Higher values indicate higher risk (shorter expected survival).
        For concordance index compatibility.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
        """

    def predict_survival_function(
        self,
        X: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict survival function S(t|X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        times : array-like of shape (n_times,), optional
            Time points at which to evaluate S(t).
            If None, uses the unique event times from training.

        Returns
        -------
        survival_function : ndarray of shape (n_samples, n_times)
            Survival probabilities. Each row is a survival curve.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "predict_survival_function"
        )

    def predict_cumulative_hazard(
        self,
        X: np.ndarray,
        times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict cumulative hazard function H(t|X).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        times : array-like of shape (n_times,), optional
            Time points at which to evaluate H(t).

        Returns
        -------
        cumulative_hazard : ndarray of shape (n_samples, n_times)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "predict_cumulative_hazard"
        )

    def predict_median_survival_time(self, X: np.ndarray) -> np.ndarray:
        """Predict median survival time for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        median_time : ndarray of shape (n_samples,)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement "
            "predict_median_survival_time"
        )

    def predict_full(
        self,
        X: np.ndarray,
        times: np.ndarray | None = None,
    ) -> SurvivalPrediction:
        """Return all available predictions in a single container.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like of shape (n_times,), optional

        Returns
        -------
        SurvivalPrediction
        """
        risk = self.predict(X)
        sf = ch = med = None
        try:
            sf = self.predict_survival_function(X, times)
        except NotImplementedError:
            pass
        try:
            ch = self.predict_cumulative_hazard(X, times)
        except NotImplementedError:
            pass
        try:
            med = self.predict_median_survival_time(X)
        except NotImplementedError:
            pass
        t = times
        if t is None and hasattr(self, "event_times_"):
            t = self.event_times_
        return SurvivalPrediction(
            risk_scores=risk,
            survival_function=sf,
            cumulative_hazard=ch,
            times=t,
            median_survival_time=med,
        )


class BaseSurvivalEstimator(BaseEstimator, SurvivalMixin, ABC):
    """Base class for all survival estimators in endgame.

    Provides common utilities for input validation, data conversion,
    and fitted state tracking.

    Parameters
    ----------
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _check_is_fitted(self) -> None:
        """Raise if estimator has not been fitted."""
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )

    def _validate_survival_data(
        self,
        X: Any,
        y: Any,
        reset: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and convert inputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or None
            If None (for non-parametric estimators), returns None.
        y : structured array, tuple, or DataFrame
            Survival target.
        reset : bool
            Whether to store metadata (during fit).

        Returns
        -------
        X : ndarray or None
        y : structured ndarray with SURVIVAL_DTYPE
        """
        y = _check_survival_y(y)

        if X is not None:
            X = self._to_numpy(X)
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            if len(X) != len(y):
                raise ValueError(
                    f"X and y must have the same number of samples, "
                    f"got {len(X)} and {len(y)}"
                )
            if reset:
                self.n_features_in_ = X.shape[1]

        if reset:
            time, event = _get_time_event(y)
            self.event_times_ = np.sort(np.unique(time[event]))
            self.n_events_ = int(event.sum())
            self.n_censored_ = int((~event).sum())

        return X, y

    @staticmethod
    def _to_numpy(X: Any) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, np.ndarray):
            return np.asarray(X, dtype=np.float64)

        if HAS_PANDAS and isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values.astype(np.float64)

        if HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            return X.to_numpy().astype(np.float64)

        return np.asarray(X, dtype=np.float64)

    def score(self, X: np.ndarray, y: Any) -> float:
        """Return concordance index on (X, y).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        c_index : float
        """
        from endgame.survival.metrics import concordance_index

        y = _check_survival_y(y)
        risk = self.predict(X)
        return concordance_index(y, risk)
