"""Survival tree ensembles wrapping scikit-survival.

Provides Random Survival Forest, Extra Survival Trees, and Gradient-Boosted
Survival Analysis behind the standard :class:`BaseSurvivalEstimator` API.

All wrappers require ``scikit-survival`` at runtime; a clear
``ImportError`` is raised if it is missing.

Example
-------
>>> from endgame.survival.trees import RandomSurvivalForestRegressor
>>> model = RandomSurvivalForestRegressor(n_estimators=50)
>>> model.fit(X, y)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np

from endgame.survival.base import (
    BaseSurvivalEstimator,
    _check_survival_y,
    _get_time_event,
    SURVIVAL_DTYPE,
)

try:
    from sksurv.ensemble import RandomSurvivalForest as _RSF
    from sksurv.ensemble import ExtraSurvivalTrees as _EST
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis as _GBSA

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_sksurv_y(y: np.ndarray) -> np.ndarray:
    """Convert endgame survival array to scikit-survival structured array.

    scikit-survival expects dtype ``[('event', bool), ('time', float)]``
    which matches our ``SURVIVAL_DTYPE``, but we rebuild to be safe.
    """
    times, events = _get_time_event(y)
    out = np.array(
        [(bool(e), float(t)) for e, t in zip(events, times)],
        dtype=[("event", bool), ("time", np.float64)],
    )
    return out


def _check_sksurv() -> None:
    """Raise an informative error when scikit-survival is missing."""
    if not HAS_SKSURV:
        raise ImportError(
            "scikit-survival is required for tree-based survival models. "
            "Install it with: pip install scikit-survival"
        )


def _extract_survival_functions(
    step_functions: list,
    times: np.ndarray | None,
    default_times: np.ndarray,
) -> np.ndarray:
    """Evaluate sksurv StepFunction objects on a common time grid.

    Parameters
    ----------
    step_functions : list of sksurv StepFunction
    times : requested evaluation times or None
    default_times : fallback event times from training

    Returns
    -------
    values : ndarray of shape (n_samples, n_times)
    """
    if times is None:
        times = default_times
    else:
        times = np.asarray(times, dtype=np.float64)

    n = len(step_functions)
    result = np.empty((n, len(times)))
    for i, fn in enumerate(step_functions):
        # sksurv StepFunction is callable
        result[i] = fn(times)
    return result


# ---------------------------------------------------------------------------
# RandomSurvivalForestRegressor
# ---------------------------------------------------------------------------


class RandomSurvivalForestRegressor(BaseSurvivalEstimator):
    """Random Survival Forest.

    Wraps ``sksurv.ensemble.RandomSurvivalForest``.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None, default=None
    min_samples_split : int, default=6
    min_samples_leaf : int, default=3
    max_features : str or int or float, default='sqrt'
    n_jobs : int, default=-1
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | int | float = "sqrt",
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: Any) -> RandomSurvivalForestRegressor:
        """Fit Random Survival Forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        _check_sksurv()
        X, y = self._validate_survival_data(X, y, reset=True)
        sksurv_y = _to_sksurv_y(y)

        self._model = _RSF(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._log(f"Fitting RSF with {self.n_estimators} trees")
        self._model.fit(X, sksurv_y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = more risk)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function S(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        sf = self._model.predict_survival_function(X)
        return _extract_survival_functions(sf, times, self.event_times_)

    def predict_cumulative_hazard(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict cumulative hazard H(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        chf = self._model.predict_cumulative_hazard_function(X)
        return _extract_survival_functions(chf, times, self.event_times_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the underlying RSF."""
        self._check_is_fitted()
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# ExtraSurvivalTreesRegressor
# ---------------------------------------------------------------------------


class ExtraSurvivalTreesRegressor(BaseSurvivalEstimator):
    """Extra Survival Trees.

    Wraps ``sksurv.ensemble.ExtraSurvivalTrees``.

    Parameters
    ----------
    n_estimators : int, default=100
    max_depth : int or None, default=None
    min_samples_split : int, default=6
    min_samples_leaf : int, default=3
    max_features : str or int or float, default='sqrt'
    n_jobs : int, default=-1
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 6,
        min_samples_leaf: int = 3,
        max_features: str | int | float = "sqrt",
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.n_jobs = n_jobs

    def fit(self, X: np.ndarray, y: Any) -> ExtraSurvivalTreesRegressor:
        """Fit Extra Survival Trees.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        _check_sksurv()
        X, y = self._validate_survival_data(X, y, reset=True)
        sksurv_y = _to_sksurv_y(y)

        self._model = _EST(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )
        self._log(f"Fitting ExtraSurvivalTrees with {self.n_estimators} trees")
        self._model.fit(X, sksurv_y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = more risk)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function S(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        sf = self._model.predict_survival_function(X)
        return _extract_survival_functions(sf, times, self.event_times_)

    def predict_cumulative_hazard(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict cumulative hazard H(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        chf = self._model.predict_cumulative_hazard_function(X)
        return _extract_survival_functions(chf, times, self.event_times_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the underlying model."""
        self._check_is_fitted()
        return self._model.feature_importances_


# ---------------------------------------------------------------------------
# GradientBoostedSurvivalRegressor
# ---------------------------------------------------------------------------


class GradientBoostedSurvivalRegressor(BaseSurvivalEstimator):
    """Gradient Boosted Survival Analysis.

    Wraps ``sksurv.ensemble.GradientBoostingSurvivalAnalysis``.

    Parameters
    ----------
    n_estimators : int, default=100
    learning_rate : float, default=0.1
    max_depth : int, default=3
    loss : str, default='coxph'
        Loss function: ``'coxph'``, ``'squared'``, ``'ipcwls'``.
    subsample : float, default=1.0
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        loss: str = "coxph",
        subsample: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.loss = loss
        self.subsample = subsample

    def fit(self, X: np.ndarray, y: Any) -> GradientBoostedSurvivalRegressor:
        """Fit gradient-boosted survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target

        Returns
        -------
        self
        """
        _check_sksurv()
        X, y = self._validate_survival_data(X, y, reset=True)
        sksurv_y = _to_sksurv_y(y)

        self._model = _GBSA(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            loss=self.loss,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._log(f"Fitting GBSA with {self.n_estimators} iterations, "
                  f"loss={self.loss}")
        self._model.fit(X, sksurv_y)
        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = more risk)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return self._model.predict(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function S(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        sf = self._model.predict_survival_function(X)
        return _extract_survival_functions(sf, times, self.event_times_)

    def predict_cumulative_hazard(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict cumulative hazard H(t|X)."""
        self._check_is_fitted()
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        chf = self._model.predict_cumulative_hazard_function(X)
        return _extract_survival_functions(chf, times, self.event_times_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the underlying model."""
        self._check_is_fitted()
        return self._model.feature_importances_
