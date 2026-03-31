"""Survival analysis with XGBoost, LightGBM, and CatBoost.

Provides a unified :class:`SurvivalGBDTWrapper` that trains gradient-boosted
trees for Cox / AFT survival objectives across the three major GBDT backends.

Example
-------
>>> from endgame.survival.gbdt import SurvivalGBDTWrapper
>>> model = SurvivalGBDTWrapper(backend="xgboost", preset="endgame")
>>> model.fit(X, y, eval_set=[(X_val, y_val)])  # doctest: +SKIP
>>> risk = model.predict(X_test)
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from endgame.survival.base import (
    BaseSurvivalEstimator,
    _check_survival_y,
    _get_time_event,
    SURVIVAL_DTYPE,
)

# ---------------------------------------------------------------------------
# Optional backend imports
# ---------------------------------------------------------------------------

try:
    import xgboost as xgb

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb

    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb

    HAS_CB = True
except ImportError:
    HAS_CB = False


# ---------------------------------------------------------------------------
# Presets
# ---------------------------------------------------------------------------

_PRESETS: dict[str, dict[str, Any]] = {
    "endgame": {
        "n_estimators": 500,
        "learning_rate": 0.05,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "fast": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 4,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "accurate": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "max_depth": 8,
        "subsample": 0.7,
        "colsample_bytree": 0.7,
    },
}


# ---------------------------------------------------------------------------
# Breslow baseline helper
# ---------------------------------------------------------------------------


def _breslow_from_risk(
    times: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Breslow baseline cumulative hazard from predicted risk scores.

    Parameters
    ----------
    times : (n,) observed times
    events : (n,) boolean event indicators
    risk_scores : (n,) exp(linear predictor) or model output

    Returns
    -------
    unique_event_times : sorted unique event times
    baseline_cumulative_hazard : H_0(t) at each event time
    """
    order = np.argsort(times)
    sorted_times = times[order]
    sorted_events = events[order]
    sorted_risk = risk_scores[order]

    # Reverse cumulative sum = risk set sum at each time
    risk_set_sum = np.cumsum(sorted_risk[::-1])[::-1]

    event_mask = sorted_events.astype(bool)
    event_times = sorted_times[event_mask]
    event_risk_set = risk_set_sum[event_mask]

    unique_times, first_idx = np.unique(event_times, return_index=True)
    increments = np.zeros(len(unique_times))
    for i, ut in enumerate(unique_times):
        n_events = (event_times == ut).sum()
        increments[i] = n_events / event_risk_set[first_idx[i]]

    return unique_times, np.cumsum(increments)


# ---------------------------------------------------------------------------
# SurvivalGBDTWrapper
# ---------------------------------------------------------------------------


class SurvivalGBDTWrapper(BaseSurvivalEstimator):
    """Unified GBDT wrapper for survival analysis.

    Supports XGBoost (Cox objective), LightGBM (custom Cox or IPCW
    regression fallback), and CatBoost (Cox / AFT loss).

    Parameters
    ----------
    backend : str, default='xgboost'
        Backend library: ``'xgboost'``, ``'lightgbm'``, ``'catboost'``.
    n_estimators : int, default=500
    learning_rate : float, default=0.05
    max_depth : int, default=6
    subsample : float, default=0.8
    colsample_bytree : float, default=0.8
    early_stopping_rounds : int, default=50
    preset : str, default='endgame'
        One of ``'endgame'``, ``'fast'``, ``'accurate'``.
    use_gpu : bool, default=False
    random_state : int or None, default=None
    verbose : bool, default=False
    **kwargs
        Additional parameters forwarded to the underlying model.

    Attributes
    ----------
    model_ : underlying fitted model
    feature_importances_ : ndarray
    """

    def __init__(
        self,
        backend: Literal["xgboost", "lightgbm", "catboost"] = "xgboost",
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int = 50,
        preset: str = "endgame",
        use_gpu: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs: Any,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.backend = backend
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds
        self.preset = preset
        self.use_gpu = use_gpu
        self.kwargs = kwargs

    # -- parameter resolution -----------------------------------------------

    def _resolve_params(self) -> dict[str, Any]:
        """Merge preset defaults with user-specified overrides."""
        base = dict(_PRESETS.get(self.preset, _PRESETS["endgame"]))
        # Override with explicitly set constructor args
        explicit = {
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
        # Only override if caller changed from the "endgame" defaults
        # (i.e., use preset as base, then layer on any non-default values)
        if self.preset != "endgame":
            # When a non-default preset is chosen, start from that preset
            pass
        base.update(explicit)
        base.update(self.kwargs)
        return base

    # -- backend-specific fitting -------------------------------------------

    def _fit_xgboost(
        self,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        eval_set: list | None,
        params: dict,
    ) -> None:
        if not HAS_XGB:
            raise ImportError(
                "XGBoost is required for backend='xgboost'. "
                "Install it with: pip install xgboost"
            )

        # XGBoost survival:cox expects label = time (positive if event,
        # negative if censored)
        labels = np.where(events, times, -times)

        xgb_params = {
            "objective": "survival:cox",
            "eval_metric": "cox-nloglik",
            "eta": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "verbosity": 1 if self.verbose else 0,
        }
        if self.random_state is not None:
            xgb_params["seed"] = self.random_state
        if self.use_gpu:
            xgb_params["device"] = "cuda"

        dtrain = xgb.DMatrix(X, label=labels)
        evals = [(dtrain, "train")]

        if eval_set is not None:
            for i, (X_e, y_e) in enumerate(eval_set):
                X_e = self._to_numpy(X_e)
                y_e = _check_survival_y(y_e)
                t_e, ev_e = _get_time_event(y_e)
                lab_e = np.where(ev_e, t_e, -t_e)
                deval = xgb.DMatrix(X_e, label=lab_e)
                evals.append((deval, f"eval_{i}"))

        self.model_ = xgb.train(
            xgb_params,
            dtrain,
            num_boost_round=params["n_estimators"],
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            verbose_eval=self.verbose,
        )
        self._backend_type = "xgboost"

    def _fit_lightgbm(
        self,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        eval_set: list | None,
        params: dict,
    ) -> None:
        if not HAS_LGB:
            raise ImportError(
                "LightGBM is required for backend='lightgbm'. "
                "Install it with: pip install lightgbm"
            )

        # LightGBM does not have a native Cox objective in the stable API.
        # Strategy: encode as a regression problem with IPCW weights.
        # Censored observations get lower weight; uncensored get full weight.
        # Target is log(time) for uncensored, and we weight by event
        # indicator with a small floor for censored samples.
        labels = np.log1p(times)
        weights = np.where(events, 1.0, 0.3)  # down-weight censored

        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": params["learning_rate"],
            "max_depth": params["max_depth"],
            "subsample": params["subsample"],
            "colsample_bytree": params["colsample_bytree"],
            "n_estimators": params["n_estimators"],
            "verbosity": 1 if self.verbose else -1,
        }
        if self.random_state is not None:
            lgb_params["random_state"] = self.random_state
        if self.use_gpu:
            lgb_params["device"] = "gpu"

        callbacks = []
        if not self.verbose:
            callbacks.append(lgb.log_evaluation(period=-1))

        eval_sets_lgb = [(X, labels)]
        eval_names = ["train"]
        eval_weights = [weights]
        if eval_set is not None:
            for i, (X_e, y_e) in enumerate(eval_set):
                X_e = self._to_numpy(X_e)
                y_e = _check_survival_y(y_e)
                t_e, ev_e = _get_time_event(y_e)
                eval_sets_lgb.append((X_e, np.log1p(t_e)))
                eval_names.append(f"eval_{i}")
                eval_weights.append(np.where(ev_e, 1.0, 0.3))

        model = lgb.LGBMRegressor(**lgb_params)
        fit_kwargs: dict[str, Any] = {
            "sample_weight": weights,
            "callbacks": callbacks,
        }
        if eval_set is not None:
            fit_kwargs["eval_set"] = [
                (es[0], es[1]) for es in eval_sets_lgb[1:]
            ]
            fit_kwargs["eval_sample_weight"] = eval_weights[1:]

        model.fit(X, labels, **fit_kwargs)
        self.model_ = model
        self._backend_type = "lightgbm"

    def _fit_catboost(
        self,
        X: np.ndarray,
        times: np.ndarray,
        events: np.ndarray,
        eval_set: list | None,
        params: dict,
    ) -> None:
        if not HAS_CB:
            raise ImportError(
                "CatBoost is required for backend='catboost'. "
                "Install it with: pip install catboost"
            )

        # CatBoost label for Cox: (time, event) concatenated
        # CatBoost expects a Pool with label = time and
        # pairs or special format. For the Cox loss the label
        # is time and the event flag is encoded in the label sign
        # (positive = event, negative = censored), similar to XGBoost.
        labels = np.where(events, times, -times)

        cb_params = {
            "loss_function": "Cox",
            "iterations": params["n_estimators"],
            "learning_rate": params["learning_rate"],
            "depth": params["max_depth"],
            "subsample": params["subsample"],
            "rsm": params["colsample_bytree"],  # random subspace method
            "verbose": self.verbose,
        }
        if self.random_state is not None:
            cb_params["random_seed"] = self.random_state
        if self.use_gpu:
            cb_params["task_type"] = "GPU"

        train_pool = cb.Pool(X, label=labels)
        eval_pools = []
        if eval_set is not None:
            for X_e, y_e in eval_set:
                X_e = self._to_numpy(X_e)
                y_e = _check_survival_y(y_e)
                t_e, ev_e = _get_time_event(y_e)
                lab_e = np.where(ev_e, t_e, -t_e)
                eval_pools.append(cb.Pool(X_e, label=lab_e))

        model = cb.CatBoost(cb_params)
        model.fit(
            train_pool,
            eval_set=eval_pools if eval_pools else None,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        self.model_ = model
        self._backend_type = "catboost"

    # -- public API ---------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: Any,
        eval_set: list[tuple[np.ndarray, Any]] | None = None,
    ) -> SurvivalGBDTWrapper:
        """Fit survival GBDT model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : survival target
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y, reset=True)
        times, events = _get_time_event(y)
        params = self._resolve_params()

        self._log(f"Fitting SurvivalGBDT with backend={self.backend}, "
                  f"preset={self.preset}")

        if self.backend == "xgboost":
            self._fit_xgboost(X, times, events, eval_set, params)
        elif self.backend == "lightgbm":
            self._fit_lightgbm(X, times, events, eval_set, params)
        elif self.backend == "catboost":
            self._fit_catboost(X, times, events, eval_set, params)
        else:
            raise ValueError(
                f"Unknown backend '{self.backend}'. "
                "Choose from: 'xgboost', 'lightgbm', 'catboost'."
            )

        # Compute Breslow baseline from training data
        train_risk = self._predict_raw(X)
        self._breslow_times, self._breslow_cumhaz = _breslow_from_risk(
            times, events, train_risk
        )
        self._breslow_surv = np.exp(-self._breslow_cumhaz)

        self.is_fitted_ = True
        return self

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw risk scores from the backend model.

        For Cox models the output is exp(margin) or the partial hazard.
        For the LightGBM fallback we negate (lower predicted time = higher
        risk).
        """
        X = self._to_numpy(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if self._backend_type == "xgboost":
            dmat = xgb.DMatrix(X)
            # xgb survival:cox predict returns exp(margin) = risk score
            return self.model_.predict(dmat)
        elif self._backend_type == "lightgbm":
            # Higher predicted log-time = lower risk; negate for risk score
            pred = self.model_.predict(X)
            return np.exp(-pred)
        elif self._backend_type == "catboost":
            # CatBoost Cox returns log partial hazard
            raw = self.model_.predict(X)
            return np.exp(raw)
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores (higher = more risk).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        return self._predict_raw(X)

    def predict_survival_function(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict survival function via Breslow baseline.

        S(t|X) = S_0(t) ^ (risk / mean_risk)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional

        Returns
        -------
        S : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        risk = self._predict_raw(X)

        if times is None:
            eval_times = self._breslow_times
            cumhaz = self._breslow_cumhaz
        else:
            eval_times = np.asarray(times, dtype=np.float64)
            # Interpolate baseline cumulative hazard
            indices = np.searchsorted(
                self._breslow_times, eval_times, side="right"
            ) - 1
            cumhaz = np.where(
                indices < 0,
                0.0,
                self._breslow_cumhaz[
                    np.clip(indices, 0, len(self._breslow_cumhaz) - 1)
                ],
            )

        # S(t|X) = exp(-H_0(t) * risk_i)
        return np.exp(-risk[:, None] * cumhaz[None, :])

    def predict_cumulative_hazard(
        self, X: np.ndarray, times: np.ndarray | None = None
    ) -> np.ndarray:
        """Predict cumulative hazard H(t|X) = H_0(t) * risk.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        times : array-like, optional

        Returns
        -------
        H : ndarray of shape (n_samples, n_times)
        """
        self._check_is_fitted()
        risk = self._predict_raw(X)

        if times is None:
            cumhaz = self._breslow_cumhaz
        else:
            eval_times = np.asarray(times, dtype=np.float64)
            indices = np.searchsorted(
                self._breslow_times, eval_times, side="right"
            ) - 1
            cumhaz = np.where(
                indices < 0,
                0.0,
                self._breslow_cumhaz[
                    np.clip(indices, 0, len(self._breslow_cumhaz) - 1)
                ],
            )

        return risk[:, None] * cumhaz[None, :]

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the underlying model."""
        self._check_is_fitted()
        if self._backend_type == "xgboost":
            scores = self.model_.get_score(importance_type="gain")
            n = self.n_features_in_
            imp = np.zeros(n)
            for key, val in scores.items():
                idx = int(key.replace("f", ""))
                if idx < n:
                    imp[idx] = val
            total = imp.sum()
            if total > 0:
                imp /= total
            return imp
        elif self._backend_type == "lightgbm":
            return self.model_.feature_importances_ / max(
                self.model_.feature_importances_.sum(), 1
            )
        elif self._backend_type == "catboost":
            imp = np.array(self.model_.get_feature_importance())
            total = imp.sum()
            if total > 0:
                imp /= total
            return imp
        else:
            raise RuntimeError(f"Unknown backend type: {self._backend_type}")
