from __future__ import annotations

"""Base classes for time series forecasting.

This module provides the foundational abstractions for all forecasters
in the timeseries module, designed with sklearn compatibility and
future signal processing integration in mind.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class ForecasterMixin:
    """Mixin providing common forecaster functionality.

    Defines the interface that all forecasters must implement.
    """

    _estimator_type = "forecaster"

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate point forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : array-like, optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        np.ndarray
            Point forecasts of shape (horizon,) or (horizon, n_series).
        """
        pass

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        coverage : float, default=0.95
            Coverage probability for the interval.
        X : array-like, optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        tuple of np.ndarray
            (point_forecast, lower_bound, upper_bound)
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support prediction intervals"
        )

    def predict_quantiles(
        self,
        horizon: int,
        quantiles: list[float] = [0.1, 0.5, 0.9],
        X: np.ndarray | None = None,
    ) -> np.ndarray:
        """Generate quantile forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        quantiles : List[float], default=[0.1, 0.5, 0.9]
            Quantile levels to predict.
        X : array-like, optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        np.ndarray
            Quantile forecasts of shape (horizon, n_quantiles).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support quantile forecasting"
        )


class UnivariateForecasterMixin(ForecasterMixin):
    """Mixin for univariate time series forecasters.

    Forecasters using this mixin expect a single time series as input.
    """

    _supports_multivariate = False

    def _validate_univariate(self, y: np.ndarray) -> np.ndarray:
        """Validate that input is univariate."""
        y = np.asarray(y).squeeze()
        if y.ndim != 1:
            raise ValueError(
                f"{self.__class__.__name__} only supports univariate time series. "
                f"Got shape {y.shape}"
            )
        return y


class MultivariateForecasterMixin(ForecasterMixin):
    """Mixin for multivariate time series forecasters.

    Forecasters using this mixin can handle multiple time series
    or a single series with multiple variables.
    """

    _supports_multivariate = True

    def _validate_multivariate(
        self,
        y: np.ndarray,
        allow_univariate: bool = True,
    ) -> np.ndarray:
        """Validate and reshape input for multivariate forecasting."""
        y = np.asarray(y)
        if y.ndim == 1:
            if not allow_univariate:
                raise ValueError(
                    f"{self.__class__.__name__} requires multivariate input"
                )
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError(
                f"Expected 1D or 2D array, got shape {y.shape}"
            )
        return y


class BaseForecaster(BaseEstimator, RegressorMixin, ABC):
    """Base class for all time series forecasters.

    Provides sklearn-compatible interface with time series specific
    extensions. Designed for integration with signal processing.

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the forecaster has been fitted.
    y_ : np.ndarray
        The training time series (stored for forecasting).
    n_samples_ : int
        Number of samples in the training series.
    freq_ : str, optional
        Inferred or specified frequency of the time series.

    Notes
    -----
    All forecasters follow the pattern:
    1. fit(y, X=None) - Learn from historical data
    2. predict(horizon) - Generate future predictions
    3. update(y_new) - Incrementally update with new data (optional)

    For sklearn pipeline compatibility, forecasters also support:
    - fit_predict(y, horizon) - Fit and predict in one call
    - score(y_true, y_pred) - Evaluate forecast accuracy

    Design Considerations for Signal Processing Integration:
    - All methods accept raw arrays (no timestamp requirements)
    - Frequency/sampling rate stored in freq_ attribute
    - Support for irregularly sampled data via interpolation
    - Hooks for spectral features (FFT, wavelets) via features parameter
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.random_state = random_state
        self.verbose = verbose

        # Fitted state
        self.is_fitted_ = False
        self.y_: np.ndarray | None = None
        self.n_samples_: int | None = None
        self.freq_: str | None = None

        # For exogenous variables
        self.X_: np.ndarray | None = None
        self.n_features_: int | None = None

    def _log(self, message: str) -> None:
        """Log message if verbose mode is enabled."""
        if self.verbose:
            print(f"[{self.__class__.__name__}] {message}")

    def _to_numpy(self, data: Any) -> np.ndarray:
        """Convert input to numpy array.

        Handles pandas Series/DataFrame and polars Series/DataFrame.
        """
        if data is None:
            return None

        if isinstance(data, np.ndarray):
            return data

        if HAS_PANDAS:
            if isinstance(data, pd.Series):
                return data.values
            if isinstance(data, pd.DataFrame):
                return data.values

        if HAS_POLARS:
            if isinstance(data, pl.Series):
                return data.to_numpy()
            if isinstance(data, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(data, pl.LazyFrame):
                    data = data.collect()
                return data.to_numpy()

        return np.asarray(data)

    def _check_is_fitted(self) -> None:
        """Raise error if not fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this forecaster."
            )

    def _validate_horizon(self, horizon: int) -> int:
        """Validate forecast horizon."""
        if not isinstance(horizon, (int, np.integer)):
            raise TypeError(f"horizon must be int, got {type(horizon)}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")
        return int(horizon)

    def _infer_frequency(self, timestamps: np.ndarray | None = None) -> str | None:
        """Infer time series frequency from timestamps or data.

        This is a placeholder for future signal processing integration
        where sampling rate inference will be important.
        """
        # TODO: Implement frequency inference from timestamps
        # For now, return None (unknown frequency)
        return None

    @abstractmethod
    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> BaseForecaster:
        """Fit the forecaster to training data.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_series)
            Training time series.
        X : array-like of shape (n_samples, n_features), optional
            Exogenous features aligned with y.
        **fit_params : dict
            Additional parameters for fitting.

        Returns
        -------
        self
            Fitted forecaster.
        """
        pass

    @abstractmethod
    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate point forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : array-like of shape (horizon, n_features), optional
            Exogenous features for the forecast horizon.

        Returns
        -------
        np.ndarray
            Point forecasts of shape (horizon,) or (horizon, n_series).
        """
        pass

    def fit_predict(
        self,
        y: Any,
        horizon: int,
        X: Any | None = None,
        X_future: Any | None = None,
        **fit_params,
    ) -> np.ndarray:
        """Fit and predict in one step.

        Parameters
        ----------
        y : array-like
            Training time series.
        horizon : int
            Forecast horizon.
        X : array-like, optional
            Historical exogenous features.
        X_future : array-like, optional
            Future exogenous features for prediction.
        **fit_params : dict
            Additional fitting parameters.

        Returns
        -------
        np.ndarray
            Forecasts.
        """
        self.fit(y, X=X, **fit_params)
        return self.predict(horizon, X=X_future)

    def update(
        self,
        y_new: Any,
        X_new: Any | None = None,
    ) -> BaseForecaster:
        """Update the forecaster with new observations.

        Default implementation re-fits with concatenated data.
        Subclasses may override for incremental updates.

        Parameters
        ----------
        y_new : array-like
            New observations to incorporate.
        X_new : array-like, optional
            Corresponding exogenous features.

        Returns
        -------
        self
            Updated forecaster.
        """
        self._check_is_fitted()

        y_new = self._to_numpy(y_new)
        y_combined = np.concatenate([self.y_, y_new])

        if X_new is not None and self.X_ is not None:
            X_new = self._to_numpy(X_new)
            X_combined = np.concatenate([self.X_, X_new])
        else:
            X_combined = None

        return self.fit(y_combined, X=X_combined)

    def score(
        self,
        y_true: Any,
        y_pred: Any | None = None,
        horizon: int | None = None,
        metric: str = "mse",
    ) -> float:
        """Score the forecaster's predictions.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like, optional
            Predicted values. If None, generates predictions.
        horizon : int, optional
            Forecast horizon (required if y_pred is None).
        metric : str, default="mse"
            Scoring metric: "mse", "rmse", "mae", "mape", "smape".

        Returns
        -------
        float
            Negative score (for sklearn compatibility, lower is better).
        """
        y_true = self._to_numpy(y_true)

        if y_pred is None:
            if horizon is None:
                horizon = len(y_true)
            self._check_is_fitted()
            y_pred = self.predict(horizon)
        else:
            y_pred = self._to_numpy(y_pred)

        # Align lengths
        min_len = min(len(y_true), len(y_pred))
        y_true = y_true[:min_len]
        y_pred = y_pred[:min_len]

        # Compute metric
        if metric == "mse":
            return -np.mean((y_true - y_pred) ** 2)
        elif metric == "rmse":
            return -np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == "mae":
            return -np.mean(np.abs(y_true - y_pred))
        elif metric == "mape":
            # Avoid division by zero
            mask = y_true != 0
            if not np.any(mask):
                return -np.inf
            return -np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        elif metric == "smape":
            denom = np.abs(y_true) + np.abs(y_pred)
            mask = denom != 0
            if not np.any(mask):
                return -np.inf
            return -np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values (one-step-ahead predictions).

        Returns
        -------
        np.ndarray
            Fitted values for the training period.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support fitted values"
        )

    def get_residuals(self) -> np.ndarray:
        """Get in-sample residuals.

        Returns
        -------
        np.ndarray
            Residuals (y - fitted_values).
        """
        self._check_is_fitted()
        fitted = self.get_fitted_values()
        return self.y_[:len(fitted)] - fitted

    def __repr__(self) -> str:
        """String representation."""
        params = self.get_params()
        param_str = ", ".join(f"{k}={v!r}" for k, v in params.items())
        return f"{self.__class__.__name__}({param_str})"


class BaseMultiSeriesForecaster(BaseForecaster, MultivariateForecasterMixin):
    """Base class for forecasters handling multiple time series.

    Supports both:
    - Panel data: Multiple independent series (item_id dimension)
    - Multivariate: Single series with multiple variables

    Parameters
    ----------
    global_model : bool, default=True
        If True, fit a single model across all series (transfer learning).
        If False, fit separate models per series.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        global_model: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.global_model = global_model

        self.n_series_: int | None = None
        self.series_names_: list[str] | None = None

    def _validate_panel_data(
        self,
        y: np.ndarray,
        series_ids: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and structure panel data.

        Parameters
        ----------
        y : array-like of shape (n_samples,) or (n_samples, n_series)
            Time series data.
        series_ids : array-like, optional
            Series identifiers for panel data format.

        Returns
        -------
        tuple
            (y_structured, series_ids) where y_structured is (n_samples, n_series)
        """
        y = self._to_numpy(y)

        if y.ndim == 1:
            y = y.reshape(-1, 1)
            series_ids = np.zeros(len(y), dtype=int) if series_ids is None else series_ids

        return y, series_ids


# Type aliases for clarity
TimeSeriesData = Union[np.ndarray, "pd.Series", "pd.DataFrame", "pl.Series", "pl.DataFrame"]
ExogenousData = Optional[Union[np.ndarray, "pd.DataFrame", "pl.DataFrame"]]


def validate_forecast_input(
    y: Any,
    X: Any | None = None,
    allow_missing: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate and convert forecast inputs.

    Parameters
    ----------
    y : array-like
        Time series data.
    X : array-like, optional
        Exogenous features.
    allow_missing : bool, default=False
        Whether to allow NaN values.

    Returns
    -------
    tuple
        (y_validated, X_validated)

    Raises
    ------
    ValueError
        If validation fails.
    """
    # Convert to numpy
    if HAS_PANDAS and isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.values
    elif HAS_POLARS and isinstance(y, (pl.Series, pl.DataFrame, pl.LazyFrame)):
        if isinstance(y, pl.LazyFrame):
            y = y.collect()
        y = y.to_numpy()
    else:
        y = np.asarray(y)

    # Ensure float type
    y = y.astype(np.float64)

    # Check for missing values
    if not allow_missing and np.any(np.isnan(y)):
        raise ValueError(
            "Input contains NaN values. Set allow_missing=True to allow."
        )

    # Handle exogenous features
    X_out = None
    if X is not None:
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            X_out = X.values
        elif HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            X_out = X.to_numpy()
        else:
            X_out = np.asarray(X)

        X_out = X_out.astype(np.float64)

        if X_out.ndim == 1:
            X_out = X_out.reshape(-1, 1)

        if len(X_out) != len(y):
            raise ValueError(
                f"X has {len(X_out)} samples but y has {len(y)} samples"
            )

    return y, X_out
