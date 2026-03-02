from __future__ import annotations

"""Statistical forecasting models via statsforecast.

This module wraps Nixtla's statsforecast library, providing sklearn-compatible
interfaces to fast statistical forecasting models.

statsforecast is 20x faster than pmdarima and provides optimized
implementations of ARIMA, ETS, Theta, and other classical methods.

Installation
------------
pip install statsforecast

Examples
--------
>>> from endgame.timeseries import AutoARIMAForecaster
>>> model = AutoARIMAForecaster(seasonal=True, m=12)
>>> model.fit(monthly_sales)
>>> forecast = model.predict(horizon=12)
"""

from typing import Any, Literal

import numpy as np

from endgame.timeseries.base import (
    BaseForecaster,
    UnivariateForecasterMixin,
    validate_forecast_input,
)

# Check for statsforecast availability
try:
    import statsforecast
    from statsforecast.models import (
        MSTL,
        AutoARIMA,
        AutoCES,
        AutoETS,
        AutoTheta,
        HistoricAverage,
        Naive,
        SeasonalNaive,
    )
    HAS_STATSFORECAST = True
except ImportError:
    HAS_STATSFORECAST = False


def _check_statsforecast():
    """Raise ImportError if statsforecast is not installed."""
    if not HAS_STATSFORECAST:
        raise ImportError(
            "statsforecast is required for statistical forecasting models. "
            "Install with: pip install endgame-ml[timeseries]"
        )


class StatsForecastWrapper(BaseForecaster, UnivariateForecasterMixin):
    """Base wrapper for statsforecast models.

    Provides common functionality for wrapping statsforecast models
    with sklearn-compatible interface.

    Parameters
    ----------
    model_class : class
        The statsforecast model class to wrap.
    model_kwargs : dict
        Keyword arguments to pass to the model constructor.
    seasonal_period : int, default=1
        Seasonal period (m in statsforecast).
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        model_class,
        model_kwargs: dict[str, Any] | None = None,
        seasonal_period: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_class = model_class
        self.model_kwargs = model_kwargs or {}
        self.seasonal_period = seasonal_period

        self._model = None
        self._fitted_model = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> StatsForecastWrapper:
        """Fit the statsforecast model.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : array-like, optional
            Exogenous features (not supported by all models).

        Returns
        -------
        self
            Fitted forecaster.
        """
        _check_statsforecast()

        y, X_arr = validate_forecast_input(y, X)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)
        if X_arr is not None:
            self.X_ = X_arr.copy()
            self.n_features_ = X_arr.shape[1]

        # Create model instance
        self._model = self.model_class(**self.model_kwargs)

        # Fit model
        self._fitted_model = self._model.fit(y, X=X_arr)

        self._log(f"Fitted {self.model_class.__name__}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : array-like, optional
            Future exogenous features.

        Returns
        -------
        np.ndarray
            Point forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        X_arr = None
        if X is not None:
            X_arr = self._to_numpy(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)

        # Generate predictions
        result = self._fitted_model.predict(h=horizon, X=X_arr)

        # Handle different return formats
        if isinstance(result, dict):
            forecast = result.get('mean', result.get('fitted', list(result.values())[0]))
        else:
            forecast = result

        return np.asarray(forecast).flatten()[:horizon]

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        coverage : float, default=0.95
            Coverage probability.
        X : array-like, optional
            Future exogenous features.

        Returns
        -------
        tuple
            (point_forecast, lower, upper)
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        X_arr = None
        if X is not None:
            X_arr = self._to_numpy(X)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)

        level = int(coverage * 100)

        try:
            result = self._fitted_model.predict(h=horizon, X=X_arr, level=[level])

            if isinstance(result, dict):
                point = np.asarray(result.get('mean', result.get('fitted', [])))
                lower = np.asarray(result.get(f'lo-{level}', result.get('lo', [])))
                upper = np.asarray(result.get(f'hi-{level}', result.get('hi', [])))
            else:
                # Fall back to point forecast only
                point = self.predict(horizon, X)
                lower = point.copy()
                upper = point.copy()

        except (TypeError, AttributeError):
            # Model doesn't support intervals
            point = self.predict(horizon, X)

            # Estimate intervals from residuals
            residuals = self.get_residuals()
            std = np.std(residuals[~np.isnan(residuals)])
            alpha = 1 - coverage
            z = 1.96 if coverage == 0.95 else -np.percentile(
                np.random.standard_normal(10000), alpha / 2 * 100
            )
            margin = z * std * np.sqrt(1 + np.arange(horizon) * 0.1)
            lower = point - margin
            upper = point + margin

        return point.flatten()[:horizon], lower.flatten()[:horizon], upper.flatten()[:horizon]

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        self._check_is_fitted()

        try:
            fitted = self._fitted_model.fitted_
            return np.asarray(fitted).flatten()
        except AttributeError:
            # Compute manually using one-step-ahead
            fitted = np.empty(self.n_samples_)
            fitted[0] = np.nan

            for t in range(1, self.n_samples_):
                temp_model = self.model_class(**self.model_kwargs)
                temp_fitted = temp_model.fit(self.y_[:t])
                pred = temp_fitted.predict(h=1)
                if isinstance(pred, dict):
                    fitted[t] = list(pred.values())[0][0]
                else:
                    fitted[t] = pred[0]

            return fitted


class AutoARIMAForecaster(StatsForecastWrapper):
    """AutoARIMA forecaster via statsforecast.

    Automatically selects the best ARIMA model using AIC/BIC.
    20x faster than pmdarima.

    Parameters
    ----------
    d : int, optional
        Order of differencing. If None, auto-selected.
    D : int, optional
        Order of seasonal differencing. If None, auto-selected.
    max_p : int, default=5
        Maximum p (AR order).
    max_q : int, default=5
        Maximum q (MA order).
    max_P : int, default=2
        Maximum seasonal P.
    max_Q : int, default=2
        Maximum seasonal Q.
    seasonal : bool, default=True
        Whether to fit seasonal ARIMA.
    m : int, default=1
        Seasonal period.
    stationary : bool, default=False
        If True, restricts search to stationary models.
    ic : str, default="aicc"
        Information criterion: "aicc", "aic", "bic".
    approximation : bool, default=True
        Use approximation for speed.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Monthly data with yearly seasonality
    >>> model = AutoARIMAForecaster(seasonal=True, m=12)
    >>> model.fit(monthly_sales)
    >>> forecast = model.predict(horizon=12)

    >>> # Non-seasonal ARIMA
    >>> model = AutoARIMAForecaster(seasonal=False)
    >>> model.fit(daily_returns)
    """

    def __init__(
        self,
        d: int | None = None,
        D: int | None = None,
        max_p: int = 5,
        max_q: int = 5,
        max_P: int = 2,
        max_Q: int = 2,
        seasonal: bool = True,
        m: int = 1,
        stationary: bool = False,
        ic: Literal["aicc", "aic", "bic"] = "aicc",
        approximation: bool = True,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_statsforecast()

        model_kwargs = {
            'd': d,
            'D': D,
            'max_p': max_p,
            'max_q': max_q,
            'max_P': max_P,
            'max_Q': max_Q,
            'seasonal': seasonal,
            'season_length': m,
            'stationary': stationary,
            'ic': ic,
            'approximation': approximation,
        }

        super().__init__(
            model_class=AutoARIMA,
            model_kwargs=model_kwargs,
            seasonal_period=m,
            random_state=random_state,
            verbose=verbose,
        )

        # Store parameters for sklearn get_params
        self.d = d
        self.D = D
        self.max_p = max_p
        self.max_q = max_q
        self.max_P = max_P
        self.max_Q = max_Q
        self.seasonal = seasonal
        self.m = m
        self.stationary = stationary
        self.ic = ic
        self.approximation = approximation


class AutoETSForecaster(StatsForecastWrapper):
    """AutoETS (Error, Trend, Seasonal) forecaster via statsforecast.

    Automatically selects the best exponential smoothing model.

    Parameters
    ----------
    model : str, default="ZZZ"
        ETS model specification (Error, Trend, Seasonal):
        - Error: A(dditive), M(ultiplicative), Z(auto)
        - Trend: N(one), A(dditive), Ad(damped additive), M(ultiplicative), Md(damped multiplicative), Z(auto)
        - Seasonal: N(one), A(dditive), M(ultiplicative), Z(auto)
        Examples: "AAN" (additive error, additive trend, no seasonal),
                  "ZZZ" (auto-select all)
    season_length : int, default=1
        Seasonal period.
    damped : bool, optional
        Whether to use damped trend. If None, auto-selected.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Auto-select ETS model
    >>> model = AutoETSForecaster(model="ZZZ", season_length=12)
    >>> model.fit(monthly_sales)
    >>> forecast = model.predict(horizon=12)

    >>> # Specific model: additive error, damped trend, multiplicative seasonal
    >>> model = AutoETSForecaster(model="AdM", season_length=12)
    """

    def __init__(
        self,
        model: str = "ZZZ",
        season_length: int = 1,
        damped: bool | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_statsforecast()

        model_kwargs = {
            'model': model,
            'season_length': season_length,
        }
        if damped is not None:
            model_kwargs['damped'] = damped

        super().__init__(
            model_class=AutoETS,
            model_kwargs=model_kwargs,
            seasonal_period=season_length,
            random_state=random_state,
            verbose=verbose,
        )

        self.model = model
        self.season_length = season_length
        self.damped = damped


class AutoThetaForecaster(StatsForecastWrapper):
    """AutoTheta forecaster via statsforecast.

    Implements optimized Theta method with automatic selection.

    Parameters
    ----------
    season_length : int, default=1
        Seasonal period.
    decomposition_type : str, default="multiplicative"
        Type of seasonal decomposition: "multiplicative" or "additive".
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> model = AutoThetaForecaster(season_length=12)
    >>> model.fit(monthly_data)
    >>> forecast = model.predict(horizon=6)
    """

    def __init__(
        self,
        season_length: int = 1,
        decomposition_type: Literal["multiplicative", "additive"] = "multiplicative",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_statsforecast()

        model_kwargs = {
            'season_length': season_length,
            'decomposition_type': decomposition_type,
        }

        super().__init__(
            model_class=AutoTheta,
            model_kwargs=model_kwargs,
            seasonal_period=season_length,
            random_state=random_state,
            verbose=verbose,
        )

        self.season_length = season_length
        self.decomposition_type = decomposition_type


class MSTLForecaster(BaseForecaster, UnivariateForecasterMixin):
    """MSTL (Multiple Seasonal-Trend decomposition using Loess) forecaster.

    Handles multiple seasonal patterns (e.g., daily + weekly + yearly).
    Decomposes series then forecasts trend and seasonal components separately.

    Parameters
    ----------
    season_lengths : List[int], default=[7]
        List of seasonal periods to model.
    trend_forecaster : str, default="auto_arima"
        Method for forecasting trend: "auto_arima", "auto_ets", "naive".
    stl_kwargs : dict, optional
        Additional arguments for STL decomposition.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Hourly data with daily and weekly seasonality
    >>> model = MSTLForecaster(season_lengths=[24, 168])
    >>> model.fit(hourly_demand)
    >>> forecast = model.predict(horizon=48)
    """

    def __init__(
        self,
        season_lengths: list[int] = [7],
        trend_forecaster: Literal["auto_arima", "auto_ets", "naive"] = "auto_arima",
        stl_kwargs: dict[str, Any] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.season_lengths = season_lengths
        self.trend_forecaster = trend_forecaster
        self.stl_kwargs = stl_kwargs or {}

        self._model = None
        self._fitted_model = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> MSTLForecaster:
        """Fit MSTL model."""
        _check_statsforecast()

        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        # Select trend forecaster
        if self.trend_forecaster == "auto_arima":
            trend_model = AutoARIMA()
        elif self.trend_forecaster == "auto_ets":
            trend_model = AutoETS()
        else:
            trend_model = Naive()

        # Create MSTL model
        self._model = MSTL(
            season_length=self.season_lengths,
            trend_forecaster=trend_model,
            **self.stl_kwargs,
        )

        # Fit
        self._fitted_model = self._model.fit(y)

        self._log(f"Fitted MSTL with seasons={self.season_lengths}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate MSTL forecasts."""
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        result = self._fitted_model.predict(h=horizon)

        if isinstance(result, dict):
            forecast = result.get('mean', list(result.values())[0])
        else:
            forecast = result

        return np.asarray(forecast).flatten()[:horizon]

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        self._check_is_fitted()
        try:
            return np.asarray(self._fitted_model.fitted_).flatten()
        except AttributeError:
            # Fall back to computing from components
            return np.full(self.n_samples_, np.nan)


class CESForecaster(StatsForecastWrapper):
    """Complex Exponential Smoothing forecaster via statsforecast.

    Uses complex-valued exponential smoothing for improved accuracy
    on some data patterns.

    Parameters
    ----------
    season_length : int, default=1
        Seasonal period.
    model : str, default="Z"
        Model type: "N" (none), "S" (simple), "P" (partial), "F" (full), "Z" (auto).
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        season_length: int = 1,
        model: Literal["N", "S", "P", "F", "Z"] = "Z",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_statsforecast()

        model_kwargs = {
            'season_length': season_length,
            'model': model,
        }

        super().__init__(
            model_class=AutoCES,
            model_kwargs=model_kwargs,
            seasonal_period=season_length,
            random_state=random_state,
            verbose=verbose,
        )

        self.season_length = season_length
        self.model = model


# Convenience function for quick forecasting
def quick_forecast(
    y: Any,
    horizon: int,
    model: str = "auto_arima",
    seasonal_period: int = 1,
    return_intervals: bool = False,
    coverage: float = 0.95,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quick forecasting with sensible defaults.

    Parameters
    ----------
    y : array-like
        Time series data.
    horizon : int
        Forecast horizon.
    model : str, default="auto_arima"
        Model to use: "auto_arima", "auto_ets", "theta", "mstl", "naive".
    seasonal_period : int, default=1
        Seasonal period.
    return_intervals : bool, default=False
        Whether to return prediction intervals.
    coverage : float, default=0.95
        Interval coverage probability.

    Returns
    -------
    np.ndarray or tuple
        Point forecasts, or (point, lower, upper) if return_intervals=True.

    Examples
    --------
    >>> forecast = quick_forecast(sales_data, horizon=30, model="auto_arima")
    >>> point, lower, upper = quick_forecast(
    ...     sales_data, horizon=30, return_intervals=True
    ... )
    """
    _check_statsforecast()

    models = {
        "auto_arima": AutoARIMAForecaster(m=seasonal_period),
        "auto_ets": AutoETSForecaster(season_length=seasonal_period),
        "theta": AutoThetaForecaster(season_length=seasonal_period),
        "mstl": MSTLForecaster(season_lengths=[seasonal_period]),
        "ces": CESForecaster(season_length=seasonal_period),
    }

    if model not in models:
        raise ValueError(f"Unknown model: {model}. Choose from {list(models.keys())}")

    forecaster = models[model]
    forecaster.fit(y)

    if return_intervals:
        return forecaster.predict_interval(horizon, coverage=coverage)
    else:
        return forecaster.predict(horizon)
