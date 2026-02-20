"""Simple forecasting baselines.

These baseline models are essential for:
1. Establishing benchmark performance
2. Sanity checking more complex models
3. Fast prototyping and feature engineering validation
4. Kaggle competitions where simple models often outperform complex ones

All baselines are implemented from scratch with no external dependencies
beyond numpy, making them lightweight and fast.
"""

from typing import Any, Literal

import numpy as np

from endgame.timeseries.base import (
    BaseForecaster,
    UnivariateForecasterMixin,
    validate_forecast_input,
)


class NaiveForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Naive forecaster using simple strategies.

    Implements several naive forecasting strategies that serve as
    strong baselines, especially for random walk-like data.

    Parameters
    ----------
    strategy : str, default="last"
        Forecasting strategy:
        - "last": Predict the last observed value (random walk)
        - "mean": Predict the mean of the training data
        - "median": Predict the median of the training data
    random_state : int, optional
        Random seed (unused, for API consistency).
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> model = NaiveForecaster(strategy="last")
    >>> model.fit(y_train)
    >>> forecast = model.predict(horizon=7)

    Notes
    -----
    The "last" strategy is equivalent to a random walk model without drift
    and often performs surprisingly well on financial and economic data.
    """

    def __init__(
        self,
        strategy: Literal["last", "mean", "median"] = "last",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.strategy = strategy

        self._forecast_value: float | None = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "NaiveForecaster":
        """Fit the naive forecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.
        X : ignored
            Exogenous features (not used).

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        # Compute forecast value based on strategy
        if self.strategy == "last":
            self._forecast_value = y[-1]
        elif self.strategy == "mean":
            self._forecast_value = np.mean(y)
        elif self.strategy == "median":
            self._forecast_value = np.median(y)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        self._log(f"Fitted with {self.strategy} strategy, value={self._forecast_value:.4f}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate naive forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Exogenous features (not used).

        Returns
        -------
        np.ndarray of shape (horizon,)
            Constant forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        return np.full(horizon, self._forecast_value)

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals using residual-based approach.

        Parameters
        ----------
        horizon : int
            Forecast horizon.
        coverage : float, default=0.95
            Coverage probability.
        X : ignored
            Not used.

        Returns
        -------
        tuple
            (point_forecast, lower, upper)
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        point = self.predict(horizon)

        # Estimate variance from in-sample residuals
        if self.strategy == "last":
            # For random walk, residuals are first differences
            residuals = np.diff(self.y_)
        else:
            residuals = self.y_ - self._forecast_value

        std = np.std(residuals)

        # Prediction intervals widen with horizon for random walk
        alpha = 1 - coverage
        z = -np.percentile(np.random.standard_normal(10000), alpha / 2 * 100)

        if self.strategy == "last":
            # Variance grows linearly with horizon for random walk
            horizon_factors = np.sqrt(np.arange(1, horizon + 1))
        else:
            horizon_factors = np.ones(horizon)

        margin = z * std * horizon_factors
        lower = point - margin
        upper = point + margin

        return point, lower, upper

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values.

        For naive forecaster, fitted values are lagged actuals (strategy=last)
        or constant values (strategy=mean/median).
        """
        self._check_is_fitted()

        if self.strategy == "last":
            # One-step-ahead forecasts are lagged values
            fitted = np.empty(self.n_samples_)
            fitted[0] = np.nan
            fitted[1:] = self.y_[:-1]
        else:
            fitted = np.full(self.n_samples_, self._forecast_value)

        return fitted


class SeasonalNaiveForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Seasonal naive forecaster.

    Predicts using values from the same season in the previous cycle.
    Essential baseline for data with strong seasonality.

    Parameters
    ----------
    seasonal_period : int, default=1
        Length of seasonal cycle (e.g., 7 for weekly, 12 for monthly, 24 for hourly).
        If None, attempts to auto-detect from data.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Weekly seasonality
    >>> model = SeasonalNaiveForecaster(seasonal_period=7)
    >>> model.fit(daily_sales)
    >>> forecast = model.predict(horizon=14)  # Next two weeks
    """

    def __init__(
        self,
        seasonal_period: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.seasonal_period = seasonal_period

        self._last_season: np.ndarray | None = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "SeasonalNaiveForecaster":
        """Fit the seasonal naive forecaster.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Training time series.
        X : ignored
            Not used.

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        sp = self.seasonal_period

        if sp < 1:
            raise ValueError(f"seasonal_period must be >= 1, got {sp}")

        if len(y) < sp:
            raise ValueError(
                f"Time series length ({len(y)}) must be >= seasonal_period ({sp})"
            )

        # Store the last complete season
        self._last_season = y[-sp:]

        self._log(f"Fitted with seasonal_period={sp}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate seasonal naive forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Not used.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Seasonal forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        sp = self.seasonal_period
        n_cycles = (horizon + sp - 1) // sp

        # Tile the last season and trim to horizon
        forecast = np.tile(self._last_season, n_cycles)[:horizon]

        return forecast

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals."""
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        point = self.predict(horizon)
        sp = self.seasonal_period

        # Compute seasonal residuals
        n_complete = self.n_samples_ - sp
        residuals = self.y_[sp:] - self.y_[:-sp]
        std = np.std(residuals)

        alpha = 1 - coverage
        z = -np.percentile(np.random.standard_normal(10000), alpha / 2 * 100)

        # Variance grows with number of complete cycles
        cycle_idx = np.arange(horizon) // sp + 1
        margin = z * std * np.sqrt(cycle_idx)

        return point, point - margin, point + margin

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values (lag-sp values)."""
        self._check_is_fitted()

        sp = self.seasonal_period
        fitted = np.empty(self.n_samples_)
        fitted[:sp] = np.nan
        fitted[sp:] = self.y_[:-sp]

        return fitted


class MovingAverageForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Moving average forecaster.

    Predicts using simple or weighted moving average of recent observations.
    Effective for smoothing noise and capturing local trends.

    Parameters
    ----------
    window : int, default=5
        Number of recent observations to average.
    weights : array-like, optional
        Custom weights for each lag. If None, uses equal weights.
        Weights are applied in reverse order (most recent first).
    center : bool, default=False
        If True, use centered moving average (for decomposition).
        Only affects fitted values, not predictions.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Simple moving average
    >>> model = MovingAverageForecaster(window=7)
    >>> model.fit(y_train)
    >>> forecast = model.predict(horizon=30)

    >>> # Weighted moving average (recent values weighted more)
    >>> model = MovingAverageForecaster(window=5, weights=[1, 2, 3, 4, 5])
    """

    def __init__(
        self,
        window: int = 5,
        weights: list[float] | None = None,
        center: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.window = window
        self.weights = weights
        self.center = center

        self._ma_value: float | None = None
        self._normalized_weights: np.ndarray | None = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "MovingAverageForecaster":
        """Fit the moving average forecaster.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : ignored
            Not used.

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        if self.window < 1:
            raise ValueError(f"window must be >= 1, got {self.window}")

        if len(y) < self.window:
            raise ValueError(
                f"Time series length ({len(y)}) must be >= window ({self.window})"
            )

        # Normalize weights
        if self.weights is not None:
            weights = np.asarray(self.weights, dtype=float)
            if len(weights) != self.window:
                raise ValueError(
                    f"weights length ({len(weights)}) must equal window ({self.window})"
                )
            self._normalized_weights = weights / weights.sum()
        else:
            self._normalized_weights = np.ones(self.window) / self.window

        # Compute moving average of last window observations
        self._ma_value = np.dot(self._normalized_weights, y[-self.window:])

        self._log(f"Fitted MA({self.window}), value={self._ma_value:.4f}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate moving average forecasts.

        Note: For multi-step forecasts, this uses a recursive approach
        where each forecast becomes input for the next.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Not used.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        # Use recursive forecasting
        forecast = np.empty(horizon)
        history = np.concatenate([self.y_[-self.window:], np.zeros(horizon)])

        for h in range(horizon):
            window_data = history[h:h + self.window]
            forecast[h] = np.dot(self._normalized_weights, window_data)
            history[self.window + h] = forecast[h]

        return forecast

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals."""
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        point = self.predict(horizon)

        # Compute residuals from in-sample MA
        fitted = self.get_fitted_values()
        valid_mask = ~np.isnan(fitted)
        residuals = self.y_[valid_mask] - fitted[valid_mask]
        std = np.std(residuals)

        alpha = 1 - coverage
        z = -np.percentile(np.random.standard_normal(10000), alpha / 2 * 100)

        # Intervals widen with horizon (approximate)
        horizon_factors = np.sqrt(1 + np.arange(horizon) * 0.1)
        margin = z * std * horizon_factors

        return point, point - margin, point + margin

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        self._check_is_fitted()

        fitted = np.empty(self.n_samples_)
        fitted[:self.window - 1] = np.nan

        for i in range(self.window - 1, self.n_samples_):
            window_data = self.y_[i - self.window + 1:i + 1]
            fitted[i] = np.dot(self._normalized_weights, window_data)

        return fitted


class ExponentialSmoothingForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Simple exponential smoothing forecaster.

    Implements single exponential smoothing (SES) for level forecasting
    and double exponential smoothing (Holt's method) for trend.

    Parameters
    ----------
    alpha : float, default=0.3
        Smoothing parameter for level (0 < alpha < 1).
        Higher values give more weight to recent observations.
    beta : float, optional
        Smoothing parameter for trend (0 < beta < 1).
        If None, uses simple exponential smoothing without trend.
    optimize : bool, default=False
        If True, optimize alpha (and beta) using grid search.
    initial_method : str, default="heuristic"
        Method for initializing level (and trend):
        - "heuristic": Use first few observations
        - "estimated": Estimate from training data
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> # Simple exponential smoothing
    >>> model = ExponentialSmoothingForecaster(alpha=0.3)
    >>> model.fit(y_train)
    >>> forecast = model.predict(horizon=10)

    >>> # Holt's linear trend method
    >>> model = ExponentialSmoothingForecaster(alpha=0.3, beta=0.1)

    >>> # Optimize parameters
    >>> model = ExponentialSmoothingForecaster(optimize=True)
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float | None = None,
        optimize: bool = False,
        initial_method: Literal["heuristic", "estimated"] = "heuristic",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.alpha = alpha
        self.beta = beta
        self.optimize = optimize
        self.initial_method = initial_method

        # Fitted parameters
        self.alpha_: float | None = None
        self.beta_: float | None = None
        self._level: float | None = None
        self._trend: float | None = None
        self._fitted_levels: np.ndarray | None = None
        self._fitted_trends: np.ndarray | None = None

    def _initialize_level_trend(self, y: np.ndarray) -> tuple[float, float | None]:
        """Initialize level and trend components."""
        if self.initial_method == "heuristic":
            level = y[0]
            trend = (y[min(len(y) - 1, 9)] - y[0]) / min(len(y) - 1, 9) if self.beta is not None else None
        else:  # estimated
            level = np.mean(y[:min(len(y), 10)])
            if self.beta is not None:
                # Simple trend estimate
                n = min(len(y), 10)
                x = np.arange(n)
                trend = np.polyfit(x, y[:n], 1)[0]
            else:
                trend = None

        return level, trend

    def _smooth(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float | None,
    ) -> tuple[np.ndarray, np.ndarray | None, float, float | None]:
        """Apply exponential smoothing."""
        n = len(y)
        levels = np.empty(n)
        trends = np.empty(n) if beta is not None else None

        level, trend = self._initialize_level_trend(y)

        for t in range(n):
            prev_level = level

            if beta is not None:
                # Holt's method
                level = alpha * y[t] + (1 - alpha) * (level + trend)
                trend = beta * (level - prev_level) + (1 - beta) * trend
                trends[t] = trend
            else:
                # Simple exponential smoothing
                level = alpha * y[t] + (1 - alpha) * level

            levels[t] = level

        return levels, trends, level, trend

    def _compute_sse(
        self,
        y: np.ndarray,
        alpha: float,
        beta: float | None,
    ) -> float:
        """Compute sum of squared errors for parameter optimization."""
        levels, trends, _, _ = self._smooth(y, alpha, beta)

        # One-step-ahead forecasts
        fitted = np.empty(len(y))
        fitted[0] = y[0]  # First fitted value

        if beta is not None:
            for t in range(1, len(y)):
                fitted[t] = levels[t - 1] + trends[t - 1]
        else:
            fitted[1:] = levels[:-1]

        sse = np.sum((y - fitted) ** 2)
        return sse

    def _optimize_parameters(self, y: np.ndarray) -> tuple[float, float | None]:
        """Optimize smoothing parameters via grid search."""
        best_alpha = self.alpha
        best_beta = self.beta
        best_sse = float('inf')

        alphas = np.linspace(0.01, 0.99, 20)

        if self.beta is not None:
            betas = np.linspace(0.01, 0.99, 20)
            for a in alphas:
                for b in betas:
                    sse = self._compute_sse(y, a, b)
                    if sse < best_sse:
                        best_sse = sse
                        best_alpha = a
                        best_beta = b
        else:
            for a in alphas:
                sse = self._compute_sse(y, a, None)
                if sse < best_sse:
                    best_sse = sse
                    best_alpha = a

        return best_alpha, best_beta

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "ExponentialSmoothingForecaster":
        """Fit the exponential smoothing forecaster.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : ignored
            Not used.

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        # Validate parameters
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")
        if self.beta is not None and not 0 < self.beta < 1:
            raise ValueError(f"beta must be in (0, 1), got {self.beta}")

        # Optimize if requested
        if self.optimize:
            self.alpha_, self.beta_ = self._optimize_parameters(y)
            self._log(f"Optimized: alpha={self.alpha_:.4f}, beta={self.beta_}")
        else:
            self.alpha_ = self.alpha
            self.beta_ = self.beta

        # Fit model
        self._fitted_levels, self._fitted_trends, self._level, self._trend = \
            self._smooth(y, self.alpha_, self.beta_)

        self._log(f"Fitted SES(alpha={self.alpha_:.2f}), level={self._level:.4f}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate exponential smoothing forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Not used.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        if self._trend is not None:
            # Holt's method: forecast = level + h * trend
            h = np.arange(1, horizon + 1)
            forecast = self._level + h * self._trend
        else:
            # Simple ES: flat forecast
            forecast = np.full(horizon, self._level)

        return forecast

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals."""
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        point = self.predict(horizon)

        # Compute residuals
        fitted = self.get_fitted_values()
        valid_mask = ~np.isnan(fitted)
        residuals = self.y_[valid_mask] - fitted[valid_mask]
        std = np.std(residuals)

        alpha = 1 - coverage
        z = -np.percentile(np.random.standard_normal(10000), alpha / 2 * 100)

        # Variance formula for SES
        a = self.alpha_
        h = np.arange(1, horizon + 1)

        if self._trend is None:
            # SES variance factor
            var_factor = 1 + (h - 1) * a ** 2
        else:
            # Holt's variance (approximate)
            b = self.beta_
            var_factor = 1 + (h - 1) * (a ** 2 + a * b * h + b ** 2 * h * (2 * h - 1) / 6)

        margin = z * std * np.sqrt(var_factor)

        return point, point - margin, point + margin

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values (one-step-ahead forecasts)."""
        self._check_is_fitted()

        fitted = np.empty(self.n_samples_)
        fitted[0] = np.nan

        if self._fitted_trends is not None:
            fitted[1:] = self._fitted_levels[:-1] + self._fitted_trends[:-1]
        else:
            fitted[1:] = self._fitted_levels[:-1]

        return fitted


class DriftForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Drift (random walk with drift) forecaster.

    Predicts using the last value plus average historical change.
    Equivalent to fitting a line between first and last observations.

    Parameters
    ----------
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> model = DriftForecaster()
    >>> model.fit(y_train)
    >>> forecast = model.predict(horizon=10)

    Notes
    -----
    The drift method is equivalent to:
    y_hat[t+h] = y[t] + h * (y[t] - y[1]) / (t - 1)

    This is a useful baseline for trending data where you expect
    the trend to continue.
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)

        self._drift: float | None = None
        self._last_value: float | None = None

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "DriftForecaster":
        """Fit the drift forecaster.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : ignored
            Not used.

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        if len(y) < 2:
            raise ValueError("Need at least 2 observations for drift method")

        # Drift = average change per period
        self._drift = (y[-1] - y[0]) / (len(y) - 1)
        self._last_value = y[-1]

        self._log(f"Fitted with drift={self._drift:.6f}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate drift forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Not used.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        h = np.arange(1, horizon + 1)
        forecast = self._last_value + h * self._drift

        return forecast

    def predict_interval(
        self,
        horizon: int,
        coverage: float = 0.95,
        X: Any | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate prediction intervals."""
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        point = self.predict(horizon)

        # Residuals from drift model (detrended differences)
        fitted = self.get_fitted_values()
        valid_mask = ~np.isnan(fitted)
        residuals = self.y_[valid_mask] - fitted[valid_mask]
        std = np.std(residuals)

        alpha = 1 - coverage
        z = -np.percentile(np.random.standard_normal(10000), alpha / 2 * 100)

        # Variance grows linearly with horizon for drift
        h = np.arange(1, horizon + 1)
        n = self.n_samples_
        var_factor = np.sqrt(h * (1 + h / n))
        margin = z * std * var_factor

        return point, point - margin, point + margin

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        self._check_is_fitted()

        fitted = np.empty(self.n_samples_)
        fitted[0] = np.nan

        # Each fitted value = previous actual + drift
        fitted[1:] = self.y_[:-1] + self._drift

        return fitted


class ThetaForecaster(BaseForecaster, UnivariateForecasterMixin):
    """Theta method forecaster.

    Implements the Theta method which won the M3 forecasting competition.
    Decomposes series into two theta lines and combines their forecasts.

    Parameters
    ----------
    theta : float, default=2.0
        Theta parameter. theta=0 gives linear regression,
        theta=2 is the standard Theta method.
    seasonal_period : int, optional
        If provided, applies seasonal adjustment before forecasting.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> model = ThetaForecaster(theta=2.0)
    >>> model.fit(y_train)
    >>> forecast = model.predict(horizon=12)

    Notes
    -----
    The Theta method:
    1. Decomposes series into theta=0 (linear trend) and theta=2 (amplified curvature)
    2. Forecasts theta=0 using linear regression
    3. Forecasts theta=2 using simple exponential smoothing
    4. Combines forecasts (equal weights)

    This simple method often outperforms complex approaches on many datasets.

    References
    ----------
    Assimakopoulos, V., & Nikolopoulos, K. (2000). The theta model:
    a decomposition approach to forecasting. International Journal
    of Forecasting, 16(4), 521-530.
    """

    def __init__(
        self,
        theta: float = 2.0,
        seasonal_period: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.theta = theta
        self.seasonal_period = seasonal_period

        # Fitted components
        self._drift: float | None = None
        self._ses_level: float | None = None
        self._ses_alpha: float = 0.5  # Could optimize
        self._seasonal_factors: np.ndarray | None = None
        self._deseasonalized: np.ndarray | None = None

    def _deseasonalize(self, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remove seasonal component using classical decomposition."""
        sp = self.seasonal_period
        n = len(y)

        # Compute seasonal indices using moving average
        if n < 2 * sp:
            return y, np.ones(sp)

        # Centered moving average for trend
        ma = np.convolve(y, np.ones(sp) / sp, mode='valid')

        # Adjust for even seasonal periods
        if sp % 2 == 0:
            ma = (ma[:-1] + ma[1:]) / 2

        # Pad MA to original length
        pad_left = (n - len(ma)) // 2
        pad_right = n - len(ma) - pad_left
        trend = np.concatenate([
            np.full(pad_left, ma[0]),
            ma,
            np.full(pad_right, ma[-1])
        ])

        # Seasonal = actual / trend (multiplicative)
        # Use additive for simplicity: seasonal = actual - trend
        seasonal_component = y - trend

        # Average seasonal factors
        factors = np.zeros(sp)
        counts = np.zeros(sp)
        for i in range(n):
            factors[i % sp] += seasonal_component[i]
            counts[i % sp] += 1
        factors = factors / np.maximum(counts, 1)

        # Normalize to sum to zero
        factors = factors - np.mean(factors)

        # Deseasonalize
        seasonal_indices = np.array([factors[i % sp] for i in range(n)])
        deseasonalized = y - seasonal_indices

        return deseasonalized, factors

    def fit(
        self,
        y: Any,
        X: Any | None = None,
        **fit_params,
    ) -> "ThetaForecaster":
        """Fit the Theta forecaster.

        Parameters
        ----------
        y : array-like
            Training time series.
        X : ignored
            Not used.

        Returns
        -------
        self
            Fitted forecaster.
        """
        y, _ = validate_forecast_input(y, X=None)
        y = self._validate_univariate(y)

        self.y_ = y.copy()
        self.n_samples_ = len(y)

        if len(y) < 3:
            raise ValueError("Need at least 3 observations for Theta method")

        # Deseasonalize if needed
        if self.seasonal_period is not None and self.seasonal_period > 1:
            y_adj, self._seasonal_factors = self._deseasonalize(y)
            self._deseasonalized = y_adj
        else:
            y_adj = y
            self._seasonal_factors = None
            self._deseasonalized = y

        n = len(y_adj)

        # Fit linear trend (theta=0 line)
        x = np.arange(n)
        slope, intercept = np.polyfit(x, y_adj, 1)
        self._drift = slope

        # Fit SES on theta=2 line
        # theta=2 line: z = theta * y - (theta-1) * linear_trend
        theta_2_line = self.theta * y_adj - (self.theta - 1) * (intercept + slope * x)

        # Simple exponential smoothing on theta_2 line
        level = theta_2_line[0]
        for t in range(1, n):
            level = self._ses_alpha * theta_2_line[t] + (1 - self._ses_alpha) * level
        self._ses_level = level

        self._log(f"Fitted Theta(theta={self.theta}), drift={self._drift:.6f}")
        self.is_fitted_ = True
        return self

    def predict(
        self,
        horizon: int,
        X: Any | None = None,
    ) -> np.ndarray:
        """Generate Theta forecasts.

        Parameters
        ----------
        horizon : int
            Number of steps ahead to forecast.
        X : ignored
            Not used.

        Returns
        -------
        np.ndarray of shape (horizon,)
            Forecasts.
        """
        self._check_is_fitted()
        horizon = self._validate_horizon(horizon)

        n = self.n_samples_
        h = np.arange(1, horizon + 1)

        # Linear trend forecast (theta=0)
        x_last = n - 1
        y_last = self._deseasonalized[-1]
        linear_forecast = y_last + self._drift * h

        # SES forecast (theta=2) - flat
        ses_forecast = np.full(horizon, self._ses_level)

        # Combine: average of theta=0 and theta=2 forecasts
        forecast = (linear_forecast + ses_forecast) / 2

        # Re-seasonalize
        if self._seasonal_factors is not None:
            sp = self.seasonal_period
            for i in range(horizon):
                forecast[i] += self._seasonal_factors[(n + i) % sp]

        return forecast

    def get_fitted_values(self) -> np.ndarray:
        """Get in-sample fitted values."""
        self._check_is_fitted()

        # For simplicity, use one-step-ahead from combined model
        fitted = np.empty(self.n_samples_)
        fitted[0] = np.nan

        y_adj = self._deseasonalized

        # One-step-ahead forecasts
        level = y_adj[0]
        for t in range(1, self.n_samples_):
            # Linear component
            linear_pred = y_adj[t - 1] + self._drift

            # SES component
            theta_2_val = self.theta * y_adj[t - 1] - (self.theta - 1) * (
                y_adj[0] + self._drift * (t - 1)
            )
            level = self._ses_alpha * theta_2_val + (1 - self._ses_alpha) * level

            # Combined
            fitted[t] = (linear_pred + level) / 2

            # Re-seasonalize
            if self._seasonal_factors is not None:
                fitted[t] += self._seasonal_factors[t % self.seasonal_period]

        return fitted
