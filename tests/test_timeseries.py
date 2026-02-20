"""Tests for endgame.timeseries module.

Covers baseline forecasters, cross-validation splitters, forecasting metrics,
and ROCKET-family classifiers (with optional dependency skipping).
"""

import numpy as np
import pytest

from endgame.timeseries import (
    NaiveForecaster,
    SeasonalNaiveForecaster,
    MovingAverageForecaster,
    ExponentialSmoothingForecaster,
    DriftForecaster,
    ThetaForecaster,
    ExpandingWindowCV,
    SlidingWindowCV,
    BlockedTimeSeriesSplit,
    mase,
    smape,
    mape,
    rmsse,
    wape,
    coverage,
    interval_width,
    winkler_score,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic time series data
# ---------------------------------------------------------------------------

@pytest.fixture
def constant_series():
    """Constant series (value = 5.0, length 50)."""
    return np.full(50, 5.0)


@pytest.fixture
def linear_trend():
    """Linearly increasing series: 0, 1, 2, ..., 99."""
    return np.arange(100, dtype=np.float64)


@pytest.fixture
def seasonal_series():
    """Series with period-7 seasonality and a linear trend."""
    n = 105  # 15 complete weeks
    trend = np.linspace(0, 10, n)
    season = 3.0 * np.sin(2 * np.pi * np.arange(n) / 7)
    return trend + season


@pytest.fixture
def random_walk():
    """Random walk of length 200."""
    rng = np.random.RandomState(42)
    return np.cumsum(rng.randn(200))


# ===================================================================
# 1. NaiveForecaster
# ===================================================================

class TestNaiveForecaster:

    def test_last_strategy(self, linear_trend):
        model = NaiveForecaster(strategy="last")
        model.fit(linear_trend)
        forecast = model.predict(horizon=10)
        assert forecast.shape == (10,)
        np.testing.assert_array_equal(forecast, np.full(10, 99.0))

    def test_mean_strategy(self, linear_trend):
        model = NaiveForecaster(strategy="mean")
        model.fit(linear_trend)
        forecast = model.predict(horizon=5)
        expected = np.mean(linear_trend)
        np.testing.assert_allclose(forecast, np.full(5, expected))

    def test_median_strategy(self):
        y = np.array([1.0, 2.0, 3.0, 100.0])
        model = NaiveForecaster(strategy="median")
        model.fit(y)
        forecast = model.predict(horizon=3)
        np.testing.assert_allclose(forecast, np.full(3, np.median(y)))

    def test_constant_series_all_strategies(self, constant_series):
        for strategy in ("last", "mean", "median"):
            model = NaiveForecaster(strategy=strategy)
            model.fit(constant_series)
            forecast = model.predict(horizon=7)
            np.testing.assert_allclose(forecast, np.full(7, 5.0))

    def test_fit_predict_roundtrip(self, random_walk):
        model = NaiveForecaster(strategy="last")
        forecast = model.fit_predict(random_walk, horizon=5)
        assert forecast.shape == (5,)
        np.testing.assert_allclose(forecast, np.full(5, random_walk[-1]))

    def test_predict_before_fit_raises(self):
        model = NaiveForecaster()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(horizon=5)

    def test_invalid_strategy_raises(self):
        model = NaiveForecaster(strategy="unknown")
        with pytest.raises(ValueError, match="Unknown strategy"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_invalid_horizon_raises(self, constant_series):
        model = NaiveForecaster()
        model.fit(constant_series)
        with pytest.raises(ValueError, match="horizon must be >= 1"):
            model.predict(horizon=0)

    def test_update(self, constant_series):
        model = NaiveForecaster(strategy="last")
        model.fit(constant_series)
        model.update(np.array([10.0]))
        forecast = model.predict(horizon=3)
        np.testing.assert_allclose(forecast, np.full(3, 10.0))

    def test_fitted_values_last(self, linear_trend):
        model = NaiveForecaster(strategy="last")
        model.fit(linear_trend)
        fitted = model.get_fitted_values()
        assert fitted.shape == (100,)
        assert np.isnan(fitted[0])
        np.testing.assert_array_equal(fitted[1:], linear_trend[:-1])

    def test_predict_interval(self, random_walk):
        model = NaiveForecaster(strategy="last")
        model.fit(random_walk)
        point, lower, upper = model.predict_interval(horizon=10, coverage=0.95)
        assert point.shape == lower.shape == upper.shape == (10,)
        assert np.all(lower <= point)
        assert np.all(upper >= point)

    def test_score(self, linear_trend):
        model = NaiveForecaster(strategy="mean")
        model.fit(linear_trend)
        # Negative MSE (sklearn convention)
        score = model.score(linear_trend[-10:], horizon=10, metric="mse")
        assert score <= 0


# ===================================================================
# 2. SeasonalNaiveForecaster
# ===================================================================

class TestSeasonalNaiveForecaster:

    def test_basic_seasonal(self, seasonal_series):
        model = SeasonalNaiveForecaster(seasonal_period=7)
        model.fit(seasonal_series)
        forecast = model.predict(horizon=7)
        assert forecast.shape == (7,)
        # Should repeat the last 7 values
        np.testing.assert_array_equal(forecast, seasonal_series[-7:])

    def test_multi_cycle_forecast(self, seasonal_series):
        model = SeasonalNaiveForecaster(seasonal_period=7)
        model.fit(seasonal_series)
        forecast = model.predict(horizon=21)
        # Three complete repetitions of the last week
        expected = np.tile(seasonal_series[-7:], 3)
        np.testing.assert_array_equal(forecast, expected)

    def test_partial_cycle_forecast(self, seasonal_series):
        model = SeasonalNaiveForecaster(seasonal_period=7)
        model.fit(seasonal_series)
        forecast = model.predict(horizon=10)
        expected = np.tile(seasonal_series[-7:], 2)[:10]
        np.testing.assert_array_equal(forecast, expected)

    def test_period_1_equals_naive_last(self, linear_trend):
        sn = SeasonalNaiveForecaster(seasonal_period=1)
        sn.fit(linear_trend)
        naive = NaiveForecaster(strategy="last")
        naive.fit(linear_trend)
        np.testing.assert_array_equal(
            sn.predict(horizon=5), naive.predict(horizon=5)
        )

    def test_series_shorter_than_period_raises(self):
        model = SeasonalNaiveForecaster(seasonal_period=10)
        with pytest.raises(ValueError, match="must be >= seasonal_period"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_invalid_period_raises(self):
        model = SeasonalNaiveForecaster(seasonal_period=0)
        with pytest.raises(ValueError, match="seasonal_period must be >= 1"):
            model.fit(np.arange(20, dtype=float))

    def test_fitted_values(self, seasonal_series):
        model = SeasonalNaiveForecaster(seasonal_period=7)
        model.fit(seasonal_series)
        fitted = model.get_fitted_values()
        assert np.all(np.isnan(fitted[:7]))
        np.testing.assert_array_equal(fitted[7:], seasonal_series[:-7])


# ===================================================================
# 3. MovingAverageForecaster
# ===================================================================

class TestMovingAverageForecaster:

    def test_constant_series(self, constant_series):
        model = MovingAverageForecaster(window=5)
        model.fit(constant_series)
        forecast = model.predict(horizon=10)
        np.testing.assert_allclose(forecast, np.full(10, 5.0))

    def test_window_1_equals_naive_last(self, linear_trend):
        model = MovingAverageForecaster(window=1)
        model.fit(linear_trend)
        forecast = model.predict(horizon=1)
        np.testing.assert_allclose(forecast, [99.0])

    def test_forecast_shape(self, random_walk):
        model = MovingAverageForecaster(window=10)
        model.fit(random_walk)
        forecast = model.predict(horizon=20)
        assert forecast.shape == (20,)

    def test_weighted_ma(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = [1, 2, 3, 4, 5]
        model = MovingAverageForecaster(window=5, weights=weights)
        model.fit(y)
        forecast = model.predict(horizon=1)
        # Weighted average: (1*1 + 2*2 + 3*3 + 4*4 + 5*5) / 15
        expected = np.dot(np.array(weights) / sum(weights), y)
        np.testing.assert_allclose(forecast, [expected])

    def test_window_larger_than_series_raises(self):
        model = MovingAverageForecaster(window=10)
        with pytest.raises(ValueError, match="must be >= window"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_weights_length_mismatch_raises(self):
        model = MovingAverageForecaster(window=3, weights=[1, 2])
        with pytest.raises(ValueError, match="weights length"):
            model.fit(np.arange(10, dtype=float))

    def test_fitted_values(self, constant_series):
        model = MovingAverageForecaster(window=5)
        model.fit(constant_series)
        fitted = model.get_fitted_values()
        assert np.all(np.isnan(fitted[:4]))
        np.testing.assert_allclose(fitted[4:], np.full(46, 5.0))


# ===================================================================
# 4. ExponentialSmoothingForecaster
# ===================================================================

class TestExponentialSmoothingForecaster:

    def test_ses_constant_series(self, constant_series):
        model = ExponentialSmoothingForecaster(alpha=0.3)
        model.fit(constant_series)
        forecast = model.predict(horizon=5)
        np.testing.assert_allclose(forecast, np.full(5, 5.0), atol=1e-10)

    def test_ses_flat_forecast(self, random_walk):
        model = ExponentialSmoothingForecaster(alpha=0.5)
        model.fit(random_walk)
        forecast = model.predict(horizon=10)
        # SES produces a flat forecast (all same value)
        np.testing.assert_allclose(forecast, np.full(10, forecast[0]))

    def test_holt_trend(self, linear_trend):
        model = ExponentialSmoothingForecaster(alpha=0.8, beta=0.2)
        model.fit(linear_trend)
        forecast = model.predict(horizon=5)
        assert forecast.shape == (5,)
        # With trend, forecasts should increase
        assert np.all(np.diff(forecast) > 0)

    def test_optimize(self, random_walk):
        model = ExponentialSmoothingForecaster(alpha=0.3, optimize=True)
        model.fit(random_walk)
        assert model.alpha_ is not None
        assert 0 < model.alpha_ < 1

    def test_invalid_alpha_raises(self):
        model = ExponentialSmoothingForecaster(alpha=1.5)
        with pytest.raises(ValueError, match="alpha must be in"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_invalid_beta_raises(self):
        model = ExponentialSmoothingForecaster(alpha=0.3, beta=-0.1)
        with pytest.raises(ValueError, match="beta must be in"):
            model.fit(np.array([1.0, 2.0, 3.0]))

    def test_predict_interval(self, random_walk):
        model = ExponentialSmoothingForecaster(alpha=0.3)
        model.fit(random_walk)
        point, lower, upper = model.predict_interval(horizon=5, coverage=0.95)
        assert np.all(lower <= point)
        assert np.all(upper >= point)

    def test_fitted_values(self, linear_trend):
        model = ExponentialSmoothingForecaster(alpha=0.5)
        model.fit(linear_trend)
        fitted = model.get_fitted_values()
        assert fitted.shape == (100,)
        assert np.isnan(fitted[0])
        assert not np.any(np.isnan(fitted[1:]))


# ===================================================================
# 5. DriftForecaster
# ===================================================================

class TestDriftForecaster:

    def test_linear_trend(self, linear_trend):
        model = DriftForecaster()
        model.fit(linear_trend)
        forecast = model.predict(horizon=5)
        # Drift = (99 - 0) / 99 = 1.0 per step
        expected = 99.0 + np.arange(1, 6) * 1.0
        np.testing.assert_allclose(forecast, expected)

    def test_constant_series_zero_drift(self, constant_series):
        model = DriftForecaster()
        model.fit(constant_series)
        forecast = model.predict(horizon=5)
        np.testing.assert_allclose(forecast, np.full(5, 5.0))

    def test_too_short_raises(self):
        model = DriftForecaster()
        with pytest.raises(ValueError):
            # Single element squeezes to scalar, which fails univariate validation
            model.fit(np.array([1.0]))

    def test_forecast_shape(self, random_walk):
        model = DriftForecaster()
        model.fit(random_walk)
        forecast = model.predict(horizon=15)
        assert forecast.shape == (15,)

    def test_fitted_values(self, linear_trend):
        model = DriftForecaster()
        model.fit(linear_trend)
        fitted = model.get_fitted_values()
        assert np.isnan(fitted[0])
        # drift = 1.0, so fitted[t] = y[t-1] + 1 = t-1+1 = t
        np.testing.assert_allclose(fitted[1:], linear_trend[1:])


# ===================================================================
# 6. ThetaForecaster
# ===================================================================

class TestThetaForecaster:

    def test_basic_forecast(self, linear_trend):
        model = ThetaForecaster(theta=2.0)
        model.fit(linear_trend)
        forecast = model.predict(horizon=5)
        assert forecast.shape == (5,)
        # For a perfectly linear series, theta forecast should be close
        # to a continuation of the trend
        assert forecast[0] > linear_trend[-1] - 5

    def test_too_short_raises(self):
        model = ThetaForecaster()
        with pytest.raises(ValueError, match="at least 3 observations"):
            model.fit(np.array([1.0, 2.0]))

    def test_with_seasonal_adjustment(self, seasonal_series):
        model = ThetaForecaster(theta=2.0, seasonal_period=7)
        model.fit(seasonal_series)
        forecast = model.predict(horizon=14)
        assert forecast.shape == (14,)
        assert np.all(np.isfinite(forecast))

    def test_constant_series(self, constant_series):
        model = ThetaForecaster(theta=2.0)
        model.fit(constant_series)
        forecast = model.predict(horizon=5)
        np.testing.assert_allclose(forecast, np.full(5, 5.0), atol=0.1)

    def test_fitted_values(self, random_walk):
        model = ThetaForecaster(theta=2.0)
        model.fit(random_walk)
        fitted = model.get_fitted_values()
        assert fitted.shape == (200,)
        assert np.isnan(fitted[0])
        assert np.all(np.isfinite(fitted[1:]))


# ===================================================================
# 7. ExpandingWindowCV
# ===================================================================

class TestExpandingWindowCV:

    def test_basic_split(self):
        X = np.arange(100)
        cv = ExpandingWindowCV(n_splits=3, initial_train_size=40, val_size=10)
        splits = list(cv.split(X))
        assert len(splits) == 3

    def test_train_expands(self):
        X = np.arange(100)
        cv = ExpandingWindowCV(n_splits=3, initial_train_size=40, val_size=10)
        splits = list(cv.split(X))
        train_sizes = [len(train) for train, _ in splits]
        # Each fold should have a larger training set
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[1] > train_sizes[0]

    def test_val_size_constant(self):
        X = np.arange(100)
        cv = ExpandingWindowCV(n_splits=3, initial_train_size=40, val_size=10)
        splits = list(cv.split(X))
        for _, val in splits:
            assert len(val) == 10

    def test_no_overlap(self):
        X = np.arange(100)
        cv = ExpandingWindowCV(n_splits=3, initial_train_size=40, val_size=10)
        for train, val in cv.split(X):
            assert len(np.intersect1d(train, val)) == 0

    def test_gap(self):
        X = np.arange(100)
        cv = ExpandingWindowCV(
            n_splits=2, initial_train_size=30, val_size=10, gap=5
        )
        for train, val in cv.split(X):
            assert val[0] - train[-1] > 1  # gap enforced

    def test_get_n_splits(self):
        cv = ExpandingWindowCV(n_splits=5)
        assert cv.get_n_splits() == 5


# ===================================================================
# 8. SlidingWindowCV
# ===================================================================

class TestSlidingWindowCV:

    def test_basic_split(self):
        X = np.arange(100)
        cv = SlidingWindowCV(n_splits=3, train_size=30, val_size=10)
        splits = list(cv.split(X))
        assert len(splits) == 3

    def test_train_size_constant(self):
        X = np.arange(100)
        cv = SlidingWindowCV(n_splits=3, train_size=30, val_size=10)
        for train, _ in cv.split(X):
            assert len(train) == 30

    def test_val_size_constant(self):
        X = np.arange(100)
        cv = SlidingWindowCV(n_splits=3, train_size=30, val_size=10)
        for _, val in cv.split(X):
            assert len(val) == 10

    def test_windows_slide(self):
        X = np.arange(200)
        cv = SlidingWindowCV(n_splits=3, train_size=30, val_size=10)
        splits = list(cv.split(X))
        starts = [train[0] for train, _ in splits]
        # Each fold's train window should start later
        assert starts == sorted(starts)
        assert len(set(starts)) == 3

    def test_no_overlap(self):
        X = np.arange(200)
        cv = SlidingWindowCV(n_splits=3, train_size=30, val_size=10)
        for train, val in cv.split(X):
            assert len(np.intersect1d(train, val)) == 0


# ===================================================================
# 9. BlockedTimeSeriesSplit
# ===================================================================

class TestBlockedTimeSeriesSplit:

    def test_basic_split(self):
        X = np.arange(120)
        cv = BlockedTimeSeriesSplit(n_splits=3)
        splits = list(cv.split(X))
        assert len(splits) == 3

    def test_gap_before(self):
        X = np.arange(120)
        cv = BlockedTimeSeriesSplit(n_splits=3, gap_before=5)
        for train, val in cv.split(X):
            if len(train) > 0 and len(val) > 0:
                assert val[0] - train[-1] > 1

    def test_no_overlap(self):
        X = np.arange(120)
        cv = BlockedTimeSeriesSplit(n_splits=3, gap_before=2)
        for train, val in cv.split(X):
            assert len(np.intersect1d(train, val)) == 0

    def test_temporal_order(self):
        X = np.arange(120)
        cv = BlockedTimeSeriesSplit(n_splits=3)
        for train, val in cv.split(X):
            assert train[-1] < val[0]


# ===================================================================
# 10. Metrics
# ===================================================================

class TestMetrics:

    def test_smape_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert smape(y, y) == 0.0

    def test_smape_range(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([3.0, 4.0, 5.0])
        result = smape(y_true, y_pred)
        assert 0 <= result <= 200

    def test_smape_symmetric(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 3.0, 4.0])
        assert smape(y_true, y_pred) == pytest.approx(smape(y_pred, y_true))

    def test_smape_known_value(self):
        y_true = np.array([100.0])
        y_pred = np.array([110.0])
        # sMAPE = 2 * |100-110| / (|100| + |110|) * 100 = 2*10/210*100
        expected = 2 * 10.0 / 210.0 * 100
        assert smape(y_true, y_pred) == pytest.approx(expected)

    def test_smape_zeros(self):
        y = np.array([0.0, 0.0])
        assert smape(y, y) == 0.0

    def test_mape_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mape(y, y) == 0.0

    def test_mape_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        # MAPE = mean(|10/100|, |10/200|) * 100 = mean(0.1, 0.05) * 100 = 7.5
        expected = 7.5
        assert mape(y_true, y_pred) == pytest.approx(expected)

    def test_mape_skips_zeros(self):
        y_true = np.array([0.0, 100.0])
        y_pred = np.array([5.0, 110.0])
        # Only uses the second element: |10/100| * 100 = 10
        assert mape(y_true, y_pred) == pytest.approx(10.0)

    def test_mape_all_zeros(self):
        y_true = np.array([0.0, 0.0])
        y_pred = np.array([1.0, 2.0])
        assert mape(y_true, y_pred) == float('inf')

    def test_mase_perfect(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        result = mase(y_true, y_true, y_train)
        assert result == 0.0

    def test_mase_naive_equals_one(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        # Naive forecast = repeat last value (5.0)
        y_pred = np.array([5.0, 5.0])
        # MAE of forecast = mean(|6-5|, |7-5|) = 1.5
        # In-sample naive MAE = mean(|2-1|, |3-2|, |4-3|, |5-4|) = 1.0
        # MASE = 1.5 / 1.0 = 1.5
        result = mase(y_true, y_pred, y_train)
        assert result == pytest.approx(1.5)

    def test_mase_seasonal(self):
        y_train = np.array([1.0, 3.0, 1.0, 3.0, 1.0, 3.0])
        y_true = np.array([1.0, 3.0])
        y_pred = np.array([1.0, 3.0])
        result = mase(y_true, y_pred, y_train, seasonal_period=2)
        assert result == 0.0

    def test_rmsse_perfect(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        result = rmsse(y_true, y_true, y_train)
        assert result == 0.0

    def test_rmsse_known_value(self):
        y_train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_true = np.array([6.0, 7.0])
        y_pred = np.array([5.0, 5.0])
        # Forecast MSE = mean((6-5)^2, (7-5)^2) = mean(1, 4) = 2.5
        # Scale = mean((2-1)^2, (3-2)^2, (4-3)^2, (5-4)^2) = 1.0
        # RMSSE = sqrt(2.5 / 1.0) = sqrt(2.5)
        result = rmsse(y_true, y_pred, y_train)
        assert result == pytest.approx(np.sqrt(2.5))

    def test_wape_perfect(self):
        y = np.array([10.0, 20.0, 30.0])
        assert wape(y, y) == 0.0

    def test_wape_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([90.0, 220.0])
        # WAPE = (|100-90| + |200-220|) / (|100| + |200|) * 100
        #      = (10 + 20) / 300 * 100 = 10.0
        assert wape(y_true, y_pred) == pytest.approx(10.0)

    def test_coverage_all_inside(self):
        y_true = np.array([1.0, 2.0, 3.0])
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert coverage(y_true, lower, upper) == 1.0

    def test_coverage_none_inside(self):
        y_true = np.array([5.0, 6.0, 7.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([1.0, 1.0, 1.0])
        assert coverage(y_true, lower, upper) == 0.0

    def test_coverage_partial(self):
        y_true = np.array([1.0, 5.0])
        lower = np.array([0.0, 0.0])
        upper = np.array([2.0, 2.0])
        assert coverage(y_true, lower, upper) == 0.5

    def test_interval_width(self):
        lower = np.array([0.0, 1.0, 2.0])
        upper = np.array([2.0, 3.0, 4.0])
        assert interval_width(lower, upper) == pytest.approx(2.0)

    def test_winkler_score_all_inside(self):
        y_true = np.array([1.5])
        lower = np.array([1.0])
        upper = np.array([2.0])
        # Width = 1.0, no penalty
        result = winkler_score(y_true, lower, upper, alpha=0.05)
        assert result == pytest.approx(1.0)

    def test_winkler_score_below(self):
        y_true = np.array([0.0])
        lower = np.array([1.0])
        upper = np.array([2.0])
        # Width = 1.0, penalty = 2/0.05 * 1.0 = 40.0
        result = winkler_score(y_true, lower, upper, alpha=0.05)
        assert result == pytest.approx(1.0 + 40.0)

    def test_winkler_score_above(self):
        y_true = np.array([3.0])
        lower = np.array([1.0])
        upper = np.array([2.0])
        # Width = 1.0, penalty = 2/0.05 * 1.0 = 40.0
        result = winkler_score(y_true, lower, upper, alpha=0.05)
        assert result == pytest.approx(1.0 + 40.0)


# ===================================================================
# 11. MiniRocketClassifier (optional dependency)
# ===================================================================

class TestMiniRocketClassifier:

    @pytest.fixture(autouse=True)
    def _skip_if_no_sktime(self):
        pytest.importorskip("sktime")

    def _make_ts_data(self, n_samples=60, n_timepoints=50, n_classes=3):
        """Create synthetic time series classification data."""
        rng = np.random.RandomState(42)
        X = rng.randn(n_samples, n_timepoints)
        y = rng.choice(n_classes, size=n_samples)
        # Inject class-specific signal
        for cls in range(n_classes):
            mask = y == cls
            X[mask] += cls * 0.5
        return X, y

    def test_fit_predict(self):
        from endgame.timeseries import MiniRocketClassifier

        X, y = self._make_ts_data(n_samples=40, n_timepoints=30, n_classes=2)
        clf = MiniRocketClassifier(n_kernels=500, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (40,)
        assert set(preds).issubset(set(y))

    def test_score(self):
        from endgame.timeseries import MiniRocketClassifier

        X, y = self._make_ts_data(n_samples=40, n_timepoints=30, n_classes=2)
        clf = MiniRocketClassifier(n_kernels=500, random_state=42)
        clf.fit(X, y)
        acc = clf.score(X, y)
        assert 0.0 <= acc <= 1.0

    def test_predict_proba(self):
        from endgame.timeseries import MiniRocketClassifier

        X, y = self._make_ts_data(n_samples=40, n_timepoints=30, n_classes=2)
        clf = MiniRocketClassifier(n_kernels=500, random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape[0] == 40
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_classes_stored(self):
        from endgame.timeseries import MiniRocketClassifier

        X, y = self._make_ts_data(n_samples=30, n_timepoints=20, n_classes=3)
        clf = MiniRocketClassifier(n_kernels=500, random_state=42)
        clf.fit(X, y)
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1, 2]))


# ===================================================================
# 12. HydraClassifier (optional dependency)
# ===================================================================

class TestHydraClassifier:

    @pytest.fixture(autouse=True)
    def _skip_if_no_sktime(self):
        pytest.importorskip("sktime")
        # Also check for hydra support specifically
        try:
            from sktime.transformations.panel.hydra import Hydra  # noqa: F401
        except ImportError:
            pytest.skip("sktime HYDRA support not available")

    def test_fit_predict(self):
        from endgame.timeseries import HydraClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(30, 40)
        y = np.array([0] * 15 + [1] * 15)
        X[y == 1] += 0.5
        clf = HydraClassifier(n_kernels=4, n_groups=8, random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (30,)


# ===================================================================
# 13. Edge cases and cross-cutting concerns
# ===================================================================

class TestEdgeCases:

    def test_single_value_series_naive_mean(self):
        # A 1-element 1D array squeezes to scalar (ndim=0), which the
        # univariate validator rejects. Use at least 2 elements.
        model = NaiveForecaster(strategy="mean")
        model.fit(np.array([42.0, 42.0]))
        forecast = model.predict(horizon=3)
        np.testing.assert_allclose(forecast, np.full(3, 42.0))

    def test_two_value_series_drift(self):
        model = DriftForecaster()
        model.fit(np.array([0.0, 10.0]))
        forecast = model.predict(horizon=3)
        np.testing.assert_allclose(forecast, [20.0, 30.0, 40.0])

    def test_list_input(self):
        model = NaiveForecaster(strategy="last")
        model.fit([1.0, 2.0, 3.0])
        forecast = model.predict(horizon=2)
        np.testing.assert_allclose(forecast, [3.0, 3.0])

    def test_integer_input_converted(self):
        model = NaiveForecaster(strategy="mean")
        model.fit(np.array([1, 2, 3, 4, 5]))
        forecast = model.predict(horizon=1)
        np.testing.assert_allclose(forecast, [3.0])

    def test_horizon_type_validation(self, constant_series):
        model = NaiveForecaster()
        model.fit(constant_series)
        with pytest.raises(TypeError, match="horizon must be int"):
            model.predict(horizon=3.5)

    def test_repr(self):
        model = NaiveForecaster(strategy="mean")
        r = repr(model)
        assert "NaiveForecaster" in r
        assert "mean" in r

    def test_get_params(self):
        model = ExponentialSmoothingForecaster(alpha=0.5, beta=0.1)
        params = model.get_params()
        assert params["alpha"] == 0.5
        assert params["beta"] == 0.1

    def test_set_params(self):
        model = NaiveForecaster(strategy="last")
        model.set_params(strategy="mean")
        assert model.strategy == "mean"

    def test_nan_input_raises(self):
        model = NaiveForecaster()
        with pytest.raises(ValueError, match="NaN"):
            model.fit(np.array([1.0, np.nan, 3.0]))

    def test_multivariate_input_raises_on_univariate_forecaster(self):
        model = NaiveForecaster()
        with pytest.raises(ValueError, match="univariate"):
            model.fit(np.array([[1.0, 2.0], [3.0, 4.0]]))
