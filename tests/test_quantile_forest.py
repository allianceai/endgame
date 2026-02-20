"""Tests for Quantile Regression Forest."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from endgame.models.trees import (
    QuantileRegressorForest,
    pinball_loss,
    interval_coverage,
    interval_width,
)


@pytest.fixture
def regression_data():
    """Generate simple regression data."""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def heteroscedastic_data():
    """Generate data with heteroscedastic noise (variance depends on X)."""
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    # Noise variance increases with X[:, 0]
    noise_scale = 1 + 2 * np.abs(X[:, 0])
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * noise_scale
    return train_test_split(X, y, test_size=0.2, random_state=42)


class TestQuantileRegressorForest:
    """Tests for QuantileRegressorForest."""
    
    def test_fit_predict_single_quantile(self, regression_data):
        """Test basic fit and predict with single quantile."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            quantiles=0.5,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        predictions = qrf.predict(X_test)
        
        # Check shape
        assert predictions.shape == (len(X_test),)
        
        # Check that predictions are reasonable
        assert np.isfinite(predictions).all()
        
        # Check fitted attributes
        assert hasattr(qrf, "estimators_")
        assert hasattr(qrf, "leaf_samples_")
        assert hasattr(qrf, "feature_importances_")
        assert len(qrf.estimators_) == 50
        assert len(qrf.leaf_samples_) == 50
    
    def test_fit_predict_multiple_quantiles(self, regression_data):
        """Test fit and predict with multiple quantiles."""
        X_train, X_test, y_train, y_test = regression_data
        
        quantiles = [0.1, 0.5, 0.9]
        qrf = QuantileRegressorForest(
            n_estimators=50,
            quantiles=quantiles,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        predictions = qrf.predict(X_test)
        
        # Check shape
        assert predictions.shape == (len(X_test), len(quantiles))
        
        # Check that quantiles are ordered
        assert np.all(predictions[:, 0] <= predictions[:, 1])
        assert np.all(predictions[:, 1] <= predictions[:, 2])
    
    def test_predict_quantiles_method(self, regression_data):
        """Test predict_quantiles with different quantiles than training."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Fit with default quantile
        qrf = QuantileRegressorForest(
            n_estimators=50,
            quantiles=0.5,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        # Predict different quantiles
        q_pred = qrf.predict_quantiles(X_test, [0.25, 0.75])
        
        assert q_pred.shape == (len(X_test), 2)
        assert np.all(q_pred[:, 0] <= q_pred[:, 1])
    
    def test_predict_interval(self, regression_data):
        """Test prediction interval generation."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=100,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        lower, upper = qrf.predict_interval(X_test, coverage=0.9)
        
        # Check shapes
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        
        # Check ordering
        assert np.all(lower <= upper)
        
        # Check coverage is reasonable (not exact due to finite samples)
        coverage = interval_coverage(y_test, lower, upper)
        # Should be approximately 0.9, allow some slack
        assert 0.7 < coverage < 1.0
    
    def test_predict_mean(self, regression_data):
        """Test mean prediction method."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        mean_pred = qrf.predict_mean(X_test)
        
        assert mean_pred.shape == (len(X_test),)
        assert np.isfinite(mean_pred).all()
        
        # Mean should be close to median for symmetric distributions
        median_pred = qrf.predict_quantiles(X_test, 0.5)
        correlation = np.corrcoef(mean_pred, median_pred)[0, 1]
        assert correlation > 0.9
    
    def test_predict_std(self, regression_data):
        """Test standard deviation prediction method."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        std_pred = qrf.predict_std(X_test)
        
        assert std_pred.shape == (len(X_test),)
        assert np.all(std_pred >= 0)
        assert np.isfinite(std_pred).all()
    
    def test_heteroscedastic_coverage(self, heteroscedastic_data):
        """Test that QRF adapts to heteroscedastic data."""
        X_train, X_test, y_train, y_test = heteroscedastic_data
        
        qrf = QuantileRegressorForest(
            n_estimators=100,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        # Get intervals
        lower, upper = qrf.predict_interval(X_test, coverage=0.8)
        
        # Intervals should be wider for samples with higher X[:, 0]
        widths = upper - lower
        
        # Split by X[:, 0] magnitude
        high_var_mask = np.abs(X_test[:, 0]) > np.median(np.abs(X_test[:, 0]))
        
        mean_width_high = np.mean(widths[high_var_mask])
        mean_width_low = np.mean(widths[~high_var_mask])
        
        # High variance region should have wider intervals
        assert mean_width_high > mean_width_low * 0.9  # Allow some slack
    
    def test_oob_score(self, regression_data):
        """Test out-of-bag score computation."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=100,
            oob_score=True,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        assert hasattr(qrf, "oob_score_")
        assert hasattr(qrf, "oob_prediction_")
        assert -1.0 <= qrf.oob_score_ <= 1.0  # R² can be negative
        assert qrf.oob_prediction_.shape == (len(X_train),)
    
    def test_feature_importances(self, regression_data):
        """Test feature importances computation."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        importances = qrf.feature_importances_
        
        assert importances.shape == (X_train.shape[1],)
        assert np.all(importances >= 0)
        assert np.isclose(np.sum(importances), 1.0)
    
    def test_apply(self, regression_data):
        """Test apply method (leaf indices)."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        leaf_indices = qrf.apply(X_test)
        
        assert leaf_indices.shape == (len(X_test), 50)
        assert leaf_indices.dtype == np.int64
    
    def test_warm_start(self, regression_data):
        """Test warm start functionality."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=30,
            warm_start=True,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        assert len(qrf.estimators_) == 30
        
        # Add more trees
        qrf.n_estimators = 50
        qrf.fit(X_train, y_train)
        
        assert len(qrf.estimators_) == 50
    
    def test_max_samples(self, regression_data):
        """Test max_samples parameter for subsampling."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(
            n_estimators=50,
            max_samples=0.5,
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        predictions = qrf.predict(X_test)
        assert predictions.shape == (len(X_test),)
    
    def test_invalid_quantiles(self, regression_data):
        """Test that invalid quantiles raise errors."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(quantiles=1.5)
        with pytest.raises(ValueError, match="quantiles must be in"):
            qrf.fit(X_train, y_train)
        
        qrf2 = QuantileRegressorForest(quantiles=-0.1)
        with pytest.raises(ValueError, match="quantiles must be in"):
            qrf2.fit(X_train, y_train)
    
    def test_invalid_coverage(self, regression_data):
        """Test that invalid coverage raises errors."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(n_estimators=10, random_state=42)
        qrf.fit(X_train, y_train)
        
        with pytest.raises(ValueError, match="coverage must be in"):
            qrf.predict_interval(X_test, coverage=1.5)
        
        with pytest.raises(ValueError, match="coverage must be in"):
            qrf.predict_interval(X_test, coverage=0.0)
    
    def test_reproducibility(self, regression_data):
        """Test that random_state ensures reproducibility."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf1 = QuantileRegressorForest(n_estimators=20, random_state=42)
        qrf1.fit(X_train, y_train)
        pred1 = qrf1.predict(X_test)
        
        qrf2 = QuantileRegressorForest(n_estimators=20, random_state=42)
        qrf2.fit(X_train, y_train)
        pred2 = qrf2.predict(X_test)
        
        np.testing.assert_array_almost_equal(pred1, pred2)
    
    def test_n_estimators_property(self, regression_data):
        """Test n_estimators_ property."""
        X_train, X_test, y_train, y_test = regression_data
        
        qrf = QuantileRegressorForest(n_estimators=50, random_state=42)
        
        # Before fitting
        assert qrf.n_estimators_ == 0
        
        qrf.fit(X_train, y_train)
        
        # After fitting
        assert qrf.n_estimators_ == 50
    
    def test_parallel_fitting(self, regression_data):
        """Test parallel fitting with n_jobs."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Sequential
        qrf1 = QuantileRegressorForest(n_estimators=30, n_jobs=1, random_state=42)
        qrf1.fit(X_train, y_train)
        
        # Parallel
        qrf2 = QuantileRegressorForest(n_estimators=30, n_jobs=2, random_state=42)
        qrf2.fit(X_train, y_train)
        
        # Results should be identical with same random_state
        np.testing.assert_array_almost_equal(
            qrf1.predict(X_test),
            qrf2.predict(X_test),
        )


class TestPinballLoss:
    """Tests for pinball loss function."""
    
    def test_pinball_loss_median(self):
        """Test pinball loss at median quantile."""
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.5, 2.0, 2.5, 4.0, 4.5])
        
        loss = pinball_loss(y_true, y_pred, 0.5)
        
        # For median, pinball loss = 0.5 * MAE
        mae = np.mean(np.abs(y_true - y_pred))
        expected = 0.5 * mae
        
        np.testing.assert_almost_equal(loss, expected)
    
    def test_pinball_loss_asymmetric(self):
        """Test asymmetric behavior of pinball loss."""
        y_true = np.array([10.0])
        
        # Under-prediction should be penalized more at high quantile
        y_pred_under = np.array([8.0])  # Under by 2
        y_pred_over = np.array([12.0])  # Over by 2
        
        loss_under_q90 = pinball_loss(y_true, y_pred_under, 0.9)
        loss_over_q90 = pinball_loss(y_true, y_pred_over, 0.9)
        
        # At q=0.9, under-prediction is penalized more
        assert loss_under_q90 > loss_over_q90
        
        # Verify exact values
        # Under: 0.9 * (10 - 8) = 1.8
        # Over: 0.1 * (12 - 10) = 0.2
        np.testing.assert_almost_equal(loss_under_q90, 1.8)
        np.testing.assert_almost_equal(loss_over_q90, 0.2)
    
    def test_pinball_loss_zero(self):
        """Test pinball loss with perfect predictions."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1, 2, 3])
        
        for q in [0.1, 0.5, 0.9]:
            loss = pinball_loss(y_true, y_pred, q)
            np.testing.assert_almost_equal(loss, 0.0)


class TestIntervalMetrics:
    """Tests for interval coverage and width metrics."""
    
    def test_interval_coverage(self):
        """Test interval coverage computation."""
        y_true = np.array([1, 2, 3, 4, 5])
        lower = np.array([0, 1.5, 2, 3.5, 4])
        upper = np.array([2, 3, 4, 5, 6])
        
        coverage = interval_coverage(y_true, lower, upper)
        
        # All values should be within bounds
        assert coverage == 1.0
        
        # Test partial coverage
        lower2 = np.array([2, 3, 4, 5, 6])  # None covered
        upper2 = np.array([0.5, 1, 2, 3, 4])
        
        coverage2 = interval_coverage(y_true, lower2, upper2)
        assert coverage2 == 0.0
    
    def test_interval_width(self):
        """Test interval width computation."""
        lower = np.array([0, 1, 2])
        upper = np.array([1, 3, 5])
        
        width = interval_width(lower, upper)
        
        # Widths: [1, 2, 3], mean = 2
        np.testing.assert_almost_equal(width, 2.0)
    
    def test_coverage_boundaries(self):
        """Test coverage at interval boundaries."""
        y_true = np.array([1.0, 2.0])
        lower = np.array([1.0, 2.0])  # Exactly at boundary
        upper = np.array([1.0, 2.0])
        
        coverage = interval_coverage(y_true, lower, upper)
        assert coverage == 1.0  # Boundaries should be inclusive


class TestIntegration:
    """Integration tests combining QRF with utility functions."""
    
    def test_full_workflow(self, regression_data):
        """Test complete workflow with all QRF methods and metrics."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Fit QRF
        qrf = QuantileRegressorForest(
            n_estimators=100,
            quantiles=[0.1, 0.5, 0.9],
            random_state=42,
        )
        qrf.fit(X_train, y_train)
        
        # Get predictions
        q_pred = qrf.predict(X_test)
        lower, upper = q_pred[:, 0], q_pred[:, 2]
        median = q_pred[:, 1]
        
        # Compute metrics
        cov = interval_coverage(y_test, lower, upper)
        width = interval_width(lower, upper)
        loss = pinball_loss(y_test, median, 0.5)
        
        # Assertions
        assert 0.5 < cov < 1.0  # Should have reasonable coverage
        assert width > 0  # Width should be positive
        assert loss >= 0  # Loss should be non-negative
        
        # Test uncertainty quantification
        std = qrf.predict_std(X_test)
        mean = qrf.predict_mean(X_test)
        
        # Higher uncertainty should correlate with wider intervals
        correlation = np.corrcoef(std, upper - lower)[0, 1]
        assert correlation > 0.5  # Should be positively correlated
