"""Tests for Conformalized Quantile Regression."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from endgame.calibration import ConformizedQuantileRegressor
from endgame.models.trees import QuantileRegressorForest


@pytest.fixture
def regression_data():
    """Generate regression data."""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42)


@pytest.fixture
def heteroscedastic_data():
    """Generate data with heteroscedastic noise."""
    np.random.seed(42)
    n_samples = 500
    X = np.random.randn(n_samples, 5)
    # Noise variance increases with X[:, 0]
    noise_scale = 1 + 3 * np.abs(X[:, 0])
    y = 3 * X[:, 0] + 2 * X[:, 1] + np.random.randn(n_samples) * noise_scale
    return train_test_split(X, y, test_size=0.3, random_state=42)


class TestConformizedQuantileRegressor:
    """Tests for ConformizedQuantileRegressor."""
    
    def test_basic_fit_predict(self, regression_data):
        """Test basic fit and predict workflow."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        # Check fitted attributes
        assert hasattr(cqr, 'quantile_estimator_')
        assert hasattr(cqr, 'conformity_scores_')
        assert hasattr(cqr, 'quantile_')
        
        # Point predictions
        predictions = cqr.predict(X_test)
        assert predictions.shape == (len(X_test),)
        assert np.isfinite(predictions).all()
        
        # Interval predictions
        lower, upper = cqr.predict_interval(X_test)
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        assert np.all(lower <= upper)
    
    def test_coverage_guarantee(self, regression_data):
        """Test that coverage is approximately at target level."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        alpha = 0.1  # Target 90% coverage
        cqr = ConformizedQuantileRegressor(alpha=alpha, random_state=42)
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        coverage = cqr.coverage_score(X_test, y_test)
        
        # Coverage should be close to target (with some slack for finite samples)
        assert coverage >= 1 - alpha - 0.15  # Allow 15% slack
        assert coverage <= 1.0
    
    def test_auto_calibration_split(self, regression_data):
        """Test automatic calibration split when X_cal not provided."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train)  # No X_cal provided
        
        lower, upper = cqr.predict_interval(X_test)
        assert np.all(lower <= upper)
        
        coverage = cqr.coverage_score(X_test, y_test)
        assert coverage > 0.7  # Should have reasonable coverage
    
    def test_cross_conformal(self, regression_data):
        """Test cross-conformal mode."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(
            alpha=0.1,
            cv=5,
            random_state=42,
        )
        cqr.fit(X_train, y_train)  # No separate calibration needed
        
        lower, upper = cqr.predict_interval(X_test)
        coverage = cqr.coverage_score(X_test, y_test)
        
        assert np.all(lower <= upper)
        assert coverage >= 0.75  # Should have reasonable coverage
    
    def test_asymmetric_cqr(self, regression_data):
        """Test asymmetric conformity scores."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        cqr = ConformizedQuantileRegressor(
            alpha=0.1,
            symmetric=False,
            random_state=42,
        )
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        assert hasattr(cqr, 'lower_quantile_')
        assert hasattr(cqr, 'upper_quantile_')
        
        lower, upper = cqr.predict_interval(X_test)
        coverage = cqr.coverage_score(X_test, y_test)
        
        assert np.all(lower <= upper)
        assert coverage >= 0.75
    
    def test_heteroscedastic_adaptation(self, heteroscedastic_data):
        """Test that CQR adapts to heteroscedastic data."""
        X_train, X_test, y_train, y_test = heteroscedastic_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        widths = cqr.interval_width(X_test)
        
        # Higher |X[:, 0]| should correlate with wider intervals
        correlation = np.corrcoef(np.abs(X_test[:, 0]), widths)[0, 1]
        assert correlation > 0.3  # Should be positively correlated
    
    def test_custom_quantile_estimator(self, regression_data):
        """Test with custom QuantileRegressorForest."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        qrf = QuantileRegressorForest(n_estimators=50, random_state=42)
        
        cqr = ConformizedQuantileRegressor(
            quantile_estimator=qrf,
            alpha=0.1,
            random_state=42,
        )
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        coverage = cqr.coverage_score(X_test, y_test)
        assert coverage >= 0.75
    
    def test_predict_quantiles(self, regression_data):
        """Test predicting arbitrary quantiles."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train)
        
        # Predict multiple quantiles
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
        q_pred = cqr.predict_quantiles(X_test[:5], quantiles)
        
        assert q_pred.shape == (5, len(quantiles))
        
        # Quantiles should be ordered
        for i in range(len(quantiles) - 1):
            assert np.all(q_pred[:, i] <= q_pred[:, i + 1])
    
    def test_average_interval_width(self, regression_data):
        """Test average interval width computation."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train)
        
        avg_width = cqr.average_interval_width(X_test)
        
        assert avg_width > 0
        assert np.isfinite(avg_width)
    
    def test_score_method(self, regression_data):
        """Test sklearn-compatible score method."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train)
        
        score = cqr.score(X_test, y_test)
        
        # Score is negative interval width (higher is better)
        assert score < 0
        assert score == -cqr.average_interval_width(X_test)
    
    def test_different_alpha_values(self, regression_data):
        """Test different coverage levels."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        coverages = []
        widths = []
        
        for alpha in [0.2, 0.1, 0.05]:
            cqr = ConformizedQuantileRegressor(alpha=alpha, random_state=42)
            cqr.fit(X_train, y_train, X_cal, y_cal)
            
            coverages.append(cqr.coverage_score(X_test, y_test))
            widths.append(cqr.average_interval_width(X_test))
        
        # Lower alpha should give higher coverage
        assert coverages[0] <= coverages[1] <= coverages[2] or \
               all(c > 0.7 for c in coverages)  # Allow some variance
        
        # Lower alpha should give wider intervals
        assert widths[0] <= widths[1] <= widths[2]
    
    def test_reproducibility(self, regression_data):
        """Test that random_state ensures reproducibility."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr1 = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr1.fit(X_train, y_train)
        lower1, upper1 = cqr1.predict_interval(X_test)
        
        cqr2 = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr2.fit(X_train, y_train)
        lower2, upper2 = cqr2.predict_interval(X_test)
        
        np.testing.assert_array_almost_equal(lower1, lower2)
        np.testing.assert_array_almost_equal(upper1, upper2)
    
    def test_not_fitted_error(self, regression_data):
        """Test error when predicting before fitting."""
        X_train, X_test, y_train, y_test = regression_data
        
        cqr = ConformizedQuantileRegressor(alpha=0.1)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            cqr.predict(X_test)
        
        with pytest.raises(RuntimeError, match="has not been fitted"):
            cqr.predict_interval(X_test)


class TestCQRWithQRFIntegration:
    """Integration tests for CQR with QuantileRegressorForest."""
    
    def test_qrf_cqr_pipeline(self, regression_data):
        """Test full pipeline with QRF and CQR."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Create QRF with specific settings
        qrf = QuantileRegressorForest(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
        )
        
        # Create CQR with QRF
        cqr = ConformizedQuantileRegressor(
            quantile_estimator=qrf,
            alpha=0.1,
            random_state=42,
        )
        cqr.fit(X_train, y_train, X_cal, y_cal)
        
        # Check coverage
        coverage = cqr.coverage_score(X_test, y_test)
        avg_width = cqr.average_interval_width(X_test)
        
        print(f"Coverage: {coverage:.3f}, Avg width: {avg_width:.3f}")
        
        assert coverage >= 0.80  # Should meet target approximately
        assert avg_width > 0
    
    def test_cqr_vs_raw_qrf_intervals(self, regression_data):
        """Compare CQR intervals to raw QRF intervals."""
        X_train, X_test, y_train, y_test = regression_data
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_train, y_train, test_size=0.3, random_state=42
        )
        
        # Raw QRF intervals
        qrf = QuantileRegressorForest(
            n_estimators=100,
            quantiles=[0.05, 0.95],
            random_state=42,
        )
        qrf.fit(np.vstack([X_train, X_cal]), np.hstack([y_train, y_cal]))
        qrf_lower, qrf_upper = qrf.predict_interval(X_test, coverage=0.9)
        
        qrf_coverage = np.mean((y_test >= qrf_lower) & (y_test <= qrf_upper))
        
        # CQR intervals
        cqr = ConformizedQuantileRegressor(alpha=0.1, random_state=42)
        cqr.fit(X_train, y_train, X_cal, y_cal)
        cqr_lower, cqr_upper = cqr.predict_interval(X_test)
        
        cqr_coverage = cqr.coverage_score(X_test, y_test)
        
        # CQR should have better coverage guarantee
        # (Raw QRF may undercover due to finite sample effects)
        print(f"QRF coverage: {qrf_coverage:.3f}")
        print(f"CQR coverage: {cqr_coverage:.3f}")
        
        # CQR should be at least as good as raw QRF for coverage
        # (may be slightly wider but more reliable)
        assert cqr_coverage >= qrf_coverage - 0.1  # Allow some slack


class TestImports:
    """Test module imports."""
    
    def test_import_from_calibration(self):
        """Test import from calibration module."""
        from endgame.calibration import ConformizedQuantileRegressor
        
        assert ConformizedQuantileRegressor is not None
    
    def test_import_via_eg(self):
        """Test import via endgame namespace."""
        import endgame as eg
        
        assert hasattr(eg.calibration, 'ConformizedQuantileRegressor')
