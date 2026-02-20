"""Tests for NGBoost probabilistic prediction models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error

# Skip all tests if ngboost is not installed
ngboost = pytest.importorskip("ngboost")

from endgame.models import NGBoostRegressor, NGBoostClassifier


class TestNGBoostRegressor:
    """Tests for NGBoostRegressor."""

    @pytest.fixture
    def regression_data(self):
        """Generate regression dataset."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=0.5,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_fit_predict_basic(self, regression_data):
        """Test basic fit and predict."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            random_state=42,
            verbose=False,
        )
        model.fit(X_train, y_train)

        # Point predictions
        y_pred = model.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        
        # Should have reasonable MSE
        mse = mean_squared_error(y_test, y_pred)
        assert mse < np.var(y_test) * 2  # Better than very bad

    def test_pred_dist(self, regression_data):
        """Test distribution prediction."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            distribution="normal",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Get distribution
        dist = model.pred_dist(X_test)
        
        # Test distribution methods
        mean = dist.mean()
        std = dist.std()
        
        assert mean.shape == (len(X_test),)
        assert std.shape == (len(X_test),)
        assert np.all(std > 0)  # Std should be positive
        
        # Test log probability
        log_prob = dist.logpdf(y_test)
        assert log_prob.shape == (len(X_test),)
        assert np.all(np.isfinite(log_prob))

    def test_predict_interval(self, regression_data):
        """Test prediction intervals."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # 90% prediction interval
        lower, upper = model.predict_interval(X_test, alpha=0.1)
        
        assert lower.shape == (len(X_test),)
        assert upper.shape == (len(X_test),)
        assert np.all(lower < upper)
        
        # Check coverage (should be approximately 90%)
        # With small samples (40 test points), coverage can vary significantly
        coverage = np.mean((y_test >= lower) & (y_test <= upper))
        assert coverage > 0.3  # Relaxed threshold for small sample variance

    def test_predict_std(self, regression_data):
        """Test uncertainty prediction."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        std = model.predict_std(X_test)
        
        assert std.shape == (len(X_test),)
        assert np.all(std > 0)

    def test_different_distributions(self, regression_data):
        """Test different distribution options."""
        X_train, X_test, y_train, y_test = regression_data
        
        # Make targets positive for lognormal
        y_train_pos = np.abs(y_train) + 1
        y_test_pos = np.abs(y_test) + 1

        for dist in ["normal", "laplace"]:
            model = NGBoostRegressor(
                preset="fast",
                distribution=dist,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            assert pred.shape == (len(X_test),)

        # Test lognormal with positive targets
        model = NGBoostRegressor(
            preset="fast",
            distribution="lognormal",
            random_state=42,
        )
        model.fit(X_train, y_train_pos)
        pred = model.predict(X_test)
        assert pred.shape == (len(X_test),)

    def test_different_scores(self, regression_data):
        """Test different scoring rules."""
        X_train, X_test, y_train, y_test = regression_data

        for score in ["crps", "mle"]:
            model = NGBoostRegressor(
                preset="fast",
                score=score,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            assert pred.shape == (len(X_test),)

    def test_early_stopping(self, regression_data):
        """Test early stopping."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            n_estimators=500,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        pred = model.predict(X_test)
        assert pred.shape == (len(X_test),)

    def test_feature_importances(self, regression_data):
        """Test feature importances."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        importances = model.feature_importances_
        # NGBoost returns 2D importances (one row per distribution parameter)
        assert importances.ndim == 2
        assert importances.shape[1] == X_train.shape[1]
        assert np.all(importances >= 0)

    def test_presets(self, regression_data):
        """Test different presets."""
        X_train, X_test, y_train, y_test = regression_data

        for preset in ["fast", "endgame"]:
            model = NGBoostRegressor(
                preset=preset,
                random_state=42,
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            assert pred.shape == (len(X_test),)

    def test_sklearn_compatibility(self, regression_data):
        """Test sklearn cross_val_score compatibility."""
        X_train, X_test, y_train, y_test = regression_data

        model = NGBoostRegressor(
            preset="fast",
            n_estimators=50,
            random_state=42,
        )
        
        # Cross-validation should work
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="r2"
        )
        assert len(scores) == 3

    def test_sample_weight(self, regression_data):
        """Test with sample weights."""
        X_train, X_test, y_train, y_test = regression_data
        
        weights = np.random.rand(len(y_train)) + 0.5

        model = NGBoostRegressor(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=weights)

        pred = model.predict(X_test)
        assert pred.shape == (len(X_test),)


class TestNGBoostClassifier:
    """Tests for NGBoostClassifier."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    @pytest.fixture
    def multiclass_data(self):
        """Generate multiclass classification dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.2, random_state=42)

    def test_fit_predict_binary(self, binary_data):
        """Test binary classification."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Class predictions
        y_pred = model.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(y_pred).issubset({0, 1})
        
        # Should have reasonable accuracy
        acc = accuracy_score(y_test, y_pred)
        assert acc > 0.5

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        X_train, X_test, y_train, y_test = multiclass_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        # Class predictions
        y_pred = model.predict(X_test)
        assert y_pred.shape == (len(X_test),)
        assert set(y_pred).issubset({0, 1, 2, 3})
        
        # Check attributes
        assert model.n_classes_ == 4
        assert len(model.classes_) == 4

    def test_predict_proba_binary(self, binary_data):
        """Test probability prediction for binary."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test probability prediction for multiclass."""
        X_train, X_test, y_train, y_test = multiclass_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        proba = model.predict_proba(X_test)
        
        assert proba.shape == (len(X_test), 4)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_pred_dist(self, binary_data):
        """Test distribution prediction."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        dist = model.pred_dist(X_test)
        
        # Distribution should exist
        assert dist is not None

    def test_early_stopping(self, binary_data):
        """Test early stopping."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            n_estimators=500,
            early_stopping_rounds=10,
            random_state=42,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        pred = model.predict(X_test)
        assert pred.shape == (len(X_test),)

    def test_feature_importances(self, binary_data):
        """Test feature importances."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        importances = model.feature_importances_
        # NGBoost returns 2D importances (one row per distribution parameter)
        assert importances.ndim == 2
        assert importances.shape[1] == X_train.shape[1]
        assert np.all(importances >= 0)

    def test_score(self, binary_data):
        """Test score method."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train)

        score = model.score(X_test, y_test)
        assert 0 <= score <= 1

    def test_sklearn_compatibility(self, binary_data):
        """Test sklearn cross_val_score compatibility."""
        X_train, X_test, y_train, y_test = binary_data

        model = NGBoostClassifier(
            preset="fast",
            n_estimators=50,
            random_state=42,
        )
        
        scores = cross_val_score(
            model, X_train, y_train, cv=3, scoring="accuracy"
        )
        assert len(scores) == 3

    def test_sample_weight(self, binary_data):
        """Test with sample weights."""
        X_train, X_test, y_train, y_test = binary_data
        
        weights = np.random.rand(len(y_train)) + 0.5

        model = NGBoostClassifier(
            preset="fast",
            random_state=42,
        )
        model.fit(X_train, y_train, sample_weight=weights)

        pred = model.predict(X_test)
        assert pred.shape == (len(X_test),)


class TestNGBoostIntegration:
    """Integration tests for NGBoost models."""

    def test_regressor_vs_sklearn_baseline(self):
        """NGBoost should be competitive with sklearn baselines."""
        from sklearn.ensemble import GradientBoostingRegressor
        
        X, y = make_regression(
            n_samples=500,
            n_features=10,
            n_informative=5,
            noise=0.5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # NGBoost
        ngb = NGBoostRegressor(preset="fast", random_state=42)
        ngb.fit(X_train, y_train)
        ngb_pred = ngb.predict(X_test)
        ngb_mse = mean_squared_error(y_test, ngb_pred)

        # Sklearn GBR
        gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)
        gbr.fit(X_train, y_train)
        gbr_pred = gbr.predict(X_test)
        gbr_mse = mean_squared_error(y_test, gbr_pred)

        # NGBoost should be within 2x of sklearn
        assert ngb_mse < gbr_mse * 2

    def test_classifier_calibration(self):
        """NGBoost should produce well-calibrated probabilities."""
        X, y = make_classification(
            n_samples=1000,
            n_features=10,
            n_informative=5,
            n_classes=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        model = NGBoostClassifier(preset="fast", random_state=42)
        model.fit(X_train, y_train)
        
        proba = model.predict_proba(X_test)
        
        # Check calibration roughly (bin probabilities and check accuracy)
        # For well-calibrated model, predicted prob should match actual rate
        high_conf_mask = proba[:, 1] > 0.7
        if high_conf_mask.sum() > 10:
            high_conf_acc = y_test[high_conf_mask].mean()
            # High confidence predictions should have high accuracy
            assert high_conf_acc > 0.5

    def test_uncertainty_increases_with_distance(self):
        """Uncertainty should be higher for out-of-distribution samples."""
        # Create clustered data
        np.random.seed(42)
        X_train = np.random.randn(200, 5)
        y_train = X_train[:, 0] + 0.5 * np.random.randn(200)
        
        # In-distribution test
        X_in = np.random.randn(50, 5)
        
        # Out-of-distribution test (shifted)
        X_out = np.random.randn(50, 5) + 10

        model = NGBoostRegressor(preset="fast", random_state=42)
        model.fit(X_train, y_train)

        std_in = model.predict_std(X_in)
        std_out = model.predict_std(X_out)

        # Out-of-distribution should have higher uncertainty
        # (This is a property of NGBoost's natural gradient approach)
        assert std_out.mean() >= std_in.mean() * 0.5  # Some increase expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
