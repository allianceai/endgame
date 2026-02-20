"""Tests for eg.quick module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


class TestQuickAPI:
    """Test quick API functions."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=0.1,
            random_state=42,
        )
        return X, y

    def test_presets_loaded(self):
        """Test that presets are available."""
        from endgame.quick import PRESETS

        assert "fast" in PRESETS
        assert "default" in PRESETS
        assert "competition" in PRESETS
        assert "interpretable" in PRESETS

    def test_classify_basic(self, classification_data):
        """Test basic classification."""
        from endgame.quick import classify, QuickResult

        X, y = classification_data

        # Use fast preset with minimal config
        result = classify(X, y, preset="fast", cv_folds=2, verbose=False)

        assert isinstance(result, QuickResult)
        assert result.cv_score > 0.5  # Better than random
        assert result.oof_predictions is not None
        assert len(result.oof_predictions) == len(y)
        assert result.metric == "roc_auc"

    def test_regress_basic(self, regression_data):
        """Test basic regression."""
        from endgame.quick import regress, QuickResult

        X, y = regression_data

        # Use fast preset with minimal config
        result = regress(X, y, preset="fast", cv_folds=2, verbose=False)

        assert isinstance(result, QuickResult)
        assert result.oof_predictions is not None
        assert len(result.oof_predictions) == len(y)
        assert result.metric == "rmse"

    def test_classify_different_metrics(self, classification_data):
        """Test classification with different metrics."""
        from endgame.quick import classify

        X, y = classification_data

        # Test accuracy metric
        result = classify(X, y, preset="fast", metric="accuracy", cv_folds=2, verbose=False)
        assert result.metric == "accuracy"
        assert 0 <= result.cv_score <= 1

        # Test f1 metric
        result = classify(X, y, preset="fast", metric="f1", cv_folds=2, verbose=False)
        assert result.metric == "f1"
        assert 0 <= result.cv_score <= 1

    def test_regress_different_metrics(self, regression_data):
        """Test regression with different metrics."""
        from endgame.quick import regress

        X, y = regression_data

        # Test R2 metric
        result = regress(X, y, preset="fast", metric="r2", cv_folds=2, verbose=False)
        assert result.metric == "r2"

        # Test MAE metric
        result = regress(X, y, preset="fast", metric="mae", cv_folds=2, verbose=False)
        assert result.metric == "mae"
        assert result.cv_score >= 0


class TestKNNBaselines:
    """Test KNN baseline models."""

    @pytest.fixture
    def data(self):
        """Generate test data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y

    def test_knn_classifier_basic(self, data):
        """Test KNNClassifier basic functionality."""
        from endgame.models.baselines import KNNClassifier

        X, y = data
        clf = KNNClassifier(n_neighbors=3, n_jobs=1)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert clf.n_features_in_ == 10

        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 2)

        preds = clf.predict(X[:5])
        assert len(preds) == 5

    def test_knn_classifier_scaling(self, data):
        """Test KNNClassifier with and without scaling."""
        from endgame.models.baselines import KNNClassifier

        X, y = data

        # With scaling (default)
        clf_scaled = KNNClassifier(n_neighbors=3, scale_features=True, n_jobs=1)
        clf_scaled.fit(X, y)

        # Without scaling
        clf_unscaled = KNNClassifier(n_neighbors=3, scale_features=False, n_jobs=1)
        clf_unscaled.fit(X, y)

        # Both should work
        assert clf_scaled.predict(X[:5]).shape == (5,)
        assert clf_unscaled.predict(X[:5]).shape == (5,)

    def test_knn_regressor_basic(self):
        """Test KNNRegressor basic functionality."""
        from endgame.models.baselines import KNNRegressor

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        reg = KNNRegressor(n_neighbors=5, n_jobs=1)
        reg.fit(X, y)

        assert reg.n_features_in_ == 10

        preds = reg.predict(X[:5])
        assert preds.shape == (5,)


class TestLinearBaselines:
    """Test Linear baseline models."""

    @pytest.fixture
    def clf_data(self):
        """Generate classification data."""
        X, y = make_classification(n_samples=100, n_features=10, random_state=42)
        return X, y

    @pytest.fixture
    def reg_data(self):
        """Generate regression data."""
        X, y = make_regression(n_samples=100, n_features=10, random_state=42)
        return X, y

    def test_linear_classifier_l2(self, clf_data):
        """Test LinearClassifier with L2 penalty."""
        from endgame.models.baselines import LinearClassifier

        X, y = clf_data
        clf = LinearClassifier(penalty="l2", n_jobs=1)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert clf.n_features_in_ == 10
        assert hasattr(clf, "coef_")

        proba = clf.predict_proba(X[:5])
        assert proba.shape == (5, 2)

    def test_linear_classifier_l1(self, clf_data):
        """Test LinearClassifier with L1 penalty."""
        from endgame.models.baselines import LinearClassifier

        X, y = clf_data
        clf = LinearClassifier(penalty="l1", n_jobs=1)
        clf.fit(X, y)

        assert hasattr(clf, "coef_")
        preds = clf.predict(X[:5])
        assert len(preds) == 5

    def test_linear_regressor_l2(self, reg_data):
        """Test LinearRegressor with L2 (Ridge) penalty."""
        from endgame.models.baselines import LinearRegressor

        X, y = reg_data
        reg = LinearRegressor(penalty="l2")
        reg.fit(X, y)

        assert reg.n_features_in_ == 10
        assert hasattr(reg, "coef_")

        preds = reg.predict(X[:5])
        assert preds.shape == (5,)

    def test_linear_regressor_l1(self, reg_data):
        """Test LinearRegressor with L1 (Lasso) penalty."""
        from endgame.models.baselines import LinearRegressor

        X, y = reg_data
        reg = LinearRegressor(penalty="l1")
        reg.fit(X, y)

        preds = reg.predict(X[:5])
        assert preds.shape == (5,)

    def test_linear_regressor_elasticnet(self, reg_data):
        """Test LinearRegressor with ElasticNet penalty."""
        from endgame.models.baselines import LinearRegressor

        X, y = reg_data
        reg = LinearRegressor(penalty="elasticnet", l1_ratio=0.5)
        reg.fit(X, y)

        preds = reg.predict(X[:5])
        assert preds.shape == (5,)

    def test_feature_importances(self, clf_data):
        """Test feature_importances_ property."""
        from endgame.models.baselines import LinearClassifier

        X, y = clf_data
        clf = LinearClassifier(penalty="l2", n_jobs=1)
        clf.fit(X, y)

        importances = clf.feature_importances_
        assert len(importances) == 10
        assert all(v >= 0 for v in importances)
