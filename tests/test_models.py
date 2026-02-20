"""Tests for models module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


class TestGBDTWrapper:
    """Tests for GBDT wrappers."""

    def test_lgbm_classifier(self):
        """Test LightGBM classifier wrapper."""
        pytest.importorskip("lightgbm")
        from endgame.models import LGBMWrapper

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = LGBMWrapper(
            preset="fast",
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict_proba(X)

        assert preds.shape == (100, 2)
        assert model.feature_importances_ is not None

    def test_lgbm_regressor(self):
        """Test LightGBM regressor wrapper."""
        pytest.importorskip("lightgbm")
        from endgame.models import LGBMWrapper

        X, y = make_regression(n_samples=100, n_features=5, random_state=42)

        model = LGBMWrapper(
            preset="fast",
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 100

    def test_xgb_classifier(self):
        """Test XGBoost classifier wrapper."""
        pytest.importorskip("xgboost")
        from endgame.models import XGBWrapper

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = XGBWrapper(
            preset="fast",
            task="classification",
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict_proba(X)

        assert preds.shape == (100, 2)

    def test_catboost_classifier(self):
        """Test CatBoost classifier wrapper."""
        pytest.importorskip("catboost")
        from endgame.models import CatBoostWrapper

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = CatBoostWrapper(
            preset="fast",
            task="classification",
            random_state=42,
            verbose=False,
        )
        model.fit(X, y)
        preds = model.predict_proba(X)

        assert preds.shape == (100, 2)

    def test_early_stopping(self):
        """Test early stopping with validation set."""
        pytest.importorskip("lightgbm")
        from endgame.models import LGBMWrapper

        X, y = make_classification(n_samples=200, n_features=5, random_state=42)
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        model = LGBMWrapper(
            preset="fast",
            task="classification",
            random_state=42,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])

        assert model.best_iteration_ > 0


class TestRotationForest:
    """Tests for Rotation Forest."""

    def test_rotation_forest_classifier(self):
        """Test Rotation Forest classifier."""
        from endgame.models.trees import RotationForestClassifier

        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        model = RotationForestClassifier(
            n_estimators=10,
            n_subsets=3,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)
        proba = model.predict_proba(X)

        assert len(preds) == 100
        assert proba.shape == (100, 2)

    def test_rotation_forest_regressor(self):
        """Test Rotation Forest regressor."""
        from endgame.models.trees import RotationForestRegressor

        X, y = make_regression(n_samples=100, n_features=10, random_state=42)

        model = RotationForestRegressor(
            n_estimators=10,
            n_subsets=3,
            random_state=42,
        )
        model.fit(X, y)
        preds = model.predict(X)

        assert len(preds) == 100

    def test_rotation_forest_pca_rotation(self):
        """Test that PCA rotation is applied."""
        from endgame.models.trees import RotationForestClassifier

        X, y = make_classification(n_samples=100, n_features=10, random_state=42)

        model = RotationForestClassifier(
            n_estimators=5,
            n_subsets=5,
            random_state=42,
        )
        model.fit(X, y)

        # Check rotation matrices were created
        assert len(model.rotation_matrices_) == 5
        for R in model.rotation_matrices_:
            assert R.shape == (10, 10)
