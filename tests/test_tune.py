"""Tests for tune module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


class TestOptunaOptimizer:
    """Tests for OptunaOptimizer."""

    def test_optimize_lightgbm(self):
        """Test LightGBM hyperparameter optimization."""
        pytest.importorskip("optuna")
        lgbm = pytest.importorskip("lightgbm")
        from endgame.tune import OptunaOptimizer

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        estimator = lgbm.LGBMClassifier(verbose=-1)
        optimizer = OptunaOptimizer(
            estimator=estimator,
            param_space="lgbm_standard",
            n_trials=5,
            cv=3,
            random_state=42,
            verbose=False,
        )
        result = optimizer.optimize(X, y)

        assert result.best_params is not None
        assert "learning_rate" in result.best_params
        assert result.best_score > 0

    def test_optimize_xgboost(self):
        """Test XGBoost hyperparameter optimization."""
        pytest.importorskip("optuna")
        xgb = pytest.importorskip("xgboost")
        from endgame.tune import OptunaOptimizer

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        estimator = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
        optimizer = OptunaOptimizer(
            estimator=estimator,
            param_space="xgb_standard",
            n_trials=5,
            cv=3,
            random_state=42,
            verbose=False,
        )
        result = optimizer.optimize(X, y)

        assert result.best_params is not None
        assert result.best_score > 0

    def test_custom_search_space(self):
        """Test with custom search space."""
        pytest.importorskip("optuna")
        lgbm = pytest.importorskip("lightgbm")
        from endgame.tune import OptunaOptimizer

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        custom_space = {
            "learning_rate": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "num_leaves": {"type": "int", "low": 10, "high": 50},
        }

        estimator = lgbm.LGBMClassifier(verbose=-1)
        optimizer = OptunaOptimizer(
            estimator=estimator,
            param_space=custom_space,
            n_trials=3,
            random_state=42,
            verbose=False,
        )
        result = optimizer.optimize(X, y)

        assert "learning_rate" in result.best_params
        assert 0.01 <= result.best_params["learning_rate"] <= 0.1


class TestSearchSpaces:
    """Tests for search space utilities."""

    def test_get_lgbm_space(self):
        """Test LightGBM search space."""
        from endgame.tune.spaces import get_space

        space = get_space("lgbm_standard")

        assert "learning_rate" in space
        assert "num_leaves" in space
        assert "max_depth" in space

    def test_get_xgb_space(self):
        """Test XGBoost search space."""
        from endgame.tune.spaces import get_space

        space = get_space("xgb_standard")

        assert "learning_rate" in space
        assert "max_depth" in space

    def test_get_catboost_space(self):
        """Test CatBoost search space."""
        from endgame.tune.spaces import get_space

        space = get_space("catboost_standard")

        assert "learning_rate" in space
        assert "depth" in space
