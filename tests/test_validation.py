"""Tests for validation module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


class TestAdversarialValidator:
    """Tests for AdversarialValidator."""

    def test_check_drift_no_drift(self):
        """Test with identical distributions (no drift)."""
        from endgame.validation import AdversarialValidator

        # Same distribution
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)
        X_test = rng.randn(50, 5)

        av = AdversarialValidator(random_state=42)
        result = av.check_drift(X_train, X_test)

        # AUC should be close to 0.5 (random)
        assert 0.4 <= result.auc_score <= 0.6
        assert result.drift_severity in ["none", "mild"]

    def test_check_drift_with_drift(self):
        """Test with different distributions (drift)."""
        from endgame.validation import AdversarialValidator

        # Different distributions
        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 5)
        X_test = rng.randn(50, 5) + 2  # Shifted distribution

        av = AdversarialValidator(random_state=42)
        result = av.check_drift(X_train, X_test)

        # AUC should be high
        assert result.auc_score > 0.7
        assert result.drift_severity in ["moderate", "severe"]

    def test_feature_importances(self):
        """Test feature importance extraction."""
        from endgame.validation import AdversarialValidator

        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 3)
        X_test = rng.randn(50, 3)
        X_test[:, 0] += 5  # Drift in first feature only

        av = AdversarialValidator(random_state=42)
        result = av.check_drift(X_train, X_test)

        # Features should be ranked by importance
        assert len(result.feature_importances) == 3
        # Feature 0 has the drift, so it should be in top-2 drifted features
        assert "feature_0" in result.drifted_features[:2]


class TestCVSplitters:
    """Tests for cross-validation splitters."""

    def test_purged_time_series_split(self):
        """Test PurgedTimeSeriesSplit."""
        from endgame.validation import PurgedTimeSeriesSplit

        X = np.arange(100).reshape(-1, 1)
        y = np.zeros(100)

        cv = PurgedTimeSeriesSplit(n_splits=3, purge_gap=5)

        folds = list(cv.split(X, y))
        assert len(folds) == 3

        for train_idx, val_idx in folds:
            # Training should end before validation with gap
            assert train_idx.max() < val_idx.min() - 5

    def test_stratified_group_kfold(self):
        """Test StratifiedGroupKFold."""
        from endgame.validation import StratifiedGroupKFold

        X = np.random.randn(100, 5)
        y = np.array([0] * 50 + [1] * 50)
        groups = np.repeat(np.arange(20), 5)  # 20 groups, 5 samples each

        cv = StratifiedGroupKFold(n_splits=5, random_state=42)

        for train_idx, val_idx in cv.split(X, y, groups):
            # No group should appear in both train and val
            train_groups = set(groups[train_idx])
            val_groups = set(groups[val_idx])
            assert len(train_groups & val_groups) == 0

    def test_multilabel_stratified_kfold(self):
        """Test MultilabelStratifiedKFold."""
        from endgame.validation import MultilabelStratifiedKFold

        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, size=(100, 3))  # 3 labels

        cv = MultilabelStratifiedKFold(n_splits=5, random_state=42)

        folds = list(cv.split(X, y))
        assert len(folds) == 5

        for train_idx, val_idx in folds:
            assert len(train_idx) > 0
            assert len(val_idx) > 0
            # Indices should not overlap
            assert len(set(train_idx) & set(val_idx)) == 0


class TestCVUtils:
    """Tests for CV utilities."""

    def test_cross_validate_oof(self):
        """Test cross_validate_oof function."""
        from endgame.validation import cross_validate_oof
        from sklearn.ensemble import RandomForestClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        result = cross_validate_oof(model, X, y, cv=3, verbose=True)

        assert len(result.oof_predictions) == 100
        assert len(result.fold_scores) == 3
        assert 0 <= result.mean_score <= 1
        assert len(result.models) == 3

    def test_cross_validate_oof_regression(self):
        """Test cross_validate_oof for regression."""
        from endgame.validation import cross_validate_oof
        from sklearn.ensemble import RandomForestRegressor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = X[:, 0] + rng.randn(100) * 0.1

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = cross_validate_oof(model, X, y, cv=3, scoring="neg_mean_squared_error")

        assert len(result.oof_predictions) == 100
        assert len(result.fold_scores) == 3
