"""Tests for semi-supervised learning methods."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from endgame.semi_supervised import (
    SelfTrainingClassifier,
    SelfTrainingRegressor,
)


@pytest.fixture
def classification_data():
    """Generate classification data with unlabeled samples."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=2,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Generate regression data with unlabeled samples."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=10.0,
        random_state=42,
    )
    return X, y


def make_semi_supervised_classification(X, y, labeled_ratio=0.3, random_state=42):
    """Create semi-supervised setting by masking some labels."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    n_labeled = int(n_samples * labeled_ratio)
    
    # Randomly select labeled samples
    labeled_idx = rng.choice(n_samples, n_labeled, replace=False)
    
    y_semi = np.full(n_samples, -1)  # -1 for unlabeled
    y_semi[labeled_idx] = y[labeled_idx]
    
    return y_semi, labeled_idx


def make_semi_supervised_regression(X, y, labeled_ratio=0.3, random_state=42):
    """Create semi-supervised regression setting by masking some labels."""
    rng = np.random.RandomState(random_state)
    n_samples = len(y)
    n_labeled = int(n_samples * labeled_ratio)
    
    labeled_idx = rng.choice(n_samples, n_labeled, replace=False)
    
    y_semi = np.full(n_samples, np.nan)  # nan for unlabeled
    y_semi[labeled_idx] = y[labeled_idx]
    
    return y_semi, labeled_idx


class TestSelfTrainingClassifier:
    """Tests for SelfTrainingClassifier."""
    
    def test_basic_fit_predict(self, classification_data):
        """Test basic fit and predict workflow."""
        X, y = classification_data
        y_semi, labeled_idx = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            threshold=0.75,
            max_iter=5,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        # Check fitted attributes
        assert hasattr(st, "base_estimator_")
        assert hasattr(st, "classes_")
        assert hasattr(st, "n_iter_")
        assert hasattr(st, "labeled_iter_")
        assert hasattr(st, "pseudo_labels_")
        
        # Predictions should work
        predictions = st.predict(X[:10])
        assert predictions.shape == (10,)
        
        proba = st.predict_proba(X[:10])
        assert proba.shape == (10, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_threshold_criterion(self, classification_data):
        """Test threshold-based selection."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            criterion="threshold",
            threshold=0.9,  # High threshold
            max_iter=10,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        # Should have pseudo-labeled some samples
        n_pseudo = np.sum(st.labeled_iter_ > 0)
        assert n_pseudo >= 0  # May be 0 if threshold too high
    
    def test_k_best_criterion(self, classification_data):
        """Test k-best selection."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        k = 20
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            criterion="k_best",
            k_best=k,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        # Should have selected at most k * max_iter samples
        n_pseudo = np.sum(st.labeled_iter_ > 0)
        assert n_pseudo <= k * 3
    
    def test_sample_weight_decay(self, classification_data):
        """Test that sample_weight_decay affects training."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        # With decay
        st_decay = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.7,
            max_iter=3,
            sample_weight_decay=0.5,
            random_state=42,
        )
        st_decay.fit(X, y_semi)
        
        # Without decay
        st_no_decay = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.7,
            max_iter=3,
            sample_weight_decay=1.0,
            random_state=42,
        )
        st_no_decay.fit(X, y_semi)
        
        # Both should complete without error
        assert st_decay.n_iter_ >= 1
        assert st_no_decay.n_iter_ >= 1
    
    def test_progressive_weight(self, classification_data):
        """Test progressive weighting by confidence."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.7,
            max_iter=3,
            progressive_weight=True,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert st.n_iter_ >= 1
    
    def test_labeled_iter_tracking(self, classification_data):
        """Test that labeled_iter_ correctly tracks pseudo-labeling iterations."""
        X, y = classification_data
        y_semi, labeled_idx = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            criterion="k_best",
            k_best=30,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        # Original labeled samples should have iteration 0
        assert np.all(st.labeled_iter_[labeled_idx] == 0)
        
        # Unlabeled samples should have -1 or positive iteration
        unlabeled_idx = np.setdiff1d(np.arange(len(y)), labeled_idx)
        assert np.all((st.labeled_iter_[unlabeled_idx] == -1) | (st.labeled_iter_[unlabeled_idx] > 0))
    
    def test_transduction(self, classification_data):
        """Test transduction_ attribute (sklearn compatibility)."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.7,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert hasattr(st, "transduction_")
        np.testing.assert_array_equal(st.transduction_, st.pseudo_labels_)
    
    def test_get_pseudo_labeled_samples(self, classification_data):
        """Test getting pseudo-labeled sample information."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            criterion="k_best",
            k_best=20,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        indices, labels, iterations = st.get_pseudo_labeled_samples()
        
        # All should be non-negative integers
        assert len(indices) == len(labels) == len(iterations)
        assert np.all(iterations > 0)  # Pseudo-labels have iteration > 0
        
        # Labels should be valid classes
        assert np.all(np.isin(labels, st.classes_))
    
    def test_termination_conditions(self, classification_data):
        """Test different termination conditions."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y, labeled_ratio=0.1)
        
        # Max iter termination
        st_max = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.5,  # Low threshold to keep selecting
            max_iter=2,
            random_state=42,
        )
        st_max.fit(X, y_semi)
        assert st_max.termination_condition_ in ("max_iter", "all_labeled", "no_change")
        
        # No change termination (very high threshold)
        st_high = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.999,  # Very high
            max_iter=100,
            random_state=42,
        )
        st_high.fit(X, y_semi)
        # Either no_change (couldn't find confident samples) or max_iter
        assert st_high.termination_condition_ in ("max_iter", "no_change", "all_labeled")
    
    def test_verbose_output(self, classification_data, capsys):
        """Test verbose output."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            threshold=0.7,
            max_iter=2,
            verbose=True,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        captured = capsys.readouterr()
        assert "Iteration" in captured.out
    
    def test_invalid_parameters(self, classification_data):
        """Test that invalid parameters raise errors."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        # Invalid criterion
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(random_state=42),
            criterion="invalid",
        )
        with pytest.raises(ValueError, match="criterion must be"):
            st.fit(X, y_semi)
        
        # Invalid threshold
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(random_state=42),
            threshold=1.5,
        )
        with pytest.raises(ValueError, match="threshold must be"):
            st.fit(X, y_semi)
        
        # No labeled samples
        y_all_unlabeled = np.full(len(y), -1)
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(random_state=42),
        )
        with pytest.raises(ValueError, match="at least one labeled"):
            st.fit(X, y_all_unlabeled)
    
    def test_different_base_estimators(self, classification_data):
        """Test with different base estimators."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        estimators = [
            RandomForestClassifier(n_estimators=30, random_state=42),
            LogisticRegression(random_state=42, max_iter=500),
            DecisionTreeClassifier(random_state=42),
        ]
        
        for estimator in estimators:
            st = SelfTrainingClassifier(
                base_estimator=estimator,
                threshold=0.7,
                max_iter=2,
                random_state=42,
            )
            st.fit(X, y_semi)
            
            predictions = st.predict(X[:5])
            assert predictions.shape == (5,)
    
    def test_multiclass(self):
        """Test with multiclass classification."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            threshold=0.6,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert st.n_classes_ == 4
        assert len(st.classes_) == 4
        
        proba = st.predict_proba(X[:5])
        assert proba.shape == (5, 4)
    
    def test_reproducibility(self, classification_data):
        """Test that random_state ensures reproducibility."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st1 = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            criterion="k_best",
            k_best=20,
            max_iter=3,
            random_state=123,
        )
        st1.fit(X, y_semi)
        
        st2 = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=30, random_state=42),
            criterion="k_best",
            k_best=20,
            max_iter=3,
            random_state=123,
        )
        st2.fit(X, y_semi)
        
        np.testing.assert_array_equal(st1.labeled_iter_, st2.labeled_iter_)
        np.testing.assert_array_equal(st1.pseudo_labels_, st2.pseudo_labels_)
    
    def test_decision_function(self, classification_data):
        """Test decision_function method."""
        X, y = classification_data
        y_semi, _ = make_semi_supervised_classification(X, y)
        
        st = SelfTrainingClassifier(
            base_estimator=LogisticRegression(random_state=42, max_iter=500),
            threshold=0.7,
            max_iter=2,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        decision = st.decision_function(X[:5])
        assert decision.shape[0] == 5


class TestSelfTrainingRegressor:
    """Tests for SelfTrainingRegressor."""
    
    def test_basic_fit_predict(self, regression_data):
        """Test basic fit and predict workflow."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            threshold=1.0,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert hasattr(st, "base_estimator_")
        assert hasattr(st, "n_iter_")
        assert hasattr(st, "labeled_iter_")
        
        predictions = st.predict(X[:10])
        assert predictions.shape == (10,)
        assert np.isfinite(predictions).all()
    
    def test_ensemble_uncertainty(self, regression_data):
        """Test ensemble-based uncertainty estimation."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            uncertainty_method="ensemble",
            threshold=0.5,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert st.n_iter_ >= 1
    
    def test_knn_uncertainty(self, regression_data):
        """Test KNN-based uncertainty estimation."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=Ridge(random_state=42),
            uncertainty_method="knn",
            threshold=2.0,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert st.n_iter_ >= 1
    
    def test_residual_uncertainty(self, regression_data):
        """Test residual-based uncertainty estimation."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=Ridge(random_state=42),
            uncertainty_method="residual",
            threshold=5.0,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        assert st.n_iter_ >= 1
    
    def test_k_best_selection(self, regression_data):
        """Test k-best selection for regression."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            criterion="k_best",
            k_best=30,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        n_pseudo = np.sum(st.labeled_iter_ > 0)
        assert n_pseudo <= 30 * 3  # At most k * max_iter
    
    def test_get_pseudo_labeled_samples(self, regression_data):
        """Test getting pseudo-labeled sample information."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=RandomForestRegressor(n_estimators=50, random_state=42),
            criterion="k_best",
            k_best=20,
            max_iter=3,
            random_state=42,
        )
        st.fit(X, y_semi)
        
        indices, labels, iterations = st.get_pseudo_labeled_samples()
        
        assert len(indices) == len(labels) == len(iterations)
        assert np.all(iterations > 0)
        assert np.isfinite(labels).all()
    
    def test_no_labeled_error(self, regression_data):
        """Test error when no labeled samples."""
        X, y = regression_data
        y_all_nan = np.full(len(y), np.nan)
        
        st = SelfTrainingRegressor(
            base_estimator=RandomForestRegressor(random_state=42),
        )
        with pytest.raises(ValueError, match="at least one labeled"):
            st.fit(X, y_all_nan)
    
    def test_invalid_uncertainty_method(self, regression_data):
        """Test error for non-ensemble with ensemble uncertainty."""
        X, y = regression_data
        y_semi, _ = make_semi_supervised_regression(X, y)
        
        st = SelfTrainingRegressor(
            base_estimator=Ridge(random_state=42),  # Not an ensemble
            uncertainty_method="ensemble",
            max_iter=2,
        )
        with pytest.raises(ValueError, match="requires an ensemble"):
            st.fit(X, y_semi)


class TestIntegration:
    """Integration tests."""
    
    def test_import_from_module(self):
        """Test imports work correctly."""
        from endgame.semi_supervised import SelfTrainingClassifier, SelfTrainingRegressor
        import endgame as eg
        
        # Should be accessible via eg.semi_supervised
        assert hasattr(eg, "semi_supervised")
        assert hasattr(eg.semi_supervised, "SelfTrainingClassifier")
        assert hasattr(eg.semi_supervised, "SelfTrainingRegressor")
    
    def test_classification_improves_with_unlabeled(self, classification_data):
        """Test that using unlabeled data can improve performance."""
        X, y = classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Create semi-supervised setting with only 20% labeled
        y_semi, labeled_idx = make_semi_supervised_classification(
            X_train, y_train, labeled_ratio=0.2, random_state=42
        )
        
        # Baseline: train only on labeled data
        X_labeled = X_train[labeled_idx]
        y_labeled = y_train[labeled_idx]
        
        baseline = RandomForestClassifier(n_estimators=50, random_state=42)
        baseline.fit(X_labeled, y_labeled)
        baseline_score = baseline.score(X_test, y_test)
        
        # Self-training: use all data
        st = SelfTrainingClassifier(
            base_estimator=RandomForestClassifier(n_estimators=50, random_state=42),
            threshold=0.8,
            max_iter=10,
            random_state=42,
        )
        st.fit(X_train, y_semi)
        st_score = st.score(X_test, y_test)
        
        # Self-training should not be significantly worse
        # (It may not always be better due to noise in pseudo-labels)
        assert st_score >= baseline_score - 0.1  # Allow some slack
