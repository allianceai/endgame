"""Tests for Bayesian Network Classifiers.

Tests cover:
- TANClassifier
- EBMCClassifier
- ESKDBClassifier
- KDBClassifier
- NeuralKDBClassifier (if PyTorch available)
- BayesianDiscretizer
- Structure learning utilities
- AutoSLE
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted

# Import Bayesian classifiers
from endgame.models.bayesian import (
    TANClassifier,
    EBMCClassifier,
    ESKDBClassifier,
    KDBClassifier,
    AutoSLE,
    compute_mutual_information,
    compute_conditional_mutual_information,
    compute_cmi_matrix,
    bdeu_score,
    bic_score,
    k2_score,
)
from endgame.preprocessing import BayesianDiscretizer

# Check if PyTorch is available
try:
    import torch
    from endgame.models.bayesian import NeuralKDBClassifier
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def discrete_classification_data():
    """Generate discrete classification dataset."""
    np.random.seed(42)
    n_samples = 500
    n_features = 5
    n_classes = 3
    
    # Generate discrete features
    X = np.random.randint(0, 4, size=(n_samples, n_features))
    
    # Generate labels with some structure
    y = (X[:, 0] + X[:, 1]) % n_classes
    
    return X, y


@pytest.fixture
def continuous_classification_data():
    """Generate continuous classification dataset for discretization tests."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def large_discrete_data():
    """Generate larger discrete dataset for ensemble/scaling tests."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = np.random.randint(0, 5, size=(n_samples, n_features))
    y = (X[:, 0] + X[:, 1] + X[:, 2]) % 2
    
    return X, y


# =============================================================================
# TANClassifier Tests
# =============================================================================


class TestTANClassifier:
    """Tests for Tree Augmented Naive Bayes classifier."""
    
    def test_fit_predict(self, discrete_classification_data):
        """Test basic fit and predict."""
        X, y = discrete_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = TANClassifier(smoothing=1.0, random_state=42)
        clf.fit(X_train, y_train)
        
        # Check fitted attributes
        assert clf.structure_ is not None
        assert clf.cpts_ is not None
        assert clf.classes_ is not None
        assert clf.n_features_in_ == X.shape[1]
        
        # Predict
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Predict proba
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), len(np.unique(y)))
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_root_selection_methods(self, discrete_classification_data):
        """Test different root selection strategies."""
        X, y = discrete_classification_data
        
        # Max MI selection
        clf1 = TANClassifier(root_selection='max_mi', random_state=42)
        clf1.fit(X, y)
        
        # Random selection
        clf2 = TANClassifier(root_selection='random', random_state=42)
        clf2.fit(X, y)
        
        # Specific index
        clf3 = TANClassifier(root_selection=2, random_state=42)
        clf3.fit(X, y)
        
        # All should produce valid structures
        assert clf1.structure_ is not None
        assert clf2.structure_ is not None
        assert clf3.structure_ is not None
    
    def test_feature_importances(self, discrete_classification_data):
        """Test feature importance computation."""
        X, y = discrete_classification_data
        
        clf = TANClassifier(random_state=42)
        clf.fit(X, y)
        
        assert clf.feature_importances_ is not None
        assert len(clf.feature_importances_) == X.shape[1]
        assert np.allclose(clf.feature_importances_.sum(), 1.0)
    
    def test_serialization(self, discrete_classification_data):
        """Test model serialization."""
        X, y = discrete_classification_data
        
        clf = TANClassifier(random_state=42)
        clf.fit(X, y)
        
        # To dict
        state = clf.to_dict()
        assert state['class_name'] == 'TANClassifier'
        assert state['fitted'] is True
        
        # From dict
        clf2 = TANClassifier.from_dict(state)
        
        # Predictions should match
        proba1 = clf.predict_proba(X[:10])
        proba2 = clf2.predict_proba(X[:10])
        assert np.allclose(proba1, proba2)
    
    def test_explain(self, discrete_classification_data):
        """Test explanation method."""
        X, y = discrete_classification_data
        
        clf = TANClassifier(random_state=42)
        clf.fit(X, y)
        
        explanation = clf.explain(X[0])
        
        assert 'prediction' in explanation
        assert 'probabilities' in explanation
        assert 'influential_features' in explanation
        assert 'local_structure' in explanation


# =============================================================================
# EBMCClassifier Tests
# =============================================================================


class TestEBMCClassifier:
    """Tests for Efficient Bayesian Multivariate Classifier."""
    
    def test_fit_predict(self, discrete_classification_data):
        """Test basic fit and predict."""
        X, y = discrete_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = EBMCClassifier(max_parents=2, random_state=42)
        clf.fit(X_train, y_train)
        
        # Check feature selection
        assert clf.selected_features_ is not None
        assert len(clf.selected_features_) <= X.shape[1]
        
        # Predict
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        
        assert y_pred.shape == y_test.shape
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_score_types(self, discrete_classification_data):
        """Test different scoring functions."""
        X, y = discrete_classification_data
        
        for score in ['bdeu', 'bic', 'k2']:
            clf = EBMCClassifier(score=score, random_state=42, verbose=False)
            clf.fit(X, y)
            assert clf.structure_ is not None
    
    def test_max_features_limit(self, large_discrete_data):
        """Test max_features parameter."""
        X, y = large_discrete_data
        
        clf = EBMCClassifier(max_features=5, random_state=42)
        clf.fit(X, y)
        
        assert len(clf.selected_features_) <= 5


# =============================================================================
# ESKDBClassifier Tests
# =============================================================================


class TestESKDBClassifier:
    """Tests for Ensemble of Selective K-Dependence Bayes."""
    
    def test_fit_predict(self, discrete_classification_data):
        """Test basic fit and predict."""
        X, y = discrete_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = ESKDBClassifier(n_estimators=10, k=2, random_state=42)
        clf.fit(X_train, y_train)
        
        assert clf.estimators_ is not None
        assert len(clf.estimators_) == 10
        
        y_pred = clf.predict(X_test)
        proba = clf.predict_proba(X_test)
        
        assert y_pred.shape == y_test.shape
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_diversity_methods(self, discrete_classification_data):
        """Test different diversity generation methods."""
        X, y = discrete_classification_data
        
        for method in ['sao', 'bootstrap', 'both']:
            clf = ESKDBClassifier(
                n_estimators=5,
                diversity_method=method,
                random_state=42
            )
            clf.fit(X, y)
            assert clf.estimators_ is not None
    
    def test_aggregation_methods(self, discrete_classification_data):
        """Test different aggregation methods."""
        X, y = discrete_classification_data
        
        for agg in ['averaging', 'voting']:
            clf = ESKDBClassifier(
                n_estimators=5,
                aggregation=agg,
                random_state=42
            )
            clf.fit(X, y)
            proba = clf.predict_proba(X[:10])
            assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_oob_score(self, discrete_classification_data):
        """Test OOB score computation with bootstrap."""
        X, y = discrete_classification_data
        
        clf = ESKDBClassifier(
            n_estimators=10,
            diversity_method='bootstrap',
            random_state=42
        )
        clf.fit(X, y)
        
        assert clf.oob_score_ is not None
        assert 0 <= clf.oob_score_ <= 1


# =============================================================================
# KDBClassifier Tests
# =============================================================================


class TestKDBClassifier:
    """Tests for K-Dependence Bayes classifier."""
    
    def test_fit_predict(self, discrete_classification_data):
        """Test basic fit and predict."""
        X, y = discrete_classification_data
        
        clf = KDBClassifier(k=2, random_state=42)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), len(np.unique(y)))
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_k_values(self, discrete_classification_data):
        """Test different K values."""
        X, y = discrete_classification_data
        
        for k in [0, 1, 2, 3]:
            clf = KDBClassifier(k=k, random_state=42)
            clf.fit(X, y)
            assert clf.structure_ is not None
    
    def test_smoothing_methods(self, discrete_classification_data):
        """Test different smoothing methods."""
        X, y = discrete_classification_data
        
        for smoothing in ['laplace', 'hdp']:
            clf = KDBClassifier(k=2, smoothing=smoothing, random_state=42)
            clf.fit(X, y)
            proba = clf.predict_proba(X[:10])
            assert np.allclose(proba.sum(axis=1), 1.0)


# =============================================================================
# NeuralKDBClassifier Tests (requires PyTorch)
# =============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestNeuralKDBClassifier:
    """Tests for Neural K-Dependence Bayes classifier."""
    
    def test_fit_predict(self, discrete_classification_data):
        """Test basic fit and predict."""
        X, y = discrete_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = NeuralKDBClassifier(
            k=2,
            epochs=5,
            batch_size=64,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train)
        
        assert clf.conditionals_ is not None
        assert clf.structure_ is not None
        
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), len(np.unique(y)))
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    
    def test_early_stopping(self, discrete_classification_data):
        """Test early stopping with validation data."""
        X, y = discrete_classification_data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        clf = NeuralKDBClassifier(
            k=2,
            epochs=50,
            early_stopping=3,
            random_state=42,
            verbose=False
        )
        clf.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        # Should have stopped early
        assert clf.training_history_ is not None
        assert len(clf.training_history_) <= 50
    
    def test_device_selection(self, discrete_classification_data):
        """Test CPU device selection."""
        X, y = discrete_classification_data
        
        clf = NeuralKDBClassifier(
            k=1,
            epochs=2,
            device='cpu',
            random_state=42,
            verbose=False
        )
        clf.fit(X, y)
        
        assert clf.device_ == torch.device('cpu')


# =============================================================================
# BayesianDiscretizer Tests
# =============================================================================


class TestBayesianDiscretizer:
    """Tests for BayesianDiscretizer."""
    
    def test_mdlp_discretization(self, continuous_classification_data):
        """Test MDLP discretization strategy."""
        X, y = continuous_classification_data
        
        disc = BayesianDiscretizer(strategy='mdlp', max_bins=10)
        X_disc = disc.fit_transform(X, y)
        
        assert X_disc.dtype == int
        assert X_disc.shape == X.shape
        assert disc.n_bins_ is not None
        assert all(disc.n_bins_ > 0)
    
    def test_equal_width_discretization(self, continuous_classification_data):
        """Test equal-width discretization."""
        X, y = continuous_classification_data
        
        disc = BayesianDiscretizer(strategy='equal_width', max_bins=5)
        X_disc = disc.fit_transform(X)
        
        assert X_disc.dtype == int
        assert all(disc.n_bins_ <= 5)
    
    def test_equal_freq_discretization(self, continuous_classification_data):
        """Test equal-frequency discretization."""
        X, y = continuous_classification_data
        
        disc = BayesianDiscretizer(strategy='equal_freq', max_bins=5)
        X_disc = disc.fit_transform(X)
        
        assert X_disc.dtype == int
    
    def test_kmeans_discretization(self, continuous_classification_data):
        """Test k-means discretization."""
        X, y = continuous_classification_data
        
        disc = BayesianDiscretizer(strategy='kmeans', max_bins=5, random_state=42)
        X_disc = disc.fit_transform(X)
        
        assert X_disc.dtype == int
    
    def test_auto_discrete_detection(self):
        """Test automatic detection of discrete features."""
        # Mix of discrete and continuous
        X = np.column_stack([
            np.random.randint(0, 3, 100),  # Discrete
            np.random.randn(100),           # Continuous
            np.random.randint(0, 5, 100),  # Discrete
            np.random.randn(100) * 10,     # Continuous
        ])
        y = np.random.randint(0, 2, 100)
        
        disc = BayesianDiscretizer(
            strategy='mdlp',
            discrete_features='auto',
            max_unique_continuous=10
        )
        X_disc = disc.fit_transform(X, y)
        
        # Check that discrete features were detected
        assert disc.discrete_features_[0] == True
        assert disc.discrete_features_[2] == True
    
    def test_transform_consistency(self, continuous_classification_data):
        """Test that transform is consistent."""
        X, y = continuous_classification_data
        X_train, X_test = X[:400], X[400:]
        y_train = y[:400]
        
        disc = BayesianDiscretizer(strategy='mdlp')
        disc.fit(X_train, y_train)
        
        X_test_disc = disc.transform(X_test)
        
        assert X_test_disc.shape == X_test.shape
        assert X_test_disc.dtype == int
    
    def test_inverse_transform(self, continuous_classification_data):
        """Test approximate inverse transform."""
        X, y = continuous_classification_data
        
        disc = BayesianDiscretizer(strategy='equal_width', max_bins=5)
        X_disc = disc.fit_transform(X)
        X_inv = disc.inverse_transform(X_disc)
        
        # Should have same shape
        assert X_inv.shape == X.shape
        # Values should be bin centers (not exact reconstruction)


# =============================================================================
# Structure Learning Tests
# =============================================================================


class TestStructureLearning:
    """Tests for structure learning utilities."""
    
    def test_mutual_information(self, discrete_classification_data):
        """Test MI computation."""
        X, y = discrete_classification_data
        
        mi = compute_mutual_information(X, y, 0)
        
        assert mi >= 0  # MI is non-negative
        assert np.isfinite(mi)
    
    def test_conditional_mutual_information(self, discrete_classification_data):
        """Test CMI computation."""
        X, y = discrete_classification_data
        
        cmi = compute_conditional_mutual_information(X, y, 0, 1)
        
        assert cmi >= 0
        assert np.isfinite(cmi)
    
    def test_cmi_matrix(self, discrete_classification_data):
        """Test CMI matrix computation."""
        X, y = discrete_classification_data
        
        cmi_matrix = compute_cmi_matrix(X, y)
        
        assert cmi_matrix.shape == (X.shape[1], X.shape[1])
        assert np.allclose(cmi_matrix, cmi_matrix.T)  # Symmetric
        assert np.all(cmi_matrix >= 0)
    
    def test_scoring_functions(self, discrete_classification_data):
        """Test BDeu, BIC, K2 scores."""
        X, y = discrete_classification_data
        data = np.column_stack([X, y])
        cardinalities = {i: len(np.unique(X[:, i])) for i in range(X.shape[1])}
        cardinalities[X.shape[1]] = len(np.unique(y))
        
        # All scores should return finite values
        bdeu = bdeu_score(data, [], 0, cardinalities)
        bic = bic_score(data, [], 0, cardinalities)
        k2 = k2_score(data, [], 0, cardinalities)
        
        assert np.isfinite(bdeu)
        assert np.isfinite(bic)
        assert np.isfinite(k2)
        
        # With parents
        bdeu_p = bdeu_score(data, [1], 0, cardinalities)
        assert np.isfinite(bdeu_p)


# =============================================================================
# AutoSLE Tests
# =============================================================================


class TestAutoSLE:
    """Tests for AutoSLE structure learning."""
    
    def test_learn_structure(self, discrete_classification_data):
        """Test basic structure learning."""
        X, y = discrete_classification_data
        data = np.column_stack([X, y])
        
        sle = AutoSLE(
            solvers=['hc'],
            max_cluster_size=10,
            random_state=42
        )
        structure = sle.learn(data)
        
        assert structure is not None
        assert structure.number_of_nodes() == data.shape[1]
    
    def test_partition_methods(self, large_discrete_data):
        """Test different partitioning methods."""
        X, y = large_discrete_data
        data = np.column_stack([X, y])
        
        for method in ['spectral', 'correlation', 'random']:
            sle = AutoSLE(
                solvers=['hc'],
                partition_method=method,
                max_cluster_size=10,
                random_state=42
            )
            structure = sle.learn(data)
            assert structure is not None
    
    def test_edge_confidence(self, discrete_classification_data):
        """Test edge confidence tracking."""
        X, y = discrete_classification_data
        data = np.column_stack([X, y])
        
        sle = AutoSLE(
            solvers=['hc'],
            max_cluster_size=10,
            random_state=42
        )
        structure = sle.learn(data)
        
        # Should have edge confidence dict
        assert sle.edge_confidence_ is not None
        
        # Get highly confident edges
        confident = sle.get_highly_confident_edges(min_confidence=0.5)
        assert isinstance(confident, list)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full pipeline."""
    
    def test_discretize_and_classify_pipeline(self, continuous_classification_data):
        """Test full pipeline: discretize -> classify."""
        X, y = continuous_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Discretize
        disc = BayesianDiscretizer(strategy='mdlp', max_bins=5)
        X_train_disc = disc.fit_transform(X_train, y_train)
        X_test_disc = disc.transform(X_test)
        
        # Classify
        clf = TANClassifier(random_state=42)
        clf.fit(X_train_disc, y_train)
        
        proba = clf.predict_proba(X_test_disc)
        y_pred = clf.predict(X_test_disc)
        
        assert proba.shape == (len(X_test), len(np.unique(y)))
        assert y_pred.shape == y_test.shape
    
    def test_sklearn_compatibility(self, discrete_classification_data):
        """Test sklearn pipeline compatibility."""
        from sklearn.pipeline import Pipeline
        from sklearn.model_selection import cross_val_score
        
        X, y = discrete_classification_data
        
        clf = TANClassifier(random_state=42)
        
        # Should work with cross_val_score
        scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)
    
    def test_accuracy_baseline(self, discrete_classification_data):
        """Test that BNCs beat random guessing."""
        X, y = discrete_classification_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random baseline
        random_accuracy = 1.0 / len(np.unique(y))
        
        # TAN
        clf = TANClassifier(random_state=42)
        clf.fit(X_train, y_train)
        tan_accuracy = (clf.predict(X_test) == y_test).mean()
        
        # Should be better than random
        assert tan_accuracy > random_accuracy


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
