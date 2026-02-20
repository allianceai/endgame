"""Tests for interpretable machine learning models.

This module tests the interpretable classifiers:
- CORELS (Certifiably Optimal Rule Lists)
- NODE-GAM (Neural Oblivious Decision Ensembles for GAMs)
- GAMI-Net (Generalized Additive Models with Structured Interactions)
- SLIM/FasterRisk (Sparse Integer Linear Models)
- pyGAM (Generalized Additive Models)
- GOSDT (Globally Optimal Sparse Decision Trees)
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

# Skip tests if dependencies not available
pytest.importorskip("sklearn")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def binary_classification_data():
    """Create binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def regression_data():
    """Create regression dataset."""
    X, y = make_regression(
        n_samples=500,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


# ============================================================================
# CORELS Tests
# ============================================================================

class TestCORELS:
    """Test CORELS classifier."""

    def test_import(self):
        """Test that CORELSClassifier can be imported."""
        from endgame.models.interpretable import CORELSClassifier
        assert CORELSClassifier is not None

    def test_init(self):
        """Test CORELSClassifier initialization."""
        from endgame.models.interpretable import CORELSClassifier
        clf = CORELSClassifier(max_card=2, c=0.001)
        assert clf.max_card == 2
        assert clf.c == 0.001

    def test_fit_predict_greedy_backend(self, binary_classification_data):
        """Test that fitting works with the greedy (pure-Python) backend."""
        from endgame.models.interpretable.corels import CORELSClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = CORELSClassifier(backend="greedy")
        clf.fit(X_train, y_train)

        assert clf.backend_used_ == "greedy"
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_get_rules_method_exists(self):
        """Test that get_rules method exists."""
        from endgame.models.interpretable import CORELSClassifier
        clf = CORELSClassifier()
        assert hasattr(clf, "get_rules")

    def test_summary_method_exists(self):
        """Test that summary method exists."""
        from endgame.models.interpretable import CORELSClassifier
        clf = CORELSClassifier()
        assert hasattr(clf, "summary")


# ============================================================================
# NODE-GAM Tests
# ============================================================================

class TestNodeGAM:
    """Test NODE-GAM classifier and regressor."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")
            return False

    def test_import(self):
        """Test that NodeGAMClassifier can be imported."""
        from endgame.models.interpretable import NodeGAMClassifier, NodeGAMRegressor
        assert NodeGAMClassifier is not None
        assert NodeGAMRegressor is not None

    def test_classifier_init(self):
        """Test NodeGAMClassifier initialization."""
        from endgame.models.interpretable import NodeGAMClassifier
        clf = NodeGAMClassifier(n_trees_per_feature=16, depth=3)
        assert clf.n_trees_per_feature == 16
        assert clf.depth == 3

    def test_classifier_fit_predict(self, torch_available, binary_classification_data):
        """Test NodeGAMClassifier fit and predict."""
        from endgame.models.interpretable import NodeGAMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = NodeGAMClassifier(
            n_trees_per_feature=8,
            depth=2,
            n_epochs=5,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        # Predictions
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        # Probabilities
        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)
        assert np.allclose(y_proba.sum(axis=1), 1.0)

    def test_classifier_feature_contributions(self, torch_available, binary_classification_data):
        """Test that feature contributions can be extracted."""
        from endgame.models.interpretable import NodeGAMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = NodeGAMClassifier(n_trees_per_feature=4, depth=2, n_epochs=3, random_state=42)
        clf.fit(X_train, y_train)

        contributions = clf.get_feature_contributions(X_test)
        assert contributions.shape == (len(X_test), X_test.shape[1])

    def test_regressor_fit_predict(self, torch_available, regression_data):
        """Test NodeGAMRegressor fit and predict."""
        from endgame.models.interpretable import NodeGAMRegressor

        X_train, X_test, y_train, y_test = regression_data
        reg = NodeGAMRegressor(
            n_trees_per_feature=8,
            depth=2,
            n_epochs=5,
            random_state=42,
        )
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape


# ============================================================================
# GAMI-Net Tests
# ============================================================================

class TestGAMINet:
    """Test GAMI-Net classifier and regressor."""

    @pytest.fixture
    def torch_available(self):
        """Check if PyTorch is available."""
        try:
            import torch
            return True
        except ImportError:
            pytest.skip("PyTorch not available")
            return False

    def test_import(self):
        """Test that GAMINetClassifier can be imported."""
        from endgame.models.interpretable import GAMINetClassifier, GAMINetRegressor
        assert GAMINetClassifier is not None
        assert GAMINetRegressor is not None

    def test_classifier_init(self):
        """Test GAMINetClassifier initialization."""
        from endgame.models.interpretable import GAMINetClassifier
        clf = GAMINetClassifier(interact_num=5, main_hidden_units=[32, 16])
        assert clf.interact_num == 5
        assert clf.main_hidden_units == [32, 16]

    def test_classifier_fit_predict(self, torch_available, binary_classification_data):
        """Test GAMINetClassifier fit and predict."""
        from endgame.models.interpretable import GAMINetClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GAMINetClassifier(
            interact_num=3,
            main_hidden_units=[16],
            interact_hidden_units=[8],
            n_epochs=5,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)

    def test_get_effects(self, torch_available, binary_classification_data):
        """Test that main effects and interactions can be extracted."""
        from endgame.models.interpretable import GAMINetClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GAMINetClassifier(
            interact_num=3,
            main_hidden_units=[16],
            n_epochs=3,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        main_effects, interactions = clf.get_effects(X_test)
        assert main_effects.shape == (len(X_test), X_test.shape[1])

    def test_regressor_fit_predict(self, torch_available, regression_data):
        """Test GAMINetRegressor fit and predict."""
        from endgame.models.interpretable import GAMINetRegressor

        X_train, X_test, y_train, y_test = regression_data
        reg = GAMINetRegressor(
            interact_num=3,
            main_hidden_units=[16],
            n_epochs=5,
            random_state=42,
        )
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape


# ============================================================================
# SLIM Tests
# ============================================================================

class TestSLIM:
    """Test SLIM classifier."""

    def test_import(self):
        """Test that SLIMClassifier can be imported."""
        from endgame.models.interpretable import SLIMClassifier
        assert SLIMClassifier is not None

    def test_init(self):
        """Test SLIMClassifier initialization."""
        from endgame.models.interpretable import SLIMClassifier
        clf = SLIMClassifier(max_coef=5, sparsity=10)
        assert clf.max_coef == 5
        assert clf.sparsity == 10

    def test_fit_predict(self, binary_classification_data):
        """Test SLIMClassifier fit and predict."""
        from endgame.models.interpretable import SLIMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = SLIMClassifier(max_coef=5, C=0.1, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)

    def test_integer_coefficients(self, binary_classification_data):
        """Test that coefficients are integers."""
        from endgame.models.interpretable import SLIMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = SLIMClassifier(max_coef=5, C=0.1, random_state=42)
        clf.fit(X_train, y_train)

        assert np.all(clf.coef_ == np.floor(clf.coef_))
        assert abs(clf.intercept_) == abs(int(clf.intercept_))

    def test_get_scorecard(self, binary_classification_data):
        """Test scorecard generation."""
        from endgame.models.interpretable import SLIMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = SLIMClassifier(max_coef=5, random_state=42)
        clf.fit(X_train, y_train)

        scorecard = clf.get_scorecard()
        assert isinstance(scorecard, str)
        assert "points" in scorecard.lower()

    def test_score_sample(self, binary_classification_data):
        """Test single sample scoring."""
        from endgame.models.interpretable import SLIMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = SLIMClassifier(max_coef=5, random_state=42)
        clf.fit(X_train, y_train)

        total_score, breakdown = clf.score_sample(X_test[0])
        assert isinstance(total_score, (int, np.integer))
        assert isinstance(breakdown, list)


# ============================================================================
# FasterRisk Tests
# ============================================================================

class TestFasterRisk:
    """Test FasterRisk classifier."""

    def test_import(self):
        """Test that FasterRiskClassifier can be imported."""
        from endgame.models.interpretable import FasterRiskClassifier
        assert FasterRiskClassifier is not None

    def test_init(self):
        """Test FasterRiskClassifier initialization."""
        from endgame.models.interpretable import FasterRiskClassifier
        clf = FasterRiskClassifier(max_coef=5, sparsity=10)
        assert clf.max_coef == 5
        assert clf.sparsity == 10

    def test_fit_predict_without_fasterrisk(self, binary_classification_data):
        """Test that fitting raises ImportError when fasterrisk not installed."""
        from endgame.models.interpretable.slim import HAS_FASTERRISK, FasterRiskClassifier

        if HAS_FASTERRISK:
            pytest.skip("fasterrisk is installed, skipping import error test")

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = FasterRiskClassifier()

        with pytest.raises(ImportError, match="fasterrisk"):
            clf.fit(X_train, y_train)


# ============================================================================
# pyGAM Tests
# ============================================================================

class TestGAM:
    """Test GAM classifier and regressor (pyGAM wrapper)."""

    @pytest.fixture
    def pygam_available(self):
        """Check if pyGAM is available."""
        try:
            import pygam
            return True
        except ImportError:
            pytest.skip("pyGAM not available")
            return False

    def test_import(self):
        """Test that GAMClassifier can be imported."""
        from endgame.models.interpretable import GAMClassifier, GAMRegressor
        assert GAMClassifier is not None
        assert GAMRegressor is not None

    def test_classifier_init(self):
        """Test GAMClassifier initialization."""
        from endgame.models.interpretable import GAMClassifier
        clf = GAMClassifier(n_splines=20, lam=0.5)
        assert clf.n_splines == 20
        assert clf.lam == 0.5

    def test_classifier_fit_predict(self, pygam_available, binary_classification_data):
        """Test GAMClassifier fit and predict."""
        from endgame.models.interpretable import GAMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GAMClassifier(n_splines=10, lam=1.0)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)

    def test_partial_dependence(self, pygam_available, binary_classification_data):
        """Test partial dependence extraction."""
        from endgame.models.interpretable import GAMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GAMClassifier(n_splines=10, lam=1.0)
        clf.fit(X_train, y_train)

        grid, effects = clf.partial_dependence(0, X_test)
        assert len(grid) == len(effects)

    def test_summary(self, pygam_available, binary_classification_data):
        """Test summary method."""
        from endgame.models.interpretable import GAMClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GAMClassifier(n_splines=10, lam=1.0)
        clf.fit(X_train, y_train)

        summary = clf.summary()
        assert isinstance(summary, str)
        assert "GAM" in summary

    def test_regressor_fit_predict(self, pygam_available, regression_data):
        """Test GAMRegressor fit and predict."""
        from endgame.models.interpretable import GAMRegressor

        X_train, X_test, y_train, y_test = regression_data
        reg = GAMRegressor(n_splines=10, lam=1.0)
        reg.fit(X_train, y_train)

        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape


# ============================================================================
# GOSDT Tests
# ============================================================================

class TestGOSDT:
    """Test GOSDT classifier."""

    def test_import(self):
        """Test that GOSDTClassifier can be imported."""
        from endgame.models.interpretable import GOSDTClassifier
        assert GOSDTClassifier is not None

    def test_init(self):
        """Test GOSDTClassifier initialization."""
        from endgame.models.interpretable import GOSDTClassifier
        clf = GOSDTClassifier(regularization=0.01, depth_budget=5)
        assert clf.regularization == 0.01
        assert clf.depth_budget == 5

    def test_fit_predict_with_fallback(self, binary_classification_data):
        """Test GOSDTClassifier fit and predict with CART fallback."""
        from endgame.models.interpretable import GOSDTClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GOSDTClassifier(
            regularization=0.01,
            depth_budget=3,
            fallback_to_cart=True,
            random_state=42,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        y_proba = clf.predict_proba(X_test)
        assert y_proba.shape == (len(X_test), 2)

    def test_get_tree_structure(self, binary_classification_data):
        """Test tree structure extraction."""
        from endgame.models.interpretable import GOSDTClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GOSDTClassifier(depth_budget=3, fallback_to_cart=True, random_state=42)
        clf.fit(X_train, y_train)

        structure = clf.get_tree_structure()
        assert isinstance(structure, str)
        assert "PREDICT" in structure or "predict" in structure.lower()

    def test_get_rules(self, binary_classification_data):
        """Test rules extraction."""
        from endgame.models.interpretable import GOSDTClassifier

        X_train, X_test, y_train, y_test = binary_classification_data
        clf = GOSDTClassifier(depth_budget=3, fallback_to_cart=True, random_state=42)
        clf.fit(X_train, y_train)

        rules = clf.get_rules()
        assert isinstance(rules, list)
        assert len(rules) > 0
        assert "conditions" in rules[0]
        assert "prediction" in rules[0]


# ============================================================================
# Model Registry Tests
# ============================================================================

class TestModelRegistryIntegration:
    """Test that interpretable models are properly registered."""

    def test_all_models_registered(self):
        """Test that all interpretable models are in the registry."""
        from endgame.automl.model_registry import MODEL_REGISTRY

        expected_models = [
            "corels", "node_gam", "gami_net", "slim", "fasterrisk", "gam", "gosdt"
        ]

        for model_name in expected_models:
            assert model_name in MODEL_REGISTRY, f"{model_name} not in registry"

    def test_interpretable_flag_set(self):
        """Test that interpretable flag is set correctly."""
        from endgame.automl.model_registry import MODEL_REGISTRY

        interpretable_models = [
            "corels", "node_gam", "gami_net", "slim", "fasterrisk", "gam", "gosdt"
        ]

        for model_name in interpretable_models:
            assert MODEL_REGISTRY[model_name].interpretable, \
                f"{model_name} should be marked as interpretable"

    def test_list_interpretable_models(self):
        """Test listing interpretable models."""
        from endgame.automl.model_registry import list_models

        interpretable = list_models(interpretable_only=True)
        assert len(interpretable) >= 7  # At least our new models plus existing

        expected_new = ["corels", "node_gam", "gami_net", "slim", "fasterrisk", "gam", "gosdt"]
        for model in expected_new:
            assert model in interpretable, f"{model} not in interpretable list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
