"""Tests for the endgame.explain module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def classification_data():
    """Generate a simple classification dataset with a fitted model."""
    X, y = make_classification(
        n_samples=200, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42,
    )
    return X_train, X_test, y_train, y_test


@pytest.fixture
def fitted_rf(classification_data):
    X_train, _, y_train, _ = classification_data
    model = RandomForestClassifier(n_estimators=20, random_state=42)
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def fitted_lr(classification_data):
    X_train, _, y_train, _ = classification_data
    model = LogisticRegression(random_state=42, max_iter=500)
    model.fit(X_train, y_train)
    return model


# ------------------------------------------------------------------
# Explanation dataclass
# ------------------------------------------------------------------

class TestExplanation:
    def test_basic_creation(self):
        from endgame.explain._base import Explanation

        values = np.array([0.3, 0.1, 0.5, 0.05, 0.05])
        exp = Explanation(
            values=values,
            base_value=0.5,
            feature_names=["a", "b", "c", "d", "e"],
            method="test",
        )
        assert exp.method == "test"
        assert exp.base_value == 0.5
        assert len(exp.feature_names) == 5

    def test_top_features(self):
        from endgame.explain._base import Explanation

        values = np.array([0.3, 0.1, 0.5, 0.05, 0.05])
        exp = Explanation(
            values=values,
            feature_names=["a", "b", "c", "d", "e"],
            method="test",
        )
        top = exp.top_features(3)
        assert len(top) == 3
        assert top[0] == "c"  # highest absolute value

    def test_top_features_2d(self):
        from endgame.explain._base import Explanation

        values = np.array([
            [0.1, 0.5, 0.2],
            [0.3, 0.4, 0.1],
        ])
        exp = Explanation(
            values=values,
            feature_names=["a", "b", "c"],
            method="test",
        )
        top = exp.top_features(2)
        assert len(top) == 2
        assert top[0] == "b"  # highest mean abs

    def test_to_dataframe(self):
        from endgame.explain._base import Explanation

        values = np.array([0.3, 0.1, 0.5])
        exp = Explanation(
            values=values,
            feature_names=["a", "b", "c"],
            method="test",
        )
        df = exp.to_dataframe()
        assert df.shape == (3, 1)
        assert "attribution" in df.columns

    def test_repr(self):
        from endgame.explain._base import Explanation

        values = np.array([0.3, 0.1])
        exp = Explanation(values=values, method="shap")
        r = repr(exp)
        assert "shap" in r
        assert "(2,)" in r


# ------------------------------------------------------------------
# SHAP Explainer
# ------------------------------------------------------------------

class TestSHAPExplainer:
    def test_tree_shap(self, fitted_rf, classification_data):
        shap = pytest.importorskip("shap")
        from endgame.explain import SHAPExplainer

        _, X_test, _, _ = classification_data
        explainer = SHAPExplainer(fitted_rf)
        explanation = explainer.explain(X_test[:10])

        assert explanation.method == "shap"
        assert explanation.values.shape[1] == 5
        assert explanation.feature_names is not None

    def test_linear_shap(self, fitted_lr, classification_data):
        shap = pytest.importorskip("shap")
        from endgame.explain import SHAPExplainer

        _, X_test, _, _ = classification_data
        explainer = SHAPExplainer(fitted_lr)
        explanation = explainer.explain(X_test[:10])

        assert explanation.method == "shap"
        assert explanation.values.shape[1] == 5

    def test_custom_feature_names(self, fitted_rf, classification_data):
        shap = pytest.importorskip("shap")
        from endgame.explain import SHAPExplainer

        _, X_test, _, _ = classification_data
        names = ["f0", "f1", "f2", "f3", "f4"]
        explainer = SHAPExplainer(fitted_rf, feature_names=names)
        explanation = explainer.explain(X_test[:5])

        assert explanation.feature_names == names


# ------------------------------------------------------------------
# PDP
# ------------------------------------------------------------------

class TestPartialDependence:
    def test_pdp_1d(self, fitted_rf, classification_data):
        from endgame.explain import PartialDependence

        X_train, _, _, _ = classification_data
        pdp = PartialDependence(fitted_rf)
        explanation = pdp.explain(X_train, features=[0])

        assert explanation.method == "pdp"
        assert explanation.values is not None

    def test_pdp_multiple_features(self, fitted_rf, classification_data):
        from endgame.explain import PartialDependence

        X_train, _, _, _ = classification_data
        pdp = PartialDependence(fitted_rf)
        explanation = pdp.explain(X_train, features=[0, 1, 2])

        assert explanation.method == "pdp"


# ------------------------------------------------------------------
# Feature Interaction
# ------------------------------------------------------------------

class TestFeatureInteraction:
    def test_h_statistic(self, fitted_rf, classification_data):
        from endgame.explain import FeatureInteraction

        X_train, _, _, _ = classification_data
        fi = FeatureInteraction(fitted_rf)
        explanation = fi.explain(X_train[:50])

        assert explanation.method == "h_statistic"
        assert explanation.values is not None


# ------------------------------------------------------------------
# explain() convenience function
# ------------------------------------------------------------------

class TestExplainFunction:
    def test_auto_tree(self, fitted_rf, classification_data):
        shap = pytest.importorskip("shap")
        from endgame.explain import explain

        _, X_test, _, _ = classification_data
        explanation = explain(fitted_rf, X_test[:10])

        assert explanation.method == "shap"
        assert explanation.values.shape[1] == 5

    def test_pdp_method(self, fitted_rf, classification_data):
        from endgame.explain import explain

        X_train, _, _, _ = classification_data
        explanation = explain(
            fitted_rf, X_train[:50], method="pdp", features=[0],
        )
        assert explanation.method == "pdp"

    def test_invalid_method(self, fitted_rf, classification_data):
        from endgame.explain import explain

        _, X_test, _, _ = classification_data
        with pytest.raises(ValueError, match="Unknown explanation method"):
            explain(fitted_rf, X_test, method="invalid")
