"""Tests for TabDPT (Tabular Discriminative Pre-trained Transformer) wrappers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.utils.validation import check_is_fitted

from endgame.models.tabular.tabdpt import (
    TabDPTClassifier,
    TabDPTRegressor,
    _check_tabdpt_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=2, random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_data():
    """Small multiclass classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=4, n_clusters_per_class=1, random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Small regression dataset."""
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    return X, y


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

class TestAvailability:
    def test_check_tabdpt_available_returns_bool(self):
        result = _check_tabdpt_available()
        assert isinstance(result, bool)

    def test_check_tabdpt_available_consistent(self):
        # Result should be consistent across calls
        result1 = _check_tabdpt_available()
        result2 = _check_tabdpt_available()
        assert result1 == result2


# ---------------------------------------------------------------------------
# Import from __init__
# ---------------------------------------------------------------------------

class TestImport:
    def test_import_classifier_from_init(self):
        from endgame.models.tabular import TabDPTClassifier as Clf
        assert Clf is TabDPTClassifier

    def test_import_regressor_from_init(self):
        from endgame.models.tabular import TabDPTRegressor as Reg
        assert Reg is TabDPTRegressor


# ---------------------------------------------------------------------------
# TabDPTClassifier
# ---------------------------------------------------------------------------

class TestTabDPTClassifierParams:
    def test_default_params(self):
        clf = TabDPTClassifier()
        assert clf.n_estimators == 8
        assert clf.device == "auto"
        assert clf.random_state == 0
        assert clf.max_samples == 10_000
        assert clf.max_features == 500
        assert clf.max_classes == 10
        assert clf.ignore_pretraining_limits is False

    def test_custom_params(self):
        clf = TabDPTClassifier(
            n_estimators=16, device="cpu", random_state=42,
            max_samples=5000, max_features=100, max_classes=5,
            ignore_pretraining_limits=True,
        )
        assert clf.n_estimators == 16
        assert clf.device == "cpu"
        assert clf.random_state == 42
        assert clf.max_samples == 5000
        assert clf.max_features == 100
        assert clf.max_classes == 5
        assert clf.ignore_pretraining_limits is True

    def test_get_params(self):
        clf = TabDPTClassifier(n_estimators=4, random_state=7)
        params = clf.get_params()
        assert params["n_estimators"] == 4
        assert params["random_state"] == 7
        assert "device" in params

    def test_set_params(self):
        clf = TabDPTClassifier()
        clf.set_params(n_estimators=32, device="cpu")
        assert clf.n_estimators == 32
        assert clf.device == "cpu"

    def test_estimator_type(self):
        clf = TabDPTClassifier()
        assert clf._estimator_type == "classifier"


class TestTabDPTClassifierFitPredict:
    def test_fit_returns_self(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier()
        result = clf.fit(X, y)
        assert result is clf

    def test_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X, y)
        assert hasattr(clf, "classes_")
        np.testing.assert_array_equal(clf.classes_, np.array([0, 1]))

    def test_n_classes_attribute(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X, y)
        assert clf.n_classes_ == 2

    def test_predict_shape(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_proba_shape(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        proba = clf.predict_proba(X[150:])
        assert proba.shape == (50, 2)

    def test_predict_proba_sums_to_one(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        proba = clf.predict_proba(X[150:])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_nonnegative(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        proba = clf.predict_proba(X[150:])
        assert np.all(proba >= 0)

    def test_predict_labels_in_classes(self, binary_data):
        X, y = binary_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert set(preds).issubset(set(clf.classes_))

    def test_not_fitted_error(self):
        clf = TabDPTClassifier()
        with pytest.raises(Exception):
            clf.predict(np.zeros((5, 10)))

    def test_not_fitted_error_proba(self):
        clf = TabDPTClassifier()
        with pytest.raises(Exception):
            clf.predict_proba(np.zeros((5, 10)))

    def test_multiclass(self, multiclass_data):
        X, y = multiclass_data
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        proba = clf.predict_proba(X[150:])
        assert proba.shape == (50, 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_string_labels(self):
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)
        y = np.array(["cat", "dog", "fish"])[rng.randint(0, 3, 200)]
        clf = TabDPTClassifier().fit(X[:150], y[:150])
        preds = clf.predict(X[150:])
        assert set(preds).issubset({"cat", "dog", "fish"})
        proba = clf.predict_proba(X[150:])
        assert proba.shape == (50, 3)


class TestTabDPTClassifierLimits:
    def test_sample_limit(self):
        X = np.zeros((10_001, 5))
        y = np.zeros(10_001, dtype=int)
        clf = TabDPTClassifier()
        with pytest.raises(ValueError, match="samples"):
            clf.fit(X, y)

    def test_feature_limit(self):
        X = np.zeros((100, 501))
        y = np.zeros(100, dtype=int)
        clf = TabDPTClassifier()
        with pytest.raises(ValueError, match="features"):
            clf.fit(X, y)

    def test_class_limit(self):
        X = np.zeros((100, 5))
        y = np.arange(100) % 11  # 11 classes
        clf = TabDPTClassifier()
        with pytest.raises(ValueError, match="classes"):
            clf.fit(X, y)

    def test_ignore_pretraining_limits_samples(self):
        X = np.zeros((10_001, 5))
        y = np.zeros(10_001, dtype=int)
        clf = TabDPTClassifier(ignore_pretraining_limits=True)
        # Should not raise
        clf.fit(X, y)
        assert clf.is_fitted_

    def test_ignore_pretraining_limits_features(self):
        X = np.zeros((100, 501))
        y = np.zeros(100, dtype=int)
        clf = TabDPTClassifier(ignore_pretraining_limits=True)
        clf.fit(X, y)
        assert clf.is_fitted_

    def test_ignore_pretraining_limits_classes(self):
        X = np.zeros((100, 5))
        y = np.arange(100) % 11
        clf = TabDPTClassifier(ignore_pretraining_limits=True)
        clf.fit(X, y)
        assert clf.is_fitted_


class TestTabDPTClassifierCheckConstraints:
    def test_valid_data(self):
        X = np.zeros((100, 10))
        y = np.zeros(100, dtype=int)
        result = TabDPTClassifier.check_constraints(X, y)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_too_many_samples(self):
        X = np.zeros((10_001, 10))
        y = np.zeros(10_001, dtype=int)
        result = TabDPTClassifier.check_constraints(X, y)
        assert result["valid"] is False
        assert any("samples" in issue.lower() for issue in result["issues"])

    def test_too_many_features(self):
        X = np.zeros((100, 501))
        y = np.zeros(100, dtype=int)
        result = TabDPTClassifier.check_constraints(X, y)
        assert result["valid"] is False
        assert any("features" in issue.lower() for issue in result["issues"])

    def test_too_many_classes(self):
        X = np.zeros((100, 10))
        y = np.arange(100) % 11
        result = TabDPTClassifier.check_constraints(X, y)
        assert result["valid"] is False
        assert any("classes" in issue.lower() for issue in result["issues"])


# ---------------------------------------------------------------------------
# TabDPTRegressor
# ---------------------------------------------------------------------------

class TestTabDPTRegressorParams:
    def test_default_params(self):
        reg = TabDPTRegressor()
        assert reg.n_estimators == 8
        assert reg.device == "auto"
        assert reg.random_state == 0
        assert reg.max_samples == 10_000
        assert reg.max_features == 500
        assert reg.ignore_pretraining_limits is False

    def test_custom_params(self):
        reg = TabDPTRegressor(
            n_estimators=16, device="cpu", random_state=42,
            max_samples=5000, max_features=100,
            ignore_pretraining_limits=True,
        )
        assert reg.n_estimators == 16
        assert reg.device == "cpu"
        assert reg.random_state == 42
        assert reg.max_samples == 5000
        assert reg.max_features == 100
        assert reg.ignore_pretraining_limits is True

    def test_get_params(self):
        reg = TabDPTRegressor(n_estimators=4)
        params = reg.get_params()
        assert params["n_estimators"] == 4

    def test_set_params(self):
        reg = TabDPTRegressor()
        reg.set_params(n_estimators=32)
        assert reg.n_estimators == 32

    def test_estimator_type(self):
        reg = TabDPTRegressor()
        assert reg._estimator_type == "regressor"


class TestTabDPTRegressorFitPredict:
    def test_fit_returns_self(self, regression_data):
        X, y = regression_data
        reg = TabDPTRegressor()
        result = reg.fit(X, y)
        assert result is reg

    def test_n_features_in(self, regression_data):
        X, y = regression_data
        reg = TabDPTRegressor().fit(X, y)
        assert reg.n_features_in_ == 10

    def test_predict_shape(self, regression_data):
        X, y = regression_data
        reg = TabDPTRegressor().fit(X[:150], y[:150])
        preds = reg.predict(X[150:])
        assert preds.shape == (50,)

    def test_predict_finite(self, regression_data):
        X, y = regression_data
        reg = TabDPTRegressor().fit(X[:150], y[:150])
        preds = reg.predict(X[150:])
        assert np.all(np.isfinite(preds))

    def test_not_fitted_error(self):
        reg = TabDPTRegressor()
        with pytest.raises(Exception):
            reg.predict(np.zeros((5, 10)))


class TestTabDPTRegressorLimits:
    def test_sample_limit(self):
        X = np.zeros((10_001, 5))
        y = np.zeros(10_001)
        reg = TabDPTRegressor()
        with pytest.raises(ValueError, match="samples"):
            reg.fit(X, y)

    def test_feature_limit(self):
        X = np.zeros((100, 501))
        y = np.zeros(100)
        reg = TabDPTRegressor()
        with pytest.raises(ValueError, match="features"):
            reg.fit(X, y)

    def test_ignore_pretraining_limits(self):
        X = np.zeros((10_001, 5))
        y = np.zeros(10_001)
        reg = TabDPTRegressor(ignore_pretraining_limits=True)
        reg.fit(X, y)
        assert reg.is_fitted_


class TestTabDPTRegressorCheckConstraints:
    def test_valid_data(self):
        X = np.zeros((100, 10))
        result = TabDPTRegressor.check_constraints(X)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_too_many_samples(self):
        X = np.zeros((10_001, 10))
        result = TabDPTRegressor.check_constraints(X)
        assert result["valid"] is False
        assert any("samples" in issue.lower() for issue in result["issues"])

    def test_too_many_features(self):
        X = np.zeros((100, 501))
        result = TabDPTRegressor.check_constraints(X)
        assert result["valid"] is False
        assert any("features" in issue.lower() for issue in result["issues"])
