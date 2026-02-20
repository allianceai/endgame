"""Tests for TabPFN 2.5 wrappers (TabPFN25Classifier, TabPFN25Regressor).

These tests exercise the fallback (kNN) code path and validation logic
without requiring the actual ``tabpfn`` package to be installed.
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression

from endgame.models.tabular.tabpfn import (
    TabPFN25Classifier,
    TabPFN25Regressor,
    _check_tabpfn_25_available,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Classification dataset with 200 samples and 10 features."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_classes=3,
        random_state=42,
    )
    return X, y


@pytest.fixture
def reg_data():
    """Regression dataset with 200 samples and 10 features."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=6,
        random_state=42,
    )
    return X, y


# ---------------------------------------------------------------------------
# Import / availability
# ---------------------------------------------------------------------------

class TestAvailability:
    """Test that the availability check returns a bool without crashing."""

    def test_check_returns_bool(self):
        result = _check_tabpfn_25_available()
        assert isinstance(result, bool)

    def test_import_from_init(self):
        """Classes are importable from the subpackage __init__."""
        from endgame.models.tabular import TabPFN25Classifier as C
        from endgame.models.tabular import TabPFN25Regressor as R
        assert C is TabPFN25Classifier
        assert R is TabPFN25Regressor


# ---------------------------------------------------------------------------
# TabPFN25Classifier
# ---------------------------------------------------------------------------

class TestTabPFN25Classifier:
    """Tests for the v2.5 classifier (fallback path)."""

    def test_init_defaults(self):
        clf = TabPFN25Classifier()
        assert clf.n_estimators == 8
        assert clf.device == "auto"
        assert clf.random_state == 0
        assert clf.fit_mode == "fit_with_cache"
        assert clf.memory_saving_mode is False
        assert clf.ignore_pretraining_limits is False
        assert clf.categorical_features_indices is None
        assert clf.model_version == "2.5_real"

    def test_init_custom(self):
        clf = TabPFN25Classifier(
            n_estimators=16,
            device="cpu",
            random_state=42,
            categorical_features_indices=[0, 2],
            fit_mode="fit_with_cache",
            memory_saving_mode=True,
            ignore_pretraining_limits=True,
            model_version="2.5",
        )
        assert clf.n_estimators == 16
        assert clf.device == "cpu"
        assert clf.random_state == 42
        assert clf.categorical_features_indices == [0, 2]
        assert clf.fit_mode == "fit_with_cache"
        assert clf.memory_saving_mode is True
        assert clf.ignore_pretraining_limits is True
        assert clf.model_version == "2.5"

    def test_fit_predict(self, clf_data):
        X, y = clf_data
        clf = TabPFN25Classifier(random_state=0)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert preds.shape == (len(X),)
        # predictions should be in the original label space
        assert set(preds).issubset(set(y))

    def test_predict_proba_shape(self, clf_data):
        X, y = clf_data
        clf = TabPFN25Classifier()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        n_classes = len(np.unique(y))
        assert proba.shape == (len(X), n_classes)

    def test_predict_proba_sums_to_one(self, clf_data):
        X, y = clf_data
        clf = TabPFN25Classifier()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_nonnegative(self, clf_data):
        X, y = clf_data
        clf = TabPFN25Classifier()
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert (proba >= 0).all()

    def test_classes_attribute(self, clf_data):
        X, y = clf_data
        clf = TabPFN25Classifier()
        clf.fit(X, y)

        np.testing.assert_array_equal(clf.classes_, np.unique(y))
        assert clf.n_classes_ == len(np.unique(y))

    def test_not_fitted_raises(self):
        clf = TabPFN25Classifier()
        with pytest.raises(Exception):
            clf.predict(np.zeros((2, 3)))

    def test_not_fitted_proba_raises(self):
        clf = TabPFN25Classifier()
        with pytest.raises(Exception):
            clf.predict_proba(np.zeros((2, 3)))

    def test_too_many_samples_raises(self):
        X = np.zeros((TabPFN25Classifier.MAX_SAMPLES + 1, 3))
        y = np.zeros(TabPFN25Classifier.MAX_SAMPLES + 1)
        clf = TabPFN25Classifier()
        with pytest.raises(ValueError, match="maximum of 50000"):
            clf.fit(X, y)

    def test_too_many_samples_override(self):
        """ignore_pretraining_limits should bypass sample limit."""
        # Use a small amount over the limit to keep test fast
        n = TabPFN25Classifier.MAX_SAMPLES + 1
        X = np.random.randn(n, 3).astype(np.float32)
        y = np.random.randint(0, 2, n)
        clf = TabPFN25Classifier(ignore_pretraining_limits=True)
        # Should not raise
        clf.fit(X, y)
        assert clf.is_fitted_

    def test_too_many_features_raises(self):
        X = np.zeros((10, TabPFN25Classifier.MAX_FEATURES + 1))
        y = np.zeros(10)
        clf = TabPFN25Classifier()
        with pytest.raises(ValueError, match="maximum of 2000"):
            clf.fit(X, y)

    def test_too_many_classes_raises(self):
        X = np.random.randn(20, 3)
        y = np.arange(20)  # 20 > MAX_CLASSES=10
        clf = TabPFN25Classifier()
        with pytest.raises(ValueError, match="maximum of 10"):
            clf.fit(X, y)

    def test_check_constraints_valid(self, clf_data):
        X, y = clf_data
        result = TabPFN25Classifier.check_constraints(X, y)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_check_constraints_too_many_samples(self):
        X = np.zeros((TabPFN25Classifier.MAX_SAMPLES + 1, 3))
        y = np.zeros(TabPFN25Classifier.MAX_SAMPLES + 1)
        result = TabPFN25Classifier.check_constraints(X, y)
        assert result["valid"] is False
        assert len(result["issues"]) >= 1
        assert "samples" in result["issues"][0].lower()

    def test_check_constraints_too_many_features(self):
        X = np.zeros((10, TabPFN25Classifier.MAX_FEATURES + 1))
        y = np.zeros(10)
        result = TabPFN25Classifier.check_constraints(X, y)
        assert result["valid"] is False
        assert any("features" in iss.lower() for iss in result["issues"])

    def test_check_constraints_too_many_classes(self):
        X = np.zeros((20, 3))
        y = np.arange(20)
        result = TabPFN25Classifier.check_constraints(X, y)
        assert result["valid"] is False
        assert any("classes" in iss.lower() for iss in result["issues"])

    def test_binary_classification(self):
        X, y = make_classification(
            n_samples=200, n_features=10, n_classes=2, random_state=0
        )
        clf = TabPFN25Classifier()
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_sklearn_estimator_type(self):
        clf = TabPFN25Classifier()
        assert clf._estimator_type == "classifier"

    def test_get_params(self):
        clf = TabPFN25Classifier(n_estimators=4, device="cpu")
        params = clf.get_params()
        assert params["n_estimators"] == 4
        assert params["device"] == "cpu"
        assert params["model_version"] == "2.5_real"
        assert params["fit_mode"] == "fit_with_cache"

    def test_set_params(self):
        clf = TabPFN25Classifier()
        clf.set_params(n_estimators=32, model_version="2.5")
        assert clf.n_estimators == 32
        assert clf.model_version == "2.5"

    def test_string_labels(self):
        """Classifier should handle string labels."""
        X = np.random.randn(200, 10)
        y = np.array(["cat", "dog", "fish"] * 66 + ["cat", "dog"])
        clf = TabPFN25Classifier()
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset({"cat", "dog", "fish"})

    def test_model_version_parameter(self):
        """model_version should be stored and retrievable."""
        clf_real = TabPFN25Classifier(model_version="2.5_real")
        assert clf_real.model_version == "2.5_real"

        clf_synth = TabPFN25Classifier(model_version="2.5")
        assert clf_synth.model_version == "2.5"

    def test_higher_limits_than_v2(self):
        """TabPFN 2.5 should have higher limits than v2."""
        from endgame.models.tabular.tabpfn import TabPFNv2Classifier
        assert TabPFN25Classifier.MAX_SAMPLES > TabPFNv2Classifier.MAX_SAMPLES
        assert TabPFN25Classifier.MAX_FEATURES > TabPFNv2Classifier.MAX_FEATURES
        assert TabPFN25Classifier.MAX_CLASSES == TabPFNv2Classifier.MAX_CLASSES


# ---------------------------------------------------------------------------
# TabPFN25Regressor
# ---------------------------------------------------------------------------

class TestTabPFN25Regressor:
    """Tests for the v2.5 regressor (fallback path)."""

    def test_init_defaults(self):
        reg = TabPFN25Regressor()
        assert reg.n_estimators == 8
        assert reg.device == "auto"
        assert reg.random_state == 0
        assert reg.fit_mode == "fit_with_cache"
        assert reg.memory_saving_mode is False
        assert reg.ignore_pretraining_limits is False
        assert reg.categorical_features_indices is None
        assert reg.model_version == "2.5_real"

    def test_init_custom(self):
        reg = TabPFN25Regressor(
            n_estimators=16,
            device="cpu",
            random_state=42,
            categorical_features_indices=[1],
            fit_mode="fit_with_cache",
            memory_saving_mode=True,
            ignore_pretraining_limits=True,
            model_version="2.5",
        )
        assert reg.n_estimators == 16
        assert reg.device == "cpu"
        assert reg.random_state == 42
        assert reg.categorical_features_indices == [1]
        assert reg.fit_mode == "fit_with_cache"
        assert reg.memory_saving_mode is True
        assert reg.ignore_pretraining_limits is True
        assert reg.model_version == "2.5"

    def test_fit_predict(self, reg_data):
        X, y = reg_data
        reg = TabPFN25Regressor(random_state=0)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert preds.shape == (len(X),)
        assert preds.dtype == np.float64 or preds.dtype == np.float32

    def test_n_features_in(self, reg_data):
        X, y = reg_data
        reg = TabPFN25Regressor()
        reg.fit(X, y)
        assert reg.n_features_in_ == X.shape[1]

    def test_not_fitted_raises(self):
        reg = TabPFN25Regressor()
        with pytest.raises(Exception):
            reg.predict(np.zeros((2, 3)))

    def test_too_many_samples_raises(self):
        X = np.zeros((TabPFN25Regressor.MAX_SAMPLES + 1, 3))
        y = np.zeros(TabPFN25Regressor.MAX_SAMPLES + 1)
        reg = TabPFN25Regressor()
        with pytest.raises(ValueError, match="maximum of 50000"):
            reg.fit(X, y)

    def test_too_many_samples_override(self):
        n = TabPFN25Regressor.MAX_SAMPLES + 1
        X = np.random.randn(n, 3).astype(np.float32)
        y = np.random.randn(n).astype(np.float32)
        reg = TabPFN25Regressor(ignore_pretraining_limits=True)
        reg.fit(X, y)
        assert reg.is_fitted_

    def test_too_many_features_raises(self):
        X = np.zeros((10, TabPFN25Regressor.MAX_FEATURES + 1))
        y = np.zeros(10)
        reg = TabPFN25Regressor()
        with pytest.raises(ValueError, match="maximum of 2000"):
            reg.fit(X, y)

    def test_check_constraints_valid(self, reg_data):
        X, y = reg_data
        result = TabPFN25Regressor.check_constraints(X)
        assert result["valid"] is True
        assert result["issues"] == []

    def test_check_constraints_too_many_samples(self):
        X = np.zeros((TabPFN25Regressor.MAX_SAMPLES + 1, 3))
        result = TabPFN25Regressor.check_constraints(X)
        assert result["valid"] is False
        assert any("samples" in iss.lower() for iss in result["issues"])

    def test_check_constraints_too_many_features(self):
        X = np.zeros((10, TabPFN25Regressor.MAX_FEATURES + 1))
        result = TabPFN25Regressor.check_constraints(X)
        assert result["valid"] is False
        assert any("features" in iss.lower() for iss in result["issues"])

    def test_sklearn_estimator_type(self):
        reg = TabPFN25Regressor()
        assert reg._estimator_type == "regressor"

    def test_get_params(self):
        reg = TabPFN25Regressor(n_estimators=4, device="cpu")
        params = reg.get_params()
        assert params["n_estimators"] == 4
        assert params["device"] == "cpu"
        assert params["model_version"] == "2.5_real"
        assert params["fit_mode"] == "fit_with_cache"

    def test_set_params(self):
        reg = TabPFN25Regressor()
        reg.set_params(n_estimators=32, memory_saving_mode=True)
        assert reg.n_estimators == 32
        assert reg.memory_saving_mode is True

    def test_predictions_reasonable(self, reg_data):
        """Predictions should be finite and in a reasonable range."""
        X, y = reg_data
        reg = TabPFN25Regressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        assert np.all(np.isfinite(preds))

    def test_predictions_vary(self, reg_data):
        """Predictions should not be constant."""
        X, y = reg_data
        reg = TabPFN25Regressor()
        reg.fit(X, y)
        preds = reg.predict(X)
        assert np.std(preds) > 0

    def test_model_version_parameter(self):
        """model_version should be stored and retrievable."""
        reg_real = TabPFN25Regressor(model_version="2.5_real")
        assert reg_real.model_version == "2.5_real"

        reg_synth = TabPFN25Regressor(model_version="2.5")
        assert reg_synth.model_version == "2.5"

    def test_higher_limits_than_v2(self):
        """TabPFN 2.5 should have higher limits than v2."""
        from endgame.models.tabular.tabpfn import TabPFNv2Regressor
        assert TabPFN25Regressor.MAX_SAMPLES > TabPFNv2Regressor.MAX_SAMPLES
        assert TabPFN25Regressor.MAX_FEATURES > TabPFNv2Regressor.MAX_FEATURES


# ---------------------------------------------------------------------------
# Class-level constants
# ---------------------------------------------------------------------------

class TestConstants:
    """Verify the documented limits are set correctly."""

    def test_classifier_limits(self):
        assert TabPFN25Classifier.MAX_SAMPLES == 50_000
        assert TabPFN25Classifier.MAX_FEATURES == 2_000
        assert TabPFN25Classifier.MAX_CLASSES == 10

    def test_regressor_limits(self):
        assert TabPFN25Regressor.MAX_SAMPLES == 50_000
        assert TabPFN25Regressor.MAX_FEATURES == 2_000
