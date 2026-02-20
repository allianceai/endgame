"""Tests for xRFM (Recursive Feature Machines) wrapper.

Tests both classifier and regressor using the kNN fallback path
(xrfm package not required to be installed).
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_redundant=2, random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def multiclass_data():
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=4, n_clusters_per_class=1, random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------


class TestAvailability:
    def test_check_returns_bool(self):
        from endgame.models.tabular.xrfm import _check_xrfm_available
        result = _check_xrfm_available()
        assert isinstance(result, bool)

    def test_import_from_init(self):
        from endgame.models.tabular import xRFMClassifier, xRFMRegressor
        assert xRFMClassifier is not None
        assert xRFMRegressor is not None


# ---------------------------------------------------------------------------
# xRFMClassifier
# ---------------------------------------------------------------------------


class TestXRFMClassifierInit:
    def test_import(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        assert xRFMClassifier is not None

    def test_default_params(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier()
        assert clf.kernel == "laplace"
        assert clf.bandwidth == 5.0
        assert clf.reg == 1e-3
        assert clf.iters == 5
        assert clf.diag is True
        assert clf.min_subset_size == 10_000
        assert clf.split_method == "top_vector_agop_on_subset"
        assert clf.M_batch_size == 1000
        assert clf.early_stop_rfm is True
        assert clf.val_size == 0.2
        assert clf.device == "auto"
        assert clf.random_state is None
        assert clf.verbose is False

    def test_custom_params(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier(
            kernel="l2", bandwidth=10.0, reg=1e-2, iters=3,
            diag=False, min_subset_size=5000,
            split_method="random_pca", device="cpu",
            random_state=123, verbose=True,
        )
        assert clf.kernel == "l2"
        assert clf.bandwidth == 10.0
        assert clf.reg == 1e-2
        assert clf.iters == 3
        assert clf.diag is False
        assert clf.min_subset_size == 5000
        assert clf.split_method == "random_pca"
        assert clf.device == "cpu"
        assert clf.random_state == 123
        assert clf.verbose is True

    def test_sklearn_estimator_type(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier()
        assert clf._estimator_type == "classifier"

    def test_get_params(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier(kernel="l2", iters=3)
        params = clf.get_params()
        assert params["kernel"] == "l2"
        assert params["iters"] == 3

    def test_set_params(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier()
        clf.set_params(kernel="l2", bandwidth=10.0)
        assert clf.kernel == "l2"
        assert clf.bandwidth == 10.0


class TestXRFMClassifierFitPredict:
    def test_fit_returns_self(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        result = clf.fit(X, y)
        assert result is clf

    def test_classes_attribute(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2

    def test_n_classes_attribute(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.n_classes_ == 2

    def test_n_features_in(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        assert clf.n_features_in_ == 10

    def test_predict_shape(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X[:20])
        assert preds.shape == (20,)

    def test_predict_proba_shape(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:20])
        assert proba.shape == (20, 2)

    def test_predict_proba_sums_to_one(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:20])
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_nonnegative(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:20])
        assert (proba >= 0).all()

    def test_predicted_labels_in_classes(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert set(preds).issubset(set(clf.classes_))

    def test_not_fitted_raises(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier()
        with pytest.raises(Exception):
            clf.predict(np.zeros((5, 10)))

    def test_not_fitted_proba_raises(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier()
        with pytest.raises(Exception):
            clf.predict_proba(np.zeros((5, 10)))

    def test_multiclass(self, multiclass_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = multiclass_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        proba = clf.predict_proba(X[:20])
        assert proba.shape == (20, 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_string_labels(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        y_str = np.array(["cat" if v == 0 else "dog" for v in y])
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y_str)
        preds = clf.predict(X[:10])
        assert all(p in ("cat", "dog") for p in preds)

    def test_feature_importances(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        X, y = classification_data
        clf = xRFMClassifier(random_state=42)
        clf.fit(X, y)
        importances = clf.feature_importances_
        assert importances.shape == (10,)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)
        assert (importances >= 0).all()

    def test_pandas_input(self, classification_data):
        from endgame.models.tabular.xrfm import xRFMClassifier
        pd = pytest.importorskip("pandas")
        X, y = classification_data
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        clf = xRFMClassifier(random_state=42)
        clf.fit(df, y)
        preds = clf.predict(df[:10])
        assert preds.shape == (10,)

    def test_build_rfm_params(self):
        from endgame.models.tabular.xrfm import xRFMClassifier
        clf = xRFMClassifier(
            kernel="l2", bandwidth=10.0, reg=1e-2, iters=3,
            diag=False, M_batch_size=500, verbose=True,
            early_stop_rfm=False,
        )
        params = clf._build_rfm_params()
        assert params["model"]["kernel"] == "l2"
        assert params["model"]["bandwidth"] == 10.0
        assert params["model"]["diag"] is False
        assert params["fit"]["reg"] == 1e-2
        assert params["fit"]["iters"] == 3
        assert params["fit"]["M_batch_size"] == 500
        assert params["fit"]["verbose"] is True
        assert params["fit"]["early_stop_rfm"] is False


# ---------------------------------------------------------------------------
# xRFMRegressor
# ---------------------------------------------------------------------------


class TestXRFMRegressorInit:
    def test_import(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        assert xRFMRegressor is not None

    def test_default_params(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor()
        assert reg.kernel == "laplace"
        assert reg.bandwidth == 5.0
        assert reg.reg == 1e-3
        assert reg.iters == 5
        assert reg.diag is True
        assert reg.device == "auto"

    def test_custom_params(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor(
            kernel="l2", bandwidth=10.0, reg=1e-2, iters=3,
            device="cpu", random_state=123,
        )
        assert reg.kernel == "l2"
        assert reg.bandwidth == 10.0
        assert reg.random_state == 123

    def test_sklearn_estimator_type(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor()
        assert reg._estimator_type == "regressor"

    def test_get_params(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor(kernel="l2", iters=3)
        params = reg.get_params()
        assert params["kernel"] == "l2"
        assert params["iters"] == 3

    def test_set_params(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor()
        reg.set_params(kernel="l2", bandwidth=10.0)
        assert reg.kernel == "l2"
        assert reg.bandwidth == 10.0


class TestXRFMRegressorFitPredict:
    def test_fit_returns_self(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        result = reg.fit(X, y)
        assert result is reg

    def test_n_features_in(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        assert reg.n_features_in_ == 10

    def test_predict_shape(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X[:20])
        assert preds.shape == (20,)

    def test_predictions_finite(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X[:20])
        assert np.all(np.isfinite(preds))

    def test_predictions_vary(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X[:20])
        assert np.std(preds) > 0

    def test_not_fitted_raises(self):
        from endgame.models.tabular.xrfm import xRFMRegressor
        reg = xRFMRegressor()
        with pytest.raises(Exception):
            reg.predict(np.zeros((5, 10)))

    def test_feature_importances(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        importances = reg.feature_importances_
        assert importances.shape == (10,)
        np.testing.assert_allclose(importances.sum(), 1.0, atol=1e-6)

    def test_pandas_input(self, regression_data):
        from endgame.models.tabular.xrfm import xRFMRegressor
        pd = pytest.importorskip("pandas")
        X, y = regression_data
        df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        reg = xRFMRegressor(random_state=42)
        reg.fit(df, y)
        preds = reg.predict(df[:10])
        assert preds.shape == (10,)

    def test_target_inverse_scaling(self, regression_data):
        """Predictions should be in original target scale, not standardized."""
        from endgame.models.tabular.xrfm import xRFMRegressor
        X, y = regression_data
        reg = xRFMRegressor(random_state=42)
        reg.fit(X, y)
        preds = reg.predict(X[:20])
        # Predictions should be in a similar range to y
        assert np.abs(preds.mean()) < np.abs(y).max() * 5
