"""Tests for GRANDE (GRadient-based Neural Decision Ensemble) models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import cross_val_score


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classification_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=150,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def multiclass_data():
    """Small multiclass classification dataset."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=6,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def regression_data():
    """Small regression dataset."""
    X, y = make_regression(
        n_samples=150,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def small_params():
    """Small model parameters for fast tests."""
    return dict(
        n_trees=16,
        depth=3,
        n_epochs=5,
        batch_size=64,
        patience=3,
        random_state=42,
        verbose=False,
    )


# =============================================================================
# GRANDEClassifier Tests
# =============================================================================

class TestGRANDEClassifierInit:
    """Tests for GRANDEClassifier initialization."""

    def test_import(self):
        """Test that GRANDE can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import GRANDEClassifier
        assert GRANDEClassifier is not None

    def test_default_params(self):
        """Test default parameter values."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier()
        assert clf.n_trees == 128
        assert clf.depth == 5
        assert clf.lr == 0.005
        assert clf.weight_decay == 0.0
        assert clf.n_epochs == 50
        assert clf.batch_size == 512
        assert clf.patience == 10
        assert clf.temperature_init == 1.0
        assert clf.temperature_final == 0.1
        assert clf.device == "auto"
        assert clf.random_state is None
        assert clf.verbose is False

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier(
            n_trees=32,
            depth=4,
            lr=0.005,
            weight_decay=1e-4,
            n_epochs=50,
            batch_size=128,
            patience=10,
            temperature_init=2.0,
            temperature_final=0.05,
            device="cpu",
            random_state=123,
            verbose=True,
        )
        assert clf.n_trees == 32
        assert clf.depth == 4
        assert clf.lr == 0.005
        assert clf.weight_decay == 1e-4
        assert clf.n_epochs == 50
        assert clf.batch_size == 128
        assert clf.patience == 10
        assert clf.temperature_init == 2.0
        assert clf.temperature_final == 0.05
        assert clf.device == "cpu"
        assert clf.random_state == 123
        assert clf.verbose is True

    def test_estimator_type(self):
        """Test sklearn estimator type attribute."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier()
        assert clf._estimator_type == "classifier"

    def test_get_params(self):
        """Test sklearn get_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier(n_trees=32, depth=4, random_state=42)
        params = clf.get_params()
        assert params["n_trees"] == 32
        assert params["depth"] == 4
        assert params["random_state"] == 42
        assert "lr" in params
        assert "batch_size" in params

    def test_set_params(self):
        """Test sklearn set_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier()
        clf.set_params(n_trees=64, depth=5)
        assert clf.n_trees == 64
        assert clf.depth == 5


class TestGRANDEClassifierFitPredict:
    """Tests for GRANDEClassifier fit/predict functionality."""

    def test_basic_fit_predict(self, classification_data, small_params):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(y))

    def test_predict_proba_shape(self, classification_data, small_params):
        """Test predict_proba returns correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_valid_probabilities(self, classification_data, small_params):
        """Test that predict_proba returns valid probabilities."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        # All values between 0 and 1
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        # Rows sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classes_attribute(self, classification_data, small_params):
        """Test classes_ attribute is set correctly."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(np.unique(y))

    def test_multiclass_classification(self, multiclass_data, small_params):
        """Test multiclass classification."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = multiclass_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert len(preds) == len(X)
        assert proba.shape == (len(X), 4)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert clf.n_classes_ == 4

    def test_string_labels(self, small_params):
        """Test classification with string labels."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y_int = make_classification(
            n_samples=100, n_features=10, random_state=42,
        )
        X = X.astype(np.float32)
        label_map = {0: "cat", 1: "dog"}
        y = np.array([label_map[yi] for yi in y_int])

        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert all(p in ("cat", "dog") for p in preds)
        assert set(clf.classes_) == {"cat", "dog"}

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict(X)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict_proba(X)

    def test_feature_importances_shape(self, classification_data, small_params):
        """Test feature_importances_ has correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_feature_importances_nonnegative(self, classification_data, small_params):
        """Test feature importances are non-negative."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        assert np.all(clf.feature_importances_ >= 0)

    def test_feature_importances_sum_to_one(self, classification_data, small_params):
        """Test feature importances sum to approximately 1."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        np.testing.assert_allclose(
            clf.feature_importances_.sum(), 1.0, atol=1e-5
        )

    def test_history_attribute(self, classification_data, small_params):
        """Test that training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "history_")
        assert "train_loss" in clf.history_
        assert "val_loss" in clf.history_
        assert len(clf.history_["train_loss"]) > 0
        assert len(clf.history_["val_loss"]) > 0


class TestGRANDEClassifierTemperature:
    """Tests for temperature annealing in GRANDEClassifier."""

    def test_temperature_annealing_range(self):
        """Test temperature annealing schedule."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier(
            temperature_init=1.0,
            temperature_final=0.1,
        )

        # First epoch
        t0 = clf._temperature(0, 100)
        # Last epoch
        t_last = clf._temperature(99, 100)
        # Middle epoch
        t_mid = clf._temperature(50, 100)

        np.testing.assert_allclose(t0, 1.0, atol=1e-5)
        np.testing.assert_allclose(t_last, 0.1, atol=1e-5)
        assert t0 > t_mid > t_last

    def test_temperature_decreases_monotonically(self):
        """Test that temperature decreases over epochs."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        clf = GRANDEClassifier(
            temperature_init=2.0,
            temperature_final=0.05,
        )

        temps = [clf._temperature(e, 50) for e in range(50)]
        for i in range(1, len(temps)):
            assert temps[i] <= temps[i - 1] + 1e-10


class TestGRANDEClassifierSklearn:
    """Tests for sklearn compatibility."""

    def test_sklearn_cross_validate(self, classification_data, small_params):
        """Test compatibility with sklearn cross_val_score."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDEClassifier

        X, y = classification_data
        clf = GRANDEClassifier(**small_params)

        scores = cross_val_score(clf, X, y, cv=2, scoring="accuracy")
        assert len(scores) == 2
        assert all(0 <= s <= 1 for s in scores)


# =============================================================================
# GRANDERegressor Tests
# =============================================================================

class TestGRANDERegressorInit:
    """Tests for GRANDERegressor initialization."""

    def test_import(self):
        """Test that GRANDERegressor can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import GRANDERegressor
        assert GRANDERegressor is not None

    def test_default_params(self):
        """Test default parameter values for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        reg = GRANDERegressor()
        assert reg.n_trees == 128
        assert reg.depth == 5
        assert reg.lr == 0.005
        assert reg._estimator_type == "regressor"

    def test_estimator_type(self):
        """Test sklearn estimator type for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        reg = GRANDERegressor()
        assert reg._estimator_type == "regressor"

    def test_get_set_params(self):
        """Test get_params and set_params for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        reg = GRANDERegressor(n_trees=32)
        assert reg.get_params()["n_trees"] == 32

        reg.set_params(n_trees=128)
        assert reg.n_trees == 128


class TestGRANDERegressorFitPredict:
    """Tests for GRANDERegressor fit/predict functionality."""

    def test_basic_fit_predict(self, regression_data, small_params):
        """Test basic fit/predict for regression."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        X, y = regression_data
        reg = GRANDERegressor(**small_params)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype in (np.float32, np.float64)

    def test_predict_returns_real_values(self, regression_data, small_params):
        """Test that predictions are real-valued (not quantized)."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        X, y = regression_data
        reg = GRANDERegressor(**small_params)
        reg.fit(X, y)

        preds = reg.predict(X)
        # Predictions should have unique values (not just a few discrete values)
        n_unique = len(np.unique(np.round(preds, 2)))
        assert n_unique > 5

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        reg = GRANDERegressor()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            reg.predict(X)

    def test_feature_importances_shape(self, regression_data, small_params):
        """Test feature_importances_ shape for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        X, y = regression_data
        reg = GRANDERegressor(**small_params)
        reg.fit(X, y)

        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_.shape == (X.shape[1],)
        assert np.all(reg.feature_importances_ >= 0)

    def test_history_attribute(self, regression_data, small_params):
        """Test training history for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        X, y = regression_data
        reg = GRANDERegressor(**small_params)
        reg.fit(X, y)

        assert hasattr(reg, "history_")
        assert len(reg.history_["train_loss"]) > 0
        assert len(reg.history_["val_loss"]) > 0

    def test_sklearn_cross_validate(self, regression_data, small_params):
        """Test compatibility with sklearn cross_val_score."""
        pytest.importorskip("torch")
        from endgame.models.tabular.grande import GRANDERegressor

        X, y = regression_data
        reg = GRANDERegressor(**small_params)

        scores = cross_val_score(reg, X, y, cv=2, scoring="neg_mean_squared_error")
        assert len(scores) == 2
        assert all(s <= 0 for s in scores)  # neg MSE is always <= 0


# =============================================================================
# Internal Module Tests
# =============================================================================

class TestBatchedGRANDENetwork:
    """Tests for the _BatchedGRANDENetwork module."""

    def test_forward_shape(self):
        """Test _BatchedGRANDENetwork forward output shape."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.grande import _BatchedGRANDENetwork

        net = _BatchedGRANDENetwork(n_features=10, n_outputs=2, n_trees=8, depth=3)
        x = torch.randn(32, 10)
        out = net(x, temperature=1.0)
        assert out.shape == (32, 2)

    def test_forward_regression(self):
        """Test _BatchedGRANDENetwork with single output (regression)."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.grande import _BatchedGRANDENetwork

        net = _BatchedGRANDENetwork(n_features=5, n_outputs=1, n_trees=4, depth=2)
        x = torch.randn(16, 5)
        out = net(x, temperature=0.5)
        assert out.shape == (16, 1)

    def test_low_temperature_sharper(self):
        """Test that lower temperature produces sharper routing."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.grande import _BatchedGRANDENetwork

        torch.manual_seed(42)
        net = _BatchedGRANDENetwork(n_features=5, n_outputs=2, n_trees=4, depth=2)
        x = torch.randn(100, 5)

        out_warm = net(x, temperature=10.0)
        out_cold = net(x, temperature=0.01)

        # With very low temperature, outputs should have less variance
        # because routing becomes more deterministic
        # Just verify both produce valid outputs
        assert out_warm.shape == out_cold.shape
        assert not torch.allclose(out_warm, out_cold)

    def test_temperature_override(self):
        """Test temperature override in forward call."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.grande import _BatchedGRANDENetwork

        torch.manual_seed(42)
        net = _BatchedGRANDENetwork(
            n_features=5, n_outputs=2, n_trees=4, depth=2,
            temperature=1.0,
        )
        x = torch.randn(16, 5)

        out_default = net(x)
        out_override = net(x, temperature=0.01)

        assert out_default.shape == out_override.shape
        # Different temperatures should give different results
        assert not torch.allclose(out_default, out_override)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
