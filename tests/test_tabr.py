"""Tests for TabR: Retrieval-Augmented Tabular Deep Learning models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classification_data():
    """Small classification dataset for testing."""
    X, y = make_classification(
        n_samples=200,
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
        n_informative=5,
        n_redundant=2,
        n_classes=4,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X.astype(np.float32), y


@pytest.fixture
def regression_data():
    """Small regression dataset for testing."""
    X, y = make_regression(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def string_label_data():
    """Classification data with string labels."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    y_str = np.array(["cat" if label == 0 else "dog" for label in y])
    return X.astype(np.float32), y_str


# Small model kwargs for fast tests
SMALL_CLF_KWARGS = dict(
    d_model=32,
    n_heads=4,
    n_layers=1,
    k_neighbors=8,
    n_epochs=5,
    batch_size=64,
    patience=3,
    random_state=42,
)

SMALL_REG_KWARGS = dict(
    d_model=32,
    n_heads=4,
    n_layers=1,
    k_neighbors=8,
    n_epochs=5,
    batch_size=64,
    patience=3,
    random_state=42,
)


# =============================================================================
# TabRClassifier Tests
# =============================================================================

class TestTabRClassifier:
    """Tests for TabRClassifier."""

    def test_import(self):
        """Test that TabR can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier
        assert TabRClassifier is not None

    def test_default_params(self):
        """Test default parameter values."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier()
        assert clf.d_model == 192
        assert clf.n_heads == 8
        assert clf.n_layers == 2
        assert clf.k_neighbors == 96
        assert clf.dropout == 0.0
        assert clf.lr == 1e-4
        assert clf.weight_decay == 1e-5
        assert clf.n_epochs == 100
        assert clf.batch_size == 256
        assert clf.patience == 16
        assert clf.device == "auto"
        assert clf.random_state is None

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier(
            d_model=64,
            n_heads=4,
            n_layers=3,
            k_neighbors=32,
            dropout=0.1,
            lr=1e-3,
            weight_decay=1e-4,
            n_epochs=50,
            batch_size=128,
            patience=10,
            device="cpu",
            random_state=123,
        )
        assert clf.d_model == 64
        assert clf.n_heads == 4
        assert clf.n_layers == 3
        assert clf.k_neighbors == 32
        assert clf.dropout == 0.1
        assert clf.lr == 1e-3
        assert clf.random_state == 123

    def test_basic_fit_predict(self, classification_data):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == 200
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data):
        """Test predict_proba returns correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_predict_proba_sums_to_one(self, classification_data):
        """Test that predicted probabilities sum to 1."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_range(self, classification_data):
        """Test that probabilities are in [0, 1]."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_classes_attribute(self, classification_data):
        """Test classes_ attribute is set correctly."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == {0, 1}

    def test_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = multiclass_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        preds = clf.predict(X)
        proba = clf.predict_proba(X)

        assert len(preds) == 200
        assert proba.shape == (200, 4)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
        assert clf.n_classes_ == 4

    def test_string_labels(self, string_label_data):
        """Test with string class labels."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = string_label_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert all(p in ("cat", "dog") for p in preds)
        assert set(clf.classes_) == {"cat", "dog"}

    def test_not_fitted_error(self):
        """Test that predict raises error before fitting."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        X_dummy = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(X_dummy)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(X_dummy)

    def test_device_cpu(self, classification_data):
        """Test explicit CPU device."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(device="cpu", **{
            k: v for k, v in SMALL_CLF_KWARGS.items() if k != "random_state"
        }, random_state=42)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == 200

    def test_device_auto(self, classification_data):
        """Test auto device selection."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        import torch
        if torch.cuda.is_available():
            assert clf._device.type == "cuda"
        else:
            assert clf._device.type == "cpu"

    def test_get_params(self):
        """Test sklearn get_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier(d_model=64, n_heads=4, random_state=42)
        params = clf.get_params()

        assert params["d_model"] == 64
        assert params["n_heads"] == 4
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier()
        clf.set_params(d_model=64, n_heads=4)

        assert clf.d_model == 64
        assert clf.n_heads == 4

    def test_estimator_type(self):
        """Test sklearn estimator type."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        clf = TabRClassifier()
        assert clf._estimator_type == "classifier"

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ is computed after fit."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_ is not None
        assert clf.feature_importances_.shape == (10,)
        assert np.all(clf.feature_importances_ >= 0)
        assert np.isclose(clf.feature_importances_.sum(), 1.0, atol=1e-5)

    def test_history(self, classification_data):
        """Test training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert "train_loss" in clf.history_
        assert "val_loss" in clf.history_
        assert len(clf.history_["train_loss"]) > 0
        assert len(clf.history_["val_loss"]) > 0

    def test_fit_returns_self(self, classification_data):
        """Test that fit returns self for chaining."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        result = clf.fit(X, y)
        assert result is clf

    def test_reproducibility(self, classification_data):
        """Test that random_state produces reproducible results."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data

        clf1 = TabRClassifier(**SMALL_CLF_KWARGS)
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)

        clf2 = TabRClassifier(**SMALL_CLF_KWARGS)
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)

        assert np.allclose(proba1, proba2, atol=1e-5)

    def test_pandas_input(self, classification_data):
        """Test that pandas DataFrame input works."""
        pytest.importorskip("torch")
        pd = pytest.importorskip("pandas")
        from endgame.models.tabular import TabRClassifier

        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        clf = TabRClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X_df, y_series)

        preds = clf.predict(X_df)
        assert len(preds) == 200


# =============================================================================
# TabRRegressor Tests
# =============================================================================

class TestTabRRegressor:
    """Tests for TabRRegressor."""

    def test_import(self):
        """Test that TabRRegressor can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor
        assert TabRRegressor is not None

    def test_default_params(self):
        """Test default parameter values."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        reg = TabRRegressor()
        assert reg.d_model == 192
        assert reg.n_heads == 8
        assert reg.n_layers == 2
        assert reg.k_neighbors == 96
        assert reg.dropout == 0.0
        assert reg.lr == 1e-4
        assert reg.weight_decay == 1e-5
        assert reg.n_epochs == 100
        assert reg.batch_size == 256
        assert reg.patience == 16

    def test_basic_fit_predict(self, regression_data):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert len(preds) == 200
        assert preds.dtype in (np.float32, np.float64)

    def test_predictions_finite(self, regression_data):
        """Test that predictions are finite."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_error(self):
        """Test that predict raises error before fitting."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        reg = TabRRegressor(**SMALL_REG_KWARGS)
        X_dummy = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            reg.predict(X_dummy)

    def test_estimator_type(self):
        """Test sklearn estimator type."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        reg = TabRRegressor()
        assert reg._estimator_type == "regressor"

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ is computed after fit."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_ is not None
        assert reg.feature_importances_.shape == (10,)
        assert np.all(reg.feature_importances_ >= 0)
        assert np.isclose(reg.feature_importances_.sum(), 1.0, atol=1e-5)

    def test_get_params(self):
        """Test sklearn get_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        reg = TabRRegressor(d_model=64, k_neighbors=32, random_state=42)
        params = reg.get_params()

        assert params["d_model"] == 64
        assert params["k_neighbors"] == 32
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        reg = TabRRegressor()
        reg.set_params(d_model=64, k_neighbors=32)

        assert reg.d_model == 64
        assert reg.k_neighbors == 32

    def test_history(self, regression_data):
        """Test training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        assert "train_loss" in reg.history_
        assert "val_loss" in reg.history_
        assert len(reg.history_["train_loss"]) > 0
        assert len(reg.history_["val_loss"]) > 0

    def test_fit_returns_self(self, regression_data):
        """Test that fit returns self for chaining."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(**SMALL_REG_KWARGS)
        result = reg.fit(X, y)
        assert result is reg

    def test_verbose_mode(self, regression_data, capsys):
        """Test verbose output."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabRRegressor

        X, y = regression_data
        reg = TabRRegressor(verbose=True, n_epochs=15, **{
            k: v for k, v in SMALL_REG_KWARGS.items()
            if k not in ("random_state", "n_epochs")
        }, random_state=42)
        reg.fit(X, y)

        captured = capsys.readouterr()
        assert "[TabR]" in captured.out
