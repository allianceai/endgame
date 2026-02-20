"""Tests for TabM: Parameter-Efficient MLP Ensembling models."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def classification_data():
    """Small binary classification dataset for testing."""
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
    k=4,
    n_blocks=1,
    d_block=32,
    n_epochs=5,
    batch_size=64,
    patience=3,
    random_state=42,
)

SMALL_REG_KWARGS = dict(
    k=4,
    n_blocks=1,
    d_block=32,
    n_epochs=5,
    batch_size=64,
    patience=3,
    random_state=42,
)


# =============================================================================
# TabMClassifier Tests
# =============================================================================

class TestTabMClassifier:
    """Tests for TabMClassifier."""

    def test_import(self):
        """Test that TabM can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier
        assert TabMClassifier is not None

    def test_default_params(self):
        """Test default parameter values."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier()
        assert clf.k == 32
        assert clf.n_blocks == 3
        assert clf.d_block == 256
        assert clf.dropout == 0.0
        assert clf.use_embeddings is False
        assert clf.n_bins == 16
        assert clf.learning_rate == 1e-3
        assert clf.weight_decay == 1e-5
        assert clf.n_epochs == 100
        assert clf.batch_size == 256
        assert clf.patience == 16
        assert clf.val_size == 0.2
        assert clf.device == "auto"
        assert clf.random_state is None
        assert clf.verbose is False

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier(
            k=8,
            n_blocks=2,
            d_block=128,
            dropout=0.1,
            use_embeddings=True,
            n_bins=8,
            learning_rate=1e-4,
            weight_decay=1e-4,
            n_epochs=50,
            batch_size=128,
            patience=10,
            val_size=0.15,
            device="cpu",
            random_state=123,
            verbose=True,
        )
        assert clf.k == 8
        assert clf.n_blocks == 2
        assert clf.d_block == 128
        assert clf.dropout == 0.1
        assert clf.use_embeddings is True
        assert clf.n_bins == 8
        assert clf.learning_rate == 1e-4
        assert clf.random_state == 123

    def test_basic_fit_predict(self, classification_data):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == 200
        assert set(preds).issubset({0, 1})

    def test_predict_proba_shape(self, classification_data):
        """Test predict_proba returns correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_predict_proba_sums_to_one(self, classification_data):
        """Test that predicted probabilities sum to 1."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_range(self, classification_data):
        """Test that probabilities are in [0, 1]."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_classes_attribute(self, classification_data):
        """Test classes_ attribute is set correctly."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == {0, 1}

    def test_multiclass(self, multiclass_data):
        """Test multiclass classification."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = multiclass_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
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
        from endgame.models.tabular import TabMClassifier

        X, y = string_label_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert all(p in ("cat", "dog") for p in preds)
        assert set(clf.classes_) == {"cat", "dog"}

    def test_not_fitted_error(self):
        """Test that predict raises error before fitting."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        X_dummy = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict(X_dummy)

        with pytest.raises(RuntimeError, match="not been fitted"):
            clf.predict_proba(X_dummy)

    def test_feature_importances(self, classification_data):
        """Test feature_importances_ is computed after fit."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_ is not None
        assert clf.feature_importances_.shape == (10,)
        assert np.all(clf.feature_importances_ >= 0)
        assert np.isclose(clf.feature_importances_.sum(), 1.0, atol=1e-5)

    def test_get_params(self):
        """Test sklearn get_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier(k=8, n_blocks=2, random_state=42)
        params = clf.get_params()

        assert params["k"] == 8
        assert params["n_blocks"] == 2
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier()
        clf.set_params(k=8, n_blocks=2)

        assert clf.k == 8
        assert clf.n_blocks == 2

    def test_estimator_type(self):
        """Test sklearn estimator type."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        clf = TabMClassifier()
        assert clf._estimator_type == "classifier"

    def test_history(self, classification_data):
        """Test training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        assert "train_loss" in clf.history_
        assert "val_loss" in clf.history_
        assert len(clf.history_["train_loss"]) > 0
        assert len(clf.history_["val_loss"]) > 0

    def test_reproducibility(self, classification_data):
        """Test that random_state produces reproducible results."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data

        clf1 = TabMClassifier(**SMALL_CLF_KWARGS)
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)

        clf2 = TabMClassifier(**SMALL_CLF_KWARGS)
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)

        assert np.allclose(proba1, proba2, atol=1e-5)

    def test_fit_returns_self(self, classification_data):
        """Test that fit returns self for chaining."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        result = clf.fit(X, y)
        assert result is clf

    def test_ensemble_averaging(self, classification_data):
        """Test that ensemble members produce different outputs that get averaged."""
        pytest.importorskip("torch")
        import torch
        from endgame.models.tabular.tabm import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X, y)

        # Get raw logits from model to verify k members produce different outputs
        clf.model_.eval()
        X_scaled = clf._scaler.transform(X[:10].astype(np.float32))
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)
        x_t = torch.tensor(X_scaled, dtype=torch.float32).to(clf._device)

        with torch.no_grad():
            logits = clf.model_(x_t)  # (10, k, n_classes)

        assert logits.shape == (10, clf.k, clf.n_classes_)
        # Verify that different ensemble members produce different outputs
        # (at least some variation across k members)
        member_std = logits.std(dim=1)  # (10, n_classes)
        assert member_std.mean().item() > 0, "Ensemble members should produce different outputs"

    def test_with_embeddings(self, classification_data):
        """Test TabM with piecewise linear embeddings enabled."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(use_embeddings=True, n_bins=8, **{
            k: v for k, v in SMALL_CLF_KWARGS.items()
        })
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == 200

        proba = clf.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_pandas_input(self, classification_data):
        """Test that pandas DataFrame input works."""
        pytest.importorskip("torch")
        pd = pytest.importorskip("pandas")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        y_series = pd.Series(y)

        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X_df, y_series)

        preds = clf.predict(X_df)
        assert len(preds) == 200

    def test_verbose_output(self, classification_data, capsys):
        """Test verbose mode produces output."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        clf = TabMClassifier(
            verbose=True, n_epochs=15, **{
                k: v for k, v in SMALL_CLF_KWARGS.items()
                if k not in ("random_state", "n_epochs")
            }, random_state=42,
        )
        clf.fit(X, y)

        captured = capsys.readouterr()
        assert "[TabM]" in captured.out

    def test_eval_set(self, classification_data):
        """Test that providing eval_set works."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMClassifier

        X, y = classification_data
        # Split into train and val
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        clf = TabMClassifier(**SMALL_CLF_KWARGS)
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))

        preds = clf.predict(X_val)
        assert len(preds) == 50


# =============================================================================
# TabMRegressor Tests
# =============================================================================

class TestTabMRegressor:
    """Tests for TabMRegressor."""

    def test_import(self):
        """Test that TabMRegressor can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor
        assert TabMRegressor is not None

    def test_default_params(self):
        """Test default parameter values."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor()
        assert reg.k == 32
        assert reg.n_blocks == 3
        assert reg.d_block == 256
        assert reg.dropout == 0.0
        assert reg.learning_rate == 1e-3
        assert reg.weight_decay == 1e-5
        assert reg.n_epochs == 100
        assert reg.batch_size == 256
        assert reg.patience == 16
        assert reg.val_size == 0.2
        assert reg.device == "auto"
        assert reg.random_state is None

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor(k=8, n_blocks=2, d_block=64, random_state=123)
        assert reg.k == 8
        assert reg.n_blocks == 2
        assert reg.d_block == 64
        assert reg.random_state == 123

    def test_basic_fit_predict(self, regression_data):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert len(preds) == 200
        assert preds.dtype in (np.float32, np.float64)

    def test_predictions_finite(self, regression_data):
        """Test that predictions are finite."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert np.all(np.isfinite(preds))

    def test_not_fitted_error(self):
        """Test that predict raises error before fitting."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor(**SMALL_REG_KWARGS)
        X_dummy = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="not been fitted"):
            reg.predict(X_dummy)

    def test_estimator_type(self):
        """Test sklearn estimator type."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor()
        assert reg._estimator_type == "regressor"

    def test_feature_importances(self, regression_data):
        """Test feature_importances_ is computed after fit."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_ is not None
        assert reg.feature_importances_.shape == (10,)
        assert np.all(reg.feature_importances_ >= 0)
        assert np.isclose(reg.feature_importances_.sum(), 1.0, atol=1e-5)

    def test_get_params(self):
        """Test sklearn get_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor(k=8, d_block=64, random_state=42)
        params = reg.get_params()

        assert params["k"] == 8
        assert params["d_block"] == 64
        assert params["random_state"] == 42

    def test_set_params(self):
        """Test sklearn set_params."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        reg = TabMRegressor()
        reg.set_params(k=8, d_block=64)

        assert reg.k == 8
        assert reg.d_block == 64

    def test_history(self, regression_data):
        """Test training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(**SMALL_REG_KWARGS)
        reg.fit(X, y)

        assert "train_loss" in reg.history_
        assert "val_loss" in reg.history_
        assert len(reg.history_["train_loss"]) > 0
        assert len(reg.history_["val_loss"]) > 0

    def test_fit_returns_self(self, regression_data):
        """Test that fit returns self for chaining."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(**SMALL_REG_KWARGS)
        result = reg.fit(X, y)
        assert result is reg

    def test_reproducibility(self, regression_data):
        """Test that random_state produces reproducible results."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data

        reg1 = TabMRegressor(**SMALL_REG_KWARGS)
        reg1.fit(X, y)
        preds1 = reg1.predict(X)

        reg2 = TabMRegressor(**SMALL_REG_KWARGS)
        reg2.fit(X, y)
        preds2 = reg2.predict(X)

        assert np.allclose(preds1, preds2, atol=1e-4)

    def test_eval_set(self, regression_data):
        """Test that providing eval_set works."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        X_train, X_val = X[:150], X[150:]
        y_train, y_val = y[:150], y[150:]

        reg = TabMRegressor(**SMALL_REG_KWARGS)
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))

        preds = reg.predict(X_val)
        assert len(preds) == 50
        assert np.all(np.isfinite(preds))

    def test_verbose_mode(self, regression_data, capsys):
        """Test verbose output."""
        pytest.importorskip("torch")
        from endgame.models.tabular import TabMRegressor

        X, y = regression_data
        reg = TabMRegressor(verbose=True, n_epochs=15, **{
            k: v for k, v in SMALL_REG_KWARGS.items()
            if k not in ("random_state", "n_epochs")
        }, random_state=42)
        reg.fit(X, y)

        captured = capsys.readouterr()
        assert "[TabM]" in captured.out
