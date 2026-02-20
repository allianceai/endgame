"""Tests for RealMLP (Holzmuller et al., NeurIPS 2024) models."""

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
        n_samples=200,
        n_features=10,
        n_informative=5,
        random_state=42,
    )
    return X.astype(np.float32), y.astype(np.float32)


@pytest.fixture
def small_params():
    """Small model parameters for fast testing."""
    return dict(
        n_blocks=1,
        d_block=32,
        n_epochs=5,
        n_bins=4,
        batch_size=64,
        early_stopping=3,
        random_state=42,
        verbose=False,
    )


# =============================================================================
# RealMLPClassifier Tests
# =============================================================================


class TestRealMLPClassifierInit:
    """Tests for RealMLPClassifier initialization."""

    def test_import(self):
        """Test that RealMLPClassifier can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import RealMLPClassifier
        assert RealMLPClassifier is not None

    def test_default_params(self):
        """Test default parameter values match meta-tuned defaults."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier()
        assert clf.n_blocks == 3
        assert clf.d_block == 256
        assert clf.dropout == 0.15
        assert clf.learning_rate == 0.04
        assert clf.weight_decay == 0.0
        assert clf.n_epochs == 256
        assert clf.batch_size == 256
        assert clf.smooth_clip_c == 3.0
        assert clf.use_embeddings is True
        assert clf.n_bins == 16
        assert clf.warmup_fraction == 0.1
        assert clf.early_stopping == 20
        assert clf.device == "auto"
        assert clf.random_state is None
        assert clf.verbose is False

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier(
            n_blocks=2,
            d_block=128,
            dropout=0.3,
            learning_rate=0.01,
            weight_decay=1e-4,
            n_epochs=50,
            batch_size=128,
            smooth_clip_c=5.0,
            use_embeddings=False,
            n_bins=8,
            warmup_fraction=0.2,
            early_stopping=10,
            device="cpu",
            random_state=123,
            verbose=True,
        )
        assert clf.n_blocks == 2
        assert clf.d_block == 128
        assert clf.dropout == 0.3
        assert clf.learning_rate == 0.01
        assert clf.weight_decay == 1e-4
        assert clf.n_epochs == 50
        assert clf.batch_size == 128
        assert clf.smooth_clip_c == 5.0
        assert clf.use_embeddings is False
        assert clf.n_bins == 8
        assert clf.warmup_fraction == 0.2
        assert clf.early_stopping == 10
        assert clf.device == "cpu"
        assert clf.random_state == 123
        assert clf.verbose is True

    def test_estimator_type(self):
        """Test sklearn estimator type attribute."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier()
        assert clf._estimator_type == "classifier"

    def test_get_params(self):
        """Test sklearn get_params compatibility."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier(n_blocks=2, d_block=64, random_state=42)
        params = clf.get_params()
        assert params["n_blocks"] == 2
        assert params["d_block"] == 64
        assert params["random_state"] == 42
        assert "learning_rate" in params
        assert "smooth_clip_c" in params

    def test_set_params(self):
        """Test sklearn set_params compatibility."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier()
        clf.set_params(n_blocks=5, d_block=512)
        assert clf.n_blocks == 5
        assert clf.d_block == 512


class TestRealMLPClassifierFitPredict:
    """Tests for RealMLPClassifier fit/predict functionality."""

    def test_basic_fit_predict(self, classification_data, small_params):
        """Test basic fit/predict cycle."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == len(X)
        assert set(preds).issubset(set(y))

    def test_predict_proba_shape(self, classification_data, small_params):
        """Test predict_proba returns correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)

    def test_predict_proba_sums_to_one(self, classification_data, small_params):
        """Test that predict_proba rows sum to 1."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_predict_proba_range(self, classification_data, small_params):
        """Test that predict_proba values are in [0, 1]."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_classes_attribute(self, classification_data, small_params):
        """Test classes_ attribute is set correctly after fit."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "classes_")
        assert len(clf.classes_) == 2
        assert set(clf.classes_) == set(np.unique(y))

    def test_multiclass_classification(self, multiclass_data, small_params):
        """Test multiclass classification."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = multiclass_data
        clf = RealMLPClassifier(**small_params)
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
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y_int = make_classification(
            n_samples=100, n_features=10, random_state=42,
        )
        X = X.astype(np.float32)
        label_map = {0: "cat", 1: "dog"}
        y = np.array([label_map[yi] for yi in y_int])

        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert all(p in ("cat", "dog") for p in preds)
        assert set(clf.classes_) == {"cat", "dog"}

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict(X)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict_proba(X)

    def test_feature_importances_shape(self, classification_data, small_params):
        """Test feature_importances_ has correct shape."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_feature_importances_nonnegative(self, classification_data, small_params):
        """Test feature importances are non-negative."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        assert np.all(clf.feature_importances_ >= 0)

    def test_feature_importances_sum_to_one(self, classification_data, small_params):
        """Test feature importances sum to approximately 1."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        np.testing.assert_allclose(
            clf.feature_importances_.sum(), 1.0, atol=1e-5
        )

    def test_feature_importances_not_fitted(self):
        """Test feature_importances_ raises error when not fitted."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        clf = RealMLPClassifier()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            _ = clf.feature_importances_

    def test_history_attribute(self, classification_data, small_params):
        """Test that training history is recorded."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        clf = RealMLPClassifier(**small_params)
        clf.fit(X, y)

        assert hasattr(clf, "history_")
        assert "train_loss" in clf.history_
        assert "val_loss" in clf.history_
        assert len(clf.history_["train_loss"]) > 0
        assert len(clf.history_["val_loss"]) > 0

    def test_no_embeddings(self, classification_data, small_params):
        """Test training without piecewise-linear embeddings."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data
        params = {**small_params, "use_embeddings": False}
        clf = RealMLPClassifier(**params)
        clf.fit(X, y)

        preds = clf.predict(X)
        assert len(preds) == len(X)
        # feature importances should still work
        assert clf.feature_importances_.shape == (X.shape[1],)

    def test_reproducibility(self, classification_data, small_params):
        """Test that results are reproducible with the same seed."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPClassifier

        X, y = classification_data

        clf1 = RealMLPClassifier(**small_params)
        clf1.fit(X, y)
        proba1 = clf1.predict_proba(X)

        clf2 = RealMLPClassifier(**small_params)
        clf2.fit(X, y)
        proba2 = clf2.predict_proba(X)

        np.testing.assert_allclose(proba1, proba2, atol=1e-5)


# =============================================================================
# Preprocessing Tests
# =============================================================================


class TestSmoothClip:
    """Tests for the _SmoothClip transform."""

    def test_smooth_clip_identity_near_zero(self):
        """Test that small values pass through nearly unchanged."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _SmoothClip

        clip = _SmoothClip(c=3.0)
        x = np.array([0.0, 0.1, -0.1, 0.5, -0.5])
        result = clip(x)
        # For small |x| << c, tanh(x/c) ~ x/c, so c*tanh(x/c) ~ x
        np.testing.assert_allclose(result, x, atol=0.05)

    def test_smooth_clip_bounds_outliers(self):
        """Test that extreme values are bounded."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _SmoothClip

        clip = _SmoothClip(c=3.0)
        x = np.array([100.0, -100.0, 50.0, -50.0])
        result = clip(x)
        # Should be bounded approximately within [-c, c]
        assert np.all(np.abs(result) <= 3.0 + 1e-5)


class TestRobustPreprocessor:
    """Tests for the _RobustPreprocessor."""

    def test_fit_transform_centers(self):
        """Test that robust scaling centers data around median."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _RobustPreprocessor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 5).astype(np.float32)
        prep = _RobustPreprocessor(smooth_clip_c=3.0)
        X_transformed = prep.fit_transform(X)

        # After scaling, the median should be approximately 0
        medians = np.median(X_transformed, axis=0)
        np.testing.assert_allclose(medians, 0.0, atol=0.3)

    def test_robust_vs_no_clip(self):
        """Test that smooth clipping reduces the range of outliers."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _RobustPreprocessor

        rng = np.random.RandomState(42)
        X = rng.randn(100, 3).astype(np.float32)
        # Add outliers
        X[0, 0] = 100.0
        X[1, 1] = -100.0

        # With clipping
        prep_clip = _RobustPreprocessor(smooth_clip_c=3.0)
        X_clip = prep_clip.fit_transform(X)

        # Without clipping (very large c effectively disables it)
        prep_noclip = _RobustPreprocessor(smooth_clip_c=1000.0)
        X_noclip = prep_noclip.fit_transform(X)

        # Clipped version should have smaller max absolute value
        assert np.max(np.abs(X_clip)) < np.max(np.abs(X_noclip))


# =============================================================================
# RealMLPRegressor Tests
# =============================================================================


class TestRealMLPRegressorInit:
    """Tests for RealMLPRegressor initialization."""

    def test_import(self):
        """Test that RealMLPRegressor can be imported."""
        pytest.importorskip("torch")
        from endgame.models.tabular import RealMLPRegressor
        assert RealMLPRegressor is not None

    def test_default_params(self):
        """Test default parameter values for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        reg = RealMLPRegressor()
        assert reg.n_blocks == 3
        assert reg.d_block == 256
        assert reg.dropout == 0.15
        assert reg.learning_rate == 0.04
        assert reg._estimator_type == "regressor"

    def test_estimator_type(self):
        """Test sklearn estimator type for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        reg = RealMLPRegressor()
        assert reg._estimator_type == "regressor"

    def test_get_set_params(self):
        """Test get_params and set_params for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        reg = RealMLPRegressor(n_blocks=2)
        assert reg.get_params()["n_blocks"] == 2

        reg.set_params(n_blocks=5)
        assert reg.n_blocks == 5


class TestRealMLPRegressorFitPredict:
    """Tests for RealMLPRegressor fit/predict functionality."""

    def test_basic_fit_predict(self, regression_data, small_params):
        """Test basic fit/predict for regression."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        X, y = regression_data
        reg = RealMLPRegressor(**small_params)
        reg.fit(X, y)

        preds = reg.predict(X)
        assert len(preds) == len(X)
        assert preds.dtype in (np.float32, np.float64)

    def test_predict_returns_real_values(self, regression_data, small_params):
        """Test that predictions are real-valued (not quantized)."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        X, y = regression_data
        reg = RealMLPRegressor(**small_params)
        reg.fit(X, y)

        preds = reg.predict(X)
        n_unique = len(np.unique(np.round(preds, 2)))
        assert n_unique > 5

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        reg = RealMLPRegressor()
        X = np.random.randn(10, 5).astype(np.float32)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            reg.predict(X)

    def test_feature_importances_shape(self, regression_data, small_params):
        """Test feature_importances_ shape for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        X, y = regression_data
        reg = RealMLPRegressor(**small_params)
        reg.fit(X, y)

        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_.shape == (X.shape[1],)
        assert np.all(reg.feature_importances_ >= 0)

    def test_history_attribute(self, regression_data, small_params):
        """Test training history for regressor."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        X, y = regression_data
        reg = RealMLPRegressor(**small_params)
        reg.fit(X, y)

        assert hasattr(reg, "history_")
        assert len(reg.history_["train_loss"]) > 0
        assert len(reg.history_["val_loss"]) > 0

    def test_sklearn_cross_validate(self, regression_data, small_params):
        """Test compatibility with sklearn cross_val_score."""
        pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import RealMLPRegressor

        X, y = regression_data
        reg = RealMLPRegressor(**small_params)

        scores = cross_val_score(reg, X, y, cv=2, scoring="neg_mean_squared_error")
        assert len(scores) == 2
        assert all(s <= 0 for s in scores)


# =============================================================================
# Internal Module Tests
# =============================================================================


class TestDiagonalLayer:
    """Tests for the _DiagonalLayer module."""

    def test_forward_shape(self):
        """Test _DiagonalLayer preserves input shape."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _DiagonalLayer

        layer = _DiagonalLayer(d_in=10)
        x = torch.randn(32, 10)
        out = layer(x)
        assert out.shape == (32, 10)

    def test_identity_init(self):
        """Test that diagonal layer is initialised to identity scaling."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _DiagonalLayer

        layer = _DiagonalLayer(d_in=5)
        x = torch.randn(16, 5)
        out = layer(x)
        # Weights initialised to 1.0, so output should equal input
        torch.testing.assert_close(out, x)


class TestPiecewiseLinearEmbedding:
    """Tests for the _PiecewiseLinearEmbedding module."""

    def test_forward_shape(self):
        """Test embedding output shape."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _PiecewiseLinearEmbedding

        emb = _PiecewiseLinearEmbedding(n_features=5, n_bins=4)
        x = torch.randn(32, 5)
        out = emb(x)
        assert out.shape == (32, 5 * 4)

    def test_output_bounded(self):
        """Test that embeddings produce finite values."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _PiecewiseLinearEmbedding

        emb = _PiecewiseLinearEmbedding(n_features=3, n_bins=8)
        x = torch.randn(64, 3) * 10  # larger values
        out = emb(x)
        assert torch.isfinite(out).all()


class TestRealMLPNetwork:
    """Tests for the _RealMLPNetwork module."""

    def test_forward_shape_classification(self):
        """Test network output shape for classification."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _RealMLPNetwork

        net = _RealMLPNetwork(
            n_features=10, n_outputs=3,
            n_blocks=2, d_block=32,
            use_embeddings=True, n_bins=4,
        )
        x = torch.randn(32, 10)
        out = net(x)
        assert out.shape == (32, 3)

    def test_forward_shape_regression(self):
        """Test network output shape for regression."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _RealMLPNetwork

        net = _RealMLPNetwork(
            n_features=5, n_outputs=1,
            n_blocks=1, d_block=16,
            use_embeddings=False,
        )
        x = torch.randn(16, 5)
        out = net(x)
        assert out.shape == (16, 1)

    def test_no_embeddings(self):
        """Test network without embeddings."""
        torch = pytest.importorskip("torch")
        from endgame.models.tabular.realmlp import _RealMLPNetwork

        net = _RealMLPNetwork(
            n_features=8, n_outputs=2,
            n_blocks=1, d_block=16,
            use_embeddings=False,
        )
        x = torch.randn(10, 8)
        out = net(x)
        assert out.shape == (10, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
