"""Tests for target transformation wrappers."""

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeRegressor

from endgame.preprocessing.target_transform import (
    TargetTransformer,
    TargetQuantileTransformer,
    _inv_yeojohnson,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def regression_data():
    """Standard regression dataset."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=5.0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def positive_regression_data():
    """Regression dataset with strictly positive, right-skewed target."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=5.0,
        random_state=42,
    )
    # Exponentiate to make positive and skewed
    y = np.exp(y / 100)
    return X, y


@pytest.fixture
def nonneg_regression_data():
    """Regression dataset with non-negative target."""
    X, y = make_regression(
        n_samples=300,
        n_features=10,
        n_informative=5,
        noise=5.0,
        random_state=42,
    )
    y = np.abs(y)
    return X, y


# ------------------------------------------------------------------
# TargetTransformer: basic usage
# ------------------------------------------------------------------


class TestTargetTransformerBasic:
    """Basic fit/predict tests for TargetTransformer."""

    def test_no_regressor_raises(self):
        """Passing no regressor raises TypeError."""
        with pytest.raises(TypeError, match="requires a regressor"):
            TargetTransformer()

    def test_invalid_method_raises(self, regression_data):
        """Passing an invalid method raises ValueError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="banana")
        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(X, y)

    def test_none_method(self, regression_data):
        """method='none' acts as passthrough."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="none")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

        # Should behave identically to an unwrapped Ridge
        ref = Ridge().fit(X_train, y_train)
        np.testing.assert_allclose(preds, ref.predict(X_test), atol=1e-10)

    def test_predict_before_fit_raises(self, regression_data):
        """Calling predict before fit raises RuntimeError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="none")
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(X)


class TestTargetTransformerMethods:
    """Test each transform method individually."""

    def test_log_transform(self, positive_regression_data):
        """Log transform on positive data."""
        X, y = positive_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="log")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "log"
        # Predictions should be positive (exp of real numbers)
        assert np.all(preds > 0)

    def test_log_negative_target_raises(self, regression_data):
        """Log transform with negative targets raises ValueError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="log")
        with pytest.raises(ValueError, match="non-positive"):
            model.fit(X, y)

    def test_log1p_transform(self, nonneg_regression_data):
        """Log1p transform on non-negative data."""
        X, y = nonneg_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="log1p")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "log1p"

    def test_log1p_negative_target_raises(self, regression_data):
        """Log1p transform with negative targets raises ValueError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="log1p")
        with pytest.raises(ValueError, match="negative values"):
            model.fit(X, y)

    def test_sqrt_transform(self, nonneg_regression_data):
        """Sqrt transform on non-negative data."""
        X, y = nonneg_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="sqrt")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "sqrt"
        # Inverse of sqrt is square, result should be non-negative
        assert np.all(preds >= 0)

    def test_sqrt_negative_target_raises(self, regression_data):
        """Sqrt transform with negative targets raises ValueError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="sqrt")
        with pytest.raises(ValueError, match="negative values"):
            model.fit(X, y)

    def test_box_cox_transform(self, positive_regression_data):
        """Box-Cox transform on positive data."""
        X, y = positive_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="box_cox")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "box_cox"
        assert model.lambda_ is not None

    def test_box_cox_negative_target_raises(self, regression_data):
        """Box-Cox with negative targets raises ValueError."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="box_cox")
        with pytest.raises(ValueError, match="non-positive"):
            model.fit(X, y)

    def test_yeo_johnson_transform(self, regression_data):
        """Yeo-Johnson transform works with any real-valued targets."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="yeo_johnson")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "yeo_johnson"
        assert model.lambda_ is not None

    def test_quantile_transform(self, regression_data):
        """Quantile transform maps to normal distribution."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(
            regressor=Ridge(), method="quantile", random_state=42
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "quantile"
        assert model.qt_ is not None

    def test_rank_transform(self, regression_data):
        """Rank transform maps to normal via ordinal ranks."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(regressor=Ridge(), method="rank")
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ == "rank"
        assert model.y_train_sorted_ is not None


class TestTargetTransformerAuto:
    """Test auto method selection."""

    def test_auto_selects_method(self, positive_regression_data):
        """Auto mode selects a transform method."""
        X, y = positive_regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetTransformer(
            regressor=Ridge(), method="auto", verbose=True
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape
        assert model.method_ in {
            "none", "box_cox", "yeo_johnson",
        }

    def test_auto_with_normal_target(self):
        """Auto mode selects 'none' for normally distributed targets."""
        rng = np.random.RandomState(42)
        X = rng.randn(500, 5)
        y = rng.randn(500)  # Already normal
        model = TargetTransformer(regressor=Ridge(), method="auto")
        model.fit(X, y)
        # Shapiro-Wilk should pass for truly normal data
        assert model.method_ == "none"

    def test_auto_with_negative_targets(self, regression_data):
        """Auto mode handles negative targets (cannot use box_cox)."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="auto")
        model.fit(X, y)
        # Should not select box_cox for mixed-sign targets
        assert model.method_ in {"none", "yeo_johnson"}


class TestTargetTransformerEdgeCases:
    """Edge case tests."""

    def test_constant_target(self):
        """Constant target defaults to 'none' method."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = np.full(100, 3.14)
        model = TargetTransformer(regressor=Ridge(), method="auto")
        model.fit(X, y)
        assert model.method_ == "none"
        preds = model.predict(X)
        np.testing.assert_allclose(preds, 3.14, atol=1e-6)

    def test_single_sample(self):
        """Fitting with a single sample still works."""
        X = np.array([[1.0, 2.0]])
        y = np.array([5.0])
        model = TargetTransformer(regressor=Ridge(), method="none")
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (1,)

    def test_sample_weight_passthrough(self, regression_data):
        """sample_weight is forwarded to the wrapped regressor."""
        X, y = regression_data
        weights = np.ones(len(y))
        model = TargetTransformer(regressor=Ridge(), method="none")
        # Should not raise
        model.fit(X, y, sample_weight=weights)

    def test_large_dataset_auto_subsamples(self):
        """Auto mode subsamples for Shapiro-Wilk when n > 5000."""
        rng = np.random.RandomState(42)
        X = rng.randn(6000, 5)
        y = np.exp(rng.randn(6000))  # Skewed positive
        model = TargetTransformer(
            regressor=Ridge(), method="auto", random_state=42
        )
        model.fit(X, y)
        assert model.method_ in {"none", "box_cox", "yeo_johnson"}


class TestTargetTransformerProperties:
    """Test property delegation."""

    def test_feature_importances_passthrough(self, regression_data):
        """feature_importances_ is delegated to the wrapped regressor."""
        X, y = regression_data
        model = TargetTransformer(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            method="none",
        )
        model.fit(X, y)
        fi = model.feature_importances_
        assert fi.shape == (X.shape[1],)
        assert np.all(fi >= 0)

    def test_feature_importances_unavailable(self, regression_data):
        """AttributeError when wrapped regressor has no feature_importances_."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="none")
        model.fit(X, y)
        with pytest.raises(AttributeError, match="does not expose"):
            _ = model.feature_importances_

    def test_predict_proba_passthrough(self, regression_data):
        """predict_proba raises AttributeError for regressors without it."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="none")
        model.fit(X, y)
        with pytest.raises(AttributeError, match="does not support predict_proba"):
            model.predict_proba(X)


# ------------------------------------------------------------------
# Round-trip invertibility tests
# ------------------------------------------------------------------


class TestInvertibility:
    """Verify that forward + inverse is approximately identity."""

    def test_log_roundtrip(self):
        """log / exp roundtrip."""
        y = np.array([1.0, 2.0, 3.0, 100.0, 0.01])
        model = TargetTransformer(regressor=Ridge(), method="log")
        model.method_ = "log"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-10)

    def test_log1p_roundtrip(self):
        """log1p / expm1 roundtrip."""
        y = np.array([0.0, 1.0, 2.0, 100.0])
        model = TargetTransformer(regressor=Ridge(), method="log1p")
        model.method_ = "log1p"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-10)

    def test_sqrt_roundtrip(self):
        """sqrt / square roundtrip."""
        y = np.array([0.0, 1.0, 4.0, 9.0, 100.0])
        model = TargetTransformer(regressor=Ridge(), method="sqrt")
        model.method_ = "sqrt"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-10)

    def test_box_cox_roundtrip(self):
        """Box-Cox roundtrip."""
        y = np.array([1.0, 2.0, 3.0, 10.0, 50.0])
        model = TargetTransformer(regressor=Ridge(), method="box_cox")
        model.method_ = "box_cox"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-6)

    def test_yeo_johnson_roundtrip(self):
        """Yeo-Johnson roundtrip (including negative values)."""
        y = np.array([-10.0, -1.0, 0.0, 1.0, 10.0, 100.0])
        model = TargetTransformer(regressor=Ridge(), method="yeo_johnson")
        model.method_ = "yeo_johnson"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, rtol=1e-6)

    def test_quantile_roundtrip(self):
        """Quantile transform roundtrip."""
        rng = np.random.RandomState(42)
        y = rng.exponential(scale=5.0, size=200)
        model = TargetTransformer(
            regressor=Ridge(), method="quantile", random_state=42
        )
        model.method_ = "quantile"
        y_t = model._forward(y, fit=True)
        y_back = model._inverse(y_t)
        np.testing.assert_allclose(y_back, y, atol=0.5)

    def test_yeo_johnson_inv_edge_lambda_zero(self):
        """Yeo-Johnson inverse with lambda=0 (log branch)."""
        y_orig = np.array([0.0, 1.0, 5.0, 10.0])
        # Forward: lam=0 => y = log(x+1)
        y_transformed = np.log(y_orig + 1)
        y_back = _inv_yeojohnson(y_transformed, lam=0.0)
        np.testing.assert_allclose(y_back, y_orig, rtol=1e-10)

    def test_yeo_johnson_inv_edge_lambda_two(self):
        """Yeo-Johnson inverse with lambda=2 (negative log branch)."""
        x_orig = np.array([-5.0, -1.0, -0.5])
        # Forward: lam=2 => y = -log(-x + 1)
        y_transformed = -np.log(-x_orig + 1)
        x_back = _inv_yeojohnson(y_transformed, lam=2.0)
        np.testing.assert_allclose(x_back, x_orig, rtol=1e-10)


# ------------------------------------------------------------------
# TargetQuantileTransformer
# ------------------------------------------------------------------


class TestTargetQuantileTransformer:
    """Tests for TargetQuantileTransformer wrapper."""

    def test_no_regressor_raises(self):
        """Passing no regressor raises TypeError."""
        with pytest.raises(TypeError, match="requires a regressor"):
            TargetQuantileTransformer()

    def test_basic_fit_predict(self, regression_data):
        """Basic fit and predict workflow."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetQuantileTransformer(
            regressor=Ridge(),
            n_quantiles=100,
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_uniform_output_distribution(self, regression_data):
        """Uniform output distribution works."""
        X, y = regression_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = TargetQuantileTransformer(
            regressor=Ridge(),
            output_distribution="uniform",
            random_state=42,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        assert preds.shape == y_test.shape

    def test_feature_importances(self, regression_data):
        """Feature importances passthrough."""
        X, y = regression_data
        model = TargetQuantileTransformer(
            regressor=RandomForestRegressor(n_estimators=10, random_state=42),
            random_state=42,
        )
        model.fit(X, y)
        fi = model.feature_importances_
        assert fi.shape == (X.shape[1],)

    def test_feature_importances_unavailable(self, regression_data):
        """AttributeError for regressors without feature_importances_."""
        X, y = regression_data
        model = TargetQuantileTransformer(regressor=Ridge(), random_state=42)
        model.fit(X, y)
        with pytest.raises(AttributeError, match="does not expose"):
            _ = model.feature_importances_

    def test_predict_before_fit_raises(self, regression_data):
        """Calling predict before fit raises RuntimeError."""
        X, y = regression_data
        model = TargetQuantileTransformer(regressor=Ridge(), random_state=42)
        with pytest.raises(RuntimeError, match="has not been fitted"):
            model.predict(X)


# ------------------------------------------------------------------
# Sklearn compatibility
# ------------------------------------------------------------------


class TestSklearnCompatibility:
    """Test that wrappers work in sklearn pipelines and utilities."""

    def test_cross_val_score(self, regression_data):
        """Works with cross_val_score."""
        X, y = regression_data
        model = TargetTransformer(regressor=Ridge(), method="yeo_johnson")
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3
        # Sanity: R^2 should be reasonable for make_regression
        assert np.mean(scores) > 0.5

    def test_get_params(self):
        """get_params returns constructor parameters."""
        model = TargetTransformer(regressor=Ridge(alpha=1.0), method="log")
        params = model.get_params()
        assert params["method"] == "log"
        assert "regressor" in params

    def test_set_params(self):
        """set_params updates parameters."""
        model = TargetTransformer(regressor=Ridge(), method="log")
        model.set_params(method="sqrt")
        assert model.method == "sqrt"

    def test_clone_works(self):
        """sklearn clone works on TargetTransformer."""
        from sklearn.base import clone
        model = TargetTransformer(regressor=Ridge(alpha=2.0), method="log")
        cloned = clone(model)
        assert cloned.method == "log"
        assert cloned is not model

    def test_quantile_cross_val_score(self, regression_data):
        """TargetQuantileTransformer works with cross_val_score."""
        X, y = regression_data
        model = TargetQuantileTransformer(
            regressor=Ridge(), random_state=42
        )
        scores = cross_val_score(model, X, y, cv=3, scoring="r2")
        assert len(scores) == 3


# ------------------------------------------------------------------
# Import from preprocessing
# ------------------------------------------------------------------


class TestImport:
    """Test that classes are importable from the preprocessing module."""

    def test_import_target_transformer(self):
        """TargetTransformer is importable from preprocessing."""
        from endgame.preprocessing import TargetTransformer as TT
        assert TT is TargetTransformer

    def test_import_target_quantile_transformer(self):
        """TargetQuantileTransformer is importable from preprocessing."""
        from endgame.preprocessing import TargetQuantileTransformer as TQT
        assert TQT is TargetQuantileTransformer
