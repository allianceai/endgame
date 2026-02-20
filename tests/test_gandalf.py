"""Tests for GANDALF (Gated Adaptive Network for Deep Automated Learning of Features)."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split


# Skip all tests if pytorch-tabular is not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("pytorch_tabular", reason="pytorch-tabular not installed"),
    reason="pytorch-tabular not installed",
)


class TestGANDALFClassifier:
    """Tests for GANDALFClassifier."""

    def test_basic_classification(self):
        """Test basic classification with numpy arrays."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = GANDALFClassifier(
            gflu_stages=2,
            n_epochs=5,  # Quick test
            batch_size=32,
            early_stopping_patience=3,
            random_state=42,
            verbose=False,
        )
        clf.fit(X_train, y_train)

        # Test predict
        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset(set(y_train))

        # Test predict_proba
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_classification_with_eval_set(self):
        """Test classification with validation set for early stopping."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        clf = GANDALFClassifier(
            gflu_stages=2,
            n_epochs=10,
            batch_size=32,
            early_stopping_patience=3,
            validation_fraction=0.0,  # Use external validation
            random_state=42,
            verbose=False,
        )
        clf.fit(X_train, y_train, eval_set=(X_val, y_val))

        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)

    def test_classification_with_dataframe(self):
        """Test classification with pandas DataFrame input."""
        pd = pytest.importorskip("pandas")
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        
        # Convert to DataFrame
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        clf = GANDALFClassifier(
            gflu_stages=2,
            n_epochs=5,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X_train, y_train)

        # Test with DataFrame
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf = GANDALFClassifier(
            gflu_stages=2,
            n_epochs=5,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X_train, y_train)

        preds = clf.predict(X_test)
        assert len(preds) == len(X_test)
        assert set(preds).issubset({0, 1, 2})

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        from endgame.models.tabular import GANDALFClassifier

        clf = GANDALFClassifier()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict(X)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict_proba(X)


class TestGANDALFRegressor:
    """Tests for GANDALFRegressor."""

    def test_basic_regression(self):
        """Test basic regression with numpy arrays."""
        from endgame.models.tabular import GANDALFRegressor

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=10.0,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg = GANDALFRegressor(
            gflu_stages=2,
            n_epochs=5,
            batch_size=32,
            early_stopping_patience=3,
            random_state=42,
            verbose=False,
        )
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)
        assert len(preds) == len(X_test)
        assert preds.dtype in [np.float32, np.float64]

    def test_regression_with_eval_set(self):
        """Test regression with validation set for early stopping."""
        from endgame.models.tabular import GANDALFRegressor

        X, y = make_regression(
            n_samples=300,
            n_features=10,
            n_informative=5,
            noise=10.0,
            random_state=42,
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        reg = GANDALFRegressor(
            gflu_stages=2,
            n_epochs=10,
            batch_size=32,
            early_stopping_patience=3,
            validation_fraction=0.0,
            random_state=42,
            verbose=False,
        )
        reg.fit(X_train, y_train, eval_set=(X_val, y_val))

        preds = reg.predict(X_test)
        assert len(preds) == len(X_test)

    def test_regression_with_target_range(self):
        """Test regression with target range constraint."""
        from endgame.models.tabular import GANDALFRegressor

        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            random_state=42,
        )
        # Normalize targets to [0, 1]
        y = (y - y.min()) / (y.max() - y.min())
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        reg = GANDALFRegressor(
            gflu_stages=2,
            n_epochs=5,
            batch_size=32,
            target_range=(0.0, 1.0),
            random_state=42,
            verbose=False,
        )
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)
        assert len(preds) == len(X_test)

    def test_regression_with_dataframe(self):
        """Test regression with pandas DataFrame input."""
        pd = pytest.importorskip("pandas")
        from endgame.models.tabular import GANDALFRegressor

        X, y = make_regression(
            n_samples=200,
            n_features=5,
            n_informative=3,
            random_state=42,
        )
        
        df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
        X_train, X_test, y_train, y_test = train_test_split(
            df, y, test_size=0.2, random_state=42
        )

        reg = GANDALFRegressor(
            gflu_stages=2,
            n_epochs=5,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        reg.fit(X_train, y_train)

        preds = reg.predict(X_test)
        assert len(preds) == len(X_test)

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        from endgame.models.tabular import GANDALFRegressor

        reg = GANDALFRegressor()
        X = np.random.randn(10, 5)

        with pytest.raises(RuntimeError, match="has not been fitted"):
            reg.predict(X)


class TestGANDALFParameters:
    """Tests for GANDALF parameter variations."""

    def test_gflu_stages_parameter(self):
        """Test different GFLU stages values."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        for stages in [2, 4, 6]:
            clf = GANDALFClassifier(
                gflu_stages=stages,
                n_epochs=3,
                batch_size=32,
                random_state=42,
                verbose=False,
            )
            clf.fit(X, y)
            preds = clf.predict(X)
            assert len(preds) == len(X)

    def test_dropout_parameter(self):
        """Test GFLU dropout regularization."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = GANDALFClassifier(
            gflu_stages=2,
            gflu_dropout=0.2,
            embedding_dropout=0.1,
            n_epochs=3,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)

    def test_sparsity_parameter(self):
        """Test feature sparsity initialization."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = GANDALFClassifier(
            gflu_stages=2,
            gflu_feature_init_sparsity=0.5,
            learnable_sparsity=True,
            n_epochs=3,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)

    def test_fixed_sparsity(self):
        """Test fixed (non-learnable) sparsity."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = GANDALFClassifier(
            gflu_stages=2,
            gflu_feature_init_sparsity=0.3,
            learnable_sparsity=False,
            n_epochs=3,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X, y)
        preds = clf.predict(X)
        assert len(preds) == len(X)


class TestGANDALFIntegration:
    """Integration tests for GANDALF."""

    def test_sklearn_api_compatibility(self):
        """Test sklearn API compatibility (estimator checks)."""
        from endgame.models.tabular import GANDALFClassifier, GANDALFRegressor

        # Test classifier attributes
        clf = GANDALFClassifier()
        assert hasattr(clf, 'fit')
        assert hasattr(clf, 'predict')
        assert hasattr(clf, 'predict_proba')
        assert clf._estimator_type == "classifier"

        # Test regressor attributes
        reg = GANDALFRegressor()
        assert hasattr(reg, 'fit')
        assert hasattr(reg, 'predict')
        assert reg._estimator_type == "regressor"

    def test_import_from_models(self):
        """Test import from endgame.models.tabular."""
        from endgame.models.tabular import GANDALFClassifier, GANDALFRegressor

        assert GANDALFClassifier is not None
        assert GANDALFRegressor is not None

    def test_classes_attribute(self):
        """Test that classes_ attribute is set correctly."""
        from endgame.models.tabular import GANDALFClassifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = GANDALFClassifier(
            gflu_stages=2,
            n_epochs=3,
            batch_size=32,
            random_state=42,
            verbose=False,
        )
        clf.fit(X, y)

        assert hasattr(clf, 'classes_')
        assert len(clf.classes_) == 2
        assert clf.n_classes_ == 2
