"""Tests for dimensionality reduction module."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification
from sklearn.utils.validation import check_is_fitted


# Test data fixtures
@pytest.fixture
def small_data():
    """Small dataset for quick tests."""
    X, y = make_blobs(n_samples=100, n_features=20, centers=3, random_state=42)
    return X, y


@pytest.fixture
def medium_data():
    """Medium dataset for more thorough tests."""
    X, y = make_blobs(n_samples=500, n_features=50, centers=5, random_state=42)
    return X, y


# =============================================================================
# Linear Methods Tests
# =============================================================================


class TestPCAReducer:
    """Tests for PCAReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import PCAReducer
        assert PCAReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import PCAReducer

        X, _ = small_data
        pca = PCAReducer(n_components=5, random_state=42)
        X_reduced = pca.fit_transform(X)

        assert X_reduced.shape == (100, 5)
        assert hasattr(pca, "components_")
        assert hasattr(pca, "explained_variance_ratio_")

    def test_variance_threshold(self, small_data):
        from endgame.dimensionality_reduction import PCAReducer

        X, _ = small_data
        pca = PCAReducer(n_components=0.95)  # Keep 95% variance
        X_reduced = pca.fit_transform(X)

        cumulative = pca.get_cumulative_variance()
        assert cumulative[-1] >= 0.95

    def test_inverse_transform(self, small_data):
        from endgame.dimensionality_reduction import PCAReducer

        X, _ = small_data
        pca = PCAReducer(n_components=10)
        X_reduced = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_reduced)

        assert X_reconstructed.shape == X.shape

    def test_get_n_components_for_variance(self, small_data):
        from endgame.dimensionality_reduction import PCAReducer

        X, _ = small_data
        pca = PCAReducer(n_components=None)  # Keep all
        pca.fit(X)

        n_95 = pca.get_n_components_for_variance(0.95)
        assert n_95 <= X.shape[1]


class TestRandomizedPCA:
    """Tests for RandomizedPCA."""

    def test_import(self):
        from endgame.dimensionality_reduction import RandomizedPCA
        assert RandomizedPCA is not None

    def test_fit_transform(self, medium_data):
        from endgame.dimensionality_reduction import RandomizedPCA

        X, _ = medium_data
        rpca = RandomizedPCA(n_components=10, random_state=42)
        X_reduced = rpca.fit_transform(X)

        assert X_reduced.shape == (500, 10)
        assert hasattr(rpca, "components_")

    def test_inverse_transform(self, medium_data):
        from endgame.dimensionality_reduction import RandomizedPCA

        X, _ = medium_data
        rpca = RandomizedPCA(n_components=20, random_state=42)
        X_reduced = rpca.fit_transform(X)
        X_recon = rpca.inverse_transform(X_reduced)

        assert X_recon.shape == X.shape


class TestTruncatedSVDReducer:
    """Tests for TruncatedSVDReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import TruncatedSVDReducer
        assert TruncatedSVDReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import TruncatedSVDReducer

        X, _ = small_data
        svd = TruncatedSVDReducer(n_components=5, random_state=42)
        X_reduced = svd.fit_transform(X)

        assert X_reduced.shape == (100, 5)
        assert hasattr(svd, "components_")
        assert hasattr(svd, "singular_values_")

    def test_sparse_input(self):
        from endgame.dimensionality_reduction import TruncatedSVDReducer
        from scipy.sparse import random as sparse_random

        X_sparse = sparse_random(100, 50, density=0.1, random_state=42)
        svd = TruncatedSVDReducer(n_components=10)
        X_reduced = svd.fit_transform(X_sparse)

        assert X_reduced.shape == (100, 10)


class TestKernelPCAReducer:
    """Tests for KernelPCAReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import KernelPCAReducer
        assert KernelPCAReducer is not None

    def test_fit_transform_rbf(self, small_data):
        from endgame.dimensionality_reduction import KernelPCAReducer

        X, _ = small_data
        kpca = KernelPCAReducer(n_components=5, kernel="rbf", gamma=0.1)
        X_reduced = kpca.fit_transform(X)

        assert X_reduced.shape == (100, 5)
        assert hasattr(kpca, "eigenvalues_")

    def test_different_kernels(self, small_data):
        from endgame.dimensionality_reduction import KernelPCAReducer

        X, _ = small_data
        for kernel in ["linear", "poly", "rbf", "sigmoid", "cosine"]:
            kpca = KernelPCAReducer(n_components=3, kernel=kernel)
            X_reduced = kpca.fit_transform(X)
            assert X_reduced.shape == (100, 3)

    def test_inverse_transform(self, small_data):
        from endgame.dimensionality_reduction import KernelPCAReducer

        X, _ = small_data
        kpca = KernelPCAReducer(
            n_components=5, kernel="rbf", fit_inverse_transform=True
        )
        X_reduced = kpca.fit_transform(X)
        X_recon = kpca.inverse_transform(X_reduced)

        assert X_recon.shape == X.shape


class TestICAReducer:
    """Tests for ICAReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import ICAReducer
        assert ICAReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import ICAReducer

        X, _ = small_data
        ica = ICAReducer(n_components=5, random_state=42)
        X_reduced = ica.fit_transform(X)

        assert X_reduced.shape == (100, 5)
        assert hasattr(ica, "components_")
        assert hasattr(ica, "mixing_")

    def test_inverse_transform(self, small_data):
        from endgame.dimensionality_reduction import ICAReducer

        X, _ = small_data
        ica = ICAReducer(n_components=10, random_state=42)
        X_reduced = ica.fit_transform(X)
        X_recon = ica.inverse_transform(X_reduced)

        assert X_recon.shape == X.shape


# =============================================================================
# Manifold Learning Tests
# =============================================================================


class TestUMAPReducer:
    """Tests for UMAPReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import UMAPReducer
        assert UMAPReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import UMAPReducer

        X, _ = small_data
        umap = UMAPReducer(n_components=2, n_neighbors=10, random_state=42)
        X_2d = umap.fit_transform(X)

        assert X_2d.shape == (100, 2)
        assert hasattr(umap, "embedding_")

    def test_transform_new_data(self, small_data):
        from endgame.dimensionality_reduction import UMAPReducer

        X, _ = small_data
        X_train, X_test = X[:80], X[80:]

        umap = UMAPReducer(n_components=2, n_neighbors=10, random_state=42)
        umap.fit(X_train)
        X_test_2d = umap.transform(X_test)

        assert X_test_2d.shape == (20, 2)

    def test_supervised_umap(self, small_data):
        from endgame.dimensionality_reduction import UMAPReducer

        X, y = small_data
        umap = UMAPReducer(n_components=2, random_state=42)
        X_2d = umap.fit_transform(X, y)  # Semi-supervised

        assert X_2d.shape == (100, 2)

    def test_different_metrics(self, small_data):
        from endgame.dimensionality_reduction import UMAPReducer

        X, _ = small_data
        for metric in ["euclidean", "manhattan", "cosine"]:
            umap = UMAPReducer(n_components=2, metric=metric, random_state=42)
            X_2d = umap.fit_transform(X)
            assert X_2d.shape == (100, 2)


class TestTriMAPReducer:
    """Tests for TriMAPReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import TriMAPReducer
        assert TriMAPReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import TriMAPReducer

        X, _ = small_data
        trimap = TriMAPReducer(n_components=2, n_inliers=8, n_iters=100, verbose=False)
        X_2d = trimap.fit_transform(X)

        assert X_2d.shape == (100, 2)
        assert hasattr(trimap, "embedding_")

    def test_transform_approximation(self, small_data):
        from endgame.dimensionality_reduction import TriMAPReducer

        X, _ = small_data
        X_train, X_test = X[:80], X[80:]

        trimap = TriMAPReducer(n_components=2, n_iters=100, verbose=False)
        trimap.fit(X_train)
        X_test_2d = trimap.transform(X_test)

        assert X_test_2d.shape == (20, 2)


class TestPHATEReducer:
    """Tests for PHATEReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import PHATEReducer
        assert PHATEReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import PHATEReducer

        X, _ = small_data
        phate = PHATEReducer(n_components=2, knn=5, verbose=0)
        X_2d = phate.fit_transform(X)

        assert X_2d.shape == (100, 2)
        assert hasattr(phate, "embedding_")

    def test_transform_new_data(self, small_data):
        from endgame.dimensionality_reduction import PHATEReducer

        X, _ = small_data
        X_train, X_test = X[:80], X[80:]

        phate = PHATEReducer(n_components=2, knn=5, verbose=0)
        phate.fit(X_train)
        X_test_2d = phate.transform(X_test)

        assert X_test_2d.shape == (20, 2)


class TestPaCMAPReducer:
    """Tests for PaCMAPReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import PaCMAPReducer
        assert PaCMAPReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import PaCMAPReducer

        X, _ = small_data
        pacmap = PaCMAPReducer(n_components=2, n_neighbors=10, num_iters=100)
        X_2d = pacmap.fit_transform(X)

        assert X_2d.shape == (100, 2)
        assert hasattr(pacmap, "embedding_")

    def test_transform_new_data(self, small_data):
        from endgame.dimensionality_reduction import PaCMAPReducer

        X, _ = small_data
        X_train, X_test = X[:80], X[80:]

        pacmap = PaCMAPReducer(n_components=2, num_iters=100)
        pacmap.fit(X_train)
        X_test_2d = pacmap.transform(X_test)

        assert X_test_2d.shape == (20, 2)


# =============================================================================
# VAE Tests
# =============================================================================


class TestVAEReducer:
    """Tests for VAEReducer."""

    def test_import(self):
        from endgame.dimensionality_reduction import VAEReducer
        assert VAEReducer is not None

    def test_fit_transform(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        vae = VAEReducer(
            n_components=5,
            encoder_layers=[32, 16],
            decoder_layers=[16, 32],
            n_epochs=10,
            batch_size=32,
            random_state=42,
        )
        X_latent = vae.fit_transform(X)

        assert X_latent.shape == (100, 5)
        assert hasattr(vae, "model_")
        assert hasattr(vae, "reconstruction_loss_")

    def test_inverse_transform(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        vae = VAEReducer(
            n_components=10,
            encoder_layers=[32],
            decoder_layers=[32],
            n_epochs=10,
            random_state=42,
        )
        X_latent = vae.fit_transform(X)
        X_recon = vae.inverse_transform(X_latent)

        assert X_recon.shape == X.shape

    def test_reconstruct(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        vae = VAEReducer(
            n_components=10,
            n_epochs=20,
            random_state=42,
        )
        vae.fit(X)
        X_recon = vae.reconstruct(X)

        assert X_recon.shape == X.shape
        # Reconstruction should be reasonably close
        error = np.mean((X - X_recon) ** 2)
        assert error < 100  # Rough sanity check

    def test_sample(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        vae = VAEReducer(n_components=5, n_epochs=10, random_state=42)
        vae.fit(X)
        samples = vae.sample(n_samples=50)

        assert samples.shape == (50, X.shape[1])

    def test_reconstruction_error(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        vae = VAEReducer(n_components=10, n_epochs=20, random_state=42)
        vae.fit(X)
        errors = vae.reconstruction_error(X)

        assert errors.shape == (100,)
        assert np.all(errors >= 0)

    def test_beta_vae(self, small_data):
        from endgame.dimensionality_reduction import VAEReducer

        X, _ = small_data
        # Test with different beta values
        vae_low = VAEReducer(n_components=5, beta=0.1, n_epochs=10, random_state=42)
        vae_high = VAEReducer(n_components=5, beta=10.0, n_epochs=10, random_state=42)

        vae_low.fit(X)
        vae_high.fit(X)

        # Both should work
        assert hasattr(vae_low, "model_")
        assert hasattr(vae_high, "model_")


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for dimensionality reduction module."""

    def test_all_imports(self):
        """Test that all classes can be imported."""
        from endgame.dimensionality_reduction import (
            PCAReducer,
            RandomizedPCA,
            TruncatedSVDReducer,
            KernelPCAReducer,
            ICAReducer,
            UMAPReducer,
            ParametricUMAP,
            TriMAPReducer,
            PHATEReducer,
            PaCMAPReducer,
            VAEReducer,
        )

        # All should be importable
        assert PCAReducer is not None
        assert RandomizedPCA is not None
        assert TruncatedSVDReducer is not None
        assert KernelPCAReducer is not None
        assert ICAReducer is not None
        assert UMAPReducer is not None
        assert ParametricUMAP is not None
        assert TriMAPReducer is not None
        assert PHATEReducer is not None
        assert PaCMAPReducer is not None
        assert VAEReducer is not None

    def test_sklearn_pipeline_compatibility(self, small_data):
        """Test that reducers work in sklearn pipelines."""
        from sklearn.pipeline import Pipeline
        from sklearn.linear_model import LogisticRegression
        from endgame.dimensionality_reduction import PCAReducer, UMAPReducer

        X, y = small_data

        # PCA + Classifier pipeline
        pipe = Pipeline([
            ("reduce", PCAReducer(n_components=5)),
            ("clf", LogisticRegression(max_iter=200)),
        ])
        pipe.fit(X, y)
        preds = pipe.predict(X)
        assert preds.shape == (100,)

    def test_compare_linear_methods(self, medium_data):
        """Compare different linear methods on same data."""
        from endgame.dimensionality_reduction import (
            PCAReducer,
            RandomizedPCA,
            TruncatedSVDReducer,
            ICAReducer,
        )

        X, _ = medium_data
        n_components = 10

        methods = {
            "PCA": PCAReducer(n_components=n_components, random_state=42),
            "RandomizedPCA": RandomizedPCA(n_components=n_components, random_state=42),
            "TruncatedSVD": TruncatedSVDReducer(n_components=n_components, random_state=42),
            "ICA": ICAReducer(n_components=n_components, random_state=42),
        }

        for name, method in methods.items():
            X_reduced = method.fit_transform(X)
            assert X_reduced.shape == (500, n_components), f"{name} failed"

    def test_compare_manifold_methods(self, small_data):
        """Compare different manifold methods on same data."""
        from endgame.dimensionality_reduction import (
            UMAPReducer,
            TriMAPReducer,
            PHATEReducer,
            PaCMAPReducer,
        )

        X, _ = small_data

        methods = {
            "UMAP": UMAPReducer(n_components=2, random_state=42),
            "TriMAP": TriMAPReducer(n_components=2, n_iters=100, verbose=False),
            "PHATE": PHATEReducer(n_components=2, verbose=0),
            "PaCMAP": PaCMAPReducer(n_components=2, num_iters=100),
        }

        for name, method in methods.items():
            X_reduced = method.fit_transform(X)
            assert X_reduced.shape == (100, 2), f"{name} failed"
