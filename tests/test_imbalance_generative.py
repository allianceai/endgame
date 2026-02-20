"""Tests for generative model-based oversamplers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

# Optional dependency checks
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ctgan
    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False

try:
    import xgboost
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


@pytest.fixture
def imbalanced_binary():
    """90/10 binary classification (small for speed)."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0,
        random_state=42,
    )
    return X, y


# ============================================================================
# CTGANResampler
# ============================================================================


@pytest.mark.skipif(not HAS_CTGAN, reason="ctgan not installed")
class TestCTGANResampler:
    """Tests for CTGANResampler."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import CTGANResampler

        X, y = imbalanced_binary
        sampler = CTGANResampler(n_epochs=2, batch_size=32, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert X_res.shape[1] == X.shape[1]

    def test_output_shape(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import CTGANResampler

        X, y = imbalanced_binary
        sampler = CTGANResampler(n_epochs=2, batch_size=32, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)


# ============================================================================
# ForestFlowResampler
# ============================================================================


@pytest.mark.skipif(not HAS_XGBOOST, reason="xgboost not installed")
class TestForestFlowResampler:
    """Tests for ForestFlowResampler."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import ForestFlowResampler

        X, y = imbalanced_binary
        sampler = ForestFlowResampler(
            n_estimators=10, n_steps=5, random_state=42
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert X_res.shape[1] == X.shape[1]

    def test_output_shape(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import ForestFlowResampler

        X, y = imbalanced_binary
        sampler = ForestFlowResampler(
            n_estimators=10, n_steps=5, random_state=42
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)

    def test_deterministic(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import ForestFlowResampler

        X, y = imbalanced_binary
        X1, y1 = ForestFlowResampler(
            n_estimators=10, n_steps=5, random_state=42
        ).fit_resample(X, y)
        X2, y2 = ForestFlowResampler(
            n_estimators=10, n_steps=5, random_state=42
        ).fit_resample(X, y)

        np.testing.assert_array_almost_equal(X1, X2)

    def test_noise_type_uniform(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import ForestFlowResampler

        X, y = imbalanced_binary
        sampler = ForestFlowResampler(
            n_estimators=10, n_steps=5, noise_type="uniform", random_state=42
        )
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


# ============================================================================
# TabDDPMResampler
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTabDDPMResampler:
    """Tests for TabDDPMResampler."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabDDPMResampler

        X, y = imbalanced_binary
        sampler = TabDDPMResampler(
            n_timesteps=10,
            hidden_dims=[32, 32],
            n_epochs=2,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert X_res.shape[1] == X.shape[1]

    def test_output_shape(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabDDPMResampler

        X, y = imbalanced_binary
        sampler = TabDDPMResampler(
            n_timesteps=10,
            hidden_dims=[32, 32],
            n_epochs=2,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)

    def test_device_cpu(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabDDPMResampler

        X, y = imbalanced_binary
        sampler = TabDDPMResampler(
            n_timesteps=10,
            hidden_dims=[32],
            n_epochs=1,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


# ============================================================================
# TabSynResampler
# ============================================================================


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")
class TestTabSynResampler:
    """Tests for TabSynResampler."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabSynResampler

        X, y = imbalanced_binary
        sampler = TabSynResampler(
            latent_dim=4,
            vae_hidden_dims=[16],
            vae_epochs=2,
            diffusion_hidden_dims=[16, 16],
            diffusion_epochs=2,
            n_timesteps=10,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert X_res.shape[1] == X.shape[1]

    def test_output_shape(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabSynResampler

        X, y = imbalanced_binary
        sampler = TabSynResampler(
            latent_dim=4,
            vae_hidden_dims=[16],
            vae_epochs=2,
            diffusion_hidden_dims=[16, 16],
            diffusion_epochs=2,
            n_timesteps=10,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)

    def test_device_cpu(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_generative import TabSynResampler

        X, y = imbalanced_binary
        sampler = TabSynResampler(
            latent_dim=4,
            vae_hidden_dims=[16],
            vae_epochs=1,
            diffusion_hidden_dims=[16],
            diffusion_epochs=1,
            n_timesteps=10,
            batch_size=32,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


# ============================================================================
# GENERATIVE_SAMPLERS dict
# ============================================================================


class TestGenerativeSamplersDict:
    """Tests for the GENERATIVE_SAMPLERS registry."""

    def test_dict_exists_and_complete(self):
        from endgame.preprocessing.imbalance_generative import GENERATIVE_SAMPLERS

        assert "ctgan" in GENERATIVE_SAMPLERS
        assert "forest_flow" in GENERATIVE_SAMPLERS
        assert "tabddpm" in GENERATIVE_SAMPLERS
        assert "tabsyn" in GENERATIVE_SAMPLERS
        assert len(GENERATIVE_SAMPLERS) == 4
