"""Tests for LLM-based oversampler."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

try:
    import transformers
    import torch
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


@pytest.fixture
def imbalanced_binary():
    """90/10 binary classification (small)."""
    X, y = make_classification(
        n_samples=50,
        n_features=3,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0,
        random_state=42,
    )
    return X, y


# ============================================================================
# Serialization / parsing roundtrip (no LLM needed)
# ============================================================================


class TestSerializationParsing:
    """Test row serialization and parsing independently of LLM."""

    def test_serialize_row(self):
        from endgame.preprocessing.imbalance_llm import _serialize_row

        row = np.array([1.5, -2.3, 0.0])
        feature_names = ["f0", "f1", "f2"]
        text = _serialize_row(row, feature_names, "Class", 1)

        assert "f0 is 1.5" in text
        assert "f1 is -2.3" in text
        assert "f2 is 0" in text
        assert "Class is 1" in text

    def test_parse_generated_text(self):
        from endgame.preprocessing.imbalance_llm import _parse_generated_text

        text = "f0 is 1.5, f1 is -2.3, f2 is 0, Class is 1"
        feature_names = ["f0", "f1", "f2"]

        row = _parse_generated_text(text, feature_names, "Class", 3)
        assert row is not None
        np.testing.assert_almost_equal(row[0], 1.5)
        np.testing.assert_almost_equal(row[1], -2.3)
        np.testing.assert_almost_equal(row[2], 0.0)

    def test_parse_incomplete_text(self):
        from endgame.preprocessing.imbalance_llm import _parse_generated_text

        text = "f0 is 1.5"
        feature_names = ["f0", "f1", "f2", "f3", "f4", "f5"]

        # Only 1 out of 6 features parsed => None (below 50% threshold)
        row = _parse_generated_text(text, feature_names, "Class", 6)
        assert row is None

    def test_roundtrip(self):
        from endgame.preprocessing.imbalance_llm import _serialize_row, _parse_generated_text

        row = np.array([3.14, -1.0, 2.718])
        feature_names = ["a", "b", "c"]
        text = _serialize_row(row, feature_names, "Y", 0)
        parsed = _parse_generated_text(text, feature_names, "Y", 3)

        assert parsed is not None
        np.testing.assert_almost_equal(parsed, row, decimal=3)


# ============================================================================
# GReaTResampler (requires transformers + torch)
# ============================================================================


@pytest.mark.skipif(not HAS_TRANSFORMERS, reason="transformers not installed")
@pytest.mark.slow
class TestGReaTResampler:
    """Tests for GReaTResampler (slow, requires model download)."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_llm import GReaTResampler

        X, y = imbalanced_binary
        sampler = GReaTResampler(
            model_name="distilgpt2",
            n_epochs=1,
            batch_size=4,
            max_length=128,
            device="cpu",
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert X_res.shape[1] == X.shape[1]
        assert len(X_res) == len(y_res)


# ============================================================================
# LLM_SAMPLERS dict
# ============================================================================


class TestLLMSamplersDict:
    """Tests for the LLM_SAMPLERS registry."""

    def test_dict_exists(self):
        from endgame.preprocessing.imbalance_llm import LLM_SAMPLERS

        assert "great" in LLM_SAMPLERS
        assert len(LLM_SAMPLERS) == 1
