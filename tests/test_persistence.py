"""Tests for endgame.persistence module."""

import os
import shutil
import tempfile
import warnings

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import endgame as eg
from endgame.core.base import EndgameEstimator
from endgame.persistence import ModelMetadata, load, save
from endgame.persistence._backends import EGM_EXT
from endgame.persistence._detection import detect_backend, has_torch_modules
from endgame.persistence._metadata import (
    CURRENT_FORMAT_VERSION,
    collect_metadata,
)


@pytest.fixture
def sample_data():
    """Create a simple classification dataset."""
    X, y = make_classification(
        n_samples=100, n_features=5, random_state=42
    )
    return X, y


@pytest.fixture
def fitted_lr(sample_data):
    """Return a fitted LogisticRegression."""
    X, y = sample_data
    return LogisticRegression(random_state=42, max_iter=200).fit(X, y)


@pytest.fixture
def tmp_dir():
    """Create a temporary directory, cleaned up after the test."""
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


# ---------- Basic save / load ----------


class TestSaveLoad:
    """Test basic save and load functionality."""

    def test_save_load_sklearn(self, fitted_lr, sample_data, tmp_dir):
        """Round-trip a plain sklearn estimator."""
        X, _ = sample_data
        path = os.path.join(tmp_dir, "model")

        saved_path = save(fitted_lr, path)
        assert saved_path.endswith(EGM_EXT)
        assert os.path.isfile(saved_path)

        loaded = load(saved_path)
        np.testing.assert_array_equal(
            fitted_lr.predict(X), loaded.predict(X)
        )
        np.testing.assert_array_almost_equal(
            fitted_lr.predict_proba(X), loaded.predict_proba(X)
        )

    def test_save_load_with_compression(self, fitted_lr, sample_data, tmp_dir):
        """Save with gzip compression and verify round-trip."""
        X, _ = sample_data
        path = os.path.join(tmp_dir, "model_compressed")

        saved = save(fitted_lr, path, compress=3)
        loaded = load(saved)
        np.testing.assert_array_equal(
            fitted_lr.predict(X), loaded.predict(X)
        )

    def test_save_load_unfitted(self, tmp_dir):
        """Saving an unfitted estimator should work."""
        lr = LogisticRegression()
        path = os.path.join(tmp_dir, "unfitted")

        saved = save(lr, path)
        loaded = load(saved)
        assert loaded.get_params() == lr.get_params()

    def test_save_load_pipeline(self, sample_data, tmp_dir):
        """Persist an sklearn Pipeline."""
        X, y = sample_data
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=42, max_iter=200)),
        ]).fit(X, y)

        path = os.path.join(tmp_dir, "pipeline")
        saved = save(pipe, path)
        loaded = load(saved)
        np.testing.assert_array_equal(
            pipe.predict(X), loaded.predict(X)
        )


# ---------- Metadata ----------


class TestMetadata:
    """Test metadata collection and storage."""

    def test_metadata_fields(self, fitted_lr):
        """Verify metadata has the expected fields."""
        meta = collect_metadata(fitted_lr, "joblib", None)
        assert meta.format_version == CURRENT_FORMAT_VERSION
        assert meta.model_class.endswith("LogisticRegression")
        assert meta.python_version
        assert meta.created_at
        assert meta.endgame_version == eg.__version__
        assert "numpy" in meta.dependencies
        assert meta.is_fitted is True
        assert meta.classes_ is not None
        assert len(meta.classes_) == 2

    def test_metadata_params(self, fitted_lr):
        """Verify model params are captured."""
        meta = collect_metadata(fitted_lr, "joblib", None)
        assert meta.model_params["random_state"] == 42

    def test_metadata_features(self, sample_data, tmp_dir):
        """n_features_in_ is captured after fit."""
        X, y = sample_data
        lr = LogisticRegression(max_iter=200).fit(X, y)
        meta = collect_metadata(lr, "joblib", None)
        assert meta.n_features_in_ == 5

    def test_metadata_roundtrip(self):
        """to_dict / from_dict should round-trip cleanly."""
        meta = ModelMetadata(
            endgame_version="0.3.0",
            model_class="sklearn.linear_model.LogisticRegression",
            model_params={"C": 1.0},
            created_at="2026-01-01T00:00:00+00:00",
            python_version="3.10.0",
            dependencies={"numpy": "1.24.0"},
            classes_=[0, 1],
        )
        d = meta.to_dict()
        restored = ModelMetadata.from_dict(d)
        assert restored.endgame_version == meta.endgame_version
        assert restored.model_class == meta.model_class
        assert restored.classes_ == meta.classes_

    def test_metadata_saved_in_file(self, fitted_lr, tmp_dir):
        """Metadata should be loadable from the saved file."""
        import joblib
        path = os.path.join(tmp_dir, "model")
        saved = save(fitted_lr, path)
        payload = joblib.load(saved)
        meta = payload["metadata"]
        assert meta["format_version"] == CURRENT_FORMAT_VERSION
        assert meta["model_class"].endswith("LogisticRegression")
        assert meta["is_fitted"] is True


# ---------- Detection ----------


class TestDetection:
    """Test backend detection logic."""

    def test_detect_joblib_for_sklearn(self, fitted_lr):
        """Plain sklearn models should use joblib."""
        assert detect_backend(fitted_lr) == "joblib"

    def test_no_torch_modules_sklearn(self, fitted_lr):
        """sklearn models have no torch modules."""
        assert has_torch_modules(fitted_lr) is False

    def test_explicit_backend_override(self, fitted_lr):
        """Explicit backend preference should be honoured."""
        assert detect_backend(fitted_lr, preferred="pickle") == "pickle"


# ---------- Convenience API ----------


class TestConvenienceAPI:
    """Test eg.save / eg.load top-level access."""

    def test_eg_save_load(self, fitted_lr, sample_data, tmp_dir):
        """eg.save() and eg.load() should work."""
        X, _ = sample_data
        path = os.path.join(tmp_dir, "model")
        saved = eg.save(fitted_lr, path)
        loaded = eg.load(saved)
        np.testing.assert_array_equal(
            fitted_lr.predict(X), loaded.predict(X)
        )

    def test_persistence_module_accessible(self):
        """eg.persistence should be importable."""
        mod = eg.persistence
        assert hasattr(mod, "save")
        assert hasattr(mod, "load")
        assert hasattr(mod, "ModelMetadata")


# ---------- EndgameEstimator instance methods ----------


class _DummyEstimator(EndgameEstimator):
    """Minimal EndgameEstimator subclass for testing."""

    def __init__(self, alpha=1.0, random_state=None):
        super().__init__(random_state=random_state)
        self.alpha = alpha

    def fit(self, X, y=None):
        self._is_fitted = True
        self.coef_ = np.ones(X.shape[1]) * self.alpha
        return self

    def predict(self, X):
        return X @ self.coef_


class TestInstanceMethods:
    """Test .save() / .load() on EndgameEstimator subclass."""

    def test_save_load_instance(self, sample_data, tmp_dir):
        X, y = sample_data
        est = _DummyEstimator(alpha=2.5, random_state=7).fit(X, y)
        path = os.path.join(tmp_dir, "dummy")
        saved = est.save(path)

        loaded = _DummyEstimator.load(saved)
        np.testing.assert_array_equal(est.predict(X), loaded.predict(X))
        assert loaded.alpha == 2.5
        assert loaded.random_state == 7


# ---------- Error handling ----------


class TestErrorHandling:
    """Test error paths."""

    def test_load_nonexistent(self, tmp_dir):
        """Loading from a nonexistent path should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load(os.path.join(tmp_dir, "nonexistent.egm"))

    def test_load_corrupted(self, tmp_dir):
        """Loading a corrupted file should raise ValueError."""
        bad_path = os.path.join(tmp_dir, "corrupted.egm")
        with open(bad_path, "wb") as f:
            f.write(b"not a valid model file")
        with pytest.raises(ValueError, match="corrupted"):
            load(bad_path)

    def test_load_invalid_directory(self, tmp_dir):
        """Loading from a dir without metadata.json should raise."""
        bad_dir = os.path.join(tmp_dir, "bad_dir")
        os.makedirs(bad_dir)
        with pytest.raises(ValueError, match="metadata.json"):
            load(bad_dir)


# ---------- Version mismatch ----------


class TestVersionWarning:
    """Test format version mismatch warning."""

    def test_version_mismatch_warns(self, fitted_lr, tmp_dir):
        """Loading a model with a different format version should warn."""
        import joblib

        path = os.path.join(tmp_dir, "model")
        saved = save(fitted_lr, path)

        # Tamper with the format version
        payload = joblib.load(saved)
        payload["metadata"]["format_version"] = 999
        joblib.dump(payload, saved)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            loaded = load(saved)
            version_warnings = [
                x for x in w if "format version" in str(x.message)
            ]
            assert len(version_warnings) >= 1


# ---------- PyTorch backend (conditional) ----------


@pytest.fixture
def _skip_no_torch():
    """Skip test if PyTorch is not available."""
    pytest.importorskip("torch")


try:
    import torch
    import torch.nn as nn
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class _TorchEstimator(EndgameEstimator):
        """Module-level torch estimator so joblib can pickle it."""

        def __init__(self, random_state=None):
            super().__init__(random_state=random_state)
            self.model_ = None

        def fit(self, X, y=None):
            self.model_ = nn.Linear(X.shape[1], 1)
            self._is_fitted = True
            return self

        def predict(self, X):
            self.model_.eval()
            with torch.no_grad():
                t = torch.tensor(X, dtype=torch.float32)
                return self.model_(t).numpy().flatten()


class TestTorchBackend:
    """Test PyTorch directory-format persistence."""

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_save_load_torch_model(self, sample_data, tmp_dir):
        """Round-trip a simple nn.Module-containing estimator."""
        X, y = sample_data

        est = _TorchEstimator(random_state=42).fit(X, y)
        preds_before = est.predict(X)

        path = os.path.join(tmp_dir, "torch_model")
        saved = save(est, path)
        assert saved.endswith(".egd")
        assert os.path.isdir(saved)
        assert os.path.isfile(os.path.join(saved, "metadata.json"))

        loaded = load(saved)
        preds_after = loaded.predict(X)
        np.testing.assert_array_almost_equal(preds_before, preds_after)

    @pytest.mark.usefixtures("_skip_no_torch")
    def test_detect_torch_backend(self):
        """Estimator with nn.Module should be detected as torch backend."""
        class _HasModule(EndgameEstimator):
            def __init__(self):
                super().__init__()
                self.model_ = nn.Linear(5, 1)

        est = _HasModule()
        assert has_torch_modules(est) is True
        assert detect_backend(est) == "torch"
