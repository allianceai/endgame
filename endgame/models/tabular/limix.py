"""LimiX-2: StableAI's in-context tabular foundation model (arXiv 2609.17488; #2 on TabArena, Oct 2026).

Wraps ``inference.v2_0.predictor.LimiXPredictor`` from the LimiX package (https://github.com/limix-ldm-ai/LimiX).
LimiX has no fit step: every predict runs the training table and the query rows through an ensemble of preprocessing
pipelines (32 for classification, 8 for regression in the packaged no-retrieval configs TabArena uses). String and
category columns and NaN are handled natively from a DataFrame; 2-10 classes. Weights ``stable-ai/LimiX-2``
(``LimiX-2.ckpt``, 1.6 GB, ~400 M parameters, not gated) download on first fit, pinned to TabArena's revision; they are
under the StableAI LimiX Non-Commercial License v1.0. LimiX v1 (LimiX-16M) is superseded by LimiX-2 and not wrapped.

Install (no dependency resolution: that commit pins ``torch==2.9.1``; it installs generic top-level packages named
``inference``, ``model``, ``utils`` and ``config``)::

    pip install --no-deps "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@774aa3e1a994cbe38f33758e3d663e9951855554"
    pip install einops kditransform nvtx
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

_REPO, _FILE, _REVISION = "stable-ai/LimiX-2", "LimiX-2.ckpt", "20c07a07801973a0aec57c31062bdb2ea1cda2b2"
_CONFIGS = {"Classification": "cls_default_noretrieval_v2.json", "Regression": "reg_default_noretrieval_v2.json"}


def _available():
    return importlib.util.find_spec("limix") is not None and importlib.util.find_spec("inference") is not None


def _patch_cache_dir(predictor_cls):
    """LimiXPredictor's constructor creates a cache directory under a hard-coded cluster path (``/mnt/public``) even
    with caching off, which fails on other machines; fall back to ``~/.cache/limix`` (TabArena's fix)."""
    cache_cls = predictor_cls.CacheManager
    if getattr(cache_cls, "_endgame_patched", False):
        return
    original = cache_cls.__init__

    def init(self, cache_dir="/mnt/public/infe_cache", *args, **kwargs):
        try:
            original(self, cache_dir, *args, **kwargs)
        except OSError:
            fallback = Path.home() / ".cache" / "limix" / "infe_cache"
            fallback.mkdir(parents=True, exist_ok=True)
            original(self, str(fallback), *args, **kwargs)

    cache_cls.__init__ = init
    cache_cls._endgame_patched = True


def _frame(X):
    """DataFrames go in with ``category``/``string`` columns as ``object`` (what LimiX encodes); arrays as float."""
    import pandas as pd
    if not isinstance(X, pd.DataFrame):
        return np.asarray(X, dtype=np.float32)
    cols = [c for c, d in X.dtypes.items() if isinstance(d, pd.CategoricalDtype)
            or (pd.api.types.is_string_dtype(d) and not pd.api.types.is_object_dtype(d))]
    return X.astype(dict.fromkeys(cols, object)).reset_index(drop=True) if cols else X.reset_index(drop=True)


class _LimiX2(BaseEstimator):
    _task = None

    def __init__(self, n_estimators=None, device="auto", random_state=0, model_path=None):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.model_path = model_path

    def _fit(self, X, y):
        if not _available():
            raise ImportError("LimiX is not installed: pip install --no-deps 'LimiX @ git+https://github.com/"
                              "limix-ldm-ai/LimiX.git@774aa3e1a994cbe38f33758e3d663e9951855554' einops kditransform nvtx")
        import json
        from importlib import resources

        import torch
        from huggingface_hub import hf_hub_download
        from inference.v2_0.predictor import LimiXPredictor

        _patch_cache_dir(LimiXPredictor)
        config = json.loads(resources.files("config").joinpath(_CONFIGS[self._task]).read_text())
        if self.n_estimators is not None:
            config["pipelines"] = config["pipelines"][:self.n_estimators]
        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        self._model = LimiXPredictor(device=torch.device(device), inference_config=config, seed=self.random_state,
                                     model_path=self.model_path or hf_hub_download(_REPO, _FILE, revision=_REVISION))
        self._X, self._y = _frame(X), y
        self.n_features_in_ = self._X.shape[1]
        return self

    def _predict(self, X):
        check_is_fitted(self, "_model")
        out = self._model.predict(self._X, self._y, _frame(X), task_type=self._task)
        return np.asarray(out.detach().float().cpu() if hasattr(out, "detach") else out, dtype=np.float64)


class LimiX2Classifier(ClassifierMixin, _LimiX2):
    """sklearn front for LimiX-2 classification (2-10 classes; ``fit`` stores the context, ``predict_proba`` runs it).

    Parameters
    ----------
    n_estimators : int or None, default=None
        Keep the first n preprocessing pipelines of the packaged config (``None``: all 32, as on TabArena).
    device : str, default="auto"
    random_state : int, default=0
    model_path : str or None, default=None
        Local checkpoint; ``None`` downloads ``stable-ai/LimiX-2`` at TabArena's pinned revision.
    """

    _task = "Classification"

    def fit(self, X, y):
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        return self._fit(X, self._le.transform(y))

    def predict_proba(self, X):
        return self._predict(X)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class LimiX2Regressor(RegressorMixin, _LimiX2):
    """sklearn front for LimiX-2 regression (8 pipelines by default); parameters as :class:`LimiX2Classifier`."""

    _task = "Regression"

    def fit(self, X, y):
        return self._fit(X, np.asarray(y, dtype=np.float32))

    def predict(self, X):
        return self._predict(X).reshape(-1)
