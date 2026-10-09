"""iLTM: integrated Large Tabular Model (Bonet et al., arXiv 2511.15941; TabArena's best "tuned + ensembled" FM after TabDPT).

Wraps ``iltm.iLTMClassifier`` / ``iltm.iLTMRegressor`` (https://github.com/AI-sandbox/iLTM, Apache-2.0 code and
weights, not gated). A hypernetwork pretrained across datasets generates an MLP from tree embeddings (XGBoost or
CatBoost) plus retrieval, then fine-tunes it on the training data; an ensemble of 12 by default. The checkpoint
(~2.2 GB for every choice except ``"r128bn"``, 307 MB) downloads on first fit into ``$ILTM_CKPT_DIR`` or
``~/.cache/iltm``. At most 100 classes. No flash-attention or Ampere requirement.

The package's own constructor downloads the checkpoint and rewrites its parameters, so this wrapper builds it in
``fit`` (keeping ``clone`` cheap), and restores the global torch flags iltm sets (TF32 on import, cuDNN determinism
on fit) so other models in the process are unaffected.

Install::

    pip install iltm      # Python >= 3.11, torch >= 2.8
"""

from __future__ import annotations

import importlib.util
from contextlib import contextmanager

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _available():
    """Installed? (no import: importing has side effects, and a broken dependency should raise its own error)"""
    return importlib.util.find_spec("iltm") is not None


@contextmanager
def _keep_torch_flags():
    import torch
    saved = (torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(saved[0])
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.benchmark,
         torch.backends.cudnn.deterministic) = saved[1:]


class _ILTM(BaseEstimator):
    _cls_name = None

    def __init__(self, checkpoint="xgbrconcat", n_ensemble=12, finetuning=True, device="auto", random_state=0):
        self.checkpoint = checkpoint
        self.n_ensemble = n_ensemble
        self.finetuning = finetuning
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, eval_set=None):
        """``eval_set=(X_val, y_val)`` enables iLTM's adaptive retrieval weight and early stopping, as TabArena runs it."""
        if not _available():
            raise ImportError("iltm is not installed: pip install iltm")
        import pandas as pd
        import torch

        with _keep_torch_flags():
            import iltm
            device = ("cuda:0" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
            cat = ([i for i, d in enumerate(X.dtypes) if not pd.api.types.is_numeric_dtype(d)]
                   if isinstance(X, pd.DataFrame) else None)    # object/string columns are also auto-detected
            self._model = getattr(iltm, self._cls_name)(checkpoint=self.checkpoint, n_ensemble=self.n_ensemble,
                                                        finetuning=self.finetuning, device=device,
                                                        seed=self.random_state, cat_features=cat)
            self._model.fit(X, y, eval_set=eval_set)
        self.n_features_in_ = np.shape(X)[1]
        return self

    def _call(self, method, X):
        check_is_fitted(self, "_model")
        with _keep_torch_flags():
            return np.asarray(getattr(self._model, method)(X))


class iLTMClassifier(ClassifierMixin, _ILTM):
    """sklearn front for ``iltm.iLTMClassifier`` (any label type; numpy or pandas; NaN allowed).

    Parameters
    ----------
    checkpoint : str, default="xgbrconcat"
        One of ``iltm.AVAILABLE_CHECKPOINTS`` (XGBoost/CatBoost tree-embedding and retrieval variants).
    n_ensemble : int, default=12
    finetuning : bool, default=True
        Fine-tune the generated network on the training data (most of the fit time).
    device : str, default="auto"
    random_state : int, default=0
    """

    _cls_name = "iLTMClassifier"

    def fit(self, X, y, eval_set=None):
        super().fit(X, y, eval_set)
        self.classes_ = np.asarray(self._model.classes_)
        return self

    def predict_proba(self, X):
        return self._call("predict_proba", X).astype(np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class iLTMRegressor(RegressorMixin, _ILTM):
    """sklearn front for ``iltm.iLTMRegressor``; parameters as :class:`iLTMClassifier`."""

    _cls_name = "iLTMRegressor"

    def predict(self, X):
        return self._call("predict", X).astype(np.float64).reshape(-1)
