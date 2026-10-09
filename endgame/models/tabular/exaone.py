"""EXAONE-Tabular: LG AI Research's in-context tabular foundation model (2026).

Wraps ``exaonetabular`` (Eo et al., "EXAONE Tabular 1.0: Technical Report", arXiv 2608.25774;
https://github.com/LGAI-Research/EXAONE-Tabular). A 20.8 M-parameter Cross-Axis Summary Transformer trained on a
synthetic prior; zero-shot in-context classification, at most 10 native classes, and the classifier keeps the 100
columns with the highest attention when given more. Weights (Hugging Face ``LG-AI-Research/EXAONE-Tabular``, not
gated) are under the EXAONE AI Model License 1.2-NC: research and education only.

Install::

    pip install "exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git"   # Python >= 3.11, torch >= 2.6
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _available():
    try:
        import exaonetabular  # noqa: F401
        return True
    except ImportError:
        return False


def _allow_pre_ampere_gpus():
    """EXAONE's attention pins the flash or memory-efficient SDPA kernel; flash needs an Ampere (sm_80) GPU, so on older
    cards (e.g. RTX 20xx) PyTorch raises "No available kernel". Route those calls to the memory-efficient kernel, or to
    the math kernel where mem-efficient cannot serve the head width."""
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] >= 8:
        return
    import exaonetabular.model.attention as att
    from torch.nn.attention import SDPBackend
    if getattr(att, "_endgame_patched", False):
        return
    original = att._select_sdpa_backend

    def select(query_length, context_length, head_width, *, on_cuda, policy):
        if not on_cuda:
            return original(query_length, context_length, head_width, on_cuda=on_cuda, policy=policy)
        return SDPBackend.EFFICIENT_ATTENTION if head_width % att.MEM_EFFICIENT_HEAD_WIDTH_MULTIPLE == 0 else SDPBackend.MATH

    att._select_sdpa_backend = select
    att._endgame_patched = True


class EXAONETabularClassifier(ClassifierMixin, BaseEstimator):
    """sklearn front for ``exaonetabular.EXAONETabularClassifier`` (fit stores the context, predict_proba runs the model).

    Parameters
    ----------
    ensemble_count : int, default=8
        Ensemble members (input permutations / preprocessings) averaged at inference.
    compute_dtype : {"float16", "float32", "bfloat16"}, default="float16"
    device : str, default="auto"
    random_state : int, default=0
    """

    def __init__(self, ensemble_count=8, compute_dtype="float16", device="auto", random_state=0):
        self.ensemble_count = ensemble_count
        self.compute_dtype = compute_dtype
        self.device = device
        self.random_state = random_state

    def _device(self):
        if self.device != "auto":
            return self.device
        import torch
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        if not _available():
            raise ImportError("exaonetabular is not installed: pip install "
                              "'exaonetabular @ git+https://github.com/LGAI-Research/EXAONE-Tabular.git'")
        from exaonetabular import EXAONETabularClassifier as _Clf

        if self._device().startswith("cuda"):
            _allow_pre_ampere_gpus()
        X = np.asarray(X, dtype=np.float32)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_features_in_ = X.shape[1]
        self._model = _Clf.from_pretrained(device=self._device(), compute_dtype=self.compute_dtype, seed=self.random_state,
                                           ensemble_count=self.ensemble_count)
        self._model.fit(X, self._le.transform(y))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
