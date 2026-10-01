"""TabFM: Google Research's zero-shot tabular foundation model (June 2026).

Wraps the ``tabfm`` package (https://github.com/google-research/tabfm; weights ``google/tabfm-1.0.0-pytorch`` on
Hugging Face, ~1.6 B parameters, hybrid row/column attention, trained on structural-causal-model synthetic data).
Zero-shot in-context prediction, at most 10 classes and 500 features per ensemble member. Code is Apache-2.0; the
weights are under the TabFM non-commercial licence v1.0 (no production use). The first ``fit`` downloads the
classification checkpoint (several GB) into the Hugging Face cache; set ``HF_HOME`` to put it elsewhere.

Install::

    pip install "tabfm[pytorch]"      # Python >= 3.11

Memory: bf16 weights alone are ~3.3 GB; keep ``batch_size=1`` and ``use_amp=True`` on an 8 GB GPU and reduce
``n_estimators`` if inference runs out of memory.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

_MODEL = {}     # one loaded network per process (1.6 B parameters)


def _available():
    try:
        import tabfm  # noqa: F401
        return True
    except ImportError:
        return False


def _load(device):
    if _MODEL.get("device") != device:
        from tabfm import tabfm_v1_0_0_pytorch as m
        _MODEL.update(cls=m.load(model_type="classification", device=device), device=device)
    return _MODEL["cls"]


class TabFMClassifier(ClassifierMixin, BaseEstimator):
    """sklearn front for ``tabfm.TabFMClassifier`` sharing one loaded network per process.

    Parameters
    ----------
    n_estimators : int, default=16
        Ensemble members (default of the package is 32; 16 halves the inference time at n ~ 1,000).
    batch_size : int, default=1
        Ensemble members per forward pass (memory).
    use_amp : bool, default=True
    softmax_temperature : float, default=0.9
    device : str, default="auto"
        Where the network lives ("cuda", "cpu" or "auto"); on CPU a fit at n ~ 300 takes minutes.
    random_state : int, default=0
    """

    def __init__(self, n_estimators=16, batch_size=1, use_amp=True, softmax_temperature=0.9, device="auto", random_state=0):
        self.n_estimators = n_estimators
        self.batch_size = batch_size
        self.use_amp = use_amp
        self.softmax_temperature = softmax_temperature
        self.device = device
        self.random_state = random_state

    def _device(self):
        if self.device != "auto":
            return self.device
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"

    def fit(self, X, y):
        if not _available():
            raise ImportError("tabfm is not installed: pip install 'tabfm[pytorch]'")
        from tabfm import TabFMClassifier as _Clf

        X = np.asarray(X, dtype=np.float32)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_features_in_ = X.shape[1]
        self._model = _Clf(model=_load(self._device()), n_estimators=self.n_estimators, batch_size=self.batch_size, use_amp=self.use_amp,
                           softmax_temperature=self.softmax_temperature, random_state=self.random_state)
        self._model.fit(X, self._le.transform(y))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
