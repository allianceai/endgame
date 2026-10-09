"""Mitra-v2: AutoGluon's tabular foundation model, second release (arXiv 2609.04540; Apache-2.0; TabArena top 10, Oct 2026).

Wraps AutoGluon's own sklearn interface ``autogluon.tabular.models.mitra.sklearn_interface.MitraClassifier`` with the
v2 checkpoint ``autogluon/mitra-classifier-2`` (~300 MB, not gated) and TabArena's v2 fine-tuning settings: 50
fine-tuning steps, lr 1e-5 (3e-6 for binary tasks up to 16,384 rows), 10 warm-up steps. TabArena additionally uses
weight decay 0.3 (AutoGluon's interface fixes 0.1), 8-fold bagging and support-size caps, so scores are close to, not
identical with, the leaderboard entry. Classification only: the v2 regressor has a 1,000-bin head that stock AutoGluon
cannot load. At most 10 classes; non-numeric DataFrame columns are ordinal-encoded here, NaN is handled inside Mitra.
Fine-tuning autocasts to bfloat16, which pre-Ampere GPUs only emulate (works, slower; checked on an RTX 2080).

Install::

    pip install "autogluon.tabular[mitra]>=1.6"
"""

from __future__ import annotations

import importlib.util

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.utils.validation import check_is_fitted


def _available():
    return importlib.util.find_spec("autogluon") is not None and importlib.util.find_spec("autogluon.tabular") is not None


class MitraClassifier(ClassifierMixin, BaseEstimator):
    """Mitra-v2 classifier (in-context prediction after a short fine-tune on the training data).

    Parameters
    ----------
    fine_tune : bool, default=True
    fine_tune_steps : int, default=50
    lr : float or None, default=None
        ``None`` uses TabArena's rule: 3e-6 for binary tasks with at most 16,384 rows, else 1e-5.
    warmup_steps : int, default=10
    n_estimators : int, default=1
    hf_model : str, default="autogluon/mitra-classifier-2"
        Hugging Face repo or local directory (``"autogluon/mitra-classifier"`` is v1).
    device : str, default="auto"
    random_state : int, default=0
    """

    def __init__(self, fine_tune=True, fine_tune_steps=50, lr=None, warmup_steps=10, n_estimators=1,
                 hf_model="autogluon/mitra-classifier-2", device="auto", random_state=0):
        self.fine_tune = fine_tune
        self.fine_tune_steps = fine_tune_steps
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.n_estimators = n_estimators
        self.hf_model = hf_model
        self.device = device
        self.random_state = random_state

    def _numeric(self, X, fit=False):
        import pandas as pd
        if not isinstance(X, pd.DataFrame):
            return np.asarray(X, dtype=np.float32)
        if fit:
            self._cat = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
            self._enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan).fit(
                X[self._cat].astype(str)) if self._cat else None
        if self._cat:
            X = X.copy()
            X[self._cat] = self._enc.transform(X[self._cat].astype(str))
        return X.to_numpy(dtype=np.float32)

    def fit(self, X, y):
        if not _available():
            raise ImportError("autogluon.tabular is not installed: pip install 'autogluon.tabular[mitra]>=1.6'")
        import torch
        from autogluon.tabular.models.mitra.sklearn_interface import MitraClassifier as _Clf

        X = self._numeric(X, fit=True)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_features_in_ = X.shape[1]
        lr = self.lr if self.lr is not None else (3e-6 if len(self.classes_) == 2 and len(X) <= 16_384 else 1e-5)
        device = ("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device
        self._model = _Clf(n_estimators=self.n_estimators, device=device, fine_tune=self.fine_tune,
                           fine_tune_steps=self.fine_tune_steps, hf_model=self.hf_model, lr=lr,
                           warmup_steps=self.warmup_steps, seed=self.random_state, verbose=False)
        self._model.fit(X, self._le.transform(y))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict_proba(self._numeric(X)), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
