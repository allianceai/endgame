"""TabICL v2: in-context tabular classifier with fully permissive weights (Qu, Holzmüller, Varoquaux, Le Morvan, ICML 2026).

Wraps ``tabicl.TabICLClassifier`` (https://github.com/soda-inria/tabicl; BSD-3 code and weights, not gated). Column-then-
row attention pretrained on synthetic data; TabArena standing on par with RealTabPFN-2.5, degrades beyond ~100 features.
The default checkpoint (``tabicl-classifier-v2-20260212.ckpt``) downloads on first use.

Install::

    pip install tabicl
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _available():
    try:
        import tabicl  # noqa: F401
        return True
    except ImportError:
        return False


class TabICLClassifier(ClassifierMixin, BaseEstimator):
    """sklearn front for ``tabicl.TabICLClassifier``.

    Parameters
    ----------
    n_estimators : int, default=32
        Ensemble members (feature/class permutations) averaged at inference.
    device : str or None, default=None
        ``None`` lets tabicl pick CUDA when available.
    random_state : int, default=0
    """

    def __init__(self, n_estimators=32, device=None, random_state=0):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state

    def fit(self, X, y):
        if not _available():
            raise ImportError("tabicl is not installed: pip install tabicl")
        from tabicl import TabICLClassifier as _Clf

        X = np.asarray(X, dtype=np.float32)
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        self.n_features_in_ = X.shape[1]
        self._model = _Clf(n_estimators=self.n_estimators, device=self.device, random_state=self.random_state)
        self._model.fit(X, self._le.transform(y))
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict_proba(np.asarray(X, dtype=np.float32)), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]
