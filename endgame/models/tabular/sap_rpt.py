"""SAP-RPT-OSS: SAP's open relational pretrained transformer (sap-rpt-1-oss, successor of ConTextTab; arXiv 2506.10707).

Wraps ``sap_rpt_oss.SAP_RPT_OSS_Classifier`` / ``SAP_RPT_OSS_Regressor`` (https://github.com/SAP-samples/sap-rpt-1-oss).
Embeds column names and cell values with a sentence encoder, so raw strings, categoricals, dates and NaN go in as-is
(pass a DataFrame with meaningful column names). The package's constructor downloads and loads the weights, so this
wrapper builds it in ``fit``. Runs on CUDA when available (fp16 before Ampere, bf16 after), else CPU.

Weights: Hugging Face ``SAP/sap-rpt-1-oss`` (65 MB) is **gated**: accept the terms on the model page and log in
(``hf auth login`` or ``HF_TOKEN``). The README states the checkpoints are for research use only.

Memory: SAP recommends an 80 GB GPU for the default 8192-row context; on a small GPU use
``max_context_size=2048, bagging=1``.

Install::

    pip install "sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git"   # Python >= 3.11
"""

from __future__ import annotations

import importlib.util

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def _available():
    """Installed? (no import: importing has side effects, and a broken dependency should raise its own error)"""
    return importlib.util.find_spec("sap_rpt_oss") is not None


class _SAPRPT(BaseEstimator):
    _cls_name = None

    def __init__(self, bagging=8, max_context_size=8192, test_chunk_size=1000, random_state=42):
        self.bagging = bagging
        self.max_context_size = max_context_size
        self.test_chunk_size = test_chunk_size
        self.random_state = random_state

    def fit(self, X, y):
        if not _available():
            raise ImportError("sap_rpt_oss is not installed: pip install "
                              "'sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git'")
        import sap_rpt_oss

        self._model = getattr(sap_rpt_oss, self._cls_name)(bagging=self.bagging, max_context_size=self.max_context_size,
                                                           test_chunk_size=self.test_chunk_size)
        self._model.seed = self.random_state     # a plain attribute upstream (context subsampling), as TabArena sets it
        self._model.fit(X, y)
        self.n_features_in_ = np.shape(X)[1]
        return self


class SAPRPTClassifier(ClassifierMixin, _SAPRPT):
    """sklearn front for ``sap_rpt_oss.SAP_RPT_OSS_Classifier``.

    Parameters
    ----------
    bagging : int or "auto", default=8
        Context subsamples averaged when the training set exceeds ``max_context_size`` (else a single pass).
    max_context_size : int, default=8192
        Training rows per subsample.
    test_chunk_size : int, default=1000
        Query rows per forward pass.
    random_state : int, default=42
    """

    _cls_name = "SAP_RPT_OSS_Classifier"

    def fit(self, X, y):
        super().fit(X, y)
        self.classes_ = np.asarray(self._model.classes_)
        return self

    def predict_proba(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict_proba(X), dtype=np.float64)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class SAPRPTRegressor(RegressorMixin, _SAPRPT):
    """sklearn front for ``sap_rpt_oss.SAP_RPT_OSS_Regressor``; parameters as :class:`SAPRPTClassifier`."""

    _cls_name = "SAP_RPT_OSS_Regressor"

    def predict(self, X):
        check_is_fitted(self, "_model")
        return np.asarray(self._model.predict(X), dtype=np.float64).reshape(-1)
