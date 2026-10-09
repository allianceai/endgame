"""Kumo-Tabular: NVIDIA's open-weights in-context tabular foundation model (Oct 2026; #1 on TabArena).

Wraps ``sdm.models.KumoTabular`` from NVIDIA's structured-data-models library
(https://github.com/NVIDIA/structured-data-models; blog https://huggingface.co/blog/nvidia/kumo-tabular). Three sizes,
``"large"`` (~214 M parameters, the TabArena "Kumo-Tabular" entry), ``"medium"`` and ``"small"``; separate
classification and regression checkpoints from the ungated Hugging Face repo ``nvidia/Kumo-Tabular`` (OpenMDW-1.1
weights, Apache-2.0 code; commercial use allowed), downloaded on first fit (0.11-0.86 GB each). Runs locally, no API
key. Categorical (string / bool / ``category``) columns and NaN are handled natively from a pandas DataFrame; more than
10 classes go through error-correcting output codes. Pretrained on contexts up to 60k rows and 100 columns (the
recipe keeps 500 columns per ensemble member). Regression predicts 999 quantiles; ``predict`` returns their mean.

Install (git only; the PyPI ``structured-data-models`` is a placeholder and ``sdm`` an unrelated package)::

    pip install "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19"

Memory: a multi-member fit on CUDA offloads the context KV cache to pinned host memory, which grows with rows x
members x classes; lower ``n_estimators`` on large data.
"""

from __future__ import annotations

import importlib.util

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted

_NETWORKS = {}      # (task, size, device) -> loaded network, shared by every fit in the process
_INSTALL = ("structured-data-models is not installed: pip install 'structured-data-models @ "
            "git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19'")


def _available():
    """Installed? (no import, so a broken dependency raises its own error at fit)"""
    return importlib.util.find_spec("sdm") is not None


def _frame(X):
    import pandas as pd
    if isinstance(X, pd.DataFrame):
        return X.reset_index(drop=True)
    X = np.asarray(X)
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


class _KumoTabular(BaseEstimator):
    _task = None

    def __init__(self, size="large", n_estimators=None, estimator_batch_size=1, predict_batch_size=4096, device="auto",
                 random_state=0):
        self.size = size
        self.n_estimators = n_estimators
        self.estimator_batch_size = estimator_batch_size
        self.predict_batch_size = predict_batch_size
        self.device = device
        self.random_state = random_state

    def _device(self):
        import torch
        return torch.device(("cuda" if torch.cuda.is_available() else "cpu") if self.device == "auto" else self.device)

    def _fit(self, X, y):
        if not _available():
            raise ImportError(_INSTALL)
        import pandas as pd
        import sdm
        import torch

        if not hasattr(sdm, "models"):
            raise ImportError("the installed 'sdm' is an unrelated PyPI package; " + _INSTALL)
        device = self._device()
        key = (self._task, self.size, str(device))
        if key not in _NETWORKS:
            _NETWORKS[key] = sdm.models.KumoTabular(task=self._task, size=self.size, device=device).models[self._task]
        # A fresh estimator around the shared network keeps each fitted wrapper's context cache its own
        self._model = sdm.models.KumoTabular(task=self._task, size=self.size, pretrained=False, device="meta")
        self._model.models[self._task] = _NETWORKS[key]
        self._model.eval()

        X = _frame(X)
        self.n_features_in_ = X.shape[1]
        self._stypes = sdm.infer_stypes(X, _low_cardinality="infer")    # as TabArena: low-cardinality ints -> categorical
        x = sdm.TableTensor.from_pandas(df=X, stypes=self._stypes, device=device)
        y = sdm.TableTensor.from_pandas(df=pd.DataFrame({"target": y}), device=device,
                                        stypes={"target": "categorical" if self._task == "classification" else "numerical"})
        n = self.n_estimators or (16 if self.size == "large" else 8)    # TabArena's leaderboard setting
        with torch.inference_mode(), torch.amp.autocast(device.type, torch.float16, enabled=device.type == "cuda"):
            self._model.fit(x=x, y=y, num_estimators=n, estimator_batch_size=self.estimator_batch_size,
                            generator=torch.Generator(device).manual_seed(self.random_state))
        self._device_ = device
        return self

    def _predict(self, X):
        """(output column names, values) with the query run in ``predict_batch_size``-row chunks."""
        check_is_fitted(self, "_model")
        import sdm
        import torch

        X, cols, parts = _frame(X), None, []
        for start in range(0, len(X), self.predict_batch_size):
            q = sdm.TableTensor.from_pandas(df=X.iloc[start:start + self.predict_batch_size], stypes=self._stypes,
                                            device=self._device_)
            with torch.inference_mode(), torch.amp.autocast(self._device_.type, torch.float16,
                                                            enabled=self._device_.type == "cuda"):
                out = self._model.predict(q)
            cols = list(out.columns[sdm.Stype.numerical])
            parts.append(out.numerical.float().cpu().numpy())
        return cols, np.concatenate(parts).astype(np.float64)


class KumoTabularClassifier(ClassifierMixin, _KumoTabular):
    """sklearn front for ``sdm.models.KumoTabular(task="classification")``; the network is shared per process.

    Parameters
    ----------
    size : {"large", "medium", "small"}, default="large"
        TabArena's Kumo-Tabular, Kumo-Tabular-Medium and Kumo-Tabular-Small.
    n_estimators : int or None, default=None
        Ensemble members (column/row permutations); ``None`` uses TabArena's 16 for large and 8 otherwise.
    estimator_batch_size : int, default=1
        Members run per forward pass (GPU memory).
    predict_batch_size : int, default=4096
        Query rows per forward pass (GPU memory).
    device : str, default="auto"
        "cuda" when available, else "cpu" (slow for large).
    random_state : int, default=0

    Examples
    --------
    >>> clf = KumoTabularClassifier(size="small").fit(X_train, y_train)   # DataFrame with categoricals is fine
    >>> proba = clf.predict_proba(X_test)
    """

    _task = "classification"

    def fit(self, X, y):
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        return self._fit(X, self._le.transform(y))

    def predict_proba(self, X):
        cols, values = self._predict(X)
        proba = np.zeros((len(values), len(self.classes_)))
        proba[:, [int(c) for c in cols]] = values      # columns are str(class code) in arbitrary order
        return proba

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


class KumoTabularRegressor(RegressorMixin, _KumoTabular):
    """sklearn front for ``sdm.models.KumoTabular(task="regression")``; ``predict`` is the mean of 999 quantiles.

    Parameters are those of :class:`KumoTabularClassifier`. ``predict_quantiles`` returns all 999 (q0.001 .. q0.999).
    """

    _task = "regression"

    def fit(self, X, y):
        return self._fit(X, np.asarray(y, dtype=np.float64))

    def predict_quantiles(self, X):
        return self._predict(X)[1]

    def predict(self, X):
        return self.predict_quantiles(X).mean(axis=1)
