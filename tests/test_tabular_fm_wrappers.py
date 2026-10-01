"""EXAONE-Tabular, TabFM and TabICL wrappers: sklearn contract, clear ImportError without the package, real fit when present."""

import importlib

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import roc_auc_score

from endgame.models.tabular.exaone import EXAONETabularClassifier
from endgame.models.tabular.tabfm import TabFMClassifier
from endgame.models.tabular.tabicl import TabICLClassifier

WRAPPERS = [(EXAONETabularClassifier, "exaonetabular"), (TabFMClassifier, "tabfm"), (TabICLClassifier, "tabicl")]


def _installed(pkg):
    try:
        importlib.import_module(pkg)
        return True
    except ImportError:
        return False


@pytest.mark.parametrize("cls,pkg", WRAPPERS)
def test_contract(cls, pkg):
    est = cls()
    params = est.get_params()
    assert clone(est).get_params() == params and "random_state" in params
    if not _installed(pkg):
        with pytest.raises(ImportError, match="pip install"):
            est.fit(np.zeros((4, 2)), [0, 1, 0, 1])


@pytest.mark.parametrize("cls,pkg", [w for w in WRAPPERS if w[1] != "tabfm"])   # TabFM weights are several GB
def test_fit_predict(cls, pkg):
    if not _installed(pkg):
        pytest.skip(f"{pkg} not installed")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((160, 8))
    y = (X[:, 0] + 0.5 * X[:, 1] + 0.5 * rng.standard_normal(160) > 0).astype(int)
    kw = {"device": "cpu"} if cls is TabICLClassifier else {"device": "cpu", "compute_dtype": "float32"}
    est = cls(**kw).fit(X[:120], y[:120])
    p = est.predict_proba(X[120:])
    assert p.shape == (40, 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-4)
    assert roc_auc_score(y[120:], p[:, 1]) > 0.7
    assert set(est.predict(X[120:])) <= {0, 1}
