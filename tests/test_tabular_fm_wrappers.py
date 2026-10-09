"""Foundation-model wrappers: sklearn contract, clear ImportError without the package, real fit when present."""

import importlib
import importlib.util

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.metrics import r2_score, roc_auc_score

from endgame.models.tabular.exaone import EXAONETabularClassifier
from endgame.models.tabular.iltm import iLTMClassifier, iLTMRegressor
from endgame.models.tabular.kumo import KumoTabularClassifier, KumoTabularRegressor
from endgame.models.tabular.limix import LimiX2Classifier, LimiX2Regressor
from endgame.models.tabular.mitra import MitraClassifier
from endgame.models.tabular.sap_rpt import SAPRPTClassifier, SAPRPTRegressor
from endgame.models.tabular.tabfm import TabFMClassifier
from endgame.models.tabular.tabicl import TabICLClassifier

WRAPPERS = [(EXAONETabularClassifier, "exaonetabular"), (TabFMClassifier, "tabfm"), (TabICLClassifier, "tabicl"),
            (KumoTabularClassifier, "sdm"), (KumoTabularRegressor, "sdm"), (LimiX2Classifier, "limix"),
            (LimiX2Regressor, "limix"), (MitraClassifier, "autogluon.tabular"), (iLTMClassifier, "iltm"),
            (iLTMRegressor, "iltm"), (SAPRPTClassifier, "sap_rpt_oss"), (SAPRPTRegressor, "sap_rpt_oss")]

# Cheap enough to fit in a test when installed (TabFM, LimiX-2 and iLTM pull multi-GB weights; SAP's are gated)
FIT_KWARGS = {
    EXAONETabularClassifier: {"device": "cpu", "compute_dtype": "float32"},
    TabICLClassifier: {"device": "cpu"},
    KumoTabularClassifier: {"size": "small", "n_estimators": 2, "device": "cpu"},
    MitraClassifier: {"fine_tune": False, "device": "cpu"},
}


def _installed(pkg):
    try:
        return importlib.util.find_spec(pkg) is not None
    except ModuleNotFoundError:     # parent package of a dotted name missing
        return False


def _data(n=160):
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, 8))
    return X, X[:, 0] + 0.5 * X[:, 1] + 0.5 * rng.standard_normal(n)


@pytest.mark.parametrize("cls,pkg", WRAPPERS)
def test_contract(cls, pkg):
    est = cls()
    params = est.get_params()
    assert clone(est).get_params() == params and "random_state" in params
    if not _installed(pkg):
        with pytest.raises(ImportError, match="pip install"):
            est.fit(np.zeros((4, 2)), [0, 1, 0, 1])


@pytest.mark.parametrize("cls", list(FIT_KWARGS))
def test_fit_predict(cls):
    pkg = dict(WRAPPERS)[cls]
    if not _installed(pkg):
        pytest.skip(f"{pkg} not installed")
    X, t = _data()
    y = (t > 0).astype(int)
    est = cls(**FIT_KWARGS[cls]).fit(X[:120], y[:120])
    p = est.predict_proba(X[120:])
    assert p.shape == (40, 2) and np.allclose(p.sum(axis=1), 1.0, atol=1e-3)
    assert roc_auc_score(y[120:], p[:, 1]) > 0.7
    assert set(est.predict(X[120:])) <= {0, 1}


def test_kumo_regressor_fit_predict():
    if not _installed("sdm"):
        pytest.skip("structured-data-models not installed")
    X, y = _data()
    est = KumoTabularRegressor(size="small", n_estimators=2, device="cpu", predict_batch_size=16).fit(X[:120], y[:120])
    assert est.predict_quantiles(X[120:]).shape == (40, 999)
    assert r2_score(y[120:], est.predict(X[120:])) > 0.3


@pytest.mark.parametrize("module,name,pkg", [
    ("endgame.models.tabular", "CausiloClassifier", "causilo"),
    ("endgame.models.tabular", "TabLDMEnhancedRegressor", "tabldm"),
    ("endgame.models.boosters", "ChimeraBoostClassifier", "chimeraboost"),
    ("endgame.models.boosters", "CTBoostRegressor", "ctboost"),
])
def test_reexport(module, name, pkg):
    mod = importlib.import_module(module)
    if _installed(pkg):
        assert getattr(mod, name).__module__.split(".")[0] == pkg
    else:
        with pytest.raises(ImportError, match="pip install"):
            getattr(mod, name)
