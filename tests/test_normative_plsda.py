"""Tests for NormativeDeviation, PLSDAClassifier and bootstrap_ci."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from endgame.models import PLSDAClassifier
from endgame.preprocessing import NormativeDeviation
from endgame.utils.metrics import bootstrap_ci


def _norm_data(seed=0, n_ref=80, n_pat=40):
    rng = np.random.RandomState(seed)
    age = rng.uniform(50, 80, n_ref + n_pat)
    sex = rng.randint(0, 2, n_ref + n_pat)
    is_ref = np.r_[np.ones(n_ref), np.zeros(n_pat)]
    # feature shrinks with age; patients have an extra deficit of 2 SD
    vol = 10 - 0.05 * age + 0.4 * sex + rng.randn(n_ref + n_pat) * 0.5 - 1.0 * (1 - is_ref)
    return pd.DataFrame({"age": age, "sex": sex, "is_ref": is_ref, "vol": vol})


def test_normative_deviation_recovers_patient_deficit():
    df = _norm_data()
    nd = NormativeDeviation(covariates=["age", "sex"], reference="is_ref").fit(df)
    z = nd.transform(df)
    assert "is_ref" not in z.columns and list(z.columns) == ["age", "sex", "vol"]
    ref_z, pat_z = z.loc[df["is_ref"] == 1, "vol"], z.loc[df["is_ref"] == 0, "vol"]
    assert abs(ref_z.mean()) < 0.15 and 0.8 < ref_z.std() < 1.2   # reference ~ N(0, 1)
    assert -2.8 < pat_z.mean() < -1.2                                # patients ~ 2 SD below the norm
    # deviation is age-independent in the reference group
    assert abs(np.corrcoef(df.loc[df["is_ref"] == 1, "age"], ref_z)[0, 1]) < 0.15


def test_normative_deviation_validation():
    df = _norm_data()
    with pytest.raises(ValueError):
        NormativeDeviation(covariates=["age"], reference="is_ref", min_reference=500).fit(df)
    with pytest.raises(ValueError):
        NormativeDeviation(covariates=["nope"]).fit(df)
    plain = NormativeDeviation(covariates=["age"]).fit(df)  # no reference column: residualize on everyone
    assert abs(plain.transform(df)["vol"].mean()) < 1e-6


def test_plsda_classifier_learns_and_exposes_scores():
    rng = np.random.RandomState(1)
    X = rng.randn(300, 60)
    y = (X[:, :5].sum(axis=1) + 0.5 * rng.randn(300) > 0).astype(int)
    clf = PLSDAClassifier(n_components=3).fit(X[:200], y[:200])
    proba = clf.predict_proba(X[200:])
    assert proba.shape == (100, 2) and np.allclose(proba.sum(axis=1), 1)
    assert roc_auc_score(y[200:], proba[:, 1]) > 0.85
    assert clf.transform(X[:3]).shape == (3, 3)
    assert clf.feature_importances_.shape == (60,) and clf.feature_importances_[:5].mean() > clf.feature_importances_[5:].mean()
    assert set(clf.predict(X[200:])) <= {0, 1}


def test_bootstrap_ci_brackets_estimate():
    rng = np.random.RandomState(0)
    y = rng.randint(0, 2, 400)
    s = y + rng.randn(400)
    auc, lo, hi = bootstrap_ci(roc_auc_score, y, s, n_boot=200)
    assert lo <= auc <= hi and 0.6 < auc < 0.9 and (hi - lo) < 0.15
