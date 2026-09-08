import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from endgame.models.block_stacking import BlockStackingClassifier


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    # block A: weakly informative, 20 features; block B: one strong feature among noise; demo: uninformative
    A = rng.normal(size=(n, 20)) + 0.3 * y[:, None]
    B = rng.normal(size=(n, 5))
    B[:, 0] += 1.5 * y
    demo = rng.normal(size=(n, 2))
    X = pd.DataFrame(np.c_[A, B, demo], columns=[f"a{i}" for i in range(20)] + [f"b{i}" for i in range(5)] + ["age", "sex"])
    blocks = {"A": [f"a{i}" for i in range(20)], "B": [f"b{i}" for i in range(5)], "demo": ["age", "sex"]}
    return X, y, blocks


def test_fit_predict_and_block_scores():
    X, y, blocks = _data()
    clf = BlockStackingClassifier(blocks, base_estimator=LogisticRegression(max_iter=1000), passthrough=["demo"], cv=3)
    clf.fit(X[:300], y[:300])
    p = clf.predict_proba(X[300:])
    assert p.shape == (100, 2) and np.allclose(p.sum(axis=1), 1)
    assert roc_auc_score(y[300:], p[:, 1]) > 0.85
    assert set(clf.predict(X[300:])) <= {0, 1}
    bs = clf.block_scores(X[300:])
    assert list(bs.columns) == ["A", "B", "demo"]
    # the strong block should carry more signal than the weak one
    assert roc_auc_score(y[300:], bs["B"]) > roc_auc_score(y[300:], bs["A"])


def test_missing_block_falls_back_to_training_mean():
    X, y, blocks = _data()
    clf = BlockStackingClassifier(blocks, cv=3).fit(X[:300], y[:300])
    Xm = X[300:].copy()
    Xm.loc[:, blocks["B"]] = np.nan          # whole block missing at prediction time
    p = clf.predict_proba(Xm)
    assert np.isfinite(p).all()
    assert np.allclose(clf.block_scores(Xm)["B"], clf.block_means_["B"])


def test_ndarray_input_needs_feature_names():
    X, y, blocks = _data(n=120)
    with pytest.raises(ValueError):
        BlockStackingClassifier(blocks, cv=3).fit(X.to_numpy(), y)
    clf = BlockStackingClassifier(blocks, cv=3, feature_names=list(X.columns)).fit(X.to_numpy(), y)
    assert clf.predict_proba(X.to_numpy()).shape == (120, 2)
