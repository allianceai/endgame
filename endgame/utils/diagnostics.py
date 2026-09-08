"""Out-of-fold batch predictability diagnostics, not certificates of no leakage."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from endgame.validation._study import aligned_vector, study_folds


def batch_leakage_check(X, batch, cv=5, model=None, min_n=10, random_state=0, *,
                        groups=None, preprocessor=None, features=None, models=None,
                        times=None, label_end_times=None):
    """Cross-validated one-vs-rest batch AUROC on common study folds.

    Imputation always fits inside each probe's training fold. An optional cloned
    preprocessor (e.g. ComBat/BlockwiseHarmonizer) also fits only on that fold's
    original input table, without batch targets supplied as y. ``features`` selects
    numeric predictors after preprocessing and should exclude scanner metadata.
    ``groups`` enforces patient separation. Explicit temporal splitters may leave
    initial history uncovered; coverage and fold indices are in result.attrs.

    ``model`` selects one probe (default standardized balanced logistic).
    ``models`` instead maps names to predeclared probes, e.g. linear and forest;
    results then include a probe column. Low linear AUROC does not exclude
    nonlinear scanner information or demonstrate preserved biological signal.
    The same splits are used for all probes and labels. An unseen site cannot
    be learned as a binary class without any positive training examples.

    Rare batch labels (< min_n rows) are pooled into an explicitly named level.
    Report conditional effects when preserved biology differs across sites.
    """
    if isinstance(min_n, bool) or not isinstance(min_n, (int, np.integer)) or min_n < 2:
        raise ValueError("min_n must be an integer >= 2")
    frame = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
    if not frame.columns.is_unique:
        raise ValueError("X must have unique feature names")
    b = pd.Series(aligned_vector(batch, len(frame), "batch")).astype(str)
    counts = b.value_counts()
    rare = b.map(counts) < min_n
    if rare.any():
        pooled = "__pooled_rare_batches__"
        while pooled in set(b):
            pooled += "_"
        b = b.where(~rare, pooled)
    levels = sorted(b.unique())
    if len(levels) < 2:
        raise ValueError("at least two batch levels are needed after pooling")
    if model is not None and models is not None:
        raise ValueError("choose model or models, not both")
    default = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=.5))
    probes = models if models is not None else {"linear" if model is None else "custom": default if model is None else model}
    if not probes:
        raise ValueError("models must be nonempty")
    # Cap only the automatic stratified row splitter. Explicit/grouped splits
    # are validated without silently changing the user's design.
    if groups is None and isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        cv = min(cv, int(b.value_counts().min()))
    folds, covered = study_folds(frame, b.to_numpy(), cv, groups, times, label_end_times, random_state)
    scores = {(probe, level): np.full(len(frame), np.nan) for probe in probes for level in levels}
    ybs = {level: (b.to_numpy() == level).astype(int) for level in levels}
    for train, valid in folds:
        tr, va = frame.iloc[train].copy(), frame.iloc[valid].copy()
        if preprocessor is not None:
            processor = clone(preprocessor).fit(tr)
            tr, va = processor.transform(tr), processor.transform(va)
        if features is not None:
            if not isinstance(tr, pd.DataFrame) or not isinstance(va, pd.DataFrame):
                raise ValueError("named features require a DataFrame preprocessor output")
            tr, va = tr.loc[:, list(features)], va.loc[:, list(features)]
        tr, va = np.asarray(tr, dtype=float), np.asarray(va, dtype=float)
        if (tr.ndim != 2 or va.ndim != 2 or not tr.shape[1] or tr.shape[1] != va.shape[1]
                or np.isinf(tr).any() or np.isinf(va).any()):
            raise ValueError("probe features must have a stable numeric schema without infinity")
        for level, yb in ybs.items():
            if len(np.unique(yb[train])) < 2:
                raise ValueError(f"batch '{level}' lacks both classes in a training fold")
            for name, estimator in probes.items():
                pipe = make_pipeline(SimpleImputer(strategy="mean", keep_empty_features=True), clone(estimator))
                pipe.fit(tr, yb[train])
                ix = np.flatnonzero(np.asarray(pipe.classes_) == 1)
                if len(ix) != 1:
                    raise ValueError("probe must expose positive class 1")
                p = np.asarray(pipe.predict_proba(va), float)
                if (p.shape != (len(valid), 2) or not np.isfinite(p).all()
                        or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1.)):
                    raise ValueError("probe returned invalid class probabilities")
                scores[name, level][valid] = p[:, ix[0]]
    rows = []
    for name in probes:
        aucs = []
        for level, yb in ybs.items():
            if min(yb[covered].sum(), (1-yb[covered]).sum()) < min_n:
                continue
            auc = float(roc_auc_score(yb[covered], scores[name, level][covered]))
            row = {"batch": level, "n": int(yb[covered].sum()), "auroc": auc}
            if models is not None:
                row["probe"] = name
            rows.append(row)
            aucs.append(auc)
        if not aucs:
            raise ValueError("insufficient held-out observations for batch AUROC")
        row = {"batch": "MACRO", "n": int(covered.sum()), "auroc": float(np.mean(aucs))}
        if models is not None:
            row["probe"] = name
        rows.append(row)
    out = pd.DataFrame(rows)
    out.attrs["covered"] = covered
    out.attrs["folds"] = folds
    return out
