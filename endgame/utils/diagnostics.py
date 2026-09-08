"""Pipeline sanity checks."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def batch_leakage_check(X, batch, cv: int = 5, model=None, min_n: int = 10, random_state: int = 0) -> pd.DataFrame:
    """How well do the features predict the batch (site / scanner / protocol)?

    A one-vs-rest cross-validated AUROC per batch level, plus the macro average. Run it on raw features and on
    harmonised features: residual batch predictability after harmonisation is the leakage a downstream model can
    exploit, and a batch that is perfectly predictable from the features is confounded with whatever it correlates
    with (cohort, site). Levels with fewer than ``min_n`` rows are pooled into "other".

    Parameters
    ----------
    X : DataFrame or ndarray
        Features (NaNs are mean-imputed).
    batch : array-like
        Batch label per row.
    model : estimator, optional
        Default: standardised logistic regression (balanced).

    Returns
    -------
    DataFrame with columns batch, n, auroc; last row "MACRO".
    """
    Xa = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
    Xa = Xa.fillna(Xa.mean()).fillna(0.0).to_numpy(dtype=float)
    b = pd.Series(np.asarray(batch)).astype(str)
    counts = b.value_counts()
    b = b.where(b.map(counts) >= min_n, "other")
    rows = []
    for level in sorted(b.unique()):
        yb = (b == level).astype(int).to_numpy()
        if yb.sum() < min_n or (1 - yb).sum() < min_n:
            continue
        est = model if model is not None else make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        skf = StratifiedKFold(n_splits=min(cv, int(yb.sum())), shuffle=True, random_state=random_state)
        p = cross_val_predict(est, Xa, yb, cv=skf, method="predict_proba")[:, 1]
        rows.append({"batch": level, "n": int(yb.sum()), "auroc": float(roc_auc_score(yb, p))})
    out = pd.DataFrame(rows)
    if len(out):
        out = pd.concat([out, pd.DataFrame([{"batch": "MACRO", "n": int(len(b)), "auroc": float(out["auroc"].mean())}])], ignore_index=True)
    return out
