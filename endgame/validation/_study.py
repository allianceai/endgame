"""Shared validation of study folds, including partial chronological coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from endgame.validation.panel import PurgedPanelSplit, _times


def aligned_vector(values, n, name):
    """Require one nonmissing value per observation, preserving label types."""
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != n or pd.isna(arr).any():
        raise ValueError(f"{name} must be an aligned nonmissing vector")
    if arr.dtype.kind in "fiu" and not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return arr


def study_folds(X, y, cv, groups=None, times=None, label_end_times=None, random_state=0):
    """Materialize folds and reject overlap, subject leakage and future labels.

    Groups, when supplied, always mean independent subjects. For forecasting
    future visits of known subjects, omit groups and supply explicit temporal
    folds, times and label ends. Initial history may remain uncovered.
    """
    n = len(X)
    g = None if groups is None else aligned_vector(groups, n, "groups")
    if (times is None) != (label_end_times is None):
        raise ValueError("times and label_end_times must be supplied together")
    t = ends = None
    if times is not None:
        t = _times(times, n, "times")
        ends = _times(label_end_times, n, "label_end_times")
        if (t.dtype.kind == "M") != (ends.dtype.kind == "M") or (ends < t).any():
            raise ValueError("label ends must match decision time types and follow decisions")
    if isinstance(cv, (int, np.integer)) and not isinstance(cv, bool):
        if cv < 2:
            raise ValueError("cv must have at least two folds")
        if t is not None:
            raise ValueError("temporal data requires an explicit chronological splitter")
        splitter = (StratifiedKFold(cv, shuffle=True, random_state=random_state) if g is None
                    else StratifiedGroupKFold(cv, shuffle=True, random_state=random_state))
        splits = splitter.split(X, y, g)
    elif isinstance(cv, PurgedPanelSplit):
        if t is None:
            raise ValueError("PurgedPanelSplit requires times and label_end_times")
        splits = cv.split(X, y, groups=t, label_end_times=ends)
    elif hasattr(cv, "split"):
        splits = cv.split(X, y, g)
    else:
        try:
            splits = iter(cv)
        except TypeError as exc:
            raise ValueError("cv must be a fold count, splitter or iterable of splits") from exc
    folds, covered = [], np.zeros(n, dtype=bool)
    for train, valid in splits:
        train, valid = np.asarray(train), np.asarray(valid)
        for idx in (train, valid):
            if (idx.ndim != 1 or not len(idx) or idx.dtype.kind not in "iu"
                    or (idx < 0).any() or (idx >= n).any() or len(np.unique(idx)) != len(idx)):
                raise ValueError("fold indices must be nonempty, unique, in-range integer vectors")
        if np.intersect1d(train, valid).size or covered[valid].any():
            raise ValueError("training/validation or OOF validation folds overlap")
        if g is not None and pd.Index(g[train]).isin(g[valid]).any():
            raise ValueError("patient groups overlap between training and validation")
        if t is not None and ((t[train] >= t[valid].min()).any()
                              or (ends[train] >= t[valid].min()).any()):
            raise ValueError("future decisions or overlapping label windows enter training")
        covered[valid] = True
        folds.append((train.copy(), valid.copy()))
    if not folds:
        raise ValueError("cv produced no folds")
    return folds, covered
