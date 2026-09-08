"""Strict forward-only validation for panels with overlapping label windows.

Unlike row-gap splitters, a decision date is indivisible. Actual label-end
times are required: an arbitrary number of adjacent stock rows is not a
financial label horizon. No training, fetching or work occurs on import.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import BaseCrossValidator


def _times(values, n, name):
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) != n or arr.dtype.kind not in 'iufM':
        raise ValueError(f'{name} must be an aligned numeric or datetime64 vector')
    invalid = np.isnat(arr) if arr.dtype.kind == 'M' else ~np.isfinite(arr)
    if invalid.any():
        raise ValueError(f'{name} contains missing/non-finite times')
    return arr


class PurgedPanelSplit(BaseCrossValidator):
    """Expanding date-group folds with strict label-end purging.

    ``groups`` passed to split is the decision-time vector, NOT ticker IDs.
    ``label_end_times`` is a mandatory keyword vector in the same units.
    The entire training date is removed if any row's label ends at or after
    the first validation date. This is deliberately conservative.

    The final ``1 - train_fraction`` share of distinct dates is divided into
    n_splits non-overlapping validation blocks. Earlier dates have no OOF
    prediction. No future dates ever enter training. ``max_train_groups``
    optionally limits retained history in distinct eligible dates, not rows.
    """

    def __init__(self, n_splits=3, train_fraction=0.75, max_train_groups=None):
        self.n_splits = n_splits
        self.train_fraction = train_fraction
        self.max_train_groups = max_train_groups

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None, *, label_end_times=None):
        if isinstance(self.n_splits, bool) or int(self.n_splits) != self.n_splits or self.n_splits < 1:
            raise ValueError('n_splits must be a positive integer')
        if not 0 < self.train_fraction < 1:
            raise ValueError('train_fraction must be in (0, 1)')
        if self.max_train_groups is not None and (
            int(self.max_train_groups) != self.max_train_groups or self.max_train_groups < 1
        ):
            raise ValueError('max_train_groups must be a positive integer')
        if groups is None or label_end_times is None:
            raise ValueError('explicit decision times and label_end_times are required')
        starts = _times(groups, len(X), 'groups')
        ends = _times(label_end_times, len(X), 'label_end_times')
        if (starts.dtype.kind == 'M') != (ends.dtype.kind == 'M') or (ends < starts).any():
            raise ValueError('label ends must use the same time units and cannot precede decisions')
        dates = np.unique(starts)
        cut = int(np.floor(len(dates) * self.train_fraction))
        if cut < 1 or len(dates) - cut < self.n_splits:
            raise ValueError('not enough distinct dates for the requested validation blocks')
        group_ends = np.array([ends[starts == date].max() for date in dates])
        for block in np.array_split(dates[cut:], int(self.n_splits)):
            eligible = dates[(dates < block[0]) & (group_ends < block[0])]
            if self.max_train_groups is not None:
                eligible = eligible[-int(self.max_train_groups):]
            train = np.flatnonzero(np.isin(starts, eligible))
            valid = np.flatnonzero(np.isin(starts, block))
            if not len(train) or not len(valid):
                raise ValueError('purging leaves an empty fold; request more history or fewer folds')
            yield train, valid


@dataclass
class PanelOOFResult:
    predictions: np.ndarray
    covered: np.ndarray
    fold_id: np.ndarray
    folds: list[dict]


def purged_panel_oof(estimator, X, y, times, label_end_times, *, cv=None, fit_groups=False):
    """Clone/refit the complete estimator for each strictly chronological fold.

    Returns NaN on the cold-start rows, never fabricated in-sample predictions.
    Use pipelines or estimators that own their preprocessing; do not preprocess
    globally before calling this helper. ``fit_groups=True`` forwards decision
    times to estimators such as GroupedRanker that require query groups.
    Predict must return one finite scalar per validation row. This helper does
    not fit ensemble weights or evaluate meta-weights on their training OOF.
    """
    splitter = PurgedPanelSplit() if cv is None else cv
    if not isinstance(splitter, PurgedPanelSplit):
        raise ValueError('purged_panel_oof requires PurgedPanelSplit, not a shuffled/fallback CV')
    target = np.asarray(y, dtype=float)
    if target.ndim != 1 or len(target) != len(X) or not np.isfinite(target).all():
        raise ValueError('y must be an aligned finite numeric target vector')
    t = _times(times, len(X), 'times')
    ends = _times(label_end_times, len(X), 'label_end_times')
    predictions = np.full(len(X), np.nan)
    fold_id = np.full(len(X), -1, dtype=int)
    folds = []
    take = lambda obj, rows: obj.iloc[rows] if hasattr(obj, 'iloc') else np.asarray(obj)[rows]
    for i, (train, valid) in enumerate(splitter.split(X, groups=t, label_end_times=ends)):
        if (fold_id[valid] >= 0).any():
            raise ValueError('OOF validation folds overlap')
        fitted = clone(estimator)
        kwargs = {'groups': t[train]} if fit_groups else {}
        fitted.fit(take(X, train), target[train], **kwargs)
        pred = np.asarray(fitted.predict(take(X, valid)), dtype=float)
        if pred.shape != (len(valid),) or not np.isfinite(pred).all():
            raise ValueError('estimator must produce one finite scalar per validation row')
        predictions[valid], fold_id[valid] = pred, i
        folds.append({'fold': i, 'train_rows': len(train), 'validation_rows': len(valid),
                      'last_train_time': str(t[train].max()), 'last_train_label_end': str(ends[train].max()),
                      'first_validation_time': str(t[valid].min()), 'last_validation_time': str(t[valid].max())})
    return PanelOOFResult(predictions, fold_id >= 0, fold_id, folds)
