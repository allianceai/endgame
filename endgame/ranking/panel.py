"""Ranking models and diagnostics for repeated cross-sections.

Groups identify contemporaneous choice sets (e.g. stocks on one decision
date), not entities over time. No implicit split, tuning, GPU, or downloading.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _groups(groups, n):
    g = np.asarray(groups)
    if g.ndim != 1 or len(g) != n or n == 0:
        raise ValueError('groups must be a nonempty aligned vector')
    # Strings, ordinal numbers and timestamps are accepted as group labels.
    if any(x is None or str(x) in ('nan', 'NaT', '<NA>', '') for x in g):
        raise ValueError('groups contain missing labels')
    try:
        labels, codes = np.unique(g, return_inverse=True)
    except TypeError as exc:
        raise ValueError('group labels must have one comparable type') from exc
    return labels, codes


def group_percentiles(values, groups):
    """Average-rank percentiles in [0,1], preserving row order.

    Endpoints are 0 and 1 for distinct minimum/maximum. Constant and singleton
    groups score 0.5; ties never depend on row ordering or ticker spelling.
    """
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('ranking requires finite one-dimensional values')
    labels, codes = _groups(groups, len(v))
    out = np.empty(len(v))
    for code in range(len(labels)):
        idx = np.flatnonzero(codes == code)
        out[idx] = (rankdata(v[idx], method='average') - 1) / (len(idx) - 1) if len(idx) > 1 else .5
    return out


class GroupedRanker(RegressorMixin, BaseEstimator):
    """Fixed-settings rank regression or query-aware LambdaMART.

    ``backend='ridge'`` predicts within-group return percentiles using shrinkage.
    ``backend='lambdamart'`` maps those ranks to integer relevance grades and
    supplies contiguous query sizes to LightGBM. Linear label gains avoid the
    default exponential gain's extreme emphasis on the highest grade.

    ``focus='bottom'`` makes bad outcomes highly relevant for LambdaMART, then
    negates predictions so HIGH SCORES ALWAYS MEAN BETTER OUTCOMES. Thus a
    low-score exclusion rule can target the predicted losers. Ridge supports
    ``focus='top'`` only: negating a linear target is not a distinct model.

    Each group receives equal total sample weight. LambdaRank has additional
    query normalization: this does not guarantee equal group gradient norms.
    Missing features use fitted training medians (all-missing columns use 0).
    Ridge standardization is fitted only on training rows. No automatic early
    stopping or parameter search. n_jobs=1 and CPU are the defaults.
    """

    def __init__(self, backend='ridge', focus='top', alpha=100., n_bins=5,
                 n_estimators=300, learning_rate=.03, num_leaves=7,
                 min_child_samples=50, reg_lambda=10., truncation_level=30,
                 colsample_bytree=.8, random_state=42, n_jobs=1):
        self.backend = backend
        self.focus = focus
        self.alpha = alpha
        self.n_bins = n_bins
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.min_child_samples = min_child_samples
        self.reg_lambda = reg_lambda
        self.truncation_level = truncation_level
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _matrix(self, X, fitting=False):
        if hasattr(X, 'columns'):
            names = list(X.columns)
            if len(set(names)) != len(names):
                raise ValueError('duplicate feature names')
            if fitting:
                self.feature_names_in_ = np.asarray(names, dtype=object)
            elif hasattr(self, 'feature_names_in_'):
                if set(names) != set(self.feature_names_in_):
                    raise ValueError('prediction feature schema differs from training')
                X = X.loc[:, list(self.feature_names_in_)]
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2 or not arr.shape[0] or not arr.shape[1] or np.isinf(arr).any():
            raise ValueError('X must be a nonempty numeric matrix without infinity (NaN is allowed)')
        if not fitting and arr.shape[1] != self.n_features_in_:
            raise ValueError('prediction feature count differs from training')
        return arr

    def fit(self, X, y, groups=None):
        if self.backend not in ('ridge', 'lambdamart') or self.focus not in ('top', 'bottom'):
            raise ValueError('unknown backend or ranking focus')
        if self.backend == 'ridge' and self.focus != 'top':
            raise ValueError('ridge focus must be top; sign reversal is not a separate linear model')
        for name in ('n_bins', 'n_estimators', 'num_leaves', 'min_child_samples', 'truncation_level', 'n_jobs'):
            value = getattr(self, name)
            if isinstance(value, bool) or int(value) != value or value < (2 if name in ('n_bins', 'num_leaves') else 1):
                raise ValueError(f'{name} must be a positive integer (bins/leaves at least 2)')
        if not np.isfinite([self.alpha, self.learning_rate, self.reg_lambda, self.colsample_bytree]).all() or self.alpha <= 0 or self.learning_rate <= 0 or self.reg_lambda < 0 or not 0 < self.colsample_bytree <= 1:
            raise ValueError('invalid regularization, learning rate or feature-sampling fraction')
        # Remove a previous fitted schema when refitting with a numpy matrix.
        if hasattr(self, 'feature_names_in_'):
            del self.feature_names_in_
        data = self._matrix(X, fitting=True)
        target = np.asarray(y, dtype=float)
        if target.shape != (len(data),) or not np.isfinite(target).all():
            raise ValueError('y must be finite and aligned; resolve missing outcomes before fitting')
        labels, codes = _groups(groups, len(data))
        counts = np.bincount(codes)
        if (counts < 2).any():
            raise ValueError('every training group needs at least two rows')
        ranks = group_percentiles(target, groups)
        if not any(np.ptp(target[codes == i]) > 0 for i in range(len(labels))):
            raise ValueError('no within-group outcome variation to learn')
        self.n_features_in_ = data.shape[1]
        self.imputer_ = SimpleImputer(strategy='median', keep_empty_features=True)
        clean = self.imputer_.fit_transform(data)
        self.group_labels_, self.group_sizes_ = labels, counts
        self.training_weight_ = len(data) / (len(labels) * counts[codes])
        self.direction_ = -1. if self.focus == 'bottom' else 1.
        if self.backend == 'ridge':
            self.scaler_ = StandardScaler().fit(clean, sample_weight=self.training_weight_)
            self.model_ = Ridge(alpha=self.alpha, solver='lsqr')
            self.model_.fit(self.scaler_.transform(clean), ranks, sample_weight=self.training_weight_)
        else:
            try:
                from lightgbm import LGBMRanker
            except ImportError as exc:
                raise ImportError('LambdaMART requires lightgbm; install endgame-ml[tabular] explicitly') from exc
            relevant = 1 - ranks if self.focus == 'bottom' else ranks
            grades = np.minimum((relevant * self.n_bins).astype(int), self.n_bins - 1)
            self.training_relevance_ = grades
            order = np.argsort(codes, kind='stable')
            self.model_ = LGBMRanker(objective='lambdarank', n_estimators=self.n_estimators,
                learning_rate=self.learning_rate, num_leaves=self.num_leaves,
                min_child_samples=self.min_child_samples, reg_lambda=self.reg_lambda,
                colsample_bytree=self.colsample_bytree, random_state=self.random_state,
                n_jobs=self.n_jobs, device_type='cpu', deterministic=True, force_col_wise=True,
                label_gain=list(range(self.n_bins)), lambdarank_truncation_level=self.truncation_level,
                importance_type='gain', verbosity=-1)
            self.model_.fit(clean[order], grades[order], group=counts.tolist(), sample_weight=self.training_weight_[order])
        return self

    def predict(self, X):
        check_is_fitted(self, 'model_')
        clean = self.imputer_.transform(self._matrix(X))
        if self.backend == 'ridge':
            clean = self.scaler_.transform(clean)
            predictions = self.model_.predict(clean)
        else:
            # Our own schema check already ran. Native prediction avoids the
            # sklearn wrapper's synthetic Column_N feature-name warnings and
            # keeps prediction threading explicit, just like training.
            predictions = self.model_.booster_.predict(clean, num_threads=self.n_jobs)
        return self.direction_ * np.asarray(predictions, dtype=float)

    def predict_rank(self, X, groups):
        return group_percentiles(self.predict(X), groups)

    def score(self, X, y, groups=None):
        """Equal-group mean Spearman IC, not accuracy or a portfolio Sharpe."""
        rows = group_rank_diagnostics(y, self.predict(X), groups)
        valid = [r['rank_ic'] for r in rows if r['rank_ic'] is not None]
        if not valid:
            raise ValueError('no groups have a defined rank correlation')
        return float(np.mean(valid))

    @property
    def feature_importances_(self):
        check_is_fitted(self, 'model_')
        return np.abs(self.model_.coef_) if self.backend == 'ridge' else self.model_.feature_importances_


def group_rank_average(predictions, groups, weights=None):
    """Outcome-blind fixed rank averaging; no fitted weights or winner search.

    Columns must be predeclared constituents, rows the identical scored keys.
    Missing component scores fail rather than silently changing the ensemble.
    """
    p = np.asarray(predictions, dtype=float)
    if p.ndim != 2 or p.shape[1] < 1 or not np.isfinite(p).all():
        raise ValueError('finite aligned prediction columns are required')
    w = np.ones(p.shape[1]) if weights is None else np.asarray(weights, dtype=float)
    if w.shape != (p.shape[1],) or not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('weights must be finite, nonnegative and have positive sum')
    ranked = np.column_stack([group_percentiles(p[:, i], groups) for i in range(p.shape[1])])
    return ranked @ (w / w.sum())


def group_rank_diagnostics(y, predictions, groups, tail_fraction=.3):
    """Per-date IC and equal-weight selected-tail outcomes; diagnostic only.

    Tail cutoffs include all boundary ties rather than breaking ties by row
    order. Publish tail counts: constant predictions produce overlapping tails
    and zero spread, not an apparently profitable ticker-order tie break.
    No annualization, cost assumption, market neutralization or t-test is implied.
    """
    actual, pred = np.asarray(y, dtype=float), np.asarray(predictions, dtype=float)
    if actual.ndim != 1 or actual.shape != pred.shape or not np.isfinite(actual).all() or not np.isfinite(pred).all():
        raise ValueError('diagnostics require aligned finite outcomes and predictions')
    if not 0 < tail_fraction <= .5:
        raise ValueError('tail_fraction must be in (0, .5]')
    labels, codes = _groups(groups, len(actual))
    rows = []
    for i, label in enumerate(labels):
        a, p = actual[codes == i], pred[codes == i]
        k = max(1, int(np.ceil(len(a) * tail_fraction)))
        lower, upper = np.sort(p)[k - 1], np.sort(p)[-k]
        bottom, top = a[p <= lower], a[p >= upper]
        ic = float(np.corrcoef(rankdata(a), rankdata(p))[0, 1]) if np.ptp(a) > 0 and np.ptp(p) > 0 else None
        rows.append({'group': str(label), 'n': len(a), 'rank_ic': ic, 'mean_outcome': float(a.mean()),
                     'bottom_n': len(bottom), 'top_n': len(top), 'bottom_mean_outcome': float(bottom.mean()),
                     'top_mean_outcome': float(top.mean()), 'top_minus_bottom': float(top.mean() - bottom.mean())})
    return rows
