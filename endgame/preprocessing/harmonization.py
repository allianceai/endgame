"""Training-fitted ComBat for cross-sectional, tabular imaging features.

This transformer does not estimate new-scanner effects or longitudinal subject
random effects. Biological covariates must be available at prediction time.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _eb_iterate(s_batch, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4, max_iter=1000):
    """Bounded parametric EB iteration on active features only."""
    n = len(s_batch)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    for _ in range(max_iter):
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_batch - g_new) ** 2).sum(axis=0)
        d_new = (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)
        if not np.isfinite(g_new).all() or not np.isfinite(d_new).all() or (d_new <= 0).any():
            raise ValueError("ComBat empirical-Bayes posterior is nonfinite or nonpositive")
        change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-8)),
                     np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-8)))
        if change <= conv:
            return g_new, d_new
        g_old, d_old = g_new, d_new
    raise ValueError("ComBat empirical-Bayes iteration did not converge within max_iter")


def _design_diagnostics(design):
    """Check identifiability using column scaling to reduce dependence on units."""
    norms = np.linalg.norm(design, axis=0)
    if (norms == 0).any():
        raise ValueError("design is rank deficient: zero covariate column")
    scaled = design / norms
    rank = np.linalg.matrix_rank(scaled)
    if rank != design.shape[1]:
        raise ValueError("design is rank deficient; batch/covariate effects are not identifiable")
    if len(design) <= rank:
        raise ValueError("design has no residual degrees of freedom")
    condition = float(np.linalg.cond(scaled))
    if condition > 1e8:
        warnings.warn("Poorly conditioned design; check covariate and batch overlap", UserWarning, stacklevel=2)
    return int(rank), condition


class ComBatHarmonizer(EndgameEstimator, TransformerMixin):
    """ComBat learned in fit and applied independently to each transform row.

    Parameters
    ----------
    batch : str
        Scanner/site column. Every training batch needs at least two rows.
    covariates, categorical : list of str, optional
        Continuous and categorical biological covariates to preserve.
    features : list of str, optional
        Numeric columns to harmonize; default excludes batch and covariates.
        Explicit feature allowlists are recommended for study tables.
    eb : bool, default=True
        Estimate parametric empirical-Bayes priors across active features.
    mean_only : bool, default=False
        Correct only location. Regular cases match neuroCombat training output.
    unknown_batch : {'raise', 'passthrough'}, default='raise'
        Unseen-scanner passthrough is raw data, not estimated harmonization.
    drop_batch : bool, default=True
        Remove the batch column from output.
    constant_tol : float, default=1e-10
        Relative variance tolerance, scaled by squared mean absolute feature
        magnitude (at least one). Globally constant features pass through and
        never contribute to EB priors.
    eb_fallback : {'raise', 'no_eb'}, default='raise'
        Policy for one active feature, degenerate priors or failed convergence.
        Explicit no_eb fallback uses unshrunk location/scale and emits a warning.
    degenerate_features : {'raise', 'passthrough'}, default='raise'
        Nonconstant features with negligible residual variance may be batch
        identifiers or perfectly explained by covariates. Reject by default;
        explicit passthrough warns and excludes them from EB priors.
    max_iter : int, default=1000
        Maximum EB iterations per batch.
    tol : float, default=1e-4
        Relative convergence tolerance.
    verbose : bool, default=False
        Endgame logging option; scientific warnings do not depend on it.

    Notes
    -----
    Inputs must be finite. No statistics are learned in transform. Inspect
    ``adjustment_report(X)`` and ``batch_status_`` to audit passthrough. This is
    cross-sectional ComBat, not a longitudinal or unseen-scanner estimator.
    """

    def __init__(self, batch, covariates=None, categorical=None, features=None,
                 eb=True, mean_only=False, unknown_batch="raise", drop_batch=True,
                 constant_tol=1e-10, verbose=False, eb_fallback="raise",
                 degenerate_features="raise", max_iter=1000, tol=1e-4):
        super().__init__(verbose=verbose)
        self.batch = batch
        self.covariates = covariates
        self.categorical = categorical
        self.features = features
        self.eb = eb
        self.mean_only = mean_only
        self.unknown_batch = unknown_batch
        self.drop_batch = drop_batch
        self.constant_tol = constant_tol
        self.eb_fallback = eb_fallback
        self.degenerate_features = degenerate_features
        self.max_iter = max_iter
        self.tol = tol

    def _check_df(self, X, fitting=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("ComBatHarmonizer expects a pandas DataFrame")
        if not X.columns.is_unique or not len(X):
            raise ValueError("X must have rows and unique column names")
        need = [self.batch, *(self.covariates or []), *(self.categorical or [])]
        if not fitting:
            need = list(self.feature_names_in_)
        missing = [c for c in need if c not in X]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        if X[[self.batch, *(self.categorical or [])]].isna().any().any():
            raise ValueError("Batch/categorical columns contain missing values")
        return X if fitting else X.loc[:, list(self.feature_names_in_)]

    def _covariate_design(self, X):
        cols = []
        for c in self.categorical or []:
            levels = self.categorical_levels_[c]
            vals = X[c].astype(str).to_numpy()
            if set(vals) - set(levels):
                raise ValueError(f"Unseen level in categorical covariate '{c}'")
            cols.extend((vals == lv).astype(float) for lv in levels[1:])
        cols.extend(X[c].to_numpy(dtype=float) for c in self.covariates or [])
        out = np.column_stack(cols) if cols else np.zeros((len(X), 0))
        if not np.isfinite(out).all():
            raise ValueError("Covariates contain NaN/inf")
        return out

    def fit(self, X, y=None):
        self._is_fitted = False
        if self.unknown_batch not in ("raise", "passthrough") or self.eb_fallback not in ("raise", "no_eb"):
            raise ValueError("invalid unknown_batch or eb_fallback policy")
        if self.degenerate_features not in ("raise", "passthrough"):
            raise ValueError("invalid degenerate_features policy")
        if (not np.isfinite(self.constant_tol) or self.constant_tol < 0
                or not np.isfinite(self.tol) or self.tol <= 0
                or isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer))
                or self.max_iter < 1):
            raise ValueError("constant_tol, tol and max_iter must be valid nonnegative/positive values")
        X = self._check_df(X, fitting=True)
        roles = [self.batch, *(self.covariates or []), *(self.categorical or [])]
        if len(set(roles)) != len(roles):
            raise ValueError("batch and covariate roles must be disjoint and unique")
        self.features_ = list(self.features) if self.features is not None else [
            c for c in X if c not in roles and pd.api.types.is_numeric_dtype(X[c])]
        if not self.features_ or len(set(self.features_)) != len(self.features_) or set(self.features_) & set(roles):
            raise ValueError("features must be nonempty, unique and distinct from batch/covariates")
        if not set(self.features_).issubset(X.columns):
            raise ValueError("feature columns are missing")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_columns_ = [c for c in X if c != self.batch or not self.drop_batch]
        data = X[self.features_].to_numpy(dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("Features contain NaN/inf; impute within training folds before ComBat")
        batch = X[self.batch].astype(str).to_numpy()
        self.batch_levels_, idx, counts = np.unique(batch, return_inverse=True, return_counts=True)
        if counts.min() < 2:
            raise ValueError("Every batch needs >= 2 samples")
        self.categorical_levels_ = {c: sorted(X[c].astype(str).unique()) for c in self.categorical or []}
        cov = self._covariate_design(X)
        nb = len(counts)
        design = np.column_stack([np.eye(nb)[idx], cov])
        self.design_rank_, self.design_condition_ = _design_diagnostics(design)
        self.covariate_ranges_ = {c: X.groupby(self.batch)[c].agg(["min", "max"])
                                  for c in self.covariates or []}
        beta = np.linalg.lstsq(design, data, rcond=None)[0]
        self.grand_mean_ = (counts / len(X)) @ beta[:nb]
        self.beta_cov_ = beta[nb:]
        pooled = ((data - design @ beta) ** 2).mean(axis=0)
        scale = np.maximum(np.abs(data).mean(axis=0), 1.0) ** 2
        self.constant_mask_ = data.var(axis=0) <= self.constant_tol * scale
        self.degenerate_mask_ = (pooled <= self.constant_tol * scale) & ~self.constant_mask_
        if self.degenerate_mask_.any():
            names = np.asarray(self.features_)[self.degenerate_mask_].tolist()
            msg = f"Nonconstant features have negligible residual variance (possible batch identifiers): {names}"
            if self.degenerate_features == "raise":
                raise ValueError(msg)
            warnings.warn(msg + "; passed through unadjusted", UserWarning, stacklevel=2)
        self.inactive_mask_ = self.constant_mask_ | self.degenerate_mask_
        active = ~self.inactive_mask_
        self.var_pooled_ = np.where(self.inactive_mask_, 1., pooled)
        s = (data - self.grand_mean_ - cov @ self.beta_cov_) / np.sqrt(self.var_pooled_)
        self.gamma_star_ = np.zeros((nb, len(self.features_)))
        self.delta_star_ = np.ones_like(self.gamma_star_)
        self.batch_status_ = {}
        for i, level in enumerate(self.batch_levels_):
            if not active.any() or nb == 1:
                self.batch_status_[level] = "no_adjustment"
                continue
            sb = s[idx == i][:, active]
            gh = sb.mean(axis=0)
            dh = np.ones_like(gh) if self.mean_only else sb.var(axis=0, ddof=1)
            if (dh <= 0).any() or not np.isfinite(dh).all():
                raise ValueError("Within-batch variance is zero; use mean_only=True or review features")
            g, d = gh, dh
            status = "location_scale"
            if self.eb:
                try:
                    if active.sum() < 2:
                        raise ValueError("empirical Bayes requires at least two active features")
                    gbar, t2 = gh.mean(), gh.var(ddof=1)
                    if self.mean_only:
                        # Preserve neuroCombat's mean-only training convention.
                        g, d = _postmean(gh, gbar, 1, 1, t2), np.ones_like(dh)
                    else:
                        m, v = dh.mean(), dh.var(ddof=1)
                        if not np.isfinite(v) or v <= np.finfo(float).eps * max(m*m, 1.):
                            raise ValueError("degenerate empirical-Bayes variance prior")
                        a, b = (2*v + m*m)/v, (m*v + m**3)/v
                        g, d = _eb_iterate(sb, gh, dh, gbar, t2, a, b, self.tol, self.max_iter)
                    status = "empirical_bayes"
                except ValueError as exc:
                    if self.eb_fallback == "raise":
                        raise ValueError(f"Batch {level}: {exc}; explicitly use eb=False or eb_fallback='no_eb'") from exc
                    warnings.warn(f"Batch {level}: {exc}; using unshrunk location/scale", UserWarning, stacklevel=2)
                    g, d, status = gh, dh, "no_eb_fallback"
            if not np.isfinite(g).all() or not np.isfinite(d).all() or (d <= 0).any():
                raise ValueError("invalid fitted ComBat parameters")
            self.gamma_star_[i, active], self.delta_star_[i, active] = g, d
            self.batch_status_[level] = status
        self._is_fitted = True
        return self

    def adjustment_report(self, X):
        """Per-row adjustment status without modifying fitted state."""
        self._check_is_fitted()
        X = self._check_df(X)
        batch = X[self.batch].astype(str)
        unknown = "unseen_batch_passthrough" if self.unknown_batch == "passthrough" else "unseen_batch_rejected"
        status = batch.map(self.batch_status_).fillna(unknown)
        return pd.DataFrame({"batch": batch, "status": status,
                             "inactive_features": int(self.inactive_mask_.sum())}, index=X.index)

    def transform(self, X):
        self._check_is_fitted()
        X = self._check_df(X)
        data = X[self.features_].to_numpy(dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("Features contain NaN/inf")
        batch = X[self.batch].astype(str).to_numpy()
        indices = {level: i for i, level in enumerate(self.batch_levels_)}
        idx = np.array([indices.get(level, -1) for level in batch])
        known = idx >= 0
        if not known.all():
            unseen = sorted(set(batch[~known]))
            if self.unknown_batch == "raise":
                raise ValueError(f"Unseen batch level(s) {unseen}")
            warnings.warn(f"Unseen batch level(s) {unseen}: raw passthrough, not harmonized", UserWarning, stacklevel=2)
        mean = self.grand_mean_ + self._covariate_design(X) @ self.beta_cov_
        sd = np.sqrt(self.var_pooled_)
        out_data = data.copy()
        out_data[known] = (((data[known] - mean[known]) / sd - self.gamma_star_[idx[known]])
                           / np.sqrt(self.delta_star_[idx[known]]) * sd + mean[known])
        out_data[:, self.inactive_mask_] = data[:, self.inactive_mask_]
        if not np.isfinite(out_data).all():
            raise ValueError("nonfinite harmonized values")
        out = X.copy()
        out[self.features_] = out_data
        return out.loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features differs from fitted schema")
        return np.asarray(self.output_columns_, dtype=object)
