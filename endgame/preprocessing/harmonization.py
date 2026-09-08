from __future__ import annotations

"""Batch-effect harmonization transformers.

- ComBatHarmonizer: leakage-safe ComBat (Johnson et al. 2007; Fortin et al. 2018)
  for multi-site / multi-scanner tabular features such as imaging-derived phenotypes.

ComBat removes additive (location) and multiplicative (scale) batch effects per
feature while preserving the effect of biological covariates (age, sex, ...).
The estimates are learned in ``fit`` and re-applied in ``transform``, so the
transformer can sit inside a cross-validation fold without leaking the test
batch statistics into training (fitting ComBat on the full dataset before
splitting is a documented source of inflated accuracy in neuroimaging ML).

Examples
--------
>>> import pandas as pd
>>> from endgame.preprocessing import ComBatHarmonizer
>>> df = pd.DataFrame({"site": ["a", "a", "b", "b"], "age": [60, 70, 65, 75],
...                    "vol1": [1.0, 1.2, 2.0, 2.3], "vol2": [5.0, 5.5, 4.0, 4.4]})
>>> h = ComBatHarmonizer(batch="site", covariates=["age"])
>>> harmonized = h.fit_transform(df)      # returns a DataFrame, batch column dropped
"""

from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False


def _aprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (2 * s2 + m**2) / s2


def _bprior(delta_hat: np.ndarray) -> float:
    m, s2 = delta_hat.mean(), delta_hat.var(ddof=1)
    return (m * s2 + m**3) / s2


def _postmean(g_hat, g_bar, n, d_star, t2):
    return (t2 * n * g_hat + d_star * g_bar) / (t2 * n + d_star)


def _postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)


def _eb_iterate(s_batch, g_hat, d_hat, g_bar, t2, a, b, conv=1e-4):
    """Parametric empirical-Bayes iteration for one batch (s_batch: samples x features)."""
    n = np.isfinite(s_batch).sum(axis=0)
    g_old, d_old = g_hat.copy(), d_hat.copy()
    change = 1.0
    while change > conv:
        g_new = _postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((s_batch - g_new) ** 2).sum(axis=0)
        d_new = _postvar(sum2, n, a, b)
        change = max(np.abs((g_new - g_old) / g_old).max(), np.abs((d_new - d_old) / d_old).max())
        g_old, d_old = g_new, d_new
    return g_new, d_new


class ComBatHarmonizer(EndgameEstimator, TransformerMixin):
    """Leakage-safe ComBat harmonization of tabular features across batches.

    Parameters
    ----------
    batch : str
        Column of ``X`` holding the batch / site / scanner label.
    covariates : list of str, optional
        Continuous biological covariates to preserve (e.g. ``["age"]``).
    categorical : list of str, optional
        Categorical biological covariates to preserve (e.g. ``["sex"]``).
    features : list of str, optional
        Feature columns to harmonize. Default: every numeric column that is not
        the batch column or a covariate.
    eb : bool, default=True
        Use empirical-Bayes shrinkage of the batch estimates (parametric priors).
    mean_only : bool, default=False
        Adjust only the batch means, not the variances.
    unknown_batch : {'raise', 'passthrough'}, default='raise'
        What to do at ``transform`` time with a batch level not seen in ``fit``.
        'passthrough' leaves those rows un-adjusted.
    drop_batch : bool, default=True
        Drop the batch column from the transformed output.
    constant_tol : float, default=1e-10
        Features whose pooled residual variance is below ``constant_tol`` times their squared mean
        scale are treated as constant and passed through unchanged (see ``constant_mask_``).

    Notes
    -----
    Features must not contain NaN; impute first. Every batch needs at least two
    samples. ``fit_transform`` on the training data reproduces ``neuroCombat``
    output; ``transform`` on new data uses each new sample's own covariates,
    unlike ``neuroCombatFromTraining`` which uses the training-average covariate
    effect.
    """

    def __init__(
        self,
        batch: str,
        covariates: list[str] | None = None,
        categorical: list[str] | None = None,
        features: list[str] | None = None,
        eb: bool = True,
        mean_only: bool = False,
        unknown_batch: str = "raise",
        drop_batch: bool = True,
        constant_tol: float = 1e-10,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.constant_tol = constant_tol
        self.batch = batch
        self.covariates = covariates
        self.categorical = categorical
        self.features = features
        self.eb = eb
        self.mean_only = mean_only
        self.unknown_batch = unknown_batch
        self.drop_batch = drop_batch

    # ------------------------------------------------------------------ helpers
    def _check_df(self, X: Any) -> pd.DataFrame:
        if not (HAS_PANDAS and isinstance(X, pd.DataFrame)):
            raise TypeError("ComBatHarmonizer expects a pandas DataFrame with batch/covariate columns.")
        missing = [c for c in [self.batch, *(self.covariates or []), *(self.categorical or [])] if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        return X

    def _covariate_design(self, X: pd.DataFrame) -> np.ndarray:
        """Covariate part of the design matrix (samples x p), using levels stored at fit."""
        cols = []
        for c in self.categorical or []:
            levels = self.categorical_levels_[c]
            vals = X[c].astype(str).to_numpy()
            unseen = set(vals) - set(levels)
            if unseen:
                raise ValueError(f"Unseen level(s) {sorted(unseen)} in categorical covariate '{c}'.")
            for lv in levels[1:]:  # drop first level, as neuroCombat does
                cols.append((vals == lv).astype(float))
        for c in self.covariates or []:
            cols.append(X[c].to_numpy(dtype=float))
        return np.column_stack(cols) if cols else np.zeros((len(X), 0))

    # ---------------------------------------------------------------------- fit
    def fit(self, X, y=None):
        X = self._check_df(X)
        excluded = {self.batch, *(self.covariates or []), *(self.categorical or [])}
        self.features_ = list(self.features) if self.features else [
            c for c in X.columns if c not in excluded and pd.api.types.is_numeric_dtype(X[c])
        ]
        if not self.features_:
            raise ValueError("No feature columns to harmonize.")
        data = X[self.features_].to_numpy(dtype=float)
        if not np.isfinite(data).all():
            raise ValueError("Features contain NaN/inf; impute before ComBatHarmonizer.")

        if X[self.batch].isna().any():
            raise ValueError(f"Batch column '{self.batch}' contains missing values; fill or drop them first.")
        batch = X[self.batch].astype(str).to_numpy()
        self.batch_levels_, batch_idx, counts = np.unique(batch, return_inverse=True, return_counts=True)
        if counts.min() < 2:
            small = self.batch_levels_[counts < 2].tolist()
            raise ValueError(f"Every batch needs >= 2 samples; too small: {small}")
        self.categorical_levels_ = {
            c: np.unique(X[c].astype(str).to_numpy()).tolist() for c in (self.categorical or [])
        }

        n, n_batch = len(X), len(self.batch_levels_)
        onehot = np.eye(n_batch)[batch_idx]
        design = np.hstack([onehot, self._covariate_design(X)])
        b_hat, *_ = np.linalg.lstsq(design, data, rcond=None)  # (p, n_features)

        self.grand_mean_ = (counts / n) @ b_hat[:n_batch]
        resid = data - design @ b_hat
        var_pooled = (resid**2).mean(axis=0)
        # (near-)constant features cannot be harmonized: dividing by ~0 variance turns them into batch
        # identifiers. They pass through unchanged (constant_mask_) and are excluded from the EB priors.
        scale = np.maximum(np.abs(data).mean(axis=0), 1.0) ** 2
        self.constant_mask_ = var_pooled <= self.constant_tol * scale
        if self.constant_mask_.any():
            self._log(f"{int(self.constant_mask_.sum())} near-constant feature(s) left un-harmonized.", "warn")
            var_pooled = np.where(self.constant_mask_, 1.0, var_pooled)
        self.var_pooled_ = var_pooled
        self.beta_cov_ = b_hat[n_batch:]

        mod_mean = design[:, n_batch:] @ self.beta_cov_
        s = (data - self.grand_mean_ - mod_mean) / np.sqrt(self.var_pooled_)

        gamma_hat = np.vstack([s[batch_idx == i].mean(axis=0) for i in range(n_batch)])
        if self.mean_only:
            delta_hat = np.ones_like(gamma_hat)
        else:
            delta_hat = np.vstack([s[batch_idx == i].var(axis=0, ddof=1) for i in range(n_batch)])
            delta_hat[delta_hat == 0] = 1.0

        if self.constant_mask_.any():  # neutral batch parameters for pass-through features
            gamma_hat[:, self.constant_mask_] = 0.0
            delta_hat[:, self.constant_mask_] = 1.0
        if self.eb:
            gamma_bar, t2 = gamma_hat.mean(axis=1), gamma_hat.var(axis=1, ddof=1)
            gamma_star, delta_star = [], []
            for i in range(n_batch):
                if self.mean_only:
                    gamma_star.append(_postmean(gamma_hat[i], gamma_bar[i], 1, 1, t2[i]))
                    delta_star.append(np.ones(len(self.features_)))
                else:
                    g, d = _eb_iterate(
                        s[batch_idx == i], gamma_hat[i], delta_hat[i], gamma_bar[i], t2[i],
                        _aprior(delta_hat[i]), _bprior(delta_hat[i]),
                    )
                    gamma_star.append(g)
                    delta_star.append(d)
            self.gamma_star_, self.delta_star_ = np.vstack(gamma_star), np.vstack(delta_star)
        else:
            self.gamma_star_, self.delta_star_ = gamma_hat, delta_hat

        self._is_fitted = True
        return self

    # ---------------------------------------------------------------- transform
    def transform(self, X):
        self._check_is_fitted()
        X = self._check_df(X)
        data = X[self.features_].to_numpy(dtype=float)
        if X[self.batch].isna().any():
            raise ValueError(f"Batch column '{self.batch}' contains missing values; fill or drop them first.")
        batch = X[self.batch].astype(str).to_numpy()
        level_index = {lv: i for i, lv in enumerate(self.batch_levels_)}
        idx = np.array([level_index.get(b, -1) for b in batch])
        if (idx < 0).any():
            unseen = sorted(set(batch[idx < 0]))
            if self.unknown_batch != "passthrough":
                raise ValueError(f"Unseen batch level(s) {unseen}; set unknown_batch='passthrough' to keep them un-adjusted.")
            self._log(f"Batch level(s) {unseen} unseen at fit; passed through un-adjusted.", "warn")

        mod_mean = self._covariate_design(X) @ self.beta_cov_
        sd = np.sqrt(self.var_pooled_)
        s = (data - self.grand_mean_ - mod_mean) / sd
        known = idx >= 0
        gamma = np.zeros_like(data)
        delta = np.ones_like(data)
        gamma[known] = self.gamma_star_[idx[known]]
        delta[known] = self.delta_star_[idx[known]]
        harmonized = ((s - gamma) / np.sqrt(delta)) * sd + self.grand_mean_ + mod_mean
        harmonized[:, self.constant_mask_] = data[:, self.constant_mask_]

        out = X.copy()
        out[self.features_] = harmonized
        if self.drop_batch:
            out = out.drop(columns=[self.batch])
        return out

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        return np.asarray(self.features_)
