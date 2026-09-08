"""Linear reference-population W scores, fitted only on training controls."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator
from endgame.preprocessing.harmonization import _design_diagnostics


class NormativeDeviation(EndgameEstimator, TransformerMixin):
    """Standardized linear residuals relative to a reference group.

    ``reference`` is a nonmissing boolean/0-1 training column, never a required
    prediction input when ``drop_reference=True``. Covariates are numeric and
    include an implicit intercept. Features default to numeric non-covariate,
    non-reference columns; explicit allowlists are recommended.

    These are homoskedastic W scores, not posterior probabilities or predictive
    intervals. Validate their tails on independent controls. Saturated or rank
    deficient designs and undefined residual scales fail explicitly.

    Parameters
    ----------
    covariates : list of str
        Numeric predictors of the reference norm.
    reference : str, optional
        Training reference membership. None uses all training rows.
    features : list of str, optional
        Features to replace by deviation scores.
    min_reference : int, default=20
        Minimum number of training reference observations.
    drop_reference : bool, default=True
        Exclude the reference marker from output and do not require it at test.
    verbose : bool, default=False
        Endgame logging option.
    extrapolation : {'warn', 'raise', 'ignore'}, default='warn'
        Policy when prediction covariates leave training-reference ranges.
    """

    def __init__(self, covariates, reference=None, features=None, min_reference=20,
                 drop_reference=True, verbose=False, extrapolation="warn"):
        super().__init__(verbose=verbose)
        self.covariates = covariates
        self.reference = reference
        self.features = features
        self.min_reference = min_reference
        self.drop_reference = drop_reference
        self.extrapolation = extrapolation

    def _check(self, X, fitting=False):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NormativeDeviation expects a pandas DataFrame")
        if not len(X) or not X.columns.is_unique:
            raise ValueError("X must have rows and unique column names")
        required = list(self.covariates) + ([self.reference] if self.reference else [])
        if not fitting:
            required = self.output_columns_
        if not set(required).issubset(X.columns):
            raise ValueError(f"Columns not found in X: {sorted(set(required) - set(X.columns))}")
        return X

    def _design(self, X):
        cov = X[list(self.covariates)].to_numpy(dtype=float)
        if not np.isfinite(cov).all():
            raise ValueError("Covariates contain NaN/inf")
        return np.column_stack([np.ones(len(X)), cov])

    def fit(self, X, y=None):
        self._is_fitted = False
        if self.extrapolation not in ("warn", "raise", "ignore"):
            raise ValueError("invalid extrapolation policy")
        if (isinstance(self.min_reference, bool) or not isinstance(self.min_reference, (int, np.integer))
                or self.min_reference < 2):
            raise ValueError("min_reference must be an integer >= 2")
        X = self._check(X, fitting=True)
        covs = list(self.covariates)
        if len(set(covs)) != len(covs) or self.reference in covs:
            raise ValueError("covariates must be unique and exclude reference membership")
        excluded = set(covs) | ({self.reference} if self.reference else set())
        self.features_ = list(self.features) if self.features is not None else [
            c for c in X if c not in excluded and pd.api.types.is_numeric_dtype(X[c])]
        if (not self.features_ or len(set(self.features_)) != len(self.features_)
                or set(self.features_) & excluded or not set(self.features_).issubset(X.columns)):
            raise ValueError("features must be present, nonempty, unique and distinct from metadata")
        if self.reference is None:
            ref = X
        else:
            marker = X[self.reference]
            if marker.isna().any() or not marker.isin([0, 1, False, True]).all():
                raise ValueError("reference membership must be nonmissing boolean or numeric 0/1")
            ref = X.loc[marker == 1]
        self.n_reference_ = len(ref)
        if len(ref) < self.min_reference:
            raise ValueError(f"Only {len(ref)} reference rows; need >= {self.min_reference}")
        D = self._design(ref)
        self.design_rank_, self.design_condition_ = _design_diagnostics(D)
        F = ref[self.features_].to_numpy(dtype=float)
        if not np.isfinite(F).all():
            raise ValueError("Reference features contain NaN/inf")
        self.beta_ = np.linalg.lstsq(D, F, rcond=None)[0]
        self.residual_dof_ = len(ref) - self.design_rank_
        resid = F - D @ self.beta_
        self.resid_sd_ = np.sqrt((resid**2).sum(axis=0) / self.residual_dof_)
        tolerance = 100 * np.finfo(float).eps * np.maximum(np.abs(F).mean(axis=0), 1.)
        if not np.isfinite(self.resid_sd_).all() or (self.resid_sd_ <= tolerance).any():
            raise ValueError("Reference residual scale is undefined or numerically zero")
        self.covariate_min_ = ref[covs].min().to_numpy(dtype=float)
        self.covariate_max_ = ref[covs].max().to_numpy(dtype=float)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_columns_ = [c for c in X if c != self.reference or not self.drop_reference]
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = self._check(X)
        D = self._design(X)
        outside = ((D[:, 1:] < self.covariate_min_) | (D[:, 1:] > self.covariate_max_)).any(axis=1)
        if outside.any() and self.extrapolation != "ignore":
            msg = f"{outside.sum()} rows extrapolate beyond training-reference covariate ranges"
            if self.extrapolation == "raise":
                raise ValueError(msg)
            warnings.warn(msg, UserWarning, stacklevel=2)
        F = X[self.features_].to_numpy(dtype=float)
        if not np.isfinite(F).all():
            raise ValueError("Features contain NaN/inf")
        out = X.loc[:, self.output_columns_].copy()
        out[self.features_] = (F - D @ self.beta_) / self.resid_sd_
        return out

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features differs from fitted schema")
        return np.asarray(self.output_columns_, dtype=object)
