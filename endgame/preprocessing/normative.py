from __future__ import annotations

"""Normative deviation scores (W-scores / z-scores against a reference population).

Standard practice in neuroimaging and other biomarker work: fit, on a reference group
(e.g. healthy controls), a per-feature linear model of the feature on covariates such as
age, sex and head size; then express every subject's value as its standardized residual
from that norm. Positive scores mean "larger than expected for this person's covariates".
The reference fit happens in ``fit`` only, so the transformer is leakage-safe inside
cross-validation folds.

References
----------
- Jack et al. (1997) W-scores; Marquand et al. (2016, 2019) normative modelling;
  Rutherford et al. (2022) "The normative modeling framework for computational psychiatry".

Examples
--------
>>> import pandas as pd
>>> from endgame.preprocessing import NormativeDeviation
>>> df = pd.DataFrame({"age": [60, 70, 65, 75, 62], "sex": [0, 1, 1, 0, 1],
...                    "is_control": [1, 1, 1, 0, 0], "vol": [5.0, 4.5, 4.7, 3.9, 4.8]})
>>> nd = NormativeDeviation(covariates=["age", "sex"], reference="is_control")
>>> z = nd.fit_transform(df)   # 'vol' replaced by its deviation score; covariates kept
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


class NormativeDeviation(EndgameEstimator, TransformerMixin):
    """Replace features by their standardized deviation from a covariate-adjusted reference norm.

    Parameters
    ----------
    covariates : list of str
        Numeric covariate columns of the norm (e.g. ``["age", "sex", "icv"]``). An intercept is
        always included.
    reference : str, optional
        Boolean/0-1 column marking reference rows (e.g. controls). If None, all training rows
        are the reference (plain covariate residualization).
    features : list of str, optional
        Columns to score. Default: every numeric column except covariates and reference.
    min_reference : int, default=20
        Minimum number of reference rows required in ``fit``.
    drop_reference : bool, default=True
        Drop the reference column from the output.

    Attributes
    ----------
    beta_ : ndarray (n_covariates + 1, n_features)
        Reference-group regression coefficients.
    resid_sd_ : ndarray (n_features,)
        Reference-group residual standard deviation per feature (floor at 1e-12).
    """

    def __init__(
        self,
        covariates: list[str],
        reference: str | None = None,
        features: list[str] | None = None,
        min_reference: int = 20,
        drop_reference: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.covariates = covariates
        self.reference = reference
        self.features = features
        self.min_reference = min_reference
        self.drop_reference = drop_reference

    def _check(self, X: Any) -> pd.DataFrame:
        if not (HAS_PANDAS and isinstance(X, pd.DataFrame)):
            raise TypeError("NormativeDeviation expects a pandas DataFrame with covariate columns.")
        need = list(self.covariates) + ([self.reference] if self.reference else [])
        missing = [c for c in need if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in X: {missing}")
        return X

    def _design(self, X: pd.DataFrame) -> np.ndarray:
        cov = X[list(self.covariates)].to_numpy(dtype=float)
        if not np.isfinite(cov).all():
            raise ValueError("Covariates contain NaN/inf.")
        return np.column_stack([np.ones(len(X)), cov])

    def fit(self, X, y=None):
        X = self._check(X)
        excluded = set(self.covariates) | ({self.reference} if self.reference else set())
        self.features_ = list(self.features) if self.features else [
            c for c in X.columns if c not in excluded and pd.api.types.is_numeric_dtype(X[c])
        ]
        if not self.features_:
            raise ValueError("No feature columns to score.")
        ref = X if self.reference is None else X[X[self.reference].astype(bool)]
        if len(ref) < self.min_reference:
            raise ValueError(f"Only {len(ref)} reference rows; need >= {self.min_reference}.")
        D = self._design(ref)
        F = ref[self.features_].to_numpy(dtype=float)
        if not np.isfinite(F).all():
            raise ValueError("Reference features contain NaN/inf; impute first.")
        self.beta_, *_ = np.linalg.lstsq(D, F, rcond=None)
        resid = F - D @ self.beta_
        dof = max(len(ref) - D.shape[1], 1)
        self.resid_sd_ = np.maximum(np.sqrt((resid**2).sum(axis=0) / dof), 1e-12)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        X = self._check(X)
        D = self._design(X)
        F = X[self.features_].to_numpy(dtype=float)
        out = X.copy()
        out[self.features_] = (F - D @ self.beta_) / self.resid_sd_
        if self.drop_reference and self.reference:
            out = out.drop(columns=[self.reference])
        return out

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        return np.asarray(self.features_)
