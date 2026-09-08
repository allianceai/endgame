"""Per-modality harmonization with explicit availability and adjustment status."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator
from endgame.preprocessing.harmonization import ComBatHarmonizer


def _validate_blocks(X, blocks, fitted=False):
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique or not len(X):
        raise ValueError("X must be a nonempty DataFrame with unique columns")
    out = X.copy()
    for name, feats in blocks.items():
        present = [f for f in feats if f in X]
        if len(present) != len(feats):
            if not fitted or present:
                raise ValueError(f"Block '{name}' feature schema is incomplete")
            out[feats] = np.nan
        if np.isinf(out[feats].to_numpy(dtype=float)).any():
            raise ValueError(f"Block '{name}' contains infinity")
    return out


def _block_schema(blocks):
    if not blocks:
        raise ValueError("blocks must be nonempty")
    seen = set()
    for feats in blocks.values():
        if not feats or len(set(feats)) != len(feats) or seen.intersection(feats):
            raise ValueError("block features must be nonempty, unique and nonoverlapping")
        seen.update(feats)


class BlockwiseHarmonizer(EndgameEstimator, TransformerMixin):
    """One train-fitted ComBat per modality and its own scanner column.

    ``blocks`` maps names to dictionaries with features, batch, and optional
    covariates/categorical/min_rows (default 10). Missing modalities, missing
    batch values, rare training batches and blocks with insufficient data pass
    through with explicit ``adjustment_report`` statuses. Unknown batches obey
    ``unknown_batch``. Batch columns remain metadata in output: select model
    predictors explicitly.

    ``partial_missing='raise'`` (default) rejects partially observed rows;
    ``'passthrough'`` excludes them from estimation and leaves them unchanged.
    Imputation, if desired, must be explicitly fitted inside training folds;
    do not turn entirely absent modalities into observations for ComBat.
    """

    def __init__(self, blocks, covariates=None, categorical=None,
                 unknown_batch="passthrough", min_batch_n=3, combat_kwargs=None,
                 partial_missing="raise"):
        super().__init__()
        self.blocks = blocks
        self.covariates = covariates
        self.categorical = categorical
        self.unknown_batch = unknown_batch
        self.min_batch_n = min_batch_n
        self.combat_kwargs = combat_kwargs
        self.partial_missing = partial_missing

    def fit(self, X, y=None):
        self._is_fitted = False
        if self.partial_missing not in ("raise", "passthrough") or self.unknown_batch not in ("raise", "passthrough"):
            raise ValueError("invalid missingness or unknown-batch policy")
        if (isinstance(self.min_batch_n, bool) or not isinstance(self.min_batch_n, (int, np.integer))
                or self.min_batch_n < 2):
            raise ValueError("min_batch_n must be an integer >= 2")
        self.blocks_ = {name: list(spec["features"]) for name, spec in self.blocks.items()}
        _block_schema(self.blocks_)
        X = _validate_blocks(X, self.blocks_)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.harmonizers_, self.block_info_ = {}, {}
        for name, spec in self.blocks.items():
            feats, batch = self.blocks_[name], spec["batch"]
            covs = list(spec.get("covariates", self.covariates or []))
            cats = list(spec.get("categorical", self.categorical or []))
            cols = feats + [batch] + covs + cats
            if len(set(cols)) != len(cols) or not set(cols).issubset(X.columns):
                raise ValueError(f"Block '{name}' needs unique features, batch and covariate columns")
            observed = X[feats].notna()
            partial = observed.any(axis=1) & ~observed.all(axis=1)
            if partial.any() and self.partial_missing == "raise":
                raise ValueError(f"Block '{name}' has partially missing features; choose an explicit policy")
            complete = observed.all(axis=1) & X[batch].notna()
            counts = X.loc[complete, batch].astype(str).value_counts()
            allowed = counts[counts >= self.min_batch_n].index.tolist()
            eligible = complete & X[batch].astype(str).isin(allowed)
            min_rows = spec.get("min_rows", 10)
            if isinstance(min_rows, bool) or not isinstance(min_rows, (int, np.integer)) or min_rows < 2:
                raise ValueError("min_rows must be an integer >= 2")
            info = {"features": feats, "batch": batch, "metadata": covs + cats,
                    "training_batches": set(X.loc[X[batch].notna(), batch].astype(str)),
                    "eligible_batches": set(allowed), "eligible_rows": int(eligible.sum()),
                    "reason": None}
            if eligible.sum() < min_rows or len(allowed) < 2:
                info["reason"] = "insufficient_block_data"
                warnings.warn(f"Block '{name}': insufficient data; left unadjusted", UserWarning, stacklevel=2)
            else:
                h = ComBatHarmonizer(batch=batch, covariates=covs or None, categorical=cats or None,
                                     features=feats, unknown_batch=self.unknown_batch, drop_batch=True,
                                     **(self.combat_kwargs or {}))
                h.fit(X.loc[eligible, cols])
                self.harmonizers_[name] = h
            self.block_info_[name] = info
        self._is_fitted = True
        return self

    def _frame(self, X):
        self._check_is_fitted()
        out = _validate_blocks(X, self.blocks_, fitted=True)
        # Absent acquisitions may omit the corresponding batch column entirely.
        for info in self.block_info_.values():
            if info["batch"] not in out:
                out[info["batch"]] = np.nan
        required = set(self.feature_names_in_) - set(out.columns)
        if required:
            raise ValueError(f"Required covariate/metadata columns missing: {sorted(required)}")
        return out.loc[:, list(self.feature_names_in_)]

    def adjustment_report(self, X):
        """Status for each row and modality, retaining the input index."""
        X = self._frame(X)
        report = pd.DataFrame(index=X.index)
        for name, info in self.block_info_.items():
            obs = X[info["features"]].notna()
            partial = obs.any(axis=1) & ~obs.all(axis=1)
            if partial.any() and self.partial_missing == "raise":
                raise ValueError(f"Block '{name}' has partially missing features")
            batch = X[info["batch"]].astype(str)
            status = pd.Series("adjusted", index=X.index)
            if info["reason"]:
                status[:] = info["reason"]
            else:
                unknown = "unseen_batch_passthrough" if self.unknown_batch == "passthrough" else "unseen_batch_rejected"
                status[~batch.isin(info["training_batches"])] = unknown
                status[batch.isin(info["training_batches"] - info["eligible_batches"])] = "insufficient_batch"
                eligible = batch.isin(info["eligible_batches"])
                statuses = batch.map(self.harmonizers_[name].batch_status_)
                status[eligible] = statuses[eligible]
            status[X[info["batch"]].isna()] = "missing_batch"
            status[partial] = "partial_missing_passthrough"
            status[~obs.any(axis=1)] = "missing_modality"
            report[name] = status
        return report

    def transform(self, X):
        out = self._frame(X)
        report = self.adjustment_report(out)
        for name, h in self.harmonizers_.items():
            info = self.block_info_[name]
            status = report[name]
            rows = status.isin(["empirical_bayes", "location_scale", "no_eb_fallback", "no_adjustment",
                                "unseen_batch_passthrough", "unseen_batch_rejected"])
            if rows.any():
                cols = info["features"] + [info["batch"]] + info["metadata"]
                harm = h.transform(out.loc[rows, cols])
                out.loc[rows, info["features"]] = harm[info["features"]].to_numpy()
        return out

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features differs from fitted schema")
        return self.feature_names_in_.copy()


class MissingBlockIndicator(EndgameEstimator, TransformerMixin):
    """Flag entirely absent modalities and optionally impute with training means.

    All-missing training features use ``empty_value`` (default 0); their names
    are exposed in ``empty_features_``. Missing entire modality columns at test
    are reconstructed. Partial schema changes fail rather than changing shape.
    """

    def __init__(self, blocks, impute=True, empty_value=0.):
        super().__init__()
        self.blocks = blocks
        self.impute = impute
        self.empty_value = empty_value

    def fit(self, X, y=None):
        self._is_fitted = False
        if not np.isfinite(self.empty_value):
            raise ValueError("empty_value must be finite")
        self.blocks_ = {name: list(feats) for name, feats in self.blocks.items()}
        _block_schema(self.blocks_)
        X = _validate_blocks(X, self.blocks_)
        flags = [f"{name}_missing" for name in self.blocks_]
        if set(flags) & set(X.columns):
            raise ValueError("missingness flag collides with an input column")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.output_columns_ = list(X.columns) + flags
        self.means_, self.empty_features_ = {}, {}
        for name, feats in self.blocks_.items():
            mean = X[feats].mean()
            self.empty_features_[name] = mean.index[mean.isna()].tolist()
            self.means_[name] = mean.fillna(self.empty_value)
        self._is_fitted = True
        return self

    def transform(self, X):
        self._check_is_fitted()
        out = _validate_blocks(X, self.blocks_, fitted=True)
        if not set(self.feature_names_in_).issubset(out.columns):
            raise ValueError("input schema differs from training")
        out = out.loc[:, list(self.feature_names_in_)]
        for name, feats in self.blocks_.items():
            out[f"{name}_missing"] = out[feats].isna().all(axis=1).astype(float)
            if self.impute:
                out[feats] = out[feats].fillna(self.means_[name])
        return out.loc[:, self.output_columns_]

    def get_feature_names_out(self, input_features=None):
        self._check_is_fitted()
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features differs from fitted schema")
        return np.asarray(self.output_columns_, dtype=object)
