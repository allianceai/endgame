from __future__ import annotations

"""Block-wise preprocessing for multi-modal tabular data.

Features that come from different acquisitions (structural MRI volumes, diffusion metrics, PET/SPECT
measures, neuromelanin MRI, FLAIR lesion loads) carry different batch effects: each modality has its own
scanner, protocol and QC. Harmonising every column by the *structural* scanner, as a single ComBat call does,
is wrong for the other modalities. `BlockwiseHarmonizer` runs one leakage-safe ComBat per block with the
block's own batch column and passes through rows where that block is absent; `MissingBlockIndicator` adds an
availability flag per block (and mean-imputes the block) so early-fusion models can learn from the missingness
pattern instead of treating an imputed modality as observed.
"""

from typing import Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator
from endgame.preprocessing.harmonization import ComBatHarmonizer


class BlockwiseHarmonizer(EndgameEstimator, TransformerMixin):
    """One ComBat per feature block, each with its own batch column.

    Parameters
    ----------
    blocks : dict[str, dict]
        Block name -> {"features": [columns], "batch": batch column name}. Optional per-block keys
        ``covariates`` (list) and ``min_rows`` (default 10) override the shared settings.
    covariates : list of str, optional
        Continuous biological covariates preserved in every block (e.g. ["age"]).
    categorical : list of str, optional
        Categorical covariates preserved in every block (e.g. ["sex"]).
    unknown_batch : {"raise", "passthrough"}, default "passthrough"
        What to do with test rows whose batch was not seen in ``fit``.
    min_batch_n : int, default 3
        Batch levels with fewer training rows are left un-adjusted (ComBat needs at least two per level).
    combat_kwargs : dict, optional
        Extra keyword arguments for every `ComBatHarmonizer`.

    Rows whose batch value is missing, or whose block features are entirely missing, are passed through
    unchanged for that block. Batch columns are kept in the output (drop them yourself if a model must not
    see them); feature columns keep their names and order.

    Examples
    --------
    >>> h = BlockwiseHarmonizer({"t1": {"features": t1_cols, "batch": "scanner_batch"},
    ...                          "dwi": {"features": dwi_cols, "batch": "dwi_batch"}}, covariates=["age", "sex"])
    >>> Xtr = h.fit_transform(train); Xte = h.transform(test)
    """

    def __init__(self, blocks: Dict[str, dict], covariates: Sequence[str] | None = None, categorical: Sequence[str] | None = None,
                 unknown_batch: str = "passthrough", min_batch_n: int = 3, combat_kwargs: dict | None = None):
        self.blocks = blocks
        self.covariates = covariates
        self.categorical = categorical
        self.unknown_batch = unknown_batch
        self.min_batch_n = min_batch_n
        self.combat_kwargs = combat_kwargs

    def fit(self, X: pd.DataFrame, y=None):
        self.harmonizers_ = {}
        for name, spec in self.blocks.items():
            feats = [f for f in spec["features"] if f in X.columns]
            batch = spec["batch"]
            covs = list(spec.get("covariates", self.covariates or []))
            cats = list(spec.get("categorical", self.categorical or []))
            if not feats or batch not in X.columns:
                continue
            rows = X[batch].notna() & X[feats].notna().any(axis=1)
            counts = X.loc[rows, batch].value_counts()
            rows &= X[batch].isin(counts[counts >= self.min_batch_n].index)   # levels too small to estimate pass through
            if rows.sum() < spec.get("min_rows", 10) or X.loc[rows, batch].nunique() < 2:
                continue
            h = ComBatHarmonizer(batch=batch, covariates=covs or None, categorical=cats or None, features=feats, unknown_batch=self.unknown_batch,
                                 drop_batch=True, **(self.combat_kwargs or {}))
            h.fit(X.loc[rows, feats + [batch] + covs + cats])
            self.harmonizers_[name] = (h, feats, batch, covs + cats)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for name, (h, feats, batch, covs) in self.harmonizers_.items():
            rows = X[batch].notna() & X[feats].notna().any(axis=1)   # unseen batch levels are handled by ComBat itself
            if not rows.any():
                continue
            harmonized = h.transform(X.loc[rows, feats + [batch] + covs])
            out.loc[rows, feats] = harmonized[feats].to_numpy()
        return out


class MissingBlockIndicator(EndgameEstimator, TransformerMixin):
    """Add ``<block>_missing`` (1 when every feature of the block is NaN) and mean-impute the block's features.

    Parameters
    ----------
    blocks : dict[str, list[str]]
        Block name -> feature columns.
    impute : bool, default True
        Fill the block's NaNs with training means (so downstream models need no imputer for these columns).

    Examples
    --------
    >>> mbi = MissingBlockIndicator({"dwi": dwi_cols, "nm": nm_cols}).fit(train)
    >>> Xtr = mbi.transform(train)      # adds dwi_missing, nm_missing
    """

    def __init__(self, blocks: Dict[str, Sequence[str]], impute: bool = True):
        self.blocks = blocks
        self.impute = impute

    def fit(self, X: pd.DataFrame, y=None):
        self.means_ = {name: X[[f for f in feats if f in X.columns]].mean() for name, feats in self.blocks.items()}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X.copy()
        for name, feats in self.blocks.items():
            feats = [f for f in feats if f in X.columns]
            if not feats:
                continue
            out[f"{name}_missing"] = X[feats].isna().all(axis=1).astype(float)
            if self.impute:
                out[feats] = out[feats].fillna(self.means_[name])
        return out

    def get_feature_names_out(self, input_features=None):
        return list(input_features or []) + [f"{name}_missing" for name in self.blocks]
