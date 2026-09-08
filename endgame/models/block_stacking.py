"""Late fusion with training-only state and explicit study folds."""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from endgame.validation._study import aligned_vector, study_folds


class BlockStackingClassifier(ClassifierMixin, BaseEstimator):
    """Binary stacking of named modalities with independently fitted inner folds.

    Parameters
    ----------
    blocks : dict[str, list[str]]
        Predictor columns per modality. Rows with any nonfinite block feature
        fall back to that fold's training class prior. A wholly absent modality
        at prediction is supported; partially missing column schemas fail.
    base_estimator : estimator or dict, optional
        Cloned model/pipeline per block. Receives a DataFrame, so preprocessing
        can use named columns. Defaults to balanced logistic regression.
    meta_estimator : estimator, optional
        Model trained on covered OOF rows only; default balanced logistic.
    cv : int, splitter or iterable of (train, valid) indices, default=5
        Integers use stratified folds, grouped by patient when groups are given
        to fit. Explicit chronological folds may leave initial history uncovered.
    passthrough : sequence of str, default=()
        Blocks also supplied raw to the meta-model. Their imputer is fitted on
        meta-training rows only and uses zero for entirely unobserved columns.
    feature_names : sequence of str, optional
        Column names for ndarray input.
    random_state : int, default=0
        Seed for default folds.
    block_metadata : dict[str, list[str]], optional
        Extra input columns for each base pipeline (e.g. scanner, covariates,
        reference membership). These do not determine modality availability or
        enter the meta-model directly. Required during fit; available metadata
        are passed during predict so train-only reference markers may be absent.

    Notes
    -----
    Supply groups to fit to enforce patient independence. For future visits of
    known patients, use explicit temporal folds with times and label_end_times;
    omit groups only when that overlap is intentional. Base pipelines must own
    all learned preprocessing. Evaluate the entire stack on independent data.
    Balanced class weights do not establish population-calibrated risk.
    """

    def __init__(self, blocks, base_estimator=None, meta_estimator=None, cv=5,
                 passthrough=(), feature_names=None, random_state=0, block_metadata=None):
        self.blocks = blocks
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.feature_names = feature_names
        self.random_state = random_state
        self.block_metadata = block_metadata

    def _frame(self, X):
        if not isinstance(X, pd.DataFrame):
            if self.feature_names is None:
                raise ValueError("X must be a DataFrame or feature_names must be given")
            X = pd.DataFrame(np.asarray(X), columns=list(self.feature_names))
        if not X.columns.is_unique or not len(X):
            raise ValueError("X must have rows and unique column names")
        return X

    def _block_frame(self, X, name, fitting=False):
        feats = self.blocks_[name]
        present = [c for c in feats if c in X]
        if len(present) != len(feats):
            if fitting or present:
                raise ValueError(f"Block '{name}' feature schema differs from training")
            block = pd.DataFrame(np.nan, index=X.index, columns=feats)
        else:
            block = X[feats].copy()
        # Availability concerns measurements, not train-only metadata.
        ok = np.isfinite(block.to_numpy(dtype=float)).all(axis=1)
        for c in self.metadata_[name]:
            if c in X:
                block[c] = X[c]
            elif fitting:
                raise ValueError(f"Block '{name}' metadata column '{c}' is missing")
        return block, ok

    def _base(self, name):
        est = self.base_estimator
        if isinstance(est, dict):
            est = est[name]
        return clone(est) if est is not None else LogisticRegression(max_iter=2000, class_weight="balanced")

    @staticmethod
    def _positive(model, X):
        classes = np.asarray(model.classes_)
        ix = np.flatnonzero(classes == 1)
        p = np.asarray(model.predict_proba(X), dtype=float)
        if (len(ix) != 1 or p.shape != (len(X), len(classes)) or not np.isfinite(p).all()
                or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(axis=1), 1.)):
            raise ValueError("estimator must produce valid probabilities with encoded positive class 1")
        return p[:, ix[0]]

    def _raw_passthrough(self, X):
        cols = []
        for name in self.passthrough:
            block, _ = self._block_frame(X, name)
            cols.append(block[self.blocks_[name]].to_numpy(dtype=float))
        raw = np.column_stack(cols) if cols else np.empty((len(X), 0))
        if np.isinf(raw).any():
            raise ValueError("passthrough contains infinity")
        return raw

    def fit(self, X, y, groups=None, *, times=None, label_end_times=None):
        if hasattr(self, "meta_"):
            del self.meta_
        X = self._frame(X)
        y = aligned_vector(y, len(X), "y")
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BlockStackingClassifier is binary")
        yb = (y == self.classes_[1]).astype(int)
        if not self.blocks:
            raise ValueError("blocks must be nonempty")
        self.blocks_ = {name: list(cols) for name, cols in self.blocks.items()}
        if any(not cols or len(set(cols)) != len(cols) for cols in self.blocks_.values()):
            raise ValueError("each block needs unique, nonempty feature columns")
        if not set(self.passthrough).issubset(self.blocks_) or len(set(self.passthrough)) != len(self.passthrough):
            raise ValueError("passthrough must contain unique block names")
        metadata = self.block_metadata or {}
        if not set(metadata).issubset(self.blocks_):
            raise ValueError("unknown block_metadata name")
        self.metadata_ = {n: list(metadata.get(n, [])) for n in self.blocks_}
        if any(len(set(cols)) != len(cols) or set(cols) & set(self.blocks_[n]) for n, cols in self.metadata_.items()):
            raise ValueError("block metadata must be unique and distinct from block features")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.folds_, self.oof_coverage_ = study_folds(X, yb, self.cv, groups, times, label_end_times, self.random_state)
        covered = self.oof_coverage_
        if len(np.unique(yb[covered])) < 2:
            raise ValueError("covered OOF rows must contain both classes")
        self.oof_predictions_ = np.full((len(X), len(self.blocks_)), np.nan)
        self.block_models_, self.block_means_, self.block_status_ = {}, {}, {}
        for j, name in enumerate(self.blocks_):
            block, ok = self._block_frame(X, name, fitting=True)
            for train, valid in self.folds_:
                if len(np.unique(yb[train])) < 2:
                    raise ValueError("every training fold must contain both outcome classes")
                eligible = train[ok[train]]
                self.oof_predictions_[valid, j] = float(yb[train].mean())
                test = valid[ok[valid]]
                if len(np.unique(yb[eligible])) == 2 and len(test):
                    model = self._base(name).fit(block.iloc[eligible], yb[eligible])
                    self.oof_predictions_[test, j] = self._positive(model, block.iloc[test])
            self.block_means_[name] = float(yb.mean())
            fitted = len(np.unique(yb[ok])) == 2
            self.block_models_[name] = self._base(name).fit(block.loc[ok], yb[ok]) if fitted else None
            self.block_status_[name] = {"available_rows": int(ok.sum()), "fitted": fitted,
                                        "fallback": None if fitted else "insufficient_available_classes"}
            if not fitted:
                warnings.warn(f"Block '{name}' lacks both available classes; using training prior", UserWarning, stacklevel=2)
        raw = self._raw_passthrough(X)
        self.passthrough_imputer_ = None
        meta_X = self.oof_predictions_[covered]
        if raw.shape[1]:
            self.passthrough_imputer_ = SimpleImputer(strategy="mean", keep_empty_features=True).fit(raw[covered])
            meta_X = np.column_stack([meta_X, self.passthrough_imputer_.transform(raw[covered])])
        meta = clone(self.meta_estimator) if self.meta_estimator is not None else LogisticRegression(max_iter=2000, class_weight="balanced")
        self.meta_ = meta.fit(meta_X, yb[covered])
        self.meta_training_rows_ = np.flatnonzero(covered)
        return self

    def block_scores(self, X):
        """Per-block scores aligned to the original patient/visit index."""
        check_is_fitted(self, "meta_")
        X = self._frame(X)
        scores = {}
        for name, model in self.block_models_.items():
            block, ok = self._block_frame(X, name)
            score = np.full(len(X), self.block_means_[name])
            if model is not None and ok.any():
                score[ok] = self._positive(model, block.loc[ok])
            scores[name] = score
        return pd.DataFrame(scores, index=X.index)

    def predict_proba(self, X):
        check_is_fitted(self, "meta_")
        X = self._frame(X)
        meta_X = self.block_scores(X).to_numpy()
        if self.passthrough_imputer_ is not None:
            meta_X = np.column_stack([meta_X, self.passthrough_imputer_.transform(self._raw_passthrough(X))])
        p = self._positive(self.meta_, meta_X)
        return np.column_stack([1-p, p])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= .5).astype(int)]
