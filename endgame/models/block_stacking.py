from __future__ import annotations

"""Block-wise stacking (late fusion) for multi-modal tabular data.

When features come in blocks from different sources (structural MRI volumes, diffusion metrics, PET/SPECT
measures, demographics and genetics), concatenating everything ("early fusion") lets a large block swamp a
small informative one and mixes blocks with different noise structure. Late fusion fits one base model per
block, turns each into out-of-fold class probabilities on the training data, and learns a small meta-model
on those probabilities (plus, optionally, one always-present block passed through raw). Missing blocks at
prediction time are handled by the meta-model's imputation of the block score with the training mean, so
subjects lacking one modality still get a prediction.

References
----------
- Wolpert (1992) "Stacked generalization", Neural Networks.
- van der Laan, Polley & Hubbard (2007) "Super Learner", Stat. Appl. Genet. Mol. Biol.
"""

from typing import Callable, Dict, Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.utils.validation import check_is_fitted


class BlockStackingClassifier(ClassifierMixin, BaseEstimator):
    """Late-fusion stacking over named feature blocks.

    Parameters
    ----------
    blocks : dict[str, list[str]]
        Block name -> column names (X must be a DataFrame, or ``feature_names`` must be given).
    base_estimator : estimator or dict[str, estimator]
        Model fitted on each block (cloned per block); a dict gives one per block name.
    meta_estimator : estimator, default=LogisticRegression(class_weight='balanced')
        Model on the stacked block probabilities.
    cv : int, default=5
        Folds for the out-of-fold block probabilities (stratified).
    passthrough : sequence of str, default=()
        Block names whose raw features are also given to the meta-model (e.g. demographics).
    feature_names : sequence of str or None
        Column names when X is an ndarray.
    random_state : int, default=0

    Attributes
    ----------
    block_models_ : dict[str, estimator]
    meta_ : estimator
    block_means_ : dict[str, float]  training mean of each block score (used when a block is entirely missing)
    classes_ : ndarray

    Examples
    --------
    >>> from endgame.models import BlockStackingClassifier
    >>> clf = BlockStackingClassifier(blocks={"t1": t1_cols, "dwi": dwi_cols, "demo": demo_cols},
    ...                               base_estimator=LogisticRegression(max_iter=1000), passthrough=["demo"])
    >>> clf.fit(X_train, y_train).predict_proba(X_test)
    """

    def __init__(self, blocks: Dict[str, Sequence[str]], base_estimator=None, meta_estimator=None, cv: int = 5,
                 passthrough: Sequence[str] = (), feature_names: Sequence[str] | None = None, random_state: int = 0):
        self.blocks = blocks
        self.base_estimator = base_estimator
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.feature_names = feature_names
        self.random_state = random_state

    # ------------------------------------------------------------------ helpers
    def _frame(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        if self.feature_names is None:
            raise ValueError("X must be a DataFrame or feature_names must be given")
        return pd.DataFrame(np.asarray(X), columns=list(self.feature_names))

    def _base(self, name):
        est = self.base_estimator
        if isinstance(est, dict):
            est = est[name]
        if est is None:
            est = LogisticRegression(max_iter=2000, class_weight="balanced")
        return clone(est)

    def _block_matrix(self, X: pd.DataFrame, name: str) -> np.ndarray:
        cols = [c for c in self.blocks[name] if c in X.columns]
        return X[cols].to_numpy(dtype=float)

    def _score_block(self, model, Xb: np.ndarray, mean: float) -> np.ndarray:
        """Positive-class probability per row; rows with any missing value in the block get the training mean."""
        ok = np.isfinite(Xb).all(axis=1) if Xb.shape[1] else np.zeros(len(Xb), bool)
        out = np.full(len(Xb), mean)
        if ok.any():
            out[ok] = model.predict_proba(Xb[ok])[:, 1]
        return out

    def _meta_matrix(self, X: pd.DataFrame, scores: Dict[str, np.ndarray]) -> np.ndarray:
        cols = [scores[n] for n in self.blocks]
        for n in self.passthrough:
            Xp = self._block_matrix(X, n)
            cols.append(np.where(np.isfinite(Xp), Xp, np.nanmean(Xp, axis=0)) if len(Xp) else Xp)
        return np.column_stack(cols)

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X, y):
        X = self._frame(X)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("BlockStackingClassifier is binary")
        yb = (y == self.classes_[1]).astype(int)
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        self.block_models_, self.block_means_, oof = {}, {}, {}
        for name in self.blocks:
            Xb = self._block_matrix(X, name)
            ok = np.isfinite(Xb).all(axis=1) if Xb.shape[1] else np.zeros(len(Xb), bool)
            model = self._base(name)
            score = np.full(len(Xb), np.nan)
            if ok.sum() >= 2 * self.cv and yb[ok].min() != yb[ok].max():
                score[ok] = cross_val_predict(clone(model), Xb[ok], yb[ok], cv=skf, method="predict_proba")[:, 1]
                model.fit(Xb[ok], yb[ok])
            self.block_models_[name] = model if ok.sum() >= 2 * self.cv else None
            mean = float(np.nanmean(score)) if np.isfinite(score).any() else float(yb.mean())
            self.block_means_[name] = mean
            oof[name] = np.where(np.isfinite(score), score, mean)
        meta = clone(self.meta_estimator) if self.meta_estimator is not None else LogisticRegression(max_iter=2000, class_weight="balanced")
        self.meta_ = meta.fit(self._meta_matrix(X, oof), yb)
        return self

    def _scores(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        return {name: (self._score_block(m, self._block_matrix(X, name), self.block_means_[name]) if m is not None
                       else np.full(len(X), self.block_means_[name])) for name, m in self.block_models_.items()}

    def predict_proba(self, X):
        check_is_fitted(self, "meta_")
        X = self._frame(X)
        p1 = self.meta_.predict_proba(self._meta_matrix(X, self._scores(X)))[:, 1]
        return np.column_stack([1 - p1, p1])

    def predict(self, X):
        return self.classes_[(self.predict_proba(X)[:, 1] >= 0.5).astype(int)]

    def block_scores(self, X) -> pd.DataFrame:
        """Per-block positive-class probabilities (the meta-model's inputs), for inspection."""
        check_is_fitted(self, "meta_")
        return pd.DataFrame(self._scores(self._frame(X)))
