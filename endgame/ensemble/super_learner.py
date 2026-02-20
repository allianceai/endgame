"""Super Learner: Cross-validated optimal ensemble combination.

The Super Learner (van der Laan et al., 2007) is an oracle-optimal
ensemble that uses cross-validation to find the best convex
combination of base learners. It minimizes cross-validated risk under
a non-negative least-squares (NNLS) constraint, guaranteeing the
ensemble is asymptotically at least as good as the best single model.

This is the gold-standard ensemble method in biostatistics and is
widely used in Kaggle competitions.

Example
-------
>>> from endgame.ensemble import SuperLearner
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> import xgboost as xgb
>>>
>>> sl = SuperLearner(
...     base_estimators=[
...         ("rf", RandomForestClassifier(n_estimators=100)),
...         ("lr", LogisticRegression()),
...         ("xgb", xgb.XGBClassifier()),
...     ],
...     meta_learner="nnls",  # non-negative least squares
...     cv=5,
... )
>>> sl.fit(X_train, y_train)
>>> sl.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict


class SuperLearner(BaseEstimator):
    """Cross-validated Super Learner ensemble.

    Parameters
    ----------
    base_estimators : list of (str, estimator) tuples
        Named base learners to combine.
    meta_learner : {'nnls', 'ridge', 'best'} or estimator, default='nnls'
        How to combine OOF predictions:
        - ``'nnls'``: Non-negative least squares (convex combination).
        - ``'ridge'``: Ridge regression on OOF predictions.
        - ``'best'``: Use the single best base learner (no blending).
        - An sklearn estimator for custom meta-learning.
    cv : int or CV splitter, default=5
        Cross-validation strategy for OOF predictions.
    use_proba : bool, default=True
        Use ``predict_proba`` for classifiers (if available).
    include_original_features : bool, default=False
        Pass original features to the meta-learner alongside OOF predictions.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    coef_ : ndarray
        Meta-learner weights (one per base estimator).
    base_estimators_ : list of estimator
        Fitted base estimators (on full training data).
    oof_predictions_ : ndarray
        Out-of-fold predictions used for meta-learning.
    cv_scores_ : dict of {name: float}
        Per-estimator cross-validated score.
    is_classifier_ : bool

    References
    ----------
    van der Laan, M.J., Polley, E.C. & Hubbard, A.E. (2007).
    Super Learner. *Statistical Applications in Genetics and
    Molecular Biology*, 6(1).
    """

    def __init__(
        self,
        base_estimators: list[tuple[str, BaseEstimator]],
        meta_learner: str | BaseEstimator = "nnls",
        cv: int | Any = 5,
        use_proba: bool = True,
        include_original_features: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.cv = cv
        self.use_proba = use_proba
        self.include_original_features = include_original_features
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg):
        if self.verbose:
            print(f"[SuperLearner] {msg}")

    def fit(self, X, y, sample_weight=None):
        """Fit the Super Learner.

        1. Generate OOF predictions for each base estimator.
        2. Solve for optimal combination weights.
        3. Refit all base estimators on the full training set.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        n_samples = X.shape[0]

        # Auto-detect task
        self.is_classifier_ = len(np.unique(y)) <= 30 and np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 10
        if self.is_classifier_:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        # Build CV splitter
        if isinstance(self.cv, int):
            if self.is_classifier_:
                cv = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
            else:
                cv = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        else:
            cv = self.cv

        # Step 1: Generate OOF predictions
        self._log("Generating out-of-fold predictions...")
        oof_list = []
        for name, est in self.base_estimators:
            self._log(f"  CV for {name}...")
            method = self._get_predict_method(est)
            try:
                oof = cross_val_predict(est, X, y, cv=cv, method=method)
            except Exception:
                oof = cross_val_predict(est, X, y, cv=cv, method="predict")

            if oof.ndim == 2 and oof.shape[1] == 2 and self.is_classifier_:
                oof = oof[:, 1]  # binary: use P(positive)
            if oof.ndim == 1:
                oof = oof.reshape(-1, 1)
            oof_list.append(oof)

        self.oof_predictions_ = np.hstack(oof_list)

        # Step 2: Solve for meta-learner weights
        self._log("Solving for optimal weights...")
        Z = self.oof_predictions_
        if self.include_original_features:
            Z = np.hstack([Z, X])

        if self.is_classifier_ and self.n_classes_ == 2:
            y_target = (y == self.classes_[1]).astype(float)
        else:
            y_target = y.astype(float)

        self._fit_meta(Z, y_target)

        # Step 3: Refit base estimators on full data
        self._log("Fitting base estimators on full data...")
        self.base_estimators_ = []
        for name, est in self.base_estimators:
            self._log(f"  Fitting {name}...")
            fitted = clone(est)
            if sample_weight is not None:
                try:
                    fitted.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    fitted.fit(X, y)
            else:
                fitted.fit(X, y)
            self.base_estimators_.append(fitted)

        # CV scores for each base learner
        self._compute_cv_scores(y_target)

        return self

    def _get_predict_method(self, est):
        if self.is_classifier_ and self.use_proba and hasattr(est, "predict_proba"):
            return "predict_proba"
        return "predict"

    def _fit_meta(self, Z, y):
        """Fit the meta-learner."""
        if isinstance(self.meta_learner, str):
            if self.meta_learner == "nnls":
                from scipy.optimize import nnls
                # Solve: min ||Z @ w - y||^2, w >= 0
                coef, _ = nnls(Z, y)
                total = coef.sum()
                self.coef_ = coef / total if total > 0 else np.ones(Z.shape[1]) / Z.shape[1]
                self._meta_fitted = None

            elif self.meta_learner == "ridge":
                from sklearn.linear_model import Ridge
                meta = Ridge(alpha=1.0)
                meta.fit(Z, y)
                self.coef_ = meta.coef_
                self._meta_fitted = meta

            elif self.meta_learner == "best":
                # Select the single best base learner
                n_base = len(self.base_estimators)
                scores = []
                for i in range(n_base):
                    col = Z[:, i] if Z.shape[1] > n_base else Z[:, i:i+1].ravel()
                    mse = float(np.mean((col - y) ** 2))
                    scores.append(mse)
                best = int(np.argmin(scores))
                self.coef_ = np.zeros(Z.shape[1])
                self.coef_[best] = 1.0
                self._meta_fitted = None

            else:
                raise ValueError(f"Unknown meta_learner: {self.meta_learner}")
        else:
            # Custom estimator
            meta = clone(self.meta_learner)
            meta.fit(Z, y)
            self.coef_ = getattr(meta, "coef_", np.ones(Z.shape[1]) / Z.shape[1])
            self._meta_fitted = meta

    def _compute_cv_scores(self, y_target):
        self.cv_scores_ = {}
        oof = self.oof_predictions_
        for i, (name, _) in enumerate(self.base_estimators):
            col = oof[:, i] if oof.shape[1] > i else oof[:, 0]
            mse = float(np.mean((col - y_target) ** 2))
            self.cv_scores_[name] = round(1 - mse / (np.var(y_target) + 1e-15), 4)

    def _get_base_predictions(self, X):
        """Get predictions from fitted base estimators."""
        X = np.asarray(X)
        pred_list = []
        for est in self.base_estimators_:
            method = self._get_predict_method(est)
            if method == "predict_proba" and hasattr(est, "predict_proba"):
                pred = est.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2 and self.is_classifier_:
                    pred = pred[:, 1]
            else:
                pred = est.predict(X)
            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)
            pred_list.append(pred)
        return np.hstack(pred_list)

    def predict(self, X):
        X = np.asarray(X)
        Z = self._get_base_predictions(X)
        if self.include_original_features:
            Z = np.hstack([Z, X])

        if self._meta_fitted is not None:
            raw = self._meta_fitted.predict(Z)
        else:
            raw = Z @ self.coef_

        if self.is_classifier_:
            if self.n_classes_ == 2:
                return self.classes_[(raw >= 0.5).astype(int)]
            return self.classes_[np.argmax(Z[:, :self.n_classes_], axis=1)]
        return raw

    def predict_proba(self, X):
        if not self.is_classifier_:
            raise ValueError("predict_proba only for classification tasks.")
        X = np.asarray(X)
        Z = self._get_base_predictions(X)
        if self.include_original_features:
            Z = np.hstack([Z, X])

        if self._meta_fitted is not None:
            raw = self._meta_fitted.predict(Z)
        else:
            raw = Z @ self.coef_

        if self.n_classes_ == 2:
            p = np.clip(raw, 0, 1)
            return np.column_stack([1 - p, p])
        # Multiclass: try averaging probabilities
        return Z[:, :self.n_classes_]

    @property
    def named_estimators(self):
        return {name: est for (name, _), est in zip(self.base_estimators, self.base_estimators_)}
