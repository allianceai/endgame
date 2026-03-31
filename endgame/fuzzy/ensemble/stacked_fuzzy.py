"""Stacked Fuzzy System -- meta-learning with a fuzzy TSK combiner.

Level-0 estimators produce out-of-fold predictions which become
inputs to a level-1 Takagi-Sugeno-Kang (TSK) fuzzy system.  The TSK
meta-learner partitions the base-learner output space with Gaussian
membership functions and fits linear consequents, enabling smooth,
interpretable combination of heterogeneous models.

Example
-------
>>> from endgame.fuzzy.ensemble import StackedFuzzySystem
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>> stk = StackedFuzzySystem(
...     base_estimators=[
...         RandomForestClassifier(n_estimators=50),
...         LogisticRegression(),
...     ],
... )
>>> stk.fit(X_train, y_train)
>>> stk.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
    is_classifier,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.membership import GaussianMF, create_uniform_mfs


# ---------------------------------------------------------------------------
# Lightweight TSK meta-learner
# ---------------------------------------------------------------------------


class _TSKMetaLearner:
    """Zero-order TSK system used as the stacking meta-learner.

    Each rule has Gaussian antecedents over the meta-features (base
    learner outputs) and a constant consequent per output dimension.
    Fitting is done with closed-form weighted least squares.

    Parameters
    ----------
    n_rules : int
        Number of fuzzy rules.
    n_mfs : int
        Number of membership functions per input.
    """

    def __init__(self, n_rules: int = 5, n_mfs: int = 3):
        self.n_rules = n_rules
        self.n_mfs = n_mfs

    def fit(self, X: np.ndarray, y: np.ndarray) -> _TSKMetaLearner:
        """Fit the TSK meta-learner.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_meta_features)
            Out-of-fold base-learner predictions.
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            True targets (encoded integer labels for classification,
            float for regression).

        Returns
        -------
        self
        """
        self.n_features_ = X.shape[1]
        if y.ndim == 1:
            y = y[:, None]
        self.n_outputs_ = y.shape[1]

        # Build membership functions from data range
        self.mfs_: list[list[GaussianMF]] = []
        for j in range(self.n_features_):
            col = X[:, j]
            x_min, x_max = float(col.min()), float(col.max())
            pad = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
            mfs = create_uniform_mfs(
                n_mfs=self.n_mfs,
                x_min=x_min - pad,
                x_max=x_max + pad,
                mf_type="gaussian",
            )
            self.mfs_.append(mfs)  # type: ignore[arg-type]

        # Generate rule antecedent indices
        # Use a subset of the full grid (up to n_rules)
        self._generate_rules()

        # Compute firing strengths
        W = self._firing_strengths(X)  # (n_samples, n_rules)

        # Fit consequents with weighted least squares
        # For zero-order TSK each rule has a constant consequent
        self.consequents_ = np.zeros(
            (self.n_rules_, self.n_outputs_), dtype=np.float64
        )
        for r in range(self.n_rules_):
            wr = W[:, r]
            total = wr.sum()
            if total < 1e-12:
                continue
            self.consequents_[r] = (wr[:, None] * y).sum(axis=0) / total

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using the TSK meta-learner.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_outputs)
        """
        W = self._firing_strengths(X)  # (n_samples, n_rules)
        total = W.sum(axis=1, keepdims=True)
        total = np.where(total > 0, total, 1.0)
        # Weighted average of consequents
        out = (W @ self.consequents_) / total  # (n_samples, n_outputs)
        if out.shape[1] == 1:
            return out.ravel()
        return out

    # -- internal ----------------------------------------------------------

    def _generate_rules(self) -> None:
        """Generate rule antecedent indices (grid subset)."""
        n_features = self.n_features_
        # Full grid has n_mfs^n_features rules, but cap at n_rules
        total_grid = self.n_mfs ** n_features
        if total_grid <= self.n_rules:
            # Use full grid
            indices = np.array(
                np.meshgrid(*[np.arange(self.n_mfs)] * n_features)
            ).T.reshape(-1, n_features)
        else:
            # Subsample: use evenly spaced indices from the grid
            rng = np.random.default_rng(42)
            all_indices = np.array(
                np.meshgrid(*[np.arange(self.n_mfs)] * n_features)
            ).T.reshape(-1, n_features)
            chosen = rng.choice(len(all_indices), size=self.n_rules, replace=False)
            indices = all_indices[chosen]
        self.antecedent_indices_ = indices.astype(int)
        self.n_rules_ = len(indices)

    def _firing_strengths(self, X: np.ndarray) -> np.ndarray:
        """Compute firing strengths (product t-norm)."""
        n_samples = X.shape[0]
        n_rules = self.n_rules_
        strengths = np.ones((n_samples, n_rules), dtype=np.float64)
        for j in range(self.n_features_):
            for r in range(n_rules):
                mf_idx = self.antecedent_indices_[r, j]
                mu = self.mfs_[j][mf_idx](X[:, j])
                strengths[:, r] *= mu
        return strengths


# ---------------------------------------------------------------------------
# StackedFuzzySystem
# ---------------------------------------------------------------------------


class StackedFuzzySystem(BaseEstimator):
    """Meta-learning with a fuzzy TSK combiner.

    Level-0: any scikit-learn estimators produce out-of-fold predictions.
    Level-1: a zero-order TSK fuzzy system combines the base outputs using
    Gaussian membership functions and constant consequents.

    The task (classification vs regression) is inferred from the first
    base estimator.

    Parameters
    ----------
    base_estimators : list of estimators
        Level-0 learners.
    meta_learner : object or None, default=None
        Level-1 combiner.  Defaults to an internal ``_TSKMetaLearner``.
    cv : int, default=5
        Number of cross-validation folds for generating OOF predictions.
    n_rules : int, default=5
        Number of fuzzy rules in the default TSK meta-learner.
    n_mfs : int, default=3
        Membership functions per input in the TSK meta-learner.
    random_state : int or None, default=None
        Random seed for CV splits.

    Attributes
    ----------
    classes_ : ndarray
        Class labels (classification only).
    n_features_in_ : int
        Number of original features.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import StackedFuzzySystem
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.linear_model import Ridge
    >>> stk = StackedFuzzySystem(
    ...     base_estimators=[RandomForestRegressor(), Ridge()],
    ...     cv=3,
    ... )
    >>> stk.fit(X_train, y_train)
    >>> stk.predict(X_test)
    """

    def __init__(
        self,
        base_estimators: list[Any] | None = None,
        meta_learner: Any = None,
        cv: int = 5,
        n_rules: int = 5,
        n_mfs: int = 3,
        random_state: int | None = None,
    ):
        self.base_estimators = base_estimators
        self.meta_learner = meta_learner
        self.cv = cv
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> StackedFuzzySystem:
        """Fit level-0 estimators and the fuzzy meta-learner.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.base_estimators is None or len(self.base_estimators) == 0:
            raise ValueError("base_estimators must be a non-empty list.")

        self.is_classifier_ = is_classifier(self.base_estimators[0])

        if self.is_classifier_:
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            self.classes_ = le.classes_
            self.n_classes_ = len(self.classes_)
            self.label_encoder_ = le
        else:
            y_enc = y.astype(np.float64)

        n_samples = X.shape[0]
        n_estimators = len(self.base_estimators)

        # Determine meta-feature width
        if self.is_classifier_ and self.n_classes_ > 2:
            meta_width = n_estimators * self.n_classes_
        else:
            meta_width = n_estimators

        oof_meta = np.zeros((n_samples, meta_width), dtype=np.float64)

        # CV for OOF predictions
        if self.is_classifier_:
            kf = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            split_iter = kf.split(X, y_enc)
        else:
            kf = KFold(
                n_splits=self.cv, shuffle=True, random_state=self.random_state
            )
            split_iter = kf.split(X)

        for train_idx, val_idx in split_iter:
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx] if self.is_classifier_ else y_enc[train_idx]

            for i, est_template in enumerate(self.base_estimators):
                est = clone(est_template)
                est.fit(X_tr, y_tr)
                if self.is_classifier_ and self.n_classes_ > 2:
                    proba = est.predict_proba(X_val)
                    start = i * self.n_classes_
                    end = start + self.n_classes_
                    # Handle missing classes
                    if proba.shape[1] == self.n_classes_:
                        oof_meta[val_idx, start:end] = proba
                    else:
                        for j, cls in enumerate(est.classes_):
                            col = np.where(self.classes_ == cls)[0]
                            if len(col) > 0:
                                oof_meta[val_idx, start + col[0]] = proba[:, j]
                elif self.is_classifier_:
                    # Binary: use P(class=1)
                    proba = est.predict_proba(X_val)
                    oof_meta[val_idx, i] = proba[:, 1]
                else:
                    oof_meta[val_idx, i] = est.predict(X_val)

        # Fit all base estimators on full data for test-time use
        self.fitted_estimators_: list[Any] = []
        for est_template in self.base_estimators:
            est = clone(est_template)
            y_full = y if self.is_classifier_ else y_enc
            est.fit(X, y_full)
            self.fitted_estimators_.append(est)

        # Fit meta-learner
        if self.meta_learner is not None:
            self.meta_learner_ = clone(self.meta_learner)
            if self.is_classifier_:
                self.meta_learner_.fit(oof_meta, y_enc)
            else:
                self.meta_learner_.fit(oof_meta, y_enc)
        else:
            # Default TSK meta-learner
            self.meta_learner_ = _TSKMetaLearner(
                n_rules=self.n_rules, n_mfs=self.n_mfs
            )
            if self.is_classifier_:
                # One-hot encode targets for TSK
                targets = np.eye(self.n_classes_)[y_enc]
                self.meta_learner_.fit(oof_meta, targets)
            else:
                self.meta_learner_.fit(oof_meta, y_enc)

        return self

    def _get_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from base estimator predictions."""
        n_samples = X.shape[0]
        n_estimators = len(self.fitted_estimators_)

        if self.is_classifier_ and self.n_classes_ > 2:
            meta_width = n_estimators * self.n_classes_
        else:
            meta_width = n_estimators

        meta = np.zeros((n_samples, meta_width), dtype=np.float64)
        for i, est in enumerate(self.fitted_estimators_):
            if self.is_classifier_ and self.n_classes_ > 2:
                proba = est.predict_proba(X)
                start = i * self.n_classes_
                end = start + self.n_classes_
                if proba.shape[1] == self.n_classes_:
                    meta[:, start:end] = proba
                else:
                    for j, cls in enumerate(est.classes_):
                        col = np.where(self.classes_ == cls)[0]
                        if len(col) > 0:
                            meta[:, start + col[0]] = proba[:, j]
            elif self.is_classifier_:
                proba = est.predict_proba(X)
                meta[:, i] = proba[:, 1]
            else:
                meta[:, i] = est.predict(X)
        return meta

    def predict(self, X: Any) -> np.ndarray:
        """Predict using the stacked fuzzy system.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["fitted_estimators_", "meta_learner_"])
        X = check_array(X)
        meta = self._get_meta_features(X)
        raw = self.meta_learner_.predict(meta)

        if self.is_classifier_:
            if raw.ndim == 2 and raw.shape[1] > 1:
                indices = np.argmax(raw, axis=1)
            else:
                indices = (raw.ravel() >= 0.5).astype(int)
            return self.classes_[indices]
        return raw

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["fitted_estimators_", "meta_learner_"])
        if not self.is_classifier_:
            raise AttributeError("predict_proba is only available for classifiers.")

        X = check_array(X)
        meta = self._get_meta_features(X)
        raw = self.meta_learner_.predict(meta)

        if raw.ndim == 1 or raw.shape[1] == 1:
            # Binary
            p = np.clip(raw.ravel(), 0, 1)
            return np.column_stack([1 - p, p])
        else:
            # Multi-class: normalise to probabilities
            raw = np.clip(raw, 0, None)
            totals = raw.sum(axis=1, keepdims=True)
            totals = np.where(totals > 0, totals, 1.0)
            return raw / totals
