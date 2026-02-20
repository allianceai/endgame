"""Cascade Ensemble: Multi-stage cascading with confidence-based routing.

In a cascade, "easy" samples are classified by earlier (cheaper) stages
and only "hard" samples proceed to later (more powerful) stages. Each
stage adds features from the previous stage's predictions, creating a
deep representation.

Inspired by gcForest (Zhou & Feng, 2019, "Deep Forest") and Viola-Jones
cascades.

Example
-------
>>> from endgame.ensemble import CascadeEnsemble
>>> cascade = CascadeEnsemble(
...     stages=[
...         [LogisticRegression(), DecisionTreeClassifier()],
...         [RandomForestClassifier(n_estimators=50)],
...         [GradientBoostingClassifier(n_estimators=100)],
...     ],
...     confidence_threshold=0.95,
... )
>>> cascade.fit(X_train, y_train)
>>> cascade.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class CascadeEnsemble(BaseEstimator, ClassifierMixin):
    """Multi-stage cascade classifier with early-exit.

    Parameters
    ----------
    stages : list of list of estimator
        Each stage is a list of base classifiers. Predictions from
        stage k are concatenated as features for stage k+1.
    confidence_threshold : float, default=0.95
        If max predicted probability exceeds this, the sample exits
        the cascade early (only at prediction time).
    cv : int, default=3
        CV folds for generating OOF features during training.
    use_proba : bool, default=True
        Use predicted probabilities as cascade features (vs. labels).
    passthrough : bool, default=True
        Include original features at every stage.
    max_stages : int or None, default=None
        Maximum number of stages. If None, use all provided stages.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    stages_ : list of list of estimator
        Fitted estimators per stage.
    classes_ : ndarray
    n_stages_ : int
        Number of fitted stages.
    stage_scores_ : list of float
        Per-stage validation accuracy.
    """

    def __init__(
        self,
        stages: list[list[BaseEstimator]],
        confidence_threshold: float = 0.95,
        cv: int = 3,
        use_proba: bool = True,
        passthrough: bool = True,
        max_stages: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.stages = stages
        self.confidence_threshold = confidence_threshold
        self.cv = cv
        self.use_proba = use_proba
        self.passthrough = passthrough
        self.max_stages = max_stages
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit the cascade stage by stage.

        At each stage, generate OOF predictions, concatenate them
        as features for the next stage, then refit on all data.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        cv_splitter = StratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        n_stages = self.max_stages or len(self.stages)
        n_stages = min(n_stages, len(self.stages))

        self.stages_ = []
        self.stage_scores_ = []
        augmented_X = X.copy()

        for stage_idx in range(n_stages):
            stage_estimators = self.stages[stage_idx]
            fitted_stage = []
            oof_features = []

            if self.verbose:
                print(f"[Cascade] Stage {stage_idx + 1}/{n_stages} "
                      f"({len(stage_estimators)} estimators, "
                      f"features: {augmented_X.shape[1]})")

            for est in stage_estimators:
                # Generate OOF predictions for this estimator
                method = "predict_proba" if self.use_proba and hasattr(est, "predict_proba") else "predict"
                try:
                    oof = cross_val_predict(est, augmented_X, y, cv=cv_splitter, method=method)
                except Exception:
                    oof = cross_val_predict(est, augmented_X, y, cv=cv_splitter, method="predict")

                if oof.ndim == 1:
                    oof = oof.reshape(-1, 1)

                oof_features.append(oof)

                # Refit on full augmented data
                fitted = clone(est)
                if sample_weight is not None:
                    try:
                        fitted.fit(augmented_X, y, sample_weight=sample_weight)
                    except TypeError:
                        fitted.fit(augmented_X, y)
                else:
                    fitted.fit(augmented_X, y)
                fitted_stage.append(fitted)

            self.stages_.append(fitted_stage)

            # Compute stage accuracy
            stage_oof = np.hstack(oof_features)
            if stage_oof.shape[1] >= self.n_classes_:
                stage_pred = self.classes_[np.argmax(stage_oof[:, :self.n_classes_], axis=1)]
            else:
                stage_pred = self.classes_[(stage_oof[:, 0] >= 0.5).astype(int)]
            stage_acc = float(np.mean(stage_pred == y))
            self.stage_scores_.append(stage_acc)

            if self.verbose:
                print(f"  Stage accuracy: {stage_acc:.4f}")

            # Build features for next stage
            new_features = np.hstack(oof_features)
            if self.passthrough:
                augmented_X = np.hstack([X, new_features])
            else:
                augmented_X = np.hstack([augmented_X, new_features])

        self.n_stages_ = len(self.stages_)
        return self

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        """Predict with early exit based on confidence.

        Samples whose max probability exceeds ``confidence_threshold``
        at any stage are assigned their prediction from that stage.
        Remaining samples proceed to the next stage.
        """
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        final_proba = np.zeros((n, self.n_classes_))
        decided = np.zeros(n, dtype=bool)
        augmented_X = X.copy()

        for stage_idx, fitted_stage in enumerate(self.stages_):
            # Only process undecided samples
            active = ~decided
            if not active.any():
                break

            # Get predictions from this stage
            stage_preds = []
            for fitted_est in fitted_stage:
                if self.use_proba and hasattr(fitted_est, "predict_proba"):
                    pred = fitted_est.predict_proba(augmented_X[active])
                else:
                    pred = fitted_est.predict(augmented_X[active])
                    if pred.ndim == 1:
                        pred = pred.reshape(-1, 1)
                stage_preds.append(pred)

            # Average probabilities from this stage's estimators
            probas = []
            for pred in stage_preds:
                if pred.shape[1] >= self.n_classes_:
                    probas.append(pred[:, :self.n_classes_])
                elif pred.shape[1] == 1:
                    p = pred[:, 0]
                    probas.append(np.column_stack([1 - p, p]))
                else:
                    probas.append(pred)

            stage_proba = np.mean(probas, axis=0)

            # Check confidence
            max_conf = stage_proba.max(axis=1)
            confident = max_conf >= self.confidence_threshold

            # Also use final stage regardless of confidence
            is_last = stage_idx == self.n_stages_ - 1

            # Assign decided samples
            active_indices = np.where(active)[0]
            if is_last:
                final_proba[active_indices] = stage_proba
                decided[active_indices] = True
            else:
                decided_now = active_indices[confident]
                final_proba[decided_now] = stage_proba[confident]
                decided[decided_now] = True

            # Build augmented features for next stage (all active samples)
            new_features = np.hstack(stage_preds)
            if self.passthrough:
                full_new = np.zeros((n, new_features.shape[1]))
                full_new[active] = new_features
                augmented_X = np.hstack([X, full_new])
            else:
                full_new = np.zeros((n, new_features.shape[1]))
                full_new[active] = new_features
                augmented_X = np.hstack([augmented_X, full_new])

        # Normalize
        row_sums = final_proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return final_proba / row_sums
