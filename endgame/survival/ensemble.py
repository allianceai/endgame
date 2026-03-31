"""Survival ensemble methods.

Provides stacking, hill climbing, and voting ensembles for survival models.
All ensembles combine risk scores from base survival estimators.

Example
-------
>>> from endgame.survival.ensemble import SurvivalVotingEnsemble
>>> # voting = SurvivalVotingEnsemble(estimators=[("cox", cox), ("rsf", rsf)])
>>> # voting.fit(X_train, y_train)
>>> # risk = voting.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from endgame.survival.base import (
    BaseSurvivalEstimator,
    SurvivalMixin,
    _check_survival_y,
    _get_time_event,
)
from endgame.survival.metrics import concordance_index
from endgame.survival.validation import SurvivalStratifiedKFold


# ---------------------------------------------------------------------------
# Stacking Ensemble
# ---------------------------------------------------------------------------


class SurvivalStackingEnsemble(BaseSurvivalEstimator):
    """Stacking ensemble for survival models.

    Level-0 estimators produce out-of-fold risk score predictions, which
    are stacked as features for a level-1 meta-learner. The meta-learner
    learns an optimal combination of base model risk scores.

    Parameters
    ----------
    base_estimators : list
        List of survival estimator instances (must implement fit/predict).
    meta_estimator : estimator or None, default=None
        Meta-learner that combines stacked risk scores. If None, uses
        ``Ridge(alpha=1.0)`` from sklearn.
    cv : int, default=5
        Number of cross-validation folds for generating OOF predictions.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimators_ : list
        Fitted base estimators (refitted on full training data).
    meta_estimator_ : estimator
        Fitted meta-learner.
    cv_scores_ : list of float
        Per-base-model OOF concordance index scores.

    Example
    -------
    >>> # stack = SurvivalStackingEnsemble(
    >>> #     base_estimators=[cox_model, rsf_model, gbsa_model],
    >>> #     cv=5,
    >>> # )
    >>> # stack.fit(X_train, y_train)
    >>> # risk = stack.predict(X_test)
    """

    def __init__(
        self,
        base_estimators: list[Any] | None = None,
        meta_estimator: Any | None = None,
        cv: int = 5,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.base_estimators = base_estimators or []
        self.meta_estimator = meta_estimator
        self.cv = cv

    def fit(self, X: np.ndarray, y: Any) -> "SurvivalStackingEnsemble":
        """Fit the stacking ensemble.

        1. Generate OOF risk score predictions for each base model.
        2. Fit meta-learner on stacked OOF predictions.
        3. Refit all base models on full training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : structured array or compatible
            Survival target.

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y)
        n_samples = len(y)
        n_models = len(self.base_estimators)

        if n_models == 0:
            raise ValueError("base_estimators must contain at least one estimator.")

        # Set up CV
        cv_splitter = SurvivalStratifiedKFold(
            n_splits=self.cv, shuffle=True, random_state=self.random_state
        )

        # Generate OOF predictions
        oof_preds = np.zeros((n_samples, n_models))
        self.cv_scores_ = []

        for model_idx, estimator in enumerate(self.base_estimators):
            self._log(f"Generating OOF predictions for model {model_idx}")
            fold_scores = []

            for train_idx, test_idx in cv_splitter.split(X, y):
                fold_model = clone(estimator)
                fold_model.fit(X[train_idx], y[train_idx])
                preds = fold_model.predict(X[test_idx])
                oof_preds[test_idx, model_idx] = preds

                score = concordance_index(y[test_idx], preds)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores))
            self.cv_scores_.append(mean_score)
            self._log(f"Model {model_idx} OOF C-index: {mean_score:.4f}")

        # Fit meta-learner on OOF predictions
        # Target: use risk ordering from time/event
        # We use negative time as target (higher risk = shorter time)
        # For censored observations, this is approximate but works in practice
        time, event = _get_time_event(y)
        meta_target = -time  # negative time so higher = more risk

        if self.meta_estimator is None:
            from sklearn.linear_model import Ridge

            self.meta_estimator_ = Ridge(alpha=1.0)
        else:
            self.meta_estimator_ = clone(self.meta_estimator)

        self.meta_estimator_.fit(oof_preds, meta_target)
        self._log("Meta-learner fitted on OOF predictions")

        # Refit all base models on full training data
        self.base_estimators_ = []
        for estimator in self.base_estimators:
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.base_estimators_.append(fitted)

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores using the stacking ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Combined risk scores (higher = more risk).
        """
        self._check_is_fitted()
        X = self._to_numpy(X)

        # Get predictions from each base model
        n_models = len(self.base_estimators_)
        stacked = np.column_stack(
            [est.predict(X) for est in self.base_estimators_]
        )

        # Meta-learner combines them
        return self.meta_estimator_.predict(stacked)


# ---------------------------------------------------------------------------
# Hill Climbing Ensemble
# ---------------------------------------------------------------------------


class SurvivalHillClimbingEnsemble(BaseSurvivalEstimator):
    """Greedy forward selection ensemble optimizing concordance index.

    Iteratively adds the model whose inclusion maximizes the concordance
    index of the weighted average of risk scores. Models can be selected
    multiple times (bagging effect).

    Parameters
    ----------
    n_iterations : int, default=50
        Maximum number of greedy selection iterations.
    patience : int, default=10
        Stop after this many iterations without improvement.
    metric : callable or None, default=None
        Metric function ``metric(y_true, risk_scores) -> float``.
        If None, uses concordance index. Higher is better.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    weights_ : dict
        Mapping from model index to its weight (proportional to selection count).
    best_score_ : float
        Best metric score achieved.
    selection_history_ : list of tuple
        List of ``(model_idx, score)`` at each iteration.

    Example
    -------
    >>> # hc = SurvivalHillClimbingEnsemble(n_iterations=50, patience=10)
    >>> # hc.fit(oof_predictions, y_train)
    >>> # combined_risk = hc.predict(X=None, base_predictions=test_predictions)
    """

    def __init__(
        self,
        n_iterations: int = 50,
        patience: int = 10,
        metric: Any | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_iterations = n_iterations
        self.patience = patience
        self.metric = metric

    def fit(
        self,
        oof_predictions: dict[int, np.ndarray] | list[np.ndarray] | np.ndarray,
        y: Any,
    ) -> "SurvivalHillClimbingEnsemble":
        """Fit by greedy forward selection on OOF predictions.

        Parameters
        ----------
        oof_predictions : dict, list, or ndarray
            Out-of-fold risk score predictions from base models.
            - dict: ``{model_idx: predictions_array}``
            - list: ``[predictions_array_0, predictions_array_1, ...]``
            - ndarray of shape (n_samples, n_models)
        y : structured array or compatible
            Survival target.

        Returns
        -------
        self
        """
        y = _check_survival_y(y)

        # Normalize oof_predictions to dict
        if isinstance(oof_predictions, np.ndarray) and oof_predictions.ndim == 2:
            preds = {
                i: oof_predictions[:, i]
                for i in range(oof_predictions.shape[1])
            }
        elif isinstance(oof_predictions, list):
            preds = {i: np.asarray(p) for i, p in enumerate(oof_predictions)}
        elif isinstance(oof_predictions, dict):
            preds = {k: np.asarray(v) for k, v in oof_predictions.items()}
        else:
            raise ValueError(
                "oof_predictions must be a dict, list, or 2D ndarray."
            )

        metric_fn = self.metric if self.metric is not None else concordance_index
        model_indices = sorted(preds.keys())
        n_models = len(model_indices)

        if n_models == 0:
            raise ValueError("oof_predictions must contain at least one model.")

        # Find best single model
        best_idx = None
        best_score = -np.inf
        for idx in model_indices:
            score = metric_fn(y, preds[idx])
            if score > best_score:
                best_score = score
                best_idx = idx

        selected = [best_idx]
        self.selection_history_ = [(best_idx, best_score)]
        self._log(f"Iteration 0: selected model {best_idx}, score={best_score:.4f}")

        no_improve_count = 0

        for iteration in range(1, self.n_iterations):
            # Current ensemble average
            current_avg = np.mean(
                [preds[idx] for idx in selected], axis=0
            )

            best_candidate = None
            best_candidate_score = -np.inf

            for idx in model_indices:
                # Compute new average if we add this model
                n = len(selected)
                candidate_avg = (current_avg * n + preds[idx]) / (n + 1)
                score = metric_fn(y, candidate_avg)

                if score > best_candidate_score:
                    best_candidate_score = score
                    best_candidate = idx

            if best_candidate_score > best_score:
                best_score = best_candidate_score
                selected.append(best_candidate)
                self.selection_history_.append(
                    (best_candidate, best_candidate_score)
                )
                no_improve_count = 0
                self._log(
                    f"Iteration {iteration}: added model {best_candidate}, "
                    f"score={best_candidate_score:.4f}"
                )
            else:
                # Add anyway (may help later) but track no improvement
                selected.append(best_candidate)
                self.selection_history_.append(
                    (best_candidate, best_candidate_score)
                )
                no_improve_count += 1
                if no_improve_count >= self.patience:
                    self._log(
                        f"Early stopping at iteration {iteration} "
                        f"(no improvement for {self.patience} iterations)"
                    )
                    # Remove last `patience` additions that didn't help
                    selected = selected[: len(selected) - self.patience]
                    break

        # Compute weights from selection counts
        from collections import Counter

        counts = Counter(selected)
        total = sum(counts.values())
        self.weights_ = {idx: count / total for idx, count in counts.items()}
        self.best_score_ = best_score
        self.is_fitted_ = True

        self._log(f"Final weights: {self.weights_}")
        self._log(f"Best score: {self.best_score_:.4f}")
        return self

    def predict(
        self,
        X: np.ndarray | None = None,
        base_predictions: dict[int, np.ndarray]
        | list[np.ndarray]
        | np.ndarray
        | None = None,
    ) -> np.ndarray:
        """Predict combined risk scores using learned weights.

        Parameters
        ----------
        X : ignored
            Present for API compatibility. Not used directly.
        base_predictions : dict, list, or ndarray
            Risk score predictions from base models, in the same format
            as ``oof_predictions`` in ``fit``.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Weighted average risk scores.
        """
        self._check_is_fitted()

        if base_predictions is None:
            raise ValueError(
                "base_predictions must be provided for hill climbing ensemble."
            )

        # Normalize to dict
        if isinstance(base_predictions, np.ndarray) and base_predictions.ndim == 2:
            preds = {
                i: base_predictions[:, i]
                for i in range(base_predictions.shape[1])
            }
        elif isinstance(base_predictions, list):
            preds = {i: np.asarray(p) for i, p in enumerate(base_predictions)}
        elif isinstance(base_predictions, dict):
            preds = {k: np.asarray(v) for k, v in base_predictions.items()}
        else:
            raise ValueError(
                "base_predictions must be a dict, list, or 2D ndarray."
            )

        # Weighted average
        combined = np.zeros_like(next(iter(preds.values())), dtype=np.float64)
        for idx, weight in self.weights_.items():
            if idx in preds:
                combined += weight * preds[idx]

        return combined


# ---------------------------------------------------------------------------
# Voting Ensemble
# ---------------------------------------------------------------------------


class SurvivalVotingEnsemble(BaseSurvivalEstimator):
    """Weighted voting ensemble for survival models.

    Computes a (weighted) average of risk scores from multiple survival
    estimators. Optionally optimizes weights on the training data.

    Parameters
    ----------
    estimators : list of (str, estimator) tuples
        Named survival estimators.
    weights : list of float or None, default=None
        Weights for each estimator. If None, uses uniform weights.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    estimators_ : list of (str, fitted_estimator)
        Fitted estimators.
    weights_ : ndarray of shape (n_estimators,)
        Final weights used for combination.

    Example
    -------
    >>> # voting = SurvivalVotingEnsemble(
    >>> #     estimators=[("cox", cox_model), ("rsf", rsf_model)],
    >>> #     weights=[0.6, 0.4],
    >>> # )
    >>> # voting.fit(X_train, y_train)
    >>> # risk = voting.predict(X_test)
    """

    def __init__(
        self,
        estimators: list[tuple[str, Any]] | None = None,
        weights: list[float] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.estimators = estimators or []
        self.weights = weights

    def fit(self, X: np.ndarray, y: Any) -> "SurvivalVotingEnsemble":
        """Fit all base estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : structured array or compatible
            Survival target.

        Returns
        -------
        self
        """
        X, y = self._validate_survival_data(X, y)

        if len(self.estimators) == 0:
            raise ValueError("estimators must contain at least one (name, estimator) tuple.")

        n_estimators = len(self.estimators)

        # Set weights
        if self.weights is not None:
            if len(self.weights) != n_estimators:
                raise ValueError(
                    f"Number of weights ({len(self.weights)}) must match "
                    f"number of estimators ({n_estimators})."
                )
            self.weights_ = np.array(self.weights, dtype=np.float64)
        else:
            self.weights_ = np.ones(n_estimators, dtype=np.float64) / n_estimators

        # Normalize weights to sum to 1
        self.weights_ = self.weights_ / self.weights_.sum()

        # Fit all estimators
        self.estimators_ = []
        for name, estimator in self.estimators:
            self._log(f"Fitting {name}")
            fitted = clone(estimator)
            fitted.fit(X, y)
            self.estimators_.append((name, fitted))

        self.is_fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict risk scores as weighted average of base models.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        risk_scores : ndarray of shape (n_samples,)
            Weighted average risk scores (higher = more risk).
        """
        self._check_is_fitted()
        X = self._to_numpy(X)

        predictions = []
        for name, estimator in self.estimators_:
            predictions.append(estimator.predict(X))

        predictions = np.column_stack(predictions)
        return predictions @ self.weights_
