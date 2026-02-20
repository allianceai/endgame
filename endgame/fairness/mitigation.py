"""Fairness mitigation methods: pre-processing, in-processing, and post-processing.

Provides sklearn-compatible estimators for bias mitigation at different stages
of the ML pipeline:

- **Pre-processing**: ``ReweighingPreprocessor`` adjusts sample weights.
- **In-processing**: ``ExponentiatedGradient`` wraps a classifier with fairness
  constraints (requires fairlearn).
- **Post-processing**: ``CalibratedEqOdds`` adjusts per-group thresholds.

References
----------
- Kamiran & Calders "Data preprocessing techniques for classification without
  discrimination" (2012)
- Agarwal et al. "A Reductions Approach to Fair Classification" (2018)
- Hardt et al. "Equality of Opportunity in Supervised Learning" (2016)
- Pleiss et al. "On Fairness and Calibration" (2017)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

if TYPE_CHECKING:
    import pandas as pd

ArrayLike = Union[np.ndarray, list, "pd.Series"]


# =============================================================================
# Pre-processing: Reweighing
# =============================================================================


class ReweighingPreprocessor(BaseEstimator, TransformerMixin):
    """Compute sample weights to achieve demographic parity.

    Assigns higher weights to under-represented (group, label) combinations
    and lower weights to over-represented ones, so that the weighted label
    distribution is independent of the sensitive attribute.

    For each (group g, label y) cell the weight is::

        w(g, y) = [ P(Y=y) * P(A=g) ] / P(Y=y, A=g)

    This is a pre-processing method: use the returned weights as the
    ``sample_weight`` argument in downstream estimators.

    Parameters
    ----------
    sensitive_attr_index : int or str, optional
        Column index (int) or column name (str) in X that contains the
        sensitive attribute. If ``None``, the ``sensitive_attr`` parameter
        must be provided to ``fit`` / ``transform``.

    Attributes
    ----------
    groups_ : np.ndarray
        Unique groups seen during fit.
    labels_ : np.ndarray
        Unique labels seen during fit.
    weight_map_ : dict
        Mapping (group, label) -> weight.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fairness import ReweighingPreprocessor
    >>> X = np.array([[1, 0], [2, 0], [3, 1], [4, 1]])
    >>> y = np.array([0, 1, 0, 1])
    >>> sensitive = np.array(["A", "A", "B", "B"])
    >>> rw = ReweighingPreprocessor()
    >>> rw.fit(X, y, sensitive_attr=sensitive)
    ReweighingPreprocessor()
    >>> weights = rw.transform(X, y, sensitive_attr=sensitive)
    >>> weights.shape
    (4,)
    """

    def __init__(
        self,
        sensitive_attr_index: int | str | None = None,
    ):
        self.sensitive_attr_index = sensitive_attr_index

    def _extract_sensitive(
        self,
        X: np.ndarray,
        sensitive_attr: ArrayLike | None = None,
    ) -> np.ndarray:
        """Extract the sensitive attribute from X or an explicit argument.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        sensitive_attr : array-like, optional
            Explicit sensitive attribute array.

        Returns
        -------
        np.ndarray
            The sensitive attribute values.

        Raises
        ------
        ValueError
            If neither ``sensitive_attr`` nor ``sensitive_attr_index`` is set.
        """
        if sensitive_attr is not None:
            return np.asarray(sensitive_attr)

        if self.sensitive_attr_index is not None:
            idx = self.sensitive_attr_index
            if isinstance(idx, str):
                # Assume pandas-like column access; fall back to int
                try:
                    return np.asarray(X[idx])
                except (KeyError, TypeError, IndexError):
                    raise ValueError(
                        f"Cannot extract column '{idx}' from X. "
                        "Pass sensitive_attr explicitly."
                    )
            return np.asarray(X[:, idx])

        raise ValueError(
            "No sensitive attribute provided. Either set "
            "sensitive_attr_index in the constructor or pass "
            "sensitive_attr to fit/transform."
        )

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        sensitive_attr: ArrayLike | None = None,
        **fit_params: Any,
    ) -> ReweighingPreprocessor:
        """Compute reweighing weights from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        sensitive_attr : array-like of shape (n_samples,), optional
            Sensitive attribute values. Required if ``sensitive_attr_index``
            was not set in the constructor.
        **fit_params : dict
            Ignored. Present for API compatibility.

        Returns
        -------
        self
            Fitted preprocessor.
        """
        X_arr = np.asarray(X) if not hasattr(X, "__array__") else np.asarray(X)
        y_arr = np.asarray(y)
        s_arr = self._extract_sensitive(X, sensitive_attr)

        n = len(y_arr)
        self.groups_ = np.unique(s_arr)
        self.labels_ = np.unique(y_arr)

        self.weight_map_: dict[tuple, float] = {}

        for g in self.groups_:
            for lab in self.labels_:
                p_y = np.sum(y_arr == lab) / n
                p_g = np.sum(s_arr == g) / n
                p_yg = np.sum((y_arr == lab) & (s_arr == g)) / n

                if p_yg > 0:
                    self.weight_map_[(g, lab)] = (p_y * p_g) / p_yg
                else:
                    # Cell is empty; assign weight 1.0 (no adjustment)
                    self.weight_map_[(g, lab)] = 1.0

        self._is_fitted = True
        return self

    def transform(
        self,
        X: Any,
        y: ArrayLike | None = None,
        sensitive_attr: ArrayLike | None = None,
        **transform_params: Any,
    ) -> np.ndarray:
        """Return per-sample weights for bias correction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,), optional
            Labels. Required to look up weights.
        sensitive_attr : array-like of shape (n_samples,), optional
            Sensitive attribute values.
        **transform_params : dict
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sample weights.

        Raises
        ------
        ValueError
            If ``y`` is not provided.
        """
        check_is_fitted(self, ["weight_map_", "groups_", "labels_"])

        if y is None:
            raise ValueError(
                "ReweighingPreprocessor.transform requires y to compute "
                "per-sample weights."
            )

        y_arr = np.asarray(y)
        s_arr = self._extract_sensitive(X, sensitive_attr)

        weights = np.ones(len(y_arr), dtype=np.float64)
        for i in range(len(y_arr)):
            key = (s_arr[i], y_arr[i])
            weights[i] = self.weight_map_.get(key, 1.0)

        return weights

    def fit_transform(
        self,
        X: Any,
        y: ArrayLike | None = None,
        sensitive_attr: ArrayLike | None = None,
        **fit_params: Any,
    ) -> np.ndarray:
        """Fit and return sample weights in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : array-like of shape (n_samples,), optional
            Labels.
        sensitive_attr : array-like of shape (n_samples,), optional
            Sensitive attribute values.
        **fit_params : dict
            Ignored.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Sample weights.
        """
        return self.fit(X, y, sensitive_attr=sensitive_attr, **fit_params).transform(
            X, y, sensitive_attr=sensitive_attr
        )


# =============================================================================
# In-processing: Exponentiated Gradient
# =============================================================================


class ExponentiatedGradient(BaseEstimator, ClassifierMixin):
    """Fairness-constrained classification via exponentiated gradient reduction.

    Wraps any sklearn-compatible binary classifier and trains it under a
    fairness constraint (demographic parity or equalized odds) using the
    fairlearn library's ``ExponentiatedGradient`` algorithm.

    This is an in-processing method: the fairness constraint is enforced
    during training.

    Parameters
    ----------
    estimator : sklearn estimator
        Base binary classifier to wrap. Must implement ``fit`` and
        ``predict``.
    constraint : str, default="demographic_parity"
        Fairness constraint to enforce. One of:

        - ``"demographic_parity"`` : equalize selection rates
        - ``"equalized_odds"`` : equalize TPR and FPR
        - ``"true_positive_rate_parity"`` : equalize TPR (equal opportunity)
        - ``"error_rate_parity"`` : equalize error rates

    constraint_weight : float, default=0.5
        Trade-off parameter. Higher values enforce the constraint more
        strictly at the cost of overall accuracy. Must be in (0, 1].
    max_iter : int, default=50
        Maximum number of iterations for the exponentiated gradient solver.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    mitigator_ : fairlearn.reductions.ExponentiatedGradient
        The fitted fairlearn mitigator.
    classes_ : np.ndarray
        Unique class labels.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from endgame.fairness import ExponentiatedGradient
    >>> clf = ExponentiatedGradient(
    ...     estimator=LogisticRegression(),
    ...     constraint="demographic_parity",
    ... )
    >>> clf.fit(X_train, y_train, sensitive_attr=sensitive_train)
    ExponentiatedGradient(...)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        estimator: Any = None,
        constraint: str = "demographic_parity",
        constraint_weight: float = 0.5,
        max_iter: int = 50,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.constraint = constraint
        self.constraint_weight = constraint_weight
        self.max_iter = max_iter
        self.random_state = random_state

    @staticmethod
    def _import_fairlearn():
        """Lazy-import fairlearn and return the module.

        Returns
        -------
        module
            The ``fairlearn.reductions`` module.

        Raises
        ------
        ImportError
            If fairlearn is not installed.
        """
        try:
            import fairlearn.reductions as reductions
            return reductions
        except ImportError:
            raise ImportError(
                "fairlearn is required for ExponentiatedGradient. "
                "Install with: pip install fairlearn"
            )

    def _get_constraint_object(self):
        """Build the fairlearn constraint object.

        Returns
        -------
        fairlearn.reductions.Moment
            The constraint object.

        Raises
        ------
        ValueError
            If ``self.constraint`` is not recognized.
        """
        reductions = self._import_fairlearn()

        constraint_map = {
            "demographic_parity": reductions.DemographicParity,
            "equalized_odds": reductions.EqualizedOdds,
            "true_positive_rate_parity": reductions.TruePositiveRateParity,
            "error_rate_parity": reductions.ErrorRateParity,
        }

        if self.constraint not in constraint_map:
            raise ValueError(
                f"Unknown constraint '{self.constraint}'. "
                f"Choose from: {list(constraint_map.keys())}"
            )

        return constraint_map[self.constraint]()

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        sensitive_attr: ArrayLike | None = None,
        **fit_params: Any,
    ) -> ExponentiatedGradient:
        """Fit the fairness-constrained classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        sensitive_attr : array-like of shape (n_samples,)
            Sensitive attribute for fairness constraint.
        **fit_params : dict
            Additional parameters (ignored).

        Returns
        -------
        self
            Fitted estimator.

        Raises
        ------
        ValueError
            If ``sensitive_attr`` is not provided.
        ImportError
            If fairlearn is not installed.
        """
        if sensitive_attr is None:
            raise ValueError(
                "sensitive_attr is required for ExponentiatedGradient.fit(). "
                "Pass the sensitive attribute array."
            )

        reductions = self._import_fairlearn()

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        s_arr = np.asarray(sensitive_attr)

        self.classes_ = np.unique(y_arr)

        base = clone(self.estimator) if self.estimator is not None else None
        if base is None:
            from sklearn.linear_model import LogisticRegression
            base = LogisticRegression()

        constraint_obj = self._get_constraint_object()

        self.mitigator_ = reductions.ExponentiatedGradient(
            estimator=base,
            constraints=constraint_obj,
            max_iter=self.max_iter,
        )
        self.mitigator_.fit(X_arr, y_arr, sensitive_features=s_arr)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["mitigator_"])
        return self.mitigator_.predict(np.asarray(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Uses the internal randomized classifier to return soft predictions.
        Falls back to hard predictions if the mitigator does not support
        ``_pmf_predict``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        check_is_fitted(self, ["mitigator_"])
        X_arr = np.asarray(X)

        if hasattr(self.mitigator_, "_pmf_predict"):
            return self.mitigator_._pmf_predict(X_arr)

        # Fallback: convert hard predictions to one-hot
        preds = self.mitigator_.predict(X_arr)
        n_classes = len(self.classes_)
        proba = np.zeros((len(preds), n_classes))
        for i, cls in enumerate(self.classes_):
            proba[preds == cls, i] = 1.0
        return proba


# =============================================================================
# Post-processing: Calibrated Equalized Odds
# =============================================================================


class CalibratedEqOdds(BaseEstimator, ClassifierMixin):
    """Post-processing threshold adjustment for equalized odds.

    Adjusts per-group classification thresholds on predicted probabilities
    to equalize true positive and false positive rates across groups.
    Finds optimal thresholds via grid search on calibration data.

    This is a post-processing method: it wraps a trained classifier and
    adjusts its decisions without retraining.

    Parameters
    ----------
    estimator : sklearn classifier
        A *fitted* classifier with ``predict_proba``.
    cost_weight : float, default=1.0
        Relative cost of false negatives vs false positives.
        Higher values favor higher TPR (at the cost of higher FPR).
    grid_size : int, default=101
        Number of threshold candidates to evaluate per group.
    random_state : int or None, default=None
        Random seed (currently unused, reserved for future stochastic
        extensions).

    Attributes
    ----------
    thresholds_ : dict
        Mapping group -> optimal classification threshold.
    groups_ : np.ndarray
        Unique groups seen during fit.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from endgame.fairness import CalibratedEqOdds
    >>> base = LogisticRegression().fit(X_train, y_train)
    >>> ceqo = CalibratedEqOdds(estimator=base)
    >>> ceqo.fit(X_cal, y_cal, sensitive_attr=sensitive_cal)
    CalibratedEqOdds(...)
    >>> y_pred = ceqo.predict(X_test, sensitive_attr=sensitive_test)
    """

    def __init__(
        self,
        estimator: Any = None,
        cost_weight: float = 1.0,
        grid_size: int = 101,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.cost_weight = cost_weight
        self.grid_size = grid_size
        self.random_state = random_state

    def fit(
        self,
        X: Any,
        y: ArrayLike,
        sensitive_attr: ArrayLike | None = None,
        **fit_params: Any,
    ) -> CalibratedEqOdds:
        """Find per-group thresholds that equalize odds on calibration data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Calibration features.
        y : array-like of shape (n_samples,)
            Calibration labels.
        sensitive_attr : array-like of shape (n_samples,)
            Sensitive attribute values.
        **fit_params : dict
            Ignored.

        Returns
        -------
        self
            Fitted post-processor.

        Raises
        ------
        ValueError
            If ``sensitive_attr`` is not provided or the base estimator
            lacks ``predict_proba``.
        """
        if sensitive_attr is None:
            raise ValueError(
                "sensitive_attr is required for CalibratedEqOdds.fit()."
            )
        if self.estimator is None:
            raise ValueError("A fitted estimator must be provided.")
        if not hasattr(self.estimator, "predict_proba"):
            raise ValueError(
                "The base estimator must implement predict_proba."
            )

        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        s_arr = np.asarray(sensitive_attr)

        self.groups_ = np.unique(s_arr)
        self.classes_ = np.unique(y_arr)

        # Get probabilities from the base model
        proba = self.estimator.predict_proba(X_arr)
        if proba.ndim == 2:
            proba_pos = proba[:, 1]
        else:
            proba_pos = proba

        # Compute global target TPR and FPR (overall model performance at 0.5)
        pred_global = (proba_pos >= 0.5).astype(int)
        pos_mask_global = y_arr == 1
        neg_mask_global = y_arr == 0

        target_tpr = (
            float(np.mean(pred_global[pos_mask_global]))
            if pos_mask_global.sum() > 0
            else 0.5
        )
        target_fpr = (
            float(np.mean(pred_global[neg_mask_global]))
            if neg_mask_global.sum() > 0
            else 0.5
        )

        # Find per-group thresholds
        thresholds = np.linspace(0.0, 1.0, self.grid_size)
        self.thresholds_: dict[Any, float] = {}

        for group in self.groups_:
            mask = s_arr == group
            y_g = y_arr[mask]
            p_g = proba_pos[mask]

            pos_mask = y_g == 1
            neg_mask = y_g == 0

            best_threshold = 0.5
            best_cost = float("inf")

            for t in thresholds:
                pred_g = (p_g >= t).astype(int)

                tpr = (
                    float(np.mean(pred_g[pos_mask]))
                    if pos_mask.sum() > 0
                    else 0.0
                )
                fpr = (
                    float(np.mean(pred_g[neg_mask]))
                    if neg_mask.sum() > 0
                    else 0.0
                )

                # Cost: weighted combination of TPR and FPR deviations
                cost = (
                    self.cost_weight * abs(tpr - target_tpr)
                    + abs(fpr - target_fpr)
                )

                if cost < best_cost:
                    best_cost = cost
                    best_threshold = t

            self.thresholds_[group] = float(best_threshold)

        self._is_fitted = True
        return self

    def predict(
        self,
        X: Any,
        sensitive_attr: ArrayLike | None = None,
    ) -> np.ndarray:
        """Predict class labels using per-group thresholds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.
        sensitive_attr : array-like of shape (n_samples,)
            Sensitive attribute values.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Predicted class labels.

        Raises
        ------
        ValueError
            If ``sensitive_attr`` is not provided.
        """
        check_is_fitted(self, ["thresholds_", "groups_"])

        if sensitive_attr is None:
            raise ValueError(
                "sensitive_attr is required for CalibratedEqOdds.predict()."
            )

        X_arr = np.asarray(X)
        s_arr = np.asarray(sensitive_attr)

        proba = self.estimator.predict_proba(X_arr)
        if proba.ndim == 2:
            proba_pos = proba[:, 1]
        else:
            proba_pos = proba

        predictions = np.zeros(len(s_arr), dtype=int)
        for group in self.groups_:
            mask = s_arr == group
            threshold = self.thresholds_.get(group, 0.5)
            predictions[mask] = (proba_pos[mask] >= threshold).astype(int)

        # Handle unseen groups with default threshold
        seen = set(self.groups_)
        for i in range(len(s_arr)):
            if s_arr[i] not in seen:
                predictions[i] = int(proba_pos[i] >= 0.5)

        return predictions

    def predict_proba(self, X: Any) -> np.ndarray:
        """Return raw probabilities from the base estimator.

        Post-processing adjusts thresholds, not probabilities. This method
        exposes the underlying predicted probabilities for transparency.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        np.ndarray of shape (n_samples, n_classes)
            Class probability estimates from the base estimator.
        """
        check_is_fitted(self, ["thresholds_"])
        return self.estimator.predict_proba(np.asarray(X))
