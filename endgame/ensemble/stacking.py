"""Stacking Ensemble: Multi-level model stacking."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

from endgame.core.base import BaseEnsemble


class StackingEnsemble(BaseEnsemble):
    """Multi-level stacking with out-of-fold prediction handling.

    Level 1: Diverse base models (GBDTs, NNs, etc.)
    Level 2: Meta-learner (typically Ridge/Linear Regression)

    The meta-learner is trained on out-of-fold predictions from Level 1
    to prevent overfitting.

    Parameters
    ----------
    base_estimators : List[estimator]
        Level 1 models.
    meta_estimator : estimator, optional
        Level 2 model. Default: Ridge for regression, LogisticRegression for classification.
    cv : int or CV splitter, default=5
        Cross-validation strategy for OOF predictions.
    passthrough : bool, default=False
        Whether to include original features in Level 2.
    use_proba : bool, default=True
        Use predict_proba for classification (if available).
    stack_method : str, default='auto'
        Method for stacking: 'auto', 'predict', 'predict_proba'.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    base_estimators_ : List[estimator]
        Fitted Level 1 models.
    meta_estimator_ : estimator
        Fitted Level 2 model.
    oof_predictions_ : ndarray
        Out-of-fold predictions used for meta-learner training.

    Examples
    --------
    >>> from endgame.ensemble import StackingEnsemble
    >>> base_models = [LGBMWrapper(), XGBWrapper(), CatBoostWrapper()]
    >>> stacker = StackingEnsemble(base_estimators=base_models)
    >>> stacker.fit(X_train, y_train)
    >>> predictions = stacker.predict(X_test)
    """

    def __init__(
        self,
        base_estimators: list[BaseEstimator] | None = None,
        meta_estimator: BaseEstimator | None = None,
        cv: int | Any = 5,
        passthrough: bool = False,
        use_proba: bool = True,
        stack_method: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            estimators=base_estimators,
            random_state=random_state,
            verbose=verbose,
        )
        self.base_estimators = base_estimators or []
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.passthrough = passthrough
        self.use_proba = use_proba
        self.stack_method = stack_method

        self.base_estimators_: list[BaseEstimator] = []
        self.meta_estimator_: BaseEstimator | None = None
        self.oof_predictions_: np.ndarray | None = None
        self._is_classifier: bool = False
        self._n_features_in: int = 0

    def _get_default_meta_estimator(self) -> BaseEstimator:
        """Get default meta-estimator based on task."""
        if self._is_classifier:
            return LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            )
        return Ridge(alpha=1.0, random_state=self.random_state)

    def _get_stack_method(self, estimator: BaseEstimator) -> str:
        """Determine stacking method for an estimator."""
        if self.stack_method != "auto":
            return self.stack_method

        if self._is_classifier and self.use_proba and hasattr(estimator, "predict_proba"):
            return "predict_proba"
        return "predict"

    def _get_cv_splitter(self, y: np.ndarray) -> Any:
        """Get cross-validation splitter."""
        if isinstance(self.cv, int):
            if self._is_classifier:
                return StratifiedKFold(
                    n_splits=self.cv,
                    shuffle=True,
                    random_state=self.random_state,
                )
            return KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        return self.cv

    def fit(
        self,
        X,
        y,
        sample_weight: np.ndarray | None = None,
        **fit_params,
    ) -> "StackingEnsemble":
        """Fit the stacking ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like, optional
            Sample weights.
        **fit_params
            Additional parameters.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self._n_features_in = X.shape[1]
        self._is_classifier = len(np.unique(y)) <= 20

        if self._is_classifier:
            self.classes_ = np.unique(y)
            self.n_classes_ = len(self.classes_)

        cv = self._get_cv_splitter(y)

        self._log(f"Fitting {len(self.base_estimators)} base estimators...")

        # Generate OOF predictions for each base estimator
        oof_list = []

        for i, estimator in enumerate(self.base_estimators):
            self._log(f"  Fitting base estimator {i + 1}/{len(self.base_estimators)}")

            method = self._get_stack_method(estimator)

            try:
                if method == "predict_proba":
                    oof_pred = cross_val_predict(
                        estimator,
                        X,
                        y,
                        cv=cv,
                        method="predict_proba",
                    )
                    # For binary classification, use probability of positive class
                    if oof_pred.ndim == 2 and oof_pred.shape[1] == 2:
                        oof_pred = oof_pred[:, 1]
                else:
                    oof_pred = cross_val_predict(
                        estimator,
                        X,
                        y,
                        cv=cv,
                        method="predict",
                    )
            except Exception as e:
                self._log(f"  Warning: {e}, falling back to predict", level="warn")
                oof_pred = cross_val_predict(
                    estimator,
                    X,
                    y,
                    cv=cv,
                    method="predict",
                )

            if oof_pred.ndim == 1:
                oof_pred = oof_pred.reshape(-1, 1)

            oof_list.append(oof_pred)

        # Stack OOF predictions
        self.oof_predictions_ = np.hstack(oof_list)

        # Add original features if passthrough
        if self.passthrough:
            meta_features = np.hstack([self.oof_predictions_, X])
        else:
            meta_features = self.oof_predictions_

        # Fit meta-estimator
        self._log("Fitting meta-estimator...")
        self.meta_estimator_ = self.meta_estimator or self._get_default_meta_estimator()
        self.meta_estimator_ = clone(self.meta_estimator_)

        if sample_weight is not None:
            self.meta_estimator_.fit(meta_features, y, sample_weight=sample_weight)
        else:
            self.meta_estimator_.fit(meta_features, y)

        # Fit base estimators on full data for prediction
        self._log("Fitting base estimators on full data...")
        self.base_estimators_ = []
        for i, estimator in enumerate(self.base_estimators):
            fitted = clone(estimator)
            if sample_weight is not None:
                fitted.fit(X, y, sample_weight=sample_weight)
            else:
                fitted.fit(X, y)
            self.base_estimators_.append(fitted)

        self._is_fitted = True
        return self

    def _get_base_predictions(self, X: np.ndarray) -> np.ndarray:
        """Get predictions from base estimators."""
        pred_list = []

        for estimator in self.base_estimators_:
            method = self._get_stack_method(estimator)

            if method == "predict_proba" and hasattr(estimator, "predict_proba"):
                pred = estimator.predict_proba(X)
                if pred.ndim == 2 and pred.shape[1] == 2:
                    pred = pred[:, 1]
            else:
                pred = estimator.predict(X)

            if pred.ndim == 1:
                pred = pred.reshape(-1, 1)

            pred_list.append(pred)

        return np.hstack(pred_list)

    def predict(self, X) -> np.ndarray:
        """Predict using the stacking ensemble.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Get base predictions
        base_predictions = self._get_base_predictions(X)

        # Add original features if passthrough
        if self.passthrough:
            meta_features = np.hstack([base_predictions, X])
        else:
            meta_features = base_predictions

        return self.meta_estimator_.predict(meta_features)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()

        if not self._is_classifier:
            raise ValueError("predict_proba only available for classification")

        X = np.asarray(X)

        # Get base predictions
        base_predictions = self._get_base_predictions(X)

        # Add original features if passthrough
        if self.passthrough:
            meta_features = np.hstack([base_predictions, X])
        else:
            meta_features = base_predictions

        if hasattr(self.meta_estimator_, "predict_proba"):
            return self.meta_estimator_.predict_proba(meta_features)

        # Fall back to decision function if available
        if hasattr(self.meta_estimator_, "decision_function"):
            decision = self.meta_estimator_.decision_function(meta_features)
            # Convert to probabilities using sigmoid
            proba = 1 / (1 + np.exp(-decision))
            if proba.ndim == 1:
                return np.vstack([1 - proba, proba]).T
            return proba

        raise ValueError("Meta-estimator doesn't support probability predictions")

    def score(self, X, y, sample_weight=None) -> float:
        """Return the mean accuracy on the given test data and labels.

        For classification, this is the accuracy score. For regression,
        this is the R^2 score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels for classification, true values for regression.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        float
            Score of the predictions.
        """
        self._check_is_fitted()

        if self._is_classifier:
            from sklearn.metrics import accuracy_score
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            from sklearn.metrics import r2_score
            y_pred = self.predict(X)
            return r2_score(y, y_pred, sample_weight=sample_weight)
