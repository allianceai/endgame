from __future__ import annotations

"""AutoML stacking ensemble builder.

This module provides automatic multi-level stacking ensemble construction
optimized for AutoML pipelines.
"""

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)


@dataclass
class StackerConfig:
    """Configuration for AutoMLStacker.

    Attributes
    ----------
    num_stack_levels : int
        Number of stacking levels.
    num_bag_folds : int
        Number of bagging folds.
    use_features_in_secondary : bool
        Whether to include original features in meta-learner.
    meta_learner : str
        Meta-learner model name.
    use_probas : bool
        Whether to use probabilities as meta-features.
    """

    num_stack_levels: int = 1
    num_bag_folds: int = 5
    use_features_in_secondary: bool = True
    meta_learner: str = "lgbm"
    use_probas: bool = True


class AutoMLStacker(BaseEstimator):
    """Multi-level stacking ensemble for AutoML.

    This class builds stacking ensembles optimized for AutoML pipelines,
    supporting multiple stacking levels and automatic meta-learner selection.

    Parameters
    ----------
    base_estimators : dict
        Dictionary mapping names to fitted base estimators.
    task_type : str, default="classification"
        Task type ("classification" or "regression").
    num_stack_levels : int, default=1
        Number of stacking levels (1 or 2).
    num_bag_folds : int, default=5
        Number of folds for OOF prediction generation.
    use_features_in_secondary : bool, default=True
        Whether to include original features in meta-learner input.
    meta_learner : str, default="lgbm"
        Meta-learner type: "lgbm", "linear", "ridge", or "xgb".
    use_probas : bool, default=True
        For classification, whether to use probabilities as meta-features.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the stacker is fitted.
    weights_ : dict
        Optimal weights for base models.
    meta_estimator_ : estimator
        Fitted meta-learner.
    oof_predictions_ : dict
        OOF predictions for each base model.

    Examples
    --------
    >>> stacker = AutoMLStacker(
    ...     base_estimators=trained_models,
    ...     task_type="classification",
    ... )
    >>> stacker.fit(X_train, y_train)
    >>> predictions = stacker.predict(X_test)
    """

    def __init__(
        self,
        base_estimators: dict[str, Any],
        task_type: str = "classification",
        num_stack_levels: int = 1,
        num_bag_folds: int = 5,
        use_features_in_secondary: bool = True,
        meta_learner: str = "lgbm",
        use_probas: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.base_estimators = base_estimators
        self.task_type = task_type
        self.num_stack_levels = num_stack_levels
        self.num_bag_folds = num_bag_folds
        self.use_features_in_secondary = use_features_in_secondary
        self.meta_learner = meta_learner
        self.use_probas = use_probas
        self.random_state = random_state
        self.verbose = verbose

        # State
        self.is_fitted_ = False
        self.weights_: dict[str, float] = {}
        self.meta_estimator_: Any | None = None
        self.oof_predictions_: dict[str, np.ndarray] = {}
        self._classes: np.ndarray | None = None

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        oof_predictions: dict[str, np.ndarray] | None = None,
    ) -> AutoMLStacker:
        """Fit the stacking ensemble.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training target.
        oof_predictions : dict, optional
            Pre-computed OOF predictions for base models.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Store classes for classification
        if self.task_type == "classification":
            self._classes = np.unique(y)

        # Generate or use provided OOF predictions
        if oof_predictions is not None:
            self.oof_predictions_ = oof_predictions
        else:
            self.oof_predictions_ = self._generate_oof_predictions(X, y)

        # Build meta-features
        meta_X = self._build_meta_features(X)

        # Fit meta-learner
        self.meta_estimator_ = self._create_meta_learner()
        self.meta_estimator_.fit(meta_X, y)

        # Compute optimal weights using hill climbing
        self.weights_ = self._compute_optimal_weights(y)

        self.is_fitted_ = True

        if self.verbose > 0:
            print(f"Stacker fitted with {len(self.base_estimators)} base models")
            print(f"Meta-learner: {self.meta_learner}")

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get base predictions
        base_preds = self._get_base_predictions(X)

        # Build meta-features
        meta_X = self._build_meta_features_from_predictions(X, base_preds)

        # Meta-learner prediction
        return self.meta_estimator_.predict(meta_X)

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------
        ndarray
            Class probabilities.
        """
        self._check_is_fitted()

        if self.task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get base predictions
        base_preds = self._get_base_predictions(X)

        # Build meta-features
        meta_X = self._build_meta_features_from_predictions(X, base_preds)

        # Meta-learner prediction
        if hasattr(self.meta_estimator_, "predict_proba"):
            return self.meta_estimator_.predict_proba(meta_X)
        else:
            # Fallback: convert predictions to one-hot
            preds = self.meta_estimator_.predict(meta_X)
            n_classes = len(self._classes)
            proba = np.zeros((len(X), n_classes))
            for i, pred in enumerate(preds):
                class_idx = np.where(self._classes == pred)[0][0]
                proba[i, class_idx] = 1.0
            return proba

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Generate out-of-fold predictions for base models.

        Parameters
        ----------
        X : ndarray
            Features.
        y : ndarray
            Target.

        Returns
        -------
        dict
            OOF predictions for each model.
        """
        oof_preds = {}

        # Choose CV strategy
        if self.task_type == "classification":
            cv = StratifiedKFold(
                n_splits=self.num_bag_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            cv = KFold(
                n_splits=self.num_bag_folds,
                shuffle=True,
                random_state=self.random_state,
            )

        for name, estimator in self.base_estimators.items():
            if self.verbose > 0:
                print(f"Generating OOF predictions for {name}")

            try:
                if self.task_type == "classification" and self.use_probas:
                    if hasattr(estimator, "predict_proba"):
                        oof = cross_val_predict(
                            estimator, X, y, cv=cv, method="predict_proba"
                        )
                    else:
                        oof = cross_val_predict(estimator, X, y, cv=cv)
                else:
                    oof = cross_val_predict(estimator, X, y, cv=cv)

                oof_preds[name] = oof

            except Exception as e:
                logger.warning(f"Failed to generate OOF for {name}: {e}")
                # Fallback to simple predictions
                oof_preds[name] = estimator.predict(X)

        return oof_preds

    def _get_base_predictions(
        self,
        X: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Get predictions from base models.

        Parameters
        ----------
        X : ndarray
            Features.

        Returns
        -------
        dict
            Predictions from each model.
        """
        predictions = {}

        for name, estimator in self.base_estimators.items():
            try:
                if self.task_type == "classification" and self.use_probas:
                    if hasattr(estimator, "predict_proba"):
                        pred = estimator.predict_proba(X)
                    else:
                        pred = estimator.predict(X)
                else:
                    pred = estimator.predict(X)

                predictions[name] = pred

            except Exception as e:
                logger.warning(f"Failed to get predictions from {name}: {e}")

        return predictions

    def _build_meta_features(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """Build meta-features from OOF predictions.

        Parameters
        ----------
        X : ndarray
            Original features.

        Returns
        -------
        ndarray
            Meta-features.
        """
        return self._build_meta_features_from_predictions(X, self.oof_predictions_)

    def _build_meta_features_from_predictions(
        self,
        X: np.ndarray,
        predictions: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Build meta-features from predictions.

        Parameters
        ----------
        X : ndarray
            Original features.
        predictions : dict
            Predictions from base models.

        Returns
        -------
        ndarray
            Meta-features.
        """
        meta_features = []

        for name, pred in predictions.items():
            if pred.ndim == 1:
                meta_features.append(pred.reshape(-1, 1))
            else:
                meta_features.append(pred)

        meta_X = np.hstack(meta_features)

        # Optionally include original features
        if self.use_features_in_secondary:
            meta_X = np.hstack([meta_X, X])

        return meta_X

    def _create_meta_learner(self):
        """Create the meta-learner estimator.

        Returns
        -------
        estimator
            Meta-learner.
        """
        if self.meta_learner == "lgbm":
            try:
                if self.task_type == "classification":
                    from lightgbm import LGBMClassifier
                    return LGBMClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=self.random_state,
                        verbose=-1,
                    )
                else:
                    from lightgbm import LGBMRegressor
                    return LGBMRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=self.random_state,
                        verbose=-1,
                    )
            except ImportError:
                logger.warning("LightGBM not available, using linear meta-learner")
                self.meta_learner = "linear"

        if self.meta_learner == "xgb":
            try:
                if self.task_type == "classification":
                    from xgboost import XGBClassifier
                    return XGBClassifier(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=self.random_state,
                        verbosity=0,
                    )
                else:
                    from xgboost import XGBRegressor
                    return XGBRegressor(
                        n_estimators=100,
                        learning_rate=0.1,
                        max_depth=4,
                        random_state=self.random_state,
                        verbosity=0,
                    )
            except ImportError:
                logger.warning("XGBoost not available, using linear meta-learner")
                self.meta_learner = "linear"

        if self.meta_learner in ("linear", "ridge"):
            if self.task_type == "classification":
                from sklearn.linear_model import LogisticRegression
                return LogisticRegression(
                    C=1.0,
                    max_iter=1000,
                    random_state=self.random_state,
                )
            else:
                from sklearn.linear_model import Ridge
                return Ridge(alpha=1.0, random_state=self.random_state)

        raise ValueError(f"Unknown meta-learner: {self.meta_learner}")

    def _compute_optimal_weights(
        self,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Compute optimal weights using hill climbing.

        Parameters
        ----------
        y : ndarray
            Target values.

        Returns
        -------
        dict
            Optimal weights for each model.
        """
        try:
            from endgame.ensemble.hill_climbing import HillClimbingEnsemble

            # Prepare predictions
            preds_list = []
            model_names = []

            for name, oof_pred in self.oof_predictions_.items():
                if oof_pred.ndim > 1:
                    # Use class 1 probability for binary classification
                    if oof_pred.shape[1] == 2:
                        preds_list.append(oof_pred[:, 1])
                    else:
                        preds_list.append(oof_pred)
                else:
                    preds_list.append(oof_pred)
                model_names.append(name)

            # Run hill climbing
            hc = HillClimbingEnsemble(
                metric="accuracy" if self.task_type == "classification" else "r2",
                n_iterations=50,
            )
            hc.fit(preds_list, y)

            weights = {name: w for name, w in zip(model_names, hc.weights_)}

        except ImportError:
            # Fallback to equal weights
            n_models = len(self.base_estimators)
            weights = {name: 1.0 / n_models for name in self.base_estimators}

        return weights

    def _check_is_fitted(self) -> None:
        """Check if the stacker is fitted."""
        if not self.is_fitted_:
            raise RuntimeError("Stacker is not fitted. Call fit() first.")

    def get_model_weights(self) -> pd.DataFrame:
        """Get model weights as a DataFrame.

        Returns
        -------
        DataFrame
            Model weights.
        """
        return pd.DataFrame([
            {"model": name, "weight": weight}
            for name, weight in self.weights_.items()
        ]).sort_values("weight", ascending=False)


class HillClimbingStacker(AutoMLStacker):
    """Stacker using hill climbing for weight optimization.

    This variant uses only hill climbing for combining predictions,
    without a separate meta-learner.
    """

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        oof_predictions: dict[str, np.ndarray] | None = None,
    ) -> HillClimbingStacker:
        """Fit using hill climbing.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training target.
        oof_predictions : dict, optional
            Pre-computed OOF predictions.

        Returns
        -------
        self
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        if self.task_type == "classification":
            self._classes = np.unique(y)

        # Generate OOF predictions if not provided
        if oof_predictions is not None:
            self.oof_predictions_ = oof_predictions
        else:
            self.oof_predictions_ = self._generate_oof_predictions(X, y)

        # Compute optimal weights
        self.weights_ = self._compute_optimal_weights(y)

        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Make predictions using weighted average.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()

        if isinstance(X, pd.DataFrame):
            X = X.values

        # Get weighted average of predictions
        base_preds = self._get_base_predictions(X)

        weighted_pred = None
        total_weight = sum(self.weights_.values())

        for name, pred in base_preds.items():
            weight = self.weights_.get(name, 0) / total_weight

            if pred.ndim > 1:
                pred = pred[:, 1] if pred.shape[1] == 2 else np.argmax(pred, axis=1)

            if weighted_pred is None:
                weighted_pred = weight * pred
            else:
                weighted_pred += weight * pred

        if self.task_type == "classification":
            return (weighted_pred > 0.5).astype(int)
        else:
            return weighted_pred

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted average.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------
        ndarray
            Probabilities.
        """
        self._check_is_fitted()

        if self.task_type != "classification":
            raise ValueError("predict_proba only for classification")

        if isinstance(X, pd.DataFrame):
            X = X.values

        base_preds = self._get_base_predictions(X)

        weighted_proba = None
        total_weight = sum(self.weights_.values())

        for name, pred in base_preds.items():
            weight = self.weights_.get(name, 0) / total_weight

            if pred.ndim == 1:
                # Convert to 2-class probabilities
                pred = np.column_stack([1 - pred, pred])

            if weighted_proba is None:
                weighted_proba = weight * pred
            else:
                weighted_proba += weight * pred

        return weighted_proba
