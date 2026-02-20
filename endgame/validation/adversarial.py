"""Adversarial Validation for detecting train/test distribution drift."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from endgame.core.base import EndgameEstimator
from endgame.core.types import AdversarialValidationResult

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class AdversarialValidator(EndgameEstimator):
    """Detects train/test distribution drift using adversarial validation.

    Trains a classifier to distinguish train from test data. High AUC (>0.5)
    indicates distribution drift. Feature importances identify drifting features.

    This is a critical technique documented across winning solutions to prevent
    leaderboard overfitting when CV doesn't correlate with public LB.

    Parameters
    ----------
    estimator : sklearn-compatible classifier, optional
        The classifier used for adversarial validation.
        If None, uses LightGBM if available, else RandomForest.
    sample_frac : float, default=1.0
        Fraction of data to use (for large datasets).
    cv : int, default=5
        Number of cross-validation folds.
    threshold : float, default=0.7
        AUC threshold above which to flag significant drift.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    auc_score_ : float
        ROC-AUC score from adversarial validation.
    feature_importances_ : Dict[str, float]
        Feature importance in distinguishing train/test.
    drifted_features_ : List[str]
        Features contributing most to drift (sorted by importance).

    Examples
    --------
    >>> from endgame.validation import AdversarialValidator
    >>> av = AdversarialValidator(threshold=0.6)
    >>> result = av.check_drift(X_train, X_test)
    >>> print(f"Drift AUC: {result.auc_score:.3f}")
    >>> if result.drift_severity == 'severe':
    ...     # Remove drifted features
    ...     drop_cols = result.drifted_features[:5]
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        sample_frac: float = 1.0,
        cv: int = 5,
        threshold: float = 0.7,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.estimator = estimator
        self.sample_frac = sample_frac
        self.cv = cv
        self.threshold = threshold

        self.auc_score_: float | None = None
        self.feature_importances_: dict[str, float] | None = None
        self.drifted_features_: list[str] | None = None
        self._estimator: BaseEstimator | None = None

    def _get_default_estimator(self) -> BaseEstimator:
        """Get the default estimator for adversarial validation."""
        # Try LightGBM first (fastest and most accurate)
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
                max_depth=5,
                verbosity=-1,
                n_jobs=-1,
                random_state=self.random_state,
            )
        except ImportError:
            pass

        # Fall back to RandomForest
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _to_numpy_with_columns(
        self,
        X: Any,
    ) -> tuple[np.ndarray, list[str]]:
        """Convert input to numpy array and extract column names."""
        if HAS_PANDAS and isinstance(X, pd.DataFrame):
            return X.values, list(X.columns)

        if HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            return X.to_numpy(), list(X.columns)

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)
        columns = [f"feature_{i}" for i in range(X_arr.shape[1])]
        return X_arr, columns

    def check_drift(
        self,
        X_train: Any,
        X_test: Any,
    ) -> AdversarialValidationResult:
        """Check for distribution drift between train and test data.

        Parameters
        ----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training features.
        X_test : array-like of shape (n_test_samples, n_features)
            Test features.

        Returns
        -------
        AdversarialValidationResult
            Result containing:
            - auc_score: float (>0.5 indicates drift)
            - drifted_features: List[str] (features with high importance)
            - feature_importances: Dict[str, float]
            - drift_severity: str ('none', 'mild', 'moderate', 'severe')
        """
        X_train_arr, columns = self._to_numpy_with_columns(X_train)
        X_test_arr, _ = self._to_numpy_with_columns(X_test)

        # Sample if requested
        if self.sample_frac < 1.0:
            rng = np.random.RandomState(self.random_state)

            n_train = int(len(X_train_arr) * self.sample_frac)
            n_test = int(len(X_test_arr) * self.sample_frac)

            train_idx = rng.choice(len(X_train_arr), n_train, replace=False)
            test_idx = rng.choice(len(X_test_arr), n_test, replace=False)

            X_train_arr = X_train_arr[train_idx]
            X_test_arr = X_test_arr[test_idx]

        # Create adversarial labels: 0 = train, 1 = test
        X_combined = np.vstack([X_train_arr, X_test_arr])
        y_combined = np.concatenate([
            np.zeros(len(X_train_arr)),
            np.ones(len(X_test_arr)),
        ])

        # Handle NaN/inf values
        X_combined = np.nan_to_num(X_combined, nan=0.0, posinf=0.0, neginf=0.0)

        # Get estimator
        self._estimator = self.estimator or self._get_default_estimator()
        if self.estimator is not None:
            self._estimator = clone(self._estimator)

        # Cross-validate to get OOF predictions
        cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        self._log(f"Running adversarial validation with {self.cv}-fold CV...")

        try:
            oof_proba = cross_val_predict(
                self._estimator,
                X_combined,
                y_combined,
                cv=cv,
                method="predict_proba",
            )[:, 1]
        except Exception as e:
            self._log(f"predict_proba failed: {e}, using predict", level="warn")
            oof_proba = cross_val_predict(
                self._estimator,
                X_combined,
                y_combined,
                cv=cv,
                method="predict",
            )

        # Compute AUC
        from sklearn.metrics import roc_auc_score
        self.auc_score_ = roc_auc_score(y_combined, oof_proba)

        self._log(f"Adversarial validation AUC: {self.auc_score_:.4f}")

        # Fit on full data to get feature importances
        self._estimator.fit(X_combined, y_combined)

        # Extract feature importances
        if hasattr(self._estimator, "feature_importances_"):
            importances = self._estimator.feature_importances_
        elif hasattr(self._estimator, "coef_"):
            importances = np.abs(self._estimator.coef_).ravel()
        else:
            # Fall back to permutation importance
            from sklearn.inspection import permutation_importance
            perm_result = permutation_importance(
                self._estimator,
                X_combined,
                y_combined,
                n_repeats=5,
                random_state=self.random_state,
            )
            importances = perm_result.importances_mean

        # Normalize importances
        if importances.sum() > 0:
            importances = importances / importances.sum()

        self.feature_importances_ = dict(zip(columns, importances))
        self.drifted_features_ = sorted(
            columns,
            key=lambda x: self.feature_importances_[x],
            reverse=True,
        )

        result = AdversarialValidationResult.from_auc(
            auc=self.auc_score_,
            importances=self.feature_importances_,
        )

        self._log(f"Drift severity: {result.drift_severity}")
        if result.drift_severity != "none":
            top_features = result.drifted_features[:5]
            self._log(f"Top drifting features: {top_features}")

        self._is_fitted = True
        return result

    def get_test_like_samples(
        self,
        X_train: Any,
        y_train: Any,
        X_test: Any,
        top_pct: float = 0.2,
    ) -> tuple[Any, Any]:
        """Get training samples most similar to test distribution.

        Uses adversarial validation predictions to identify training samples
        that the classifier thinks look like test samples.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        X_test : array-like
            Test features.
        top_pct : float, default=0.2
            Top percentage of test-like samples to return.

        Returns
        -------
        X_selected : array-like
            Selected training features.
        y_selected : array-like
            Selected training labels.
        """
        # Run adversarial validation if not already done
        if not self._is_fitted or self._estimator is None:
            self.check_drift(X_train, X_test)

        X_train_arr, _ = self._to_numpy_with_columns(X_train)
        y_train_arr = np.asarray(y_train)

        # Handle NaN/inf values
        X_train_arr_clean = np.nan_to_num(X_train_arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Predict probability of being "test-like"
        proba = self._estimator.predict_proba(X_train_arr_clean)[:, 1]

        # Select top samples
        n_select = int(len(X_train_arr) * top_pct)
        top_indices = np.argsort(proba)[-n_select:]

        # Return in original format if possible
        if HAS_PANDAS and isinstance(X_train, pd.DataFrame):
            return X_train.iloc[top_indices], y_train_arr[top_indices]

        if HAS_POLARS and isinstance(X_train, pl.DataFrame):
            return X_train[top_indices], y_train_arr[top_indices]

        return X_train_arr[top_indices], y_train_arr[top_indices]

    def suggest_features_to_drop(
        self,
        X_train: Any,
        X_test: Any,
        max_features: int = 10,
        min_importance: float = 0.05,
    ) -> list[str]:
        """Suggest features to drop to reduce drift.

        Parameters
        ----------
        X_train : array-like
            Training features.
        X_test : array-like
            Test features.
        max_features : int, default=10
            Maximum number of features to suggest.
        min_importance : float, default=0.05
            Minimum importance threshold.

        Returns
        -------
        List[str]
            Features suggested for removal.
        """
        if not self._is_fitted:
            self.check_drift(X_train, X_test)

        suggestions = []
        for feature in self.drifted_features_:
            if len(suggestions) >= max_features:
                break
            if self.feature_importances_[feature] >= min_importance:
                suggestions.append(feature)

        return suggestions
