from __future__ import annotations

"""Linear models for classification and regression.

Linear models provide a fundamentally different inductive bias from
tree-based and neural network models:
- Global linear decision boundaries
- Strong regularization prevents overfitting
- Fast training and inference
- Feature importance via coefficients

These characteristics make linear models valuable for ensemble diversity.

References
----------
- Ridge: Hoerl & Kennard, "Ridge Regression: Biased Estimation" (1970)
- Logistic: Cox, "The Regression Analysis of Binary Sequences" (1958)
- sklearn.linear_model documentation
"""

from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LogisticRegression,
    Ridge,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler

from endgame.core.glassbox import GlassboxMixin


class LinearClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """Linear Classifier with competition-tuned defaults.

    Wraps LogisticRegression with automatic feature scaling and
    sensible defaults for competitive ML. Supports both L1, L2,
    and ElasticNet regularization.

    Parameters
    ----------
    penalty : str, default='l2'
        Regularization: 'l1', 'l2', 'elasticnet', or 'none'.
    C : float, default=1.0
        Inverse of regularization strength. Smaller values = stronger regularization.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only used when penalty='elasticnet').
    solver : str, default='lbfgs'
        Optimization algorithm. 'saga' required for L1/ElasticNet.
    max_iter : int, default=1000
        Maximum iterations for solver.
    class_weight : str or dict, default='balanced'
        Class weights: 'balanced' adjusts for class imbalance.
    scale_features : bool, default=True
        Whether to standardize features before fitting.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    coef_ : ndarray
        Feature coefficients.
    intercept_ : ndarray
        Intercept term.

    Examples
    --------
    >>> from endgame.models.baselines import LinearClassifier
    >>> clf = LinearClassifier(penalty='l2', C=1.0)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    Linear classifiers are different from tree-based models because:
    1. Global decision boundary - same coefficients for all regions
    2. Monotonic feature relationships
    3. Implicit feature selection with L1 penalty
    4. Well-calibrated probabilities (especially with Platt scaling)

    The class_weight='balanced' default helps with imbalanced datasets.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet", "none"] = "l2",
        C: float = 1.0,
        l1_ratio: float = 0.5,
        solver: str = "lbfgs",
        max_iter: int = 1000,
        class_weight: str | dict | None = "balanced",
        scale_features: bool = True,
        n_jobs: int = -1,
        random_state: int | None = None,
    ):
        self.penalty = penalty
        self.C = C
        self.l1_ratio = l1_ratio
        self.solver = solver
        self.max_iter = max_iter
        self.class_weight = class_weight
        self.scale_features = scale_features
        self.n_jobs = n_jobs
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: LogisticRegression | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, sample_weight=None, **fit_params) -> LinearClassifier:
        """Fit the linear classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Scale features
        if self.scale_features:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean

        # Determine solver based on penalty
        solver = self.solver
        if self.penalty in ("l1", "elasticnet") and solver not in ("saga", "liblinear"):
            solver = "saga"

        # Handle penalty=None for sklearn compatibility
        penalty = self.penalty if self.penalty != "none" else None

        # Create and fit model
        self.model_ = LogisticRegression(
            penalty=penalty,
            C=self.C,
            l1_ratio=self.l1_ratio if self.penalty == "elasticnet" else None,
            solver=solver,
            max_iter=self.max_iter,
            class_weight=self.class_weight,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
        )

        self.model_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        self._is_fitted = True

        return self

    def _preprocess(self, X) -> np.ndarray:
        """Preprocess features for prediction."""
        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.scale_features and self._scaler is not None:
            return self._scaler.transform(X_clean)
        return X_clean

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        y_pred = self.model_.predict(X_proc)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.predict_proba(X_proc)

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.predict_log_proba(X_proc)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.decision_function(X_proc)

    @property
    def coef_(self):
        """Feature coefficients."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")
        return self.model_.coef_

    @property
    def intercept_(self):
        """Intercept term."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")
        return self.model_.intercept_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (absolute value of coefficients)."""
        if not self._is_fitted:
            raise RuntimeError("LinearClassifier has not been fitted.")
        # Average absolute coefficients across classes for multiclass
        return np.mean(np.abs(self.model_.coef_), axis=0)


class LinearRegressor(GlassboxMixin, RegressorMixin, BaseEstimator):
    """Linear Regressor with competition-tuned defaults.

    Wraps Ridge/Lasso/ElasticNet with automatic feature scaling and
    sensible defaults for competitive ML.

    Parameters
    ----------
    penalty : str, default='l2'
        Regularization: 'l1' (Lasso), 'l2' (Ridge), 'elasticnet'.
    alpha : float, default=1.0
        Regularization strength. Larger values = stronger regularization.
    l1_ratio : float, default=0.5
        ElasticNet mixing parameter (only used when penalty='elasticnet').
    max_iter : int, default=1000
        Maximum iterations for solver (only for L1/ElasticNet).
    scale_features : bool, default=True
        Whether to standardize features before fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    coef_ : ndarray
        Feature coefficients.
    intercept_ : float
        Intercept term.

    Examples
    --------
    >>> from endgame.models.baselines import LinearRegressor
    >>> reg = LinearRegressor(penalty='l2', alpha=1.0)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)

    Notes
    -----
    Linear regression provides:
    1. Interpretable coefficients
    2. Fast training and inference
    3. L1 penalty for feature selection
    4. L2 penalty for multicollinearity
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        penalty: Literal["l1", "l2", "elasticnet"] = "l2",
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        max_iter: int = 1000,
        scale_features: bool = True,
        random_state: int | None = None,
    ):
        self.penalty = penalty
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.scale_features = scale_features
        self.random_state = random_state

        self.n_features_in_: int = 0
        self.model_: Any | None = None
        self._scaler: StandardScaler | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, sample_weight=None, **fit_params) -> LinearRegressor:
        """Fit the linear regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        self.n_features_in_ = X.shape[1]

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)
        y_clean = np.nan_to_num(y, nan=0.0)

        # Scale features
        if self.scale_features:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X_clean)
        else:
            X_scaled = X_clean

        # Create model based on penalty
        if self.penalty == "l2":
            self.model_ = Ridge(
                alpha=self.alpha,
                random_state=self.random_state,
            )
        elif self.penalty == "l1":
            self.model_ = Lasso(
                alpha=self.alpha,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )
        else:  # elasticnet
            self.model_ = ElasticNet(
                alpha=self.alpha,
                l1_ratio=self.l1_ratio,
                max_iter=self.max_iter,
                random_state=self.random_state,
            )

        # Fit model (sample_weight only supported by Ridge)
        if self.penalty == "l2" and sample_weight is not None:
            self.model_.fit(X_scaled, y_clean, sample_weight=sample_weight)
        else:
            self.model_.fit(X_scaled, y_clean)

        self._is_fitted = True
        return self

    def _preprocess(self, X) -> np.ndarray:
        """Preprocess features for prediction."""
        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.scale_features and self._scaler is not None:
            return self._scaler.transform(X_clean)
        return X_clean

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("LinearRegressor has not been fitted.")

        X_proc = self._preprocess(X)
        return self.model_.predict(X_proc)

    @property
    def coef_(self):
        """Feature coefficients."""
        if not self._is_fitted:
            raise RuntimeError("LinearRegressor has not been fitted.")
        return self.model_.coef_

    @property
    def intercept_(self):
        """Intercept term."""
        if not self._is_fitted:
            raise RuntimeError("LinearRegressor has not been fitted.")
        return self.model_.intercept_

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (absolute value of coefficients)."""
        if not self._is_fitted:
            raise RuntimeError("LinearRegressor has not been fitted.")
        return np.abs(self.model_.coef_)


def _linear_structure(self, *, is_classifier: bool) -> dict:
    if not self._is_fitted:
        raise RuntimeError(f"{self.__class__.__name__} has not been fitted.")
    feature_names = self._structure_feature_names(self.n_features_in_)
    coef = np.asarray(self.coef_)
    intercept = np.asarray(self.intercept_).ravel().tolist()
    if coef.ndim == 1:
        coefficients = {feature_names[i]: float(c) for i, c in enumerate(coef)}
    else:
        coefficients = [
            {feature_names[i]: float(c) for i, c in enumerate(row)}
            for row in coef
        ]
    return {
        "link": "logit" if is_classifier else "identity",
        "coefficients": coefficients,
        "intercept": intercept if len(intercept) > 1 else float(intercept[0]),
        "solver": getattr(self, "solver", None),
        "penalty": getattr(self, "penalty", None),
        "feature_importances": self.feature_importances_.tolist() if np.asarray(self.feature_importances_).ndim == 1 else np.asarray(self.feature_importances_).tolist(),
    }


LinearClassifier._structure_type = "linear"
LinearClassifier._structure_content = lambda self: _linear_structure(self, is_classifier=True)
LinearRegressor._structure_type = "linear"
LinearRegressor._structure_content = lambda self: _linear_structure(self, is_classifier=False)
