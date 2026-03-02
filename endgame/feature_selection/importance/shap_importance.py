from __future__ import annotations

"""SHAP-based feature selection."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    shap = None


def _check_shap_installed():
    """Raise ImportError if shap is not installed."""
    if not HAS_SHAP:
        raise ImportError(
            "The 'shap' package is required for SHAPSelector. "
            "Install it with: pip install shap"
        )


class SHAPSelector(TransformerMixin, BaseEstimator):
    """Feature selection based on SHAP values.

    Uses mean absolute SHAP values as feature importance. More
    theoretically grounded than permutation importance.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted model.

    n_features : int or float, default=10
        Number of features to select:
        - If int, select that many features.
        - If float (0-1), select that fraction.

    explainer_type : str, default='auto'
        Type of SHAP explainer:
        - 'auto': Auto-detect based on model type
        - 'tree': TreeExplainer (fast for tree models)
        - 'linear': LinearExplainer
        - 'kernel': KernelExplainer (model-agnostic, slow)
        - 'deep': DeepExplainer (for neural networks)

    background_samples : int, default=100
        Number of background samples for KernelExplainer.

    max_samples : int, optional
        Maximum samples to use for SHAP computation.

    check_additivity : bool, default=False
        Whether to verify SHAP additivity (slower).

    random_state : int, optional
        Random seed.

    Attributes
    ----------
    shap_values_ : ndarray
        SHAP values for each sample and feature.

    feature_importances_ : ndarray
        Mean absolute SHAP values for each feature.

    selected_features_ : ndarray
        Indices of selected features.

    Example
    -------
    >>> from endgame.feature_selection import SHAPSelector
    >>> model.fit(X_train, y_train)
    >>> selector = SHAPSelector(estimator=model, n_features=20)
    >>> X_selected = selector.fit_transform(X_val, y_val)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        n_features: int | float = 10,
        explainer_type: Literal["auto", "tree", "linear", "kernel", "deep"] = "auto",
        background_samples: int = 100,
        max_samples: int | None = None,
        check_additivity: bool = False,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.explainer_type = explainer_type
        self.background_samples = background_samples
        self.max_samples = max_samples
        self.check_additivity = check_additivity
        self.random_state = random_state

    def _get_explainer(self, X: np.ndarray):
        """Get the appropriate SHAP explainer."""
        _check_shap_installed()

        model = self.estimator

        if self.explainer_type == "auto":
            # Auto-detect explainer type
            model_name = type(model).__name__.lower()

            if any(name in model_name for name in [
                "lgbm", "xgb", "catboost", "randomforest",
                "gradientboosting", "extratrees", "decisiontree"
            ]):
                return shap.TreeExplainer(model)
            elif any(name in model_name for name in [
                "linear", "logistic", "ridge", "lasso", "elasticnet"
            ]):
                return shap.LinearExplainer(model, X[:self.background_samples])
            else:
                # Fall back to KernelExplainer
                rng = np.random.RandomState(self.random_state)
                idx = rng.choice(
                    len(X), size=min(self.background_samples, len(X)),
                    replace=False
                )
                background = X[idx]
                return shap.KernelExplainer(
                    model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                    background
                )

        elif self.explainer_type == "tree":
            return shap.TreeExplainer(model)

        elif self.explainer_type == "linear":
            return shap.LinearExplainer(model, X[:self.background_samples])

        elif self.explainer_type == "kernel":
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(
                len(X), size=min(self.background_samples, len(X)),
                replace=False
            )
            background = X[idx]
            return shap.KernelExplainer(
                model.predict_proba if hasattr(model, "predict_proba") else model.predict,
                background
            )

        elif self.explainer_type == "deep":
            return shap.DeepExplainer(model, X[:self.background_samples])

        else:
            raise ValueError(f"Unknown explainer_type: {self.explainer_type}")

    def fit(self, X, y=None):
        """Fit the SHAP selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to compute SHAP values on.
        y : Ignored

        Returns
        -------
        self : SHAPSelector
        """
        _check_shap_installed()

        X = check_array(X)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Subsample if needed
        if self.max_samples and n_samples > self.max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(n_samples, size=self.max_samples, replace=False)
            X_shap = X[idx]
        else:
            X_shap = X

        # Get explainer
        explainer = self._get_explainer(X)

        # Compute SHAP values
        shap_values = explainer.shap_values(
            X_shap,
            check_additivity=self.check_additivity
        )

        # Handle multi-class (take absolute values across classes)
        if isinstance(shap_values, list):
            # Multi-class: average across classes
            shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            shap_values = np.abs(shap_values)

        self.shap_values_ = shap_values

        # Feature importance = mean absolute SHAP value
        self.feature_importances_ = np.mean(np.abs(shap_values), axis=0)

        # Select top features
        if isinstance(self.n_features, float):
            n_select = max(1, int(n_features * self.n_features))
        else:
            n_select = min(self.n_features, n_features)

        ranking = np.argsort(self.feature_importances_)[::-1]
        self.selected_features_ = ranking[:n_select]
        self._support_mask = np.isin(np.arange(n_features), self.selected_features_)
        self.n_features_ = len(self.selected_features_)

        return self

    def transform(self, X) -> np.ndarray:
        """Select features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with selected features.
        """
        check_is_fitted(self, "selected_features_")
        X = check_array(X)
        return X[:, self.selected_features_]

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, "_support_mask")
        if indices:
            return self.selected_features_
        return self._support_mask

    def get_feature_ranking(self) -> np.ndarray:
        """Get feature ranking by SHAP importance."""
        check_is_fitted(self, "feature_importances_")
        return np.argsort(self.feature_importances_)[::-1]

    def get_interaction_values(self, X: np.ndarray) -> np.ndarray:
        """Get SHAP interaction values.

        Only available for TreeExplainer.

        Parameters
        ----------
        X : array-like
            Data to compute interactions on.

        Returns
        -------
        interaction_values : ndarray of shape (n_samples, n_features, n_features)
        """
        _check_shap_installed()
        check_is_fitted(self, "feature_importances_")

        X = check_array(X)

        if self.explainer_type not in ("auto", "tree"):
            raise ValueError(
                "Interaction values only available for tree models"
            )

        explainer = shap.TreeExplainer(self.estimator)
        return explainer.shap_interaction_values(X)
