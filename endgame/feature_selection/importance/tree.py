from __future__ import annotations

"""Tree-based feature importance selection."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class TreeImportanceSelector(TransformerMixin, BaseEstimator):
    """Feature selection based on tree-based importance.

    Uses Gini/entropy importance from tree-based models. Fast but
    can be biased toward high-cardinality features.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Tree-based model with `feature_importances_`.
        Default: RandomForestClassifier.

    n_features : int, float, or str, default='mean'
        Number of features to select:
        - If int, select that many features.
        - If float (0-1), select that fraction.
        - If 'mean', select features with importance > mean.
        - If 'median', select features with importance > median.

    importance_type : str, default='native'
        Type of importance to use:
        - 'native': Use model's feature_importances_
        - 'gain': Gain-based (LightGBM/XGBoost specific)
        - 'split': Split count-based

    threshold : float, optional
        Explicit importance threshold.

    prefit : bool, default=False
        Whether the estimator is already fitted.

    random_state : int, optional
        Random seed.

    Attributes
    ----------
    feature_importances_ : ndarray
        Feature importance scores.

    selected_features_ : ndarray
        Indices of selected features.

    threshold_ : float
        Actual threshold used for selection.

    Example
    -------
    >>> from endgame.feature_selection import TreeImportanceSelector
    >>> selector = TreeImportanceSelector(n_features=20)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_features: int | float | str = "mean",
        importance_type: Literal["native", "gain", "split"] = "native",
        threshold: float | None = None,
        prefit: bool = False,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.importance_type = importance_type
        self.threshold = threshold
        self.prefit = prefit
        self.random_state = random_state

    def _get_estimator(self):
        """Get the estimator to use."""
        if self.estimator is not None:
            if self.prefit:
                return self.estimator
            return clone(self.estimator)

        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100,
                importance_type="gain" if self.importance_type == "gain" else "split",
                verbosity=-1,
                n_jobs=-1,
                random_state=self.random_state,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                n_jobs=-1,
                random_state=self.random_state,
            )

    def fit(self, X, y):
        """Fit the tree importance selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : TreeImportanceSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Get and fit estimator
        estimator = self._get_estimator()

        if not self.prefit:
            estimator.fit(X, y)

        self.estimator_ = estimator

        # Get feature importances
        if self.importance_type == "native" or not hasattr(estimator, "booster_"):
            self.feature_importances_ = estimator.feature_importances_
        else:
            # LightGBM/XGBoost specific
            booster = estimator.booster_
            if hasattr(booster, "feature_importance"):
                self.feature_importances_ = booster.feature_importance(
                    importance_type=self.importance_type
                )
            else:
                self.feature_importances_ = estimator.feature_importances_

        # Normalize
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ = (
                self.feature_importances_ / self.feature_importances_.sum()
            )

        # Determine threshold
        if self.threshold is not None:
            self.threshold_ = self.threshold
        elif self.n_features == "mean":
            self.threshold_ = np.mean(self.feature_importances_)
        elif self.n_features == "median":
            self.threshold_ = np.median(self.feature_importances_)
        elif isinstance(self.n_features, float):
            n_select = max(1, int(n_features * self.n_features))
            sorted_imp = np.sort(self.feature_importances_)[::-1]
            self.threshold_ = sorted_imp[n_select - 1] if n_select <= n_features else 0
        else:  # int
            n_select = min(self.n_features, n_features)
            sorted_imp = np.sort(self.feature_importances_)[::-1]
            self.threshold_ = sorted_imp[n_select - 1] if n_select <= n_features else 0

        # Select features
        self._support_mask = self.feature_importances_ >= self.threshold_
        self.selected_features_ = np.where(self._support_mask)[0]
        self.n_features_ = len(self.selected_features_)

        # Ensure at least one feature
        if self.n_features_ == 0:
            best_idx = np.argmax(self.feature_importances_)
            self._support_mask[best_idx] = True
            self.selected_features_ = np.array([best_idx])
            self.n_features_ = 1

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

    def fit_transform(self, X, y) -> np.ndarray:
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
        """Get feature ranking by importance."""
        check_is_fitted(self, "feature_importances_")
        return np.argsort(self.feature_importances_)[::-1]
