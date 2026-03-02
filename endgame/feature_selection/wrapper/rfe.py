from __future__ import annotations

"""Recursive Feature Elimination (RFE) selection."""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import RFE, RFECV
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class RFESelector(TransformerMixin, BaseEstimator):
    """Recursive Feature Elimination feature selection.

    RFE iteratively removes the least important features based on
    model coefficients or feature importances.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Model to use for feature ranking. Must have `coef_` or
        `feature_importances_` attribute. Default: RandomForest.

    n_features : int, float, or None, default=None
        Number of features to select:
        - If int, select that many features.
        - If float (0-1), select that fraction of features.
        - If None, use cross-validation to find optimal.

    step : int or float, default=1
        Number of features to remove at each iteration:
        - If int > 1, remove that many features.
        - If float (0-1), remove that fraction.

    cv : int, default=5
        Cross-validation folds (used when n_features=None).

    scoring : str, optional
        Scoring metric for RFECV.

    min_features_to_select : int, default=1
        Minimum features for RFECV.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    selected_features_ : ndarray
        Indices of selected features.

    ranking_ : ndarray
        Feature ranking (1 = selected).

    n_features_ : int
        Number of selected features.

    estimator_ : BaseEstimator
        Fitted estimator used for final ranking.

    Example
    -------
    >>> from endgame.feature_selection import RFESelector
    >>> selector = RFESelector(n_features=20)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_features: int | float | None = None,
        step: int | float = 1,
        cv: int = 5,
        scoring: str | None = None,
        min_features_to_select: int = 1,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.step = step
        self.cv = cv
        self.scoring = scoring
        self.min_features_to_select = min_features_to_select
        self.verbose = verbose

    def _get_estimator(self):
        """Get the estimator to use."""
        if self.estimator is not None:
            return clone(self.estimator)

        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                verbosity=-1,
                n_jobs=-1,
            )
        except ImportError:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                n_jobs=-1,
            )

    def fit(self, X, y):
        """Fit the RFE selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RFESelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        estimator = self._get_estimator()

        # Determine number of features
        if self.n_features is None:
            # Use cross-validation
            self._rfe = RFECV(
                estimator=estimator,
                step=self.step,
                cv=self.cv,
                scoring=self.scoring,
                min_features_to_select=self.min_features_to_select,
                verbose=self.verbose,
            )
        else:
            if isinstance(self.n_features, float):
                n_select = max(1, int(n_features * self.n_features))
            else:
                n_select = self.n_features

            self._rfe = RFE(
                estimator=estimator,
                n_features_to_select=n_select,
                step=self.step,
                verbose=self.verbose,
            )

        self._rfe.fit(X, y)

        self.ranking_ = self._rfe.ranking_
        self.n_features_ = self._rfe.n_features_
        self.selected_features_ = np.where(self._rfe.support_)[0]
        self._support_mask = self._rfe.support_
        self.estimator_ = self._rfe.estimator_

        # Store CV scores if available
        if hasattr(self._rfe, "cv_results_"):
            self.cv_results_ = self._rfe.cv_results_

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
        check_is_fitted(self, "_rfe")
        X = check_array(X)
        return self._rfe.transform(X)

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
        """Get feature ranking (1 = selected)."""
        check_is_fitted(self, "ranking_")
        return self.ranking_
