from __future__ import annotations

"""Sequential feature selection (forward/backward)."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SequentialSelector(TransformerMixin, BaseEstimator):
    """Sequential feature selection.

    Implements forward selection, backward elimination, or bidirectional
    search for optimal feature subsets.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Model to use for evaluation. Default: LogisticRegression.

    n_features : int, float, or 'auto', default='auto'
        Number of features to select:
        - If int, select that many features.
        - If float (0-1), select that fraction.
        - If 'auto', use cross-validation to find optimal.

    direction : str, default='forward'
        Search direction:
        - 'forward': Start empty, add features
        - 'backward': Start full, remove features
        - 'bidirectional': Both directions (floating)

    scoring : str, optional
        Scoring metric.

    cv : int, default=5
        Cross-validation folds.

    tol : float, optional
        Tolerance for early stopping (only for sklearn >= 1.1).

    n_jobs : int, default=None
        Number of parallel jobs.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    selected_features_ : ndarray
        Indices of selected features.

    n_features_ : int
        Number of selected features.

    scores_ : dict
        Cross-validation scores at each step.

    Example
    -------
    >>> from endgame.feature_selection import SequentialSelector
    >>> selector = SequentialSelector(n_features=10, direction='forward')
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_features: int | float | str = "auto",
        direction: Literal["forward", "backward", "bidirectional"] = "forward",
        scoring: str | None = None,
        cv: int = 5,
        tol: float | None = None,
        n_jobs: int | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.direction = direction
        self.scoring = scoring
        self.cv = cv
        self.tol = tol
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _get_estimator(self):
        """Get the estimator to use."""
        if self.estimator is not None:
            return clone(self.estimator)

        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        """Fit the sequential selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SequentialSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        estimator = self._get_estimator()

        if self.direction == "bidirectional":
            # Implement floating search
            self._floating_search(X, y, estimator)
        else:
            # Use sklearn's SequentialFeatureSelector
            if self.n_features == "auto":
                n_select = "auto"
            elif isinstance(self.n_features, float):
                n_select = max(1, int(n_features * self.n_features))
            else:
                n_select = self.n_features

            kwargs = {
                "estimator": estimator,
                "n_features_to_select": n_select,
                "direction": self.direction,
                "scoring": self.scoring,
                "cv": self.cv,
                "n_jobs": self.n_jobs,
            }

            # Add tol if supported (sklearn >= 1.1)
            if self.tol is not None:
                kwargs["tol"] = self.tol

            self._selector = SequentialFeatureSelector(**kwargs)
            self._selector.fit(X, y)

            self._support_mask = self._selector.support_
            self.selected_features_ = np.where(self._support_mask)[0]
            self.n_features_ = len(self.selected_features_)

        return self

    def _floating_search(self, X, y, estimator):
        """Bidirectional floating search."""
        n_features = X.shape[1]

        # Determine target number of features
        if self.n_features == "auto":
            n_target = n_features // 2
        elif isinstance(self.n_features, float):
            n_target = max(1, int(n_features * self.n_features))
        else:
            n_target = self.n_features

        # Initialize
        selected = set()
        available = set(range(n_features))
        best_score = -np.inf
        self.scores_ = {}

        while len(selected) < n_target:
            # Forward step: add best feature
            best_feature = None
            best_forward_score = -np.inf

            for f in available:
                test_set = list(selected | {f})
                score = np.mean(cross_val_score(
                    estimator, X[:, test_set], y,
                    cv=self.cv, scoring=self.scoring
                ))
                if score > best_forward_score:
                    best_forward_score = score
                    best_feature = f

            if best_feature is not None:
                selected.add(best_feature)
                available.remove(best_feature)
                best_score = best_forward_score
                self.scores_[len(selected)] = best_score

                if self.verbose:
                    print(f"Added feature {best_feature}, score={best_score:.4f}")

            # Backward step: try removing each selected feature
            if len(selected) > 1:
                improved = True
                while improved and len(selected) > 1:
                    improved = False
                    worst_feature = None
                    best_remove_score = -np.inf

                    for f in selected:
                        test_set = list(selected - {f})
                        score = np.mean(cross_val_score(
                            estimator, X[:, test_set], y,
                            cv=self.cv, scoring=self.scoring
                        ))
                        if score > best_remove_score:
                            best_remove_score = score
                            worst_feature = f

                    # Remove if it improves score and won't go below target
                    if (best_remove_score > best_score and
                        len(selected) > n_target // 2):
                        selected.remove(worst_feature)
                        available.add(worst_feature)
                        best_score = best_remove_score
                        improved = True

                        if self.verbose:
                            print(
                                f"Removed feature {worst_feature}, "
                                f"score={best_score:.4f}"
                            )

        self.selected_features_ = np.array(sorted(selected))
        self.n_features_ = len(self.selected_features_)
        self._support_mask = np.isin(np.arange(n_features), self.selected_features_)

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
