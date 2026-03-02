from __future__ import annotations

"""Univariate statistical feature selection methods."""

from collections.abc import Callable
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import (
    SelectKBest,
    SelectPercentile,
    chi2,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression,
)
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class UnivariateSelector(TransformerMixin, BaseEstimator):
    """Unified univariate feature selection.

    Selects features based on univariate statistical tests. Supports
    various scoring functions and selection modes.

    Parameters
    ----------
    score_func : str or callable, default='f_classif'
        Scoring function. Options:
        - 'f_classif': ANOVA F-test for classification
        - 'f_regression': F-test for regression
        - 'mutual_info_classif': Mutual information for classification
        - 'mutual_info_regression': Mutual information for regression
        - 'chi2': Chi-squared test (requires non-negative features)
        - callable: Custom function (X, y) -> (scores, pvalues)

    mode : str, default='k_best'
        Selection mode:
        - 'k_best': Select top k features
        - 'percentile': Select top percentile
        - 'fpr': Select by false positive rate
        - 'fdr': Select by false discovery rate
        - 'fwe': Select by family-wise error

    k : int, default=10
        Number of features to select (for k_best mode).

    percentile : int, default=10
        Percentile of features to select (for percentile mode).

    alpha : float, default=0.05
        Threshold for fpr/fdr/fwe modes.

    random_state : int, optional
        Random seed for mutual information estimation.

    Attributes
    ----------
    scores_ : ndarray
        Scores for each feature.

    pvalues_ : ndarray
        P-values for each feature (if available).

    selected_features_ : ndarray
        Indices of selected features.

    Example
    -------
    >>> from endgame.feature_selection import UnivariateSelector
    >>> selector = UnivariateSelector(score_func='mutual_info_classif', k=20)
    >>> X_selected = selector.fit_transform(X, y)
    """

    SCORE_FUNCS = {
        "f_classif": f_classif,
        "f_regression": f_regression,
        "mutual_info_classif": mutual_info_classif,
        "mutual_info_regression": mutual_info_regression,
        "chi2": chi2,
    }

    def __init__(
        self,
        score_func: str | Callable = "f_classif",
        mode: Literal["k_best", "percentile", "fpr", "fdr", "fwe"] = "k_best",
        k: int = 10,
        percentile: int = 10,
        alpha: float = 0.05,
        random_state: int | None = None,
    ):
        self.score_func = score_func
        self.mode = mode
        self.k = k
        self.percentile = percentile
        self.alpha = alpha
        self.random_state = random_state

    def _get_score_func(self) -> Callable:
        """Get the scoring function."""
        if callable(self.score_func):
            return self.score_func

        if self.score_func not in self.SCORE_FUNCS:
            raise ValueError(
                f"Unknown score_func: {self.score_func}. "
                f"Options: {list(self.SCORE_FUNCS.keys())}"
            )

        func = self.SCORE_FUNCS[self.score_func]

        # Wrap mutual information functions to accept random_state
        if "mutual_info" in self.score_func:
            original_func = func
            def wrapped_func(X, y):
                return original_func(X, y, random_state=self.random_state)
            return wrapped_func

        return func

    def fit(self, X, y):
        """Fit the selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : UnivariateSelector
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        score_func = self._get_score_func()

        if self.mode == "k_best":
            self._selector = SelectKBest(score_func=score_func, k=self.k)
        elif self.mode == "percentile":
            self._selector = SelectPercentile(
                score_func=score_func, percentile=self.percentile
            )
        else:
            # Use SelectKBest and filter by p-value threshold
            self._selector = SelectKBest(score_func=score_func, k="all")

        self._selector.fit(X, y)

        self.scores_ = self._selector.scores_
        self.pvalues_ = getattr(self._selector, "pvalues_", None)

        # For fpr/fdr/fwe modes, select based on p-values
        if self.mode in ("fpr", "fdr", "fwe") and self.pvalues_ is not None:
            if self.mode == "fpr":
                mask = self.pvalues_ < self.alpha
            elif self.mode == "fdr":
                # Benjamini-Hochberg procedure
                mask = self._benjamini_hochberg(self.pvalues_, self.alpha)
            else:  # fwe
                mask = self.pvalues_ < (self.alpha / len(self.pvalues_))
            self._custom_mask = mask
        else:
            self._custom_mask = None

        self.selected_features_ = self.get_support(indices=True)

        return self

    def _benjamini_hochberg(self, pvalues: np.ndarray, alpha: float) -> np.ndarray:
        """Benjamini-Hochberg procedure for FDR control."""
        n = len(pvalues)
        sorted_idx = np.argsort(pvalues)
        sorted_pvals = pvalues[sorted_idx]

        # Calculate BH threshold
        thresholds = alpha * np.arange(1, n + 1) / n
        mask = sorted_pvals <= thresholds

        # Find the largest k where p_(k) <= k*alpha/n
        if mask.any():
            k = np.max(np.where(mask)[0]) + 1
            selected = np.zeros(n, dtype=bool)
            selected[sorted_idx[:k]] = True
            return selected
        return np.zeros(n, dtype=bool)

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
        check_is_fitted(self, "_selector")
        X = check_array(X)

        if self._custom_mask is not None:
            return X[:, self._custom_mask]
        return self._selector.transform(X)

    def fit_transform(self, X, y) -> np.ndarray:
        """Fit and transform."""
        self.fit(X, y)
        return self.transform(X)

    def get_support(self, indices: bool = False) -> np.ndarray:
        """Get mask or indices of selected features."""
        check_is_fitted(self, "_selector")

        if self._custom_mask is not None:
            mask = self._custom_mask
        else:
            mask = self._selector.get_support()

        if indices:
            return np.where(mask)[0]
        return mask

    def get_feature_scores(self) -> np.ndarray:
        """Get feature scores."""
        check_is_fitted(self, "scores_")
        return self.scores_


class MutualInfoSelector(UnivariateSelector):
    """Mutual information-based feature selection.

    Convenience wrapper for mutual information scoring, which captures
    nonlinear dependencies.

    Parameters
    ----------
    k : int, default=10
        Number of features to select.

    task : str, default='classification'
        Task type: 'classification' or 'regression'.

    n_neighbors : int, default=3
        Number of neighbors for MI estimation.

    random_state : int, optional
        Random seed.

    Example
    -------
    >>> selector = MutualInfoSelector(k=20, task='classification')
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        k: int = 10,
        task: Literal["classification", "regression"] = "classification",
        n_neighbors: int = 3,
        random_state: int | None = None,
    ):
        score_func = (
            "mutual_info_classif" if task == "classification"
            else "mutual_info_regression"
        )
        super().__init__(
            score_func=score_func,
            mode="k_best",
            k=k,
            random_state=random_state,
        )
        self.task = task
        self.n_neighbors = n_neighbors


class FTestSelector(UnivariateSelector):
    """F-test based feature selection.

    Uses ANOVA F-test for classification or F-regression for regression.
    Fast linear baseline.

    Parameters
    ----------
    k : int, default=10
        Number of features to select.

    task : str, default='classification'
        Task type: 'classification' or 'regression'.

    Example
    -------
    >>> selector = FTestSelector(k=20)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        k: int = 10,
        task: Literal["classification", "regression"] = "classification",
    ):
        score_func = "f_classif" if task == "classification" else "f_regression"
        super().__init__(
            score_func=score_func,
            mode="k_best",
            k=k,
        )
        self.task = task


class Chi2Selector(UnivariateSelector):
    """Chi-squared feature selection.

    For categorical features vs categorical target. Requires non-negative
    feature values.

    Parameters
    ----------
    k : int, default=10
        Number of features to select.

    Example
    -------
    >>> selector = Chi2Selector(k=20)
    >>> X_selected = selector.fit_transform(X_categorical, y)
    """

    def __init__(self, k: int = 10):
        super().__init__(
            score_func="chi2",
            mode="k_best",
            k=k,
        )
