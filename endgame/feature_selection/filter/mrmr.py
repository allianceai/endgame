"""Minimum Redundancy Maximum Relevance (MRMR) feature selection."""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MRMRSelector(TransformerMixin, BaseEstimator):
    """Minimum Redundancy Maximum Relevance feature selection.

    MRMR balances feature relevance (high mutual information with target)
    with redundancy (low mutual information with already-selected features).

    The selection criterion is: max(relevance - redundancy)

    Parameters
    ----------
    n_features : int, default=10
        Number of features to select.

    task : str, default='classification'
        Task type: 'classification' or 'regression'.

    relevance_func : str, default='mutual_info'
        Function for computing relevance:
        - 'mutual_info': Mutual information
        - 'f_test': F-statistic

    redundancy_func : str, default='pearson'
        Function for computing redundancy:
        - 'pearson': Absolute Pearson correlation
        - 'mutual_info': Mutual information between features

    n_neighbors : int, default=3
        Number of neighbors for MI estimation.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Whether to print selection progress.

    Attributes
    ----------
    selected_features_ : ndarray
        Indices of selected features in order of selection.

    relevance_scores_ : ndarray
        Relevance scores for all features.

    ranking_ : ndarray
        Full feature ranking.

    Example
    -------
    >>> from endgame.feature_selection import MRMRSelector
    >>> selector = MRMRSelector(n_features=20)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(f"Selected features: {selector.selected_features_}")
    """

    def __init__(
        self,
        n_features: int = 10,
        task: Literal["classification", "regression"] = "classification",
        relevance_func: Literal["mutual_info", "f_test"] = "mutual_info",
        redundancy_func: Literal["pearson", "mutual_info"] = "pearson",
        n_neighbors: int = 3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_features = n_features
        self.task = task
        self.relevance_func = relevance_func
        self.redundancy_func = redundancy_func
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.verbose = verbose

    def _compute_relevance(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute relevance scores (feature-target dependency)."""
        if self.relevance_func == "mutual_info":
            if self.task == "classification":
                return mutual_info_classif(
                    X, y,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )
            else:
                return mutual_info_regression(
                    X, y,
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )
        elif self.relevance_func == "f_test":
            from sklearn.feature_selection import f_classif, f_regression
            if self.task == "classification":
                scores, _ = f_classif(X, y)
            else:
                scores, _ = f_regression(X, y)
            # Normalize to [0, 1]
            scores = np.nan_to_num(scores, nan=0.0)
            if scores.max() > 0:
                scores = scores / scores.max()
            return scores
        else:
            raise ValueError(f"Unknown relevance_func: {self.relevance_func}")

    def _compute_redundancy(
        self, X: np.ndarray, selected_idx: list[int], candidate_idx: int
    ) -> float:
        """Compute redundancy between candidate and selected features."""
        if not selected_idx:
            return 0.0

        candidate = X[:, candidate_idx]

        if self.redundancy_func == "pearson":
            redundancies = []
            for idx in selected_idx:
                corr = np.abs(np.corrcoef(candidate, X[:, idx])[0, 1])
                redundancies.append(corr if not np.isnan(corr) else 0.0)
            return np.mean(redundancies)

        elif self.redundancy_func == "mutual_info":
            # Compute MI between candidate and each selected feature
            from sklearn.feature_selection import mutual_info_regression
            redundancies = []
            for idx in selected_idx:
                mi = mutual_info_regression(
                    candidate.reshape(-1, 1),
                    X[:, idx],
                    n_neighbors=self.n_neighbors,
                    random_state=self.random_state,
                )[0]
                redundancies.append(mi)
            return np.mean(redundancies)

        else:
            raise ValueError(f"Unknown redundancy_func: {self.redundancy_func}")

    def fit(self, X, y):
        """Fit the MRMR selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : MRMRSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Compute relevance for all features
        self.relevance_scores_ = self._compute_relevance(X, y)

        # Greedy forward selection with MRMR criterion
        selected = []
        remaining = list(range(n_features))
        ranking = []

        # Select features one by one
        n_to_select = min(self.n_features, n_features)

        for i in range(n_to_select):
            best_score = -np.inf
            best_idx = None

            for idx in remaining:
                relevance = self.relevance_scores_[idx]
                redundancy = self._compute_redundancy(X, selected, idx)

                # MRMR criterion: maximize (relevance - redundancy)
                score = relevance - redundancy

                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                ranking.append(best_idx)

                if self.verbose:
                    print(
                        f"Selected feature {best_idx} "
                        f"(relevance={self.relevance_scores_[best_idx]:.4f}, "
                        f"score={best_score:.4f})"
                    )

        # Add remaining features to ranking (by relevance)
        remaining_sorted = sorted(
            remaining, key=lambda x: self.relevance_scores_[x], reverse=True
        )
        ranking.extend(remaining_sorted)

        self.selected_features_ = np.array(selected)
        self.ranking_ = np.array(ranking)
        self._support_mask = np.isin(np.arange(n_features), selected)

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
        """Get complete feature ranking."""
        check_is_fitted(self, "ranking_")
        return self.ranking_
