"""Fuzzy-Rough Feature Selection (FRFS).

Uses fuzzy-rough dependency measure for greedy forward feature selection.
The dependency is based on the fuzzy positive region, which measures how
well a set of features approximates the decision classes.

References
----------
Jensen, R., & Shen, Q. (2009). New approaches to fuzzy-rough feature
selection. IEEE Transactions on Fuzzy Systems, 17(4), 824-838.

Example
-------
>>> from endgame.fuzzy.rough.frfs import FuzzyRoughFeatureSelector
>>> selector = FuzzyRoughFeatureSelector(n_features=5)
>>> selector.fit(X_train, y_train)
>>> X_selected = selector.transform(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FuzzyRoughFeatureSelector(BaseEstimator, TransformerMixin):
    """Fuzzy-Rough Feature Selection using dependency measure.

    Performs greedy forward selection by iteratively adding the feature
    that most increases the fuzzy-rough dependency degree (positive region
    size). The dependency measures how well a subset of features
    approximates the decision classes via lower approximations.

    Parameters
    ----------
    n_features : int, default=10
        Maximum number of features to select.
    similarity_metric : str, default='gaussian'
        Similarity metric for fuzzy relations: 'gaussian' or 'linear'.
    sigma : float, default=1.0
        Width parameter for Gaussian similarity.
    threshold : float, default=1e-4
        Minimum improvement in dependency to continue selection.

    Attributes
    ----------
    selected_features_ : ndarray of shape (n_selected,)
        Indices of selected features in order of selection.
    ranking_ : ndarray of shape (n_selected,)
        Dependency gain when each feature was added.
    n_features_in_ : int
        Number of features seen during fit.
    dependency_ : float
        Final dependency degree of the selected feature subset.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.rough.frfs import FuzzyRoughFeatureSelector
    >>> X = np.random.randn(100, 20)
    >>> y = (X[:, 0] + X[:, 3] > 0).astype(int)
    >>> selector = FuzzyRoughFeatureSelector(n_features=5, sigma=1.0)
    >>> selector.fit(X, y)
    FuzzyRoughFeatureSelector(n_features=5)
    >>> X_sel = selector.transform(X)
    >>> X_sel.shape[1] <= 5
    True
    """

    def __init__(
        self,
        n_features: int = 10,
        similarity_metric: str = "gaussian",
        sigma: float = 1.0,
        threshold: float = 1e-4,
    ):
        self.n_features = n_features
        self.similarity_metric = similarity_metric
        self.sigma = sigma
        self.threshold = threshold

    def fit(self, X: Any, y: Any) -> FuzzyRoughFeatureSelector:
        """Fit the feature selector using fuzzy-rough dependency.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted selector.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_total_features = X.shape

        classes = np.unique(y)
        n_select = min(self.n_features, n_total_features)

        # Precompute per-feature similarity matrices
        sim_matrices = []
        for j in range(n_total_features):
            sim_matrices.append(self._feature_similarity(X[:, j]))

        selected: list[int] = []
        ranking: list[float] = []
        remaining = set(range(n_total_features))
        current_dep = 0.0

        for _ in range(n_select):
            best_feat = -1
            best_dep = current_dep

            for feat in remaining:
                candidate = selected + [feat]
                dep = self._compute_dependency(
                    sim_matrices, candidate, y, classes, n_samples
                )
                if dep > best_dep:
                    best_dep = dep
                    best_feat = feat

            if best_feat < 0 or (best_dep - current_dep) < self.threshold:
                break

            ranking.append(best_dep - current_dep)
            selected.append(best_feat)
            remaining.discard(best_feat)
            current_dep = best_dep

            # Early stop if dependency is near-perfect
            if current_dep >= 1.0 - 1e-10:
                break

        self.selected_features_ = np.array(selected, dtype=int)
        self.ranking_ = np.array(ranking, dtype=np.float64)
        self.dependency_ = current_dep
        return self

    def transform(self, X: Any) -> np.ndarray:
        """Select features from X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_selected_features)
            Data with only selected features.
        """
        check_is_fitted(self, ["selected_features_"])
        X = check_array(X)
        return X[:, self.selected_features_]

    def _feature_similarity(self, x_col: np.ndarray) -> np.ndarray:
        """Compute pairwise fuzzy similarity for one feature.

        Parameters
        ----------
        x_col : ndarray of shape (n_samples,)
            Single feature column.

        Returns
        -------
        ndarray of shape (n_samples, n_samples)
            Pairwise similarity matrix.
        """
        diff = np.abs(x_col[:, None] - x_col[None, :])
        if self.similarity_metric == "gaussian":
            return np.exp(-0.5 * (diff / self.sigma) ** 2)
        elif self.similarity_metric == "linear":
            max_d = diff.max()
            if max_d == 0:
                return np.ones_like(diff)
            return 1.0 - diff / max_d
        else:
            raise ValueError(
                f"Unknown similarity_metric: {self.similarity_metric}"
            )

    def _compute_dependency(
        self,
        sim_matrices: list[np.ndarray],
        feature_indices: list[int],
        y: np.ndarray,
        classes: np.ndarray,
        n_samples: int,
    ) -> float:
        """Compute fuzzy-rough dependency for a feature subset.

        The dependency is the mean positive region membership over all samples.
        For each sample, it measures the degree to which the sample belongs
        to the lower approximation of its decision class.

        Parameters
        ----------
        sim_matrices : list of ndarray
            Precomputed per-feature similarity matrices.
        feature_indices : list of int
            Feature indices in the candidate subset.
        y : ndarray of shape (n_samples,)
            Class labels.
        classes : ndarray
            Unique class labels.
        n_samples : int
            Number of samples.

        Returns
        -------
        float
            Dependency degree in [0, 1].
        """
        # Compute combined similarity as t-norm (min) over features
        combined_sim = np.ones((n_samples, n_samples))
        for j in feature_indices:
            np.minimum(combined_sim, sim_matrices[j], out=combined_sim)

        # Compute positive region membership for each sample
        pos_region = np.zeros(n_samples)
        for i in range(n_samples):
            c = y[i]
            # Lower approximation of class c at sample i:
            # min over all j of: (1 - R(i,j)) OR (j in class c)
            # = min over j not in c of: 1 - R(i,j)
            not_c_mask = y != c
            if not np.any(not_c_mask):
                pos_region[i] = 1.0
            else:
                implication = 1.0 - combined_sim[i, not_c_mask]
                pos_region[i] = np.min(implication)

        return float(np.mean(pos_region))
