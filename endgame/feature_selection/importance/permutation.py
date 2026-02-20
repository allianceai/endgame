"""Permutation importance-based feature selection."""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.inspection import permutation_importance
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class PermutationSelector(TransformerMixin, BaseEstimator):
    """Feature selection based on permutation importance.

    More reliable than model-specific importances as it measures
    actual predictive contribution. Can optionally compute p-values
    (PIMP) for statistical significance.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted model or model to fit.

    n_features : int or float, default=10
        Number of features to select:
        - If int, select that many features.
        - If float (0-1), select that fraction.

    n_repeats : int, default=10
        Number of permutation repetitions.

    scoring : str, optional
        Scoring metric.

    threshold : float, optional
        Minimum importance threshold. If set, overrides n_features.

    use_pimp : bool, default=False
        Whether to compute p-values using PIMP (permutation importance
        with p-values). More statistically rigorous.

    alpha : float, default=0.05
        Significance level for PIMP.

    random_state : int, optional
        Random seed.

    n_jobs : int, default=None
        Number of parallel jobs.

    Attributes
    ----------
    feature_importances_ : ndarray
        Permutation importance for each feature.

    importance_std_ : ndarray
        Standard deviation of importance across repeats.

    pvalues_ : ndarray
        P-values for each feature (if use_pimp=True).

    selected_features_ : ndarray
        Indices of selected features.

    Example
    -------
    >>> from endgame.feature_selection import PermutationSelector
    >>> model.fit(X, y)
    >>> selector = PermutationSelector(estimator=model, n_features=20)
    >>> X_selected = selector.fit_transform(X, y)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        n_features: int | float = 10,
        n_repeats: int = 10,
        scoring: str | None = None,
        threshold: float | None = None,
        use_pimp: bool = False,
        alpha: float = 0.05,
        random_state: int | None = None,
        n_jobs: int | None = None,
    ):
        self.estimator = estimator
        self.n_features = n_features
        self.n_repeats = n_repeats
        self.scoring = scoring
        self.threshold = threshold
        self.use_pimp = use_pimp
        self.alpha = alpha
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """Fit the permutation selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (should be validation set for fitted model).
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PermutationSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Compute permutation importance
        result = permutation_importance(
            self.estimator,
            X, y,
            n_repeats=self.n_repeats,
            scoring=self.scoring,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self.feature_importances_ = result.importances_mean
        self.importance_std_ = result.importances_std
        self._importances_raw = result.importances

        # PIMP: compute p-values
        if self.use_pimp:
            self.pvalues_ = self._compute_pimp_pvalues(X, y)
        else:
            self.pvalues_ = None

        # Select features
        if self.threshold is not None:
            # Select by threshold
            if self.pvalues_ is not None:
                mask = self.pvalues_ < self.alpha
            else:
                mask = self.feature_importances_ >= self.threshold
            self.selected_features_ = np.where(mask)[0]
        else:
            # Select top n
            if isinstance(self.n_features, float):
                n_select = max(1, int(n_features * self.n_features))
            else:
                n_select = min(self.n_features, n_features)

            # Sort by importance
            ranking = np.argsort(self.feature_importances_)[::-1]
            self.selected_features_ = ranking[:n_select]

        self._support_mask = np.isin(np.arange(n_features), self.selected_features_)
        self.n_features_ = len(self.selected_features_)

        return self

    def _compute_pimp_pvalues(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Compute p-values using PIMP methodology.

        PIMP (Permutation Importance with P-values) computes a null
        distribution by permuting the target and comparing actual
        importances to this distribution.
        """
        n_features = X.shape[1]
        rng = np.random.RandomState(self.random_state)

        # Null distribution: permute y and compute importances
        n_null = 100  # Number of null permutations
        null_importances = np.zeros((n_null, n_features))

        for i in range(n_null):
            y_perm = rng.permutation(y)

            # Fit new model on permuted target
            model = clone(self.estimator)
            model.fit(X, y_perm)

            # Compute importance
            result = permutation_importance(
                model, X, y_perm,
                n_repeats=5,
                scoring=self.scoring,
                random_state=self.random_state,
            )
            null_importances[i] = result.importances_mean

        # Compute p-values
        pvalues = np.zeros(n_features)
        for j in range(n_features):
            # P-value: fraction of null >= actual
            pvalues[j] = np.mean(
                null_importances[:, j] >= self.feature_importances_[j]
            )

        return pvalues

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
