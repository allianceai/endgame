"""Boruta feature selection algorithm."""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class BorutaSelector(TransformerMixin, BaseEstimator):
    """Boruta all-relevant feature selection algorithm.

    Boruta is a wrapper around Random Forest. It creates "shadow" features
    (shuffled copies of real features) and selects features that have
    significantly higher importance than the best shadow feature.

    This is a statistically principled method that finds ALL relevant
    features, not just the minimal set.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Tree-based model with `feature_importances_`. Default: RandomForest.

    n_estimators : int or 'auto', default='auto'
        Number of trees. 'auto' uses heuristic based on features.

    max_iter : int, default=100
        Maximum iterations.

    alpha : float, default=0.05
        Significance level for the binomial test.

    perc : int, default=100
        Percentile of shadow feature importance distribution to use
        as threshold. 100 = max (original Boruta).

    two_step : bool, default=True
        If True, use two-step correction for multiple testing.

    random_state : int, optional
        Random seed.

    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    selected_features_ : ndarray
        Indices of confirmed features.

    tentative_features_ : ndarray
        Indices of tentative features (borderline).

    rejected_features_ : ndarray
        Indices of rejected features.

    ranking_ : ndarray
        Feature ranking (1 = confirmed, 2 = tentative, 3 = rejected).

    n_features_ : int
        Number of confirmed features.

    feature_importances_ : ndarray
        Mean feature importances across iterations.

    Example
    -------
    >>> from endgame.feature_selection import BorutaSelector
    >>> selector = BorutaSelector(max_iter=100)
    >>> X_selected = selector.fit_transform(X, y)
    >>> print(f"Confirmed: {len(selector.selected_features_)}")
    >>> print(f"Tentative: {len(selector.tentative_features_)}")
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_estimators: str = "auto",
        max_iter: int = 100,
        alpha: float = 0.05,
        perc: int = 100,
        two_step: bool = True,
        random_state: int | None = None,
        verbose: int = 0,
    ):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.alpha = alpha
        self.perc = perc
        self.two_step = two_step
        self.random_state = random_state
        self.verbose = verbose

    def _get_estimator(self, n_features: int):
        """Get the estimator to use."""
        if self.estimator is not None:
            return clone(self.estimator)

        # Auto-select n_estimators
        if self.n_estimators == "auto":
            n_est = max(100, 2 * n_features)
        else:
            n_est = self.n_estimators

        return RandomForestClassifier(
            n_estimators=n_est,
            max_depth=5,
            n_jobs=-1,
            random_state=self.random_state,
        )

    def _create_shadow_features(self, X: np.ndarray, rng) -> np.ndarray:
        """Create shadow features by shuffling each column."""
        X_shadow = np.copy(X)
        for i in range(X.shape[1]):
            rng.shuffle(X_shadow[:, i])
        return X_shadow

    def _get_importance(
        self, X: np.ndarray, X_shadow: np.ndarray, y: np.ndarray
    ) -> tuple:
        """Get importances for real and shadow features."""
        # Combine real and shadow features
        X_combined = np.hstack([X, X_shadow])

        # Fit model
        estimator = self._get_estimator(X.shape[1])
        estimator.fit(X_combined, y)

        # Get importances
        imp = estimator.feature_importances_
        n_features = X.shape[1]

        return imp[:n_features], imp[n_features:]

    def fit(self, X, y):
        """Fit the Boruta selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : BorutaSelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)

        # Track hits: how many times each feature beats max shadow
        hits = np.zeros(n_features, dtype=int)
        importance_history = []

        # Feature status: 0 = undecided, 1 = confirmed, -1 = rejected
        status = np.zeros(n_features, dtype=int)

        for iteration in range(self.max_iter):
            if self.verbose and (iteration + 1) % 10 == 0:
                print(
                    f"Iteration {iteration + 1}/{self.max_iter}: "
                    f"confirmed={np.sum(status == 1)}, "
                    f"rejected={np.sum(status == -1)}, "
                    f"undecided={np.sum(status == 0)}"
                )

            # Only process undecided features
            undecided_mask = status == 0
            if not undecided_mask.any():
                break

            # Create shadow features
            X_shadow = self._create_shadow_features(X, rng)

            # Get importances
            real_imp, shadow_imp = self._get_importance(X, X_shadow, y)
            importance_history.append(real_imp)

            # Shadow threshold
            if self.perc == 100:
                shadow_threshold = shadow_imp.max()
            else:
                shadow_threshold = np.percentile(shadow_imp, self.perc)

            # Update hits for undecided features
            for i in range(n_features):
                if status[i] == 0 and real_imp[i] > shadow_threshold:
                    hits[i] += 1

            # Statistical test (binomial test)
            # Under null hypothesis, P(beat shadow) = 0.5
            from scipy.stats import binom

            for i in range(n_features):
                if status[i] != 0:
                    continue

                n_trials = iteration + 1

                # Test for confirmation
                p_confirm = 1 - binom.cdf(hits[i] - 1, n_trials, 0.5)
                if p_confirm < self.alpha:
                    status[i] = 1  # Confirmed

                # Test for rejection
                p_reject = binom.cdf(hits[i], n_trials, 0.5)
                if p_reject < self.alpha:
                    status[i] = -1  # Rejected

        # Two-step correction for tentative features
        if self.two_step:
            # Bonferroni correction for remaining undecided
            n_undecided = np.sum(status == 0)
            if n_undecided > 0:
                corrected_alpha = self.alpha / n_undecided
                from scipy.stats import binom

                for i in range(n_features):
                    if status[i] == 0:
                        p_confirm = 1 - binom.cdf(hits[i] - 1, self.max_iter, 0.5)
                        if p_confirm < corrected_alpha:
                            status[i] = 1

        # Store results
        self.selected_features_ = np.where(status == 1)[0]
        self.tentative_features_ = np.where(status == 0)[0]
        self.rejected_features_ = np.where(status == -1)[0]

        self.n_features_ = len(self.selected_features_)
        self._support_mask = status == 1

        # Ranking: 1 = confirmed, 2 = tentative, 3 = rejected
        self.ranking_ = np.zeros(n_features, dtype=int)
        self.ranking_[status == 1] = 1
        self.ranking_[status == 0] = 2
        self.ranking_[status == -1] = 3

        # Mean importances
        if importance_history:
            self.feature_importances_ = np.mean(importance_history, axis=0)
        else:
            self.feature_importances_ = np.zeros(n_features)

        self.hits_ = hits

        return self

    def transform(self, X) -> np.ndarray:
        """Select confirmed features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with confirmed features.
        """
        check_is_fitted(self, "selected_features_")
        X = check_array(X)

        if len(self.selected_features_) == 0:
            # Return tentative if no confirmed
            if len(self.tentative_features_) > 0:
                return X[:, self.tentative_features_]
            # Return all if nothing selected
            return X

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

    def get_all_relevant(self, include_tentative: bool = True) -> np.ndarray:
        """Get all potentially relevant features.

        Parameters
        ----------
        include_tentative : bool, default=True
            Whether to include tentative features.

        Returns
        -------
        features : ndarray
            Indices of relevant features.
        """
        check_is_fitted(self, "selected_features_")
        if include_tentative:
            return np.concatenate([
                self.selected_features_,
                self.tentative_features_
            ])
        return self.selected_features_
