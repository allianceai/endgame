from __future__ import annotations

"""Stability selection wrapper."""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class StabilitySelector(TransformerMixin, BaseEstimator):
    """Stability selection wrapper for any feature selection method.

    Runs feature selection multiple times on bootstrap samples and
    keeps features that are consistently selected. This addresses
    the instability problem in most selection methods.

    Based on Meinshausen & Buhlmann (2010).

    Parameters
    ----------
    base_selector : TransformerMixin
        Base feature selector to wrap (must have fit/get_support).

    n_bootstrap : int, default=100
        Number of bootstrap iterations.

    sample_fraction : float, default=0.5
        Fraction of samples to use in each bootstrap.

    threshold : float, default=0.6
        Selection frequency threshold. Features selected in more
        than this fraction of bootstraps are kept.

    lambda_grid : array-like, optional
        For LASSO-style selectors, grid of regularization values.

    max_features : int, optional
        Maximum number of features to select.

    random_state : int, optional
        Random seed.

    n_jobs : int, default=None
        Number of parallel jobs.

    verbose : bool, default=False
        Whether to print progress.

    Attributes
    ----------
    selection_frequencies_ : ndarray
        Selection frequency for each feature.

    selected_features_ : ndarray
        Indices of stable features.

    bootstrap_results_ : list
        Selected features in each bootstrap.

    Example
    -------
    >>> from endgame.feature_selection import StabilitySelector, MRMRSelector
    >>> base = MRMRSelector(n_features=20)
    >>> stable = StabilitySelector(base, n_bootstrap=50, threshold=0.7)
    >>> X_selected = stable.fit_transform(X, y)
    """

    def __init__(
        self,
        base_selector: TransformerMixin,
        n_bootstrap: int = 100,
        sample_fraction: float = 0.5,
        threshold: float = 0.6,
        lambda_grid: np.ndarray | None = None,
        max_features: int | None = None,
        random_state: int | None = None,
        n_jobs: int | None = None,
        verbose: bool = False,
    ):
        self.base_selector = base_selector
        self.n_bootstrap = n_bootstrap
        self.sample_fraction = sample_fraction
        self.threshold = threshold
        self.lambda_grid = lambda_grid
        self.max_features = max_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _run_bootstrap(
        self, X: np.ndarray, y: np.ndarray, rng
    ) -> np.ndarray:
        """Run one bootstrap selection."""
        n_samples = X.shape[0]
        n_select = int(n_samples * self.sample_fraction)

        # Bootstrap sample
        indices = rng.choice(n_samples, size=n_select, replace=False)
        X_boot = X[indices]
        y_boot = y[indices]

        # Run selector
        selector = clone(self.base_selector)
        selector.fit(X_boot, y_boot)

        return selector.get_support()

    def fit(self, X, y):
        """Fit the stability selector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : StabilitySelector
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        rng = np.random.RandomState(self.random_state)

        # Run bootstrap selections
        selection_counts = np.zeros(n_features)
        self.bootstrap_results_ = []

        if self.n_jobs is not None and self.n_jobs != 1:
            # Parallel execution
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs)(
                delayed(self._run_bootstrap)(X, y, np.random.RandomState(rng.randint(0, 2**32)))
                for _ in range(self.n_bootstrap)
            )

            for mask in results:
                selection_counts += mask.astype(int)
                self.bootstrap_results_.append(np.where(mask)[0])
        else:
            # Sequential execution
            for i in range(self.n_bootstrap):
                if self.verbose and (i + 1) % 10 == 0:
                    print(f"Bootstrap {i + 1}/{self.n_bootstrap}")

                mask = self._run_bootstrap(X, y, rng)
                selection_counts += mask.astype(int)
                self.bootstrap_results_.append(np.where(mask)[0])

        # Compute selection frequencies
        self.selection_frequencies_ = selection_counts / self.n_bootstrap

        # Select features above threshold
        self._support_mask = self.selection_frequencies_ >= self.threshold
        self.selected_features_ = np.where(self._support_mask)[0]

        # Apply max_features constraint
        if self.max_features and len(self.selected_features_) > self.max_features:
            # Keep top max_features by frequency
            top_indices = np.argsort(self.selection_frequencies_)[::-1][:self.max_features]
            self._support_mask = np.zeros(n_features, dtype=bool)
            self._support_mask[top_indices] = True
            self.selected_features_ = top_indices

        self.n_features_ = len(self.selected_features_)

        # Handle case where nothing is selected
        if self.n_features_ == 0:
            # Select the most frequently chosen feature
            best_idx = np.argmax(self.selection_frequencies_)
            self._support_mask[best_idx] = True
            self.selected_features_ = np.array([best_idx])
            self.n_features_ = 1

        return self

    def transform(self, X) -> np.ndarray:
        """Select stable features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_selected : ndarray
            Data with stable features.
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

    def get_selection_frequencies(self) -> np.ndarray:
        """Get selection frequency for all features."""
        check_is_fitted(self, "selection_frequencies_")
        return self.selection_frequencies_

    def plot_selection_frequencies(self, feature_names: list | None = None):
        """Plot selection frequencies.

        Parameters
        ----------
        feature_names : list, optional
            Names for features.

        Returns
        -------
        fig : matplotlib Figure
        """
        check_is_fitted(self, "selection_frequencies_")

        import matplotlib.pyplot as plt

        n_features = len(self.selection_frequencies_)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        # Sort by frequency
        sorted_idx = np.argsort(self.selection_frequencies_)[::-1]

        fig, ax = plt.subplots(figsize=(10, max(6, n_features * 0.3)))

        y_pos = np.arange(min(30, n_features))
        top_idx = sorted_idx[:30]

        colors = ['green' if self._support_mask[i] else 'gray' for i in top_idx]

        ax.barh(y_pos, self.selection_frequencies_[top_idx], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_idx])
        ax.axvline(self.threshold, color='red', linestyle='--', label='Threshold')
        ax.set_xlabel('Selection Frequency')
        ax.set_title('Feature Stability Selection')
        ax.legend()

        plt.tight_layout()
        return fig
