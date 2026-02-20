"""Feature selection methods for competitive ML."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from endgame.core.base import PolarsTransformer
from endgame.core.polars_ops import HAS_POLARS

if HAS_POLARS:
    pass


class AdversarialFeatureSelector(PolarsTransformer):
    """Removes features that contribute to train/test drift.

    Uses adversarial validation to identify and remove features
    that differ significantly between train and test distributions.

    Parameters
    ----------
    threshold : float, default=0.05
        Remove features with importance above this threshold.
    max_features_to_remove : int, default=10
        Maximum number of features to remove.
    estimator : BaseEstimator, optional
        Classifier for adversarial validation.

    Examples
    --------
    >>> selector = AdversarialFeatureSelector(threshold=0.05)
    >>> selector.fit(X_train, X_test=X_test)
    >>> X_train_clean = selector.transform(X_train)
    """

    def __init__(
        self,
        threshold: float = 0.05,
        max_features_to_remove: int = 10,
        estimator: BaseEstimator | None = None,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.threshold = threshold
        self.max_features_to_remove = max_features_to_remove
        self.estimator = estimator

        self._features_to_drop: list[str] = []
        self._feature_importances: dict[str, float] = {}

    def fit(
        self,
        X,
        y=None,
        X_test: Any | None = None,
        **fit_params,
    ) -> "AdversarialFeatureSelector":
        """Identify features to remove based on adversarial validation.

        Parameters
        ----------
        X : array-like
            Training features.
        y : ignored
        X_test : array-like
            Test features for adversarial validation.

        Returns
        -------
        self
        """
        if X_test is None:
            raise ValueError("X_test must be provided for adversarial feature selection")

        from endgame.validation.adversarial import AdversarialValidator

        lf = self._to_lazyframe(X, store_metadata=True)

        av = AdversarialValidator(
            estimator=self.estimator,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        result = av.check_drift(X, X_test)

        self._feature_importances = result.feature_importances

        # Select features to drop
        self._features_to_drop = []
        for feature in result.drifted_features:
            if len(self._features_to_drop) >= self.max_features_to_remove:
                break
            if self._feature_importances[feature] >= self.threshold:
                self._features_to_drop.append(feature)

        self._log(f"Dropping {len(self._features_to_drop)} drifted features")
        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Remove drifted features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        # Drop selected features
        cols_to_keep = [c for c in df.columns if c not in self._features_to_drop]
        result = df.select(cols_to_keep)

        return self._from_lazyframe(result.lazy())

    @property
    def features_to_drop_(self) -> list[str]:
        """Features identified for removal."""
        return self._features_to_drop

    @property
    def feature_importances_(self) -> dict[str, float]:
        """Adversarial validation feature importances."""
        return self._feature_importances


class PermutationImportanceSelector(PolarsTransformer):
    """Selects features based on permutation importance.

    More robust than model-specific importance measures because
    it measures actual predictive contribution.

    Parameters
    ----------
    estimator : BaseEstimator
        Fitted estimator to evaluate.
    threshold : float, default=0.0
        Minimum importance to keep a feature.
    n_repeats : int, default=10
        Number of permutation repetitions.
    scoring : str, optional
        Scoring metric for importance calculation.

    Examples
    --------
    >>> selector = PermutationImportanceSelector(estimator=model)
    >>> selector.fit(X_val, y_val)
    >>> X_selected = selector.transform(X_train)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        threshold: float = 0.0,
        n_repeats: int = 10,
        scoring: str | None = None,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.estimator = estimator
        self.threshold = threshold
        self.n_repeats = n_repeats
        self.scoring = scoring

        self._selected_features: list[str] = []
        self._importances: dict[str, float] = {}

    def fit(self, X, y, **fit_params) -> "PermutationImportanceSelector":
        """Compute permutation importances and select features.

        Parameters
        ----------
        X : array-like
            Validation features.
        y : array-like
            Validation targets.

        Returns
        -------
        self
        """
        from sklearn.inspection import permutation_importance

        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        X_arr = df.to_numpy()
        y_arr = np.asarray(y)
        columns = list(df.columns)

        if self.estimator is None:
            raise ValueError("estimator must be provided")

        result = permutation_importance(
            self.estimator,
            X_arr,
            y_arr,
            n_repeats=self.n_repeats,
            random_state=self.random_state,
            scoring=self.scoring,
        )

        self._importances = dict(zip(columns, result.importances_mean))

        # Select features above threshold
        self._selected_features = [
            col for col, imp in self._importances.items()
            if imp >= self.threshold
        ]

        self._log(f"Selected {len(self._selected_features)} features")
        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Keep only selected features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        cols_to_keep = [c for c in df.columns if c in self._selected_features]
        result = df.select(cols_to_keep)

        return self._from_lazyframe(result.lazy())

    @property
    def selected_features_(self) -> list[str]:
        """Features selected based on importance."""
        return self._selected_features

    @property
    def importances_(self) -> dict[str, float]:
        """Permutation importance for each feature."""
        return self._importances


class NullImportanceSelector(PolarsTransformer):
    """Selects features based on null importance distribution.

    Features must significantly outperform a shuffled-target baseline.
    Robust method for identifying truly predictive features.

    Parameters
    ----------
    estimator : BaseEstimator, optional
        Model to use. If None, uses LightGBM.
    n_iterations : int, default=100
        Number of null importance iterations.
    significance_threshold : float, default=0.95
        Percentile threshold for significance.

    Examples
    --------
    >>> selector = NullImportanceSelector(n_iterations=100)
    >>> selector.fit(X, y)
    >>> X_selected = selector.transform(X)
    """

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        n_iterations: int = 100,
        significance_threshold: float = 0.95,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.estimator = estimator
        self.n_iterations = n_iterations
        self.significance_threshold = significance_threshold

        self._selected_features: list[str] = []
        self._actual_importance: dict[str, float] = {}
        self._null_importance_threshold: dict[str, float] = {}

    def _get_default_estimator(self) -> BaseEstimator:
        """Get default estimator."""
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=100,
                learning_rate=0.1,
                num_leaves=31,
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

    def fit(self, X, y, **fit_params) -> "NullImportanceSelector":
        """Compute actual and null importances.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target values.

        Returns
        -------
        self
        """
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        X_arr = df.to_numpy()
        y_arr = np.asarray(y)
        columns = list(df.columns)

        estimator = self.estimator or self._get_default_estimator()

        # Train on actual target
        model = clone(estimator)
        model.fit(X_arr, y_arr)

        if hasattr(model, "feature_importances_"):
            self._actual_importance = dict(zip(columns, model.feature_importances_))
        else:
            # Use permutation importance
            from sklearn.inspection import permutation_importance
            result = permutation_importance(
                model, X_arr, y_arr, n_repeats=5, random_state=self.random_state
            )
            self._actual_importance = dict(zip(columns, result.importances_mean))

        # Compute null importances
        null_importances = {col: [] for col in columns}
        rng = np.random.RandomState(self.random_state)

        self._log(f"Computing null importances over {self.n_iterations} iterations...")

        for i in range(self.n_iterations):
            if self.verbose and (i + 1) % 20 == 0:
                self._log(f"  Iteration {i + 1}/{self.n_iterations}")

            # Shuffle target
            y_shuffled = rng.permutation(y_arr)

            # Train on shuffled target
            model = clone(estimator)
            model.fit(X_arr, y_shuffled)

            if hasattr(model, "feature_importances_"):
                for j, col in enumerate(columns):
                    null_importances[col].append(model.feature_importances_[j])

        # Compute threshold for each feature
        for col in columns:
            if null_importances[col]:
                threshold = np.percentile(
                    null_importances[col],
                    self.significance_threshold * 100
                )
                self._null_importance_threshold[col] = threshold

        # Select features that beat null threshold
        self._selected_features = [
            col for col in columns
            if self._actual_importance[col] > self._null_importance_threshold.get(col, 0)
        ]

        self._log(f"Selected {len(self._selected_features)} significant features")
        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Keep only significant features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        cols_to_keep = [c for c in df.columns if c in self._selected_features]
        result = df.select(cols_to_keep)

        return self._from_lazyframe(result.lazy())

    @property
    def selected_features_(self) -> list[str]:
        """Features that passed significance test."""
        return self._selected_features

    @property
    def actual_importance_(self) -> dict[str, float]:
        """Actual feature importances."""
        return self._actual_importance

    @property
    def null_threshold_(self) -> dict[str, float]:
        """Null importance thresholds."""
        return self._null_importance_threshold
