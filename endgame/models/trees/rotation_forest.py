from __future__ import annotations

"""Rotation Forest: Ensemble of trees trained on PCA-rotated feature subsets."""


import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.base import EndgameEstimator


class BaseRotationForest(EndgameEstimator):
    """Base class for Rotation Forest.

    Rotation Forest is an ensemble method where each tree sees a
    different "view" of the data through PCA rotation of feature subsets.
    This provides ensemble diversity without feature subsampling.

    Algorithm:
    1. For each tree:
       a. Randomly split features into K subsets
       b. Apply PCA to each subset → get rotation matrix components
       c. Combine into full rotation matrix R
       d. Project X → X @ R
       e. Train tree on rotated data
    2. Predictions: Average tree predictions

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees in the forest.
    n_subsets : int, default=3
        Number of feature subsets per tree.
    max_features : float or int, default=0.5
        Features per subset for PCA rotation.
        If float, fraction of total features.
        If int, absolute number.
    base_estimator : estimator, optional
        Base tree estimator.
    bootstrap : bool, default=True
        Whether to bootstrap samples for each tree.
    random_state : int, optional
        Random seed.
    n_jobs : int, default=1
        Number of parallel jobs.

    References
    ----------
    Rodriguez, J.J., Kuncheva, L.I., and Alonso, C.J., 2006.
    Rotation Forest: A New Classifier Ensemble Method.
    IEEE Transactions on Pattern Analysis and Machine Intelligence.
    """

    def __init__(
        self,
        n_estimators: int = 10,
        n_subsets: int = 3,
        max_features: float = 0.5,
        base_estimator: BaseEstimator | None = None,
        bootstrap: bool = True,
        random_state: int | None = None,
        n_jobs: int = 1,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_estimators = n_estimators
        self.n_subsets = n_subsets
        self.max_features = max_features
        self.base_estimator = base_estimator
        self.bootstrap = bootstrap
        self.n_jobs = n_jobs

        self.estimators_: list[BaseEstimator] = []
        self.rotation_matrices_: list[np.ndarray] = []
        self.feature_subsets_: list[list[list[int]]] = []

    def _get_base_estimator(self) -> BaseEstimator:
        """Get the base tree estimator."""
        raise NotImplementedError("Subclasses must implement _get_base_estimator")

    def _compute_n_features_per_subset(self, n_features: int) -> int:
        """Compute number of features per subset."""
        if isinstance(self.max_features, float):
            return max(1, int(n_features * self.max_features / self.n_subsets))
        return max(1, self.max_features // self.n_subsets)

    def _create_rotation_matrix(
        self,
        X: np.ndarray,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, list[list[int]]]:
        """Create rotation matrix from PCA on feature subsets.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        rng : RandomState
            Random number generator.

        Returns
        -------
        rotation_matrix : ndarray of shape (n_features, n_features)
            Full rotation matrix.
        subsets : List[List[int]]
            Feature indices for each subset.
        """
        n_samples, n_features = X.shape
        features_per_subset = self._compute_n_features_per_subset(n_features)

        # Randomly partition features into subsets
        feature_indices = rng.permutation(n_features)
        subsets = []

        start = 0
        for i in range(self.n_subsets):
            if i == self.n_subsets - 1:
                # Last subset gets remaining features
                subset = feature_indices[start:].tolist()
            else:
                end = start + features_per_subset
                subset = feature_indices[start:end].tolist()
                start = end

            if len(subset) > 0:
                subsets.append(subset)

        # Build rotation matrix
        rotation_matrix = np.zeros((n_features, n_features))

        for subset in subsets:
            if len(subset) == 0:
                continue

            # Extract features for this subset
            X_subset = X[:, subset]

            # Bootstrap sample for PCA
            if self.bootstrap:
                boot_idx = rng.choice(n_samples, n_samples, replace=True)
                X_subset = X_subset[boot_idx]

            # Fit PCA
            n_components = min(len(subset), X_subset.shape[0])
            pca = PCA(n_components=n_components)

            try:
                pca.fit(X_subset)
                # Place PCA components in rotation matrix
                components = pca.components_.T  # (n_subset_features, n_components)

                for i, feat_idx in enumerate(subset):
                    for j, comp_idx in enumerate(subset[:n_components]):
                        if j < components.shape[1]:
                            rotation_matrix[feat_idx, comp_idx] = components[i, j]
            except Exception:
                # Fall back to identity for this subset
                for feat_idx in subset:
                    rotation_matrix[feat_idx, feat_idx] = 1.0

        # Fill diagonal for unused features
        for i in range(n_features):
            if np.all(rotation_matrix[i, :] == 0) and np.all(rotation_matrix[:, i] == 0):
                rotation_matrix[i, i] = 1.0

        return rotation_matrix, subsets

    def _fit_single_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_estimator: BaseEstimator,
        seed: int,
    ) -> tuple[BaseEstimator, np.ndarray, list[list[int]]]:
        """Fit a single rotation tree.

        Parameters
        ----------
        X : ndarray
            Training data.
        y : ndarray
            Target values.
        base_estimator : estimator
            Base tree to clone and fit.
        seed : int
            Random seed for this tree.

        Returns
        -------
        tree : estimator
            Fitted tree.
        rotation_matrix : ndarray
            Rotation matrix for this tree.
        subsets : List[List[int]]
            Feature subsets used.
        """
        n_samples = X.shape[0]
        rng = np.random.RandomState(seed)

        # Create rotation matrix
        rotation_matrix, subsets = self._create_rotation_matrix(X, rng)

        # Rotate data
        X_rotated = X @ rotation_matrix

        # Bootstrap samples
        if self.bootstrap:
            boot_idx = rng.choice(n_samples, n_samples, replace=True)
            X_boot = X_rotated[boot_idx]
            y_boot = y[boot_idx]
        else:
            X_boot = X_rotated
            y_boot = y

        # Train tree
        tree = clone(base_estimator)
        tree.fit(X_boot, y_boot)

        return tree, rotation_matrix, subsets

    def fit(self, X, y, **fit_params) -> BaseRotationForest:
        """Fit the rotation forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)
        # Generate seeds for each tree to ensure reproducibility with parallel execution
        seeds = rng.randint(0, 2**31 - 1, size=self.n_estimators)

        base_estimator = self.base_estimator or self._get_base_estimator()

        self._log(f"Training {self.n_estimators} rotation trees...")

        # For large datasets, use sequential processing to avoid pickling issues
        n_jobs = self.n_jobs
        if n_samples > 100000 and n_jobs != 1:
            n_jobs = 1  # Fall back to sequential for large datasets

        # Fit trees in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._fit_single_tree)(X, y, base_estimator, seed)
            for seed in seeds
        )

        # Unpack results
        self.estimators_ = [r[0] for r in results]
        self.rotation_matrices_ = [r[1] for r in results]
        self.feature_subsets_ = [r[2] for r in results]

        self._n_features_in = n_features
        self._is_fitted = True
        return self


class RotationForestClassifier(ClassifierMixin, BaseRotationForest):
    """Rotation Forest for classification.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.
    n_subsets : int, default=3
        Number of feature subsets per tree.
    max_features : float, default=0.5
        Fraction of features per subset.
    base_estimator : estimator, optional
        Base tree. Default: DecisionTreeClassifier.
    bootstrap : bool, default=True
        Bootstrap samples.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from endgame.models import RotationForestClassifier
    >>> clf = RotationForestClassifier(n_estimators=20)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    _estimator_type = "classifier"

    def _get_base_estimator(self) -> BaseEstimator:
        """Get default decision tree classifier."""
        return DecisionTreeClassifier(random_state=self.random_state)

    def fit(self, X, y, **fit_params) -> RotationForestClassifier:
        """Fit the classifier."""
        X, y = check_X_y(X, y)
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return super().fit(X, y, **fit_params)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def _predict_proba_single_tree(
        self,
        X: np.ndarray,
        tree: BaseEstimator,
        rotation_matrix: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get predictions from a single tree.

        Returns
        -------
        tree_proba : ndarray
            Probability predictions from this tree.
        tree_classes : ndarray
            Classes known to this tree.
        """
        X_rotated = X @ rotation_matrix
        return tree.predict_proba(X_rotated), tree.classes_

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["estimators_", "rotation_matrices_"])
        X = check_array(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        # For large datasets, use sequential processing to avoid pickling issues
        # The rotation matrices can be very large and cause memory issues with joblib
        n_jobs = self.n_jobs
        if n_samples > 100000 and n_jobs != 1:
            n_jobs = 1  # Fall back to sequential for large datasets

        # Get predictions from all trees in parallel
        results = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_proba_single_tree)(X, tree, rotation_matrix)
            for tree, rotation_matrix in zip(self.estimators_, self.rotation_matrices_)
        )

        # Aggregate predictions
        for tree_proba, tree_classes in results:
            # Handle case where tree didn't see all classes
            for i, cls in enumerate(tree_classes):
                cls_idx = np.where(self.classes_ == cls)[0][0]
                proba[:, cls_idx] += tree_proba[:, i]

        proba /= len(self.estimators_)
        return proba


class RotationForestRegressor(BaseRotationForest, RegressorMixin):
    """Rotation Forest for regression.

    Parameters
    ----------
    n_estimators : int, default=10
        Number of trees.
    n_subsets : int, default=3
        Number of feature subsets per tree.
    max_features : float, default=0.5
        Fraction of features per subset.
    base_estimator : estimator, optional
        Base tree. Default: DecisionTreeRegressor.
    bootstrap : bool, default=True
        Bootstrap samples.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from endgame.models import RotationForestRegressor
    >>> reg = RotationForestRegressor(n_estimators=20)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def _get_base_estimator(self) -> BaseEstimator:
        """Get default decision tree regressor."""
        return DecisionTreeRegressor(random_state=self.random_state)

    def _predict_single_tree(
        self,
        X: np.ndarray,
        tree: BaseEstimator,
        rotation_matrix: np.ndarray,
    ) -> np.ndarray:
        """Get predictions from a single tree."""
        X_rotated = X @ rotation_matrix
        return tree.predict(X_rotated)

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["estimators_", "rotation_matrices_"])
        X = check_array(X)

        # For large datasets, use sequential processing to avoid pickling issues
        n_jobs = self.n_jobs
        if X.shape[0] > 100000 and n_jobs != 1:
            n_jobs = 1  # Fall back to sequential for large datasets

        # Get predictions from all trees in parallel
        all_predictions = Parallel(n_jobs=n_jobs)(
            delayed(self._predict_single_tree)(X, tree, rotation_matrix)
            for tree, rotation_matrix in zip(self.estimators_, self.rotation_matrices_)
        )

        # Average predictions
        predictions = np.mean(all_predictions, axis=0)
        return predictions
