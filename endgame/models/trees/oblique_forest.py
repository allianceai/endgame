from __future__ import annotations

"""Oblique Random Forest: Random Forest with linear combination splits.

Standard Random Forests use axis-aligned splits — each split considers only a
single feature (e.g., "x₃ > 5.2"). Oblique Random Forests use oblique splits —
linear combinations of features (e.g., "0.3x₁ + 0.7x₃ - 0.2x₅ > 2.1"). This
allows the forest to capture linear decision boundaries that would require many
axis-aligned splits to approximate.

Why this matters for competitive ML:
- Provides ensemble diversity (errors uncorrelated with axis-aligned trees/GBDTs)
- Often outperforms standard RF on data with linear structure
- Valuable as a diverse base learner in stacking ensembles

References
----------
Menze, B. H., et al. (2011). "On Oblique Random Forests."
Machine Learning and Knowledge Discovery in Databases, ECML PKDD.

Breiman, L. (2001). "Random Forests." Machine Learning, 45(1), 5-32.
"""


import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from typing import Any

from endgame.models.trees.oblique_tree import (
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
)


def _fit_single_tree_classifier(
    tree_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    bootstrap: bool,
    base_seed: int,
    tree_params: dict,
    n_samples: int,
) -> tuple[ObliqueDecisionTreeClassifier, np.ndarray]:
    """Fit a single classifier tree (for parallel execution).

    Returns the fitted tree and the bootstrap indices (for OOB calculation).
    """
    rng = np.random.RandomState(base_seed + tree_idx)

    # Bootstrap sampling
    if bootstrap:
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        if sample_weight is not None:
            weight_sample = sample_weight[indices]
        else:
            weight_sample = None
    else:
        indices = np.arange(n_samples)
        X_sample = X
        y_sample = y
        weight_sample = sample_weight

    # Create and fit tree
    tree = ObliqueDecisionTreeClassifier(
        random_state=rng.randint(2**31),
        **tree_params
    )
    tree.fit(X_sample, y_sample, sample_weight=weight_sample)

    return tree, indices


def _fit_single_tree_regressor(
    tree_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    bootstrap: bool,
    base_seed: int,
    tree_params: dict,
    n_samples: int,
) -> tuple[ObliqueDecisionTreeRegressor, np.ndarray]:
    """Fit a single regressor tree (for parallel execution)."""
    rng = np.random.RandomState(base_seed + tree_idx)

    if bootstrap:
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        if sample_weight is not None:
            weight_sample = sample_weight[indices]
        else:
            weight_sample = None
    else:
        indices = np.arange(n_samples)
        X_sample = X
        y_sample = y
        weight_sample = sample_weight

    tree = ObliqueDecisionTreeRegressor(
        random_state=rng.randint(2**31),
        **tree_params
    )
    tree.fit(X_sample, y_sample, sample_weight=weight_sample)

    return tree, indices


class ObliqueRandomForestClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """Random Forest with oblique (linear combination) splits.

    Unlike standard Random Forest which uses axis-aligned splits on single
    features, Oblique Random Forest finds linear combinations of features
    for each split. This enables better decision boundaries for data with
    linear structure while maintaining the ensemble benefits of RF.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    oblique_method : str, default='ridge'
        Method for finding oblique split directions:
        - 'ridge': Ridge regression on class labels (recommended)
        - 'pca': Principal Component Analysis
        - 'lda': Linear Discriminant Analysis (classification only)
        - 'random': Random projections (fastest)
        - 'svm': Linear SVM hyperplane
        - 'householder': Householder reflections

    criterion : str, default='gini'
        Splitting criterion: 'gini' or 'entropy'.

    max_depth : int, default=None
        Maximum depth of each tree. None means nodes expand until
        all leaves are pure or contain < min_samples_split samples.

    min_samples_split : int or float, default=2
        Minimum samples required to split an internal node.
        If float, interpreted as fraction of total samples.

    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf node.
        If float, interpreted as fraction of total samples.

    max_features : int, float, str, or None, default='sqrt'
        Number of features to consider for each oblique split:
        - int: Use exactly max_features
        - float: Use max_features * n_features (fraction)
        - 'sqrt': Use sqrt(n_features)
        - 'log2': Use log2(n_features)
        - None: Use all features

    max_leaf_nodes : int, default=None
        Maximum number of leaf nodes per tree. None means unlimited.
        (Not yet implemented - placeholder for sklearn compatibility)

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a split.

    bootstrap : bool, default=True
        Whether to use bootstrap sampling for each tree.

    oob_score : bool, default=False
        Whether to compute out-of-bag score.

    n_jobs : int, default=None
        Number of parallel jobs for fitting trees.
        None means 1, -1 means all processors.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level.

    warm_start : bool, default=False
        If True, reuse previous fit and add more trees.

    class_weight : dict, 'balanced', 'balanced_subsample', or None, default=None
        Weights for classes. 'balanced' adjusts weights inversely
        proportional to class frequencies.
        (Not yet implemented - placeholder for sklearn compatibility)

    feature_combinations : int, default=2
        Number of features to combine in each oblique split candidate.
        Higher values allow more complex splits but increase computation.
        Only used when oblique_method='random'.

    ridge_alpha : float, default=1.0
        Regularization strength for ridge regression method.

    Attributes
    ----------
    estimators_ : list of ObliqueDecisionTreeClassifier
        The fitted tree estimators.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Feature names seen during fit (if provided).

    feature_importances_ : ndarray of shape (n_features_in_,)
        Impurity-based feature importances.

    oob_score_ : float
        Out-of-bag accuracy score (if oob_score=True).

    oob_decision_function_ : ndarray of shape (n_samples, n_classes)
        OOB predicted class probabilities (if oob_score=True).

    Examples
    --------
    >>> from endgame.models.trees import ObliqueRandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> X, y = make_classification(n_samples=1000, n_features=10,
    ...                            n_informative=5, random_state=42)
    >>> clf = ObliqueRandomForestClassifier(n_estimators=100, random_state=42)
    >>> clf.fit(X, y)
    >>> print(clf.score(X, y))

    Notes
    -----
    Oblique Random Forests typically outperform axis-aligned Random Forests
    when the true decision boundary is linear or approximately linear. They
    provide valuable diversity when ensembled with standard GBDTs.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_estimators: int = 100,
        oblique_method: str = "ridge",
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = "sqrt",
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        class_weight: dict | str | None = None,
        feature_combinations: int = 2,
        ridge_alpha: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.oblique_method = oblique_method
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.class_weight = class_weight
        self.feature_combinations = feature_combinations
        self.ridge_alpha = ridge_alpha

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ObliqueRandomForestClassifier:
        """Build an oblique random forest from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for fitting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Encode classes
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Tree parameters to pass to base estimator
        tree_params = {
            "oblique_method": self.oblique_method,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_impurity_decrease": self.min_impurity_decrease,
            "ridge_alpha": self.ridge_alpha,
            "feature_combinations": self.feature_combinations,
        }

        # Random state handling
        random_state = check_random_state(self.random_state)
        base_seed = random_state.randint(2**31)

        # Handle warm start
        if self.warm_start and hasattr(self, "estimators_"):
            n_existing = len(self.estimators_)
            if n_existing >= self.n_estimators:
                return self
            n_new = self.n_estimators - n_existing
        else:
            self.estimators_ = []
            self._bootstrap_indices = []
            n_existing = 0
            n_new = self.n_estimators

        # Fit trees in parallel
        if self.verbose > 0:
            print(f"Fitting {n_new} oblique trees...")

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single_tree_classifier)(
                i + n_existing, X, y_encoded, sample_weight,
                self.bootstrap, base_seed, tree_params, n_samples
            )
            for i in range(n_new)
        )

        # Collect results
        for tree, indices in results:
            self.estimators_.append(tree)
            self._bootstrap_indices.append(indices)

        # Compute OOB score if requested
        if self.oob_score:
            self._compute_oob_score(X, y_encoded)

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute out-of-bag score."""
        n_samples = X.shape[0]
        oob_decision_function = np.zeros((n_samples, self.n_classes_))
        n_oob_predictions = np.zeros(n_samples)

        for tree, indices in zip(self.estimators_, self._bootstrap_indices):
            # Find samples not in bootstrap
            unsampled_mask = np.ones(n_samples, dtype=bool)
            unsampled_mask[indices] = False
            unsampled_indices = np.where(unsampled_mask)[0]

            if len(unsampled_indices) == 0:
                continue

            # Get OOB predictions
            proba = tree.predict_proba(X[unsampled_indices])
            oob_decision_function[unsampled_indices] += proba
            n_oob_predictions[unsampled_indices] += 1

        # Normalize
        valid_mask = n_oob_predictions > 0
        if np.sum(valid_mask) == 0:
            self.oob_score_ = 0.0
            self.oob_decision_function_ = oob_decision_function
            return

        oob_decision_function[valid_mask] /= n_oob_predictions[valid_mask, np.newaxis]
        self.oob_decision_function_ = oob_decision_function

        # Compute accuracy
        oob_predictions = np.argmax(oob_decision_function[valid_mask], axis=1)
        self.oob_score_ = np.mean(oob_predictions == y[valid_mask])

    def _compute_feature_importances(self) -> None:
        """Compute feature importances from all trees."""
        importances = np.zeros(self.n_features_in_)

        for tree in self.estimators_:
            importances += tree.feature_importances_

        # Normalize
        importances = importances / len(self.estimators_)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["estimators_", "classes_"])
        X = check_array(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        for tree in self.estimators_:
            tree_proba = tree.predict_proba(X)
            proba += tree_proba

        proba /= len(self.estimators_)
        return proba

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply trees to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            Leaf indices for each sample in each tree.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        n_samples = X.shape[0]
        leaves = np.zeros((n_samples, len(self.estimators_)), dtype=np.int64)

        for i, tree in enumerate(self.estimators_):
            leaves[:, i] = tree.apply(X)

        return leaves

    def decision_path(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return decision path through the forest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        indicator : list of ndarrays
            Decision path indicators for each tree.
        n_nodes_ptr : ndarray of shape (n_estimators + 1,)
            Cumulative node count for each tree.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        indicators = []
        n_nodes = [0]

        for tree in self.estimators_:
            indicator = tree.decision_path(X)
            indicators.append(indicator)
            n_nodes.append(n_nodes[-1] + tree.n_nodes_)

        return indicators, np.array(n_nodes)

    @property
    def n_estimators_(self) -> int:
        """Number of fitted estimators."""
        return len(self.estimators_) if hasattr(self, "estimators_") else 0


class ObliqueRandomForestRegressor(GlassboxMixin, BaseEstimator, RegressorMixin):
    """Oblique Random Forest for regression.

    Same as ObliqueRandomForestClassifier but for continuous targets.
    Uses variance reduction (MSE) as the splitting criterion.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    oblique_method : str, default='ridge'
        Method for finding oblique split directions:
        - 'ridge': Ridge regression (recommended)
        - 'pca': Principal Component Analysis
        - 'random': Random projections (fastest)
        - 'householder': Householder reflections
        Note: 'lda' and 'svm' fall back to 'ridge' for regression.

    criterion : str, default='squared_error'
        Splitting criterion:
        - 'squared_error': Mean squared error (variance reduction)
        - 'absolute_error': Mean absolute error

    max_depth : int, default=None
        Maximum depth of each tree.

    min_samples_split : int or float, default=2
        Minimum samples required to split a node.

    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf.

    max_features : int, float, str, or None, default='sqrt'
        Features to consider per split.

    max_leaf_nodes : int, default=None
        Maximum leaf nodes per tree.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease for split.

    bootstrap : bool, default=True
        Whether to use bootstrap sampling.

    oob_score : bool, default=False
        Whether to compute out-of-bag R² score.

    n_jobs : int, default=None
        Number of parallel jobs.

    random_state : int, RandomState, or None, default=None
        Random seed.

    verbose : int, default=0
        Verbosity level.

    warm_start : bool, default=False
        If True, reuse previous fit and add more trees.

    feature_combinations : int, default=2
        Features per random combination.

    ridge_alpha : float, default=1.0
        Ridge regularization strength.

    Attributes
    ----------
    estimators_ : list of ObliqueDecisionTreeRegressor
        The fitted tree estimators.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Impurity-based feature importances.

    oob_score_ : float
        Out-of-bag R² score (if oob_score=True).

    oob_prediction_ : ndarray of shape (n_samples,)
        OOB predictions (if oob_score=True).

    Examples
    --------
    >>> from endgame.models.trees import ObliqueRandomForestRegressor
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>> reg = ObliqueRandomForestRegressor(n_estimators=100, random_state=42)
    >>> reg.fit(X, y)
    >>> print(reg.score(X, y))
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_estimators: int = 100,
        oblique_method: str = "ridge",
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = "sqrt",
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
        feature_combinations: int = 2,
        ridge_alpha: float = 1.0,
    ):
        self.n_estimators = n_estimators
        self.oblique_method = oblique_method
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start
        self.feature_combinations = feature_combinations
        self.ridge_alpha = ridge_alpha

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> ObliqueRandomForestRegressor:
        """Build an oblique random forest from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Tree parameters
        tree_params = {
            "oblique_method": self.oblique_method,
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "min_impurity_decrease": self.min_impurity_decrease,
            "ridge_alpha": self.ridge_alpha,
            "feature_combinations": self.feature_combinations,
        }

        # Random state
        random_state = check_random_state(self.random_state)
        base_seed = random_state.randint(2**31)

        # Handle warm start
        if self.warm_start and hasattr(self, "estimators_"):
            n_existing = len(self.estimators_)
            if n_existing >= self.n_estimators:
                return self
            n_new = self.n_estimators - n_existing
        else:
            self.estimators_ = []
            self._bootstrap_indices = []
            n_existing = 0
            n_new = self.n_estimators

        # Fit trees in parallel
        if self.verbose > 0:
            print(f"Fitting {n_new} oblique trees...")

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single_tree_regressor)(
                i + n_existing, X, y, sample_weight,
                self.bootstrap, base_seed, tree_params, n_samples
            )
            for i in range(n_new)
        )

        # Collect results
        for tree, indices in results:
            self.estimators_.append(tree)
            self._bootstrap_indices.append(indices)

        # Compute OOB score if requested
        if self.oob_score:
            self._compute_oob_score(X, y)

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute out-of-bag R² score."""
        n_samples = X.shape[0]
        oob_prediction = np.zeros(n_samples)
        n_oob_predictions = np.zeros(n_samples)

        for tree, indices in zip(self.estimators_, self._bootstrap_indices):
            unsampled_mask = np.ones(n_samples, dtype=bool)
            unsampled_mask[indices] = False
            unsampled_indices = np.where(unsampled_mask)[0]

            if len(unsampled_indices) == 0:
                continue

            predictions = tree.predict(X[unsampled_indices])
            oob_prediction[unsampled_indices] += predictions
            n_oob_predictions[unsampled_indices] += 1

        # Normalize
        valid_mask = n_oob_predictions > 0
        if np.sum(valid_mask) == 0:
            self.oob_score_ = 0.0
            self.oob_prediction_ = oob_prediction
            return

        oob_prediction[valid_mask] /= n_oob_predictions[valid_mask]
        self.oob_prediction_ = oob_prediction

        # Compute R²
        y_valid = y[valid_mask]
        pred_valid = oob_prediction[valid_mask]
        ss_res = np.sum((y_valid - pred_valid) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        self.oob_score_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _compute_feature_importances(self) -> None:
        """Compute feature importances from all trees."""
        importances = np.zeros(self.n_features_in_)

        for tree in self.estimators_:
            importances += tree.feature_importances_

        importances = importances / len(self.estimators_)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        predictions = np.zeros(X.shape[0])

        for tree in self.estimators_:
            predictions += tree.predict(X)

        predictions /= len(self.estimators_)
        return predictions

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply trees to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            Leaf indices for each sample in each tree.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        n_samples = X.shape[0]
        leaves = np.zeros((n_samples, len(self.estimators_)), dtype=np.int64)

        for i, tree in enumerate(self.estimators_):
            leaves[:, i] = tree.apply(X)

        return leaves

    @property
    def n_estimators_(self) -> int:
        """Number of fitted estimators."""
        return len(self.estimators_) if hasattr(self, "estimators_") else 0


def _oblique_forest_structure(self, class_names: list[Any] | None) -> dict[str, Any]:
    from endgame.models.trees.oblique_tree import _oblique_node_to_dict
    check_is_fitted(self, ["estimators_"])
    feature_names = self._structure_feature_names(self.n_features_in_)
    trees = [
        {
            "tree": {
                "root": _oblique_node_to_dict(est.tree_, feature_names, class_names),
                "max_depth": int(est.get_depth()),
                "n_leaves": int(est.get_n_leaves()),
            },
            "oblique_method": est.oblique_method,
        }
        for est in self.estimators_
    ]
    return {
        "trees": trees,
        "n_trees": len(self.estimators_),
        "feature_importances": self.feature_importances_.tolist(),
    }


ObliqueRandomForestClassifier._structure_type = "tree_ensemble"
ObliqueRandomForestClassifier._structure_content = lambda self: _oblique_forest_structure(
    self, self.classes_.tolist()
)
ObliqueRandomForestRegressor._structure_type = "tree_ensemble"
ObliqueRandomForestRegressor._structure_content = lambda self: _oblique_forest_structure(self, None)
