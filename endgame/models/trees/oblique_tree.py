"""Oblique Decision Trees for Oblique Random Forests.

Oblique decision trees use linear combinations of features for splits,
allowing them to capture diagonal decision boundaries more efficiently
than axis-aligned trees.

These trees are primarily used as base estimators for ObliqueRandomForest
but can be used standalone for interpretable oblique splits.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.models.trees.oblique_splits import (
    ObliqueSplit,
    compute_entropy,
    compute_gini,
    compute_mae,
    compute_mse,
    find_best_oblique_split,
)


@dataclass
class ObliqueTreeNode:
    """A node in an oblique decision tree.

    Attributes
    ----------
    is_leaf : bool
        Whether this is a leaf node.
    split : ObliqueSplit or None
        Split information (None for leaf nodes).
    left : ObliqueTreeNode or None
        Left child (samples where split condition is True).
    right : ObliqueTreeNode or None
        Right child (samples where split condition is False).
    value : ndarray
        For classification: class counts or probabilities.
        For regression: mean target value.
    n_samples : int
        Number of training samples at this node.
    impurity : float
        Impurity (Gini, entropy, or MSE) at this node.
    depth : int
        Depth of this node in the tree.
    node_id : int
        Unique identifier for this node.
    """

    is_leaf: bool = True
    split: ObliqueSplit | None = None
    left: Optional["ObliqueTreeNode"] = None
    right: Optional["ObliqueTreeNode"] = None
    value: np.ndarray | None = None
    n_samples: int = 0
    impurity: float = 0.0
    depth: int = 0
    node_id: int = 0


class ObliqueDecisionTreeClassifier(ClassifierMixin, BaseEstimator):
    """A single oblique decision tree for classification.

    This is the base estimator used by ObliqueRandomForestClassifier.
    Uses linear combinations of features for splits, enabling better
    capture of diagonal decision boundaries.

    Parameters
    ----------
    oblique_method : str, default='ridge'
        Method for finding oblique splits:
        - 'ridge': Ridge regression on class labels (recommended)
        - 'pca': Principal Component Analysis
        - 'lda': Linear Discriminant Analysis
        - 'random': Random projections (fastest)
        - 'svm': Linear SVM hyperplane
        - 'householder': Householder reflections

    criterion : str, default='gini'
        Splitting criterion: 'gini' or 'entropy'.

    max_depth : int, default=None
        Maximum tree depth. None means unlimited.

    min_samples_split : int or float, default=2
        Minimum samples required to split a node.
        If float, fraction of total samples.

    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf.
        If float, fraction of total samples.

    max_features : int, float, str, or None, default=None
        Features to consider per split:
        - int: Use exactly max_features
        - float: Use max_features * n_features (fraction)
        - 'sqrt': Use sqrt(n_features)
        - 'log2': Use log2(n_features)
        - None: Use all features

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for split.

    random_state : int, RandomState, or None, default=None
        Random seed.

    ridge_alpha : float, default=1.0
        Ridge regularization for 'ridge' method.

    feature_combinations : int, default=2
        Features per random combination (for 'random' method).

    Attributes
    ----------
    tree_ : ObliqueTreeNode
        The root node of the fitted tree.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Impurity-based feature importances.

    n_nodes_ : int
        Number of nodes in the tree.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        oblique_method: str = "ridge",
        criterion: str = "gini",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        random_state: int | np.random.RandomState | None = None,
        ridge_alpha: float = 1.0,
        feature_combinations: int = 2,
    ):
        self.oblique_method = oblique_method
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.feature_combinations = feature_combinations

    def _resolve_max_features(self, n_features: int) -> int:
        """Resolve max_features parameter to an integer."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _resolve_min_samples(self, param: int | float, n_samples: int) -> int:
        """Resolve min_samples parameter to an integer."""
        if isinstance(param, float):
            return max(1, int(param * n_samples))
        return param

    def _get_impurity_func(self) -> Callable:
        """Get the impurity function based on criterion."""
        if self.criterion == "gini":
            return compute_gini
        elif self.criterion == "entropy":
            return compute_entropy
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ObliqueDecisionTreeClassifier":
        """Build the oblique decision tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Encode class labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Resolve parameters
        self._max_features = self._resolve_max_features(n_features)
        self._min_samples_split = self._resolve_min_samples(
            self.min_samples_split, n_samples
        )
        self._min_samples_leaf = self._resolve_min_samples(
            self.min_samples_leaf, n_samples
        )

        # Random state
        self._rng = check_random_state(self.random_state)

        # Impurity function
        self._impurity_func = self._get_impurity_func()

        # Build tree
        self._node_count = 0
        self.tree_ = self._build_tree(X, y_encoded, sample_weight, depth=0)
        self.n_nodes_ = self._node_count

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        depth: int,
    ) -> ObliqueTreeNode:
        """Recursively build the oblique decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data at this node.
        y : ndarray of shape (n_samples,)
            Target values (encoded).
        sample_weight : ndarray or None
            Sample weights.
        depth : int
            Current depth in tree.

        Returns
        -------
        node : ObliqueTreeNode
            The constructed tree node.
        """
        n_samples = len(y)

        # Create node
        node = ObliqueTreeNode()
        node.n_samples = n_samples
        node.depth = depth
        node.node_id = self._node_count
        self._node_count += 1

        # Compute node value (class distribution)
        if sample_weight is None:
            class_counts = np.bincount(y, minlength=self.n_classes_)
        else:
            class_counts = np.zeros(self.n_classes_)
            for c in range(self.n_classes_):
                mask = y == c
                class_counts[c] = np.sum(sample_weight[mask])

        node.value = class_counts / class_counts.sum() if class_counts.sum() > 0 else class_counts
        node.impurity = self._impurity_func(y, sample_weight)

        # Check stopping criteria
        if self._should_stop(node, depth, n_samples):
            node.is_leaf = True
            return node

        # Find best oblique split
        best_split = find_best_oblique_split(
            X, y, sample_weight,
            oblique_method=self.oblique_method,
            max_features=self._max_features,
            random_state=self._rng,
            min_samples_leaf=self._min_samples_leaf,
            impurity_func=self._impurity_func,
            include_axis_aligned=True,
            ridge_alpha=self.ridge_alpha,
            feature_combinations=self.feature_combinations,
        )

        if best_split is None:
            node.is_leaf = True
            return node

        # Check minimum impurity decrease
        impurity_decrease = (
            node.impurity -
            (best_split.n_samples_left / n_samples) * best_split.impurity_left -
            (best_split.n_samples_right / n_samples) * best_split.impurity_right
        )

        if impurity_decrease < self.min_impurity_decrease:
            node.is_leaf = True
            return node

        # Apply split
        left_mask = best_split.apply(X)
        right_mask = ~left_mask

        # Check min_samples_leaf again (should already be satisfied, but double-check)
        if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
            node.is_leaf = True
            return node

        # Create children
        node.is_leaf = False
        node.split = best_split

        # Left child
        if sample_weight is not None:
            left_weight = sample_weight[left_mask]
            right_weight = sample_weight[right_mask]
        else:
            left_weight = None
            right_weight = None

        node.left = self._build_tree(
            X[left_mask], y[left_mask], left_weight, depth + 1
        )
        node.right = self._build_tree(
            X[right_mask], y[right_mask], right_weight, depth + 1
        )

        return node

    def _should_stop(self, node: ObliqueTreeNode, depth: int, n_samples: int) -> bool:
        """Check if we should stop splitting."""
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self._min_samples_split:
            return True
        if node.impurity <= 0:  # Pure node
            return True
        return False

    def _compute_feature_importances(self) -> None:
        """Compute impurity-based feature importances."""
        importances = np.zeros(self.n_features_in_)
        self._accumulate_importances(self.tree_, importances, self.tree_.n_samples)

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def _accumulate_importances(
        self,
        node: ObliqueTreeNode,
        importances: np.ndarray,
        total_samples: int,
    ) -> None:
        """Recursively accumulate feature importances."""
        if node.is_leaf or node.split is None:
            return

        # Impurity decrease at this node
        n_left = node.split.n_samples_left
        n_right = node.split.n_samples_right
        n_node = node.n_samples

        impurity_decrease = (
            node.impurity -
            (n_left / n_node) * node.split.impurity_left -
            (n_right / n_node) * node.split.impurity_right
        )

        # Weight by fraction of samples
        weighted_decrease = impurity_decrease * (n_node / total_samples)

        # Distribute importance to features based on coefficient magnitude
        coeffs = np.abs(node.split.coefficients)
        coeff_sum = np.sum(coeffs)
        if coeff_sum > 0:
            coeffs = coeffs / coeff_sum

        for idx, feat_idx in enumerate(node.split.feature_indices):
            importances[feat_idx] += weighted_decrease * coeffs[idx]

        # Recurse
        self._accumulate_importances(node.left, importances, total_samples)
        self._accumulate_importances(node.right, importances, total_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

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
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["tree_", "classes_"])
        X = check_array(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        for i in range(n_samples):
            leaf = self._traverse_to_leaf(self.tree_, X[i:i+1])
            proba[i] = leaf.value

        return proba

    def _traverse_to_leaf(self, node: ObliqueTreeNode, x: np.ndarray) -> ObliqueTreeNode:
        """Traverse tree to find the leaf node for a sample."""
        if node.is_leaf:
            return node

        goes_left = node.split.apply(x)[0]

        if goes_left:
            return self._traverse_to_leaf(node.left, x)
        else:
            return self._traverse_to_leaf(node.right, x)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf indices for samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples,)
            Leaf node id for each sample.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        n_samples = X.shape[0]
        leaves = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            leaf = self._traverse_to_leaf(self.tree_, X[i:i+1])
            leaves[i] = leaf.node_id

        return leaves

    def decision_path(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return decision path through the tree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        indicator : ndarray of shape (n_samples, n_nodes)
            Dense matrix where element [i, j] = 1 if sample i passes
            through node j.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        n_samples = X.shape[0]
        indicator = np.zeros((n_samples, self.n_nodes_), dtype=np.int32)

        for i in range(n_samples):
            self._trace_path(self.tree_, X[i:i+1], indicator, i)

        return indicator

    def _trace_path(
        self,
        node: ObliqueTreeNode,
        x: np.ndarray,
        indicator: np.ndarray,
        sample_idx: int,
    ) -> None:
        """Trace path from root to leaf for a sample."""
        indicator[sample_idx, node.node_id] = 1

        if node.is_leaf:
            return

        goes_left = node.split.apply(x)[0]

        if goes_left:
            self._trace_path(node.left, x, indicator, sample_idx)
        else:
            self._trace_path(node.right, x, indicator, sample_idx)

    def get_depth(self) -> int:
        """Return the maximum depth of the tree."""
        check_is_fitted(self, ["tree_"])
        return self._get_max_depth(self.tree_)

    def _get_max_depth(self, node: ObliqueTreeNode) -> int:
        """Recursively compute max depth."""
        if node.is_leaf:
            return node.depth
        return max(
            self._get_max_depth(node.left),
            self._get_max_depth(node.right),
        )

    def get_n_leaves(self) -> int:
        """Return the number of leaves."""
        check_is_fitted(self, ["tree_"])
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node: ObliqueTreeNode) -> int:
        """Recursively count leaves."""
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)


class ObliqueDecisionTreeRegressor(BaseEstimator, RegressorMixin):
    """A single oblique decision tree for regression.

    This is the base estimator used by ObliqueRandomForestRegressor.
    Uses linear combinations of features for splits, enabling better
    capture of diagonal decision boundaries.

    Parameters
    ----------
    oblique_method : str, default='ridge'
        Method for finding oblique splits:
        - 'ridge': Ridge regression (recommended)
        - 'pca': Principal Component Analysis
        - 'random': Random projections (fastest)
        - 'householder': Householder reflections
        Note: 'lda' and 'svm' are not available for regression.

    criterion : str, default='squared_error'
        Splitting criterion: 'squared_error' or 'absolute_error'.

    max_depth : int, default=None
        Maximum tree depth. None means unlimited.

    min_samples_split : int or float, default=2
        Minimum samples required to split a node.

    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf.

    max_features : int, float, str, or None, default=None
        Features to consider per split.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for split.

    random_state : int, RandomState, or None, default=None
        Random seed.

    ridge_alpha : float, default=1.0
        Ridge regularization for 'ridge' method.

    feature_combinations : int, default=2
        Features per random combination (for 'random' method).

    Attributes
    ----------
    tree_ : ObliqueTreeNode
        The root node of the fitted tree.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Impurity-based feature importances.

    n_nodes_ : int
        Number of nodes in the tree.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        oblique_method: str = "ridge",
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = None,
        min_impurity_decrease: float = 0.0,
        random_state: int | np.random.RandomState | None = None,
        ridge_alpha: float = 1.0,
        feature_combinations: int = 2,
    ):
        self.oblique_method = oblique_method
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.min_impurity_decrease = min_impurity_decrease
        self.random_state = random_state
        self.ridge_alpha = ridge_alpha
        self.feature_combinations = feature_combinations

    def _resolve_max_features(self, n_features: int) -> int:
        """Resolve max_features parameter to an integer."""
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")

    def _resolve_min_samples(self, param: int | float, n_samples: int) -> int:
        """Resolve min_samples parameter to an integer."""
        if isinstance(param, float):
            return max(1, int(param * n_samples))
        return param

    def _get_impurity_func(self) -> Callable:
        """Get the impurity function based on criterion."""
        if self.criterion == "squared_error":
            return compute_mse
        elif self.criterion == "absolute_error":
            return compute_mae
        else:
            raise ValueError(f"Invalid criterion: {self.criterion}")

    def _get_oblique_method(self) -> str:
        """Get validated oblique method for regression."""
        # LDA and SVM are classification-only
        if self.oblique_method in ("lda", "svm"):
            return "ridge"
        return self.oblique_method

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "ObliqueDecisionTreeRegressor":
        """Build the oblique decision tree.

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

        # Resolve parameters
        self._max_features = self._resolve_max_features(n_features)
        self._min_samples_split = self._resolve_min_samples(
            self.min_samples_split, n_samples
        )
        self._min_samples_leaf = self._resolve_min_samples(
            self.min_samples_leaf, n_samples
        )
        self._oblique_method = self._get_oblique_method()

        # Random state
        self._rng = check_random_state(self.random_state)

        # Impurity function
        self._impurity_func = self._get_impurity_func()

        # Build tree
        self._node_count = 0
        self.tree_ = self._build_tree(X, y, sample_weight, depth=0)
        self.n_nodes_ = self._node_count

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None,
        depth: int,
    ) -> ObliqueTreeNode:
        """Recursively build the oblique decision tree."""
        n_samples = len(y)

        # Create node
        node = ObliqueTreeNode()
        node.n_samples = n_samples
        node.depth = depth
        node.node_id = self._node_count
        self._node_count += 1

        # Compute node value (mean prediction)
        if sample_weight is None:
            node.value = np.array([np.mean(y)])
        else:
            node.value = np.array([np.average(y, weights=sample_weight)])

        node.impurity = self._impurity_func(y, sample_weight)

        # Check stopping criteria
        if self._should_stop(node, depth, n_samples):
            node.is_leaf = True
            return node

        # Find best oblique split
        best_split = find_best_oblique_split(
            X, y, sample_weight,
            oblique_method=self._oblique_method,
            max_features=self._max_features,
            random_state=self._rng,
            min_samples_leaf=self._min_samples_leaf,
            impurity_func=self._impurity_func,
            include_axis_aligned=True,
            ridge_alpha=self.ridge_alpha,
            feature_combinations=self.feature_combinations,
        )

        if best_split is None:
            node.is_leaf = True
            return node

        # Check minimum impurity decrease
        impurity_decrease = (
            node.impurity -
            (best_split.n_samples_left / n_samples) * best_split.impurity_left -
            (best_split.n_samples_right / n_samples) * best_split.impurity_right
        )

        if impurity_decrease < self.min_impurity_decrease:
            node.is_leaf = True
            return node

        # Apply split
        left_mask = best_split.apply(X)
        right_mask = ~left_mask

        if np.sum(left_mask) < self._min_samples_leaf or np.sum(right_mask) < self._min_samples_leaf:
            node.is_leaf = True
            return node

        # Create children
        node.is_leaf = False
        node.split = best_split

        if sample_weight is not None:
            left_weight = sample_weight[left_mask]
            right_weight = sample_weight[right_mask]
        else:
            left_weight = None
            right_weight = None

        node.left = self._build_tree(
            X[left_mask], y[left_mask], left_weight, depth + 1
        )
        node.right = self._build_tree(
            X[right_mask], y[right_mask], right_weight, depth + 1
        )

        return node

    def _should_stop(self, node: ObliqueTreeNode, depth: int, n_samples: int) -> bool:
        """Check if we should stop splitting."""
        if self.max_depth is not None and depth >= self.max_depth:
            return True
        if n_samples < self._min_samples_split:
            return True
        if node.impurity <= 1e-10:  # Nearly pure node
            return True
        return False

    def _compute_feature_importances(self) -> None:
        """Compute impurity-based feature importances."""
        importances = np.zeros(self.n_features_in_)
        self._accumulate_importances(self.tree_, importances, self.tree_.n_samples)

        # Normalize
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def _accumulate_importances(
        self,
        node: ObliqueTreeNode,
        importances: np.ndarray,
        total_samples: int,
    ) -> None:
        """Recursively accumulate feature importances."""
        if node.is_leaf or node.split is None:
            return

        n_left = node.split.n_samples_left
        n_right = node.split.n_samples_right
        n_node = node.n_samples

        impurity_decrease = (
            node.impurity -
            (n_left / n_node) * node.split.impurity_left -
            (n_right / n_node) * node.split.impurity_right
        )

        weighted_decrease = impurity_decrease * (n_node / total_samples)

        coeffs = np.abs(node.split.coefficients)
        coeff_sum = np.sum(coeffs)
        if coeff_sum > 0:
            coeffs = coeffs / coeff_sum

        for idx, feat_idx in enumerate(node.split.feature_indices):
            importances[feat_idx] += weighted_decrease * coeffs[idx]

        self._accumulate_importances(node.left, importances, total_samples)
        self._accumulate_importances(node.right, importances, total_samples)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            leaf = self._traverse_to_leaf(self.tree_, X[i:i+1])
            predictions[i] = leaf.value[0]

        return predictions

    def _traverse_to_leaf(self, node: ObliqueTreeNode, x: np.ndarray) -> ObliqueTreeNode:
        """Traverse tree to find the leaf node for a sample."""
        if node.is_leaf:
            return node

        goes_left = node.split.apply(x)[0]

        if goes_left:
            return self._traverse_to_leaf(node.left, x)
        else:
            return self._traverse_to_leaf(node.right, x)

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Return leaf indices for samples."""
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        n_samples = X.shape[0]
        leaves = np.zeros(n_samples, dtype=np.int64)

        for i in range(n_samples):
            leaf = self._traverse_to_leaf(self.tree_, X[i:i+1])
            leaves[i] = leaf.node_id

        return leaves

    def get_depth(self) -> int:
        """Return the maximum depth of the tree."""
        check_is_fitted(self, ["tree_"])
        return self._get_max_depth(self.tree_)

    def _get_max_depth(self, node: ObliqueTreeNode) -> int:
        """Recursively compute max depth."""
        if node.is_leaf:
            return node.depth
        return max(
            self._get_max_depth(node.left),
            self._get_max_depth(node.right),
        )

    def get_n_leaves(self) -> int:
        """Return the number of leaves."""
        check_is_fitted(self, ["tree_"])
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node: ObliqueTreeNode) -> int:
        """Recursively count leaves."""
        if node.is_leaf:
            return 1
        return self._count_leaves(node.left) + self._count_leaves(node.right)
