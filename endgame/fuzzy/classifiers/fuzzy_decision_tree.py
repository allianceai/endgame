"""Fuzzy Decision Tree classifier and regressor with soft splits.

Instead of hard axis-aligned splits, uses membership functions to create
soft partitions. Each sample traverses ALL branches with membership-weighted
degrees, producing smooth decision boundaries.

Example
-------
>>> from endgame.fuzzy.classifiers import FuzzyDecisionTreeClassifier
>>> clf = FuzzyDecisionTreeClassifier(max_depth=3, n_splits=2)
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyClassifier, BaseFuzzyRegressor
from endgame.fuzzy.core.membership import (
    BaseMembershipFunction,
    create_uniform_mfs,
)


@dataclass
class FuzzyTreeNode:
    """Node in a fuzzy decision tree.

    Attributes
    ----------
    is_leaf : bool
        Whether this is a leaf node.
    feature_idx : int
        Index of the splitting feature (-1 for leaves).
    membership_functions : list of BaseMembershipFunction or None
        One membership function per child branch.
    children : list of FuzzyTreeNode or None
        Child nodes (one per membership function).
    value : ndarray or None
        Class distribution (classifier) or mean value (regressor) at this node.
    n_samples : float
        Weighted sample count reaching this node.
    depth : int
        Depth of this node in the tree.
    """

    is_leaf: bool = True
    feature_idx: int = -1
    membership_functions: list[BaseMembershipFunction] | None = None
    children: list[FuzzyTreeNode] | None = None
    value: np.ndarray | None = None
    n_samples: float = 0.0
    depth: int = 0


class _FuzzyDecisionTreeBase:
    """Shared logic for fuzzy decision tree classifier and regressor."""

    def _build_tree(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        depth: int,
    ) -> FuzzyTreeNode:
        """Recursively build the fuzzy decision tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Target values or class distributions.
        weights : ndarray of shape (n_samples,)
            Sample weights (fuzzy membership degrees from parent).
        depth : int
            Current depth in the tree.

        Returns
        -------
        FuzzyTreeNode
            The constructed node (leaf or internal).
        """
        n_samples = X.shape[0]
        total_weight = np.sum(weights)

        node = FuzzyTreeNode(depth=depth, n_samples=total_weight)
        node.value = self._compute_node_value(y, weights)

        # Stopping criteria
        if (
            depth >= self.max_depth
            or total_weight < self.min_samples_leaf * 2
            or n_samples < 2
        ):
            return node

        # Find best split
        best_gain = self.min_impurity_decrease
        best_feature = -1
        best_mfs = None
        best_child_weights = None

        current_impurity = self._compute_impurity(y, weights)

        for j in range(self.n_features_in_):
            # Create membership functions for this feature
            x_col = X[:, j]
            x_min = float(np.min(x_col))
            x_max = float(np.max(x_col))

            if x_max - x_min < 1e-10:
                continue  # Skip constant features

            padding = (x_max - x_min) * 0.05
            mfs = create_uniform_mfs(
                n_mfs=self.n_splits,
                x_min=x_min - padding,
                x_max=x_max + padding,
                mf_type=self._mf_type,
            )

            # Compute membership degrees for each split
            child_memberships = np.column_stack(
                [mf(x_col) for mf in mfs]
            )  # (n_samples, n_splits)

            # Weighted memberships incorporating parent weights
            child_weights_list = child_memberships * weights[:, np.newaxis]

            # Compute fuzzy information gain
            child_weight_sums = np.sum(child_weights_list, axis=0)
            weighted_child_impurity = 0.0

            valid_split = True
            for k in range(self.n_splits):
                if child_weight_sums[k] < self.min_samples_leaf:
                    valid_split = False
                    break
                child_imp = self._compute_impurity(y, child_weights_list[:, k])
                weighted_child_impurity += (
                    child_weight_sums[k] / total_weight * child_imp
                )

            if not valid_split:
                continue

            gain = current_impurity - weighted_child_impurity

            if gain > best_gain:
                best_gain = gain
                best_feature = j
                best_mfs = mfs
                best_child_weights = child_weights_list

        # If no good split found, make this a leaf
        if best_feature == -1:
            return node

        # Record feature importance
        self.feature_importances_array_[best_feature] += (
            best_gain * total_weight
        )

        # Create internal node
        node.is_leaf = False
        node.feature_idx = best_feature
        node.membership_functions = best_mfs
        node.children = []

        for k in range(self.n_splits):
            # Recurse with updated weights
            child_w = best_child_weights[:, k]
            # Filter out samples with negligible weight
            mask = child_w > 1e-10
            if np.sum(mask) < 1:
                # Create leaf with parent value
                leaf = FuzzyTreeNode(
                    depth=depth + 1,
                    n_samples=float(np.sum(child_w)),
                )
                leaf.value = node.value
                node.children.append(leaf)
            else:
                child_node = self._build_tree(
                    X[mask], y[mask], child_w[mask], depth + 1
                )
                node.children.append(child_node)

        return node

    def _traverse_tree(
        self,
        x: np.ndarray,
        node: FuzzyTreeNode,
        weight: float,
    ) -> list[tuple[np.ndarray, float]]:
        """Traverse tree for a single sample, collecting weighted leaf values.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single sample.
        node : FuzzyTreeNode
            Current node.
        weight : float
            Current accumulated weight.

        Returns
        -------
        list of (value, weight) tuples
            Weighted values from all reached leaf nodes.
        """
        if node.is_leaf or node.children is None:
            return [(node.value, weight)]

        results = []
        feature_val = x[node.feature_idx]

        for k, (child, mf) in enumerate(
            zip(node.children, node.membership_functions)
        ):
            membership = float(mf(np.array([feature_val]))[0])
            child_weight = weight * membership

            if child_weight > 1e-10:
                results.extend(
                    self._traverse_tree(x, child, child_weight)
                )

        return results

    def _predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Compute raw predictions by traversing the tree.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Accumulated weighted values for each sample.
        """
        n_samples = X.shape[0]
        result_shape = self.tree_.value.shape
        predictions = np.zeros((n_samples, *result_shape))

        for i in range(n_samples):
            leaf_results = self._traverse_tree(X[i], self.tree_, 1.0)
            for value, w in leaf_results:
                predictions[i] += w * value

        return predictions


class FuzzyDecisionTreeClassifier(_FuzzyDecisionTreeBase, BaseFuzzyClassifier):
    """Fuzzy decision tree classifier with soft splits.

    Uses membership functions instead of hard thresholds, allowing samples
    to traverse multiple branches with different membership degrees. This
    produces smoother decision boundaries than crisp decision trees.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_leaf : float, default=5.0
        Minimum weighted sample count in a leaf node.
    n_splits : int, default=2
        Number of fuzzy partitions per split (e.g., 2 for "low"/"high").
    min_impurity_decrease : float, default=0.01
        Minimum impurity decrease required for a split.
    n_mfs : int, default=3
        Number of membership functions per feature (inherited, used for
        compatibility but n_splits controls actual split count).
    mf_type : str, default='gaussian'
        Type of membership function for splits.
    t_norm : str, default='product'
        T-norm for combining memberships (not used directly in tree).
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    tree_ : FuzzyTreeNode
        The fitted fuzzy decision tree.
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on weighted information gain.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.classifiers import FuzzyDecisionTreeClassifier
    >>> X = np.random.randn(100, 4)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> clf = FuzzyDecisionTreeClassifier(max_depth=3)
    >>> clf.fit(X, y)
    FuzzyDecisionTreeClassifier(max_depth=3)
    >>> clf.predict_proba(X[:5]).shape
    (5, 2)
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: float = 5.0,
        n_splits: int = 2,
        min_impurity_decrease: float = 0.01,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
        self.min_impurity_decrease = min_impurity_decrease

    @property
    def _mf_type(self) -> str:
        return self.mf_type

    def fit(self, X: Any, y: Any) -> FuzzyDecisionTreeClassifier:
        """Build the fuzzy decision tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        y_encoded = self._encode_labels(y)

        # Initialize feature importances accumulator
        self.feature_importances_array_ = np.zeros(self.n_features_in_)

        # Initial uniform weights
        weights = np.ones(X.shape[0])

        # Build tree
        self.tree_ = self._build_tree(X, y_encoded, weights, depth=0)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_array_)
        if total_imp > 0:
            self.feature_importances_array_ /= total_imp

        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on weighted information gain."""
        check_is_fitted(self, ["feature_importances_array_"])
        return self.feature_importances_array_

    def _compute_node_value(
        self, y: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute class distribution at a node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Encoded class labels.
        weights : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        ndarray of shape (n_classes,)
            Weighted class distribution.
        """
        distribution = np.zeros(self.n_classes_)
        for c in range(self.n_classes_):
            distribution[c] = np.sum(weights[y == c])
        total = np.sum(distribution)
        if total > 0:
            distribution /= total
        return distribution

    def _compute_impurity(
        self, y: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute weighted entropy (impurity) at a node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Encoded class labels.
        weights : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        float
            Entropy value.
        """
        total = np.sum(weights)
        if total < 1e-10:
            return 0.0

        entropy = 0.0
        for c in range(self.n_classes_):
            p_c = np.sum(weights[y == c]) / total
            if p_c > 1e-10:
                entropy -= p_c * np.log2(p_c)
        return entropy

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities using fuzzy tree traversal.

        Each sample traverses all branches weighted by membership degrees.
        Leaf distributions are aggregated by their accumulated weights.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        predictions = self._predict_raw(X)

        # Normalize to probabilities
        row_sums = predictions.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        return predictions / row_sums


class FuzzyDecisionTreeRegressor(_FuzzyDecisionTreeBase, BaseFuzzyRegressor):
    """Fuzzy decision tree regressor with soft splits.

    Uses membership functions for soft partitions, allowing samples to
    contribute to multiple branches. Predictions are weighted averages
    of leaf values.

    Parameters
    ----------
    max_depth : int, default=5
        Maximum depth of the tree.
    min_samples_leaf : float, default=5.0
        Minimum weighted sample count in a leaf node.
    n_splits : int, default=2
        Number of fuzzy partitions per split.
    min_impurity_decrease : float, default=0.01
        Minimum impurity decrease required for a split.
    n_mfs : int, default=3
        Number of membership functions per feature.
    mf_type : str, default='gaussian'
        Type of membership function for splits.
    t_norm : str, default='product'
        T-norm for combining memberships.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    tree_ : FuzzyTreeNode
        The fitted fuzzy decision tree.
    n_features_in_ : int
        Number of features seen during fit.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on weighted variance reduction.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.classifiers import FuzzyDecisionTreeRegressor
    >>> X = np.random.randn(100, 4)
    >>> y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(100) * 0.1
    >>> reg = FuzzyDecisionTreeRegressor(max_depth=3)
    >>> reg.fit(X, y)
    FuzzyDecisionTreeRegressor(max_depth=3)
    >>> reg.predict(X[:5]).shape
    (5,)
    """

    def __init__(
        self,
        max_depth: int = 5,
        min_samples_leaf: float = 5.0,
        n_splits: int = 2,
        min_impurity_decrease: float = 0.01,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_splits = n_splits
        self.min_impurity_decrease = min_impurity_decrease

    @property
    def _mf_type(self) -> str:
        return self.mf_type

    def fit(self, X: Any, y: Any) -> FuzzyDecisionTreeRegressor:
        """Build the fuzzy regression tree from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        y = y.astype(np.float64)

        # Initialize feature importances accumulator
        self.feature_importances_array_ = np.zeros(self.n_features_in_)

        # Initial uniform weights
        weights = np.ones(X.shape[0])

        # Build tree
        self.tree_ = self._build_tree(X, y, weights, depth=0)

        # Normalize feature importances
        total_imp = np.sum(self.feature_importances_array_)
        if total_imp > 0:
            self.feature_importances_array_ /= total_imp

        return self

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on weighted variance reduction."""
        check_is_fitted(self, ["feature_importances_array_"])
        return self.feature_importances_array_

    def _compute_node_value(
        self, y: np.ndarray, weights: np.ndarray
    ) -> np.ndarray:
        """Compute weighted mean at a node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        weights : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        ndarray of shape (1,)
            Weighted mean value.
        """
        total = np.sum(weights)
        if total < 1e-10:
            return np.array([0.0])
        return np.array([np.sum(weights * y) / total])

    def _compute_impurity(
        self, y: np.ndarray, weights: np.ndarray
    ) -> float:
        """Compute weighted variance (impurity) at a node.

        Parameters
        ----------
        y : ndarray of shape (n_samples,)
            Target values.
        weights : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        float
            Weighted variance.
        """
        total = np.sum(weights)
        if total < 1e-10:
            return 0.0
        mean = np.sum(weights * y) / total
        return float(np.sum(weights * (y - mean) ** 2) / total)

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using fuzzy tree traversal.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["tree_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            leaf_results = self._traverse_tree(X[i], self.tree_, 1.0)
            total_weight = sum(w for _, w in leaf_results)
            if total_weight > 0:
                predictions[i] = sum(
                    float(v[0]) * w for v, w in leaf_results
                ) / total_weight
            else:
                predictions[i] = float(self.tree_.value[0])

        return predictions
