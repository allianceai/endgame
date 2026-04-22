"""Alternating Decision Tree (ADTree) and Alternating Model Tree (AMT) Implementation.

This module provides:
- AlternatingDecisionTreeClassifier: ADTree for binary/multiclass classification
- AlternatingModelTreeRegressor: AMT for regression with linear models

The Alternating Decision Tree (Freund & Mason, 1999) is a boosting-based
classification algorithm that represents a vote of features. The tree
alternates between "splitter" nodes (testing conditions) and "prediction"
nodes (containing partial predictions). The final prediction is the sum
of all predictions along all applicable paths.

The Alternating Model Tree extends this concept to regression, using
linear models at prediction nodes instead of simple constants.

References
----------
.. [1] Freund, Y., & Mason, L. (1999). The Alternating Decision Tree
       Learning Algorithm. ICML.
.. [2] Pfahringer, B., Holmes, G., & Kirkby, R. (2001). Optimizing the
       Induction of Alternating Decision Trees. PAKDD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from typing import Any


class SplitType(Enum):
    """Type of split condition."""
    THRESHOLD = 0  # value <= threshold
    EQUALITY = 1   # value == category


@dataclass
class ADTreeCondition:
    """A condition (splitter node) in the ADTree.

    Attributes
    ----------
    feature_idx : int
        Index of the feature to test.
    threshold : float
        Threshold value for comparison.
    split_type : SplitType
        Type of split (threshold or equality).
    """
    feature_idx: int
    threshold: float
    split_type: SplitType = SplitType.THRESHOLD

    def evaluate(self, X: NDArray) -> NDArray:
        """Evaluate condition on samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples to evaluate.

        Returns
        -------
        mask : ndarray of shape (n_samples,)
            Boolean mask, True where condition is satisfied.
        """
        if self.split_type == SplitType.THRESHOLD:
            return X[:, self.feature_idx] <= self.threshold
        else:
            return X[:, self.feature_idx] == self.threshold


@dataclass
class ADTreePredictionNode:
    """A prediction node in the ADTree.

    Each prediction node contains a prediction value that gets added
    to the final score for all samples that reach it.

    Attributes
    ----------
    prediction : float
        The prediction value (half of log-odds for binary classification).
    n_samples : float
        Weighted number of samples at this node.
    parent_condition : ADTreeCondition or None
        The condition that leads to this node (None for root).
    parent_satisfied : bool
        Whether the parent condition must be satisfied (True) or not (False).
    children : list of ADTreeSplitterNode
        Child splitter nodes.
    depth : int
        Depth in the tree.
    node_id : int
        Unique identifier for this node.
    """
    prediction: float = 0.0
    n_samples: float = 0.0
    parent_condition: ADTreeCondition | None = None
    parent_satisfied: bool = True
    children: list[ADTreeSplitterNode] = field(default_factory=list)
    depth: int = 0
    node_id: int = 0


@dataclass
class ADTreeSplitterNode:
    """A splitter node in the ADTree.

    Each splitter node contains a condition and two prediction children:
    one for when the condition is satisfied, one for when it's not.

    Attributes
    ----------
    condition : ADTreeCondition
        The split condition.
    yes_child : ADTreePredictionNode
        Prediction node when condition is satisfied.
    no_child : ADTreePredictionNode
        Prediction node when condition is not satisfied.
    z_score : float
        The z-score used to select this split.
    depth : int
        Depth in the tree.
    node_id : int
        Unique identifier for this node.
    """
    condition: ADTreeCondition | None = None
    yes_child: ADTreePredictionNode | None = None
    no_child: ADTreePredictionNode | None = None
    z_score: float = 0.0
    depth: int = 0
    node_id: int = 0


class AlternatingDecisionTreeClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """Alternating Decision Tree Classifier.

    The ADTree is a boosting-based classification algorithm that combines
    the interpretability of decision trees with the accuracy of boosting.
    The tree alternates between splitter nodes (testing conditions) and
    prediction nodes (containing partial votes).

    The final prediction is computed by summing the predictions from all
    prediction nodes that are "active" for a given sample (i.e., whose
    ancestor conditions are all satisfied).

    Parameters
    ----------
    n_iterations : int, default=10
        Number of boosting iterations (splitter nodes to add).

    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.

    min_samples_split : int, default=2
        Minimum samples required to consider a split.

    min_samples_leaf : int, default=1
        Minimum samples required at a leaf/prediction node.

    categorical_features : list of int or None, default=None
        Indices of categorical features. If None, all features are
        treated as continuous.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    root_ : ADTreePredictionNode
        The root prediction node of the tree.

    classes_ : ndarray of shape (n_classes,)
        Unique class labels.

    n_classes_ : int
        Number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Feature importances based on z-scores.

    n_nodes_ : int
        Total number of nodes in the tree.

    Examples
    --------
    >>> from endgame.models.trees import AlternatingDecisionTreeClassifier
    >>> clf = AlternatingDecisionTreeClassifier(n_iterations=10)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> probas = clf.predict_proba(X_test)

    Notes
    -----
    The ADTree works best for binary classification. For multiclass problems,
    it uses a one-vs-all strategy internally.

    References
    ----------
    .. [1] Freund, Y., & Mason, L. (1999). The Alternating Decision Tree
           Learning Algorithm. ICML.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_iterations: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        categorical_features: list[int] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> AlternatingDecisionTreeClassifier:
        """Fit the Alternating Decision Tree.

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
        self : AlternatingDecisionTreeClassifier
            Fitted classifier.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self._rng = check_random_state(self.random_state)

        # Encode class labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Sample weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight)

        # Categorical features
        self._categorical = np.zeros(n_features, dtype=bool)
        if self.categorical_features is not None:
            for idx in self.categorical_features:
                self._categorical[idx] = True

        # For binary classification, use standard ADTree
        if self.n_classes_ == 2:
            self._binary = True
            # Convert to -1/+1 labels
            y_binary = 2 * y_encoded - 1
            self.root_ = self._fit_binary(X, y_binary, sample_weight)
        else:
            # For multiclass, use one-vs-all with separate ADTrees
            self._binary = False
            self._ovr_trees: list[ADTreePredictionNode] = []
            for c in range(self.n_classes_):
                y_binary = np.where(y_encoded == c, 1, -1)
                tree = self._fit_binary(X, y_binary, sample_weight)
                self._ovr_trees.append(tree)
            self.root_ = self._ovr_trees[0]  # For compatibility

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _fit_binary(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray,
    ) -> ADTreePredictionNode:
        """Fit ADTree for binary classification.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Binary labels (-1 or +1).
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        root : ADTreePredictionNode
            Root of the fitted tree.
        """
        n_samples = len(y)
        self._node_count = 0
        self._splitter_count = 0

        # Initialize weights (w+ and w- for each sample)
        # These are adjusted after each boosting iteration
        weights = sample_weight.copy()

        # Create root prediction node
        # Root prediction is 0.5 * log(sum(w+) / sum(w-))
        w_plus = np.sum(weights[y == 1])
        w_minus = np.sum(weights[y == -1])
        root_pred = 0.5 * np.log((w_plus + 1e-10) / (w_minus + 1e-10))

        root = ADTreePredictionNode(
            prediction=root_pred,
            n_samples=float(n_samples),
            depth=0,
            node_id=self._node_count,
        )
        self._node_count += 1

        # Track all prediction nodes for potential splitting
        prediction_nodes: list[tuple[ADTreePredictionNode, NDArray]] = [(root, np.ones(n_samples, dtype=bool))]

        # Feature importances accumulator
        self._importance_accumulator = np.zeros(self.n_features_in_)

        # Boosting iterations
        for iteration in range(self.n_iterations):
            # Update instance weights based on current predictions
            current_predictions = self._predict_scores_single(root, X)
            weights = sample_weight * np.exp(-y * current_predictions)
            weights = weights / weights.sum() * n_samples  # Normalize

            # Find best split across all prediction nodes
            best_split = None
            best_z_score = 0.0
            best_pred_node = None
            best_pred_mask = None

            for pred_node, pred_mask in prediction_nodes:
                # Check depth limit
                if self.max_depth is not None and pred_node.depth >= self.max_depth:
                    continue

                # Check minimum samples
                if pred_mask.sum() < self.min_samples_split:
                    continue

                # Find best condition for this prediction node
                split, z_score = self._find_best_split(
                    X, y, weights, pred_mask
                )

                if split is not None and z_score > best_z_score:
                    best_split = split
                    best_z_score = z_score
                    best_pred_node = pred_node
                    best_pred_mask = pred_mask

            if best_split is None:
                break

            # Create splitter node
            splitter = ADTreeSplitterNode(
                condition=best_split,
                depth=best_pred_node.depth + 1,
                z_score=best_z_score,
                node_id=self._splitter_count,
            )
            self._splitter_count += 1

            # Update feature importance
            self._importance_accumulator[best_split.feature_idx] += best_z_score

            # Compute predictions for yes/no children
            yes_mask = best_pred_mask & best_split.evaluate(X)
            no_mask = best_pred_mask & ~best_split.evaluate(X)

            # Prediction for yes branch
            w_plus_yes = np.sum(weights[yes_mask & (y == 1)])
            w_minus_yes = np.sum(weights[yes_mask & (y == -1)])
            yes_pred = 0.5 * np.log((w_plus_yes + 1e-10) / (w_minus_yes + 1e-10))

            # Prediction for no branch
            w_plus_no = np.sum(weights[no_mask & (y == 1)])
            w_minus_no = np.sum(weights[no_mask & (y == -1)])
            no_pred = 0.5 * np.log((w_plus_no + 1e-10) / (w_minus_no + 1e-10))

            # Create prediction nodes
            yes_child = ADTreePredictionNode(
                prediction=yes_pred,
                n_samples=float(yes_mask.sum()),
                parent_condition=best_split,
                parent_satisfied=True,
                depth=splitter.depth + 1,
                node_id=self._node_count,
            )
            self._node_count += 1

            no_child = ADTreePredictionNode(
                prediction=no_pred,
                n_samples=float(no_mask.sum()),
                parent_condition=best_split,
                parent_satisfied=False,
                depth=splitter.depth + 1,
                node_id=self._node_count,
            )
            self._node_count += 1

            splitter.yes_child = yes_child
            splitter.no_child = no_child

            # Add splitter to parent prediction node
            best_pred_node.children.append(splitter)

            # Add new prediction nodes to the list
            if yes_mask.sum() >= self.min_samples_leaf:
                prediction_nodes.append((yes_child, yes_mask))
            if no_mask.sum() >= self.min_samples_leaf:
                prediction_nodes.append((no_child, no_mask))

        self.n_nodes_ = self._node_count + self._splitter_count
        return root

    def _find_best_split(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        mask: NDArray,
    ) -> tuple[ADTreeCondition | None, float]:
        """Find the best split condition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Binary labels (-1 or +1).
        weights : ndarray of shape (n_samples,)
            Current instance weights.
        mask : ndarray of shape (n_samples,)
            Boolean mask for samples in current prediction node.

        Returns
        -------
        condition : ADTreeCondition or None
            Best split condition.
        z_score : float
            Z-score of the split.
        """
        best_condition = None
        best_z_score = 0.0

        n_features = X.shape[1]
        X_masked = X[mask]
        y_masked = y[mask]
        w_masked = weights[mask]

        if len(y_masked) < self.min_samples_split:
            return None, 0.0

        for feature_idx in range(n_features):
            if self._categorical[feature_idx]:
                condition, z_score = self._find_best_categorical_split(
                    X_masked, y_masked, w_masked, feature_idx
                )
            else:
                condition, z_score = self._find_best_threshold_split(
                    X_masked, y_masked, w_masked, feature_idx
                )

            if condition is not None and z_score > best_z_score:
                best_condition = condition
                best_z_score = z_score

        return best_condition, best_z_score

    def _find_best_threshold_split(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        feature_idx: int,
    ) -> tuple[ADTreeCondition | None, float]:
        """Find best threshold split for a continuous feature.

        The z-score for a split is computed following Freund & Mason (1999):
        Z = 2 * (sqrt(W+_l * W-_l) + sqrt(W+_r * W-_r))

        We want to MINIMIZE Z, which corresponds to maximizing the margin.
        However, we return -Z as the score so that higher = better.

        where W+_l is the sum of weights for positive samples on the left, etc.
        """
        values = X[:, feature_idx]
        sorted_indices = np.argsort(values)

        sorted_values = values[sorted_indices]
        sorted_y = y[sorted_indices]
        sorted_w = weights[sorted_indices]

        # Cumulative sums for left partition
        w_plus_total = np.sum(sorted_w[sorted_y == 1])
        w_minus_total = np.sum(sorted_w[sorted_y == -1])

        w_plus_left = 0.0
        w_minus_left = 0.0

        best_threshold = None
        best_z_score = -np.inf  # We want highest z (which means lowest Z from paper)

        n_samples = len(sorted_values)

        # Baseline Z score for no split
        z_baseline = 2 * np.sqrt((w_plus_total + 1e-10) * (w_minus_total + 1e-10))

        for i in range(n_samples - 1):
            if sorted_y[i] == 1:
                w_plus_left += sorted_w[i]
            else:
                w_minus_left += sorted_w[i]

            # Skip if same value as next
            if sorted_values[i] == sorted_values[i + 1]:
                continue

            # Check minimum leaf size
            n_left = i + 1
            n_right = n_samples - n_left
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            w_plus_right = w_plus_total - w_plus_left
            w_minus_right = w_minus_total - w_minus_left

            # Compute Z-score (lower is better according to paper)
            z_split = 2 * (
                np.sqrt((w_plus_left + 1e-10) * (w_minus_left + 1e-10)) +
                np.sqrt((w_plus_right + 1e-10) * (w_minus_right + 1e-10))
            )

            # Convert to "higher is better" by computing improvement over baseline
            z_improvement = z_baseline - z_split

            if z_improvement > best_z_score:
                best_z_score = z_improvement
                best_threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

        if best_threshold is None or best_z_score <= 0:
            return None, 0.0

        condition = ADTreeCondition(
            feature_idx=feature_idx,
            threshold=best_threshold,
            split_type=SplitType.THRESHOLD,
        )
        return condition, best_z_score

    def _find_best_categorical_split(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        feature_idx: int,
    ) -> tuple[ADTreeCondition | None, float]:
        """Find best equality split for a categorical feature.

        Uses the same Z-score criterion as threshold splits.
        """
        values = X[:, feature_idx]
        unique_values = np.unique(values[np.isfinite(values)])

        if len(unique_values) < 2:
            return None, 0.0

        # Baseline Z score for no split
        w_plus_total = np.sum(weights[y == 1])
        w_minus_total = np.sum(weights[y == -1])
        z_baseline = 2 * np.sqrt((w_plus_total + 1e-10) * (w_minus_total + 1e-10))

        best_value = None
        best_z_score = -np.inf

        for val in unique_values:
            mask = values == val
            n_left = mask.sum()
            n_right = len(values) - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            w_plus_left = np.sum(weights[mask & (y == 1)])
            w_minus_left = np.sum(weights[mask & (y == -1)])
            w_plus_right = np.sum(weights[~mask & (y == 1)])
            w_minus_right = np.sum(weights[~mask & (y == -1)])

            # Compute Z-score (lower is better according to paper)
            z_split = 2 * (
                np.sqrt((w_plus_left + 1e-10) * (w_minus_left + 1e-10)) +
                np.sqrt((w_plus_right + 1e-10) * (w_minus_right + 1e-10))
            )

            # Convert to "higher is better" by computing improvement over baseline
            z_improvement = z_baseline - z_split

            if z_improvement > best_z_score:
                best_z_score = z_improvement
                best_value = val

        if best_value is None or best_z_score <= 0:
            return None, 0.0

        condition = ADTreeCondition(
            feature_idx=feature_idx,
            threshold=best_value,
            split_type=SplitType.EQUALITY,
        )
        return condition, best_z_score

    def _predict_scores_single(
        self,
        root: ADTreePredictionNode,
        X: NDArray,
    ) -> NDArray:
        """Compute raw prediction scores for a single tree.

        Parameters
        ----------
        root : ADTreePredictionNode
            Root of the tree.
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Raw scores (sum of all applicable predictions).
        """
        n_samples = X.shape[0]
        scores = np.full(n_samples, root.prediction)

        # Traverse tree and accumulate predictions
        self._accumulate_predictions(root, X, np.ones(n_samples, dtype=bool), scores)

        return scores

    def _accumulate_predictions(
        self,
        node: ADTreePredictionNode,
        X: NDArray,
        active_mask: NDArray,
        scores: NDArray,
    ) -> None:
        """Recursively accumulate predictions from all active paths.

        Parameters
        ----------
        node : ADTreePredictionNode
            Current prediction node.
        X : ndarray of shape (n_samples, n_features)
            Samples.
        active_mask : ndarray of shape (n_samples,)
            Mask of samples that have reached this node.
        scores : ndarray of shape (n_samples,)
            Scores to update.
        """
        for splitter in node.children:
            condition_mask = splitter.condition.evaluate(X)

            # Yes child (condition satisfied)
            yes_mask = active_mask & condition_mask
            if yes_mask.any():
                scores[yes_mask] += splitter.yes_child.prediction
                self._accumulate_predictions(splitter.yes_child, X, yes_mask, scores)

            # No child (condition not satisfied)
            no_mask = active_mask & ~condition_mask
            if no_mask.any():
                scores[no_mask] += splitter.no_child.prediction
                self._accumulate_predictions(splitter.no_child, X, no_mask, scores)

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["root_", "classes_"])
        X = check_array(X, dtype=np.float64)

        if self._binary:
            scores = self._predict_scores_single(self.root_, X)
            pred_indices = (scores >= 0).astype(int)
        else:
            # One-vs-all: get scores from each tree
            all_scores = np.zeros((X.shape[0], self.n_classes_))
            for c, tree in enumerate(self._ovr_trees):
                all_scores[:, c] = self._predict_scores_single(tree, X)
            pred_indices = np.argmax(all_scores, axis=1)

        return self.classes_[pred_indices]

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["root_", "classes_"])
        X = check_array(X, dtype=np.float64)

        if self._binary:
            scores = self._predict_scores_single(self.root_, X)
            # Convert to probabilities using sigmoid
            prob_positive = 1.0 / (1.0 + np.exp(-2.0 * scores))
            proba = np.column_stack([1 - prob_positive, prob_positive])
        else:
            # One-vs-all with softmax
            all_scores = np.zeros((X.shape[0], self.n_classes_))
            for c, tree in enumerate(self._ovr_trees):
                all_scores[:, c] = self._predict_scores_single(tree, X)
            # Softmax normalization
            exp_scores = np.exp(all_scores - np.max(all_scores, axis=1, keepdims=True))
            proba = exp_scores / exp_scores.sum(axis=1, keepdims=True)

        return proba

    def decision_function(self, X: ArrayLike) -> NDArray:
        """Compute raw decision function scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Decision function scores.
        """
        check_is_fitted(self, ["root_", "classes_"])
        X = check_array(X, dtype=np.float64)

        if self._binary:
            return self._predict_scores_single(self.root_, X)
        else:
            all_scores = np.zeros((X.shape[0], self.n_classes_))
            for c, tree in enumerate(self._ovr_trees):
                all_scores[:, c] = self._predict_scores_single(tree, X)
            return all_scores

    def _compute_feature_importances(self) -> None:
        """Compute feature importances based on z-scores."""
        total = self._importance_accumulator.sum()
        if total > 0:
            self.feature_importances_ = self._importance_accumulator / total
        else:
            self.feature_importances_ = np.zeros(self.n_features_in_)

    @property
    def feature_importances_(self) -> NDArray:
        """Return feature importances."""
        check_is_fitted(self, ["_importance_accumulator"])
        total = self._importance_accumulator.sum()
        if total > 0:
            return self._importance_accumulator / total
        return np.zeros(self.n_features_in_)

    @feature_importances_.setter
    def feature_importances_(self, value: NDArray) -> None:
        """Set feature importances."""
        self._feature_importances = value

    _structure_type = "tree"

    def summary(self, feature_names: list[str] | None = None) -> str:
        """Human-readable representation of the alternating decision tree."""
        check_is_fitted(self, ["root_"])

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]

        lines = []
        lines.append("Alternating Decision Tree")
        lines.append("=" * 50)
        lines.append(f"Classes: {list(self.classes_)}")
        lines.append(f"Number of iterations: {self.n_iterations}")
        lines.append(f"Total nodes: {self.n_nodes_}")
        lines.append("")
        lines.append("Tree Structure:")
        lines.append("-" * 50)

        if self._binary:
            self._format_prediction_node(self.root_, feature_names, lines, indent=0)
        else:
            for c in range(self.n_classes_):
                lines.append(f"\n--- Class {self.classes_[c]} vs Rest ---")
                self._format_prediction_node(self._ovr_trees[c], feature_names, lines, indent=0)

        return "\n".join(lines)

    def _prediction_node_to_dict(
        self,
        node: ADTreePredictionNode,
        feature_names: list[str],
        depth: int = 0,
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "type": "prediction",
            "depth": depth,
            "prediction": float(node.prediction),
            "n_samples": float(node.n_samples),
            "splitters": [],
        }
        for splitter in node.children:
            cond = splitter.condition
            feat_idx = cond.feature_idx
            out["splitters"].append({
                "feature_index": int(feat_idx),
                "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"x{feat_idx}",
                "split_type": cond.split_type.name.lower(),
                "threshold": float(cond.threshold) if cond.threshold is not None else None,
                "yes": self._prediction_node_to_dict(splitter.yes_child, feature_names, depth + 1),
                "no": self._prediction_node_to_dict(splitter.no_child, feature_names, depth + 1),
            })
        return out

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, ["root_"])
        feature_names = self._structure_feature_names(self.n_features_in_)
        if self._binary:
            tree = self._prediction_node_to_dict(self.root_, feature_names)
            return {
                "tree": tree,
                "n_iterations": int(self.n_iterations),
                "n_nodes": int(self.n_nodes_),
                "strategy": "binary",
                "feature_importances": self.feature_importances_.tolist(),
            }
        trees = [
            {
                "class_index": i,
                "class": self.classes_[i].item() if hasattr(self.classes_[i], "item") else self.classes_[i],
                "tree": self._prediction_node_to_dict(self._ovr_trees[i], feature_names),
            }
            for i in range(self.n_classes_)
        ]
        return {
            "trees": trees,
            "n_iterations": int(self.n_iterations),
            "n_nodes": int(self.n_nodes_),
            "strategy": "ovr",
            "feature_importances": self.feature_importances_.tolist(),
        }

    def _format_prediction_node(
        self,
        node: ADTreePredictionNode,
        feature_names: list[str],
        lines: list[str],
        indent: int,
    ) -> None:
        """Recursively format prediction node for display."""
        prefix = "  " * indent
        lines.append(f"{prefix}[Prediction: {node.prediction:.4f}, samples: {node.n_samples:.0f}]")

        for splitter in node.children:
            cond = splitter.condition
            feat_name = feature_names[cond.feature_idx]

            if cond.split_type == SplitType.THRESHOLD:
                lines.append(f"{prefix}  IF {feat_name} <= {cond.threshold:.4g}:")
            else:
                lines.append(f"{prefix}  IF {feat_name} == {cond.threshold}:")

            self._format_prediction_node(splitter.yes_child, feature_names, lines, indent + 2)

            if cond.split_type == SplitType.THRESHOLD:
                lines.append(f"{prefix}  ELSE ({feat_name} > {cond.threshold:.4g}):")
            else:
                lines.append(f"{prefix}  ELSE ({feat_name} != {cond.threshold}):")

            self._format_prediction_node(splitter.no_child, feature_names, lines, indent + 2)

@dataclass
class AMTreeLinearModel:
    """A linear model at a prediction node in the AMT.

    Attributes
    ----------
    coefficients : ndarray of shape (n_features,)
        Linear regression coefficients.
    intercept : float
        Intercept term.
    feature_indices : list of int
        Indices of features used in this model.
    """
    coefficients: NDArray = field(default_factory=lambda: np.array([]))
    intercept: float = 0.0
    feature_indices: list[int] = field(default_factory=list)

    def predict(self, X: NDArray) -> NDArray:
        """Make predictions with the linear model.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        if len(self.feature_indices) == 0:
            return np.full(X.shape[0], self.intercept)

        X_subset = X[:, self.feature_indices]
        return X_subset @ self.coefficients + self.intercept


@dataclass
class AMTreePredictionNode:
    """A prediction node in the Alternating Model Tree.

    Each prediction node contains a linear model that contributes to
    the final prediction for all samples that reach it.

    Attributes
    ----------
    model : AMTreeLinearModel
        The linear model at this node.
    n_samples : float
        Weighted number of samples at this node.
    parent_condition : ADTreeCondition or None
        The condition that leads to this node (None for root).
    parent_satisfied : bool
        Whether the parent condition must be satisfied (True) or not (False).
    children : list of AMTreeSplitterNode
        Child splitter nodes.
    depth : int
        Depth in the tree.
    node_id : int
        Unique identifier for this node.
    """
    model: AMTreeLinearModel | None = None
    n_samples: float = 0.0
    parent_condition: ADTreeCondition | None = None
    parent_satisfied: bool = True
    children: list[AMTreeSplitterNode] = field(default_factory=list)
    depth: int = 0
    node_id: int = 0


@dataclass
class AMTreeSplitterNode:
    """A splitter node in the Alternating Model Tree.

    Attributes
    ----------
    condition : ADTreeCondition
        The split condition.
    yes_child : AMTreePredictionNode
        Prediction node when condition is satisfied.
    no_child : AMTreePredictionNode
        Prediction node when condition is not satisfied.
    improvement : float
        The improvement in error from this split.
    depth : int
        Depth in the tree.
    node_id : int
        Unique identifier for this node.
    """
    condition: ADTreeCondition | None = None
    yes_child: AMTreePredictionNode | None = None
    no_child: AMTreePredictionNode | None = None
    improvement: float = 0.0
    depth: int = 0
    node_id: int = 0


class AlternatingModelTreeRegressor(GlassboxMixin, BaseEstimator, RegressorMixin):
    """Alternating Model Tree Regressor.

    The AMT extends the concept of Alternating Decision Trees to regression
    by using linear models at prediction nodes instead of constant values.
    The final prediction is the sum of all linear model predictions along
    all applicable paths.

    This creates a model that can capture complex non-linear relationships
    while still being interpretable as a sum of linear functions.

    Parameters
    ----------
    n_iterations : int, default=10
        Number of boosting iterations (splitter nodes to add).

    max_depth : int or None, default=None
        Maximum depth of the tree. None means unlimited.

    min_samples_split : int, default=5
        Minimum samples required to consider a split.

    min_samples_leaf : int, default=2
        Minimum samples required at a leaf/prediction node.

    model_type : str, default='constant'
        Type of model at prediction nodes:
        - 'constant': Simple mean (like standard ADTree)
        - 'linear': Full linear regression
        - 'ridge': Ridge regression (more stable)

    ridge_alpha : float, default=1.0
        Regularization parameter for ridge regression.
        Only used when model_type='ridge'.

    max_features_per_model : int or None, default=None
        Maximum features to include in each linear model.
        None means use all features.

    categorical_features : list of int or None, default=None
        Indices of categorical features.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    root_ : AMTreePredictionNode
        The root prediction node of the tree.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Feature importances based on split improvements.

    n_nodes_ : int
        Total number of nodes in the tree.

    Examples
    --------
    >>> from endgame.models.trees import AlternatingModelTreeRegressor
    >>> reg = AlternatingModelTreeRegressor(n_iterations=10, model_type='ridge')
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)

    Notes
    -----
    The AMT with linear models can capture more complex patterns than
    a standard regression tree but may be more prone to overfitting.
    Using model_type='ridge' with appropriate regularization helps.

    References
    ----------
    .. [1] Freund, Y., & Mason, L. (1999). The Alternating Decision Tree
           Learning Algorithm. ICML.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_iterations: int = 10,
        max_depth: int | None = None,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        model_type: str = "constant",
        ridge_alpha: float = 1.0,
        max_features_per_model: int | None = None,
        categorical_features: list[int] | None = None,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.n_iterations = n_iterations
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.model_type = model_type
        self.ridge_alpha = ridge_alpha
        self.max_features_per_model = max_features_per_model
        self.categorical_features = categorical_features
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> AlternatingModelTreeRegressor:
        """Fit the Alternating Model Tree.

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
        self : AlternatingModelTreeRegressor
            Fitted regressor.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        y = y.astype(np.float64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features
        self._rng = check_random_state(self.random_state)

        # Sample weights
        if sample_weight is None:
            sample_weight = np.ones(n_samples)
        else:
            sample_weight = np.asarray(sample_weight, dtype=np.float64)

        # Categorical features
        self._categorical = np.zeros(n_features, dtype=bool)
        if self.categorical_features is not None:
            for idx in self.categorical_features:
                self._categorical[idx] = True

        # Max features per model
        if self.max_features_per_model is None:
            self._max_features_model = n_features
        else:
            self._max_features_model = min(self.max_features_per_model, n_features)

        self._node_count = 0
        self._splitter_count = 0

        # Feature importances accumulator
        self._importance_accumulator = np.zeros(n_features)

        # Initialize residuals
        residuals = y.copy()

        # Create root prediction node with initial model
        root_model = self._fit_model(X, residuals, sample_weight)
        root_predictions = root_model.predict(X)
        residuals = residuals - root_predictions

        self.root_ = AMTreePredictionNode(
            model=root_model,
            n_samples=float(n_samples),
            depth=0,
            node_id=self._node_count,
        )
        self._node_count += 1

        # Track all prediction nodes for potential splitting
        prediction_nodes: list[tuple[AMTreePredictionNode, NDArray]] = [
            (self.root_, np.ones(n_samples, dtype=bool))
        ]

        # Boosting iterations
        for iteration in range(self.n_iterations):
            # Find best split across all prediction nodes
            best_split = None
            best_improvement = 0.0
            best_pred_node = None
            best_pred_mask = None
            best_yes_model = None
            best_no_model = None

            for pred_node, pred_mask in prediction_nodes:
                # Check depth limit
                if self.max_depth is not None and pred_node.depth >= self.max_depth:
                    continue

                # Check minimum samples
                if pred_mask.sum() < self.min_samples_split:
                    continue

                # Find best condition for this prediction node
                split_result = self._find_best_split(
                    X, residuals, sample_weight, pred_mask
                )

                if split_result is not None:
                    split, improvement, yes_model, no_model = split_result

                    if improvement > best_improvement:
                        best_split = split
                        best_improvement = improvement
                        best_pred_node = pred_node
                        best_pred_mask = pred_mask
                        best_yes_model = yes_model
                        best_no_model = no_model

            if best_split is None or best_improvement <= 0:
                break

            # Create splitter node
            splitter = AMTreeSplitterNode(
                condition=best_split,
                depth=best_pred_node.depth + 1,
                improvement=best_improvement,
                node_id=self._splitter_count,
            )
            self._splitter_count += 1

            # Update feature importance
            self._importance_accumulator[best_split.feature_idx] += best_improvement

            # Compute masks for children
            yes_mask = best_pred_mask & best_split.evaluate(X)
            no_mask = best_pred_mask & ~best_split.evaluate(X)

            # Create prediction nodes
            yes_child = AMTreePredictionNode(
                model=best_yes_model,
                n_samples=float(yes_mask.sum()),
                parent_condition=best_split,
                parent_satisfied=True,
                depth=splitter.depth + 1,
                node_id=self._node_count,
            )
            self._node_count += 1

            no_child = AMTreePredictionNode(
                model=best_no_model,
                n_samples=float(no_mask.sum()),
                parent_condition=best_split,
                parent_satisfied=False,
                depth=splitter.depth + 1,
                node_id=self._node_count,
            )
            self._node_count += 1

            splitter.yes_child = yes_child
            splitter.no_child = no_child

            # Add splitter to parent prediction node
            best_pred_node.children.append(splitter)

            # Update residuals
            residuals[yes_mask] -= best_yes_model.predict(X[yes_mask])
            residuals[no_mask] -= best_no_model.predict(X[no_mask])

            # Add new prediction nodes to the list
            if yes_mask.sum() >= self.min_samples_leaf:
                prediction_nodes.append((yes_child, yes_mask))
            if no_mask.sum() >= self.min_samples_leaf:
                prediction_nodes.append((no_child, no_mask))

        self.n_nodes_ = self._node_count + self._splitter_count

        return self

    def _fit_model(
        self,
        X: NDArray,
        y: NDArray,
        sample_weight: NDArray,
    ) -> AMTreeLinearModel:
        """Fit a linear model for a prediction node.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (residuals).
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.

        Returns
        -------
        model : AMTreeLinearModel
            Fitted linear model.
        """
        if self.model_type == "constant":
            # Simple weighted mean
            weighted_mean = np.average(y, weights=sample_weight)
            return AMTreeLinearModel(
                coefficients=np.array([]),
                intercept=weighted_mean,
                feature_indices=[],
            )

        # Linear or ridge regression
        n_samples, n_features = X.shape

        if n_samples < n_features + 1:
            # Not enough samples for linear regression, fall back to constant
            weighted_mean = np.average(y, weights=sample_weight)
            return AMTreeLinearModel(
                coefficients=np.array([]),
                intercept=weighted_mean,
                feature_indices=[],
            )

        # Select features for the model
        if self._max_features_model < n_features:
            # Select features with highest correlation to residuals
            correlations = np.abs([
                np.corrcoef(X[:, i], y)[0, 1] if np.std(X[:, i]) > 0 else 0
                for i in range(n_features)
            ])
            feature_indices = list(np.argsort(correlations)[-self._max_features_model:])
        else:
            feature_indices = list(range(n_features))

        X_subset = X[:, feature_indices]

        # Weighted least squares
        W = np.diag(np.sqrt(sample_weight))
        X_w = W @ X_subset
        y_w = W @ y

        if self.model_type == "ridge":
            # Ridge regression
            XtX = X_w.T @ X_w
            XtX_reg = XtX + self.ridge_alpha * np.eye(len(feature_indices))
            try:
                coefficients = np.linalg.solve(XtX_reg, X_w.T @ y_w)
            except np.linalg.LinAlgError:
                # Fall back to constant
                weighted_mean = np.average(y, weights=sample_weight)
                return AMTreeLinearModel(
                    coefficients=np.array([]),
                    intercept=weighted_mean,
                    feature_indices=[],
                )
        else:
            # Ordinary least squares
            try:
                coefficients, _, _, _ = np.linalg.lstsq(X_w, y_w, rcond=None)
            except np.linalg.LinAlgError:
                # Fall back to constant
                weighted_mean = np.average(y, weights=sample_weight)
                return AMTreeLinearModel(
                    coefficients=np.array([]),
                    intercept=weighted_mean,
                    feature_indices=[],
                )

        # Compute intercept
        intercept = np.average(y - X_subset @ coefficients, weights=sample_weight)

        return AMTreeLinearModel(
            coefficients=coefficients,
            intercept=intercept,
            feature_indices=feature_indices,
        )

    def _find_best_split(
        self,
        X: NDArray,
        residuals: NDArray,
        sample_weight: NDArray,
        mask: NDArray,
    ) -> tuple[ADTreeCondition, float, AMTreeLinearModel, AMTreeLinearModel] | None:
        """Find the best split condition.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data.
        residuals : ndarray of shape (n_samples,)
            Current residuals.
        sample_weight : ndarray of shape (n_samples,)
            Sample weights.
        mask : ndarray of shape (n_samples,)
            Boolean mask for samples in current prediction node.

        Returns
        -------
        result : tuple or None
            (condition, improvement, yes_model, no_model) or None if no valid split.
        """
        best_result = None
        best_improvement = 0.0

        n_features = X.shape[1]
        X_masked = X[mask]
        r_masked = residuals[mask]
        w_masked = sample_weight[mask]

        if len(r_masked) < self.min_samples_split:
            return None

        # Current error
        current_error = np.sum(w_masked * r_masked ** 2)

        for feature_idx in range(n_features):
            if self._categorical[feature_idx]:
                result = self._find_best_categorical_split_regression(
                    X_masked, r_masked, w_masked, feature_idx, current_error
                )
            else:
                result = self._find_best_threshold_split_regression(
                    X_masked, r_masked, w_masked, feature_idx, current_error
                )

            if result is not None:
                condition, improvement, yes_model, no_model = result
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_result = (condition, improvement, yes_model, no_model)

        return best_result

    def _find_best_threshold_split_regression(
        self,
        X: NDArray,
        residuals: NDArray,
        weights: NDArray,
        feature_idx: int,
        current_error: float,
    ) -> tuple[ADTreeCondition, float, AMTreeLinearModel, AMTreeLinearModel] | None:
        """Find best threshold split for regression."""
        values = X[:, feature_idx]
        sorted_indices = np.argsort(values)

        sorted_values = values[sorted_indices]
        sorted_r = residuals[sorted_indices]
        sorted_w = weights[sorted_indices]
        sorted_X = X[sorted_indices]

        best_result = None
        best_improvement = 0.0

        n_samples = len(sorted_values)

        # Try a subset of thresholds for efficiency
        n_thresholds = min(20, n_samples - 1)
        step = max(1, (n_samples - 1) // n_thresholds)

        for i in range(self.min_samples_leaf - 1, n_samples - self.min_samples_leaf, step):
            # Skip if same value as next
            if sorted_values[i] == sorted_values[i + 1]:
                continue

            n_left = i + 1
            n_right = n_samples - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            threshold = (sorted_values[i] + sorted_values[i + 1]) / 2

            # Fit models for left and right
            yes_model = self._fit_model(sorted_X[:n_left], sorted_r[:n_left], sorted_w[:n_left])
            no_model = self._fit_model(sorted_X[n_left:], sorted_r[n_left:], sorted_w[n_left:])

            # Compute improvement
            yes_pred = yes_model.predict(sorted_X[:n_left])
            no_pred = no_model.predict(sorted_X[n_left:])

            new_error = (
                np.sum(sorted_w[:n_left] * (sorted_r[:n_left] - yes_pred) ** 2) +
                np.sum(sorted_w[n_left:] * (sorted_r[n_left:] - no_pred) ** 2)
            )

            improvement = current_error - new_error

            if improvement > best_improvement:
                best_improvement = improvement
                condition = ADTreeCondition(
                    feature_idx=feature_idx,
                    threshold=threshold,
                    split_type=SplitType.THRESHOLD,
                )
                best_result = (condition, improvement, yes_model, no_model)

        return best_result

    def _find_best_categorical_split_regression(
        self,
        X: NDArray,
        residuals: NDArray,
        weights: NDArray,
        feature_idx: int,
        current_error: float,
    ) -> tuple[ADTreeCondition, float, AMTreeLinearModel, AMTreeLinearModel] | None:
        """Find best equality split for categorical regression."""
        values = X[:, feature_idx]
        unique_values = np.unique(values[np.isfinite(values)])

        if len(unique_values) < 2:
            return None

        best_result = None
        best_improvement = 0.0

        for val in unique_values:
            mask = values == val
            n_left = mask.sum()
            n_right = len(values) - n_left

            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue

            # Fit models
            yes_model = self._fit_model(X[mask], residuals[mask], weights[mask])
            no_model = self._fit_model(X[~mask], residuals[~mask], weights[~mask])

            # Compute improvement
            yes_pred = yes_model.predict(X[mask])
            no_pred = no_model.predict(X[~mask])

            new_error = (
                np.sum(weights[mask] * (residuals[mask] - yes_pred) ** 2) +
                np.sum(weights[~mask] * (residuals[~mask] - no_pred) ** 2)
            )

            improvement = current_error - new_error

            if improvement > best_improvement:
                best_improvement = improvement
                condition = ADTreeCondition(
                    feature_idx=feature_idx,
                    threshold=val,
                    split_type=SplitType.EQUALITY,
                )
                best_result = (condition, improvement, yes_model, no_model)

        return best_result

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["root_"])
        X = check_array(X, dtype=np.float64)

        n_samples = X.shape[0]
        predictions = self.root_.model.predict(X)

        # Traverse tree and accumulate predictions
        self._accumulate_predictions(self.root_, X, np.ones(n_samples, dtype=bool), predictions)

        return predictions

    def _accumulate_predictions(
        self,
        node: AMTreePredictionNode,
        X: NDArray,
        active_mask: NDArray,
        predictions: NDArray,
    ) -> None:
        """Recursively accumulate predictions from all active paths.

        Parameters
        ----------
        node : AMTreePredictionNode
            Current prediction node.
        X : ndarray of shape (n_samples, n_features)
            Samples.
        active_mask : ndarray of shape (n_samples,)
            Mask of samples that have reached this node.
        predictions : ndarray of shape (n_samples,)
            Predictions to update.
        """
        for splitter in node.children:
            condition_mask = splitter.condition.evaluate(X)

            # Yes child (condition satisfied)
            yes_mask = active_mask & condition_mask
            if yes_mask.any():
                predictions[yes_mask] += splitter.yes_child.model.predict(X[yes_mask])
                self._accumulate_predictions(splitter.yes_child, X, yes_mask, predictions)

            # No child (condition not satisfied)
            no_mask = active_mask & ~condition_mask
            if no_mask.any():
                predictions[no_mask] += splitter.no_child.model.predict(X[no_mask])
                self._accumulate_predictions(splitter.no_child, X, no_mask, predictions)

    @property
    def feature_importances_(self) -> NDArray:
        """Return feature importances."""
        check_is_fitted(self, ["_importance_accumulator"])
        total = self._importance_accumulator.sum()
        if total > 0:
            return self._importance_accumulator / total
        return np.zeros(self.n_features_in_)

    _structure_type = "tree"

    def summary(self, feature_names: list[str] | None = None) -> str:
        """Human-readable representation of the alternating model tree."""
        check_is_fitted(self, ["root_"])

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]

        lines = []
        lines.append("Alternating Model Tree")
        lines.append("=" * 50)
        lines.append(f"Model type: {self.model_type}")
        lines.append(f"Number of iterations: {self.n_iterations}")
        lines.append(f"Total nodes: {self.n_nodes_}")
        lines.append("")
        lines.append("Tree Structure:")
        lines.append("-" * 50)

        self._format_prediction_node(self.root_, feature_names, lines, indent=0)

        return "\n".join(lines)

    def _prediction_node_to_dict(
        self,
        node: AMTreePredictionNode,
        feature_names: list[str],
        depth: int = 0,
    ) -> dict[str, Any]:
        model = node.model
        model_dict = {
            "intercept": float(model.intercept),
            "feature_indices": [int(i) for i in model.feature_indices],
            "coefficients": [float(c) for c in model.coefficients],
            "features": [
                feature_names[i] if i < len(feature_names) else f"x{i}"
                for i in model.feature_indices
            ],
        }
        out: dict[str, Any] = {
            "type": "prediction",
            "depth": depth,
            "n_samples": float(node.n_samples),
            "linear_model": model_dict,
            "splitters": [],
        }
        for splitter in node.children:
            cond = splitter.condition
            feat_idx = cond.feature_idx
            out["splitters"].append({
                "feature_index": int(feat_idx),
                "feature": feature_names[feat_idx] if feat_idx < len(feature_names) else f"x{feat_idx}",
                "split_type": cond.split_type.name.lower(),
                "threshold": float(cond.threshold) if cond.threshold is not None else None,
                "yes": self._prediction_node_to_dict(splitter.yes_child, feature_names, depth + 1),
                "no": self._prediction_node_to_dict(splitter.no_child, feature_names, depth + 1),
            })
        return out

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, ["root_"])
        feature_names = self._structure_feature_names(self.n_features_in_)
        return {
            "tree": self._prediction_node_to_dict(self.root_, feature_names),
            "n_iterations": int(self.n_iterations),
            "n_nodes": int(self.n_nodes_),
            "model_type": self.model_type,
            "feature_importances": self.feature_importances_.tolist(),
        }

    def _format_prediction_node(
        self,
        node: AMTreePredictionNode,
        feature_names: list[str],
        lines: list[str],
        indent: int,
    ) -> None:
        """Recursively format prediction node for display."""
        prefix = "  " * indent

        model = node.model
        if len(model.feature_indices) == 0:
            lines.append(f"{prefix}[Constant: {model.intercept:.4f}, samples: {node.n_samples:.0f}]")
        else:
            model_str = f"{model.intercept:.4f}"
            for idx, feat_idx in enumerate(model.feature_indices):
                coef = model.coefficients[idx]
                feat_name = feature_names[feat_idx]
                if coef >= 0:
                    model_str += f" + {coef:.4f}*{feat_name}"
                else:
                    model_str += f" - {abs(coef):.4f}*{feat_name}"
            lines.append(f"{prefix}[Linear: {model_str}, samples: {node.n_samples:.0f}]")

        for splitter in node.children:
            cond = splitter.condition
            feat_name = feature_names[cond.feature_idx]

            if cond.split_type == SplitType.THRESHOLD:
                lines.append(f"{prefix}  IF {feat_name} <= {cond.threshold:.4g}:")
            else:
                lines.append(f"{prefix}  IF {feat_name} == {cond.threshold}:")

            self._format_prediction_node(splitter.yes_child, feature_names, lines, indent + 2)

            if cond.split_type == SplitType.THRESHOLD:
                lines.append(f"{prefix}  ELSE ({feat_name} > {cond.threshold:.4g}):")
            else:
                lines.append(f"{prefix}  ELSE ({feat_name} != {cond.threshold}):")

            self._format_prediction_node(splitter.no_child, feature_names, lines, indent + 2)
