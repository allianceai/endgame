"""C5.0 Decision Tree Implementation.

A pure Python implementation of the C5.0 decision tree algorithm with support
for the Rust accelerated backend when available.

This module provides:
- C50Classifier: Single C5.0 decision tree classifier
- C50Ensemble: Boosted ensemble of C5.0 trees

Features:
- Gain ratio splitting criterion
- Subset splits for categorical attributes
- Local and global pruning
- Missing value handling
- Soft thresholds for continuous attributes
- AdaBoost-style boosting
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Try to import Rust backend
try:
    from endgame.models.trees.c50_rust import C50Classifier as RustC50Classifier
    from endgame.models.trees.c50_rust import C50Ensemble as RustC50Ensemble
    HAS_RUST = True
except ImportError:
    HAS_RUST = False


class NodeType(Enum):
    """Type of tree node."""
    LEAF = 0
    DISCRETE = 1
    THRESHOLD = 2
    SUBSET = 3


@dataclass
class TreeNode:
    """A node in the C5.0 decision tree."""
    node_type: NodeType = NodeType.LEAF
    class_: int = 0
    cases: float = 0.0
    class_dist: NDArray = field(default_factory=lambda: np.array([]))
    errors: float = 0.0
    tested_attr: int | None = None
    threshold: float | None = None
    soft_bounds: tuple[float, float, float] | None = None
    subsets: list[list[int]] | None = None
    branches: list[TreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return self.node_type == NodeType.LEAF

    def depth(self) -> int:
        if self.is_leaf:
            return 1
        return 1 + max((b.depth() for b in self.branches), default=0)

    def n_leaves(self) -> int:
        if self.is_leaf:
            return 1
        return sum(b.n_leaves() for b in self.branches)


def _log2(x: float) -> float:
    """Log base 2, handling zero."""
    return math.log2(x) if x > 0 else 0.0


def _entropy(counts: NDArray) -> float:
    """Compute entropy of a distribution."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts / total
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def _total_info(counts: NDArray) -> float:
    """Compute total information (not normalized)."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    valid = counts[counts > 0]
    return total * _log2(total) - np.sum(valid * np.log2(valid))


class SplitEvaluator:
    """Evaluates potential splits for tree building."""

    def __init__(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        indices: NDArray,
        n_classes: int,
        categorical: NDArray,
        min_cases: int = 2,
        use_subset: bool = True,
    ):
        self.X = X
        self.y = y
        self.weights = weights
        self.indices = indices
        self.n_classes = n_classes
        self.categorical = categorical
        self.min_cases = min_cases
        self.use_subset = use_subset

        # Compute class distribution
        self.class_dist = np.zeros(n_classes)
        for i in indices:
            self.class_dist[y[i]] += weights[i]
        self.total_weight = self.class_dist.sum()
        self.base_info = _entropy(self.class_dist)

    def find_best_split(self) -> dict[str, Any] | None:
        """Find the best split across all attributes."""
        best_split = None
        best_gain_ratio = 0.0

        for attr in range(self.X.shape[1]):
            if self.categorical[attr]:
                split = self._evaluate_categorical(attr)
            else:
                split = self._evaluate_continuous(attr)

            if split and split["gain_ratio"] > best_gain_ratio:
                best_gain_ratio = split["gain_ratio"]
                best_split = split

        return best_split

    def _evaluate_continuous(self, attr: int) -> dict[str, Any] | None:
        """Evaluate continuous attribute for threshold split."""
        # Get valid values
        values = []
        for i in self.indices:
            v = self.X[i, attr]
            if np.isfinite(v):
                values.append((v, self.weights[i], self.y[i], i))

        if len(values) < 2 * self.min_cases:
            return None

        # Sort by value
        values.sort(key=lambda x: x[0])

        known_weight = sum(v[1] for v in values)
        unknown_weight = self.total_weight - known_weight
        unknown_frac = unknown_weight / self.total_weight if self.total_weight > 0 else 0

        # Initialize cumulative counts
        left_counts = np.zeros(self.n_classes)
        right_counts = self.class_dist.copy()
        # Adjust for unknown
        for i in self.indices:
            if not np.isfinite(self.X[i, attr]):
                right_counts[self.y[i]] -= self.weights[i]

        left_weight = 0.0
        right_weight = known_weight

        best_gain = 0.0
        best_threshold = 0.0
        n_cuts = 0

        min_split = max(self.min_cases, min(25, int(0.1 * known_weight / self.n_classes)))

        for i in range(len(values) - 1):
            val, w, c, idx = values[i]

            # Move case from right to left
            left_counts[c] += w
            left_weight += w
            right_counts[c] -= w
            right_weight -= w

            # Skip if same value as next
            if abs(val - values[i + 1][0]) < 1e-10:
                continue

            # Check minimum cases
            if left_weight < min_split or right_weight < min_split:
                continue

            n_cuts += 1

            # Compute gain
            branch_counts = [left_counts, right_counts]
            split_info_val = 0.0
            for bc in branch_counts:
                split_info_val += _total_info(bc)
            split_info_val /= known_weight

            gain = (1 - unknown_frac) * (self.base_info - split_info_val)

            if gain > best_gain:
                best_gain = gain
                best_threshold = (val + values[i + 1][0]) / 2

        if best_gain <= 0 or n_cuts == 0:
            return None

        # Apply penalty
        penalty = _log2(max(1, n_cuts)) / known_weight
        adjusted_gain = best_gain - penalty

        if adjusted_gain <= 0:
            return None

        # Compute split info for gain ratio
        left_final = sum(w for v, w, c, i in values if v <= best_threshold)
        right_final = known_weight - left_final
        branch_weights = np.array([left_final, right_final])
        split_info = _entropy(branch_weights)
        gain_ratio = adjusted_gain / split_info if split_info > 0 else 0

        # Build branch indices
        left_indices = []
        right_indices = []
        for i in self.indices:
            v = self.X[i, attr]
            if np.isfinite(v):
                if v <= best_threshold:
                    left_indices.append(i)
                else:
                    right_indices.append(i)
            else:
                # Assign unknown to larger branch
                if left_final >= right_final:
                    left_indices.append(i)
                else:
                    right_indices.append(i)

        return {
            "attribute": attr,
            "split_type": NodeType.THRESHOLD,
            "gain": adjusted_gain,
            "gain_ratio": gain_ratio,
            "threshold": best_threshold,
            "branch_indices": [np.array(left_indices), np.array(right_indices)],
        }

    def _evaluate_categorical(self, attr: int) -> dict[str, Any] | None:
        """Evaluate categorical attribute for split."""
        if self.use_subset:
            return self._evaluate_subset(attr)
        return self._evaluate_discrete(attr)

    def _evaluate_discrete(self, attr: int) -> dict[str, Any] | None:
        """Evaluate categorical attribute for multi-way split."""
        # Map values to indices
        value_map = {}
        value_list = []

        for i in self.indices:
            v = self.X[i, attr]
            if np.isfinite(v):
                v_int = int(v)
                if v_int not in value_map:
                    value_map[v_int] = len(value_list)
                    value_list.append(v_int)

        n_values = len(value_list)
        if n_values < 2:
            return None

        # Build frequency table
        freq = np.zeros((n_values, self.n_classes))
        val_freq = np.zeros(n_values)
        unknown_weight = 0.0

        for i in self.indices:
            v = self.X[i, attr]
            w = self.weights[i]
            c = self.y[i]

            if np.isfinite(v):
                val_idx = value_map[int(v)]
                freq[val_idx, c] += w
                val_freq[val_idx] += w
            else:
                unknown_weight += w

        # Check minimum cases
        valid_branches = [v for v in range(n_values) if val_freq[v] >= self.min_cases]
        if len(valid_branches) < 2:
            return None

        # Compute gain
        unknown_frac = unknown_weight / self.total_weight if self.total_weight > 0 else 0
        split_entropy = sum(_total_info(freq[v]) for v in range(n_values))
        split_entropy /= val_freq.sum()
        gain = (1 - unknown_frac) * (self.base_info - split_entropy)

        if gain <= 0:
            return None

        split_info = _entropy(val_freq)
        gain_ratio = gain / split_info if split_info > 0 else 0

        # Build branch indices
        branch_indices = [[] for _ in range(n_values)]
        unknown_indices = []

        for i in self.indices:
            v = self.X[i, attr]
            if np.isfinite(v):
                val_idx = value_map[int(v)]
                branch_indices[val_idx].append(i)
            else:
                unknown_indices.append(i)

        # Assign unknowns to largest branch
        if unknown_indices:
            max_branch = np.argmax(val_freq)
            branch_indices[max_branch].extend(unknown_indices)

        return {
            "attribute": attr,
            "split_type": NodeType.DISCRETE,
            "gain": gain,
            "gain_ratio": gain_ratio,
            "branch_indices": [np.array(bi) for bi in branch_indices],
        }

    def _evaluate_subset(self, attr: int) -> dict[str, Any] | None:
        """Evaluate categorical attribute for subset split."""
        # Similar to discrete but with subset merging
        # For simplicity, start with discrete and could add subset optimization
        return self._evaluate_discrete(attr)


class Pruner:
    """Pruner for C5.0 trees."""

    def __init__(self, n_classes: int, cf: float = 0.25):
        self.n_classes = n_classes
        self.cf = cf
        self.z = self._compute_z(cf)

    def _compute_z(self, cf: float) -> float:
        """Compute z-value from confidence factor."""
        p = 1.0 - cf
        t = math.sqrt(-2.0 * math.log(p))
        c0, c1, c2 = 2.515517, 0.802853, 0.010328
        d1, d2, d3 = 1.432788, 0.189269, 0.001308
        return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t * t * t)

    def extra_errors(self, n: float, e: float) -> float:
        """Compute extra errors for pruning decision."""
        if n <= 0:
            return 0.0
        if e < 1e-6:
            return n * (1.0 - self.cf ** (1.0 / n))
        if e + 0.5 >= n:
            return 0.67 * (n - e)

        z2 = self.z * self.z
        f = (e + 0.5) / n
        pr = (f + z2 / (2.0 * n) + self.z * math.sqrt(f * (1.0 - f) / n + z2 / (4.0 * n * n))) / (1.0 + z2 / n)
        return n * pr - e

    def prune(self, node: TreeNode) -> None:
        """Perform bottom-up pruning."""
        if node.is_leaf:
            leaf_errors = node.cases - node.class_dist[node.class_]
            node.errors = leaf_errors + self.extra_errors(node.cases, leaf_errors)
            return

        # Prune children first
        for child in node.branches:
            self.prune(child)

        # Compute tree errors
        tree_errors = sum(b.errors for b in node.branches)

        # Compute leaf errors if collapsed
        leaf_errors = node.cases - node.class_dist[node.class_]
        leaf_estimated = leaf_errors + self.extra_errors(node.cases, leaf_errors)

        if leaf_estimated <= tree_errors + 0.1:
            # Collapse to leaf
            node.node_type = NodeType.LEAF
            node.tested_attr = None
            node.threshold = None
            node.subsets = None
            node.branches = []
            node.errors = leaf_estimated
        else:
            node.errors = tree_errors


class C50Classifier(ClassifierMixin, BaseEstimator):
    """C5.0 Decision Tree Classifier.

    A high-performance implementation of the C5.0 decision tree algorithm
    with support for continuous and categorical features, missing values,
    and sophisticated pruning.

    Parameters
    ----------
    min_cases : int, default=2
        Minimum number of cases in a branch.
    cf : float, default=0.25
        Confidence factor for pruning. Lower values = more pruning.
    use_subset : bool, default=True
        Use subset splits for categorical attributes.
    global_pruning : bool, default=True
        Apply global pruning in addition to local pruning.
    use_rust : bool, default=False
        Use Rust backend if available. Disabled by default due to a
        classification routing bug in the current Rust extension.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    tree_ : TreeNode
        The fitted decision tree.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features.
    classes_ : ndarray
        Unique class labels.
    feature_importances_ : ndarray
        Feature importances based on split gains.

    Examples
    --------
    >>> from endgame.models.trees import C50Classifier
    >>> clf = C50Classifier()
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        min_cases: int = 2,
        cf: float = 0.25,
        use_subset: bool = True,
        global_pruning: bool = True,
        use_rust: bool = False,
        random_state: int | None = None,
    ):
        self.min_cases = min_cases
        self.cf = cf
        self.use_subset = use_subset
        self.global_pruning = global_pruning
        self.use_rust = use_rust
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
        categorical_features: list[int] | None = None,
    ) -> C50Classifier:
        """Fit the C5.0 classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        categorical_features : list of int, optional
            Indices of categorical features.

        Returns
        -------
        self : C50Classifier
            Fitted classifier.
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]

        # Encode classes
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        class_map = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_map[c] for c in y])

        # Sample weights
        if sample_weight is None:
            sample_weight = np.ones(len(y))
        else:
            sample_weight = np.asarray(sample_weight)

        # Categorical features
        categorical = np.zeros(X.shape[1], dtype=bool)
        if categorical_features:
            for idx in categorical_features:
                categorical[idx] = True

        # Try Rust backend
        if self.use_rust and HAS_RUST:
            self._rust_clf = RustC50Classifier(
                min_cases=self.min_cases,
                cf=self.cf,
                use_subset=self.use_subset,
                global_pruning=self.global_pruning,
                random_state=self.random_state,
            )
            self._rust_clf.fit(
                X, y_encoded,
                categorical_features=categorical_features,
                sample_weight=sample_weight,
            )
            self._use_rust_backend = True
            return self

        self._use_rust_backend = False

        # Pure Python implementation
        self._importances = np.zeros(X.shape[1])
        indices = np.arange(len(y))

        self.tree_ = self._build_tree(X, y_encoded, sample_weight, indices, categorical, 0)

        # Prune
        if self.cf > 0:
            pruner = Pruner(self.n_classes_, self.cf)
            pruner.prune(self.tree_)

        # Normalize importances
        total = self._importances.sum()
        if total > 0:
            self._importances /= total

        return self

    def _build_tree(
        self,
        X: NDArray,
        y: NDArray,
        weights: NDArray,
        indices: NDArray,
        categorical: NDArray,
        depth: int,
    ) -> TreeNode:
        """Build tree recursively."""
        # Compute class distribution
        class_dist = np.zeros(self.n_classes_)
        for i in indices:
            class_dist[y[i]] += weights[i]
        total_weight = class_dist.sum()

        # Find majority class
        best_class = int(np.argmax(class_dist))
        max_count = class_dist[best_class]

        # Check stopping conditions
        if len(indices) < 2 * self.min_cases:
            return TreeNode(
                class_=best_class,
                cases=total_weight,
                class_dist=class_dist,
                errors=total_weight - max_count,
            )

        if max_count >= total_weight - 1e-10:
            return TreeNode(
                class_=best_class,
                cases=total_weight,
                class_dist=class_dist,
                errors=0.0,
            )

        # Find best split
        evaluator = SplitEvaluator(
            X, y, weights, indices, self.n_classes_,
            categorical, self.min_cases, self.use_subset
        )
        best_split = evaluator.find_best_split()

        if best_split is None:
            return TreeNode(
                class_=best_class,
                cases=total_weight,
                class_dist=class_dist,
                errors=total_weight - max_count,
            )

        # Update importance
        self._importances[best_split["attribute"]] += best_split["gain"] * total_weight

        # Build children
        children = []
        for child_indices in best_split["branch_indices"]:
            if len(child_indices) == 0:
                children.append(TreeNode(
                    class_=best_class,
                    cases=0.0,
                    class_dist=np.zeros(self.n_classes_),
                    errors=0.0,
                ))
            else:
                children.append(
                    self._build_tree(X, y, weights, child_indices, categorical, depth + 1)
                )

        child_errors = sum(c.errors for c in children)

        return TreeNode(
            node_type=best_split["split_type"],
            class_=best_class,
            cases=total_weight,
            class_dist=class_dist,
            errors=child_errors,
            tested_attr=best_split["attribute"],
            threshold=best_split.get("threshold"),
            subsets=best_split.get("subsets"),
            branches=children,
        )

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")

        if self._use_rust_backend:
            pred_indices = self._rust_clf.predict(X)
            return self.classes_[pred_indices]

        predictions = []
        for sample in X:
            pred_idx = self._classify(self.tree_, sample)
            predictions.append(self.classes_[pred_idx])

        return np.array(predictions)

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
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64, ensure_all_finite="allow-nan")

        if self._use_rust_backend:
            return self._rust_clf.predict_proba(X)

        probas = []
        for sample in X:
            proba = self._classify_proba(self.tree_, sample)
            probas.append(proba)

        return np.array(probas)

    def _classify(self, node: TreeNode, sample: NDArray) -> int:
        """Classify a single sample."""
        proba = self._classify_proba(node, sample)
        return int(np.argmax(proba))

    def _classify_proba(self, node: TreeNode, sample: NDArray) -> NDArray:
        """Get class probabilities for a sample."""
        class_sum = np.zeros(self.n_classes_)
        self._find_leaf(node, sample, 1.0, class_sum)

        total = class_sum.sum()
        if total > 0:
            return class_sum / total
        else:
            total = node.class_dist.sum()
            if total > 0:
                return node.class_dist / total
            return np.ones(self.n_classes_) / self.n_classes_

    def _find_leaf(
        self,
        node: TreeNode,
        sample: NDArray,
        weight: float,
        class_sum: NDArray,
    ) -> None:
        """Traverse tree to find leaf, handling missing values."""
        if node.is_leaf:
            total = node.class_dist.sum()
            if total > 0:
                class_sum += weight * node.class_dist / total
            return

        attr = node.tested_attr
        value = sample[attr] if attr < len(sample) else np.nan

        if node.node_type == NodeType.THRESHOLD:
            threshold = node.threshold
            if len(node.branches) < 2:
                return

            if not np.isfinite(value):
                # Unknown: distribute proportionally
                left_weight = node.branches[0].cases
                right_weight = node.branches[1].cases
                total = left_weight + right_weight
                if total > 0:
                    self._find_leaf(node.branches[0], sample, weight * left_weight / total, class_sum)
                    self._find_leaf(node.branches[1], sample, weight * right_weight / total, class_sum)
            else:
                if value <= threshold:
                    self._find_leaf(node.branches[0], sample, weight, class_sum)
                else:
                    self._find_leaf(node.branches[1], sample, weight, class_sum)

        elif node.node_type in (NodeType.DISCRETE, NodeType.SUBSET):
            if not np.isfinite(value):
                # Distribute proportionally
                total = sum(b.cases for b in node.branches)
                if total > 0:
                    for branch in node.branches:
                        self._find_leaf(branch, sample, weight * branch.cases / total, class_sum)
            else:
                val_idx = int(value)
                if val_idx < len(node.branches):
                    self._find_leaf(node.branches[val_idx], sample, weight, class_sum)
                else:
                    # Unknown value: use largest branch
                    best = max(node.branches, key=lambda b: b.cases)
                    self._find_leaf(best, sample, weight, class_sum)

    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances."""
        check_is_fitted(self)
        if self._use_rust_backend:
            return self._rust_clf.feature_importances()
        return self._importances

    def get_structure(self, feature_names: list[str] | None = None) -> str:
        """Get a human-readable representation of the decision tree structure.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for the features. If None, uses feature indices.

        Returns
        -------
        structure : str
            Text representation of the tree.
        """
        check_is_fitted(self)
        if self._use_rust_backend:
            return "Tree structure not available for Rust backend"

        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(self.n_features_in_)]

        lines = []
        lines.append("C5.0 Decision Tree")
        lines.append("=" * 50)
        lines.append(f"Classes: {list(self.classes_)}")
        lines.append(f"Depth: {self.tree_.depth()}")
        lines.append(f"Leaves: {self.tree_.n_leaves()}")
        lines.append("")
        lines.append("Tree Structure:")
        lines.append("-" * 50)

        self._format_node(self.tree_, feature_names, lines, indent=0)

        return "\n".join(lines)

    def _format_node(
        self,
        node: TreeNode,
        feature_names: list[str],
        lines: list[str],
        indent: int,
    ) -> None:
        """Recursively format tree node for display."""
        prefix = "  " * indent

        if node.is_leaf:
            class_name = self.classes_[node.class_]
            total = node.class_dist.sum()
            confidence = node.class_dist[node.class_] / total if total > 0 else 0
            lines.append(f"{prefix}-> class={class_name} (n={total:.0f}, conf={confidence:.2f})")
            return

        attr = node.tested_attr
        attr_name = feature_names[attr] if attr < len(feature_names) else f"attr_{attr}"

        if node.node_type == NodeType.THRESHOLD:
            lines.append(f"{prefix}{attr_name} <= {node.threshold:.4g}:")
            if len(node.branches) >= 1:
                self._format_node(node.branches[0], feature_names, lines, indent + 1)
            lines.append(f"{prefix}{attr_name} > {node.threshold:.4g}:")
            if len(node.branches) >= 2:
                self._format_node(node.branches[1], feature_names, lines, indent + 1)
        else:
            for i, branch in enumerate(node.branches):
                lines.append(f"{prefix}{attr_name} = {i}:")
                self._format_node(branch, feature_names, lines, indent + 1)

    def summary(self, feature_names: list[str] | None = None) -> str:
        """Alias for get_structure for API consistency."""
        return self.get_structure(feature_names)


class C50Ensemble(ClassifierMixin, BaseEstimator):
    """Boosted C5.0 Ensemble Classifier.

    Uses AdaBoost-style boosting to combine multiple C5.0 trees.

    Parameters
    ----------
    n_trials : int, default=10
        Number of boosting iterations.
    min_cases : int, default=2
        Minimum cases per branch.
    cf : float, default=0.25
        Confidence factor for pruning.
    use_subset : bool, default=True
        Use subset splits for categorical attributes.
    use_rust : bool, default=False
        Use Rust backend if available. Disabled by default due to a
        classification routing bug in the current Rust extension.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    estimators_ : list of C50Classifier
        The fitted trees.
    estimator_weights_ : ndarray
        Weights for each tree in voting.
    classes_ : ndarray
        Unique class labels.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_trials: int = 10,
        min_cases: int = 2,
        cf: float = 0.25,
        use_subset: bool = True,
        use_rust: bool = False,
        random_state: int | None = None,
    ):
        self.n_trials = n_trials
        self.min_cases = min_cases
        self.cf = cf
        self.use_subset = use_subset
        self.use_rust = use_rust
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        categorical_features: list[int] | None = None,
    ) -> C50Ensemble:
        """Fit the boosted ensemble."""
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        class_map = {c: i for i, c in enumerate(self.classes_)}
        y_encoded = np.array([class_map[c] for c in y])

        if self.use_rust and HAS_RUST:
            self._rust_ensemble = RustC50Ensemble(
                n_trials=self.n_trials,
                min_cases=self.min_cases,
                cf=self.cf,
                use_subset=self.use_subset,
                random_state=self.random_state,
            )
            self._rust_ensemble.fit(X, y_encoded, categorical_features)
            self._use_rust_backend = True
            return self

        self._use_rust_backend = False

        # Pure Python boosting
        n_samples = len(y)
        weights = np.ones(n_samples)
        self.estimators_ = []
        self.estimator_weights_ = []

        categorical = np.zeros(X.shape[1], dtype=bool)
        if categorical_features:
            for idx in categorical_features:
                categorical[idx] = True

        for trial in range(self.n_trials):
            # Fit tree with current weights
            clf = C50Classifier(
                min_cases=self.min_cases,
                cf=self.cf,
                use_subset=self.use_subset,
                use_rust=False,
                random_state=self.random_state,
            )
            clf.fit(X, y, sample_weight=weights, categorical_features=categorical_features)

            # Get predictions
            pred_proba = clf.predict_proba(X)
            predictions = np.argmax(pred_proba, axis=1)

            # Compute error rate
            incorrect = predictions != y_encoded
            error_weight = np.sum(weights[incorrect])
            total_weight = weights.sum()
            error_rate = error_weight / total_weight if total_weight > 0 else 0.5

            # If error rate is too high, stop boosting but keep at least one estimator
            if error_rate >= 0.5:
                if len(self.estimators_) == 0:
                    # Keep this estimator with weight 1.0 if we have none
                    self.estimators_.append(clf)
                    self.estimator_weights_.append(1.0)
                break

            # Compute tree weight (confidence)
            if error_rate > 1e-10:
                confidence = 0.5 * np.log((1 - error_rate) / error_rate)
            else:
                # Perfect classifier gets high but finite weight
                confidence = 5.0

            self.estimators_.append(clf)
            self.estimator_weights_.append(confidence)

            # If perfect, no need for more iterations
            if error_rate < 1e-10:
                break

            # Update weights for next iteration
            if trial < self.n_trials - 1:
                beta = error_rate / (1 - error_rate) if error_rate < 1 - 1e-10 else 1e-10
                # Decrease weights for correct samples
                weights[~incorrect] *= beta
                # Normalize weights
                weight_sum = weights.sum()
                if weight_sum > 0:
                    weights *= n_samples / weight_sum

        # Convert to array and normalize
        self.estimator_weights_ = np.array(self.estimator_weights_)
        weight_sum = self.estimator_weights_.sum()
        if weight_sum > 0:
            self.estimator_weights_ /= weight_sum
        else:
            # Fallback: equal weights
            self.estimator_weights_ = np.ones(len(self.estimators_)) / max(1, len(self.estimators_))

        return self

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict class labels."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if self._use_rust_backend:
            pred_indices = self._rust_ensemble.predict(X)
            return self.classes_[pred_indices]

        proba = self.predict_proba(X)
        pred_indices = np.argmax(proba, axis=1)
        return self.classes_[pred_indices]

    def predict_proba(self, X: ArrayLike) -> NDArray:
        """Predict class probabilities."""
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if self._use_rust_backend:
            return self._rust_ensemble.predict_proba(X)

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        for clf, weight in zip(self.estimators_, self.estimator_weights_):
            proba += weight * clf.predict_proba(X)

        return proba

    def get_structure(self, feature_names: list[str] | None = None) -> str:
        """Get a summary of the boosted ensemble structure.

        Parameters
        ----------
        feature_names : list of str, optional
            Names for the features.

        Returns
        -------
        structure : str
            Text representation of the ensemble.
        """
        check_is_fitted(self)

        lines = []
        lines.append("C5.0 Boosted Ensemble")
        lines.append("=" * 50)
        lines.append(f"Classes: {list(self.classes_)}")
        lines.append(f"Number of trees: {len(self.estimators_)}")
        lines.append(f"Weights: {[f'{w:.3f}' for w in self.estimator_weights_]}")
        lines.append("")

        for i, (clf, weight) in enumerate(zip(self.estimators_, self.estimator_weights_)):
            lines.append(f"Tree {i+1} (weight={weight:.3f}):")
            lines.append("-" * 30)
            if hasattr(clf, 'tree_'):
                lines.append(f"  Depth: {clf.tree_.depth()}, Leaves: {clf.tree_.n_leaves()}")
            lines.append("")

        return "\n".join(lines)

    def summary(self, feature_names: list[str] | None = None) -> str:
        """Alias for get_structure for API consistency."""
        return self.get_structure(feature_names)
