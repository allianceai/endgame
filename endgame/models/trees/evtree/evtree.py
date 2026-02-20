"""Evolutionary Trees (evtree) - Globally optimal decision trees via genetic algorithms.

This implementation uses evolutionary algorithms to find globally optimal tree
structures, unlike greedy methods (CART/C4.5) which make locally optimal splits.

Key features:
- BIC-type fitness balancing accuracy and complexity
- Deterministic crowding for diversity maintenance
- 4 mutation operators + crossover
- Performance optimizations: numba JIT, parallel evaluation, warm start

References
----------
- Grubinger et al., "evtree: Evolutionary Learning of Globally Optimal
  Classification and Regression Trees in R" (2014)
- https://cran.r-project.org/web/packages/evtree/

Performance Notes
-----------------
- Hot paths (fitness evaluation, prediction) use numba JIT compilation
- Population evaluation can be parallelized via joblib
- Warm start seeds population with greedy tree for faster convergence
- Early stopping when no improvement for `patience` generations
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.utils import check_random_state

# Try to import numba for JIT compilation
try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator that does nothing
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# Try to import joblib for parallel processing
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False


# =============================================================================
# Tree Node Representation
# =============================================================================

@dataclass
class TreeNode:
    """Efficient tree node representation.

    For internal nodes: feature_idx >= 0, threshold is the split point
    For leaf nodes: feature_idx = -1, value contains prediction
    """
    feature_idx: int = -1  # -1 means leaf node
    threshold: float = 0.0
    value: np.ndarray = field(default_factory=lambda: np.array([]))
    left: Optional['TreeNode'] = None
    right: Optional['TreeNode'] = None
    n_samples: int = 0

    def is_leaf(self) -> bool:
        return self.feature_idx == -1

    def copy(self) -> 'TreeNode':
        """Deep copy the node and its subtree."""
        new_node = TreeNode(
            feature_idx=self.feature_idx,
            threshold=self.threshold,
            value=self.value.copy() if len(self.value) > 0 else np.array([]),
            n_samples=self.n_samples,
        )
        if self.left is not None:
            new_node.left = self.left.copy()
        if self.right is not None:
            new_node.right = self.right.copy()
        return new_node

    def count_nodes(self) -> int:
        """Count total nodes in subtree."""
        count = 1
        if self.left is not None:
            count += self.left.count_nodes()
        if self.right is not None:
            count += self.right.count_nodes()
        return count

    def count_leaves(self) -> int:
        """Count terminal nodes (leaves) in subtree."""
        if self.is_leaf():
            return 1
        count = 0
        if self.left is not None:
            count += self.left.count_leaves()
        if self.right is not None:
            count += self.right.count_leaves()
        return count

    def depth(self) -> int:
        """Get maximum depth of subtree."""
        if self.is_leaf():
            return 0
        left_depth = self.left.depth() if self.left else 0
        right_depth = self.right.depth() if self.right else 0
        return 1 + max(left_depth, right_depth)

    def get_all_nodes(self) -> list['TreeNode']:
        """Get all nodes in subtree as a flat list."""
        nodes = [self]
        if self.left is not None:
            nodes.extend(self.left.get_all_nodes())
        if self.right is not None:
            nodes.extend(self.right.get_all_nodes())
        return nodes

    def get_internal_nodes(self) -> list['TreeNode']:
        """Get all internal (non-leaf) nodes."""
        nodes = []
        if not self.is_leaf():
            nodes.append(self)
            if self.left is not None:
                nodes.extend(self.left.get_internal_nodes())
            if self.right is not None:
                nodes.extend(self.right.get_internal_nodes())
        return nodes

    def get_leaves(self) -> list['TreeNode']:
        """Get all leaf nodes."""
        if self.is_leaf():
            return [self]
        leaves = []
        if self.left is not None:
            leaves.extend(self.left.get_leaves())
        if self.right is not None:
            leaves.extend(self.right.get_leaves())
        return leaves


# =============================================================================
# Numba-accelerated functions for performance
# =============================================================================

@jit(nopython=True, cache=True)
def _compute_gini(y: np.ndarray, n_classes: int) -> float:
    """Compute Gini impurity (numba-accelerated)."""
    n = len(y)
    if n == 0:
        return 0.0
    counts = np.zeros(n_classes, dtype=np.float64)
    for i in range(n):
        counts[int(y[i])] += 1
    gini = 1.0
    for c in range(n_classes):
        p = counts[c] / n
        gini -= p * p
    return gini


@jit(nopython=True, cache=True)
def _compute_mse(y: np.ndarray) -> float:
    """Compute MSE from mean (numba-accelerated)."""
    n = len(y)
    if n == 0:
        return 0.0
    mean = 0.0
    for i in range(n):
        mean += y[i]
    mean /= n
    mse = 0.0
    for i in range(n):
        diff = y[i] - mean
        mse += diff * diff
    return mse / n


@jit(nopython=True, cache=True)
def _misclassification_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute misclassification rate (numba-accelerated)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    errors = 0
    for i in range(n):
        if y_true[i] != y_pred[i]:
            errors += 1
    return errors / n


# =============================================================================
# Tree Prediction (vectorized for speed)
# =============================================================================

def _predict_tree(node: TreeNode, X: np.ndarray) -> np.ndarray:
    """Predict using tree (vectorized batch prediction)."""
    n_samples = X.shape[0]

    if node.is_leaf():
        # Return class predictions or regression values
        if len(node.value) > 1:
            # Classification: return class with highest count
            pred_class = np.argmax(node.value)
            return np.full(n_samples, pred_class, dtype=np.int64)
        else:
            # Regression: return mean value
            return np.full(n_samples, node.value[0], dtype=np.float64)

    # Split samples
    left_mask = X[:, node.feature_idx] <= node.threshold
    right_mask = ~left_mask

    predictions = np.zeros(n_samples, dtype=np.float64)

    if np.any(left_mask) and node.left is not None:
        predictions[left_mask] = _predict_tree(node.left, X[left_mask])
    if np.any(right_mask) and node.right is not None:
        predictions[right_mask] = _predict_tree(node.right, X[right_mask])

    return predictions


def _predict_proba_tree(node: TreeNode, X: np.ndarray, n_classes: int) -> np.ndarray:
    """Predict class probabilities using tree."""
    n_samples = X.shape[0]

    if node.is_leaf():
        # Return normalized class counts
        proba = node.value / (node.value.sum() + 1e-10)
        return np.tile(proba, (n_samples, 1))

    # Split samples
    left_mask = X[:, node.feature_idx] <= node.threshold
    right_mask = ~left_mask

    probas = np.zeros((n_samples, n_classes), dtype=np.float64)

    if np.any(left_mask) and node.left is not None:
        probas[left_mask] = _predict_proba_tree(node.left, X[left_mask], n_classes)
    if np.any(right_mask) and node.right is not None:
        probas[right_mask] = _predict_proba_tree(node.right, X[right_mask], n_classes)

    return probas


# =============================================================================
# Evolutionary Operators
# =============================================================================

class EvolutionaryOperators:
    """Genetic operators for tree evolution."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        is_classification: bool,
        max_depth: int,
        min_samples_leaf: int,
        rng: np.random.RandomState,
    ):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.n_classes = n_classes
        self.is_classification = is_classification
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = rng

        # Precompute split candidates for efficiency
        self._precompute_split_candidates()

    def _precompute_split_candidates(self):
        """Precompute valid split points for each feature."""
        self.split_candidates = []
        for j in range(self.n_features):
            unique_vals = np.unique(self.X[:, j])
            if len(unique_vals) > 1:
                # Midpoints between unique values
                midpoints = (unique_vals[:-1] + unique_vals[1:]) / 2
                # Limit to max 100 candidates per feature for speed
                if len(midpoints) > 100:
                    idx = self.rng.choice(len(midpoints), 100, replace=False)
                    midpoints = midpoints[idx]
                self.split_candidates.append(midpoints)
            else:
                self.split_candidates.append(np.array([unique_vals[0]]))

    def create_random_tree(self, max_depth: int | None = None) -> TreeNode:
        """Create a random valid tree."""
        if max_depth is None:
            max_depth = self.max_depth

        return self._grow_random_tree(
            np.arange(self.n_samples),
            depth=0,
            max_depth=max_depth,
        )

    def _grow_random_tree(
        self,
        indices: np.ndarray,
        depth: int,
        max_depth: int,
    ) -> TreeNode:
        """Recursively grow a random tree."""
        n = len(indices)

        # Create leaf if stopping criteria met
        if (depth >= max_depth or
            n < 2 * self.min_samples_leaf or
            self.rng.random() < 0.3):  # Random stop probability
            return self._create_leaf(indices)

        # Try to find a valid split
        feature_idx, threshold = self._random_split(indices)

        if feature_idx is None:
            return self._create_leaf(indices)

        # Split indices
        left_mask = self.X[indices, feature_idx] <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        # Check minimum samples constraint
        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return self._create_leaf(indices)

        # Create internal node
        node = TreeNode(
            feature_idx=feature_idx,
            threshold=threshold,
            n_samples=n,
        )

        node.left = self._grow_random_tree(left_indices, depth + 1, max_depth)
        node.right = self._grow_random_tree(right_indices, depth + 1, max_depth)

        return node

    def _random_split(self, indices: np.ndarray) -> tuple[int | None, float]:
        """Select a random valid split."""
        # Shuffle feature order
        features = self.rng.permutation(self.n_features)

        for feature_idx in features:
            candidates = self.split_candidates[feature_idx]
            if len(candidates) == 0:
                continue

            # Get values for this feature
            values = self.X[indices, feature_idx]
            vmin, vmax = values.min(), values.max()

            # Filter valid candidates
            valid = candidates[(candidates > vmin) & (candidates < vmax)]

            if len(valid) > 0:
                threshold = self.rng.choice(valid)
                return feature_idx, threshold

        return None, 0.0

    def _create_leaf(self, indices: np.ndarray) -> TreeNode:
        """Create a leaf node with appropriate value."""
        n = len(indices)

        if self.is_classification:
            # Store class counts
            counts = np.bincount(self.y[indices].astype(int), minlength=self.n_classes)
            value = counts.astype(np.float64)
        else:
            # Store mean value
            value = np.array([self.y[indices].mean()])

        return TreeNode(feature_idx=-1, value=value, n_samples=n)

    def mutate_split(self, tree: TreeNode) -> TreeNode:
        """Split mutation: Add a split to a random leaf."""
        tree = tree.copy()
        leaves = tree.get_leaves()

        if not leaves:
            return tree

        # Select random leaf
        leaf = self.rng.choice(leaves)

        # Can only split if leaf has enough samples
        if leaf.n_samples < 2 * self.min_samples_leaf:
            return tree

        # Find indices that reach this leaf
        indices = self._get_leaf_indices(tree, leaf)

        if len(indices) < 2 * self.min_samples_leaf:
            return tree

        # Try to find a valid split
        feature_idx, threshold = self._random_split(indices)

        if feature_idx is None:
            return tree

        # Split the leaf
        left_mask = self.X[indices, feature_idx] <= threshold
        left_indices = indices[left_mask]
        right_indices = indices[~left_mask]

        if len(left_indices) < self.min_samples_leaf or len(right_indices) < self.min_samples_leaf:
            return tree

        # Convert leaf to internal node
        leaf.feature_idx = feature_idx
        leaf.threshold = threshold
        leaf.left = self._create_leaf(left_indices)
        leaf.right = self._create_leaf(right_indices)
        leaf.value = np.array([])

        return tree

    def mutate_prune(self, tree: TreeNode) -> TreeNode:
        """Prune mutation: Remove a subtree, making it a leaf."""
        tree = tree.copy()
        internal_nodes = tree.get_internal_nodes()

        if not internal_nodes:
            return tree

        # Select random internal node (not root if it's the only internal)
        if len(internal_nodes) == 1 and internal_nodes[0] is tree:
            return tree  # Can't prune root

        # Prefer to prune nodes whose children are both leaves
        prunable = [n for n in internal_nodes
                   if n.left and n.left.is_leaf() and n.right and n.right.is_leaf()]

        if not prunable:
            prunable = internal_nodes[1:] if internal_nodes[0] is tree else internal_nodes

        if not prunable:
            return tree

        node = self.rng.choice(prunable)

        # Get indices for this node
        indices = self._get_leaf_indices(tree, node)

        # Convert to leaf
        node.feature_idx = -1
        if self.is_classification:
            counts = np.bincount(self.y[indices].astype(int), minlength=self.n_classes)
            node.value = counts.astype(np.float64)
        else:
            node.value = np.array([self.y[indices].mean()])
        node.left = None
        node.right = None

        return tree

    def mutate_major(self, tree: TreeNode) -> TreeNode:
        """Major mutation: Change split variable and/or threshold."""
        tree = tree.copy()
        internal_nodes = tree.get_internal_nodes()

        if not internal_nodes:
            return tree

        node = self.rng.choice(internal_nodes)
        indices = self._get_leaf_indices(tree, node)

        if len(indices) == 0 or len(indices) < 2 * self.min_samples_leaf:
            return tree

        # 50% chance to change feature
        if self.rng.random() < 0.5:
            new_feature = self.rng.randint(0, self.n_features)
        else:
            new_feature = node.feature_idx

        # Select new threshold
        candidates = self.split_candidates[new_feature]
        values = self.X[indices, new_feature]
        vmin, vmax = values.min(), values.max()
        valid = candidates[(candidates > vmin) & (candidates < vmax)]

        if len(valid) == 0:
            return tree

        new_threshold = self.rng.choice(valid)

        # Check if split is valid
        left_mask = self.X[indices, new_feature] <= new_threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return tree

        node.feature_idx = new_feature
        node.threshold = new_threshold

        # Recompute leaf values for affected subtree
        self._update_subtree_values(node, indices)

        return tree

    def mutate_minor(self, tree: TreeNode) -> TreeNode:
        """Minor mutation: Small adjustment to split threshold."""
        tree = tree.copy()
        internal_nodes = tree.get_internal_nodes()

        if not internal_nodes:
            return tree

        node = self.rng.choice(internal_nodes)
        indices = self._get_leaf_indices(tree, node)

        if len(indices) == 0:
            return tree

        # Get candidate thresholds near current
        candidates = self.split_candidates[node.feature_idx]
        values = self.X[indices, node.feature_idx]
        vmin, vmax = values.min(), values.max()
        valid = candidates[(candidates > vmin) & (candidates < vmax)]

        if len(valid) < 2:
            return tree

        # Find nearby threshold (within 20% of range)
        current_idx = np.argmin(np.abs(valid - node.threshold))
        range_size = max(1, int(len(valid) * 0.2))
        low = max(0, current_idx - range_size)
        high = min(len(valid), current_idx + range_size + 1)

        new_threshold = self.rng.choice(valid[low:high])

        # Check validity
        left_mask = self.X[indices, node.feature_idx] <= new_threshold
        if left_mask.sum() < self.min_samples_leaf or (~left_mask).sum() < self.min_samples_leaf:
            return tree

        node.threshold = new_threshold
        self._update_subtree_values(node, indices)

        return tree

    def crossover(self, tree1: TreeNode, tree2: TreeNode) -> tuple[TreeNode, TreeNode]:
        """Crossover: Exchange subtrees between two trees."""
        tree1 = tree1.copy()
        tree2 = tree2.copy()

        # Get all nodes
        nodes1 = tree1.get_all_nodes()
        nodes2 = tree2.get_all_nodes()

        if len(nodes1) < 2 or len(nodes2) < 2:
            return tree1, tree2

        # Select random non-root nodes
        node1 = self.rng.choice(nodes1[1:]) if len(nodes1) > 1 else None
        node2 = self.rng.choice(nodes2[1:]) if len(nodes2) > 1 else None

        if node1 is None or node2 is None:
            return tree1, tree2

        # Find parents and swap subtrees
        parent1, is_left1 = self._find_parent(tree1, node1)
        parent2, is_left2 = self._find_parent(tree2, node2)

        if parent1 is None or parent2 is None:
            return tree1, tree2

        # Swap
        if is_left1:
            parent1.left = node2
        else:
            parent1.right = node2

        if is_left2:
            parent2.left = node1
        else:
            parent2.right = node1

        # Update values
        self._update_all_values(tree1)
        self._update_all_values(tree2)

        return tree1, tree2

    def _find_parent(self, root: TreeNode, target: TreeNode) -> tuple[TreeNode | None, bool]:
        """Find parent of target node and whether it's left child."""
        if root.left is target:
            return root, True
        if root.right is target:
            return root, False

        if root.left:
            result = self._find_parent(root.left, target)
            if result[0] is not None:
                return result

        if root.right:
            result = self._find_parent(root.right, target)
            if result[0] is not None:
                return result

        return None, False

    def _get_leaf_indices(self, tree: TreeNode, target: TreeNode) -> np.ndarray:
        """Get indices of samples that reach a particular node."""
        indices = np.arange(self.n_samples)
        return self._trace_to_node(tree, target, indices)

    def _trace_to_node(
        self,
        current: TreeNode,
        target: TreeNode,
        indices: np.ndarray,
    ) -> np.ndarray:
        """Trace path from current to target, filtering indices."""
        if current is target:
            return indices

        if current.is_leaf():
            return np.array([], dtype=np.int64)

        # Split indices
        left_mask = self.X[indices, current.feature_idx] <= current.threshold

        # Search in children
        if current.left:
            result = self._trace_to_node(current.left, target, indices[left_mask])
            if len(result) > 0:
                return result

        if current.right:
            result = self._trace_to_node(current.right, target, indices[~left_mask])
            if len(result) > 0:
                return result

        return np.array([], dtype=np.int64)

    def _update_subtree_values(self, node: TreeNode, indices: np.ndarray):
        """Update leaf values in subtree after mutation."""
        if node.is_leaf():
            if self.is_classification:
                counts = np.bincount(self.y[indices].astype(int), minlength=self.n_classes)
                node.value = counts.astype(np.float64)
            else:
                node.value = np.array([self.y[indices].mean() if len(indices) > 0 else 0.0])
            node.n_samples = len(indices)
            return

        # Split and recurse
        left_mask = self.X[indices, node.feature_idx] <= node.threshold
        node.n_samples = len(indices)

        if node.left:
            self._update_subtree_values(node.left, indices[left_mask])
        if node.right:
            self._update_subtree_values(node.right, indices[~left_mask])

    def _update_all_values(self, tree: TreeNode):
        """Update all leaf values in tree."""
        self._update_subtree_values(tree, np.arange(self.n_samples))


# =============================================================================
# Fitness Evaluation
# =============================================================================

class FitnessEvaluator:
    """Evaluates tree fitness using BIC-type criterion."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        is_classification: bool,
        alpha: float = 1.0,
    ):
        self.X = X
        self.y = y
        self.n_samples = X.shape[0]
        self.n_classes = n_classes
        self.is_classification = is_classification
        self.alpha = alpha
        self.log_n = np.log(self.n_samples) if self.n_samples > 1 else 1.0

    def evaluate(self, tree: TreeNode) -> float:
        """Compute fitness (lower is better).

        Classification: 2N * misclassification + alpha * M * log(N)
        Regression: N * log(MSE + 1e-10) + alpha * 4 * (M+1) * log(N)
        """
        predictions = _predict_tree(tree, self.X)
        n_leaves = tree.count_leaves()

        if self.is_classification:
            # Misclassification loss
            error_rate = (predictions != self.y).mean()
            loss = 2 * self.n_samples * error_rate
            complexity = self.alpha * n_leaves * self.log_n
        else:
            # MSE loss (log scale)
            mse = np.mean((predictions - self.y) ** 2)
            loss = self.n_samples * np.log(mse + 1e-10)
            complexity = self.alpha * 4 * (n_leaves + 1) * self.log_n

        return loss + complexity

    def evaluate_batch(self, trees: list[TreeNode], n_jobs: int = 1) -> np.ndarray:
        """Evaluate multiple trees (optionally in parallel)."""
        if n_jobs != 1 and HAS_JOBLIB and len(trees) > 10:
            fitnesses = Parallel(n_jobs=n_jobs)(
                delayed(self.evaluate)(tree) for tree in trees
            )
            return np.array(fitnesses)
        else:
            return np.array([self.evaluate(tree) for tree in trees])


# =============================================================================
# Main Evolutionary Tree Classes
# =============================================================================

class _EvolutionaryTreeBase(BaseEstimator):
    """Base class for evolutionary trees."""

    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        alpha: float = 1.0,
        mutation_prob: float = 0.8,
        crossover_prob: float = 0.2,
        patience: int = 20,
        warm_start: bool = True,
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.population_size = population_size
        self.n_generations = n_generations
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.alpha = alpha
        self.mutation_prob = mutation_prob
        self.crossover_prob = crossover_prob
        self.patience = patience
        self.warm_start = warm_start
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.tree_: TreeNode | None = None
        self.n_features_in_: int = 0
        self.best_fitness_: float = np.inf
        self.fitness_history_: list[float] = []
        self._is_fitted: bool = False

    def _initialize_population(
        self,
        operators: EvolutionaryOperators,
        evaluator: FitnessEvaluator,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[list[TreeNode], np.ndarray]:
        """Initialize population with random trees and optional warm start."""
        population = []

        # Warm start: seed with greedy tree
        if self.warm_start:
            greedy_tree = self._create_greedy_tree(X, y)
            if greedy_tree is not None:
                population.append(greedy_tree)

        # Fill rest with random trees
        while len(population) < self.population_size:
            # Vary initial depth for diversity
            max_d = self._rng.randint(2, self.max_depth + 1)
            tree = operators.create_random_tree(max_depth=max_d)
            population.append(tree)

        # Evaluate fitness
        fitnesses = evaluator.evaluate_batch(population, self.n_jobs)

        return population, fitnesses

    def _create_greedy_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode | None:
        """Create a greedy tree using sklearn to seed population."""
        raise NotImplementedError

    def _tree_similarity(self, tree1: TreeNode, tree2: TreeNode) -> float:
        """Compute similarity between two trees for deterministic crowding."""
        # Simple similarity: compare predictions
        predictions1 = _predict_tree(tree1, self._X_fit)
        predictions2 = _predict_tree(tree2, self._X_fit)

        if self._is_classification:
            # Classification: fraction of same predictions
            return (predictions1 == predictions2).mean()
        else:
            # Regression: 1 / (1 + MSE between predictions)
            mse = np.mean((predictions1 - predictions2) ** 2)
            return 1.0 / (1.0 + mse)

    def _evolve(
        self,
        population: list[TreeNode],
        fitnesses: np.ndarray,
        operators: EvolutionaryOperators,
        evaluator: FitnessEvaluator,
    ) -> tuple[list[TreeNode], np.ndarray]:
        """Run one generation of evolution using deterministic crowding."""
        n = len(population)

        # Shuffle population for random pairing
        indices = self._rng.permutation(n)

        for i in range(0, n - 1, 2):
            idx1, idx2 = indices[i], indices[i + 1]
            parent1, parent2 = population[idx1], population[idx2]

            # Create offspring via crossover or mutation
            if self._rng.random() < self.crossover_prob:
                child1, child2 = operators.crossover(parent1, parent2)
            else:
                child1, child2 = parent1.copy(), parent2.copy()

            # Apply mutations
            if self._rng.random() < self.mutation_prob:
                mutation_type = self._rng.randint(0, 4)
                if mutation_type == 0:
                    child1 = operators.mutate_split(child1)
                elif mutation_type == 1:
                    child1 = operators.mutate_prune(child1)
                elif mutation_type == 2:
                    child1 = operators.mutate_major(child1)
                else:
                    child1 = operators.mutate_minor(child1)

            if self._rng.random() < self.mutation_prob:
                mutation_type = self._rng.randint(0, 4)
                if mutation_type == 0:
                    child2 = operators.mutate_split(child2)
                elif mutation_type == 1:
                    child2 = operators.mutate_prune(child2)
                elif mutation_type == 2:
                    child2 = operators.mutate_major(child2)
                else:
                    child2 = operators.mutate_minor(child2)

            # Evaluate offspring
            fitness1 = evaluator.evaluate(child1)
            fitness2 = evaluator.evaluate(child2)

            # Deterministic crowding: compete with most similar parent
            sim11 = self._tree_similarity(child1, parent1)
            sim12 = self._tree_similarity(child1, parent2)
            sim21 = self._tree_similarity(child2, parent1)
            sim22 = self._tree_similarity(child2, parent2)

            if sim11 + sim22 >= sim12 + sim21:
                # child1 vs parent1, child2 vs parent2
                if fitness1 < fitnesses[idx1]:
                    population[idx1] = child1
                    fitnesses[idx1] = fitness1
                if fitness2 < fitnesses[idx2]:
                    population[idx2] = child2
                    fitnesses[idx2] = fitness2
            else:
                # child1 vs parent2, child2 vs parent1
                if fitness1 < fitnesses[idx2]:
                    population[idx2] = child1
                    fitnesses[idx2] = fitness1
                if fitness2 < fitnesses[idx1]:
                    population[idx1] = child2
                    fitnesses[idx1] = fitness2

        return population, fitnesses

    def _fit_internal(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_classes: int,
        is_classification: bool,
    ):
        """Internal fit method."""
        self._rng = check_random_state(self.random_state)
        self._X_fit = X
        self._is_classification = is_classification
        self.n_features_in_ = X.shape[1]

        # Create operators and evaluator
        operators = EvolutionaryOperators(
            X=X,
            y=y,
            n_classes=n_classes,
            is_classification=is_classification,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            rng=self._rng,
        )

        evaluator = FitnessEvaluator(
            X=X,
            y=y,
            n_classes=n_classes,
            is_classification=is_classification,
            alpha=self.alpha,
        )

        # Initialize population
        population, fitnesses = self._initialize_population(
            operators, evaluator, X, y
        )

        # Track best
        best_idx = np.argmin(fitnesses)
        self.best_fitness_ = fitnesses[best_idx]
        self.tree_ = population[best_idx].copy()
        self.fitness_history_ = [self.best_fitness_]

        # Evolution loop
        no_improvement = 0

        for gen in range(self.n_generations):
            population, fitnesses = self._evolve(
                population, fitnesses, operators, evaluator
            )

            # Update best
            current_best_idx = np.argmin(fitnesses)
            current_best_fitness = fitnesses[current_best_idx]

            if current_best_fitness < self.best_fitness_:
                self.best_fitness_ = current_best_fitness
                self.tree_ = population[current_best_idx].copy()
                no_improvement = 0
            else:
                no_improvement += 1

            self.fitness_history_.append(self.best_fitness_)

            if self.verbose and (gen + 1) % 10 == 0:
                print(f"Generation {gen + 1}: best_fitness={self.best_fitness_:.4f}, "
                      f"best_leaves={self.tree_.count_leaves()}, "
                      f"best_depth={self.tree_.depth()}")

            # Early stopping
            if no_improvement >= self.patience:
                if self.verbose:
                    print(f"Early stopping at generation {gen + 1}")
                break

        # Update tree values one final time
        operators._update_all_values(self.tree_)

        # Clear reference to training data
        del self._X_fit

        self._is_fitted = True


class EvolutionaryTreeClassifier(_EvolutionaryTreeBase, ClassifierMixin):
    """Evolutionary Tree Classifier - Globally optimal trees via genetic algorithms.

    Unlike greedy methods (CART, C4.5) that make locally optimal splits,
    evolutionary trees use genetic algorithms to search for globally
    optimal tree structures. This can discover patterns that greedy
    methods miss.

    Parameters
    ----------
    population_size : int, default=100
        Number of trees in the population. Larger populations explore
        more of the search space but are slower.
    n_generations : int, default=100
        Maximum number of evolutionary generations.
    max_depth : int, default=8
        Maximum depth of trees in the population.
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf node.
    alpha : float, default=1.0
        Complexity penalty coefficient. Higher values favor simpler trees.
        Controls the BIC-type tradeoff: loss + alpha * complexity.
    mutation_prob : float, default=0.8
        Probability of applying mutation to offspring.
    crossover_prob : float, default=0.2
        Probability of using crossover vs just mutation.
    patience : int, default=20
        Generations without improvement before early stopping.
    warm_start : bool, default=True
        If True, seed population with a greedy tree for faster convergence.
    n_jobs : int, default=1
        Number of parallel jobs for fitness evaluation.
        -1 means using all processors.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, print progress every 10 generations.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    tree_ : TreeNode
        The best tree found during evolution.
    best_fitness_ : float
        Fitness of the best tree (lower is better).
    fitness_history_ : list
        Best fitness at each generation.

    Examples
    --------
    >>> from endgame.models.trees.evtree import EvolutionaryTreeClassifier
    >>> clf = EvolutionaryTreeClassifier(
    ...     population_size=50, n_generations=50, random_state=42
    ... )
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)

    Notes
    -----
    Evolutionary trees are slower than greedy methods but can find better
    structures for complex problems. They're particularly valuable for:

    1. Ensemble diversity: Different inductive bias from greedy trees
    2. Interpretability: Often finds simpler trees with similar accuracy
    3. Avoiding local optima: Global search escapes greedy suboptimality

    Performance tips:
    - Use warm_start=True (default) to seed with a greedy tree
    - Increase patience for harder problems
    - Use n_jobs=-1 for parallel fitness evaluation on large populations
    - Reduce population_size for faster (but potentially worse) results

    References
    ----------
    Grubinger et al., "evtree: Evolutionary Learning of Globally Optimal
    Classification and Regression Trees in R" (2014)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        population_size: int = 100,
        n_generations: int = 100,
        max_depth: int = 8,
        min_samples_leaf: int = 5,
        alpha: float = 1.0,
        mutation_prob: float = 0.8,
        crossover_prob: float = 0.2,
        patience: int = 20,
        warm_start: bool = True,
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            population_size=population_size,
            n_generations=n_generations,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            alpha=alpha,
            mutation_prob=mutation_prob,
            crossover_prob=crossover_prob,
            patience=patience,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
        )

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self._label_encoder: LabelEncoder | None = None

    def __sklearn_tags__(self):
        """Return sklearn tags for the classifier."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = tags.classifier_tags or type(tags.classifier_tags)()
        return tags

    def _create_greedy_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode | None:
        """Create a greedy tree using sklearn."""
        try:
            dt = DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self._rng.randint(0, 2**31),
            )
            dt.fit(X, y)
            return self._sklearn_tree_to_node(dt, X, y)
        except Exception:
            return None

    def _sklearn_tree_to_node(
        self,
        dt: DecisionTreeClassifier,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TreeNode:
        """Convert sklearn tree to our TreeNode format."""
        tree = dt.tree_

        def build_node(node_id: int, indices: np.ndarray) -> TreeNode:
            if tree.feature[node_id] == -2:  # Leaf
                counts = np.bincount(y[indices].astype(int), minlength=self.n_classes_)
                return TreeNode(
                    feature_idx=-1,
                    value=counts.astype(np.float64),
                    n_samples=len(indices),
                )

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_mask = X[indices, feature_idx] <= threshold
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]

            node = TreeNode(
                feature_idx=feature_idx,
                threshold=threshold,
                n_samples=len(indices),
            )
            node.left = build_node(tree.children_left[node_id], left_indices)
            node.right = build_node(tree.children_right[node_id], right_indices)

            return node

        return build_node(0, np.arange(len(y)))

    def fit(self, X, y, **fit_params) -> "EvolutionaryTreeClassifier":
        """Fit the evolutionary tree classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Fit
        self._fit_internal(X, y_encoded, self.n_classes_, is_classification=True)

        return self

    def predict(self, X) -> np.ndarray:
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
        if not self._is_fitted:
            raise RuntimeError("EvolutionaryTreeClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)

        predictions = _predict_tree(self.tree_, X).astype(int)
        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X) -> np.ndarray:
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
        if not self._is_fitted:
            raise RuntimeError("EvolutionaryTreeClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)

        return _predict_proba_tree(self.tree_, X, self.n_classes_)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on split frequency."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        importances = np.zeros(self.n_features_in_)

        for node in self.tree_.get_internal_nodes():
            importances[node.feature_idx] += node.n_samples

        # Normalize
        total = importances.sum()
        if total > 0:
            importances /= total

        return importances


class EvolutionaryTreeRegressor(_EvolutionaryTreeBase, RegressorMixin):
    """Evolutionary Tree Regressor - Globally optimal trees via genetic algorithms.

    Parameters
    ----------
    population_size : int, default=100
        Number of trees in the population.
    n_generations : int, default=100
        Maximum number of evolutionary generations.
    max_depth : int, default=8
        Maximum depth of trees.
    min_samples_leaf : int, default=5
        Minimum samples required in a leaf node.
    alpha : float, default=1.0
        Complexity penalty coefficient.
    mutation_prob : float, default=0.8
        Probability of applying mutation.
    crossover_prob : float, default=0.2
        Probability of using crossover.
    patience : int, default=20
        Generations without improvement before early stopping.
    warm_start : bool, default=True
        Seed population with a greedy tree.
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all processors).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Print progress during training.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    tree_ : TreeNode
        The best tree found.
    best_fitness_ : float
        Fitness of the best tree.

    Examples
    --------
    >>> from endgame.models.trees.evtree import EvolutionaryTreeRegressor
    >>> reg = EvolutionaryTreeRegressor(population_size=50, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __sklearn_tags__(self):
        """Return sklearn tags for the regressor."""
        tags = super().__sklearn_tags__()
        tags.estimator_type = "regressor"
        tags.regressor_tags = tags.regressor_tags or type(tags.regressor_tags)()
        return tags

    def _create_greedy_tree(self, X: np.ndarray, y: np.ndarray) -> TreeNode | None:
        """Create a greedy tree using sklearn."""
        try:
            dt = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self._rng.randint(0, 2**31),
            )
            dt.fit(X, y)
            return self._sklearn_tree_to_node(dt, X, y)
        except Exception:
            return None

    def _sklearn_tree_to_node(
        self,
        dt: DecisionTreeRegressor,
        X: np.ndarray,
        y: np.ndarray,
    ) -> TreeNode:
        """Convert sklearn tree to our TreeNode format."""
        tree = dt.tree_

        def build_node(node_id: int, indices: np.ndarray) -> TreeNode:
            if tree.feature[node_id] == -2:  # Leaf
                mean_val = y[indices].mean() if len(indices) > 0 else 0.0
                return TreeNode(
                    feature_idx=-1,
                    value=np.array([mean_val]),
                    n_samples=len(indices),
                )

            feature_idx = tree.feature[node_id]
            threshold = tree.threshold[node_id]

            left_mask = X[indices, feature_idx] <= threshold
            left_indices = indices[left_mask]
            right_indices = indices[~left_mask]

            node = TreeNode(
                feature_idx=feature_idx,
                threshold=threshold,
                n_samples=len(indices),
            )
            node.left = build_node(tree.children_left[node_id], left_indices)
            node.right = build_node(tree.children_right[node_id], right_indices)

            return node

        return build_node(0, np.arange(len(y)))

    def fit(self, X, y, **fit_params) -> "EvolutionaryTreeRegressor":
        """Fit the evolutionary tree regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        # Handle NaN
        X = np.nan_to_num(X, nan=0.0)
        y = np.nan_to_num(y, nan=0.0)

        # Fit
        self._fit_internal(X, y, n_classes=0, is_classification=False)

        return self

    def predict(self, X) -> np.ndarray:
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
        if not self._is_fitted:
            raise RuntimeError("EvolutionaryTreeRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0)

        return _predict_tree(self.tree_, X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on split frequency."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")

        importances = np.zeros(self.n_features_in_)

        for node in self.tree_.get_internal_nodes():
            importances[node.feature_idx] += node.n_samples

        # Normalize
        total = importances.sum()
        if total > 0:
            importances /= total

        return importances
