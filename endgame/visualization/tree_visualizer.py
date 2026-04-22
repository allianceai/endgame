"""Interactive Decision Tree Visualization.

Generates gorgeous, self-contained HTML/JavaScript visualizations of decision
trees with expandable/collapsible branches, zoom, pan, and rich node tooltips.

Supports:
- sklearn DecisionTreeClassifier/Regressor
- C5.0 (C50Classifier, C50Ensemble)
- ObliqueDecisionTree (linear combination splits)
- EvolutionaryTree
- Any tree model with a `tree_` attribute following sklearn conventions

Example
-------
>>> from sklearn.tree import DecisionTreeClassifier
>>> from endgame.visualization import TreeVisualizer
>>>
>>> clf = DecisionTreeClassifier(max_depth=4).fit(X, y)
>>> viz = TreeVisualizer(clf, feature_names=['age', 'income', 'score'])
>>> viz.save("tree.html")  # Open in browser for interactive visualization
"""

from __future__ import annotations

import html as html_module
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Universal tree node representation for visualization
# ---------------------------------------------------------------------------

@dataclass
class VizNode:
    """Universal tree node for visualization.

    All tree types are converted to this format before rendering.
    """
    node_id: int = 0
    is_leaf: bool = True

    # Split info (internal nodes)
    split_label: str = ""          # e.g. "age <= 30.5" or "0.3*x1 + 0.7*x3 <= 2.1"
    split_feature: str = ""        # Primary feature name (for coloring)
    split_type: str = "threshold"  # "threshold", "oblique", "subset", "discrete"

    # Node statistics
    n_samples: int = 0
    impurity: float = 0.0
    impurity_name: str = "gini"    # "gini", "entropy", "mse"

    # Prediction info
    predicted_class: str = ""
    predicted_value: float | None = None
    class_distribution: list[float] = field(default_factory=list)
    class_names: list[str] = field(default_factory=list)

    # Children
    children: list[VizNode] = field(default_factory=list)
    child_labels: list[str] = field(default_factory=list)  # Edge labels

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d: dict[str, Any] = {
            "id": int(self.node_id),
            "leaf": bool(self.is_leaf),
            "split": str(self.split_label),
            "feature": str(self.split_feature),
            "splitType": str(self.split_type),
            "samples": int(self.n_samples),
            "impurity": round(float(self.impurity), 4),
            "impurityName": str(self.impurity_name),
            "prediction": str(self.predicted_class),
            "classDist": [round(float(v), 4) for v in self.class_distribution],
            "classNames": [str(n) for n in self.class_names],
            "children": [c.to_dict() for c in self.children],
            "childLabels": [str(l) for l in self.child_labels],
        }
        if self.predicted_value is not None:
            d["predValue"] = round(float(self.predicted_value), 4)
        return d


# ---------------------------------------------------------------------------
# Tree extraction adapters
# ---------------------------------------------------------------------------

def _extract_sklearn_tree(model, feature_names, class_names, is_classifier):
    """Extract tree structure from a sklearn DecisionTreeClassifier/Regressor."""
    tree = model.tree_
    n_nodes = tree.node_count
    features = tree.feature
    thresholds = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right
    values = tree.value
    n_samples_arr = tree.n_node_samples
    impurities = tree.impurity

    criterion = getattr(model, "criterion", "gini")

    def build_node(node_idx):
        is_leaf = bool(children_left[node_idx] == children_right[node_idx])
        n_samp = int(n_samples_arr[node_idx])
        imp = float(impurities[node_idx])

        node = VizNode(
            node_id=node_idx,
            is_leaf=is_leaf,
            n_samples=n_samp,
            impurity=imp,
            impurity_name=criterion,
        )

        if is_classifier:
            class_counts = values[node_idx].flatten()
            total = class_counts.sum()
            dist = (class_counts / total).tolist() if total > 0 else []
            pred_idx = int(np.argmax(class_counts))
            node.class_distribution = dist
            node.class_names = list(class_names) if class_names else [str(i) for i in range(len(class_counts))]
            node.predicted_class = node.class_names[pred_idx] if node.class_names else str(pred_idx)
        else:
            val = float(values[node_idx].flatten()[0])
            node.predicted_value = val
            node.predicted_class = f"{val:.4f}"

        if not is_leaf:
            feat_idx = int(features[node_idx])
            thresh = float(thresholds[node_idx])
            feat_name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"feature_{feat_idx}"
            node.split_label = f"{feat_name} \u2264 {thresh:.4g}"
            node.split_feature = feat_name
            node.split_type = "threshold"

            left_child = build_node(int(children_left[node_idx]))
            right_child = build_node(int(children_right[node_idx]))
            node.children = [left_child, right_child]
            node.child_labels = ["True", "False"]

        return node

    return build_node(0)


def _extract_c50_tree(tree_node, feature_names, class_names, node_counter=None):
    """Extract tree structure from a C5.0 TreeNode."""
    if node_counter is None:
        node_counter = [0]

    node_id = node_counter[0]
    node_counter[0] += 1

    n_samp = int(tree_node.cases) if tree_node.cases else 0
    dist = tree_node.class_dist.tolist() if hasattr(tree_node, 'class_dist') and len(tree_node.class_dist) > 0 else []
    total = sum(dist) if dist else 0
    norm_dist = [d / total for d in dist] if total > 0 else dist

    node = VizNode(
        node_id=node_id,
        is_leaf=tree_node.is_leaf,
        n_samples=n_samp,
        impurity=float(tree_node.errors) if hasattr(tree_node, 'errors') else 0.0,
        impurity_name="errors",
        class_distribution=norm_dist,
        class_names=list(class_names) if class_names else [str(i) for i in range(len(dist))],
        predicted_class=class_names[tree_node.class_] if class_names and tree_node.class_ < len(class_names) else str(tree_node.class_),
    )

    if not tree_node.is_leaf and tree_node.branches:
        feat_idx = tree_node.tested_attr
        feat_name = feature_names[feat_idx] if feature_names and feat_idx is not None and feat_idx < len(feature_names) else f"feature_{feat_idx}"
        node.split_feature = feat_name

        from endgame.models.trees.c50 import NodeType
        if tree_node.node_type == NodeType.THRESHOLD:
            thresh = tree_node.threshold
            node.split_label = f"{feat_name} \u2264 {thresh:.4g}"
            node.split_type = "threshold"
            node.child_labels = ["True", "False"]
        elif tree_node.node_type == NodeType.DISCRETE:
            node.split_label = f"{feat_name}"
            node.split_type = "discrete"
            node.child_labels = [f"= {i}" for i in range(len(tree_node.branches))]
        elif tree_node.node_type == NodeType.SUBSET:
            node.split_label = f"{feat_name} \u2208 subset"
            node.split_type = "subset"
            node.child_labels = [f"subset {i}" for i in range(len(tree_node.branches))]
        else:
            node.split_label = feat_name
            node.split_type = "threshold"

        for branch in tree_node.branches:
            child = _extract_c50_tree(branch, feature_names, class_names, node_counter)
            node.children.append(child)

        if len(node.child_labels) < len(node.children):
            node.child_labels.extend(["" for _ in range(len(node.children) - len(node.child_labels))])

    return node


def _extract_oblique_tree(tree_node, feature_names, class_names, is_classifier, node_counter=None):
    """Extract tree structure from an ObliqueTreeNode."""
    if node_counter is None:
        node_counter = [0]

    node_id = node_counter[0]
    node_counter[0] += 1

    node = VizNode(
        node_id=node_id,
        is_leaf=tree_node.is_leaf,
        n_samples=int(tree_node.n_samples),
        impurity=float(tree_node.impurity),
        impurity_name="gini",
    )

    if tree_node.value is not None:
        if is_classifier:
            class_counts = tree_node.value.flatten()
            total = class_counts.sum()
            dist = (class_counts / total).tolist() if total > 0 else []
            pred_idx = int(np.argmax(class_counts))
            node.class_distribution = dist
            node.class_names = list(class_names) if class_names else [str(i) for i in range(len(class_counts))]
            node.predicted_class = node.class_names[pred_idx] if node.class_names else str(pred_idx)
        else:
            val = float(tree_node.value.flatten()[0])
            node.predicted_value = val
            node.predicted_class = f"{val:.4f}"

    if not tree_node.is_leaf and tree_node.split is not None:
        split = tree_node.split
        # Build oblique split label
        terms = []
        for idx, coef in zip(split.feature_indices, split.coefficients):
            if abs(coef) > 1e-10:
                feat_name = feature_names[idx] if feature_names and idx < len(feature_names) else f"x{idx}"
                terms.append(f"{coef:+.3g}\u00b7{feat_name}")
        equation = " ".join(terms)
        node.split_label = f"{equation} \u2264 {split.threshold:.4g}"
        node.split_type = "oblique"

        # Use dominant feature for coloring
        dominant_idx = split.feature_indices[np.argmax(np.abs(split.coefficients))]
        node.split_feature = feature_names[dominant_idx] if feature_names and dominant_idx < len(feature_names) else f"feature_{dominant_idx}"

        if tree_node.left is not None:
            left = _extract_oblique_tree(tree_node.left, feature_names, class_names, is_classifier, node_counter)
            node.children.append(left)
        if tree_node.right is not None:
            right = _extract_oblique_tree(tree_node.right, feature_names, class_names, is_classifier, node_counter)
            node.children.append(right)
        node.child_labels = ["True", "False"][:len(node.children)]

    return node


def _extract_evtree(tree_node, feature_names, class_names, is_classifier, node_counter=None):
    """Extract tree structure from an EvolutionaryTree TreeNode."""
    if node_counter is None:
        node_counter = [0]

    node_id = node_counter[0]
    node_counter[0] += 1

    is_leaf = tree_node.is_leaf() if callable(tree_node.is_leaf) else tree_node.is_leaf
    n_samp = int(tree_node.n_samples) if hasattr(tree_node, 'n_samples') else 0

    node = VizNode(
        node_id=node_id,
        is_leaf=is_leaf,
        n_samples=n_samp,
        impurity_name="gini",
    )

    if tree_node.value is not None and len(tree_node.value) > 0:
        if is_classifier:
            class_counts = tree_node.value.flatten()
            total = class_counts.sum()
            dist = (class_counts / total).tolist() if total > 0 else []
            pred_idx = int(np.argmax(class_counts))
            node.class_distribution = dist
            node.class_names = list(class_names) if class_names else [str(i) for i in range(len(class_counts))]
            node.predicted_class = node.class_names[pred_idx] if node.class_names else str(pred_idx)
        else:
            val = float(tree_node.value.flatten()[0])
            node.predicted_value = val
            node.predicted_class = f"{val:.4f}"

    if not is_leaf:
        feat_idx = tree_node.feature_idx
        thresh = tree_node.threshold
        feat_name = feature_names[feat_idx] if feature_names and feat_idx < len(feature_names) else f"feature_{feat_idx}"
        node.split_label = f"{feat_name} \u2264 {thresh:.4g}"
        node.split_feature = feat_name
        node.split_type = "threshold"

        if tree_node.left is not None:
            left = _extract_evtree(tree_node.left, feature_names, class_names, is_classifier, node_counter)
            node.children.append(left)
        if tree_node.right is not None:
            right = _extract_evtree(tree_node.right, feature_names, class_names, is_classifier, node_counter)
            node.children.append(right)
        node.child_labels = ["True", "False"][:len(node.children)]

    return node


def _extract_adt_prediction_node(
    pred_node,
    feature_names,
    class_names,
    is_classifier,
    node_counter=None,
):
    """Extract an AlternatingDecisionTree prediction node (clf/reg).

    ADT's native shape is prediction-node → splitters → yes/no prediction-nodes,
    which we flatten: each splitter becomes a split node under the prediction
    node whose yes/no children are the next prediction nodes.
    """
    if node_counter is None:
        node_counter = [0]

    pred_id = node_counter[0]
    node_counter[0] += 1

    is_leaf = not getattr(pred_node, "children", None)
    node = VizNode(
        node_id=pred_id,
        is_leaf=is_leaf,
        n_samples=int(getattr(pred_node, "n_samples", 0) or 0),
        impurity_name="error",
    )

    # Prediction payload — handles both ADT (scalar) and AMT (linear model).
    if hasattr(pred_node, "model"):  # AMT: linear model at node
        lm = pred_node.model
        val = float(lm.intercept)
        node.predicted_value = val
        if len(lm.feature_indices) == 0:
            node.predicted_class = f"{val:.4g}"
        else:
            parts = [f"{val:.3g}"]
            for idx, coef in zip(lm.feature_indices, lm.coefficients):
                name = feature_names[idx] if feature_names and idx < len(feature_names) else f"x{idx}"
                parts.append(f"{float(coef):+.3g}·{name}")
            node.predicted_class = " ".join(parts)
    else:
        val = float(getattr(pred_node, "prediction", 0.0))
        if is_classifier:
            # Binary ADT: sign of prediction picks the class.
            node.predicted_value = val
            if class_names is not None and len(class_names) >= 2:
                node.predicted_class = str(class_names[1 if val >= 0 else 0])
            else:
                node.predicted_class = f"{val:+.3g}"
        else:
            node.predicted_value = val
            node.predicted_class = f"{val:.4g}"

    for splitter in getattr(pred_node, "children", []) or []:
        cond = splitter.condition
        feat_idx = cond.feature_idx
        feat_name = (
            feature_names[feat_idx]
            if feature_names and feat_idx < len(feature_names)
            else f"feature_{feat_idx}"
        )
        split_id = node_counter[0]
        node_counter[0] += 1

        split_type = getattr(cond, "split_type", None)
        split_name = getattr(split_type, "name", "").lower() or "threshold"
        if split_name == "threshold":
            label = f"{feat_name} ≤ {float(cond.threshold):.4g}"
        else:
            label = f"{feat_name} = {cond.threshold}"

        split_node = VizNode(
            node_id=split_id,
            is_leaf=False,
            split_label=label,
            split_feature=feat_name,
            split_type="threshold" if split_name == "threshold" else "discrete",
            n_samples=int(getattr(pred_node, "n_samples", 0) or 0),
            child_labels=["yes", "no"],
        )
        split_node.children.append(
            _extract_adt_prediction_node(splitter.yes_child, feature_names, class_names, is_classifier, node_counter)
        )
        split_node.children.append(
            _extract_adt_prediction_node(splitter.no_child, feature_names, class_names, is_classifier, node_counter)
        )
        node.children.append(split_node)

    if node.children and not node.child_labels:
        node.child_labels = ["" for _ in node.children]

    return node


def _extract_gosdt_json(tree_json, classes, node_counter=None):
    """Extract a tree from GOSDT's JSON export (optimal path)."""
    if node_counter is None:
        node_counter = [0]
    node_id = node_counter[0]
    node_counter[0] += 1

    if "prediction" in tree_json:
        pred_idx = int(tree_json["prediction"])
        pred = classes[pred_idx] if classes is not None and pred_idx < len(classes) else pred_idx
        return VizNode(
            node_id=node_id,
            is_leaf=True,
            predicted_class=str(pred),
        )

    feat = str(tree_json.get("feature", "?"))
    node = VizNode(
        node_id=node_id,
        is_leaf=False,
        split_label=feat,
        split_feature=feat,
        split_type="discrete",
        child_labels=["True", "False"],
    )
    if "true" in tree_json:
        node.children.append(_extract_gosdt_json(tree_json["true"], classes, node_counter))
    if "false" in tree_json:
        node.children.append(_extract_gosdt_json(tree_json["false"], classes, node_counter))
    return node


# ---------------------------------------------------------------------------
# Main TreeVisualizer class
# ---------------------------------------------------------------------------

class TreeVisualizer:
    """Interactive decision tree visualizer.

    Generates self-contained HTML files with D3.js-powered interactive
    tree visualizations featuring:
    - Expandable/collapsible branches (click nodes)
    - Zoom in/out and pan (mouse wheel + drag)
    - Rich tooltips with node statistics
    - Color-coded nodes by prediction or impurity
    - Responsive layout

    Parameters
    ----------
    model : estimator
        A fitted tree model. Supports:
        - sklearn DecisionTreeClassifier/Regressor
        - sklearn ensembles (extracts individual trees)
        - C50Classifier / C50Ensemble
        - ObliqueDecisionTreeClassifier/Regressor
        - ObliqueRandomForestClassifier/Regressor
        - EvolutionaryTreeClassifier/Regressor
    feature_names : list of str, optional
        Names for each feature. If None, uses "feature_0", "feature_1", etc.
    class_names : list of str, optional
        Names for each class (classification only).
    tree_index : int, default=0
        For ensemble models, which tree to visualize.
    title : str, optional
        Title displayed above the visualization.
    color_by : str, default='prediction'
        How to color nodes: 'prediction' (class color), 'impurity' (heatmap),
        or 'samples' (by sample count).
    max_depth : int, optional
        Maximum depth to display. Deeper nodes are collapsed by default.
    palette : str, default='tableau'
        Color palette: 'tableau', 'viridis', 'pastel', or 'dark'.

    Example
    -------
    >>> from sklearn.datasets import load_iris
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> from endgame.visualization import TreeVisualizer
    >>>
    >>> X, y = load_iris(return_X_y=True)
    >>> clf = DecisionTreeClassifier(max_depth=4, random_state=42).fit(X, y)
    >>> viz = TreeVisualizer(
    ...     clf,
    ...     feature_names=load_iris().feature_names,
    ...     class_names=load_iris().target_names.tolist(),
    ...     title="Iris Decision Tree"
    ... )
    >>> viz.save("iris_tree.html")
    """

    def __init__(
        self,
        model,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        tree_index: int = 0,
        title: str | None = None,
        color_by: str = "prediction",
        max_depth: int | None = None,
        palette: str = "tableau",
    ):
        self.model = model
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.class_names = list(class_names) if class_names is not None else None
        self.tree_index = tree_index
        self.title = title
        self.color_by = color_by
        self.max_depth = max_depth
        self.palette = palette

        # Extract the tree
        self._root = self._extract_tree()

    def _detect_model_type(self):
        """Detect what type of tree model we have."""
        model = self.model
        cls_name = type(model).__name__

        # Named endgame ensembles (match before generic estimators_ check so
        # nested tree backends aren't misrouted through sklearn extraction).
        if cls_name == 'C50Ensemble':
            return 'c50_ensemble'
        if cls_name in ('ObliqueRandomForestClassifier', 'ObliqueRandomForestRegressor'):
            return 'oblique_forest'
        if cls_name in ('RotationForestClassifier', 'RotationForestRegressor'):
            return 'rotation_forest'
        if cls_name == 'QuantileRegressorForest':
            return 'sklearn_ensemble'
        if cls_name == 'CubistRegressor':
            return 'cubist'
        if cls_name in ('EvolutionaryTreeClassifier', 'EvolutionaryTreeRegressor'):
            return 'evtree'
        if cls_name == 'AlternatingDecisionTreeClassifier':
            return 'adt'
        if cls_name == 'AlternatingModelTreeRegressor':
            return 'amt'
        if cls_name == 'GOSDTClassifier':
            return 'gosdt'

        # Generic sklearn ensembles (random forest, gradient boosting, etc.)
        if hasattr(model, 'estimators_'):
            est = model.estimators_
            if hasattr(est, 'shape') and len(est.shape) == 2:
                if hasattr(est[0, 0], 'tree_'):
                    return 'sklearn_gb_ensemble'
            elif hasattr(est, '__len__') and len(est) > 0:
                first = est[0]
                if hasattr(first, 'tree_'):
                    return 'sklearn_ensemble'

        # Single trees
        if cls_name == 'C50Classifier' and hasattr(model, 'tree_'):
            return 'c50'
        if cls_name in ('ObliqueDecisionTreeClassifier', 'ObliqueDecisionTreeRegressor'):
            return 'oblique'
        if hasattr(model, 'tree_') and hasattr(model.tree_, 'feature'):
            return 'sklearn'

        # Fallback: try sklearn-like
        if hasattr(model, 'tree_'):
            return 'sklearn'

        raise ValueError(
            f"Unsupported model type: {cls_name}. "
            "TreeVisualizer supports sklearn trees, C5.0, ObliqueTree, "
            "EvolutionaryTree, AlternatingDecisionTree, Cubist, GOSDT, "
            "and their ensemble variants."
        )

    def _is_classifier(self) -> bool:
        """Check if the model is a classifier."""
        from sklearn.base import is_classifier
        model = self.model
        if hasattr(model, 'estimators_'):
            return is_classifier(model)
        return is_classifier(model)

    def _extract_tree(self) -> VizNode:
        """Extract tree from the fitted model into VizNode format."""
        model_type = self._detect_model_type()
        is_clf = self._is_classifier()

        if model_type == 'sklearn_gb_ensemble':
            # GradientBoosting: estimators_ is 2D (n_estimators, n_classes)
            est = self.model.estimators_
            idx = min(self.tree_index, est.shape[0] - 1)
            tree_model = est[idx, 0]
            return _extract_sklearn_tree(tree_model, self.feature_names, self.class_names, False)

        elif model_type == 'sklearn_ensemble':
            tree_model = self.model.estimators_[self.tree_index]
            if hasattr(tree_model, 'estimators_'):
                # Nested ensemble (e.g., bagging)
                tree_model = tree_model.estimators_[0]
            return _extract_sklearn_tree(tree_model, self.feature_names, self.class_names, is_clf)

        elif model_type == 'sklearn':
            return _extract_sklearn_tree(self.model, self.feature_names, self.class_names, is_clf)

        elif model_type == 'c50':
            return _extract_c50_tree(self.model.tree_, self.feature_names, self.class_names)

        elif model_type == 'c50_ensemble':
            # C50Ensemble stores fitted C50Classifier instances in estimators_.
            estimators = self.model.estimators_
            idx = min(self.tree_index, len(estimators) - 1)
            return _extract_c50_tree(estimators[idx].tree_, self.feature_names, self.class_names)

        elif model_type == 'oblique':
            root = self.model.tree_ if hasattr(self.model, 'tree_') else self.model.root_
            return _extract_oblique_tree(root, self.feature_names, self.class_names, is_clf)

        elif model_type == 'oblique_forest':
            tree_model = self.model.estimators_[self.tree_index]
            root = tree_model.tree_ if hasattr(tree_model, 'tree_') else tree_model.root_
            return _extract_oblique_tree(root, self.feature_names, self.class_names, is_clf)

        elif model_type == 'rotation_forest':
            # Each member is a sklearn tree trained on rotated features.
            tree_model = self.model.estimators_[self.tree_index]
            rotated_names = [f"rot_{i}" for i in range(self.model.n_features_in_)] if hasattr(self.model, 'n_features_in_') else None
            return _extract_sklearn_tree(tree_model, rotated_names or self.feature_names, self.class_names, is_clf)

        elif model_type == 'evtree':
            root = self.model.tree_ if hasattr(self.model, 'tree_') else self.model.root_
            return _extract_evtree(root, self.feature_names, self.class_names, is_clf)

        elif model_type == 'adt':
            if getattr(self.model, '_binary', True):
                root = self.model.root_
            else:
                trees = self.model._ovr_trees
                root = trees[min(self.tree_index, len(trees) - 1)]
            return _extract_adt_prediction_node(
                root, self.feature_names, self.class_names, is_clf,
            )

        elif model_type == 'amt':
            return _extract_adt_prediction_node(
                self.model.root_, self.feature_names, self.class_names, False,
            )

        elif model_type == 'cubist':
            # Cubist is committees of sklearn trees with per-leaf linear models.
            trees = self.model._trees
            idx = min(self.tree_index, len(trees) - 1)
            return _extract_sklearn_tree(trees[idx], self.feature_names, self.class_names, False)

        elif model_type == 'gosdt':
            # Optimal path: the native GOSDT JSON. Fallback path: CART inside self.model.tree_.
            if getattr(self.model, "_using_gosdt", False) and hasattr(self.model, "_tree_json"):
                return _extract_gosdt_json(self.model._tree_json, self.class_names)
            return _extract_sklearn_tree(self.model.tree_, self.feature_names, self.class_names, is_clf)

        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def to_json(self) -> str:
        """Export tree data as JSON string."""
        return json.dumps(self._root.to_dict(), indent=2)

    def save(self, filepath: str | Path, open_browser: bool = False) -> Path:
        """Save interactive visualization as a self-contained HTML file.

        Parameters
        ----------
        filepath : str or Path
            Output file path (should end in .html).
        open_browser : bool, default=False
            If True, open the file in the default web browser.

        Returns
        -------
        Path
            The absolute path to the saved file.
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix('.html')

        tree_json = self._root.to_dict()
        title = self.title or "Decision Tree Visualization"
        title_escaped = html_module.escape(title)

        html_content = _generate_html(
            tree_data=tree_json,
            title=title_escaped,
            color_by=self.color_by,
            max_depth=self.max_depth,
            palette=self.palette,
        )

        filepath.write_text(html_content, encoding='utf-8')

        if open_browser:
            import webbrowser
            webbrowser.open(filepath.resolve().as_uri())

        return filepath.resolve()

    def _repr_html_(self) -> str:
        """Jupyter notebook display support."""
        tree_json = self._root.to_dict()
        title = self.title or "Decision Tree Visualization"
        return _generate_html(
            tree_data=tree_json,
            title=html_module.escape(title),
            color_by=self.color_by,
            max_depth=self.max_depth,
            palette=self.palette,
            embedded=True,
        )


# ---------------------------------------------------------------------------
# HTML/JS/CSS generation
# ---------------------------------------------------------------------------

_PALETTES = {
    "tableau": [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ],
    "viridis": [
        "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ],
    "pastel": [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
    ],
    "dark": [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#666666", "#e41a1c", "#377eb8",
    ],
}


def _generate_html(
    tree_data: dict,
    title: str,
    color_by: str,
    max_depth: int | None,
    palette: str,
    embedded: bool = False,
) -> str:
    """Generate the complete self-contained HTML visualization."""
    palette_colors = _PALETTES.get(palette, _PALETTES["tableau"])
    tree_json_str = json.dumps(tree_data)
    colors_json = json.dumps(palette_colors)
    max_depth_js = str(max_depth) if max_depth is not None else "null"

    # Height for embedded (Jupyter) vs standalone
    container_height = "600px" if embedded else "100vh"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}

body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: #0f1117;
  color: #e0e0e0;
  overflow: hidden;
}}

#app {{
  width: 100vw;
  height: {container_height};
  display: flex;
  flex-direction: column;
}}

/* Header */
.header {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 24px;
  background: linear-gradient(135deg, #1a1d29, #252836);
  border-bottom: 1px solid rgba(255,255,255,0.06);
  z-index: 100;
  flex-shrink: 0;
}}

.header h1 {{
  font-size: 18px;
  font-weight: 600;
  color: #f0f0f0;
  letter-spacing: -0.3px;
}}

.controls {{
  display: flex;
  gap: 8px;
  align-items: center;
}}

.btn {{
  padding: 6px 14px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px;
  background: rgba(255,255,255,0.05);
  color: #c0c0c0;
  font-size: 13px;
  cursor: pointer;
  transition: all 0.2s;
  user-select: none;
}}
.btn:hover {{
  background: rgba(255,255,255,0.1);
  color: #fff;
  border-color: rgba(255,255,255,0.2);
}}
.btn.active {{
  background: rgba(78, 121, 167, 0.3);
  border-color: #4e79a7;
  color: #a0c4e8;
}}

.stats {{
  font-size: 12px;
  color: #888;
  padding: 0 12px;
}}

/* Tree container */
#tree-container {{
  flex: 1;
  overflow: hidden;
  position: relative;
  cursor: grab;
}}
#tree-container:active {{ cursor: grabbing; }}

#tree-svg {{
  width: 100%;
  height: 100%;
}}

/* Links */
.link {{
  fill: none;
  stroke: rgba(255,255,255,0.12);
  stroke-width: 1.5px;
  transition: stroke 0.3s, stroke-width 0.3s;
}}
.link:hover {{
  stroke: rgba(255,255,255,0.3);
  stroke-width: 2.5px;
}}

/* Nodes */
.node {{
  cursor: pointer;
}}

.node-rect {{
  rx: 8;
  ry: 8;
  stroke-width: 1.5px;
  transition: all 0.3s;
  filter: drop-shadow(0 2px 6px rgba(0,0,0,0.4));
}}
.node:hover .node-rect {{
  stroke-width: 2.5px;
  filter: drop-shadow(0 4px 12px rgba(0,0,0,0.6));
}}
.node.collapsed .node-rect {{
  stroke-dasharray: 4 2;
}}

.node-label {{
  font-size: 11px;
  fill: #f0f0f0;
  text-anchor: middle;
  pointer-events: none;
  font-weight: 500;
}}
.node-sublabel {{
  font-size: 9.5px;
  fill: #aaa;
  text-anchor: middle;
  pointer-events: none;
}}

.edge-label {{
  font-size: 9px;
  fill: #888;
  text-anchor: middle;
  pointer-events: none;
  font-weight: 500;
}}

/* Collapse indicator */
.collapse-badge {{
  fill: rgba(255,255,255,0.15);
  stroke: rgba(255,255,255,0.2);
  stroke-width: 1px;
}}
.collapse-text {{
  font-size: 9px;
  fill: #ccc;
  text-anchor: middle;
  dominant-baseline: central;
  pointer-events: none;
  font-weight: 600;
}}

/* Tooltip */
.tooltip {{
  position: absolute;
  background: linear-gradient(135deg, #252836, #1e2130);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 10px;
  padding: 14px 18px;
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.2s;
  z-index: 1000;
  min-width: 220px;
  max-width: 360px;
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  backdrop-filter: blur(10px);
}}
.tooltip.visible {{ opacity: 1; }}
.tooltip h3 {{
  font-size: 13px;
  font-weight: 600;
  color: #f0f0f0;
  margin-bottom: 8px;
  border-bottom: 1px solid rgba(255,255,255,0.08);
  padding-bottom: 6px;
}}
.tooltip .row {{
  display: flex;
  justify-content: space-between;
  font-size: 12px;
  padding: 2px 0;
}}
.tooltip .row .label {{ color: #888; }}
.tooltip .row .value {{ color: #ddd; font-weight: 500; }}

/* Distribution bar */
.dist-bar {{
  display: flex;
  height: 8px;
  border-radius: 4px;
  overflow: hidden;
  margin: 8px 0 4px;
}}
.dist-bar .segment {{
  transition: width 0.3s;
}}
.dist-legend {{
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 4px;
}}
.dist-legend .item {{
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 10px;
  color: #aaa;
}}
.dist-legend .swatch {{
  width: 8px;
  height: 8px;
  border-radius: 2px;
}}

/* Minimap */
.minimap {{
  position: absolute;
  bottom: 16px;
  right: 16px;
  width: 180px;
  height: 120px;
  background: rgba(15, 17, 23, 0.85);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 8px;
  overflow: hidden;
  z-index: 50;
}}
.minimap-viewport {{
  stroke: #4e79a7;
  stroke-width: 1.5px;
  fill: rgba(78, 121, 167, 0.1);
}}

/* Search */
.search-box {{
  position: relative;
}}
.search-box input {{
  padding: 6px 12px;
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 6px;
  background: rgba(255,255,255,0.05);
  color: #e0e0e0;
  font-size: 13px;
  width: 160px;
  outline: none;
  transition: all 0.2s;
}}
.search-box input:focus {{
  border-color: #4e79a7;
  background: rgba(255,255,255,0.08);
  width: 200px;
}}
.search-box input::placeholder {{ color: #666; }}

/* Zoom indicator */
.zoom-level {{
  position: absolute;
  bottom: 16px;
  left: 16px;
  background: rgba(15, 17, 23, 0.8);
  border: 1px solid rgba(255,255,255,0.1);
  border-radius: 6px;
  padding: 6px 12px;
  font-size: 11px;
  color: #888;
  z-index: 50;
}}
</style>
</head>
<body>
<div id="app">
  <div class="header">
    <h1>{title}</h1>
    <div class="controls">
      <div class="search-box">
        <input type="text" id="search-input" placeholder="Search features..." />
      </div>
      <button class="btn" onclick="expandAll()" title="Expand all nodes">Expand All</button>
      <button class="btn" onclick="collapseAll()" title="Collapse all nodes">Collapse All</button>
      <button class="btn" onclick="resetZoom()" title="Reset view">Reset View</button>
      <button class="btn" id="btn-color" onclick="cycleColor()" title="Change color mode">Color: prediction</button>
      <span class="stats" id="stats-text"></span>
    </div>
  </div>
  <div id="tree-container">
    <svg id="tree-svg"></svg>
    <div class="tooltip" id="tooltip"></div>
    <div class="zoom-level" id="zoom-level">100%</div>
  </div>
</div>

<script>
// ===== D3.js v7 (minified inline) =====
// Using a lightweight subset of D3 functionality built from scratch
// to keep the file self-contained without external CDN dependencies.
</script>
<script>
(function() {{
"use strict";

// ===== Data =====
const treeData = {tree_json_str};
const PALETTE = {colors_json};
const MAX_DEPTH = {max_depth_js};
const COLOR_MODES = ['prediction', 'impurity', 'samples'];
let colorMode = '{color_by}';

// ===== State =====
let root;
let nodeId = 0;
let transform = {{ x: 0, y: 0, k: 1 }};
let dragging = false;
let dragStart = {{ x: 0, y: 0 }};
let selectedNode = null;

// Layout constants
const NODE_W = 160;
const NODE_H = 52;
const H_GAP = 24;
const V_GAP = 72;
const ANIMATION_MS = 400;

// ===== Tree data processing =====
function processNode(data, depth, parent) {{
  const node = {{
    id: nodeId++,
    data: data,
    depth: depth,
    parent: parent,
    children: [],
    _children: null,
    x: 0,
    y: 0,
    collapsed: false,
  }};
  if (data.children && data.children.length > 0) {{
    for (const childData of data.children) {{
      node.children.push(processNode(childData, depth + 1, node));
    }}
    // Auto-collapse beyond max depth
    if (MAX_DEPTH !== null && depth >= MAX_DEPTH) {{
      node._children = node.children;
      node.children = [];
      node.collapsed = true;
    }}
  }}
  return node;
}}

function countDescendants(node) {{
  if (!node.children || node.children.length === 0) {{
    if (!node._children || node._children.length === 0) return 1;
  }}
  let count = 0;
  const kids = node.children.length > 0 ? node.children : (node._children || []);
  for (const c of kids) count += countDescendants(c);
  return Math.max(count, 1);
}}

function countVisible(node) {{
  if (node.children.length === 0) return 1;
  let count = 0;
  for (const c of node.children) count += countVisible(c);
  return Math.max(count, 1);
}}

// ===== Layout (Reingold-Tilford inspired) =====
function layoutTree(root) {{
  // First pass: compute leaf counts for spacing
  assignY(root, 0);
  // Second pass: assign X based on leaf positions
  let leafIndex = 0;
  assignX(root);

  function assignY(node, depth) {{
    node.y = depth * (NODE_H + V_GAP);
    node.depth = depth;
    for (const child of node.children) {{
      assignY(child, depth + 1);
    }}
  }}

  function assignX(node) {{
    if (node.children.length === 0) {{
      node.x = leafIndex * (NODE_W + H_GAP);
      leafIndex++;
      return;
    }}
    for (const child of node.children) {{
      assignX(child);
    }}
    // Center parent above children
    const first = node.children[0];
    const last = node.children[node.children.length - 1];
    node.x = (first.x + last.x) / 2;
  }}
}}

// ===== Rendering =====
const svg = document.getElementById('tree-svg');
const container = document.getElementById('tree-container');
const tooltip = document.getElementById('tooltip');
const zoomLabel = document.getElementById('zoom-level');
const statsText = document.getElementById('stats-text');

function createSVGElement(tag) {{
  return document.createElementNS('http://www.w3.org/2000/svg', tag);
}}

function clearSVG() {{
  while (svg.firstChild) svg.removeChild(svg.firstChild);
}}

function getNodeColor(node) {{
  const d = node.data;
  if (colorMode === 'prediction') {{
    if (d.classDist && d.classDist.length > 0) {{
      const maxIdx = d.classDist.indexOf(Math.max(...d.classDist));
      return PALETTE[maxIdx % PALETTE.length];
    }}
    return PALETTE[0];
  }} else if (colorMode === 'impurity') {{
    // Heatmap: low impurity = blue, high = red
    const imp = Math.min(d.impurity, 0.5) / 0.5;
    const r = Math.round(30 + imp * 200);
    const g = Math.round(80 - imp * 50);
    const b = Math.round(180 - imp * 150);
    return `rgb(${{r}},${{g}},${{b}})`;
  }} else {{ // samples
    // Size-based: more samples = brighter
    const maxSamples = root.data.samples || 1;
    const ratio = Math.min((d.samples || 0) / maxSamples, 1);
    const intensity = Math.round(40 + ratio * 160);
    return `rgb(${{Math.round(30 + ratio * 60)}}, ${{intensity}}, ${{Math.round(120 + ratio * 80)}})`;
  }}
}}

function getNodeStroke(node) {{
  const color = getNodeColor(node);
  return color;
}}

function getNodeFill(node) {{
  const color = getNodeColor(node);
  // Parse and darken
  if (color.startsWith('rgb')) {{
    const m = color.match(/\\d+/g);
    if (m) {{
      return `rgba(${{Math.round(m[0]*0.25)}},${{Math.round(m[1]*0.25)}},${{Math.round(m[2]*0.25)}},0.85)`;
    }}
  }}
  // Hex
  const r = parseInt(color.slice(1,3), 16);
  const g = parseInt(color.slice(3,5), 16);
  const b = parseInt(color.slice(5,7), 16);
  return `rgba(${{Math.round(r*0.25)}},${{Math.round(g*0.25)}},${{Math.round(b*0.25)}},0.85)`;
}}

function render() {{
  clearSVG();
  layoutTree(root);

  const g = createSVGElement('g');
  g.setAttribute('id', 'tree-group');
  svg.appendChild(g);

  // Apply transform
  updateTransform();

  // Draw links first (under nodes)
  drawLinks(g, root);
  // Draw nodes on top
  drawNodes(g, root);

  updateStats();
}}

function drawLinks(parent, node) {{
  for (let i = 0; i < node.children.length; i++) {{
    const child = node.children[i];
    const link = createSVGElement('path');

    const x1 = node.x + NODE_W / 2;
    const y1 = node.y + NODE_H;
    const x2 = child.x + NODE_W / 2;
    const y2 = child.y;
    const midY = (y1 + y2) / 2;

    link.setAttribute('d', `M${{x1}},${{y1}} C${{x1}},${{midY}} ${{x2}},${{midY}} ${{x2}},${{y2}}`);
    link.setAttribute('class', 'link');
    parent.appendChild(link);

    // Edge label
    const edgeLabels = node.data.childLabels || [];
    if (edgeLabels[i]) {{
      const label = createSVGElement('text');
      label.setAttribute('x', (x1 + x2) / 2);
      label.setAttribute('y', midY - 4);
      label.setAttribute('class', 'edge-label');
      label.textContent = edgeLabels[i];
      parent.appendChild(label);
    }}

    drawLinks(parent, child);
  }}
}}

function drawNodes(parent, node) {{
  const g = createSVGElement('g');
  g.setAttribute('class', 'node' + (node.collapsed ? ' collapsed' : ''));
  g.setAttribute('transform', `translate(${{node.x}},${{node.y}})`);

  // Node rectangle
  const rect = createSVGElement('rect');
  rect.setAttribute('width', NODE_W);
  rect.setAttribute('height', NODE_H);
  rect.setAttribute('class', 'node-rect');
  rect.setAttribute('fill', getNodeFill(node));
  rect.setAttribute('stroke', getNodeStroke(node));
  g.appendChild(rect);

  // Main label (split condition or prediction)
  const label = createSVGElement('text');
  label.setAttribute('x', NODE_W / 2);
  label.setAttribute('y', 20);
  label.setAttribute('class', 'node-label');
  const labelText = node.data.leaf
    ? (node.data.prediction || 'leaf')
    : truncateText(node.data.split || 'split', 22);
  label.textContent = labelText;
  g.appendChild(label);

  // Sub-label (samples / impurity)
  const sublabel = createSVGElement('text');
  sublabel.setAttribute('x', NODE_W / 2);
  sublabel.setAttribute('y', 38);
  sublabel.setAttribute('class', 'node-sublabel');
  sublabel.textContent = `n=${{node.data.samples}}`;
  if (!node.data.leaf) {{
    sublabel.textContent += ` | ${{node.data.impurityName}}=${{node.data.impurity.toFixed(3)}}`;
  }}
  g.appendChild(sublabel);

  // Collapse indicator badge
  if (node.collapsed && node._children && node._children.length > 0) {{
    const badgeR = 10;
    const badge = createSVGElement('circle');
    badge.setAttribute('cx', NODE_W / 2);
    badge.setAttribute('cy', NODE_H + 8);
    badge.setAttribute('r', badgeR);
    badge.setAttribute('class', 'collapse-badge');
    g.appendChild(badge);

    const badgeText = createSVGElement('text');
    badgeText.setAttribute('x', NODE_W / 2);
    badgeText.setAttribute('y', NODE_H + 8);
    badgeText.setAttribute('class', 'collapse-text');
    const hiddenCount = countDescendants({{ children: node._children, _children: null }});
    badgeText.textContent = `+${{hiddenCount}}`;
    g.appendChild(badgeText);
  }}

  // Click handler: expand/collapse
  g.addEventListener('click', (e) => {{
    e.stopPropagation();
    toggleNode(node);
  }});

  // Hover handlers for tooltip
  g.addEventListener('mouseenter', (e) => showTooltip(e, node));
  g.addEventListener('mouseleave', hideTooltip);

  parent.appendChild(g);

  // Recurse for visible children
  for (const child of node.children) {{
    drawNodes(parent, child);
  }}
}}

function truncateText(text, maxLen) {{
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen - 1) + '\u2026';
}}

// ===== Node toggle =====
function toggleNode(node) {{
  if (node.collapsed) {{
    // Expand
    node.children = node._children || [];
    node._children = null;
    node.collapsed = false;
  }} else if (node.children.length > 0) {{
    // Collapse
    node._children = node.children;
    node.children = [];
    node.collapsed = true;
  }}
  render();
}}

function expandAll() {{
  function expand(node) {{
    if (node._children) {{
      node.children = node._children;
      node._children = null;
      node.collapsed = false;
    }}
    for (const c of node.children) expand(c);
  }}
  expand(root);
  render();
  fitToView();
}}

function collapseAll() {{
  function collapse(node) {{
    for (const c of node.children) collapse(c);
    if (node.children.length > 0) {{
      node._children = node.children;
      node.children = [];
      node.collapsed = true;
    }}
  }}
  if (root.children.length > 0) {{
    for (const c of root.children) {{
      collapse(c);
    }}
  }}
  render();
  fitToView();
}}

// ===== Tooltip =====
function showTooltip(event, node) {{
  const d = node.data;
  let html = `<h3>${{d.leaf ? 'Leaf Node' : 'Decision Node'}}</h3>`;

  if (!d.leaf) {{
    html += `<div class="row"><span class="label">Split</span><span class="value">${{escapeHtml(d.split)}}</span></div>`;
    if (d.splitType === 'oblique') {{
      html += `<div class="row"><span class="label">Type</span><span class="value">Oblique (multi-feature)</span></div>`;
    }}
  }}

  html += `<div class="row"><span class="label">Samples</span><span class="value">${{d.samples.toLocaleString()}}</span></div>`;
  html += `<div class="row"><span class="label">${{capitalize(d.impurityName)}}</span><span class="value">${{d.impurity.toFixed(4)}}</span></div>`;
  html += `<div class="row"><span class="label">Prediction</span><span class="value">${{escapeHtml(d.prediction)}}</span></div>`;

  if (d.predValue !== undefined) {{
    html += `<div class="row"><span class="label">Value</span><span class="value">${{d.predValue.toFixed(4)}}</span></div>`;
  }}

  // Class distribution bar
  if (d.classDist && d.classDist.length > 0) {{
    html += '<div class="dist-bar">';
    for (let i = 0; i < d.classDist.length; i++) {{
      const pct = (d.classDist[i] * 100).toFixed(1);
      html += `<div class="segment" style="width:${{pct}}%;background:${{PALETTE[i % PALETTE.length]}}"></div>`;
    }}
    html += '</div>';
    html += '<div class="dist-legend">';
    for (let i = 0; i < d.classDist.length; i++) {{
      const name = d.classNames[i] || `Class ${{i}}`;
      const pct = (d.classDist[i] * 100).toFixed(1);
      html += `<span class="item"><span class="swatch" style="background:${{PALETTE[i % PALETTE.length]}}"></span>${{escapeHtml(name)}}: ${{pct}}%</span>`;
    }}
    html += '</div>';
  }}

  tooltip.innerHTML = html;
  tooltip.classList.add('visible');

  const rect = container.getBoundingClientRect();
  let left = event.clientX - rect.left + 16;
  let top = event.clientY - rect.top - 10;

  // Keep tooltip in bounds
  const tw = tooltip.offsetWidth;
  const th = tooltip.offsetHeight;
  if (left + tw > rect.width - 16) left = event.clientX - rect.left - tw - 16;
  if (top + th > rect.height - 16) top = rect.height - th - 16;
  if (top < 8) top = 8;

  tooltip.style.left = left + 'px';
  tooltip.style.top = top + 'px';
}}

function hideTooltip() {{
  tooltip.classList.remove('visible');
}}

function escapeHtml(str) {{
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}}

function capitalize(s) {{
  return s.charAt(0).toUpperCase() + s.slice(1);
}}

// ===== Zoom & Pan =====
function updateTransform() {{
  const g = document.getElementById('tree-group');
  if (g) {{
    g.setAttribute('transform', `translate(${{transform.x}},${{transform.y}}) scale(${{transform.k}})`);
  }}
  zoomLabel.textContent = Math.round(transform.k * 100) + '%';
}}

container.addEventListener('wheel', (e) => {{
  e.preventDefault();
  const rect = container.getBoundingClientRect();
  const mx = e.clientX - rect.left;
  const my = e.clientY - rect.top;

  const delta = e.deltaY > 0 ? 0.9 : 1.1;
  const newK = Math.max(0.05, Math.min(5, transform.k * delta));

  // Zoom toward cursor
  transform.x = mx - (mx - transform.x) * (newK / transform.k);
  transform.y = my - (my - transform.y) * (newK / transform.k);
  transform.k = newK;

  updateTransform();
}}, {{ passive: false }});

container.addEventListener('mousedown', (e) => {{
  if (e.target.closest('.node')) return;
  dragging = true;
  dragStart.x = e.clientX - transform.x;
  dragStart.y = e.clientY - transform.y;
}});

window.addEventListener('mousemove', (e) => {{
  if (!dragging) return;
  transform.x = e.clientX - dragStart.x;
  transform.y = e.clientY - dragStart.y;
  updateTransform();
}});

window.addEventListener('mouseup', () => {{ dragging = false; }});

// Touch support
let lastTouchDist = 0;
container.addEventListener('touchstart', (e) => {{
  if (e.touches.length === 1) {{
    dragging = true;
    dragStart.x = e.touches[0].clientX - transform.x;
    dragStart.y = e.touches[0].clientY - transform.y;
  }} else if (e.touches.length === 2) {{
    lastTouchDist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
  }}
}}, {{ passive: true }});

container.addEventListener('touchmove', (e) => {{
  e.preventDefault();
  if (e.touches.length === 1 && dragging) {{
    transform.x = e.touches[0].clientX - dragStart.x;
    transform.y = e.touches[0].clientY - dragStart.y;
    updateTransform();
  }} else if (e.touches.length === 2) {{
    const dist = Math.hypot(
      e.touches[0].clientX - e.touches[1].clientX,
      e.touches[0].clientY - e.touches[1].clientY
    );
    const delta = dist / lastTouchDist;
    transform.k = Math.max(0.05, Math.min(5, transform.k * delta));
    lastTouchDist = dist;
    updateTransform();
  }}
}}, {{ passive: false }});

container.addEventListener('touchend', () => {{ dragging = false; }});

function fitToView() {{
  // Find bounds of all visible nodes
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  function traverse(node) {{
    minX = Math.min(minX, node.x);
    maxX = Math.max(maxX, node.x + NODE_W);
    minY = Math.min(minY, node.y);
    maxY = Math.max(maxY, node.y + NODE_H);
    for (const c of node.children) traverse(c);
  }}
  traverse(root);

  const rect = container.getBoundingClientRect();
  const padding = 60;
  const treeW = maxX - minX + padding * 2;
  const treeH = maxY - minY + padding * 2;

  const scaleX = rect.width / treeW;
  const scaleY = rect.height / treeH;
  const k = Math.min(scaleX, scaleY, 1.5);

  transform.k = k;
  transform.x = (rect.width - (maxX - minX) * k) / 2 - minX * k;
  transform.y = (rect.height - (maxY - minY) * k) / 2 - minY * k + 20;

  updateTransform();
}}

window.resetZoom = fitToView;
window.expandAll = expandAll;
window.collapseAll = collapseAll;

// ===== Color mode cycling =====
window.cycleColor = function() {{
  const idx = COLOR_MODES.indexOf(colorMode);
  colorMode = COLOR_MODES[(idx + 1) % COLOR_MODES.length];
  document.getElementById('btn-color').textContent = 'Color: ' + colorMode;
  render();
  // Preserve current transform after re-render
  updateTransform();
}};

// ===== Search =====
const searchInput = document.getElementById('search-input');
searchInput.addEventListener('input', (e) => {{
  const query = e.target.value.toLowerCase().trim();
  if (!query) {{
    // Reset highlight
    document.querySelectorAll('.node-rect').forEach(r => r.style.opacity = '1');
    return;
  }}
  function searchNode(node, parentG, idx) {{
    const g = parentG.querySelectorAll(':scope > g.node');
    // Highlight matching nodes
    const allNodes = document.querySelectorAll('.node');
    allNodes.forEach((nodeEl, i) => {{
      const rect = nodeEl.querySelector('.node-rect');
      const label = nodeEl.querySelector('.node-label');
      if (label && label.textContent.toLowerCase().includes(query)) {{
        rect.style.opacity = '1';
        rect.style.strokeWidth = '3px';
      }} else {{
        rect.style.opacity = '0.3';
        rect.style.strokeWidth = '1.5px';
      }}
    }});
  }}
  // Simple approach: check all node labels
  const allNodes = document.querySelectorAll('.node');
  allNodes.forEach((nodeEl) => {{
    const rect = nodeEl.querySelector('.node-rect');
    const labels = nodeEl.querySelectorAll('.node-label, .node-sublabel');
    let match = false;
    labels.forEach(l => {{
      if (l.textContent.toLowerCase().includes(query)) match = true;
    }});
    if (rect) {{
      rect.style.opacity = match ? '1' : '0.2';
      rect.style.strokeWidth = match ? '3px' : '1.5px';
    }}
  }});
}});

// ===== Stats =====
function updateStats() {{
  let totalNodes = 0, visibleNodes = 0, leaves = 0, maxDepth = 0;
  function count(node, depth) {{
    totalNodes++;
    visibleNodes++;
    if (node.data.leaf || (node.children.length === 0 && (!node._children || node._children.length === 0))) leaves++;
    maxDepth = Math.max(maxDepth, depth);
    for (const c of node.children) count(c, depth + 1);
    if (node._children) {{
      function countHidden(n) {{
        totalNodes++;
        const kids = n.children || n._children || [];
        if (n.data && n.data.children) {{
          for (const cd of n.data.children) countHidden({{ data: cd }});
        }}
      }}
      // Count hidden via _children tree structure
      for (const hc of node._children) {{
        function countAll(nd) {{
          totalNodes++;
          for (const c of nd.children) countAll(c);
          if (nd._children) for (const c of nd._children) countAll(c);
        }}
        countAll(hc);
      }}
    }}
  }}
  totalNodes = 0; visibleNodes = 0; leaves = 0; maxDepth = 0;
  // Simple count of visible
  function simpleCount(node, depth) {{
    visibleNodes++;
    if (node.children.length === 0) leaves++;
    maxDepth = Math.max(maxDepth, depth);
    for (const c of node.children) simpleCount(c, depth + 1);
  }}
  simpleCount(root, 0);
  statsText.textContent = `${{visibleNodes}} nodes | ${{leaves}} leaves | depth ${{maxDepth}}`;
}}

// ===== Keyboard shortcuts =====
document.addEventListener('keydown', (e) => {{
  if (e.target === searchInput) return;
  if (e.key === '+' || e.key === '=') {{
    transform.k = Math.min(5, transform.k * 1.2);
    updateTransform();
  }} else if (e.key === '-') {{
    transform.k = Math.max(0.05, transform.k / 1.2);
    updateTransform();
  }} else if (e.key === '0') {{
    fitToView();
  }} else if (e.key === 'e') {{
    expandAll();
  }} else if (e.key === 'c') {{
    collapseAll();
  }} else if (e.key === '/') {{
    e.preventDefault();
    searchInput.focus();
  }}
}});

// ===== Init =====
root = processNode(treeData, 0, null);
render();

// Auto fit after initial render
requestAnimationFrame(() => {{
  fitToView();
}});

// Handle window resize
window.addEventListener('resize', () => {{ fitToView(); }});

}})();
</script>
</body>
</html>"""

    return html
