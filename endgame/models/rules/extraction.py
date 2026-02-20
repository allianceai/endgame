"""Rule extraction from tree ensembles.

This module contains functions to extract rules from sklearn-compatible
decision tree ensembles (RandomForest, GradientBoosting, ExtraTrees).
"""


import numpy as np

from endgame.models.rules.rule import Condition, Operator, Rule, RuleEnsemble


def extract_rules_from_tree(
    tree,
    feature_names: list[str],
    tree_idx: int = 0,
    X_train: np.ndarray | None = None,
) -> list[Rule]:
    """
    Extract all rules from a single decision tree.

    A rule is extracted for EVERY node (not just leaves).
    Each rule is the conjunction of conditions along the path
    from root to that node.

    Parameters
    ----------
    tree : sklearn tree object
        A fitted decision tree (tree_ attribute of DecisionTreeClassifier/Regressor
        or the estimator itself).
    feature_names : list of str
        Names of input features.
    tree_idx : int
        Index of this tree in the ensemble (for tracking).
    X_train : ndarray, optional
        Training data for computing support.

    Returns
    -------
    rules : list of Rule
        All rules extracted from this tree.
    """
    # Handle both tree estimators and tree_ attributes
    if hasattr(tree, "tree_"):
        tree_ = tree.tree_
    else:
        tree_ = tree

    feature = tree_.feature
    threshold = tree_.threshold
    children_left = tree_.children_left
    children_right = tree_.children_right
    n_node_samples = tree_.n_node_samples

    rules = []

    def recurse(node_idx: int, current_conditions: list[Condition]):
        """Recursively traverse tree and extract rules."""
        # Create rule for current node (if not root with empty conditions)
        if len(current_conditions) > 0:
            rule = Rule(
                conditions=current_conditions.copy(),
                tree_idx=tree_idx,
                node_idx=node_idx,
            )

            # Compute support
            if X_train is not None:
                satisfied = rule.evaluate(X_train)
                rule.support = float(np.mean(satisfied))
            else:
                # Estimate from tree node samples
                rule.support = n_node_samples[node_idx] / n_node_samples[0]

            rules.append(rule)

        # If not a leaf, recurse to children
        left_child = children_left[node_idx]
        right_child = children_right[node_idx]

        # In sklearn, leaf nodes have children_left[node] == children_right[node] == TREE_LEAF (-1)
        # or we can check if both children point to same value (which only happens for undefined)
        if left_child != right_child:  # Not a leaf
            feat_idx = feature[node_idx]
            thresh = threshold[node_idx]
            feat_name = feature_names[feat_idx]

            # Left child: feature <= threshold
            left_condition = Condition(
                feature_idx=feat_idx,
                feature_name=feat_name,
                operator=Operator.LE,
                threshold=thresh,
            )
            recurse(left_child, current_conditions + [left_condition])

            # Right child: feature > threshold
            right_condition = Condition(
                feature_idx=feat_idx,
                feature_name=feat_name,
                operator=Operator.GT,
                threshold=thresh,
            )
            recurse(right_child, current_conditions + [right_condition])

    # Start recursion from root
    recurse(0, [])

    return rules


def extract_rules_from_ensemble(
    ensemble,
    feature_names: list[str],
    X_train: np.ndarray | None = None,
) -> RuleEnsemble:
    """
    Extract all rules from a tree ensemble.

    Parameters
    ----------
    ensemble : fitted ensemble estimator
        Must have `estimators_` attribute containing trees.
        Supports: RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting.
    feature_names : list of str
        Names of input features.
    X_train : ndarray, optional
        Training data for computing support.

    Returns
    -------
    rule_ensemble : RuleEnsemble
        Collection of all extracted rules.
    """
    all_rules = []
    trees = _get_trees_from_ensemble(ensemble)

    # Extract rules from each tree
    for tree_idx, tree in enumerate(trees):
        tree_rules = extract_rules_from_tree(tree, feature_names, tree_idx, X_train)
        all_rules.extend(tree_rules)

    return RuleEnsemble(
        rules=all_rules,
        n_features=len(feature_names),
        feature_names=feature_names,
    )


def _get_trees_from_ensemble(ensemble) -> list:
    """
    Extract individual trees from various ensemble types.

    Parameters
    ----------
    ensemble : fitted ensemble estimator
        Supports RandomForest, ExtraTrees, GradientBoosting, HistGradientBoosting,
        and other sklearn-compatible tree ensembles.

    Returns
    -------
    trees : list
        List of individual tree estimators.
    """
    trees = []

    if hasattr(ensemble, "estimators_"):
        estimators = ensemble.estimators_

        # Check if it's a 2D array (GradientBoosting has shape [n_estimators, n_classes])
        if hasattr(estimators, "ndim") and estimators.ndim == 2:
            # GradientBoosting: array of shape (n_estimators, n_classes)
            for stage in estimators:
                for tree in stage:
                    if hasattr(tree, "tree_"):
                        trees.append(tree)
        elif hasattr(estimators, "__iter__"):
            # List of estimators (RandomForest, ExtraTrees)
            for item in estimators:
                if hasattr(item, "tree_"):
                    # Direct tree estimator
                    trees.append(item)
                elif hasattr(item, "__iter__"):
                    # Nested list (like GradientBoosting)
                    for tree in item:
                        if hasattr(tree, "tree_"):
                            trees.append(tree)
                else:
                    trees.append(item)

    elif hasattr(ensemble, "_predictors"):
        # HistGradientBoosting: has _predictors attribute
        for stage in ensemble._predictors:
            for predictor in stage:
                if hasattr(predictor, "get_nodes"):
                    # This is a TreePredictor - needs different handling
                    # For now, skip HistGradientBoosting as it uses a different tree format
                    pass

    else:
        raise ValueError(
            f"Ensemble type {type(ensemble).__name__} is not supported. "
            "Must have estimators_ attribute containing trees."
        )

    return trees
