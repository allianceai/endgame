from __future__ import annotations

"""GOSDT: Generalized and Scalable Optimal Sparse Decision Trees.

GOSDT produces globally optimal decision trees that are provably optimal
within the space of all trees of a given complexity. Unlike greedy CART,
GOSDT finds the best tree, not just a locally optimal one.

References
----------
- Lin et al. "Generalized and Scalable Optimal Sparse Decision Trees"
  (ICML 2020)
- https://github.com/Jimmy-Lin/GOSDT

Example
-------
>>> from endgame.models.interpretable import GOSDTClassifier
>>> clf = GOSDTClassifier(regularization=0.01, depth_budget=5)
>>> clf.fit(X_train_binary, y_train)
>>> print(clf.get_tree_structure())
>>> predictions = clf.predict(X_test_binary)
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    from gosdt import GOSDT
    HAS_GOSDT = True
except ImportError:
    HAS_GOSDT = False
    GOSDT = None


def _check_gosdt_installed():
    """Raise ImportError if gosdt is not installed."""
    if not HAS_GOSDT:
        raise ImportError(
            "The 'gosdt' package is required for GOSDTClassifier. "
            "Install with: pip install gosdt"
        )


class GOSDTClassifier(ClassifierMixin, BaseEstimator):
    """Globally Optimal Sparse Decision Tree Classifier.

    GOSDT produces decision trees that are provably optimal within the
    search space defined by regularization and depth constraints. This
    guarantees finding the best tree, not just a locally optimal one.

    Note: GOSDT requires binary features. Use auto_discretize=True for
    continuous features.

    If GOSDT is not installed, falls back to sklearn DecisionTreeClassifier
    with similar constraints.

    Parameters
    ----------
    regularization : float, default=0.01
        Regularization parameter. Higher values produce smaller trees.
        This is the cost of adding a leaf to the tree.

    depth_budget : int, default=5
        Maximum depth of the tree. Smaller values produce simpler trees.

    time_limit : int, default=60
        Maximum time in seconds for optimization.

    similar_support : bool, default=True
        Whether to use similar support bound for pruning.

    look_ahead : bool, default=True
        Whether to use look-ahead bound for pruning.

    auto_discretize : bool, default=True
        If True, automatically discretize continuous features.

    n_bins : int, default=5
        Number of bins for discretization.

    random_state : int, optional
        Random seed.

    fallback_to_cart : bool, default=True
        If True, use sklearn DecisionTree when GOSDT not installed.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    n_features_in_ : int
        Number of features.

    tree_ : GOSDT or DecisionTreeClassifier
        Fitted tree model.

    feature_names_ : list of str
        Feature names (after discretization).

    feature_importances_ : ndarray
        Feature importance scores.

    Examples
    --------
    >>> from endgame.models.interpretable import GOSDTClassifier
    >>> clf = GOSDTClassifier(regularization=0.01, depth_budget=5)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.get_tree_structure())
    >>> predictions = clf.predict(X_test)

    Notes
    -----
    For best interpretability:
    - Use small depth_budget (3-5) for simple trees
    - Use higher regularization for sparser trees
    - Provide meaningful feature names
    - Binary features produce the most interpretable output
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        regularization: float = 0.01,
        depth_budget: int = 5,
        time_limit: int = 60,
        similar_support: bool = True,
        look_ahead: bool = True,
        auto_discretize: bool = True,
        n_bins: int = 5,
        random_state: int | None = None,
        fallback_to_cart: bool = True,
    ):
        self.regularization = regularization
        self.depth_budget = depth_budget
        self.time_limit = time_limit
        self.similar_support = similar_support
        self.look_ahead = look_ahead
        self.auto_discretize = auto_discretize
        self.n_bins = n_bins
        self.random_state = random_state
        self.fallback_to_cart = fallback_to_cart

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> GOSDTClassifier:
        """Fit the GOSDT classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. If auto_discretize=False, should be binary.
        y : array-like of shape (n_samples,)
            Target labels (binary classification only).
        feature_names : list of str, optional
            Names for features.
        sample_weight : array-like, optional
            Sample weights (not supported by GOSDT, ignored).

        Returns
        -------
        self : GOSDTClassifier
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        if len(self.classes_) != 2:
            raise ValueError("GOSDT only supports binary classification.")

        # Feature names
        if feature_names is not None:
            self._original_feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self._original_feature_names = list(X.columns)
        else:
            self._original_feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        # Discretize if needed
        if self.auto_discretize:
            X_binary, self.feature_names_ = self._discretize(X)
        else:
            X_binary = (X > 0.5).astype(int)
            self.feature_names_ = self._original_feature_names
            self._discretizer = None

        # Fit model
        if HAS_GOSDT:
            self._fit_gosdt(X_binary, y_encoded)
        elif self.fallback_to_cart:
            self._fit_cart(X_binary, y_encoded, sample_weight)
        else:
            _check_gosdt_installed()  # Will raise ImportError

        return self

    def _discretize(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Discretize continuous features into binary features."""
        self._discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="onehot-dense",
            strategy="quantile",
            random_state=self.random_state,
        )

        X_binned = self._discretizer.fit_transform(X).astype(int)

        # Generate feature names
        feature_names = []
        for feat_idx, edges in enumerate(self._discretizer.bin_edges_):
            feat_name = self._original_feature_names[feat_idx]
            for bin_idx in range(len(edges) - 1):
                if bin_idx == 0:
                    name = f"{feat_name} <= {edges[bin_idx + 1]:.3g}"
                elif bin_idx == len(edges) - 2:
                    name = f"{feat_name} > {edges[bin_idx]:.3g}"
                else:
                    name = f"{edges[bin_idx]:.3g} < {feat_name} <= {edges[bin_idx + 1]:.3g}"
                feature_names.append(name)

        return X_binned, feature_names

    def _fit_gosdt(self, X_binary: np.ndarray, y: np.ndarray):
        """Fit using GOSDT algorithm."""
        import pandas as pd

        # GOSDT expects DataFrame
        df = pd.DataFrame(X_binary, columns=self.feature_names_)
        df["target"] = y

        # Configure GOSDT
        config = {
            "regularization": self.regularization,
            "depth_budget": self.depth_budget,
            "time_limit": self.time_limit,
            "similar_support": self.similar_support,
            "look_ahead": self.look_ahead,
        }

        # Fit
        self.tree_ = GOSDT(config)
        self.tree_.fit(df, "target")

        # Get tree structure
        self._tree_json = self.tree_.tree.json()

        # Compute feature importances from tree structure
        self._compute_feature_importances_gosdt()

        self._using_gosdt = True

    def _fit_cart(self, X_binary: np.ndarray, y: np.ndarray, sample_weight):
        """Fallback to sklearn DecisionTree."""
        # Approximate GOSDT behavior with CART
        # Use regularization to set min_impurity_decrease
        min_impurity = self.regularization * 0.1

        self.tree_ = DecisionTreeClassifier(
            max_depth=self.depth_budget,
            min_impurity_decrease=min_impurity,
            random_state=self.random_state,
        )
        self.tree_.fit(X_binary, y, sample_weight=sample_weight)

        # Get feature importances from CART
        self.feature_importances_ = self.tree_.feature_importances_

        self._using_gosdt = False

    def _compute_feature_importances_gosdt(self):
        """Compute feature importances from GOSDT tree structure."""
        importances = np.zeros(len(self.feature_names_))

        # Parse tree JSON to count feature usage
        def count_features(node):
            if node is None:
                return

            if "feature" in node:
                feat_name = node["feature"]
                if feat_name in self.feature_names_:
                    feat_idx = self.feature_names_.index(feat_name)
                    importances[feat_idx] += 1

            if "true" in node:
                count_features(node["true"])
            if "false" in node:
                count_features(node["false"])

        if hasattr(self, "_tree_json"):
            count_features(self._tree_json)

        # Normalize
        if importances.sum() > 0:
            importances = importances / importances.sum()

        self.feature_importances_ = importances

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "tree_")
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, but GOSDTClassifier expects "
                f"{self.n_features_in_} features."
            )

        # Discretize
        if self.auto_discretize and self._discretizer is not None:
            X_binary = self._discretizer.transform(X).astype(int)
        else:
            X_binary = (X > 0.5).astype(int)

        # Predict
        if self._using_gosdt:
            import pandas as pd
            df = pd.DataFrame(X_binary, columns=self.feature_names_)
            y_pred = self.tree_.predict(df).values
        else:
            y_pred = self.tree_.predict(X_binary)

        return self._label_encoder.inverse_transform(y_pred.astype(int))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Note: GOSDT produces hard predictions, so probabilities are
        approximated from leaf statistics.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self, "tree_")
        X = check_array(X)

        # Discretize
        if self.auto_discretize and self._discretizer is not None:
            X_binary = self._discretizer.transform(X).astype(int)
        else:
            X_binary = (X > 0.5).astype(int)

        if self._using_gosdt:
            # GOSDT doesn't have predict_proba, use hard predictions
            y_pred = self.predict(X)
            y_encoded = self._label_encoder.transform(y_pred)
            proba = np.zeros((len(y_pred), 2))
            proba[np.arange(len(y_pred)), y_encoded] = 1.0
        else:
            proba = self.tree_.predict_proba(X_binary)

        return proba

    def get_tree_structure(self) -> str:
        """Get a human-readable tree structure.

        Returns
        -------
        structure : str
            Formatted tree representation.
        """
        check_is_fitted(self, "tree_")

        lines = []
        lines.append("=" * 60)
        lines.append("GOSDT Decision Tree" if self._using_gosdt else "Decision Tree (CART fallback)")
        lines.append("=" * 60)
        lines.append("")

        if self._using_gosdt and hasattr(self, "_tree_json"):
            # Parse GOSDT tree JSON
            def print_node(node, indent=0):
                prefix = "  " * indent

                if "prediction" in node:
                    # Leaf node
                    pred_class = self.classes_[node["prediction"]]
                    lines.append(f"{prefix}PREDICT: {pred_class}")
                elif "feature" in node:
                    # Split node
                    feat = node["feature"]
                    lines.append(f"{prefix}IF {feat}:")
                    if "true" in node:
                        print_node(node["true"], indent + 1)
                    lines.append(f"{prefix}ELSE:")
                    if "false" in node:
                        print_node(node["false"], indent + 1)

            print_node(self._tree_json)
        else:
            # CART tree
            tree = self.tree_.tree_

            def print_cart_node(node_id, indent=0):
                prefix = "  " * indent

                if tree.children_left[node_id] == tree.children_right[node_id]:
                    # Leaf
                    values = tree.value[node_id][0]
                    pred_class = self.classes_[np.argmax(values)]
                    lines.append(f"{prefix}PREDICT: {pred_class}")
                else:
                    # Split
                    feat_idx = tree.feature[node_id]
                    thresh = tree.threshold[node_id]
                    feat_name = self.feature_names_[feat_idx] if feat_idx < len(self.feature_names_) else f"feature_{feat_idx}"

                    lines.append(f"{prefix}IF {feat_name} <= {thresh:.3g}:")
                    print_cart_node(tree.children_left[node_id], indent + 1)
                    lines.append(f"{prefix}ELSE:")
                    print_cart_node(tree.children_right[node_id], indent + 1)

            print_cart_node(0)

        lines.append("")
        lines.append("-" * 60)
        lines.append(f"Regularization: {self.regularization}")
        lines.append(f"Depth budget: {self.depth_budget}")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_rules(self) -> list[dict]:
        """Extract rules from the tree.

        Returns
        -------
        rules : list of dict
            Each dict contains:
            - 'conditions': list of condition strings
            - 'prediction': predicted class
        """
        check_is_fitted(self, "tree_")

        rules = []

        if self._using_gosdt and hasattr(self, "_tree_json"):
            def extract_rules(node, conditions=[]):
                if "prediction" in node:
                    rules.append({
                        "conditions": list(conditions),
                        "prediction": self.classes_[node["prediction"]],
                    })
                elif "feature" in node:
                    feat = node["feature"]
                    if "true" in node:
                        extract_rules(node["true"], conditions + [f"{feat} = True"])
                    if "false" in node:
                        extract_rules(node["false"], conditions + [f"{feat} = False"])

            extract_rules(self._tree_json)
        else:
            tree = self.tree_.tree_

            def extract_cart_rules(node_id, conditions=[]):
                if tree.children_left[node_id] == tree.children_right[node_id]:
                    values = tree.value[node_id][0]
                    pred_class = self.classes_[np.argmax(values)]
                    rules.append({
                        "conditions": list(conditions),
                        "prediction": pred_class,
                    })
                else:
                    feat_idx = tree.feature[node_id]
                    thresh = tree.threshold[node_id]
                    feat_name = self.feature_names_[feat_idx] if feat_idx < len(self.feature_names_) else f"feature_{feat_idx}"

                    extract_cart_rules(
                        tree.children_left[node_id],
                        conditions + [f"{feat_name} <= {thresh:.3g}"]
                    )
                    extract_cart_rules(
                        tree.children_right[node_id],
                        conditions + [f"{feat_name} > {thresh:.3g}"]
                    )

            extract_cart_rules(0)

        return rules

    def __repr__(self) -> str:
        return (
            f"GOSDTClassifier(regularization={self.regularization}, "
            f"depth_budget={self.depth_budget}, auto_discretize={self.auto_discretize})"
        )
