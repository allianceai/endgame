"""Tests for the interactive decision tree visualization module."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import load_iris, make_regression, make_classification
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from endgame.visualization import TreeVisualizer
from endgame.visualization.tree_visualizer import (
    VizNode,
    _extract_sklearn_tree,
)


# ===== Fixtures =====

@pytest.fixture
def iris_data():
    X, y = load_iris(return_X_y=True)
    return X, y, load_iris().feature_names, load_iris().target_names.tolist()


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    return X, y


@pytest.fixture
def binary_data():
    X, y = make_classification(n_samples=200, n_features=6, n_classes=2, random_state=42)
    return X, y


@pytest.fixture
def fitted_clf(iris_data):
    X, y, _, _ = iris_data
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X, y)
    return clf


@pytest.fixture
def fitted_reg(regression_data):
    X, y = regression_data
    reg = DecisionTreeRegressor(max_depth=3, random_state=42)
    reg.fit(X, y)
    return reg


# ===== VizNode Tests =====

class TestVizNode:
    def test_to_dict_leaf(self):
        node = VizNode(node_id=0, is_leaf=True, n_samples=50, predicted_class="setosa")
        d = node.to_dict()
        assert d["id"] == 0
        assert d["leaf"] is True
        assert d["samples"] == 50
        assert d["prediction"] == "setosa"

    def test_to_dict_internal(self):
        child = VizNode(node_id=1, is_leaf=True, n_samples=25)
        node = VizNode(
            node_id=0, is_leaf=False, n_samples=50,
            split_label="age <= 30.5", split_feature="age",
            children=[child], child_labels=["True"],
        )
        d = node.to_dict()
        assert d["leaf"] is False
        assert d["split"] == "age <= 30.5"
        assert len(d["children"]) == 1

    def test_to_dict_regression(self):
        node = VizNode(node_id=0, is_leaf=True, predicted_value=3.14)
        d = node.to_dict()
        assert d["predValue"] == 3.14

    def test_class_distribution_rounding(self):
        node = VizNode(class_distribution=[0.33333, 0.66667])
        d = node.to_dict()
        assert d["classDist"] == [0.3333, 0.6667]


# ===== Sklearn Tree Extraction Tests =====

class TestSklearnExtraction:
    def test_classifier_extraction(self, fitted_clf, iris_data):
        _, _, feature_names, class_names = iris_data
        root = _extract_sklearn_tree(fitted_clf, feature_names, class_names, True)
        assert not root.is_leaf
        assert root.n_samples > 0
        assert len(root.children) == 2
        assert root.split_label  # has split condition
        assert root.split_feature in feature_names

    def test_regressor_extraction(self, fitted_reg):
        root = _extract_sklearn_tree(fitted_reg, None, None, False)
        assert not root.is_leaf
        # Find a leaf node
        def find_leaf(node):
            if node.is_leaf:
                return node
            for c in node.children:
                result = find_leaf(c)
                if result:
                    return result
            return None
        leaf = find_leaf(root)
        assert leaf is not None
        assert leaf.predicted_value is not None

    def test_no_feature_names(self, fitted_clf):
        root = _extract_sklearn_tree(fitted_clf, None, None, True)
        assert root.split_feature.startswith("feature_")

    def test_tree_depth(self, iris_data):
        X, y, fnames, cnames = iris_data
        clf = DecisionTreeClassifier(max_depth=2, random_state=42).fit(X, y)
        root = _extract_sklearn_tree(clf, fnames, cnames, True)
        def max_depth(node, d=0):
            if node.is_leaf:
                return d
            return max(max_depth(c, d+1) for c in node.children)
        assert max_depth(root) <= 2


# ===== TreeVisualizer Tests =====

class TestTreeVisualizer:
    def test_basic_classifier(self, fitted_clf, iris_data):
        _, _, fnames, cnames = iris_data
        viz = TreeVisualizer(fitted_clf, feature_names=fnames, class_names=cnames)
        assert viz._root is not None
        assert not viz._root.is_leaf

    def test_basic_regressor(self, fitted_reg):
        viz = TreeVisualizer(fitted_reg, feature_names=[f"f{i}" for i in range(5)])
        assert viz._root is not None

    def test_save_html(self, fitted_clf, iris_data):
        _, _, fnames, cnames = iris_data
        viz = TreeVisualizer(fitted_clf, feature_names=fnames, class_names=cnames,
                           title="Test Tree")
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = viz.save(f.name)
            assert path.exists()
            content = path.read_text()
            assert 'Test Tree' in content
            assert 'tree-container' in content
            assert 'expandAll' in content
            assert 'collapseAll' in content
            # Check it contains tree data
            assert 'treeData' in content

    def test_save_adds_extension(self, fitted_clf):
        viz = TreeVisualizer(fitted_clf)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = viz.save(Path(tmpdir) / "tree")
            assert path.suffix == '.html'
            assert path.exists()

    def test_to_json(self, fitted_clf, iris_data):
        _, _, fnames, cnames = iris_data
        viz = TreeVisualizer(fitted_clf, feature_names=fnames, class_names=cnames)
        j = viz.to_json()
        data = json.loads(j)
        assert "id" in data
        assert "children" in data
        assert "split" in data

    def test_json_valid_structure(self, fitted_clf):
        viz = TreeVisualizer(fitted_clf)
        data = json.loads(viz.to_json())
        # Check required fields
        for key in ["id", "leaf", "split", "samples", "impurity", "children"]:
            assert key in data, f"Missing key: {key}"

    def test_color_by_options(self, fitted_clf, iris_data):
        _, _, fnames, cnames = iris_data
        for color_by in ['prediction', 'impurity', 'samples']:
            viz = TreeVisualizer(fitted_clf, feature_names=fnames,
                               class_names=cnames, color_by=color_by)
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                path = viz.save(f.name)
                assert path.exists()

    def test_max_depth(self, iris_data):
        X, y, fnames, cnames = iris_data
        clf = DecisionTreeClassifier(max_depth=6, random_state=42).fit(X, y)
        viz = TreeVisualizer(clf, feature_names=fnames, class_names=cnames,
                           max_depth=2)
        # Should still produce a valid tree
        assert viz._root is not None
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = viz.save(f.name)
            content = path.read_text()
            assert 'MAX_DEPTH = 2' in content

    def test_palettes(self, fitted_clf):
        for palette in ['tableau', 'viridis', 'pastel', 'dark']:
            viz = TreeVisualizer(fitted_clf, palette=palette)
            with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
                path = viz.save(f.name)
                assert path.exists()

    def test_title_escaping(self, fitted_clf):
        viz = TreeVisualizer(fitted_clf, title="Tree <script>alert('xss')</script>")
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = viz.save(f.name)
            content = path.read_text()
            assert "<script>alert" not in content.split("</style>")[0]

    def test_repr_html(self, fitted_clf, iris_data):
        _, _, fnames, cnames = iris_data
        viz = TreeVisualizer(fitted_clf, feature_names=fnames, class_names=cnames)
        html = viz._repr_html_()
        assert 'tree-container' in html
        assert len(html) > 1000  # Should be substantial HTML

    def test_random_forest(self, iris_data):
        X, y, fnames, cnames = iris_data
        rf = RandomForestClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
        viz = TreeVisualizer(rf, feature_names=fnames, class_names=cnames, tree_index=0)
        assert viz._root is not None
        # Can also visualize tree_index=2
        viz2 = TreeVisualizer(rf, feature_names=fnames, class_names=cnames, tree_index=2)
        assert viz2._root is not None

    def test_gradient_boosting(self, iris_data):
        X, y, fnames, cnames = iris_data
        gb = GradientBoostingClassifier(n_estimators=5, max_depth=3, random_state=42).fit(X, y)
        viz = TreeVisualizer(gb, feature_names=fnames, class_names=cnames, tree_index=0)
        assert viz._root is not None

    def test_unsupported_model(self):
        class FakeModel:
            pass
        with pytest.raises(ValueError, match="Unsupported model type"):
            TreeVisualizer(FakeModel())

    def test_binary_classification(self, binary_data):
        X, y = binary_data
        clf = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        viz = TreeVisualizer(clf, class_names=["negative", "positive"])
        data = json.loads(viz.to_json())
        assert len(data["classDist"]) == 2

    def test_large_tree(self, iris_data):
        """Test with a deeper tree to verify rendering handles large trees."""
        X, y, fnames, cnames = iris_data
        clf = DecisionTreeClassifier(max_depth=10, random_state=42).fit(X, y)
        viz = TreeVisualizer(clf, feature_names=fnames, class_names=cnames)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = viz.save(f.name)
            assert path.exists()
            # File should be self-contained (no external deps)
            content = path.read_text()
            assert 'src=' not in content or 'tree-svg' in content

    def test_stump(self, iris_data):
        """Test with max_depth=1 (stump)."""
        X, y, fnames, cnames = iris_data
        clf = DecisionTreeClassifier(max_depth=1, random_state=42).fit(X, y)
        viz = TreeVisualizer(clf, feature_names=fnames, class_names=cnames)
        data = json.loads(viz.to_json())
        assert len(data["children"]) == 2
        assert data["children"][0]["leaf"] or data["children"][1]["leaf"]

    def test_html_self_contained(self, fitted_clf, iris_data):
        """Verify the HTML has no external dependencies."""
        _, _, fnames, cnames = iris_data
        viz = TreeVisualizer(fitted_clf, feature_names=fnames, class_names=cnames)
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            path = viz.save(f.name)
            content = path.read_text()
            # No external script/link tags
            assert 'cdn.' not in content
            assert 'unpkg.com' not in content
            assert 'jsdelivr' not in content
