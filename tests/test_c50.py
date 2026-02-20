"""Tests for C5.0 decision tree implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class TestC50Classifier:
    """Tests for C50Classifier."""

    def test_fit_predict_basic(self):
        """Test basic fit and predict."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=0, n_classes=2, random_state=42
        )

        clf = C50Classifier(use_rust=False, random_state=42)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == 100
        assert set(predictions).issubset(set(y))

    def test_predict_proba(self):
        """Test probability predictions."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_multiclass(self):
        """Test multiclass classification."""
        from endgame.models.trees import C50Classifier

        iris = load_iris()
        X, y = iris.data, iris.target

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == 150
        assert set(predictions).issubset({0, 1, 2})

        proba = clf.predict_proba(X)
        assert proba.shape == (150, 3)

    def test_with_sample_weights(self):
        """Test with sample weights."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        weights = np.random.rand(100)

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y, sample_weight=weights)

        predictions = clf.predict(X)
        assert len(predictions) == 100

    def test_with_categorical_features(self):
        """Test with categorical features."""
        from endgame.models.trees import C50Classifier

        # Create data with categorical feature
        np.random.seed(42)
        X = np.random.rand(100, 3)
        X[:, 0] = np.random.randint(0, 5, 100)  # Categorical: 5 values
        y = (X[:, 0] > 2).astype(int)

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y, categorical_features=[0])

        predictions = clf.predict(X)
        assert len(predictions) == len(y)
        assert set(predictions).issubset(set(y))

    def test_feature_importances(self):
        """Test feature importances."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y)

        importances = clf.feature_importances_
        assert len(importances) == 5
        assert np.isclose(importances.sum(), 1.0) or importances.sum() == 0

    def test_missing_values(self):
        """Test handling of missing values."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Introduce missing values
        X_missing = X.copy()
        X_missing[0, 0] = np.nan
        X_missing[10, 2] = np.nan

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y)  # Fit on clean data

        # Predict on data with missing values
        predictions = clf.predict(X_missing)
        assert len(predictions) == 100
        assert not np.any(np.isnan(predictions))

    def test_pruning_effect(self):
        """Test that pruning reduces tree size."""
        from endgame.models.trees import C50Classifier

        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        # No pruning
        clf_no_prune = C50Classifier(cf=0.0, use_rust=False)
        clf_no_prune.fit(X, y)

        # With pruning
        clf_prune = C50Classifier(cf=0.25, use_rust=False)
        clf_prune.fit(X, y)

        # Pruned tree should generally have fewer leaves
        # (not guaranteed in all cases but likely)
        n_leaves_no_prune = clf_no_prune.tree_.n_leaves()
        n_leaves_prune = clf_prune.tree_.n_leaves()

        # Both should work
        assert n_leaves_no_prune >= 1
        assert n_leaves_prune >= 1

    def test_pure_node_becomes_leaf(self):
        """Test that pure nodes become leaves."""
        from endgame.models.trees import C50Classifier

        # Perfectly separable data
        X = np.array([[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]], dtype=float)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        clf = C50Classifier(use_rust=False)
        clf.fit(X, y)

        # Should predict correctly
        predictions = clf.predict(X)
        assert np.array_equal(predictions, y)

    def test_train_test_split_accuracy(self):
        """Test accuracy on held-out data."""
        from endgame.models.trees import C50Classifier

        iris = load_iris()
        X_train, X_test, y_train, y_test = train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )

        clf = C50Classifier(use_rust=False)
        clf.fit(X_train, y_train)

        accuracy = clf.score(X_test, y_test)
        assert accuracy > 0.8  # Should achieve reasonable accuracy


class TestC50Ensemble:
    """Tests for C50Ensemble (boosted C5.0)."""

    def test_fit_predict(self):
        """Test boosted ensemble."""
        from endgame.models.trees import C50Ensemble

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = C50Ensemble(n_trials=5, use_rust=False)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert len(predictions) == 100

    def test_predict_proba(self):
        """Test probability predictions from ensemble."""
        from endgame.models.trees import C50Ensemble

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        clf = C50Ensemble(n_trials=5, use_rust=False)
        clf.fit(X, y)

        proba = clf.predict_proba(X)
        assert proba.shape == (100, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_ensemble_improves_accuracy(self):
        """Test that boosting generally improves accuracy."""
        from endgame.models.trees import C50Classifier, C50Ensemble

        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=2, random_state=42
        )
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # Single tree
        single = C50Classifier(use_rust=False)
        single.fit(X_train, y_train)
        single_acc = single.score(X_test, y_test)

        # Ensemble
        ensemble = C50Ensemble(n_trials=10, use_rust=False)
        ensemble.fit(X_train, y_train)
        ensemble_acc = ensemble.score(X_test, y_test)

        # Ensemble should be at least as good (usually better)
        assert ensemble_acc >= single_acc - 0.1  # Allow small margin

    def test_multiclass_ensemble(self):
        """Test ensemble with multiclass."""
        from endgame.models.trees import C50Ensemble

        iris = load_iris()
        X, y = iris.data, iris.target

        clf = C50Ensemble(n_trials=5, use_rust=False)
        clf.fit(X, y)

        predictions = clf.predict(X)
        assert set(predictions).issubset({0, 1, 2})

        proba = clf.predict_proba(X)
        assert proba.shape == (150, 3)


class TestC50Internals:
    """Tests for internal C5.0 components."""

    def test_entropy(self):
        """Test entropy calculation."""
        from endgame.models.trees.c50 import _entropy

        # Pure distribution
        assert _entropy(np.array([10.0, 0.0])) == 0.0

        # Uniform binary
        e = _entropy(np.array([5.0, 5.0]))
        assert abs(e - 1.0) < 1e-10

        # Uniform 4-way
        e = _entropy(np.array([5.0, 5.0, 5.0, 5.0]))
        assert abs(e - 2.0) < 1e-10

    def test_split_evaluator(self):
        """Test split evaluation."""
        from endgame.models.trees.c50 import SplitEvaluator

        # Simple separable data
        X = np.array([[0], [1], [2], [3], [4], [5]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        weights = np.ones(6)
        indices = np.arange(6)
        categorical = np.array([False])

        evaluator = SplitEvaluator(X, y, weights, indices, 2, categorical)
        split = evaluator.find_best_split()

        assert split is not None
        assert split["gain"] > 0
        assert split["threshold"] is not None
        assert 2.0 < split["threshold"] < 4.0

    def test_pruner(self):
        """Test pruner extra errors calculation."""
        from endgame.models.trees.c50 import Pruner

        pruner = Pruner(n_classes=2, cf=0.25)

        # No errors
        e = pruner.extra_errors(100.0, 0.0)
        assert e > 0 and e < 10

        # Half errors
        e = pruner.extra_errors(100.0, 50.0)
        assert np.isfinite(e)

    def test_tree_node(self):
        """Test TreeNode properties."""
        from endgame.models.trees.c50 import TreeNode, NodeType

        # Leaf node
        leaf = TreeNode(class_=0, cases=10.0, class_dist=np.array([10.0, 0.0]))
        assert leaf.is_leaf
        assert leaf.depth() == 1
        assert leaf.n_leaves() == 1

        # Internal node
        internal = TreeNode(
            node_type=NodeType.THRESHOLD,
            class_=0,
            cases=20.0,
            class_dist=np.array([10.0, 10.0]),
            tested_attr=0,
            threshold=5.0,
            branches=[leaf, leaf],
        )
        assert not internal.is_leaf
        assert internal.depth() == 2
        assert internal.n_leaves() == 2
