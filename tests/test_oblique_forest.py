"""Tests for Oblique Random Forest implementation."""

import numpy as np
import pytest
from sklearn.datasets import (
    make_classification,
    make_regression,
    make_moons,
    make_circles,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.estimator_checks import parametrize_with_checks

from endgame.models.trees import (
    ObliqueRandomForestClassifier,
    ObliqueRandomForestRegressor,
    ObliqueDecisionTreeClassifier,
    ObliqueDecisionTreeRegressor,
)
from endgame.models.trees import _HAS_TREEPLE
from endgame.models.trees.oblique_splits import (
    ObliqueSplit,
    compute_gini,
    compute_entropy,
    compute_mse,
    get_ridge_directions,
    get_pca_directions,
    get_lda_directions,
    get_random_directions,
    get_svm_directions,
    get_householder_directions,
)


class TestObliqueSplit:
    """Tests for ObliqueSplit dataclass."""

    def test_compute_projection(self):
        """Test projection computation."""
        split = ObliqueSplit(
            feature_indices=np.array([0, 2]),
            coefficients=np.array([0.5, 0.5]),
            threshold=1.0,
        )
        X = np.array([[1, 0, 1], [2, 0, 2], [0, 0, 0]])
        projection = split.compute_projection(X)
        np.testing.assert_array_almost_equal(projection, [1.0, 2.0, 0.0])

    def test_apply(self):
        """Test split application."""
        split = ObliqueSplit(
            feature_indices=np.array([0]),
            coefficients=np.array([1.0]),
            threshold=1.5,
        )
        X = np.array([[1], [2], [3], [0]])
        goes_left = split.apply(X)
        np.testing.assert_array_equal(goes_left, [True, False, False, True])

    def test_str_representation(self):
        """Test string representation."""
        split = ObliqueSplit(
            feature_indices=np.array([0, 1]),
            coefficients=np.array([0.5, -0.3]),
            threshold=2.0,
        )
        s = str(split)
        assert "x0" in s
        assert "x1" in s
        assert "2.0" in s


class TestImpurityFunctions:
    """Tests for impurity computation functions."""

    def test_gini_pure_node(self):
        """Gini should be 0 for pure nodes."""
        y = np.array([0, 0, 0, 0, 0])
        assert compute_gini(y) == 0.0

    def test_gini_impure_node(self):
        """Gini should be 0.5 for maximally impure binary."""
        y = np.array([0, 0, 1, 1])
        assert abs(compute_gini(y) - 0.5) < 1e-10

    def test_gini_with_weights(self):
        """Test Gini with sample weights."""
        y = np.array([0, 0, 1, 1])
        weights = np.array([2, 2, 1, 1])  # More weight on class 0
        gini = compute_gini(y, weights)
        # Class 0: 4/6, Class 1: 2/6
        expected = 1 - (4/6)**2 - (2/6)**2
        assert abs(gini - expected) < 1e-10

    def test_entropy_pure_node(self):
        """Entropy should be 0 for pure nodes."""
        y = np.array([1, 1, 1, 1])
        assert compute_entropy(y) == 0.0

    def test_entropy_impure_node(self):
        """Entropy should be 1 for maximally impure binary."""
        y = np.array([0, 0, 1, 1])
        assert abs(compute_entropy(y) - 1.0) < 1e-10

    def test_mse_constant(self):
        """MSE should be 0 for constant target."""
        y = np.array([5.0, 5.0, 5.0])
        assert compute_mse(y) == 0.0

    def test_mse_variance(self):
        """MSE should equal variance."""
        y = np.array([1.0, 2.0, 3.0])
        expected = np.var(y)
        assert abs(compute_mse(y) - expected) < 1e-10


class TestDirectionMethods:
    """Tests for oblique direction finding methods."""

    @pytest.fixture
    def classification_data(self):
        """Generate classification data."""
        X, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            n_redundant=0, random_state=42
        )
        return X, y

    @pytest.fixture
    def regression_data(self):
        """Generate regression data."""
        X, y = make_regression(
            n_samples=100, n_features=5, n_informative=3,
            random_state=42
        )
        return X, y

    def test_ridge_directions_shape(self, classification_data):
        """Ridge should return correct shape."""
        X, y = classification_data
        directions = get_ridge_directions(X, y)
        assert directions.shape == (1, 5)

    def test_ridge_directions_normalized(self, classification_data):
        """Ridge directions should be normalized."""
        X, y = classification_data
        directions = get_ridge_directions(X, y)
        norm = np.linalg.norm(directions)
        assert abs(norm - 1.0) < 1e-10

    def test_pca_directions_shape(self, classification_data):
        """PCA should return correct shape."""
        X, _ = classification_data
        directions = get_pca_directions(X, n_directions=2)
        assert directions.shape == (2, 5)

    def test_pca_directions_orthogonal(self, classification_data):
        """PCA directions should be approximately orthogonal."""
        X, _ = classification_data
        directions = get_pca_directions(X, n_directions=2)
        dot_product = np.dot(directions[0], directions[1])
        assert abs(dot_product) < 1e-10

    def test_lda_directions_shape(self, classification_data):
        """LDA should return correct shape."""
        X, y = classification_data
        directions = get_lda_directions(X, y, n_directions=1)
        assert directions.shape[1] == 5

    def test_random_directions_sparsity(self, classification_data):
        """Random directions should be sparse."""
        X, _ = classification_data
        rng = np.random.RandomState(42)
        directions = get_random_directions(
            X, n_directions=3, feature_combinations=2, random_state=rng
        )
        # Each direction should have at most 2 non-zero coefficients
        for direction in directions:
            n_nonzero = np.sum(np.abs(direction) > 1e-10)
            assert n_nonzero <= 2

    def test_svm_directions(self, classification_data):
        """SVM should produce valid directions."""
        X, y = classification_data
        directions = get_svm_directions(X, y)
        assert directions.shape[1] == 5
        # Should be normalized
        for direction in directions:
            norm = np.linalg.norm(direction)
            assert abs(norm - 1.0) < 1e-6

    def test_householder_directions(self, classification_data):
        """Householder should produce valid directions."""
        X, _ = classification_data
        rng = np.random.RandomState(42)
        directions = get_householder_directions(X, n_directions=3, random_state=rng)
        assert directions.shape == (3, 5)
        # Should be normalized
        for direction in directions:
            norm = np.linalg.norm(direction)
            assert abs(norm - 1.0) < 1e-6


class TestObliqueDecisionTreeClassifier:
    """Tests for ObliqueDecisionTreeClassifier."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        assert predictions.shape == (100,)
        assert set(predictions).issubset(set(y))

    def test_predict_proba(self):
        """Test probability predictions."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        
        proba = tree.predict_proba(X)
        assert proba.shape == (100, 2)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_max_depth(self):
        """Test max_depth constraint."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        tree = ObliqueDecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        assert tree.get_depth() <= 3

    def test_min_samples_split(self):
        """Test min_samples_split constraint."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeClassifier(min_samples_split=50, random_state=42)
        tree.fit(X, y)
        
        # Tree should be shallow
        assert tree.get_n_leaves() < 10

    def test_feature_importances(self):
        """Test feature importances computation."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        
        assert hasattr(tree, "feature_importances_")
        assert tree.feature_importances_.shape == (5,)
        assert np.all(tree.feature_importances_ >= 0)
        assert abs(tree.feature_importances_.sum() - 1.0) < 1e-10 or tree.feature_importances_.sum() == 0

    def test_apply(self):
        """Test apply method returns leaf indices."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeClassifier(random_state=42)
        tree.fit(X, y)
        
        leaves = tree.apply(X)
        assert leaves.shape == (100,)
        assert leaves.dtype in [np.int64, np.int32]

    def test_all_oblique_methods(self):
        """Test all oblique split methods."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        methods = ["ridge", "pca", "lda", "random", "svm", "householder"]
        
        for method in methods:
            tree = ObliqueDecisionTreeClassifier(
                oblique_method=method, random_state=42
            )
            tree.fit(X, y)
            score = tree.score(X, y)
            assert score > 0.5, f"Method {method} failed with score {score}"


class TestObliqueDecisionTreeRegressor:
    """Tests for ObliqueDecisionTreeRegressor."""

    def test_basic_fit_predict(self):
        """Test basic fit and predict."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeRegressor(random_state=42)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        assert predictions.shape == (100,)

    def test_max_depth(self):
        """Test max_depth constraint."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        tree = ObliqueDecisionTreeRegressor(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        assert tree.get_depth() <= 3

    def test_criterion_squared_error(self):
        """Test squared_error criterion."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeRegressor(
            criterion="squared_error", random_state=42
        )
        tree.fit(X, y)
        score = tree.score(X, y)
        assert score > 0.5

    def test_criterion_absolute_error(self):
        """Test absolute_error criterion."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        tree = ObliqueDecisionTreeRegressor(
            criterion="absolute_error", random_state=42
        )
        tree.fit(X, y)
        score = tree.score(X, y)
        assert score > 0.3


class TestObliqueRandomForestClassifier:
    """Tests for ObliqueRandomForestClassifier."""

    def test_basic_classification(self):
        """Basic classification test."""
        X, y = make_classification(
            n_samples=200, n_features=10, n_informative=5,
            n_redundant=2, random_state=42
        )
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        accuracy = clf.score(X, y)
        assert accuracy > 0.8

    def test_oblique_advantage_on_linear_boundary(self):
        """Oblique RF should handle linear boundaries well."""
        np.random.seed(42)
        
        # Create data with a diagonal decision boundary
        n_samples = 500
        X = np.random.randn(n_samples, 10)
        # Boundary: x0 + x1 + x2 > 0
        y = (X[:, 0] + X[:, 1] + X[:, 2] > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        # Standard RF
        rf = RandomForestClassifier(n_estimators=50, random_state=42)
        rf.fit(X_train, y_train)
        rf_score = rf.score(X_test, y_test)
        
        # Oblique RF (treeple's version doesn't have oblique_method param;
        # it always uses oblique splits by default)
        oblique_rf = ObliqueRandomForestClassifier(
            n_estimators=50, random_state=42
        )
        oblique_rf.fit(X_train, y_train)
        oblique_score = oblique_rf.score(X_test, y_test)
        
        # Oblique should do at least as well (or close)
        assert oblique_score >= rf_score - 0.1

    @pytest.mark.skipif(
        _HAS_TREEPLE,
        reason="treeple's ObliqueRandomForestClassifier does not expose oblique_method",
    )
    def test_all_oblique_methods(self):
        """Test all oblique split methods (pure-Python fallback only)."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)

        methods = ["ridge", "pca", "lda", "random", "svm", "householder"]

        for method in methods:
            clf = ObliqueRandomForestClassifier(
                n_estimators=10, oblique_method=method, random_state=42
            )
            clf.fit(X, y)
            score = clf.score(X, y)
            assert score > 0.7, f"Method {method} failed with score {score}"

    def test_multiclass(self):
        """Test multi-class classification."""
        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=5,
            n_classes=5, n_clusters_per_class=1, random_state=42
        )
        
        clf = ObliqueRandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)
        
        assert clf.n_classes_ == 5
        assert clf.predict(X).shape == (300,)
        assert clf.predict_proba(X).shape == (300, 5)

    def test_feature_importances(self):
        """Feature importances should be computed."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        assert hasattr(clf, "feature_importances_")
        assert clf.feature_importances_.shape == (10,)
        assert np.allclose(clf.feature_importances_.sum(), 1.0)
        assert np.all(clf.feature_importances_ >= 0)

    def test_oob_score(self):
        """OOB score computation."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        clf = ObliqueRandomForestClassifier(
            n_estimators=50, oob_score=True, random_state=42
        )
        clf.fit(X, y)
        
        assert hasattr(clf, "oob_score_")
        assert 0 <= clf.oob_score_ <= 1
        assert hasattr(clf, "oob_decision_function_")

    def test_predict_proba_sums_to_one(self):
        """Predicted probabilities should sum to 1."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        proba = clf.predict_proba(X)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_sample_weight(self):
        """Sample weights should affect fit."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        # Heavy weight on first half
        weights = np.ones(200)
        weights[:100] = 10.0
        
        clf1 = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf1.fit(X, y)
        
        clf2 = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf2.fit(X, y, sample_weight=weights)
        
        # Predictions should differ
        pred1 = clf1.predict_proba(X)
        pred2 = clf2.predict_proba(X)
        assert not np.allclose(pred1, pred2)

    def test_warm_start(self):
        """Warm start should add trees."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = ObliqueRandomForestClassifier(
            n_estimators=10, warm_start=True, random_state=42
        )
        clf.fit(X, y)
        assert len(clf.estimators_) == 10
        
        clf.n_estimators = 20
        clf.fit(X, y)
        assert len(clf.estimators_) == 20

    def test_parallel_training(self):
        """Parallel training should work."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        clf1 = ObliqueRandomForestClassifier(
            n_estimators=20, n_jobs=1, random_state=42
        )
        clf1.fit(X, y)
        
        clf2 = ObliqueRandomForestClassifier(
            n_estimators=20, n_jobs=2, random_state=42
        )
        clf2.fit(X, y)
        
        # Both should produce valid results
        assert clf1.score(X, y) > 0.8
        assert clf2.score(X, y) > 0.8

    def test_apply(self):
        """Apply should return leaf indices."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        leaves = clf.apply(X)
        assert leaves.shape == (100, 10)
        assert leaves.dtype in [np.int32, np.int64]

    def test_moons_dataset(self):
        """Test on moons (nonlinear but with some structure)."""
        X, y = make_moons(n_samples=300, noise=0.2, random_state=42)
        
        clf = ObliqueRandomForestClassifier(n_estimators=50, random_state=42)
        scores = cross_val_score(clf, X, y, cv=5)
        
        assert np.mean(scores) > 0.80

    def test_max_depth_constraint(self):
        """max_depth should be respected."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        clf = ObliqueRandomForestClassifier(
            n_estimators=10, max_depth=3, random_state=42
        )
        clf.fit(X, y)
        
        # Check that all trees have depth <= 3
        for tree in clf.estimators_:
            max_tree_depth = tree.get_depth()
            assert max_tree_depth <= 3

    def test_bootstrap_false(self):
        """Test without bootstrap sampling."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        clf = ObliqueRandomForestClassifier(
            n_estimators=10, bootstrap=False, random_state=42
        )
        clf.fit(X, y)
        
        # Should still work
        assert clf.score(X, y) > 0.8


class TestObliqueRandomForestRegressor:
    """Tests for ObliqueRandomForestRegressor."""

    def test_basic_regression(self):
        """Test regression variant."""
        X, y = make_regression(
            n_samples=200, n_features=10, n_informative=5,
            noise=0.1, random_state=42
        )
        
        reg = ObliqueRandomForestRegressor(n_estimators=20, random_state=42)
        reg.fit(X, y)
        
        # Should have reasonable R²
        score = reg.score(X, y)
        assert score > 0.8

    def test_oob_score_regression(self):
        """OOB R² score computation."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        
        reg = ObliqueRandomForestRegressor(
            n_estimators=50, oob_score=True, random_state=42
        )
        reg.fit(X, y)
        
        assert hasattr(reg, "oob_score_")
        assert hasattr(reg, "oob_prediction_")

    def test_feature_importances_regression(self):
        """Feature importances for regression."""
        X, y = make_regression(n_samples=200, n_features=10, random_state=42)
        
        reg = ObliqueRandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X, y)
        
        assert hasattr(reg, "feature_importances_")
        assert reg.feature_importances_.shape == (10,)
        assert np.all(reg.feature_importances_ >= 0)

    def test_apply_regression(self):
        """Apply should return leaf indices."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        reg = ObliqueRandomForestRegressor(n_estimators=10, random_state=42)
        reg.fit(X, y)
        
        leaves = reg.apply(X)
        assert leaves.shape == (100, 10)

    def test_warm_start_regression(self):
        """Warm start for regressor."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        reg = ObliqueRandomForestRegressor(
            n_estimators=10, warm_start=True, random_state=42
        )
        reg.fit(X, y)
        assert len(reg.estimators_) == 10
        
        reg.n_estimators = 15
        reg.fit(X, y)
        assert len(reg.estimators_) == 15

    def test_criterion_squared_error(self):
        """Test squared_error criterion."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        reg = ObliqueRandomForestRegressor(
            n_estimators=10, criterion="squared_error", random_state=42
        )
        reg.fit(X, y)
        assert reg.score(X, y) > 0.7

    def test_criterion_absolute_error(self):
        """Test absolute_error criterion."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        reg = ObliqueRandomForestRegressor(
            n_estimators=10, criterion="absolute_error", random_state=42
        )
        reg.fit(X, y)
        assert reg.score(X, y) > 0.5


class TestSklearnCompatibility:
    """Tests for sklearn estimator contract compliance."""

    def test_classifier_get_params(self):
        """Test get_params."""
        clf = ObliqueRandomForestClassifier(n_estimators=50, max_depth=5)
        params = clf.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 5

    def test_classifier_set_params(self):
        """Test set_params."""
        clf = ObliqueRandomForestClassifier()
        clf.set_params(n_estimators=30, max_depth=10)
        assert clf.n_estimators == 30
        assert clf.max_depth == 10

    def test_regressor_get_params(self):
        """Test get_params for regressor."""
        reg = ObliqueRandomForestRegressor(n_estimators=50)
        params = reg.get_params()
        assert params["n_estimators"] == 50

    def test_regressor_set_params(self):
        """Test set_params for regressor."""
        reg = ObliqueRandomForestRegressor()
        reg.set_params(n_estimators=30)
        assert reg.n_estimators == 30

    def test_clone_classifier(self):
        """Test that classifier can be cloned."""
        from sklearn.base import clone
        
        clf = ObliqueRandomForestClassifier(n_estimators=50, random_state=42)
        clf_clone = clone(clf)
        
        assert clf_clone.n_estimators == 50
        assert clf_clone.random_state == 42
        assert not hasattr(clf_clone, "estimators_")

    def test_clone_regressor(self):
        """Test that regressor can be cloned."""
        from sklearn.base import clone
        
        reg = ObliqueRandomForestRegressor(n_estimators=50, random_state=42)
        reg_clone = clone(reg)
        
        assert reg_clone.n_estimators == 50

    def test_classifier_in_pipeline(self):
        """Test classifier in sklearn pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", ObliqueRandomForestClassifier(n_estimators=10, random_state=42)),
        ])
        
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_regressor_in_pipeline(self):
        """Test regressor in sklearn pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)
        
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ObliqueRandomForestRegressor(n_estimators=10, random_state=42)),
        ])
        
        pipe.fit(X, y)
        predictions = pipe.predict(X)
        assert predictions.shape == (100,)

    def test_cross_val_score(self):
        """Test with cross_val_score."""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3)
        
        assert len(scores) == 3
        assert np.mean(scores) > 0.7


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_sample(self):
        """Test with minimal data."""
        X = np.array([[1, 2, 3]])
        y = np.array([0])
        
        clf = ObliqueRandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        
        pred = clf.predict(X)
        assert pred[0] == 0

    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        
        clf = ObliqueRandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        
        assert clf.score(X, y) > 0.8

    def test_pure_class_node(self):
        """Test when a node becomes pure."""
        X = np.array([[1, 1], [1, 2], [1, 3], [10, 1], [10, 2], [10, 3]])
        y = np.array([0, 0, 0, 1, 1, 1])
        
        clf = ObliqueRandomForestClassifier(n_estimators=5, random_state=42)
        clf.fit(X, y)
        
        assert clf.score(X, y) == 1.0

    def test_many_classes(self):
        """Test with many classes."""
        X, y = make_classification(
            n_samples=500, n_features=20, n_informative=10,
            n_classes=10, n_clusters_per_class=1, random_state=42
        )
        
        clf = ObliqueRandomForestClassifier(n_estimators=20, random_state=42)
        clf.fit(X, y)
        
        assert clf.n_classes_ == 10
        assert clf.predict_proba(X).shape == (500, 10)

    def test_high_dimensional(self):
        """Test with many features."""
        X, y = make_classification(
            n_samples=100, n_features=100, n_informative=20,
            n_redundant=10, random_state=42
        )
        
        clf = ObliqueRandomForestClassifier(
            n_estimators=10, max_features="sqrt", random_state=42
        )
        clf.fit(X, y)
        
        assert clf.score(X, y) > 0.7

    def test_constant_feature(self):
        """Test with constant features."""
        X = np.random.randn(100, 5)
        X[:, 2] = 0  # Constant feature
        y = (X[:, 0] > 0).astype(int)
        
        clf = ObliqueRandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)
        
        assert clf.score(X, y) > 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
