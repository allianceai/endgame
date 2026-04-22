"""Tests for Alternating Decision Tree and Alternating Model Tree."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

from endgame.models.trees.adtree import (
    AlternatingDecisionTreeClassifier,
    AlternatingModelTreeRegressor,
    ADTreeCondition,
    SplitType,
)


class TestADTreeCondition:
    """Tests for ADTreeCondition."""
    
    def test_threshold_condition(self):
        """Test threshold-based condition."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        condition = ADTreeCondition(
            feature_idx=0,
            threshold=2.5,
            split_type=SplitType.THRESHOLD,
        )
        result = condition.evaluate(X)
        expected = np.array([True, False, False])
        np.testing.assert_array_equal(result, expected)
    
    def test_equality_condition(self):
        """Test equality-based condition."""
        X = np.array([[1.0, 2.0], [2.0, 4.0], [1.0, 6.0]])
        condition = ADTreeCondition(
            feature_idx=0,
            threshold=1.0,
            split_type=SplitType.EQUALITY,
        )
        result = condition.evaluate(X)
        expected = np.array([True, False, True])
        np.testing.assert_array_equal(result, expected)


class TestAlternatingDecisionTreeClassifier:
    """Tests for AlternatingDecisionTreeClassifier."""
    
    @pytest.fixture
    def binary_data(self):
        """Create binary classification dataset."""
        X, y = make_classification(
            n_samples=200,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    @pytest.fixture
    def multiclass_data(self):
        """Create multiclass classification dataset."""
        X, y = make_classification(
            n_samples=300,
            n_features=10,
            n_informative=5,
            n_redundant=2,
            n_classes=3,
            n_clusters_per_class=1,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_binary_classification(self, binary_data):
        """Test binary classification."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        # Check attributes
        assert clf.n_features_in_ == 10
        assert clf.n_classes_ == 2
        assert len(clf.classes_) == 2
        assert clf.n_nodes_ > 0
        
        # Check predictions
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred).issubset(set(clf.classes_))
        
        # Check accuracy is reasonable
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.5  # Better than random
    
    def test_multiclass_classification(self, multiclass_data):
        """Test multiclass classification."""
        X_train, X_test, y_train, y_test = multiclass_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        # Check attributes
        assert clf.n_classes_ == 3
        
        # Check predictions
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Check probabilities
        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(y_test), 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    
    def test_predict_proba(self, binary_data):
        """Test probability predictions."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        proba = clf.predict_proba(X_test)
        
        assert proba.shape == (len(y_test), 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
    
    def test_decision_function(self, binary_data):
        """Test decision function."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        scores = clf.decision_function(X_test)
        assert scores.shape == (len(y_test),)
    
    def test_feature_importances(self, binary_data):
        """Test feature importances."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        importances = clf.feature_importances_
        assert importances.shape == (10,)
        assert np.all(importances >= 0)
        np.testing.assert_almost_equal(importances.sum(), 1.0, decimal=5)
    
    def test_sample_weight(self, binary_data):
        """Test with sample weights."""
        X_train, X_test, y_train, y_test = binary_data
        
        # Create sample weights
        sample_weight = np.random.RandomState(42).rand(len(y_train)) + 0.5
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            random_state=42,
        )
        clf.fit(X_train, y_train, sample_weight=sample_weight)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_max_depth(self, binary_data):
        """Test max_depth parameter."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=20,
            max_depth=3,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_get_structure(self, binary_data):
        """Test tree structure visualization."""
        X_train, X_test, y_train, y_test = binary_data
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=5,
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        text = clf.summary()
        assert isinstance(text, str)
        assert "Alternating Decision Tree" in text

        # Test with feature names
        feature_names = [f"feat_{i}" for i in range(10)]
        text = clf.summary(feature_names=feature_names)
        assert "feat_" in text

        # Machine-readable dict API
        struct = clf.get_structure()
        assert isinstance(struct, dict)
        assert struct["structure_type"] == "tree"
        assert "feature_importances" in struct
    
    def test_categorical_features(self):
        """Test with categorical features."""
        # Create data with a categorical feature
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        X[:, 0] = rng.choice([0, 1, 2], size=200)  # Categorical
        y = (X[:, 0] == 1).astype(int) ^ (X[:, 1] > 0).astype(int)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            categorical_features=[0],
            random_state=42,
        )
        clf.fit(X_train, y_train)
        
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        assert accuracy > 0.5


class TestAlternatingModelTreeRegressor:
    """Tests for AlternatingModelTreeRegressor."""
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset."""
        X, y = make_regression(
            n_samples=200,
            n_features=10,
            n_informative=5,
            noise=0.5,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)
    
    def test_constant_model(self, regression_data):
        """Test with constant (mean) prediction at nodes."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            model_type='constant',
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        # Check attributes
        assert reg.n_features_in_ == 10
        assert reg.n_nodes_ > 0
        
        # Check predictions
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
        
        # Check R2 score is reasonable
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.0  # Better than mean prediction
    
    def test_linear_model(self, regression_data):
        """Test with linear models at nodes."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            model_type='linear',
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_ridge_model(self, regression_data):
        """Test with ridge regression at nodes."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            model_type='ridge',
            ridge_alpha=1.0,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_feature_importances(self, regression_data):
        """Test feature importances."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        importances = reg.feature_importances_
        assert importances.shape == (10,)
        assert np.all(importances >= 0)
    
    def test_sample_weight(self, regression_data):
        """Test with sample weights."""
        X_train, X_test, y_train, y_test = regression_data
        
        sample_weight = np.random.RandomState(42).rand(len(y_train)) + 0.5
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            random_state=42,
        )
        reg.fit(X_train, y_train, sample_weight=sample_weight)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_max_depth(self, regression_data):
        """Test max_depth parameter."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=20,
            max_depth=3,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_max_features_per_model(self, regression_data):
        """Test max_features_per_model parameter."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            model_type='ridge',
            max_features_per_model=3,
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        assert y_pred.shape == y_test.shape
    
    def test_get_structure(self, regression_data):
        """Test tree structure visualization."""
        X_train, X_test, y_train, y_test = regression_data
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=5,
            model_type='constant',
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        text = reg.summary()
        assert isinstance(text, str)
        assert "Alternating Model Tree" in text

        # Test with feature names
        feature_names = [f"feat_{i}" for i in range(10)]
        text = reg.summary(feature_names=feature_names)
        assert "feat_" in text

        # Machine-readable dict API
        struct = reg.get_structure()
        assert isinstance(struct, dict)
        assert struct["structure_type"] == "tree"
    
    def test_categorical_features(self):
        """Test with categorical features."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 5)
        X[:, 0] = rng.choice([0, 1, 2], size=200)  # Categorical
        y = X[:, 0] * 2 + X[:, 1] + rng.randn(200) * 0.1
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            categorical_features=[0],
            random_state=42,
        )
        reg.fit(X_train, y_train)
        
        y_pred = reg.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        assert r2 > 0.0


class TestSklearnCompatibility:
    """Test sklearn compatibility."""
    
    def test_classifier_clone(self):
        """Test that classifier can be cloned."""
        from sklearn.base import clone
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=10,
            max_depth=5,
            random_state=42,
        )
        clf_clone = clone(clf)
        
        assert clf_clone.n_iterations == clf.n_iterations
        assert clf_clone.max_depth == clf.max_depth
        assert clf_clone.random_state == clf.random_state
    
    def test_regressor_clone(self):
        """Test that regressor can be cloned."""
        from sklearn.base import clone
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=10,
            model_type='ridge',
            ridge_alpha=2.0,
            random_state=42,
        )
        reg_clone = clone(reg)
        
        assert reg_clone.n_iterations == reg.n_iterations
        assert reg_clone.model_type == reg.model_type
        assert reg_clone.ridge_alpha == reg.ridge_alpha
    
    def test_classifier_cross_val(self):
        """Test classifier with cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        X, y = make_classification(
            n_samples=100,
            n_features=10,
            n_classes=2,
            random_state=42,
        )
        
        clf = AlternatingDecisionTreeClassifier(
            n_iterations=5,
            random_state=42,
        )
        
        scores = cross_val_score(clf, X, y, cv=3)
        assert len(scores) == 3
        assert np.mean(scores) > 0.5
    
    def test_regressor_cross_val(self):
        """Test regressor with cross-validation."""
        from sklearn.model_selection import cross_val_score
        
        X, y = make_regression(
            n_samples=100,
            n_features=10,
            random_state=42,
        )
        
        reg = AlternatingModelTreeRegressor(
            n_iterations=5,
            random_state=42,
        )
        
        scores = cross_val_score(reg, X, y, cv=3, scoring='r2')
        assert len(scores) == 3
