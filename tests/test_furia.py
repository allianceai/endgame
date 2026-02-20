"""Tests for FURIA fuzzy rule-based classifier."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.exceptions import NotFittedError

from endgame.models.rules import FURIAClassifier, FuzzyRule, FuzzyCondition


class TestFuzzyCondition:
    """Tests for FuzzyCondition class."""

    def test_crisp_condition_lower_bound(self):
        """Test crisp condition with lower bound only."""
        cond = FuzzyCondition(feature_idx=0, feature_name="x0", lower_bound=5.0)
        X = np.array([[3.0], [5.0], [7.0]])
        membership = cond.membership(X)
        np.testing.assert_array_equal(membership, [0.0, 1.0, 1.0])

    def test_crisp_condition_upper_bound(self):
        """Test crisp condition with upper bound only."""
        cond = FuzzyCondition(feature_idx=0, feature_name="x0", upper_bound=5.0)
        X = np.array([[3.0], [5.0], [7.0]])
        membership = cond.membership(X)
        np.testing.assert_array_equal(membership, [1.0, 1.0, 0.0])

    def test_crisp_condition_both_bounds(self):
        """Test crisp condition with both bounds."""
        cond = FuzzyCondition(
            feature_idx=0, feature_name="x0", lower_bound=3.0, upper_bound=7.0
        )
        X = np.array([[2.0], [3.0], [5.0], [7.0], [8.0]])
        membership = cond.membership(X)
        np.testing.assert_array_equal(membership, [0.0, 1.0, 1.0, 1.0, 0.0])

    def test_fuzzy_condition_trapezoidal(self):
        """Test fuzzy trapezoidal membership function."""
        cond = FuzzyCondition(
            feature_idx=0,
            feature_name="x0",
            lower_bound=3.0,
            upper_bound=7.0,
            lower_support=1.0,
            upper_support=9.0,
        )
        X = np.array([[0.0], [1.0], [2.0], [3.0], [5.0], [7.0], [8.0], [9.0], [10.0]])
        membership = cond.membership(X)
        # Check key points
        assert membership[0] == 0.0  # Before lower support
        assert membership[1] == 0.0  # At lower support
        assert membership[3] == 1.0  # At lower bound (core)
        assert membership[4] == 1.0  # Inside core
        assert membership[5] == 1.0  # At upper bound (core)
        assert membership[7] == 0.0  # At upper support
        assert membership[8] == 0.0  # After upper support
        # Check linear interpolation in ramps
        assert 0 < membership[2] < 1  # In lower ramp

    def test_condition_str(self):
        """Test string representation of conditions."""
        cond1 = FuzzyCondition(feature_idx=0, feature_name="x0", lower_bound=5.0)
        assert "x0 >= 5" in str(cond1)

        cond2 = FuzzyCondition(feature_idx=0, feature_name="x0", upper_bound=5.0)
        assert "x0 <= 5" in str(cond2)

        cond3 = FuzzyCondition(
            feature_idx=0, feature_name="x0", lower_bound=3.0, upper_bound=7.0
        )
        assert "3" in str(cond3) and "7" in str(cond3)


class TestFuzzyRule:
    """Tests for FuzzyRule class."""

    def test_empty_rule(self):
        """Test rule with no conditions (covers all)."""
        rule = FuzzyRule(consequent=0)
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        firing = rule.firing_strength(X)
        np.testing.assert_array_equal(firing, [1.0, 1.0])

    def test_single_condition(self):
        """Test rule with single condition."""
        cond = FuzzyCondition(feature_idx=0, feature_name="x0", lower_bound=2.0)
        rule = FuzzyRule(consequent=1, conditions=[cond])
        X = np.array([[1.0], [2.0], [3.0]])
        firing = rule.firing_strength(X)
        np.testing.assert_array_equal(firing, [0.0, 1.0, 1.0])

    def test_multiple_conditions_and(self):
        """Test rule with multiple conditions (fuzzy AND = min)."""
        cond1 = FuzzyCondition(feature_idx=0, feature_name="x0", lower_bound=2.0)
        cond2 = FuzzyCondition(feature_idx=1, feature_name="x1", upper_bound=5.0)
        rule = FuzzyRule(consequent=0, conditions=[cond1, cond2])
        X = np.array([[1.0, 3.0], [2.0, 3.0], [3.0, 6.0], [3.0, 4.0]])
        firing = rule.firing_strength(X)
        np.testing.assert_array_equal(firing, [0.0, 1.0, 0.0, 1.0])

    def test_rule_covers(self):
        """Test rule coverage check."""
        cond = FuzzyCondition(feature_idx=0, feature_name="x0", lower_bound=2.0)
        rule = FuzzyRule(consequent=1, conditions=[cond])
        X = np.array([[1.0], [2.0], [3.0]])
        covers = rule.covers(X)
        np.testing.assert_array_equal(covers, [False, True, True])


class TestFURIAClassifier:
    """Tests for FURIAClassifier."""

    @pytest.fixture
    def binary_data(self):
        """Generate binary classification data."""
        X, y = make_classification(
            n_samples=200,
            n_features=5,
            n_informative=3,
            n_redundant=1,
            n_classes=2,
            random_state=42,
        )
        return train_test_split(X, y, test_size=0.3, random_state=42)

    @pytest.fixture
    def multiclass_data(self):
        """Load iris dataset for multiclass testing."""
        iris = load_iris()
        return train_test_split(
            iris.data, iris.target, test_size=0.3, random_state=42
        )

    def test_fit_predict_binary(self, binary_data):
        """Test basic fit and predict on binary data."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(max_rules=20, random_state=42)
        clf.fit(X_train, y_train)

        # Check fitted attributes
        assert hasattr(clf, "rules_")
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_features_in_")
        assert len(clf.classes_) == 2

        # Predict
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred).issubset(set(y_train))

    def test_fit_predict_multiclass(self, multiclass_data):
        """Test fit and predict on multiclass data."""
        X_train, X_test, y_train, y_test = multiclass_data
        clf = FURIAClassifier(max_rules=30, random_state=42)
        clf.fit(X_train, y_train)

        assert len(clf.classes_) == 3
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape
        assert set(y_pred).issubset(set(y_train))

    def test_predict_proba_binary(self, binary_data):
        """Test probability predictions for binary classification."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 2)
        # Probabilities should sum to 1
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)
        # Probabilities should be in [0, 1]
        assert (proba >= 0).all() and (proba <= 1).all()

    def test_predict_proba_multiclass(self, multiclass_data):
        """Test probability predictions for multiclass."""
        X_train, X_test, y_train, y_test = multiclass_data
        clf = FURIAClassifier(random_state=42)
        clf.fit(X_train, y_train)

        proba = clf.predict_proba(X_test)
        assert proba.shape == (len(X_test), 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0)

    def test_not_fitted_error(self):
        """Test that predict raises error when not fitted."""
        clf = FURIAClassifier()
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(NotFittedError):
            clf.predict(X)

    def test_feature_names(self, binary_data):
        """Test that feature names are auto-generated."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(random_state=42)
        clf.fit(X_train, y_train)

        # Check that rules have feature names (auto-generated as X0, X1, ...)
        for rule in clf.rules_:
            for cond in rule.conditions:
                assert cond.feature_name.startswith("X")

    def test_fuzzification_disabled(self, binary_data):
        """Test classifier with fuzzification disabled."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(fuzzify=False, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

        # Check that rules have no fuzzy support bounds
        for rule in clf.rules_:
            for cond in rule.conditions:
                assert cond.lower_support is None
                assert cond.upper_support is None

    def test_max_rules_parameter(self, binary_data):
        """Test max_rules parameter limits rule count."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(max_rules=5, random_state=42)
        clf.fit(X_train, y_train)
        assert len(clf.rules_) <= 5

    def test_min_support_parameter(self, binary_data):
        """Test min_support parameter."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(min_support=10, random_state=42)
        clf.fit(X_train, y_train)
        # Classifier should still work
        y_pred = clf.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_max_conditions_parameter(self, binary_data):
        """Test max_conditions parameter limits condition count per rule."""
        X_train, X_test, y_train, y_test = binary_data
        max_cond = 3
        clf = FURIAClassifier(max_conditions=max_cond, random_state=42)
        clf.fit(X_train, y_train)
        for rule in clf.rules_:
            assert len(rule.conditions) <= max_cond

    def test_sklearn_compatible(self, binary_data):
        """Test sklearn compatibility via cross_val_score."""
        X_train, X_test, y_train, y_test = binary_data
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])

        clf = FURIAClassifier(max_rules=10, random_state=42)
        scores = cross_val_score(clf, X, y, cv=3, scoring="accuracy")
        assert len(scores) == 3
        assert all(0 <= s <= 1 for s in scores)

    def test_clone_and_set_params(self, binary_data):
        """Test sklearn clone and set_params."""
        from sklearn.base import clone

        clf = FURIAClassifier(max_rules=10, random_state=42)
        clf_clone = clone(clf)
        assert clf_clone.max_rules == 10
        assert clf_clone.random_state == 42

        clf.set_params(max_rules=20)
        assert clf.max_rules == 20

    def test_get_rules_method(self, binary_data):
        """Test get_rules method for rule inspection."""
        X_train, X_test, y_train, y_test = binary_data
        clf = FURIAClassifier(random_state=42)
        clf.fit(X_train, y_train)

        rules = clf.get_rules()
        assert isinstance(rules, list)
        for rule in rules:
            assert isinstance(rule, FuzzyRule)
            assert hasattr(rule, "consequent")
            assert hasattr(rule, "conditions")
            assert hasattr(rule, "support")
            assert hasattr(rule, "weight")

    def test_simple_dataset(self):
        """Test on simple dataset where rules are obvious."""
        # Create dataset where class depends clearly on feature boundaries
        np.random.seed(42)
        X_class0 = np.random.randn(50, 2) + np.array([0, 0])
        X_class1 = np.random.randn(50, 2) + np.array([5, 5])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * 50 + [1] * 50)

        clf = FURIAClassifier(random_state=42)
        clf.fit(X, y)

        # Should achieve high accuracy on this separable data
        y_pred = clf.predict(X)
        accuracy = (y_pred == y).mean()
        assert accuracy > 0.8

    def test_handles_single_class_remaining(self):
        """Test that classifier handles edge cases gracefully."""
        X = np.array([[1, 2], [2, 3], [3, 4], [10, 20], [11, 21], [12, 22]])
        y = np.array([0, 0, 0, 1, 1, 1])

        clf = FURIAClassifier(max_rules=10, random_state=42)
        clf.fit(X, y)

        # Should predict both classes
        y_pred = clf.predict(X)
        assert set(y_pred) == {0, 1}

    def test_reproducibility(self, binary_data):
        """Test that random_state ensures reproducibility."""
        X_train, X_test, y_train, y_test = binary_data

        clf1 = FURIAClassifier(random_state=42)
        clf1.fit(X_train, y_train)
        pred1 = clf1.predict(X_test)

        clf2 = FURIAClassifier(random_state=42)
        clf2.fit(X_train, y_train)
        pred2 = clf2.predict(X_test)

        np.testing.assert_array_equal(pred1, pred2)


class TestFURIAIntegration:
    """Integration tests for FURIA with endgame ecosystem."""

    def test_import_from_endgame(self):
        """Test that FURIA can be imported from endgame."""
        from endgame.models.rules import FURIAClassifier

        assert FURIAClassifier is not None

    def test_in_sklearn_pipeline(self):
        """Test FURIA in sklearn pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", FURIAClassifier(random_state=42))
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        assert y_pred.shape == y_test.shape

    def test_with_grid_search(self):
        """Test FURIA with GridSearchCV."""
        from sklearn.model_selection import GridSearchCV

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        param_grid = {
            "max_rules": [5, 10],
            "min_support": [2, 5],
        }

        clf = FURIAClassifier(random_state=42)
        grid_search = GridSearchCV(clf, param_grid, cv=3, scoring="accuracy")
        grid_search.fit(X, y)

        assert hasattr(grid_search, "best_params_")
        assert hasattr(grid_search, "best_score_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
