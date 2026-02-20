"""Comprehensive tests for RuleFit implementation."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score

from endgame.models.rules import (
    Condition,
    Operator,
    Rule,
    RuleEnsemble,
    RuleFitClassifier,
    RuleFitRegressor,
    extract_rules_from_ensemble,
    extract_rules_from_tree,
)


class TestCondition:
    """Tests for the Condition class."""

    def test_le_operator(self):
        """Test less-than-or-equal operator."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.LE, threshold=50
        )
        X = np.array([[30], [50], [60]])
        result = cond.evaluate(X)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_gt_operator(self):
        """Test greater-than operator."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        X = np.array([[30], [50], [60]])
        result = cond.evaluate(X)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_eq_operator(self):
        """Test equality operator."""
        cond = Condition(
            feature_idx=0, feature_name="color", operator=Operator.EQ, threshold=1
        )
        X = np.array([[1], [2], [1]])
        result = cond.evaluate(X)
        np.testing.assert_array_equal(result, [True, False, True])

    def test_in_operator(self):
        """Test 'in set' operator."""
        cond = Condition(
            feature_idx=0, feature_name="color", operator=Operator.IN, threshold={1, 2}
        )
        X = np.array([[1], [2], [3]])
        result = cond.evaluate(X)
        np.testing.assert_array_equal(result, [True, True, False])

    def test_str_representation(self):
        """Test string representation."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.LE, threshold=50.5
        )
        assert "age" in str(cond)
        assert "<=" in str(cond)
        assert "50.5" in str(cond)

    def test_hash_and_equality(self):
        """Test hashing and equality."""
        cond1 = Condition(
            feature_idx=0, feature_name="age", operator=Operator.LE, threshold=50
        )
        cond2 = Condition(
            feature_idx=0, feature_name="age", operator=Operator.LE, threshold=50
        )
        cond3 = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )

        assert cond1 == cond2
        assert cond1 != cond3
        assert hash(cond1) == hash(cond2)


class TestRule:
    """Tests for the Rule class."""

    def test_empty_rule(self):
        """Test empty rule (always true)."""
        rule = Rule()
        X = np.array([[1, 2], [3, 4]])
        result = rule.evaluate(X)
        np.testing.assert_array_equal(result, [True, True])

    def test_single_condition_rule(self):
        """Test rule with single condition."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        rule = Rule(conditions=[cond])
        X = np.array([[30, 100], [60, 50]])
        result = rule.evaluate(X)
        np.testing.assert_array_equal(result, [False, True])

    def test_multi_condition_rule(self):
        """Test rule with multiple conditions (AND logic)."""
        cond1 = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        cond2 = Condition(
            feature_idx=1, feature_name="income", operator=Operator.GT, threshold=80000
        )
        rule = Rule(conditions=[cond1, cond2])

        X = np.array(
            [
                [30, 100000],  # age <= 50
                [60, 50000],  # income <= 80k
                [60, 100000],  # both satisfied
            ]
        )
        result = rule.evaluate(X)
        np.testing.assert_array_equal(result, [False, False, True])

    def test_rule_properties(self):
        """Test rule properties."""
        cond1 = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        cond2 = Condition(
            feature_idx=1, feature_name="income", operator=Operator.GT, threshold=80000
        )
        rule = Rule(conditions=[cond1, cond2], support=0.3, coefficient=2.5)

        assert rule.length == 2
        assert set(rule.feature_indices) == {0, 1}
        assert rule.importance == 0.75  # |2.5| * 0.3

    def test_rule_str(self):
        """Test rule string representation."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        rule = Rule(conditions=[cond])
        assert "age > 50" in str(rule)

    def test_rule_to_dict(self):
        """Test rule to_dict method."""
        cond = Condition(
            feature_idx=0, feature_name="age", operator=Operator.GT, threshold=50
        )
        rule = Rule(conditions=[cond], support=0.5, coefficient=1.0)
        d = rule.to_dict()

        assert "rule" in d
        assert d["support"] == 0.5
        assert d["coefficient"] == 1.0
        assert "importance" in d


class TestRuleEnsemble:
    """Tests for the RuleEnsemble class."""

    def test_transform(self):
        """Test rule ensemble transform."""
        cond1 = Condition(
            feature_idx=0, feature_name="x", operator=Operator.GT, threshold=0.5
        )
        cond2 = Condition(
            feature_idx=0, feature_name="x", operator=Operator.LE, threshold=0.5
        )
        rule1 = Rule(conditions=[cond1])
        rule2 = Rule(conditions=[cond2])

        ensemble = RuleEnsemble(
            rules=[rule1, rule2], n_features=1, feature_names=["x"]
        )

        X = np.array([[0.3], [0.7]])
        X_rules = ensemble.transform(X)

        assert X_rules.shape == (2, 2)
        np.testing.assert_array_equal(X_rules[0], [0, 1])  # 0.3 <= 0.5
        np.testing.assert_array_equal(X_rules[1], [1, 0])  # 0.7 > 0.5

    def test_filter_by_support(self):
        """Test support filtering."""
        rule1 = Rule(conditions=[], support=0.05)
        rule2 = Rule(conditions=[], support=0.5)
        rule3 = Rule(conditions=[], support=0.95)

        ensemble = RuleEnsemble(rules=[rule1, rule2, rule3])
        filtered = ensemble.filter_by_support(min_support=0.1, max_support=0.9)

        assert len(filtered) == 1
        assert filtered.rules[0].support == 0.5

    def test_deduplicate(self):
        """Test deduplication."""
        cond = Condition(
            feature_idx=0, feature_name="x", operator=Operator.GT, threshold=0.5
        )
        rule1 = Rule(conditions=[cond])
        rule2 = Rule(conditions=[cond])  # Same rule

        ensemble = RuleEnsemble(rules=[rule1, rule2])
        deduped = ensemble.deduplicate()

        assert len(deduped) == 1

    def test_limit_rules(self):
        """Test rule limiting."""
        rules = [Rule(conditions=[], support=0.1 * (i + 1)) for i in range(10)]
        ensemble = RuleEnsemble(rules=rules)
        limited = ensemble.limit_rules(max_rules=5)

        assert len(limited) == 5
        # Should keep highest support rules
        assert limited.rules[0].support == 1.0


class TestRuleExtraction:
    """Tests for rule extraction functions."""

    def test_extract_from_single_tree(self):
        """Test rule extraction from a single decision tree."""
        from sklearn.tree import DecisionTreeRegressor

        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1]

        tree = DecisionTreeRegressor(max_depth=2, random_state=42)
        tree.fit(X, y)

        feature_names = ["a", "b", "c"]
        rules = extract_rules_from_tree(tree, feature_names, X_train=X)

        assert len(rules) > 0
        for rule in rules:
            assert isinstance(rule, Rule)
            assert 0 <= rule.support <= 1
            for cond in rule.conditions:
                assert cond.feature_name in feature_names

    def test_extract_from_random_forest(self):
        """Test rule extraction from random forest."""
        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1]

        rf = RandomForestRegressor(n_estimators=5, max_depth=2, random_state=42)
        rf.fit(X, y)

        feature_names = ["a", "b", "c"]
        ensemble = extract_rules_from_ensemble(rf, feature_names, X_train=X)

        assert len(ensemble) > 0
        assert ensemble.n_features == 3

    def test_extract_from_gradient_boosting(self):
        """Test rule extraction from gradient boosting."""
        from sklearn.ensemble import GradientBoostingRegressor

        X = np.random.randn(100, 3)
        y = X[:, 0] + 2 * X[:, 1]

        gb = GradientBoostingRegressor(n_estimators=5, max_depth=2, random_state=42)
        gb.fit(X, y)

        feature_names = ["a", "b", "c"]
        ensemble = extract_rules_from_ensemble(gb, feature_names, X_train=X)

        assert len(ensemble) > 0


class TestRuleFitRegressor:
    """Tests for RuleFitRegressor."""

    def test_basic_fit_predict(self):
        """Basic regression test."""
        X, y = make_regression(
            n_samples=300, n_features=10, n_informative=5, noise=0.1, random_state=42
        )

        model = RuleFitRegressor(n_estimators=30, tree_max_depth=3, random_state=42)
        model.fit(X, y)

        # Should have decent R²
        score = model.score(X, y)
        assert score > 0.7

        # Should have extracted rules
        assert model.n_rules_ > 0
        assert model.n_rules_selected_ > 0

    def test_predict_shape(self):
        """Test prediction shape."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        preds = model.predict(X)
        assert preds.shape == (100,)

    def test_transform(self):
        """Test transform method."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        X_rules = model.transform(X)
        assert X_rules.shape[0] == 100
        assert X_rules.shape[1] == model.n_rules_

        # Should be binary
        assert np.all((X_rules == 0) | (X_rules == 1))

    def test_get_rules(self):
        """Test rule extraction and formatting."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y, feature_names=["a", "b", "c", "d", "e"])

        rules = model.get_rules(exclude_zero_coef=True)

        assert len(rules) > 0

        for rule_dict in rules:
            assert "rule" in rule_dict
            assert "coefficient" in rule_dict
            assert "support" in rule_dict
            assert "importance" in rule_dict

            # Rule should be a readable string
            assert isinstance(rule_dict["rule"], str)

            # Support should be in [0, 1]
            assert 0 <= rule_dict["support"] <= 1

    def test_get_rules_sorting(self):
        """Test rule sorting options."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        rules_imp = model.get_rules(sort_by="importance")
        rules_sup = model.get_rules(sort_by="support")
        rules_len = model.get_rules(sort_by="length")

        # Check sorting is correct
        if len(rules_imp) > 1:
            assert rules_imp[0]["importance"] >= rules_imp[-1]["importance"]
        if len(rules_sup) > 1:
            assert rules_sup[0]["support"] >= rules_sup[-1]["support"]
        if len(rules_len) > 1:
            assert rules_len[0]["length"] <= rules_len[-1]["length"]

    def test_summary(self):
        """Test summary output."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y)

        summary = model.summary()

        assert isinstance(summary, str)
        assert "RuleFit" in summary
        assert "Rules" in summary or "rules" in summary.lower()

    def test_feature_names(self):
        """Feature names should appear in rules."""
        X, y = make_regression(n_samples=200, n_features=3, random_state=42)

        model = RuleFitRegressor(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y, feature_names=["age", "income", "score"])

        rules = model.get_rules()

        # At least some rules should mention feature names
        all_rule_text = " ".join(r["rule"] for r in rules)
        has_feature_name = (
            "age" in all_rule_text
            or "income" in all_rule_text
            or "score" in all_rule_text
        )
        assert has_feature_name

    def test_support_filtering(self):
        """Test that support filtering works."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)

        model = RuleFitRegressor(
            n_estimators=50,
            tree_max_depth=3,
            min_support=0.1,
            max_support=0.9,
            random_state=42,
        )
        model.fit(X, y)

        # All rules should have support in [0.1, 0.9]
        for rule in model.rule_ensemble_.rules:
            assert 0.1 <= rule.support <= 0.9

    def test_max_rules(self):
        """Test max_rules parameter."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)

        model = RuleFitRegressor(
            n_estimators=100, tree_max_depth=4, max_rules=50, random_state=42
        )
        model.fit(X, y)

        assert model.n_rules_ <= 50

    def test_no_linear_features(self):
        """Test with include_linear=False."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)

        model = RuleFitRegressor(
            n_estimators=50, tree_max_depth=3, include_linear=False, random_state=42
        )
        model.fit(X, y)

        # Linear coefficients should be all zeros
        assert np.allclose(model.linear_coef_, 0)

        # Should still make reasonable predictions
        score = model.score(X, y)
        assert score > 0.3

    def test_custom_tree_generator(self):
        """Test with custom tree generator."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)

        rf = RandomForestRegressor(n_estimators=20, max_depth=3, random_state=42)

        model = RuleFitRegressor(tree_generator=rf, random_state=42)
        model.fit(X, y)

        assert model.n_rules_ > 0
        assert model.score(X, y) > 0.5

    def test_feature_importances(self):
        """Feature importances should be computed."""
        X, y = make_regression(n_samples=300, n_features=10, random_state=42)

        model = RuleFitRegressor(n_estimators=50, tree_max_depth=3, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "feature_importances_")
        assert model.feature_importances_.shape == (10,)
        assert np.isclose(model.feature_importances_.sum(), 1.0)
        assert np.all(model.feature_importances_ >= 0)

    def test_reproducibility(self):
        """Same random_state should give same results."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        model1 = RuleFitRegressor(n_estimators=20, random_state=42)
        model1.fit(X, y)

        model2 = RuleFitRegressor(n_estimators=20, random_state=42)
        model2.fit(X, y)

        np.testing.assert_allclose(model1.predict(X), model2.predict(X), rtol=1e-5)

    def test_cv_alpha_selection(self):
        """Test cross-validation alpha selection."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)

        model = RuleFitRegressor(
            n_estimators=30, tree_max_depth=3, alpha=None, cv=3, random_state=42
        )
        model.fit(X, y)

        assert model.alpha_ > 0
        assert "alphas" in model.cv_results_

    def test_fixed_alpha(self):
        """Test with fixed alpha."""
        X, y = make_regression(n_samples=300, n_features=5, random_state=42)

        model = RuleFitRegressor(
            n_estimators=50, tree_max_depth=3, alpha=0.1, random_state=42
        )
        model.fit(X, y)

        assert model.alpha_ == 0.1

    def test_sparse_output_high_alpha(self):
        """High alpha should produce sparse (few rules) output."""
        X, y = make_regression(n_samples=500, n_features=10, random_state=42)

        model_sparse = RuleFitRegressor(
            n_estimators=50, tree_max_depth=3, alpha=1.0, random_state=42
        )
        model_sparse.fit(X, y)

        model_dense = RuleFitRegressor(
            n_estimators=50, tree_max_depth=3, alpha=0.001, random_state=42
        )
        model_dense.fit(X, y)

        # Sparse model should have fewer selected rules
        assert model_sparse.n_rules_selected_ < model_dense.n_rules_selected_

    def test_get_equation(self):
        """Test equation output."""
        X, y = make_regression(n_samples=200, n_features=3, random_state=42)

        model = RuleFitRegressor(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y, feature_names=["a", "b", "c"])

        equation = model.get_equation()

        assert isinstance(equation, str)
        assert "y =" in equation


class TestRuleFitClassifier:
    """Tests for RuleFitClassifier."""

    def test_basic_binary_classification(self):
        """Basic binary classification test."""
        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=5, random_state=42
        )

        model = RuleFitClassifier(n_estimators=30, tree_max_depth=3, random_state=42)
        model.fit(X, y)

        # Should have decent accuracy
        score = model.score(X, y)
        assert score > 0.7

        # Probabilities should sum to 1
        proba = model.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)

    def test_multiclass_classification(self):
        """Test multiclass classification."""
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=5,
            n_classes=4,
            n_clusters_per_class=1,
            random_state=42,
        )

        model = RuleFitClassifier(n_estimators=50, tree_max_depth=3, random_state=42)
        model.fit(X, y)

        assert model.n_classes_ == 4
        assert model.predict(X).shape == (500,)
        assert model.predict_proba(X).shape == (500, 4)

    def test_predict_shape(self):
        """Test prediction shapes."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        assert model.predict(X).shape == (100,)
        assert model.predict_proba(X).shape == (100, 2)
        assert model.predict_log_proba(X).shape == (100, 2)

    def test_classes_attribute(self):
        """Test classes_ attribute."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        assert hasattr(model, "classes_")
        assert len(model.classes_) == 2

    def test_get_rules(self):
        """Test rule extraction for classifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y)

        rules = model.get_rules()
        assert len(rules) > 0

    def test_summary(self):
        """Test summary for classifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        summary = model.summary()
        assert isinstance(summary, str)
        assert "Classifier" in summary or "classes" in summary.lower()

    def test_custom_tree_generator(self):
        """Test with custom tree generator."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        rf = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)

        model = RuleFitClassifier(tree_generator=rf, random_state=42)
        model.fit(X, y)

        assert model.n_rules_ > 0
        assert model.score(X, y) > 0.6

    def test_class_weight(self):
        """Test class_weight parameter."""
        X, y = make_classification(
            n_samples=200, n_features=5, weights=[0.9, 0.1], random_state=42
        )

        model = RuleFitClassifier(
            n_estimators=30, tree_max_depth=3, class_weight="balanced", random_state=42
        )
        model.fit(X, y)

        # Should fit without error
        assert model.score(X, y) > 0.5

    def test_transform(self):
        """Test transform method for classifier."""
        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, random_state=42)
        model.fit(X, y)

        X_rules = model.transform(X)
        assert X_rules.shape[0] == 100
        assert X_rules.shape[1] == model.n_rules_

    def test_get_equation_binary(self):
        """Test equation output for binary classifier."""
        X, y = make_classification(
            n_samples=200, n_features=5, n_informative=3, random_state=42
        )

        model = RuleFitClassifier(n_estimators=20, tree_max_depth=2, random_state=42)
        model.fit(X, y)

        equation = model.get_equation()
        assert isinstance(equation, str)
        assert "logit" in equation or "P(y=1)" in equation


class TestInterpretability:
    """Tests focused on interpretability aspects."""

    def test_interpretable_rules_regression(self):
        """
        Test on a problem where rules should be interpretable.
        """
        np.random.seed(42)
        n_samples = 1000

        age = np.random.uniform(20, 70, n_samples)
        income = np.random.uniform(20000, 150000, n_samples)

        # Target with clear rules:
        # - Base: 0.01 * income
        # - Bonus if age > 50 AND income > 80000
        # - Penalty if age < 30 AND income < 40000

        y = 0.01 * income
        y += 10000 * ((age > 50) & (income > 80000))
        y -= 5000 * ((age < 30) & (income < 40000))
        y += np.random.normal(0, 1000, n_samples)

        X = np.column_stack([age, income])

        model = RuleFitRegressor(
            n_estimators=100, tree_max_depth=2, alpha=0.01, random_state=42
        )
        model.fit(X, y, feature_names=["age", "income"])

        # Should have good fit
        assert model.score(X, y) > 0.85

        # Check that meaningful rules were found
        rules = model.get_rules(sort_by="importance")
        rule_texts = [r["rule"] for r in rules[:15]]

        # At least some rules should involve age/income thresholds
        found_relevant = any(
            ("age" in r and ">" in r) or ("income" in r and ">" in r)
            for r in rule_texts
        )
        assert found_relevant, f"Expected rules about age/income, got: {rule_texts}"

    def test_rule_coefficient_signs(self):
        """Test that rule coefficients have expected signs."""
        np.random.seed(42)
        n_samples = 500

        x = np.random.randn(n_samples, 1)
        y = (x[:, 0] > 0).astype(float) * 5 + np.random.randn(n_samples) * 0.1

        model = RuleFitRegressor(
            n_estimators=50, tree_max_depth=2, alpha=0.01, random_state=42
        )
        model.fit(x, y, feature_names=["x"])

        # Rules involving "x > 0" should have positive coefficients
        rules = model.get_rules()
        for r in rules:
            if "x > 0" in r["rule"] and r["coefficient"] != 0:
                assert r["coefficient"] > 0, f"Rule '{r['rule']}' should be positive"


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature(self):
        """Test with single feature."""
        X = np.random.randn(100, 1)
        y = X[:, 0] ** 2

        model = RuleFitRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        assert model.predict(X).shape == (100,)

    def test_many_features(self):
        """Test with many features."""
        X = np.random.randn(200, 50)
        y = X[:, 0] + X[:, 1]

        model = RuleFitRegressor(n_estimators=30, max_rules=100, random_state=42)
        model.fit(X, y)

        assert model.predict(X).shape == (200,)

    def test_few_samples(self):
        """Test with few samples."""
        X = np.random.randn(50, 5)
        y = X[:, 0]

        model = RuleFitRegressor(n_estimators=10, cv=3, random_state=42)
        model.fit(X, y)

        assert model.predict(X).shape == (50,)

    def test_constant_feature(self):
        """Test with constant feature."""
        X = np.random.randn(100, 5)
        X[:, 2] = 1.0  # Constant feature
        y = X[:, 0]

        model = RuleFitRegressor(n_estimators=20, random_state=42)
        model.fit(X, y)

        # Should still work
        assert model.predict(X).shape == (100,)

    def test_not_fitted_error(self):
        """Test error when predicting without fitting."""
        model = RuleFitRegressor()
        X = np.random.randn(10, 5)

        with pytest.raises(Exception):  # NotFittedError
            model.predict(X)

    def test_wrong_n_features_error(self):
        """Test error when predicting with wrong number of features."""
        X_train = np.random.randn(100, 5)
        y = X_train[:, 0]

        model = RuleFitRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y)

        X_test = np.random.randn(10, 3)  # Wrong number of features

        with pytest.raises(ValueError):
            model.predict(X_test)


class TestSklearnCompatibility:
    """Tests for sklearn compatibility."""

    def test_cross_val_score_regressor(self):
        """Test cross_val_score with regressor."""
        X, y = make_regression(n_samples=200, n_features=5, random_state=42)

        model = RuleFitRegressor(n_estimators=20, random_state=42, alpha=0.1)
        scores = cross_val_score(model, X, y, cv=3)

        assert len(scores) == 3
        assert all(s > -1 for s in scores)  # Not terrible scores

    def test_cross_val_score_classifier(self):
        """Test cross_val_score with classifier."""
        X, y = make_classification(n_samples=200, n_features=5, random_state=42)

        model = RuleFitClassifier(n_estimators=20, random_state=42, alpha=0.1)
        scores = cross_val_score(model, X, y, cv=3)

        assert len(scores) == 3
        assert all(s > 0.3 for s in scores)

    def test_get_params_set_params(self):
        """Test get_params and set_params."""
        model = RuleFitRegressor(n_estimators=50, tree_max_depth=4)

        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["tree_max_depth"] == 4

        model.set_params(n_estimators=100)
        assert model.n_estimators == 100

    def test_clone(self):
        """Test that model can be cloned."""
        from sklearn.base import clone

        model = RuleFitRegressor(n_estimators=50, random_state=42)
        cloned = clone(model)

        assert cloned.n_estimators == 50
        assert cloned.random_state == 42
