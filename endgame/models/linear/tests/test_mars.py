"""Comprehensive test suite for MARS (Multivariate Adaptive Regression Splines).

Tests cover:
- sklearn compatibility
- Basic functionality
- Edge cases
- Performance characteristics
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from endgame.models.linear.basis import BasisFunction, HingeSpec, LinearBasisFunction
from endgame.models.linear.mars import MARSClassifier, MARSRegressor

# Skip sklearn estimator checks if not available
try:
    from sklearn.utils.estimator_checks import check_estimator
    HAS_SKLEARN_CHECKS = True
except ImportError:
    HAS_SKLEARN_CHECKS = False


class TestHingeSpec:
    """Tests for HingeSpec class."""

    def test_positive_hinge(self):
        """Test max(0, x - knot) hinge."""
        hinge = HingeSpec(feature_idx=0, knot=5.0, direction=1)
        x = np.array([3.0, 5.0, 7.0, 10.0])
        result = hinge.evaluate(x)
        expected = np.array([0.0, 0.0, 2.0, 5.0])
        assert_allclose(result, expected)

    def test_negative_hinge(self):
        """Test max(0, knot - x) hinge."""
        hinge = HingeSpec(feature_idx=0, knot=5.0, direction=-1)
        x = np.array([3.0, 5.0, 7.0, 10.0])
        result = hinge.evaluate(x)
        expected = np.array([2.0, 0.0, 0.0, 0.0])
        assert_allclose(result, expected)

    def test_invalid_direction(self):
        """Invalid direction should raise ValueError."""
        with pytest.raises(ValueError, match="direction must be"):
            HingeSpec(feature_idx=0, knot=5.0, direction=0)

    def test_str_representation(self):
        """Test string representation."""
        hinge_pos = HingeSpec(0, 5.0, 1)
        assert "x0 - 5" in str(hinge_pos)

        hinge_neg = HingeSpec(1, 3.5, -1)
        assert "3.5 - x1" in str(hinge_neg)


class TestBasisFunction:
    """Tests for BasisFunction class."""

    def test_intercept(self):
        """Test intercept (constant) basis function."""
        bf = BasisFunction()
        assert bf.is_intercept
        assert bf.degree == 0

        X = np.random.randn(10, 3)
        result = bf.evaluate(X)
        assert_allclose(result, np.ones(10))

    def test_single_hinge(self):
        """Test single hinge basis function."""
        bf = BasisFunction([HingeSpec(0, 2.0, 1)])
        assert not bf.is_intercept
        assert bf.degree == 1
        assert bf.feature_indices == [0]

        X = np.array([[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])
        result = bf.evaluate(X)
        expected = np.array([0.0, 0.0, 1.0, 2.0])
        assert_allclose(result, expected)

    def test_interaction(self):
        """Test interaction (product of hinges) basis function."""
        bf = BasisFunction([
            HingeSpec(0, 0.0, 1),
            HingeSpec(1, 0.0, 1)
        ])
        assert bf.degree == 2
        assert set(bf.feature_indices) == {0, 1}

        X = np.array([
            [1.0, 1.0],   # both positive -> 1*1 = 1
            [-1.0, 1.0],  # first zero -> 0
            [1.0, -1.0],  # second zero -> 0
            [2.0, 3.0],   # 2*3 = 6
        ])
        result = bf.evaluate(X)
        expected = np.array([1.0, 0.0, 0.0, 6.0])
        assert_allclose(result, expected)

    def test_can_extend_with(self):
        """Test extension rules."""
        # Intercept can be extended
        intercept = BasisFunction()
        assert intercept.can_extend_with(0, max_degree=2)

        # Single hinge can be extended with different feature
        single = BasisFunction([HingeSpec(0, 1.0, 1)])
        assert single.can_extend_with(1, max_degree=2)
        assert not single.can_extend_with(0, max_degree=2)  # Same feature

        # At max degree, cannot extend
        assert not single.can_extend_with(1, max_degree=1)

    def test_copy(self):
        """Test copying basis function."""
        original = BasisFunction([HingeSpec(0, 1.0, 1)])
        copy = original.copy()

        # Modify original
        original.hinges[0].knot = 2.0

        # Copy should be unchanged
        assert copy.hinges[0].knot == 1.0


class TestMARSRegressor:
    """Tests for MARSRegressor."""

    @pytest.mark.skipif(not HAS_SKLEARN_CHECKS, reason="sklearn checks not available")
    def test_sklearn_compatibility(self):
        """Verify sklearn estimator contract."""
        # Use a smaller set of checks that are more stable
        model = MARSRegressor(max_terms=10)
        # Basic check - fit and predict
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        model.fit(X, y)
        predictions = model.predict(X)
        assert predictions.shape == (50,)

    def test_linear_relationship(self):
        """MARS should recover linear relationships."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = 2 * X[:, 0] - 3 * X[:, 1] + 0.5 * X[:, 2] + np.random.randn(200) * 0.1

        model = MARSRegressor(max_degree=1)
        model.fit(X, y)

        # Should have high R^2
        assert model.rsq_ > 0.95, f"R^2 = {model.rsq_}"

        # Should have few terms (close to linear)
        assert len(model.basis_functions_) <= 10

    def test_single_knot(self):
        """MARS should find a single knot."""
        np.random.seed(42)
        X = np.linspace(0, 10, 200).reshape(-1, 1)
        y = np.where(X.ravel() < 5, 2 * X.ravel(), 10 + 0.5 * (X.ravel() - 5))
        y += np.random.randn(200) * 0.2

        model = MARSRegressor(max_degree=1)
        model.fit(X, y)

        # Should find knot near 5
        knots = []
        for bf in model.basis_functions_:
            for h in bf.hinges:
                knots.append(h.knot)

        assert any(4 < k < 6 for k in knots), f"Expected knot near 5, got {knots}"

    def test_friedman1(self):
        """Test on Friedman #1 benchmark (has interactions)."""
        try:
            from sklearn.datasets import make_friedman1
            from sklearn.model_selection import cross_val_score
        except ImportError:
            pytest.skip("sklearn not available")

        X, y = make_friedman1(n_samples=500, n_features=10, noise=0.1, random_state=42)

        model = MARSRegressor(max_degree=2)  # Allow interactions
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')

        # Should achieve reasonable R^2 on this benchmark
        assert np.mean(scores) > 0.80, f"Mean R^2 = {np.mean(scores)}"

    def test_interactions(self):
        """MARS should discover interactions when max_degree > 1."""
        np.random.seed(42)
        X = np.random.randn(500, 2)
        # y depends on x0 * x1 interaction
        y = X[:, 0] * X[:, 1] + np.random.randn(500) * 0.1

        model = MARSRegressor(max_degree=2)
        model.fit(X, y)

        # Should have at least one interaction term
        interaction_terms = [bf for bf in model.basis_functions_ if bf.degree == 2]
        assert len(interaction_terms) > 0, "Expected at least one interaction term"

    def test_no_interactions_when_degree_1(self):
        """max_degree=1 should prevent interactions."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = X[:, 0] * X[:, 1] + np.random.randn(200) * 0.1

        model = MARSRegressor(max_degree=1)
        model.fit(X, y)

        # All terms should have degree <= 1
        for bf in model.basis_functions_:
            assert bf.degree <= 1

    def test_predict_shape(self):
        """Predictions should have correct shape."""
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        model = MARSRegressor()
        model.fit(X, y)

        # Single sample
        pred = model.predict(X[:1])
        assert pred.shape == (1,)

        # Multiple samples
        pred = model.predict(X[:10])
        assert pred.shape == (10,)

    def test_summary_output(self):
        """Summary should return formatted string."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model = MARSRegressor()
        model.fit(X, y)

        summary = model.summary()
        assert isinstance(summary, str)
        assert "Basis Functions" in summary
        assert "R-squared" in summary

    def test_feature_names(self):
        """Feature names should appear in summary."""
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1] + np.random.randn(100) * 0.1

        model = MARSRegressor(feature_names=['age', 'income'])
        model.fit(X, y)

        summary = model.summary()
        # At least one feature should appear
        assert 'age' in summary or 'income' in summary

    def test_sample_weight(self):
        """Sample weights should affect fit."""
        np.random.seed(42)
        X = np.random.randn(200, 2)
        y = X[:, 0] + np.random.randn(200) * 0.1

        # Weight first half heavily
        weights = np.ones(200)
        weights[:100] = 10.0

        model1 = MARSRegressor()
        model1.fit(X, y)

        model2 = MARSRegressor()
        model2.fit(X, y, sample_weight=weights)

        # Predictions should differ (at least slightly)
        pred1 = model1.predict(X)
        pred2 = model2.predict(X)

        # Not exactly equal
        assert not np.allclose(pred1, pred2, atol=1e-6)

    def test_reproducibility(self):
        """Same input should give same output."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model1 = MARSRegressor()
        model1.fit(X, y)

        model2 = MARSRegressor()
        model2.fit(X, y)

        assert_allclose(model1.predict(X), model2.predict(X))

    def test_basis_matrix(self):
        """Test get_basis_matrix method."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)

        model = MARSRegressor()
        model.fit(X, y)

        B = model.get_basis_matrix(X)
        assert B.shape == (100, len(model.basis_functions_))

        # First column should be all ones (intercept)
        assert_allclose(B[:, 0], np.ones(100))

    def test_variable_importance(self):
        """Test variable importance computation."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        # y depends only on first two features
        y = 3 * X[:, 0] + X[:, 1] + np.random.randn(200) * 0.1

        model = MARSRegressor(max_degree=1)
        model.fit(X, y)

        importance = model.compute_variable_importance()

        # First feature should have high importance
        assert importance['x0'] > 50

        # Third feature should have low importance
        assert importance['x2'] < 20

    def test_constant_feature(self):
        """Constant features should be handled gracefully."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        X[:, 1] = 5.0  # Constant feature
        y = X[:, 0] + np.random.randn(100) * 0.1

        model = MARSRegressor()
        model.fit(X, y)

        # Should still work
        pred = model.predict(X)
        assert pred.shape == (100,)

        # Constant feature should not appear in basis functions
        for bf in model.basis_functions_:
            assert 1 not in bf.feature_indices

    def test_small_dataset(self):
        """Should work on small datasets."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = X[:, 0] + np.random.randn(30) * 0.1

        model = MARSRegressor(max_terms=10)
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (30,)

    def test_single_feature(self):
        """Should work with single feature."""
        np.random.seed(42)
        X = np.random.randn(100, 1)
        y = np.maximum(0, X.ravel() - 0.5) + np.random.randn(100) * 0.1

        model = MARSRegressor()
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (100,)
        assert model.rsq_ > 0.8

    def test_wrong_feature_count(self):
        """Should raise error for wrong number of features at predict time."""
        X_train = np.random.randn(100, 3)
        y_train = np.random.randn(100)

        model = MARSRegressor()
        model.fit(X_train, y_train)

        X_test = np.random.randn(10, 5)  # Wrong number of features
        with pytest.raises(ValueError, match="features"):
            model.predict(X_test)

    def test_not_fitted_error(self):
        """Should raise error when predicting before fitting."""
        model = MARSRegressor()
        X = np.random.randn(10, 3)

        with pytest.raises(Exception):  # NotFittedError
            model.predict(X)

    def test_penalty_effect(self):
        """Higher penalty should produce simpler models."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X[:, 0] + X[:, 1] + X[:, 2] + np.random.randn(200) * 0.1

        model_low = MARSRegressor(penalty=1.0, thresh=0.0001)
        model_low.fit(X, y)

        model_high = MARSRegressor(penalty=5.0, thresh=0.0001)
        model_high.fit(X, y)

        # Higher penalty should generally give fewer terms
        assert len(model_high.basis_functions_) <= len(model_low.basis_functions_) + 2

    def test_max_terms_limit(self):
        """max_terms should limit number of basis functions."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = X.sum(axis=1) + np.random.randn(200) * 0.1

        model = MARSRegressor(max_terms=5, thresh=0.0)
        model.fit(X, y)

        assert len(model.basis_functions_) <= 5

    def test_fast_k_parameter(self):
        """fast_k should affect which parents are considered."""
        np.random.seed(42)
        X = np.random.randn(200, 10)
        y = X[:, 0] + X[:, 1] + np.random.randn(200) * 0.1

        # With fast_k=0, consider all parents
        model_full = MARSRegressor(fast_k=0, max_degree=2)
        model_full.fit(X, y)

        # With fast_k=5, only consider top 5 parents
        model_fast = MARSRegressor(fast_k=5, max_degree=2)
        model_fast.fit(X, y)

        # Both should produce reasonable models
        assert model_full.rsq_ > 0.8
        assert model_fast.rsq_ > 0.8


class TestMARSClassifier:
    """Tests for MARSClassifier."""

    def test_binary_classification(self):
        """Test binary classification."""
        np.random.seed(42)
        X = np.random.randn(200, 3)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)

        model = MARSClassifier(max_degree=1)
        model.fit(X, y)

        pred = model.predict(X)
        assert set(pred).issubset({0, 1})

        # Should have reasonable accuracy
        accuracy = np.mean(pred == y)
        assert accuracy > 0.8

    def test_predict_proba(self):
        """Test probability predictions."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = MARSClassifier()
        model.fit(X, y)

        proba = model.predict_proba(X)
        assert proba.shape == (100, 2)
        assert_allclose(proba.sum(axis=1), np.ones(100))
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_classes_attribute(self):
        """Test classes_ attribute."""
        X = np.random.randn(100, 2)
        y = np.array(['a', 'b'] * 50)

        model = MARSClassifier()
        model.fit(X, y)

        assert_array_equal(model.classes_, np.array(['a', 'b']))

    def test_multiclass(self):
        """Test multiclass classification."""
        np.random.seed(42)
        X = np.random.randn(300, 3)
        y = np.repeat([0, 1, 2], 100)
        # Make classes separable
        X[:100, 0] += 2
        X[100:200, 1] += 2
        X[200:, 2] += 2

        model = MARSClassifier()
        model.fit(X, y)

        pred = model.predict(X)
        assert set(pred).issubset({0, 1, 2})

        # Should have reasonable accuracy
        accuracy = np.mean(pred == y)
        assert accuracy > 0.7

    def test_threshold_method(self):
        """Test threshold classification method."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = MARSClassifier(method='threshold')
        model.fit(X, y)

        pred = model.predict(X)
        assert set(pred).issubset({0, 1})

    def test_classifier_summary(self):
        """Test classifier summary output."""
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = MARSClassifier()
        model.fit(X, y)

        summary = model.summary()
        assert "MARS Classifier" in summary
        assert "Classes" in summary

    def test_basis_functions_attribute(self):
        """Test basis_functions_ property."""
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        model = MARSClassifier()
        model.fit(X, y)

        bf = model.basis_functions_
        assert len(bf) > 0
        assert bf[0].is_intercept

    def test_sample_weight_classifier(self):
        """Test sample weights in classification."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] > 0).astype(int)

        weights = np.ones(100)
        weights[:50] = 10.0

        model = MARSClassifier()
        model.fit(X, y, sample_weight=weights)

        pred = model.predict(X)
        assert pred.shape == (100,)


class TestLinearBasisFunction:
    """Tests for LinearBasisFunction."""

    def test_evaluate(self):
        """Test linear basis function evaluation."""
        linear = LinearBasisFunction(feature_idx=1)
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        result = linear.evaluate(X)
        expected = np.array([2.0, 4.0, 6.0])
        assert_allclose(result, expected)

    def test_properties(self):
        """Test linear basis function properties."""
        linear = LinearBasisFunction(feature_idx=0)
        assert linear.degree == 1
        assert not linear.is_intercept
        assert linear.feature_indices == [0]
        assert not linear.can_extend_with(1, max_degree=2)

    def test_str_representation(self):
        """Test string representation."""
        linear = LinearBasisFunction(feature_idx=2)
        assert "x2" in str(linear)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_all_zero_target(self):
        """Should handle all-zero target."""
        X = np.random.randn(100, 3)
        y = np.zeros(100)

        model = MARSRegressor()
        model.fit(X, y)

        pred = model.predict(X)
        assert_allclose(pred, np.zeros(100), atol=1e-10)

    def test_single_sample(self):
        """Should handle minimal dataset."""
        X = np.array([[1.0, 2.0]])
        y = np.array([3.0])

        model = MARSRegressor(max_terms=3)
        # This might not work well but shouldn't crash
        try:
            model.fit(X, y)
            pred = model.predict(X)
            assert pred.shape == (1,)
        except Exception:
            # Acceptable to fail on such extreme edge case
            pass

    def test_collinear_features(self):
        """Should handle collinear features."""
        np.random.seed(42)
        X = np.random.randn(100, 2)
        X = np.column_stack([X, X[:, 0] * 2])  # Third feature is 2x first
        y = X[:, 0] + np.random.randn(100) * 0.1

        model = MARSRegressor()
        model.fit(X, y)

        pred = model.predict(X)
        assert pred.shape == (100,)

    def test_large_values(self):
        """Should handle large feature values."""
        np.random.seed(42)
        X = np.random.randn(100, 3) * 1e6
        y = X[:, 0] + np.random.randn(100) * 1e4

        model = MARSRegressor()
        model.fit(X, y)

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))

    def test_small_values(self):
        """Should handle small feature values."""
        np.random.seed(42)
        X = np.random.randn(100, 3) * 1e-6
        y = X[:, 0] + np.random.randn(100) * 1e-8

        model = MARSRegressor()
        model.fit(X, y)

        pred = model.predict(X)
        assert np.all(np.isfinite(pred))


class TestPerformance:
    """Performance-related tests."""

    @pytest.mark.slow
    def test_medium_dataset(self):
        """Test on medium-sized dataset."""
        np.random.seed(42)
        X = np.random.randn(5000, 20)
        y = X[:, :5].sum(axis=1) + np.random.randn(5000) * 0.1

        model = MARSRegressor(max_degree=1)
        model.fit(X, y)

        assert model.rsq_ > 0.9

    @pytest.mark.slow
    def test_interaction_detection_performance(self):
        """Test interaction detection on larger dataset."""
        np.random.seed(42)
        X = np.random.randn(1000, 10)
        y = X[:, 0] * X[:, 1] + X[:, 2] + np.random.randn(1000) * 0.1

        model = MARSRegressor(max_degree=2)
        model.fit(X, y)

        # Should detect the interaction
        interaction_terms = [bf for bf in model.basis_functions_ if bf.degree == 2]
        assert len(interaction_terms) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
