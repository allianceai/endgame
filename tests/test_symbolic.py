"""Tests for Symbolic Regression models (pure-Python GP engine)."""

import pytest
import numpy as np
from sklearn.metrics import r2_score, accuracy_score

from endgame.models.symbolic import SymbolicRegressor, SymbolicClassifier, PRESETS, OPERATOR_SETS
from endgame.models.symbolic._expression import (
    Binary,
    Constant,
    Node,
    Unary,
    Variable,
    complexity,
    depth,
    evaluate,
    get_constants,
    set_constants,
    get_variables_used,
    to_string,
    to_sympy,
)
from endgame.models.symbolic._operators import validate_operators
from endgame.models.symbolic._population import (
    clone_tree,
    compute_fitness,
    random_tree,
    ramped_half_and_half,
    tournament_select,
    subtree_crossover,
    mutate,
    evolve_population,
)
from endgame.models.symbolic._pareto import ParetoFrontier
from endgame.models.symbolic._constant_optimizer import optimize_constants


# ============================================================
# Expression tree tests
# ============================================================

class TestExpressionTree:
    """Test expression tree construction, evaluation, and export."""

    def test_constant_eval(self):
        node = Constant(3.14)
        X = np.random.randn(10, 2)
        result = evaluate(node, X)
        assert result.shape == (10,)
        np.testing.assert_allclose(result, 3.14)

    def test_variable_eval(self):
        node = Variable(1, "x1")
        X = np.random.randn(10, 3)
        result = evaluate(node, X)
        np.testing.assert_array_equal(result, X[:, 1])

    def test_binary_eval(self):
        # 2 + x0
        node = Binary("+", Constant(2.0), Variable(0))
        X = np.array([[1.0], [2.0], [3.0]])
        result = evaluate(node, X)
        np.testing.assert_allclose(result, [3.0, 4.0, 5.0])

    def test_unary_eval(self):
        # sin(x0)
        node = Unary("sin", Variable(0))
        X = np.array([[0.0], [np.pi / 2], [np.pi]])
        result = evaluate(node, X)
        np.testing.assert_allclose(result, np.sin(X[:, 0]), atol=1e-10)

    def test_nested_eval(self):
        # x0^2 + x1
        node = Binary("+", Unary("square", Variable(0)), Variable(1))
        X = np.array([[2.0, 3.0], [1.0, -1.0]])
        result = evaluate(node, X)
        np.testing.assert_allclose(result, [7.0, 0.0])

    def test_complexity(self):
        # x0 + x1 has 3 nodes
        node = Binary("+", Variable(0), Variable(1))
        assert complexity(node) == 3

        # sin(x0) has 2 nodes
        node2 = Unary("sin", Variable(0))
        assert complexity(node2) == 2

        assert complexity(Constant(1.0)) == 1

    def test_depth(self):
        node = Binary("+", Unary("sin", Variable(0)), Constant(1.0))
        assert depth(node) == 2

    def test_get_set_constants(self):
        node = Binary("+", Constant(1.0), Binary("*", Constant(2.0), Variable(0)))
        consts = get_constants(node)
        assert consts == [1.0, 2.0]

        set_constants(node, [10.0, 20.0])
        assert get_constants(node) == [10.0, 20.0]

    def test_get_variables_used(self):
        node = Binary("+", Variable(0), Binary("*", Variable(2), Constant(3.0)))
        assert get_variables_used(node) == {0, 2}

    def test_to_string(self):
        node = Binary("+", Variable(0), Constant(2.0))
        s = to_string(node, ["x0"])
        assert "x0" in s
        assert "2" in s
        assert "+" in s

    def test_to_string_feature_names(self):
        node = Binary("*", Variable(0), Variable(1))
        s = to_string(node, ["height", "weight"])
        assert "height" in s
        assert "weight" in s

    def test_to_sympy(self):
        sympy = pytest.importorskip("sympy")
        node = Binary("+", Unary("square", Variable(0)), Variable(1))
        expr = to_sympy(node, ["x0", "x1"])
        assert "x0" in str(expr)
        assert "x1" in str(expr)

    def test_clone(self):
        node = Binary("+", Constant(1.0), Variable(0))
        cloned = clone_tree(node)
        # Modify original constant
        node.left.value = 999.0
        assert cloned.left.value == 1.0  # Clone is independent

    def test_safe_division(self):
        node = Binary("/", Constant(1.0), Constant(0.0))
        X = np.random.randn(5, 1)
        result = evaluate(node, X)
        assert np.all(np.isfinite(result))


# ============================================================
# Population / GP tests
# ============================================================

class TestPopulation:
    """Test GP engine primitives."""

    def test_random_tree(self):
        rng = np.random.default_rng(42)
        tree = random_tree(rng, 3, ["+", "*"], ["sin"], 3, "grow")
        assert isinstance(tree, (Constant, Variable, Unary, Binary))

    def test_ramped_half_and_half(self):
        rng = np.random.default_rng(42)
        pop = ramped_half_and_half(rng, 20, 3, ["+", "*", "/"], ["sin"], 1, 4)
        assert len(pop) == 20
        # Should have variety of depths
        depths = [depth(t) for t in pop]
        assert len(set(depths)) > 1

    def test_compute_fitness(self):
        tree = Binary("+", Variable(0), Variable(1))
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([3.0, 7.0])
        loss_fn = lambda yt, yp: float(np.mean((yt - yp) ** 2))
        fitness = compute_fitness(tree, X, y, loss_fn, 0.0)
        assert fitness == 0.0

    def test_tournament_select(self):
        rng = np.random.default_rng(42)
        pop = [Constant(float(i)) for i in range(10)]
        fitnesses = np.array([float(9 - i) for i in range(10)])  # Last is best
        selected = tournament_select(pop, fitnesses, rng, tournament_size=3)
        assert isinstance(selected, Constant)

    def test_subtree_crossover(self):
        rng = np.random.default_rng(42)
        p1 = Binary("+", Variable(0), Constant(1.0))
        p2 = Binary("*", Variable(1), Constant(2.0))
        c1, c2 = subtree_crossover(p1, p2, rng)
        assert isinstance(c1, (Constant, Variable, Unary, Binary))
        assert isinstance(c2, (Constant, Variable, Unary, Binary))

    def test_mutate(self):
        rng = np.random.default_rng(42)
        tree = Binary("+", Variable(0), Constant(1.0))
        mutated = mutate(tree, rng, ["+", "*"], ["sin"], 3)
        assert isinstance(mutated, (Constant, Variable, Unary, Binary))

    def test_evolve_population(self):
        rng = np.random.default_rng(42)
        pop = ramped_half_and_half(rng, 10, 2, ["+", "*"], [], 1, 3)
        X = np.random.randn(50, 2)
        y = X[:, 0] + X[:, 1]
        loss_fn = lambda yt, yp: float(np.mean((yt - yp) ** 2))
        fitnesses = np.array([compute_fitness(t, X, y, loss_fn) for t in pop])

        new_pop = evolve_population(pop, fitnesses, rng, ["+", "*"], [], 2)
        assert len(new_pop) == 10

    def test_validate_operators_invalid(self):
        with pytest.raises(ValueError, match="Unknown binary operator"):
            validate_operators(["++"], [])
        with pytest.raises(ValueError, match="Unknown unary operator"):
            validate_operators(["+"], ["nonexist"])


# ============================================================
# Constant optimizer tests
# ============================================================

class TestConstantOptimizer:
    """Test constant optimization."""

    def test_optimize_improves_loss(self):
        # y = 3*x + 2, tree starts with wrong constants
        tree = Binary("+", Binary("*", Constant(1.0), Variable(0)), Constant(0.0))
        X = np.random.RandomState(42).randn(100, 1)
        y = 3.0 * X[:, 0] + 2.0
        loss_fn = lambda yt, yp: float(np.mean((yt - yp) ** 2))

        initial_loss = loss_fn(y, evaluate(tree, X))
        optimized = optimize_constants(tree, X, y, loss_fn, n_restarts=2, max_steps=30)
        optimized_loss = loss_fn(y, evaluate(optimized, X))

        assert optimized_loss <= initial_loss

    def test_no_constants_unchanged(self):
        tree = Binary("+", Variable(0), Variable(1))
        X = np.random.randn(20, 2)
        y = X[:, 0] + X[:, 1]
        loss_fn = lambda yt, yp: float(np.mean((yt - yp) ** 2))
        result = optimize_constants(tree, X, y, loss_fn)
        # Should return same structure since no constants to optimize
        assert isinstance(result, Binary)


# ============================================================
# Pareto frontier tests
# ============================================================

class TestParetoFrontier:
    """Test Pareto frontier tracking."""

    def test_update_and_get_best(self):
        frontier = ParetoFrontier()
        tree1 = Binary("+", Variable(0), Constant(1.0))
        tree2 = Variable(0)

        frontier.update(tree1, 0.5, ["x0"])
        frontier.update(tree2, 1.0, ["x0"])

        best = frontier.get_best("best")
        assert best.loss == 0.5

    def test_better_replaces(self):
        frontier = ParetoFrontier()
        tree = Binary("+", Variable(0), Constant(1.0))
        frontier.update(tree, 1.0, ["x0"])
        frontier.update(tree, 0.5, ["x0"])

        entry = frontier.get_at_complexity(complexity(tree))
        assert entry.loss == 0.5

    def test_to_dataframe(self):
        frontier = ParetoFrontier()
        frontier.update(Variable(0), 1.0, ["x0"])
        frontier.update(Binary("+", Variable(0), Constant(1.0)), 0.5, ["x0"])

        df = frontier.to_dataframe()
        assert "equation" in df.columns
        assert "loss" in df.columns
        assert "complexity" in df.columns
        assert len(df) == 2


# ============================================================
# SymbolicRegressor sklearn interface tests
# ============================================================

class TestSymbolicRegressorBasic:
    """Test basic functionality."""

    def test_import(self):
        assert SymbolicRegressor is not None
        assert SymbolicClassifier is not None

    def test_presets_exist(self):
        assert "fast" in PRESETS
        assert "default" in PRESETS
        assert "competition" in PRESETS
        assert "interpretable" in PRESETS

    def test_operator_sets_exist(self):
        assert "basic" in OPERATOR_SETS
        assert "arithmetic" in OPERATOR_SETS
        assert "trigonometric" in OPERATOR_SETS
        assert "scientific" in OPERATOR_SETS
        assert "full" in OPERATOR_SETS

    def test_init_regressor(self):
        sr = SymbolicRegressor(preset="fast", operators="basic")
        assert sr.preset == "fast"
        assert sr.operators == "basic"

    def test_init_classifier(self):
        clf = SymbolicClassifier(preset="fast", operators="basic")
        assert clf.preset == "fast"
        assert clf.operators == "basic"

    def test_get_operators_basic(self):
        sr = SymbolicRegressor(operators="basic")
        binary_ops, unary_ops = sr._get_operators()
        assert "+" in binary_ops
        assert "*" in binary_ops
        assert len(unary_ops) == 0

    def test_get_operators_scientific(self):
        sr = SymbolicRegressor(operators="scientific")
        binary_ops, unary_ops = sr._get_operators()
        assert "+" in binary_ops
        assert "sin" in unary_ops
        assert "exp" in unary_ops

    def test_get_operators_custom(self):
        sr = SymbolicRegressor(binary_operators=["+", "*"], unary_operators=["sin"])
        binary_ops, unary_ops = sr._get_operators()
        assert binary_ops == ["+", "*"]
        assert unary_ops == ["sin"]

    def test_build_params(self):
        sr = SymbolicRegressor(preset="fast", operators="basic", niterations=10, maxsize=15)
        params = sr._build_params()
        assert params["niterations"] == 10
        assert params["maxsize"] == 15
        assert params["populations"] == 15  # From preset

    def test_invalid_preset(self):
        sr = SymbolicRegressor(preset="nonexistent")
        with pytest.raises(ValueError, match="Unknown preset"):
            sr._build_params()

    def test_invalid_operators(self):
        sr = SymbolicRegressor(operators="nonexistent")
        with pytest.raises(ValueError, match="Unknown operator set"):
            sr._get_operators()


class TestSymbolicRegressorFitting:
    """Tests that exercise the GP engine for actual fitting."""

    @pytest.fixture
    def regression_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = 2 * X[:, 0] ** 2 + 3 * X[:, 1] + 0.1 * np.random.randn(100)
        return X, y

    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        X = np.random.randn(100, 2)
        y = (X[:, 0] ** 2 + X[:, 1] > 0).astype(int)
        return X, y

    def test_fit_regressor(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="arithmetic",
            niterations=3, maxsize=12, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        assert hasattr(sr, "model_")
        assert hasattr(sr, "equations_")
        assert hasattr(sr, "best_equation_")
        assert sr.n_features_in_ == 2

    def test_predict_regressor(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="arithmetic",
            niterations=3, maxsize=12, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        y_pred = sr.predict(X)
        assert y_pred.shape == y.shape
        assert r2_score(y, y_pred) > 0.5

    def test_get_best_equation(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        eq = sr.get_best_equation()
        assert isinstance(eq, str)
        assert len(eq) > 0

    def test_sympy_export(self, regression_data):
        pytest.importorskip("sympy")
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        expr = sr.sympy()
        assert expr is not None

    def test_feature_importances(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        importances = sr.feature_importances_
        assert importances.shape == (2,)
        assert np.isclose(importances.sum(), 1.0) or importances.sum() == 0

    def test_pareto_frontier(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        frontier = sr.get_pareto_frontier()
        assert frontier is not None
        assert "equation" in frontier.columns
        assert "loss" in frontier.columns
        assert "complexity" in frontier.columns

    def test_summary(self, regression_data):
        X, y = regression_data
        sr = SymbolicRegressor(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        sr.fit(X, y)
        summary = sr.summary()
        assert isinstance(summary, str)
        assert "Best equation" in summary

    def test_fit_classifier_binary(self, classification_data):
        X, y = classification_data
        clf = SymbolicClassifier(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        clf.fit(X, y)
        assert hasattr(clf, "model_")
        assert clf.n_classes_ == 2
        assert len(clf.classes_) == 2

    def test_predict_classifier(self, classification_data):
        X, y = classification_data
        clf = SymbolicClassifier(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        clf.fit(X, y)
        y_pred = clf.predict(X)
        assert y_pred.shape == y.shape
        assert set(y_pred).issubset(set(y))

    def test_predict_proba_classifier(self, classification_data):
        X, y = classification_data
        clf = SymbolicClassifier(
            preset="fast", operators="basic",
            niterations=3, maxsize=10, verbosity=0, random_state=42,
        )
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0) and np.all(proba <= 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
