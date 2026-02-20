"""Tests for multi-output ensemble wrappers."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clf_data():
    """Multi-output classification dataset (3 outputs, binary each)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    Y = np.column_stack([
        (X[:, 0] + rng.randn(200) * 0.3 > 0).astype(int),
        (X[:, 1] + rng.randn(200) * 0.3 > 0).astype(int),
        (X[:, 2] + rng.randn(200) * 0.3 > 0).astype(int),
    ])
    return X, Y


@pytest.fixture
def reg_data():
    """Multi-output regression dataset (3 outputs)."""
    rng = np.random.RandomState(42)
    X = rng.randn(200, 5)
    Y = np.column_stack([
        X[:, 0] * 2 + rng.randn(200) * 0.1,
        X[:, 1] * -1 + rng.randn(200) * 0.1,
        X[:, 2] + X[:, 0] + rng.randn(200) * 0.1,
    ])
    return X, Y


# ---------------------------------------------------------------------------
# MultiOutputClassifier
# ---------------------------------------------------------------------------

class TestMultiOutputClassifier:
    """Tests for MultiOutputClassifier."""

    def test_fit_predict_shape(self, clf_data):
        """Predictions have correct shape (n_samples, n_outputs)."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
        clf.fit(X, Y)
        preds = clf.predict(X)
        assert preds.shape == Y.shape

    def test_predict_proba_returns_list(self, clf_data):
        """predict_proba returns a list of arrays, one per output."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
        clf.fit(X, Y)
        probas = clf.predict_proba(X)
        assert isinstance(probas, list)
        assert len(probas) == Y.shape[1]
        for p in probas:
            assert p.shape[0] == X.shape[0]
            assert p.shape[1] == 2  # binary outputs

    def test_classes_attribute(self, clf_data):
        """classes_ stores unique labels for each output."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
        clf.fit(X, Y)
        assert len(clf.classes_) == 3
        for c in clf.classes_:
            np.testing.assert_array_equal(c, [0, 1])

    def test_n_outputs(self, clf_data):
        """n_outputs_ is set correctly."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(DecisionTreeClassifier())
        clf.fit(X, Y)
        assert clf.n_outputs_ == 3

    def test_not_fitted_raises(self):
        """Calling predict before fit raises RuntimeError."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        clf = MultiOutputClassifier(DecisionTreeClassifier())
        with pytest.raises(RuntimeError, match="has not been fitted"):
            clf.predict(np.zeros((5, 3)))

    def test_parallel_fitting(self, clf_data):
        """Parallel fitting produces the same shape as sequential."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(
            DecisionTreeClassifier(random_state=42), n_jobs=2
        )
        clf.fit(X, Y)
        preds = clf.predict(X)
        assert preds.shape == Y.shape

    def test_score(self, clf_data):
        """score returns a float between 0 and 1."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
        clf.fit(X, Y)
        s = clf.score(X, Y)
        assert 0.0 <= s <= 1.0

    def test_random_state_propagation(self, clf_data):
        """random_state is propagated to cloned estimators."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        clf = MultiOutputClassifier(
            DecisionTreeClassifier(), random_state=42
        )
        clf.fit(X, Y)
        for est in clf.estimators_:
            assert est.random_state == 42

    def test_1d_target_promoted(self):
        """1-D target Y is promoted to (n_samples, 1)."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        rng = np.random.RandomState(0)
        X = rng.randn(50, 3)
        y = rng.randint(0, 2, 50)
        clf = MultiOutputClassifier(DecisionTreeClassifier())
        clf.fit(X, y)
        assert clf.n_outputs_ == 1
        assert clf.predict(X).shape == (50, 1)

    def test_sample_weight(self, clf_data):
        """Fitting with sample_weight does not raise."""
        from endgame.ensemble.multi_output import MultiOutputClassifier

        X, Y = clf_data
        w = np.ones(X.shape[0])
        clf = MultiOutputClassifier(DecisionTreeClassifier(random_state=42))
        clf.fit(X, Y, sample_weight=w)
        preds = clf.predict(X)
        assert preds.shape == Y.shape


# ---------------------------------------------------------------------------
# MultiOutputRegressor
# ---------------------------------------------------------------------------

class TestMultiOutputRegressor:
    """Tests for MultiOutputRegressor."""

    def test_fit_predict_shape(self, reg_data):
        """Predictions have correct shape."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        reg = MultiOutputRegressor(Ridge())
        reg.fit(X, Y)
        preds = reg.predict(X)
        assert preds.shape == Y.shape

    def test_feature_importances(self, reg_data):
        """feature_importances_ averages across outputs."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        reg = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=10, random_state=42)
        )
        reg.fit(X, Y)
        imp = reg.feature_importances_
        assert imp.shape == (X.shape[1],)
        assert np.all(imp >= 0)

    def test_feature_importances_raises_if_unsupported(self, reg_data):
        """feature_importances_ raises if base estimator lacks it."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        reg = MultiOutputRegressor(Ridge())
        reg.fit(X, Y)
        with pytest.raises(AttributeError, match="does not provide"):
            _ = reg.feature_importances_

    def test_score(self, reg_data):
        """score returns a reasonable R^2 value."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        reg = MultiOutputRegressor(Ridge())
        reg.fit(X, Y)
        s = reg.score(X, Y)
        # Ridge should fit this well
        assert s > 0.5

    def test_not_fitted_raises(self):
        """Calling predict before fit raises RuntimeError."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        reg = MultiOutputRegressor(Ridge())
        with pytest.raises(RuntimeError, match="has not been fitted"):
            reg.predict(np.zeros((5, 3)))

    def test_parallel_fitting(self, reg_data):
        """Parallel fitting produces the same shape as sequential."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        reg = MultiOutputRegressor(Ridge(), n_jobs=2)
        reg.fit(X, Y)
        preds = reg.predict(X)
        assert preds.shape == Y.shape

    def test_sample_weight(self, reg_data):
        """Fitting with sample_weight does not raise."""
        from endgame.ensemble.multi_output import MultiOutputRegressor

        X, Y = reg_data
        w = np.ones(X.shape[0])
        reg = MultiOutputRegressor(Ridge())
        reg.fit(X, Y, sample_weight=w)
        preds = reg.predict(X)
        assert preds.shape == Y.shape


# ---------------------------------------------------------------------------
# ClassifierChain
# ---------------------------------------------------------------------------

class TestClassifierChain:
    """Tests for ClassifierChain."""

    def test_fit_predict_shape(self, clf_data):
        """Predictions have correct shape."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(random_state=42), order="auto"
        )
        chain.fit(X, Y)
        preds = chain.predict(X)
        assert preds.shape == Y.shape

    def test_predict_proba(self, clf_data):
        """predict_proba returns a list per output in original column order."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(random_state=42), order="auto"
        )
        chain.fit(X, Y)
        probas = chain.predict_proba(X)
        assert isinstance(probas, list)
        assert len(probas) == 3
        for p in probas:
            assert p.shape[0] == X.shape[0]

    def test_order_auto(self, clf_data):
        """'auto' order produces a valid permutation."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(DecisionTreeClassifier(), order="auto")
        chain.fit(X, Y)
        assert sorted(chain.order_) == [0, 1, 2]

    def test_order_random(self, clf_data):
        """'random' order produces a valid permutation."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(), order="random", random_state=42
        )
        chain.fit(X, Y)
        assert sorted(chain.order_) == [0, 1, 2]

    def test_order_explicit(self, clf_data):
        """Explicit order is respected."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(random_state=42), order=[2, 0, 1]
        )
        chain.fit(X, Y)
        assert chain.order_ == [2, 0, 1]

    def test_order_invalid_raises(self, clf_data):
        """Invalid explicit order raises ValueError."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(DecisionTreeClassifier(), order=[0, 0, 1])
        with pytest.raises(ValueError, match="permutation"):
            chain.fit(X, Y)

    def test_chain_augments_features(self, clf_data):
        """Each successive estimator sees more features (X + previous preds)."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(random_state=42), order=[0, 1, 2]
        )
        chain.fit(X, Y)
        # First estimator sees 5 features, second 6, third 7
        assert chain.estimators_[0].n_features_in_ == 5
        assert chain.estimators_[1].n_features_in_ == 6
        assert chain.estimators_[2].n_features_in_ == 7

    def test_not_fitted_raises(self):
        """Calling predict before fit raises RuntimeError."""
        from endgame.ensemble.multi_output import ClassifierChain

        chain = ClassifierChain(DecisionTreeClassifier())
        with pytest.raises(RuntimeError, match="has not been fitted"):
            chain.predict(np.zeros((5, 3)))

    def test_score(self, clf_data):
        """score returns a float between 0 and 1."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain = ClassifierChain(
            DecisionTreeClassifier(random_state=42), order="auto"
        )
        chain.fit(X, Y)
        s = chain.score(X, Y)
        assert 0.0 <= s <= 1.0

    def test_random_order_reproducible(self, clf_data):
        """Same random_state produces the same order."""
        from endgame.ensemble.multi_output import ClassifierChain

        X, Y = clf_data
        chain1 = ClassifierChain(
            DecisionTreeClassifier(), order="random", random_state=99
        )
        chain1.fit(X, Y)

        chain2 = ClassifierChain(
            DecisionTreeClassifier(), order="random", random_state=99
        )
        chain2.fit(X, Y)

        assert chain1.order_ == chain2.order_


# ---------------------------------------------------------------------------
# RegressorChain
# ---------------------------------------------------------------------------

class TestRegressorChain:
    """Tests for RegressorChain."""

    def test_fit_predict_shape(self, reg_data):
        """Predictions have correct shape."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(Ridge(), order="auto")
        chain.fit(X, Y)
        preds = chain.predict(X)
        assert preds.shape == Y.shape

    def test_order_auto(self, reg_data):
        """'auto' order produces a valid permutation."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(Ridge(), order="auto")
        chain.fit(X, Y)
        assert sorted(chain.order_) == [0, 1, 2]

    def test_order_explicit(self, reg_data):
        """Explicit order is respected."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(Ridge(), order=[1, 2, 0])
        chain.fit(X, Y)
        assert chain.order_ == [1, 2, 0]

    def test_chain_augments_features(self, reg_data):
        """Each successive estimator sees more features."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(
            DecisionTreeRegressor(random_state=42), order=[0, 1, 2]
        )
        chain.fit(X, Y)
        assert chain.estimators_[0].n_features_in_ == 5
        assert chain.estimators_[1].n_features_in_ == 6
        assert chain.estimators_[2].n_features_in_ == 7

    def test_feature_importances(self, reg_data):
        """feature_importances_ returns original-feature-only importances."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(
            RandomForestRegressor(n_estimators=10, random_state=42),
            order=[0, 1, 2],
        )
        chain.fit(X, Y)
        imp = chain.feature_importances_
        # Should have length = original n_features, not augmented
        assert imp.shape == (X.shape[1],)

    def test_score(self, reg_data):
        """score returns a reasonable R^2 value."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        chain = RegressorChain(Ridge(), order="auto")
        chain.fit(X, Y)
        s = chain.score(X, Y)
        assert s > 0.5

    def test_not_fitted_raises(self):
        """Calling predict before fit raises RuntimeError."""
        from endgame.ensemble.multi_output import RegressorChain

        chain = RegressorChain(Ridge())
        with pytest.raises(RuntimeError, match="has not been fitted"):
            chain.predict(np.zeros((5, 3)))

    def test_sample_weight(self, reg_data):
        """Fitting with sample_weight does not raise."""
        from endgame.ensemble.multi_output import RegressorChain

        X, Y = reg_data
        w = np.ones(X.shape[0])
        chain = RegressorChain(Ridge(), order="auto")
        chain.fit(X, Y, sample_weight=w)
        preds = chain.predict(X)
        assert preds.shape == Y.shape


# ---------------------------------------------------------------------------
# Import from top-level ensemble
# ---------------------------------------------------------------------------

class TestEnsembleExports:
    """Verify all four classes are importable from endgame.ensemble."""

    def test_import_multi_output_classifier(self):
        from endgame.ensemble import MultiOutputClassifier
        assert MultiOutputClassifier is not None

    def test_import_multi_output_regressor(self):
        from endgame.ensemble import MultiOutputRegressor
        assert MultiOutputRegressor is not None

    def test_import_classifier_chain(self):
        from endgame.ensemble import ClassifierChain
        assert ClassifierChain is not None

    def test_import_regressor_chain(self):
        from endgame.ensemble import RegressorChain
        assert RegressorChain is not None


# ---------------------------------------------------------------------------
# _determine_chain_order helper
# ---------------------------------------------------------------------------

class TestDetermineChainOrder:
    """Tests for the _determine_chain_order helper."""

    def test_auto_returns_permutation(self):
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        Y = rng.randn(100, 4)
        order = _determine_chain_order(Y, "auto")
        assert sorted(order) == [0, 1, 2, 3]

    def test_random_returns_permutation(self):
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        Y = rng.randn(100, 4)
        order = _determine_chain_order(Y, "random", random_state=0)
        assert sorted(order) == [0, 1, 2, 3]

    def test_explicit_passthrough(self):
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        Y = rng.randn(100, 3)
        order = _determine_chain_order(Y, [2, 1, 0])
        assert order == [2, 1, 0]

    def test_invalid_order_string_raises(self):
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        Y = rng.randn(100, 3)
        with pytest.raises(ValueError, match="must be"):
            _determine_chain_order(Y, "foobar")

    def test_invalid_explicit_raises(self):
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        Y = rng.randn(100, 3)
        with pytest.raises(ValueError, match="permutation"):
            _determine_chain_order(Y, [0, 1, 5])

    def test_auto_correlated_outputs(self):
        """Auto ordering should place correlated columns adjacently."""
        from endgame.ensemble.multi_output import _determine_chain_order

        rng = np.random.RandomState(42)
        base = rng.randn(200)
        # col0 and col2 are highly correlated, col1 is independent
        Y = np.column_stack([
            base + rng.randn(200) * 0.01,
            rng.randn(200),
            base + rng.randn(200) * 0.01,
        ])
        order = _determine_chain_order(Y, "auto")
        # col0 and col2 should be adjacent in the ordering
        pos0 = order.index(0)
        pos2 = order.index(2)
        assert abs(pos0 - pos2) == 1, (
            f"Correlated columns 0 and 2 should be adjacent, "
            f"but got positions {pos0} and {pos2}"
        )
