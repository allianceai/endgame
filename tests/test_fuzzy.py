"""Tests for endgame.fuzzy module — all 39 fuzzy learning algorithms."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=120, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    return X[:100], X[100:], y[:100], y[100:]


@pytest.fixture
def regression_data():
    X, y = make_regression(
        n_samples=120, n_features=5, noise=0.1, random_state=42,
    )
    return X[:100], X[100:], y[:100], y[100:]


@pytest.fixture
def small_classification_data():
    X, y = make_classification(
        n_samples=60, n_features=5, n_informative=3,
        n_redundant=1, random_state=42,
    )
    return X[:50], X[50:], y[:50], y[50:]


@pytest.fixture
def small_regression_data():
    X, y = make_regression(
        n_samples=60, n_features=4, noise=0.1, random_state=42,
    )
    return X[:50], X[50:], y[:50], y[50:]


# ── Phase 1: Core Utilities ──────────────────────────────────────────────────

class TestMembershipFunctions:
    def test_gaussian(self):
        from endgame.fuzzy.core.membership import GaussianMF
        mf = GaussianMF(center=0.0, sigma=1.0)
        x = np.array([-2, -1, 0, 1, 2], dtype=float)
        result = mf(x)
        assert result[2] == 1.0
        assert np.all(result >= 0) and np.all(result <= 1)
        assert result[1] == result[3]  # Symmetry

    def test_triangular(self):
        from endgame.fuzzy.core.membership import TriangularMF
        mf = TriangularMF(a=-1.0, b=0.0, c=1.0)
        result = mf(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert result[4] == 0.0
        assert abs(result[1] - 0.5) < 1e-10
        assert abs(result[3] - 0.5) < 1e-10

    def test_trapezoidal(self):
        from endgame.fuzzy.core.membership import TrapezoidalMF
        mf = TrapezoidalMF(a=0.0, b=1.0, c=3.0, d=4.0)
        result = mf(np.array([0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0]))
        assert result[0] == 0.0
        assert abs(result[1] - 0.5) < 1e-10
        assert result[2] == 1.0
        assert result[3] == 1.0
        assert result[4] == 1.0
        assert result[6] == 0.0

    def test_generalized_bell(self):
        from endgame.fuzzy.core.membership import GeneralizedBellMF
        mf = GeneralizedBellMF(a=1.0, b=2.0, c=0.0)
        result = mf(np.array([0.0]))
        assert result[0] == 1.0

    def test_sigmoidal(self):
        from endgame.fuzzy.core.membership import SigmoidalMF
        mf = SigmoidalMF(a=10.0, c=0.0)
        result = mf(np.array([-5.0, 0.0, 5.0]))
        assert abs(result[1] - 0.5) < 1e-10
        assert result[2] > 0.99

    def test_pi_mf(self):
        from endgame.fuzzy.core.membership import PiMF
        mf = PiMF(a=0.0, b=2.0, c=4.0, d=6.0)
        result = mf(np.array([0.0, 1.0, 3.0, 5.0, 6.0]))
        assert result[0] == 0.0
        assert result[2] == 1.0
        assert result[4] == 0.0

    def test_interval_type2_gaussian(self):
        from endgame.fuzzy.core.membership import IntervalType2GaussianMF
        mf = IntervalType2GaussianMF(center=0.0, sigma_lower=0.5, sigma_upper=1.0)
        x = np.array([0.0, 1.0])
        upper = mf.upper(x)
        lower = mf.lower(x)
        assert upper[0] == 1.0
        assert lower[0] == 1.0
        assert upper[1] >= lower[1]  # Upper is wider

    def test_create_uniform_mfs(self):
        from endgame.fuzzy.core.membership import create_uniform_mfs
        mfs = create_uniform_mfs(n_mfs=5, x_min=0, x_max=10, mf_type='gaussian')
        assert len(mfs) == 5
        # Center MF should peak at 5
        assert abs(mfs[2](np.array([5.0]))[0] - 1.0) < 1e-6

    def test_differentiable_mf(self):
        pytest.importorskip("torch")
        from endgame.fuzzy.core.membership import GaussianMF
        mf = GaussianMF(center=0.0, sigma=1.0)
        torch_mf = mf.to_torch()
        import torch
        x = torch.tensor([0.0, 1.0], requires_grad=True)
        y = torch_mf(x)
        y.sum().backward()
        assert x.grad is not None


class TestOperators:
    def test_min_tnorm(self):
        from endgame.fuzzy.core.operators import t_norm
        result = t_norm(0.7, 0.5, method='min')
        assert float(result) == 0.5

    def test_product_tnorm(self):
        from endgame.fuzzy.core.operators import t_norm
        result = t_norm(0.7, 0.5, method='product')
        assert abs(float(result) - 0.35) < 1e-10

    def test_lukasiewicz_tnorm(self):
        from endgame.fuzzy.core.operators import t_norm
        result = t_norm(0.7, 0.5, method='lukasiewicz')
        assert abs(float(result) - 0.2) < 1e-10

    def test_max_tconorm(self):
        from endgame.fuzzy.core.operators import t_conorm
        result = t_conorm(0.3, 0.7, method='max')
        assert float(result) == 0.7

    def test_probabilistic_sum(self):
        from endgame.fuzzy.core.operators import t_conorm
        result = t_conorm(0.3, 0.7, method='probabilistic_sum')
        assert abs(float(result) - 0.79) < 1e-10

    def test_hamacher(self):
        from endgame.fuzzy.core.operators import HamacherTNorm, HamacherTConorm
        tn = HamacherTNorm(p=1.0)
        result = tn(np.array(0.5), np.array(0.5))
        assert float(result) > 0

    def test_reduce(self):
        from endgame.fuzzy.core.operators import MinTNorm
        tn = MinTNorm()
        values = np.array([[0.8, 0.6, 0.9], [0.3, 0.7, 0.5]])
        result = tn.reduce(values, axis=1)
        assert result[0] == 0.6
        assert result[1] == 0.3


class TestDefuzzification:
    def test_centroid(self):
        from endgame.fuzzy.core.defuzzification import centroid
        x = np.linspace(0, 10, 1000)
        mf = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)
        result = centroid(x, mf)
        assert abs(result - 5.0) < 0.1

    def test_bisector(self):
        from endgame.fuzzy.core.defuzzification import bisector
        x = np.linspace(0, 10, 1000)
        mf = np.exp(-0.5 * ((x - 5) / 1.0) ** 2)
        result = bisector(x, mf)
        assert abs(result - 5.0) < 0.5

    def test_mean_of_maxima(self):
        from endgame.fuzzy.core.defuzzification import mean_of_maxima
        x = np.array([1, 2, 3, 4, 5], dtype=float)
        mf = np.array([0.0, 0.5, 1.0, 1.0, 0.5])
        result = mean_of_maxima(x, mf)
        assert abs(result - 3.5) < 1e-10

    def test_weighted_average(self):
        from endgame.fuzzy.core.defuzzification import weighted_average
        centers = np.array([2.0, 5.0, 8.0])
        heights = np.array([0.3, 0.6, 0.1])
        result = weighted_average(centers, heights)
        expected = (2 * 0.3 + 5 * 0.6 + 8 * 0.1) / 1.0
        assert abs(result - expected) < 1e-10

    def test_defuzzify_function(self):
        from endgame.fuzzy.core.defuzzification import defuzzify
        x = np.linspace(0, 10, 100)
        mf = np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
        result = defuzzify(x, mf, method='centroid')
        assert abs(result - 5.0) < 0.1


# ── Phase 2: Inference Systems ───────────────────────────────────────────────

class TestMamdaniFIS:
    def test_regression(self, regression_data):
        from endgame.fuzzy.inference import MamdaniFIS
        X_train, X_test, y_train, y_test = regression_data
        model = MamdaniFIS(n_mfs=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.inference import MamdaniClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = MamdaniClassifier(n_mfs=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)
        assert set(y_pred).issubset(set(y_train))


class TestTSK:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.inference import TSKRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = TSKRegressor(n_rules=5, order=1, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.inference import TSKClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = TSKClassifier(n_rules=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_order_zero(self, regression_data):
        from endgame.fuzzy.inference import TSKRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = TSKRegressor(n_rules=5, order=0, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestANFIS:
    def test_regressor(self, small_regression_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.inference import ANFISRegressor
        X_train, X_test, y_train, y_test = small_regression_data
        model = ANFISRegressor(n_rules=3, n_epochs=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)


class TestWangMendel:
    def test_regression(self, regression_data):
        from endgame.fuzzy.inference import WangMendelRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = WangMendelRegressor(n_mfs=3)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


# ── Phase 3: Fuzzy Classifiers ──────────────────────────────────────────────

class TestFuzzyKNN:
    def test_basic(self, classification_data):
        from endgame.fuzzy.classifiers import FuzzyKNNClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyKNNClassifier(n_neighbors=5, m=2.0)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)

    def test_pipeline(self, classification_data):
        from endgame.fuzzy.classifiers import FuzzyKNNClassifier
        X_train, X_test, y_train, y_test = classification_data
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('fknn', FuzzyKNNClassifier(n_neighbors=5)),
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        assert y_pred.shape == (20,)


class TestFuzzyDecisionTree:
    def test_classifier(self, classification_data):
        from endgame.fuzzy.classifiers import FuzzyDecisionTreeClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyDecisionTreeClassifier(max_depth=4, min_samples_leaf=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)

    def test_regressor(self, regression_data):
        from endgame.fuzzy.classifiers import FuzzyDecisionTreeRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FuzzyDecisionTreeRegressor(max_depth=4, min_samples_leaf=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestNEFCLASS:
    def test_basic(self, classification_data):
        from endgame.fuzzy.classifiers import NEFCLASSClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = NEFCLASSClassifier(n_rules_per_class=5, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)


# ── Phase 4: Neuro-Fuzzy ────────────────────────────────────────────────────

class TestFALCON:
    def test_regressor(self, small_regression_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.neurofuzzy import FALCONRegressor
        X_train, X_test, y_train, y_test = small_regression_data
        model = FALCONRegressor(n_rules=5, n_epochs=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)

    def test_classifier(self, small_classification_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.neurofuzzy import FALCONClassifier
        X_train, X_test, y_train, y_test = small_classification_data
        model = FALCONClassifier(n_rules=5, n_epochs=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (10,)
        assert proba.shape[0] == 10


class TestSOFNN:
    def test_regression(self, regression_data):
        from endgame.fuzzy.neurofuzzy import SOFNNRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = SOFNNRegressor(threshold_add=0.5, max_rules=20, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)
        assert model.n_rules_ > 0

    def test_partial_fit(self, regression_data):
        from endgame.fuzzy.neurofuzzy import SOFNNRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = SOFNNRegressor(threshold_add=0.5, max_rules=20, n_epochs=5, random_state=42)
        model.fit(X_train[:50], y_train[:50])
        model.partial_fit(X_train[50:], y_train[50:])
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestDENFIS:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.neurofuzzy import DENFISRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = DENFISRegressor(distance_threshold=0.5, max_rules=20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.neurofuzzy import DENFISClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = DENFISClassifier(distance_threshold=0.5, max_rules=20)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFNNTSK:
    def test_regressor(self, small_regression_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.neurofuzzy import FNNTSKRegressor
        X_train, X_test, y_train, y_test = small_regression_data
        model = FNNTSKRegressor(n_layers=2, n_rules_per_layer=3, n_epochs=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)

    def test_classifier(self, small_classification_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.neurofuzzy import FNNTSKClassifier
        X_train, X_test, y_train, y_test = small_classification_data
        model = FNNTSKClassifier(n_layers=2, n_rules_per_layer=3, n_epochs=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)


# ── Phase 5: Evolving Systems ───────────────────────────────────────────────

class TestEvolvingTSK:
    def test_basic(self, regression_data):
        from endgame.fuzzy.evolving import EvolvingTSK
        X_train, X_test, y_train, y_test = regression_data
        model = EvolvingTSK(radius=0.5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)
        assert model.n_rules_ > 0

    def test_partial_fit(self, regression_data):
        from endgame.fuzzy.evolving import EvolvingTSK
        X_train, X_test, y_train, y_test = regression_data
        model = EvolvingTSK(radius=0.5, random_state=42)
        model.fit(X_train[:50], y_train[:50])
        model.partial_fit(X_train[50:], y_train[50:])
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestEvolvingTSKPlus:
    def test_basic(self, regression_data):
        from endgame.fuzzy.evolving import EvolvingTSKPlus
        X_train, X_test, y_train, y_test = regression_data
        model = EvolvingTSKPlus(radius=0.5, merge_threshold=0.3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestPANFIS:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.evolving import PANFISRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = PANFISRegressor(max_rules=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.evolving import PANFISClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = PANFISClassifier(max_rules=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestAutoCloud:
    def test_basic(self, classification_data):
        from endgame.fuzzy.evolving import AutoCloudClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = AutoCloudClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFLEXFIS:
    def test_basic(self, regression_data):
        from endgame.fuzzy.evolving import FLEXFISRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FLEXFISRegressor(max_rules=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


# ── Phase 6: Type-2 Systems ─────────────────────────────────────────────────

class TestIT2FLS:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.type2 import IT2FLSRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = IT2FLSRegressor(n_rules=5, n_mfs=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.type2 import IT2FLSClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = IT2FLSClassifier(n_rules=5, n_mfs=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestIT2TSK:
    def test_basic(self, regression_data):
        from endgame.fuzzy.type2 import IT2TSKRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = IT2TSKRegressor(n_rules=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestIT2ANFIS:
    def test_basic(self, small_regression_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.type2 import IT2ANFISRegressor
        X_train, X_test, y_train, y_test = small_regression_data
        model = IT2ANFISRegressor(n_rules=3, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)


class TestGeneralType2:
    def test_basic(self, regression_data):
        from endgame.fuzzy.type2 import GeneralType2FLS
        X_train, X_test, y_train, y_test = regression_data
        model = GeneralType2FLS(n_rules=5, n_alpha_planes=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


# ── Phase 7: Fuzzy Ensembles ────────────────────────────────────────────────

class TestFuzzyRandomForest:
    def test_classifier(self, classification_data):
        from endgame.fuzzy.ensemble import FuzzyRandomForestClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyRandomForestClassifier(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)

    def test_regressor(self, regression_data):
        from endgame.fuzzy.ensemble import FuzzyRandomForestRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FuzzyRandomForestRegressor(n_estimators=10, max_depth=3, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFuzzyBoostedTrees:
    def test_classifier(self, classification_data):
        from endgame.fuzzy.ensemble import FuzzyBoostedTreesClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyBoostedTreesClassifier(
            n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)

    def test_regressor(self, regression_data):
        from endgame.fuzzy.ensemble import FuzzyBoostedTreesRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FuzzyBoostedTreesRegressor(
            n_estimators=10, learning_rate=0.1, max_depth=3, random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFuzzyBagging:
    def test_classifier(self, classification_data):
        from endgame.fuzzy.ensemble import FuzzyBaggingClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyBaggingClassifier(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_regressor(self, regression_data):
        from endgame.fuzzy.ensemble import FuzzyBaggingRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FuzzyBaggingRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestStackedFuzzy:
    def test_basic(self, classification_data):
        from endgame.fuzzy.ensemble import StackedFuzzySystem
        X_train, X_test, y_train, y_test = classification_data
        model = StackedFuzzySystem(
            base_estimators=[
                DecisionTreeClassifier(random_state=42),
                KNeighborsClassifier(),
            ],
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


# ── Phase 8: Rough, Modern, Extraction ──────────────────────────────────────

class TestFuzzyRoughNN:
    def test_basic(self, classification_data):
        from endgame.fuzzy.rough import FuzzyRoughNNClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyRoughNNClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        assert y_pred.shape == (20,)
        assert proba.shape == (20, 2)


class TestFuzzyRoughFeatureSelection:
    def test_basic(self, classification_data):
        from endgame.fuzzy.rough import FuzzyRoughFeatureSelector
        X_train, X_test, y_train, y_test = classification_data
        model = FuzzyRoughFeatureSelector(n_features=3)
        model.fit(X_train, y_train)
        X_selected = model.transform(X_test)
        assert X_selected.shape == (20, 3)
        assert len(model.selected_features_) == 3


class TestHTSK:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.modern import HTSKRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = HTSKRegressor(n_layers=2, n_rules_per_layer=5, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.modern import HTSKClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = HTSKClassifier(n_layers=2, n_rules_per_layer=5, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestMBGDRDA:
    def test_basic(self, regression_data):
        from endgame.fuzzy.modern import MBGDRDARegressor
        X_train, X_test, y_train, y_test = regression_data
        model = MBGDRDARegressor(n_rules=5, droprule_rate=0.3, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFCMRDpA:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.modern import FCMRDpARegressor
        X_train, X_test, y_train, y_test = regression_data
        model = FCMRDpARegressor(n_rules=5, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_classifier(self, classification_data):
        from endgame.fuzzy.modern import FCMRDpAClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FCMRDpAClassifier(n_rules=5, n_epochs=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestDifferentiableFuzzy:
    def test_regression(self, small_regression_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.modern import DifferentiableFuzzySystem
        X_train, X_test, y_train, y_test = small_regression_data
        model = DifferentiableFuzzySystem(n_rules=5, n_epochs=20, task='regression', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)

    def test_classification(self, small_classification_data):
        pytest.importorskip("torch")
        from endgame.fuzzy.modern import DifferentiableFuzzySystem
        X_train, X_test, y_train, y_test = small_classification_data
        model = DifferentiableFuzzySystem(n_rules=5, n_epochs=20, task='classification', random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (10,)


class TestTSKPlus:
    def test_regressor(self, regression_data):
        from endgame.fuzzy.modern import TSKPlusRegressor
        X_train, X_test, y_train, y_test = regression_data
        model = TSKPlusRegressor(n_rules=5, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_with_privileged(self, regression_data):
        from endgame.fuzzy.modern import TSKPlusRegressor
        X_train, X_test, y_train, y_test = regression_data
        X_priv = np.random.randn(100, 3)
        model = TSKPlusRegressor(n_rules=5, random_state=42)
        model.fit(X_train, y_train, X_privileged=X_priv)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestSEIT2FNN:
    def test_basic(self, classification_data):
        from endgame.fuzzy.modern import SEIT2FNNClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = SEIT2FNNClassifier(max_rules=20, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)


class TestFuzzyRuleExtractor:
    def test_classifier_extraction(self, classification_data):
        from endgame.fuzzy.extraction import FuzzyRuleExtractor
        from sklearn.ensemble import RandomForestClassifier
        X_train, X_test, y_train, y_test = classification_data
        rf = RandomForestClassifier(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        extractor = FuzzyRuleExtractor(n_rules=10, n_mfs=3)
        extractor.fit(rf, X_train, y_train)
        fidelity = extractor.score_fidelity(rf, X_test)
        assert 0 <= fidelity <= 1
        assert len(extractor.rules_) > 0
        rules_str = extractor.get_rules_str()
        assert "Rule 1" in rules_str

    def test_regressor_extraction(self, regression_data):
        from endgame.fuzzy.extraction import FuzzyRuleExtractor
        from sklearn.ensemble import RandomForestRegressor
        X_train, X_test, y_train, y_test = regression_data
        rf = RandomForestRegressor(n_estimators=10, random_state=42)
        rf.fit(X_train, y_train)
        extractor = FuzzyRuleExtractor(n_rules=10, n_mfs=3)
        extractor.fit(rf, X_train, y_train)
        y_pred = extractor.transform(X_test)
        assert y_pred.shape == (20,)


# ── Re-exports ───────────────────────────────────────────────────────────────

class TestReExports:
    def test_furia(self, classification_data):
        from endgame.fuzzy import FURIAClassifier
        X_train, X_test, y_train, y_test = classification_data
        model = FURIAClassifier(max_rules=10)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_fcm(self):
        from endgame.fuzzy import FuzzyCMeansClusterer
        X = np.random.randn(100, 3)
        model = FuzzyCMeansClusterer(n_clusters=3, random_state=42)
        labels = model.fit_predict(X)
        assert labels.shape == (100,)
        assert model.membership_.shape == (100, 3)


# ── Integration ──────────────────────────────────────────────────────────────

class TestLazyLoading:
    def test_endgame_fuzzy_access(self, classification_data):
        import endgame as eg
        X_train, X_test, y_train, y_test = classification_data
        model = eg.fuzzy.FuzzyKNNClassifier(n_neighbors=5)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        assert y_pred.shape == (20,)

    def test_submodule_access(self):
        import endgame as eg
        assert hasattr(eg.fuzzy, 'core')
        assert hasattr(eg.fuzzy, 'inference')
        assert hasattr(eg.fuzzy, 'ensemble')
