"""Tests for ensemble module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class TestHillClimbingEnsemble:
    """Tests for HillClimbingEnsemble."""

    def test_fit_optimize_weights(self):
        """Test weight optimization via hill climbing."""
        from endgame.ensemble import HillClimbingEnsemble

        # Create synthetic predictions
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)

        # Three "models" with different quality
        pred1 = y_true + rng.randn(100) * 0.3
        pred2 = y_true + rng.randn(100) * 0.5
        pred3 = y_true + rng.randn(100) * 0.7

        predictions = [pred1, pred2, pred3]

        ensemble = HillClimbingEnsemble(
            metric="auc",
            n_iterations=50,
            random_state=42,
        )
        ensemble.fit(predictions, y_true)

        assert len(ensemble.weights_) > 0
        assert abs(sum(ensemble.weights_.values()) - 1.0) < 0.01  # Weights sum to 1

    def test_predict(self):
        """Test ensemble prediction."""
        from endgame.ensemble import HillClimbingEnsemble

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)

        pred1 = y_true + rng.randn(100) * 0.3
        pred2 = y_true + rng.randn(100) * 0.5

        ensemble = HillClimbingEnsemble(metric="auc", random_state=42)
        ensemble.fit([pred1, pred2], y_true)

        combined = ensemble.predict([pred1, pred2])
        assert len(combined) == 100


class TestStackingEnsemble:
    """Tests for StackingEnsemble."""

    def test_fit_predict(self):
        """Test stacking ensemble."""
        from endgame.ensemble import StackingEnsemble
        from sklearn.linear_model import LogisticRegression

        X, y = make_classification(n_samples=100, n_features=5, random_state=42)

        # Mock OOF predictions
        rng = np.random.RandomState(42)
        oof_preds = [
            rng.rand(100),
            rng.rand(100),
        ]

        stacker = StackingEnsemble(
            base_estimators=[
                LogisticRegression(max_iter=200),
                DecisionTreeClassifier(max_depth=3),
            ],
            meta_estimator=LogisticRegression(max_iter=200),
            passthrough=False,
            random_state=42,
        )
        stacker.fit(X, y)

        result = stacker.predict(X)
        assert len(result) == 100


class TestBlenders:
    """Tests for blending methods."""

    def test_optimized_blender(self):
        """Test Optuna-optimized blender."""
        pytest.importorskip("optuna")
        from endgame.ensemble import OptimizedBlender

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)

        predictions = [
            y_true + rng.randn(100) * 0.3,
            y_true + rng.randn(100) * 0.5,
        ]

        blender = OptimizedBlender(
            metric="roc_auc",
            n_trials=10,
            random_state=42,
        )
        result = blender.fit(predictions, y_true)

        assert len(result.weights_) == 2
        assert result.best_score_ > 0.5

    def test_rank_average_blender(self):
        """Test rank averaging."""
        from endgame.ensemble import RankAverageBlender

        predictions = [
            np.array([0.1, 0.5, 0.9]),
            np.array([0.2, 0.4, 0.8]),
        ]

        blender = RankAverageBlender()
        result = blender.blend(predictions)

        assert len(result) == 3
        # Result should be rank-averaged
        assert result[0] < result[1] < result[2]

    def test_power_blender(self):
        """Test power averaging."""
        from endgame.ensemble import PowerBlender

        predictions = [
            np.array([0.1, 0.5, 0.9]),
            np.array([0.2, 0.4, 0.8]),
        ]

        blender = PowerBlender(scores=[0.85, 0.9], power=2.0)
        blender.fit()
        result = blender.predict(predictions)

        assert len(result) == 3


class TestThresholdOptimizer:
    """Tests for threshold optimization."""

    def test_optimize_f1(self):
        """Test F1 threshold optimization."""
        from endgame.ensemble import ThresholdOptimizer

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_proba = y_true * 0.6 + rng.rand(100) * 0.4

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_true, y_proba)

        assert 0 < optimizer.threshold_ < 1
        assert optimizer.best_score_ > 0.5

    def test_apply_threshold(self):
        """Test applying optimized threshold."""
        from endgame.ensemble import ThresholdOptimizer

        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, 100)
        y_proba = y_true * 0.6 + rng.rand(100) * 0.4

        optimizer = ThresholdOptimizer(metric="f1")
        optimizer.fit(y_proba, y_true)

        predictions = optimizer.predict(y_proba)
        assert set(predictions).issubset({0, 1})


# ===================================================================
# Classic Ensemble Tests
# ===================================================================

@pytest.fixture
def clf_data():
    X, y = make_classification(n_samples=200, n_features=10, random_state=42)
    return X, y


@pytest.fixture
def reg_data():
    X, y = make_regression(n_samples=200, n_features=10, random_state=42)
    return X, y


class TestVotingClassifier:
    def test_hard_voting(self, clf_data):
        from endgame.ensemble import VotingClassifier
        X, y = clf_data
        vc = VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            voting="hard",
        )
        vc.fit(X, y)
        preds = vc.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})

    def test_soft_voting(self, clf_data):
        from endgame.ensemble import VotingClassifier
        X, y = clf_data
        vc = VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            voting="soft",
        )
        vc.fit(X, y)
        proba = vc.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_weighted_voting(self, clf_data):
        from endgame.ensemble import VotingClassifier
        X, y = clf_data
        vc = VotingClassifier(
            estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            weights=[2, 1],
            voting="soft",
        )
        vc.fit(X, y)
        preds = vc.predict(X)
        assert len(preds) == 200

    def test_named_estimators(self, clf_data):
        from endgame.ensemble import VotingClassifier
        X, y = clf_data
        vc = VotingClassifier(
            estimators=[("dt", DecisionTreeClassifier()), ("lr", LogisticRegression(max_iter=200))],
        )
        vc.fit(X, y)
        assert "dt" in vc.named_estimators
        assert "lr" in vc.named_estimators

    def test_transform(self, clf_data):
        from endgame.ensemble import VotingClassifier
        X, y = clf_data
        vc = VotingClassifier(
            estimators=[("dt", DecisionTreeClassifier()), ("lr", LogisticRegression(max_iter=200))],
            voting="soft",
        )
        vc.fit(X, y)
        t = vc.transform(X)
        assert t.shape[0] == 200


class TestVotingRegressor:
    def test_basic(self, reg_data):
        from endgame.ensemble import VotingRegressor
        X, y = reg_data
        vr = VotingRegressor(
            estimators=[
                ("dt", DecisionTreeRegressor(max_depth=3)),
                ("lr", Ridge()),
            ],
        )
        vr.fit(X, y)
        preds = vr.predict(X)
        assert preds.shape == (200,)

    def test_weighted(self, reg_data):
        from endgame.ensemble import VotingRegressor
        X, y = reg_data
        vr = VotingRegressor(
            estimators=[("dt", DecisionTreeRegressor()), ("lr", Ridge())],
            weights=[1, 3],
        )
        vr.fit(X, y)
        preds = vr.predict(X)
        assert preds.shape == (200,)


class TestBaggingClassifier:
    def test_basic(self, clf_data):
        from endgame.ensemble import BaggingClassifier
        X, y = clf_data
        bag = BaggingClassifier(n_estimators=5, random_state=42)
        bag.fit(X, y)
        preds = bag.predict(X)
        assert preds.shape == (200,)

    def test_proba(self, clf_data):
        from endgame.ensemble import BaggingClassifier
        X, y = clf_data
        bag = BaggingClassifier(n_estimators=5, random_state=42)
        bag.fit(X, y)
        proba = bag.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_oob_score(self, clf_data):
        from endgame.ensemble import BaggingClassifier
        X, y = clf_data
        bag = BaggingClassifier(n_estimators=10, oob_score=True, random_state=42)
        bag.fit(X, y)
        assert hasattr(bag, "oob_score_")
        assert 0 <= bag.oob_score_ <= 1

    def test_feature_subsampling(self, clf_data):
        from endgame.ensemble import BaggingClassifier
        X, y = clf_data
        bag = BaggingClassifier(
            n_estimators=5, max_features=0.5, random_state=42
        )
        bag.fit(X, y)
        preds = bag.predict(X)
        assert preds.shape == (200,)

    def test_feature_importances(self, clf_data):
        from endgame.ensemble import BaggingClassifier
        X, y = clf_data
        bag = BaggingClassifier(n_estimators=5, random_state=42)
        bag.fit(X, y)
        fi = bag.feature_importances_
        assert len(fi) == 10


class TestBaggingRegressor:
    def test_basic(self, reg_data):
        from endgame.ensemble import BaggingRegressor
        X, y = reg_data
        bag = BaggingRegressor(n_estimators=5, random_state=42)
        bag.fit(X, y)
        preds = bag.predict(X)
        assert preds.shape == (200,)

    def test_oob_score(self, reg_data):
        from endgame.ensemble import BaggingRegressor
        X, y = reg_data
        bag = BaggingRegressor(n_estimators=10, oob_score=True, random_state=42)
        bag.fit(X, y)
        assert hasattr(bag, "oob_score_")


class TestAdaBoostClassifier:
    def test_samme(self, clf_data):
        from endgame.ensemble import AdaBoostClassifier
        X, y = clf_data
        ada = AdaBoostClassifier(
            n_estimators=10, algorithm="SAMME", random_state=42
        )
        ada.fit(X, y)
        preds = ada.predict(X)
        assert preds.shape == (200,)
        assert len(ada.estimators_) <= 10

    def test_sammer(self, clf_data):
        from endgame.ensemble import AdaBoostClassifier
        X, y = clf_data
        ada = AdaBoostClassifier(
            n_estimators=10, algorithm="SAMME.R", random_state=42
        )
        ada.fit(X, y)
        proba = ada.predict_proba(X)
        assert proba.shape == (200, 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)

    def test_feature_importances(self, clf_data):
        from endgame.ensemble import AdaBoostClassifier
        X, y = clf_data
        ada = AdaBoostClassifier(n_estimators=10, random_state=42)
        ada.fit(X, y)
        fi = ada.feature_importances_
        assert len(fi) == 10

    def test_learning_rate(self, clf_data):
        from endgame.ensemble import AdaBoostClassifier
        X, y = clf_data
        ada = AdaBoostClassifier(
            n_estimators=20, learning_rate=0.5, random_state=42
        )
        ada.fit(X, y)
        preds = ada.predict(X)
        assert preds.shape == (200,)


class TestAdaBoostRegressor:
    def test_basic(self, reg_data):
        from endgame.ensemble import AdaBoostRegressor
        X, y = reg_data
        ada = AdaBoostRegressor(n_estimators=10, random_state=42)
        ada.fit(X, y)
        preds = ada.predict(X)
        assert preds.shape == (200,)

    def test_loss_functions(self, reg_data):
        from endgame.ensemble import AdaBoostRegressor
        X, y = reg_data
        for loss in ["linear", "square", "exponential"]:
            ada = AdaBoostRegressor(n_estimators=5, loss=loss, random_state=42)
            ada.fit(X, y)
            preds = ada.predict(X)
            assert preds.shape == (200,)


# ===================================================================
# SOTA Ensemble Tests
# ===================================================================

class TestSuperLearner:
    def test_classification(self, clf_data):
        from endgame.ensemble import SuperLearner
        X, y = clf_data
        sl = SuperLearner(
            base_estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            meta_learner="nnls",
            cv=3,
            random_state=42,
        )
        sl.fit(X, y)
        preds = sl.predict(X)
        assert preds.shape == (200,)
        assert set(preds).issubset({0, 1})
        assert len(sl.coef_) > 0
        assert "dt" in sl.cv_scores_
        assert "lr" in sl.cv_scores_

    def test_regression(self, reg_data):
        from endgame.ensemble import SuperLearner
        X, y = reg_data
        sl = SuperLearner(
            base_estimators=[
                ("dt", DecisionTreeRegressor(max_depth=3, random_state=42)),
                ("lr", Ridge()),
            ],
            meta_learner="nnls",
            cv=3,
            random_state=42,
        )
        sl.fit(X, y)
        preds = sl.predict(X)
        assert preds.shape == (200,)

    def test_meta_learner_best(self, clf_data):
        from endgame.ensemble import SuperLearner
        X, y = clf_data
        sl = SuperLearner(
            base_estimators=[
                ("dt", DecisionTreeClassifier(max_depth=3, random_state=42)),
                ("lr", LogisticRegression(max_iter=200)),
            ],
            meta_learner="best",
            cv=3,
            random_state=42,
        )
        sl.fit(X, y)
        assert sl.coef_.sum() == 1.0  # exactly one model selected

    def test_named_estimators(self, clf_data):
        from endgame.ensemble import SuperLearner
        X, y = clf_data
        sl = SuperLearner(
            base_estimators=[("dt", DecisionTreeClassifier()), ("lr", LogisticRegression(max_iter=200))],
            cv=3,
        )
        sl.fit(X, y)
        assert "dt" in sl.named_estimators


class TestBayesianModelAveraging:
    def test_classification(self, clf_data):
        from endgame.ensemble import BayesianModelAveraging
        X, y = clf_data
        dt = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, y)
        lr = LogisticRegression(max_iter=200).fit(X, y)
        rf = RandomForestClassifier(n_estimators=5, random_state=42).fit(X, y)

        bma = BayesianModelAveraging(criterion="bic")
        bma.fit(estimators=[dt, lr, rf], X_val=X, y_val=y)

        assert bma.weights_.shape == (3,)
        assert np.isclose(bma.weights_.sum(), 1.0)

        preds = bma.predict(X)
        assert preds.shape == (200,)

        proba = bma.predict_proba(X)
        assert proba.shape == (200, 2)

    def test_regression(self, reg_data):
        from endgame.ensemble import BayesianModelAveraging
        X, y = reg_data
        dt = DecisionTreeRegressor(max_depth=3, random_state=42).fit(X, y)
        lr = Ridge().fit(X, y)

        bma = BayesianModelAveraging(criterion="aic", task="regression")
        bma.fit(estimators=[dt, lr], X_val=X, y_val=y)

        preds = bma.predict(X)
        assert preds.shape == (200,)
        assert np.isclose(bma.weights_.sum(), 1.0)


class TestNegativeCorrelationEnsemble:
    def test_basic(self, reg_data):
        from endgame.ensemble import NegativeCorrelationEnsemble
        X, y = reg_data
        nce = NegativeCorrelationEnsemble(
            base_estimators=[Ridge(), LinearRegression(), Ridge(alpha=10)],
            lambda_ncl=0.5,
            n_iterations=3,
            random_state=42,
        )
        nce.fit(X, y)
        preds = nce.predict(X)
        assert preds.shape == (200,)
        assert nce.diversity_ >= 0

    def test_lambda_zero_is_standard(self, reg_data):
        from endgame.ensemble import NegativeCorrelationEnsemble
        X, y = reg_data
        nce = NegativeCorrelationEnsemble(
            base_estimators=[Ridge()],
            lambda_ncl=0.0,
            n_iterations=1,
        )
        nce.fit(X, y)
        preds = nce.predict(X)
        assert preds.shape == (200,)


class TestSnapshotEnsemble:
    def test_basic(self, clf_data):
        from endgame.ensemble import SnapshotEnsemble
        from sklearn.neural_network import MLPClassifier
        X, y = clf_data
        snap = SnapshotEnsemble(
            base_estimator=MLPClassifier(hidden_layer_sizes=(20,), max_iter=1, random_state=42),
            n_snapshots=2,
            epochs_per_cycle=5,
            initial_lr=0.01,
        )
        snap.fit(X, y)
        assert len(snap.snapshots_) == 2
        preds = snap.predict(X)
        assert preds.shape == (200,)
        proba = snap.predict_proba(X)
        assert proba.shape == (200, 2)


class TestCascadeEnsemble:
    def test_basic(self, clf_data):
        from endgame.ensemble import CascadeEnsemble
        X, y = clf_data
        cascade = CascadeEnsemble(
            stages=[
                [LogisticRegression(max_iter=200), DecisionTreeClassifier(max_depth=2)],
                [RandomForestClassifier(n_estimators=5, random_state=42)],
            ],
            confidence_threshold=0.95,
            cv=3,
            random_state=42,
        )
        cascade.fit(X, y)
        preds = cascade.predict(X)
        assert preds.shape == (200,)
        assert cascade.n_stages_ == 2
        assert len(cascade.stage_scores_) == 2

    def test_proba(self, clf_data):
        from endgame.ensemble import CascadeEnsemble
        X, y = clf_data
        cascade = CascadeEnsemble(
            stages=[
                [DecisionTreeClassifier(max_depth=3)],
                [DecisionTreeClassifier(max_depth=5)],
            ],
            cv=3,
            random_state=42,
        )
        cascade.fit(X, y)
        proba = cascade.predict_proba(X)
        assert proba.shape == (200, 2)
