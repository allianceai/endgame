"""Tests for the anomaly detection module."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs
from sklearn.metrics import roc_auc_score


@pytest.fixture
def anomaly_data():
    """Generate test data with anomalies."""
    np.random.seed(42)
    X_normal, _ = make_blobs(n_samples=200, centers=2, n_features=5, random_state=42)
    X_anomaly = np.random.uniform(-10, 10, size=(20, 5))
    X = np.vstack([X_normal, X_anomaly])
    y_true = np.array([0] * 200 + [1] * 20)
    return X, y_true


class TestIsolationForestDetector:
    """Tests for IsolationForestDetector."""

    def test_fit_predict(self, anomaly_data):
        """Test basic fit and predict."""
        from endgame.anomaly import IsolationForestDetector

        X, y_true = anomaly_data
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        detector.fit(X)

        predictions = detector.predict(X)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})

    def test_decision_function(self, anomaly_data):
        """Test that decision function returns scores."""
        from endgame.anomaly import IsolationForestDetector

        X, y_true = anomaly_data
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        detector.fit(X)

        scores = detector.decision_function(X)
        assert scores.shape == (len(X),)
        # Higher scores should indicate anomalies
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.7

    def test_fit_predict_combined(self, anomaly_data):
        """Test fit_predict method."""
        from endgame.anomaly import IsolationForestDetector

        X, _ = anomaly_data
        detector = IsolationForestDetector(contamination=0.1, random_state=42)
        predictions = detector.fit_predict(X)
        assert predictions.shape == (len(X),)


class TestExtendedIsolationForest:
    """Tests for ExtendedIsolationForest."""

    def test_fit_predict(self, anomaly_data):
        """Test basic fit and predict."""
        from endgame.anomaly import ExtendedIsolationForest

        X, y_true = anomaly_data
        detector = ExtendedIsolationForest(
            contamination=0.1, n_estimators=50, random_state=42
        )
        detector.fit(X)

        predictions = detector.predict(X)
        assert predictions.shape == (len(X),)
        # treeple's ExtendedIsolationForest follows sklearn convention: -1=outlier, 1=inlier
        assert set(predictions).issubset({-1, 1})

    def test_decision_function(self, anomaly_data):
        """Test decision function quality."""
        from endgame.anomaly import ExtendedIsolationForest

        X, y_true = anomaly_data
        detector = ExtendedIsolationForest(
            contamination=0.1, n_estimators=100, random_state=42
        )
        detector.fit(X)

        scores = detector.decision_function(X)
        # sklearn convention: lower (more negative) scores = more anomalous,
        # so negate scores for AUC where y_true=1 means anomaly
        auc = roc_auc_score(y_true, -scores)
        assert auc > 0.7

    def test_feature_combinations(self, anomaly_data):
        """Test different feature_combinations settings."""
        from endgame.anomaly import ExtendedIsolationForest

        X, _ = anomaly_data

        # feature_combinations=None uses default oblique splits
        det_default = ExtendedIsolationForest(
            feature_combinations=None, n_estimators=50, random_state=42
        )
        det_default.fit(X)
        assert det_default.feature_combinations is None

        # Explicit feature_combinations value
        det_fc2 = ExtendedIsolationForest(
            feature_combinations=2, n_estimators=50, random_state=42
        )
        det_fc2.fit(X)
        assert det_fc2.feature_combinations == 2


class TestLocalOutlierFactorDetector:
    """Tests for LocalOutlierFactorDetector."""

    def test_fit_predict(self, anomaly_data):
        """Test basic fit and predict."""
        from endgame.anomaly import LocalOutlierFactorDetector

        X, y_true = anomaly_data
        detector = LocalOutlierFactorDetector(contamination=0.1)
        detector.fit(X)

        predictions = detector.predict(X)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})

    def test_decision_function(self, anomaly_data):
        """Test decision function quality."""
        from endgame.anomaly import LocalOutlierFactorDetector

        X, y_true = anomaly_data
        detector = LocalOutlierFactorDetector(contamination=0.1)
        detector.fit(X)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.7

    def test_fit_predict_transductive(self, anomaly_data):
        """Test fit_predict (transductive mode)."""
        from endgame.anomaly import LocalOutlierFactorDetector

        X, _ = anomaly_data
        detector = LocalOutlierFactorDetector(contamination=0.1)
        predictions = detector.fit_predict(X)
        assert predictions.shape == (len(X),)


class TestPyODDetector:
    """Tests for PyODDetector wrapper."""

    def test_ecod(self, anomaly_data):
        """Test ECOD algorithm."""
        from endgame.anomaly import PyODDetector

        X, y_true = anomaly_data
        detector = PyODDetector(algorithm="ECOD", contamination=0.1)
        detector.fit(X)

        predictions = detector.predict(X)
        assert predictions.shape == (len(X),)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.6

    def test_copod(self, anomaly_data):
        """Test COPOD algorithm."""
        from endgame.anomaly import PyODDetector

        X, y_true = anomaly_data
        detector = PyODDetector(algorithm="COPOD", contamination=0.1)
        detector.fit(X)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.6

    def test_iforest(self, anomaly_data):
        """Test PyOD's IForest."""
        from endgame.anomaly import PyODDetector

        X, y_true = anomaly_data
        detector = PyODDetector(algorithm="IForest", contamination=0.1)
        detector.fit(X)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.7

    def test_hbos(self, anomaly_data):
        """Test HBOS algorithm."""
        from endgame.anomaly import PyODDetector

        X, y_true = anomaly_data
        detector = PyODDetector(algorithm="HBOS", contamination=0.1)
        detector.fit(X)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.6

    def test_available_algorithms(self):
        """Test that PYOD_ALGORITHMS is accessible."""
        from endgame.anomaly import PYOD_ALGORITHMS

        assert len(PYOD_ALGORITHMS) > 30
        assert "ECOD" in PYOD_ALGORITHMS
        assert "IForest" in PYOD_ALGORITHMS
        assert "LOF" in PYOD_ALGORITHMS

    def test_invalid_algorithm(self, anomaly_data):
        """Test error on invalid algorithm."""
        from endgame.anomaly import PyODDetector

        X, _ = anomaly_data
        detector = PyODDetector(algorithm="InvalidAlgo")

        with pytest.raises(ValueError, match="Unknown algorithm"):
            detector.fit(X)

    def test_predict_proba(self, anomaly_data):
        """Test predict_proba method."""
        from endgame.anomaly import PyODDetector

        X, _ = anomaly_data
        detector = PyODDetector(algorithm="ECOD", contamination=0.1)
        detector.fit(X)

        proba = detector.predict_proba(X)
        assert proba.shape == (len(X), 2)
        assert np.allclose(proba.sum(axis=1), 1.0)


class TestDetectorEnsemble:
    """Tests for create_detector_ensemble."""

    def test_default_ensemble(self, anomaly_data):
        """Test creating default ensemble."""
        from endgame.anomaly import create_detector_ensemble

        X, y_true = anomaly_data
        detectors = create_detector_ensemble(contamination=0.1)

        assert len(detectors) == 6  # Default is 6 detectors

        # Fit all and combine scores
        all_scores = []
        for det in detectors:
            det.fit(X)
            all_scores.append(det.decision_function(X))

        # Ensemble should perform well
        ensemble_scores = np.mean(all_scores, axis=0)
        auc = roc_auc_score(y_true, ensemble_scores)
        assert auc > 0.7

    def test_custom_ensemble(self, anomaly_data):
        """Test creating custom ensemble."""
        from endgame.anomaly import create_detector_ensemble

        X, _ = anomaly_data
        detectors = create_detector_ensemble(
            algorithms=["ECOD", "COPOD"],
            contamination=0.05,
        )

        assert len(detectors) == 2
        for det in detectors:
            assert det.contamination == 0.05


class TestGritBotDetector:
    """Tests for GritBotDetector."""

    def test_fit_predict(self, anomaly_data):
        """Test basic fit and predict."""
        from endgame.anomaly import GritBotDetector

        X, y_true = anomaly_data
        detector = GritBotDetector(
            max_conditions=2, filtering_level=50, contamination=0.1
        )
        detector.fit(X)

        predictions = detector.predict(X)
        assert predictions.shape == (len(X),)
        assert set(predictions).issubset({0, 1})

    def test_decision_function(self, anomaly_data):
        """Test decision function quality."""
        from endgame.anomaly import GritBotDetector

        X, y_true = anomaly_data
        detector = GritBotDetector(
            max_conditions=2, filtering_level=30, contamination=0.1
        )
        detector.fit(X)

        scores = detector.decision_function(X)
        auc = roc_auc_score(y_true, scores)
        assert auc > 0.6  # GritBot is context-based, may not be best for simple data

    def test_anomaly_report(self, anomaly_data):
        """Test that anomaly report is generated."""
        from endgame.anomaly import GritBotDetector

        X, _ = anomaly_data
        detector = GritBotDetector(filtering_level=30)
        detector.fit(X)

        report = detector.get_anomaly_report(max_anomalies=5)
        assert isinstance(report, str)
        assert "GritBot" in report or "No anomalies" in report

    def test_filtering_level(self, anomaly_data):
        """Test different filtering levels."""
        from endgame.anomaly import GritBotDetector

        X, _ = anomaly_data

        # Low filtering (more sensitive)
        det_low = GritBotDetector(filtering_level=10)
        det_low.fit(X)

        # High filtering (more conservative)
        det_high = GritBotDetector(filtering_level=90)
        det_high.fit(X)

        # Both should work
        assert det_low._get_minabnorm() < det_high._get_minabnorm()


class TestModuleImports:
    """Test module-level imports work correctly."""

    def test_import_from_endgame(self):
        """Test imports from main endgame package."""
        import endgame as eg

        assert hasattr(eg, "anomaly")
        assert hasattr(eg.anomaly, "IsolationForestDetector")
        assert hasattr(eg.anomaly, "ExtendedIsolationForest")
        assert hasattr(eg.anomaly, "LocalOutlierFactorDetector")
        assert hasattr(eg.anomaly, "GritBotDetector")
        assert hasattr(eg.anomaly, "PyODDetector")
        assert hasattr(eg.anomaly, "PYOD_ALGORITHMS")
        assert hasattr(eg.anomaly, "create_detector_ensemble")

    def test_direct_imports(self):
        """Test direct imports from anomaly module."""
        from endgame.anomaly import (
            IsolationForestDetector,
            ExtendedIsolationForest,
            LocalOutlierFactorDetector,
            GritBotDetector,
            PyODDetector,
            PYOD_ALGORITHMS,
            create_detector_ensemble,
        )

        assert IsolationForestDetector is not None
        assert ExtendedIsolationForest is not None
        assert LocalOutlierFactorDetector is not None
        assert GritBotDetector is not None
        assert PyODDetector is not None
        assert isinstance(PYOD_ALGORITHMS, dict)
        assert callable(create_detector_ensemble)
