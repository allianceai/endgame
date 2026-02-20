"""Tests for the endgame.fairness module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

class TestDemographicParity:
    def test_perfect_parity(self):
        from endgame.fairness import demographic_parity

        # Both groups predict exactly 50% positive
        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B", "A", "B"])

        result = demographic_parity(y_true, y_pred, sensitive)
        assert "group_rates" in result
        assert "max_disparity" in result
        # A: [1,0,1] → 2/3, B: [0,0,0] → wait, let me just check equal rates
        # Use a setup that guarantees equal rates per group
        y_pred2 = np.array([1, 0, 1, 0])
        sensitive2 = np.array(["A", "A", "B", "B"])
        result2 = demographic_parity(y_true[:4], y_pred2, sensitive2)
        assert abs(result2["max_disparity"]) < 1e-10

    def test_full_disparity(self):
        from endgame.fairness import demographic_parity

        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = demographic_parity(y_true, y_pred, sensitive)
        assert result["max_disparity"] == pytest.approx(1.0)
        assert result["privileged_group"] == "A"
        assert result["unprivileged_group"] == "B"

    def test_validation_error(self):
        from endgame.fairness import demographic_parity

        with pytest.raises(ValueError, match="samples"):
            demographic_parity(
                np.array([1, 0]),
                np.array([1, 0, 1]),
                np.array([0, 1]),
            )

    def test_empty_arrays(self):
        from endgame.fairness import demographic_parity

        with pytest.raises(ValueError, match="empty"):
            demographic_parity(
                np.array([]),
                np.array([]),
                np.array([]),
            )


class TestEqualizedOdds:
    def test_perfect_model(self):
        from endgame.fairness import equalized_odds

        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B", "A", "B"])

        result = equalized_odds(y_true, y_pred, sensitive)
        assert result["tpr_disparity"] == pytest.approx(0.0)
        assert result["fpr_disparity"] == pytest.approx(0.0)
        assert result["satisfied"] is True

    def test_unequal_tpr(self):
        from endgame.fairness import equalized_odds

        # Group A: TPR=1.0 (predicts all positives correctly)
        # Group B: TPR=0.0 (misses all positives)
        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        result = equalized_odds(y_true, y_pred, sensitive)
        assert result["tpr_disparity"] == pytest.approx(1.0)
        assert result["satisfied"] is False

    def test_output_keys(self):
        from endgame.fairness import equalized_odds

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        result = equalized_odds(y_true, y_pred, sensitive)
        expected_keys = {"group_tpr", "group_fpr", "tpr_disparity",
                         "fpr_disparity", "max_disparity", "satisfied"}
        assert expected_keys.issubset(result.keys())


class TestDisparateImpact:
    def test_equal_rates(self):
        from endgame.fairness import disparate_impact

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        result = disparate_impact(y_true, y_pred, sensitive)
        assert result["disparate_impact_ratio"] == pytest.approx(1.0)
        assert result["four_fifths_satisfied"] is True

    def test_below_four_fifths(self):
        from endgame.fairness import disparate_impact

        # Group A: 100% positive, Group B: 25% positive → ratio = 0.25
        y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        sensitive = np.array(["A", "A", "A", "A", "B", "B", "B", "B"])

        result = disparate_impact(y_true, y_pred, sensitive)
        assert result["disparate_impact_ratio"] < 0.8
        assert result["four_fifths_satisfied"] is False


class TestCalibrationByGroup:
    def test_basic(self):
        from endgame.fairness import calibration_by_group

        y_true = np.array([1, 0, 1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0, 1, 0])
        sensitive = np.array(["A", "A", "A", "B", "B", "B"])

        result = calibration_by_group(y_true, y_pred, sensitive)
        assert "group_accuracy" in result
        # Perfect predictions → all groups have 1.0 accuracy
        for acc in result["group_accuracy"].values():
            assert acc == pytest.approx(1.0)

    def test_with_probabilities(self):
        from endgame.fairness import calibration_by_group

        y_true = np.array([1, 0, 1, 0])
        y_pred = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])
        y_proba = np.array([0.9, 0.1, 0.8, 0.2])

        result = calibration_by_group(
            y_true, y_pred, sensitive, y_proba=y_proba,
        )
        assert "group_brier_score" in result
        assert "group_ece" in result


# ------------------------------------------------------------------
# Mitigation
# ------------------------------------------------------------------

class TestReweighingPreprocessor:
    def test_fit_transform(self):
        from endgame.fairness import ReweighingPreprocessor

        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 0, 1, 0])
        sensitive = np.array(["A", "A", "B", "B"])

        rw = ReweighingPreprocessor()
        weights = rw.fit_transform(X, y, sensitive_attr=sensitive)

        assert weights.shape == (4,)
        assert np.all(weights > 0)


class TestCalibratedEqOdds:
    def test_fit_predict(self):
        from endgame.fairness import CalibratedEqOdds

        np.random.seed(42)
        n = 200
        X = np.random.randn(n, 5)
        y = np.random.randint(0, 2, size=n)
        sensitive = np.random.choice(["A", "B"], size=n)

        # Need a fitted estimator with predict_proba
        model = LogisticRegression(random_state=42, max_iter=500)
        model.fit(X, y)

        cal = CalibratedEqOdds(estimator=model)
        cal.fit(X, y, sensitive_attr=sensitive)

        # Predict should return adjusted labels
        y_adjusted = cal.predict(X, sensitive_attr=sensitive)
        assert y_adjusted.shape == (n,)
        assert set(np.unique(y_adjusted)).issubset({0, 1})


# ------------------------------------------------------------------
# Import smoke test
# ------------------------------------------------------------------

class TestImports:
    def test_all_exports(self):
        from endgame.fairness import (
            demographic_parity,
            equalized_odds,
            disparate_impact,
            calibration_by_group,
            ReweighingPreprocessor,
            ExponentiatedGradient,
            CalibratedEqOdds,
            FairnessReport,
        )
        assert callable(demographic_parity)
        assert callable(equalized_odds)

    def test_import_via_endgame(self):
        import endgame as eg
        assert hasattr(eg, "fairness")
        assert hasattr(eg.fairness, "demographic_parity")
