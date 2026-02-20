"""Tests for core module."""

import numpy as np
import pytest

from endgame.core.config import get_preset, LGBM_ENDGAME_DEFAULTS
from endgame.core.types import AdversarialValidationResult, OOFResult


class TestConfig:
    """Tests for configuration module."""

    def test_get_preset_lgbm(self):
        """Test getting LightGBM preset."""
        params = get_preset("lightgbm", "endgame")
        assert "learning_rate" in params
        assert params["learning_rate"] == 0.01
        assert params["n_estimators"] == 10000

    def test_get_preset_xgb(self):
        """Test getting XGBoost preset."""
        params = get_preset("xgboost", "endgame")
        assert "learning_rate" in params
        assert "max_depth" in params

    def test_get_preset_catboost(self):
        """Test getting CatBoost preset."""
        params = get_preset("catboost", "endgame")
        assert "iterations" in params
        assert "depth" in params

    def test_get_preset_fast(self):
        """Test getting fast preset."""
        params = get_preset("lightgbm", "fast")
        assert params["learning_rate"] > LGBM_ENDGAME_DEFAULTS["learning_rate"]

    def test_get_preset_unknown_backend(self):
        """Test error on unknown backend."""
        with pytest.raises(ValueError, match="Unknown backend"):
            get_preset("unknown", "endgame")

    def test_get_preset_unknown_preset(self):
        """Test error on unknown preset."""
        with pytest.raises(ValueError, match="Unknown preset"):
            get_preset("lightgbm", "unknown")


class TestTypes:
    """Tests for type definitions."""

    def test_adversarial_validation_result(self):
        """Test AdversarialValidationResult creation."""
        importances = {"feat1": 0.3, "feat2": 0.5, "feat3": 0.2}
        result = AdversarialValidationResult.from_auc(0.75, importances)

        assert result.auc_score == 0.75
        assert result.drift_severity == "severe"
        assert result.drifted_features[0] == "feat2"  # Highest importance first

    def test_adversarial_validation_severity(self):
        """Test drift severity classification."""
        importances = {"feat1": 0.5}

        # No drift
        result = AdversarialValidationResult.from_auc(0.50, importances)
        assert result.drift_severity == "none"

        # Mild drift
        result = AdversarialValidationResult.from_auc(0.60, importances)
        assert result.drift_severity == "mild"

        # Moderate drift
        result = AdversarialValidationResult.from_auc(0.70, importances)
        assert result.drift_severity == "moderate"

        # Severe drift
        result = AdversarialValidationResult.from_auc(0.85, importances)
        assert result.drift_severity == "severe"

    def test_oof_result(self):
        """Test OOFResult creation."""
        result = OOFResult(
            oof_predictions=np.array([0.1, 0.9, 0.5]),
            fold_scores=[0.85, 0.87, 0.86],
            mean_score=0.86,
            std_score=0.01,
        )

        assert len(result.oof_predictions) == 3
        assert result.mean_score == 0.86
        assert "0.86" in repr(result)


class TestPolarsOps:
    """Tests for Polars operations."""

    def test_to_lazyframe_numpy(self):
        """Test converting numpy array to LazyFrame."""
        pytest.importorskip("polars")
        from endgame.core.polars_ops import to_lazyframe

        X = np.array([[1, 2], [3, 4]])
        lf = to_lazyframe(X)

        assert lf is not None
        df = lf.collect()
        assert df.shape == (2, 2)

    def test_to_lazyframe_with_columns(self):
        """Test specifying column names."""
        pytest.importorskip("polars")
        from endgame.core.polars_ops import to_lazyframe

        X = np.array([[1, 2], [3, 4]])
        lf = to_lazyframe(X, column_names=["a", "b"])

        df = lf.collect()
        assert list(df.columns) == ["a", "b"]

    def test_from_lazyframe(self):
        """Test converting LazyFrame to different formats."""
        pl = pytest.importorskip("polars")
        from endgame.core.polars_ops import from_lazyframe

        lf = pl.DataFrame({"a": [1, 2], "b": [3, 4]}).lazy()

        # To polars
        df = from_lazyframe(lf, "polars")
        assert isinstance(df, pl.DataFrame)

        # To numpy
        arr = from_lazyframe(lf, "numpy")
        assert isinstance(arr, np.ndarray)

    def test_infer_categorical_columns(self):
        """Test categorical column inference."""
        pl = pytest.importorskip("polars")
        from endgame.core.polars_ops import infer_categorical_columns

        df = pl.DataFrame({
            "cat": ["a", "b", "a"],
            "num": [1.0, 2.0, 3.0],
            "low_card": [1, 2, 1],
        })

        cat_cols = infer_categorical_columns(df, max_cardinality=5)
        assert "cat" in cat_cols
        assert "low_card" in cat_cols
        assert "num" not in cat_cols

    def test_infer_numeric_columns(self):
        """Test numeric column inference."""
        pl = pytest.importorskip("polars")
        from endgame.core.polars_ops import infer_numeric_columns

        df = pl.DataFrame({
            "cat": ["a", "b", "c"],
            "num": [1.0, 2.0, 3.0],
        })

        num_cols = infer_numeric_columns(df)
        assert "num" in num_cols
        assert "cat" not in num_cols
