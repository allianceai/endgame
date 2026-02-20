"""Tests for preprocessing module."""

import numpy as np
import pytest


class TestSafeTargetEncoder:
    """Tests for SafeTargetEncoder."""

    def test_fit_transform(self):
        """Test basic fit_transform."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import SafeTargetEncoder

        df = pl.DataFrame({
            "cat": ["a", "b", "a", "b", "a", "b"],
            "num": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })
        y = np.array([0, 1, 0, 1, 0, 1])

        encoder = SafeTargetEncoder(cols=["cat"], random_state=42)
        result = encoder.fit_transform(df, y)

        assert result is not None
        # Cat column should be encoded
        assert result["cat"].dtype in [pl.Float32, pl.Float64]

    def test_transform(self):
        """Test transform on new data."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import SafeTargetEncoder

        df_train = pl.DataFrame({"cat": ["a", "b", "a", "b"]})
        y = np.array([0, 1, 0, 1])

        df_test = pl.DataFrame({"cat": ["a", "b", "c"]})

        encoder = SafeTargetEncoder(cols=["cat"], handle_unknown="global_mean")
        encoder.fit(df_train, y)
        result = encoder.transform(df_test)

        assert len(result) == 3
        # Unknown "c" should get global mean
        assert not np.isnan(result["cat"][2])

    def test_smoothing(self):
        """Test smoothing effect on rare categories."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import SafeTargetEncoder

        # Category "rare" appears once
        df = pl.DataFrame({"cat": ["common"] * 10 + ["rare"]})
        y = np.array([0] * 10 + [1])

        encoder = SafeTargetEncoder(cols=["cat"], smoothing=10, random_state=42)
        encoder.fit(df, y)

        # Rare category should be smoothed toward global mean
        global_mean = y.mean()
        rare_encoding = encoder._encodings["cat"]["rare"]

        # Should be between raw mean (1.0) and global mean
        assert rare_encoding < 1.0
        assert rare_encoding > global_mean


class TestAutoAggregator:
    """Tests for AutoAggregator."""

    def test_fit_transform(self):
        """Test aggregation feature generation."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import AutoAggregator

        df = pl.DataFrame({
            "group": ["a", "a", "a", "b", "b", "b"],
            "value": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        })

        agg = AutoAggregator(
            group_cols=["group"],
            agg_cols=["value"],
            methods=["mean", "std"],
            rank_features=True,
        )
        result = agg.fit_transform(df)

        assert "value_mean" in result.columns
        assert "value_std" in result.columns
        assert "value_rank" in result.columns

    def test_transform_new_data(self):
        """Test applying aggregations to new data."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import AutoAggregator

        df_train = pl.DataFrame({
            "group": ["a", "a", "b", "b"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })

        df_test = pl.DataFrame({
            "group": ["a", "b"],
            "value": [5.0, 6.0],
        })

        agg = AutoAggregator(
            group_cols=["group"],
            agg_cols=["value"],
            methods=["mean"],
        )
        agg.fit(df_train)
        result = agg.transform(df_test)

        # Group "a" should have mean 1.5, group "b" should have mean 3.5
        assert result["value_mean"][0] == 1.5
        assert result["value_mean"][1] == 3.5


class TestInteractionFeatures:
    """Tests for InteractionFeatures."""

    def test_create_interactions(self):
        """Test interaction feature creation."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import InteractionFeatures

        df = pl.DataFrame({
            "a": [1.0, 2.0, 3.0],
            "b": [4.0, 5.0, 6.0],
        })

        inter = InteractionFeatures(operations=["multiply", "divide"])
        result = inter.fit_transform(df)

        assert "a*b" in result.columns
        assert "a/b" in result.columns
        assert result["a*b"][0] == 4.0


class TestRankFeatures:
    """Tests for RankFeatures."""

    def test_compute_ranks(self):
        """Test rank computation."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import RankFeatures

        df = pl.DataFrame({
            "value": [3.0, 1.0, 2.0],
        })

        ranker = RankFeatures(pct=True)
        result = ranker.fit_transform(df)

        assert "value_rank" in result.columns
        # Ranks should be normalized to [0, 1]
        ranks = result["value_rank"].to_numpy()
        assert ranks.min() >= 0
        assert ranks.max() <= 1


class TestTemporalFeatures:
    """Tests for TemporalFeatures."""

    def test_extract_features(self):
        """Test temporal feature extraction."""
        pl = pytest.importorskip("polars")
        from endgame.preprocessing import TemporalFeatures
        from datetime import datetime

        df = pl.DataFrame({
            "date": [
                datetime(2024, 1, 15, 10, 30),
                datetime(2024, 6, 20, 15, 45),
            ],
        })

        tf = TemporalFeatures(cyclical=True)
        result = tf.fit_transform(df)

        assert "date_year" in result.columns
        assert "date_month" in result.columns
        assert "date_month_sin" in result.columns
        assert "date_month_cos" in result.columns
