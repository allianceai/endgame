"""Tests for Combinatorial Purged Cross-Validation (CPCV)."""

import pytest
import numpy as np
import math
from endgame.validation import CombinatorialPurgedKFold


class TestCombinatorialPurgedKFoldBasic:
    """Basic tests for CPCV initialization and properties."""

    def test_init_default(self):
        """Test default initialization."""
        cpcv = CombinatorialPurgedKFold()
        assert cpcv.n_folds == 10
        assert cpcv.n_test_folds == 2
        assert cpcv.purge_gap == 0
        assert cpcv.embargo_pct == 0.0

    def test_init_custom(self):
        """Test custom initialization."""
        cpcv = CombinatorialPurgedKFold(
            n_folds=6,
            n_test_folds=3,
            purge_gap=5,
            embargo_pct=0.02,
        )
        assert cpcv.n_folds == 6
        assert cpcv.n_test_folds == 3
        assert cpcv.purge_gap == 5
        assert cpcv.embargo_pct == 0.02

    def test_n_splits_property(self):
        """Test n_splits matches C(n_folds, n_test_folds)."""
        # C(10, 2) = 45
        cpcv = CombinatorialPurgedKFold(n_folds=10, n_test_folds=2)
        assert cpcv.n_splits == math.comb(10, 2)
        assert cpcv.n_splits == 45

        # C(6, 2) = 15
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)
        assert cpcv.n_splits == math.comb(6, 2)
        assert cpcv.n_splits == 15

        # C(5, 3) = 10
        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=3)
        assert cpcv.n_splits == math.comb(5, 3)
        assert cpcv.n_splits == 10

    def test_n_test_paths_property(self):
        """Test n_test_paths matches C(n_folds-1, n_test_folds-1)."""
        # C(9, 1) = 9
        cpcv = CombinatorialPurgedKFold(n_folds=10, n_test_folds=2)
        assert cpcv.n_test_paths == math.comb(9, 1)
        assert cpcv.n_test_paths == 9

        # C(5, 1) = 5
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)
        assert cpcv.n_test_paths == math.comb(5, 1)
        assert cpcv.n_test_paths == 5

    def test_invalid_n_folds(self):
        """Test error on invalid n_folds."""
        with pytest.raises(ValueError, match="n_folds must be >= 3"):
            CombinatorialPurgedKFold(n_folds=2)

    def test_invalid_n_test_folds_too_small(self):
        """Test error on n_test_folds < 1."""
        with pytest.raises(ValueError, match="n_test_folds must be >= 1"):
            CombinatorialPurgedKFold(n_folds=5, n_test_folds=0)

    def test_invalid_n_test_folds_too_large(self):
        """Test error on n_test_folds >= n_folds."""
        with pytest.raises(ValueError, match="n_test_folds must be < n_folds"):
            CombinatorialPurgedKFold(n_folds=5, n_test_folds=5)

    def test_repr(self):
        """Test string representation."""
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, purge_gap=5, embargo_pct=0.01)
        repr_str = repr(cpcv)
        assert "CombinatorialPurgedKFold" in repr_str
        assert "n_folds=6" in repr_str
        assert "n_test_folds=2" in repr_str
        assert "purge_gap=5" in repr_str
        assert "embargo_pct=0.01" in repr_str


class TestCombinatorialPurgedKFoldSplit:
    """Tests for the split functionality."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        y = np.random.randn(n_samples)
        times = np.arange(n_samples)
        return X, y, times

    def test_split_generates_correct_number(self, sample_data):
        """Test that split generates the correct number of folds."""
        X, y, times = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)

        splits = list(cpcv.split(X, y, times))
        assert len(splits) == cpcv.n_splits
        assert len(splits) == 15

    def test_split_train_test_disjoint(self, sample_data):
        """Test that train and test sets are disjoint."""
        X, y, times = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, purge_gap=2, embargo_pct=0.01)

        for train_idx, test_idx in cpcv.split(X, y, times):
            # No overlap between train and test
            assert len(np.intersect1d(train_idx, test_idx)) == 0

    def test_split_all_test_covered(self, sample_data):
        """Test that all samples appear in at least one test set."""
        X, y, times = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)

        all_test_idx = set()
        for train_idx, test_idx in cpcv.split(X, y, times):
            all_test_idx.update(test_idx)

        # All samples should appear in test at least once
        assert all_test_idx == set(range(len(X)))

    def test_split_without_times(self, sample_data):
        """Test split works without explicit times (uses index)."""
        X, y, _ = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=2)

        splits = list(cpcv.split(X, y))
        assert len(splits) == cpcv.n_splits
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_split_purge_effect(self, sample_data):
        """Test that purging reduces training set size."""
        X, y, times = sample_data

        # Without purging
        cpcv_no_purge = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, purge_gap=0)
        splits_no_purge = list(cpcv_no_purge.split(X, y, times))

        # With purging
        cpcv_purge = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, purge_gap=5)
        splits_purge = list(cpcv_purge.split(X, y, times))

        # Training sets should be smaller with purging
        total_train_no_purge = sum(len(train) for train, _ in splits_no_purge)
        total_train_purge = sum(len(train) for train, _ in splits_purge)
        assert total_train_purge < total_train_no_purge

    def test_split_embargo_effect(self, sample_data):
        """Test that embargo reduces training set size."""
        X, y, times = sample_data

        # Without embargo
        cpcv_no_embargo = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, embargo_pct=0.0)
        splits_no_embargo = list(cpcv_no_embargo.split(X, y, times))

        # With embargo
        cpcv_embargo = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, embargo_pct=0.05)
        splits_embargo = list(cpcv_embargo.split(X, y, times))

        # Training sets should be smaller with embargo
        total_train_no_embargo = sum(len(train) for train, _ in splits_no_embargo)
        total_train_embargo = sum(len(train) for train, _ in splits_embargo)
        assert total_train_embargo < total_train_no_embargo

    def test_get_n_splits(self, sample_data):
        """Test get_n_splits method."""
        X, y, times = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)

        # Without args
        assert cpcv.get_n_splits() == 15

        # With args (should be same)
        assert cpcv.get_n_splits(X, y, times) == 15


class TestCombinatorialPurgedKFoldInfo:
    """Tests for information methods."""

    @pytest.fixture
    def sample_data(self):
        """Generate sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 5)
        return X

    def test_get_fold_info(self, sample_data):
        """Test get_fold_info method."""
        X = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2, purge_gap=2, embargo_pct=0.01)

        info = cpcv.get_fold_info(X)

        assert info["n_samples"] == len(X)
        assert info["n_folds"] == 6
        assert info["n_test_folds"] == 2
        assert info["n_splits"] == 15
        assert info["n_test_paths"] == 5
        assert len(info["fold_sizes"]) == 6
        assert sum(info["fold_sizes"]) == len(X)
        assert info["purge_gap"] == 2
        assert info["embargo_size"] >= 0

    def test_get_test_paths(self, sample_data):
        """Test get_test_paths method."""
        X = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=2)

        paths = cpcv.get_test_paths(X)

        # Should have C(n_folds, n_test_folds) paths
        assert len(paths) == cpcv.n_splits

        # Each path should have n_test_folds segments
        for path in paths:
            assert len(path) == 2

    def test_paths_cover_all_samples(self, sample_data):
        """Test that combining path segments covers all indices."""
        X = sample_data
        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=2)

        paths = cpcv.get_test_paths(X)

        # Each path when combined should cover a portion of the data
        # (2 folds out of 5)
        for path in paths:
            combined = np.concatenate(path)
            # Should be roughly 2/5 of data
            assert len(combined) >= len(X) * 0.3  # Allow some margin


class TestCombinatorialPurgedKFoldEdgeCases:
    """Edge case tests."""

    def test_small_dataset(self):
        """Test with minimal dataset."""
        X = np.random.randn(15, 3)
        y = np.random.randn(15)
        times = np.arange(15)

        cpcv = CombinatorialPurgedKFold(n_folds=3, n_test_folds=1)

        splits = list(cpcv.split(X, y, times))
        assert len(splits) == 3

        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_single_test_fold(self):
        """Test with n_test_folds=1 (like standard K-fold)."""
        X = np.random.randn(50, 3)
        y = np.random.randn(50)
        times = np.arange(50)

        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=1)

        # C(5,1) = 5 splits
        assert cpcv.n_splits == 5

        splits = list(cpcv.split(X, y, times))
        assert len(splits) == 5

    def test_many_test_folds(self):
        """Test with many test folds."""
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        times = np.arange(100)

        cpcv = CombinatorialPurgedKFold(n_folds=6, n_test_folds=4)

        # C(6,4) = 15 splits
        assert cpcv.n_splits == 15

        splits = list(cpcv.split(X, y, times))
        assert len(splits) == 15

        for train_idx, test_idx in splits:
            # Test should be larger than train
            assert len(test_idx) > len(train_idx)

    def test_high_purge_and_embargo(self):
        """Test with aggressive purging and embargo."""
        X = np.random.randn(200, 3)
        y = np.random.randn(200)
        times = np.arange(200)

        cpcv = CombinatorialPurgedKFold(
            n_folds=5,
            n_test_folds=2,
            purge_gap=10,
            embargo_pct=0.05,  # 5% embargo
        )

        splits = list(cpcv.split(X, y, times))

        for train_idx, test_idx in splits:
            # Should still have non-empty sets
            assert len(train_idx) > 0
            assert len(test_idx) > 0
            # Train should be significantly reduced
            assert len(train_idx) < len(X) * 0.7


class TestCombinatorialPurgedKFoldSKLearnCompat:
    """Tests for scikit-learn compatibility."""

    def test_sklearn_cross_val_score(self):
        """Test that CPCV works with sklearn cross_val_score."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import Ridge

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = np.random.randn(100)

        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=2)
        model = Ridge()

        # This should work without errors
        scores = cross_val_score(model, X, y, cv=cpcv)

        assert len(scores) == cpcv.n_splits
        assert all(np.isfinite(scores))

    def test_sklearn_cross_validate(self):
        """Test that CPCV works with sklearn cross_validate."""
        from sklearn.model_selection import cross_validate
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X = np.random.randn(100, 5)
        y = (np.random.randn(100) > 0).astype(int)

        cpcv = CombinatorialPurgedKFold(n_folds=5, n_test_folds=2)
        model = LogisticRegression(max_iter=1000)

        results = cross_validate(model, X, y, cv=cpcv, return_train_score=True)

        assert len(results["test_score"]) == cpcv.n_splits
        assert len(results["train_score"]) == cpcv.n_splits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
