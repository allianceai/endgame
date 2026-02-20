"""Tests for modern geometric SMOTE extensions."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def imbalanced_binary():
    """90/10 binary classification."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],
        flip_y=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def imbalanced_severe():
    """95/5 binary classification."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],
        flip_y=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_imbalanced():
    """70/20/10 multiclass."""
    X, y = make_classification(
        n_samples=300,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=3,
        n_clusters_per_class=1,
        weights=[0.7, 0.2, 0.1],
        flip_y=0,
        random_state=42,
    )
    return X, y


# ============================================================================
# _compute_sampling_targets
# ============================================================================


class TestComputeSamplingTargets:
    """Tests for the shared utility function."""

    def test_auto_strategy(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

        _, y = imbalanced_binary
        targets = _compute_sampling_targets(y, "auto")
        assert len(targets) >= 1
        for cls, n in targets.items():
            assert n > 0

    def test_float_strategy(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

        _, y = imbalanced_binary
        targets = _compute_sampling_targets(y, 0.5)
        assert len(targets) >= 1

    def test_dict_strategy(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

        _, y = imbalanced_binary
        targets = _compute_sampling_targets(y, {1: 100})
        assert 1 in targets

    def test_all_strategy(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

        _, y = imbalanced_binary
        targets = _compute_sampling_targets(y, "all")
        assert len(targets) >= 1

    def test_invalid_strategy(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

        _, y = imbalanced_binary
        with pytest.raises(ValueError, match="Unknown sampling_strategy"):
            _compute_sampling_targets(y, "invalid")


# ============================================================================
# MultivariateGaussianSMOTE
# ============================================================================


class TestMultivariateGaussianSMOTE:
    """Tests for MultivariateGaussianSMOTE."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = imbalanced_binary
        sampler = MultivariateGaussianSMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert len(y_res) > len(y)
        # Check class balance
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_sampling_strategy_float(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = imbalanced_binary
        sampler = MultivariateGaussianSMOTE(sampling_strategy=0.5, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)

    def test_multiclass(self, multiclass_imbalanced):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = multiclass_imbalanced
        sampler = MultivariateGaussianSMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        assert len(np.unique(y_res)) == 3

    def test_random_state_reproducibility(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = imbalanced_binary
        X1, y1 = MultivariateGaussianSMOTE(random_state=42).fit_resample(X, y)
        X2, y2 = MultivariateGaussianSMOTE(random_state=42).fit_resample(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_regularization(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = imbalanced_binary
        sampler = MultivariateGaussianSMOTE(regularization=1.0, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)

    def test_k_neighbors(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import MultivariateGaussianSMOTE

        X, y = imbalanced_binary
        sampler = MultivariateGaussianSMOTE(k_neighbors=3, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


# ============================================================================
# SimplicialSMOTE
# ============================================================================


class TestSimplicialSMOTE:
    """Tests for SimplicialSMOTE."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import SimplicialSMOTE

        X, y = imbalanced_binary
        sampler = SimplicialSMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_simplex_dim(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import SimplicialSMOTE

        X, y = imbalanced_binary
        for dim in [1, 2, 3]:
            sampler = SimplicialSMOTE(simplex_dim=dim, random_state=42)
            X_res, y_res = sampler.fit_resample(X, y)
            assert len(X_res) > len(X)

    def test_multiclass(self, multiclass_imbalanced):
        from endgame.preprocessing.imbalance_geometric import SimplicialSMOTE

        X, y = multiclass_imbalanced
        sampler = SimplicialSMOTE(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(np.unique(y_res)) == 3

    def test_random_state_reproducibility(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import SimplicialSMOTE

        X, y = imbalanced_binary
        X1, y1 = SimplicialSMOTE(random_state=42).fit_resample(X, y)
        X2, y2 = SimplicialSMOTE(random_state=42).fit_resample(X, y)
        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)


# ============================================================================
# CVSMOTEResampler
# ============================================================================


class TestCVSMOTEResampler:
    """Tests for CVSMOTEResampler."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import CVSMOTEResampler

        X, y = imbalanced_binary
        sampler = CVSMOTEResampler(cv=2, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)

    def test_sampling_strategy_float(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import CVSMOTEResampler

        X, y = imbalanced_binary
        sampler = CVSMOTEResampler(sampling_strategy=0.5, cv=2, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)

    def test_custom_estimator(self, imbalanced_binary):
        from sklearn.tree import DecisionTreeClassifier
        from endgame.preprocessing.imbalance_geometric import CVSMOTEResampler

        X, y = imbalanced_binary
        sampler = CVSMOTEResampler(
            estimator=DecisionTreeClassifier(max_depth=3),
            cv=2,
            random_state=42,
        )
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


# ============================================================================
# OverlapRegionDetector
# ============================================================================


class TestOverlapRegionDetector:
    """Tests for OverlapRegionDetector."""

    def test_fit_resample(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import OverlapRegionDetector

        X, y = imbalanced_binary
        sampler = OverlapRegionDetector(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > 0
        assert set(np.unique(y_res)).issubset(set(np.unique(y)))

    def test_base_sampler_string(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import OverlapRegionDetector

        X, y = imbalanced_binary
        for sampler_name in ["smote", "borderline_smote", "random_over"]:
            sampler = OverlapRegionDetector(
                base_sampler=sampler_name, random_state=42
            )
            X_res, y_res = sampler.fit_resample(X, y)
            assert len(X_res) > 0

    def test_invalid_base_sampler(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import OverlapRegionDetector

        X, y = imbalanced_binary
        sampler = OverlapRegionDetector(base_sampler="nonexistent")
        with pytest.raises(ValueError, match="Unknown base_sampler"):
            sampler.fit_resample(X, y)

    def test_threshold(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import OverlapRegionDetector

        X, y = imbalanced_binary
        sampler = OverlapRegionDetector(threshold=0.5, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > 0

    def test_overlap_count_stored(self, imbalanced_binary):
        from endgame.preprocessing.imbalance_geometric import OverlapRegionDetector

        X, y = imbalanced_binary
        sampler = OverlapRegionDetector(random_state=42)
        sampler.fit_resample(X, y)
        assert hasattr(sampler, "n_overlap_")
        assert sampler.n_overlap_ >= 0


# ============================================================================
# GEOMETRIC_SAMPLERS dict
# ============================================================================


class TestGeometricSamplersDict:
    """Tests for the GEOMETRIC_SAMPLERS registry."""

    def test_dict_exists_and_complete(self):
        from endgame.preprocessing.imbalance_geometric import GEOMETRIC_SAMPLERS

        assert "multivariate_gaussian_smote" in GEOMETRIC_SAMPLERS
        assert "simplicial_smote" in GEOMETRIC_SAMPLERS
        assert "cv_smote" in GEOMETRIC_SAMPLERS
        assert "overlap_region_detector" in GEOMETRIC_SAMPLERS
        assert len(GEOMETRIC_SAMPLERS) == 4
