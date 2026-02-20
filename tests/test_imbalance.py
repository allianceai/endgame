"""Tests for the class imbalance handling module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def imbalanced_data():
    """Generate imbalanced binary classification data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.9, 0.1],  # 90% class 0, 10% class 1
        flip_y=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def severely_imbalanced_data():
    """Generate severely imbalanced data."""
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.95, 0.05],  # 95% class 0, 5% class 1
        flip_y=0,
        random_state=42,
    )
    return X, y


@pytest.fixture
def multiclass_imbalanced_data():
    """Generate imbalanced multiclass data."""
    X, y = make_classification(
        n_samples=1000,
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


class TestSMOTEResampler:
    """Tests for SMOTEResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import SMOTEResampler

        X, y = imbalanced_data
        smote = SMOTEResampler(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        # Should have more samples
        assert len(X_res) > len(X)
        assert len(y_res) > len(y)

        # Classes should be balanced
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_sampling_strategy_float(self, imbalanced_data):
        """Test with float sampling strategy."""
        from endgame.preprocessing import SMOTEResampler

        X, y = imbalanced_data
        smote = SMOTEResampler(sampling_strategy=0.5, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        unique, counts = np.unique(y_res, return_counts=True)
        # Minority class should have ~50% of majority
        ratio = counts.min() / counts.max()
        assert 0.45 <= ratio <= 0.55

    def test_k_neighbors(self, imbalanced_data):
        """Test different k_neighbors values."""
        from endgame.preprocessing import SMOTEResampler

        X, y = imbalanced_data
        smote = SMOTEResampler(k_neighbors=3, random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        assert len(X_res) > len(X)


class TestBorderlineSMOTEResampler:
    """Tests for BorderlineSMOTEResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import BorderlineSMOTEResampler

        X, y = imbalanced_data
        sampler = BorderlineSMOTEResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)

    def test_kind_borderline2(self, imbalanced_data):
        """Test borderline-2 mode."""
        from endgame.preprocessing import BorderlineSMOTEResampler

        X, y = imbalanced_data
        sampler = BorderlineSMOTEResampler(kind='borderline-2', random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)


class TestADASYNResampler:
    """Tests for ADASYNResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import ADASYNResampler

        X, y = imbalanced_data
        sampler = ADASYNResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)


class TestSVMSMOTEResampler:
    """Tests for SVMSMOTEResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import SVMSMOTEResampler

        X, y = imbalanced_data
        sampler = SVMSMOTEResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)


class TestKMeansSMOTEResampler:
    """Tests for KMeansSMOTEResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import KMeansSMOTEResampler

        X, y = imbalanced_data
        sampler = KMeansSMOTEResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)


class TestRandomOverSampler:
    """Tests for RandomOverSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import RandomOverSampler

        X, y = imbalanced_data
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) > len(X)
        # Classes should be balanced
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]


class TestEditedNearestNeighbours:
    """Tests for EditedNearestNeighbours."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import EditedNearestNeighbours

        X, y = imbalanced_data
        sampler = EditedNearestNeighbours()
        X_res, y_res = sampler.fit_resample(X, y)

        # Should remove some samples
        assert len(X_res) <= len(X)

    def test_kind_sel_mode(self, imbalanced_data):
        """Test with kind_sel='mode'."""
        from endgame.preprocessing import EditedNearestNeighbours

        X, y = imbalanced_data
        sampler = EditedNearestNeighbours(kind_sel='mode')
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) <= len(X)


class TestAllKNNUnderSampler:
    """Tests for AllKNNUnderSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import AllKNNUnderSampler

        X, y = imbalanced_data
        sampler = AllKNNUnderSampler()
        X_res, y_res = sampler.fit_resample(X, y)

        # Should remove some samples
        assert len(X_res) <= len(X)


class TestTomekLinksUnderSampler:
    """Tests for TomekLinksUnderSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import TomekLinksUnderSampler

        X, y = imbalanced_data
        sampler = TomekLinksUnderSampler()
        X_res, y_res = sampler.fit_resample(X, y)

        # May or may not remove samples depending on data
        assert len(X_res) <= len(X)


class TestRandomUnderSampler:
    """Tests for RandomUnderSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import RandomUnderSampler

        X, y = imbalanced_data
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        # Should remove majority samples
        assert len(X_res) < len(X)
        # Classes should be balanced
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]


class TestNearMissUnderSampler:
    """Tests for NearMissUnderSampler."""

    def test_fit_resample_v1(self, imbalanced_data):
        """Test NearMiss version 1."""
        from endgame.preprocessing import NearMissUnderSampler

        X, y = imbalanced_data
        sampler = NearMissUnderSampler(version=1)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_fit_resample_v2(self, imbalanced_data):
        """Test NearMiss version 2."""
        from endgame.preprocessing import NearMissUnderSampler

        X, y = imbalanced_data
        sampler = NearMissUnderSampler(version=2)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_fit_resample_v3(self, imbalanced_data):
        """Test NearMiss version 3."""
        from endgame.preprocessing import NearMissUnderSampler

        X, y = imbalanced_data
        sampler = NearMissUnderSampler(version=3)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) < len(X)


class TestCondensedNearestNeighbour:
    """Tests for CondensedNearestNeighbour."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import CondensedNearestNeighbour

        X, y = imbalanced_data
        sampler = CondensedNearestNeighbour(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        # Should remove samples
        assert len(X_res) < len(X)


class TestOneSidedSelectionUnderSampler:
    """Tests for OneSidedSelectionUnderSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import OneSidedSelectionUnderSampler

        X, y = imbalanced_data
        sampler = OneSidedSelectionUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) <= len(X)


class TestNeighbourhoodCleaningRule:
    """Tests for NeighbourhoodCleaningRule."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import NeighbourhoodCleaningRule

        X, y = imbalanced_data
        sampler = NeighbourhoodCleaningRule()
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) <= len(X)


class TestInstanceHardnessThresholdSampler:
    """Tests for InstanceHardnessThresholdSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import InstanceHardnessThresholdSampler

        X, y = imbalanced_data
        sampler = InstanceHardnessThresholdSampler(random_state=42, cv=2)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) <= len(X)


class TestClusterCentroidsUnderSampler:
    """Tests for ClusterCentroidsUnderSampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import ClusterCentroidsUnderSampler

        X, y = imbalanced_data
        sampler = ClusterCentroidsUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        assert len(X_res) < len(X)


class TestSMOTEENNResampler:
    """Tests for SMOTEENNResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import SMOTEENNResampler

        X, y = imbalanced_data
        sampler = SMOTEENNResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        # SMOTE increases, ENN may decrease
        # Just check it runs
        assert len(X_res) > 0


class TestSMOTETomekResampler:
    """Tests for SMOTETomekResampler."""

    def test_fit_resample(self, imbalanced_data):
        """Test basic fit and resample."""
        from endgame.preprocessing import SMOTETomekResampler

        X, y = imbalanced_data
        sampler = SMOTETomekResampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)

        # SMOTE increases, Tomek may decrease slightly
        assert len(X_res) > 0


class TestAutoBalancer:
    """Tests for AutoBalancer."""

    def test_auto_strategy_balanced_data(self):
        """Test that auto strategy doesn't resample balanced data."""
        from endgame.preprocessing import AutoBalancer

        # Create balanced data
        X, y = make_classification(
            n_samples=100, n_features=5, weights=[0.5, 0.5], random_state=42
        )

        balancer = AutoBalancer(strategy='auto', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        # Should not resample balanced data
        assert len(X_res) == len(X)
        assert balancer.selected_strategy_ == 'none'

    def test_auto_strategy_imbalanced(self, imbalanced_data):
        """Test auto strategy on imbalanced data."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='auto', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        # Should have selected a strategy and resampled
        assert balancer.selected_strategy_ != 'none'
        assert len(X_res) != len(X) or not np.array_equal(y_res, y)

    def test_auto_strategy_severe_imbalance(self, severely_imbalanced_data):
        """Test auto strategy on severely imbalanced data."""
        from endgame.preprocessing import AutoBalancer

        X, y = severely_imbalanced_data
        balancer = AutoBalancer(strategy='auto', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        # Should select combined approach for severe imbalance
        assert balancer.selected_strategy_ == 'smoteenn'

    def test_specific_strategy(self, imbalanced_data):
        """Test with specific strategy."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='smote', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'smote'
        assert len(X_res) > len(X)

    def test_oversample_strategy(self, imbalanced_data):
        """Test oversample strategy."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='oversample', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'smote'

    def test_undersample_strategy(self, imbalanced_data):
        """Test undersample strategy."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='undersample', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'enn'

    def test_combine_strategy(self, imbalanced_data):
        """Test combine strategy."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='combine', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'smoteenn'

    def test_get_sampler(self, imbalanced_data):
        """Test get_sampler method."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='smote', random_state=42)
        balancer.fit_resample(X, y)

        sampler = balancer.get_sampler()
        assert sampler is not None

    def test_invalid_strategy(self, imbalanced_data):
        """Test error on invalid strategy."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='invalid_strategy')

        with pytest.raises(ValueError, match="Unknown strategy"):
            balancer.fit_resample(X, y)

    def test_multiclass(self, multiclass_imbalanced_data):
        """Test with multiclass data."""
        from endgame.preprocessing import AutoBalancer

        X, y = multiclass_imbalanced_data
        balancer = AutoBalancer(strategy='smote', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        # Should work with multiclass
        assert len(np.unique(y_res)) == 3

    def test_geometric_strategy(self, imbalanced_data):
        """Test geometric strategy alias."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='geometric', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'multivariate_gaussian_smote'
        assert len(X_res) > len(X)

    def test_generative_strategy(self, imbalanced_data):
        """Test generative strategy alias."""
        from endgame.preprocessing import AutoBalancer

        X, y = imbalanced_data
        balancer = AutoBalancer(strategy='generative', random_state=42)
        X_res, y_res = balancer.fit_resample(X, y)

        assert balancer.selected_strategy_ == 'forest_flow'
        assert len(X_res) > len(X)


class TestUtilities:
    """Tests for utility functions."""

    def test_get_imbalance_ratio(self, imbalanced_data):
        """Test get_imbalance_ratio function."""
        from endgame.preprocessing import get_imbalance_ratio

        _, y = imbalanced_data
        ratio = get_imbalance_ratio(y)

        # Should be around 0.1 (10% minority)
        assert 0.05 <= ratio <= 0.2

    def test_get_imbalance_ratio_balanced(self):
        """Test get_imbalance_ratio on balanced data."""
        from endgame.preprocessing import get_imbalance_ratio

        y = [0, 0, 0, 1, 1, 1]
        ratio = get_imbalance_ratio(y)
        assert ratio == 1.0

    def test_get_imbalance_ratio_single_class(self):
        """Test get_imbalance_ratio with single class."""
        from endgame.preprocessing import get_imbalance_ratio

        y = [0, 0, 0, 0]
        ratio = get_imbalance_ratio(y)
        assert ratio == 1.0

    def test_get_class_distribution(self, imbalanced_data):
        """Test get_class_distribution function."""
        from endgame.preprocessing import get_class_distribution

        _, y = imbalanced_data
        dist = get_class_distribution(y)

        assert isinstance(dist, dict)
        assert len(dist) == 2
        assert 0 in dist
        assert 1 in dist

    def test_get_class_distribution_multiclass(self, multiclass_imbalanced_data):
        """Test get_class_distribution on multiclass data."""
        from endgame.preprocessing import get_class_distribution

        _, y = multiclass_imbalanced_data
        dist = get_class_distribution(y)

        assert len(dist) == 3


class TestSamplerDictionaries:
    """Tests for sampler dictionaries."""

    def test_over_samplers_dict(self):
        """Test OVER_SAMPLERS dictionary."""
        from endgame.preprocessing import OVER_SAMPLERS

        assert 'smote' in OVER_SAMPLERS
        assert 'adasyn' in OVER_SAMPLERS
        assert 'borderline_smote' in OVER_SAMPLERS
        assert len(OVER_SAMPLERS) >= 5

    def test_under_samplers_dict(self):
        """Test UNDER_SAMPLERS dictionary."""
        from endgame.preprocessing import UNDER_SAMPLERS

        assert 'enn' in UNDER_SAMPLERS
        assert 'tomek' in UNDER_SAMPLERS
        assert 'allknn' in UNDER_SAMPLERS
        assert len(UNDER_SAMPLERS) >= 8

    def test_combined_samplers_dict(self):
        """Test COMBINED_SAMPLERS dictionary."""
        from endgame.preprocessing import COMBINED_SAMPLERS

        assert 'smoteenn' in COMBINED_SAMPLERS
        assert 'smotetomek' in COMBINED_SAMPLERS

    def test_all_samplers_dict(self):
        """Test ALL_SAMPLERS dictionary contains all category dicts."""
        from endgame.preprocessing import (
            ALL_SAMPLERS, OVER_SAMPLERS, UNDER_SAMPLERS, COMBINED_SAMPLERS,
            GEOMETRIC_SAMPLERS, GENERATIVE_SAMPLERS, LLM_SAMPLERS,
        )

        # ALL_SAMPLERS should contain at least the base categories
        for key in OVER_SAMPLERS:
            assert key in ALL_SAMPLERS
        for key in UNDER_SAMPLERS:
            assert key in ALL_SAMPLERS
        for key in COMBINED_SAMPLERS:
            assert key in ALL_SAMPLERS
        for key in GEOMETRIC_SAMPLERS:
            assert key in ALL_SAMPLERS

    def test_geometric_samplers_dict(self):
        """Test GEOMETRIC_SAMPLERS dictionary."""
        from endgame.preprocessing import GEOMETRIC_SAMPLERS

        assert 'multivariate_gaussian_smote' in GEOMETRIC_SAMPLERS
        assert 'simplicial_smote' in GEOMETRIC_SAMPLERS
        assert 'cv_smote' in GEOMETRIC_SAMPLERS
        assert 'overlap_region_detector' in GEOMETRIC_SAMPLERS
        assert len(GEOMETRIC_SAMPLERS) == 4


class TestModuleImports:
    """Test module-level imports work correctly."""

    def test_import_from_endgame(self):
        """Test imports from main endgame package."""
        import endgame as eg

        assert hasattr(eg.preprocessing, "SMOTEResampler")
        assert hasattr(eg.preprocessing, "AutoBalancer")
        assert hasattr(eg.preprocessing, "EditedNearestNeighbours")
        assert hasattr(eg.preprocessing, "SMOTEENNResampler")
        assert hasattr(eg.preprocessing, "get_imbalance_ratio")

    def test_direct_imports(self):
        """Test direct imports from preprocessing module."""
        from endgame.preprocessing import (
            SMOTEResampler,
            BorderlineSMOTEResampler,
            ADASYNResampler,
            EditedNearestNeighbours,
            AllKNNUnderSampler,
            TomekLinksUnderSampler,
            SMOTEENNResampler,
            SMOTETomekResampler,
            AutoBalancer,
            get_imbalance_ratio,
            get_class_distribution,
            ALL_SAMPLERS,
        )

        assert SMOTEResampler is not None
        assert AutoBalancer is not None
        assert callable(get_imbalance_ratio)
