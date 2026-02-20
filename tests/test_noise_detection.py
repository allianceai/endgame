"""Tests for Label Noise Detection."""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from endgame.preprocessing.noise_detection import (
    ConfidentLearningFilter,
    ConsensusFilter,
    CrossValNoiseDetector,
)


@pytest.fixture
def noisy_data():
    """Create classification data with 10% label noise."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        n_classes=3, random_state=42,
    )
    rng = np.random.RandomState(42)
    noise_idx = rng.choice(len(y), size=20, replace=False)
    y_noisy = y.copy()
    for idx in noise_idx:
        choices = [c for c in range(3) if c != y[idx]]
        y_noisy[idx] = rng.choice(choices)
    return X, y, y_noisy, noise_idx


class TestConfidentLearningFilter:
    def test_basic_fit_detect(self, noisy_data):
        X, y_true, y_noisy, noise_idx = noisy_data
        clf = ConfidentLearningFilter(cv=3, random_state=42)
        noise_mask = clf.fit_detect(X, y_noisy)
        assert noise_mask.shape == (len(y_noisy),)
        assert noise_mask.dtype == bool
        assert noise_mask.sum() > 0  # Should find some noise

    def test_attributes(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        clf = ConfidentLearningFilter(cv=3, random_state=42)
        clf.fit(X, y_noisy)
        assert hasattr(clf, 'noise_mask_')
        assert hasattr(clf, 'noise_indices_')
        assert hasattr(clf, 'confident_joint_')
        assert hasattr(clf, 'noise_rate_')
        assert hasattr(clf, 'per_class_noise_rate_')
        assert hasattr(clf, 'pred_proba_')
        assert 0 <= clf.noise_rate_ <= 1

    def test_clean(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        clf = ConfidentLearningFilter(cv=3, random_state=42)
        X_clean, y_clean = clf.clean(X, y_noisy)
        assert len(X_clean) < len(X)
        assert len(X_clean) == len(y_clean)

    def test_methods(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        for method in ['prune_by_class', 'prune_by_noise_rate', 'both']:
            clf = ConfidentLearningFilter(cv=3, method=method, random_state=42)
            mask = clf.fit_detect(X, y_noisy)
            assert mask.shape == (len(y_noisy),)

    def test_threshold(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        clf = ConfidentLearningFilter(cv=3, threshold=0.3, random_state=42)
        mask = clf.fit_detect(X, y_noisy)
        assert mask.shape == (len(y_noisy),)

    def test_binary_classification(self):
        X, y = make_classification(n_samples=150, n_features=5, random_state=42)
        rng = np.random.RandomState(42)
        y[rng.choice(150, 15, replace=False)] = 1 - y[rng.choice(150, 15, replace=False)]
        clf = ConfidentLearningFilter(cv=3, random_state=42)
        mask = clf.fit_detect(X, y)
        assert mask.sum() > 0


class TestConsensusFilter:
    def test_basic(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        cf = ConsensusFilter(cv=3, random_state=42)
        mask = cf.fit_detect(X, y_noisy)
        assert mask.shape == (len(y_noisy),)
        assert hasattr(cf, 'noise_rate_')
        assert hasattr(cf, 'disagreement_ratio_')

    def test_clean(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        cf = ConsensusFilter(cv=3, random_state=42)
        X_clean, y_clean = cf.clean(X, y_noisy)
        assert len(X_clean) <= len(X)

    def test_threshold(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        # High threshold = fewer flagged
        cf_high = ConsensusFilter(cv=3, consensus_threshold=0.9, random_state=42)
        cf_low = ConsensusFilter(cv=3, consensus_threshold=0.3, random_state=42)
        mask_high = cf_high.fit_detect(X, y_noisy)
        mask_low = cf_low.fit_detect(X, y_noisy)
        assert mask_high.sum() <= mask_low.sum()


class TestCrossValNoiseDetector:
    def test_basic(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        detector = CrossValNoiseDetector(
            cv=3, n_repeats=2, random_state=42
        )
        mask = detector.fit_detect(X, y_noisy)
        assert mask.shape == (len(y_noisy),)
        assert hasattr(detector, 'misclassification_rate_')

    def test_clean(self, noisy_data):
        X, _, y_noisy, _ = noisy_data
        detector = CrossValNoiseDetector(cv=3, n_repeats=2, random_state=42)
        X_clean, y_clean = detector.clean(X, y_noisy)
        assert len(X_clean) <= len(X)
