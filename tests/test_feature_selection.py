"""Tests for feature selection module."""

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import all feature selection methods
from endgame.feature_selection import (
    UnivariateSelector,
    MutualInfoSelector,
    FTestSelector,
    Chi2Selector,
    MRMRSelector,
    ReliefFSelector,
    CorrelationSelector,
    RFESelector,
    BorutaSelector,
    SequentialSelector,
    GeneticSelector,
    PermutationSelector,
    TreeImportanceSelector,
    StabilitySelector,
    KnockoffSelector,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def classification_data():
    """Create synthetic classification data with informative and noise features."""
    X, y = make_classification(
        n_samples=200,
        n_features=20,
        n_informative=10,
        n_redundant=5,
        n_clusters_per_class=1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def regression_data():
    """Create synthetic regression data."""
    X, y = make_regression(
        n_samples=200,
        n_features=20,
        n_informative=10,
        noise=0.1,
        random_state=42,
    )
    return X, y


@pytest.fixture
def small_classification_data():
    """Create smaller dataset for slow methods."""
    X, y = make_classification(
        n_samples=100,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )
    return X, y


# ============================================================================
# Filter Methods Tests
# ============================================================================


class TestUnivariateSelector:
    """Tests for UnivariateSelector."""

    def test_fit_transform(self, classification_data):
        """Test basic fit_transform."""
        X, y = classification_data
        selector = UnivariateSelector(k=5)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)
        assert len(selector.selected_features_) == 5

    def test_get_support(self, classification_data):
        """Test get_support method."""
        X, y = classification_data
        selector = UnivariateSelector(k=5)
        selector.fit(X, y)

        mask = selector.get_support(indices=False)
        indices = selector.get_support(indices=True)

        assert mask.shape == (20,)
        assert mask.sum() == 5
        assert len(indices) == 5

    def test_different_score_funcs(self, classification_data):
        """Test different scoring functions."""
        X, y = classification_data

        for score_func in ["f_classif", "mutual_info_classif", "chi2"]:
            # chi2 requires non-negative features
            X_test = np.abs(X) if score_func == "chi2" else X
            selector = UnivariateSelector(k=5, score_func=score_func)
            X_selected = selector.fit_transform(X_test, y)
            assert X_selected.shape[1] == 5


class TestMutualInfoSelector:
    """Tests for MutualInfoSelector."""

    def test_fit_transform(self, classification_data):
        """Test basic fit_transform."""
        X, y = classification_data
        selector = MutualInfoSelector(k=5, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_regression(self, regression_data):
        """Test with regression target."""
        X, y = regression_data
        selector = MutualInfoSelector(k=5, task="regression", random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)


class TestFTestSelector:
    """Tests for FTestSelector."""

    def test_classification(self, classification_data):
        """Test F-test for classification."""
        X, y = classification_data
        selector = FTestSelector(k=5, task="classification")
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)
        assert hasattr(selector, "scores_")
        assert hasattr(selector, "pvalues_")

    def test_regression(self, regression_data):
        """Test F-test for regression."""
        X, y = regression_data
        selector = FTestSelector(k=5, task="regression")
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)


class TestChi2Selector:
    """Tests for Chi2Selector."""

    def test_fit_transform(self, classification_data):
        """Test chi2 selection."""
        X, y = classification_data
        X_positive = np.abs(X)  # chi2 requires non-negative features

        selector = Chi2Selector(k=5)
        X_selected = selector.fit_transform(X_positive, y)

        assert X_selected.shape == (200, 5)


class TestMRMRSelector:
    """Tests for MRMRSelector (Minimum Redundancy Maximum Relevance)."""

    def test_fit_transform(self, classification_data):
        """Test basic MRMR selection."""
        X, y = classification_data
        selector = MRMRSelector(n_features=5, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)
        assert len(selector.selected_features_) == 5

    def test_f_test_relevance(self, classification_data):
        """Test with F-test relevance function."""
        X, y = classification_data
        selector = MRMRSelector(n_features=5, relevance_func="f_test", random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_feature_ranking(self, classification_data):
        """Test feature ranking is stored."""
        X, y = classification_data
        selector = MRMRSelector(n_features=5, random_state=42)
        selector.fit(X, y)

        ranking = selector.get_feature_ranking()
        assert len(ranking) == X.shape[1]
        # All features should be unique in ranking
        assert len(set(ranking)) == X.shape[1]


class TestReliefFSelector:
    """Tests for ReliefFSelector."""

    def test_fit_transform(self, small_classification_data):
        """Test basic ReliefF selection."""
        X, y = small_classification_data
        selector = ReliefFSelector(n_features=3, n_neighbors=5, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (100, 3)

    def test_feature_importances(self, small_classification_data):
        """Test feature importances are computed."""
        X, y = small_classification_data
        selector = ReliefFSelector(n_features=3, n_neighbors=5, random_state=42)
        selector.fit(X, y)

        assert hasattr(selector, "feature_importances_")
        assert len(selector.feature_importances_) == X.shape[1]


class TestCorrelationSelector:
    """Tests for CorrelationSelector."""

    def test_remove_correlated(self, classification_data):
        """Test removing highly correlated features."""
        X, y = classification_data
        selector = CorrelationSelector(threshold=0.8)
        X_selected = selector.fit_transform(X, y)

        # Should remove some correlated features
        assert X_selected.shape[1] <= X.shape[1]

    def test_variance_keep_strategy(self, classification_data):
        """Test keeping by variance."""
        X, y = classification_data
        selector = CorrelationSelector(threshold=0.8, keep="variance")
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] <= X.shape[1]


# ============================================================================
# Wrapper Methods Tests
# ============================================================================


class TestRFESelector:
    """Tests for RFESelector (Recursive Feature Elimination)."""

    def test_fit_transform(self, classification_data):
        """Test basic RFE selection."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFESelector(estimator=estimator, n_features=5)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_with_cv(self, small_classification_data):
        """Test RFECV with cross-validation (n_features=None)."""
        X, y = small_classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFESelector(estimator=estimator, n_features=None, cv=3)
        X_selected = selector.fit_transform(X, y)

        # CV selects optimal number
        assert X_selected.shape[1] >= 1

    def test_feature_ranking(self, classification_data):
        """Test feature ranking."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=10, random_state=42)
        selector = RFESelector(estimator=estimator, n_features=5)
        selector.fit(X, y)

        ranking = selector.get_feature_ranking()
        assert len(ranking) == X.shape[1]


class TestBorutaSelector:
    """Tests for BorutaSelector."""

    def test_fit_transform(self, small_classification_data):
        """Test basic Boruta selection."""
        X, y = small_classification_data
        selector = BorutaSelector(
            n_estimators=20,
            max_iter=20,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        # Boruta selects features that beat shadow features
        assert X_selected.shape[1] >= 1

    def test_feature_categories(self, small_classification_data):
        """Test feature categorization (confirmed/tentative/rejected)."""
        X, y = small_classification_data
        selector = BorutaSelector(
            n_estimators=20,
            max_iter=20,
            random_state=42,
        )
        selector.fit(X, y)

        assert hasattr(selector, "selected_features_")
        assert hasattr(selector, "tentative_features_")
        assert hasattr(selector, "rejected_features_")


class TestSequentialSelector:
    """Tests for SequentialSelector."""

    def test_forward_selection(self, classification_data):
        """Test forward selection."""
        X, y = classification_data
        estimator = LogisticRegression(max_iter=200, random_state=42)
        selector = SequentialSelector(
            estimator=estimator,
            n_features=5,
            direction="forward",
            cv=3,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_backward_selection(self, classification_data):
        """Test backward selection."""
        X, y = classification_data
        estimator = LogisticRegression(max_iter=200, random_state=42)
        selector = SequentialSelector(
            estimator=estimator,
            n_features=5,
            direction="backward",
            cv=3,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)


class TestGeneticSelector:
    """Tests for GeneticSelector."""

    def test_fit_transform(self, small_classification_data):
        """Test genetic algorithm selection."""
        X, y = small_classification_data
        estimator = LogisticRegression(max_iter=200, random_state=42)
        selector = GeneticSelector(
            estimator=estimator,
            population_size=10,
            n_generations=5,
            cv=3,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] >= 1

    def test_evolution_history(self, small_classification_data):
        """Test evolution history is stored."""
        X, y = small_classification_data
        estimator = LogisticRegression(max_iter=200, random_state=42)
        selector = GeneticSelector(
            estimator=estimator,
            population_size=10,
            n_generations=5,
            cv=3,
            random_state=42,
        )
        selector.fit(X, y)

        assert hasattr(selector, "history_")
        assert len(selector.history_) == 5


# ============================================================================
# Importance-Based Methods Tests
# ============================================================================


class TestPermutationSelector:
    """Tests for PermutationSelector."""

    def test_fit_transform(self, classification_data):
        """Test permutation importance selection."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=20, random_state=42)
        estimator.fit(X, y)  # Estimator must be fitted

        selector = PermutationSelector(
            estimator=estimator,
            n_features=5,
            n_repeats=5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_importance_scores(self, classification_data):
        """Test importance scores are computed."""
        X, y = classification_data
        estimator = RandomForestClassifier(n_estimators=20, random_state=42)
        estimator.fit(X, y)

        selector = PermutationSelector(
            estimator=estimator,
            n_features=5,
            n_repeats=5,
            random_state=42,
        )
        selector.fit(X, y)

        assert hasattr(selector, "feature_importances_")
        assert len(selector.feature_importances_) == X.shape[1]


class TestTreeImportanceSelector:
    """Tests for TreeImportanceSelector."""

    def test_fit_transform(self, classification_data):
        """Test tree importance selection."""
        X, y = classification_data
        selector = TreeImportanceSelector(n_features=5, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_with_custom_estimator(self, classification_data):
        """Test with custom estimator."""
        X, y = classification_data
        from sklearn.ensemble import ExtraTreesClassifier

        estimator = ExtraTreesClassifier(n_estimators=20, random_state=42)
        selector = TreeImportanceSelector(
            estimator=estimator,
            n_features=5,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 5)

    def test_mean_threshold(self, classification_data):
        """Test mean importance threshold."""
        X, y = classification_data
        selector = TreeImportanceSelector(n_features="mean", random_state=42)
        selector.fit(X, y)

        # Features above mean importance
        assert selector.n_features_ >= 1

    def test_feature_ranking(self, classification_data):
        """Test feature ranking method."""
        X, y = classification_data
        selector = TreeImportanceSelector(n_features=5, random_state=42)
        selector.fit(X, y)

        ranking = selector.get_feature_ranking()
        assert len(ranking) == X.shape[1]


# ============================================================================
# Advanced Methods Tests
# ============================================================================


class TestStabilitySelector:
    """Tests for StabilitySelector."""

    def test_fit_transform(self, small_classification_data):
        """Test stability selection wrapper."""
        X, y = small_classification_data
        base_selector = TreeImportanceSelector(n_features=3, random_state=42)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstrap=20,
            threshold=0.5,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        # Should select stable features
        assert X_selected.shape[1] >= 1

    def test_selection_frequencies(self, small_classification_data):
        """Test selection frequencies are computed."""
        X, y = small_classification_data
        base_selector = TreeImportanceSelector(n_features=3, random_state=42)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstrap=20,
            threshold=0.5,
            random_state=42,
        )
        selector.fit(X, y)

        freqs = selector.get_selection_frequencies()
        assert len(freqs) == X.shape[1]
        assert all(0 <= f <= 1 for f in freqs)

    def test_max_features_constraint(self, small_classification_data):
        """Test max_features parameter."""
        X, y = small_classification_data
        base_selector = TreeImportanceSelector(n_features=5, random_state=42)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstrap=20,
            threshold=0.3,
            max_features=3,
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[1] <= 3


class TestKnockoffSelector:
    """Tests for KnockoffSelector (FDR control)."""

    def test_fit_transform(self, classification_data):
        """Test knockoff filter."""
        X, y = classification_data
        selector = KnockoffSelector(fdr=0.2, random_state=42)
        X_selected = selector.fit_transform(X, y)

        # Knockoff may select varying numbers
        assert X_selected.shape[1] >= 0
        assert X_selected.shape[1] <= X.shape[1]

    def test_knockoff_statistics(self, classification_data):
        """Test knockoff W statistics are computed."""
        X, y = classification_data
        selector = KnockoffSelector(fdr=0.2, random_state=42)
        selector.fit(X, y)

        stats = selector.get_statistics()
        assert len(stats) == X.shape[1]

    def test_equicorrelated_knockoffs(self, classification_data):
        """Test equicorrelated knockoff generation."""
        X, y = classification_data
        selector = KnockoffSelector(
            fdr=0.2,
            method="equicorrelated",
            random_state=42,
        )
        selector.fit(X, y)

        assert hasattr(selector, "knockoffs_")
        assert selector.knockoffs_.shape == X.shape

    def test_different_statistics(self, classification_data):
        """Test different knockoff statistics."""
        X, y = classification_data

        for statistic in ["lasso_cv", "ridge"]:
            selector = KnockoffSelector(
                fdr=0.2,
                statistic=statistic,
                random_state=42,
            )
            selector.fit(X, y)
            assert hasattr(selector, "statistics_")


# ============================================================================
# SHAP Selector Tests (conditional on shap being installed)
# ============================================================================


class TestSHAPSelector:
    """Tests for SHAPSelector."""

    def test_fit_transform(self, small_classification_data):
        """Test SHAP-based selection."""
        pytest.importorskip("shap")

        X, y = small_classification_data
        from sklearn.ensemble import GradientBoostingClassifier

        estimator = GradientBoostingClassifier(
            n_estimators=20,
            max_depth=3,
            random_state=42,
        )
        estimator.fit(X, y)

        from endgame.feature_selection import SHAPSelector
        selector = SHAPSelector(
            estimator=estimator,
            n_features=3,
            max_samples=50,
            random_state=42,
        )
        X_selected = selector.fit_transform(X)

        assert X_selected.shape == (100, 3)

    def test_feature_importances(self, small_classification_data):
        """Test SHAP feature importances."""
        pytest.importorskip("shap")

        X, y = small_classification_data
        from sklearn.ensemble import GradientBoostingClassifier

        estimator = GradientBoostingClassifier(
            n_estimators=20,
            max_depth=3,
            random_state=42,
        )
        estimator.fit(X, y)

        from endgame.feature_selection import SHAPSelector
        selector = SHAPSelector(
            estimator=estimator,
            n_features=3,
            max_samples=50,
            random_state=42,
        )
        selector.fit(X)

        assert hasattr(selector, "feature_importances_")
        assert hasattr(selector, "shap_values_")


# ============================================================================
# Integration Tests
# ============================================================================


class TestSelectorPipelineCompatibility:
    """Test that selectors work in sklearn pipelines."""

    def test_in_pipeline(self, classification_data):
        """Test selectors in sklearn Pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = classification_data

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("selector", MRMRSelector(n_features=5, random_state=42)),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ])

        pipe.fit(X, y)
        preds = pipe.predict(X)

        assert len(preds) == len(y)

    def test_selector_transform(self, classification_data):
        """Test transform on new data."""
        X, y = classification_data

        # Split
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]

        selector = TreeImportanceSelector(n_features=5, random_state=42)
        selector.fit(X_train, y_train)

        X_test_selected = selector.transform(X_test)

        assert X_test_selected.shape == (50, 5)


class TestSelectorConsistency:
    """Test selector consistency across calls."""

    def test_deterministic_with_seed(self, classification_data):
        """Test that same seed produces same results."""
        X, y = classification_data

        selector1 = MRMRSelector(n_features=5, random_state=42)
        selector1.fit(X, y)

        selector2 = MRMRSelector(n_features=5, random_state=42)
        selector2.fit(X, y)

        np.testing.assert_array_equal(
            selector1.selected_features_,
            selector2.selected_features_,
        )


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_more_features_than_samples(self):
        """Test with more features than samples."""
        X = np.random.randn(50, 100)
        y = np.random.randint(0, 2, 50)

        selector = TreeImportanceSelector(n_features=10, random_state=42)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape[0] == 50
        assert X_selected.shape[1] <= 100  # fewer than original features

    def test_single_feature_selection(self, classification_data):
        """Test selecting single feature."""
        X, y = classification_data

        selector = UnivariateSelector(k=1)
        X_selected = selector.fit_transform(X, y)

        assert X_selected.shape == (200, 1)

    def test_empty_selection_fallback(self, classification_data):
        """Test that selectors handle edge case of no features meeting criteria."""
        X, y = classification_data

        # StabilitySelector with very high threshold
        base_selector = TreeImportanceSelector(n_features=3, random_state=42)
        selector = StabilitySelector(
            base_selector=base_selector,
            n_bootstrap=10,
            threshold=0.99,  # Very high threshold
            random_state=42,
        )
        X_selected = selector.fit_transform(X, y)

        # Should fallback to at least one feature
        assert X_selected.shape[1] >= 1
