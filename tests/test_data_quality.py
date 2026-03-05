from __future__ import annotations

"""Tests for the endgame.data_quality module."""

import numpy as np
import pandas as pd
import pytest

from endgame.data_quality import (
    DataProfile,
    DataProfiler,
    DriftDetector,
    DriftReport,
    DuplicateDetector,
    DuplicateReport,
    DataValuator,
    FeatureDriftResult,
    profile_data,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def clean_df():
    """200 rows, 3 numeric + 1 categorical, no duplicates or NaN."""
    rng = np.random.RandomState(42)
    return pd.DataFrame({
        "num1": rng.randn(200),
        "num2": rng.rand(200) * 100,
        "num3": rng.randint(0, 50, 200).astype(float),
        "cat1": rng.choice(["a", "b", "c", "d"], 200),
    })


@pytest.fixture
def messy_df():
    """100 rows with duplicates, NaN, and outliers."""
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "num1": rng.randn(n),
        "num2": rng.rand(n),
        "cat1": rng.choice(["x", "y"], n),
    })
    # Add NaN
    df.loc[0:4, "num1"] = np.nan
    # Add outliers
    df.loc[95:99, "num2"] = 1000.0
    # Add exact duplicates
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    return df


# ---------------------------------------------------------------------------
# TestDataProfiler
# ---------------------------------------------------------------------------

class TestDataProfiler:
    def test_profile_returns_dataprofile(self, clean_df):
        profiler = DataProfiler()
        result = profiler.profile(clean_df)
        assert isinstance(result, DataProfile)

    def test_per_column_stats(self, clean_df):
        result = DataProfiler().profile(clean_df)
        assert "num1" in result.column_stats
        stats = result.column_stats["num1"]
        assert stats["dtype"] == "numeric"
        assert stats["missing_count"] == 0
        assert stats["n_unique"] > 0
        assert stats["mean"] is not None

    def test_missing_detection(self, messy_df):
        result = DataProfiler().profile(messy_df)
        stats = result.column_stats["num1"]
        assert stats["missing_count"] == 10  # 5 original + 5 from duplicated rows

    def test_categorical_handling(self, clean_df):
        result = DataProfiler().profile(clean_df)
        assert "cat1" in result.column_stats
        stats = result.column_stats["cat1"]
        assert stats["dtype"] == "categorical"
        assert stats["n_unique"] == 4
        assert "top_values" in stats

    def test_correlation_matrix(self, clean_df):
        result = DataProfiler(correlation=True).profile(clean_df)
        assert result.correlation_matrix is not None
        assert result.correlation_matrix.shape == (3, 3)

    def test_no_correlation_by_default(self, clean_df):
        result = DataProfiler().profile(clean_df)
        assert result.correlation_matrix is None

    def test_to_dataframe(self, clean_df):
        result = DataProfiler().profile(clean_df)
        df = result.to_dataframe()
        assert len(df) == 4  # 3 numeric + 1 categorical

    def test_repr_html(self, clean_df):
        result = DataProfiler().profile(clean_df)
        html = result._repr_html_()
        assert "<table>" in html
        assert "num1" in html

    def test_summary_stats(self, clean_df):
        result = DataProfiler().profile(clean_df)
        s = result.summary_stats
        assert s["n_rows"] == 200
        assert s["n_cols"] == 4
        assert s["n_numeric"] == 3
        assert s["n_categorical"] == 1
        assert s["total_missing_pct"] == 0.0

    def test_numpy_input(self):
        X = np.random.randn(50, 3)
        result = DataProfiler().profile(X)
        assert result.summary_stats["n_cols"] == 3
        assert result.summary_stats["n_numeric"] == 3

    def test_repr(self, clean_df):
        result = DataProfiler().profile(clean_df)
        text = repr(result)
        assert "200 rows" in text

    def test_outlier_detection(self, messy_df):
        result = DataProfiler().profile(messy_df)
        stats = result.column_stats["num2"]
        assert stats["n_outliers"] > 0


# ---------------------------------------------------------------------------
# TestDuplicateDetector
# ---------------------------------------------------------------------------

class TestDuplicateDetector:
    def test_exact_detection(self, messy_df):
        detector = DuplicateDetector(method="exact")
        report = detector.fit_detect(messy_df)
        assert isinstance(report, DuplicateReport)
        assert report.n_duplicates == 10
        assert report.method == "exact"

    def test_no_false_positives(self, clean_df):
        detector = DuplicateDetector(method="exact")
        report = detector.fit_detect(clean_df)
        assert report.n_duplicates == 0
        assert report.duplicate_groups == 0

    def test_fuzzy_detection(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        # Add near-duplicates
        X_dup = np.vstack([X, X[:5] + 1e-10])
        detector = DuplicateDetector(method="fuzzy", threshold=0.999)
        report = detector.fit_detect(X_dup)
        assert report.n_duplicates >= 5

    def test_transform_removes_dups(self, messy_df):
        detector = DuplicateDetector(method="exact")
        detector.fit(messy_df)
        result = detector.transform(messy_df)
        assert len(result) == len(messy_df) - 10

    def test_sklearn_pipeline_compat(self):
        from sklearn.pipeline import Pipeline
        X = np.array([[1, 2], [3, 4], [1, 2], [5, 6]])
        pipe = Pipeline([("dedup", DuplicateDetector(method="exact"))])
        result = pipe.fit_transform(X)
        assert len(result) == 3

    def test_repr(self, messy_df):
        report = DuplicateDetector().fit_detect(messy_df)
        text = repr(report)
        assert "10 duplicates" in text

    def test_repr_html(self, messy_df):
        report = DuplicateDetector().fit_detect(messy_df)
        html = report._repr_html_()
        assert "DuplicateReport" in html


# ---------------------------------------------------------------------------
# TestDriftDetector
# ---------------------------------------------------------------------------

class TestDriftDetector:
    def test_ks_no_drift(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(500, 3)
        X_test = rng.randn(500, 3)
        report = DriftDetector(method="ks").detect(X_ref, X_test)
        assert isinstance(report, DriftReport)
        assert len(report.drifted_features) == 0

    def test_ks_drift_detected(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(500, 3)
        X_test = rng.randn(500, 3) + 2.0  # Shift distribution
        report = DriftDetector(method="ks").detect(X_ref, X_test)
        assert len(report.drifted_features) == 3

    def test_psi_method(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(1000, 2)
        X_test = rng.randn(1000, 2) + 3.0
        report = DriftDetector(method="psi").detect(X_ref, X_test)
        assert report.method == "psi"
        assert len(report.drifted_features) > 0

    def test_chi2_categorical(self):
        ref = pd.DataFrame({"cat": np.random.choice(["a", "b"], 500)})
        test = pd.DataFrame({"cat": np.random.choice(["a", "b", "c"], 500)})
        report = DriftDetector(method="chi2").detect(ref, test)
        assert report.method == "chi2"
        assert len(report.feature_results) > 0

    def test_auto_mixed_types(self):
        rng = np.random.RandomState(42)
        ref = pd.DataFrame({
            "num": rng.randn(300),
            "cat": rng.choice(["a", "b"], 300),
        })
        test = pd.DataFrame({
            "num": rng.randn(300) + 5.0,
            "cat": rng.choice(["a", "b", "c"], 300),
        })
        report = DriftDetector(method="auto").detect(ref, test)
        assert len(report.feature_results) == 2

    def test_adversarial_smoke(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(200, 5)
        X_test = rng.randn(200, 5) + 3.0
        report = DriftDetector(method="adversarial").detect(X_ref, X_test)
        assert report.adversarial_auc is not None
        assert report.method == "adversarial"

    def test_repr_html(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(100, 2)
        X_test = rng.randn(100, 2) + 1.0
        report = DriftDetector(method="ks").detect(X_ref, X_test)
        html = report._repr_html_()
        assert "<table>" in html

    def test_to_dataframe(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(100, 2)
        X_test = rng.randn(100, 2)
        report = DriftDetector(method="ks").detect(X_ref, X_test)
        df = report.to_dataframe()
        assert len(df) == 2
        assert "feature" in df.columns

    def test_summary(self):
        rng = np.random.RandomState(42)
        X_ref = rng.randn(100, 2)
        X_test = rng.randn(100, 2)
        report = DriftDetector(method="ks").detect(X_ref, X_test)
        text = report.summary()
        assert "DriftReport" in text


# ---------------------------------------------------------------------------
# TestDataValuator
# ---------------------------------------------------------------------------

class TestDataValuator:
    @pytest.fixture
    def classification_data(self):
        rng = np.random.RandomState(42)
        X = rng.randn(50, 3)
        y = (X[:, 0] > 0).astype(int)
        return X, y

    def test_knn_shapley_length(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="knn_shapley")
        v.fit(X, y)
        assert v.values_.shape == (50,)
        assert not np.all(v.values_ == 0)

    def test_high_value_indices(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="knn_shapley").fit(X, y)
        top = v.get_high_value_indices(top_k=5)
        assert len(top) == 5
        # Should be sorted descending by value
        assert v.values_[top[0]] >= v.values_[top[-1]]

    def test_low_value_indices(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="knn_shapley").fit(X, y)
        bottom = v.get_low_value_indices(bottom_k=5)
        assert len(bottom) == 5
        assert v.values_[bottom[0]] <= v.values_[bottom[-1]]

    def test_tmc_shapley_smoke(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="tmc_shapley", n_mc_iterations=10, random_state=42)
        v.fit(X, y)
        assert v.values_.shape == (50,)

    def test_loo_smoke(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="loo", random_state=42)
        v.fit(X, y)
        assert v.values_.shape == (50,)

    def test_values_before_fit_raises(self):
        v = DataValuator()
        with pytest.raises(RuntimeError, match="fit"):
            _ = v.values

    def test_threshold_selection(self, classification_data):
        X, y = classification_data
        v = DataValuator(method="knn_shapley").fit(X, y)
        median_val = np.median(v.values_)
        high = v.get_high_value_indices(threshold=median_val)
        assert len(high) > 0
        assert all(v.values_[i] >= median_val for i in high)


# ---------------------------------------------------------------------------
# TestImports
# ---------------------------------------------------------------------------

class TestImports:
    def test_module_import(self):
        from endgame import data_quality
        assert hasattr(data_quality, "DataProfiler")
        assert hasattr(data_quality, "DuplicateDetector")
        assert hasattr(data_quality, "DriftDetector")
        assert hasattr(data_quality, "DataValuator")
        assert hasattr(data_quality, "profile_data")

    def test_lazy_import(self):
        import endgame as eg
        dq = eg.data_quality
        assert hasattr(dq, "DataProfiler")

    def test_convenience_function(self):
        X = np.random.randn(30, 3)
        result = profile_data(X)
        assert isinstance(result, DataProfile)
        assert result.summary_stats["n_rows"] == 30

    def test_top_level_profile_data(self):
        import endgame as eg
        assert callable(eg.profile_data)
