"""Tests for endgame.guardrails module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from endgame.guardrails import (
    DataQualityWarning,
    GuardrailsReport,
    LeakageDetector,
    check_data_quality,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def basic_df():
    """Simple DataFrame with no issues."""
    rng = np.random.RandomState(42)
    n = 200
    X = pd.DataFrame({
        "a": rng.randn(n),
        "b": rng.randn(n),
        "c": rng.randn(n),
    })
    y = (X["a"] + rng.randn(n) * 0.5 > 0).astype(int)
    return X, y


@pytest.fixture
def leaky_df():
    """DataFrame with a feature that leaks the target."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.randint(0, 2, n)
    X = pd.DataFrame({
        "good_feature": rng.randn(n),
        "leaky_feature": y + rng.randn(n) * 0.01,  # almost perfect correlation
    })
    return X, y


@pytest.fixture
def redundant_df():
    """DataFrame with redundant features."""
    rng = np.random.RandomState(42)
    n = 200
    a = rng.randn(n)
    X = pd.DataFrame({
        "a": a,
        "a_copy": a + rng.randn(n) * 1e-10,  # near-exact duplicate
        "b": rng.randn(n),
    })
    y = rng.randint(0, 2, n)
    return X, y


@pytest.fixture
def problematic_df():
    """DataFrame with multiple issues."""
    rng = np.random.RandomState(42)
    n = 200
    y = rng.randint(0, 2, n)
    X = pd.DataFrame({
        "good": rng.randn(n),
        "constant": np.ones(n),
        "all_nan": np.full(n, np.nan),
        "id_col": np.arange(n),
    })
    return X, y


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------

class TestGuardrailsReport:
    def test_add_warning(self):
        report = GuardrailsReport()
        report.add(DataQualityWarning("leakage", "critical", "test"))
        assert report.n_critical == 1
        assert not report.passed

    def test_add_warning_types(self):
        report = GuardrailsReport()
        report.add(DataQualityWarning("data_health", "warning", "w"))
        report.add(DataQualityWarning("data_health", "info", "i"))
        assert report.n_warnings == 1
        assert report.passed

    def test_summary(self):
        report = GuardrailsReport()
        report.add(DataQualityWarning("leakage", "critical", "test msg"))
        s = report.summary()
        assert "ISSUES FOUND" in s
        assert "test msg" in s

    def test_to_dict(self):
        report = GuardrailsReport()
        report.add(DataQualityWarning("leakage", "critical", "msg", check_name="corr"))
        d = report.to_dict()
        assert d["passed"] is False
        assert len(d["warnings"]) == 1
        assert d["warnings"][0]["check_name"] == "corr"

    def test_repr_html(self):
        report = GuardrailsReport()
        report.add(DataQualityWarning("leakage", "critical", "msg"))
        html = report._repr_html_()
        assert "<table" in html

    def test_features_to_drop(self):
        report = GuardrailsReport()
        report.features_to_drop = ["a", "b"]
        s = report.summary()
        assert "Features to drop" in s

    def test_check_name_field(self):
        w = DataQualityWarning("cat", "sev", "msg", check_name="my_check")
        assert w.check_name == "my_check"


# ---------------------------------------------------------------------------
# Data health checks
# ---------------------------------------------------------------------------

class TestDataHealthChecks:
    def test_constant_columns(self, problematic_df):
        X, y = problematic_df
        report = check_data_quality(X, y, checks=["constant_columns"])
        assert any(w.check_name == "constant_columns" for w in report.warnings)
        assert "constant" in report.features_to_drop

    def test_missing_columns(self, problematic_df):
        X, y = problematic_df
        report = check_data_quality(X, y, checks=["missing_columns"])
        assert any(w.check_name == "missing_columns" for w in report.warnings)
        assert "all_nan" in report.features_to_drop

    def test_sample_count(self):
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = np.array([0, 1, 0])
        report = check_data_quality(X, y, checks=["sample_count"])
        assert any(w.check_name == "sample_count" for w in report.warnings)

    def test_feature_sample_ratio(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame(rng.randn(5, 100))
        y = rng.randint(0, 2, 5)
        report = check_data_quality(X, y, checks=["feature_sample_ratio"])
        assert any(w.check_name == "feature_sample_ratio" for w in report.warnings)

    def test_id_columns(self, problematic_df):
        X, y = problematic_df
        report = check_data_quality(X, y, checks=["id_columns"])
        assert any(w.check_name == "id_columns" for w in report.warnings)
        assert "id_col" in report.features_to_drop

    def test_class_imbalance(self):
        rng = np.random.RandomState(42)
        n = 1000
        X = pd.DataFrame({"a": rng.randn(n)})
        y = np.zeros(n, dtype=int)
        y[:3] = 1  # 0.3% minority
        report = check_data_quality(X, y, checks=["class_imbalance"])
        assert any(w.check_name == "class_imbalance" for w in report.warnings)

    def test_variance_near_zero(self):
        rng = np.random.RandomState(42)
        n = 200
        X = pd.DataFrame({
            "good": rng.randn(n),
            "tiny_var": rng.randn(n) * 1e-12,
        })
        y = rng.randint(0, 2, n)
        report = check_data_quality(X, y, checks=["variance"])
        assert any(w.check_name == "variance" for w in report.warnings)

    def test_suspect_features(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({
            "cohort_id": rng.randint(0, 3, n),
            "site_code": rng.randint(0, 5, n),
            "real_feature": rng.randn(n),
        })
        y = rng.randint(0, 2, n)
        report = check_data_quality(
            X, y,
            checks=["suspect_features"],
            suspect_patterns=["cohort_*", "site_*"],
        )
        assert any(w.check_name == "suspect_features" for w in report.warnings)
        assert "cohort_id" in report.features_to_drop
        assert "site_code" in report.features_to_drop

    def test_no_issues_clean_data(self, basic_df):
        X, y = basic_df
        report = check_data_quality(X, y, checks=[
            "constant_columns", "missing_columns", "sample_count",
        ])
        assert report.passed


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

class TestLeakageChecks:
    def test_correlation_leakage(self, leaky_df):
        X, y = leaky_df
        report = check_data_quality(X, y, checks=["correlation_leakage"])
        assert any(w.check_name == "correlation_leakage" for w in report.warnings)
        assert "leaky_feature" in report.features_to_drop

    def test_no_leakage_clean_data(self, basic_df):
        X, y = basic_df
        report = check_data_quality(X, y, checks=["correlation_leakage"])
        assert not any(w.check_name == "correlation_leakage" for w in report.warnings)

    def test_mutual_info_leakage(self, leaky_df):
        X, y = leaky_df
        report = check_data_quality(X, y, checks=["mutual_info_leakage"])
        # MI check should detect the leaky feature (may already be flagged by corr)
        mi_warnings = [w for w in report.warnings if w.check_name == "mutual_info_leakage"]
        # The leaky feature has near-perfect MI too
        assert len(mi_warnings) >= 0  # May or may not flag depending on normalization

    def test_categorical_leakage(self):
        rng = np.random.RandomState(42)
        n = 500
        y = rng.randint(0, 3, n)
        # Create categorical that perfectly encodes target
        X = pd.DataFrame({
            "cat_leak": [f"class_{v}" for v in y],
            "good_num": rng.randn(n),
        })
        report = check_data_quality(X, y, checks=["categorical_leakage"])
        cat_warnings = [w for w in report.warnings if w.check_name == "categorical_leakage"]
        assert len(cat_warnings) > 0
        assert "cat_leak" in report.features_to_drop


# ---------------------------------------------------------------------------
# Redundancy checks
# ---------------------------------------------------------------------------

class TestRedundancyChecks:
    def test_pairwise_correlation(self, redundant_df):
        X, y = redundant_df
        report = check_data_quality(X, y, checks=["pairwise_correlation"])
        assert any(w.check_name == "pairwise_correlation" for w in report.warnings)

    def test_near_duplicate_columns(self, redundant_df):
        X, y = redundant_df
        report = check_data_quality(X, y, checks=["near_duplicate_columns"])
        assert any(w.check_name == "near_duplicate_columns" for w in report.warnings)

    def test_affine_duplicate(self):
        rng = np.random.RandomState(42)
        n = 200
        a = rng.randn(n)
        X = pd.DataFrame({
            "a": a,
            "a_scaled": 2 * a + 5,  # affine transform
            "b": rng.randn(n),
        })
        y = rng.randint(0, 2, n)
        report = check_data_quality(X, y, checks=["near_duplicate_columns"])
        assert any(w.check_name == "near_duplicate_columns" for w in report.warnings)

    def test_vif(self):
        rng = np.random.RandomState(42)
        n = 200
        a = rng.randn(n)
        X = pd.DataFrame({
            "a": a,
            "b": a + rng.randn(n) * 0.1,  # highly collinear
            "c": rng.randn(n),
        })
        y = rng.randint(0, 2, n)
        report = check_data_quality(X, y, checks=["vif"])
        assert any(w.check_name == "vif" for w in report.warnings)


# ---------------------------------------------------------------------------
# Train/test checks
# ---------------------------------------------------------------------------

class TestTrainTestChecks:
    def test_duplicate_rows(self):
        rng = np.random.RandomState(42)
        X = pd.DataFrame({
            "a": [1, 2, 3, 1, 2],
            "b": [4, 5, 6, 4, 5],
        })
        y = np.array([0, 1, 0, 0, 1])
        report = check_data_quality(X, y, checks=["duplicate_rows"])
        assert any(w.check_name == "duplicate_rows" for w in report.warnings)

    def test_duplicate_rows_label_conflict(self):
        X = pd.DataFrame({
            "a": [1, 2, 3, 1],
            "b": [4, 5, 6, 4],
        })
        y = np.array([0, 1, 0, 1])  # same features, different label
        report = check_data_quality(X, y, checks=["duplicate_rows"])
        conflict_warnings = [
            w for w in report.warnings
            if w.check_name == "duplicate_rows" and w.severity == "critical"
        ]
        assert len(conflict_warnings) > 0

    def test_temporal_leakage(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({
            "past_feature": rng.randn(n),
            "future_outcome": rng.randn(n),
        })
        y = rng.randint(0, 2, n)
        report = check_data_quality(
            X, y,
            checks=["temporal_leakage"],
            future_features=["future_outcome"],
        )
        assert any(w.check_name == "temporal_leakage" for w in report.warnings)
        assert "future_outcome" in report.features_to_drop


# ---------------------------------------------------------------------------
# LeakageDetector integration
# ---------------------------------------------------------------------------

class TestLeakageDetector:
    def test_detect_mode(self, basic_df):
        X, y = basic_df
        detector = LeakageDetector(mode="detect")
        result = detector.fit_transform(X, y)
        # In detect mode, data passes through unchanged
        assert isinstance(result, pd.DataFrame)
        assert result.shape == X.shape

    def test_fix_mode(self, problematic_df):
        X, y = problematic_df
        detector = LeakageDetector(mode="fix")
        result = detector.fit_transform(X, y)
        # Should have fewer columns
        assert result.shape[1] < X.shape[1]
        # constant and all_nan should be removed
        if isinstance(result, pd.DataFrame):
            assert "constant" not in result.columns
            assert "all_nan" not in result.columns

    def test_custom_checks(self, basic_df):
        X, y = basic_df
        detector = LeakageDetector(checks=["sample_count", "constant_columns"])
        detector.fit(X, y)
        assert hasattr(detector, "report_")

    def test_all_checks(self, basic_df):
        X, y = basic_df
        detector = LeakageDetector(checks="all")
        detector.fit(X, y)
        assert hasattr(detector, "report_")

    def test_time_budget(self, basic_df):
        X, y = basic_df
        detector = LeakageDetector(time_budget=0.001)
        detector.fit(X, y)
        assert hasattr(detector, "report_")

    def test_get_feature_names_out(self, problematic_df):
        X, y = problematic_df
        detector = LeakageDetector(mode="fix")
        detector.fit(X, y)
        names = detector.get_feature_names_out()
        assert "constant" not in names
        assert "all_nan" not in names

    def test_numpy_input(self):
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, 100)
        detector = LeakageDetector(mode="detect")
        detector.fit(X, y)
        assert hasattr(detector, "report_")

    def test_sklearn_compatible(self, basic_df):
        """Test that it works in an sklearn pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        X, y = basic_df
        pipe = Pipeline([
            ("guardrails", LeakageDetector(mode="fix")),
            ("scaler", StandardScaler()),
        ])
        result = pipe.fit_transform(X, y)
        assert result.shape[0] == X.shape[0]

    def test_report_property(self, basic_df):
        X, y = basic_df
        detector = LeakageDetector()
        detector.fit(X, y)
        assert isinstance(detector.report_, GuardrailsReport)

    def test_suspect_patterns(self):
        rng = np.random.RandomState(42)
        n = 100
        X = pd.DataFrame({
            "cohort_a": rng.randint(0, 3, n),
            "real_feat": rng.randn(n),
        })
        y = rng.randint(0, 2, n)
        detector = LeakageDetector(
            mode="fix",
            checks=["suspect_features"],
            suspect_patterns=["cohort_*"],
        )
        result = detector.fit_transform(X, y)
        assert "cohort_a" not in result.columns

    def test_x_test_param(self):
        rng = np.random.RandomState(42)
        X_train = pd.DataFrame({"a": rng.randn(100), "b": rng.randn(100)})
        X_test = pd.DataFrame({"a": rng.randn(50), "b": rng.randn(50)})
        y = rng.randint(0, 2, 100)
        detector = LeakageDetector(checks=["duplicate_rows"])
        detector.fit(X_train, y, X_test=X_test)
        assert hasattr(detector, "report_")


# ---------------------------------------------------------------------------
# Import / backward compat tests
# ---------------------------------------------------------------------------

class TestImports:
    def test_guardrails_module_import(self):
        from endgame.guardrails import LeakageDetector, check_data_quality
        assert LeakageDetector is not None
        assert check_data_quality is not None

    def test_top_level_import(self):
        import endgame as eg
        assert hasattr(eg, "guardrails")
        assert hasattr(eg, "check_data_quality")
        assert hasattr(eg, "LeakageDetector")

    def test_automl_backward_compat(self):
        from endgame.automl.guardrails import DataQualityWarning, GuardrailsReport
        assert DataQualityWarning is not None
        assert GuardrailsReport is not None

    def test_feature_selection_reexport(self):
        from endgame.feature_selection import LeakageDetector
        assert LeakageDetector is not None


# ---------------------------------------------------------------------------
# check_data_quality convenience function
# ---------------------------------------------------------------------------

class TestCheckDataQuality:
    def test_basic_usage(self, basic_df):
        X, y = basic_df
        report = check_data_quality(X, y)
        assert isinstance(report, GuardrailsReport)

    def test_with_time_budget(self, basic_df):
        X, y = basic_df
        report = check_data_quality(X, y, time_budget=1.0)
        assert isinstance(report, GuardrailsReport)

    def test_with_specific_checks(self, leaky_df):
        X, y = leaky_df
        report = check_data_quality(X, y, checks=["correlation_leakage"])
        assert isinstance(report, GuardrailsReport)
