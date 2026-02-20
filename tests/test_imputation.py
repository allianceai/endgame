"""Tests for the imputation module."""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_data_with_missing(n_samples=50, n_features=4, missing_frac=0.15, seed=42):
    """Create a random array with a controlled fraction of NaN values."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    n_missing = int(n_samples * n_features * missing_frac)
    rows = rng.randint(0, n_samples, size=n_missing)
    cols = rng.randint(0, n_features, size=n_missing)
    X[rows, cols] = np.nan
    return X


def _make_low_missing(seed=42):
    """Data with <5% missing (triggers SimpleImputer in auto mode)."""
    return _make_data_with_missing(n_samples=100, n_features=4, missing_frac=0.01, seed=seed)


def _make_high_missing(seed=42):
    """Data with >30% missing (triggers MICEImputer in auto mode)."""
    return _make_data_with_missing(n_samples=50, n_features=4, missing_frac=0.40, seed=seed)


# ---------------------------------------------------------------------------
# SimpleImputer
# ---------------------------------------------------------------------------

class TestSimpleImputer:
    """Tests for SimpleImputer."""

    def test_median_strategy(self):
        """Default median strategy fills NaN with column medians."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan]])
        imp = SimpleImputer(strategy="median")
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()
        # Median of col 0 = 4.0, col 1 = 2.5
        assert X_out[1, 0] == pytest.approx(4.0)
        assert X_out[2, 1] == pytest.approx(2.5)

    def test_mean_strategy(self):
        """Mean strategy fills NaN with column means."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [5.0, np.nan]])
        imp = SimpleImputer(strategy="mean")
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()
        assert X_out[1, 0] == pytest.approx(3.0)
        assert X_out[2, 1] == pytest.approx(2.5)

    def test_constant_strategy(self):
        """Constant strategy fills NaN with fill_value."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = np.array([[1.0], [np.nan], [3.0]])
        imp = SimpleImputer(strategy="constant", fill_value=-999.0)
        X_out = imp.fit_transform(X)

        assert X_out[1, 0] == pytest.approx(-999.0)

    def test_preserves_shape(self):
        """Output shape matches input shape (without indicator)."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = _make_data_with_missing()
        imp = SimpleImputer()
        X_out = imp.fit_transform(X)

        assert X_out.shape == X.shape

    def test_no_missing_passthrough(self):
        """Data without NaN passes through unchanged."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        imp = SimpleImputer()
        X_out = imp.fit_transform(X)

        np.testing.assert_array_equal(X_out, X)

    def test_statistics_attribute(self):
        """statistics_ attribute is set after fit."""
        from endgame.preprocessing.imputation import SimpleImputer

        X = np.array([[1.0, 10.0], [np.nan, 20.0], [3.0, np.nan]])
        imp = SimpleImputer(strategy="median")
        imp.fit(X)

        assert hasattr(imp, "statistics_")
        assert len(imp.statistics_) == 2

    def test_pandas_input(self):
        """Accepts pandas DataFrame and returns DataFrame."""
        pd = pytest.importorskip("pandas")
        from endgame.preprocessing.imputation import SimpleImputer

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        imp = SimpleImputer()
        result = imp.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b"]
        assert not result.isna().any().any()

    def test_not_fitted_raises(self):
        """Calling transform before fit raises RuntimeError."""
        from endgame.preprocessing.imputation import SimpleImputer

        imp = SimpleImputer()
        with pytest.raises(RuntimeError, match="has not been fitted"):
            imp.transform(np.array([[1.0]]))


# ---------------------------------------------------------------------------
# IndicatorImputer
# ---------------------------------------------------------------------------

class TestIndicatorImputer:
    """Tests for IndicatorImputer."""

    def test_adds_indicator_columns(self):
        """Indicator columns are appended for features with missing values."""
        from endgame.preprocessing.imputation import IndicatorImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan]])
        imp = IndicatorImputer()
        X_out = imp.fit_transform(X)

        # 2 original + 2 indicator columns (both features have missing values)
        assert X_out.shape == (3, 4)

    def test_indicator_values(self):
        """Indicator columns contain 1 where missing, 0 otherwise."""
        from endgame.preprocessing.imputation import IndicatorImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan]])
        imp = IndicatorImputer()
        X_out = imp.fit_transform(X)

        # Indicator for col 0: [0, 1, 0]
        np.testing.assert_array_equal(X_out[:, 2], [0.0, 1.0, 0.0])
        # Indicator for col 1: [0, 0, 1]
        np.testing.assert_array_equal(X_out[:, 3], [0.0, 0.0, 1.0])

    def test_only_missing_flag(self):
        """only_missing=True only adds indicators for features that had NaN."""
        from endgame.preprocessing.imputation import IndicatorImputer

        # Only col 0 has missing values
        X = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]])
        imp = IndicatorImputer(only_missing=True)
        X_out = imp.fit_transform(X)

        # 2 original + 1 indicator (only col 0)
        assert X_out.shape == (3, 3)

    def test_only_missing_false(self):
        """only_missing=False adds indicators for all features."""
        from endgame.preprocessing.imputation import IndicatorImputer

        X = np.array([[np.nan, 2.0], [3.0, 4.0], [5.0, 6.0]])
        imp = IndicatorImputer(only_missing=False)
        X_out = imp.fit_transform(X)

        # 2 original + 2 indicators (all features)
        assert X_out.shape == (3, 4)

    def test_imputed_values_correct(self):
        """The imputed portion of the output is correctly filled."""
        from endgame.preprocessing.imputation import IndicatorImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [7.0, np.nan]])
        imp = IndicatorImputer(base_strategy="median")
        X_out = imp.fit_transform(X)

        # Median of col 0 = 4.0, col 1 = 2.5
        assert X_out[1, 0] == pytest.approx(4.0)
        assert X_out[2, 1] == pytest.approx(2.5)

    def test_pandas_column_names(self):
        """DataFrame output has correct column names including indicators."""
        pd = pytest.importorskip("pandas")
        from endgame.preprocessing.imputation import IndicatorImputer

        df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, np.nan]})
        imp = IndicatorImputer()
        result = imp.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert list(result.columns) == ["a", "b", "a_missing", "b_missing"]


# ---------------------------------------------------------------------------
# KNNImputer
# ---------------------------------------------------------------------------

class TestKNNImputer:
    """Tests for KNNImputer."""

    def test_basic_imputation(self):
        """KNN imputer fills missing values."""
        from endgame.preprocessing.imputation import KNNImputer

        X = np.array([
            [1.0, 2.0],
            [np.nan, 3.0],
            [7.0, 6.0],
            [5.0, np.nan],
            [3.0, 4.0],
        ])
        imp = KNNImputer(n_neighbors=2)
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()
        assert X_out.shape == X.shape

    def test_preserves_observed(self):
        """Observed (non-NaN) values are not changed."""
        from endgame.preprocessing.imputation import KNNImputer

        X = np.array([[1.0, 2.0], [np.nan, 3.0], [5.0, 4.0]])
        imp = KNNImputer(n_neighbors=2)
        X_out = imp.fit_transform(X)

        assert X_out[0, 0] == pytest.approx(1.0)
        assert X_out[0, 1] == pytest.approx(2.0)
        assert X_out[2, 0] == pytest.approx(5.0)

    def test_default_neighbors(self):
        """Default n_neighbors is 5."""
        from endgame.preprocessing.imputation import KNNImputer

        imp = KNNImputer()
        assert imp.n_neighbors == 5

    def test_pandas_input(self):
        """Accepts pandas DataFrame."""
        pd = pytest.importorskip("pandas")
        from endgame.preprocessing.imputation import KNNImputer

        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [2.0, 3.0, np.nan, 5.0, 6.0],
        })
        imp = KNNImputer(n_neighbors=2)
        result = imp.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()


# ---------------------------------------------------------------------------
# MICEImputer
# ---------------------------------------------------------------------------

class TestMICEImputer:
    """Tests for MICEImputer."""

    def test_basic_imputation(self):
        """MICE imputer fills missing values."""
        from endgame.preprocessing.imputation import MICEImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp = MICEImputer(max_iter=5, random_state=42)
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()
        assert X_out.shape == X.shape

    def test_n_iter_attribute(self):
        """n_iter_ attribute is set after fit."""
        from endgame.preprocessing.imputation import MICEImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp = MICEImputer(max_iter=5, random_state=42)
        imp.fit(X)

        assert hasattr(imp, "n_iter_")
        assert imp.n_iter_ <= 5

    def test_custom_estimator(self):
        """MICE imputer accepts a custom base estimator."""
        from sklearn.linear_model import Ridge
        from endgame.preprocessing.imputation import MICEImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp = MICEImputer(estimator=Ridge(), max_iter=3, random_state=42)
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()

    def test_deterministic_with_seed(self):
        """Same random_state produces identical results."""
        from endgame.preprocessing.imputation import MICEImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp1 = MICEImputer(max_iter=5, random_state=42)
        imp2 = MICEImputer(max_iter=5, random_state=42)

        X_out1 = imp1.fit_transform(X)
        X_out2 = imp2.fit_transform(X)

        np.testing.assert_array_almost_equal(X_out1, X_out2)


# ---------------------------------------------------------------------------
# MissForestImputer
# ---------------------------------------------------------------------------

class TestMissForestImputer:
    """Tests for MissForestImputer."""

    def test_basic_imputation(self):
        """MissForest imputer fills missing values."""
        from endgame.preprocessing.imputation import MissForestImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp = MissForestImputer(n_estimators=10, max_iter=3, random_state=42)
        X_out = imp.fit_transform(X)

        assert not np.isnan(X_out).any()
        assert X_out.shape == X.shape

    def test_preserves_observed(self):
        """Observed values remain unchanged."""
        from endgame.preprocessing.imputation import MissForestImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        observed_mask = ~np.isnan(X)

        imp = MissForestImputer(n_estimators=10, max_iter=3, random_state=42)
        X_out = imp.fit_transform(X)

        np.testing.assert_array_almost_equal(
            X_out[observed_mask], X[observed_mask]
        )

    def test_n_estimators_parameter(self):
        """n_estimators parameter controls tree count."""
        from endgame.preprocessing.imputation import MissForestImputer

        imp = MissForestImputer(n_estimators=50)
        assert imp.n_estimators == 50

    def test_separate_fit_transform(self):
        """Separate fit() and transform() produce valid results."""
        from endgame.preprocessing.imputation import MissForestImputer

        X_train = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1, seed=42)
        X_test = _make_data_with_missing(n_samples=10, n_features=3, missing_frac=0.1, seed=99)

        imp = MissForestImputer(n_estimators=10, max_iter=3, random_state=42)
        imp.fit(X_train)
        X_out = imp.transform(X_test)

        assert not np.isnan(X_out).any()
        assert X_out.shape == X_test.shape


# ---------------------------------------------------------------------------
# AutoImputer
# ---------------------------------------------------------------------------

class TestAutoImputer:
    """Tests for AutoImputer."""

    def test_auto_selects_simple_for_low_missing(self):
        """Auto mode selects SimpleImputer when <5% missing."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_low_missing()
        imp = AutoImputer(strategy="auto", random_state=42)
        imp.fit(X)

        assert imp.selected_strategy_ == "simple"
        assert imp.missingness_fraction_ < 0.05

    def test_auto_selects_knn_for_medium_missing(self):
        """Auto mode selects KNNImputer when 5-30% missing."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_data_with_missing(n_samples=50, n_features=4, missing_frac=0.15)
        imp = AutoImputer(strategy="auto", random_state=42)
        imp.fit(X)

        assert imp.selected_strategy_ == "knn"
        assert 0.05 <= imp.missingness_fraction_ <= 0.30

    def test_auto_selects_mice_for_high_missing(self):
        """Auto mode selects MICEImputer when >30% missing."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_high_missing()
        imp = AutoImputer(strategy="auto", random_state=42)
        imp.fit(X)

        assert imp.selected_strategy_ == "mice"
        assert imp.missingness_fraction_ > 0.30

    def test_manual_strategy_override(self):
        """Manual strategy overrides auto-selection."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_low_missing()  # Would auto-select 'simple'
        imp = AutoImputer(strategy="knn", random_state=42)
        X_out = imp.fit_transform(X)

        assert imp.selected_strategy_ == "knn"
        assert not np.isnan(X_out).any()

    def test_missforest_strategy(self):
        """Manual 'missforest' strategy works."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_data_with_missing(n_samples=30, n_features=3, missing_frac=0.1)
        imp = AutoImputer(strategy="missforest", random_state=42)
        X_out = imp.fit_transform(X)

        assert imp.selected_strategy_ == "missforest"
        assert not np.isnan(X_out).any()

    def test_missingness_type_attribute(self):
        """missingness_type_ is set to a valid value after fit."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_data_with_missing()
        imp = AutoImputer(random_state=42)
        imp.fit(X)

        assert imp.missingness_type_ in ("MCAR", "MAR", "MNAR")

    def test_littles_mcar_test_no_missing(self):
        """MCAR test returns MCAR with statistic=0 when no missing values."""
        from endgame.preprocessing.imputation import AutoImputer

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        mtype, stat = AutoImputer._littles_mcar_test_approx(X)

        assert mtype == "MCAR"
        assert stat == 0.0

    def test_invalid_strategy_raises(self):
        """Invalid strategy string raises ValueError."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_data_with_missing()
        imp = AutoImputer(strategy="bogus")
        with pytest.raises(ValueError, match="Unknown strategy"):
            imp.fit(X)

    def test_fit_transform_no_nan(self):
        """AutoImputer handles data with no missing values gracefully."""
        from endgame.preprocessing.imputation import AutoImputer

        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        imp = AutoImputer(random_state=42)
        X_out = imp.fit_transform(X)

        np.testing.assert_array_equal(X_out, X)
        assert imp.missingness_fraction_ == 0.0

    def test_pandas_roundtrip(self):
        """AutoImputer preserves DataFrame when given DataFrame input."""
        pd = pytest.importorskip("pandas")
        from endgame.preprocessing.imputation import AutoImputer

        df = pd.DataFrame({
            "a": [1.0, np.nan, 3.0, 4.0, 5.0],
            "b": [2.0, 3.0, np.nan, 5.0, 6.0],
        })
        imp = AutoImputer(strategy="simple", random_state=42)
        result = imp.fit_transform(df)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any()

    def test_custom_thresholds(self):
        """Custom low/high thresholds are respected."""
        from endgame.preprocessing.imputation import AutoImputer

        X = _make_data_with_missing(n_samples=50, n_features=4, missing_frac=0.10)
        # Set thresholds so 10% triggers MICE instead of KNN
        imp = AutoImputer(
            strategy="auto",
            low_threshold=0.02,
            high_threshold=0.08,
            random_state=42,
        )
        imp.fit(X)

        assert imp.selected_strategy_ == "mice"
