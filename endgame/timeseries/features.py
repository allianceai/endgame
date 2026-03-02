from __future__ import annotations

"""Time series feature extraction.

This module provides feature extraction utilities for time series data,
designed to feed into GBDT models for Kaggle competitions.

Integrates with tsfresh for comprehensive feature extraction and provides
custom extractors optimized for competition use.

Installation
------------
pip install tsfresh

Examples
--------
>>> from endgame.timeseries import TSFreshFeatureExtractor
>>> extractor = TSFreshFeatureExtractor(preset="efficient")
>>> features = extractor.fit_transform(time_series_df, column_id="id", column_sort="time")
"""

from typing import Literal

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check for tsfresh availability
try:
    import tsfresh
    from tsfresh import extract_features, select_features
    from tsfresh.feature_extraction import (
        ComprehensiveFCParameters,
        EfficientFCParameters,
        MinimalFCParameters,
    )
    from tsfresh.utilities.dataframe_functions import impute
    HAS_TSFRESH = True
except ImportError:
    HAS_TSFRESH = False

from sklearn.base import BaseEstimator, TransformerMixin


def _check_tsfresh():
    """Raise ImportError if tsfresh is not installed."""
    if not HAS_TSFRESH:
        raise ImportError(
            "tsfresh is required for TSFreshFeatureExtractor. "
            "Install with: pip install endgame-ml[timeseries]"
        )


class TSFreshFeatureExtractor(BaseEstimator, TransformerMixin):
    """Sklearn-compatible tsfresh feature extractor.

    Wraps tsfresh's feature extraction with automatic relevance filtering
    and imputation.

    Parameters
    ----------
    preset : str, default="efficient"
        Feature calculation preset:
        - "minimal": ~10 features per column (fast)
        - "efficient": ~100 features per column (balanced)
        - "comprehensive": ~800 features per column (slow but thorough)
    column_id : str, default="id"
        Name of the column containing series identifiers.
    column_sort : str, optional
        Name of the column to sort by (e.g., timestamp).
    column_value : str, optional
        Name of the column containing values (if single-column series).
    select_relevant : bool, default=True
        Whether to filter features by relevance to target.
    fdr_level : float, default=0.05
        False discovery rate for feature selection.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 for all cores).
    show_warnings : bool, default=False
        Whether to show tsfresh warnings.

    Attributes
    ----------
    selected_features_ : List[str]
        Names of selected features after fitting.
    feature_settings_ : dict
        Feature calculation settings used.

    Examples
    --------
    >>> # Basic usage
    >>> extractor = TSFreshFeatureExtractor(preset="efficient")
    >>> features = extractor.fit_transform(df, y=target)

    >>> # Without relevance filtering
    >>> extractor = TSFreshFeatureExtractor(select_relevant=False)
    >>> features = extractor.transform(df)
    """

    def __init__(
        self,
        preset: Literal["minimal", "efficient", "comprehensive"] = "efficient",
        column_id: str = "id",
        column_sort: str | None = None,
        column_value: str | None = None,
        select_relevant: bool = True,
        fdr_level: float = 0.05,
        n_jobs: int = -1,
        show_warnings: bool = False,
    ):
        self.preset = preset
        self.column_id = column_id
        self.column_sort = column_sort
        self.column_value = column_value
        self.select_relevant = select_relevant
        self.fdr_level = fdr_level
        self.n_jobs = n_jobs
        self.show_warnings = show_warnings

        self.selected_features_: list[str] | None = None
        self.feature_settings_: dict | None = None
        self._is_fitted = False

    def _get_feature_settings(self):
        """Get feature calculation settings based on preset."""
        _check_tsfresh()

        if self.preset == "minimal":
            return MinimalFCParameters()
        elif self.preset == "efficient":
            return EfficientFCParameters()
        elif self.preset == "comprehensive":
            return ComprehensiveFCParameters()
        else:
            raise ValueError(f"Unknown preset: {self.preset}")

    def _to_pandas(self, X):
        """Convert input to pandas DataFrame."""
        if HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            return X.to_pandas()
        elif HAS_PANDAS and isinstance(X, pd.DataFrame):
            return X
        else:
            raise TypeError(
                f"Expected DataFrame, got {type(X)}. "
                "TSFreshFeatureExtractor requires tabular data."
            )

    def fit(
        self,
        X,
        y=None,
        **fit_params,
    ) -> TSFreshFeatureExtractor:
        """Fit the feature extractor.

        Parameters
        ----------
        X : DataFrame
            Input data with columns for id, sort, and values.
        y : array-like, optional
            Target variable for relevance filtering.
            Required if select_relevant=True.

        Returns
        -------
        self
            Fitted extractor.
        """
        _check_tsfresh()

        df = self._to_pandas(X)

        # Get feature settings
        self.feature_settings_ = self._get_feature_settings()

        # Extract features
        features = extract_features(
            df,
            column_id=self.column_id,
            column_sort=self.column_sort,
            column_value=self.column_value,
            default_fc_parameters=self.feature_settings_,
            n_jobs=self.n_jobs if self.n_jobs > 0 else None,
            disable_progressbar=not self.show_warnings,
        )

        # Impute missing values
        features = impute(features)

        # Select relevant features if target provided
        if self.select_relevant and y is not None:
            y = np.asarray(y)
            features = select_features(
                features,
                y,
                fdr_level=self.fdr_level,
            )
            self.selected_features_ = list(features.columns)
        else:
            self.selected_features_ = list(features.columns)

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Transform time series to features.

        Parameters
        ----------
        X : DataFrame
            Input data.

        Returns
        -------
        np.ndarray
            Extracted features.
        """
        if not self._is_fitted:
            raise RuntimeError("TSFreshFeatureExtractor has not been fitted")

        _check_tsfresh()

        df = self._to_pandas(X)

        # Extract features using fitted settings
        features = extract_features(
            df,
            column_id=self.column_id,
            column_sort=self.column_sort,
            column_value=self.column_value,
            default_fc_parameters=self.feature_settings_,
            n_jobs=self.n_jobs if self.n_jobs > 0 else None,
            disable_progressbar=not self.show_warnings,
        )

        # Impute missing values
        features = impute(features)

        # Select only fitted features
        if self.selected_features_:
            available = [f for f in self.selected_features_ if f in features.columns]
            features = features[available]

            # Add missing columns as zeros
            for f in self.selected_features_:
                if f not in features.columns:
                    features[f] = 0.0

            # Ensure column order
            features = features[self.selected_features_]

        return features.values

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if not self._is_fitted:
            raise RuntimeError("TSFreshFeatureExtractor has not been fitted")
        return self.selected_features_ or []


class TimeSeriesFeatureExtractor(BaseEstimator, TransformerMixin):
    """Fast time series feature extractor without tsfresh dependency.

    Extracts common time series features that are useful for GBDT models.
    Faster than tsfresh but with fewer features.

    Parameters
    ----------
    features : List[str], optional
        Features to extract. If None, extracts all.
        Options: "statistics", "trend", "seasonality", "autocorr", "entropy",
                 "spectral", "peaks", "crossings".
    window_sizes : List[int], default=[7, 14, 30]
        Window sizes for rolling features.
    lag_range : tuple, default=(1, 10)
        Range of lags for autocorrelation features.

    Examples
    --------
    >>> extractor = TimeSeriesFeatureExtractor(features=["statistics", "trend"])
    >>> features = extractor.fit_transform(time_series_array)
    """

    def __init__(
        self,
        features: list[str] | None = None,
        window_sizes: list[int] = [7, 14, 30],
        lag_range: tuple = (1, 10),
    ):
        self.features = features or [
            "statistics", "trend", "autocorr", "entropy", "peaks", "crossings"
        ]
        self.window_sizes = window_sizes
        self.lag_range = lag_range

        self.feature_names_: list[str] | None = None
        self._is_fitted = False

    def _extract_statistics(self, y: np.ndarray) -> dict[str, float]:
        """Extract basic statistical features."""
        features = {
            "mean": np.mean(y),
            "std": np.std(y),
            "min": np.min(y),
            "max": np.max(y),
            "median": np.median(y),
            "q25": np.percentile(y, 25),
            "q75": np.percentile(y, 75),
            "iqr": np.percentile(y, 75) - np.percentile(y, 25),
            "skewness": self._skewness(y),
            "kurtosis": self._kurtosis(y),
            "range": np.max(y) - np.min(y),
            "variation_coef": np.std(y) / np.mean(y) if np.mean(y) != 0 else 0,
        }
        return features

    def _skewness(self, y: np.ndarray) -> float:
        """Compute skewness."""
        n = len(y)
        if n < 3:
            return 0.0
        mean = np.mean(y)
        std = np.std(y)
        if std == 0:
            return 0.0
        return np.mean(((y - mean) / std) ** 3)

    def _kurtosis(self, y: np.ndarray) -> float:
        """Compute kurtosis."""
        n = len(y)
        if n < 4:
            return 0.0
        mean = np.mean(y)
        std = np.std(y)
        if std == 0:
            return 0.0
        return np.mean(((y - mean) / std) ** 4) - 3

    def _extract_trend(self, y: np.ndarray) -> dict[str, float]:
        """Extract trend-related features."""
        n = len(y)
        x = np.arange(n)

        # Linear regression
        slope, intercept = np.polyfit(x, y, 1)

        # Residuals
        trend_line = slope * x + intercept
        residuals = y - trend_line
        r_squared = 1 - np.sum(residuals ** 2) / np.sum((y - np.mean(y)) ** 2)

        features = {
            "trend_slope": slope,
            "trend_intercept": intercept,
            "trend_r_squared": max(0, r_squared),
            "trend_strength": abs(slope) / (np.std(y) + 1e-10),
            "first_value": y[0],
            "last_value": y[-1],
            "change": y[-1] - y[0],
            "pct_change": (y[-1] - y[0]) / (abs(y[0]) + 1e-10),
        }
        return features

    def _extract_autocorr(self, y: np.ndarray) -> dict[str, float]:
        """Extract autocorrelation features."""
        features = {}
        n = len(y)
        y_centered = y - np.mean(y)
        var = np.var(y)

        if var == 0:
            for lag in range(self.lag_range[0], min(self.lag_range[1] + 1, n)):
                features[f"autocorr_lag_{lag}"] = 0.0
            return features

        for lag in range(self.lag_range[0], min(self.lag_range[1] + 1, n)):
            if lag >= n:
                features[f"autocorr_lag_{lag}"] = 0.0
            else:
                acf = np.correlate(y_centered, y_centered, mode='full')[n - 1:]
                features[f"autocorr_lag_{lag}"] = acf[lag] / (acf[0] + 1e-10)

        # Partial autocorrelation (simplified)
        features["partial_autocorr_1"] = features.get("autocorr_lag_1", 0.0)

        return features

    def _extract_entropy(self, y: np.ndarray) -> dict[str, float]:
        """Extract entropy-based features."""
        # Sample entropy (simplified)
        n = len(y)
        if n < 10:
            return {"sample_entropy": 0.0, "approx_entropy": 0.0}

        # Binned entropy
        bins = min(10, n // 5)
        hist, _ = np.histogram(y, bins=bins)
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        binned_entropy = -np.sum(hist * np.log(hist))

        # Permutation entropy (simplified)
        m = 3  # embedding dimension
        if n < m:
            perm_entropy = 0.0
        else:
            patterns = []
            for i in range(n - m + 1):
                pattern = tuple(np.argsort(y[i:i+m]))
                patterns.append(pattern)
            unique, counts = np.unique(patterns, axis=0, return_counts=True)
            probs = counts / len(patterns)
            perm_entropy = -np.sum(probs * np.log(probs + 1e-10))

        features = {
            "binned_entropy": binned_entropy,
            "permutation_entropy": perm_entropy,
        }
        return features

    def _extract_peaks(self, y: np.ndarray) -> dict[str, float]:
        """Extract peak-related features."""
        # Simple peak detection
        n = len(y)
        if n < 3:
            return {"n_peaks": 0, "n_troughs": 0}

        peaks = 0
        troughs = 0

        for i in range(1, n - 1):
            if y[i] > y[i-1] and y[i] > y[i+1]:
                peaks += 1
            if y[i] < y[i-1] and y[i] < y[i+1]:
                troughs += 1

        features = {
            "n_peaks": peaks,
            "n_troughs": troughs,
            "peaks_per_sample": peaks / n,
            "troughs_per_sample": troughs / n,
        }
        return features

    def _extract_crossings(self, y: np.ndarray) -> dict[str, float]:
        """Extract zero/mean crossing features."""
        n = len(y)
        mean_val = np.mean(y)
        y_centered = y - mean_val

        # Mean crossings
        mean_crossings = np.sum(np.diff(np.sign(y_centered)) != 0)

        # Zero crossings (if not already centered)
        zero_crossings = np.sum(np.diff(np.sign(y)) != 0)

        features = {
            "mean_crossings": mean_crossings,
            "zero_crossings": zero_crossings,
            "mean_crossing_rate": mean_crossings / (n - 1),
        }
        return features

    def _extract_for_series(self, y: np.ndarray) -> np.ndarray:
        """Extract all features for a single series."""
        all_features = {}

        if "statistics" in self.features:
            all_features.update(self._extract_statistics(y))

        if "trend" in self.features:
            all_features.update(self._extract_trend(y))

        if "autocorr" in self.features:
            all_features.update(self._extract_autocorr(y))

        if "entropy" in self.features:
            all_features.update(self._extract_entropy(y))

        if "peaks" in self.features:
            all_features.update(self._extract_peaks(y))

        if "crossings" in self.features:
            all_features.update(self._extract_crossings(y))

        # Sort by key for consistent ordering
        sorted_features = dict(sorted(all_features.items()))
        return np.array(list(sorted_features.values())), list(sorted_features.keys())

    def fit(self, X, y=None, **fit_params) -> TimeSeriesFeatureExtractor:
        """Fit the extractor (learn feature names).

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_samples, n_series)
            Time series data.

        Returns
        -------
        self
            Fitted extractor.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Extract from first series to get feature names
        _, self.feature_names_ = self._extract_for_series(X[:, 0])

        self._is_fitted = True
        return self

    def transform(self, X) -> np.ndarray:
        """Extract features from time series.

        Parameters
        ----------
        X : array-like of shape (n_samples,) or (n_series, n_samples)
            Time series data. If 2D, each row is a separate series.

        Returns
        -------
        np.ndarray
            Extracted features of shape (n_series, n_features).
        """
        if not self._is_fitted:
            raise RuntimeError("TimeSeriesFeatureExtractor has not been fitted")

        X = np.asarray(X)

        # Handle single series
        if X.ndim == 1:
            features, _ = self._extract_for_series(X)
            return features.reshape(1, -1)

        # Handle multiple series (each row is a series)
        if X.shape[0] > X.shape[1]:
            # Likely (n_samples, n_series) - transpose
            X = X.T

        n_series = X.shape[0]
        all_features = []

        for i in range(n_series):
            features, _ = self._extract_for_series(X[i])
            all_features.append(features)

        return np.array(all_features)

    def fit_transform(self, X, y=None, **fit_params) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y, **fit_params).transform(X)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names."""
        if not self._is_fitted:
            raise RuntimeError("TimeSeriesFeatureExtractor has not been fitted")
        return self.feature_names_ or []
