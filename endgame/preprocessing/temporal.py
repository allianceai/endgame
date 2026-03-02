from __future__ import annotations

"""Temporal feature extraction for time series and datetime columns."""

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.core.base import PolarsTransformer
from endgame.core.polars_ops import HAS_POLARS

if HAS_POLARS:
    import polars as pl


class TemporalFeatures(PolarsTransformer):
    """Extracts temporal features from datetime columns.

    Generates comprehensive datetime features including cyclical
    encodings for periodic patterns.

    Features generated:
    - Basic: year, month, day, dayofweek, hour, minute, second
    - Boolean: is_weekend, is_month_start, is_month_end, is_year_start, is_year_end
    - Derived: quarter, week_of_year, day_of_year
    - Cyclical: sin/cos encodings for month, day, hour, dayofweek

    Parameters
    ----------
    datetime_cols : List[str], optional
        Datetime columns to extract features from.
        If None, auto-detects datetime columns.
    features : List[str], optional
        Features to extract. If None, extracts all.
        Options: 'year', 'month', 'day', 'dayofweek', 'hour', 'minute',
        'second', 'is_weekend', 'quarter', 'week_of_year', 'day_of_year',
        'is_month_start', 'is_month_end', 'cyclical'.
    cyclical : bool, default=True
        Whether to add cyclical (sin/cos) encodings.
    drop_original : bool, default=False
        Whether to drop the original datetime columns.

    Examples
    --------
    >>> tf = TemporalFeatures(cyclical=True)
    >>> X_temporal = tf.fit_transform(X)
    """

    def __init__(
        self,
        datetime_cols: list[str] | None = None,
        features: list[str] | None = None,
        cyclical: bool = True,
        drop_original: bool = False,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.datetime_cols = datetime_cols
        self.features = features
        self.cyclical = cyclical
        self.drop_original = drop_original

        self._target_cols: list[str] = []
        self._new_feature_names: list[str] = []

    def _detect_datetime_cols(self, df: pl.DataFrame) -> list[str]:
        """Auto-detect datetime columns."""
        datetime_cols = []
        for col in df.columns:
            dtype = df[col].dtype
            if dtype in (pl.Datetime, pl.Date):
                datetime_cols.append(col)
        return datetime_cols

    def fit(self, X, y=None, **fit_params) -> TemporalFeatures:
        """Identify datetime columns."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        if self.datetime_cols is None:
            self._target_cols = self._detect_datetime_cols(df)
        else:
            self._target_cols = [c for c in self.datetime_cols if c in df.columns]

        self._log(f"Found {len(self._target_cols)} datetime columns")
        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Extract temporal features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        new_cols = []
        self._new_feature_names = []

        default_features = [
            'year', 'month', 'day', 'dayofweek', 'hour',
            'is_weekend', 'quarter', 'week_of_year', 'day_of_year'
        ]
        features_to_extract = self.features or default_features

        for col in self._target_cols:
            dt_col = pl.col(col)

            # Basic features
            if 'year' in features_to_extract:
                new_cols.append(dt_col.dt.year().alias(f"{col}_year"))
                self._new_feature_names.append(f"{col}_year")

            if 'month' in features_to_extract:
                new_cols.append(dt_col.dt.month().alias(f"{col}_month"))
                self._new_feature_names.append(f"{col}_month")

            if 'day' in features_to_extract:
                new_cols.append(dt_col.dt.day().alias(f"{col}_day"))
                self._new_feature_names.append(f"{col}_day")

            if 'dayofweek' in features_to_extract:
                new_cols.append(dt_col.dt.weekday().alias(f"{col}_dayofweek"))
                self._new_feature_names.append(f"{col}_dayofweek")

            if 'hour' in features_to_extract:
                new_cols.append(dt_col.dt.hour().alias(f"{col}_hour"))
                self._new_feature_names.append(f"{col}_hour")

            if 'minute' in features_to_extract:
                new_cols.append(dt_col.dt.minute().alias(f"{col}_minute"))
                self._new_feature_names.append(f"{col}_minute")

            if 'second' in features_to_extract:
                new_cols.append(dt_col.dt.second().alias(f"{col}_second"))
                self._new_feature_names.append(f"{col}_second")

            # Boolean features
            if 'is_weekend' in features_to_extract:
                new_cols.append(
                    (dt_col.dt.weekday() >= 5).cast(pl.Int8).alias(f"{col}_is_weekend")
                )
                self._new_feature_names.append(f"{col}_is_weekend")

            if 'is_month_start' in features_to_extract:
                new_cols.append(
                    (dt_col.dt.day() == 1).cast(pl.Int8).alias(f"{col}_is_month_start")
                )
                self._new_feature_names.append(f"{col}_is_month_start")

            if 'is_month_end' in features_to_extract:
                # Approximate: day >= 28
                new_cols.append(
                    (dt_col.dt.day() >= 28).cast(pl.Int8).alias(f"{col}_is_month_end")
                )
                self._new_feature_names.append(f"{col}_is_month_end")

            # Derived features
            if 'quarter' in features_to_extract:
                new_cols.append(dt_col.dt.quarter().alias(f"{col}_quarter"))
                self._new_feature_names.append(f"{col}_quarter")

            if 'week_of_year' in features_to_extract:
                new_cols.append(dt_col.dt.week().alias(f"{col}_week_of_year"))
                self._new_feature_names.append(f"{col}_week_of_year")

            if 'day_of_year' in features_to_extract:
                new_cols.append(dt_col.dt.ordinal_day().alias(f"{col}_day_of_year"))
                self._new_feature_names.append(f"{col}_day_of_year")

            # Cyclical encodings
            if self.cyclical or 'cyclical' in features_to_extract:
                # Month (1-12)
                month = dt_col.dt.month().cast(pl.Float64)
                new_cols.append(
                    (2 * np.pi * month / 12).sin().alias(f"{col}_month_sin")
                )
                new_cols.append(
                    (2 * np.pi * month / 12).cos().alias(f"{col}_month_cos")
                )
                self._new_feature_names.extend([f"{col}_month_sin", f"{col}_month_cos"])

                # Day of week (0-6)
                dow = dt_col.dt.weekday().cast(pl.Float64)
                new_cols.append(
                    (2 * np.pi * dow / 7).sin().alias(f"{col}_dow_sin")
                )
                new_cols.append(
                    (2 * np.pi * dow / 7).cos().alias(f"{col}_dow_cos")
                )
                self._new_feature_names.extend([f"{col}_dow_sin", f"{col}_dow_cos"])

                # Hour (0-23)
                hour = dt_col.dt.hour().cast(pl.Float64)
                new_cols.append(
                    (2 * np.pi * hour / 24).sin().alias(f"{col}_hour_sin")
                )
                new_cols.append(
                    (2 * np.pi * hour / 24).cos().alias(f"{col}_hour_cos")
                )
                self._new_feature_names.extend([f"{col}_hour_sin", f"{col}_hour_cos"])

                # Day of month (1-31)
                day = dt_col.dt.day().cast(pl.Float64)
                new_cols.append(
                    (2 * np.pi * day / 31).sin().alias(f"{col}_day_sin")
                )
                new_cols.append(
                    (2 * np.pi * day / 31).cos().alias(f"{col}_day_cos")
                )
                self._new_feature_names.extend([f"{col}_day_sin", f"{col}_day_cos"])

        if new_cols:
            result = df.with_columns(new_cols)
        else:
            result = df

        if self.drop_original:
            result = result.drop(self._target_cols)

        return self._from_lazyframe(result.lazy())


class LagFeatures(PolarsTransformer):
    """Generate lag features for time series data.

    Creates shifted versions of features to capture temporal dependencies.

    Parameters
    ----------
    cols : List[str], optional
        Columns to create lags for. If None, uses all numeric columns.
    lags : List[int], default=[1, 2, 3]
        Lag periods to create.
    group_cols : List[str], optional
        Columns to group by when computing lags.
    fill_value : float, optional
        Value to fill NaN from lagging. If None, keeps NaN.

    Examples
    --------
    >>> lf = LagFeatures(cols=['price'], lags=[1, 7, 30])
    >>> X_lagged = lf.fit_transform(X)
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        lags: Sequence[int] = (1, 2, 3),
        group_cols: list[str] | None = None,
        fill_value: float | None = None,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.lags = list(lags)
        self.group_cols = group_cols
        self.fill_value = fill_value

        self._target_cols: list[str] = []

    def fit(self, X, y=None, **fit_params) -> LagFeatures:
        """Identify columns to lag."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        from endgame.core.polars_ops import infer_numeric_columns

        if self.cols is None:
            self._target_cols = infer_numeric_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Create lag features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        lag_cols = []

        for col in self._target_cols:
            for lag in self.lags:
                if self.group_cols:
                    lag_expr = pl.col(col).shift(lag).over(self.group_cols)
                else:
                    lag_expr = pl.col(col).shift(lag)

                if self.fill_value is not None:
                    lag_expr = lag_expr.fill_null(self.fill_value)

                lag_cols.append(lag_expr.alias(f"{col}_lag_{lag}"))

        if lag_cols:
            result = df.with_columns(lag_cols)
        else:
            result = df

        return self._from_lazyframe(result.lazy())


class RollingFeatures(PolarsTransformer):
    """Generate rolling window statistics.

    Creates rolling aggregations for time series data.

    Parameters
    ----------
    cols : List[str], optional
        Columns to compute rolling stats for.
    windows : List[int], default=[3, 7, 14]
        Window sizes.
    methods : List[str], default=['mean', 'std']
        Aggregation methods: 'mean', 'std', 'min', 'max', 'sum'.
    group_cols : List[str], optional
        Columns to group by.
    min_periods : int, default=1
        Minimum observations in window required.

    Examples
    --------
    >>> rf = RollingFeatures(cols=['price'], windows=[7, 30])
    >>> X_rolling = rf.fit_transform(X)
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        windows: Sequence[int] = (3, 7, 14),
        methods: Sequence[str] = ("mean", "std"),
        group_cols: list[str] | None = None,
        min_periods: int = 1,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.windows = list(windows)
        self.methods = list(methods)
        self.group_cols = group_cols
        self.min_periods = min_periods

        self._target_cols: list[str] = []

    def fit(self, X, y=None, **fit_params) -> RollingFeatures:
        """Identify columns for rolling statistics."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        from endgame.core.polars_ops import infer_numeric_columns

        if self.cols is None:
            self._target_cols = infer_numeric_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Compute rolling features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        rolling_cols = []

        for col in self._target_cols:
            for window in self.windows:
                for method in self.methods:
                    # Build rolling expression
                    if self.group_cols:
                        base_expr = pl.col(col).over(self.group_cols)
                    else:
                        base_expr = pl.col(col)

                    if method == "mean":
                        roll_expr = base_expr.rolling_mean(
                            window_size=window,
                            min_periods=self.min_periods,
                        )
                    elif method == "std":
                        roll_expr = base_expr.rolling_std(
                            window_size=window,
                            min_periods=self.min_periods,
                        )
                    elif method == "min":
                        roll_expr = base_expr.rolling_min(
                            window_size=window,
                            min_periods=self.min_periods,
                        )
                    elif method == "max":
                        roll_expr = base_expr.rolling_max(
                            window_size=window,
                            min_periods=self.min_periods,
                        )
                    elif method == "sum":
                        roll_expr = base_expr.rolling_sum(
                            window_size=window,
                            min_periods=self.min_periods,
                        )
                    else:
                        raise ValueError(f"Unknown method: {method}")

                    rolling_cols.append(
                        roll_expr.alias(f"{col}_rolling_{method}_{window}")
                    )

        if rolling_cols:
            result = df.with_columns(rolling_cols)
        else:
            result = df

        return self._from_lazyframe(result.lazy())
