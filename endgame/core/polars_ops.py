from __future__ import annotations

"""Polars utility functions for efficient data processing."""

from collections.abc import Sequence
from typing import Literal, Union

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

OutputFormat = Literal["polars", "pandas", "numpy"]


def to_lazyframe(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    column_names: list[str] | None = None,
) -> pl.LazyFrame:
    """Convert input data to Polars LazyFrame for optimized processing.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data. Can be numpy array, pandas DataFrame, Polars DataFrame,
        or Polars LazyFrame.
    column_names : List[str], optional
        Column names for numpy arrays. If None, generates col_0, col_1, etc.

    Returns
    -------
    pl.LazyFrame
        Polars LazyFrame for lazy evaluation.

    Raises
    ------
    ImportError
        If polars is not installed.
    TypeError
        If input type is not recognized.

    Examples
    --------
    >>> import numpy as np
    >>> X = np.array([[1, 2], [3, 4]])
    >>> lf = to_lazyframe(X)
    >>> lf.collect()
    shape: (2, 2)
    """
    if not HAS_POLARS:
        raise ImportError(
            "Polars is required for this operation. "
            "Install with: pip install polars"
        )

    # Already a LazyFrame
    if isinstance(X, pl.LazyFrame):
        return X

    # Polars DataFrame
    if isinstance(X, pl.DataFrame):
        return X.lazy()

    # Pandas DataFrame
    if HAS_PANDAS and isinstance(X, pd.DataFrame):
        return pl.from_pandas(X).lazy()

    # Pandas Series
    if HAS_PANDAS and isinstance(X, pd.Series):
        return pl.from_pandas(X.to_frame()).lazy()

    # NumPy array
    if isinstance(X, np.ndarray):
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        if column_names is None:
            column_names = [f"col_{i}" for i in range(X.shape[1])]

        if len(column_names) != X.shape[1]:
            raise ValueError(
                f"column_names length ({len(column_names)}) must match "
                f"number of columns ({X.shape[1]})"
            )

        return pl.DataFrame(
            {name: X[:, i] for i, name in enumerate(column_names)}
        ).lazy()

    # List or sequence
    if isinstance(X, (list, tuple)):
        X = np.array(X)
        return to_lazyframe(X, column_names)

    raise TypeError(
        f"Cannot convert type {type(X).__name__} to LazyFrame. "
        "Expected numpy array, pandas DataFrame, or Polars DataFrame/LazyFrame."
    )


def from_lazyframe(
    lf: pl.LazyFrame,
    output_format: OutputFormat = "polars",
) -> Union[np.ndarray, pd.DataFrame, pl.DataFrame]:
    """Convert LazyFrame to requested output format.

    Parameters
    ----------
    lf : pl.LazyFrame
        Polars LazyFrame to convert.
    output_format : str, default='polars'
        Output format: 'polars', 'pandas', 'numpy'.

    Returns
    -------
    DataFrame or array
        Data in requested format.

    Raises
    ------
    ValueError
        If output_format is not recognized.
    ImportError
        If pandas is requested but not installed.
    """
    if not HAS_POLARS:
        raise ImportError("Polars is required for this operation.")

    # Collect the lazy frame
    df = lf.collect()

    if output_format == "polars":
        return df

    if output_format == "pandas":
        if not HAS_PANDAS:
            raise ImportError(
                "Pandas is required for output_format='pandas'. "
                "Install with: pip install pandas"
            )
        return df.to_pandas()

    if output_format == "numpy":
        return df.to_numpy()

    raise ValueError(
        f"Unknown output_format '{output_format}'. "
        "Expected 'polars', 'pandas', or 'numpy'."
    )


def detect_input_format(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame]
) -> OutputFormat:
    """Detect the format of input data.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data.

    Returns
    -------
    str
        Format: 'polars', 'pandas', or 'numpy'.
    """
    if HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
        return "polars"
    if HAS_PANDAS and isinstance(X, (pd.DataFrame, pd.Series)):
        return "pandas"
    return "numpy"


def infer_categorical_columns(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    max_cardinality: int = 50,
    include_string: bool = True,
) -> list[str]:
    """Infer which columns should be treated as categorical.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data.
    max_cardinality : int, default=50
        Maximum unique values for a numeric column to be considered categorical.
    include_string : bool, default=True
        Whether to include string/object columns as categorical.

    Returns
    -------
    List[str]
        Column names that appear to be categorical.
    """
    lf = to_lazyframe(X)
    df = lf.collect()

    categorical_cols = []

    for col in df.columns:
        dtype = df[col].dtype

        # String columns are categorical
        if include_string and dtype == pl.Utf8:
            categorical_cols.append(col)
            continue

        # Polars categorical type
        if dtype == pl.Categorical:
            categorical_cols.append(col)
            continue

        # Check cardinality for numeric types
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            n_unique = df[col].n_unique()
            if n_unique <= max_cardinality:
                categorical_cols.append(col)

    return categorical_cols


def infer_numeric_columns(
    X: Union[np.ndarray, pd.DataFrame, pl.DataFrame, pl.LazyFrame],
    exclude_categorical: bool = True,
    max_cardinality: int = 50,
) -> list[str]:
    """Infer which columns are numeric.

    Parameters
    ----------
    X : array-like or DataFrame
        Input data.
    exclude_categorical : bool, default=True
        Whether to exclude low-cardinality integers that appear categorical.
    max_cardinality : int, default=50
        Threshold for excluding categorical-like integers.

    Returns
    -------
    List[str]
        Column names that are numeric.
    """
    lf = to_lazyframe(X)
    df = lf.collect()

    numeric_cols = []
    categorical_cols = set()

    if exclude_categorical:
        categorical_cols = set(infer_categorical_columns(X, max_cardinality))

    for col in df.columns:
        if col in categorical_cols:
            continue

        dtype = df[col].dtype

        # Float types are always numeric
        if dtype in (pl.Float32, pl.Float64):
            numeric_cols.append(col)
            continue

        # Integer types (if not excluded as categorical)
        if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
            numeric_cols.append(col)

    return numeric_cols


def safe_divide(
    numerator: pl.Expr,
    denominator: pl.Expr,
    fill_value: float = 0.0,
) -> pl.Expr:
    """Safe division that handles division by zero.

    Parameters
    ----------
    numerator : pl.Expr
        Numerator expression.
    denominator : pl.Expr
        Denominator expression.
    fill_value : float, default=0.0
        Value to use when denominator is zero.

    Returns
    -------
    pl.Expr
        Division result with zeros handled.
    """
    if not HAS_POLARS:
        raise ImportError("Polars is required for this operation.")

    return pl.when(denominator != 0).then(numerator / denominator).otherwise(fill_value)


def compute_statistics(
    lf: pl.LazyFrame,
    group_cols: list[str],
    agg_cols: list[str],
    methods: Sequence[str] = ("mean", "std", "min", "max"),
) -> pl.LazyFrame:
    """Compute grouped statistics efficiently.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input data.
    group_cols : List[str]
        Columns to group by.
    agg_cols : List[str]
        Columns to aggregate.
    methods : Sequence[str], default=('mean', 'std', 'min', 'max')
        Aggregation methods to apply.

    Returns
    -------
    pl.LazyFrame
        Aggregated statistics.
    """
    if not HAS_POLARS:
        raise ImportError("Polars is required for this operation.")

    agg_exprs = []

    for col in agg_cols:
        for method in methods:
            if method == "mean":
                agg_exprs.append(pl.col(col).mean().alias(f"{col}_mean"))
            elif method == "std":
                agg_exprs.append(pl.col(col).std().alias(f"{col}_std"))
            elif method == "min":
                agg_exprs.append(pl.col(col).min().alias(f"{col}_min"))
            elif method == "max":
                agg_exprs.append(pl.col(col).max().alias(f"{col}_max"))
            elif method == "sum":
                agg_exprs.append(pl.col(col).sum().alias(f"{col}_sum"))
            elif method == "count":
                agg_exprs.append(pl.col(col).count().alias(f"{col}_count"))
            elif method == "median":
                agg_exprs.append(pl.col(col).median().alias(f"{col}_median"))
            elif method == "skew":
                agg_exprs.append(pl.col(col).skew().alias(f"{col}_skew"))
            elif method == "kurtosis":
                agg_exprs.append(pl.col(col).kurtosis().alias(f"{col}_kurtosis"))
            elif method == "first":
                agg_exprs.append(pl.col(col).first().alias(f"{col}_first"))
            elif method == "last":
                agg_exprs.append(pl.col(col).last().alias(f"{col}_last"))
            elif method == "nunique":
                agg_exprs.append(pl.col(col).n_unique().alias(f"{col}_nunique"))
            else:
                raise ValueError(f"Unknown aggregation method: {method}")

    return lf.group_by(group_cols).agg(agg_exprs)
