from __future__ import annotations

"""Input normalization and utility helpers for guardrails checks."""

import time
from typing import Any

import numpy as np


def normalize_input(
    X: Any, y: Any = None
) -> tuple[np.ndarray, list[str], np.ndarray | None, list[str], np.ndarray | None]:
    """Normalize inputs to arrays with feature name tracking.

    Parameters
    ----------
    X : array-like
        Features as pandas DataFrame, polars DataFrame, or numpy array.
    y : array-like, optional
        Target variable.

    Returns
    -------
    X_numeric : np.ndarray
        Numeric columns as float array.
    numeric_names : list of str
        Names of numeric columns.
    X_cat : np.ndarray or None
        Categorical columns as object array, or None if no categoricals.
    cat_names : list of str
        Names of categorical columns.
    y_arr : np.ndarray or None
        Target as 1-d array, or None.
    """
    import pandas as pd

    X_numeric: np.ndarray
    numeric_names: list[str]
    X_cat: np.ndarray | None = None
    cat_names: list[str] = []

    # Handle polars
    try:
        import polars as pl
        if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            X = X.collect() if isinstance(X, pl.LazyFrame) else X
            X = X.to_pandas()
    except ImportError:
        pass

    if isinstance(X, pd.DataFrame):
        num_df = X.select_dtypes(include=[np.number])
        X_numeric = num_df.values.astype(float)
        numeric_names = num_df.columns.tolist()

        cat_df = X.select_dtypes(exclude=[np.number])
        if cat_df.shape[1] > 0:
            X_cat = cat_df.values
            cat_names = cat_df.columns.tolist()
    else:
        X_numeric = np.asarray(X, dtype=float)
        numeric_names = [f"feature_{i}" for i in range(X_numeric.shape[1])]

    y_arr = None
    if y is not None:
        y_arr = np.asarray(y).ravel()

    return X_numeric, numeric_names, X_cat, cat_names, y_arr


def check_time_budget(start: float, budget: float | None, fraction: float) -> bool:
    """Return True if we have exceeded fraction of the time budget."""
    if budget is None:
        return False
    return (time.time() - start) > budget * fraction


def reconstruct_output(
    X_original: Any, columns_to_keep: list[str]
) -> Any:
    """Filter columns from X preserving the original type.

    Parameters
    ----------
    X_original : DataFrame or array
        Original input.
    columns_to_keep : list of str
        Column names to retain.

    Returns
    -------
    Filtered X in the same type as input.
    """
    import pandas as pd

    try:
        import polars as pl
        if isinstance(X_original, pl.DataFrame):
            keep = [c for c in columns_to_keep if c in X_original.columns]
            return X_original.select(keep)
        if isinstance(X_original, pl.LazyFrame):
            keep = [c for c in columns_to_keep if c in X_original.columns]
            return X_original.select(keep)
    except ImportError:
        pass

    if isinstance(X_original, pd.DataFrame):
        keep = [c for c in columns_to_keep if c in X_original.columns]
        return X_original[keep]

    # numpy array — filter by index
    all_names = [f"feature_{i}" for i in range(X_original.shape[1])]
    keep_idx = [i for i, name in enumerate(all_names) if name in columns_to_keep]
    return np.asarray(X_original)[:, keep_idx]
