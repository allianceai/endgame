"""Data loading and splitting utilities.

Thin wrappers around sklearn so users never need to import sklearn directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def load_dataset(
    name_or_id: str | int,
    *,
    as_frame: bool = True,
) -> tuple[Any, Any]:
    """Load a dataset from OpenML by name or ID.

    Parameters
    ----------
    name_or_id : str or int
        Dataset name (str) or OpenML dataset ID (int).
    as_frame : bool, default=True
        If True, return pandas DataFrames; otherwise numpy arrays.

    Returns
    -------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target vector.

    Examples
    --------
    >>> import endgame as eg
    >>> X, y = eg.utils.load_dataset("SpeedDating")
    >>> X, y = eg.utils.load_dataset(40536)
    """
    from sklearn.datasets import fetch_openml

    kwargs: dict[str, Any] = {
        "return_X_y": True,
        "as_frame": as_frame,
        "parser": "auto",
    }

    if isinstance(name_or_id, int):
        kwargs["data_id"] = name_or_id
    else:
        kwargs["name"] = name_or_id

    X, y = fetch_openml(**kwargs)
    return X, y


def split(
    X: Any,
    y: Any,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Any = "auto",
) -> tuple[Any, Any, Any, Any]:
    """Split data into train and test sets.

    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target vector.
    test_size : float, default=0.2
        Fraction of data to use for testing.
    random_state : int, default=42
        Random seed.
    stratify : array-like or 'auto', default='auto'
        If 'auto', stratifies for classification targets (categorical or
        integer with few unique values) and skips for regression.
        Pass None to disable, or an array to stratify on.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple
        Train/test splits.

    Examples
    --------
    >>> import endgame as eg
    >>> X, y = eg.utils.load_dataset(40536)
    >>> X_train, X_test, y_train, y_test = eg.utils.split(X, y)
    """
    from sklearn.model_selection import train_test_split

    if stratify == "auto":
        stratify = _auto_stratify(y)

    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )


def _auto_stratify(y: Any) -> Any:
    """Decide whether to stratify based on target type."""
    y_arr = np.asarray(y)

    # String/object dtype -> classification
    if y_arr.dtype.kind in ("U", "S", "O"):
        return y

    # Boolean -> classification
    if y_arr.dtype.kind == "b":
        return y

    # Integer with few unique values -> classification
    if y_arr.dtype.kind in ("i", "u"):
        n_unique = len(np.unique(y_arr))
        if n_unique <= 30:
            return y

    # Categorical pandas dtype
    try:
        import pandas as pd
        if hasattr(y, "dtype") and isinstance(y.dtype, pd.CategoricalDtype):
            return y
    except ImportError:
        pass

    # Otherwise regression -> no stratification
    return None
