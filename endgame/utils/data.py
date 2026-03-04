"""Data loading and splitting utilities.

Thin wrappers around sklearn so users never need to import sklearn directly.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_SKLEARN_DATASETS = {
    "breast_cancer": "load_breast_cancer",
    "iris": "load_iris",
    "wine": "load_wine",
    "digits": "load_digits",
    "diabetes": "load_diabetes",
}


def load_dataset(
    name_or_id: str | int,
    *,
    as_frame: bool = True,
    return_names: bool = False,
) -> tuple[Any, ...]:
    """Load a dataset from sklearn or OpenML.

    Parameters
    ----------
    name_or_id : str or int
        Dataset name (str) or OpenML dataset ID (int).
        Built-in sklearn datasets can be loaded by name:
        ``'breast_cancer'``, ``'iris'``, ``'wine'``, ``'digits'``, ``'diabetes'``.
    as_frame : bool, default=True
        If True, return pandas DataFrames; otherwise numpy arrays.
    return_names : bool, default=False
        If True, also return feature names and class names (if available).

    Returns
    -------
    X : DataFrame or ndarray
        Feature matrix.
    y : Series or ndarray
        Target vector.
    feature_names : list of str
        Only returned when ``return_names=True``.
    class_names : list of str
        Only returned when ``return_names=True``. Empty list for regression.

    Examples
    --------
    >>> import endgame as eg
    >>> X, y = eg.utils.load_dataset("breast_cancer")
    >>> X, y, fnames, cnames = eg.utils.load_dataset("iris", return_names=True)
    >>> X, y = eg.utils.load_dataset(40536)  # OpenML ID
    """
    # Check for built-in sklearn datasets first
    if isinstance(name_or_id, str) and name_or_id in _SKLEARN_DATASETS:
        import sklearn.datasets as skds

        loader = getattr(skds, _SKLEARN_DATASETS[name_or_id])
        bunch = loader()
        X = bunch.data
        y = bunch.target
        feature_names = list(bunch.feature_names)
        class_names = [str(c) for c in bunch.target_names] if hasattr(bunch, "target_names") else []

        if as_frame:
            import pandas as pd
            X = pd.DataFrame(X, columns=feature_names)
            y = pd.Series(y, name="target")

        if return_names:
            return X, y, feature_names, class_names
        return X, y

    # OpenML
    from sklearn.datasets import fetch_openml

    kwargs: dict[str, Any] = {
        "return_X_y": not return_names,
        "as_frame": as_frame,
        "parser": "auto",
    }

    if isinstance(name_or_id, int):
        kwargs["data_id"] = name_or_id
    else:
        kwargs["name"] = name_or_id

    if return_names:
        del kwargs["return_X_y"]
        bunch = fetch_openml(**kwargs)
        X = bunch.data
        y = bunch.target
        feature_names = list(bunch.feature_names) if hasattr(bunch, "feature_names") else []
        class_names = []
        if hasattr(bunch, "target_names"):
            class_names = [str(c) for c in bunch.target_names]
        elif hasattr(y, "cat") and hasattr(y.cat, "categories"):
            class_names = [str(c) for c in y.cat.categories]
        elif hasattr(y, "unique"):
            class_names = [str(c) for c in sorted(y.unique())]
        return X, y, feature_names, class_names

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
