"""Unified data loading utilities for AutoML.

This module provides utilities for loading data from various sources
(CSV, Parquet, DataFrames) and inferring task types.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def load_data(
    data: str | Path | pd.DataFrame | np.ndarray,
    label: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Load data from various sources.

    Parameters
    ----------
    data : str, Path, DataFrame, or ndarray
        Input data. Can be:
        - Path to CSV or Parquet file
        - pandas DataFrame
        - numpy array (label must be last column or specified separately)
    label : str, optional
        Name of the target column.
    **kwargs
        Additional arguments passed to pandas read functions.

    Returns
    -------
    tuple of (DataFrame, Series or None)
        Features and target (if label specified).

    Examples
    --------
    >>> X, y = load_data("train.csv", label="target")
    >>> X, y = load_data(df, label="price")
    >>> X, _ = load_data("test.csv")  # No label for test data
    """
    # Load from file path
    if isinstance(data, (str, Path)):
        data = Path(data)

        if not data.exists():
            raise FileNotFoundError(f"Data file not found: {data}")

        suffix = data.suffix.lower()

        if suffix == ".csv":
            df = pd.read_csv(data, **kwargs)
        elif suffix in (".parquet", ".pq"):
            df = pd.read_parquet(data, **kwargs)
        elif suffix in (".feather", ".ftr"):
            df = pd.read_feather(data, **kwargs)
        elif suffix == ".json":
            df = pd.read_json(data, **kwargs)
        elif suffix in (".xlsx", ".xls"):
            df = pd.read_excel(data, **kwargs)
        elif suffix == ".pkl":
            df = pd.read_pickle(data, **kwargs)
        else:
            # Try CSV as default
            logger.warning(f"Unknown file extension {suffix}, trying CSV")
            df = pd.read_csv(data, **kwargs)

    # Convert numpy array to DataFrame
    elif isinstance(data, np.ndarray):
        n_cols = data.shape[1] if data.ndim > 1 else 1

        if label is not None and isinstance(label, int):
            # Label is column index
            columns = [f"feature_{i}" for i in range(n_cols)]
            columns[label] = "target"
            label = "target"
        else:
            columns = [f"feature_{i}" for i in range(n_cols)]
            if label is None:
                # Assume last column is target
                columns[-1] = "target"
                label = "target"

        df = pd.DataFrame(data, columns=columns)

    # Already a DataFrame
    elif isinstance(data, pd.DataFrame):
        df = data.copy()

    else:
        raise TypeError(f"Unsupported data type: {type(data)}")

    # Split features and target
    if label is not None and label in df.columns:
        y = df[label]
        X = df.drop(columns=[label])
        return X, y
    else:
        return df, None


def split_features_target(
    df: pd.DataFrame,
    label: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split DataFrame into features and target.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    label : str
        Name of the target column.

    Returns
    -------
    tuple of (DataFrame, Series)
        Features and target.
    """
    if label not in df.columns:
        raise ValueError(f"Label column '{label}' not found in DataFrame")

    y = df[label]
    X = df.drop(columns=[label])

    return X, y


def infer_task_type(
    y: pd.Series | np.ndarray,
    threshold: int = 10,
) -> str:
    """Infer task type from target variable.

    Parameters
    ----------
    y : array-like
        Target variable.
    threshold : int, default=10
        Maximum number of unique values to consider as classification.

    Returns
    -------
    str
        Task type: "binary", "multiclass", or "regression".

    Examples
    --------
    >>> infer_task_type(pd.Series([0, 1, 0, 1]))
    'binary'
    >>> infer_task_type(pd.Series([0, 1, 2, 3]))
    'multiclass'
    >>> infer_task_type(pd.Series([1.5, 2.3, 4.1, 5.7]))
    'regression'
    """
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)

    # Check for string/object types -> classification
    if y_arr.dtype == object or isinstance(y_arr[0], str):
        n_unique = len(np.unique(y_arr))
        return "binary" if n_unique == 2 else "multiclass"

    # Check for boolean -> binary classification
    if y_arr.dtype == bool:
        return "binary"

    # Check number of unique values
    n_unique = len(np.unique(y_arr[~np.isnan(y_arr)] if np.issubdtype(y_arr.dtype, np.floating) else y_arr))

    # Few unique values -> classification
    if n_unique <= threshold:
        return "binary" if n_unique == 2 else "multiclass"

    # Check if all values are integers
    if np.issubdtype(y_arr.dtype, np.integer):
        # Could still be classification with many classes
        if n_unique <= 100:
            return "binary" if n_unique == 2 else "multiclass"

    # Default to regression
    return "regression"


def infer_column_types(
    df: pd.DataFrame,
    max_unique_ratio: float = 0.05,
    max_unique_absolute: int = 50,
) -> dict[str, list[str]]:
    """Infer column types for preprocessing.

    Parameters
    ----------
    df : DataFrame
        Input DataFrame.
    max_unique_ratio : float, default=0.05
        Maximum ratio of unique values to consider as categorical.
    max_unique_absolute : int, default=50
        Maximum absolute number of unique values for categorical.

    Returns
    -------
    dict
        Dictionary with keys: "numeric", "categorical", "datetime", "text".

    Examples
    --------
    >>> types = infer_column_types(df)
    >>> print(types["numeric"])
    ['age', 'income', 'balance']
    >>> print(types["categorical"])
    ['gender', 'country', 'product_type']
    """
    column_types = {
        "numeric": [],
        "categorical": [],
        "datetime": [],
        "text": [],
        "boolean": [],
    }

    n_rows = len(df)

    for col in df.columns:
        dtype = df[col].dtype
        n_unique = df[col].nunique()
        unique_ratio = n_unique / max(n_rows, 1)

        # Boolean columns
        if dtype == bool or (n_unique == 2 and set(df[col].dropna().unique()).issubset({0, 1})):  # noqa: E721
            column_types["boolean"].append(col)

        # Datetime columns
        elif pd.api.types.is_datetime64_any_dtype(dtype):
            column_types["datetime"].append(col)

        # Object/string columns
        elif dtype == object:  # noqa: E721
            # Check if it looks like datetime
            sample = df[col].dropna().head(100)
            try:
                pd.to_datetime(sample)
                column_types["datetime"].append(col)
                continue
            except (ValueError, TypeError):
                pass

            # Check if it's categorical or text
            avg_len = sample.astype(str).str.len().mean()

            if avg_len > 50 or n_unique > max_unique_absolute:
                column_types["text"].append(col)
            else:
                column_types["categorical"].append(col)

        # Numeric columns
        elif pd.api.types.is_numeric_dtype(dtype):
            # Check if it should be categorical
            if unique_ratio <= max_unique_ratio and n_unique <= max_unique_absolute:
                column_types["categorical"].append(col)
            else:
                column_types["numeric"].append(col)

        else:
            # Default to numeric
            column_types["numeric"].append(col)

    return column_types


class DataLoader:
    """Unified data loader for AutoML.

    Handles loading from multiple sources and caches loaded data.

    Parameters
    ----------
    label : str
        Name of the target column.
    problem_type : str, default="auto"
        Problem type: "auto", "binary", "multiclass", "regression".
    sample_size : int, optional
        If set, sample this many rows for faster development.
    random_state : int, default=42
        Random seed for sampling.

    Attributes
    ----------
    train_data_ : DataFrame
        Loaded training data.
    X_train_ : DataFrame
        Training features.
    y_train_ : Series
        Training target.
    column_types_ : dict
        Inferred column types.
    task_type_ : str
        Inferred or specified task type.

    Examples
    --------
    >>> loader = DataLoader(label="target")
    >>> loader.load_train("train.csv")
    >>> X, y = loader.get_train()
    >>> X_test = loader.load_test("test.csv")
    """

    def __init__(
        self,
        label: str,
        problem_type: str = "auto",
        sample_size: int | None = None,
        random_state: int = 42,
    ):
        self.label = label
        self.problem_type = problem_type
        self.sample_size = sample_size
        self.random_state = random_state

        # State
        self.train_data_: pd.DataFrame | None = None
        self.X_train_: pd.DataFrame | None = None
        self.y_train_: pd.Series | None = None
        self.test_data_: pd.DataFrame | None = None
        self.X_test_: pd.DataFrame | None = None
        self.column_types_: dict[str, list[str]] | None = None
        self.task_type_: str | None = None

    def load_train(
        self,
        data: str | Path | pd.DataFrame | np.ndarray,
        **kwargs,
    ) -> "DataLoader":
        """Load training data.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Training data source.
        **kwargs
            Additional arguments for data loading.

        Returns
        -------
        self
        """
        X, y = load_data(data, label=self.label, **kwargs)

        if y is None:
            raise ValueError(f"Label column '{self.label}' not found in training data")

        # Sample if requested
        if self.sample_size is not None and len(X) > self.sample_size:
            np.random.seed(self.random_state)
            indices = np.random.choice(len(X), self.sample_size, replace=False)
            X = X.iloc[indices].reset_index(drop=True)
            y = y.iloc[indices].reset_index(drop=True)
            logger.info(f"Sampled {self.sample_size} rows from training data")

        self.train_data_ = pd.concat([X, y.rename(self.label)], axis=1)
        self.X_train_ = X
        self.y_train_ = y

        # Infer column types
        self.column_types_ = infer_column_types(X)

        # Infer task type
        if self.problem_type == "auto":
            self.task_type_ = infer_task_type(y)
        else:
            self.task_type_ = self.problem_type

        logger.info(
            f"Loaded training data: {X.shape[0]} rows, {X.shape[1]} features, "
            f"task_type={self.task_type_}"
        )

        return self

    def load_test(
        self,
        data: str | Path | pd.DataFrame | np.ndarray,
        **kwargs,
    ) -> pd.DataFrame:
        """Load test data.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Test data source.
        **kwargs
            Additional arguments for data loading.

        Returns
        -------
        DataFrame
            Test features.
        """
        X, _ = load_data(data, label=None, **kwargs)

        # Remove label column if present
        if self.label in X.columns:
            X = X.drop(columns=[self.label])

        self.test_data_ = X
        self.X_test_ = X

        logger.info(f"Loaded test data: {X.shape[0]} rows, {X.shape[1]} features")

        return X

    def get_train(self) -> tuple[pd.DataFrame, pd.Series]:
        """Get training features and target.

        Returns
        -------
        tuple of (DataFrame, Series)
            Training features and target.
        """
        if self.X_train_ is None:
            raise ValueError("Training data not loaded. Call load_train() first.")

        return self.X_train_, self.y_train_

    def get_test(self) -> pd.DataFrame:
        """Get test features.

        Returns
        -------
        DataFrame
            Test features.
        """
        if self.X_test_ is None:
            raise ValueError("Test data not loaded. Call load_test() first.")

        return self.X_test_

    def get_validation_split(
        self,
        val_frac: float = 0.2,
        stratify: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Split training data into train and validation sets.

        Parameters
        ----------
        val_frac : float, default=0.2
            Fraction of data for validation.
        stratify : bool, default=True
            Whether to use stratified split for classification.

        Returns
        -------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        from sklearn.model_selection import train_test_split

        if self.X_train_ is None:
            raise ValueError("Training data not loaded. Call load_train() first.")

        stratify_arg = self.y_train_ if (stratify and self.task_type_ != "regression") else None

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train_,
            self.y_train_,
            test_size=val_frac,
            random_state=self.random_state,
            stratify=stratify_arg,
        )

        return X_train, X_val, y_train, y_val

    def get_summary(self) -> dict[str, Any]:
        """Get summary of loaded data.

        Returns
        -------
        dict
            Data summary.
        """
        summary = {
            "label": self.label,
            "task_type": self.task_type_,
            "problem_type": self.problem_type,
        }

        if self.X_train_ is not None:
            summary.update({
                "n_train_samples": len(self.X_train_),
                "n_features": self.X_train_.shape[1],
                "column_types": {k: len(v) for k, v in self.column_types_.items()},
            })

            if self.task_type_ in ("binary", "multiclass"):
                summary["n_classes"] = self.y_train_.nunique()
                summary["class_distribution"] = self.y_train_.value_counts().to_dict()

        if self.X_test_ is not None:
            summary["n_test_samples"] = len(self.X_test_)

        return summary
