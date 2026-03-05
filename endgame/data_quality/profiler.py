from __future__ import annotations

"""Data profiling: per-column statistics and dataset summaries."""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats

from endgame.guardrails.utils import normalize_input


@dataclass
class DataProfile:
    """Result of profiling a dataset.

    Attributes
    ----------
    column_stats : dict[str, dict]
        Per-column statistics keyed by column name.
    summary_stats : dict
        Dataset-level summary statistics.
    correlation_matrix : np.ndarray or None
        Pearson correlation matrix of numeric columns, if requested.
    numeric_names : list[str]
        Names of numeric columns.
    categorical_names : list[str]
        Names of categorical columns.
    """

    column_stats: dict[str, dict]
    summary_stats: dict
    correlation_matrix: np.ndarray | None = None
    numeric_names: list[str] = field(default_factory=list)
    categorical_names: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a human-readable summary string."""
        s = self.summary_stats
        lines = [
            f"DataProfile: {s['n_rows']} rows x {s['n_cols']} cols",
            f"  Numeric: {s['n_numeric']}  Categorical: {s['n_categorical']}",
            f"  Total missing: {s['total_missing_pct']:.1%}",
            f"  Memory: {s['memory_bytes'] / 1024:.1f} KB",
        ]
        return "\n".join(lines)

    def to_dataframe(self):
        """Return column stats as a pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame(self.column_stats).T

    def _repr_html_(self) -> str:
        """Rich HTML representation for Jupyter notebooks."""
        s = self.summary_stats
        rows_html = ""
        for name, col in self.column_stats.items():
            dtype = col.get("dtype", "")
            missing = col.get("missing_pct", 0)
            n_unique = col.get("n_unique", "")
            rows_html += (
                f"<tr><td>{name}</td><td>{dtype}</td>"
                f"<td>{missing:.1%}</td><td>{n_unique}</td></tr>"
            )
        return (
            f"<div><strong>DataProfile</strong>: {s['n_rows']} rows x {s['n_cols']} cols "
            f"({s['n_numeric']} numeric, {s['n_categorical']} categorical, "
            f"{s['total_missing_pct']:.1%} missing)</div>"
            "<table><tr><th>Column</th><th>Type</th><th>Missing</th><th>Unique</th></tr>"
            f"{rows_html}</table>"
        )

    def __repr__(self) -> str:
        return self.summary()


class DataProfiler:
    """Profile a dataset to compute per-column and dataset-level statistics.

    Parameters
    ----------
    correlation : bool
        Whether to compute a Pearson correlation matrix for numeric columns.
    max_top_values : int
        Number of most-common values to report for categorical columns.
    outlier_method : str
        Method for outlier detection. Currently only ``'iqr'`` is supported.

    Examples
    --------
    >>> import endgame as eg
    >>> profiler = eg.data_quality.DataProfiler()
    >>> profile = profiler.profile(X)
    >>> print(profile.summary())
    """

    def __init__(
        self,
        correlation: bool = False,
        max_top_values: int = 10,
        outlier_method: str = "iqr",
    ):
        self.correlation = correlation
        self.max_top_values = max_top_values
        self.outlier_method = outlier_method

    def profile(self, X: Any, y: Any = None) -> DataProfile:
        """Profile the dataset.

        Parameters
        ----------
        X : array-like
            Input features (pandas/polars DataFrame or numpy array).
        y : array-like, optional
            Target variable (unused, reserved for API consistency).

        Returns
        -------
        DataProfile
            Profiling results with per-column and summary statistics.
        """
        X_numeric, numeric_names, X_cat, cat_names, _ = normalize_input(X, y)

        n_rows = X_numeric.shape[0] if X_numeric.size > 0 else (X_cat.shape[0] if X_cat is not None else 0)
        n_cols = len(numeric_names) + len(cat_names)

        column_stats: dict[str, dict] = {}
        total_missing = 0

        # Numeric columns
        for i, name in enumerate(numeric_names):
            col = X_numeric[:, i]
            missing = int(np.isnan(col).sum())
            total_missing += missing
            valid = col[~np.isnan(col)]

            # Outlier count via IQR
            n_outliers = 0
            if len(valid) > 0:
                q1, q3 = np.percentile(valid, [25, 75])
                iqr = q3 - q1
                n_outliers = int(np.sum((valid < q1 - 1.5 * iqr) | (valid > q3 + 1.5 * iqr)))

            column_stats[name] = {
                "dtype": "numeric",
                "missing_count": missing,
                "missing_pct": missing / n_rows if n_rows > 0 else 0.0,
                "n_unique": int(len(np.unique(valid))) if len(valid) > 0 else 0,
                "cardinality_pct": int(len(np.unique(valid))) / n_rows if n_rows > 0 and len(valid) > 0 else 0.0,
                "min": float(np.nanmin(col)) if len(valid) > 0 else None,
                "max": float(np.nanmax(col)) if len(valid) > 0 else None,
                "mean": float(np.nanmean(col)) if len(valid) > 0 else None,
                "std": float(np.nanstd(col, ddof=1)) if len(valid) > 1 else None,
                "median": float(np.nanmedian(col)) if len(valid) > 0 else None,
                "skewness": float(stats.skew(valid, nan_policy="omit")) if len(valid) > 2 else None,
                "kurtosis": float(stats.kurtosis(valid, nan_policy="omit")) if len(valid) > 2 else None,
                "n_zeros": int(np.sum(valid == 0)),
                "n_outliers": n_outliers,
            }

        # Categorical columns
        if X_cat is not None:
            for i, name in enumerate(cat_names):
                col = X_cat[:, i]
                missing = int(sum(1 for v in col if v is None or (isinstance(v, float) and np.isnan(v)) or str(v) == "nan"))
                total_missing += missing
                valid = [v for v in col if v is not None and str(v) != "nan"]

                counter = Counter(valid)
                top = counter.most_common(self.max_top_values)

                column_stats[name] = {
                    "dtype": "categorical",
                    "missing_count": missing,
                    "missing_pct": missing / n_rows if n_rows > 0 else 0.0,
                    "n_unique": len(counter),
                    "cardinality_pct": len(counter) / n_rows if n_rows > 0 else 0.0,
                    "top_values": dict(top),
                }

        # Correlation matrix
        corr = None
        if self.correlation and X_numeric.shape[1] > 0:
            # Use only rows with no NaN for correlation
            mask = ~np.any(np.isnan(X_numeric), axis=1)
            if mask.sum() > 1:
                corr = np.corrcoef(X_numeric[mask].T)

        # Memory estimate
        memory = X_numeric.nbytes
        if X_cat is not None:
            memory += X_cat.nbytes

        total_cells = n_rows * n_cols if n_rows > 0 else 1
        summary_stats = {
            "n_rows": n_rows,
            "n_cols": n_cols,
            "n_numeric": len(numeric_names),
            "n_categorical": len(cat_names),
            "total_missing_pct": total_missing / total_cells,
            "memory_bytes": memory,
        }

        return DataProfile(
            column_stats=column_stats,
            summary_stats=summary_stats,
            correlation_matrix=corr,
            numeric_names=list(numeric_names),
            categorical_names=list(cat_names),
        )
