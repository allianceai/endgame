from __future__ import annotations

"""Duplicate detection and deduplication."""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from endgame.guardrails.utils import normalize_input


def _hash_rows(arr: np.ndarray) -> np.ndarray:
    """Hash each row of a 2-D array for fast exact-duplicate detection."""
    filled = np.nan_to_num(arr, nan=-999999.0)
    return np.array([hash(row.tobytes()) for row in filled])


@dataclass
class DuplicateReport:
    """Result of duplicate detection.

    Attributes
    ----------
    duplicate_indices : list[list[int]]
        Groups of row indices that are duplicates of each other.
    duplicate_groups : int
        Number of distinct duplicate groups found.
    n_duplicates : int
        Total number of duplicate rows (excluding one representative per group).
    pct_duplicates : float
        Fraction of rows that are duplicates.
    method : str
        Detection method used (``'exact'`` or ``'fuzzy'``).
    """

    duplicate_indices: list[list[int]]
    duplicate_groups: int
    n_duplicates: int
    pct_duplicates: float
    method: str

    def summary(self) -> str:
        lines = [
            f"DuplicateReport ({self.method}): {self.n_duplicates} duplicates "
            f"({self.pct_duplicates:.1%}) in {self.duplicate_groups} groups",
        ]
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            f"<div><strong>DuplicateReport</strong> ({self.method}): "
            f"{self.n_duplicates} duplicates ({self.pct_duplicates:.1%}) "
            f"in {self.duplicate_groups} groups</div>"
        )

    def __repr__(self) -> str:
        return self.summary()


class DuplicateDetector(TransformerMixin, BaseEstimator):
    """Detect and optionally remove duplicate rows.

    Parameters
    ----------
    method : str
        ``'exact'`` for hash-based exact matching, ``'fuzzy'`` for
        cosine-similarity based near-duplicate detection.
    threshold : float
        Similarity threshold for fuzzy matching (0-1). Only used when
        ``method='fuzzy'``.

    Examples
    --------
    >>> detector = DuplicateDetector()
    >>> report = detector.fit_detect(X)
    >>> X_clean = detector.transform(X)
    """

    def __init__(self, method: str = "exact", threshold: float = 0.8):
        self.method = method
        self.threshold = threshold
        self.report_: DuplicateReport | None = None

    def fit_detect(self, X: Any) -> DuplicateReport:
        """Detect duplicates and store the report.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        DuplicateReport
        """
        X_numeric, numeric_names, X_cat, cat_names, _ = normalize_input(X)

        # Build a combined array for hashing
        if X_cat is not None:
            # Convert categoricals to string bytes for hashing
            cat_str = np.array([[str(v) for v in row] for row in X_cat])
            parts = []
            if X_numeric.shape[1] > 0:
                parts.append(X_numeric)
            parts.append(cat_str)
            combined = np.column_stack(parts) if len(parts) > 1 else parts[0]
        else:
            combined = X_numeric

        n_rows = combined.shape[0]

        if self.method == "exact":
            groups = self._detect_exact(combined)
        elif self.method == "fuzzy":
            groups = self._detect_fuzzy(X_numeric, n_rows)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        n_duplicates = sum(len(g) - 1 for g in groups)
        self.report_ = DuplicateReport(
            duplicate_indices=groups,
            duplicate_groups=len(groups),
            n_duplicates=n_duplicates,
            pct_duplicates=n_duplicates / n_rows if n_rows > 0 else 0.0,
            method=self.method,
        )
        return self.report_

    def fit(self, X: Any, y: Any = None) -> DuplicateDetector:
        """Fit the detector (runs detection internally).

        Parameters
        ----------
        X : array-like
            Input data.
        y : ignored

        Returns
        -------
        self
        """
        self.fit_detect(X)
        return self

    def transform(self, X: Any) -> Any:
        """Remove duplicate rows, keeping the first occurrence per group.

        Parameters
        ----------
        X : array-like
            Input data (same type is preserved in output).

        Returns
        -------
        X_deduped
            Data with duplicate rows removed.
        """
        if self.report_ is None:
            raise RuntimeError("Call fit() or fit_detect() before transform().")

        # Collect indices to drop (all but first in each group)
        drop = set()
        for group in self.report_.duplicate_indices:
            for idx in group[1:]:
                drop.add(idx)

        if not drop:
            return X

        keep_mask = np.ones(self._n_rows(X), dtype=bool)
        for idx in drop:
            keep_mask[idx] = False

        return self._select_rows(X, keep_mask)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _n_rows(X: Any) -> int:
        if hasattr(X, "shape"):
            return X.shape[0]
        return len(X)

    @staticmethod
    def _select_rows(X: Any, mask: np.ndarray) -> Any:
        """Select rows from X preserving input type."""
        import pandas as pd

        try:
            import polars as pl
            if isinstance(X, pl.DataFrame):
                return X.filter(pl.Series(mask))
        except ImportError:
            pass

        if isinstance(X, pd.DataFrame):
            return X.loc[mask].reset_index(drop=True)

        return np.asarray(X)[mask]

    @staticmethod
    def _detect_exact(combined: np.ndarray) -> list[list[int]]:
        """Group rows by hash for exact duplicate detection."""
        if combined.dtype.kind in ("U", "O"):
            hashes = np.array([hash(tuple(row)) for row in combined])
        else:
            hashes = _hash_rows(combined)

        hash_to_indices: dict[int, list[int]] = {}
        for i, h in enumerate(hashes):
            hash_to_indices.setdefault(h, []).append(i)

        return [indices for indices in hash_to_indices.values() if len(indices) > 1]

    def _detect_fuzzy(self, X_numeric: np.ndarray, n_rows: int) -> list[list[int]]:
        """Near-duplicate detection via cosine similarity with union-find."""
        if X_numeric.shape[1] == 0:
            return []

        # Subsample for large datasets
        max_n = 10000
        if n_rows > max_n:
            rng = np.random.RandomState(42)
            sample_idx = rng.choice(n_rows, max_n, replace=False)
            X_sub = X_numeric[sample_idx]
        else:
            sample_idx = np.arange(n_rows)
            X_sub = X_numeric

        # Normalize rows
        X_filled = np.nan_to_num(X_sub, nan=0.0)
        norms = np.linalg.norm(X_filled, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        X_normed = X_filled / norms

        # Cosine similarity
        sim = X_normed @ X_normed.T

        # Union-find
        n = len(X_sub)
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        for i in range(n):
            for j in range(i + 1, n):
                if sim[i, j] >= self.threshold:
                    union(i, j)

        # Collect groups
        groups: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            groups.setdefault(root, []).append(int(sample_idx[i]))

        return [indices for indices in groups.values() if len(indices) > 1]
