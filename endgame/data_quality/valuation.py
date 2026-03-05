from __future__ import annotations

"""Data valuation: estimate the contribution of each training sample."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator


def _knn_accuracy(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    k: int = 5,
) -> float:
    """Pure-numpy KNN accuracy (no sklearn dependency)."""
    if len(X_val) == 0 or len(X_train) == 0:
        return 0.0
    # Compute pairwise distances
    # (n_val, 1, d) - (1, n_train, d) -> (n_val, n_train)
    diffs = X_val[:, None, :] - X_train[None, :, :]
    dists = np.sum(diffs ** 2, axis=2)
    k_actual = min(k, len(X_train))
    if k_actual >= len(X_train):
        # All training points are neighbors
        nn_idx = np.tile(np.arange(len(X_train)), (len(X_val), 1))
    else:
        nn_idx = np.argpartition(dists, k_actual, axis=1)[:, :k_actual]
    # Majority vote
    nn_labels = y_train[nn_idx]
    preds = np.array([
        np.bincount(row.astype(int), minlength=int(y_train.max()) + 1).argmax()
        for row in nn_labels
    ])
    return float(np.mean(preds == y_val))


class DataValuator(BaseEstimator):
    """Estimate the value of each training sample using data valuation methods.

    Parameters
    ----------
    method : str
        Valuation method: ``'knn_shapley'``, ``'tmc_shapley'``, or ``'loo'``.
    n_neighbors : int
        Number of neighbors for KNN-based methods.
    n_mc_iterations : int
        Number of Monte Carlo iterations for ``'tmc_shapley'``.
    random_state : int or None
        Random seed for reproducibility.

    Attributes
    ----------
    values_ : np.ndarray
        Per-sample value scores after calling :meth:`fit`.

    Examples
    --------
    >>> valuator = DataValuator(method='knn_shapley')
    >>> valuator.fit(X_train, y_train)
    >>> high = valuator.get_high_value_indices(top_k=10)
    """

    def __init__(
        self,
        method: str = "knn_shapley",
        n_neighbors: int = 5,
        n_mc_iterations: int = 500,
        random_state: int | None = None,
    ):
        self.method = method
        self.n_neighbors = n_neighbors
        self.n_mc_iterations = n_mc_iterations
        self.random_state = random_state

    @property
    def values(self) -> np.ndarray:
        """Per-sample value scores (raises if not fitted)."""
        if not hasattr(self, "values_"):
            raise RuntimeError("Call fit() before accessing values.")
        return self.values_

    def fit(self, X: Any, y: Any) -> DataValuator:
        """Compute per-sample data values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels (classification).

        Returns
        -------
        self
        """
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y).ravel()

        X_arr = np.nan_to_num(X_arr, nan=0.0)

        if self.method == "knn_shapley":
            self.values_ = self._knn_shapley(X_arr, y_arr)
        elif self.method == "tmc_shapley":
            self.values_ = self._tmc_shapley(X_arr, y_arr)
        elif self.method == "loo":
            self.values_ = self._loo(X_arr, y_arr)
        else:
            raise ValueError(f"Unknown method: {self.method!r}")

        return self

    def get_high_value_indices(
        self, top_k: int | None = None, threshold: float | None = None
    ) -> np.ndarray:
        """Return indices of highest-value samples.

        Parameters
        ----------
        top_k : int, optional
            Return the top-k samples by value.
        threshold : float, optional
            Return samples with value above this threshold.

        Returns
        -------
        np.ndarray
            Indices sorted by descending value.
        """
        vals = self.values
        order = np.argsort(vals)[::-1]
        if top_k is not None:
            return order[:top_k]
        if threshold is not None:
            return order[vals[order] >= threshold]
        return order

    def get_low_value_indices(
        self, bottom_k: int | None = None, threshold: float | None = None
    ) -> np.ndarray:
        """Return indices of lowest-value samples.

        Parameters
        ----------
        bottom_k : int, optional
            Return the bottom-k samples by value.
        threshold : float, optional
            Return samples with value below this threshold.

        Returns
        -------
        np.ndarray
            Indices sorted by ascending value.
        """
        vals = self.values
        order = np.argsort(vals)
        if bottom_k is not None:
            return order[:bottom_k]
        if threshold is not None:
            return order[vals[order] <= threshold]
        return order

    # ------------------------------------------------------------------
    # Private methods
    # ------------------------------------------------------------------

    def _knn_shapley(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Exact KNN-Shapley via the recursive formula of Jia et al. (2019).

        Complexity: O(n^2 log n) per test point (using all training data as
        test points via leave-one-out).
        """
        n = len(X)
        k = min(self.n_neighbors, n - 1)
        values = np.zeros(n)

        # For each test point (leave-one-out)
        for i in range(n):
            x_test = X[i]
            y_test = y[i]

            # All other points
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            X_others = X[mask]
            y_others = y[mask]
            m = len(X_others)

            # Sort by distance to test point
            dists = np.sum((X_others - x_test) ** 2, axis=1)
            order = np.argsort(dists)

            # Recursive Shapley computation
            # phi[j] = Shapley value of the j-th nearest neighbor
            phi = np.zeros(m)

            # Base case: furthest point
            phi[m - 1] = (1.0 / m) * (1.0 if y_others[order[m - 1]] == y_test else 0.0)

            # Recurse from second-furthest to nearest
            for j in range(m - 2, -1, -1):
                phi[j] = phi[j + 1]
                kj = min(k, j + 1)
                indicator_j = 1.0 if y_others[order[j]] == y_test else 0.0
                indicator_j1 = 1.0 if y_others[order[j + 1]] == y_test else 0.0
                phi[j] += (1.0 / kj) * (indicator_j - indicator_j1)

            # Map back to original indices
            original_idx = np.where(mask)[0]
            for j in range(m):
                values[original_idx[order[j]]] += phi[j]

        # Average over test points
        values /= n
        return values

    def _tmc_shapley(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Truncated Monte Carlo Shapley approximation."""
        rng = np.random.RandomState(self.random_state)
        n = len(X)
        values = np.zeros(n)

        # Hold out 10% as validation
        n_val = max(1, n // 10)
        val_idx = rng.choice(n, n_val, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        X_val, y_val = X[val_idx], y[val_idx]
        train_indices = np.where(train_mask)[0]

        for _ in range(self.n_mc_iterations):
            perm = rng.permutation(train_indices)
            prev_acc = 0.0

            for t, idx in enumerate(perm, 1):
                subset = perm[:t]
                acc = _knn_accuracy(X[subset], y[subset], X_val, y_val, self.n_neighbors)
                marginal = acc - prev_acc
                values[idx] += marginal
                prev_acc = acc

                # Early truncation if accuracy stabilizes
                if t > 10 and abs(marginal) < 1e-6:
                    break

        values /= self.n_mc_iterations
        return values

    def _loo(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Leave-one-out valuation: value = accuracy_all - accuracy_without_i."""
        n = len(X)
        values = np.zeros(n)

        # Use 10% holdout for evaluation
        rng = np.random.RandomState(self.random_state)
        n_val = max(1, n // 10)
        val_idx = rng.choice(n, n_val, replace=False)
        train_mask = np.ones(n, dtype=bool)
        train_mask[val_idx] = False
        X_val, y_val = X[val_idx], y[val_idx]
        train_indices = np.where(train_mask)[0]

        # Full accuracy
        full_acc = _knn_accuracy(X[train_indices], y[train_indices], X_val, y_val, self.n_neighbors)

        for idx in train_indices:
            loo_mask = train_mask.copy()
            loo_mask[idx] = False
            loo_indices = np.where(loo_mask)[0]
            loo_acc = _knn_accuracy(X[loo_indices], y[loo_indices], X_val, y_val, self.n_neighbors)
            values[idx] = full_acc - loo_acc

        return values
