"""Time series cross-validation and metrics.

This module extends the validation strategies in endgame.validation
with time series specific implementations.

Provides:
- Expanding window CV
- Sliding window CV
- Blocked time series split
- Forecasting metrics (MASE, SMAPE, etc.)
"""

from collections.abc import Generator
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class ExpandingWindowCV(BaseCrossValidator):
    """Expanding window cross-validation for time series.

    Training window expands with each fold while validation
    window remains fixed size.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    initial_train_size : int, optional
        Initial training set size. If None, computed from n_splits.
    val_size : int, optional
        Validation set size. If None, computed from n_splits.
    gap : int, default=0
        Number of samples to skip between train and validation.
    step_size : int, optional
        How many samples to add per fold. If None, uses val_size.

    Examples
    --------
    >>> cv = ExpandingWindowCV(n_splits=5, initial_train_size=100, val_size=20)
    >>> for train_idx, val_idx in cv.split(X):
    ...     # train_idx grows with each fold
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        initial_train_size: int | None = None,
        val_size: int | None = None,
        gap: int = 0,
        step_size: int | None = None,
    ):
        self.n_splits = n_splits
        self.initial_train_size = initial_train_size
        self.val_size = val_size
        self.gap = gap
        self.step_size = step_size

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices.

        Parameters
        ----------
        X : array-like
            Input data.
        y : ignored
        groups : ignored

        Yields
        ------
        train_idx, val_idx : ndarray
            Training and validation indices for this fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Compute sizes if not provided
        val_size = self.val_size or (n_samples // (self.n_splits + 1))
        step_size = self.step_size or val_size

        # Initial train size
        if self.initial_train_size is not None:
            initial_size = self.initial_train_size
        else:
            initial_size = n_samples - (self.n_splits * (val_size + self.gap))

        if initial_size < 1:
            raise ValueError(
                f"initial_train_size ({initial_size}) must be > 0. "
                "Try reducing n_splits or providing explicit sizes."
            )

        for fold in range(self.n_splits):
            train_end = initial_size + fold * step_size
            val_start = train_end + self.gap
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = indices[:train_end]
            val_idx = indices[val_start:val_end]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


class SlidingWindowCV(BaseCrossValidator):
    """Sliding window cross-validation for time series.

    Both training and validation windows have fixed sizes
    and slide through the data.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    train_size : int, optional
        Training window size.
    val_size : int, optional
        Validation window size.
    gap : int, default=0
        Gap between train and validation.
    step_size : int, optional
        How many samples to slide per fold.

    Examples
    --------
    >>> cv = SlidingWindowCV(n_splits=5, train_size=100, val_size=20)
    >>> for train_idx, val_idx in cv.split(X):
    ...     # Both windows slide forward
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        train_size: int | None = None,
        val_size: int | None = None,
        gap: int = 0,
        step_size: int | None = None,
    ):
        self.n_splits = n_splits
        self.train_size = train_size
        self.val_size = val_size
        self.gap = gap
        self.step_size = step_size

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Compute sizes
        total_per_fold = n_samples // (self.n_splits + 1)
        val_size = self.val_size or (total_per_fold // 3)
        train_size = self.train_size or (total_per_fold - val_size - self.gap)
        step_size = self.step_size or val_size

        for fold in range(self.n_splits):
            train_start = fold * step_size
            train_end = train_start + train_size
            val_start = train_end + self.gap
            val_end = val_start + val_size

            if val_end > n_samples:
                break

            train_idx = indices[train_start:train_end]
            val_idx = indices[val_start:val_end]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


class BlockedTimeSeriesSplit(BaseCrossValidator):
    """Blocked time series split for reducing temporal leakage.

    Splits data into blocks that respect temporal structure
    and adds gaps to prevent leakage from adjacent periods.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    gap_before : int, default=0
        Gap samples before validation set.
    gap_after : int, default=0
        Gap samples after validation set (embargo).

    Examples
    --------
    >>> cv = BlockedTimeSeriesSplit(n_splits=5, gap_before=10, gap_after=5)
    >>> for train_idx, val_idx in cv.split(X):
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        gap_before: int = 0,
        gap_after: int = 0,
    ):
        self.n_splits = n_splits
        self.gap_before = gap_before
        self.gap_after = gap_after

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices with gaps."""
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Divide into n_splits + 1 blocks (training gets multiple blocks)
        fold_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            val_start = (fold + 1) * fold_size
            val_end = val_start + fold_size if fold < self.n_splits - 1 else n_samples

            # Training: all blocks before validation, minus gap
            train_end = val_start - self.gap_before
            train_idx = indices[:max(0, train_end)]

            # Validation
            val_idx = indices[val_start:val_end]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


# Forecasting metrics
def mase(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
    seasonal_period: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales MAE by the in-sample MAE of a naive seasonal forecast.
    MASE < 1 means the model outperforms seasonal naive.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Training data for computing scaling factor.
    seasonal_period : int, default=1
        Seasonal period for naive forecast (1 = random walk).

    Returns
    -------
    float
        MASE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Compute MAE of forecast
    forecast_mae = np.mean(np.abs(y_true - y_pred))

    # Compute in-sample MAE of seasonal naive
    if len(y_train) > seasonal_period:
        naive_errors = y_train[seasonal_period:] - y_train[:-seasonal_period]
        scale = np.mean(np.abs(naive_errors))
    else:
        scale = np.mean(np.abs(np.diff(y_train)))

    if scale == 0:
        return float('inf') if forecast_mae > 0 else 0.0

    return forecast_mae / scale


def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        SMAPE in percentage (0-200%).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    denom = np.abs(y_true) + np.abs(y_pred)
    mask = denom != 0

    if not np.any(mask):
        return 0.0

    return np.mean(2 * np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        MAPE in percentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    mask = y_true != 0

    if not np.any(mask):
        return float('inf')

    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def rmsse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: np.ndarray,
) -> float:
    """Root Mean Squared Scaled Error.

    Used in M5 forecasting competition.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.
    y_train : array-like
        Training data for scaling.

    Returns
    -------
    float
        RMSSE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_train = np.asarray(y_train)

    # Compute MSE of forecast
    forecast_mse = np.mean((y_true - y_pred) ** 2)

    # Compute in-sample MSE of naive (random walk)
    if len(y_train) > 1:
        naive_errors = y_train[1:] - y_train[:-1]
        scale = np.mean(naive_errors ** 2)
    else:
        scale = np.var(y_train)

    if scale == 0:
        return float('inf') if forecast_mse > 0 else 0.0

    return np.sqrt(forecast_mse / scale)


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Weighted Absolute Percentage Error.

    Parameters
    ----------
    y_true : array-like
        True values.
    y_pred : array-like
        Predicted values.

    Returns
    -------
    float
        WAPE in percentage.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    total_true = np.sum(np.abs(y_true))

    if total_true == 0:
        return 0.0 if np.sum(np.abs(y_pred)) == 0 else float('inf')

    return np.sum(np.abs(y_true - y_pred)) / total_true * 100


def coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute prediction interval coverage.

    Parameters
    ----------
    y_true : array-like
        True values.
    lower : array-like
        Lower bounds of prediction intervals.
    upper : array-like
        Upper bounds of prediction intervals.

    Returns
    -------
    float
        Coverage probability (0-1).
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    in_interval = (y_true >= lower) & (y_true <= upper)
    return np.mean(in_interval)


def interval_width(lower: np.ndarray, upper: np.ndarray) -> float:
    """Compute average prediction interval width.

    Parameters
    ----------
    lower : array-like
        Lower bounds.
    upper : array-like
        Upper bounds.

    Returns
    -------
    float
        Mean interval width.
    """
    return np.mean(np.asarray(upper) - np.asarray(lower))


def winkler_score(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """Compute Winkler score for prediction intervals.

    Combines coverage and sharpness. Lower is better.

    Parameters
    ----------
    y_true : array-like
        True values.
    lower : array-like
        Lower bounds.
    upper : array-like
        Upper bounds.
    alpha : float, default=0.05
        Significance level (1 - coverage).

    Returns
    -------
    float
        Winkler score.
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    width = upper - lower

    # Penalty for below lower bound
    below = np.maximum(0, lower - y_true)

    # Penalty for above upper bound
    above = np.maximum(0, y_true - upper)

    score = width + (2 / alpha) * below + (2 / alpha) * above

    return np.mean(score)
