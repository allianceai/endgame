"""Survival-specific cross-validation utilities.

Provides stratified CV splitters that preserve censoring rates across folds,
temporal splits for longitudinal data, and convenience evaluation functions.

Example
-------
>>> from endgame.survival.validation import SurvivalStratifiedKFold
>>> from endgame.survival.base import make_survival_y
>>> import numpy as np
>>> y = make_survival_y([5, 3, 8, 1, 6, 2], [True, False, True, True, False, True])
>>> X = np.random.randn(6, 3)
>>> cv = SurvivalStratifiedKFold(n_splits=2)
>>> for train_idx, test_idx in cv.split(X, y):
...     print(len(train_idx), len(test_idx))
3 3
3 3
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np
from sklearn.model_selection import StratifiedKFold

from endgame.survival.base import _check_survival_y, _get_time_event
from endgame.survival.metrics import concordance_index


# ---------------------------------------------------------------------------
# Stratified K-Fold for survival
# ---------------------------------------------------------------------------


class SurvivalStratifiedKFold(StratifiedKFold):
    """Stratified K-Fold cross-validation for survival data.

    Stratifies by the event indicator to ensure each fold has a similar
    censoring rate. This is important because highly imbalanced censoring
    can lead to unreliable metric estimates.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int or None, default=None
        Random seed when shuffle is True.

    Example
    -------
    >>> from endgame.survival.validation import SurvivalStratifiedKFold
    >>> from endgame.survival.base import make_survival_y
    >>> import numpy as np
    >>> y = make_survival_y([5, 3, 8, 1], [True, False, True, True])
    >>> cv = SurvivalStratifiedKFold(n_splits=2, random_state=0)
    >>> X = np.random.randn(4, 2)
    >>> for train, test in cv.split(X, y):
    ...     pass  # train and test indices
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            n_splits=n_splits, shuffle=shuffle, random_state=random_state
        )

    def split(
        self,
        X: np.ndarray,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices stratified by event indicator.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : structured array or compatible
            Survival target with 'event' and 'time' fields.
        groups : ignored
            Not used, present for API compatibility.

        Yields
        ------
        train_idx : ndarray
            Training set indices.
        test_idx : ndarray
            Test set indices.
        """
        y_checked = _check_survival_y(y)
        _, event = _get_time_event(y_checked)
        # Use event indicator as stratification variable
        yield from super().split(X, event, groups)


# ---------------------------------------------------------------------------
# Time Series Split for survival
# ---------------------------------------------------------------------------


class SurvivalTimeSeriesSplit:
    """Time-series cross-validation for longitudinal survival data.

    Splits data by observation time using an expanding window: training
    on earlier observations and testing on later ones.

    Parameters
    ----------
    n_splits : int, default=5
        Number of train/test splits.

    Example
    -------
    >>> from endgame.survival.validation import SurvivalTimeSeriesSplit
    >>> from endgame.survival.base import make_survival_y
    >>> import numpy as np
    >>> y = make_survival_y([1, 2, 3, 4, 5, 6], [True]*6)
    >>> X = np.random.randn(6, 2)
    >>> cv = SurvivalTimeSeriesSplit(n_splits=3)
    >>> for train, test in cv.split(X, y):
    ...     print(len(train), len(test))
    3 1
    4 1
    5 1
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def split(
        self,
        X: np.ndarray,
        y: Any = None,
        groups: Any = None,
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test indices sorted by observation time.

        Training set expands progressively while test set moves forward.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        y : structured array or compatible
            Survival target with 'event' and 'time' fields.
        groups : ignored
            Not used, present for API compatibility.

        Yields
        ------
        train_idx : ndarray
            Training set indices (earlier observations).
        test_idx : ndarray
            Test set indices (later observations).
        """
        y_checked = _check_survival_y(y)
        time, _ = _get_time_event(y_checked)
        n = len(time)

        # Sort indices by time
        sorted_idx = np.argsort(time)

        # Minimum training size: enough to have at least one split
        min_train = max(2, n // (self.n_splits + 1))
        test_size = max(1, (n - min_train) // self.n_splits)

        for i in range(self.n_splits):
            train_end = min_train + i * test_size
            test_end = min(train_end + test_size, n)

            if train_end >= n:
                break

            train_idx = sorted_idx[:train_end]
            test_idx = sorted_idx[train_end:test_end]

            if len(test_idx) == 0:
                break

            yield train_idx, test_idx

    def get_n_splits(
        self, X: Any = None, y: Any = None, groups: Any = None
    ) -> int:
        """Return the number of splits."""
        return self.n_splits


# ---------------------------------------------------------------------------
# Convenience evaluation function
# ---------------------------------------------------------------------------


def evaluate_survival(
    model: Any,
    X: np.ndarray,
    y: Any,
    cv: int | Any = 5,
    metrics: list[str] | None = None,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Cross-validated evaluation of a survival model.

    Parameters
    ----------
    model : survival estimator
        Must implement ``fit(X, y)`` and ``predict(X)`` returning risk scores.
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : structured array or compatible
        Survival target.
    cv : int or CV splitter, default=5
        Number of folds or a cross-validation splitter. If int, uses
        :class:`SurvivalStratifiedKFold`.
    metrics : list of str or None, default=None
        Metrics to compute. Default: ``["concordance_index"]``.
        Available: ``"concordance_index"``, ``"integrated_brier_score"``.
    return_predictions : bool, default=False
        If True, also return out-of-fold risk score predictions.

    Returns
    -------
    results : dict
        Dictionary with metric names as keys. Each value is a dict with
        ``"mean"``, ``"std"``, and ``"folds"`` (list of per-fold scores).
        If ``return_predictions=True``, also contains ``"oof_predictions"``
        (ndarray of out-of-fold risk scores).

    Example
    -------
    >>> # results = evaluate_survival(cox_model, X, y, cv=5)
    >>> # results["concordance_index"]["mean"]
    """
    from sklearn.base import clone

    y_checked = _check_survival_y(y)
    X = np.asarray(X, dtype=np.float64)

    if metrics is None:
        metrics = ["concordance_index"]

    # Set up CV splitter
    if isinstance(cv, int):
        cv_splitter = SurvivalStratifiedKFold(
            n_splits=cv, shuffle=True, random_state=42
        )
    else:
        cv_splitter = cv

    # Storage for fold results
    metric_scores: dict[str, list[float]] = {m: [] for m in metrics}
    oof_predictions = np.full(len(y_checked), np.nan)

    for train_idx, test_idx in cv_splitter.split(X, y_checked):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_checked[train_idx], y_checked[test_idx]

        # Clone and fit
        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        # Predict risk scores
        risk_scores = fold_model.predict(X_test)
        oof_predictions[test_idx] = risk_scores

        # Compute metrics
        for metric_name in metrics:
            if metric_name == "concordance_index":
                score = concordance_index(y_test, risk_scores)
            elif metric_name == "integrated_brier_score":
                try:
                    from endgame.survival.metrics import integrated_brier_score

                    sf = fold_model.predict_survival_function(X_test)
                    times = fold_model.event_times_
                    # Filter times within test range
                    test_time, _ = _get_time_event(y_test)
                    t_min, t_max = test_time.min(), test_time.max()
                    mask = (times > t_min) & (times < t_max)
                    if mask.sum() < 2:
                        score = np.nan
                    else:
                        score = integrated_brier_score(
                            y_train, y_test, sf[:, mask], times[mask]
                        )
                except (NotImplementedError, ImportError, Exception):
                    score = np.nan
            else:
                raise ValueError(f"Unknown metric: {metric_name!r}")

            metric_scores[metric_name].append(score)

    # Aggregate results
    results: dict[str, Any] = {}
    for metric_name, scores in metric_scores.items():
        arr = np.array(scores)
        results[metric_name] = {
            "mean": float(np.nanmean(arr)),
            "std": float(np.nanstd(arr)),
            "folds": [float(s) for s in arr],
        }

    if return_predictions:
        results["oof_predictions"] = oof_predictions

    return results


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------


def survival_train_test_split(
    X: np.ndarray,
    y: Any,
    test_size: float = 0.2,
    random_state: int | None = None,
    stratify_event: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train/test split that preserves event rate.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    y : structured array or compatible
        Survival target.
    test_size : float, default=0.2
        Fraction of samples for the test set.
    random_state : int or None, default=None
        Random seed for reproducibility.
    stratify_event : bool, default=True
        If True, stratify by event indicator to preserve censoring rate.

    Returns
    -------
    X_train : ndarray
        Training features.
    X_test : ndarray
        Test features.
    y_train : structured ndarray
        Training survival target.
    y_test : structured ndarray
        Test survival target.

    Example
    -------
    >>> from endgame.survival.validation import survival_train_test_split
    >>> from endgame.survival.base import make_survival_y
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = make_survival_y(np.random.exponential(10, 100), np.random.binomial(1, 0.7, 100))
    >>> X_tr, X_te, y_tr, y_te = survival_train_test_split(X, y, test_size=0.2, random_state=42)
    >>> X_tr.shape[0]
    80
    """
    from sklearn.model_selection import train_test_split

    y_checked = _check_survival_y(y)
    X = np.asarray(X, dtype=np.float64)

    if stratify_event:
        _, event = _get_time_event(y_checked)
        stratify = event
    else:
        stratify = None

    indices = np.arange(len(y_checked))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return X[train_idx], X[test_idx], y_checked[train_idx], y_checked[test_idx]
