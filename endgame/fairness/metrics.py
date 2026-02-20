"""Fairness metrics for evaluating model bias across sensitive groups.

Provides group-level fairness metrics that follow a consistent signature:
``f(y_true, y_pred, sensitive_attr) -> dict``.

All metrics accept numpy arrays, pandas Series, or lists and return a
dictionary containing per-group statistics plus an aggregate disparity
measure.

References
----------
- Hardt et al. "Equality of Opportunity in Supervised Learning" (2016)
- Feldman et al. "Certifying and Removing Disparate Impact" (2015)
- Chouldechova "Fair Prediction with Disparate Impact" (2017)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Union

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

ArrayLike = Union[np.ndarray, Sequence, "pd.Series"]


def _validate_inputs(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and convert inputs to numpy arrays.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (binary: 0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute defining groups.

    Returns
    -------
    tuple of np.ndarray
        Validated (y_true, y_pred, sensitive_attr).

    Raises
    ------
    ValueError
        If array lengths do not match or arrays are empty.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sensitive_attr = np.asarray(sensitive_attr)

    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} samples but y_pred has "
            f"{y_pred.shape[0]} samples."
        )
    if y_true.shape[0] != sensitive_attr.shape[0]:
        raise ValueError(
            f"y_true has {y_true.shape[0]} samples but sensitive_attr has "
            f"{sensitive_attr.shape[0]} samples."
        )
    if y_true.shape[0] == 0:
        raise ValueError("Input arrays must not be empty.")

    return y_true, y_pred, sensitive_attr


def _get_groups(
    sensitive_attr: np.ndarray,
) -> list:
    """Return sorted unique groups from the sensitive attribute.

    Parameters
    ----------
    sensitive_attr : np.ndarray
        Sensitive attribute array.

    Returns
    -------
    list
        Sorted unique group values.
    """
    groups = np.unique(sensitive_attr)
    return list(groups)


def demographic_parity(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> dict[str, Any]:
    """Compute demographic parity (statistical parity) across groups.

    Demographic parity requires that the positive prediction rate is equal
    across all groups defined by the sensitive attribute:

        P(Y_hat = 1 | A = a) = P(Y_hat = 1 | A = b)  for all a, b

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (not used in computation but included for
        API consistency).
    y_pred : array-like of shape (n_samples,)
        Predicted labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute defining groups.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"group_rates"`` : dict mapping group -> positive prediction rate
        - ``"max_disparity"`` : float, max difference between any two group rates
        - ``"ratio"`` : float, min(rate) / max(rate), 1.0 means perfect parity
        - ``"privileged_group"`` : group with highest positive rate
        - ``"unprivileged_group"`` : group with lowest positive rate

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0, 0])
    >>> sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    >>> result = demographic_parity(y_true, y_pred, sensitive)
    >>> result["group_rates"]
    {'A': 0.6666666666666666, 'B': 0.0}
    """
    y_true, y_pred, sensitive_attr = _validate_inputs(
        y_true, y_pred, sensitive_attr
    )
    groups = _get_groups(sensitive_attr)

    group_rates: dict[str, float] = {}
    for group in groups:
        mask = sensitive_attr == group
        group_preds = y_pred[mask]
        rate = float(np.mean(group_preds)) if len(group_preds) > 0 else 0.0
        group_rates[str(group)] = rate

    rates = list(group_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)

    privileged = max(group_rates, key=group_rates.get)  # type: ignore[arg-type]
    unprivileged = min(group_rates, key=group_rates.get)  # type: ignore[arg-type]

    return {
        "group_rates": group_rates,
        "max_disparity": max_rate - min_rate,
        "ratio": min_rate / max_rate if max_rate > 0 else 0.0,
        "privileged_group": privileged,
        "unprivileged_group": unprivileged,
    }


def equalized_odds(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> dict[str, Any]:
    """Compute equalized odds across groups.

    Equalized odds requires that the true positive rate (TPR) and false
    positive rate (FPR) are equal across groups:

        P(Y_hat = 1 | Y = y, A = a) = P(Y_hat = 1 | Y = y, A = b)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (binary: 0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute defining groups.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"group_tpr"`` : dict mapping group -> true positive rate
        - ``"group_fpr"`` : dict mapping group -> false positive rate
        - ``"tpr_disparity"`` : float, max difference in TPR across groups
        - ``"fpr_disparity"`` : float, max difference in FPR across groups
        - ``"max_disparity"`` : float, max of tpr_disparity and fpr_disparity
        - ``"satisfied"`` : bool, True if max_disparity < 0.05

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 1, 0, 0])
    >>> sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    >>> result = equalized_odds(y_true, y_pred, sensitive)
    >>> "group_tpr" in result and "group_fpr" in result
    True
    """
    y_true, y_pred, sensitive_attr = _validate_inputs(
        y_true, y_pred, sensitive_attr
    )
    groups = _get_groups(sensitive_attr)

    group_tpr: dict[str, float] = {}
    group_fpr: dict[str, float] = {}

    for group in groups:
        mask = sensitive_attr == group
        yt = y_true[mask]
        yp = y_pred[mask]

        # True positive rate
        pos_mask = yt == 1
        if pos_mask.sum() > 0:
            group_tpr[str(group)] = float(np.mean(yp[pos_mask]))
        else:
            group_tpr[str(group)] = float("nan")

        # False positive rate
        neg_mask = yt == 0
        if neg_mask.sum() > 0:
            group_fpr[str(group)] = float(np.mean(yp[neg_mask]))
        else:
            group_fpr[str(group)] = float("nan")

    tpr_values = [v for v in group_tpr.values() if not np.isnan(v)]
    fpr_values = [v for v in group_fpr.values() if not np.isnan(v)]

    tpr_disparity = (max(tpr_values) - min(tpr_values)) if len(tpr_values) >= 2 else 0.0
    fpr_disparity = (max(fpr_values) - min(fpr_values)) if len(fpr_values) >= 2 else 0.0
    max_disparity = max(tpr_disparity, fpr_disparity)

    return {
        "group_tpr": group_tpr,
        "group_fpr": group_fpr,
        "tpr_disparity": tpr_disparity,
        "fpr_disparity": fpr_disparity,
        "max_disparity": max_disparity,
        "satisfied": max_disparity < 0.05,
    }


def disparate_impact(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
) -> dict[str, Any]:
    """Compute disparate impact ratio across groups.

    Disparate impact measures the ratio of positive prediction rates between
    the least and most favored groups. The four-fifths rule considers a
    ratio below 0.8 as evidence of adverse impact.

        DI = P(Y_hat = 1 | A = unprivileged) / P(Y_hat = 1 | A = privileged)

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (not used in computation but included for
        API consistency).
    y_pred : array-like of shape (n_samples,)
        Predicted labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute defining groups.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"group_rates"`` : dict mapping group -> positive prediction rate
        - ``"disparate_impact_ratio"`` : float, min(rate)/max(rate)
        - ``"four_fifths_satisfied"`` : bool, True if ratio >= 0.8
        - ``"privileged_group"`` : group with highest positive rate
        - ``"unprivileged_group"`` : group with lowest positive rate

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 1, 1, 0, 1, 0])
    >>> sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    >>> result = disparate_impact(y_true, y_pred, sensitive)
    >>> result["four_fifths_satisfied"]
    True
    """
    y_true, y_pred, sensitive_attr = _validate_inputs(
        y_true, y_pred, sensitive_attr
    )
    groups = _get_groups(sensitive_attr)

    group_rates: dict[str, float] = {}
    for group in groups:
        mask = sensitive_attr == group
        group_preds = y_pred[mask]
        rate = float(np.mean(group_preds)) if len(group_preds) > 0 else 0.0
        group_rates[str(group)] = rate

    rates = list(group_rates.values())
    max_rate = max(rates)
    min_rate = min(rates)

    ratio = min_rate / max_rate if max_rate > 0 else 0.0

    privileged = max(group_rates, key=group_rates.get)  # type: ignore[arg-type]
    unprivileged = min(group_rates, key=group_rates.get)  # type: ignore[arg-type]

    return {
        "group_rates": group_rates,
        "disparate_impact_ratio": ratio,
        "four_fifths_satisfied": ratio >= 0.8,
        "privileged_group": privileged,
        "unprivileged_group": unprivileged,
    }


def calibration_by_group(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sensitive_attr: ArrayLike,
    y_proba: ArrayLike | None = None,
    n_bins: int = 10,
) -> dict[str, Any]:
    """Compute calibration metrics per sensitive group.

    Evaluates whether predicted probabilities (or predicted labels) are
    equally well-calibrated across groups. When ``y_proba`` is provided,
    computes Brier score and expected calibration error (ECE) per group.
    When only ``y_pred`` is provided, computes accuracy per group.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth labels (binary: 0 or 1).
    y_pred : array-like of shape (n_samples,)
        Predicted labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute defining groups.
    y_proba : array-like of shape (n_samples,), optional
        Predicted probabilities for the positive class. If provided,
        Brier score and ECE are computed per group.
    n_bins : int, default=10
        Number of bins for ECE computation.

    Returns
    -------
    dict
        Dictionary with keys:

        - ``"group_accuracy"`` : dict mapping group -> accuracy
        - ``"group_brier_score"`` : dict mapping group -> Brier score
          (only if y_proba is provided)
        - ``"group_ece"`` : dict mapping group -> ECE
          (only if y_proba is provided)
        - ``"accuracy_disparity"`` : float, max difference in accuracy
        - ``"max_brier_disparity"`` : float, max difference in Brier score
          (only if y_proba is provided)

    Examples
    --------
    >>> import numpy as np
    >>> y_true = np.array([1, 0, 1, 0, 1, 0])
    >>> y_pred = np.array([1, 0, 1, 0, 0, 0])
    >>> sensitive = np.array(["A", "A", "A", "B", "B", "B"])
    >>> result = calibration_by_group(y_true, y_pred, sensitive)
    >>> "group_accuracy" in result
    True
    """
    y_true, y_pred, sensitive_attr = _validate_inputs(
        y_true, y_pred, sensitive_attr
    )
    groups = _get_groups(sensitive_attr)

    has_proba = y_proba is not None
    if has_proba:
        y_proba = np.asarray(y_proba)
        if y_proba.shape[0] != y_true.shape[0]:
            raise ValueError(
                f"y_proba has {y_proba.shape[0]} samples but y_true has "
                f"{y_true.shape[0]} samples."
            )

    group_accuracy: dict[str, float] = {}
    group_brier: dict[str, float] = {}
    group_ece: dict[str, float] = {}

    for group in groups:
        mask = sensitive_attr == group
        yt = y_true[mask]
        yp = y_pred[mask]
        n = mask.sum()

        group_accuracy[str(group)] = float(np.mean(yt == yp)) if n > 0 else 0.0

        if has_proba:
            yprob = y_proba[mask]  # type: ignore[index]
            # Brier score: mean squared error of probabilities
            group_brier[str(group)] = float(np.mean((yprob - yt) ** 2)) if n > 0 else 0.0
            # ECE: expected calibration error
            group_ece[str(group)] = _compute_ece(yt, yprob, n_bins) if n > 0 else 0.0

    accuracies = list(group_accuracy.values())
    accuracy_disparity = max(accuracies) - min(accuracies) if len(accuracies) >= 2 else 0.0

    result: dict[str, Any] = {
        "group_accuracy": group_accuracy,
        "accuracy_disparity": accuracy_disparity,
    }

    if has_proba:
        brier_values = list(group_brier.values())
        result["group_brier_score"] = group_brier
        result["group_ece"] = group_ece
        result["max_brier_disparity"] = (
            max(brier_values) - min(brier_values) if len(brier_values) >= 2 else 0.0
        )

    return result


def _compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error for a single group.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    y_proba : np.ndarray
        Predicted probabilities.
    n_bins : int
        Number of bins.

    Returns
    -------
    float
        Expected calibration error.
    """
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = float(np.mean(y_proba[mask]))
        avg_accuracy = float(np.mean(y_true[mask]))
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return ece
