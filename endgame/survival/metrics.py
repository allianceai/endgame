"""Survival analysis metrics.

Provides concordance index (pure NumPy), plus wrappers for scikit-survival
metrics when available.

Example
-------
>>> from endgame.survival.metrics import concordance_index
>>> from endgame.survival.base import make_survival_y
>>> import numpy as np
>>> y = make_survival_y([5, 3, 8, 1], [True, False, True, True])
>>> risk = np.array([0.8, 0.3, 0.1, 0.9])  # higher = more risk
>>> concordance_index(y, risk)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np

from endgame.survival.base import _check_survival_y, _get_time_event

try:
    import sksurv.metrics as sksurv_metrics

    HAS_SKSURV = True
except ImportError:
    HAS_SKSURV = False


# ---------------------------------------------------------------------------
# Concordance index (pure NumPy)
# ---------------------------------------------------------------------------


def concordance_index(
    y_true: Any,
    risk_scores: np.ndarray,
) -> float:
    """Harrell's concordance index (C-index).

    Measures the fraction of concordant pairs: pairs where the subject
    with the higher risk score had a shorter survival time.

    Parameters
    ----------
    y_true : structured array or compatible
        Survival target with 'event' and 'time' fields.
    risk_scores : array-like of shape (n_samples,)
        Predicted risk scores (higher = more risk / shorter survival).

    Returns
    -------
    c_index : float
        Value in [0, 1]. 1.0 = perfect discrimination, 0.5 = random.
    """
    y = _check_survival_y(y_true)
    time, event = _get_time_event(y)
    risk_scores = np.asarray(risk_scores, dtype=np.float64).ravel()

    if len(time) != len(risk_scores):
        raise ValueError(
            f"y_true and risk_scores must have the same length, "
            f"got {len(time)} and {len(risk_scores)}"
        )

    return _concordance_index_numpy(time, event, risk_scores)


def concordance_index_censored(
    event: np.ndarray,
    time: np.ndarray,
    risk_scores: np.ndarray,
) -> float:
    """Concordance index from separate event and time arrays.

    Parameters
    ----------
    event : array-like of shape (n_samples,)
        Event indicator (True = event, False = censored).
    time : array-like of shape (n_samples,)
        Observed time.
    risk_scores : array-like of shape (n_samples,)
        Predicted risk scores.

    Returns
    -------
    c_index : float
    """
    time = np.asarray(time, dtype=np.float64)
    event = np.asarray(event, dtype=bool)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)
    return _concordance_index_numpy(time, event, risk_scores)


def _concordance_index_numpy(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,
) -> float:
    """Pure NumPy concordance index computation.

    A pair (i, j) is comparable if:
    - Both have events: t_i != t_j
    - One has event, one censored: t_i < t_j and i had event

    Concordant if risk_i > risk_j when t_i < t_j.
    """
    n = len(time)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # i had an event; pair is comparable if t_i < t_j
            # or if both had events and t_j < t_i (handled when j is outer)
            if time[i] < time[j]:
                # i died first — concordant if risk_i > risk_j
                if risk[i] > risk[j]:
                    concordant += 1
                elif risk[i] < risk[j]:
                    discordant += 1
                else:
                    tied_risk += 1
            elif time[i] == time[j] and event[j]:
                # Both events at same time — skip (not comparable)
                pass

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + 0.5 * tied_risk) / total


# ---------------------------------------------------------------------------
# Integrated Brier Score (requires sksurv)
# ---------------------------------------------------------------------------


def integrated_brier_score(
    y_train: Any,
    y_test: Any,
    survival_probs: np.ndarray,
    times: np.ndarray,
) -> float:
    """Integrated Brier Score (IBS) for survival predictions.

    Lower is better. Measures calibration and discrimination.

    Parameters
    ----------
    y_train : structured array
        Training survival data (for IPCW censoring weights).
    y_test : structured array
        Test survival data.
    survival_probs : ndarray of shape (n_test_samples, n_times)
        Predicted survival probabilities S(t|X) at each time point.
    times : array-like of shape (n_times,)
        Time points for evaluation.

    Returns
    -------
    ibs : float

    Raises
    ------
    ImportError
        If scikit-survival is not installed.
    """
    if not HAS_SKSURV:
        raise ImportError(
            "integrated_brier_score requires scikit-survival. "
            "Install with: pip install scikit-survival"
        )
    y_train = _check_survival_y(y_train)
    y_test = _check_survival_y(y_test)
    times = np.asarray(times, dtype=np.float64)
    survival_probs = np.asarray(survival_probs, dtype=np.float64)

    # sksurv expects structured arrays — convert
    y_train_s = _to_sksurv_structured(y_train)
    y_test_s = _to_sksurv_structured(y_test)

    _, ibs = sksurv_metrics.integrated_brier_score(
        y_train_s, y_test_s, survival_probs, times
    )
    return float(ibs)


def brier_score(
    y_train: Any,
    y_test: Any,
    survival_probs: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Time-point Brier scores for survival predictions.

    Parameters
    ----------
    y_train : structured array
        Training survival data (for IPCW censoring weights).
    y_test : structured array
        Test survival data.
    survival_probs : ndarray of shape (n_test_samples, n_times)
        Predicted survival probabilities S(t|X) at each time point.
    times : array-like of shape (n_times,)
        Time points for evaluation.

    Returns
    -------
    scores : ndarray of shape (n_times,)
        Brier score at each time point.
    """
    if not HAS_SKSURV:
        raise ImportError(
            "brier_score requires scikit-survival. "
            "Install with: pip install scikit-survival"
        )
    y_train = _check_survival_y(y_train)
    y_test = _check_survival_y(y_test)
    times = np.asarray(times, dtype=np.float64)
    survival_probs = np.asarray(survival_probs, dtype=np.float64)

    y_train_s = _to_sksurv_structured(y_train)
    y_test_s = _to_sksurv_structured(y_test)

    times_out, scores = sksurv_metrics.brier_score(
        y_train_s, y_test_s, survival_probs, times
    )
    return scores


# ---------------------------------------------------------------------------
# Cumulative/Dynamic AUC (requires sksurv)
# ---------------------------------------------------------------------------


def cumulative_dynamic_auc(
    y_train: Any,
    y_test: Any,
    risk_scores: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Cumulative/dynamic AUC at specified time points.

    Parameters
    ----------
    y_train : structured array
        Training survival data (for IPCW censoring weights).
    y_test : structured array
        Test survival data.
    risk_scores : array-like of shape (n_test_samples,)
        Predicted risk scores.
    times : array-like of shape (n_times,)
        Time points for evaluation.

    Returns
    -------
    auc_scores : ndarray of shape (n_times,)
        AUC at each time point.
    mean_auc : float
        Mean AUC across all time points.
    """
    if not HAS_SKSURV:
        raise ImportError(
            "cumulative_dynamic_auc requires scikit-survival. "
            "Install with: pip install scikit-survival"
        )
    y_train = _check_survival_y(y_train)
    y_test = _check_survival_y(y_test)
    times = np.asarray(times, dtype=np.float64)
    risk_scores = np.asarray(risk_scores, dtype=np.float64)

    y_train_s = _to_sksurv_structured(y_train)
    y_test_s = _to_sksurv_structured(y_test)

    auc, mean_auc = sksurv_metrics.cumulative_dynamic_auc(
        y_train_s, y_test_s, risk_scores, times
    )
    return auc, float(mean_auc)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibration_curve_survival(
    y_true: Any,
    survival_probs_at_t: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibration curve for survival predictions at a single time point.

    Groups predictions into bins and compares predicted vs observed survival.

    Parameters
    ----------
    y_true : structured array
        Survival target.
    survival_probs_at_t : array-like of shape (n_samples,)
        Predicted S(t) at a specific time point for each sample.
    n_bins : int, default=10
        Number of bins for calibration.

    Returns
    -------
    predicted : ndarray of shape (n_bins,)
        Mean predicted survival probability in each bin.
    observed : ndarray of shape (n_bins,)
        Observed (Kaplan-Meier) survival probability in each bin.
    """
    y = _check_survival_y(y_true)
    probs = np.asarray(survival_probs_at_t, dtype=np.float64)
    time, event = _get_time_event(y)

    # Bin by predicted survival probability
    bin_edges = np.linspace(0, 1, n_bins + 1)
    predicted = np.zeros(n_bins)
    observed = np.zeros(n_bins)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)

        if mask.sum() == 0:
            predicted[i] = (lo + hi) / 2
            observed[i] = np.nan
            continue

        predicted[i] = probs[mask].mean()
        # Kaplan-Meier estimate for observed survival in this bin
        bin_time = time[mask]
        bin_event = event[mask]
        observed[i] = _km_survival_at_max_time(bin_time, bin_event)

    return predicted, observed


def _km_survival_at_max_time(time: np.ndarray, event: np.ndarray) -> float:
    """Quick KM estimate at the maximum observed time in the bin."""
    if len(time) == 0:
        return np.nan
    order = np.argsort(time)
    time_sorted = time[order]
    event_sorted = event[order]

    unique_times = np.unique(time_sorted)
    surv = 1.0
    for t in unique_times:
        at_risk = np.sum(time_sorted >= t)
        events = np.sum((time_sorted == t) & event_sorted)
        if at_risk > 0 and events > 0:
            surv *= 1.0 - events / at_risk
    return surv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_sksurv_structured(y: np.ndarray) -> np.ndarray:
    """Convert endgame survival array to sksurv structured array format."""
    # sksurv uses the same format: structured array with (event, time)
    out = np.empty(
        len(y), dtype=np.dtype([("event", bool), ("time", np.float64)])
    )
    out["event"] = y["event"]
    out["time"] = y["time"]
    return out
