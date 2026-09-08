from __future__ import annotations

"""Competition-specific metrics not in sklearn."""

from collections.abc import Callable

import numpy as np
from sklearn.metrics import cohen_kappa_score


def quadratic_weighted_kappa(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int] | None = None,
) -> float:
    """Quadratic Weighted Kappa (QWK) metric.

    Used in education competitions (e.g., essay scoring).
    Measures agreement between two ratings with quadratic weighting.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    labels : List[int], optional
        List of labels to use for the confusion matrix.

    Returns
    -------
    float
        QWK score in range [-1, 1], where 1 is perfect agreement.

    Examples
    --------
    >>> y_true = [1, 2, 3, 4, 5]
    >>> y_pred = [1, 2, 3, 4, 4]
    >>> qwk = quadratic_weighted_kappa(y_true, y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Round predictions if they are floats
    if y_pred.dtype in [np.float32, np.float64]:
        y_pred = np.round(y_pred).astype(int)

    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)


def mean_average_precision(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int | None = None,
) -> float:
    """Mean Average Precision (MAP).

    Computes the mean of average precision scores for each sample.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_classes) or (n_samples,)
        True relevance labels (binary).
    y_pred : array-like of shape (n_samples, n_classes) or (n_samples,)
        Predicted scores.
    k : int, optional
        Consider only top k predictions.

    Returns
    -------
    float
        MAP score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    n_samples = y_true.shape[0]
    avg_precisions = []

    for i in range(n_samples):
        true_i = y_true[i]
        pred_i = y_pred[i]

        # Sort by predicted scores
        sorted_indices = np.argsort(pred_i)[::-1]

        if k is not None:
            sorted_indices = sorted_indices[:k]

        # Compute AP
        n_relevant = 0
        precision_sum = 0.0

        for j, idx in enumerate(sorted_indices):
            if true_i[idx] == 1:
                n_relevant += 1
                precision_sum += n_relevant / (j + 1)

        if n_relevant > 0:
            avg_precisions.append(precision_sum / min(n_relevant, len(sorted_indices)))
        else:
            avg_precisions.append(0.0)

    return np.mean(avg_precisions)


def map_at_k(
    y_true: list[list[int]] | np.ndarray,
    y_pred: list[list[int]] | np.ndarray,
    k: int = 5,
) -> float:
    """Mean Average Precision @ K.

    For ranking competitions where each sample has multiple relevant items.

    Parameters
    ----------
    y_true : List[List[int]]
        List of relevant item indices for each sample.
    y_pred : List[List[int]]
        List of predicted item indices (ranked) for each sample.
    k : int, default=5
        Number of predictions to consider.

    Returns
    -------
    float
        MAP@K score.

    Examples
    --------
    >>> y_true = [[1, 2, 3], [4, 5]]
    >>> y_pred = [[1, 3, 5, 2, 4], [4, 1, 5, 2, 3]]
    >>> score = map_at_k(y_true, y_pred, k=5)
    """
    n_samples = len(y_true)
    avg_precisions = []

    for true_items, pred_items in zip(y_true, y_pred):
        true_set = set(true_items)
        pred_items = list(pred_items)[:k]

        if not true_set:
            avg_precisions.append(0.0)
            continue

        n_relevant = 0
        precision_sum = 0.0

        for i, item in enumerate(pred_items):
            if item in true_set:
                n_relevant += 1
                precision_sum += n_relevant / (i + 1)

        avg_precisions.append(precision_sum / min(len(true_set), k))

    return np.mean(avg_precisions)


def apk(actual: list[int], predicted: list[int], k: int = 10) -> float:
    """Average Precision @ K for a single sample.

    Parameters
    ----------
    actual : List[int]
        List of relevant items.
    predicted : List[int]
        List of predicted items (ranked).
    k : int, default=10
        Number of predictions to consider.

    Returns
    -------
    float
        AP@K score.
    """
    if not actual:
        return 0.0

    predicted = predicted[:k]
    actual_set = set(actual)

    score = 0.0
    num_hits = 0.0

    for i, p in enumerate(predicted):
        if p in actual_set and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / min(len(actual), k)


def ndcg_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """Normalized Discounted Cumulative Gain @ K.

    Used in ranking competitions.

    Parameters
    ----------
    y_true : array-like
        True relevance scores.
    y_pred : array-like
        Predicted scores.
    k : int, default=10
        Number of predictions to consider.

    Returns
    -------
    float
        NDCG@K score in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # DCG
    def dcg(scores: np.ndarray, k: int) -> float:
        scores = scores[:k]
        gains = 2 ** scores - 1
        discounts = np.log2(np.arange(len(scores)) + 2)
        return np.sum(gains / discounts)

    # Sort by predicted scores
    sorted_indices = np.argsort(y_pred)[::-1]
    sorted_true = y_true[sorted_indices]

    # Ideal sorting
    ideal_sorted = np.sort(y_true)[::-1]

    dcg_score = dcg(sorted_true, k)
    idcg_score = dcg(ideal_sorted, k)

    if idcg_score == 0:
        return 0.0

    return dcg_score / idcg_score


def mcrmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """Mean Columnwise Root Mean Squared Error.

    Used in multi-target regression competitions.

    Parameters
    ----------
    y_true : array-like of shape (n_samples, n_targets)
        True values.
    y_pred : array-like of shape (n_samples, n_targets)
        Predicted values.

    Returns
    -------
    float
        MCRMSE score.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    rmse_per_col = np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
    return np.mean(rmse_per_col)


def competition_metric(metric_name: str) -> Callable:
    """Get metric function by name.

    Handles both sklearn metrics and competition-specific metrics.

    Parameters
    ----------
    metric_name : str
        Metric name: 'qwk', 'map_at_k', 'ndcg', 'mcrmse', etc.

    Returns
    -------
    Callable
        Metric function.
    """
    custom_metrics = {
        "qwk": quadratic_weighted_kappa,
        "quadratic_weighted_kappa": quadratic_weighted_kappa,
        "map": mean_average_precision,
        "map_at_k": map_at_k,
        "ndcg": ndcg_at_k,
        "ndcg_at_k": ndcg_at_k,
        "mcrmse": mcrmse,
    }

    if metric_name.lower() in custom_metrics:
        return custom_metrics[metric_name.lower()]

    # Try sklearn metrics
    try:
        from sklearn.metrics import get_scorer
        scorer = get_scorer(metric_name)
        return scorer._score_func
    except Exception:
        raise ValueError(
            f"Unknown metric: {metric_name}. "
            f"Available custom metrics: {list(custom_metrics.keys())}"
        )


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    y_score,
    n_boot: int = 1000,
    ci: float = 0.95,
    stratified: bool = True,
    random_state: int | None = 0,
) -> tuple[float, float, float]:
    """Point estimate and percentile bootstrap confidence interval of a metric.

    Resamples (y_true, y_score) pairs with replacement, by class when ``stratified``
    (keeps the class balance of every replicate, which matters for AUROC/AUPRC on
    imbalanced data). Returns (estimate, lower, upper).

    Examples
    --------
    >>> from sklearn.metrics import roc_auc_score
    >>> from endgame.utils.metrics import bootstrap_ci
    >>> auc, lo, hi = bootstrap_ci(roc_auc_score, y_test, proba[:, 1])
    """
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    rng = np.random.RandomState(random_state)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)] if stratified else [np.arange(len(y_true))]
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        try:
            stats.append(metric(y_true[idx], y_score[idx]))
        except ValueError:  # e.g. a replicate with a single class
            continue
    alpha = (1 - ci) / 2
    lo, hi = np.percentile(stats, [100 * alpha, 100 * (1 - alpha)])
    return float(metric(y_true, y_score)), float(lo), float(hi)


def paired_bootstrap_diff(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y_true,
    score_a,
    score_b,
    n_boot: int = 1000,
    ci: float = 0.95,
    stratified: bool = True,
    random_state: int | None = 0,
) -> dict:
    """Paired bootstrap of ``metric(a) - metric(b)`` for two score vectors on the *same* subjects.

    Two overlapping confidence intervals do not tell whether model A beats model B; resampling the same
    subjects for both keeps the pairing and gives the interval of the difference directly. Returns a dict
    with the point estimates, the difference, its percentile CI, and a two-sided bootstrap p-value
    (fraction of replicates on the other side of zero, doubled, floored at 1/n_boot).

    Examples
    --------
    >>> from sklearn.metrics import roc_auc_score
    >>> paired_bootstrap_diff(roc_auc_score, y, p_imaging, p_genetics_only)
    {'metric_a': 0.80, 'metric_b': 0.65, 'diff': 0.15, 'ci_lo': 0.09, 'ci_hi': 0.21, 'p_value': 0.001, 'n_boot': 1000}
    """
    y_true, a, b = np.asarray(y_true), np.asarray(score_a), np.asarray(score_b)
    rng = np.random.RandomState(random_state)
    groups = [np.flatnonzero(y_true == c) for c in np.unique(y_true)] if stratified else [np.arange(len(y_true))]
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(g, size=len(g), replace=True) for g in groups])
        try:
            diffs.append(metric(y_true[idx], a[idx]) - metric(y_true[idx], b[idx]))
        except ValueError:
            continue
    diffs = np.asarray(diffs)
    d = metric(y_true, a) - metric(y_true, b)
    alpha = (1 - ci) / 2
    p = 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()) if len(diffs) else np.nan
    return {"metric_a": float(metric(y_true, a)), "metric_b": float(metric(y_true, b)), "diff": float(d),
            "ci_lo": float(np.quantile(diffs, alpha)), "ci_hi": float(np.quantile(diffs, 1 - alpha)),
            "p_value": float(max(p, 1.0 / max(len(diffs), 1))), "n_boot": int(len(diffs))}


def _auc_placements(y_true, score):
    pos, neg = score[y_true == 1], score[y_true == 0]
    # placement values (DeLong 1988 / Sun & Xu 2014 fast implementation)
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order))
    ranks[order] = np.arange(1, len(order) + 1)
    all_scores = np.concatenate([pos, neg])
    # midranks for ties
    _, inv, counts = np.unique(all_scores, return_inverse=True, return_counts=True)
    start = np.cumsum(np.r_[0, counts[:-1]]) + 1
    mid = start + (counts - 1) / 2.0
    ranks = mid[inv]
    m, n = len(pos), len(neg)
    v10 = (ranks[:m] - np.arange(1, m + 1)) / n  # placement of positives among negatives (with tie midranks)
    # exact placements: fraction of negatives below each positive (+0.5 ties)
    v10 = np.array([((neg < p_).mean() + 0.5 * (neg == p_).mean()) for p_ in pos])
    v01 = np.array([((pos > n_).mean() + 0.5 * (pos == n_).mean()) for n_ in neg])
    return v10, v01


def delong_test(y_true, score_a, score_b) -> dict:
    """DeLong (1988) test for the difference of two correlated AUROCs on the same subjects.

    Returns the two AUCs, their difference, its standard error, the 95 % CI and the two-sided p-value.
    Exact placement values (O(m*n)); fine for the sample sizes of tabular biomarker studies.
    """
    from scipy import stats

    y_true, a, b = np.asarray(y_true).astype(int), np.asarray(score_a, float), np.asarray(score_b, float)
    va10, va01 = _auc_placements(y_true, a)
    vb10, vb01 = _auc_placements(y_true, b)
    auc_a, auc_b = va10.mean(), vb10.mean()
    m, n = len(va10), len(va01)
    s10 = np.cov(np.vstack([va10, vb10]))
    s01 = np.cov(np.vstack([va01, vb01]))
    S = s10 / m + s01 / n
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    se = float(np.sqrt(max(var_diff, 1e-12)))
    z = (auc_a - auc_b) / se
    p = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": float(auc_a), "auc_b": float(auc_b), "diff": float(auc_a - auc_b), "se": se,
            "ci_lo": float(auc_a - auc_b - 1.96 * se), "ci_hi": float(auc_a - auc_b + 1.96 * se), "z": float(z), "p_value": p}


def decision_curve(y_true, proba, thresholds=None) -> "pd.DataFrame":
    """Decision-curve analysis (Vickers & Elkin 2006): net benefit of treating when P(positive) >= threshold,
    against treating everyone and treating no one.

    net benefit = TP/N - FP/N * t/(1-t). Returns a DataFrame with columns threshold, net_benefit_model,
    net_benefit_all, net_benefit_none, and the fraction flagged.
    """
    import pandas as pd

    y_true, proba = np.asarray(y_true).astype(int), np.asarray(proba, float)
    thresholds = np.linspace(0.01, 0.99, 99) if thresholds is None else np.asarray(thresholds, float)
    N, prev = len(y_true), y_true.mean()
    rows = []
    for t in thresholds:
        flag = proba >= t
        tp, fp = float((flag & (y_true == 1)).sum()), float((flag & (y_true == 0)).sum())
        w = t / (1 - t)
        rows.append({"threshold": t, "net_benefit_model": tp / N - fp / N * w, "net_benefit_all": prev - (1 - prev) * w,
                     "net_benefit_none": 0.0, "fraction_flagged": float(flag.mean())})
    return pd.DataFrame(rows)
