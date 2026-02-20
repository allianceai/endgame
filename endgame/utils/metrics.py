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
