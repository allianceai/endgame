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


def _score_inputs(y_true, *scores, binary=False):
    import pandas as pd

    y = np.asarray(y_true)
    if y.ndim != 1 or not len(y) or pd.isna(y).any():
        raise ValueError("y_true must be a nonempty, nonmissing one-dimensional vector")
    if y.dtype.kind in "fiu" and not np.isfinite(y).all():
        raise ValueError("y_true must be finite")
    if binary and (not np.isin(y, [0, 1]).all() or len(np.unique(y)) != 2):
        raise ValueError("y_true must contain both binary labels 0 and 1")
    arrays = []
    for values in scores:
        a = np.asarray(values, dtype=float)
        if a.ndim not in (1, 2) or len(a) != len(y) or not np.isfinite(a).all():
            raise ValueError("scores must be finite and aligned with y_true")
        arrays.append(a)
    return y, arrays


def _resampling_units(y, stratified, groups, strata):
    """Lists of independent row/cluster units and unit indices per stratum."""
    import pandas as pd
    from sklearn.utils.multiclass import type_of_target

    from endgame.validation._study import aligned_vector

    if not isinstance(stratified, (bool, np.bool_)):
        raise ValueError("stratified must be boolean; use strata for explicit labels")
    if stratified and strata is not None:
        raise ValueError("choose stratified=True or explicit strata, not both")
    if stratified:
        if type_of_target(y) not in ("binary", "multiclass"):
            raise ValueError("class stratification is invalid for continuous outcomes; use stratified=False")
        strata = y
    labels = np.zeros(len(y), dtype=int) if strata is None else aligned_vector(strata, len(y), "strata")
    if groups is None:
        units = [np.array([i]) for i in range(len(y))]
    else:
        g = aligned_vector(groups, len(y), "groups")
        codes, unique = pd.factorize(g, sort=False)
        units = [np.flatnonzero(codes == k) for k in range(len(unique))]
    if len(units) < 2:
        raise ValueError("at least two independent resampling units are required")
    unit_strata = []
    for unit in units:
        values = pd.unique(labels[unit])
        if len(values) != 1:
            raise ValueError("each patient/group must have a single stratum; use explicit group-level strata")
        unit_strata.append(values[0])
    codes, unique = pd.factorize(np.asarray(unit_strata, dtype=object), sort=False)
    strata_units = [np.flatnonzero(codes == k) for k in range(len(unique))]
    if all(len(s) == 1 for s in strata_units):
        raise ValueError("all strata are singletons: bootstrap cannot estimate sampling uncertainty")
    return units, strata_units


def _bootstrap_options(n_boot, ci):
    if isinstance(n_boot, bool) or not isinstance(n_boot, (int, np.integer)) or n_boot < 2:
        raise ValueError("n_boot must be an integer >= 2")
    if not np.isfinite(ci) or not 0 < ci < 1:
        raise ValueError("ci must be in (0, 1)")


def _metric_value(metric, y, score):
    value = float(metric(y, score))
    if not np.isfinite(value):
        raise ValueError("metric returned a nonfinite value")
    return value


def _replicate_value(metric, y, score):
    import warnings

    from sklearn.exceptions import UndefinedMetricWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UndefinedMetricWarning)
        return _metric_value(metric, y, score)


def _sample_rows(rng, units, strata_units):
    selected = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata_units])
    return np.concatenate([units[i] for i in selected])


def _check_replicates(values, requested):
    import warnings

    if len(values) < max(2, requested // 2):
        raise ValueError("too few valid resamples to estimate uncertainty")
    if len(values) < requested:
        warnings.warn(f"Only {len(values)}/{requested} resamples had a defined finite metric", UserWarning, stacklevel=3)


def bootstrap_ci(metric, y_true, y_score, n_boot=1000, ci=0.95, stratified=False,
                 random_state=0, *, groups=None, strata=None):
    """Percentile interval from paired observation or whole-patient resampling.

    Default resampling is unstratified and works for continuous outcomes.
    ``stratified=True`` explicitly requests class strata; ``strata`` supplies
    other row-aligned strata. With ``groups``, entire patients are resampled,
    and a stratum must be constant within each patient. This interval conditions
    on the supplied predictions; it does not include model-fitting uncertainty.
    """
    _bootstrap_options(n_boot, ci)
    y, (s,) = _score_inputs(y_true, y_score)
    estimate = _metric_value(metric, y, s)
    units, strata_units = _resampling_units(y, stratified, groups, strata)
    rng = np.random.RandomState(random_state)
    values = []
    for _ in range(n_boot):
        idx = _sample_rows(rng, units, strata_units)
        try:
            values.append(_replicate_value(metric, y[idx], s[idx]))
        except ValueError:
            continue
    _check_replicates(values, n_boot)
    alpha = (1-ci)/2
    lo, hi = np.quantile(values, [alpha, 1-alpha])
    return estimate, float(lo), float(hi)


def paired_bootstrap_diff(metric, y_true, score_a, score_b, n_boot=1000, ci=0.95,
                          stratified=False, random_state=0, *, groups=None, strata=None):
    """Paired bootstrap CI and paired-randomization test for two fixed models.

    The CI resamples the same observations/patients for both models. The p-value
    is a two-sided Monte Carlo paired permutation test: swap model predictions
    within each independent unit with probability 1/2, count absolute metric
    differences at least as extreme, and use (extreme+1)/(valid+1). Its null is
    exchangeability of the paired model predictions, not an unrestricted test
    of equal metrics. All visits of a patient swap together when groups is set.
    Training/model selection must be evaluated separately on independent data.
    """
    _bootstrap_options(n_boot, ci)
    y, (a, b) = _score_inputs(y_true, score_a, score_b)
    if a.shape != b.shape:
        raise ValueError("paired model score shapes must agree")
    ma, mb = _metric_value(metric, y, a), _metric_value(metric, y, b)
    diff = ma-mb
    units, strata_units = _resampling_units(y, stratified, groups, strata)
    rng = np.random.RandomState(random_state)
    diffs = []
    for _ in range(n_boot):
        idx = _sample_rows(rng, units, strata_units)
        try:
            diffs.append(_replicate_value(metric, y[idx], a[idx]) - _replicate_value(metric, y[idx], b[idx]))
        except ValueError:
            continue
    _check_replicates(diffs, n_boot)
    extreme = valid = 0
    for _ in range(n_boot):
        swap = np.zeros(len(y), bool)
        for unit, choose in zip(units, rng.randint(0, 2, len(units))):
            swap[unit] = choose
        mask = swap if a.ndim == 1 else swap[:, None]
        pa, pb = np.where(mask, b, a), np.where(mask, a, b)
        try:
            null_diff = _replicate_value(metric, y, pa) - _replicate_value(metric, y, pb)
        except ValueError:
            continue
        valid += 1
        extreme += abs(null_diff) >= abs(diff) - 1e-12
    if valid < max(2, n_boot//2):
        raise ValueError("too few valid paired permutations")
    alpha = (1-ci)/2
    lo, hi = np.quantile(diffs, [alpha, 1-alpha])
    return {"metric_a": ma, "metric_b": mb, "diff": diff, "ci_lo": float(lo), "ci_hi": float(hi),
            "p_value": float((extreme+1)/(valid+1)), "n_boot": len(diffs), "n_permutations": valid,
            "p_value_method": "paired_permutation"}


def _auc_placements(y_true, score):
    pos, neg = score[y_true == 1], score[y_true == 0]
    v10 = np.array([((neg < p).mean() + .5*(neg == p).mean()) for p in pos])
    v01 = np.array([((pos > n).mean() + .5*(pos == n).mean()) for n in neg])
    return v10, v01


def delong_test(y_true, score_a, score_b):
    """Correlated binary AUROC comparison for independent test subjects.

    Requires finite one-dimensional scores and at least two subjects per class.
    Repeated visits require a clustered method instead. Exact zero empirical
    variance is reported as zero (with a warning), not an invented variance floor.
    """
    import warnings

    from scipy import stats

    y, (a, b) = _score_inputs(y_true, score_a, score_b, binary=True)
    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("DeLong scores must be one-dimensional")
    m, n = int((y == 1).sum()), int((y == 0).sum())
    if min(m, n) < 2:
        raise ValueError("DeLong needs at least two subjects per class")
    va10, va01 = _auc_placements(y, a)
    vb10, vb01 = _auc_placements(y, b)
    diff = float(va10.mean()-vb10.mean())
    # Covariance of the paired difference, algebraically equivalent to c' S c.
    var = float(np.var(va10-vb10, ddof=1)/m + np.var(va01-vb01, ddof=1)/n)
    se = float(np.sqrt(var))
    if se == 0:
        z, p = (0., 1.) if diff == 0 else (float(np.copysign(np.inf, diff)), 0.)
        if diff != 0:
            warnings.warn("Zero empirical AUROC-difference variance; asymptotic inference is degenerate", UserWarning, stacklevel=2)
    else:
        z = diff/se
        p = float(2*stats.norm.sf(abs(z)))
    return {"auc_a": float(va10.mean()), "auc_b": float(vb10.mean()), "diff": diff, "se": se,
            "ci_lo": diff-1.96*se, "ci_hi": diff+1.96*se, "z": z, "p_value": p}


def decision_curve(y_true, proba, thresholds=None, *, prevalence=None):
    """Binary uncensored net benefit with explicit probability/shape validation.

    Pass positive-class probabilities as a vector, not an (n,1)/(n,2) matrix.
    Optional target-population prevalence weights sensitivity and false-positive
    rate for case-control sampling; it assumes these rates transport. It does not
    calibrate model probabilities. fraction_flagged describes the observed sample.
    Continuous progression or censored outcomes need different estimators.
    """
    import pandas as pd

    y, (p,) = _score_inputs(y_true, proba, binary=True)
    if p.ndim != 1 or (p < 0).any() or (p > 1).any():
        raise ValueError("proba must be a one-dimensional probability vector in [0, 1]")
    thresholds = np.linspace(.01, .99, 99) if thresholds is None else np.asarray(thresholds, float)
    if (thresholds.ndim != 1 or not len(thresholds) or not np.isfinite(thresholds).all()
            or (thresholds <= 0).any() or (thresholds >= 1).any()):
        raise ValueError("thresholds must be finite and strictly in (0, 1)")
    prev = float(y.mean()) if prevalence is None else float(prevalence)
    if not np.isfinite(prev) or not 0 < prev < 1:
        raise ValueError("prevalence must be in (0, 1)")
    pos, neg = y == 1, y == 0
    rows = []
    for t in thresholds:
        flag = p >= t
        w = t/(1-t)
        benefit = prev*flag[pos].mean() - (1-prev)*flag[neg].mean()*w
        rows.append({"threshold": t, "net_benefit_model": float(benefit),
                     "net_benefit_all": prev-(1-prev)*w, "net_benefit_none": 0.,
                     "fraction_flagged": float(flag.mean())})
    return pd.DataFrame(rows)
