from __future__ import annotations

"""Cross-validation utilities and helpers."""

from collections.abc import Callable
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier
from sklearn.metrics import get_scorer
from sklearn.model_selection import BaseCrossValidator, KFold, StratifiedKFold

from endgame.core.types import OOFResult


def cross_validate_oof(
    estimator: BaseEstimator,
    X: Any,
    y: Any,
    cv: int | BaseCrossValidator = 5,
    scoring: str | Callable | None = None,
    fit_params: dict[str, Any] | None = None,
    return_models: bool = True,
    return_indices: bool = False,
    groups: Any | None = None,
    verbose: bool = False,
) -> OOFResult:
    """Perform cross-validation and return out-of-fold predictions.

    This is the standard approach for building stacked ensembles and
    getting unbiased training set predictions.

    Parameters
    ----------
    estimator : sklearn-compatible estimator
        The model to cross-validate.
    X : array-like of shape (n_samples, n_features)
        Training features.
    y : array-like of shape (n_samples,)
        Target values.
    cv : int or CV splitter, default=5
        Cross-validation strategy.
    scoring : str or callable, optional
        Scoring metric. If None, uses estimator's default.
    fit_params : dict, optional
        Additional parameters to pass to estimator.fit().
    return_models : bool, default=True
        Whether to return trained models from each fold.
    return_indices : bool, default=False
        Whether to return train/val indices for each fold.
    groups : array-like, optional
        Group labels for group-aware CV.
    verbose : bool, default=False
        Print fold scores during cross-validation.

    Returns
    -------
    OOFResult
        - oof_predictions: Out-of-fold predictions
        - fold_scores: Validation score for each fold
        - mean_score: Mean score across folds
        - std_score: Standard deviation of scores
        - models: List of trained models (if return_models=True)
        - fold_indices: List of (train_idx, val_idx) tuples

    Examples
    --------
    >>> from endgame.validation import cross_validate_oof
    >>> result = cross_validate_oof(model, X, y, cv=5, scoring='roc_auc')
    >>> print(f"CV Score: {result.mean_score:.4f} ± {result.std_score:.4f}")
    """
    X = np.asarray(X)
    y = np.asarray(y)

    fit_params = fit_params or {}

    # Set up cross-validator
    if isinstance(cv, int):
        if is_classifier(estimator):
            cv = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        else:
            cv = KFold(n_splits=cv, shuffle=True, random_state=42)

    # Set up scorer
    if scoring is None:
        if is_classifier(estimator):
            scoring = "roc_auc"
        else:
            scoring = "neg_mean_squared_error"

    if isinstance(scoring, str):
        scorer = get_scorer(scoring)
    else:
        scorer = scoring

    # Initialize outputs
    n_samples = len(y)
    n_classes = len(np.unique(y)) if is_classifier(estimator) else None

    if is_classifier(estimator) and hasattr(estimator, "predict_proba"):
        if n_classes == 2:
            oof_predictions = np.zeros(n_samples)
        else:
            oof_predictions = np.zeros((n_samples, n_classes))
    else:
        oof_predictions = np.zeros(n_samples)

    fold_scores = []
    models = []
    fold_indices = []

    # Cross-validation loop
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Clone and fit model
        model = clone(estimator)

        # Handle early stopping for boosting models
        model_fit_params = fit_params.copy()
        if hasattr(model, "fit") and "eval_set" not in model_fit_params:
            # Check if model supports eval_set
            try:
                import inspect
                sig = inspect.signature(model.fit)
                if "eval_set" in sig.parameters:
                    model_fit_params["eval_set"] = [(X_val, y_val)]
            except Exception:
                pass

        model.fit(X_train, y_train, **model_fit_params)

        # Get predictions
        if is_classifier(estimator) and hasattr(model, "predict_proba"):
            val_pred = model.predict_proba(X_val)
            if n_classes == 2:
                val_pred = val_pred[:, 1]
            oof_predictions[val_idx] = val_pred
        else:
            oof_predictions[val_idx] = model.predict(X_val)

        # Compute fold score
        if callable(scorer):
            try:
                score = scorer(model, X_val, y_val)
            except Exception:
                # Fall back to using predictions
                if is_classifier(estimator):
                    from sklearn.metrics import roc_auc_score
                    try:
                        if n_classes == 2:
                            score = roc_auc_score(y_val, oof_predictions[val_idx])
                        else:
                            score = roc_auc_score(
                                y_val, oof_predictions[val_idx], multi_class="ovr"
                            )
                    except Exception:
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val, model.predict(X_val))
                else:
                    from sklearn.metrics import mean_squared_error
                    score = -mean_squared_error(y_val, oof_predictions[val_idx])
        else:
            score = scorer(model, X_val, y_val)

        fold_scores.append(score)

        if verbose:
            print(f"Fold {fold_idx + 1}: {score:.4f}")

        if return_models:
            models.append(model)

        if return_indices:
            fold_indices.append((train_idx, val_idx))

    mean_score = np.mean(fold_scores)
    std_score = np.std(fold_scores)

    if verbose:
        print(f"Mean: {mean_score:.4f} ± {std_score:.4f}")

    return OOFResult(
        oof_predictions=oof_predictions,
        fold_scores=fold_scores,
        mean_score=mean_score,
        std_score=std_score,
        models=models,
        fold_indices=fold_indices,
    )


def check_cv_lb_correlation(
    cv_scores: list[float],
    lb_scores: list[float],
) -> dict[str, float]:
    """Compute correlation between CV and leaderboard scores.

    Helps validate CV strategy by checking if CV improvements
    translate to LB improvements.

    Parameters
    ----------
    cv_scores : List[float]
        Cross-validation scores from different experiments.
    lb_scores : List[float]
        Corresponding public leaderboard scores.

    Returns
    -------
    Dict[str, float]
        - pearson: Pearson correlation coefficient
        - spearman: Spearman rank correlation
        - rmse: RMSE between normalized scores

    Examples
    --------
    >>> cv_scores = [0.85, 0.86, 0.87, 0.88]
    >>> lb_scores = [0.82, 0.83, 0.84, 0.85]
    >>> result = check_cv_lb_correlation(cv_scores, lb_scores)
    >>> print(f"Correlation: {result['pearson']:.3f}")
    """
    from scipy import stats

    cv_arr = np.array(cv_scores)
    lb_arr = np.array(lb_scores)

    if len(cv_arr) != len(lb_arr):
        raise ValueError("cv_scores and lb_scores must have same length")

    if len(cv_arr) < 3:
        raise ValueError("Need at least 3 data points for correlation")

    # Pearson correlation
    pearson_r, pearson_p = stats.pearsonr(cv_arr, lb_arr)

    # Spearman rank correlation
    spearman_r, spearman_p = stats.spearmanr(cv_arr, lb_arr)

    # RMSE of normalized scores
    cv_norm = (cv_arr - cv_arr.mean()) / (cv_arr.std() + 1e-8)
    lb_norm = (lb_arr - lb_arr.mean()) / (lb_arr.std() + 1e-8)
    rmse = np.sqrt(np.mean((cv_norm - lb_norm) ** 2))

    return {
        "pearson": pearson_r,
        "pearson_pvalue": pearson_p,
        "spearman": spearman_r,
        "spearman_pvalue": spearman_p,
        "rmse": rmse,
    }


def compute_oof_score(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    metric: str | Callable,
    **metric_kwargs,
) -> float:
    """Compute score from out-of-fold predictions.

    Parameters
    ----------
    oof_predictions : np.ndarray
        Out-of-fold predictions.
    y_true : np.ndarray
        True target values.
    metric : str or callable
        Metric to compute. String options: 'auc', 'logloss', 'accuracy',
        'f1', 'rmse', 'mae', 'r2'.
    **metric_kwargs
        Additional arguments for metric function.

    Returns
    -------
    float
        Computed score.
    """
    if callable(metric):
        return metric(y_true, oof_predictions, **metric_kwargs)

    metric = metric.lower()

    if metric in ("auc", "roc_auc"):
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y_true, oof_predictions, **metric_kwargs)

    if metric in ("logloss", "log_loss"):
        from sklearn.metrics import log_loss
        return log_loss(y_true, oof_predictions, **metric_kwargs)

    if metric == "accuracy":
        from sklearn.metrics import accuracy_score
        if oof_predictions.ndim == 1 and not np.all(np.isin(oof_predictions, [0, 1])):
            preds = (oof_predictions >= 0.5).astype(int)
        else:
            preds = oof_predictions
        return accuracy_score(y_true, preds, **metric_kwargs)

    if metric in ("f1", "f1_score"):
        from sklearn.metrics import f1_score
        if oof_predictions.ndim == 1 and not np.all(np.isin(oof_predictions, [0, 1])):
            preds = (oof_predictions >= 0.5).astype(int)
        else:
            preds = oof_predictions
        return f1_score(y_true, preds, **metric_kwargs)

    if metric == "rmse":
        from sklearn.metrics import mean_squared_error
        return np.sqrt(mean_squared_error(y_true, oof_predictions))

    if metric == "mae":
        from sklearn.metrics import mean_absolute_error
        return mean_absolute_error(y_true, oof_predictions)

    if metric == "r2":
        from sklearn.metrics import r2_score
        return r2_score(y_true, oof_predictions)

    raise ValueError(f"Unknown metric: {metric}")


def get_best_threshold(
    oof_predictions: np.ndarray,
    y_true: np.ndarray,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find optimal classification threshold.

    Parameters
    ----------
    oof_predictions : np.ndarray
        Probability predictions.
    y_true : np.ndarray
        True binary labels.
    metric : str, default='f1'
        Metric to optimize: 'f1', 'accuracy', 'balanced_accuracy'.
    thresholds : np.ndarray, optional
        Thresholds to search. Default: np.arange(0.1, 0.9, 0.01).

    Returns
    -------
    best_threshold : float
        Optimal threshold.
    best_score : float
        Score at optimal threshold.
    """
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    if thresholds is None:
        thresholds = np.arange(0.1, 0.9, 0.01)

    metric_funcs = {
        "f1": f1_score,
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
    }

    if metric not in metric_funcs:
        raise ValueError(f"Unknown metric: {metric}. Choose from {list(metric_funcs.keys())}")

    metric_func = metric_funcs[metric]

    best_threshold = 0.5
    best_score = -np.inf

    for threshold in thresholds:
        preds = (oof_predictions >= threshold).astype(int)
        score = metric_func(y_true, preds)

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
