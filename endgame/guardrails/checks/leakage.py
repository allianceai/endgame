from __future__ import annotations

"""Leakage detection checks: correlation, mutual information, categorical."""

from typing import Any

import numpy as np

from endgame.guardrails.report import DataQualityWarning, GuardrailsReport


def check_correlation_leakage(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    y_arr: np.ndarray | None,
    report: GuardrailsReport,
    leakage_threshold: float = 0.95,
    **kwargs: Any,
) -> None:
    """Flag numeric features with |Pearson correlation| > threshold with target."""
    if y_arr is None or X_numeric.shape[1] == 0:
        return

    try:
        y_numeric = y_arr.astype(float)
    except (ValueError, TypeError):
        return

    valid_mask = ~(np.isnan(X_numeric).any(axis=1) | np.isnan(y_numeric))
    if valid_mask.sum() < 10:
        return

    X_valid = X_numeric[valid_mask]
    y_valid = y_numeric[valid_mask]

    for i in range(X_valid.shape[1]):
        col = X_valid[:, i]
        if np.std(col) == 0:
            continue
        corr = np.abs(np.corrcoef(col, y_valid)[0, 1])
        if np.isnan(corr):
            continue
        if corr > leakage_threshold:
            name = numeric_names[i]
            report.add(DataQualityWarning(
                category="leakage",
                severity="critical",
                message=(
                    f"Potential target leakage: '{name}' has "
                    f"|corr| = {corr:.3f} with target."
                ),
                details={"feature": name, "correlation": float(corr)},
                check_name="correlation_leakage",
            ))
            if name not in report.features_to_drop:
                report.features_to_drop.append(name)


def check_mutual_info_leakage(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    y_arr: np.ndarray | None,
    report: GuardrailsReport,
    mi_threshold: float = 0.95,
    **kwargs: Any,
) -> None:
    """Flag features with high normalized mutual information with target.

    Catches non-linear leakage that Pearson correlation misses.
    """
    if y_arr is None or X_numeric.shape[1] == 0:
        return

    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

    # Remove rows with any NaN
    valid_mask = ~np.isnan(X_numeric).any(axis=1)
    if y_arr.dtype.kind == 'f':
        valid_mask &= ~np.isnan(y_arr)
    if valid_mask.sum() < 10:
        return

    X_valid = X_numeric[valid_mask]
    y_valid = y_arr[valid_mask]

    # Subsample for speed
    n = min(X_valid.shape[0], 5000)
    if n < X_valid.shape[0]:
        rng = np.random.RandomState(kwargs.get("random_state", 42))
        idx = rng.choice(X_valid.shape[0], n, replace=False)
        X_valid = X_valid[idx]
        y_valid = y_valid[idx]

    # Determine if classification or regression
    unique_values = np.unique(y_valid[~np.isnan(y_valid)] if y_valid.dtype.kind == 'f' else y_valid)
    is_classification = len(unique_values) <= 50 and np.all(unique_values == unique_values.astype(int))

    try:
        if is_classification:
            mi = mutual_info_classif(X_valid, y_valid.astype(int), random_state=42)
        else:
            mi = mutual_info_regression(X_valid, y_valid, random_state=42)
    except Exception:
        return

    # Normalize MI to [0, 1] by dividing by max
    mi_max = mi.max()
    if mi_max > 0:
        mi_normalized = mi / mi_max
    else:
        return

    for i in range(len(mi_normalized)):
        if mi_normalized[i] > mi_threshold:
            name = numeric_names[i]
            # Skip if already flagged by correlation check
            if name in report.features_to_drop:
                continue
            report.add(DataQualityWarning(
                category="leakage",
                severity="critical",
                message=(
                    f"Potential non-linear leakage: '{name}' has "
                    f"normalized MI = {mi_normalized[i]:.3f} with target."
                ),
                details={
                    "feature": name,
                    "mutual_info_normalized": float(mi_normalized[i]),
                    "mutual_info_raw": float(mi[i]),
                },
                check_name="mutual_info_leakage",
            ))
            report.features_to_drop.append(name)


def check_categorical_leakage(
    X_cat: np.ndarray | None,
    cat_names: list[str],
    y_arr: np.ndarray | None,
    report: GuardrailsReport,
    leakage_threshold: float = 0.95,
    **kwargs: Any,
) -> None:
    """Flag categorical features with high Cramer's V with target.

    For regression targets, bins into 10 quantiles first.
    """
    if X_cat is None or y_arr is None or X_cat.shape[1] == 0:
        return

    from scipy.stats import chi2_contingency

    # For regression targets, bin into quantiles
    try:
        y_float = y_arr.astype(float)
        unique_y = np.unique(y_float[~np.isnan(y_float)])
        if len(unique_y) > 20:
            # Regression — bin target
            y_cat = np.digitize(
                y_float,
                np.nanquantile(y_float, np.linspace(0, 1, 11)[1:-1]),
            )
        else:
            y_cat = y_arr
    except (ValueError, TypeError):
        y_cat = y_arr

    for j in range(X_cat.shape[1]):
        col = X_cat[:, j]
        # Build contingency table
        try:
            # Handle NaN
            mask = ~(col.astype(str) == 'nan')
            if mask.sum() < 10:
                continue
            col_valid = col[mask]
            y_valid = np.asarray(y_cat)[mask] if hasattr(y_cat, '__len__') else y_cat[mask]

            # Create contingency table
            unique_x = np.unique(col_valid)
            unique_y_vals = np.unique(y_valid)
            if len(unique_x) < 2 or len(unique_y_vals) < 2:
                continue

            import pandas as pd
            ct = pd.crosstab(pd.Series(col_valid), pd.Series(y_valid))
            chi2, p_val, _, _ = chi2_contingency(ct.values)

            # Cramer's V
            n = ct.values.sum()
            min_dim = min(ct.shape[0] - 1, ct.shape[1] - 1)
            if min_dim == 0 or n == 0:
                continue
            cramers_v = np.sqrt(chi2 / (n * min_dim))

            if cramers_v > leakage_threshold:
                name = cat_names[j]
                report.add(DataQualityWarning(
                    category="leakage",
                    severity="critical",
                    message=(
                        f"Potential categorical leakage: '{name}' has "
                        f"Cramer's V = {cramers_v:.3f} with target."
                    ),
                    details={
                        "feature": name,
                        "cramers_v": float(cramers_v),
                        "chi2": float(chi2),
                        "p_value": float(p_val),
                    },
                    check_name="categorical_leakage",
                ))
                if name not in report.features_to_drop:
                    report.features_to_drop.append(name)

        except Exception:
            continue
