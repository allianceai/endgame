from __future__ import annotations

"""Redundancy detection checks: pairwise correlation, near-duplicates, VIF."""

from typing import Any

import numpy as np

from endgame.guardrails.report import DataQualityWarning, GuardrailsReport


def check_pairwise_correlation(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    redundancy_threshold: float = 0.98,
    **kwargs: Any,
) -> None:
    """Flag highly correlated feature pairs."""
    if X_numeric.shape[1] < 2:
        return

    # Subsample for speed
    n = min(X_numeric.shape[0], 5000)
    X_sub = X_numeric[:n].copy()

    # Remove constant columns
    stds = np.nanstd(X_sub, axis=0)
    nonconst = stds > 0
    if nonconst.sum() < 2:
        return

    X_sub = X_sub[:, nonconst]
    names_sub = [numeric_names[i] for i in range(len(numeric_names)) if i < len(nonconst) and nonconst[i]]

    # Fill NaN with column mean
    col_means = np.nanmean(X_sub, axis=0)
    for j in range(X_sub.shape[1]):
        mask = np.isnan(X_sub[:, j])
        if mask.any():
            X_sub[mask, j] = col_means[j]

    try:
        corr_matrix = np.corrcoef(X_sub.T)
    except Exception:
        return

    redundant_pairs = []
    n_cols = corr_matrix.shape[0]
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if np.abs(corr_matrix[i, j]) > redundancy_threshold:
                redundant_pairs.append((names_sub[i], names_sub[j]))

    if redundant_pairs:
        # Add the second feature of each pair to features_to_drop
        for _, feat_b in redundant_pairs:
            if feat_b not in report.features_to_drop:
                report.features_to_drop.append(feat_b)
        report.add(DataQualityWarning(
            category="redundancy",
            severity="warning",
            message=(
                f"{len(redundant_pairs)} redundant feature pair(s) detected "
                f"(|corr| > {redundancy_threshold})."
            ),
            details={"pairs": redundant_pairs[:10]},
            check_name="pairwise_correlation",
        ))


def check_near_duplicate_columns(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Detect exact and affine duplicate columns (a*x + b = y).

    Uses subsampling and sort-by-mean for efficiency.
    """
    if X_numeric.shape[1] < 2:
        return

    # Subsample rows for speed
    n = min(X_numeric.shape[0], 5000)
    X_sub = X_numeric[:n].copy()

    # Fill NaN with column mean
    col_means = np.nanmean(X_sub, axis=0)
    for j in range(X_sub.shape[1]):
        mask = np.isnan(X_sub[:, j])
        if mask.any():
            X_sub[mask, j] = col_means[j]

    # Sort columns by mean for O(n) comparison of nearby columns
    means = np.mean(X_sub, axis=0)
    stds = np.std(X_sub, axis=0)
    order = np.argsort(means)

    dup_pairs: list[tuple[str, str]] = []
    tol = 1e-8

    for idx in range(len(order) - 1):
        i = order[idx]
        # Check next few columns (sorted by mean)
        for offset in range(1, min(4, len(order) - idx)):
            j = order[idx + offset]
            col_i = X_sub[:, i]
            col_j = X_sub[:, j]

            # Exact duplicate check (L-inf)
            if np.max(np.abs(col_i - col_j)) < tol:
                dup_pairs.append((numeric_names[i], numeric_names[j]))
                continue

            # Affine duplicate: normalize both and check
            if stds[i] > tol and stds[j] > tol:
                norm_i = (col_i - means[i]) / stds[i]
                norm_j = (col_j - means[j]) / stds[j]
                if np.max(np.abs(norm_i - norm_j)) < tol:
                    dup_pairs.append((numeric_names[i], numeric_names[j]))
                elif np.max(np.abs(norm_i + norm_j)) < tol:
                    # Negative correlation duplicate
                    dup_pairs.append((numeric_names[i], numeric_names[j]))

    if dup_pairs:
        for _, feat_b in dup_pairs:
            if feat_b not in report.features_to_drop:
                report.features_to_drop.append(feat_b)
        report.add(DataQualityWarning(
            category="redundancy",
            severity="warning",
            message=f"{len(dup_pairs)} near-duplicate column pair(s) detected.",
            details={"pairs": dup_pairs[:10]},
            check_name="near_duplicate_columns",
        ))


def check_vif(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    vif_threshold: float = 10.0,
    **kwargs: Any,
) -> None:
    """Variance Inflation Factor for multicollinearity detection.

    O(p^2 * n) — off by default, guarded to max 100 features.
    """
    max_features = 100
    if X_numeric.shape[1] < 2 or X_numeric.shape[1] > max_features:
        return

    # Remove NaN rows
    valid = ~np.isnan(X_numeric).any(axis=1)
    X_valid = X_numeric[valid]
    if X_valid.shape[0] < X_valid.shape[1] + 1:
        return

    # Remove constant columns
    stds = np.std(X_valid, axis=0)
    nonconst = stds > 0
    if nonconst.sum() < 2:
        return

    X_valid = X_valid[:, nonconst]
    names = [numeric_names[i] for i in range(len(numeric_names)) if i < len(nonconst) and nonconst[i]]

    # Compute VIF via OLS: VIF_j = 1 / (1 - R^2_j)
    high_vif: list[tuple[str, float]] = []
    for j in range(X_valid.shape[1]):
        y_j = X_valid[:, j]
        X_other = np.delete(X_valid, j, axis=1)
        # Add intercept
        X_other = np.column_stack([np.ones(X_other.shape[0]), X_other])
        try:
            beta, residuals, _, _ = np.linalg.lstsq(X_other, y_j, rcond=None)
            y_pred = X_other @ beta
            ss_res = np.sum((y_j - y_pred) ** 2)
            ss_tot = np.sum((y_j - np.mean(y_j)) ** 2)
            if ss_tot == 0:
                continue
            r_squared = 1 - ss_res / ss_tot
            if r_squared >= 1.0:
                vif = float("inf")
            else:
                vif = 1.0 / (1.0 - r_squared)
            if vif > vif_threshold:
                high_vif.append((names[j], vif))
        except Exception:
            continue

    if high_vif:
        report.add(DataQualityWarning(
            category="redundancy",
            severity="warning",
            message=(
                f"{len(high_vif)} feature(s) with VIF > {vif_threshold} "
                f"(multicollinearity)."
            ),
            details={
                "features": [(name, round(v, 1)) for name, v in high_vif[:10]],
            },
            check_name="vif",
        ))
