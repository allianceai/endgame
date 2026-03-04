from __future__ import annotations

"""Data health checks: constant columns, missing, class imbalance, etc."""

import re
from typing import Any

import numpy as np
import pandas as pd

from endgame.guardrails.report import DataQualityWarning, GuardrailsReport


def check_constant_columns(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Flag columns with zero standard deviation."""
    if X_numeric.shape[1] == 0:
        return
    stds = np.nanstd(X_numeric, axis=0)
    constant_mask = stds == 0
    constant_cols = [numeric_names[i] for i in np.where(constant_mask)[0]]
    if constant_cols:
        report.add(DataQualityWarning(
            category="data_health",
            severity="warning",
            message=f"{len(constant_cols)} constant column(s) detected.",
            details={"columns": constant_cols[:10]},
            check_name="constant_columns",
        ))
        report.features_to_drop.extend(constant_cols)


def check_missing_columns(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Flag columns that are entirely NaN."""
    if X_numeric.shape[1] == 0:
        return
    all_missing = np.all(np.isnan(X_numeric), axis=0)
    missing_cols = [numeric_names[i] for i in np.where(all_missing)[0]]
    if missing_cols:
        report.add(DataQualityWarning(
            category="data_health",
            severity="critical",
            message=f"{len(missing_cols)} all-missing column(s) detected.",
            details={"columns": missing_cols[:10]},
            check_name="missing_columns",
        ))
        report.features_to_drop.extend(missing_cols)


def check_sample_count(
    X_numeric: np.ndarray,
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Warn if fewer than 20 samples."""
    n_samples = X_numeric.shape[0]
    if n_samples < 20:
        report.add(DataQualityWarning(
            category="data_health",
            severity="critical",
            message=f"Very few samples ({n_samples}). Results will be unreliable.",
            details={"n_samples": n_samples},
            check_name="sample_count",
        ))


def check_feature_sample_ratio(
    X_numeric: np.ndarray,
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Warn if feature-to-sample ratio exceeds 10."""
    n_samples, n_features = X_numeric.shape
    if n_features > 0 and n_samples > 0 and n_features / n_samples > 10:
        ratio = n_features / n_samples
        report.add(DataQualityWarning(
            category="data_health",
            severity="warning",
            message=(
                f"High feature-to-sample ratio ({n_features}/{n_samples} = "
                f"{ratio:.1f}). Consider feature selection."
            ),
            details={"ratio": ratio},
            check_name="feature_sample_ratio",
        ))


def check_id_columns(
    X_original: Any,
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Flag columns where every value is unique (possible ID columns)."""
    if not isinstance(X_original, pd.DataFrame):
        return
    n_samples = len(X_original)
    if n_samples <= 20:
        return
    for col in X_original.columns:
        if X_original[col].nunique() == n_samples:
            dtype = X_original[col].dtype
            is_likely_id = dtype.kind in ("i", "u", "O", "U") or dtype.name == "category"
            report.add(DataQualityWarning(
                category="data_health",
                severity="info",
                message=f"Column '{col}' has all unique values (possible ID column).",
                details={"column": col},
                check_name="id_columns",
            ))
            # Only auto-drop integer/string columns, not float features
            if is_likely_id:
                report.features_to_drop.append(col)


def check_class_imbalance(
    y_arr: np.ndarray | None,
    report: GuardrailsReport,
    **kwargs: Any,
) -> None:
    """Warn on extreme class imbalance (minority < 1%)."""
    if y_arr is None:
        return
    valid = y_arr[~pd.isna(y_arr)]
    unique, counts = np.unique(valid, return_counts=True)
    if len(unique) > 1 and len(unique) <= 100:
        min_frac = counts.min() / counts.sum()
        if min_frac < 0.01:
            report.add(DataQualityWarning(
                category="data_health",
                severity="warning",
                message=(
                    f"Extreme class imbalance: minority class has "
                    f"{min_frac:.2%} of samples ({counts.min()} samples)."
                ),
                details={"min_class_fraction": float(min_frac)},
                check_name="class_imbalance",
            ))


def check_variance(
    X_numeric: np.ndarray,
    numeric_names: list[str],
    report: GuardrailsReport,
    variance_low_threshold: float = 1e-10,
    **kwargs: Any,
) -> None:
    """Flag near-zero variance columns and high-cardinality numeric ID-like columns."""
    if X_numeric.shape[1] == 0:
        return
    n_samples = X_numeric.shape[0]
    variances = np.nanvar(X_numeric, axis=0)

    # Near-zero variance (but not exactly zero — those are caught by constant_columns)
    near_zero = (variances > 0) & (variances < variance_low_threshold)
    near_zero_cols = [numeric_names[i] for i in np.where(near_zero)[0]]
    if near_zero_cols:
        report.add(DataQualityWarning(
            category="data_health",
            severity="warning",
            message=f"{len(near_zero_cols)} near-zero variance column(s) detected.",
            details={"columns": near_zero_cols[:10]},
            check_name="variance",
        ))
        report.features_to_drop.extend(near_zero_cols)

    # High-cardinality numeric ID-like: integers with nearly all unique values
    if n_samples > 20:
        for i in range(X_numeric.shape[1]):
            col = X_numeric[:, i]
            valid = col[~np.isnan(col)]
            if len(valid) == 0:
                continue
            # Check if integer-valued
            if not np.allclose(valid, np.round(valid)):
                continue
            n_unique = len(np.unique(valid))
            if n_unique > 0.95 * len(valid) and len(valid) > 20:
                name = numeric_names[i]
                if name not in report.features_to_drop:
                    report.add(DataQualityWarning(
                        category="data_health",
                        severity="info",
                        message=(
                            f"Column '{name}' is integer-valued with {n_unique}/{len(valid)} "
                            f"unique values (possible numeric ID)."
                        ),
                        details={"column": name, "n_unique": n_unique, "n_samples": len(valid)},
                        check_name="variance",
                    ))


def check_suspect_features(
    X_original: Any,
    report: GuardrailsReport,
    suspect_patterns: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Flag features matching user-supplied regex patterns."""
    if suspect_patterns is None or not suspect_patterns:
        return

    # Get column names
    if isinstance(X_original, pd.DataFrame):
        col_names = X_original.columns.tolist()
    else:
        col_names = kwargs.get("numeric_names", []) + kwargs.get("cat_names", [])

    matched: list[str] = []
    for pattern in suspect_patterns:
        # Convert glob-style wildcards to regex
        regex = pattern.replace("*", ".*").replace("?", ".")
        compiled = re.compile(regex, re.IGNORECASE)
        for col in col_names:
            if compiled.fullmatch(col) and col not in matched:
                matched.append(col)

    if matched:
        report.add(DataQualityWarning(
            category="data_health",
            severity="warning",
            message=f"{len(matched)} suspect feature(s) matched patterns: {matched[:10]}",
            details={"columns": matched, "patterns": suspect_patterns},
            check_name="suspect_features",
        ))
        report.features_to_drop.extend(matched)
