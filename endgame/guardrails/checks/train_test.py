from __future__ import annotations

"""Train/test checks: duplicate rows, adversarial drift, temporal leakage."""

from typing import Any

import numpy as np

from endgame.guardrails.report import DataQualityWarning, GuardrailsReport


def check_duplicate_rows(
    X_numeric: np.ndarray,
    y_arr: np.ndarray | None,
    report: GuardrailsReport,
    X_test_numeric: np.ndarray | None = None,
    **kwargs: Any,
) -> None:
    """Hash-based row deduplication within train and across train/test.

    Flags label conflicts in duplicate rows as critical.
    """
    if X_numeric.shape[0] == 0:
        return

    # Hash rows for fast comparison
    def _hash_rows(arr: np.ndarray) -> np.ndarray:
        # Fill NaN with a sentinel for consistent hashing
        filled = np.nan_to_num(arr, nan=-999999.0)
        return np.array([hash(row.tobytes()) for row in filled])

    train_hashes = _hash_rows(X_numeric)

    # Check within-train duplicates
    unique_hashes, counts = np.unique(train_hashes, return_counts=True)
    n_dup_groups = (counts > 1).sum()
    n_dup_rows = counts[counts > 1].sum() - n_dup_groups  # extra rows beyond first

    if n_dup_rows > 0:
        # Check for label conflicts in duplicates
        label_conflicts = 0
        if y_arr is not None:
            hash_to_labels: dict[int, set] = {}
            for i, h in enumerate(train_hashes):
                h_int = int(h)
                if h_int not in hash_to_labels:
                    hash_to_labels[h_int] = set()
                hash_to_labels[h_int].add(str(y_arr[i]))
            label_conflicts = sum(1 for labels in hash_to_labels.values() if len(labels) > 1)

        severity = "critical" if label_conflicts > 0 else "info"
        msg = f"{n_dup_rows} duplicate row(s) in training data ({n_dup_groups} groups)."
        if label_conflicts > 0:
            msg += f" {label_conflicts} group(s) have conflicting labels."

        report.add(DataQualityWarning(
            category="train_test",
            severity=severity,
            message=msg,
            details={
                "n_duplicate_rows": int(n_dup_rows),
                "n_groups": int(n_dup_groups),
                "label_conflicts": label_conflicts,
            },
            check_name="duplicate_rows",
        ))

    # Check train/test overlap
    if X_test_numeric is not None and X_test_numeric.shape[0] > 0:
        test_hashes = _hash_rows(X_test_numeric)
        overlap = np.isin(test_hashes, train_hashes)
        n_overlap = overlap.sum()
        if n_overlap > 0:
            report.add(DataQualityWarning(
                category="train_test",
                severity="warning",
                message=(
                    f"{n_overlap} test row(s) are duplicates of training rows."
                ),
                details={"n_overlap": int(n_overlap)},
                check_name="duplicate_rows",
            ))


def check_adversarial_drift(
    X_numeric: np.ndarray,
    X_test_numeric: np.ndarray | None,
    report: GuardrailsReport,
    adversarial_threshold: float = 0.7,
    **kwargs: Any,
) -> None:
    """Wrap AdversarialValidator to detect train/test distribution shift."""
    if X_test_numeric is None or X_test_numeric.shape[0] == 0:
        return

    try:
        from endgame.validation.adversarial import AdversarialValidator

        av = AdversarialValidator()
        result = av.check_drift(X_numeric, X_test_numeric)

        auc = result.get("auc", 0.5) if isinstance(result, dict) else getattr(result, "auc", 0.5)

        if auc > adversarial_threshold:
            top_features = (
                result.get("top_features", [])
                if isinstance(result, dict)
                else getattr(result, "top_features", [])
            )
            report.add(DataQualityWarning(
                category="train_test",
                severity="warning",
                message=(
                    f"Significant train/test drift detected (AUC = {auc:.3f}). "
                    f"Top drifting features: {top_features[:5]}"
                ),
                details={"auc": float(auc), "top_features": top_features[:10]},
                check_name="adversarial_drift",
            ))
    except Exception:
        pass


def check_temporal_leakage(
    X_original: Any,
    report: GuardrailsReport,
    future_features: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """Flag user-specified features known to contain future information."""
    if future_features is None or not future_features:
        return

    import pandas as pd

    if isinstance(X_original, pd.DataFrame):
        col_names = X_original.columns.tolist()
    else:
        col_names = kwargs.get("numeric_names", []) + kwargs.get("cat_names", [])

    found = [f for f in future_features if f in col_names]
    if found:
        report.add(DataQualityWarning(
            category="train_test",
            severity="critical",
            message=f"Temporal leakage: {len(found)} future feature(s) present: {found}",
            details={"features": found},
            check_name="temporal_leakage",
        ))
        report.features_to_drop.extend(f for f in found if f not in report.features_to_drop)
