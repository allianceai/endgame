from __future__ import annotations

"""Quality guardrails for AutoML pipelines.

This module provides data quality checks that run early in the pipeline
to detect issues like target leakage, redundant features, and data health
problems before expensive model training begins.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


@dataclass
class DataQualityWarning:
    """A single data quality issue.

    Attributes
    ----------
    category : str
        Issue category: "leakage", "redundancy", "data_health".
    severity : str
        Severity level: "critical", "warning", "info".
    message : str
        Human-readable description.
    details : dict
        Additional details (feature names, values, etc.).
    """

    category: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class GuardrailsReport:
    """Aggregated result from all guardrail checks.

    Attributes
    ----------
    warnings : list of DataQualityWarning
        All detected issues.
    passed : bool
        True if no critical issues found.
    n_critical : int
        Number of critical issues.
    n_warnings : int
        Number of warning-level issues.
    """

    warnings: list[DataQualityWarning] = field(default_factory=list)
    passed: bool = True
    n_critical: int = 0
    n_warnings: int = 0

    def add(self, warning: DataQualityWarning) -> None:
        """Add a warning and update counts."""
        self.warnings.append(warning)
        if warning.severity == "critical":
            self.n_critical += 1
            self.passed = False
        elif warning.severity == "warning":
            self.n_warnings += 1


class QualityGuardrailsExecutor(BaseStageExecutor):
    """Performs data quality checks early in the pipeline.

    Checks for target leakage, feature redundancy, and general data
    health issues. By default issues are logged as warnings; set
    ``strict=True`` to abort on critical problems.

    Parameters
    ----------
    strict : bool, default=False
        If True, sets ``fail_fast=True`` in context metadata on critical
        issues, causing the orchestrator to abort early.
    leakage_threshold : float, default=0.95
        Absolute correlation with target above which a feature is
        flagged as potential leakage.
    redundancy_threshold : float, default=0.98
        Absolute pairwise correlation above which a feature pair is
        flagged as redundant.
    """

    def __init__(
        self,
        strict: bool = False,
        leakage_threshold: float = 0.95,
        redundancy_threshold: float = 0.98,
    ):
        self.strict = strict
        self.leakage_threshold = leakage_threshold
        self.redundancy_threshold = redundancy_threshold

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Run all guardrail checks.

        Parameters
        ----------
        context : dict
            Pipeline context containing ``X``, ``y``, and ``task_type``.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains ``guardrails_report`` in output.
        """
        start = time.time()
        report = GuardrailsReport()

        X = context.get("X")
        y = context.get("y")

        if X is None or y is None:
            return StageResult(
                stage_name="quality_guardrails",
                success=True,
                duration=time.time() - start,
                output={"guardrails_report": report},
            )

        # Convert to numpy for uniform handling
        if isinstance(X, pd.DataFrame):
            feature_names = X.columns.tolist()
            X_arr = X.select_dtypes(include=[np.number]).values
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        else:
            X_arr = np.asarray(X, dtype=float)
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            numeric_cols = feature_names

        y_arr = np.asarray(y).ravel()
        n_samples, n_features = X_arr.shape

        # --- Data health checks ---
        self._check_data_health(
            X, X_arr, y_arr, n_samples, n_features, feature_names, report
        )

        # --- Target leakage ---
        if time.time() - start < time_budget * 0.5:
            self._check_leakage(X_arr, y_arr, numeric_cols, report)

        # --- Feature redundancy ---
        if time.time() - start < time_budget * 0.8 and n_features <= 500:
            self._check_redundancy(X_arr, numeric_cols, report)

        # Log results
        for w in report.warnings:
            if w.severity == "critical":
                logger.warning(f"[Guardrails CRITICAL] {w.message}")
            elif w.severity == "warning":
                logger.warning(f"[Guardrails] {w.message}")
            else:
                logger.debug(f"[Guardrails] {w.message}")

        # Set fail_fast if strict and critical issues found
        output: dict[str, Any] = {"guardrails_report": report}
        if self.strict and not report.passed:
            output["fail_fast"] = True

        duration = time.time() - start
        return StageResult(
            stage_name="quality_guardrails",
            success=True,
            duration=duration,
            output=output,
            metadata={
                "n_critical": report.n_critical,
                "n_warnings": report.n_warnings,
            },
        )

    def _check_data_health(
        self,
        X_raw: Any,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        n_samples: int,
        n_features: int,
        feature_names: list[str],
        report: GuardrailsReport,
    ) -> None:
        """Check general data health."""
        # Too few samples
        if n_samples < 20:
            report.add(DataQualityWarning(
                category="data_health",
                severity="critical",
                message=f"Very few samples ({n_samples}). Results will be unreliable.",
                details={"n_samples": n_samples},
            ))

        # Feature-to-sample ratio
        if n_features > 0 and n_samples > 0 and n_features / n_samples > 10:
            report.add(DataQualityWarning(
                category="data_health",
                severity="warning",
                message=(
                    f"High feature-to-sample ratio ({n_features}/{n_samples} = "
                    f"{n_features / n_samples:.1f}). Consider feature selection."
                ),
                details={"ratio": n_features / n_samples},
            ))

        # Constant columns
        if X_arr.shape[1] > 0:
            stds = np.nanstd(X_arr, axis=0)
            constant_mask = stds == 0
            constant_cols = [
                feature_names[i] if i < len(feature_names) else f"col_{i}"
                for i in np.where(constant_mask)[0]
            ]
            if constant_cols:
                report.add(DataQualityWarning(
                    category="data_health",
                    severity="warning",
                    message=f"{len(constant_cols)} constant column(s) detected.",
                    details={"columns": constant_cols[:10]},
                ))

        # All-missing columns
        if X_arr.shape[1] > 0:
            all_missing = np.all(np.isnan(X_arr), axis=0)
            missing_cols = [
                feature_names[i] if i < len(feature_names) else f"col_{i}"
                for i in np.where(all_missing)[0]
            ]
            if missing_cols:
                report.add(DataQualityWarning(
                    category="data_health",
                    severity="critical",
                    message=f"{len(missing_cols)} all-missing column(s) detected.",
                    details={"columns": missing_cols[:10]},
                ))

        # Minority class check (classification)
        unique, counts = np.unique(y_arr[~pd.isna(y_arr)], return_counts=True)
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
                ))

        # ID-like columns (unique count == n_samples for integer/object cols)
        if isinstance(X_raw, pd.DataFrame):
            for col in X_raw.columns:
                if X_raw[col].nunique() == n_samples and n_samples > 20:
                    report.add(DataQualityWarning(
                        category="data_health",
                        severity="info",
                        message=f"Column '{col}' has all unique values (possible ID column).",
                        details={"column": col},
                    ))

    def _check_leakage(
        self,
        X_arr: np.ndarray,
        y_arr: np.ndarray,
        feature_names: list[str],
        report: GuardrailsReport,
    ) -> None:
        """Flag features highly correlated with target."""
        if X_arr.shape[1] == 0:
            return

        # Only check numeric target
        try:
            y_numeric = y_arr.astype(float)
        except (ValueError, TypeError):
            return

        # Compute correlations with target
        valid_mask = ~(np.isnan(X_arr).any(axis=1) | np.isnan(y_numeric))
        if valid_mask.sum() < 10:
            return

        X_valid = X_arr[valid_mask]
        y_valid = y_numeric[valid_mask]

        for i in range(X_valid.shape[1]):
            col = X_valid[:, i]
            if np.std(col) == 0:
                continue
            corr = np.abs(np.corrcoef(col, y_valid)[0, 1])
            if np.isnan(corr):
                continue
            if corr > self.leakage_threshold:
                name = feature_names[i] if i < len(feature_names) else f"feature_{i}"
                report.add(DataQualityWarning(
                    category="leakage",
                    severity="critical",
                    message=(
                        f"Potential target leakage: '{name}' has "
                        f"|corr| = {corr:.3f} with target."
                    ),
                    details={"feature": name, "correlation": float(corr)},
                ))

    def _check_redundancy(
        self,
        X_arr: np.ndarray,
        feature_names: list[str],
        report: GuardrailsReport,
    ) -> None:
        """Flag highly correlated feature pairs."""
        if X_arr.shape[1] < 2:
            return

        # Subsample for speed
        n = min(X_arr.shape[0], 5000)
        X_sub = X_arr[:n]

        # Remove constant columns
        stds = np.nanstd(X_sub, axis=0)
        nonconst = stds > 0
        if nonconst.sum() < 2:
            return

        X_sub = X_sub[:, nonconst]
        names_sub = [
            feature_names[i] for i in range(len(feature_names)) if nonconst[i]
        ] if len(feature_names) == len(nonconst) else [
            f"feature_{i}" for i in range(X_sub.shape[1])
        ]

        # Fill NaN with column mean for correlation
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
                if np.abs(corr_matrix[i, j]) > self.redundancy_threshold:
                    redundant_pairs.append((names_sub[i], names_sub[j]))

        if redundant_pairs:
            report.add(DataQualityWarning(
                category="redundancy",
                severity="warning",
                message=(
                    f"{len(redundant_pairs)} redundant feature pair(s) detected "
                    f"(|corr| > {self.redundancy_threshold})."
                ),
                details={"pairs": redundant_pairs[:10]},
            ))
