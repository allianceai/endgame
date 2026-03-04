from __future__ import annotations

"""Quality guardrails for AutoML pipelines.

This module provides data quality checks that run early in the pipeline
to detect issues like target leakage, redundant features, and data health
problems before expensive model training begins.

The actual check logic is now in ``endgame.guardrails``. This module
delegates to ``LeakageDetector`` while preserving the
``QualityGuardrailsExecutor`` interface for the AutoML orchestrator.
"""

import logging
import time
from typing import Any

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

# Re-export for backward compatibility
from endgame.guardrails.report import DataQualityWarning, GuardrailsReport  # noqa: F401

logger = logging.getLogger(__name__)


class QualityGuardrailsExecutor(BaseStageExecutor):
    """Performs data quality checks early in the pipeline.

    Delegates to :class:`endgame.guardrails.LeakageDetector` for all
    check logic. Results are wrapped in a :class:`StageResult` for the
    AutoML orchestrator.

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
        from endgame.guardrails import LeakageDetector

        start = time.time()

        X = context.get("X")
        y = context.get("y")

        if X is None or y is None:
            return StageResult(
                stage_name="quality_guardrails",
                success=True,
                duration=time.time() - start,
                output={"guardrails_report": GuardrailsReport()},
            )

        detector = LeakageDetector(
            mode="detect",
            checks="default",
            leakage_threshold=self.leakage_threshold,
            redundancy_threshold=self.redundancy_threshold,
            time_budget=time_budget,
            verbose=True,
        )
        detector.fit(X, y)
        report = detector.report_

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
