"""Performance report generation for AutoML pipelines.

Generates a structured summary of the AutoML run including model
leaderboard, stage timing, quality warnings, tuning results, and
feature importances.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class AutoMLReport:
    """Structured report from an AutoML run.

    Attributes
    ----------
    summary : dict
        Overall statistics (time, n_models, best score, preset).
    stage_summary : pd.DataFrame
        Per-stage timing and success status.
    model_leaderboard : pd.DataFrame
        Trained models sorted by score.
    quality_warnings : list
        Warnings from guardrails stage.
    feature_importances : pd.DataFrame or None
        Feature importance from explainability stage.
    tuning_summary : list
        HPO results per model.
    constraint_violations : list
        Deployment constraint violations.
    """

    summary: dict[str, Any] = field(default_factory=dict)
    stage_summary: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    model_leaderboard: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    quality_warnings: list[Any] = field(default_factory=list)
    feature_importances: pd.DataFrame | None = None
    tuning_summary: list[dict[str, Any]] = field(default_factory=list)
    constraint_violations: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a plain dict."""
        return {
            "summary": self.summary,
            "stage_summary": self.stage_summary.to_dict("records")
            if not self.stage_summary.empty
            else [],
            "model_leaderboard": self.model_leaderboard.to_dict("records")
            if not self.model_leaderboard.empty
            else [],
            "quality_warnings": [
                {"category": w.category, "severity": w.severity, "message": w.message}
                for w in self.quality_warnings
            ],
            "feature_importances": self.feature_importances.to_dict("records")
            if self.feature_importances is not None
            else None,
            "tuning_summary": self.tuning_summary,
            "constraint_violations": [
                {"model": v.model_name, "constraint": v.constraint, "message": v.message}
                for v in self.constraint_violations
            ],
        }

    def to_markdown(self) -> str:
        """Render the report as a markdown string."""
        lines = []
        lines.append("# AutoML Report")
        lines.append("")

        # Summary
        lines.append("## Summary")
        for k, v in self.summary.items():
            if isinstance(v, float):
                lines.append(f"- **{k}**: {v:.4f}")
            else:
                lines.append(f"- **{k}**: {v}")
        lines.append("")

        # Stage summary
        if not self.stage_summary.empty:
            lines.append("## Pipeline Stages")
            lines.append("")
            lines.append(
                "| Stage | Success | Duration (s) |"
            )
            lines.append("| --- | --- | --- |")
            for _, row in self.stage_summary.iterrows():
                success = "Y" if row.get("success") else "N"
                duration = f"{row.get('duration', 0):.1f}"
                lines.append(f"| {row.get('stage', '')} | {success} | {duration} |")
            lines.append("")

        # Leaderboard
        if not self.model_leaderboard.empty:
            lines.append("## Model Leaderboard")
            lines.append("")
            lines.append("| Rank | Model | Score | Fit Time (s) |")
            lines.append("| --- | --- | --- | --- |")
            for i, row in self.model_leaderboard.iterrows():
                score = f"{row.get('score', 0):.4f}"
                fit_time = f"{row.get('fit_time', 0):.1f}"
                lines.append(
                    f"| {i + 1} | {row.get('model', '')} | {score} | {fit_time} |"
                )
            lines.append("")

        # Quality warnings
        if self.quality_warnings:
            lines.append("## Quality Warnings")
            lines.append("")
            for w in self.quality_warnings:
                severity = w.severity.upper()
                lines.append(f"- **[{severity}]** {w.message}")
            lines.append("")

        # Feature importances
        if self.feature_importances is not None and not self.feature_importances.empty:
            lines.append("## Top Features")
            lines.append("")
            top = self.feature_importances.head(10)
            lines.append("| Feature | Importance |")
            lines.append("| --- | --- |")
            for _, row in top.iterrows():
                lines.append(
                    f"| {row.get('feature', '')} | {row.get('importance', 0):.4f} |"
                )
            lines.append("")

        # Tuning summary
        if self.tuning_summary:
            lines.append("## Hyperparameter Tuning")
            lines.append("")
            for entry in self.tuning_summary:
                model = entry.get("model", "?")
                orig = entry.get("original_score")
                tuned = entry.get("tuned_score")
                improved = entry.get("improved", False)
                status = "improved" if improved else "no improvement"
                orig_str = f"{orig:.4f}" if orig is not None else "N/A"
                tuned_str = f"{tuned:.4f}" if tuned is not None else "N/A"
                lines.append(
                    f"- **{model}**: {orig_str} -> {tuned_str} ({status})"
                )
            lines.append("")

        # Constraint violations
        if self.constraint_violations:
            lines.append("## Constraint Violations")
            lines.append("")
            for v in self.constraint_violations:
                lines.append(f"- {v.message}")
            lines.append("")

        return "\n".join(lines)

    def display(self) -> None:
        """Print the report to stdout."""
        print(self.to_markdown())


class ReportGenerator:
    """Generates an AutoMLReport from pipeline results."""

    def generate(
        self,
        pipeline_result: Any,
        orchestrator: Any,
        models: dict[str, Any] | None = None,
    ) -> AutoMLReport:
        """Generate a report from pipeline execution results.

        Parameters
        ----------
        pipeline_result : PipelineResult
            Result from orchestrator.run().
        orchestrator : PipelineOrchestrator
            The orchestrator that executed the pipeline.
        models : dict, optional
            Model info dict from TabularPredictor._models.

        Returns
        -------
        AutoMLReport
            The generated report.
        """
        report = AutoMLReport()

        # Summary
        report.summary = {
            "preset": pipeline_result.metadata.get("preset", "unknown"),
            "task_type": pipeline_result.metadata.get("task_type", "unknown"),
            "time_limit": pipeline_result.metadata.get("time_limit"),
            "total_time": pipeline_result.total_time,
            "best_score": pipeline_result.score,
            "n_stages": len(pipeline_result.stage_results),
        }

        # Stage summary
        stage_rows = []
        for stage_name, result in pipeline_result.stage_results.items():
            stage_rows.append({
                "stage": stage_name,
                "success": result.success,
                "duration": result.duration,
                "error": result.error,
            })
        report.stage_summary = pd.DataFrame(stage_rows)

        # Model leaderboard
        if models:
            lb_rows = []
            for name, info in models.items():
                lb_rows.append({
                    "model": name,
                    "score": info.get("score", 0.0),
                    "fit_time": info.get("fit_time", 0.0),
                })
            report.model_leaderboard = (
                pd.DataFrame(lb_rows)
                .sort_values("score", ascending=False)
                .reset_index(drop=True)
            )
            report.summary["n_models"] = len(lb_rows)

        # Quality warnings from guardrails
        guardrails_result = orchestrator.stage_results_.get("quality_guardrails")
        if guardrails_result and guardrails_result.output:
            guardrails_report = guardrails_result.output.get("guardrails_report")
            if guardrails_report is not None:
                report.quality_warnings = guardrails_report.warnings

        # Feature importances from explainability
        explain_result = orchestrator.stage_results_.get("explainability")
        if explain_result and explain_result.output:
            explanations = explain_result.output.get("explanations", {})
            report.feature_importances = explanations.get("feature_importance_df")

        # Tuning summary from HPO
        hpo_result = orchestrator.stage_results_.get("hyperparameter_tuning")
        if hpo_result and hpo_result.output:
            report.tuning_summary = hpo_result.output.get("tuning_results", [])

        # Constraint violations
        constraint_result = orchestrator.stage_results_.get("constraint_check")
        if constraint_result and constraint_result.output:
            report.constraint_violations = constraint_result.output.get(
                "constraint_violations", []
            )

        return report
