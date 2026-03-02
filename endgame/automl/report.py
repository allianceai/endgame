from __future__ import annotations

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


def _h(text: str) -> str:
    """HTML-escape a string."""
    import html
    return html.escape(str(text))


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

    def to_html(self, title: str = "AutoML Report") -> str:
        """Render the report as a self-contained HTML page.

        Returns a single HTML string with embedded CSS — no external
        dependencies required.  Suitable for saving as a standalone
        ``.html`` file or embedding in a dashboard.

        Parameters
        ----------
        title : str, default="AutoML Report"
            Page title.

        Returns
        -------
        str
            Complete HTML document.
        """
        css = """
        <style>
            * { box-sizing: border-box; margin: 0; padding: 0; }
            body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI',
                   Roboto, sans-serif; background: #f5f7fa; color: #1a1a2e;
                   padding: 2rem; line-height: 1.6; }
            .container { max-width: 960px; margin: 0 auto; }
            h1 { font-size: 1.8rem; margin-bottom: 1.5rem; color: #16213e; }
            h2 { font-size: 1.3rem; margin: 2rem 0 0.8rem; color: #0f3460;
                 border-bottom: 2px solid #e0e0e0; padding-bottom: 0.3rem; }
            .cards { display: grid; grid-template-columns: repeat(auto-fit,
                     minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }
            .card { background: #fff; border-radius: 8px; padding: 1rem;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            .card-label { font-size: 0.75rem; text-transform: uppercase;
                          color: #666; letter-spacing: 0.05em; }
            .card-value { font-size: 1.4rem; font-weight: 600; color: #16213e; }
            table { width: 100%; border-collapse: collapse; margin: 0.5rem 0 1rem;
                    background: #fff; border-radius: 8px; overflow: hidden;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
            th { background: #16213e; color: #fff; padding: 0.6rem 1rem;
                 text-align: left; font-size: 0.85rem; font-weight: 500; }
            td { padding: 0.5rem 1rem; border-bottom: 1px solid #eee;
                 font-size: 0.9rem; }
            tr:last-child td { border-bottom: none; }
            tr:hover td { background: #f0f4ff; }
            .best-row td { background: #e8f5e9; font-weight: 600; }
            .bar-cell { position: relative; }
            .bar-fill { position: absolute; left: 0; top: 0; bottom: 0;
                        background: rgba(15,52,96,0.08); border-radius: 0 4px 4px 0; }
            .warning { background: #fff3e0; border-left: 4px solid #ff9800;
                       padding: 0.5rem 1rem; margin: 0.3rem 0; border-radius: 0 4px 4px 0;
                       font-size: 0.9rem; }
            .warning-high { border-left-color: #f44336; background: #fce4ec; }
            .tuning-item { background: #fff; padding: 0.6rem 1rem; margin: 0.3rem 0;
                           border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                           font-size: 0.9rem; }
            .improved { color: #2e7d32; }
            .no-improve { color: #666; }
            footer { margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #ddd;
                     font-size: 0.8rem; color: #888; text-align: center; }
        </style>
        """

        parts = [
            "<!DOCTYPE html>",
            "<html lang='en'><head><meta charset='utf-8'>",
            "<meta name='viewport' content='width=device-width,initial-scale=1'>",
            f"<title>{_h(title)}</title>",
            css,
            "</head><body><div class='container'>",
            f"<h1>{_h(title)}</h1>",
        ]

        # ── Summary cards ──────────────────────────────────────────
        parts.append("<h2>Summary</h2><div class='cards'>")
        card_items = [
            ("Best Score", f"{self.summary.get('best_score', 0):.4f}"),
            ("Models", str(self.summary.get("n_models", "?"))),
            ("Total Time", f"{self.summary.get('total_time', 0):.1f}s"),
            ("Preset", str(self.summary.get("preset", "?"))),
            ("Task", str(self.summary.get("task_type", "?"))),
        ]
        for label, value in card_items:
            parts.append(
                f"<div class='card'>"
                f"<div class='card-label'>{_h(label)}</div>"
                f"<div class='card-value'>{_h(value)}</div></div>"
            )
        parts.append("</div>")

        # ── Pipeline stages ────────────────────────────────────────
        if not self.stage_summary.empty:
            parts.append("<h2>Pipeline Stages</h2><table>")
            parts.append(
                "<tr><th>Stage</th><th>Status</th><th>Duration</th></tr>"
            )
            for _, row in self.stage_summary.iterrows():
                status = "OK" if row.get("success") else "FAIL"
                color = "#2e7d32" if row.get("success") else "#c62828"
                dur = f"{row.get('duration', 0):.1f}s"
                parts.append(
                    f"<tr><td>{_h(row.get('stage', ''))}</td>"
                    f"<td style='color:{color};font-weight:600'>{status}</td>"
                    f"<td>{dur}</td></tr>"
                )
            parts.append("</table>")

        # ── Model leaderboard ──────────────────────────────────────
        if not self.model_leaderboard.empty:
            parts.append("<h2>Model Leaderboard</h2><table>")
            parts.append(
                "<tr><th>#</th><th>Model</th><th>Score</th><th>Fit Time</th></tr>"
            )
            max_score = self.model_leaderboard["score"].max() if "score" in self.model_leaderboard else 1.0
            for i, row in self.model_leaderboard.iterrows():
                score = row.get("score", 0)
                fit_time = row.get("fit_time", 0)
                bar_pct = (score / max_score * 100) if max_score > 0 else 0
                cls = " class='best-row'" if i == 0 else ""
                parts.append(
                    f"<tr{cls}><td>{i + 1}</td>"
                    f"<td>{_h(row.get('model', ''))}</td>"
                    f"<td class='bar-cell'>"
                    f"<span class='bar-fill' style='width:{bar_pct:.0f}%'></span>"
                    f"{score:.4f}</td>"
                    f"<td>{fit_time:.1f}s</td></tr>"
                )
            parts.append("</table>")

        # ── Feature importances ────────────────────────────────────
        if self.feature_importances is not None and not self.feature_importances.empty:
            top = self.feature_importances.head(15)
            max_imp = top["importance"].max() if "importance" in top else 1.0
            parts.append("<h2>Top Features</h2><table>")
            parts.append("<tr><th>Feature</th><th>Importance</th></tr>")
            for _, row in top.iterrows():
                imp = row.get("importance", 0)
                bar_pct = (imp / max_imp * 100) if max_imp > 0 else 0
                parts.append(
                    f"<tr><td>{_h(str(row.get('feature', '')))}</td>"
                    f"<td class='bar-cell'>"
                    f"<span class='bar-fill' style='width:{bar_pct:.0f}%'></span>"
                    f"{imp:.4f}</td></tr>"
                )
            parts.append("</table>")

        # ── Quality warnings ───────────────────────────────────────
        if self.quality_warnings:
            parts.append("<h2>Quality Warnings</h2>")
            for w in self.quality_warnings:
                sev = getattr(w, "severity", "info").lower()
                cls = "warning-high" if sev in ("high", "critical") else ""
                msg = getattr(w, "message", str(w))
                parts.append(
                    f"<div class='warning {cls}'>"
                    f"<strong>[{_h(sev.upper())}]</strong> {_h(msg)}</div>"
                )

        # ── Tuning summary ─────────────────────────────────────────
        if self.tuning_summary:
            parts.append("<h2>Hyperparameter Tuning</h2>")
            for entry in self.tuning_summary:
                model = entry.get("model", "?")
                orig = entry.get("original_score")
                tuned = entry.get("tuned_score")
                improved = entry.get("improved", False)
                cls = "improved" if improved else "no-improve"
                orig_s = f"{orig:.4f}" if orig is not None else "N/A"
                tuned_s = f"{tuned:.4f}" if tuned is not None else "N/A"
                arrow = "improved" if improved else "no improvement"
                parts.append(
                    f"<div class='tuning-item'>"
                    f"<strong>{_h(model)}</strong>: "
                    f"{orig_s} &rarr; {tuned_s} "
                    f"<span class='{cls}'>({arrow})</span></div>"
                )

        # ── Constraint violations ──────────────────────────────────
        if self.constraint_violations:
            parts.append("<h2>Constraint Violations</h2>")
            for v in self.constraint_violations:
                msg = getattr(v, "message", str(v))
                parts.append(f"<div class='warning warning-high'>{_h(msg)}</div>")

        # ── Footer ─────────────────────────────────────────────────
        parts.append(
            "<footer>Generated by endgame AutoML</footer>"
            "</div></body></html>"
        )

        return "\n".join(parts)

    def save_html(self, path: str, title: str = "AutoML Report") -> None:
        """Save the report as a standalone HTML file.

        Parameters
        ----------
        path : str
            Output file path (e.g. ``"report.html"``).
        title : str, default="AutoML Report"
            Page title.
        """
        html = self.to_html(title=title)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

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
