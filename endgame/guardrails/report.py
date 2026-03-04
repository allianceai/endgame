from __future__ import annotations

"""Data quality warning and report dataclasses."""

import html
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DataQualityWarning:
    """A single data quality issue.

    Attributes
    ----------
    category : str
        Issue category: "leakage", "redundancy", "data_health", "train_test".
    severity : str
        Severity level: "critical", "warning", "info".
    message : str
        Human-readable description.
    details : dict
        Additional details (feature names, values, etc.).
    check_name : str
        Name of the check that produced this warning.
    """

    category: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    check_name: str = ""


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
    features_to_drop : list of str
        Feature names recommended for removal.
    rows_to_drop : list of int
        Row indices recommended for removal.
    """

    warnings: list[DataQualityWarning] = field(default_factory=list)
    passed: bool = True
    n_critical: int = 0
    n_warnings: int = 0
    features_to_drop: list[str] = field(default_factory=list)
    rows_to_drop: list[int] = field(default_factory=list)

    def add(self, warning: DataQualityWarning) -> None:
        """Add a warning and update counts."""
        self.warnings.append(warning)
        if warning.severity == "critical":
            self.n_critical += 1
            self.passed = False
        elif warning.severity == "warning":
            self.n_warnings += 1

    def summary(self) -> str:
        """Return a human-readable summary of the report."""
        lines = []
        lines.append(f"Guardrails Report: {'PASSED' if self.passed else 'ISSUES FOUND'}")
        lines.append(f"  Critical: {self.n_critical}  Warnings: {self.n_warnings}  "
                      f"Info: {len(self.warnings) - self.n_critical - self.n_warnings}")
        if self.features_to_drop:
            lines.append(f"  Features to drop ({len(self.features_to_drop)}): "
                          f"{', '.join(self.features_to_drop[:10])}"
                          + ("..." if len(self.features_to_drop) > 10 else ""))
        if self.rows_to_drop:
            lines.append(f"  Rows to drop: {len(self.rows_to_drop)}")
        for w in self.warnings:
            icon = {"critical": "[!]", "warning": "[W]", "info": "[i]"}.get(w.severity, "[ ]")
            lines.append(f"  {icon} {w.message}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary."""
        return {
            "passed": self.passed,
            "n_critical": self.n_critical,
            "n_warnings": self.n_warnings,
            "features_to_drop": self.features_to_drop,
            "rows_to_drop": self.rows_to_drop,
            "warnings": [
                {
                    "category": w.category,
                    "severity": w.severity,
                    "message": w.message,
                    "check_name": w.check_name,
                    "details": w.details,
                }
                for w in self.warnings
            ],
        }

    def _repr_html_(self) -> str:
        """HTML representation for Jupyter notebooks."""
        status_color = "#28a745" if self.passed else "#dc3545"
        status_text = "PASSED" if self.passed else "ISSUES FOUND"
        rows = []
        for w in self.warnings:
            sev_color = {"critical": "#dc3545", "warning": "#ffc107", "info": "#17a2b8"}.get(
                w.severity, "#6c757d"
            )
            rows.append(
                f"<tr><td><span style='color:{sev_color};font-weight:bold'>"
                f"{html.escape(w.severity.upper())}</span></td>"
                f"<td>{html.escape(w.category)}</td>"
                f"<td>{html.escape(w.check_name)}</td>"
                f"<td>{html.escape(w.message)}</td></tr>"
            )
        table = (
            "<table style='border-collapse:collapse;width:100%'>"
            "<tr><th>Severity</th><th>Category</th><th>Check</th><th>Message</th></tr>"
            + "".join(rows) + "</table>"
        )
        return (
            f"<div style='border:1px solid #ddd;padding:12px;border-radius:8px'>"
            f"<h3 style='color:{status_color}'>Guardrails: {status_text}</h3>"
            f"<p>Critical: {self.n_critical} | Warnings: {self.n_warnings} | "
            f"Features to drop: {len(self.features_to_drop)}</p>"
            f"{table}</div>"
        )

    def __repr__(self) -> str:
        return self.summary()
