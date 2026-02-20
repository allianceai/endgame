"""Fairness report generation: HTML summaries of model fairness.

Generates standalone HTML reports with group-level fairness metrics,
threshold analysis, and summary tables using basic string templates
(no dependency on the visualization module or any plotting libraries).

Example
-------
>>> from endgame.fairness import FairnessReport
>>> report = FairnessReport(model, X_test, y_test, sensitive_attr)
>>> html = report.generate()
>>> report.save("fairness_report.html")
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Union

import numpy as np

from endgame.fairness.metrics import (
    calibration_by_group,
    demographic_parity,
    disparate_impact,
    equalized_odds,
)

if TYPE_CHECKING:
    import pandas as pd

ArrayLike = Union[np.ndarray, list, "pd.Series"]


# =============================================================================
# HTML Template
# =============================================================================

_CSS = """\
<style>
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    margin: 0; padding: 20px 40px;
    background: #fafafa; color: #222;
  }
  h1 { color: #1a1a2e; border-bottom: 2px solid #e94560; padding-bottom: 8px; }
  h2 { color: #1a1a2e; margin-top: 32px; }
  .summary-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 16px; margin: 16px 0;
  }
  .card {
    background: #fff; border-radius: 8px; padding: 16px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.12);
  }
  .card .label { font-size: 13px; color: #666; text-transform: uppercase; letter-spacing: 0.5px; }
  .card .value { font-size: 28px; font-weight: 700; margin-top: 4px; }
  .pass { color: #27ae60; }
  .warn { color: #f39c12; }
  .fail { color: #e74c3c; }
  table {
    border-collapse: collapse; width: 100%; margin: 12px 0;
    background: #fff; border-radius: 8px; overflow: hidden;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
  }
  th, td { padding: 10px 14px; text-align: left; border-bottom: 1px solid #eee; }
  th { background: #1a1a2e; color: #fff; font-weight: 600; font-size: 13px; text-transform: uppercase; }
  tr:last-child td { border-bottom: none; }
  tr:hover { background: #f5f5f5; }
  .bar-container { width: 100%; background: #eee; border-radius: 4px; height: 18px; position: relative; }
  .bar-fill {
    height: 18px; border-radius: 4px; position: absolute; top: 0; left: 0;
    transition: width 0.3s ease;
  }
  .bar-text {
    position: absolute; top: 0; left: 6px; line-height: 18px; font-size: 11px;
    font-weight: 600; color: #fff; text-shadow: 0 1px 1px rgba(0,0,0,0.3);
  }
  .footer { margin-top: 40px; padding-top: 12px; border-top: 1px solid #ddd; font-size: 12px; color: #999; }
</style>
"""


def _bar_html(value: float, max_val: float = 1.0, color: str = "#3498db") -> str:
    """Generate an inline bar chart cell.

    Parameters
    ----------
    value : float
        The value to display.
    max_val : float
        The maximum value for scaling.
    color : str
        CSS color for the bar.

    Returns
    -------
    str
        HTML string for the bar.
    """
    pct = min(value / max_val, 1.0) * 100 if max_val > 0 else 0
    return (
        f'<div class="bar-container">'
        f'<div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div>'
        f'<div class="bar-text">{value:.4f}</div>'
        f'</div>'
    )


def _status_badge(passed: bool) -> str:
    """Return a pass/fail badge.

    Parameters
    ----------
    passed : bool
        Whether the check passed.

    Returns
    -------
    str
        HTML span element.
    """
    if passed:
        return '<span class="pass">PASS</span>'
    return '<span class="fail">FAIL</span>'


# =============================================================================
# FairnessReport
# =============================================================================


class FairnessReport:
    """Generate an HTML fairness report for a binary classifier.

    Computes demographic parity, equalized odds, disparate impact, and
    per-group calibration metrics, then renders them into a standalone
    HTML document.

    Parameters
    ----------
    estimator : sklearn classifier
        A fitted classifier with ``predict`` and optionally ``predict_proba``.
    X : array-like of shape (n_samples, n_features)
        Evaluation features.
    y : array-like of shape (n_samples,)
        Ground truth labels (binary: 0 or 1).
    sensitive_attr : array-like of shape (n_samples,)
        Sensitive attribute values.
    sensitive_name : str, default="sensitive_attr"
        Human-readable name for the sensitive attribute (used in report
        headings).
    threshold : float, default=0.5
        Classification threshold for converting probabilities to labels
        (only used when the estimator supports ``predict_proba``).

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from endgame.fairness import FairnessReport
    >>> model = LogisticRegression().fit(X_train, y_train)
    >>> report = FairnessReport(model, X_test, y_test, sensitive_test)
    >>> html = report.generate()
    >>> report.save("report.html")
    """

    def __init__(
        self,
        estimator: Any,
        X: Any,
        y: ArrayLike,
        sensitive_attr: ArrayLike,
        sensitive_name: str = "sensitive_attr",
        threshold: float = 0.5,
    ):
        self.estimator = estimator
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.sensitive_attr = np.asarray(sensitive_attr)
        self.sensitive_name = sensitive_name
        self.threshold = threshold

        self._metrics: dict[str, Any] | None = None

    def _compute_predictions(self) -> tuple[np.ndarray, np.ndarray | None]:
        """Generate predictions and probabilities from the estimator.

        Returns
        -------
        tuple
            (y_pred, y_proba) where y_proba may be None.
        """
        has_proba = hasattr(self.estimator, "predict_proba")

        if has_proba:
            proba = self.estimator.predict_proba(self.X)
            if proba.ndim == 2:
                y_proba = proba[:, 1]
            else:
                y_proba = proba
            y_pred = (y_proba >= self.threshold).astype(int)
        else:
            y_pred = np.asarray(self.estimator.predict(self.X))
            y_proba = None

        return y_pred, y_proba

    def compute_metrics(self) -> dict[str, Any]:
        """Compute all fairness metrics.

        Returns
        -------
        dict
            Dictionary with keys ``"demographic_parity"``,
            ``"equalized_odds"``, ``"disparate_impact"``,
            ``"calibration"``, ``"y_pred"``, ``"y_proba"``.
        """
        y_pred, y_proba = self._compute_predictions()

        dp = demographic_parity(self.y, y_pred, self.sensitive_attr)
        eo = equalized_odds(self.y, y_pred, self.sensitive_attr)
        di = disparate_impact(self.y, y_pred, self.sensitive_attr)
        cal = calibration_by_group(
            self.y, y_pred, self.sensitive_attr, y_proba=y_proba
        )

        self._metrics = {
            "demographic_parity": dp,
            "equalized_odds": eo,
            "disparate_impact": di,
            "calibration": cal,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
        return self._metrics

    def generate(self) -> str:
        """Generate the full HTML fairness report.

        Returns
        -------
        str
            Complete HTML document as a string.
        """
        if self._metrics is None:
            self.compute_metrics()

        metrics = self._metrics  # type: ignore[assignment]
        dp = metrics["demographic_parity"]
        eo = metrics["equalized_odds"]
        di = metrics["disparate_impact"]
        cal = metrics["calibration"]

        groups = sorted(dp["group_rates"].keys())
        n_samples = len(self.y)
        n_positive = int(np.sum(self.y == 1))
        prevalence = n_positive / n_samples if n_samples > 0 else 0.0

        # Build HTML sections
        parts: list[str] = []
        parts.append("<!DOCTYPE html>")
        parts.append('<html lang="en"><head><meta charset="utf-8">')
        parts.append("<title>Endgame Fairness Report</title>")
        parts.append(_CSS)
        parts.append("</head><body>")

        # Header
        parts.append("<h1>Endgame Fairness Report</h1>")
        parts.append(
            f"<p>Sensitive attribute: <strong>{self.sensitive_name}</strong> "
            f"| Groups: <strong>{len(groups)}</strong> "
            f"| Samples: <strong>{n_samples:,}</strong> "
            f"| Prevalence: <strong>{prevalence:.1%}</strong></p>"
        )

        # Summary cards
        di_ratio = di["disparate_impact_ratio"]
        dp_disp = dp["max_disparity"]
        eo_disp = eo["max_disparity"]
        acc_disp = cal["accuracy_disparity"]

        parts.append('<div class="summary-grid">')
        parts.append(self._card(
            "Disparate Impact Ratio",
            f"{di_ratio:.3f}",
            "pass" if di["four_fifths_satisfied"] else "fail",
        ))
        parts.append(self._card(
            "Demographic Parity Gap",
            f"{dp_disp:.3f}",
            "pass" if dp_disp < 0.1 else ("warn" if dp_disp < 0.2 else "fail"),
        ))
        parts.append(self._card(
            "Equalized Odds Gap",
            f"{eo_disp:.3f}",
            "pass" if eo["satisfied"] else ("warn" if eo_disp < 0.1 else "fail"),
        ))
        parts.append(self._card(
            "Accuracy Parity Gap",
            f"{acc_disp:.3f}",
            "pass" if acc_disp < 0.05 else ("warn" if acc_disp < 0.1 else "fail"),
        ))
        parts.append("</div>")

        # Demographic Parity table
        parts.append("<h2>Demographic Parity</h2>")
        parts.append(
            "<p>Positive prediction rate per group. "
            f"Four-fifths rule: {_status_badge(di['four_fifths_satisfied'])}</p>"
        )
        parts.append("<table><thead><tr>")
        parts.append("<th>Group</th><th>N</th><th>Selection Rate</th><th>Bar</th>")
        parts.append("</tr></thead><tbody>")
        for g in groups:
            mask = self.sensitive_attr == g
            n_g = int(mask.sum())
            rate = dp["group_rates"][g]
            parts.append(
                f"<tr><td>{g}</td><td>{n_g:,}</td>"
                f"<td>{rate:.4f}</td><td>{_bar_html(rate)}</td></tr>"
            )
        parts.append("</tbody></table>")

        # Equalized Odds table
        parts.append("<h2>Equalized Odds</h2>")
        parts.append(
            f"<p>TPR disparity: <strong>{eo['tpr_disparity']:.4f}</strong> | "
            f"FPR disparity: <strong>{eo['fpr_disparity']:.4f}</strong> | "
            f"Satisfied (&lt;0.05): {_status_badge(eo['satisfied'])}</p>"
        )
        parts.append("<table><thead><tr>")
        parts.append("<th>Group</th><th>TPR</th><th>FPR</th><th>TPR Bar</th><th>FPR Bar</th>")
        parts.append("</tr></thead><tbody>")
        for g in groups:
            tpr = eo["group_tpr"].get(g, float("nan"))
            fpr = eo["group_fpr"].get(g, float("nan"))
            tpr_str = f"{tpr:.4f}" if not np.isnan(tpr) else "N/A"
            fpr_str = f"{fpr:.4f}" if not np.isnan(fpr) else "N/A"
            tpr_bar = _bar_html(tpr, color="#27ae60") if not np.isnan(tpr) else ""
            fpr_bar = _bar_html(fpr, color="#e74c3c") if not np.isnan(fpr) else ""
            parts.append(
                f"<tr><td>{g}</td><td>{tpr_str}</td><td>{fpr_str}</td>"
                f"<td>{tpr_bar}</td><td>{fpr_bar}</td></tr>"
            )
        parts.append("</tbody></table>")

        # Calibration table
        parts.append("<h2>Calibration by Group</h2>")
        has_proba = "group_brier_score" in cal
        parts.append("<table><thead><tr>")
        parts.append("<th>Group</th><th>Accuracy</th>")
        if has_proba:
            parts.append("<th>Brier Score</th><th>ECE</th>")
        parts.append("</tr></thead><tbody>")
        for g in groups:
            acc = cal["group_accuracy"].get(g, 0.0)
            row = f"<tr><td>{g}</td><td>{acc:.4f}</td>"
            if has_proba:
                brier = cal["group_brier_score"].get(g, 0.0)
                ece = cal["group_ece"].get(g, 0.0)
                row += f"<td>{brier:.4f}</td><td>{ece:.4f}</td>"
            row += "</tr>"
            parts.append(row)
        parts.append("</tbody></table>")

        # Group size breakdown
        parts.append("<h2>Group Composition</h2>")
        parts.append("<table><thead><tr>")
        parts.append("<th>Group</th><th>Count</th><th>Proportion</th><th>Positive Rate (True)</th>")
        parts.append("</tr></thead><tbody>")
        for g in groups:
            mask = self.sensitive_attr == g
            n_g = int(mask.sum())
            prop = n_g / n_samples if n_samples > 0 else 0.0
            pos_rate = float(np.mean(self.y[mask])) if n_g > 0 else 0.0
            parts.append(
                f"<tr><td>{g}</td><td>{n_g:,}</td>"
                f"<td>{prop:.1%}</td><td>{pos_rate:.4f}</td></tr>"
            )
        parts.append("</tbody></table>")

        # Footer
        parts.append('<div class="footer">')
        parts.append(
            "Generated by <strong>endgame.fairness.FairnessReport</strong> "
            "| Endgame ML Toolkit"
        )
        parts.append("</div>")

        parts.append("</body></html>")
        return "\n".join(parts)

    def save(self, path: str) -> str:
        """Save the HTML report to a file.

        Parameters
        ----------
        path : str
            Output file path.

        Returns
        -------
        str
            The path the report was saved to.
        """
        html = self.generate()
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)
        return path

    @staticmethod
    def _card(label: str, value: str, status: str = "pass") -> str:
        """Render a summary card.

        Parameters
        ----------
        label : str
            Card label text.
        value : str
            Card value text.
        status : str
            One of "pass", "warn", "fail".

        Returns
        -------
        str
            HTML string.
        """
        return (
            f'<div class="card">'
            f'<div class="label">{label}</div>'
            f'<div class="value {status}">{value}</div>'
            f'</div>'
        )
