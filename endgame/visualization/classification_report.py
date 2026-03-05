"""Classification report — comprehensive single-page evaluation.

Generates a self-contained HTML report with performance metrics,
confusion matrix, ROC curve, PR curve, calibration plot, feature
importances, class distribution, prediction histogram, and model
interpretability (decision tree rules, etc.).

Example
-------
>>> from endgame.visualization import ClassificationReport
>>> report = ClassificationReport(model, X_test, y_test, feature_names=fnames)
>>> report.save("classification_report.html", open_browser=True)
"""

from __future__ import annotations

import html as html_module
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from endgame.visualization._palettes import DEFAULT_CATEGORICAL, get_palette
from endgame.visualization._report_template import render_report


class ClassificationReport:
    """Comprehensive classification model evaluation report.

    Generates a multi-section HTML report with metrics, charts, and
    model interpretability for any sklearn-compatible classifier.

    Parameters
    ----------
    model : estimator
        Fitted sklearn-compatible classifier.
    X : array-like
        Test features.
    y : array-like
        True labels.
    feature_names : list of str, optional
        Feature names.
    class_names : list of str, optional
        Class label names. Auto-detected from ``model.classes_`` if absent.
    model_name : str, optional
        Display name for the model.
    dataset_name : str, optional
        Display name for the dataset.
    palette : str, default='tableau'
        Color palette.
    theme : str, default='dark'
        'dark' or 'light'.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> clf = RandomForestClassifier().fit(X_train, y_train)
    >>> report = ClassificationReport(clf, X_test, y_test)
    >>> report.save("report.html")
    """

    def __init__(
        self,
        model: Any,
        X: Any,
        y: Any,
        *,
        feature_names: Sequence[str] | None = None,
        class_names: Sequence[str] | None = None,
        model_name: str | None = None,
        dataset_name: str | None = None,
        palette: str = DEFAULT_CATEGORICAL,
        theme: str = "dark",
    ):
        self.model = model
        self.X = np.asarray(X)
        self.y = np.asarray(y)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.model_name = model_name or type(model).__name__
        self.dataset_name = dataset_name or ""
        self.palette = palette
        self.theme = theme

        # Resolve class names
        self.classes_ = np.unique(self.y)
        if class_names is not None:
            self.class_names = list(class_names)
        elif hasattr(model, "classes_"):
            self.class_names = [str(c) for c in model.classes_]
        else:
            self.class_names = [str(c) for c in self.classes_]

        self.n_classes = len(self.classes_)
        self.is_binary = self.n_classes == 2

        # Predictions
        self.y_pred = model.predict(self.X)
        self.has_proba = hasattr(model, "predict_proba")
        self.y_proba = model.predict_proba(self.X) if self.has_proba else None

        # Compute all metrics
        self._metrics = self._compute_metrics()

    # ------------------------------------------------------------------
    # Metrics computation
    # ------------------------------------------------------------------

    def _compute_metrics(self) -> dict[str, Any]:
        m = {}
        avg = "binary" if self.is_binary else "weighted"
        pos_label = self.classes_[1] if self.is_binary else None

        m["accuracy"] = round(accuracy_score(self.y, self.y_pred), 4)
        m["balanced_accuracy"] = round(balanced_accuracy_score(self.y, self.y_pred), 4)
        m["precision"] = round(precision_score(self.y, self.y_pred, average=avg, pos_label=pos_label, zero_division=0), 4)
        m["recall"] = round(recall_score(self.y, self.y_pred, average=avg, pos_label=pos_label, zero_division=0), 4)
        m["f1"] = round(f1_score(self.y, self.y_pred, average=avg, pos_label=pos_label, zero_division=0), 4)
        m["mcc"] = round(matthews_corrcoef(self.y, self.y_pred), 4)
        m["cohen_kappa"] = round(cohen_kappa_score(self.y, self.y_pred), 4)

        # Confusion-matrix derived metrics
        cm = confusion_matrix(self.y, self.y_pred, labels=self.classes_)
        if self.is_binary:
            tn, fp_cnt, fn_cnt, tp_cnt = cm.ravel()
            m["specificity"] = round(tn / (tn + fp_cnt) if (tn + fp_cnt) > 0 else 0, 4)
            m["npv"] = round(tn / (tn + fn_cnt) if (tn + fn_cnt) > 0 else 0, 4)
            m["informedness"] = round(m["recall"] + m["specificity"] - 1, 4)
            m["markedness"] = round(m["precision"] + m["npv"] - 1, 4)
        m["prevalence"] = round(float(np.mean(self.y == (self.classes_[1] if self.is_binary else self.classes_[0]))), 4)

        if self.has_proba:
            try:
                if self.is_binary:
                    m["auc"] = round(roc_auc_score(self.y, self.y_proba[:, 1]), 4)
                    m["log_loss"] = round(log_loss(self.y, self.y_proba), 4)
                    m["brier"] = round(brier_score_loss(self.y == self.classes_[1], self.y_proba[:, 1]), 4)
                else:
                    m["auc"] = round(roc_auc_score(self.y, self.y_proba, multi_class="ovr", average="weighted"), 4)
                    m["log_loss"] = round(log_loss(self.y, self.y_proba), 4)
            except Exception:
                pass

        m["n_samples"] = len(self.y)
        m["n_classes"] = self.n_classes

        # Per-class metrics
        per_class = []
        for i, cls in enumerate(self.classes_):
            mask_true = self.y == cls
            mask_pred = self.y_pred == cls
            tp = int((mask_true & mask_pred).sum())
            fp = int((~mask_true & mask_pred).sum())
            fn = int((mask_true & ~mask_pred).sum())
            tn_c = int((~mask_true & ~mask_pred).sum())
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            spec = tn_c / (tn_c + fp) if (tn_c + fp) > 0 else 0
            per_class.append({
                "class": self.class_names[i],
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1_c, 4),
                "specificity": round(spec, 4),
                "support": int(mask_true.sum()),
            })
        m["per_class"] = per_class

        return m

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def metrics(self) -> dict[str, Any]:
        """Access computed metrics dictionary."""
        return self._metrics

    def save(self, filepath: str | Path, open_browser: bool = False) -> Path:
        """Save report as self-contained HTML.

        Parameters
        ----------
        filepath : str or Path
            Output path.
        open_browser : bool, default=False
            Open in default browser after saving.

        Returns
        -------
        Path
            Absolute path to the saved file.
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".html")

        html = self._render()
        filepath.write_text(html, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(filepath.resolve().as_uri())

        return filepath.resolve()

    def _repr_html_(self) -> str:
        """Jupyter inline display."""
        return self._render()

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render(self) -> str:
        colors = get_palette(self.palette)
        m = self._metrics

        # Subtitle
        parts = [self.model_name]
        if self.dataset_name:
            parts.append(self.dataset_name)
        parts.append(f"{m['n_samples']} samples · {m['n_classes']} classes")
        subtitle = html_module.escape(" — ".join(parts))

        # Hero metric — AUC for binary, balanced accuracy for multiclass
        hero_html = self._build_hero_metric(m)

        # Metrics panel with gradient coloring
        # Each tuple: (label, display_value, score_for_grading, tooltip_description)
        metrics_cards = [
            ("Accuracy", f"{m['accuracy']:.2%}", m["accuracy"],
             "Fraction of all predictions that are correct: (TP+TN) / Total"),
            ("Balanced Acc", f"{m['balanced_accuracy']:.2%}", m["balanced_accuracy"],
             "Average per-class recall, correcting for class imbalance"),
            ("Precision", f"{m['precision']:.4f}", m["precision"],
             "Of predicted positives, how many are correct: TP / (TP+FP)"),
            ("Recall", f"{m['recall']:.4f}", m["recall"],
             "Of actual positives, how many are found: TP / (TP+FN). Also called sensitivity"),
            ("F1 Score", f"{m['f1']:.4f}", m["f1"],
             "Harmonic mean of precision and recall: 2·P·R / (P+R)"),
            ("MCC", f"{m['mcc']:.4f}", (m["mcc"] + 1) / 2,
             "Matthews Correlation Coefficient: balanced measure even with imbalanced classes. Range [-1, 1]"),
            ("Cohen κ", f"{m['cohen_kappa']:.4f}", (m["cohen_kappa"] + 1) / 2,
             "Agreement beyond chance between predictions and true labels. Range [-1, 1]"),
        ]
        if "auc" in m:
            metrics_cards.append(("AUC", f"{m['auc']:.4f}", m["auc"],
                                  "Area Under the ROC Curve: probability that a random positive ranks above a random negative"))
        if "log_loss" in m:
            metrics_cards.append(("Log Loss", f"{m['log_loss']:.4f}", max(0, 1 - m["log_loss"]),
                                  "Negative log-likelihood of predicted probabilities. Lower is better"))
        if "brier" in m:
            metrics_cards.append(("Brier Score", f"{m['brier']:.4f}", 1 - m["brier"],
                                  "Mean squared error of predicted probabilities. Lower is better. Range [0, 1]"))
        if "specificity" in m:
            metrics_cards.append(("Specificity", f"{m['specificity']:.4f}", m["specificity"],
                                  "Of actual negatives, how many are correctly identified: TN / (TN+FP)"))
        if "npv" in m:
            metrics_cards.append(("NPV", f"{m['npv']:.4f}", m["npv"],
                                  "Negative Predictive Value: of predicted negatives, how many are correct: TN / (TN+FN)"))
        if "informedness" in m:
            metrics_cards.append(("Informedness", f"{m['informedness']:.4f}", (m["informedness"] + 1) / 2,
                                  "Recall + Specificity − 1. How much the model informs beyond chance. Also called Youden's J"))
        if "markedness" in m:
            metrics_cards.append(("Markedness", f"{m['markedness']:.4f}", (m["markedness"] + 1) / 2,
                                  "Precision + NPV − 1. How marked the predictions are beyond chance"))

        metrics_html = "\n".join(
            f'<div class="metric-card {_metric_grade(score)}" data-tip="{html_module.escape(tip)}">'
            f'<div class="metric-value">{val}</div>'
            f'<div class="metric-label">{lbl}</div>'
            f'<div class="metric-tooltip">{html_module.escape(tip)}</div></div>'
            for lbl, val, score, tip in metrics_cards
        )

        # Chart sections
        sections = []
        chart_w, chart_h = 600, 420

        # 1. Confusion Matrix
        sections.append(self._section_confusion_matrix(chart_w, chart_h, colors))

        # 2. ROC Curve (if proba available)
        if self.has_proba:
            sections.append(self._section_roc(chart_w, chart_h, colors))

        # 3. PR Curve (if proba available)
        if self.has_proba:
            sections.append(self._section_pr(chart_w, chart_h, colors))

        # 4. Calibration (if proba + binary)
        if self.has_proba and self.is_binary:
            sections.append(self._section_calibration(chart_w, 500, colors))

        # 5. Threshold Analysis (if proba + binary)
        if self.has_proba and self.is_binary:
            sections.append(self._section_threshold_analysis(chart_w, chart_h, colors))

        # 6. Per-Class Metrics Bar Chart
        sections.append(self._section_per_class_bars(chart_w, chart_h, colors))

        # 7. Cumulative Gains (if proba + binary)
        if self.has_proba and self.is_binary:
            sections.append(self._section_cumulative_gains(chart_w, chart_h, colors))

        # 8. Feature Importances (if available)
        if hasattr(self.model, "feature_importances_"):
            sections.append(self._section_importances(chart_w, chart_h, colors))

        # 9. Class Distribution
        sections.append(self._section_class_distribution(chart_w, chart_h, colors))

        # 10. Confidence Histogram (if proba + binary)
        if self.has_proba and self.is_binary:
            sections.append(self._section_prediction_hist(chart_w, chart_h, colors))

        # Footer: interpretability
        footer_html = self._build_interpretability_footer()

        return render_report(
            title="Classification Report",
            subtitle=subtitle,
            theme=self.theme,
            hero_html=hero_html,
            metrics_html=metrics_html,
            sections=sections,
            footer_html=footer_html,
        )

    def _build_hero_metric(self, m: dict[str, Any]) -> str:
        """Build the hero metric card with a ring gauge."""
        if self.is_binary and "auc" in m:
            value = m["auc"]
            label = "AUC"
            desc = "Area Under the ROC Curve"
        else:
            value = m["balanced_accuracy"]
            label = "Balanced Accuracy"
            desc = "Average per-class recall"

        pct = max(0, min(1, value))
        # SVG ring gauge
        r, stroke_w = 38, 7
        circ = 2 * 3.14159 * r
        offset = circ * (1 - pct)
        color = _ring_color(pct)

        ring_svg = (
            f'<svg width="96" height="96" viewBox="0 0 96 96">'
            f'<circle cx="48" cy="48" r="{r}" fill="none" stroke="var(--border)" stroke-width="{stroke_w}"/>'
            f'<circle cx="48" cy="48" r="{r}" fill="none" stroke="{color}" stroke-width="{stroke_w}" '
            f'stroke-dasharray="{circ:.1f}" stroke-dashoffset="{offset:.1f}" '
            f'stroke-linecap="round" transform="rotate(-90 48 48)"/>'
            f'<text x="48" y="52" text-anchor="middle" fill="var(--text-primary)" '
            f'font-size="16" font-weight="700" font-family="SF Mono,Fira Code,Consolas,monospace">'
            f'{value:.3f}</text></svg>'
        )

        return (
            f'<div class="hero-metric">'
            f'<div class="hero-ring">{ring_svg}</div>'
            f'<div class="hero-info">'
            f'<div class="hero-value">{value:.4f}</div>'
            f'<div class="hero-label">{label}</div>'
            f'<div class="hero-desc">{desc}</div>'
            f'</div></div>'
        )

    # ------------------------------------------------------------------
    # Chart sections
    # ------------------------------------------------------------------

    def _section_confusion_matrix(self, w, h, colors):
        cm = confusion_matrix(self.y, self.y_pred, labels=self.classes_)
        n = len(self.classes_)
        matrix = [[int(cm[i][j]) for j in range(n)] for i in range(n)]
        total = int(cm.sum())

        data = {
            "matrix": matrix,
            "classNames": self.class_names,
            "total": total,
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Confusion Matrix",
            "chart_id": "cm",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _CM_SECTION_JS,
        }

    def _section_roc(self, w, h, colors):
        from sklearn.metrics import auc, roc_curve

        curves = []
        if self.is_binary:
            fpr, tpr, _ = roc_curve(self.y, self.y_proba[:, 1], pos_label=self.classes_[1])
            roc_auc = auc(fpr, tpr)
            curves.append({
                "fpr": _ds(fpr), "tpr": _ds(tpr),
                "auc": round(float(roc_auc), 4),
                "label": f"ROC (AUC = {roc_auc:.3f})",
            })
        else:
            y_bin = label_binarize(self.y, classes=self.classes_)
            for i, cls in enumerate(self.classes_):
                fpr, tpr, _ = roc_curve(y_bin[:, i], self.y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                curves.append({
                    "fpr": _ds(fpr), "tpr": _ds(tpr),
                    "auc": round(float(roc_auc), 4),
                    "label": f"{self.class_names[i]} (AUC = {roc_auc:.3f})",
                })

        data = {"curves": curves}
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "ROC Curve",
            "chart_id": "roc",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _ROC_SECTION_JS,
        }

    def _section_pr(self, w, h, colors):
        from sklearn.metrics import average_precision_score, precision_recall_curve

        curves = []
        if self.is_binary:
            prec, rec, _ = precision_recall_curve(self.y, self.y_proba[:, 1], pos_label=self.classes_[1])
            ap = average_precision_score(self.y == self.classes_[1], self.y_proba[:, 1])
            curves.append({
                "precision": _ds(prec), "recall": _ds(rec),
                "ap": round(float(ap), 4),
                "label": f"PR (AP = {ap:.3f})",
            })
            prevalence = float(np.mean(self.y == self.classes_[1]))
        else:
            y_bin = label_binarize(self.y, classes=self.classes_)
            prevalence = None
            for i, cls in enumerate(self.classes_):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], self.y_proba[:, i])
                ap = average_precision_score(y_bin[:, i], self.y_proba[:, i])
                curves.append({
                    "precision": _ds(prec), "recall": _ds(rec),
                    "ap": round(float(ap), 4),
                    "label": f"{self.class_names[i]} (AP = {ap:.3f})",
                })

        data = {"curves": curves, "prevalence": prevalence if self.is_binary else None}
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Precision-Recall Curve",
            "chart_id": "pr",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _PR_SECTION_JS,
        }

    def _section_calibration(self, w, h, colors):
        n_bins = 10
        y_binary = (self.y == self.classes_[1]).astype(float)
        y_prob = self.y_proba[:, 1]
        bin_edges = np.linspace(0, 1, n_bins + 1)

        prob_true, prob_pred, counts, hist_bins = [], [], [], []
        for i in range(n_bins):
            lo, hi = bin_edges[i], bin_edges[i + 1]
            mask = (y_prob >= lo) & (y_prob < hi) if i < n_bins - 1 else (y_prob >= lo) & (y_prob <= hi)
            n_in = int(mask.sum())
            counts.append(n_in)
            hist_bins.append(round(float((lo + hi) / 2), 4))
            if n_in > 0:
                prob_true.append(round(float(y_binary[mask].mean()), 6))
                prob_pred.append(round(float(y_prob[mask].mean()), 6))
            else:
                prob_true.append(None)
                prob_pred.append(None)

        ece = sum(
            abs(pt - pp) * c / len(y_prob)
            for pt, pp, c in zip(prob_true, prob_pred, counts)
            if pt is not None and pp is not None and c > 0
        )

        curve = {
            "probTrue": prob_true, "probPred": prob_pred,
            "counts": counts, "histBins": hist_bins,
            "ece": round(float(ece), 4), "mce": 0,
            "label": self.model_name,
        }
        data = {"curves": [curve], "nBins": n_bins}
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Calibration Plot",
            "chart_id": "cal",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _CAL_SECTION_JS,
        }

    def _section_importances(self, w, h, colors):
        raw_imp = self.model.feature_importances_
        # Handle dict-style importances (e.g., LGBMWrapper returns {name: value})
        if isinstance(raw_imp, dict):
            names = list(raw_imp.keys())
            imp = np.array(list(raw_imp.values()))
        else:
            imp = np.asarray(raw_imp)
            names = self.feature_names or [f"Feature {i}" for i in range(len(imp))]
        top_n = min(20, len(imp))
        idx = np.argsort(imp)[::-1][:top_n]

        data = {
            "labels": [names[i] for i in idx],
            "values": [round(float(imp[i]), 6) for i in idx],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": f"Feature Importances (Top {top_n})",
            "chart_id": "imp",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _IMP_SECTION_JS,
        }

    def _section_class_distribution(self, w, h, colors):
        unique, counts = np.unique(self.y, return_counts=True)
        data = {
            "labels": self.class_names,
            "values": [int(c) for c in counts],
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Class Distribution (Test Set)",
            "chart_id": "classdist",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _CLASSDIST_SECTION_JS,
        }

    def _section_prediction_hist(self, w, h, colors):
        y_prob = self.y_proba[:, 1]
        n_bins = 30
        counts_hist, edges = np.histogram(y_prob, bins=n_bins, range=(0, 1))
        bin_centers = [(edges[i] + edges[i + 1]) / 2 for i in range(n_bins)]

        # Split by class
        mask_pos = self.y == self.classes_[1]
        counts_pos, _ = np.histogram(y_prob[mask_pos], bins=n_bins, range=(0, 1))
        counts_neg, _ = np.histogram(y_prob[~mask_pos], bins=n_bins, range=(0, 1))

        data = {
            "bins": [round(float(b), 4) for b in bin_centers],
            "countsPos": [int(c) for c in counts_pos],
            "countsNeg": [int(c) for c in counts_neg],
            "posLabel": self.class_names[1] if len(self.class_names) > 1 else "Positive",
            "negLabel": self.class_names[0] if len(self.class_names) > 0 else "Negative",
        }
        config = {"width": w, "height": h, "palette": colors}

        return {
            "title": "Prediction Score Distribution",
            "chart_id": "predhist",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _PREDHIST_SECTION_JS,
        }

    def _section_threshold_analysis(self, w, h, colors):
        """Precision, recall, F1 vs classification threshold (binary only)."""
        from sklearn.metrics import (
            f1_score as _f1,
        )
        from sklearn.metrics import (
            precision_score as _prec,
        )
        from sklearn.metrics import (
            recall_score as _rec,
        )

        y_prob = self.y_proba[:, 1]
        y_bin = (self.y == self.classes_[1]).astype(int)
        thresholds = np.linspace(0.01, 0.99, 80)
        precs, recs, f1s = [], [], []
        best_f1, best_t = 0, 0.5
        for t in thresholds:
            preds = (y_prob >= t).astype(int)
            p = _prec(y_bin, preds, zero_division=0)
            r = _rec(y_bin, preds, zero_division=0)
            f = _f1(y_bin, preds, zero_division=0)
            precs.append(round(float(p), 4))
            recs.append(round(float(r), 4))
            f1s.append(round(float(f), 4))
            if f > best_f1:
                best_f1, best_t = f, t

        data = {
            "thresholds": [round(float(t), 4) for t in thresholds],
            "precision": precs,
            "recall": recs,
            "f1": f1s,
            "bestThreshold": round(float(best_t), 4),
            "bestF1": round(float(best_f1), 4),
        }
        config = {"width": w, "height": h, "palette": colors}
        return {
            "title": "Threshold Analysis",
            "chart_id": "thresh",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _THRESHOLD_SECTION_JS,
        }

    def _section_per_class_bars(self, w, h, colors):
        """Grouped bar chart of precision, recall, F1 per class."""
        per_class = self._metrics["per_class"]
        data = {
            "classes": [pc["class"] for pc in per_class],
            "precision": [pc["precision"] for pc in per_class],
            "recall": [pc["recall"] for pc in per_class],
            "f1": [pc["f1"] for pc in per_class],
        }
        config = {"width": w, "height": h, "palette": colors}
        return {
            "title": "Per-Class Metrics",
            "chart_id": "perclass",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _PERCLASS_SECTION_JS,
        }

    def _section_cumulative_gains(self, w, h, colors):
        """Cumulative gains curve (binary only)."""
        y_prob = self.y_proba[:, 1]
        y_bin = (self.y == self.classes_[1]).astype(int)
        n = len(y_bin)
        n_pos = int(y_bin.sum())

        # Sort by predicted probability descending
        order = np.argsort(-y_prob)
        y_sorted = y_bin[order]
        cum_pos = np.cumsum(y_sorted)

        # Normalize
        pct_samples = np.arange(1, n + 1) / n
        pct_captured = cum_pos / max(n_pos, 1)

        # Downsample for JSON
        step = max(1, n // 200)
        data = {
            "pctSamples": _ds(pct_samples[::step]),
            "pctCaptured": _ds(pct_captured[::step]),
            "nPos": n_pos,
            "nTotal": n,
        }
        config = {"width": w, "height": h, "palette": colors}
        return {
            "title": "Cumulative Gains",
            "chart_id": "gains",
            "width": w,
            "height": h,
            "data_json": json.dumps(data),
            "config_json": json.dumps(config),
            "chart_js": _GAINS_SECTION_JS,
        }

    # ------------------------------------------------------------------
    # Interpretability
    # ------------------------------------------------------------------

    def _build_interpretability_footer(self) -> str:
        parts = []

        # Decision tree text rules
        if _is_decision_tree(self.model):
            rules = _extract_tree_rules(self.model, self.feature_names, self.class_names)
            if rules:
                parts.append('<div class="interp-section">')
                parts.append("<h2>Decision Tree Rules</h2>")
                parts.append('<ol class="rules-list">')
                for rule in rules[:30]:  # cap at 30
                    parts.append(f"<li>{html_module.escape(rule)}</li>")
                if len(rules) > 30:
                    parts.append(f"<li>... and {len(rules) - 30} more rules</li>")
                parts.append("</ol></div>")

        # Linear model coefficients
        if _is_linear(self.model):
            coefs = _extract_linear_coefs(self.model, self.feature_names)
            if coefs:
                parts.append('<div class="interp-section">')
                parts.append("<h2>Model Coefficients (Top 20 by |coef|)</h2>")
                parts.append('<ol class="rules-list">')
                for name, coef in coefs[:20]:
                    sign = "+" if coef >= 0 else ""
                    parts.append(f"<li>{html_module.escape(name)}: {sign}{coef:.4f}</li>")
                parts.append("</ol></div>")

        # Per-class breakdown table
        per_class = self._metrics.get("per_class", [])
        if per_class:
            parts.append('<div class="report-footer">')
            parts.append("<h3>Per-Class Metrics</h3>")
            parts.append("<pre>")
            header = f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}"
            parts.append(header)
            parts.append("-" * len(header))
            for pc in per_class:
                parts.append(
                    f"{pc['class']:<20} {pc['precision']:>10.4f} {pc['recall']:>10.4f} "
                    f"{pc['f1']:>10.4f} {pc['support']:>10d}"
                )
            parts.append("</pre></div>")

        return "\n".join(parts)


# ===================================================================
# Helpers
# ===================================================================

def _ds(arr, max_pts=400):
    """Downsample array for JSON."""
    arr = np.asarray(arr)
    if len(arr) <= max_pts:
        return [round(float(v), 6) for v in arr]
    idx = np.linspace(0, len(arr) - 1, max_pts, dtype=int)
    return [round(float(arr[i]), 6) for i in idx]


def _is_decision_tree(model):
    cls_name = type(model).__name__
    return cls_name in ("DecisionTreeClassifier", "DecisionTreeRegressor")


def _is_linear(model):
    return hasattr(model, "coef_") and hasattr(model, "intercept_")


def _extract_tree_rules(model, feature_names, class_names):
    """Extract human-readable rules from a sklearn decision tree."""
    try:
        from sklearn.tree import export_text
        names = feature_names or [f"feature_{i}" for i in range(model.n_features_in_)]
        text = export_text(model, feature_names=names, max_depth=6)
        rules = [line for line in text.strip().split("\n") if line.strip()]
        return rules
    except Exception:
        return []


def _extract_linear_coefs(model, feature_names):
    """Extract sorted (name, coef) pairs from a linear model."""
    try:
        coefs = np.asarray(model.coef_).ravel()
        names = feature_names or [f"feature_{i}" for i in range(len(coefs))]
        if len(names) != len(coefs):
            names = [f"feature_{i}" for i in range(len(coefs))]
        pairs = list(zip(names, coefs))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs
    except Exception:
        return []


def _metric_grade(score: float) -> str:
    """Return a CSS class based on metric quality."""
    if score >= 0.8:
        return "metric-good"
    if score >= 0.6:
        return "metric-ok"
    return "metric-poor"


def _ring_color(pct: float) -> str:
    """Color for the hero ring gauge."""
    if pct >= 0.9:
        return "#4caf50"
    if pct >= 0.8:
        return "#66bb6a"
    if pct >= 0.7:
        return "#ffc107"
    if pct >= 0.6:
        return "#ff9800"
    return "#f44336"


# ===================================================================
# Section JavaScript renderers
# ===================================================================

_CM_SECTION_JS = r"""
function renderChart_cm(data, config, container) {
  const n = data.classNames.length;
  const W = config.width, H = config.height;
  const margin = {top: 30, right: 20, bottom: 60, left: 80};
  const svg = EG.svg('svg', {width: W, height: H});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW = W - margin.left - margin.right;
  const iH = H - margin.top - margin.bottom;
  const cellW = iW / n, cellH = iH / n;
  let maxVal = 0;
  data.matrix.forEach(r => r.forEach(v => { if(v > maxVal) maxVal = v; }));
  if(maxVal===0) maxVal=1;

  function heatColor(v) {
    const t = v / maxVal;
    const r = Math.round(30 + 200*t), gg = Math.round(60 + 80*(1-t)), b = Math.round(180 - 80*t);
    return `rgb(${r},${gg},${b})`;
  }

  for(let i=0;i<n;i++){
    for(let j=0;j<n;j++){
      const v = data.matrix[i][j];
      const rect = EG.svg('rect', {x:j*cellW, y:i*cellH, width:cellW-2, height:cellH-2, fill:heatColor(v), rx:4});
      rect.addEventListener('mouseenter', e => {
        EG.tooltip.show(e, `<b>True:</b> ${data.classNames[i]}<br><b>Pred:</b> ${data.classNames[j]}<br><b>Count:</b> ${v}`);
      });
      rect.addEventListener('mouseleave', () => EG.tooltip.hide());
      g.appendChild(rect);
      const txt = EG.svg('text', {x:j*cellW+cellW/2-1, y:i*cellH+cellH/2+5, 'text-anchor':'middle', fill:v/maxVal>0.5?'#fff':'var(--text-primary)', 'font-size':'14px', 'font-weight':'600'});
      txt.textContent = v;
      g.appendChild(txt);
    }
  }
  // Labels
  for(let i=0;i<n;i++){
    g.appendChild(EG.svg('text', {x:-8, y:i*cellH+cellH/2+4, 'text-anchor':'end', fill:'var(--text-secondary)', 'font-size':'11px'})).textContent = data.classNames[i];
    g.appendChild(EG.svg('text', {x:i*cellW+cellW/2, y:iH+18, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'11px'})).textContent = data.classNames[i];
  }
  g.appendChild(EG.svg('text', {x:iW/2, y:iH+42, 'text-anchor':'middle', fill:'var(--text-muted)', 'font-size':'12px'})).textContent = 'Predicted';
  g.appendChild(EG.svg('text', {'text-anchor':'middle', fill:'var(--text-muted)', 'font-size':'12px', transform:`translate(-55,${iH/2}) rotate(-90)`})).textContent = 'Actual';
}
"""

_ROC_SECTION_JS = r"""
function renderChart_roc(data, config, container) {
  const margin = {top:10, right:15, bottom:50, left:50};
  const W = config.width, H = config.height;
  const svg = EG.svg('svg', {width:W, height:H});
  container.appendChild(svg);
  const g = EG.svg('g', {transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right, iH=H-margin.top-margin.bottom;
  const xS=EG.scaleLinear([0,1],[0,iW]), yS=EG.scaleLinear([0,1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'False Positive Rate');
  EG.drawYAxis(g,yS,iW,'True Positive Rate');
  g.appendChild(EG.svg('line',{x1:xS(0),y1:yS(0),x2:xS(1),y2:yS(1),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));
  data.curves.forEach((c,ci)=>{
    const color=config.palette[ci%config.palette.length];
    let d='';
    const n=Math.min(c.fpr.length,c.tpr.length);
    for(let i=0;i<n;i++) d+=(i===0?'M':' L')+xS(c.fpr[i])+' '+yS(c.tpr[i]);
    let fillD=d+' L'+xS(c.fpr[n-1])+' '+iH+' L'+xS(c.fpr[0])+' '+iH+' Z';
    g.appendChild(EG.svg('path',{d:fillD,fill:color,opacity:0.06}));
    const path=EG.svg('path',{d:d,fill:'none',stroke:color,'stroke-width':2.5});
    path.addEventListener('mouseenter',e=>{path.setAttribute('stroke-width','4');EG.tooltip.show(e,'<b>'+EG.esc(c.label)+'</b>');});
    path.addEventListener('mouseleave',()=>{path.setAttribute('stroke-width','2.5');EG.tooltip.hide();});
    g.appendChild(path);
  });
  const items=data.curves.map((c,i)=>({label:c.label,color:config.palette[i%config.palette.length]}));
  EG.drawLegend(container,items);
}
"""

_PR_SECTION_JS = r"""
function renderChart_pr(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const xS=EG.scaleLinear([0,1],[0,iW]),yS=EG.scaleLinear([0,1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Recall');
  EG.drawYAxis(g,yS,iW,'Precision');
  if(data.prevalence!=null){
    const py=yS(data.prevalence);
    g.appendChild(EG.svg('line',{x1:0,y1:py,x2:iW,y2:py,stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));
  }
  data.curves.forEach((c,ci)=>{
    const color=config.palette[ci%config.palette.length];
    const n=Math.min(c.precision.length,c.recall.length);
    let d='';
    for(let i=0;i<n;i++) d+=(i===0?'M':' L')+xS(c.recall[i])+' '+yS(c.precision[i]);
    g.appendChild(EG.svg('path',{d:d+' L'+xS(c.recall[n-1])+' '+iH+' L'+xS(c.recall[0])+' '+iH+' Z',fill:color,opacity:0.06}));
    const path=EG.svg('path',{d:d,fill:'none',stroke:color,'stroke-width':2.5});
    path.addEventListener('mouseenter',e=>{path.setAttribute('stroke-width','4');EG.tooltip.show(e,'<b>'+EG.esc(c.label)+'</b>');});
    path.addEventListener('mouseleave',()=>{path.setAttribute('stroke-width','2.5');EG.tooltip.hide();});
    g.appendChild(path);
  });
  EG.drawLegend(container,data.curves.map((c,i)=>({label:c.label,color:config.palette[i%config.palette.length]})));
}
"""

_CAL_SECTION_JS = r"""
function renderChart_cal(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height*0.65;
  const svg=EG.svg('svg',{width:config.width,height:config.height});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-10;
  const xS=EG.scaleLinear([0,1],[0,iW]),yS=EG.scaleLinear([0,1],[iH,0]);
  const ticks=[0,0.2,0.4,0.6,0.8,1.0];
  ticks.forEach(v=>{
    g.appendChild(EG.svg('line',{x1:0,y1:yS(v),x2:iW,y2:yS(v),stroke:'var(--grid-line)'}));
    g.appendChild(EG.svg('text',{x:-8,y:yS(v)+4,'text-anchor':'end',fill:'var(--text-secondary)','font-size':'10px'})).textContent=v.toFixed(1);
    g.appendChild(EG.svg('text',{x:xS(v),y:iH+15,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':'10px'})).textContent=v.toFixed(1);
  });
  g.appendChild(EG.svg('line',{x1:xS(0),y1:yS(0),x2:xS(1),y2:yS(1),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.6}));
  g.appendChild(EG.svg('text',{x:iW/2,y:iH+35,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':'11px'})).textContent='Mean Predicted Probability';

  data.curves.forEach((c,ci)=>{
    const color=config.palette[ci%config.palette.length];
    let d='';const pts=[];
    for(let i=0;i<c.probTrue.length;i++){
      if(c.probTrue[i]===null||c.probPred[i]===null)continue;
      pts.push({x:xS(c.probPred[i]),y:yS(c.probTrue[i]),pred:c.probPred[i],t:c.probTrue[i],cnt:c.counts[i]});
    }
    pts.forEach((p,i)=>{d+=(i===0?'M':' L')+p.x+' '+p.y;});
    if(d)g.appendChild(EG.svg('path',{d:d,fill:'none',stroke:color,'stroke-width':2.5}));
    pts.forEach(p=>{
      const dot=EG.svg('circle',{cx:p.x,cy:p.y,r:5,fill:color,stroke:'var(--bg-card)','stroke-width':2});
      dot.addEventListener('mouseenter',e=>{dot.setAttribute('r','7');EG.tooltip.show(e,'Pred: '+EG.fmt(p.pred,3)+'<br>Actual: '+EG.fmt(p.t,3)+'<br>n='+p.cnt);});
      dot.addEventListener('mouseleave',()=>{dot.setAttribute('r','5');EG.tooltip.hide();});
      g.appendChild(dot);
    });
    g.appendChild(EG.svg('text',{x:10,y:16,fill:color,'font-size':'11px','font-weight':'600'})).textContent='ECE='+c.ece.toFixed(3);
  });

  // Histogram
  const gH=EG.svg('g',{transform:`translate(${margin.left},${H+15})`});
  svg.appendChild(gH);
  const hH=config.height-H-40;
  let maxC=0;data.curves.forEach(c=>c.counts.forEach(v=>{if(v>maxC)maxC=v;}));if(!maxC)maxC=1;
  const hY=EG.scaleLinear([0,maxC],[hH,0]);
  gH.appendChild(EG.svg('line',{x1:0,y1:hH,x2:iW,y2:hH,stroke:'var(--border)'}));
  data.curves.forEach((c,ci)=>{
    const color=config.palette[ci%config.palette.length];
    const barW=iW/c.histBins.length*0.7;
    c.counts.forEach((cnt,i)=>{
      const x=xS(c.histBins[i])-barW/2;
      const bH=hH-hY(cnt);
      gH.appendChild(EG.svg('rect',{x:x,y:hH-bH,width:barW,height:Math.max(bH,0),fill:color,opacity:0.5,rx:2}));
    });
  });
}
"""

_IMP_SECTION_JS = r"""
function renderChart_imp(data, config, container) {
  const margin={top:10,right:30,bottom:30,left:140};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const n=data.labels.length;
  const rowH=iH/n;
  const maxV=Math.max.apply(null,data.values)||1;
  const xS=EG.scaleLinear([0,maxV],[0,iW]);

  for(let i=0;i<n;i++){
    const y=i*rowH,v=data.values[i];
    const color=config.palette[i%config.palette.length];
    const bW=xS(v);
    const rect=EG.svg('rect',{x:0,y:y+2,width:Math.max(bW,2),height:rowH-4,fill:color,rx:3,opacity:0.8});
    rect.addEventListener('mouseenter',e=>{rect.setAttribute('opacity','1');EG.tooltip.show(e,'<b>'+EG.esc(data.labels[i])+'</b><br>'+EG.fmt(v,4));});
    rect.addEventListener('mouseleave',()=>{rect.setAttribute('opacity','0.8');EG.tooltip.hide();});
    g.appendChild(rect);
    g.appendChild(EG.svg('text',{x:bW+5,y:y+rowH/2+4,fill:'var(--text-secondary)','font-size':'10px'})).textContent=EG.fmt(v,4);
    g.appendChild(EG.svg('text',{x:-6,y:y+rowH/2+4,'text-anchor':'end',fill:'var(--text-primary)','font-size':'11px'})).textContent=data.labels[i].length>20?data.labels[i].slice(0,18)+'…':data.labels[i];
  }
}
"""

_CLASSDIST_SECTION_JS = r"""
function renderChart_classdist(data, config, container) {
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const cx=W/2,cy=H/2-10,R=Math.min(W,H)*0.35;
  const total=data.values.reduce((a,b)=>a+b,0)||1;
  let angle=-Math.PI/2;
  data.values.forEach((v,i)=>{
    const sweep=v/total*Math.PI*2;
    const x1=cx+R*Math.cos(angle),y1=cy+R*Math.sin(angle);
    const x2=cx+R*Math.cos(angle+sweep),y2=cy+R*Math.sin(angle+sweep);
    const large=sweep>Math.PI?1:0;
    const d=`M${cx} ${cy} L${x1} ${y1} A${R} ${R} 0 ${large} 1 ${x2} ${y2} Z`;
    const color=config.palette[i%config.palette.length];
    const path=EG.svg('path',{d:d,fill:color,stroke:'var(--bg-card)','stroke-width':2,opacity:0.85});
    const midA=angle+sweep/2;
    path.addEventListener('mouseenter',e=>{path.setAttribute('opacity','1');EG.tooltip.show(e,'<b>'+EG.esc(data.labels[i])+'</b><br>'+v+' ('+(v/total*100).toFixed(1)+'%)');});
    path.addEventListener('mouseleave',()=>{path.setAttribute('opacity','0.85');EG.tooltip.hide();});
    svg.appendChild(path);
    // Label
    const lR=R*0.65;
    const lx=cx+lR*Math.cos(midA),ly=cy+lR*Math.sin(midA);
    if(sweep>0.2) svg.appendChild(EG.svg('text',{x:lx,y:ly+4,'text-anchor':'middle',fill:'#fff','font-size':'12px','font-weight':'600'})).textContent=(v/total*100).toFixed(0)+'%';
    angle+=sweep;
  });
  EG.drawLegend(container,data.labels.map((l,i)=>({label:l+' ('+data.values[i]+')',color:config.palette[i%config.palette.length]})));
}
"""

_PREDHIST_SECTION_JS = r"""
function renderChart_predhist(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const bins=data.bins,cp=data.countsPos,cn=data.countsNeg;
  const maxC=Math.max(Math.max.apply(null,cp),Math.max.apply(null,cn))||1;
  const xS=EG.scaleLinear([0,1],[0,iW]),yS=EG.scaleLinear([0,maxC*1.1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Predicted Probability');
  EG.drawYAxis(g,yS,iW,'Count');
  const barW=iW/bins.length*0.4;
  const posColor=config.palette[0],negColor=config.palette[1%config.palette.length];
  bins.forEach((b,i)=>{
    const x=xS(b);
    const hN=iH-yS(cn[i]),hP=iH-yS(cp[i]);
    g.appendChild(EG.svg('rect',{x:x-barW-1,y:iH-hN,width:barW,height:Math.max(hN,0),fill:negColor,opacity:0.6,rx:2}));
    g.appendChild(EG.svg('rect',{x:x+1,y:iH-hP,width:barW,height:Math.max(hP,0),fill:posColor,opacity:0.6,rx:2}));
  });
  EG.drawLegend(container,[{label:data.negLabel,color:negColor},{label:data.posLabel,color:posColor}]);
}
"""

_THRESHOLD_SECTION_JS = r"""
function renderChart_thresh(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const xS=EG.scaleLinear([0,1],[0,iW]),yS=EG.scaleLinear([0,1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'Classification Threshold');
  EG.drawYAxis(g,yS,iW,'Score');

  const series=[
    {vals:data.precision, label:'Precision', color:config.palette[0]},
    {vals:data.recall, label:'Recall', color:config.palette[1%config.palette.length]},
    {vals:data.f1, label:'F1', color:config.palette[2%config.palette.length]},
  ];

  series.forEach(s=>{
    let d='';
    for(let i=0;i<data.thresholds.length;i++){
      d+=(i===0?'M':' L')+xS(data.thresholds[i])+' '+yS(s.vals[i]);
    }
    const path=EG.svg('path',{d:d,fill:'none',stroke:s.color,'stroke-width':2.5});
    path.addEventListener('mouseenter',e=>{path.setAttribute('stroke-width','4');EG.tooltip.show(e,'<b>'+s.label+'</b>');});
    path.addEventListener('mouseleave',()=>{path.setAttribute('stroke-width','2.5');EG.tooltip.hide();});
    g.appendChild(path);
  });

  // Mark optimal threshold
  const bx=xS(data.bestThreshold),by=yS(data.bestF1);
  g.appendChild(EG.svg('line',{x1:bx,y1:0,x2:bx,y2:iH,stroke:'var(--text-muted)','stroke-width':1,'stroke-dasharray':'4,3'}));
  const dot=EG.svg('circle',{cx:bx,cy:by,r:6,fill:config.palette[2%config.palette.length],stroke:'var(--bg-card)','stroke-width':2});
  dot.addEventListener('mouseenter',e=>{EG.tooltip.show(e,'<b>Best F1</b><br>Threshold: '+data.bestThreshold.toFixed(3)+'<br>F1: '+data.bestF1.toFixed(3));});
  dot.addEventListener('mouseleave',()=>{EG.tooltip.hide();});
  g.appendChild(dot);

  EG.drawLegend(container,series.map(s=>({label:s.label,color:s.color})));
}
"""

_PERCLASS_SECTION_JS = r"""
function renderChart_perclass(data, config, container) {
  const margin={top:10,right:20,bottom:60,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const n=data.classes.length;
  const metrics=['precision','recall','f1'];
  const colors=[config.palette[0],config.palette[1%config.palette.length],config.palette[2%config.palette.length]];
  const groupW=iW/n;
  const barW=groupW*0.25;
  const yS=EG.scaleLinear([0,1],[iH,0]);
  EG.drawYAxis(g,yS,iW,'Score');

  // Grid lines
  [0,0.25,0.5,0.75,1.0].forEach(v=>{
    g.appendChild(EG.svg('line',{x1:0,y1:yS(v),x2:iW,y2:yS(v),stroke:'var(--grid-line)'}));
  });

  for(let i=0;i<n;i++){
    const gx=i*groupW+groupW*0.15;
    metrics.forEach((m,mi)=>{
      const v=data[m][i];
      const bH=iH-yS(v);
      const x=gx+mi*barW+mi*2;
      const rect=EG.svg('rect',{x:x,y:iH-bH,width:barW,height:Math.max(bH,0),fill:colors[mi],rx:3,opacity:0.8});
      rect.addEventListener('mouseenter',e=>{
        rect.setAttribute('opacity','1');
        EG.tooltip.show(e,'<b>'+data.classes[i]+'</b><br>'+m+': '+v.toFixed(4));
      });
      rect.addEventListener('mouseleave',()=>{rect.setAttribute('opacity','0.8');EG.tooltip.hide();});
      g.appendChild(rect);
    });
    // Class label
    const lbl=EG.svg('text',{x:gx+barW*1.5+2,y:iH+18,'text-anchor':'middle',fill:'var(--text-secondary)','font-size':'11px'});
    lbl.textContent=data.classes[i].length>12?data.classes[i].slice(0,10)+'…':data.classes[i];
    g.appendChild(lbl);
  }

  EG.drawLegend(container,metrics.map((m,i)=>({label:m.charAt(0).toUpperCase()+m.slice(1),color:colors[i]})));
}
"""

_GAINS_SECTION_JS = r"""
function renderChart_gains(data, config, container) {
  const margin={top:10,right:15,bottom:50,left:50};
  const W=config.width,H=config.height;
  const svg=EG.svg('svg',{width:W,height:H});
  container.appendChild(svg);
  const g=EG.svg('g',{transform:`translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const iW=W-margin.left-margin.right,iH=H-margin.top-margin.bottom;
  const xS=EG.scaleLinear([0,1],[0,iW]),yS=EG.scaleLinear([0,1],[iH,0]);
  EG.drawXAxis(g,xS,iH,'% of Samples (sorted by score)');
  EG.drawYAxis(g,yS,iW,'% of Positives Captured');

  // Random baseline
  g.appendChild(EG.svg('line',{x1:xS(0),y1:yS(0),x2:xS(1),y2:yS(1),stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));

  // Model curve
  const color=config.palette[0];
  let d='';
  const n=Math.min(data.pctSamples.length,data.pctCaptured.length);
  for(let i=0;i<n;i++) d+=(i===0?'M':' L')+xS(data.pctSamples[i])+' '+yS(data.pctCaptured[i]);
  // Fill
  g.appendChild(EG.svg('path',{d:d+' L'+xS(data.pctSamples[n-1])+' '+iH+' L'+xS(data.pctSamples[0])+' '+iH+' Z',fill:color,opacity:0.08}));
  const path=EG.svg('path',{d:d,fill:'none',stroke:color,'stroke-width':2.5});
  path.addEventListener('mouseenter',e=>{path.setAttribute('stroke-width','4');EG.tooltip.show(e,'<b>Cumulative Gains</b><br>'+data.nPos+' positives / '+data.nTotal+' total');});
  path.addEventListener('mouseleave',()=>{path.setAttribute('stroke-width','2.5');EG.tooltip.hide();});
  g.appendChild(path);

  EG.drawLegend(container,[{label:'Model',color:color},{label:'Random',color:'var(--text-muted)'}]);
}
"""
