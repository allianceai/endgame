"""ROC curve visualizer.

Interactive Receiver Operating Characteristic curves for classification
evaluation. Supports binary and multi-class (One-vs-Rest) with AUC
annotation, operating point markers, and the random baseline.

Example
-------
>>> from endgame.visualization import ROCCurveVisualizer
>>> from sklearn.linear_model import LogisticRegression
>>> clf = LogisticRegression().fit(X_train, y_train)
>>> viz = ROCCurveVisualizer.from_estimator(clf, X_test, y_test)
>>> viz.save("roc.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class ROCCurveVisualizer(BaseVisualizer):
    """Interactive ROC curve visualizer.

    Parameters
    ----------
    curves : list of dict
        Each dict has keys 'fpr' (list of float), 'tpr' (list of float),
        'auc' (float), 'label' (str).
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=650
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        curves: Sequence[dict[str, Any]],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 650,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title or "ROC Curve", palette=palette, width=width, height=height, theme=theme)
        self._curves = list(curves)

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        y: Any,
        *,
        class_names: Sequence[str] | None = None,
        **kwargs,
    ) -> ROCCurveVisualizer:
        """Create ROC curves from a fitted classifier.

        For binary classifiers, plots a single curve. For multiclass,
        plots one-vs-rest curves for each class.

        Parameters
        ----------
        model : estimator
            Fitted sklearn-compatible classifier with ``predict_proba``.
        X : array-like
            Test features.
        y : array-like
            True labels.
        class_names : list of str, optional
            Class names.
        **kwargs
            Additional keyword arguments.
        """
        from sklearn.metrics import auc, roc_curve
        from sklearn.preprocessing import label_binarize

        y_arr = np.asarray(y)
        classes = np.unique(y_arr)
        n_classes = len(classes)

        if class_names is None:
            if hasattr(model, "classes_"):
                class_names = [str(c) for c in model.classes_]
            else:
                class_names = [str(c) for c in classes]

        y_proba = model.predict_proba(X)

        curves = []
        if n_classes == 2:
            # Binary: single ROC
            fpr, tpr, thresholds = roc_curve(y_arr, y_proba[:, 1])
            roc_auc = auc(fpr, tpr)

            # Find optimal threshold (Youden's J)
            j_scores = tpr - fpr
            best_idx = int(np.argmax(j_scores))

            curves.append({
                "fpr": _downsample(fpr),
                "tpr": _downsample(tpr),
                "auc": round(float(roc_auc), 4),
                "label": f"ROC (AUC = {roc_auc:.3f})",
                "optimalPoint": {
                    "fpr": round(float(fpr[best_idx]), 4),
                    "tpr": round(float(tpr[best_idx]), 4),
                    "threshold": round(float(thresholds[best_idx]), 4),
                },
            })
        else:
            # Multiclass OVR
            y_bin = label_binarize(y_arr, classes=classes)
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                curves.append({
                    "fpr": _downsample(fpr),
                    "tpr": _downsample(tpr),
                    "auc": round(float(roc_auc), 4),
                    "label": f"{class_names[i]} (AUC = {roc_auc:.3f})",
                })

            # Micro-average
            fpr_micro, tpr_micro, _ = roc_curve(y_bin.ravel(), y_proba.ravel())
            micro_auc = auc(fpr_micro, tpr_micro)
            curves.append({
                "fpr": _downsample(fpr_micro),
                "tpr": _downsample(tpr_micro),
                "auc": round(float(micro_auc), 4),
                "label": f"Micro-avg (AUC = {micro_auc:.3f})",
            })

        return cls(curves, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_score: Any,
        *,
        label: str = "Model",
        **kwargs,
    ) -> ROCCurveVisualizer:
        """Create ROC curve from predictions (binary).

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_score : array-like
            Predicted probabilities or decision scores for the positive class.
        label : str, default='Model'
            Curve label.
        **kwargs
            Additional keyword arguments.
        """
        from sklearn.metrics import auc, roc_curve

        fpr, tpr, thresholds = roc_curve(np.asarray(y_true), np.asarray(y_score))
        roc_auc = auc(fpr, tpr)

        j_scores = tpr - fpr
        best_idx = int(np.argmax(j_scores))

        curves = [{
            "fpr": _downsample(fpr),
            "tpr": _downsample(tpr),
            "auc": round(float(roc_auc), 4),
            "label": f"{label} (AUC = {roc_auc:.3f})",
            "optimalPoint": {
                "fpr": round(float(fpr[best_idx]), 4),
                "tpr": round(float(tpr[best_idx]), 4),
                "threshold": round(float(thresholds[best_idx]), 4),
            },
        }]
        return cls(curves, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        return {"curves": self._curves}

    def _chart_type(self) -> str:
        return "roc_curve"

    def _get_chart_js(self) -> str:
        return _ROC_JS


def _downsample(arr, max_points: int = 500) -> list[float]:
    """Downsample an array for rendering efficiency."""
    arr = np.asarray(arr)
    if len(arr) <= max_points:
        return [round(float(v), 6) for v in arr]
    idx = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return [round(float(arr[i]), 6) for i in idx]


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_ROC_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 55, left: 55};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const curves = data.curves;

  const xScale = EG.scaleLinear([0, 1], [0, W]);
  const yScale = EG.scaleLinear([0, 1], [H, 0]);

  EG.drawXAxis(g, xScale, H, 'False Positive Rate');
  EG.drawYAxis(g, yScale, W, 'True Positive Rate');

  // Random baseline (diagonal)
  g.appendChild(EG.svg('line', {
    x1: xScale(0), y1: yScale(0), x2: xScale(1), y2: yScale(1),
    stroke: 'var(--text-muted)', 'stroke-width': 1.5,
    'stroke-dasharray': '6,4', opacity: 0.5
  }));

  // Draw curves
  curves.forEach(function(c, ci) {
    const color = palette[ci % palette.length];
    const n = Math.min(c.fpr.length, c.tpr.length);
    let d = '';
    for (let i = 0; i < n; i++) {
      d += (i === 0 ? 'M' : ' L') + xScale(c.fpr[i]) + ' ' + yScale(c.tpr[i]);
    }
    const path = EG.svg('path', {
      d: d, fill: 'none', stroke: color,
      'stroke-width': 2.5, 'stroke-linejoin': 'round'
    });
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('stroke-width', '4');
      EG.tooltip.show(e, '<b>' + EG.esc(c.label) + '</b>');
    });
    path.addEventListener('mouseleave', function() {
      path.setAttribute('stroke-width', '2.5');
      EG.tooltip.hide();
    });
    g.appendChild(path);

    // AUC fill (subtle)
    let fillD = 'M' + xScale(c.fpr[0]) + ' ' + yScale(c.tpr[0]);
    for (let i = 1; i < n; i++) {
      fillD += ' L' + xScale(c.fpr[i]) + ' ' + yScale(c.tpr[i]);
    }
    fillD += ' L' + xScale(c.fpr[n-1]) + ' ' + H + ' L' + xScale(c.fpr[0]) + ' ' + H + ' Z';
    g.appendChild(EG.svg('path', {d: fillD, fill: color, opacity: 0.06}));

    // Optimal operating point
    if (c.optimalPoint) {
      const op = c.optimalPoint;
      const cx2 = xScale(op.fpr), cy2 = yScale(op.tpr);
      const marker = EG.svg('circle', {
        cx: cx2, cy: cy2, r: 6,
        fill: 'none', stroke: color, 'stroke-width': 2.5
      });
      marker.addEventListener('mouseenter', function(e) {
        marker.setAttribute('r', '8');
        EG.tooltip.show(e,
          '<b>Optimal Point</b><br>' +
          'FPR: ' + EG.fmt(op.fpr, 3) + '<br>' +
          'TPR: ' + EG.fmt(op.tpr, 3) + '<br>' +
          'Threshold: ' + EG.fmt(op.threshold, 3));
      });
      marker.addEventListener('mouseleave', function() {
        marker.setAttribute('r', '6');
        EG.tooltip.hide();
      });
      g.appendChild(marker);
      // Crosshair
      g.appendChild(EG.svg('line', {x1: cx2, y1: cy2, x2: cx2, y2: H, stroke: color, 'stroke-width': 1, 'stroke-dasharray': '3,3', opacity: 0.4}));
      g.appendChild(EG.svg('line', {x1: 0, y1: cy2, x2: cx2, y2: cy2, stroke: color, 'stroke-width': 1, 'stroke-dasharray': '3,3', opacity: 0.4}));
    }
  });

  // Legend
  const items = curves.map(function(c, i) {
    return {label: c.label, color: palette[i % palette.length]};
  });
  EG.drawLegend(container, items);
}
"""
