"""Precision-Recall curve visualizer.

Interactive precision-recall curves for classification evaluation,
particularly important for imbalanced datasets. Supports binary and
multi-class with AP (average precision) annotation.

Example
-------
>>> from endgame.visualization import PRCurveVisualizer
>>> from sklearn.linear_model import LogisticRegression
>>> clf = LogisticRegression().fit(X_train, y_train)
>>> viz = PRCurveVisualizer.from_estimator(clf, X_test, y_test)
>>> viz.save("pr_curve.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class PRCurveVisualizer(BaseVisualizer):
    """Interactive Precision-Recall curve visualizer.

    Parameters
    ----------
    curves : list of dict
        Each dict has keys 'precision' (list of float), 'recall'
        (list of float), 'ap' (float), 'label' (str).
    prevalence : float, optional
        Positive class prevalence (shown as baseline).
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
        prevalence: float | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 650,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Precision-Recall Curve", palette=palette, width=width, height=height, theme=theme)
        self._curves = list(curves)
        self.prevalence = prevalence

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        y: Any,
        *,
        class_names: Sequence[str] | None = None,
        **kwargs,
    ) -> PRCurveVisualizer:
        """Create PR curves from a fitted classifier.

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
        from sklearn.metrics import average_precision_score, precision_recall_curve
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
            prec, rec, thresholds = precision_recall_curve(y_arr, y_proba[:, 1])
            ap = average_precision_score(y_arr, y_proba[:, 1])

            # F1-optimal point
            f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
            best_idx = int(np.argmax(f1_scores[:-1]))  # last element has rec=0

            curves.append({
                "precision": _downsample(prec),
                "recall": _downsample(rec),
                "ap": round(float(ap), 4),
                "label": f"PR (AP = {ap:.3f})",
                "optimalPoint": {
                    "precision": round(float(prec[best_idx]), 4),
                    "recall": round(float(rec[best_idx]), 4),
                    "f1": round(float(f1_scores[best_idx]), 4),
                    "threshold": round(float(thresholds[best_idx]), 4),
                },
            })
            prevalence = float(np.mean(y_arr == classes[1]))
        else:
            y_bin = label_binarize(y_arr, classes=classes)
            prevalence = None
            for i in range(n_classes):
                prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
                ap = average_precision_score(y_bin[:, i], y_proba[:, i])
                curves.append({
                    "precision": _downsample(prec),
                    "recall": _downsample(rec),
                    "ap": round(float(ap), 4),
                    "label": f"{class_names[i]} (AP = {ap:.3f})",
                })

        kwargs.setdefault("prevalence", prevalence)
        return cls(curves, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_score: Any,
        *,
        label: str = "Model",
        **kwargs,
    ) -> PRCurveVisualizer:
        """Create PR curve from binary predictions.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_score : array-like
            Predicted probabilities for the positive class.
        label : str, default='Model'
            Curve label.
        **kwargs
            Additional keyword arguments.
        """
        from sklearn.metrics import average_precision_score, precision_recall_curve

        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        prec, rec, thresholds = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)

        f1_scores = 2 * prec * rec / (prec + rec + 1e-10)
        best_idx = int(np.argmax(f1_scores[:-1]))

        prevalence = float(np.mean(y_true))

        curves = [{
            "precision": _downsample(prec),
            "recall": _downsample(rec),
            "ap": round(float(ap), 4),
            "label": f"{label} (AP = {ap:.3f})",
            "optimalPoint": {
                "precision": round(float(prec[best_idx]), 4),
                "recall": round(float(rec[best_idx]), 4),
                "f1": round(float(f1_scores[best_idx]), 4),
                "threshold": round(float(thresholds[best_idx]), 4),
            },
        }]
        kwargs.setdefault("prevalence", prevalence)
        return cls(curves, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "curves": self._curves,
            "prevalence": self.prevalence,
        }

    def _chart_type(self) -> str:
        return "pr_curve"

    def _get_chart_js(self) -> str:
        return _PR_JS


def _downsample(arr, max_points: int = 500) -> list[float]:
    """Downsample an array for rendering efficiency."""
    arr = np.asarray(arr)
    if len(arr) <= max_points:
        return [round(float(v), 6) for v in arr]
    idx = np.linspace(0, len(arr) - 1, max_points, dtype=int)
    return [round(float(arr[i]), 6) for i in idx]


_PR_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 55, left: 55};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const xScale = EG.scaleLinear([0, 1], [0, W]);
  const yScale = EG.scaleLinear([0, 1], [H, 0]);

  EG.drawXAxis(g, xScale, H, 'Recall');
  EG.drawYAxis(g, yScale, W, 'Precision');

  // Prevalence baseline (no-skill classifier)
  if (data.prevalence !== null && data.prevalence !== undefined) {
    const py = yScale(data.prevalence);
    g.appendChild(EG.svg('line', {
      x1: 0, y1: py, x2: W, y2: py,
      stroke: 'var(--text-muted)', 'stroke-width': 1.5,
      'stroke-dasharray': '6,4', opacity: 0.5
    }));
    g.appendChild(EG.svg('text', {
      x: W - 5, y: py - 6, 'text-anchor': 'end',
      fill: 'var(--text-muted)', 'font-size': '10px'
    })).textContent = 'No-skill (' + EG.fmt(data.prevalence, 2) + ')';
  }

  // Draw curves
  data.curves.forEach(function(c, ci) {
    const color = palette[ci % palette.length];
    const n = Math.min(c.precision.length, c.recall.length);
    let d = '';
    for (let i = 0; i < n; i++) {
      d += (i === 0 ? 'M' : ' L') + xScale(c.recall[i]) + ' ' + yScale(c.precision[i]);
    }

    // Fill under curve
    let fillD = d + ' L' + xScale(c.recall[n-1]) + ' ' + H + ' L' + xScale(c.recall[0]) + ' ' + H + ' Z';
    g.appendChild(EG.svg('path', {d: fillD, fill: color, opacity: 0.06}));

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

    // F1-optimal point
    if (c.optimalPoint) {
      const op = c.optimalPoint;
      const px = xScale(op.recall), py2 = yScale(op.precision);
      const marker = EG.svg('circle', {
        cx: px, cy: py2, r: 6,
        fill: 'none', stroke: color, 'stroke-width': 2.5
      });
      marker.addEventListener('mouseenter', function(e) {
        marker.setAttribute('r', '8');
        EG.tooltip.show(e,
          '<b>Best F1 Point</b><br>' +
          'Precision: ' + EG.fmt(op.precision, 3) + '<br>' +
          'Recall: ' + EG.fmt(op.recall, 3) + '<br>' +
          'F1: ' + EG.fmt(op.f1, 3) + '<br>' +
          'Threshold: ' + EG.fmt(op.threshold, 3));
      });
      marker.addEventListener('mouseleave', function() { marker.setAttribute('r', '6'); EG.tooltip.hide(); });
      g.appendChild(marker);
    }
  });

  // Legend
  const items = data.curves.map(function(c, i) {
    return {label: c.label, color: palette[i % palette.length]};
  });
  EG.drawLegend(container, items);
}
"""
