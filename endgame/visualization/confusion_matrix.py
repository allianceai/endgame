"""Confusion matrix visualizer.

Interactive confusion matrix for classification evaluation with
normalized/raw views and per-class metrics.

Example
-------
>>> from endgame.visualization import ConfusionMatrixVisualizer
>>> viz = ConfusionMatrixVisualizer(
...     matrix=[[50, 3], [7, 40]],
...     class_names=["negative", "positive"],
... )
>>> viz.save("confusion_matrix.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer
from endgame.visualization._palettes import DEFAULT_SEQUENTIAL


class ConfusionMatrixVisualizer(BaseVisualizer):
    """Interactive confusion matrix visualizer.

    Parameters
    ----------
    matrix : array-like, shape (n_classes, n_classes)
        Confusion matrix (rows = true, columns = predicted).
    class_names : list of str, optional
        Class label names.
    normalize : bool, default=False
        If True, show row-normalized values (recall per class).
    title : str, optional
        Chart title.
    cmap : str, default='blues'
        Color palette for the matrix cells.
    width : int, default=650
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        matrix: Any,
        *,
        class_names: Sequence[str] | None = None,
        normalize: bool = False,
        title: str = "",
        cmap: str = DEFAULT_SEQUENTIAL,
        width: int = 650,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Confusion Matrix", palette=cmap, width=width, height=height, theme=theme)
        self._matrix = np.asarray(matrix, dtype=float)
        if self._matrix.ndim != 2 or self._matrix.shape[0] != self._matrix.shape[1]:
            raise ValueError("matrix must be square 2D array")

        n = self._matrix.shape[0]
        self.class_names = list(class_names) if class_names else [str(i) for i in range(n)]
        self.normalize = normalize

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
    ) -> ConfusionMatrixVisualizer:
        """Create a confusion matrix from a fitted classifier.

        Parameters
        ----------
        model : estimator
            Fitted sklearn-compatible classifier.
        X : array-like
            Features.
        y : array-like
            True labels.
        class_names : list of str, optional
            Class names.
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        from sklearn.metrics import confusion_matrix
        y_pred = model.predict(X)
        cm = confusion_matrix(y, y_pred)
        if class_names is None and hasattr(model, "classes_"):
            class_names = [str(c) for c in model.classes_]
        return cls(cm, class_names=class_names, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_pred: Any,
        *,
        class_names: Sequence[str] | None = None,
        **kwargs,
    ) -> ConfusionMatrixVisualizer:
        """Create a confusion matrix from predictions.

        Parameters
        ----------
        y_true : array-like
            True labels.
        y_pred : array-like
            Predicted labels.
        class_names : list of str, optional
            Class names.
        **kwargs
            Additional keyword arguments.
        """
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        return cls(cm, class_names=class_names, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        cm = self._matrix
        n = cm.shape[0]

        # Row-normalized version
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_norm = cm / row_sums

        # Per-class metrics
        precision = np.zeros(n)
        recall = np.zeros(n)
        f1 = np.zeros(n)
        for i in range(n):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() > 0 else 0

        return {
            "matrix": cm.tolist(),
            "matrixNorm": [[round(float(v), 4) for v in row] for row in cm_norm],
            "classNames": self.class_names,
            "normalize": self.normalize,
            "precision": [round(float(v), 4) for v in precision],
            "recall": [round(float(v), 4) for v in recall],
            "f1": [round(float(v), 4) for v in f1],
            "accuracy": round(accuracy, 4),
        }

    def _chart_type(self) -> str:
        return "confusion_matrix"

    def _get_chart_js(self) -> str:
        return _CM_JS


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_CM_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const n = data.classNames.length;
  const cellSize = Math.min(70, Math.floor((config.width - 180) / n), Math.floor((config.height - 140) / n));
  const margin = {top: 30, right: 30, bottom: 70, left: 100};

  const plotW = n * cellSize;
  const plotH = n * cellSize;
  const totalW = plotW + margin.left + margin.right;
  const totalH = plotH + margin.top + margin.bottom;

  const svg = EG.svg('svg', {width: totalW, height: totalH});
  container.appendChild(svg);
  container.style.width = totalW + 'px';
  container.style.height = totalH + 'px';

  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);

  const palette = config.palette;
  const cm = data.normalize ? data.matrixNorm : data.matrix;
  const cmRaw = data.matrix;
  const cmNorm = data.matrixNorm;

  // Find max for color scaling
  let maxVal = 0;
  for (let r = 0; r < n; r++)
    for (let c = 0; c < n; c++)
      if (cm[r][c] > maxVal) maxVal = cm[r][c];

  const colorFn = EG.colorScale(palette, 0, maxVal || 1);

  // Draw cells
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      const v = cm[r][c];
      const raw = cmRaw[r][c];
      const norm = cmNorm[r][c];
      const x = c * cellSize;
      const y = r * cellSize;
      const isDiag = r === c;

      const rect = EG.svg('rect', {
        x: x, y: y, width: cellSize - 2, height: cellSize - 2,
        fill: colorFn(v), rx: 3,
        stroke: isDiag ? 'var(--accent)' : 'none',
        'stroke-width': isDiag ? '2' : '0'
      });

      rect.addEventListener('mouseenter', function(e) {
        rect.setAttribute('stroke', 'var(--text-primary)');
        rect.setAttribute('stroke-width', '2');
        EG.tooltip.show(e,
          '<b>True: ' + EG.esc(data.classNames[r]) + '</b><br>' +
          '<b>Pred: ' + EG.esc(data.classNames[c]) + '</b><br>' +
          'Count: ' + Math.round(raw) + '<br>' +
          'Rate: ' + EG.pct(norm)
        );
      });
      rect.addEventListener('mousemove', function(e) {
        EG.tooltip.show(e,
          '<b>True: ' + EG.esc(data.classNames[r]) + '</b><br>' +
          '<b>Pred: ' + EG.esc(data.classNames[c]) + '</b><br>' +
          'Count: ' + Math.round(raw) + '<br>' +
          'Rate: ' + EG.pct(norm)
        );
      });
      rect.addEventListener('mouseleave', function() {
        rect.setAttribute('stroke', isDiag ? 'var(--accent)' : 'none');
        rect.setAttribute('stroke-width', isDiag ? '2' : '0');
        EG.tooltip.hide();
      });
      g.appendChild(rect);

      // Annotation
      const brightness = v / (maxVal || 1);
      const textColor = brightness > 0.5 ? '#1a1a2e' : '#e0e0e0';
      const valStr = data.normalize ? EG.pct(v) : Math.round(v).toString();
      const txt = EG.svg('text', {
        x: x + cellSize / 2, y: y + cellSize / 2 + 5,
        'text-anchor': 'middle', fill: textColor,
        'font-size': cellSize < 40 ? '10px' : '13px',
        'font-weight': isDiag ? '700' : '500'
      });
      txt.textContent = valStr;
      g.appendChild(txt);
    }
  }

  // Y labels (True)
  for (let r = 0; r < n; r++) {
    const lbl = EG.svg('text', {
      x: -10, y: r * cellSize + cellSize / 2 + 4,
      'text-anchor': 'end', fill: 'var(--text-secondary)', 'font-size': '11px'
    });
    lbl.textContent = data.classNames[r];
    g.appendChild(lbl);
  }
  g.appendChild(EG.svg('text', {
    x: -10, y: -10, 'text-anchor': 'end',
    fill: 'var(--text-muted)', 'font-size': '10px', 'font-style': 'italic'
  })).textContent = 'True ↓';

  // X labels (Predicted)
  for (let c = 0; c < n; c++) {
    const lbl = EG.svg('text', {
      x: c * cellSize + cellSize / 2, y: plotH + 20,
      'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '11px'
    });
    lbl.textContent = data.classNames[c];
    g.appendChild(lbl);
  }
  g.appendChild(EG.svg('text', {
    x: plotW / 2, y: plotH + 40, 'text-anchor': 'middle',
    fill: 'var(--text-muted)', 'font-size': '10px', 'font-style': 'italic'
  })).textContent = 'Predicted →';

  // Accuracy badge
  g.appendChild(EG.svg('text', {
    x: plotW / 2, y: plotH + 58, 'text-anchor': 'middle',
    fill: 'var(--accent)', 'font-size': '12px', 'font-weight': '600'
  })).textContent = 'Accuracy: ' + EG.pct(data.accuracy);
}
"""
