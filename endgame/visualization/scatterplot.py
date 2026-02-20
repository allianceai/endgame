"""Scatterplot and bubble chart visualizer.

Interactive scatterplots for embeddings, actual-vs-predicted plots,
and multi-dimensional data exploration with zoom/pan support.

Example
-------
>>> from endgame.visualization import ScatterplotVisualizer
>>> import numpy as np
>>> x = np.random.randn(200)
>>> y = x + np.random.randn(200) * 0.3
>>> viz = ScatterplotVisualizer(x, y, title="Correlation")
>>> viz.save("scatter.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class ScatterplotVisualizer(BaseVisualizer):
    """Interactive scatterplot visualizer.

    Parameters
    ----------
    x : array-like
        X coordinates.
    y : array-like
        Y coordinates.
    labels : array-like of str, optional
        Point labels for coloring (categorical).
    sizes : array-like of float, optional
        Point sizes for bubble mode.
    colors : array-like of float, optional
        Continuous values for color mapping.
    point_labels : list of str, optional
        Individual point hover labels.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    show_diagonal : bool, default=False
        Show y=x diagonal line.
    show_regression : bool, default=False
        Show linear regression line.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette name.
    width : int, default=700
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        x: Any,
        y: Any,
        *,
        labels: Any | None = None,
        sizes: Any | None = None,
        colors: Any | None = None,
        point_labels: Sequence[str] | None = None,
        x_label: str = "",
        y_label: str = "",
        show_diagonal: bool = False,
        show_regression: bool = False,
        title: str = "",
        palette: str = "tableau",
        width: int = 700,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._x = np.asarray(x, dtype=float).ravel()
        self._y = np.asarray(y, dtype=float).ravel()
        self._labels = np.asarray(labels).ravel() if labels is not None else None
        self._sizes = np.asarray(sizes, dtype=float).ravel() if sizes is not None else None
        self._colors = np.asarray(colors, dtype=float).ravel() if colors is not None else None
        self._point_labels = list(point_labels) if point_labels is not None else None
        self.x_label = x_label
        self.y_label = y_label
        self.show_diagonal = show_diagonal
        self.show_regression = show_regression

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_pred: Any,
        **kwargs,
    ) -> ScatterplotVisualizer:
        """Create an actual-vs-predicted scatter plot.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Actual vs Predicted")
        kwargs.setdefault("x_label", "Actual")
        kwargs.setdefault("y_label", "Predicted")
        kwargs.setdefault("show_diagonal", True)
        kwargs.setdefault("show_regression", True)
        return cls(np.asarray(y_true), np.asarray(y_pred), **kwargs)

    @classmethod
    def from_embedding(
        cls,
        embedding: Any,
        labels: Any | None = None,
        **kwargs,
    ) -> ScatterplotVisualizer:
        """Create a scatter plot from 2D embeddings (t-SNE, UMAP, PCA).

        Parameters
        ----------
        embedding : array-like, shape (n_samples, 2)
            2D embedding coordinates.
        labels : array-like, optional
            Labels for coloring.
        **kwargs
            Additional keyword arguments.
        """
        emb = np.asarray(embedding)
        kwargs.setdefault("title", "2D Embedding")
        kwargs.setdefault("x_label", "Dimension 1")
        kwargs.setdefault("y_label", "Dimension 2")
        return cls(emb[:, 0], emb[:, 1], labels=labels, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        n = len(self._x)
        mask = ~(np.isnan(self._x) | np.isnan(self._y))
        x = self._x[mask].tolist()
        y = self._y[mask].tolist()

        result: dict[str, Any] = {
            "x": x,
            "y": y,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
            "showDiagonal": self.show_diagonal,
            "showRegression": self.show_regression,
        }

        if self._labels is not None:
            labels = self._labels[mask]
            unique = sorted(set(str(l) for l in labels))
            result["labels"] = [str(l) for l in labels]
            result["uniqueLabels"] = unique
        if self._sizes is not None:
            s = self._sizes[mask]
            # Normalize sizes to 3-20px radius
            smin, smax = float(s.min()), float(s.max())
            if smax > smin:
                norm = ((s - smin) / (smax - smin) * 17 + 3).tolist()
            else:
                norm = [6.0] * len(s)
            result["sizes"] = norm
        if self._colors is not None:
            c = self._colors[mask]
            result["colorValues"] = c.tolist()
            result["colorMin"] = float(np.nanmin(c))
            result["colorMax"] = float(np.nanmax(c))
        if self._point_labels is not None:
            result["pointLabels"] = [self._point_labels[i] for i in range(n) if mask[i]]

        # Regression line
        if self.show_regression and len(x) >= 2:
            xa, ya = np.array(x), np.array(y)
            slope, intercept = np.polyfit(xa, ya, 1)
            r2 = 1 - np.sum((ya - (slope * xa + intercept))**2) / np.sum((ya - ya.mean())**2) if ya.std() > 0 else 0
            result["regression"] = {"slope": round(float(slope), 6), "intercept": round(float(intercept), 6), "r2": round(float(r2), 4)}

        return result

    def _chart_type(self) -> str:
        return "scatter"

    def _get_chart_js(self) -> str:
        return _SCATTER_JS


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_SCATTER_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 50, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const x = data.x, y = data.y;
  if (x.length === 0) return;

  // Scales
  let xMin = Math.min.apply(null, x), xMax = Math.max.apply(null, x);
  let yMin = Math.min.apply(null, y), yMax = Math.max.apply(null, y);
  const xPad = (xMax - xMin) * 0.05 || 1;
  const yPad = (yMax - yMin) * 0.05 || 1;
  xMin -= xPad; xMax += xPad; yMin -= yPad; yMax += yPad;

  const xScale = EG.scaleLinear([xMin, xMax], [0, W]);
  const yScale = EG.scaleLinear([yMin, yMax], [H, 0]);

  EG.drawXAxis(g, xScale, H, data.xLabel);
  EG.drawYAxis(g, yScale, W, data.yLabel);

  // Diagonal line
  if (data.showDiagonal) {
    const dMin = Math.max(xMin, yMin), dMax = Math.min(xMax, yMax);
    g.appendChild(EG.svg('line', {
      x1: xScale(dMin), y1: yScale(dMin), x2: xScale(dMax), y2: yScale(dMax),
      stroke: 'var(--text-muted)', 'stroke-width': 1, 'stroke-dasharray': '6,4', opacity: 0.6
    }));
  }

  // Regression line
  if (data.regression) {
    const r = data.regression;
    const rx1 = xMin, rx2 = xMax;
    const ry1 = r.slope * rx1 + r.intercept;
    const ry2 = r.slope * rx2 + r.intercept;
    g.appendChild(EG.svg('line', {
      x1: xScale(rx1), y1: yScale(ry1), x2: xScale(rx2), y2: yScale(ry2),
      stroke: 'var(--accent)', 'stroke-width': 2, opacity: 0.7
    }));
    const regLabel = EG.svg('text', {
      x: W - 5, y: 15, 'text-anchor': 'end',
      fill: 'var(--accent)', 'font-size': '11px'
    });
    regLabel.textContent = 'R² = ' + r.r2.toFixed(3);
    g.appendChild(regLabel);
  }

  // Color setup
  let colorFn;
  if (data.uniqueLabels) {
    const labelMap = {};
    data.uniqueLabels.forEach(function(l, i) { labelMap[l] = palette[i % palette.length]; });
    colorFn = function(i) { return labelMap[data.labels[i]]; };
  } else if (data.colorValues) {
    const cs = EG.colorScale(palette, data.colorMin, data.colorMax);
    colorFn = function(i) { return cs(data.colorValues[i]); };
  } else {
    colorFn = function() { return palette[0]; };
  }

  // Points
  for (let i = 0; i < x.length; i++) {
    const cx = xScale(x[i]), cy = yScale(y[i]);
    const r = data.sizes ? data.sizes[i] : 4;
    const color = colorFn(i);
    const circle = EG.svg('circle', {cx: cx, cy: cy, r: r, fill: color, opacity: 0.75, stroke: 'none'});
    circle.addEventListener('mouseenter', function(e) {
      circle.setAttribute('opacity', '1');
      circle.setAttribute('stroke', 'var(--text-primary)');
      circle.setAttribute('stroke-width', '1.5');
      let html = '<b>x:</b> ' + EG.fmt(x[i], 4) + '<br><b>y:</b> ' + EG.fmt(y[i], 4);
      if (data.labels) html = '<b>' + EG.esc(data.labels[i]) + '</b><br>' + html;
      if (data.pointLabels) html = '<b>' + EG.esc(data.pointLabels[i]) + '</b><br>' + html;
      EG.tooltip.show(e, html);
    });
    circle.addEventListener('mouseleave', function() {
      circle.setAttribute('opacity', '0.75');
      circle.removeAttribute('stroke');
      circle.removeAttribute('stroke-width');
      EG.tooltip.hide();
    });
    g.appendChild(circle);
  }

  // Legend for categorical labels
  if (data.uniqueLabels && data.uniqueLabels.length <= 20) {
    const items = data.uniqueLabels.map(function(l, i) {
      return {label: l, color: palette[i % palette.length]};
    });
    EG.drawLegend(container, items);
  }
}
"""
