"""Error bars visualizer.

Interactive model comparison chart with confidence intervals,
ideal for comparing multiple models with mean + CI bars.

Example
-------
>>> from endgame.visualization import ErrorBarsVisualizer
>>> viz = ErrorBarsVisualizer(
...     labels=["LGBM", "XGB", "CatBoost"],
...     means=[0.912, 0.908, 0.915],
...     errors=[0.003, 0.005, 0.004],
...     title="Model Comparison",
... )
>>> viz.save("error_bars.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class ErrorBarsVisualizer(BaseVisualizer):
    """Interactive error bar chart for model comparison.

    Parameters
    ----------
    labels : list of str
        Model/category names.
    means : list of float
        Mean values.
    errors : list of float or list of tuple of float
        Symmetric errors (single value) or asymmetric (lo, hi) per point.
    orientation : str, default='horizontal'
        'horizontal' or 'vertical'.
    sort : bool, default=True
        Sort by mean value.
    x_label : str, optional
        Value axis label.
    y_label : str, optional
        Category axis label.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=800
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        means: Sequence[float],
        errors: Sequence,
        *,
        orientation: str = "horizontal",
        sort: bool = True,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.means = list(means)
        self.orientation = orientation
        self.sort = sort
        self.x_label = x_label
        self.y_label = y_label

        # Normalize errors to (lo, hi) tuples
        self._errors = []
        for e in errors:
            if isinstance(e, (list, tuple)):
                self._errors.append((float(e[0]), float(e[1])))
            else:
                self._errors.append((float(e), float(e)))

    @classmethod
    def from_cv_results(
        cls,
        results: dict[str, Sequence[float]],
        **kwargs,
    ) -> ErrorBarsVisualizer:
        """Create from CV results (mean +/- std).

        Parameters
        ----------
        results : dict of str → list of float
            Model name → fold scores.
        **kwargs
            Additional keyword arguments.
        """
        labels, means, errors = [], [], []
        for name, scores in results.items():
            arr = np.asarray(scores, dtype=float)
            labels.append(name)
            means.append(float(arr.mean()))
            errors.append(float(arr.std()))

        kwargs.setdefault("title", "Model Comparison (mean ± std)")
        kwargs.setdefault("x_label", "Score")
        return cls(labels, means, errors, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        labels = list(self.labels)
        means = list(self.means)
        errors = list(self._errors)

        if self.sort:
            indices = sorted(range(len(means)), key=lambda i: means[i])
            labels = [labels[i] for i in indices]
            means = [means[i] for i in indices]
            errors = [errors[i] for i in indices]

        return {
            "labels": labels,
            "means": means,
            "errors": errors,
            "orientation": self.orientation,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
        }

    def _chart_type(self) -> str:
        return "error_bars"

    def _get_chart_js(self) -> str:
        return _ERROR_BARS_JS


_ERROR_BARS_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const isHoriz = data.orientation === 'horizontal';
  const margin = isHoriz
    ? {top: 20, right: 30, bottom: 50, left: 120}
    : {top: 20, right: 20, bottom: 60, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const labels = data.labels;
  const means = data.means;
  const errors = data.errors;
  const n = labels.length;

  // Value range
  let vMin = Infinity, vMax = -Infinity;
  for (let i = 0; i < n; i++) {
    const lo = means[i] - errors[i][0];
    const hi = means[i] + errors[i][1];
    if (lo < vMin) vMin = lo;
    if (hi > vMax) vMax = hi;
  }
  const pad = (vMax - vMin) * 0.1 || 0.01;
  vMin -= pad; vMax += pad;

  const catScale = EG.scaleBand(labels, [0, isHoriz ? H : W], 0.25);
  const valScale = EG.scaleLinear([vMin, vMax], isHoriz ? [0, W] : [H, 0]);
  const bw = catScale.bandwidth();

  if (isHoriz) {
    EG.drawXAxis(g, valScale, H, data.xLabel);
    EG.drawYAxis(g, catScale, W, data.yLabel, true);
  } else {
    EG.drawXAxis(g, catScale, H, data.xLabel, true);
    EG.drawYAxis(g, valScale, W, data.yLabel);
  }

  for (let i = 0; i < n; i++) {
    const color = palette[i % palette.length];
    const m = means[i];
    const eLo = errors[i][0], eHi = errors[i][1];

    if (isHoriz) {
      const cy = catScale(labels[i]) + bw / 2;
      const xLo = valScale(m - eLo), xHi = valScale(m + eHi), xM = valScale(m);
      // Error bar line
      g.appendChild(EG.svg('line', {x1: xLo, y1: cy, x2: xHi, y2: cy, stroke: color, 'stroke-width': 2}));
      // Caps
      const capH = bw * 0.25;
      g.appendChild(EG.svg('line', {x1: xLo, y1: cy-capH, x2: xLo, y2: cy+capH, stroke: color, 'stroke-width': 2}));
      g.appendChild(EG.svg('line', {x1: xHi, y1: cy-capH, x2: xHi, y2: cy+capH, stroke: color, 'stroke-width': 2}));
      // Mean dot
      const dot = EG.svg('circle', {cx: xM, cy: cy, r: 6, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2});
      dot.addEventListener('mouseenter', function(e) {
        EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>Mean: ' + EG.fmt(m, 4) +
          '<br>CI: [' + EG.fmt(m - eLo, 4) + ', ' + EG.fmt(m + eHi, 4) + ']');
      });
      dot.addEventListener('mouseleave', function() { EG.tooltip.hide(); });
      g.appendChild(dot);
    } else {
      const cx = catScale(labels[i]) + bw / 2;
      const yLo = valScale(m - eLo), yHi = valScale(m + eHi), yM = valScale(m);
      g.appendChild(EG.svg('line', {x1: cx, y1: yLo, x2: cx, y2: yHi, stroke: color, 'stroke-width': 2}));
      const capW = bw * 0.25;
      g.appendChild(EG.svg('line', {x1: cx-capW, y1: yLo, x2: cx+capW, y2: yLo, stroke: color, 'stroke-width': 2}));
      g.appendChild(EG.svg('line', {x1: cx-capW, y1: yHi, x2: cx+capW, y2: yHi, stroke: color, 'stroke-width': 2}));
      const dot = EG.svg('circle', {cx: cx, cy: yM, r: 6, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2});
      dot.addEventListener('mouseenter', function(e) {
        EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>Mean: ' + EG.fmt(m, 4) +
          '<br>CI: [' + EG.fmt(m - eLo, 4) + ', ' + EG.fmt(m + eHi, 4) + ']');
      });
      dot.addEventListener('mouseleave', function() { EG.tooltip.hide(); });
      g.appendChild(dot);
    }
  }
}
"""
