"""Lollipop chart visualizer.

A cleaner alternative to bar charts for displaying ranked values, such as
feature importances. Each value is shown as a dot on a stem line, reducing
visual clutter while preserving precision.

Example
-------
>>> from endgame.visualization import LollipopChartVisualizer
>>> viz = LollipopChartVisualizer.from_importances(model, feature_names)
>>> viz.save("lollipop.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class LollipopChartVisualizer(BaseVisualizer):
    """Interactive lollipop chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float
        Values for each category.
    orientation : str, default='horizontal'
        'horizontal' (labels on Y) or 'vertical' (labels on X).
    sort : bool, default=False
        Sort by value (descending).
    highlight_top : int, optional
        Number of top items to highlight with larger dots.
    baseline : float, default=0
        Baseline value where stems start.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=750
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        orientation: str = "horizontal",
        sort: bool = False,
        highlight_top: int | None = None,
        baseline: float = 0,
        title: str = "",
        palette: str = "tableau",
        width: int = 750,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Lollipop Chart", palette=palette, width=width, height=height, theme=theme)

        labs = list(labels)
        vals = [float(v) for v in values]

        if sort:
            order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
            labs = [labs[i] for i in order]
            vals = [vals[i] for i in order]

        self._labels = labs
        self._values = vals
        self.orientation = orientation
        self.highlight_top = highlight_top
        self.baseline = baseline

    @classmethod
    def from_importances(
        cls,
        model: Any,
        *,
        feature_names: Sequence[str] | None = None,
        top_n: int | None = None,
        **kwargs,
    ) -> LollipopChartVisualizer:
        """Create from model feature importances.

        Parameters
        ----------
        model : estimator
            Fitted model with ``feature_importances_``.
        feature_names : list of str, optional
            Feature names.
        top_n : int, optional
            Show only top N features.
        **kwargs
            Additional keyword arguments.
        """
        imp = np.asarray(model.feature_importances_)
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(len(imp))]

        if top_n is not None:
            idx = np.argsort(imp)[::-1][:top_n]
            imp = imp[idx]
            feature_names = [feature_names[i] for i in idx]

        kwargs.setdefault("sort", True)
        kwargs.setdefault("title", "Feature Importances")
        kwargs.setdefault("highlight_top", 3)
        return cls(list(feature_names), imp.tolist(), **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, float], **kwargs) -> LollipopChartVisualizer:
        """Create from a dictionary.

        Parameters
        ----------
        data : dict of str → float
            Label → value pairs.
        **kwargs
            Additional keyword arguments.
        """
        return cls(list(data.keys()), list(data.values()), **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "labels": self._labels,
            "values": self._values,
            "orientation": self.orientation,
            "highlightTop": self.highlight_top,
            "baseline": self.baseline,
        }

    def _chart_type(self) -> str:
        return "lollipop"

    def _get_chart_js(self) -> str:
        return _LOLLIPOP_JS


_LOLLIPOP_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const palette = config.palette;
  const labels = data.labels;
  const values = data.values;
  const n = labels.length;
  const horiz = data.orientation === 'horizontal';
  const baseline = data.baseline;
  const hlTop = data.highlightTop || 0;

  if (horiz) {
    const margin = {top: 15, right: 30, bottom: 45, left: 140};
    const ctx = EG.createSVG(container, config.width, config.height, margin);
    const {g, width: W, height: H} = ctx;

    let vMin = Math.min(baseline, Math.min.apply(null, values));
    let vMax = Math.max(baseline, Math.max.apply(null, values));
    const pad = (vMax - vMin) * 0.08 || 0.01;
    vMin -= pad; vMax += pad;

    const xScale = EG.scaleLinear([vMin, vMax], [0, W]);
    const bx = xScale(baseline);
    const rowH = H / n;

    EG.drawXAxis(g, xScale, H, 'Value');

    // Baseline line
    g.appendChild(EG.svg('line', {x1: bx, y1: 0, x2: bx, y2: H, stroke: 'var(--text-muted)', 'stroke-width': 1, 'stroke-dasharray': '4,3', opacity: 0.5}));

    for (let i = 0; i < n; i++) {
      const y = i * rowH + rowH / 2;
      const vx = xScale(values[i]);
      const color = palette[i % palette.length];
      const isHighlighted = i < hlTop;
      const dotR = isHighlighted ? 7 : 5;

      // Stem
      g.appendChild(EG.svg('line', {
        x1: bx, y1: y, x2: vx, y2: y,
        stroke: color, 'stroke-width': 2
      }));

      // Dot
      const dot = EG.svg('circle', {
        cx: vx, cy: y, r: dotR,
        fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2
      });
      dot.addEventListener('mouseenter', function(e) {
        dot.setAttribute('r', String(dotR + 2));
        EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>Value: ' + EG.fmt(values[i], 4));
      });
      dot.addEventListener('mouseleave', function() {
        dot.setAttribute('r', String(dotR));
        EG.tooltip.hide();
      });
      g.appendChild(dot);

      // Label
      g.appendChild(EG.svg('text', {
        x: -8, y: y + 4, 'text-anchor': 'end',
        fill: 'var(--text-primary)', 'font-size': '11px',
        'font-weight': isHighlighted ? '600' : '400'
      })).textContent = labels[i].length > 20 ? labels[i].slice(0, 18) + '…' : labels[i];
    }
  } else {
    // Vertical orientation
    const margin = {top: 15, right: 20, bottom: 65, left: 55};
    const ctx = EG.createSVG(container, config.width, config.height, margin);
    const {g, width: W, height: H} = ctx;

    let vMin = Math.min(baseline, Math.min.apply(null, values));
    let vMax = Math.max(baseline, Math.max.apply(null, values));
    const pad = (vMax - vMin) * 0.08 || 0.01;
    vMin -= pad; vMax += pad;

    const yScale = EG.scaleLinear([vMin, vMax], [H, 0]);
    const by = yScale(baseline);
    const colW = W / n;

    EG.drawYAxis(g, yScale, W, 'Value');

    // Baseline line
    g.appendChild(EG.svg('line', {x1: 0, y1: by, x2: W, y2: by, stroke: 'var(--text-muted)', 'stroke-width': 1, 'stroke-dasharray': '4,3', opacity: 0.5}));

    for (let i = 0; i < n; i++) {
      const x = i * colW + colW / 2;
      const vy = yScale(values[i]);
      const color = palette[i % palette.length];
      const isHighlighted = i < hlTop;
      const dotR = isHighlighted ? 7 : 5;

      g.appendChild(EG.svg('line', {x1: x, y1: by, x2: x, y2: vy, stroke: color, 'stroke-width': 2}));

      const dot = EG.svg('circle', {cx: x, cy: vy, r: dotR, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2});
      dot.addEventListener('mouseenter', function(e) {
        dot.setAttribute('r', String(dotR + 2));
        EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>Value: ' + EG.fmt(values[i], 4));
      });
      dot.addEventListener('mouseleave', function() { dot.setAttribute('r', String(dotR)); EG.tooltip.hide(); });
      g.appendChild(dot);

      // Rotated label
      var txt = EG.svg('text', {x: x, y: H + 12, 'text-anchor': 'end', fill: 'var(--text-secondary)', 'font-size': '10px', transform: 'rotate(-35,' + x + ',' + (H + 12) + ')'});
      txt.textContent = labels[i].length > 15 ? labels[i].slice(0, 13) + '…' : labels[i];
      g.appendChild(txt);
    }
  }
}
"""
