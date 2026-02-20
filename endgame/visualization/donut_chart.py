"""Donut chart visualizer.

Interactive donut (ring) charts for part-to-whole relationships,
class distributions, and proportion visualization.

Example
-------
>>> from endgame.visualization import DonutChartVisualizer
>>> viz = DonutChartVisualizer(
...     labels=["Class A", "Class B", "Class C"],
...     values=[45, 35, 20],
...     title="Class Distribution",
... )
>>> viz.save("donut.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class DonutChartVisualizer(BaseVisualizer):
    """Interactive donut chart visualizer.

    Parameters
    ----------
    labels : list of str
        Slice labels.
    values : list of float
        Slice values (proportions are computed automatically).
    inner_radius_ratio : float, default=0.55
        Ratio of inner to outer radius (0 = pie, ~0.6 = donut).
    center_text : str, optional
        Text displayed in the center hole.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=600
        Chart width.
    height : int, default=550
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        inner_radius_ratio: float = 0.55,
        center_text: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 550,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.values = [float(v) for v in values]
        self.inner_radius_ratio = inner_radius_ratio
        self.center_text = center_text

    @classmethod
    def from_class_distribution(
        cls,
        y: Any,
        *,
        class_names: Sequence[str] | None = None,
        **kwargs,
    ) -> DonutChartVisualizer:
        """Create from a target array's class distribution.

        Parameters
        ----------
        y : array-like
            Target labels.
        class_names : list of str, optional
            Class names.
        **kwargs
            Additional keyword arguments.
        """
        y_arr = np.asarray(y)
        unique, counts = np.unique(y_arr, return_counts=True)
        if class_names is None:
            class_names = [str(c) for c in unique]
        kwargs.setdefault("title", "Class Distribution")
        return cls(class_names, counts.tolist(), **kwargs)

    def _build_data(self) -> dict[str, Any]:
        total = sum(self.values)
        slices = []
        for lbl, val in zip(self.labels, self.values):
            slices.append({
                "label": lbl,
                "value": round(val, 4),
                "pct": round(val / total * 100, 1) if total > 0 else 0,
            })

        return {
            "slices": slices,
            "total": round(total, 4),
            "innerRatio": self.inner_radius_ratio,
            "centerText": self.center_text,
        }

    def _chart_type(self) -> str:
        return "donut"

    def _get_chart_js(self) -> str:
        return _DONUT_JS


_DONUT_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';
  const palette = config.palette;
  const cx = size / 2, cy = size / 2;
  const outerR = size / 2 - 50;
  const innerR = outerR * data.innerRatio;
  const slices = data.slices;
  const total = data.total;
  if (total <= 0 || slices.length === 0) return;

  let angle = -Math.PI / 2;

  slices.forEach(function(sl, i) {
    const span = (sl.value / total) * Math.PI * 2;
    const endAngle = angle + span;
    const color = palette[i % palette.length];
    const large = span > Math.PI ? 1 : 0;

    // Outer arc
    const ox1 = cx + outerR * Math.cos(angle);
    const oy1 = cy + outerR * Math.sin(angle);
    const ox2 = cx + outerR * Math.cos(endAngle);
    const oy2 = cy + outerR * Math.sin(endAngle);
    // Inner arc
    const ix1 = cx + innerR * Math.cos(angle);
    const iy1 = cy + innerR * Math.sin(angle);
    const ix2 = cx + innerR * Math.cos(endAngle);
    const iy2 = cy + innerR * Math.sin(endAngle);

    const d = [
      'M', ox1, oy1,
      'A', outerR, outerR, 0, large, 1, ox2, oy2,
      'L', ix2, iy2,
      'A', innerR, innerR, 0, large, 0, ix1, iy1,
      'Z'
    ].join(' ');

    const path = EG.svg('path', {d: d, fill: color, opacity: 0.85, stroke: 'var(--bg-card)', 'stroke-width': 2});
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('opacity', '1');
      path.setAttribute('transform', function() {
        const mid = angle + span / 2;
        return 'translate(' + Math.cos(mid) * 5 + ',' + Math.sin(mid) * 5 + ')';
      }());
      EG.tooltip.show(e, '<b>' + EG.esc(sl.label) + '</b><br>' + EG.fmt(sl.value) + ' (' + sl.pct + '%)');
    });
    path.addEventListener('mouseleave', function() {
      path.setAttribute('opacity', '0.85');
      path.setAttribute('transform', '');
      EG.tooltip.hide();
    });
    svg.appendChild(path);

    // Label line for larger slices
    if (span > 0.15) {
      const mid = angle + span / 2;
      const lx = cx + (outerR + 15) * Math.cos(mid);
      const ly = cy + (outerR + 15) * Math.sin(mid);
      const anchor = lx > cx ? 'start' : 'end';
      const label = EG.svg('text', {
        x: lx, y: ly + 4, 'text-anchor': anchor,
        fill: 'var(--text-secondary)', 'font-size': '11px'
      });
      label.textContent = sl.label + ' ' + sl.pct + '%';
      svg.appendChild(label);
    }

    angle = endAngle;
  });

  // Center text
  if (data.centerText) {
    const ct = EG.svg('text', {
      x: cx, y: cy + 5, 'text-anchor': 'middle',
      fill: 'var(--text-primary)', 'font-size': '16px', 'font-weight': '700'
    });
    ct.textContent = data.centerText;
    svg.appendChild(ct);
  }
}
"""
