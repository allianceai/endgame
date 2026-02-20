"""Radial bar chart visualizer.

Interactive radial bar charts where bars extend outward from a center
point. Visually striking alternative to horizontal bar charts when
comparing many categories.

Example
-------
>>> from endgame.visualization import RadialBarVisualizer
>>> viz = RadialBarVisualizer(
...     labels=["LGBM", "XGB", "CatBoost", "RF", "MLP", "SVM"],
...     values=[0.923, 0.918, 0.915, 0.901, 0.890, 0.875],
...     title="Model Scores",
... )
>>> viz.save("radial_bar.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class RadialBarVisualizer(BaseVisualizer):
    """Interactive radial bar chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float
        Bar values.
    inner_radius_ratio : float, default=0.3
        Ratio of inner radius to outer radius.
    sort : bool, default=False
        If True, sort bars by value descending.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=650
        Chart width.
    height : int, default=650
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        inner_radius_ratio: float = 0.3,
        sort: bool = False,
        title: str = "",
        palette: str = "tableau",
        width: int = 650,
        height: int = 650,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.values = [float(v) for v in values]
        self.inner_radius_ratio = inner_radius_ratio
        self.sort = sort

    def _build_data(self) -> dict[str, Any]:
        labels = list(self.labels)
        values = list(self.values)

        if self.sort:
            pairs = sorted(zip(labels, values), key=lambda p: p[1], reverse=True)
            labels = [p[0] for p in pairs]
            values = [p[1] for p in pairs]

        v_max = max(values) if values else 1

        return {
            "labels": labels,
            "values": values,
            "vMax": v_max,
            "innerRatio": self.inner_radius_ratio,
        }

    def _chart_type(self) -> str:
        return "radial_bar"

    def _get_chart_js(self) -> str:
        return _RADIAL_BAR_JS


_RADIAL_BAR_JS = r"""
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
  const labels = data.labels;
  const values = data.values;
  const n = labels.length;
  const vMax = data.vMax || 1;
  if (n === 0) return;

  const angleStep = (2 * Math.PI) / n;
  const barGap = 0.02; // gap between bars in radians

  // Background ring
  svg.appendChild(EG.svg('circle', {
    cx: cx, cy: cy, r: outerR,
    fill: 'none', stroke: 'var(--grid-line)', 'stroke-width': 1
  }));
  svg.appendChild(EG.svg('circle', {
    cx: cx, cy: cy, r: innerR,
    fill: 'var(--bg-secondary)', stroke: 'var(--border)', 'stroke-width': 1
  }));

  // Grid rings
  for (let ring = 1; ring <= 3; ring++) {
    const r = innerR + (outerR - innerR) * ring / 4;
    svg.appendChild(EG.svg('circle', {
      cx: cx, cy: cy, r: r,
      fill: 'none', stroke: 'var(--grid-line)', 'stroke-width': 0.5
    }));
  }

  // Draw bars
  for (let i = 0; i < n; i++) {
    const v = values[i];
    const barR = innerR + (v / vMax) * (outerR - innerR);
    const startAngle = i * angleStep - Math.PI / 2 + barGap;
    const endAngle = (i + 1) * angleStep - Math.PI / 2 - barGap;
    const color = palette[i % palette.length];
    const large = (endAngle - startAngle > Math.PI) ? 1 : 0;

    // Arc from innerR to barR
    const ix1 = cx + innerR * Math.cos(startAngle);
    const iy1 = cy + innerR * Math.sin(startAngle);
    const ix2 = cx + innerR * Math.cos(endAngle);
    const iy2 = cy + innerR * Math.sin(endAngle);
    const ox1 = cx + barR * Math.cos(startAngle);
    const oy1 = cy + barR * Math.sin(startAngle);
    const ox2 = cx + barR * Math.cos(endAngle);
    const oy2 = cy + barR * Math.sin(endAngle);

    const d = [
      'M', ix1, iy1,
      'L', ox1, oy1,
      'A', barR, barR, 0, large, 1, ox2, oy2,
      'L', ix2, iy2,
      'A', innerR, innerR, 0, large, 0, ix1, iy1,
      'Z'
    ].join(' ');

    const bar = EG.svg('path', {d: d, fill: color, opacity: 0.8, stroke: 'var(--bg-card)', 'stroke-width': 1});
    bar.addEventListener('mouseenter', function(e) {
      bar.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>Value: ' + EG.fmt(v, 4));
    });
    bar.addEventListener('mouseleave', function() {
      bar.setAttribute('opacity', '0.8');
      EG.tooltip.hide();
    });
    svg.appendChild(bar);

    // Label
    const midAngle = (startAngle + endAngle) / 2;
    const lR = outerR + 15;
    const lx = cx + lR * Math.cos(midAngle);
    const ly = cy + lR * Math.sin(midAngle);
    const rotate = (midAngle * 180 / Math.PI);
    const flip = midAngle > Math.PI / 2 && midAngle < 3 * Math.PI / 2;
    const anchor = flip ? 'end' : 'start';
    const actualRotate = flip ? rotate + 180 : rotate;

    const label = EG.svg('text', {
      x: lx, y: ly + 3, 'text-anchor': anchor,
      fill: 'var(--text-secondary)', 'font-size': n > 10 ? '9px' : '11px',
      transform: 'rotate(' + actualRotate + ',' + lx + ',' + ly + ')'
    });
    label.textContent = labels[i];
    svg.appendChild(label);
  }
}
"""
