"""Radar chart visualizer.

Interactive radar (spider) charts for multi-metric model comparison.

Example
-------
>>> from endgame.visualization import RadarChartVisualizer
>>> viz = RadarChartVisualizer(
...     dimensions=["Accuracy", "Precision", "Recall", "F1", "AUC"],
...     series={"ModelA": [0.92, 0.88, 0.95, 0.91, 0.96],
...             "ModelB": [0.89, 0.91, 0.87, 0.89, 0.93]},
... )
>>> viz.save("radar.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class RadarChartVisualizer(BaseVisualizer):
    """Interactive radar chart visualizer.

    Parameters
    ----------
    dimensions : list of str
        Axis labels.
    series : dict of str → list of float
        Mapping of series name to values (one per dimension).
    ranges : list of tuple of float, optional
        (min, max) for each dimension. If None, auto-computed.
    fill : bool, default=True
        Fill polygon area.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=600
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        dimensions: Sequence[str],
        series: dict[str, Sequence[float]],
        *,
        ranges: Sequence[tuple] | None = None,
        fill: bool = True,
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.dimensions = list(dimensions)
        self.series = {k: list(v) for k, v in series.items()}
        self.ranges = list(ranges) if ranges else None
        self.fill = fill

    def _build_data(self) -> dict[str, Any]:
        dims = self.dimensions
        n_dims = len(dims)
        all_vals = []
        for v in self.series.values():
            all_vals.extend(v[:n_dims])

        if self.ranges:
            axis_ranges = [(float(lo), float(hi)) for lo, hi in self.ranges]
        else:
            lo = min(all_vals) if all_vals else 0
            hi = max(all_vals) if all_vals else 1
            margin = (hi - lo) * 0.05
            axis_ranges = [(lo - margin, hi + margin)] * n_dims

        series_list = []
        for name, values in self.series.items():
            # Normalize to [0, 1] per axis
            normalized = []
            for i, v in enumerate(values[:n_dims]):
                rng = axis_ranges[i]
                span = rng[1] - rng[0]
                normalized.append(round((v - rng[0]) / span if span > 0 else 0.5, 6))
            series_list.append({
                "name": name,
                "values": values[:n_dims],
                "normalized": normalized,
            })

        return {
            "dimensions": dims,
            "series": series_list,
            "ranges": axis_ranges,
            "fill": self.fill,
        }

    def _chart_type(self) -> str:
        return "radar"

    def _get_chart_js(self) -> str:
        return _RADAR_JS


_RADAR_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const margin = 60;
  const R = (size - 2 * margin) / 2;
  const cx = size / 2, cy = size / 2;

  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';

  const g = EG.svg('g');
  svg.appendChild(g);

  const dims = data.dimensions;
  const n = dims.length;
  const palette = config.palette;
  const angleStep = (2 * Math.PI) / n;

  function polarToXY(angle, radius) {
    return {
      x: cx + radius * Math.sin(angle),
      y: cy - radius * Math.cos(angle)
    };
  }

  // Grid rings
  const nRings = 5;
  for (let r = 1; r <= nRings; r++) {
    const radius = R * r / nRings;
    let d = '';
    for (let i = 0; i <= n; i++) {
      const p = polarToXY(i * angleStep, radius);
      d += (i === 0 ? 'M' : ' L') + p.x + ' ' + p.y;
    }
    g.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: 'var(--grid-line)', 'stroke-width': 1}));
  }

  // Axis lines and labels
  for (let i = 0; i < n; i++) {
    const angle = i * angleStep;
    const p = polarToXY(angle, R);
    g.appendChild(EG.svg('line', {x1: cx, y1: cy, x2: p.x, y2: p.y, stroke: 'var(--border)', 'stroke-width': 1}));

    // Label
    const lp = polarToXY(angle, R + 18);
    const anchor = Math.abs(lp.x - cx) < 5 ? 'middle' : (lp.x > cx ? 'start' : 'end');
    const label = EG.svg('text', {
      x: lp.x, y: lp.y + 4, 'text-anchor': anchor,
      fill: 'var(--text-secondary)', 'font-size': '11px', 'font-weight': '500'
    });
    label.textContent = dims[i];
    g.appendChild(label);
  }

  // Draw series
  data.series.forEach(function(s, si) {
    const color = palette[si % palette.length];
    let d = '';
    const points = [];
    for (let i = 0; i < n; i++) {
      const angle = i * angleStep;
      const radius = R * s.normalized[i];
      const p = polarToXY(angle, radius);
      points.push(p);
      d += (i === 0 ? 'M' : ' L') + p.x + ' ' + p.y;
    }
    d += ' Z';

    // Fill
    if (data.fill) {
      g.appendChild(EG.svg('path', {d: d, fill: color, opacity: 0.15, stroke: 'none'}));
    }
    // Outline
    g.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: color, 'stroke-width': 2.5, 'stroke-linejoin': 'round'}));

    // Points
    points.forEach(function(p, i) {
      const circle = EG.svg('circle', {cx: p.x, cy: p.y, r: 4, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2});
      circle.addEventListener('mouseenter', function(e) {
        circle.setAttribute('r', '6');
        EG.tooltip.show(e, '<b>' + EG.esc(s.name) + '</b><br>' + EG.esc(dims[i]) + ': ' + EG.fmt(s.values[i], 4));
      });
      circle.addEventListener('mouseleave', function() { circle.setAttribute('r', '4'); EG.tooltip.hide(); });
      g.appendChild(circle);
    });
  });

  // Legend
  if (data.series.length > 1) {
    const items = data.series.map(function(s, i) {
      return {label: s.name, color: palette[i % palette.length]};
    });
    EG.drawLegend(container, items);
  }
}
"""
