"""Nightingale Rose (polar area) chart visualizer.

Interactive Nightingale Rose charts — a polar area chart where each
sector has equal angle but the radius encodes magnitude. Named after
Florence Nightingale's famous visualization of mortality causes.

Ideal for comparing magnitudes across categories when the number of
categories is moderate (4-12).

Example
-------
>>> from endgame.visualization import NightingaleRoseVisualizer
>>> viz = NightingaleRoseVisualizer(
...     labels=["Acc", "Prec", "Rec", "F1", "AUC", "LogLoss"],
...     values=[0.95, 0.88, 0.92, 0.90, 0.97, 0.35],
...     title="Model Metrics",
... )
>>> viz.save("rose.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class NightingaleRoseVisualizer(BaseVisualizer):
    """Interactive Nightingale Rose chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float or dict of str → list of float
        Values for each category. Pass a dict for multiple series
        (overlaid petals).
    series_names : list of str, optional
        Series names (when values is a dict).
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
        labels: Sequence[str],
        values: Any,
        *,
        series_names: Sequence[str] | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)

        if isinstance(values, dict):
            self._series = {k: [float(v) for v in vs] for k, vs in values.items()}
        else:
            name = (series_names[0] if series_names else "")
            self._series = {name: [float(v) for v in values]}

    @classmethod
    def from_metrics(
        cls,
        metrics: dict[str, float],
        **kwargs,
    ) -> NightingaleRoseVisualizer:
        """Create from a metrics dictionary.

        Parameters
        ----------
        metrics : dict of str → float
            Metric name → value.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Model Metrics")
        return cls(list(metrics.keys()), list(metrics.values()), **kwargs)

    def _build_data(self) -> dict[str, Any]:
        n = len(self.labels)
        series = []
        for name, vals in self._series.items():
            series.append({"name": name, "values": vals[:n]})

        # Global max for scaling
        all_vals = []
        for s in series:
            all_vals.extend(s["values"])
        v_max = max(all_vals) if all_vals else 1

        return {
            "labels": self.labels,
            "series": series,
            "vMax": v_max,
        }

    def _chart_type(self) -> str:
        return "nightingale_rose"

    def _get_chart_js(self) -> str:
        return _ROSE_JS


_ROSE_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';
  const palette = config.palette;
  const cx = size / 2, cy = size / 2;
  const maxR = size / 2 - 55;
  const labels = data.labels;
  const n = labels.length;
  const vMax = data.vMax || 1;
  const angleStep = (2 * Math.PI) / n;
  const nSeries = data.series.length;

  // Grid rings
  const nRings = 4;
  for (let r = 1; r <= nRings; r++) {
    const radius = maxR * r / nRings;
    let d = '';
    for (let i = 0; i <= 60; i++) {
      const a = (i / 60) * Math.PI * 2;
      const px = cx + radius * Math.cos(a);
      const py = cy + radius * Math.sin(a);
      d += (i === 0 ? 'M' : ' L') + px + ' ' + py;
    }
    d += ' Z';
    svg.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: 'var(--grid-line)', 'stroke-width': 1}));
    // Value label on ring
    const val = vMax * r / nRings;
    svg.appendChild(EG.svg('text', {
      x: cx + 4, y: cy - radius + 12,
      fill: 'var(--text-muted)', 'font-size': '9px'
    })).textContent = EG.fmt(val);
  }

  // Axis lines
  for (let i = 0; i < n; i++) {
    const angle = i * angleStep - Math.PI / 2;
    const ex = cx + maxR * Math.cos(angle);
    const ey = cy + maxR * Math.sin(angle);
    svg.appendChild(EG.svg('line', {x1: cx, y1: cy, x2: ex, y2: ey, stroke: 'var(--border)', 'stroke-width': 1}));
  }

  // Draw petals per series
  data.series.forEach(function(s, si) {
    for (let i = 0; i < n; i++) {
      const v = s.values[i] || 0;
      const r = (v / vMax) * maxR;
      const startAngle = i * angleStep - Math.PI / 2;
      const endAngle = startAngle + angleStep;
      const color = nSeries > 1 ? palette[si % palette.length] : palette[i % palette.length];

      const x1 = cx + r * Math.cos(startAngle);
      const y1 = cy + r * Math.sin(startAngle);
      const x2 = cx + r * Math.cos(endAngle);
      const y2 = cy + r * Math.sin(endAngle);
      const large = angleStep > Math.PI ? 1 : 0;

      const d = 'M' + cx + ',' + cy + ' L' + x1 + ',' + y1 +
        ' A' + r + ',' + r + ' 0 ' + large + ',1 ' + x2 + ',' + y2 + ' Z';

      const petal = EG.svg('path', {
        d: d, fill: color, opacity: nSeries > 1 ? 0.5 : 0.7,
        stroke: color, 'stroke-width': 1.5
      });
      petal.addEventListener('mouseenter', function(e) {
        petal.setAttribute('opacity', '0.95');
        const sName = s.name ? '<b>' + EG.esc(s.name) + '</b><br>' : '';
        EG.tooltip.show(e, sName + '<b>' + EG.esc(labels[i]) + '</b>: ' + EG.fmt(v, 4));
      });
      petal.addEventListener('mouseleave', function() {
        petal.setAttribute('opacity', nSeries > 1 ? '0.5' : '0.7');
        EG.tooltip.hide();
      });
      svg.appendChild(petal);
    }
  });

  // Labels
  for (let i = 0; i < n; i++) {
    const angle = i * angleStep + angleStep / 2 - Math.PI / 2;
    const lx = cx + (maxR + 22) * Math.cos(angle);
    const ly = cy + (maxR + 22) * Math.sin(angle);
    const anchor = Math.abs(lx - cx) < 10 ? 'middle' : (lx > cx ? 'start' : 'end');
    const label = EG.svg('text', {
      x: lx, y: ly + 4, 'text-anchor': anchor,
      fill: 'var(--text-secondary)', 'font-size': '11px', 'font-weight': '500'
    });
    label.textContent = labels[i];
    svg.appendChild(label);
  }

  // Legend for multi-series
  if (nSeries > 1) {
    const items = data.series.map(function(s, i) {
      return {label: s.name, color: palette[i % palette.length]};
    });
    EG.drawLegend(container, items);
  }
}
"""
