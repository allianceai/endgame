"""Spiral plot visualizer.

Interactive spiral plots for time-series data, periodic patterns,
and sequential data visualization along an Archimedean spiral.

Particularly useful for showing cyclical patterns in data (daily,
weekly, seasonal) or for displaying long sequences compactly.

Example
-------
>>> from endgame.visualization import SpiralPlotVisualizer
>>> import numpy as np
>>> values = np.sin(np.linspace(0, 6 * np.pi, 200)) + 1.5
>>> viz = SpiralPlotVisualizer(values, title="Periodic Signal")
>>> viz.save("spiral.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer
from endgame.visualization._palettes import DEFAULT_SEQUENTIAL


class SpiralPlotVisualizer(BaseVisualizer):
    """Interactive spiral plot visualizer.

    Parameters
    ----------
    values : array-like
        Sequential data values to plot along the spiral.
    labels : list of str, optional
        Labels for each data point (shown on hover).
    n_turns : float, optional
        Number of spiral turns. If None, auto-computed.
    color_by_value : bool, default=True
        If True, color points by value. Otherwise use position.
    cmap : str, default='viridis_seq'
        Color palette for value mapping.
    title : str, optional
        Chart title.
    width : int, default=650
        Chart width.
    height : int, default=650
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        values: Any,
        *,
        labels: Sequence[str] | None = None,
        n_turns: float | None = None,
        color_by_value: bool = True,
        cmap: str = DEFAULT_SEQUENTIAL,
        title: str = "",
        width: int = 650,
        height: int = 650,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=cmap, width=width, height=height, theme=theme)
        self._values = np.asarray(values, dtype=float).ravel()
        self._labels = list(labels) if labels else None
        self.n_turns = n_turns
        self.color_by_value = color_by_value

    @classmethod
    def from_time_series(
        cls,
        values: Any,
        timestamps: Sequence[str] | None = None,
        **kwargs,
    ) -> SpiralPlotVisualizer:
        """Create from a time series.

        Parameters
        ----------
        values : array-like
            Time series values.
        timestamps : list of str, optional
            Timestamp labels for each point.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Time Series (Spiral)")
        return cls(values, labels=timestamps, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        vals = self._values
        n = len(vals)
        if n == 0:
            return {"points": [], "vMin": 0, "vMax": 1}

        clean = vals[~np.isnan(vals)]
        v_min = float(clean.min()) if len(clean) > 0 else 0
        v_max = float(clean.max()) if len(clean) > 0 else 1

        n_turns = self.n_turns or max(2, n / 30)

        points = []
        for i in range(n):
            t = i / (n - 1) if n > 1 else 0
            angle = t * n_turns * 2 * np.pi
            # Archimedean spiral: r = a + b*theta
            radius = 0.1 + t * 0.4
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)

            v = vals[i]
            label = self._labels[i] if self._labels and i < len(self._labels) else f"#{i}"

            points.append({
                "x": round(float(x), 5),
                "y": round(float(y), 5),
                "value": None if np.isnan(v) else round(float(v), 6),
                "label": label,
                "idx": i,
            })

        return {
            "points": points,
            "vMin": round(v_min, 6),
            "vMax": round(v_max, 6),
            "colorByValue": self.color_by_value,
        }

    def _chart_type(self) -> str:
        return "spiral"

    def _get_chart_js(self) -> str:
        return _SPIRAL_JS


_SPIRAL_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';
  const palette = config.palette;
  const points = data.points;
  const n = points.length;
  if (n === 0) return;

  const colorFn = data.colorByValue
    ? EG.colorScale(palette, data.vMin, data.vMax)
    : EG.colorScale(palette, 0, n - 1);

  // Draw connecting line
  if (n > 1) {
    let d = '';
    for (let i = 0; i < n; i++) {
      const px = points[i].x * size;
      const py = points[i].y * size;
      d += (i === 0 ? 'M' : ' L') + px + ' ' + py;
    }
    svg.appendChild(EG.svg('path', {
      d: d, fill: 'none', stroke: 'var(--text-muted)',
      'stroke-width': 1, opacity: 0.3
    }));
  }

  // Draw points
  const dotR = Math.max(2, Math.min(5, 200 / n));
  points.forEach(function(p, i) {
    if (p.value === null) return;
    const px = p.x * size;
    const py = p.y * size;
    const color = data.colorByValue ? colorFn(p.value) : colorFn(i);

    const dot = EG.svg('circle', {
      cx: px, cy: py, r: dotR,
      fill: color, opacity: 0.85
    });
    dot.addEventListener('mouseenter', function(e) {
      dot.setAttribute('r', String(dotR + 3));
      dot.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(p.label) + '</b><br>Value: ' + EG.fmt(p.value, 4) + '<br>Index: ' + p.idx);
    });
    dot.addEventListener('mouseleave', function() {
      dot.setAttribute('r', String(dotR));
      dot.setAttribute('opacity', '0.85');
      EG.tooltip.hide();
    });
    svg.appendChild(dot);
  });

  // Color legend bar
  const cbH = size * 0.5, cbW = 12;
  const cbX = size - 30, cbY = (size - cbH) / 2;
  const cbG = EG.svg('g', {transform: `translate(${cbX},${cbY})`});
  svg.appendChild(cbG);
  const nSteps = 30;
  for (let i = 0; i < nSteps; i++) {
    const v = data.vMax - (i / nSteps) * (data.vMax - data.vMin);
    cbG.appendChild(EG.svg('rect', {
      x: 0, y: i * (cbH / nSteps), width: cbW, height: cbH / nSteps + 1,
      fill: colorFn(v)
    }));
  }
  cbG.appendChild(EG.svg('text', {x: cbW + 4, y: 10, fill: 'var(--text-muted)', 'font-size': '9px'})).textContent = EG.fmt(data.vMax);
  cbG.appendChild(EG.svg('text', {x: cbW + 4, y: cbH, fill: 'var(--text-muted)', 'font-size': '9px'})).textContent = EG.fmt(data.vMin);
}
"""
