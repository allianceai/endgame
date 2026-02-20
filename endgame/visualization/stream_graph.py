"""Stream graph (stacked area) visualizer.

Interactive stream graphs for showing how multiple series evolve over
time with smooth, organic shapes. Commonly used for topic evolution,
feature importance over time, or model performance across stages.

Example
-------
>>> from endgame.visualization import StreamGraphVisualizer
>>> viz = StreamGraphVisualizer(
...     x=[1, 2, 3, 4, 5],
...     series={"LGBM": [5, 8, 12, 15, 18],
...             "XGB": [3, 5, 7, 8, 10],
...             "CatBoost": [4, 6, 9, 11, 14]},
...     title="Model Usage Over Time",
... )
>>> viz.save("stream.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class StreamGraphVisualizer(BaseVisualizer):
    """Interactive stream graph visualizer.

    Parameters
    ----------
    x : list of float or list of str
        X-axis values.
    series : dict of str → list of float
        Mapping of series name to values at each x point.
    baseline : str, default='wiggle'
        Baseline algorithm: 'zero' (stacked from zero), 'center'
        (centered around zero), 'wiggle' (minimizes wiggle).
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=900
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        x: Sequence,
        series: dict[str, Sequence[float]],
        *,
        baseline: str = "wiggle",
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.x = list(x)
        self.series = {k: [float(v) for v in vals] for k, vals in series.items()}
        self.baseline = baseline

    def _build_data(self) -> dict[str, Any]:
        names = list(self.series.keys())
        n_series = len(names)
        n_points = len(self.x)

        # Build value matrix
        vals = np.zeros((n_series, n_points))
        for i, name in enumerate(names):
            sv = self.series[name]
            for j in range(min(len(sv), n_points)):
                vals[i, j] = sv[j]

        # Compute stack layout
        if self.baseline == "center":
            # Centered: offset so total is centered around zero
            totals = vals.sum(axis=0)
            base = -totals / 2
        elif self.baseline == "wiggle":
            # ThemeRiver / wiggle: minimize derivative
            n = n_series
            base = np.zeros(n_points)
            if n > 0:
                totals = vals.sum(axis=0)
                for j in range(n_points):
                    base[j] = -totals[j] / 2
                # Additional wiggle offset
                for j in range(1, n_points):
                    shift = 0
                    for i in range(n):
                        shift += (vals[i, j] - vals[i, j-1]) * (i + 0.5) / n
                    base[j] -= shift * 0.3
        else:
            base = np.zeros(n_points)

        # Compute y0/y1 for each layer
        layers = []
        cumulative = base.copy()
        for i in range(n_series):
            y0 = cumulative.copy()
            cumulative = cumulative + vals[i]
            y1 = cumulative.copy()
            layers.append({
                "name": names[i],
                "y0": [round(float(v), 4) for v in y0],
                "y1": [round(float(v), 4) for v in y1],
                "values": vals[i].tolist(),
            })

        # Y range
        all_y = np.concatenate([base, cumulative])
        y_min = float(all_y.min())
        y_max = float(all_y.max())

        return {
            "x": self.x,
            "layers": layers,
            "yMin": y_min,
            "yMax": y_max,
        }

    def _chart_type(self) -> str:
        return "stream_graph"

    def _get_chart_js(self) -> str:
        return _STREAM_JS


_STREAM_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 50, left: 50};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const xVals = data.x;
  const layers = data.layers;
  const n = xVals.length;
  if (n === 0 || layers.length === 0) return;

  // X scale
  const isNumericX = typeof xVals[0] === 'number';
  let xScale;
  if (isNumericX) {
    xScale = EG.scaleLinear([Math.min.apply(null, xVals), Math.max.apply(null, xVals)], [0, W]);
  } else {
    xScale = function(v) {
      const idx = xVals.indexOf(v);
      return idx / (n - 1) * W;
    };
    xScale.domain = [0, n-1];
    xScale.range = [0, W];
  }

  const yScale = EG.scaleLinear([data.yMin, data.yMax], [H, 0]);

  // Draw axes
  if (isNumericX) {
    EG.drawXAxis(g, xScale, H, '');
  } else {
    const axG = EG.svg('g', {transform: `translate(0,${H})`});
    g.appendChild(axG);
    axG.appendChild(EG.svg('line', {x1:0,y1:0,x2:W,y2:0, stroke:'var(--border)'}));
    const step = Math.max(1, Math.floor(n / 8));
    for (let i = 0; i < n; i += step) {
      const x = xScale(xVals[i]);
      const t = EG.svg('text', {x:x, y:20, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'10px'});
      t.textContent = xVals[i];
      axG.appendChild(t);
    }
  }

  // Draw layers (bottom to top)
  layers.forEach(function(layer, li) {
    const color = palette[li % palette.length];
    let d = 'M';

    // Top edge (y1, left to right)
    for (let i = 0; i < n; i++) {
      const x = xScale(xVals[i]);
      const y = yScale(layer.y1[i]);
      d += (i === 0 ? '' : ' L') + x + ' ' + y;
    }
    // Bottom edge (y0, right to left)
    for (let i = n - 1; i >= 0; i--) {
      const x = xScale(xVals[i]);
      const y = yScale(layer.y0[i]);
      d += ' L' + x + ' ' + y;
    }
    d += ' Z';

    const path = EG.svg('path', {d: d, fill: color, opacity: 0.7, stroke: color, 'stroke-width': 0.5});
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('opacity', '0.95');
      EG.tooltip.show(e, '<b>' + EG.esc(layer.name) + '</b>');
    });
    path.addEventListener('mousemove', function(e) {
      // Find nearest x index
      const rect = container.getBoundingClientRect();
      const mx = e.clientX - rect.left - margin.left;
      let bestI = 0, bestDist = Infinity;
      for (let i = 0; i < n; i++) {
        const d = Math.abs(xScale(xVals[i]) - mx);
        if (d < bestDist) { bestDist = d; bestI = i; }
      }
      const xLabel = typeof xVals[bestI] === 'number' ? EG.fmt(xVals[bestI]) : xVals[bestI];
      EG.tooltip.show(e, '<b>' + EG.esc(layer.name) + '</b><br>' + xLabel + ': ' + EG.fmt(layer.values[bestI]));
    });
    path.addEventListener('mouseleave', function() {
      path.setAttribute('opacity', '0.7');
      EG.tooltip.hide();
    });
    g.appendChild(path);
  });

  // Legend
  const items = layers.map(function(l, i) {
    return {label: l.name, color: palette[i % palette.length]};
  });
  EG.drawLegend(container, items);
}
"""
