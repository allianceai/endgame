"""Line chart and area chart visualizer.

Interactive line charts for learning curves, loss curves, CV scores,
and time-series metrics. Supports error bands and area fill.

Example
-------
>>> from endgame.visualization import LineChartVisualizer
>>> viz = LineChartVisualizer(
...     x=[1, 2, 3, 4, 5],
...     series={"train": [0.9, 0.92, 0.95, 0.96, 0.97],
...             "valid": [0.88, 0.89, 0.90, 0.91, 0.90]},
...     title="Learning Curve",
... )
>>> viz.save("learning_curve.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class LineChartVisualizer(BaseVisualizer):
    """Interactive line chart visualizer.

    Parameters
    ----------
    x : list of float or list of str
        X-axis values (shared across series).
    series : dict of str → list of float
        Mapping of series name to Y values.
    error_bands : dict of str → (list of float, list of float), optional
        Mapping of series name to (lower, upper) error bands.
    area : bool, default=False
        If True, fill area under each line.
    markers : bool, default=True
        Show data point markers.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette name.
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
        error_bands: dict[str, tuple] | None = None,
        area: bool = False,
        markers: bool = True,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.x = list(x)
        self.series = {k: list(v) for k, v in series.items()}
        self.error_bands = error_bands or {}
        self.area = area
        self.markers = markers
        self.x_label = x_label
        self.y_label = y_label

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_learning_curve(
        cls,
        train_sizes: Sequence[float],
        train_scores: Any,
        test_scores: Any,
        **kwargs,
    ) -> LineChartVisualizer:
        """Create a learning curve plot.

        Parameters
        ----------
        train_sizes : array-like
            Training set sizes.
        train_scores : array-like, shape (n_sizes,) or (n_sizes, n_folds)
            Training scores.
        test_scores : array-like, shape (n_sizes,) or (n_sizes, n_folds)
            Test/validation scores.
        **kwargs
            Additional keyword arguments.
        """
        train_arr = np.asarray(train_scores)
        test_arr = np.asarray(test_scores)

        if train_arr.ndim == 2:
            train_mean = train_arr.mean(axis=1).tolist()
            test_mean = test_arr.mean(axis=1).tolist()
            train_std = train_arr.std(axis=1)
            test_std = test_arr.std(axis=1)
            error_bands = {
                "Train": (
                    (train_arr.mean(axis=1) - train_std).tolist(),
                    (train_arr.mean(axis=1) + train_std).tolist(),
                ),
                "Validation": (
                    (test_arr.mean(axis=1) - test_std).tolist(),
                    (test_arr.mean(axis=1) + test_std).tolist(),
                ),
            }
        else:
            train_mean = train_arr.tolist()
            test_mean = test_arr.tolist()
            error_bands = None

        kwargs.setdefault("title", "Learning Curve")
        kwargs.setdefault("x_label", "Training Size")
        kwargs.setdefault("y_label", "Score")
        return cls(
            x=list(train_sizes),
            series={"Train": train_mean, "Validation": test_mean},
            error_bands=error_bands,
            **kwargs,
        )

    @classmethod
    def from_cv_scores(
        cls,
        scores: dict[str, Sequence[float]],
        **kwargs,
    ) -> LineChartVisualizer:
        """Create a line chart from cross-validation scores.

        Parameters
        ----------
        scores : dict of str → list of float
            Mapping of model name to fold scores.
        **kwargs
            Additional keyword arguments.
        """
        n_folds = max(len(v) for v in scores.values())
        x = [f"Fold {i+1}" for i in range(n_folds)]
        kwargs.setdefault("title", "CV Scores")
        kwargs.setdefault("x_label", "Fold")
        kwargs.setdefault("y_label", "Score")
        return cls(x=x, series=scores, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        series_list = []
        for name, values in self.series.items():
            entry: dict[str, Any] = {"name": name, "values": values}
            if name in self.error_bands:
                lo, hi = self.error_bands[name]
                entry["errorLo"] = list(lo)
                entry["errorHi"] = list(hi)
            series_list.append(entry)

        return {
            "x": self.x,
            "series": series_list,
            "area": self.area,
            "markers": self.markers,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
        }

    def _chart_type(self) -> str:
        return "line"

    def _get_chart_js(self) -> str:
        return _LINE_CHART_JS


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_LINE_CHART_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 50, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const xVals = data.x;
  const isNumericX = typeof xVals[0] === 'number';

  // X scale
  let xScale;
  if (isNumericX) {
    const xMin = Math.min.apply(null, xVals);
    const xMax = Math.max.apply(null, xVals);
    xScale = EG.scaleLinear([xMin, xMax], [0, W]);
  } else {
    // Categorical X (fold names, etc.)
    xScale = function(v) {
      const idx = xVals.indexOf(v);
      return (idx + 0.5) * (W / xVals.length);
    };
    xScale.domain = [0, xVals.length];
    xScale.range = [0, W];
  }

  // Y range
  let yMin = Infinity, yMax = -Infinity;
  data.series.forEach(function(s) {
    s.values.forEach(function(v) {
      if (v < yMin) yMin = v;
      if (v > yMax) yMax = v;
    });
    if (s.errorLo) s.errorLo.forEach(function(v) { if (v < yMin) yMin = v; });
    if (s.errorHi) s.errorHi.forEach(function(v) { if (v > yMax) yMax = v; });
  });
  const pad = (yMax - yMin) * 0.08 || 0.1;
  yMin -= pad; yMax += pad;
  const yScale = EG.scaleLinear([yMin, yMax], [H, 0]);

  // Axes
  if (isNumericX) {
    EG.drawXAxis(g, xScale, H, data.xLabel);
  } else {
    // Custom categorical X axis
    const axG = EG.svg('g', {transform: `translate(0,${H})`});
    g.appendChild(axG);
    axG.appendChild(EG.svg('line', {x1:0,y1:0,x2:W,y2:0, stroke:'var(--border)', 'stroke-width':1}));
    xVals.forEach(function(v, i) {
      const x = xScale(v);
      const t = EG.svg('text', {x:x, y:20, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'11px'});
      t.textContent = v;
      axG.appendChild(t);
    });
    if (data.xLabel) {
      const lbl = EG.svg('text', {x:W/2, y:40, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'12px','font-weight':'500'});
      lbl.textContent = data.xLabel;
      axG.appendChild(lbl);
    }
  }
  EG.drawYAxis(g, yScale, W, data.yLabel);

  // Draw each series
  data.series.forEach(function(s, si) {
    const color = palette[si % palette.length];
    const n = Math.min(s.values.length, xVals.length);

    // Error band
    if (s.errorLo && s.errorHi) {
      let d = 'M';
      for (let i = 0; i < n; i++) {
        d += (i === 0 ? '' : ' L') + xScale(xVals[i]) + ' ' + yScale(s.errorHi[i]);
      }
      for (let i = n - 1; i >= 0; i--) {
        d += ' L' + xScale(xVals[i]) + ' ' + yScale(s.errorLo[i]);
      }
      d += ' Z';
      g.appendChild(EG.svg('path', {d: d, fill: color, opacity: 0.15}));
    }

    // Area fill
    if (data.area) {
      let d = 'M' + xScale(xVals[0]) + ' ' + yScale(s.values[0]);
      for (let i = 1; i < n; i++) {
        d += ' L' + xScale(xVals[i]) + ' ' + yScale(s.values[i]);
      }
      d += ' L' + xScale(xVals[n-1]) + ' ' + H;
      d += ' L' + xScale(xVals[0]) + ' ' + H + ' Z';
      g.appendChild(EG.svg('path', {d: d, fill: color, opacity: 0.15}));
    }

    // Line
    let d = 'M';
    for (let i = 0; i < n; i++) {
      d += (i === 0 ? '' : ' L') + xScale(xVals[i]) + ' ' + yScale(s.values[i]);
    }
    g.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: color, 'stroke-width': 2.5, 'stroke-linejoin': 'round'}));

    // Markers
    if (data.markers) {
      for (let i = 0; i < n; i++) {
        const cx = xScale(xVals[i]);
        const cy = yScale(s.values[i]);
        const circle = EG.svg('circle', {cx: cx, cy: cy, r: 4, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2});
        circle.addEventListener('mouseenter', function(e) {
          circle.setAttribute('r', '6');
          const xStr = typeof xVals[i] === 'number' ? EG.fmt(xVals[i]) : xVals[i];
          EG.tooltip.show(e, '<b>' + EG.esc(s.name) + '</b><br>' + xStr + ': ' + EG.fmt(s.values[i], 4));
        });
        circle.addEventListener('mouseleave', function() { circle.setAttribute('r', '4'); EG.tooltip.hide(); });
        g.appendChild(circle);
      }
    }
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
