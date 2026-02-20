"""Box plot and violin plot visualizer.

Interactive box plots for CV score distributions and violin plots for
full distribution shape comparison across models.

Example
-------
>>> from endgame.visualization import BoxPlotVisualizer, ViolinPlotVisualizer
>>> data = {"ModelA": [0.89, 0.91, 0.88, 0.92, 0.90],
...         "ModelB": [0.85, 0.86, 0.84, 0.87, 0.85]}
>>> viz = BoxPlotVisualizer(data, title="CV Scores")
>>> viz.save("boxplot.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


def _compute_box_stats(data: np.ndarray) -> dict[str, float]:
    """Compute box plot statistics."""
    d = data[~np.isnan(data)]
    if len(d) == 0:
        return {"min": 0, "q1": 0, "median": 0, "q3": 0, "max": 0, "mean": 0, "outliers": []}
    q1 = float(np.percentile(d, 25))
    q3 = float(np.percentile(d, 75))
    iqr = q3 - q1
    whisker_lo = float(d[d >= q1 - 1.5 * iqr].min()) if len(d[d >= q1 - 1.5 * iqr]) > 0 else q1
    whisker_hi = float(d[d <= q3 + 1.5 * iqr].max()) if len(d[d <= q3 + 1.5 * iqr]) > 0 else q3
    outliers = d[(d < q1 - 1.5 * iqr) | (d > q3 + 1.5 * iqr)]
    return {
        "min": round(whisker_lo, 6),
        "q1": round(q1, 6),
        "median": round(float(np.median(d)), 6),
        "q3": round(q3, 6),
        "max": round(whisker_hi, 6),
        "mean": round(float(np.mean(d)), 6),
        "outliers": [round(float(v), 6) for v in outliers],
    }


def _compute_violin_kde(data: np.ndarray, n_points: int = 100) -> list[list[float]]:
    """Compute KDE for violin plot."""
    d = data[~np.isnan(data)]
    if len(d) < 2:
        return []
    std = float(np.std(d, ddof=1))
    bw = 1.06 * std * len(d) ** (-1 / 5) if std > 0 else 1.0
    lo, hi = float(d.min()), float(d.max())
    margin = (hi - lo) * 0.1 if hi > lo else 1.0
    x_pts = np.linspace(lo - margin, hi + margin, n_points)
    y_pts = np.zeros_like(x_pts)
    for xi in d:
        y_pts += np.exp(-0.5 * ((x_pts - xi) / bw) ** 2)
    y_pts /= len(d) * bw * np.sqrt(2 * np.pi)
    return [[round(float(x), 6), round(float(y), 8)] for x, y in zip(x_pts, y_pts)]


class BoxPlotVisualizer(BaseVisualizer):
    """Interactive box plot visualizer.

    Parameters
    ----------
    data : dict of str → list of float
        Mapping of group name to values.
    orientation : str, default='vertical'
        'vertical' or 'horizontal'.
    show_points : bool, default=False
        Show individual data points (jittered).
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
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
        data: dict[str, Sequence[float]],
        *,
        orientation: str = "vertical",
        show_points: bool = False,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.orientation = orientation
        self.show_points = show_points
        self.x_label = x_label
        self.y_label = y_label

    @classmethod
    def from_cv_results(
        cls,
        results: dict[str, Sequence[float]],
        **kwargs,
    ) -> BoxPlotVisualizer:
        """Create box plot from CV results.

        Parameters
        ----------
        results : dict of str → list of float
            Model name → fold scores.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "CV Score Distributions")
        kwargs.setdefault("y_label", "Score")
        return cls(results, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        groups = []
        for name, vals in self._data.items():
            stats = _compute_box_stats(vals)
            stats["name"] = name
            if self.show_points:
                clean = vals[~np.isnan(vals)]
                stats["points"] = [round(float(v), 6) for v in clean]
            groups.append(stats)

        return {
            "groups": groups,
            "orientation": self.orientation,
            "showPoints": self.show_points,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
            "chartType": "box",
        }

    def _chart_type(self) -> str:
        return "box_violin"

    def _get_chart_js(self) -> str:
        return _BOX_VIOLIN_JS


class ViolinPlotVisualizer(BaseVisualizer):
    """Interactive violin plot visualizer.

    Parameters
    ----------
    data : dict of str → list of float
        Mapping of group name to values.
    show_box : bool, default=True
        Show mini box plot inside violin.
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
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
        data: dict[str, Sequence[float]],
        *,
        show_box: bool = True,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._data = {k: np.asarray(v, dtype=float) for k, v in data.items()}
        self.show_box = show_box
        self.x_label = x_label
        self.y_label = y_label

    def _build_data(self) -> dict[str, Any]:
        groups = []
        for name, vals in self._data.items():
            stats = _compute_box_stats(vals)
            stats["name"] = name
            stats["kde"] = _compute_violin_kde(vals)
            groups.append(stats)

        return {
            "groups": groups,
            "showBox": self.show_box,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
            "chartType": "violin",
        }

    def _chart_type(self) -> str:
        return "box_violin"

    def _get_chart_js(self) -> str:
        return _BOX_VIOLIN_JS


# ---------------------------------------------------------------------------
# JavaScript renderer (handles both box and violin)
# ---------------------------------------------------------------------------

_BOX_VIOLIN_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 50, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const groups = data.groups;
  const isViolin = data.chartType === 'violin';
  const isHoriz = data.orientation === 'horizontal';

  const names = groups.map(function(gr) { return gr.name; });

  // Value range
  let vMin = Infinity, vMax = -Infinity;
  groups.forEach(function(gr) {
    if (gr.min < vMin) vMin = gr.min;
    if (gr.max > vMax) vMax = gr.max;
    gr.outliers.forEach(function(o) {
      if (o < vMin) vMin = o;
      if (o > vMax) vMax = o;
    });
    if (gr.kde) {
      gr.kde.forEach(function(p) {
        if (p[0] < vMin) vMin = p[0];
        if (p[0] > vMax) vMax = p[0];
      });
    }
  });
  const pad = (vMax - vMin) * 0.08 || 0.5;
  vMin -= pad; vMax += pad;

  const catScale = EG.scaleBand(names, [0, isHoriz ? H : W], 0.2);
  const valScale = EG.scaleLinear([vMin, vMax], isHoriz ? [0, W] : [H, 0]);

  if (isHoriz) {
    EG.drawXAxis(g, valScale, H, data.yLabel || data.xLabel);
    EG.drawYAxis(g, catScale, W, data.xLabel || data.yLabel, true);
  } else {
    EG.drawXAxis(g, catScale, H, data.xLabel, true);
    EG.drawYAxis(g, valScale, W, data.yLabel);
  }

  const bw = catScale.bandwidth();

  groups.forEach(function(gr, gi) {
    const color = palette[gi % palette.length];
    const cx = catScale(gr.name) + bw / 2;

    if (isViolin && gr.kde && gr.kde.length > 1) {
      // Violin shape
      let maxDensity = 0;
      gr.kde.forEach(function(p) { if (p[1] > maxDensity) maxDensity = p[1]; });
      const halfW = bw * 0.45;

      let dLeft = 'M', dRight = 'M';
      gr.kde.forEach(function(p, j) {
        const valPos = valScale(p[0]);
        const w = (p[1] / maxDensity) * halfW;
        if (j === 0) {
          dLeft += (cx - w) + ' ' + valPos;
          dRight += (cx + w) + ' ' + valPos;
        } else {
          dLeft += ' L' + (cx - w) + ' ' + valPos;
          dRight += ' L' + (cx + w) + ' ' + valPos;
        }
      });
      // Close path
      let fullPath = dRight;
      for (let j = gr.kde.length - 1; j >= 0; j--) {
        const valPos = valScale(gr.kde[j][0]);
        const w = (gr.kde[j][1] / maxDensity) * halfW;
        fullPath += ' L' + (cx - w) + ' ' + valPos;
      }
      fullPath += ' Z';
      g.appendChild(EG.svg('path', {d: fullPath, fill: color, opacity: 0.35, stroke: color, 'stroke-width': 1.5}));

      // Mini box inside
      if (data.showBox) {
        const boxW = bw * 0.08;
        g.appendChild(EG.svg('rect', {
          x: cx - boxW, y: valScale(gr.q3),
          width: boxW * 2, height: Math.abs(valScale(gr.q1) - valScale(gr.q3)),
          fill: color, opacity: 0.7, rx: 2
        }));
        g.appendChild(EG.svg('line', {
          x1: cx - boxW, y1: valScale(gr.median), x2: cx + boxW, y2: valScale(gr.median),
          stroke: '#fff', 'stroke-width': 2
        }));
      }
    } else {
      // Box plot
      const boxW = bw * 0.5;
      const x0 = cx - boxW / 2;

      // Whiskers
      g.appendChild(EG.svg('line', {
        x1: cx, y1: valScale(gr.min), x2: cx, y2: valScale(gr.q1),
        stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '3,2'
      }));
      g.appendChild(EG.svg('line', {
        x1: cx, y1: valScale(gr.q3), x2: cx, y2: valScale(gr.max),
        stroke: color, 'stroke-width': 1.5, 'stroke-dasharray': '3,2'
      }));
      // Whisker caps
      const capW = boxW * 0.4;
      g.appendChild(EG.svg('line', {x1: cx-capW, y1: valScale(gr.min), x2: cx+capW, y2: valScale(gr.min), stroke: color, 'stroke-width': 1.5}));
      g.appendChild(EG.svg('line', {x1: cx-capW, y1: valScale(gr.max), x2: cx+capW, y2: valScale(gr.max), stroke: color, 'stroke-width': 1.5}));

      // Box
      const boxRect = EG.svg('rect', {
        x: x0, y: valScale(gr.q3),
        width: boxW, height: Math.abs(valScale(gr.q1) - valScale(gr.q3)),
        fill: color, opacity: 0.4, stroke: color, 'stroke-width': 1.5, rx: 3
      });
      boxRect.addEventListener('mouseenter', function(e) {
        EG.tooltip.show(e,
          '<b>' + EG.esc(gr.name) + '</b><br>' +
          'Median: ' + EG.fmt(gr.median, 4) + '<br>' +
          'Q1: ' + EG.fmt(gr.q1, 4) + ', Q3: ' + EG.fmt(gr.q3, 4) + '<br>' +
          'Min: ' + EG.fmt(gr.min, 4) + ', Max: ' + EG.fmt(gr.max, 4) + '<br>' +
          'Mean: ' + EG.fmt(gr.mean, 4)
        );
      });
      boxRect.addEventListener('mouseleave', function() { EG.tooltip.hide(); });
      g.appendChild(boxRect);

      // Median line
      g.appendChild(EG.svg('line', {
        x1: x0, y1: valScale(gr.median), x2: x0 + boxW, y2: valScale(gr.median),
        stroke: '#fff', 'stroke-width': 2.5
      }));

      // Mean dot
      g.appendChild(EG.svg('circle', {
        cx: cx, cy: valScale(gr.mean), r: 3,
        fill: '#fff', stroke: color, 'stroke-width': 1.5
      }));
    }

    // Outliers
    gr.outliers.forEach(function(o) {
      g.appendChild(EG.svg('circle', {
        cx: cx, cy: valScale(o), r: 3,
        fill: 'none', stroke: color, 'stroke-width': 1.5, opacity: 0.7
      }));
    });

    // Individual points
    if (data.showPoints && gr.points) {
      gr.points.forEach(function(p) {
        const jitter = (Math.random() - 0.5) * bw * 0.3;
        g.appendChild(EG.svg('circle', {
          cx: cx + jitter, cy: valScale(p), r: 2.5,
          fill: color, opacity: 0.5
        }));
      });
    }
  });
}
"""
