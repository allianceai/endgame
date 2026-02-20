"""Ridgeline (Joy) plot visualizer.

Overlapping density plots — beautiful for comparing distributions across
models, folds, or feature groups. Each group gets its own KDE density
curve stacked vertically with configurable overlap.

Example
-------
>>> from endgame.visualization import RidgelinePlotVisualizer
>>> data = {"Fold 1": scores_1, "Fold 2": scores_2, "Fold 3": scores_3}
>>> viz = RidgelinePlotVisualizer(data)
>>> viz.save("ridgeline.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class RidgelinePlotVisualizer(BaseVisualizer):
    """Interactive ridgeline (joy) plot visualizer.

    Parameters
    ----------
    data : dict of str → list of float
        Group name → raw values.
    overlap : float, default=0.5
        Overlap ratio between adjacent ridges (0 = no overlap, 1 = full).
    kde_points : int, default=100
        Number of points for KDE evaluation.
    bandwidth : float, optional
        KDE bandwidth. If None, uses Silverman's rule.
    show_quantiles : bool, default=True
        Show median and Q1/Q3 markers.
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
        overlap: float = 0.5,
        kde_points: int = 100,
        bandwidth: float | None = None,
        show_quantiles: bool = True,
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Ridgeline Plot", palette=palette, width=width, height=height, theme=theme)
        self._data = {k: list(v) for k, v in data.items()}
        self.overlap = overlap
        self.kde_points = kde_points
        self.bandwidth = bandwidth
        self.show_quantiles = show_quantiles

    @classmethod
    def from_cv_results(
        cls,
        results: dict[str, Sequence[float]],
        **kwargs,
    ) -> RidgelinePlotVisualizer:
        """Create from cross-validation results.

        Parameters
        ----------
        results : dict of str → list of float
            Model name → fold scores.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "CV Score Distributions")
        return cls(results, **kwargs)

    @classmethod
    def from_feature_distributions(
        cls,
        X: Any,
        feature_names: Sequence[str] | None = None,
        *,
        max_features: int = 15,
        **kwargs,
    ) -> RidgelinePlotVisualizer:
        """Create from feature column distributions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.
        feature_names : list of str, optional
            Feature names.
        max_features : int, default=15
            Max number of features to show.
        **kwargs
            Additional keyword arguments.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        n_features = min(X_arr.shape[1], max_features)
        if feature_names is None:
            if hasattr(X, "columns"):
                feature_names = list(X.columns)[:n_features]
            else:
                feature_names = [f"Feature {i}" for i in range(n_features)]

        data = {}
        for i in range(n_features):
            col = X_arr[:, i]
            col = col[~np.isnan(col)]
            data[feature_names[i]] = col.tolist()

        kwargs.setdefault("title", "Feature Distributions")
        return cls(data, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        ridges = []
        global_min = float("inf")
        global_max = float("-inf")

        for name, values in self._data.items():
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            if len(arr) == 0:
                continue

            lo, hi = float(arr.min()), float(arr.max())
            global_min = min(global_min, lo)
            global_max = max(global_max, hi)

            # KDE
            bw = self.bandwidth
            if bw is None:
                std = float(np.std(arr))
                n = len(arr)
                bw = 1.06 * std * n ** (-1 / 5) if std > 0 else 0.1

            pad = bw * 2
            x_grid = np.linspace(lo - pad, hi + pad, self.kde_points)
            density = _kde(arr, x_grid, bw)

            # Quantiles
            q1 = float(np.percentile(arr, 25))
            median = float(np.median(arr))
            q3 = float(np.percentile(arr, 75))
            mean = float(np.mean(arr))

            ridges.append({
                "name": name,
                "x": [round(float(v), 6) for v in x_grid],
                "density": [round(float(v), 8) for v in density],
                "q1": round(q1, 6),
                "median": round(median, 6),
                "q3": round(q3, 6),
                "mean": round(mean, 6),
                "n": len(arr),
            })

        return {
            "ridges": ridges,
            "overlap": self.overlap,
            "showQuantiles": self.show_quantiles,
        }

    def _chart_type(self) -> str:
        return "ridgeline"

    def _get_chart_js(self) -> str:
        return _RIDGELINE_JS


def _kde(data: np.ndarray, x_grid: np.ndarray, bandwidth: float) -> np.ndarray:
    """Simple Gaussian KDE."""
    n = len(data)
    if n == 0 or bandwidth <= 0:
        return np.zeros_like(x_grid)
    density = np.zeros_like(x_grid, dtype=float)
    for xi in data:
        density += np.exp(-0.5 * ((x_grid - xi) / bandwidth) ** 2)
    density /= n * bandwidth * np.sqrt(2 * np.pi)
    return density


_RIDGELINE_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const ridges = data.ridges;
  const n = ridges.length;
  if (n === 0) return;

  const margin = {top: 20, right: 30, bottom: 45, left: 130};
  const W = config.width - margin.left - margin.right;
  const ridgeH = Math.min(80, (config.height - margin.top - margin.bottom) / (n * (1 - data.overlap * 0.5)));
  const stepY = ridgeH * (1 - data.overlap * 0.5);
  const totalH = stepY * n + ridgeH * 0.5 + margin.top + margin.bottom;

  const svg = EG.svg('svg', {width: config.width, height: totalH});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: 'translate(' + margin.left + ',' + margin.top + ')'});
  svg.appendChild(g);

  const palette = config.palette;

  // Global x range
  let xMin = Infinity, xMax = -Infinity;
  ridges.forEach(function(r) {
    r.x.forEach(function(v) { if (v < xMin) xMin = v; if (v > xMax) xMax = v; });
  });
  const xScale = EG.scaleLinear([xMin, xMax], [0, W]);

  // Find max density for normalization
  let dMax = 0;
  ridges.forEach(function(r) {
    r.density.forEach(function(v) { if (v > dMax) dMax = v; });
  });
  if (dMax === 0) dMax = 1;

  // Draw ridges from bottom to top so earlier ones render behind
  for (let i = n - 1; i >= 0; i--) {
    const r = ridges[i];
    const color = palette[i % palette.length];
    const baseY = i * stepY;
    const yScale = function(d) { return baseY + ridgeH - (d / dMax) * ridgeH * 0.85; };

    // Area path
    let d = 'M' + xScale(r.x[0]) + ' ' + (baseY + ridgeH);
    for (let j = 0; j < r.x.length; j++) {
      d += ' L' + xScale(r.x[j]) + ' ' + yScale(r.density[j]);
    }
    d += ' L' + xScale(r.x[r.x.length - 1]) + ' ' + (baseY + ridgeH) + ' Z';

    const area = EG.svg('path', {
      d: d, fill: color, opacity: 0.55,
      stroke: color, 'stroke-width': 1.5
    });
    area.addEventListener('mouseenter', function(e) {
      area.setAttribute('opacity', '0.8');
      EG.tooltip.show(e,
        '<b>' + EG.esc(r.name) + '</b> (n=' + r.n + ')<br>' +
        'Mean: ' + EG.fmt(r.mean, 4) + '<br>' +
        'Median: ' + EG.fmt(r.median, 4) + '<br>' +
        'Q1: ' + EG.fmt(r.q1, 4) + ' | Q3: ' + EG.fmt(r.q3, 4));
    });
    area.addEventListener('mouseleave', function() {
      area.setAttribute('opacity', '0.55');
      EG.tooltip.hide();
    });
    g.appendChild(area);

    // Quantile markers
    if (data.showQuantiles) {
      [r.q1, r.median, r.q3].forEach(function(qv, qi) {
        const qx = xScale(qv);
        const dAtQ = _interpDensity(r.x, r.density, qv);
        const qy1 = baseY + ridgeH;
        const qy2 = yScale(dAtQ);
        g.appendChild(EG.svg('line', {
          x1: qx, y1: qy1, x2: qx, y2: qy2,
          stroke: qi === 1 ? '#ffffff' : 'rgba(255,255,255,0.5)',
          'stroke-width': qi === 1 ? 2 : 1,
          'stroke-dasharray': qi === 1 ? 'none' : '3,2'
        }));
      });
    }

    // Label
    g.appendChild(EG.svg('text', {
      x: -10, y: baseY + ridgeH * 0.6,
      'text-anchor': 'end', fill: 'var(--text-primary)',
      'font-size': '11px'
    })).textContent = r.name.length > 18 ? r.name.slice(0, 16) + '…' : r.name;
  }

  // X axis
  var axG = EG.svg('g', {transform: 'translate(0,' + (n * stepY + ridgeH * 0.3) + ')'});
  g.appendChild(axG);
  axG.appendChild(EG.svg('line', {x1: 0, y1: 0, x2: W, y2: 0, stroke: 'var(--border)'}));
  var ticks = EG.niceTicks(xMin, xMax, 6);
  ticks.forEach(function(v) {
    axG.appendChild(EG.svg('line', {x1: xScale(v), y1: 0, x2: xScale(v), y2: 5, stroke: 'var(--text-muted)'}));
    axG.appendChild(EG.svg('text', {x: xScale(v), y: 18, 'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '10px'})).textContent = EG.fmt(v, 3);
  });

  function _interpDensity(xs, ds, val) {
    for (var j = 0; j < xs.length - 1; j++) {
      if (xs[j] <= val && xs[j+1] >= val) {
        var t = (val - xs[j]) / (xs[j+1] - xs[j]);
        return ds[j] + t * (ds[j+1] - ds[j]);
      }
    }
    return 0;
  }
}
"""
