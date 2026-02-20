"""Histogram and density plot visualizer.

Interactive histograms for feature distributions, residuals, and
prediction probabilities. Supports KDE overlay and density-only mode.

Example
-------
>>> from endgame.visualization import HistogramVisualizer
>>> import numpy as np
>>> data = np.random.randn(1000)
>>> viz = HistogramVisualizer(data, title="Distribution")
>>> viz.save("histogram.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class HistogramVisualizer(BaseVisualizer):
    """Interactive histogram visualizer.

    Parameters
    ----------
    data : array-like or list of array-like
        Data values. Pass multiple arrays for overlaid histograms.
    series_names : list of str, optional
        Names for each series.
    bins : int or str, default='auto'
        Number of bins or 'auto' (Freedman-Diaconis rule).
    density : bool, default=False
        If True, show density instead of counts.
    kde : bool, default=False
        If True, overlay KDE curve.
    cumulative : bool, default=False
        If True, show cumulative histogram.
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
        data: Any | Sequence[Any],
        *,
        series_names: Sequence[str] | None = None,
        bins: int | str = "auto",
        density: bool = False,
        kde: bool = False,
        cumulative: bool = False,
        x_label: str = "",
        y_label: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.bins = bins
        self.density = density
        self.kde = kde
        self.cumulative = cumulative
        self.x_label = x_label
        self.y_label = y_label

        # Normalize to list-of-arrays
        if isinstance(data, np.ndarray) and data.ndim == 1:
            self._datasets = [data]
        elif isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (list, tuple, np.ndarray)):
            self._datasets = [np.asarray(d, dtype=float) for d in data]
        else:
            self._datasets = [np.asarray(data, dtype=float)]

        self._series_names = list(series_names) if series_names else [f"Series {i}" for i in range(len(self._datasets))]

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_residuals(
        cls,
        y_true: Any,
        y_pred: Any,
        **kwargs,
    ) -> HistogramVisualizer:
        """Create a histogram of residuals.

        Parameters
        ----------
        y_true : array-like
            True values.
        y_pred : array-like
            Predicted values.
        **kwargs
            Additional keyword arguments.
        """
        residuals = np.asarray(y_true) - np.asarray(y_pred)
        kwargs.setdefault("title", "Residual Distribution")
        kwargs.setdefault("x_label", "Residual")
        kwargs.setdefault("kde", True)
        return cls(residuals, series_names=["Residuals"], **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_proba: Any,
        **kwargs,
    ) -> HistogramVisualizer:
        """Create a histogram of prediction probabilities.

        Parameters
        ----------
        y_proba : array-like
            Predicted probabilities.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Prediction Distribution")
        kwargs.setdefault("x_label", "Probability")
        return cls(np.asarray(y_proba).ravel(), **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        # Compute bins
        all_vals = np.concatenate(self._datasets)
        all_vals = all_vals[~np.isnan(all_vals)]
        if len(all_vals) == 0:
            return {"series": [], "xLabel": self.x_label, "yLabel": self.y_label}

        if isinstance(self.bins, str) and self.bins == "auto":
            n_bins = self._freedman_diaconis(all_vals)
        else:
            n_bins = int(self.bins)

        lo, hi = float(np.min(all_vals)), float(np.max(all_vals))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        bin_edges = np.linspace(lo, hi, n_bins + 1)

        series = []
        for i, ds in enumerate(self._datasets):
            clean = ds[~np.isnan(ds)]
            counts, _ = np.histogram(clean, bins=bin_edges)
            width = float(bin_edges[1] - bin_edges[0])

            if self.density:
                total = float(counts.sum()) * width
                vals = (counts / total).tolist() if total > 0 else counts.tolist()
            elif self.cumulative:
                vals = np.cumsum(counts).tolist()
            else:
                vals = counts.tolist()

            entry: dict[str, Any] = {
                "name": self._series_names[i],
                "counts": vals,
                "binEdges": bin_edges.tolist(),
            }

            if self.kde:
                entry["kde"] = self._compute_kde(clean, bin_edges)

            series.append(entry)

        y_label = self.y_label
        if not y_label:
            y_label = "Density" if self.density else ("Cumulative Count" if self.cumulative else "Count")

        return {
            "series": series,
            "xLabel": self.x_label,
            "yLabel": y_label,
            "density": self.density,
            "kde": self.kde,
        }

    def _chart_type(self) -> str:
        return "histogram"

    def _get_chart_js(self) -> str:
        return _HISTOGRAM_JS

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _freedman_diaconis(data: np.ndarray) -> int:
        """Compute optimal number of bins using Freedman-Diaconis rule."""
        q75, q25 = np.percentile(data, [75, 25])
        iqr = q75 - q25
        n = len(data)
        if iqr == 0 or n == 0:
            return max(10, int(np.sqrt(n)))
        bin_width = 2 * iqr * n ** (-1 / 3)
        n_bins = int(np.ceil((data.max() - data.min()) / bin_width))
        return max(5, min(n_bins, 200))

    @staticmethod
    def _compute_kde(data: np.ndarray, bin_edges: np.ndarray) -> list[list[float]]:
        """Compute Gaussian KDE for overlay."""
        if len(data) < 2:
            return []

        # Scott's bandwidth
        std = float(np.std(data, ddof=1))
        bw = 1.06 * std * len(data) ** (-1 / 5) if std > 0 else 1.0

        lo, hi = float(bin_edges[0]), float(bin_edges[-1])
        margin = (hi - lo) * 0.05
        x_pts = np.linspace(lo - margin, hi + margin, 200)
        y_pts = np.zeros_like(x_pts)

        for xi in data:
            y_pts += np.exp(-0.5 * ((x_pts - xi) / bw) ** 2)
        y_pts /= len(data) * bw * np.sqrt(2 * np.pi)

        return [[round(float(x), 6), round(float(y), 8)] for x, y in zip(x_pts, y_pts)]


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_HISTOGRAM_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  if (!data.series || data.series.length === 0) return;

  const margin = {top: 20, right: 20, bottom: 50, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  // Find global ranges
  let xMin = Infinity, xMax = -Infinity, yMax = 0;
  data.series.forEach(function(s) {
    const edges = s.binEdges;
    if (edges[0] < xMin) xMin = edges[0];
    if (edges[edges.length-1] > xMax) xMax = edges[edges.length-1];
    s.counts.forEach(function(c) { if (c > yMax) yMax = c; });
    if (s.kde) {
      s.kde.forEach(function(p) { if (p[1] > yMax) yMax = p[1]; });
    }
  });
  yMax *= 1.1;
  if (yMax === 0) yMax = 1;

  const xScale = EG.scaleLinear([xMin, xMax], [0, W]);
  const yScale = EG.scaleLinear([0, yMax], [H, 0]);

  EG.drawXAxis(g, xScale, H, data.xLabel);
  EG.drawYAxis(g, yScale, W, data.yLabel);

  const nSeries = data.series.length;

  data.series.forEach(function(s, si) {
    const color = palette[si % palette.length];
    const edges = s.binEdges;
    const counts = s.counts;
    const alpha = nSeries > 1 ? 0.6 : 0.8;

    // Draw bars
    for (let i = 0; i < counts.length; i++) {
      const x1 = xScale(edges[i]);
      const x2 = xScale(edges[i+1]);
      const barH = H - yScale(counts[i]);
      const rect = EG.svg('rect', {
        x: x1, y: H - barH, width: Math.max(x2 - x1 - 1, 1), height: Math.max(barH, 0),
        fill: color, opacity: alpha, rx: 1
      });
      rect.addEventListener('mouseenter', function(e) {
        rect.setAttribute('opacity', '1');
        const lo = Number(edges[i]).toFixed(2);
        const hi = Number(edges[i+1]).toFixed(2);
        EG.tooltip.show(e, (s.name ? '<b>' + EG.esc(s.name) + '</b><br>' : '') +
          'Range: [' + lo + ', ' + hi + ')<br>Value: ' + EG.fmt(counts[i]));
      });
      rect.addEventListener('mouseleave', function() {
        rect.setAttribute('opacity', String(alpha));
        EG.tooltip.hide();
      });
      g.appendChild(rect);
    }

    // Draw KDE
    if (s.kde && s.kde.length > 1) {
      let d = 'M';
      s.kde.forEach(function(p, j) {
        const px = xScale(p[0]);
        const py = yScale(p[1]);
        d += (j === 0 ? '' : ' L') + px + ' ' + py;
      });
      g.appendChild(EG.svg('path', {
        d: d, fill: 'none', stroke: color,
        'stroke-width': 2.5, 'stroke-dasharray': '6,3', opacity: 0.9
      }));
    }
  });

  // Legend
  if (nSeries > 1) {
    const items = data.series.map(function(s, i) {
      return {label: s.name, color: palette[i % palette.length]};
    });
    EG.drawLegend(container, items);
  }
}
"""
