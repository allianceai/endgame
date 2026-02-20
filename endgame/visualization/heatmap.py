"""Heatmap visualizer.

Interactive heatmap for correlation matrices, hyperparameter grids,
and general 2D data visualization.

Example
-------
>>> from endgame.visualization import HeatmapVisualizer
>>> import numpy as np
>>> data = np.random.randn(5, 5)
>>> viz = HeatmapVisualizer(data, x_labels=["a","b","c","d","e"],
...                         y_labels=["a","b","c","d","e"])
>>> viz.save("heatmap.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer
from endgame.visualization._palettes import DEFAULT_DIVERGING


class HeatmapVisualizer(BaseVisualizer):
    """Interactive heatmap visualizer.

    Parameters
    ----------
    data : array-like, shape (n_rows, n_cols)
        2D data matrix.
    x_labels : list of str, optional
        Column labels.
    y_labels : list of str, optional
        Row labels.
    annotate : bool, default=True
        Show cell values.
    fmt : str, default='.2f'
        Format string for annotations.
    vmin, vmax : float, optional
        Color scale range. Auto-computed if None.
    cmap : str, default='rdbu'
        Color palette for the heatmap (diverging palettes work best).
    title : str, optional
        Chart title.
    width : int, default=700
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        data: Any,
        *,
        x_labels: Sequence[str] | None = None,
        y_labels: Sequence[str] | None = None,
        annotate: bool = True,
        fmt: str = ".2f",
        vmin: float | None = None,
        vmax: float | None = None,
        cmap: str = DEFAULT_DIVERGING,
        title: str = "",
        width: int = 700,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=cmap, width=width, height=height, theme=theme)
        self._data = np.asarray(data, dtype=float)
        if self._data.ndim != 2:
            raise ValueError(f"data must be 2D, got shape {self._data.shape}")

        n_rows, n_cols = self._data.shape
        self.x_labels = list(x_labels) if x_labels is not None else [str(i) for i in range(n_cols)]
        self.y_labels = list(y_labels) if y_labels is not None else [str(i) for i in range(n_rows)]
        self.annotate = annotate
        self.fmt = fmt
        self.vmin = vmin
        self.vmax = vmax

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_correlation(
        cls,
        X: Any,
        *,
        feature_names: Sequence[str] | None = None,
        method: str = "pearson",
        **kwargs,
    ) -> HeatmapVisualizer:
        """Create a heatmap from a correlation matrix.

        Parameters
        ----------
        X : array-like or DataFrame
            Data to compute correlations from.
        feature_names : list of str, optional
            Feature names.
        method : str, default='pearson'
            Correlation method (only used if X is a pandas DataFrame).
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        # Handle pandas DataFrame
        if hasattr(X, "corr"):
            corr = X.corr(method=method)
            if feature_names is None:
                feature_names = list(corr.columns)
            corr_data = corr.values
        else:
            arr = np.asarray(X)
            corr_data = np.corrcoef(arr, rowvar=False)
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(arr.shape[1])]

        kwargs.setdefault("title", "Correlation Matrix")
        kwargs.setdefault("vmin", -1.0)
        kwargs.setdefault("vmax", 1.0)

        return cls(
            corr_data,
            x_labels=feature_names,
            y_labels=feature_names,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        data = self._data
        # Handle NaN
        data_list = []
        for row in data:
            data_list.append([None if np.isnan(v) else round(float(v), 6) for v in row])

        vmin = self.vmin if self.vmin is not None else float(np.nanmin(data))
        vmax = self.vmax if self.vmax is not None else float(np.nanmax(data))

        return {
            "matrix": data_list,
            "xLabels": self.x_labels,
            "yLabels": self.y_labels,
            "annotate": self.annotate,
            "fmt": self.fmt,
            "vmin": vmin,
            "vmax": vmax,
        }

    def _chart_type(self) -> str:
        return "heatmap"

    def _get_chart_js(self) -> str:
        return _HEATMAP_JS


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_HEATMAP_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const nRows = data.matrix.length;
  const nCols = data.matrix[0] ? data.matrix[0].length : 0;
  if (nRows === 0 || nCols === 0) return;

  const cellW = Math.min(60, Math.floor((config.width - 120) / nCols));
  const cellH = Math.min(60, Math.floor((config.height - 100) / nRows));
  const margin = {top: 20, right: 60, bottom: 60 + (nCols > 8 ? 20 : 0), left: 90};

  const plotW = nCols * cellW;
  const plotH = nRows * cellH;
  const totalW = plotW + margin.left + margin.right;
  const totalH = plotH + margin.top + margin.bottom;

  const svg = EG.svg('svg', {width: totalW, height: totalH});
  container.appendChild(svg);
  container.style.width = totalW + 'px';
  container.style.height = totalH + 'px';

  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);

  const palette = config.palette;
  const vmin = data.vmin;
  const vmax = data.vmax;
  const colorFn = EG.colorScale(palette, vmin, vmax);

  // Draw cells
  for (let r = 0; r < nRows; r++) {
    for (let c = 0; c < nCols; c++) {
      const v = data.matrix[r][c];
      const x = c * cellW;
      const y = r * cellH;
      const fill = v === null ? 'var(--bg-secondary)' : colorFn(v);

      const rect = EG.svg('rect', {
        x: x, y: y, width: cellW - 1, height: cellH - 1,
        fill: fill, rx: 2
      });
      rect.addEventListener('mouseenter', function(e) {
        rect.setAttribute('stroke', 'var(--text-primary)');
        rect.setAttribute('stroke-width', '2');
        const valStr = v === null ? 'N/A' : Number(v).toFixed(3);
        EG.tooltip.show(e, '<b>' + EG.esc(data.yLabels[r]) + ' × ' + EG.esc(data.xLabels[c]) + '</b><br>Value: ' + valStr);
      });
      rect.addEventListener('mousemove', function(e) {
        const valStr = v === null ? 'N/A' : Number(v).toFixed(3);
        EG.tooltip.show(e, '<b>' + EG.esc(data.yLabels[r]) + ' × ' + EG.esc(data.xLabels[c]) + '</b><br>Value: ' + valStr);
      });
      rect.addEventListener('mouseleave', function() {
        rect.removeAttribute('stroke');
        rect.removeAttribute('stroke-width');
        EG.tooltip.hide();
      });
      g.appendChild(rect);

      // Annotation
      if (data.annotate && v !== null) {
        const textColor = Math.abs(v - vmin) / (vmax - vmin || 1) > 0.4 &&
                          Math.abs(v - vmin) / (vmax - vmin || 1) < 0.6
                          ? 'var(--text-primary)' : (v > (vmin + vmax)/2 ? '#1a1a2e' : '#e0e0e0');
        const txt = EG.svg('text', {
          x: x + cellW / 2, y: y + cellH / 2 + 4,
          'text-anchor': 'middle', fill: textColor,
          'font-size': cellW < 35 ? '9px' : '11px', 'font-weight': '500'
        });
        txt.textContent = Number(v).toFixed(2);
        g.appendChild(txt);
      }
    }
  }

  // Y labels
  for (let r = 0; r < nRows; r++) {
    const lbl = EG.svg('text', {
      x: -8, y: r * cellH + cellH / 2 + 4,
      'text-anchor': 'end', fill: 'var(--text-secondary)', 'font-size': '11px'
    });
    lbl.textContent = data.yLabels[r].length > 12 ? data.yLabels[r].slice(0,10) + '…' : data.yLabels[r];
    g.appendChild(lbl);
  }

  // X labels
  for (let c = 0; c < nCols; c++) {
    const lbl = EG.svg('text', {
      x: c * cellW + cellW / 2, y: plotH + 16,
      'text-anchor': nCols > 8 ? 'end' : 'middle',
      fill: 'var(--text-secondary)', 'font-size': '11px',
      transform: nCols > 8 ? `rotate(-45,${c * cellW + cellW / 2},${plotH + 16})` : ''
    });
    lbl.textContent = data.xLabels[c].length > 12 ? data.xLabels[c].slice(0,10) + '…' : data.xLabels[c];
    g.appendChild(lbl);
  }

  // Color bar
  const cbW = 15, cbH = plotH;
  const cbX = plotW + 20;
  const cbG = EG.svg('g', {transform: `translate(${cbX}, 0)`});
  g.appendChild(cbG);
  const nSteps = 50;
  for (let i = 0; i < nSteps; i++) {
    const v = vmax - (i / nSteps) * (vmax - vmin);
    cbG.appendChild(EG.svg('rect', {
      x: 0, y: i * (cbH / nSteps), width: cbW, height: cbH / nSteps + 1,
      fill: colorFn(v)
    }));
  }
  cbG.appendChild(EG.svg('text', {x: cbW + 5, y: 10, fill: 'var(--text-secondary)', 'font-size': '10px'})).textContent = EG.fmt(vmax);
  cbG.appendChild(EG.svg('text', {x: cbW + 5, y: cbH, fill: 'var(--text-secondary)', 'font-size': '10px'})).textContent = EG.fmt(vmin);
}
"""
