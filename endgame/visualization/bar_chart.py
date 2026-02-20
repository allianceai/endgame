"""Bar chart and stacked bar chart visualizer.

Interactive bar charts for feature importances, model comparisons, and
categorical distributions. Supports horizontal/vertical orientation,
stacked mode, and sorting.

Example
-------
>>> from endgame.visualization import BarChartVisualizer
>>> viz = BarChartVisualizer(
...     labels=["feat_1", "feat_2", "feat_3"],
...     values=[0.35, 0.25, 0.15],
...     title="Feature Importances",
... )
>>> viz.save("importances.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class BarChartVisualizer(BaseVisualizer):
    """Interactive bar chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float or list of list of float
        Bar values. For stacked bars, pass a list of lists where each
        inner list is one stack group.
    series_names : list of str, optional
        Names for each series (stacked mode).
    orientation : str, default='vertical'
        'vertical' or 'horizontal'.
    sort : bool, default=False
        If True, sort bars by value (descending).
    x_label : str, optional
        X-axis label.
    y_label : str, optional
        Y-axis label.
    show_values : bool, default=True
        Show value labels on bars.
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
        labels: Sequence[str],
        values: Sequence[float] | Sequence[Sequence[float]],
        *,
        series_names: Sequence[str] | None = None,
        orientation: str = "vertical",
        sort: bool = False,
        x_label: str = "",
        y_label: str = "",
        show_values: bool = True,
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.orientation = orientation
        self.sort = sort
        self.x_label = x_label
        self.y_label = y_label
        self.show_values = show_values

        # Normalize values to list-of-lists for stacked support
        arr = np.asarray(values)
        if arr.ndim == 1:
            self._values = [arr.tolist()]
            self._series_names = series_names or [""]
        else:
            self._values = arr.tolist()
            self._series_names = list(series_names) if series_names else [f"Series {i}" for i in range(len(self._values))]

        self._is_stacked = len(self._values) > 1

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_importances(
        cls,
        model: Any,
        *,
        feature_names: Sequence[str] | None = None,
        top_n: int = 20,
        **kwargs,
    ) -> BarChartVisualizer:
        """Create a bar chart from a fitted model's feature importances.

        Parameters
        ----------
        model : estimator
            Fitted sklearn-compatible model with ``feature_importances_``.
        feature_names : list of str, optional
            Feature names. If None, uses ``feature_names_in_`` or generic names.
        top_n : int, default=20
            Show top N features.
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        importances = np.asarray(model.feature_importances_)
        if feature_names is None:
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        # Sort and take top_n
        idx = np.argsort(importances)[::-1][:top_n]
        labels = [feature_names[i] for i in idx]
        vals = importances[idx].tolist()

        kwargs.setdefault("title", "Feature Importances")
        kwargs.setdefault("y_label", "Importance")
        kwargs.setdefault("orientation", "horizontal")
        # Reverse for horizontal (most important at top)
        if kwargs.get("orientation") == "horizontal":
            labels = labels[::-1]
            vals = vals[::-1]

        return cls(labels=labels, values=vals, **kwargs)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, float],
        **kwargs,
    ) -> BarChartVisualizer:
        """Create a bar chart from a dictionary.

        Parameters
        ----------
        data : dict
            Mapping of label → value.
        **kwargs
            Additional keyword arguments passed to the constructor.
        """
        labels = list(data.keys())
        values = list(data.values())
        return cls(labels=labels, values=values, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        labels = list(self.labels)
        values = [list(s) for s in self._values]

        if self.sort and not self._is_stacked:
            # Sort by value descending
            pairs = sorted(zip(labels, values[0]), key=lambda p: p[1], reverse=True)
            labels = [p[0] for p in pairs]
            values = [[p[1] for p in pairs]]

        return {
            "labels": labels,
            "values": values,
            "seriesNames": list(self._series_names),
            "stacked": self._is_stacked,
            "orientation": self.orientation,
            "showValues": self.show_values,
            "xLabel": self.x_label,
            "yLabel": self.y_label,
        }

    def _chart_type(self) -> str:
        return "bar"

    def _get_chart_js(self) -> str:
        return _BAR_CHART_JS


# ---------------------------------------------------------------------------
# JavaScript renderer
# ---------------------------------------------------------------------------

_BAR_CHART_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const isHorizontal = data.orientation === 'horizontal';
  const margin = isHorizontal
    ? {top: 20, right: 40, bottom: 40, left: 120}
    : {top: 20, right: 20, bottom: 60, left: 60};

  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const labels = data.labels;
  const allValues = data.values;
  const stacked = data.stacked;
  const nSeries = allValues.length;

  // Compute totals for stacking or max for simple
  let maxVal = 0;
  if (stacked) {
    for (let i = 0; i < labels.length; i++) {
      let sum = 0;
      for (let s = 0; s < nSeries; s++) sum += allValues[s][i];
      if (sum > maxVal) maxVal = sum;
    }
  } else {
    for (let s = 0; s < nSeries; s++) {
      for (let i = 0; i < labels.length; i++) {
        if (allValues[s][i] > maxVal) maxVal = allValues[s][i];
      }
    }
  }
  maxVal = maxVal * 1.08 || 1;

  if (isHorizontal) {
    const catScale = EG.scaleBand(labels, [0, H], 0.15);
    const valScale = EG.scaleLinear([0, maxVal], [0, W]);

    EG.drawXAxis(g, valScale, H, data.xLabel || data.yLabel);
    EG.drawYAxis(g, catScale, W, data.yLabel || data.xLabel, true);

    const bw = catScale.bandwidth();
    labels.forEach(function(lbl, i) {
      let xOffset = 0;
      for (let s = 0; s < nSeries; s++) {
        const v = allValues[s][i];
        const barW = valScale(v);
        const color = palette[s % palette.length];
        const y = catScale(lbl);
        const rect = EG.svg('rect', {
          x: xOffset, y: y, width: Math.max(barW, 0), height: bw,
          fill: color, rx: 3, opacity: 0.9
        });
        rect.addEventListener('mouseenter', function(e) {
          rect.setAttribute('opacity', '1');
          const sName = data.seriesNames[s] ? data.seriesNames[s] + ': ' : '';
          EG.tooltip.show(e, '<b>' + EG.esc(lbl) + '</b><br>' + sName + EG.fmt(v));
        });
        rect.addEventListener('mousemove', function(e) { EG.tooltip.show(e, '<b>' + EG.esc(lbl) + '</b><br>' + EG.fmt(v)); });
        rect.addEventListener('mouseleave', function() { rect.setAttribute('opacity', '0.9'); EG.tooltip.hide(); });
        g.appendChild(rect);

        if (data.showValues && !stacked) {
          const txt = EG.svg('text', {
            x: barW + 5, y: y + bw / 2 + 4,
            fill: 'var(--text-secondary)', 'font-size': '10px'
          });
          txt.textContent = EG.fmt(v);
          g.appendChild(txt);
        }
        xOffset += barW;
      }
    });
  } else {
    // Vertical
    const catScale = EG.scaleBand(labels, [0, W], 0.15);
    const valScale = EG.scaleLinear([0, maxVal], [H, 0]);

    EG.drawXAxis(g, catScale, H, data.xLabel, true);
    EG.drawYAxis(g, valScale, W, data.yLabel);

    const bw = catScale.bandwidth();
    labels.forEach(function(lbl, i) {
      let yOffset = H;
      for (let s = 0; s < nSeries; s++) {
        const v = allValues[s][i];
        const barH = H - valScale(v);
        const color = palette[s % palette.length];
        const x = catScale(lbl);
        const rect = EG.svg('rect', {
          x: x, y: yOffset - barH, width: bw, height: Math.max(barH, 0),
          fill: color, rx: 3, opacity: 0.9
        });
        rect.addEventListener('mouseenter', function(e) {
          rect.setAttribute('opacity', '1');
          const sName = data.seriesNames[s] ? data.seriesNames[s] + ': ' : '';
          EG.tooltip.show(e, '<b>' + EG.esc(lbl) + '</b><br>' + sName + EG.fmt(v));
        });
        rect.addEventListener('mousemove', function(e) { EG.tooltip.show(e, '<b>' + EG.esc(lbl) + '</b><br>' + EG.fmt(v)); });
        rect.addEventListener('mouseleave', function() { rect.setAttribute('opacity', '0.9'); EG.tooltip.hide(); });
        g.appendChild(rect);

        if (data.showValues && !stacked && nSeries === 1) {
          const txt = EG.svg('text', {
            x: x + bw / 2, y: yOffset - barH - 5,
            'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '10px'
          });
          txt.textContent = EG.fmt(v);
          g.appendChild(txt);
        }
        yOffset -= barH;
      }
    });
  }

  // Legend for stacked
  if (stacked) {
    const items = data.seriesNames.map(function(name, i) {
      return {label: name, color: palette[i % palette.length]};
    });
    EG.drawLegend(container, items);
  }
}
"""
