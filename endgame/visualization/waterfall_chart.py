"""Waterfall chart visualizer.

Interactive waterfall charts for explaining individual predictions
(SHAP-style) or showing sequential value contributions. Essential
for model interpretability.

Example
-------
>>> from endgame.visualization import WaterfallVisualizer
>>> viz = WaterfallVisualizer.from_shap(shap_values, feature_names, base_value)
>>> viz.save("waterfall.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class WaterfallVisualizer(BaseVisualizer):
    """Interactive waterfall chart visualizer.

    Parameters
    ----------
    categories : list of str
        Category/feature labels.
    values : list of float
        Contribution values (positive = increase, negative = decrease).
    base_value : float, optional
        Starting/baseline value (shown at bottom).
    final_value : float, optional
        Final value (shown at top). If None, computed as base + sum(values).
    show_connectors : bool, default=True
        Show connector lines between bars.
    sort_by : str, optional
        'abs' to sort by absolute contribution, None for given order.
    max_display : int, optional
        Maximum features to display. Remaining are grouped into "Other".
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=750
        Chart width.
    height : int, default=550
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        categories: Sequence[str],
        values: Sequence[float],
        *,
        base_value: float | None = None,
        final_value: float | None = None,
        show_connectors: bool = True,
        sort_by: str | None = None,
        max_display: int | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 750,
        height: int = 550,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Waterfall Chart", palette=palette, width=width, height=height, theme=theme)

        cats = list(categories)
        vals = list(values)

        # Sort by absolute value if requested
        if sort_by == "abs":
            order = sorted(range(len(vals)), key=lambda i: abs(vals[i]), reverse=True)
            cats = [cats[i] for i in order]
            vals = [vals[i] for i in order]

        # Truncate if max_display
        if max_display is not None and len(vals) > max_display:
            display_cats = cats[:max_display]
            display_vals = vals[:max_display]
            other_val = sum(vals[max_display:])
            display_cats.append(f"Other ({len(vals) - max_display} features)")
            display_vals.append(other_val)
            cats = display_cats
            vals = display_vals

        self._categories = cats
        self._values = [round(float(v), 6) for v in vals]
        self.base_value = float(base_value) if base_value is not None else 0.0
        self.final_value = (
            float(final_value) if final_value is not None
            else self.base_value + sum(vals)
        )
        self.show_connectors = show_connectors

    @classmethod
    def from_shap(
        cls,
        shap_values: Any,
        feature_names: Sequence[str],
        base_value: float,
        *,
        max_display: int = 15,
        **kwargs,
    ) -> WaterfallVisualizer:
        """Create from SHAP values for a single prediction.

        Parameters
        ----------
        shap_values : array-like of shape (n_features,)
            SHAP values for one sample.
        feature_names : list of str
            Feature names.
        base_value : float
            Expected value (model output mean).
        max_display : int, default=15
            Max features to display.
        **kwargs
            Additional keyword arguments.
        """
        sv = np.asarray(shap_values).ravel()
        cats = list(feature_names)

        if len(cats) != len(sv):
            raise ValueError(f"Length mismatch: {len(cats)} names vs {len(sv)} values")

        kwargs.setdefault("sort_by", "abs")
        kwargs.setdefault("max_display", max_display)
        return cls(cats, sv, base_value=base_value, **kwargs)

    @classmethod
    def from_contributions(
        cls,
        categories: Sequence[str],
        values: Sequence[float],
        *,
        base_value: float = 0.0,
        **kwargs,
    ) -> WaterfallVisualizer:
        """Create from sequential contributions.

        Parameters
        ----------
        categories : list of str
            Step/category labels.
        values : list of float
            Contribution values.
        base_value : float, default=0.0
            Starting value.
        **kwargs
            Additional keyword arguments.
        """
        return cls(categories, values, base_value=base_value, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "categories": self._categories,
            "values": self._values,
            "baseValue": round(self.base_value, 6),
            "finalValue": round(self.final_value, 6),
            "showConnectors": self.show_connectors,
        }

    def _chart_type(self) -> str:
        return "waterfall"

    def _get_chart_js(self) -> str:
        return _WATERFALL_JS


_WATERFALL_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const n = data.categories.length;
  const needsBase = data.baseValue !== 0;
  const totalRows = n + (needsBase ? 2 : 1); // bars + base + final
  const rowH = Math.min(32, (config.height - 120) / totalRows);
  const actualH = rowH * totalRows + 120;
  const margin = {top: 20, right: 30, bottom: 55, left: 180};
  const W = config.width - margin.left - margin.right;
  const H = rowH * totalRows;

  const svg = EG.svg('svg', {width: config.width, height: actualH});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);

  const palette = config.palette;
  const posColor = '#2ca02c';
  const negColor = '#d62728';
  const totalColor = palette[0];

  // Build cumulative running values
  const baseVal = data.baseValue;
  const vals = data.values;
  const cats = data.categories;
  const running = [baseVal];
  for (let i = 0; i < n; i++) {
    running.push(running[i] + vals[i]);
  }
  const finalVal = data.finalValue;

  // X scale range: min/max of all running values + base + final
  let allVals = running.concat([finalVal, baseVal]);
  let xMin = Math.min.apply(null, allVals);
  let xMax = Math.max.apply(null, allVals);
  const xPad = (xMax - xMin) * 0.12 || 0.1;
  xMin -= xPad; xMax += xPad;

  const xScale = EG.scaleLinear([xMin, xMax], [0, W]);
  const barH = rowH * 0.65;

  // Helper
  function drawBar(yIdx, startVal, endVal, color, label, tooltip) {
    const y = yIdx * rowH;
    const x1 = xScale(Math.min(startVal, endVal));
    const x2 = xScale(Math.max(startVal, endVal));
    const w = Math.max(x2 - x1, 2);

    const rect = EG.svg('rect', {
      x: x1, y: y + (rowH - barH) / 2,
      width: w, height: barH,
      fill: color, rx: 3, opacity: 0.85
    });
    rect.addEventListener('mouseenter', function(e) {
      rect.setAttribute('opacity', '1');
      EG.tooltip.show(e, tooltip);
    });
    rect.addEventListener('mouseleave', function() {
      rect.setAttribute('opacity', '0.85');
      EG.tooltip.hide();
    });
    g.appendChild(rect);

    // Label on bar
    const textX = endVal >= startVal ? x2 + 5 : x1 - 5;
    const anchor = endVal >= startVal ? 'start' : 'end';
    const valText = EG.svg('text', {
      x: textX, y: y + rowH / 2 + 4,
      'text-anchor': anchor, fill: color,
      'font-size': '10px', 'font-weight': '600'
    });
    valText.textContent = (endVal - startVal >= 0 ? '+' : '') + EG.fmt(endVal - startVal, 3);
    g.appendChild(valText);

    // Category label (left)
    const catText = EG.svg('text', {
      x: -8, y: y + rowH / 2 + 4,
      'text-anchor': 'end', fill: 'var(--text-primary)',
      'font-size': '11px'
    });
    catText.textContent = label.length > 25 ? label.slice(0, 23) + '…' : label;
    g.appendChild(catText);
  }

  // Draw rows
  let row = 0;

  // Base value
  if (needsBase) {
    drawBar(row, 0, baseVal, totalColor, 'Base value',
      '<b>Base value</b><br>' + EG.fmt(baseVal, 4));
    row++;
  }

  // Contribution bars
  for (let i = 0; i < n; i++) {
    const startV = running[i];
    const endV = running[i + 1];
    const color = vals[i] >= 0 ? posColor : negColor;
    drawBar(row + i, startV, endV, color, cats[i],
      '<b>' + EG.esc(cats[i]) + '</b><br>' +
      'Contribution: ' + (vals[i] >= 0 ? '+' : '') + EG.fmt(vals[i], 4) + '<br>' +
      'Running: ' + EG.fmt(endV, 4));

    // Connector line
    if (data.showConnectors && i < n - 1) {
      const y1 = (row + i) * rowH + (rowH + barH) / 2;
      const y2 = (row + i + 1) * rowH + (rowH - barH) / 2;
      g.appendChild(EG.svg('line', {
        x1: xScale(endV), y1: y1, x2: xScale(endV), y2: y2,
        stroke: 'var(--text-muted)', 'stroke-width': 1,
        'stroke-dasharray': '3,2', opacity: 0.5
      }));
    }
  }
  row += n;

  // Connector to final
  if (data.showConnectors && n > 0) {
    const lastEnd = running[n];
    const y1 = (row - 1) * rowH + (rowH + barH) / 2;
    const y2 = row * rowH + (rowH - barH) / 2;
    g.appendChild(EG.svg('line', {
      x1: xScale(lastEnd), y1: y1, x2: xScale(lastEnd), y2: y2,
      stroke: 'var(--text-muted)', 'stroke-width': 1,
      'stroke-dasharray': '3,2', opacity: 0.5
    }));
  }

  // Final total
  drawBar(row, 0, finalVal, totalColor, 'Prediction',
    '<b>Final prediction</b><br>' + EG.fmt(finalVal, 4));

  // X axis
  const axG = EG.svg('g', {transform: `translate(0,${H + 10})`});
  g.appendChild(axG);
  axG.appendChild(EG.svg('line', {x1: 0, y1: 0, x2: W, y2: 0, stroke: 'var(--border)'}));
  const ticks = EG.niceTicks(xMin, xMax, 6);
  ticks.forEach(function(v) {
    axG.appendChild(EG.svg('line', {x1: xScale(v), y1: 0, x2: xScale(v), y2: 5, stroke: 'var(--text-muted)'}));
    axG.appendChild(EG.svg('text', {x: xScale(v), y: 18, 'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '10px'})).textContent = EG.fmt(v, 2);
  });
  axG.appendChild(EG.svg('text', {x: W/2, y: 38, 'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '11px', 'font-weight': '500'})).textContent = 'Model Output';

  // Zero line
  if (xMin <= 0 && xMax >= 0) {
    g.appendChild(EG.svg('line', {
      x1: xScale(0), y1: 0, x2: xScale(0), y2: H,
      stroke: 'var(--text-muted)', 'stroke-width': 1, 'stroke-dasharray': '4,3', opacity: 0.4
    }));
  }
}
"""
