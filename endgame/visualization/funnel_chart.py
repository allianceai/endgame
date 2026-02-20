"""Funnel chart visualizer.

Interactive funnel charts for showing progressive reduction through
stages — perfect for data pipeline attrition, conversion funnels,
or feature selection cascades.

Example
-------
>>> from endgame.visualization import FunnelChartVisualizer
>>> viz = FunnelChartVisualizer(
...     stages=["Raw Data", "Cleaned", "Featured", "Trained", "Predicted"],
...     values=[10000, 8500, 7200, 6800, 6500],
... )
>>> viz.save("funnel.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class FunnelChartVisualizer(BaseVisualizer):
    """Interactive funnel chart visualizer.

    Parameters
    ----------
    stages : list of str
        Stage names (top to bottom).
    values : list of float
        Values at each stage (should generally decrease).
    show_percentages : bool, default=True
        Show percentage of initial value and step-to-step retention.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=700
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        stages: Sequence[str],
        values: Sequence[float],
        *,
        show_percentages: bool = True,
        title: str = "",
        palette: str = "tableau",
        width: int = 700,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Funnel Chart", palette=palette, width=width, height=height, theme=theme)
        self._stages = list(stages)
        self._values = [float(v) for v in values]
        self.show_percentages = show_percentages

    @classmethod
    def from_pipeline(
        cls,
        stages: Sequence[str],
        sample_counts: Sequence[int],
        **kwargs,
    ) -> FunnelChartVisualizer:
        """Create from a data pipeline with sample counts.

        Parameters
        ----------
        stages : list of str
            Pipeline stage names.
        sample_counts : list of int
            Number of samples at each stage.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Data Pipeline")
        return cls(stages, sample_counts, **kwargs)

    @classmethod
    def from_feature_selection(
        cls,
        stages: Sequence[str],
        feature_counts: Sequence[int],
        **kwargs,
    ) -> FunnelChartVisualizer:
        """Create from feature selection stages.

        Parameters
        ----------
        stages : list of str
            Selection stage names (e.g., "All Features", "Variance Filter", etc.).
        feature_counts : list of int
            Number of features remaining at each stage.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Feature Selection Pipeline")
        return cls(stages, feature_counts, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "stages": self._stages,
            "values": self._values,
            "showPercentages": self.show_percentages,
        }

    def _chart_type(self) -> str:
        return "funnel"

    def _get_chart_js(self) -> str:
        return _FUNNEL_JS


_FUNNEL_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 30, bottom: 20, left: 30};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const stages = data.stages;
  const values = data.values;
  const n = stages.length;
  if (n === 0) return;

  const maxVal = Math.max.apply(null, values);
  const stageH = H / n;
  const gap = 3;
  const maxW = W * 0.85;

  // Width scale
  function barW(v) { return (v / maxVal) * maxW; }

  for (let i = 0; i < n; i++) {
    const w1 = barW(values[i]);
    const w2 = i < n - 1 ? barW(values[i + 1]) : w1 * 0.85;
    const x1 = (W - w1) / 2;
    const x2 = (W - w2) / 2;
    const y1 = i * stageH + gap;
    const y2 = (i + 1) * stageH - gap;
    const color = palette[i % palette.length];

    // Trapezoid
    const d = 'M' + x1 + ' ' + y1 +
              ' L' + (x1 + w1) + ' ' + y1 +
              ' L' + (x2 + w2) + ' ' + y2 +
              ' L' + x2 + ' ' + y2 + ' Z';

    const shape = EG.svg('path', {d: d, fill: color, opacity: 0.8, rx: 3});

    const pctOfTotal = maxVal > 0 ? (values[i] / maxVal * 100).toFixed(1) : 0;
    const retention = i > 0 && values[i - 1] > 0 ? (values[i] / values[i - 1] * 100).toFixed(1) : 100;

    shape.addEventListener('mouseenter', function(e) {
      shape.setAttribute('opacity', '1');
      let html = '<b>' + EG.esc(stages[i]) + '</b><br>Value: ' + EG.fmt(values[i], 0);
      if (data.showPercentages) {
        html += '<br>Of initial: ' + pctOfTotal + '%';
        if (i > 0) html += '<br>Retention: ' + retention + '%';
      }
      EG.tooltip.show(e, html);
    });
    shape.addEventListener('mouseleave', function() { shape.setAttribute('opacity', '0.8'); EG.tooltip.hide(); });
    g.appendChild(shape);

    // Stage label
    const cy = (y1 + y2) / 2;
    g.appendChild(EG.svg('text', {
      x: W / 2, y: cy - 4,
      'text-anchor': 'middle', fill: '#ffffff',
      'font-size': '12px', 'font-weight': '600'
    })).textContent = stages[i];

    // Value label
    let valText = EG.fmt(values[i], 0);
    if (data.showPercentages && i > 0) {
      valText += ' (' + pctOfTotal + '%)';
    }
    g.appendChild(EG.svg('text', {
      x: W / 2, y: cy + 14,
      'text-anchor': 'middle', fill: 'rgba(255,255,255,0.7)',
      'font-size': '10px'
    })).textContent = valText;
  }
}
"""
