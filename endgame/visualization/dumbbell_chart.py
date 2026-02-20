"""Dumbbell chart visualizer.

Interactive dumbbell (connected dot) charts for before/after or paired
comparisons. Shows two dots per category connected by a line, making
differences immediately visible.

Example
-------
>>> from endgame.visualization import DumbbellChartVisualizer
>>> viz = DumbbellChartVisualizer(
...     labels=["Model A", "Model B", "Model C"],
...     values_start=[0.85, 0.82, 0.88],
...     values_end=[0.91, 0.89, 0.93],
...     start_label="Baseline", end_label="Tuned",
... )
>>> viz.save("dumbbell.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class DumbbellChartVisualizer(BaseVisualizer):
    """Interactive dumbbell chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values_start : list of float
        Starting (left) values.
    values_end : list of float
        Ending (right) values.
    start_label : str, default='Start'
        Legend label for starting dots.
    end_label : str, default='End'
        Legend label for ending dots.
    sort_by : str, optional
        'diff' to sort by difference, 'end' to sort by end value,
        'start' to sort by start value, None for given order.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=750
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values_start: Sequence[float],
        values_end: Sequence[float],
        *,
        start_label: str = "Start",
        end_label: str = "End",
        sort_by: str | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 750,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Dumbbell Chart", palette=palette, width=width, height=height, theme=theme)

        labs = list(labels)
        vs = [float(v) for v in values_start]
        ve = [float(v) for v in values_end]

        if sort_by == "diff":
            order = sorted(range(len(labs)), key=lambda i: ve[i] - vs[i], reverse=True)
        elif sort_by == "end":
            order = sorted(range(len(labs)), key=lambda i: ve[i], reverse=True)
        elif sort_by == "start":
            order = sorted(range(len(labs)), key=lambda i: vs[i], reverse=True)
        else:
            order = list(range(len(labs)))

        self._labels = [labs[i] for i in order]
        self._values_start = [vs[i] for i in order]
        self._values_end = [ve[i] for i in order]
        self.start_label = start_label
        self.end_label = end_label

    @classmethod
    def from_metrics(
        cls,
        labels: Sequence[str],
        before: dict[str, float],
        after: dict[str, float],
        *,
        start_label: str = "Before",
        end_label: str = "After",
        **kwargs,
    ) -> DumbbellChartVisualizer:
        """Create from before/after metric dicts.

        Parameters
        ----------
        labels : list of str
            Metric names to compare.
        before : dict of str → float
            Metric name → value before.
        after : dict of str → float
            Metric name → value after.
        **kwargs
            Additional keyword arguments.
        """
        vs = [before[k] for k in labels]
        ve = [after[k] for k in labels]
        return cls(labels, vs, ve, start_label=start_label, end_label=end_label, **kwargs)

    @classmethod
    def from_train_test(
        cls,
        labels: Sequence[str],
        train_scores: Sequence[float],
        test_scores: Sequence[float],
        **kwargs,
    ) -> DumbbellChartVisualizer:
        """Create from train/test score pairs.

        Parameters
        ----------
        labels : list of str
            Model or metric names.
        train_scores : list of float
            Training scores.
        test_scores : list of float
            Test scores.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("start_label", "Train")
        kwargs.setdefault("end_label", "Test")
        kwargs.setdefault("title", "Train vs Test")
        return cls(labels, train_scores, test_scores, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "labels": self._labels,
            "valuesStart": self._values_start,
            "valuesEnd": self._values_end,
            "startLabel": self.start_label,
            "endLabel": self.end_label,
        }

    def _chart_type(self) -> str:
        return "dumbbell"

    def _get_chart_js(self) -> str:
        return _DUMBBELL_JS


_DUMBBELL_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 15, right: 30, bottom: 45, left: 140};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const labels = data.labels;
  const vs = data.valuesStart;
  const ve = data.valuesEnd;
  const n = labels.length;

  const startColor = palette[0];
  const endColor = palette[1 % palette.length];

  // X range
  let allVals = vs.concat(ve);
  let vMin = Math.min.apply(null, allVals);
  let vMax = Math.max.apply(null, allVals);
  const pad = (vMax - vMin) * 0.1 || 0.01;
  vMin -= pad; vMax += pad;

  const xScale = EG.scaleLinear([vMin, vMax], [0, W]);
  const rowH = H / n;

  EG.drawXAxis(g, xScale, H, '');

  for (let i = 0; i < n; i++) {
    const y = i * rowH + rowH / 2;
    const x1 = xScale(vs[i]);
    const x2 = xScale(ve[i]);
    const diff = ve[i] - vs[i];
    const diffStr = (diff >= 0 ? '+' : '') + EG.fmt(diff, 4);

    // Connector line
    g.appendChild(EG.svg('line', {
      x1: x1, y1: y, x2: x2, y2: y,
      stroke: 'var(--text-muted)', 'stroke-width': 2.5, opacity: 0.4
    }));

    // Start dot
    const d1 = EG.svg('circle', {cx: x1, cy: y, r: 6, fill: startColor, stroke: 'var(--bg-card)', 'stroke-width': 2});
    d1.addEventListener('mouseenter', function(e) {
      d1.setAttribute('r', '8');
      EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>' + data.startLabel + ': ' + EG.fmt(vs[i], 4));
    });
    d1.addEventListener('mouseleave', function() { d1.setAttribute('r', '6'); EG.tooltip.hide(); });
    g.appendChild(d1);

    // End dot
    const d2 = EG.svg('circle', {cx: x2, cy: y, r: 6, fill: endColor, stroke: 'var(--bg-card)', 'stroke-width': 2});
    d2.addEventListener('mouseenter', function(e) {
      d2.setAttribute('r', '8');
      EG.tooltip.show(e, '<b>' + EG.esc(labels[i]) + '</b><br>' + data.endLabel + ': ' + EG.fmt(ve[i], 4) + '<br>Δ = ' + diffStr);
    });
    d2.addEventListener('mouseleave', function() { d2.setAttribute('r', '6'); EG.tooltip.hide(); });
    g.appendChild(d2);

    // Label
    g.appendChild(EG.svg('text', {
      x: -8, y: y + 4, 'text-anchor': 'end',
      fill: 'var(--text-primary)', 'font-size': '11px'
    })).textContent = labels[i].length > 20 ? labels[i].slice(0, 18) + '…' : labels[i];
  }

  // Legend
  EG.drawLegend(container, [
    {label: data.startLabel, color: startColor},
    {label: data.endLabel, color: endColor},
  ]);
}
"""
