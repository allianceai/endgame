"""Lift and cumulative gains chart visualizer.

Interactive lift charts and cumulative gains plots for evaluating
ranking quality of classifiers. Shows how much better the model is
compared to a random baseline at various thresholds.

Example
-------
>>> from endgame.visualization import LiftChartVisualizer
>>> viz = LiftChartVisualizer.from_predictions(y_true, y_score)
>>> viz.save("lift.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class LiftChartVisualizer(BaseVisualizer):
    """Interactive lift / cumulative gains chart visualizer.

    Parameters
    ----------
    curves : list of dict
        Each dict has keys 'percentiles' (list of float 0-1),
        'gains' (list of float), 'lift' (list of float),
        'label' (str).
    mode : str, default='both'
        'gains' (cumulative gains only), 'lift' (lift only), or
        'both' (gains on left, lift on right — dual axis).
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=850
        Chart width.
    height : int, default=550
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        curves: Sequence[dict[str, Any]],
        *,
        mode: str = "both",
        title: str = "",
        palette: str = "tableau",
        width: int = 850,
        height: int = 550,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Lift & Cumulative Gains", palette=palette, width=width, height=height, theme=theme)
        self._curves = list(curves)
        self.mode = mode

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        y: Any,
        *,
        label: str | None = None,
        **kwargs,
    ) -> LiftChartVisualizer:
        """Create from a fitted classifier.

        Parameters
        ----------
        model : estimator
            Fitted classifier with ``predict_proba``.
        X : array-like
            Test features.
        y : array-like
            True binary labels.
        label : str, optional
            Model label.
        **kwargs
            Additional keyword arguments.
        """
        y_score = model.predict_proba(X)[:, 1]
        label = label or type(model).__name__
        return cls.from_predictions(np.asarray(y), y_score, label=label, **kwargs)

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_score: Any,
        *,
        label: str = "Model",
        n_points: int = 100,
        **kwargs,
    ) -> LiftChartVisualizer:
        """Create from predictions.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_score : array-like
            Predicted probabilities or decision scores.
        label : str, default='Model'
            Model label.
        n_points : int, default=100
            Number of evaluation points.
        **kwargs
            Additional keyword arguments.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_score = np.asarray(y_score, dtype=float)

        # Sort by score descending
        order = np.argsort(-y_score)
        y_sorted = y_true[order]

        n = len(y_sorted)
        total_pos = y_sorted.sum()
        prevalence = total_pos / n if n > 0 else 0

        percentiles = np.linspace(0, 1, n_points + 1)[1:]  # skip 0
        gains = []
        lifts = []

        for pct in percentiles:
            k = max(1, int(np.ceil(pct * n)))
            captured = y_sorted[:k].sum()
            gain = captured / total_pos if total_pos > 0 else 0
            lift = gain / pct if pct > 0 else 1
            gains.append(round(float(gain), 6))
            lifts.append(round(float(lift), 4))

        curves = [{
            "percentiles": [round(float(p), 4) for p in percentiles],
            "gains": gains,
            "lift": lifts,
            "label": label,
        }]
        return cls(curves, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "curves": self._curves,
            "mode": self.mode,
        }

    def _chart_type(self) -> str:
        return "lift_chart"

    def _get_chart_js(self) -> str:
        return _LIFT_JS


_LIFT_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const palette = config.palette;
  const curves = data.curves;
  const mode = data.mode;
  const showGains = mode === 'both' || mode === 'gains';
  const showLift = mode === 'both' || mode === 'lift';

  if (mode === 'both') {
    // Side by side: gains left, lift right
    const halfW = Math.floor(config.width / 2);
    const margin = {top: 20, right: 15, bottom: 55, left: 55};

    const svg = EG.svg('svg', {width: config.width, height: config.height});
    container.appendChild(svg);

    // ---- Cumulative Gains (left) ----
    const gL = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
    svg.appendChild(gL);
    const W1 = halfW - margin.left - margin.right;
    const H1 = config.height - margin.top - margin.bottom;

    const xS = EG.scaleLinear([0, 1], [0, W1]);
    const yS = EG.scaleLinear([0, 1], [H1, 0]);
    drawAxes(gL, xS, yS, W1, H1, '% Population', 'Cumulative Gains');

    // Random baseline
    gL.appendChild(EG.svg('line', {
      x1: xS(0), y1: yS(0), x2: xS(1), y2: yS(1),
      stroke: 'var(--text-muted)', 'stroke-width': 1.5, 'stroke-dasharray': '6,4', opacity: 0.5
    }));

    curves.forEach(function(c, ci) {
      drawLine(gL, c.percentiles, c.gains, xS, yS, palette[ci % palette.length], c.label);
    });

    gL.appendChild(EG.svg('text', {x: W1/2, y: -5, 'text-anchor':'middle', fill:'var(--text-primary)', 'font-size':'13px', 'font-weight':'600'})).textContent = 'Cumulative Gains';

    // ---- Lift Chart (right) ----
    const gR = EG.svg('g', {transform: `translate(${halfW + margin.left},${margin.top})`});
    svg.appendChild(gR);

    let maxLift = 1;
    curves.forEach(function(c) { c.lift.forEach(function(v) { if (v > maxLift) maxLift = v; }); });
    maxLift = Math.ceil(maxLift * 1.1);

    const xS2 = EG.scaleLinear([0, 1], [0, W1]);
    const yS2 = EG.scaleLinear([0, maxLift], [H1, 0]);
    drawAxes(gR, xS2, yS2, W1, H1, '% Population', 'Lift');

    // Baseline lift = 1
    gR.appendChild(EG.svg('line', {
      x1: 0, y1: yS2(1), x2: W1, y2: yS2(1),
      stroke: 'var(--text-muted)', 'stroke-width': 1.5, 'stroke-dasharray': '6,4', opacity: 0.5
    }));

    curves.forEach(function(c, ci) {
      drawLine(gR, c.percentiles, c.lift, xS2, yS2, palette[ci % palette.length], c.label + ' (Lift)');
    });

    gR.appendChild(EG.svg('text', {x: W1/2, y: -5, 'text-anchor':'middle', fill:'var(--text-primary)', 'font-size':'13px', 'font-weight':'600'})).textContent = 'Lift Chart';

  } else {
    // Single chart
    const margin = {top: 20, right: 20, bottom: 55, left: 55};
    const ctx = EG.createSVG(container, config.width, config.height, margin);
    const {g, width: W, height: H} = ctx;

    if (showGains) {
      const xS = EG.scaleLinear([0, 1], [0, W]);
      const yS = EG.scaleLinear([0, 1], [H, 0]);
      drawAxes(g, xS, yS, W, H, '% Population', 'Cumulative Gains');
      g.appendChild(EG.svg('line', {x1:xS(0),y1:yS(0),x2:xS(1),y2:yS(1), stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));
      curves.forEach(function(c, ci) { drawLine(g, c.percentiles, c.gains, xS, yS, palette[ci % palette.length], c.label); });
    } else {
      let maxLift = 1;
      curves.forEach(function(c) { c.lift.forEach(function(v) { if (v > maxLift) maxLift = v; }); });
      maxLift = Math.ceil(maxLift * 1.1);
      const xS = EG.scaleLinear([0, 1], [0, W]);
      const yS = EG.scaleLinear([0, maxLift], [H, 0]);
      drawAxes(g, xS, yS, W, H, '% Population', 'Lift');
      g.appendChild(EG.svg('line', {x1:0,y1:yS(1),x2:W,y2:yS(1), stroke:'var(--text-muted)','stroke-width':1.5,'stroke-dasharray':'6,4',opacity:0.5}));
      curves.forEach(function(c, ci) { drawLine(g, c.percentiles, c.lift, xS, yS, palette[ci % palette.length], c.label); });
    }
  }

  // Legend
  const items = curves.map(function(c, i) { return {label: c.label, color: palette[i % palette.length]}; });
  EG.drawLegend(container, items);

  function drawAxes(g, xS, yS, W, H, xLabel, yLabel) {
    // X axis
    var axG = EG.svg('g', {transform: `translate(0,${H})`});
    g.appendChild(axG);
    axG.appendChild(EG.svg('line', {x1:0,y1:0,x2:W,y2:0,stroke:'var(--border)'}));
    var ticks = EG.niceTicks(xS.domain[0], xS.domain[1], 5);
    ticks.forEach(function(v) {
      axG.appendChild(EG.svg('text', {x:xS(v), y:18, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'10px'})).textContent = EG.pct(v);
    });
    axG.appendChild(EG.svg('text', {x:W/2, y:40, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'11px', 'font-weight':'500'})).textContent = xLabel;
    // Y axis
    var yTicks = EG.niceTicks(yS.domain[0], yS.domain[1], 5);
    yTicks.forEach(function(v) {
      var y = yS(v);
      g.appendChild(EG.svg('line', {x1:0,y1:y,x2:W,y2:y,stroke:'var(--grid-line)'}));
      g.appendChild(EG.svg('text', {x:-8,y:y+4,'text-anchor':'end',fill:'var(--text-secondary)','font-size':'10px'})).textContent = EG.fmt(v, v >= 10 ? 0 : 2);
    });
    g.appendChild(EG.svg('text', {'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'11px', 'font-weight':'500',
      transform:`translate(-40,${(yS.range[0]+yS.range[1])/2}) rotate(-90)`})).textContent = yLabel;
  }

  function drawLine(g, xData, yData, xS, yS, color, label) {
    var d = '';
    for (var i = 0; i < xData.length; i++) {
      d += (i === 0 ? 'M' : ' L') + xS(xData[i]) + ' ' + yS(yData[i]);
    }
    // Fill
    var fillD = d + ' L' + xS(xData[xData.length-1]) + ' ' + yS.range[0] + ' L' + xS(xData[0]) + ' ' + yS.range[0] + ' Z';
    g.appendChild(EG.svg('path', {d:fillD, fill:color, opacity:0.06}));
    var path = EG.svg('path', {d:d, fill:'none', stroke:color, 'stroke-width':2.5});
    path.addEventListener('mouseenter', function(e) { path.setAttribute('stroke-width','4'); EG.tooltip.show(e, '<b>'+EG.esc(label)+'</b>'); });
    path.addEventListener('mouseleave', function() { path.setAttribute('stroke-width','2.5'); EG.tooltip.hide(); });
    g.appendChild(path);
  }
}
"""
