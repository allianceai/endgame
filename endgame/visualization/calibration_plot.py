"""Calibration plot (reliability diagram) visualizer.

Interactive calibration plots that show how well predicted probabilities
match actual frequencies. Essential companion to the ``eg.calibration``
module. Shows the calibration curve, histogram of predictions, and
calibration metrics (ECE, MCE).

Example
-------
>>> from endgame.visualization import CalibrationPlotVisualizer
>>> from sklearn.linear_model import LogisticRegression
>>> clf = LogisticRegression().fit(X_train, y_train)
>>> viz = CalibrationPlotVisualizer.from_estimator(clf, X_test, y_test)
>>> viz.save("calibration.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class CalibrationPlotVisualizer(BaseVisualizer):
    """Interactive calibration (reliability) diagram visualizer.

    Parameters
    ----------
    curves : list of dict
        Each dict has keys 'prob_true' (list of float),
        'prob_pred' (list of float), 'counts' (list of int),
        'label' (str), 'ece' (float), 'mce' (float).
    n_bins : int, default=10
        Number of bins (for reference; actual binning in constructors).
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=650
        Chart width.
    height : int, default=700
        Chart height (taller to fit histogram below).
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        curves: Sequence[dict[str, Any]],
        *,
        n_bins: int = 10,
        title: str = "",
        palette: str = "tableau",
        width: int = 650,
        height: int = 700,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Calibration Plot", palette=palette, width=width, height=height, theme=theme)
        self._curves = list(curves)
        self.n_bins = n_bins

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        y: Any,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        label: str | None = None,
        **kwargs,
    ) -> CalibrationPlotVisualizer:
        """Create from a fitted classifier.

        Parameters
        ----------
        model : estimator
            Fitted classifier with ``predict_proba``.
        X : array-like
            Test features.
        y : array-like
            True binary labels.
        n_bins : int, default=10
            Number of calibration bins.
        strategy : str, default='uniform'
            Binning strategy: 'uniform' or 'quantile'.
        label : str, optional
            Model label.
        **kwargs
            Additional keyword arguments.
        """
        y_prob = model.predict_proba(X)[:, 1]
        label = label or type(model).__name__
        return cls.from_predictions(
            np.asarray(y), y_prob,
            n_bins=n_bins, strategy=strategy, label=label, **kwargs,
        )

    @classmethod
    def from_predictions(
        cls,
        y_true: Any,
        y_prob: Any,
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        label: str = "Model",
        **kwargs,
    ) -> CalibrationPlotVisualizer:
        """Create from predictions.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        y_prob : array-like
            Predicted probabilities.
        n_bins : int, default=10
            Number of calibration bins.
        strategy : str, default='uniform'
            Binning strategy.
        label : str, default='Model'
            Model label.
        **kwargs
            Additional keyword arguments.
        """
        curve = _compute_calibration(
            np.asarray(y_true, dtype=float),
            np.asarray(y_prob, dtype=float),
            n_bins=n_bins,
            strategy=strategy,
            label=label,
        )
        kwargs.setdefault("n_bins", n_bins)
        return cls([curve], **kwargs)

    @classmethod
    def from_multiple(
        cls,
        y_true: Any,
        predictions: dict[str, Any],
        *,
        n_bins: int = 10,
        strategy: str = "uniform",
        **kwargs,
    ) -> CalibrationPlotVisualizer:
        """Compare calibration of multiple models.

        Parameters
        ----------
        y_true : array-like
            True binary labels.
        predictions : dict of str → array-like
            Model name → predicted probabilities.
        n_bins : int, default=10
            Number of bins.
        strategy : str, default='uniform'
            Binning strategy.
        **kwargs
            Additional keyword arguments.
        """
        y_arr = np.asarray(y_true, dtype=float)
        curves = []
        for name, y_prob in predictions.items():
            curve = _compute_calibration(y_arr, np.asarray(y_prob, dtype=float),
                                         n_bins=n_bins, strategy=strategy, label=name)
            curves.append(curve)
        kwargs.setdefault("n_bins", n_bins)
        kwargs.setdefault("title", "Calibration Comparison")
        return cls(curves, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "curves": self._curves,
            "nBins": self.n_bins,
        }

    def _chart_type(self) -> str:
        return "calibration"

    def _get_chart_js(self) -> str:
        return _CALIBRATION_JS


def _compute_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    strategy: str,
    label: str,
) -> dict[str, Any]:
    """Compute calibration curve data."""
    if strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(y_prob, quantiles * 100)
        bin_edges = np.unique(bin_edges)
    else:
        bin_edges = np.linspace(0, 1, n_bins + 1)

    prob_true = []
    prob_pred = []
    counts = []
    hist_bins = []

    for i in range(len(bin_edges) - 1):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == len(bin_edges) - 2:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        n_in_bin = int(mask.sum())
        counts.append(n_in_bin)
        hist_bins.append(round(float((lo + hi) / 2), 4))

        if n_in_bin > 0:
            prob_true.append(round(float(y_true[mask].mean()), 6))
            prob_pred.append(round(float(y_prob[mask].mean()), 6))
        else:
            prob_true.append(None)
            prob_pred.append(None)

    # ECE and MCE
    ece = 0.0
    mce = 0.0
    total = len(y_prob)
    for pt, pp, c in zip(prob_true, prob_pred, counts):
        if pt is not None and pp is not None and c > 0:
            gap = abs(pt - pp)
            ece += gap * c / total
            mce = max(mce, gap)

    return {
        "probTrue": prob_true,
        "probPred": prob_pred,
        "counts": counts,
        "histBins": hist_bins,
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4),
        "label": label,
    }


_CALIBRATION_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const palette = config.palette;
  const curves = data.curves;

  // Main calibration plot (top 65%)
  const mainH = config.height * 0.62;
  const histH = config.height * 0.25;
  const margin = {top: 20, right: 20, bottom: 35, left: 55};

  const svg = EG.svg('svg', {width: config.width, height: config.height});
  container.appendChild(svg);

  // ---- Calibration diagram ----
  const gMain = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(gMain);
  const W = config.width - margin.left - margin.right;
  const H = mainH - margin.top - 15;

  const xScale = EG.scaleLinear([0, 1], [0, W]);
  const yScale = EG.scaleLinear([0, 1], [H, 0]);

  // Grid
  const ticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0];
  ticks.forEach(function(v) {
    const y = yScale(v);
    gMain.appendChild(EG.svg('line', {x1:0, y1:y, x2:W, y2:y, stroke:'var(--grid-line)', 'stroke-width':1}));
    gMain.appendChild(EG.svg('text', {x:-8, y:y+4, 'text-anchor':'end', fill:'var(--text-secondary)', 'font-size':'10px'})).textContent = v.toFixed(1);
    const x = xScale(v);
    gMain.appendChild(EG.svg('text', {x:x, y:H+15, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'10px'})).textContent = v.toFixed(1);
  });

  // Perfect calibration line
  gMain.appendChild(EG.svg('line', {
    x1: xScale(0), y1: yScale(0), x2: xScale(1), y2: yScale(1),
    stroke: 'var(--text-muted)', 'stroke-width': 1.5,
    'stroke-dasharray': '6,4', opacity: 0.6
  }));
  gMain.appendChild(EG.svg('text', {x: W-5, y: yScale(0.95), 'text-anchor': 'end', fill: 'var(--text-muted)', 'font-size': '10px'})).textContent = 'Perfect';

  // Axes labels
  gMain.appendChild(EG.svg('text', {x: W/2, y: H+30, 'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '12px', 'font-weight': '500'})).textContent = 'Mean Predicted Probability';
  gMain.appendChild(EG.svg('text', {'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '12px', 'font-weight': '500',
    transform: `translate(-40,${H/2}) rotate(-90)`})).textContent = 'Fraction of Positives';

  // Draw curves
  curves.forEach(function(c, ci) {
    const color = palette[ci % palette.length];
    const n = c.probTrue.length;

    // Calibration line
    let d = '';
    let pts = [];
    for (let i = 0; i < n; i++) {
      if (c.probTrue[i] === null || c.probPred[i] === null) continue;
      const px = xScale(c.probPred[i]);
      const py = yScale(c.probTrue[i]);
      pts.push({x: px, y: py, pred: c.probPred[i], true_: c.probTrue[i], count: c.counts[i]});
    }
    for (let i = 0; i < pts.length; i++) {
      d += (i === 0 ? 'M' : ' L') + pts[i].x + ' ' + pts[i].y;
    }
    if (d) {
      gMain.appendChild(EG.svg('path', {d: d, fill:'none', stroke:color, 'stroke-width':2.5, 'stroke-linejoin':'round'}));
    }

    // Points
    pts.forEach(function(p) {
      const dot = EG.svg('circle', {cx:p.x, cy:p.y, r:5, fill:color, stroke:'var(--bg-card)', 'stroke-width':2});
      dot.addEventListener('mouseenter', function(e) {
        dot.setAttribute('r', '7');
        EG.tooltip.show(e,
          '<b>' + EG.esc(c.label) + '</b><br>' +
          'Predicted: ' + EG.fmt(p.pred, 3) + '<br>' +
          'Actual: ' + EG.fmt(p.true_, 3) + '<br>' +
          'n = ' + p.count);
      });
      dot.addEventListener('mouseleave', function() { dot.setAttribute('r', '5'); EG.tooltip.hide(); });
      gMain.appendChild(dot);
    });
  });

  // ECE/MCE annotation
  const metricsY = 16;
  curves.forEach(function(c, ci) {
    const color = palette[ci % palette.length];
    const x = 10 + ci * 200;
    gMain.appendChild(EG.svg('text', {x: x, y: metricsY, fill: color, 'font-size': '11px', 'font-weight': '600'}))
      .textContent = c.label + ': ECE=' + c.ece.toFixed(3) + ', MCE=' + c.mce.toFixed(3);
  });

  // ---- Histogram below ----
  const gHist = EG.svg('g', {transform: `translate(${margin.left},${mainH + 15})`});
  svg.appendChild(gHist);
  const hH = histH - 20;

  // Find max count
  let maxCount = 0;
  curves.forEach(function(c) { c.counts.forEach(function(cnt) { if (cnt > maxCount) maxCount = cnt; }); });
  if (maxCount === 0) maxCount = 1;

  const hYScale = EG.scaleLinear([0, maxCount], [hH, 0]);

  gHist.appendChild(EG.svg('line', {x1:0, y1:hH, x2:W, y2:hH, stroke:'var(--border)'}));
  gHist.appendChild(EG.svg('text', {x:W/2, y:hH+18, 'text-anchor':'middle', fill:'var(--text-muted)', 'font-size':'10px'})).textContent = 'Prediction Distribution';

  curves.forEach(function(c, ci) {
    const color = palette[ci % palette.length];
    const n = c.histBins.length;
    const barW = W / n * 0.7;
    const offset = ci * barW * 0.3;

    c.counts.forEach(function(cnt, i) {
      const x = xScale(c.histBins[i]) - barW/2 + offset;
      const barH = hH - hYScale(cnt);
      gHist.appendChild(EG.svg('rect', {
        x: x, y: hH - barH, width: barW, height: Math.max(barH, 0),
        fill: color, opacity: 0.5, rx: 2
      }));
    });
  });

  // Legend
  const items = curves.map(function(c, i) {
    return {label: c.label, color: palette[i % palette.length]};
  });
  EG.drawLegend(container, items);
}
"""
