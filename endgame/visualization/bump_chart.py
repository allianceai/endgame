"""Bump chart visualizer.

Interactive bump charts for visualizing ranking changes over time or
across categories. Perfect for leaderboard tracking, experiment
comparison, or any ordinal ranking evolution.

Example
-------
>>> from endgame.visualization import BumpChartVisualizer
>>> rankings = {
...     "LGBM":    [1, 1, 2, 1],
...     "XGBoost": [2, 3, 1, 2],
...     "CatBoost": [3, 2, 3, 3],
... }
>>> viz = BumpChartVisualizer(x=["Fold 1", "Fold 2", "Fold 3", "Fold 4"], rankings=rankings)
>>> viz.save("bump.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class BumpChartVisualizer(BaseVisualizer):
    """Interactive bump chart visualizer.

    Parameters
    ----------
    x : list
        X-axis labels (e.g. time points, fold names, rounds).
    rankings : dict of str → list of int
        Series name → ranking at each x point (1 = best).
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
        x: Sequence,
        rankings: dict[str, Sequence[int]],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title or "Bump Chart", palette=palette, width=width, height=height, theme=theme)
        self._x = [str(v) for v in x]
        self._rankings = {k: [int(v) for v in vals] for k, vals in rankings.items()}

    @classmethod
    def from_scores(
        cls,
        x: Sequence,
        scores: dict[str, Sequence[float]],
        *,
        higher_is_better: bool = True,
        **kwargs,
    ) -> BumpChartVisualizer:
        """Create from raw scores, converting to rankings.

        Parameters
        ----------
        x : list
            X-axis labels.
        scores : dict of str → list of float
            Series name → scores at each point.
        higher_is_better : bool, default=True
            If True, highest score → rank 1.
        **kwargs
            Additional keyword arguments.
        """
        names = list(scores.keys())
        n_points = len(x)
        rankings = {name: [] for name in names}

        for t in range(n_points):
            vals = [(name, scores[name][t]) for name in names]
            vals.sort(key=lambda p: p[1], reverse=higher_is_better)
            for rank, (name, _) in enumerate(vals, 1):
                rankings[name].append(rank)

        kwargs.setdefault("title", "Rankings by Score")
        return cls(x, rankings, **kwargs)

    @classmethod
    def from_cv_scores(
        cls,
        scores: dict[str, Sequence[float]],
        *,
        higher_is_better: bool = True,
        **kwargs,
    ) -> BumpChartVisualizer:
        """Create from cross-validation fold scores.

        Parameters
        ----------
        scores : dict of str → list of float
            Model name → fold scores.
        higher_is_better : bool, default=True
            If True, highest score → rank 1.
        **kwargs
            Additional keyword arguments.
        """
        n_folds = len(next(iter(scores.values())))
        x = [f"Fold {i+1}" for i in range(n_folds)]
        kwargs.setdefault("title", "Model Rankings by Fold")
        return cls.from_scores(x, scores, higher_is_better=higher_is_better, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        max_rank = 1
        series = []
        for name, ranks in self._rankings.items():
            for r in ranks:
                if r > max_rank:
                    max_rank = r
            series.append({"name": name, "ranks": ranks})

        return {
            "x": self._x,
            "series": series,
            "maxRank": max_rank,
        }

    def _chart_type(self) -> str:
        return "bump_chart"

    def _get_chart_js(self) -> str:
        return _BUMP_JS


_BUMP_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 25, right: 120, bottom: 45, left: 55};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;

  const xs = data.x;
  const nX = xs.length;
  const maxRank = data.maxRank;
  const series = data.series;

  // X: evenly spaced
  const xStep = W / Math.max(nX - 1, 1);
  function xPos(i) { return i * xStep; }

  // Y: rank 1 at top, maxRank at bottom
  const yScale = EG.scaleLinear([1, maxRank], [20, H - 20]);

  // X axis
  xs.forEach(function(label, i) {
    const x = xPos(i);
    g.appendChild(EG.svg('line', {x1: x, y1: H + 5, x2: x, y2: -5, stroke: 'var(--grid-line)', 'stroke-width': 1}));
    g.appendChild(EG.svg('text', {x: x, y: H + 22, 'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '11px'})).textContent = label;
  });

  // Y axis (rank labels)
  for (var r = 1; r <= maxRank; r++) {
    var y = yScale(r);
    g.appendChild(EG.svg('line', {x1: -5, y1: y, x2: W + 5, y2: y, stroke: 'var(--grid-line)', 'stroke-width': 0.5}));
    g.appendChild(EG.svg('text', {x: -12, y: y + 4, 'text-anchor': 'end', fill: 'var(--text-muted)', 'font-size': '10px'})).textContent = '#' + r;
  }

  // Draw series
  series.forEach(function(s, si) {
    const color = palette[si % palette.length];
    const ranks = s.ranks;

    // Smooth path using cardinal-ish curve
    let d = '';
    for (let i = 0; i < nX; i++) {
      const x = xPos(i), y = yScale(ranks[i]);
      if (i === 0) { d += 'M' + x + ' ' + y; }
      else {
        // Simple cubic bezier for smoothness
        const px = xPos(i - 1), py = yScale(ranks[i - 1]);
        const cx1 = px + xStep * 0.4, cx2 = x - xStep * 0.4;
        d += ' C' + cx1 + ' ' + py + ' ' + cx2 + ' ' + y + ' ' + x + ' ' + y;
      }
    }

    const path = EG.svg('path', {
      d: d, fill: 'none', stroke: color,
      'stroke-width': 3, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'
    });
    path.addEventListener('mouseenter', function() { path.setAttribute('stroke-width', '5'); });
    path.addEventListener('mouseleave', function() { path.setAttribute('stroke-width', '3'); });
    g.appendChild(path);

    // Dots at each point
    for (let i = 0; i < nX; i++) {
      const x = xPos(i), y = yScale(ranks[i]);
      const dot = EG.svg('circle', {cx: x, cy: y, r: 7, fill: color, stroke: 'var(--bg-card)', 'stroke-width': 2.5});
      dot.addEventListener('mouseenter', function(e) {
        dot.setAttribute('r', '9');
        EG.tooltip.show(e, '<b>' + EG.esc(s.name) + '</b><br>' + xs[i] + ': Rank #' + ranks[i]);
      });
      dot.addEventListener('mouseleave', function() { dot.setAttribute('r', '7'); EG.tooltip.hide(); });
      g.appendChild(dot);
    }

    // End label
    const lastY = yScale(ranks[nX - 1]);
    g.appendChild(EG.svg('text', {
      x: W + 12, y: lastY + 4, fill: color,
      'font-size': '11px', 'font-weight': '600'
    })).textContent = s.name;
  });
}
"""
