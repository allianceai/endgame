"""Dot matrix (waffle) chart visualizer.

Interactive dot matrix charts for showing proportions and
part-to-whole relationships using unit dots.

Example
-------
>>> from endgame.visualization import DotMatrixVisualizer
>>> viz = DotMatrixVisualizer(
...     labels=["Correct", "Incorrect"],
...     values=[85, 15],
...     title="Model Accuracy",
... )
>>> viz.save("dot_matrix.html")
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class DotMatrixVisualizer(BaseVisualizer):
    """Interactive dot matrix (waffle) chart visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float
        Values for each category (will be normalized to n_dots total).
    n_dots : int, default=100
        Total number of dots.
    dot_shape : str, default='circle'
        'circle' or 'square'.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=600
        Chart width.
    height : int, default=400
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        values: Sequence[float],
        *,
        n_dots: int = 100,
        dot_shape: str = "circle",
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 400,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.values = [float(v) for v in values]
        self.n_dots = n_dots
        self.dot_shape = dot_shape

    def _build_data(self) -> dict[str, Any]:
        total = sum(self.values)
        if total <= 0:
            return {"dots": [], "labels": self.labels, "counts": [0] * len(self.labels)}

        # Allocate dots proportionally
        n = self.n_dots
        raw = [v / total * n for v in self.values]
        counts = [int(r) for r in raw]
        remainders = [r - int(r) for r in raw]

        # Distribute remaining dots by largest remainder
        remaining = n - sum(counts)
        for _ in range(remaining):
            idx = max(range(len(remainders)), key=lambda i: remainders[i])
            counts[idx] += 1
            remainders[idx] = 0

        # Build dot array
        dots = []
        for i, count in enumerate(counts):
            dots.extend([i] * count)

        # Grid layout
        cols = int(math.ceil(math.sqrt(n * 1.5)))
        rows = int(math.ceil(n / cols))

        return {
            "dots": dots,
            "labels": self.labels,
            "counts": counts,
            "cols": cols,
            "rows": rows,
            "shape": self.dot_shape,
            "pcts": [round(v / total * 100, 1) for v in self.values],
        }

    def _chart_type(self) -> str:
        return "dot_matrix"

    def _get_chart_js(self) -> str:
        return _DOT_MATRIX_JS


_DOT_MATRIX_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const palette = config.palette;
  const dots = data.dots;
  const cols = data.cols;
  const isCircle = data.shape === 'circle';

  const dotSize = Math.min(
    Math.floor((config.width - 40) / cols),
    Math.floor((config.height - 80) / data.rows),
    20
  );
  const gap = 3;
  const totalW = cols * (dotSize + gap);
  const totalH = data.rows * (dotSize + gap);

  const margin = {top: 10, right: 20, bottom: 50, left: 20};
  const svg = EG.svg('svg', {width: totalW + margin.left + margin.right, height: totalH + margin.top + margin.bottom});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);

  dots.forEach(function(catIdx, i) {
    const col = i % cols;
    const row = Math.floor(i / cols);
    const x = col * (dotSize + gap);
    const y = row * (dotSize + gap);
    const color = palette[catIdx % palette.length];

    let el;
    if (isCircle) {
      el = EG.svg('circle', {
        cx: x + dotSize / 2, cy: y + dotSize / 2, r: dotSize / 2 - 1,
        fill: color, opacity: 0.85
      });
    } else {
      el = EG.svg('rect', {
        x: x + 1, y: y + 1, width: dotSize - 2, height: dotSize - 2,
        fill: color, opacity: 0.85, rx: 2
      });
    }
    el.addEventListener('mouseenter', function(e) {
      el.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(data.labels[catIdx]) + '</b><br>' +
        data.counts[catIdx] + ' / ' + dots.length + ' (' + data.pcts[catIdx] + '%)');
    });
    el.addEventListener('mouseleave', function() { el.setAttribute('opacity', '0.85'); EG.tooltip.hide(); });
    g.appendChild(el);
  });

  // Legend
  const items = data.labels.map(function(l, i) {
    return {label: l + ' (' + data.pcts[i] + '%)', color: palette[i % palette.length]};
  });
  EG.drawLegend(container, items);
}
"""
