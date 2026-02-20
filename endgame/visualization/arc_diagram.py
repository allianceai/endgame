"""Arc diagram visualizer.

Interactive arc diagrams for showing pairwise connections between nodes
arranged on a line. Useful for feature co-occurrence, model ensemble
relationships, and dependency visualization.

Example
-------
>>> from endgame.visualization import ArcDiagramVisualizer
>>> viz = ArcDiagramVisualizer(
...     nodes=["feat_1", "feat_2", "feat_3", "feat_4"],
...     edges=[("feat_1", "feat_2", 0.9), ("feat_2", "feat_3", 0.7),
...            ("feat_1", "feat_4", 0.5)],
...     title="Feature Correlations",
... )
>>> viz.save("arc_diagram.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class ArcDiagramVisualizer(BaseVisualizer):
    """Interactive arc diagram visualizer.

    Nodes are placed along a horizontal line; edges are drawn as
    semicircular arcs above (and optionally below) the line. Arc
    thickness and opacity encode edge weight.

    Parameters
    ----------
    nodes : list of str
        Node labels (placed left-to-right).
    edges : list of (str, str, float)
        Edges as (source, target, weight) tuples.
    sort_by : str, default='input'
        Node order: 'input' (given order), 'degree' (most connected first),
        'name' (alphabetical).
    thickness_range : tuple of float, default=(1, 8)
        Min and max arc stroke width (maps to weight).
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=900
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        nodes: Sequence[str],
        edges: Sequence[tuple[str, str, float]],
        *,
        sort_by: str = "input",
        thickness_range: tuple[float, float] = (1.0, 8.0),
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._nodes = list(nodes)
        self._edges = [(str(s), str(t), float(w)) for s, t, w in edges]
        self.sort_by = sort_by
        self.thickness_range = thickness_range

    @classmethod
    def from_correlation_matrix(
        cls,
        corr: Any,
        feature_names: Sequence[str],
        *,
        threshold: float = 0.3,
        **kwargs,
    ) -> ArcDiagramVisualizer:
        """Create from a correlation matrix, showing edges above a threshold.

        Parameters
        ----------
        corr : array-like, shape (n, n)
            Correlation matrix.
        feature_names : list of str
            Feature names.
        threshold : float, default=0.3
            Minimum absolute correlation to draw an edge.
        **kwargs
            Additional keyword arguments.
        """
        corr = np.asarray(corr)
        n = corr.shape[0]
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                w = abs(float(corr[i, j]))
                if w >= threshold:
                    edges.append((feature_names[i], feature_names[j], w))
        kwargs.setdefault("title", "Feature Correlations (Arc Diagram)")
        return cls(list(feature_names), edges, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        nodes = list(self._nodes)

        if self.sort_by == "degree":
            degree = {n: 0 for n in nodes}
            for s, t, w in self._edges:
                if s in degree:
                    degree[s] += 1
                if t in degree:
                    degree[t] += 1
            nodes.sort(key=lambda n: degree.get(n, 0), reverse=True)
        elif self.sort_by == "name":
            nodes.sort()

        node_idx = {n: i for i, n in enumerate(nodes)}
        edges = []
        weights = [w for _, _, w in self._edges]
        w_min = min(weights) if weights else 0
        w_max = max(weights) if weights else 1

        for s, t, w in self._edges:
            if s in node_idx and t in node_idx:
                # Normalize thickness
                if w_max > w_min:
                    norm = (w - w_min) / (w_max - w_min)
                else:
                    norm = 0.5
                thickness = self.thickness_range[0] + norm * (self.thickness_range[1] - self.thickness_range[0])
                edges.append({
                    "source": node_idx[s],
                    "target": node_idx[t],
                    "weight": round(w, 4),
                    "thickness": round(thickness, 2),
                })

        return {
            "nodes": nodes,
            "edges": edges,
        }

    def _chart_type(self) -> str:
        return "arc_diagram"

    def _get_chart_js(self) -> str:
        return _ARC_JS


_ARC_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 30, right: 40, bottom: 60, left: 40};
  const W = config.width - margin.left - margin.right;
  const H = config.height - margin.top - margin.bottom;
  const svg = EG.svg('svg', {width: config.width, height: config.height});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const palette = config.palette;
  const nodes = data.nodes;
  const edges = data.edges;
  const n = nodes.length;
  if (n === 0) return;

  const spacing = W / (n - 1 || 1);
  const baseline = H * 0.75;

  // Draw arcs
  edges.forEach(function(e, ei) {
    const x1 = e.source * spacing;
    const x2 = e.target * spacing;
    const midX = (x1 + x2) / 2;
    const dist = Math.abs(x2 - x1);
    const arcH = Math.min(dist * 0.6, H * 0.65);
    const color = palette[ei % palette.length];

    const d = 'M' + x1 + ',' + baseline +
      ' A' + (dist/2) + ',' + arcH + ' 0 0,1 ' + x2 + ',' + baseline;

    const path = EG.svg('path', {
      d: d, fill: 'none', stroke: color,
      'stroke-width': e.thickness, opacity: 0.5,
      'stroke-linecap': 'round'
    });
    path.addEventListener('mouseenter', function(ev) {
      path.setAttribute('opacity', '0.9');
      path.setAttribute('stroke-width', String(e.thickness + 2));
      EG.tooltip.show(ev, '<b>' + EG.esc(nodes[e.source]) + ' ↔ ' + EG.esc(nodes[e.target]) + '</b><br>Weight: ' + EG.fmt(e.weight, 4));
    });
    path.addEventListener('mouseleave', function() {
      path.setAttribute('opacity', '0.5');
      path.setAttribute('stroke-width', String(e.thickness));
      EG.tooltip.hide();
    });
    g.appendChild(path);
  });

  // Draw nodes
  nodes.forEach(function(name, i) {
    const x = i * spacing;
    const circle = EG.svg('circle', {
      cx: x, cy: baseline, r: 6,
      fill: palette[i % palette.length],
      stroke: 'var(--bg-card)', 'stroke-width': 2
    });
    circle.addEventListener('mouseenter', function(ev) {
      circle.setAttribute('r', '9');
      const deg = edges.filter(function(e) { return e.source === i || e.target === i; }).length;
      EG.tooltip.show(ev, '<b>' + EG.esc(name) + '</b><br>Connections: ' + deg);
    });
    circle.addEventListener('mouseleave', function() {
      circle.setAttribute('r', '6');
      EG.tooltip.hide();
    });
    g.appendChild(circle);

    // Label
    const label = EG.svg('text', {
      x: x, y: baseline + 22,
      'text-anchor': 'middle', fill: 'var(--text-secondary)',
      'font-size': n > 15 ? '9px' : '11px',
      transform: n > 10 ? `rotate(-45,${x},${baseline + 22})` : ''
    });
    label.textContent = name.length > 10 ? name.slice(0,8) + '…' : name;
    g.appendChild(label);
  });
}
"""
