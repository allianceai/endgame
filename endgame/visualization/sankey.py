"""Sankey diagram visualizer.

Interactive Sankey/flow diagrams for showing data flow between
categories, feature transformations, or model pipelines.

Example
-------
>>> from endgame.visualization import SankeyVisualizer
>>> viz = SankeyVisualizer(
...     nodes=["Train", "Valid", "Test", "Passed", "Failed"],
...     links=[("Train", "Passed", 80), ("Train", "Failed", 20),
...            ("Valid", "Passed", 70), ("Valid", "Failed", 30)],
... )
>>> viz.save("sankey.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class SankeyVisualizer(BaseVisualizer):
    """Interactive Sankey diagram visualizer.

    Parameters
    ----------
    nodes : list of str
        Node labels.
    links : list of (str, str, float)
        Links as (source, target, value) tuples.
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
        links: Sequence[tuple[str, str, float]],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.nodes = list(nodes)
        self.links = [(str(s), str(t), float(v)) for s, t, v in links]

    def _build_data(self) -> dict[str, Any]:
        node_idx = {n: i for i, n in enumerate(self.nodes)}
        links = []
        for src, tgt, val in self.links:
            if src in node_idx and tgt in node_idx:
                links.append({
                    "source": node_idx[src],
                    "target": node_idx[tgt],
                    "value": round(val, 4),
                })
        return {
            "nodes": self.nodes,
            "links": links,
        }

    def _chart_type(self) -> str:
        return "sankey"

    def _get_chart_js(self) -> str:
        return _SANKEY_JS


_SANKEY_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 100, bottom: 20, left: 100};
  const W = config.width - margin.left - margin.right;
  const H = config.height - margin.top - margin.bottom;
  const svg = EG.svg('svg', {width: config.width, height: config.height});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const palette = config.palette;
  const nodes = data.nodes;
  const links = data.links;

  // Compute node levels (simple left-to-right assignment)
  const nNodes = nodes.length;
  const inDegree = new Array(nNodes).fill(0);
  const outDegree = new Array(nNodes).fill(0);
  const nodeTotal = new Array(nNodes).fill(0);

  links.forEach(function(l) {
    outDegree[l.source] += l.value;
    inDegree[l.target] += l.value;
  });
  for (let i = 0; i < nNodes; i++) {
    nodeTotal[i] = Math.max(inDegree[i], outDegree[i]);
  }

  // Assign levels via topological order
  const level = new Array(nNodes).fill(0);
  const processed = new Array(nNodes).fill(false);
  let changed = true;
  while (changed) {
    changed = false;
    links.forEach(function(l) {
      if (level[l.target] <= level[l.source]) {
        level[l.target] = level[l.source] + 1;
        changed = true;
      }
    });
  }
  const maxLevel = Math.max.apply(null, level) || 1;

  // Group by level
  const levelGroups = [];
  for (let i = 0; i <= maxLevel; i++) levelGroups.push([]);
  for (let i = 0; i < nNodes; i++) levelGroups[level[i]].push(i);

  // Position nodes
  const nodeWidth = 18;
  const nodeX = new Array(nNodes);
  const nodeY = new Array(nNodes);
  const nodeH = new Array(nNodes);

  for (let lv = 0; lv <= maxLevel; lv++) {
    const group = levelGroups[lv];
    const totalH = group.reduce(function(s, ni) { return s + nodeTotal[ni]; }, 0);
    const scale = totalH > 0 ? (H * 0.8) / totalH : 1;
    let y = (H - totalH * scale) / 2;
    const x = (lv / maxLevel) * (W - nodeWidth);
    group.forEach(function(ni) {
      nodeX[ni] = x;
      nodeH[ni] = Math.max(nodeTotal[ni] * scale, 4);
      nodeY[ni] = y;
      y += nodeH[ni] + 8;
    });
  }

  // Track offsets for link placement
  const srcOffset = new Array(nNodes).fill(0);
  const tgtOffset = new Array(nNodes).fill(0);

  // Draw links
  links.forEach(function(l, li) {
    const s = l.source, t = l.target;
    const sx = nodeX[s] + nodeWidth;
    const sy = nodeY[s] + srcOffset[s];
    const sh = (l.value / (nodeTotal[s] || 1)) * nodeH[s];
    const tx = nodeX[t];
    const ty = nodeY[t] + tgtOffset[t];
    const th = (l.value / (nodeTotal[t] || 1)) * nodeH[t];

    srcOffset[s] += sh;
    tgtOffset[t] += th;

    const cpx = (sx + tx) / 2;
    const d = 'M' + sx + ',' + sy +
      ' C' + cpx + ',' + sy + ' ' + cpx + ',' + ty + ' ' + tx + ',' + ty +
      ' L' + tx + ',' + (ty + th) +
      ' C' + cpx + ',' + (ty + th) + ' ' + cpx + ',' + (sy + sh) + ' ' + sx + ',' + (sy + sh) + ' Z';

    const color = palette[s % palette.length];
    const path = EG.svg('path', {d: d, fill: color, opacity: 0.3, stroke: 'none'});
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('opacity', '0.6');
      EG.tooltip.show(e, '<b>' + EG.esc(nodes[s]) + ' → ' + EG.esc(nodes[t]) + '</b><br>Value: ' + EG.fmt(l.value));
    });
    path.addEventListener('mouseleave', function() { path.setAttribute('opacity', '0.3'); EG.tooltip.hide(); });
    g.appendChild(path);
  });

  // Draw nodes
  for (let i = 0; i < nNodes; i++) {
    const color = palette[i % palette.length];
    const rect = EG.svg('rect', {
      x: nodeX[i], y: nodeY[i], width: nodeWidth, height: nodeH[i],
      fill: color, rx: 3, opacity: 0.9
    });
    rect.addEventListener('mouseenter', function(e) {
      EG.tooltip.show(e, '<b>' + EG.esc(nodes[i]) + '</b><br>Total: ' + EG.fmt(nodeTotal[i]));
    });
    rect.addEventListener('mouseleave', function() { EG.tooltip.hide(); });
    g.appendChild(rect);

    // Label
    const lx = level[i] < maxLevel / 2 ? nodeX[i] + nodeWidth + 6 : nodeX[i] - 6;
    const anchor = level[i] < maxLevel / 2 ? 'start' : 'end';
    const label = EG.svg('text', {
      x: lx, y: nodeY[i] + nodeH[i] / 2 + 4,
      'text-anchor': anchor, fill: 'var(--text-primary)', 'font-size': '11px', 'font-weight': '500'
    });
    label.textContent = nodes[i];
    g.appendChild(label);
  }
}
"""
