"""Flow chart visualizer.

Interactive flow chart for visualizing ML pipelines, data processing
workflows, and model architecture diagrams.

Example
-------
>>> from endgame.visualization import FlowChartVisualizer
>>> viz = FlowChartVisualizer(
...     nodes=[
...         {"id": "data", "label": "Raw Data", "type": "input"},
...         {"id": "preprocess", "label": "Preprocessing", "type": "process"},
...         {"id": "model", "label": "LightGBM", "type": "process"},
...         {"id": "output", "label": "Predictions", "type": "output"},
...     ],
...     edges=[("data", "preprocess"), ("preprocess", "model"), ("model", "output")],
...     title="ML Pipeline",
... )
>>> viz.save("flow.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class FlowChartVisualizer(BaseVisualizer):
    """Interactive flow chart visualizer.

    Parameters
    ----------
    nodes : list of dict
        Each dict must have 'id' and 'label'. Optional keys:
        'type' ('input', 'process', 'decision', 'output'),
        'description' (tooltip text).
    edges : list of (str, str) or (str, str, str)
        Edges as (source_id, target_id) or (source_id, target_id, label).
    direction : str, default='LR'
        Flow direction: 'LR' (left-to-right), 'TB' (top-to-bottom).
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
        nodes: Sequence[dict[str, str]],
        edges: Sequence[tuple[str, str] | tuple[str, str, str]],
        *,
        direction: str = "LR",
        title: str = "",
        palette: str = "tableau",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._nodes = list(nodes)
        self._edges = list(edges)
        self.direction = direction

    @classmethod
    def from_pipeline(
        cls,
        steps: Sequence[tuple[str, str]],
        **kwargs,
    ) -> FlowChartVisualizer:
        """Create from a list of (name, description) pipeline steps.

        Parameters
        ----------
        steps : list of (str, str)
            Pipeline step names and descriptions.
        **kwargs
            Additional keyword arguments.
        """
        nodes = []
        edges = []
        for i, (name, desc) in enumerate(steps):
            t = "input" if i == 0 else ("output" if i == len(steps) - 1 else "process")
            nodes.append({"id": f"step_{i}", "label": name, "type": t, "description": desc})
            if i > 0:
                edges.append((f"step_{i-1}", f"step_{i}"))
        kwargs.setdefault("title", "Pipeline")
        return cls(nodes, edges, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        node_map = {}
        for n in self._nodes:
            node_map[n["id"]] = {
                "id": n["id"],
                "label": n.get("label", n["id"]),
                "type": n.get("type", "process"),
                "description": n.get("description", ""),
            }

        # Topological layering
        adj: dict[str, list[str]] = {n["id"]: [] for n in self._nodes}
        in_deg: dict[str, int] = {n["id"]: 0 for n in self._nodes}
        edges_data = []
        for e in self._edges:
            src, tgt = e[0], e[1]
            label = e[2] if len(e) > 2 else ""
            if src in adj:
                adj[src].append(tgt)
            if tgt in in_deg:
                in_deg[tgt] += 1
            edges_data.append({"source": src, "target": tgt, "label": label})

        # BFS layering
        layers: dict[str, int] = {}
        queue = [nid for nid, d in in_deg.items() if d == 0]
        for nid in queue:
            if nid not in layers:
                layers[nid] = 0
        while queue:
            nid = queue.pop(0)
            for child in adj.get(nid, []):
                layers[child] = max(layers.get(child, 0), layers[nid] + 1)
                in_deg[child] -= 1
                if in_deg[child] <= 0:
                    queue.append(child)

        # Assign positions
        max_layer = max(layers.values()) if layers else 0
        layer_groups: dict[int, list[str]] = {}
        for nid, layer in layers.items():
            layer_groups.setdefault(layer, []).append(nid)

        nodes_out = []
        is_lr = self.direction == "LR"
        for layer, nids in sorted(layer_groups.items()):
            for pos, nid in enumerate(nids):
                nd = node_map.get(nid, {"id": nid, "label": nid, "type": "process", "description": ""})
                nd["layer"] = layer
                nd["pos"] = pos
                nd["layerSize"] = len(nids)
                nodes_out.append(nd)

        return {
            "nodes": nodes_out,
            "edges": edges_data,
            "maxLayer": max_layer,
            "direction": self.direction,
        }

    def _chart_type(self) -> str:
        return "flow_chart"

    def _get_chart_js(self) -> str:
        return _FLOW_JS


_FLOW_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 20, left: 20};
  const W = config.width - margin.left - margin.right;
  const H = config.height - margin.top - margin.bottom;
  const svg = EG.svg('svg', {width: config.width, height: config.height});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const palette = config.palette;
  const nodes = data.nodes;
  const edges = data.edges;
  const isLR = data.direction === 'LR';
  const maxLayer = data.maxLayer || 1;

  const nodeW = 130, nodeH = 50;

  // Position map
  const posMap = {};
  const typeColors = {input: palette[0], process: palette[1], decision: palette[2], output: palette[3]};

  nodes.forEach(function(n) {
    let x, y;
    if (isLR) {
      x = (n.layer / maxLayer) * (W - nodeW);
      y = (n.pos + 0.5) / n.layerSize * H - nodeH / 2;
    } else {
      x = (n.pos + 0.5) / n.layerSize * W - nodeW / 2;
      y = (n.layer / maxLayer) * (H - nodeH);
    }
    posMap[n.id] = {x: x, y: y};
  });

  // Draw edges first (behind nodes)
  edges.forEach(function(e) {
    const s = posMap[e.source];
    const t = posMap[e.target];
    if (!s || !t) return;

    const sx = s.x + nodeW, sy = s.y + nodeH / 2;
    const tx = t.x, ty = t.y + nodeH / 2;

    if (isLR) {
      const cpx = (sx + tx) / 2;
      const d = 'M' + sx + ',' + sy + ' C' + cpx + ',' + sy + ' ' + cpx + ',' + ty + ' ' + tx + ',' + ty;
      g.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: 'var(--text-muted)', 'stroke-width': 2, opacity: 0.6}));
      // Arrow
      g.appendChild(EG.svg('polygon', {
        points: (tx-8) + ',' + (ty-4) + ' ' + tx + ',' + ty + ' ' + (tx-8) + ',' + (ty+4),
        fill: 'var(--text-muted)', opacity: 0.6
      }));
    } else {
      const bsy = s.y + nodeH, btx = t.x + nodeW / 2, bty = t.y;
      const bsx = s.x + nodeW / 2;
      const cpy = (bsy + bty) / 2;
      const d = 'M' + bsx + ',' + bsy + ' C' + bsx + ',' + cpy + ' ' + btx + ',' + cpy + ' ' + btx + ',' + bty;
      g.appendChild(EG.svg('path', {d: d, fill: 'none', stroke: 'var(--text-muted)', 'stroke-width': 2, opacity: 0.6}));
      g.appendChild(EG.svg('polygon', {
        points: (btx-4) + ',' + (bty-8) + ' ' + btx + ',' + bty + ' ' + (btx+4) + ',' + (bty-8),
        fill: 'var(--text-muted)', opacity: 0.6
      }));
    }

    // Edge label
    if (e.label) {
      const lx = (s.x + nodeW + t.x) / 2;
      const ly = (s.y + t.y + nodeH) / 2 - 5;
      const txt = EG.svg('text', {x: lx, y: ly, 'text-anchor': 'middle', fill: 'var(--text-muted)', 'font-size': '9px'});
      txt.textContent = e.label;
      g.appendChild(txt);
    }
  });

  // Draw nodes
  nodes.forEach(function(n) {
    const p = posMap[n.id];
    const color = typeColors[n.type] || palette[1];
    let shape;

    if (n.type === 'decision') {
      // Diamond
      const cx = p.x + nodeW/2, cy2 = p.y + nodeH/2;
      const pts = [cx, p.y, p.x+nodeW, cy2, cx, p.y+nodeH, p.x, cy2].join(',');
      shape = EG.svg('polygon', {points: pts, fill: color, opacity: 0.2, stroke: color, 'stroke-width': 2});
    } else if (n.type === 'input' || n.type === 'output') {
      shape = EG.svg('rect', {x: p.x, y: p.y, width: nodeW, height: nodeH, rx: nodeH/2, fill: color, opacity: 0.2, stroke: color, 'stroke-width': 2});
    } else {
      shape = EG.svg('rect', {x: p.x, y: p.y, width: nodeW, height: nodeH, rx: 8, fill: color, opacity: 0.2, stroke: color, 'stroke-width': 2});
    }

    shape.addEventListener('mouseenter', function(e) {
      shape.setAttribute('opacity', '0.5');
      let html = '<b>' + EG.esc(n.label) + '</b>';
      if (n.description) html += '<br>' + EG.esc(n.description);
      html += '<br><i>' + n.type + '</i>';
      EG.tooltip.show(e, html);
    });
    shape.addEventListener('mouseleave', function() {
      shape.setAttribute('opacity', '0.2');
      EG.tooltip.hide();
    });
    g.appendChild(shape);

    // Label
    const txt = EG.svg('text', {
      x: p.x + nodeW/2, y: p.y + nodeH/2 + 4,
      'text-anchor': 'middle', fill: 'var(--text-primary)',
      'font-size': '12px', 'font-weight': '600'
    });
    txt.textContent = n.label.length > 16 ? n.label.slice(0,14) + '…' : n.label;
    g.appendChild(txt);
  });
}
"""
