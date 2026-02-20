"""Network diagram visualizer.

Interactive force-directed network diagrams, specifically designed for
Bayesian Network visualization but general enough for any DAG or graph.

Supports:
- Bayesian Network structure (from endgame Bayesian classifiers)
- General directed and undirected graphs
- Node sizing by importance/degree
- Edge coloring by weight/strength
- Force-directed layout with draggable nodes

Example
-------
>>> from endgame.visualization import NetworkDiagramVisualizer
>>> viz = NetworkDiagramVisualizer(
...     nodes=["Rain", "Sprinkler", "Wet Grass"],
...     edges=[("Rain", "Wet Grass"), ("Sprinkler", "Wet Grass"),
...            ("Rain", "Sprinkler")],
...     title="Bayesian Network: Wet Grass",
... )
>>> viz.save("bayesian_network.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class NetworkDiagramVisualizer(BaseVisualizer):
    """Interactive network diagram visualizer.

    Parameters
    ----------
    nodes : list of str or list of dict
        Node labels (strings) or dicts with 'id', 'label', and optional
        'group', 'size', 'description' keys.
    edges : list of (str, str) or (str, str, float)
        Edges as (source, target) or (source, target, weight) tuples.
    directed : bool, default=True
        If True, draw arrows on edges (for DAGs like Bayesian Networks).
    node_sizes : dict of str → float, optional
        Custom node sizes. If None, sized by degree.
    layout : str, default='force'
        Layout algorithm: 'force' (force-directed), 'hierarchical'
        (top-down DAG), 'circular'.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=800
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        nodes: Sequence[str] | Sequence[dict[str, Any]],
        edges: Sequence[tuple[str, str] | tuple[str, str, float]],
        *,
        directed: bool = True,
        node_sizes: dict[str, float] | None = None,
        layout: str = "force",
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        # Normalize nodes
        if nodes and isinstance(nodes[0], str):
            self._nodes = [{"id": n, "label": n} for n in nodes]
        else:
            self._nodes = [dict(n) for n in nodes]

        self._edges = list(edges)
        self.directed = directed
        self.node_sizes = node_sizes or {}
        self.layout = layout

    # ------------------------------------------------------------------
    # Classmethod constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_bayesian_network(
        cls,
        model: Any,
        *,
        feature_names: Sequence[str] | None = None,
        **kwargs,
    ) -> NetworkDiagramVisualizer:
        """Create from an endgame Bayesian classifier (TAN, KDB, ESKDB).

        Extracts the DAG structure from the model's learned network.

        Parameters
        ----------
        model : estimator
            Fitted Bayesian network classifier with structure information.
            Supports endgame TAN, KDB, ESKDB classifiers and any model
            with ``edges_`` or ``dag_`` attribute.
        feature_names : list of str, optional
            Feature names.
        **kwargs
            Additional keyword arguments.
        """
        nodes = []
        edges = []

        # Try to extract from endgame Bayesian classifiers
        if hasattr(model, "feature_names_in_"):
            feature_names = list(model.feature_names_in_)

        if hasattr(model, "edges_"):
            # Direct edge list [(parent_idx, child_idx), ...]
            edge_list = model.edges_
            n_features = len(feature_names) if feature_names else 0

            # Determine feature names
            if feature_names is None:
                max_idx = max(max(e) for e in edge_list) if edge_list else 0
                feature_names = [f"X{i}" for i in range(max_idx + 1)]

            nodes = list(feature_names)
            # Add class node
            if hasattr(model, "classes_"):
                nodes.append("Class")

            for parent, child in edge_list:
                src = feature_names[parent] if parent < len(feature_names) else f"X{parent}"
                tgt = feature_names[child] if child < len(feature_names) else f"X{child}"
                edges.append((src, tgt))

            # In TAN/KDB, class is parent of all features
            if hasattr(model, "classes_"):
                for fname in feature_names:
                    edges.append(("Class", fname))

        elif hasattr(model, "dag_"):
            # Adjacency matrix
            dag = np.asarray(model.dag_)
            n = dag.shape[0]
            if feature_names is None:
                feature_names = [f"X{i}" for i in range(n)]
            nodes = list(feature_names)
            for i in range(n):
                for j in range(n):
                    if dag[i, j] != 0:
                        edges.append((feature_names[i], feature_names[j]))

        elif hasattr(model, "tree_") and hasattr(model, "parents_"):
            # Tree-augmented structure
            parents = model.parents_
            if feature_names is None:
                feature_names = [f"X{i}" for i in range(len(parents))]
            nodes = list(feature_names)
            for i, p in enumerate(parents):
                if p >= 0:
                    edges.append((feature_names[p], feature_names[i]))
        else:
            raise ValueError(
                f"Cannot extract network structure from {type(model).__name__}. "
                "Model must have 'edges_', 'dag_', or 'parents_' attribute."
            )

        kwargs.setdefault("title", f"Bayesian Network ({type(model).__name__})")
        kwargs.setdefault("directed", True)
        kwargs.setdefault("layout", "hierarchical")

        # Compute node importance from degree
        degree = {n: 0 for n in nodes}
        for e in edges:
            if e[0] in degree:
                degree[e[0]] += 1
            if e[1] in degree:
                degree[e[1]] += 1
        max_deg = max(degree.values()) if degree else 1
        node_sizes = {n: 8 + (d / max_deg) * 20 for n, d in degree.items()}

        return cls(nodes, edges, node_sizes=node_sizes, **kwargs)

    @classmethod
    def from_adjacency_matrix(
        cls,
        matrix: Any,
        labels: Sequence[str],
        *,
        threshold: float = 0.0,
        **kwargs,
    ) -> NetworkDiagramVisualizer:
        """Create from an adjacency/weight matrix.

        Parameters
        ----------
        matrix : array-like, shape (n, n)
            Adjacency or weight matrix.
        labels : list of str
            Node labels.
        threshold : float, default=0.0
            Minimum absolute value to create an edge.
        **kwargs
            Additional keyword arguments.
        """
        m = np.asarray(matrix)
        n = m.shape[0]
        edges = []
        for i in range(n):
            for j in range(n):
                if i != j and abs(m[i, j]) > threshold:
                    edges.append((labels[i], labels[j], float(m[i, j])))
        return cls(list(labels), edges, **kwargs)

    @classmethod
    def from_edge_list(
        cls,
        edges: Sequence[tuple[str, str]],
        **kwargs,
    ) -> NetworkDiagramVisualizer:
        """Create from a simple edge list, auto-discovering nodes.

        Parameters
        ----------
        edges : list of (str, str)
            Edge tuples.
        **kwargs
            Additional keyword arguments.
        """
        node_set = set()
        for e in edges:
            node_set.add(e[0])
            node_set.add(e[1])
        return cls(sorted(node_set), edges, **kwargs)

    # ------------------------------------------------------------------
    # BaseVisualizer interface
    # ------------------------------------------------------------------

    def _build_data(self) -> dict[str, Any]:
        # Node index map
        node_ids = [n["id"] for n in self._nodes]
        node_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Compute degree for sizing
        in_deg = {nid: 0 for nid in node_ids}
        out_deg = {nid: 0 for nid in node_ids}
        for e in self._edges:
            src, tgt = e[0], e[1]
            if src in out_deg:
                out_deg[src] += 1
            if tgt in in_deg:
                in_deg[tgt] += 1

        nodes = []
        for n in self._nodes:
            nid = n["id"]
            degree = in_deg.get(nid, 0) + out_deg.get(nid, 0)
            size = self.node_sizes.get(nid, 8 + degree * 3)
            group = n.get("group", "")
            nodes.append({
                "id": nid,
                "label": n.get("label", nid),
                "group": group,
                "size": round(float(size), 2),
                "inDeg": in_deg.get(nid, 0),
                "outDeg": out_deg.get(nid, 0),
                "description": n.get("description", ""),
            })

        edges = []
        for e in self._edges:
            src, tgt = e[0], e[1]
            weight = float(e[2]) if len(e) > 2 else 1.0
            if src in node_idx and tgt in node_idx:
                edges.append({
                    "source": node_idx[src],
                    "target": node_idx[tgt],
                    "weight": round(weight, 4),
                })

        # Initial positions (hierarchical or circular)
        positions = self._compute_layout(nodes, edges)

        return {
            "nodes": nodes,
            "edges": edges,
            "directed": self.directed,
            "layout": self.layout,
            "positions": positions,
        }

    def _compute_layout(self, nodes, edges):
        """Compute initial node positions."""
        n = len(nodes)
        if n == 0:
            return []

        if self.layout == "circular":
            import math
            positions = []
            for i in range(n):
                angle = 2 * math.pi * i / n - math.pi / 2
                positions.append({
                    "x": round(0.5 + 0.35 * math.cos(angle), 4),
                    "y": round(0.5 + 0.35 * math.sin(angle), 4),
                })
            return positions

        if self.layout == "hierarchical":
            # Topological layering
            node_ids = [nd["id"] for nd in nodes]
            idx = {nid: i for i, nid in enumerate(node_ids)}
            in_deg = [0] * n
            adj = [[] for _ in range(n)]
            for e in edges:
                adj[e["source"]].append(e["target"])
                in_deg[e["target"]] += 1

            layers = [0] * n
            queue = [i for i in range(n) if in_deg[i] == 0]
            visited = [False] * n
            for i in queue:
                visited[i] = True
            while queue:
                cur = queue.pop(0)
                for child in adj[cur]:
                    layers[child] = max(layers[child], layers[cur] + 1)
                    in_deg[child] -= 1
                    if in_deg[child] <= 0 and not visited[child]:
                        visited[child] = True
                        queue.append(child)

            # Unvisited nodes (cycles) — assign to last layer + 1
            max_layer = max(layers) if layers else 0
            for i in range(n):
                if not visited[i]:
                    layers[i] = max_layer + 1

            max_layer = max(layers)
            layer_groups: dict[int, list[int]] = {}
            for i, layer in enumerate(layers):
                layer_groups.setdefault(layer, []).append(i)

            positions = [{"x": 0.5, "y": 0.5}] * n
            for layer, group in layer_groups.items():
                for pos, ni in enumerate(group):
                    x = (layer + 0.5) / (max_layer + 1) if max_layer > 0 else 0.5
                    y = (pos + 0.5) / len(group)
                    positions[ni] = {"x": round(x, 4), "y": round(y, 4)}
            return positions

        # Force layout: random initial positions
        rng = np.random.RandomState(42)
        return [
            {"x": round(float(rng.uniform(0.2, 0.8)), 4),
             "y": round(float(rng.uniform(0.2, 0.8)), 4)}
            for _ in range(n)
        ]

    def _chart_type(self) -> str:
        return "network"

    def _get_chart_js(self) -> str:
        return _NETWORK_JS


# ---------------------------------------------------------------------------
# JavaScript renderer with force-directed simulation
# ---------------------------------------------------------------------------

_NETWORK_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const W = config.width, H = config.height;
  const svg = EG.svg('svg', {width: W, height: H});
  container.appendChild(svg);
  const palette = config.palette;
  const nodes = data.nodes;
  const edges = data.edges;
  const directed = data.directed;
  const n = nodes.length;
  if (n === 0) return;

  // Initialize positions from layout
  const pos = data.positions.map(function(p) {
    return {x: p.x * W, y: p.y * H};
  });

  // Arrow marker for directed edges
  if (directed) {
    const defs = EG.svg('defs');
    svg.appendChild(defs);
    const marker = EG.svg('marker', {
      id: 'arrowhead', markerWidth: 10, markerHeight: 7,
      refX: 10, refY: 3.5, orient: 'auto', fill: 'var(--text-muted)'
    });
    marker.appendChild(EG.svg('polygon', {points: '0 0, 10 3.5, 0 7'}));
    defs.appendChild(marker);
  }

  // Edge group (drawn first, behind nodes)
  const edgeGroup = EG.svg('g');
  svg.appendChild(edgeGroup);
  const nodeGroup = EG.svg('g');
  svg.appendChild(nodeGroup);
  const labelGroup = EG.svg('g');
  svg.appendChild(labelGroup);

  // Create edge elements
  const edgeEls = [];
  edges.forEach(function(e, ei) {
    const line = EG.svg('line', {
      'stroke': 'var(--text-muted)', 'stroke-width': Math.max(1, Math.min(e.weight * 2, 5)),
      'opacity': 0.4, 'marker-end': directed ? 'url(#arrowhead)' : ''
    });
    line.addEventListener('mouseenter', function(ev) {
      line.setAttribute('opacity', '0.8');
      line.setAttribute('stroke', palette[ei % palette.length]);
      EG.tooltip.show(ev, '<b>' + EG.esc(nodes[e.source].label) + ' → ' + EG.esc(nodes[e.target].label) + '</b>' +
        (e.weight !== 1 ? '<br>Weight: ' + EG.fmt(e.weight) : ''));
    });
    line.addEventListener('mouseleave', function() {
      line.setAttribute('opacity', '0.4');
      line.setAttribute('stroke', 'var(--text-muted)');
      EG.tooltip.hide();
    });
    edgeGroup.appendChild(line);
    edgeEls.push(line);
  });

  // Create node elements
  const nodeEls = [];
  const labelEls = [];
  nodes.forEach(function(nd, i) {
    const color = nd.group ? palette[nd.group.charCodeAt(0) % palette.length] : palette[i % palette.length];
    const r = Math.max(nd.size, 6);

    const circle = EG.svg('circle', {
      r: r, fill: color, opacity: 0.85,
      stroke: 'var(--bg-card)', 'stroke-width': 2,
      style: 'cursor: grab;'
    });
    circle.addEventListener('mouseenter', function(ev) {
      circle.setAttribute('opacity', '1');
      circle.setAttribute('r', String(r + 3));
      let html = '<b>' + EG.esc(nd.label) + '</b>';
      if (nd.description) html += '<br>' + EG.esc(nd.description);
      html += '<br>In-degree: ' + nd.inDeg + ', Out-degree: ' + nd.outDeg;
      EG.tooltip.show(ev, html);
    });
    circle.addEventListener('mouseleave', function() {
      circle.setAttribute('opacity', '0.85');
      circle.setAttribute('r', String(r));
      EG.tooltip.hide();
    });

    // Drag support
    let dragging = false;
    circle.addEventListener('mousedown', function(ev) {
      dragging = true;
      circle.style.cursor = 'grabbing';
      ev.preventDefault();
    });
    document.addEventListener('mousemove', function(ev) {
      if (!dragging) return;
      const rect = svg.getBoundingClientRect();
      pos[i].x = ev.clientX - rect.left;
      pos[i].y = ev.clientY - rect.top;
      updatePositions();
    });
    document.addEventListener('mouseup', function() {
      if (dragging) {
        dragging = false;
        circle.style.cursor = 'grab';
      }
    });

    nodeGroup.appendChild(circle);
    nodeEls.push(circle);

    // Label
    const label = EG.svg('text', {
      'text-anchor': 'middle', fill: 'var(--text-primary)',
      'font-size': '11px', 'font-weight': '500',
      'pointer-events': 'none'
    });
    label.textContent = nd.label.length > 14 ? nd.label.slice(0,12) + '…' : nd.label;
    labelGroup.appendChild(label);
    labelEls.push(label);
  });

  function updatePositions() {
    edges.forEach(function(e, ei) {
      const sx = pos[e.source].x, sy = pos[e.source].y;
      const tx = pos[e.target].x, ty = pos[e.target].y;
      // Shorten line to account for node radius
      const sr = nodes[e.source].size || 6;
      const tr = nodes[e.target].size || 6;
      const dx = tx - sx, dy = ty - sy;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const ux = dx / dist, uy = dy / dist;
      edgeEls[ei].setAttribute('x1', sx + ux * sr);
      edgeEls[ei].setAttribute('y1', sy + uy * sr);
      edgeEls[ei].setAttribute('x2', tx - ux * (tr + 5));
      edgeEls[ei].setAttribute('y2', ty - uy * (tr + 5));
    });
    nodes.forEach(function(nd, i) {
      nodeEls[i].setAttribute('cx', pos[i].x);
      nodeEls[i].setAttribute('cy', pos[i].y);
      labelEls[i].setAttribute('x', pos[i].x);
      labelEls[i].setAttribute('y', pos[i].y + nd.size + 14);
    });
  }

  // Force simulation (simple Fruchterman-Reingold style)
  if (data.layout === 'force') {
    const area = W * H;
    const k = Math.sqrt(area / n) * 0.8;
    const iterations = 120;
    const temp0 = W / 5;

    for (let iter = 0; iter < iterations; iter++) {
      const temp = temp0 * (1 - iter / iterations);
      const disp = pos.map(function() { return {x: 0, y: 0}; });

      // Repulsive forces
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          let dx = pos[i].x - pos[j].x;
          let dy = pos[i].y - pos[j].y;
          let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
          let force = k * k / dist;
          let fx = dx / dist * force;
          let fy = dy / dist * force;
          disp[i].x += fx; disp[i].y += fy;
          disp[j].x -= fx; disp[j].y -= fy;
        }
      }

      // Attractive forces
      edges.forEach(function(e) {
        let dx = pos[e.target].x - pos[e.source].x;
        let dy = pos[e.target].y - pos[e.source].y;
        let dist = Math.sqrt(dx * dx + dy * dy) || 0.1;
        let force = dist * dist / k;
        let fx = dx / dist * force;
        let fy = dy / dist * force;
        disp[e.source].x += fx; disp[e.source].y += fy;
        disp[e.target].x -= fx; disp[e.target].y -= fy;
      });

      // Apply with temperature
      for (let i = 0; i < n; i++) {
        let dist = Math.sqrt(disp[i].x * disp[i].x + disp[i].y * disp[i].y) || 0.1;
        pos[i].x += disp[i].x / dist * Math.min(dist, temp);
        pos[i].y += disp[i].y / dist * Math.min(dist, temp);
        // Keep in bounds
        pos[i].x = Math.max(40, Math.min(W - 40, pos[i].x));
        pos[i].y = Math.max(40, Math.min(H - 40, pos[i].y));
      }
    }
  }

  updatePositions();
}
"""
