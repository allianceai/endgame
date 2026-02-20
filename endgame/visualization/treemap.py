"""Treemap and sunburst visualizer.

Interactive treemap for hierarchical feature importance and
sunburst charts for hierarchical data exploration.

Example
-------
>>> from endgame.visualization import TreemapVisualizer
>>> viz = TreemapVisualizer(
...     labels=["cat_feats", "num_feats", "text_feats"],
...     values=[0.45, 0.35, 0.20],
...     title="Feature Group Importances",
... )
>>> viz.save("treemap.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


def _squarify(values, x, y, w, h):
    """Simple squarified treemap layout algorithm."""
    if not values:
        return []

    total = sum(v for v, _ in values)
    if total <= 0:
        return [(x, y, w, h, lbl) for _, lbl in values]

    rects = []
    remaining = list(values)

    def layout_row(row, rx, ry, rw, rh, is_vertical):
        row_total = sum(v for v, _ in row)
        if is_vertical:
            row_w = (row_total / total) * rw if total > 0 else rw
            offset = 0
            for v, lbl in row:
                rect_h = (v / row_total) * rh if row_total > 0 else rh / len(row)
                rects.append((rx, ry + offset, row_w, rect_h, lbl))
                offset += rect_h
            return rx + row_w, ry, rw - row_w, rh
        else:
            row_h = (row_total / total) * rh if total > 0 else rh
            offset = 0
            for v, lbl in row:
                rect_w = (v / row_total) * rw if row_total > 0 else rw / len(row)
                rects.append((rx + offset, ry, rect_w, row_h, lbl))
                offset += rect_w
            return rx, ry + row_h, rw, rh - row_h

    # Simple greedy approach
    is_vertical = w >= h
    row = []
    cx, cy, cw, ch = x, y, w, h

    for item in remaining:
        row.append(item)
        if len(row) >= max(1, len(remaining) // 3):
            cx, cy, cw, ch = layout_row(row, cx, cy, cw, ch, is_vertical)
            is_vertical = cw >= ch
            row = []

    if row:
        layout_row(row, cx, cy, cw, ch, is_vertical)

    return rects


class TreemapVisualizer(BaseVisualizer):
    """Interactive treemap visualizer.

    Parameters
    ----------
    labels : list of str
        Category labels.
    values : list of float
        Values (sizes) for each category.
    parents : list of str, optional
        Parent labels for hierarchical treemaps.
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
        labels: Sequence[str],
        values: Sequence[float],
        *,
        parents: Sequence[str] | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self.values = [float(v) for v in values]
        self.parents = list(parents) if parents else None

    @classmethod
    def from_importances(
        cls,
        model: Any,
        *,
        feature_names: Sequence[str] | None = None,
        top_n: int = 30,
        **kwargs,
    ) -> TreemapVisualizer:
        """Create from model feature importances.

        Parameters
        ----------
        model : estimator
            Fitted model with ``feature_importances_``.
        feature_names : list of str, optional
            Feature names.
        top_n : int, default=30
            Show top N features.
        **kwargs
            Additional keyword arguments.
        """
        importances = np.asarray(model.feature_importances_)
        if feature_names is None:
            if hasattr(model, "feature_names_in_"):
                feature_names = list(model.feature_names_in_)
            else:
                feature_names = [f"feature_{i}" for i in range(len(importances))]

        idx = np.argsort(importances)[::-1][:top_n]
        labels = [feature_names[i] for i in idx]
        vals = importances[idx].tolist()
        kwargs.setdefault("title", "Feature Importances (Treemap)")
        return cls(labels, vals, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        # Sort by value descending for better layout
        pairs = sorted(zip(self.values, self.labels), reverse=True)
        values = [v for v, _ in pairs]
        labels = [l for _, l in pairs]
        total = sum(values) if values else 1

        # Compute layout
        rects = _squarify(
            list(zip(values, labels)),
            0, 0, self.width - 40, self.height - 60,
        )

        items = []
        for i, (rx, ry, rw, rh, lbl) in enumerate(rects):
            idx = labels.index(lbl)
            items.append({
                "label": lbl,
                "value": round(values[idx], 6),
                "pct": round(values[idx] / total * 100, 1) if total > 0 else 0,
                "x": round(rx, 1),
                "y": round(ry, 1),
                "w": round(rw, 1),
                "h": round(rh, 1),
            })

        return {"items": items}

    def _chart_type(self) -> str:
        return "treemap"

    def _get_chart_js(self) -> str:
        return _TREEMAP_JS


class SunburstVisualizer(BaseVisualizer):
    """Interactive sunburst chart visualizer.

    Parameters
    ----------
    labels : list of str
        Node labels.
    parents : list of str
        Parent label for each node ('' for root).
    values : list of float
        Values/sizes for each node.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=600
        Chart width.
    height : int, default=600
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        labels: Sequence[str],
        parents: Sequence[str],
        values: Sequence[float],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.labels = list(labels)
        self._parents = list(parents)
        self.values = [float(v) for v in values]

    def _build_data(self) -> dict[str, Any]:
        nodes = []
        for i, (lbl, parent, val) in enumerate(zip(self.labels, self._parents, self.values)):
            nodes.append({
                "label": lbl,
                "parent": parent,
                "value": round(val, 6),
            })
        return {"nodes": nodes}

    def _chart_type(self) -> str:
        return "sunburst"

    def _get_chart_js(self) -> str:
        return _SUNBURST_JS


# ---------------------------------------------------------------------------
# Treemap JS
# ---------------------------------------------------------------------------

_TREEMAP_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 10, right: 10, bottom: 10, left: 10};
  const svg = EG.svg('svg', {width: config.width, height: config.height});
  container.appendChild(svg);
  const g = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(g);
  const palette = config.palette;

  data.items.forEach(function(item, i) {
    const color = palette[i % palette.length];
    const rect = EG.svg('rect', {
      x: item.x, y: item.y, width: Math.max(item.w - 2, 0), height: Math.max(item.h - 2, 0),
      fill: color, opacity: 0.8, rx: 4, stroke: 'var(--bg-card)', 'stroke-width': 2
    });
    rect.addEventListener('mouseenter', function(e) {
      rect.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(item.label) + '</b><br>Value: ' + EG.fmt(item.value) + '<br>' + item.pct + '%');
    });
    rect.addEventListener('mouseleave', function() { rect.setAttribute('opacity', '0.8'); EG.tooltip.hide(); });
    g.appendChild(rect);

    // Label
    if (item.w > 40 && item.h > 20) {
      const txt = EG.svg('text', {
        x: item.x + item.w / 2, y: item.y + item.h / 2,
        'text-anchor': 'middle', 'dominant-baseline': 'middle',
        fill: '#fff', 'font-size': Math.min(12, item.w / 8) + 'px', 'font-weight': '600'
      });
      const dispLabel = item.label.length > item.w / 7 ? item.label.slice(0, Math.floor(item.w / 7)) + '…' : item.label;
      txt.textContent = dispLabel;
      g.appendChild(txt);

      if (item.h > 35) {
        const pctTxt = EG.svg('text', {
          x: item.x + item.w / 2, y: item.y + item.h / 2 + 14,
          'text-anchor': 'middle', fill: 'rgba(255,255,255,0.7)', 'font-size': '10px'
        });
        pctTxt.textContent = item.pct + '%';
        g.appendChild(pctTxt);
      }
    }
  });
}
"""

# ---------------------------------------------------------------------------
# Sunburst JS
# ---------------------------------------------------------------------------

_SUNBURST_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';

  const cx = size / 2, cy = size / 2;
  const maxR = size / 2 - 30;
  const palette = config.palette;
  const nodes = data.nodes;

  // Build tree structure
  const nodeMap = {};
  nodes.forEach(function(n) { nodeMap[n.label] = {label: n.label, value: n.value, children: [], parent: n.parent}; });
  const roots = [];
  nodes.forEach(function(n) {
    if (n.parent && nodeMap[n.parent]) {
      nodeMap[n.parent].children.push(nodeMap[n.label]);
    } else if (!n.parent) {
      roots.push(nodeMap[n.label]);
    }
  });

  // Compute depths
  function setDepth(node, d) {
    node.depth = d;
    node.children.forEach(function(c) { setDepth(c, d + 1); });
  }
  let maxDepth = 0;
  roots.forEach(function(r) { setDepth(r, 0); });
  function findMax(node) { maxDepth = Math.max(maxDepth, node.depth); node.children.forEach(findMax); }
  roots.forEach(findMax);
  maxDepth = maxDepth || 1;

  const ringW = maxR / (maxDepth + 1);

  // Layout arcs
  function layoutNode(node, startAngle, endAngle, colorIdx) {
    const innerR = node.depth * ringW + ringW * 0.2;
    const outerR = (node.depth + 1) * ringW - 2;
    const color = palette[colorIdx % palette.length];

    drawArc(node, innerR, outerR, startAngle, endAngle, color);

    const total = node.children.reduce(function(s, c) { return s + c.value; }, 0) || 1;
    let angle = startAngle;
    node.children.forEach(function(c, i) {
      const span = (c.value / total) * (endAngle - startAngle);
      layoutNode(c, angle, angle + span, colorIdx + i);
      angle += span;
    });
  }

  function drawArc(node, innerR, outerR, startAngle, endAngle, color) {
    if (endAngle - startAngle < 0.01) return;
    const large = (endAngle - startAngle > Math.PI) ? 1 : 0;
    const x1 = cx + innerR * Math.cos(startAngle);
    const y1 = cy + innerR * Math.sin(startAngle);
    const x2 = cx + outerR * Math.cos(startAngle);
    const y2 = cy + outerR * Math.sin(startAngle);
    const x3 = cx + outerR * Math.cos(endAngle);
    const y3 = cy + outerR * Math.sin(endAngle);
    const x4 = cx + innerR * Math.cos(endAngle);
    const y4 = cy + innerR * Math.sin(endAngle);

    const d = [
      'M', x1, y1,
      'L', x2, y2,
      'A', outerR, outerR, 0, large, 1, x3, y3,
      'L', x4, y4,
      'A', innerR, innerR, 0, large, 0, x1, y1,
      'Z'
    ].join(' ');

    const path = EG.svg('path', {d: d, fill: color, opacity: 0.8, stroke: 'var(--bg-card)', 'stroke-width': 1.5});
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(node.label) + '</b><br>Value: ' + EG.fmt(node.value));
    });
    path.addEventListener('mouseleave', function() { path.setAttribute('opacity', '0.8'); EG.tooltip.hide(); });
    svg.appendChild(path);
  }

  const totalVal = roots.reduce(function(s, r) { return s + r.value; }, 0) || 1;
  let angle = -Math.PI / 2;
  roots.forEach(function(r, i) {
    const span = (r.value / totalVal) * Math.PI * 2;
    layoutNode(r, angle, angle + span, i);
    angle += span;
  });
}
"""
