"""Chord diagram visualizer.

Interactive chord diagrams for showing pairwise relationships in a
circular layout. Ideal for confusion-style matrices, feature
interaction strengths, and model agreement visualization.

Example
-------
>>> from endgame.visualization import ChordDiagramVisualizer
>>> import numpy as np
>>> matrix = np.array([[0, 5, 3], [5, 0, 4], [3, 4, 0]])
>>> viz = ChordDiagramVisualizer(
...     matrix=matrix,
...     labels=["Model A", "Model B", "Model C"],
...     title="Model Agreement",
... )
>>> viz.save("chord.html")
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class ChordDiagramVisualizer(BaseVisualizer):
    """Interactive chord diagram visualizer.

    Parameters
    ----------
    matrix : array-like, shape (n, n)
        Symmetric or asymmetric flow matrix. ``matrix[i][j]``
        is the flow from group *i* to group *j*.
    labels : list of str
        Group labels.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=650
        Chart width.
    height : int, default=650
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        matrix: Any,
        labels: Sequence[str],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 650,
        height: int = 650,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._matrix = np.asarray(matrix, dtype=float)
        if self._matrix.ndim != 2:
            raise ValueError("matrix must be 2D")
        self.labels = list(labels)

    @classmethod
    def from_confusion_matrix(
        cls,
        matrix: Any,
        class_names: Sequence[str],
        **kwargs,
    ) -> ChordDiagramVisualizer:
        """Create a chord diagram from a confusion matrix.

        Shows misclassification flows between classes.

        Parameters
        ----------
        matrix : array-like, shape (n, n)
            Confusion matrix.
        class_names : list of str
            Class names.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("title", "Classification Flow")
        return cls(matrix, class_names, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        m = self._matrix
        n = m.shape[0]
        row_sums = m.sum(axis=1)

        # Precompute arc angles per group
        total = float(m.sum())
        gap = 0.04  # gap between groups in radians
        available = 2 * math.pi - n * gap
        if total <= 0:
            return {"groups": [], "chords": []}

        groups = []
        angle = 0.0
        for i in range(n):
            span = (row_sums[i] / total) * available if total > 0 else 0
            groups.append({
                "label": self.labels[i],
                "startAngle": round(angle, 6),
                "endAngle": round(angle + span, 6),
                "total": round(float(row_sums[i]), 4),
            })
            angle += span + gap

        # Chords
        chords = []
        for i in range(n):
            for j in range(i, n):
                val_ij = float(m[i, j])
                val_ji = float(m[j, i])
                if val_ij <= 0 and val_ji <= 0:
                    continue
                chords.append({
                    "source": i,
                    "target": j,
                    "valueIJ": round(val_ij, 4),
                    "valueJI": round(val_ji, 4),
                })

        return {
            "groups": groups,
            "chords": chords,
            "labels": self.labels,
        }

    def _chart_type(self) -> str:
        return "chord"

    def _get_chart_js(self) -> str:
        return _CHORD_JS


_CHORD_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';
  const palette = config.palette;
  const cx = size / 2, cy = size / 2;
  const outerR = size / 2 - 50;
  const innerR = outerR - 20;
  const groups = data.groups;
  const chords = data.chords;
  const labels = data.labels;

  if (groups.length === 0) return;

  function arcPath(cx, cy, r, startAngle, endAngle) {
    const x1 = cx + r * Math.cos(startAngle - Math.PI/2);
    const y1 = cy + r * Math.sin(startAngle - Math.PI/2);
    const x2 = cx + r * Math.cos(endAngle - Math.PI/2);
    const y2 = cy + r * Math.sin(endAngle - Math.PI/2);
    const large = (endAngle - startAngle > Math.PI) ? 1 : 0;
    return {x1, y1, x2, y2, large};
  }

  // Draw group arcs
  groups.forEach(function(gr, i) {
    const color = palette[i % palette.length];
    const o = arcPath(cx, cy, outerR, gr.startAngle, gr.endAngle);
    const inn = arcPath(cx, cy, innerR, gr.startAngle, gr.endAngle);

    const d = [
      'M', o.x1, o.y1,
      'A', outerR, outerR, 0, o.large, 1, o.x2, o.y2,
      'L', inn.x2, inn.y2,
      'A', innerR, innerR, 0, inn.large, 0, inn.x1, inn.y1,
      'Z'
    ].join(' ');

    const arc = EG.svg('path', {d: d, fill: color, opacity: 0.85, stroke: 'var(--bg-card)', 'stroke-width': 1});
    arc.addEventListener('mouseenter', function(e) {
      arc.setAttribute('opacity', '1');
      EG.tooltip.show(e, '<b>' + EG.esc(gr.label) + '</b><br>Total: ' + EG.fmt(gr.total));
    });
    arc.addEventListener('mouseleave', function() { arc.setAttribute('opacity', '0.85'); EG.tooltip.hide(); });
    svg.appendChild(arc);

    // Label
    const midAngle = (gr.startAngle + gr.endAngle) / 2 - Math.PI / 2;
    const lx = cx + (outerR + 18) * Math.cos(midAngle);
    const ly = cy + (outerR + 18) * Math.sin(midAngle);
    const anchor = lx > cx ? 'start' : 'end';
    const label = EG.svg('text', {
      x: lx, y: ly + 4, 'text-anchor': Math.abs(lx - cx) < 10 ? 'middle' : anchor,
      fill: 'var(--text-primary)', 'font-size': '11px', 'font-weight': '500'
    });
    label.textContent = gr.label;
    svg.appendChild(label);
  });

  // Draw chords
  chords.forEach(function(ch) {
    const sg = groups[ch.source];
    const tg = groups[ch.target];
    const color = palette[ch.source % palette.length];

    // Simplified: draw a bezier from source arc midpoint to target arc midpoint
    const sa = (sg.startAngle + sg.endAngle) / 2 - Math.PI / 2;
    const ta = (tg.startAngle + tg.endAngle) / 2 - Math.PI / 2;
    const sx = cx + innerR * Math.cos(sa);
    const sy = cy + innerR * Math.sin(sa);
    const tx = cx + innerR * Math.cos(ta);
    const ty = cy + innerR * Math.sin(ta);

    const totalVal = ch.valueIJ + ch.valueJI;
    const thickness = Math.max(1, Math.min(totalVal / (data.groups[0].total || 1) * 30, 15));

    const d = 'M' + sx + ',' + sy + ' Q' + cx + ',' + cy + ' ' + tx + ',' + ty;
    const path = EG.svg('path', {
      d: d, fill: 'none', stroke: color,
      'stroke-width': thickness, opacity: 0.25,
      'stroke-linecap': 'round'
    });
    path.addEventListener('mouseenter', function(e) {
      path.setAttribute('opacity', '0.6');
      EG.tooltip.show(e,
        '<b>' + EG.esc(labels[ch.source]) + ' ↔ ' + EG.esc(labels[ch.target]) + '</b><br>' +
        labels[ch.source] + ' → ' + labels[ch.target] + ': ' + EG.fmt(ch.valueIJ) + '<br>' +
        labels[ch.target] + ' → ' + labels[ch.source] + ': ' + EG.fmt(ch.valueJI));
    });
    path.addEventListener('mouseleave', function() { path.setAttribute('opacity', '0.25'); EG.tooltip.hide(); });
    svg.appendChild(path);
  });
}
"""
