"""Venn diagram visualizer.

Interactive Venn diagrams for showing set overlaps (2 or 3 sets).

Example
-------
>>> from endgame.visualization import VennDiagramVisualizer
>>> viz = VennDiagramVisualizer(
...     sets={"Model A": 150, "Model B": 120},
...     intersections={"Model A&Model B": 80},
...     title="Prediction Overlap",
... )
>>> viz.save("venn.html")
"""

from __future__ import annotations

from typing import Any

from endgame.visualization._base import BaseVisualizer


class VennDiagramVisualizer(BaseVisualizer):
    """Interactive Venn diagram visualizer (2 or 3 sets).

    Parameters
    ----------
    sets : dict of str → int
        Set name → total count.
    intersections : dict of str → int
        Intersection keys (e.g., 'A&B', 'A&B&C') → count.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=600
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        sets: dict[str, int],
        intersections: dict[str, int],
        *,
        title: str = "",
        palette: str = "tableau",
        width: int = 600,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._sets = {k: int(v) for k, v in sets.items()}
        self._intersections = {k: int(v) for k, v in intersections.items()}

    def _build_data(self) -> dict[str, Any]:
        names = list(self._sets.keys())
        sizes = [self._sets[n] for n in names]
        intersections = {}
        for key, val in self._intersections.items():
            intersections[key] = val

        return {
            "names": names,
            "sizes": sizes,
            "intersections": intersections,
            "nSets": len(names),
        }

    def _chart_type(self) -> str:
        return "venn"

    def _get_chart_js(self) -> str:
        return _VENN_JS


_VENN_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const size = Math.min(config.width, config.height);
  const svg = EG.svg('svg', {width: size, height: size});
  container.appendChild(svg);
  container.style.width = size + 'px';
  container.style.height = size + 'px';
  const palette = config.palette;
  const names = data.names;
  const sizes = data.sizes;
  const n = data.nSets;
  const cx = size / 2, cy = size / 2;

  // Circle positions
  const R = size * 0.22;
  const offset = size * 0.12;
  const circles = [];

  if (n === 2) {
    circles.push({x: cx - offset * 0.6, y: cy, r: R});
    circles.push({x: cx + offset * 0.6, y: cy, r: R});
  } else if (n >= 3) {
    const angleStep = (2 * Math.PI) / Math.min(n, 3);
    for (let i = 0; i < Math.min(n, 3); i++) {
      const a = -Math.PI / 2 + i * angleStep;
      circles.push({x: cx + offset * Math.cos(a), y: cy + offset * Math.sin(a), r: R});
    }
  }

  // Draw circles
  circles.forEach(function(c, i) {
    const color = palette[i % palette.length];
    const circle = EG.svg('circle', {
      cx: c.x, cy: c.y, r: c.r,
      fill: color, opacity: 0.25,
      stroke: color, 'stroke-width': 2.5
    });
    circle.addEventListener('mouseenter', function(e) {
      circle.setAttribute('opacity', '0.4');
      EG.tooltip.show(e, '<b>' + EG.esc(names[i]) + '</b><br>Size: ' + sizes[i]);
    });
    circle.addEventListener('mouseleave', function() {
      circle.setAttribute('opacity', '0.25');
      EG.tooltip.hide();
    });
    svg.appendChild(circle);

    // Label
    const lx = c.x + (c.x > cx ? 20 : c.x < cx ? -20 : 0);
    const ly = c.y + (c.y > cy ? R + 20 : c.y < cy ? -R - 10 : -R - 10);
    const label = EG.svg('text', {
      x: lx, y: ly, 'text-anchor': 'middle',
      fill: 'var(--text-primary)', 'font-size': '13px', 'font-weight': '600'
    });
    label.textContent = names[i];
    svg.appendChild(label);

    // Size text on circle
    const sx = c.x + (c.x > cx ? R*0.3 : c.x < cx ? -R*0.3 : 0);
    const sy = c.y + (n === 2 ? 0 : (c.y > cy ? R*0.3 : -R*0.3));
    const sizeText = EG.svg('text', {
      x: sx, y: sy + 4, 'text-anchor': 'middle',
      fill: 'var(--text-secondary)', 'font-size': '14px', 'font-weight': '500'
    });
    sizeText.textContent = sizes[i];
    svg.appendChild(sizeText);
  });

  // Intersection labels
  const intKeys = Object.keys(data.intersections);
  intKeys.forEach(function(key) {
    const val = data.intersections[key];
    // Show at center for now
    const txt = EG.svg('text', {
      x: cx, y: cy + 4, 'text-anchor': 'middle',
      fill: 'var(--text-primary)', 'font-size': '16px', 'font-weight': '700'
    });
    txt.textContent = val;
    svg.appendChild(txt);
  });
}
"""
