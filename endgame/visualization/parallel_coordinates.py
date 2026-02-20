"""Parallel coordinates visualizer.

Interactive parallel coordinates plot for hyperparameter search
visualization and multi-dimensional data exploration.

Example
-------
>>> from endgame.visualization import ParallelCoordinatesVisualizer
>>> data = [
...     {"lr": 0.001, "depth": 6, "n_est": 100, "score": 0.92},
...     {"lr": 0.01, "depth": 4, "n_est": 200, "score": 0.89},
... ]
>>> viz = ParallelCoordinatesVisualizer(data, color_by="score")
>>> viz.save("parallel_coords.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class ParallelCoordinatesVisualizer(BaseVisualizer):
    """Interactive parallel coordinates visualizer.

    Parameters
    ----------
    data : list of dict
        List of records, each a dict of dimension_name → value.
    dimensions : list of str, optional
        Which dimensions to show (and in what order). If None, use all.
    color_by : str, optional
        Dimension name to use for line coloring.
    cmap : str, default='viridis_seq'
        Color palette for continuous coloring.
    title : str, optional
        Chart title.
    width : int, default=900
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        data: Sequence[dict[str, Any]],
        *,
        dimensions: Sequence[str] | None = None,
        color_by: str | None = None,
        cmap: str = "viridis_seq",
        title: str = "",
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=cmap, width=width, height=height, theme=theme)
        self._records = list(data)
        if dimensions is None:
            # Use all keys from first record
            dimensions = list(self._records[0].keys()) if self._records else []
        self.dimensions = list(dimensions)
        self.color_by = color_by

    @classmethod
    def from_optuna_study(
        cls,
        study: Any,
        *,
        n_trials: int | None = None,
        **kwargs,
    ) -> ParallelCoordinatesVisualizer:
        """Create from an Optuna study.

        Parameters
        ----------
        study : optuna.Study
            Completed Optuna study.
        n_trials : int, optional
            Max number of trials to show.
        **kwargs
            Additional keyword arguments.
        """
        trials = [t for t in study.trials if t.state.name == "COMPLETE"]
        if n_trials:
            trials = trials[:n_trials]

        records = []
        for t in trials:
            rec = dict(t.params)
            rec["objective"] = t.value
            records.append(rec)

        kwargs.setdefault("color_by", "objective")
        kwargs.setdefault("title", "Hyperparameter Search")
        return cls(records, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        dims = self.dimensions
        axes = []
        for dim in dims:
            values = [r.get(dim) for r in self._records]
            # Determine if numeric
            numeric = all(isinstance(v, (int, float)) for v in values if v is not None)
            if numeric:
                clean = [float(v) for v in values if v is not None]
                axes.append({
                    "name": dim,
                    "type": "numeric",
                    "values": [float(v) if v is not None else None for v in values],
                    "min": min(clean) if clean else 0,
                    "max": max(clean) if clean else 1,
                })
            else:
                unique = sorted(set(str(v) for v in values if v is not None))
                axes.append({
                    "name": dim,
                    "type": "categorical",
                    "values": [str(v) if v is not None else "" for v in values],
                    "categories": unique,
                })

        color_data = None
        if self.color_by:
            color_vals = [r.get(self.color_by) for r in self._records]
            clean = [float(v) for v in color_vals if v is not None and isinstance(v, (int, float))]
            if clean:
                color_data = {
                    "values": [float(v) if isinstance(v, (int, float)) else None for v in color_vals],
                    "min": min(clean),
                    "max": max(clean),
                    "name": self.color_by,
                }

        return {
            "axes": axes,
            "nRecords": len(self._records),
            "colorData": color_data,
        }

    def _chart_type(self) -> str:
        return "parallel_coordinates"

    def _get_chart_js(self) -> str:
        return _PARCOORD_JS


_PARCOORD_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 30, right: 30, bottom: 30, left: 30};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const axes = data.axes;
  const nAxes = axes.length;
  const nRec = data.nRecords;
  if (nAxes < 2 || nRec === 0) return;

  const axisSpacing = W / (nAxes - 1);

  // Color function
  let colorFn;
  if (data.colorData) {
    const cs = EG.colorScale(palette, data.colorData.min, data.colorData.max);
    colorFn = function(i) {
      const v = data.colorData.values[i];
      return v !== null ? cs(v) : 'var(--text-muted)';
    };
  } else {
    colorFn = function() { return palette[0] || 'var(--accent)'; };
  }

  // Scale functions per axis
  const scales = axes.map(function(ax, ai) {
    const x = ai * axisSpacing;
    if (ax.type === 'numeric') {
      return function(v) {
        if (v === null) return H / 2;
        return H - (v - ax.min) / (ax.max - ax.min || 1) * H;
      };
    } else {
      const cats = ax.categories;
      return function(v) {
        const idx = cats.indexOf(String(v));
        return H - (idx + 0.5) / cats.length * H;
      };
    }
  });

  // Draw lines
  for (let i = 0; i < nRec; i++) {
    let d = '';
    for (let a = 0; a < nAxes; a++) {
      const x = a * axisSpacing;
      const y = scales[a](axes[a].values[i]);
      d += (a === 0 ? 'M' : ' L') + x + ' ' + y;
    }
    const line = EG.svg('path', {
      d: d, fill: 'none', stroke: colorFn(i),
      'stroke-width': 1.5, opacity: 0.4
    });
    line.addEventListener('mouseenter', function(e) {
      line.setAttribute('opacity', '1');
      line.setAttribute('stroke-width', '3');
      let html = '';
      axes.forEach(function(ax) {
        html += '<b>' + EG.esc(ax.name) + ':</b> ' + ax.values[i] + '<br>';
      });
      EG.tooltip.show(e, html);
    });
    line.addEventListener('mouseleave', function() {
      line.setAttribute('opacity', '0.4');
      line.setAttribute('stroke-width', '1.5');
      EG.tooltip.hide();
    });
    g.appendChild(line);
  }

  // Draw axes
  for (let a = 0; a < nAxes; a++) {
    const x = a * axisSpacing;
    const ax = axes[a];
    g.appendChild(EG.svg('line', {x1: x, y1: 0, x2: x, y2: H, stroke: 'var(--border)', 'stroke-width': 1.5}));

    // Axis label
    const label = EG.svg('text', {x: x, y: -10, 'text-anchor': 'middle', fill: 'var(--text-primary)', 'font-size': '11px', 'font-weight': '600'});
    label.textContent = ax.name.length > 12 ? ax.name.slice(0,10) + '…' : ax.name;
    g.appendChild(label);

    // Ticks
    if (ax.type === 'numeric') {
      const ticks = EG.niceTicks(ax.min, ax.max, 4);
      ticks.forEach(function(v) {
        const y = scales[a](v);
        const t = EG.svg('text', {x: x - 5, y: y + 3, 'text-anchor': 'end', fill: 'var(--text-muted)', 'font-size': '9px'});
        t.textContent = EG.fmt(v);
        g.appendChild(t);
      });
    } else {
      ax.categories.forEach(function(cat, ci) {
        const y = H - (ci + 0.5) / ax.categories.length * H;
        const t = EG.svg('text', {x: x - 5, y: y + 3, 'text-anchor': 'end', fill: 'var(--text-muted)', 'font-size': '9px'});
        t.textContent = cat.length > 8 ? cat.slice(0,6) + '…' : cat;
        g.appendChild(t);
      });
    }
  }

  // Color bar legend
  if (data.colorData) {
    const cbH = H * 0.6, cbW = 12;
    const cbG = EG.svg('g', {transform: `translate(${W + 15}, ${(H - cbH)/2})`});
    g.appendChild(cbG);
    const nSteps = 30;
    const cs = EG.colorScale(palette, data.colorData.min, data.colorData.max);
    for (let i = 0; i < nSteps; i++) {
      const v = data.colorData.max - (i / nSteps) * (data.colorData.max - data.colorData.min);
      cbG.appendChild(EG.svg('rect', {x:0, y: i * (cbH/nSteps), width: cbW, height: cbH/nSteps+1, fill: cs(v)}));
    }
    cbG.appendChild(EG.svg('text', {x: cbW+4, y: 10, fill:'var(--text-secondary)', 'font-size':'9px'})).textContent = EG.fmt(data.colorData.max);
    cbG.appendChild(EG.svg('text', {x: cbW+4, y: cbH, fill:'var(--text-secondary)', 'font-size':'9px'})).textContent = EG.fmt(data.colorData.min);
    cbG.appendChild(EG.svg('text', {x: cbW/2, y: -8, 'text-anchor':'middle', fill:'var(--text-muted)', 'font-size':'9px'})).textContent = data.colorData.name;
  }
}
"""
