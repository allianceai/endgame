"""Partial Dependence Plot (PDP) and Individual Conditional Expectation (ICE) visualizer.

Interactive PDP/ICE plots for understanding feature effects in any
sklearn-compatible model. Supports 1D PDP with optional ICE lines, and
2D PDP as a heatmap.

Example
-------
>>> from endgame.visualization import PDPVisualizer
>>> from sklearn.ensemble import RandomForestClassifier
>>> clf = RandomForestClassifier().fit(X_train, y_train)
>>> viz = PDPVisualizer.from_estimator(clf, X_train, feature=0)
>>> viz.save("pdp.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from endgame.visualization._base import BaseVisualizer


class PDPVisualizer(BaseVisualizer):
    """Interactive partial dependence / ICE plot visualizer.

    Parameters
    ----------
    grid_values : list of float
        Feature values on the grid.
    pdp_values : list of float
        Partial dependence values (mean ICE).
    ice_lines : list of list of float, optional
        Individual conditional expectation curves.
    feature_name : str, default=''
        Feature name for axis label.
    is_categorical : bool, default=False
        Whether feature is categorical.
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=750
        Chart width.
    height : int, default=500
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        grid_values: Sequence[float],
        pdp_values: Sequence[float],
        *,
        ice_lines: Sequence[Sequence[float]] | None = None,
        feature_name: str = "",
        is_categorical: bool = False,
        title: str = "",
        palette: str = "tableau",
        width: int = 750,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(
            title=title or f"PDP – {feature_name}" if feature_name else "Partial Dependence",
            palette=palette, width=width, height=height, theme=theme,
        )
        self._grid = list(grid_values)
        self._pdp = list(pdp_values)
        self._ice = [list(line) for line in ice_lines] if ice_lines else None
        self.feature_name = feature_name
        self.is_categorical = is_categorical

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        *,
        feature: int | str = 0,
        ice: bool = True,
        n_ice_lines: int = 50,
        grid_resolution: int = 50,
        percentiles: tuple[float, float] = (0.05, 0.95),
        target_class: int = 1,
        **kwargs,
    ) -> PDPVisualizer:
        """Create PDP/ICE from a fitted model.

        Parameters
        ----------
        model : estimator
            Fitted sklearn-compatible estimator.
        X : array-like of shape (n_samples, n_features)
            Training data (used to build the grid and compute ICE).
        feature : int or str
            Feature index or name.
        ice : bool, default=True
            Whether to include ICE lines.
        n_ice_lines : int, default=50
            Number of ICE lines to sample.
        grid_resolution : int, default=50
            Number of grid points.
        percentiles : tuple, default=(0.05, 0.95)
            Feature range percentiles.
        target_class : int, default=1
            For classifiers, which class probability to show.
        **kwargs
            Additional keyword arguments.
        """
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Resolve feature index
        if isinstance(feature, str):
            if hasattr(X, "columns"):
                feat_idx = list(X.columns).index(feature)
                feat_name = feature
            else:
                raise ValueError(f"Cannot resolve feature name '{feature}' from non-DataFrame input")
        else:
            feat_idx = int(feature)
            if hasattr(X, "columns"):
                feat_name = str(X.columns[feat_idx])
            else:
                feat_name = f"Feature {feat_idx}"

        col = X_arr[:, feat_idx]

        # Check if categorical (few unique values)
        unique_vals = np.unique(col)
        is_cat = len(unique_vals) <= 20 and np.all(unique_vals == unique_vals.astype(int))

        if is_cat:
            grid = sorted(unique_vals.tolist())
        else:
            lo, hi = np.percentile(col, [percentiles[0] * 100, percentiles[1] * 100])
            grid = np.linspace(lo, hi, grid_resolution).tolist()

        # Compute ICE / PDP
        has_proba = hasattr(model, "predict_proba")
        n_samples = X_arr.shape[0]

        # Sample subset for ICE
        if ice and n_samples > n_ice_lines:
            ice_idx = np.random.choice(n_samples, size=n_ice_lines, replace=False)
        else:
            ice_idx = np.arange(min(n_samples, n_ice_lines)) if ice else np.array([], dtype=int)

        all_ice = []
        pdp_vals = []

        for g_val in grid:
            X_mod = X_arr.copy()
            X_mod[:, feat_idx] = g_val
            if has_proba:
                preds = model.predict_proba(X_mod)[:, target_class]
            else:
                preds = model.predict(X_mod)

            pdp_vals.append(round(float(preds.mean()), 6))

            if ice and len(ice_idx) > 0:
                ice_at_g = preds[ice_idx]
                all_ice.append([round(float(v), 6) for v in ice_at_g])

        # Transpose ICE: all_ice is [grid_points x n_ice] → we want [n_ice x grid_points]
        ice_lines_out = None
        if all_ice:
            n_ice_actual = len(all_ice[0])
            ice_lines_out = []
            for j in range(n_ice_actual):
                ice_lines_out.append([all_ice[i][j] for i in range(len(grid))])

        grid_out = [round(float(v), 4) for v in grid]

        kwargs.setdefault("feature_name", feat_name)
        kwargs.setdefault("is_categorical", bool(is_cat))
        return cls(grid_out, pdp_vals, ice_lines=ice_lines_out, **kwargs)

    @classmethod
    def from_precomputed(
        cls,
        grid_values: Sequence[float],
        pdp_values: Sequence[float],
        *,
        ice_lines: Sequence[Sequence[float]] | None = None,
        feature_name: str = "",
        **kwargs,
    ) -> PDPVisualizer:
        """Create from precomputed PDP/ICE values.

        Parameters
        ----------
        grid_values : list of float
            Feature grid values.
        pdp_values : list of float
            Partial dependence values.
        ice_lines : list of list of float, optional
            ICE lines.
        feature_name : str, optional
            Feature name.
        **kwargs
            Additional keyword arguments.
        """
        return cls(grid_values, pdp_values, ice_lines=ice_lines,
                   feature_name=feature_name, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        return {
            "grid": self._grid,
            "pdp": self._pdp,
            "ice": self._ice,
            "featureName": self.feature_name,
            "isCategorical": self.is_categorical,
        }

    def _chart_type(self) -> str:
        return "pdp"

    def _get_chart_js(self) -> str:
        return _PDP_JS


class PDP2DVisualizer(BaseVisualizer):
    """2D partial dependence plot (heatmap).

    Parameters
    ----------
    grid_x : list of float
        Grid values for feature x.
    grid_y : list of float
        Grid values for feature y.
    values : list of list of float
        2D matrix of PD values (rows = y, cols = x).
    feature_x : str, default=''
        Feature x name.
    feature_y : str, default=''
        Feature y name.
    """

    def __init__(
        self,
        grid_x: Sequence[float],
        grid_y: Sequence[float],
        values: Sequence[Sequence[float]],
        *,
        feature_x: str = "",
        feature_y: str = "",
        title: str = "",
        palette: str = "tableau",
        width: int = 700,
        height: int = 600,
        theme: str = "dark",
    ):
        super().__init__(
            title=title or f"2D PDP – {feature_x} × {feature_y}",
            palette=palette, width=width, height=height, theme=theme,
        )
        self._grid_x = list(grid_x)
        self._grid_y = list(grid_y)
        self._values = [list(row) for row in values]
        self.feature_x = feature_x
        self.feature_y = feature_y

    @classmethod
    def from_estimator(
        cls,
        model: Any,
        X: Any,
        *,
        features: tuple[int | str, int | str] = (0, 1),
        grid_resolution: int = 25,
        percentiles: tuple[float, float] = (0.05, 0.95),
        target_class: int = 1,
        **kwargs,
    ) -> PDP2DVisualizer:
        """Create 2D PDP from a fitted model.

        Parameters
        ----------
        model : estimator
            Fitted sklearn-compatible estimator.
        X : array-like
            Training data.
        features : tuple of (int or str, int or str)
            Two feature indices or names.
        grid_resolution : int, default=25
            Grid resolution per axis.
        percentiles : tuple, default=(0.05, 0.95)
            Feature range percentiles.
        target_class : int, default=1
            For classifiers, which class probability.
        **kwargs
            Additional keyword arguments.
        """
        X_arr = np.asarray(X)
        has_cols = hasattr(X, "columns")

        feat_idxs = []
        feat_names = []
        for f in features:
            if isinstance(f, str):
                if has_cols:
                    idx = list(X.columns).index(f)
                    feat_idxs.append(idx)
                    feat_names.append(f)
                else:
                    raise ValueError(f"Cannot resolve feature name '{f}'")
            else:
                feat_idxs.append(int(f))
                feat_names.append(str(X.columns[int(f)]) if has_cols else f"Feature {f}")

        grids = []
        for fi in feat_idxs:
            col = X_arr[:, fi]
            lo, hi = np.percentile(col, [percentiles[0] * 100, percentiles[1] * 100])
            grids.append(np.linspace(lo, hi, grid_resolution))

        has_proba = hasattr(model, "predict_proba")
        values = []

        for y_val in grids[1]:
            row = []
            for x_val in grids[0]:
                X_mod = X_arr.copy()
                X_mod[:, feat_idxs[0]] = x_val
                X_mod[:, feat_idxs[1]] = y_val
                if has_proba:
                    preds = model.predict_proba(X_mod)[:, target_class]
                else:
                    preds = model.predict(X_mod)
                row.append(round(float(preds.mean()), 6))
            values.append(row)

        kwargs.setdefault("feature_x", feat_names[0])
        kwargs.setdefault("feature_y", feat_names[1])
        return cls(
            [round(float(v), 4) for v in grids[0]],
            [round(float(v), 4) for v in grids[1]],
            values, **kwargs,
        )

    def _build_data(self) -> dict[str, Any]:
        return {
            "gridX": self._grid_x,
            "gridY": self._grid_y,
            "values": self._values,
            "featureX": self.feature_x,
            "featureY": self.feature_y,
        }

    def _chart_type(self) -> str:
        return "pdp2d"

    def _get_chart_js(self) -> str:
        return _PDP2D_JS


# ---------------------------------------------------------------------------
# 1D PDP/ICE JavaScript
# ---------------------------------------------------------------------------

_PDP_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 20, bottom: 55, left: 60};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;
  const palette = config.palette;
  const grid = data.grid;
  const pdp = data.pdp;
  const ice = data.ice;
  const n = grid.length;

  // Find y range from PDP + ICE
  let yMin = Infinity, yMax = -Infinity;
  pdp.forEach(function(v) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; });
  if (ice) {
    ice.forEach(function(line) {
      line.forEach(function(v) { if (v < yMin) yMin = v; if (v > yMax) yMax = v; });
    });
  }
  const yPad = (yMax - yMin) * 0.08 || 0.01;
  yMin -= yPad; yMax += yPad;

  let xScale, isBand = false;
  if (data.isCategorical) {
    isBand = true;
    xScale = EG.scaleBand(grid.map(String), [0, W], 0.3);
  } else {
    xScale = EG.scaleLinear([grid[0], grid[n-1]], [0, W]);
  }
  const yScale = EG.scaleLinear([yMin, yMax], [H, 0]);

  EG.drawXAxis(g, xScale, H, data.featureName || 'Feature Value', isBand);
  EG.drawYAxis(g, yScale, W, 'Partial Dependence');

  // ICE lines (subtle background)
  if (ice && ice.length > 0) {
    const iceG = EG.svg('g', {opacity: 0.15});
    g.appendChild(iceG);
    ice.forEach(function(line) {
      let d = '';
      for (let i = 0; i < n; i++) {
        const x = isBand ? xScale(String(grid[i])) + xScale.bandwidth / 2 : xScale(grid[i]);
        d += (i === 0 ? 'M' : ' L') + x + ' ' + yScale(line[i]);
      }
      iceG.appendChild(EG.svg('path', {d:d, fill:'none', stroke: palette[0], 'stroke-width': 1}));
    });
  }

  // PDP main line
  let pdpD = '';
  for (let i = 0; i < n; i++) {
    const x = isBand ? xScale(String(grid[i])) + xScale.bandwidth / 2 : xScale(grid[i]);
    pdpD += (i === 0 ? 'M' : ' L') + x + ' ' + yScale(pdp[i]);
  }
  g.appendChild(EG.svg('path', {
    d: pdpD, fill: 'none', stroke: palette[0],
    'stroke-width': 3.5, 'stroke-linejoin': 'round'
  }));

  // Interactive dots on PDP
  for (let i = 0; i < n; i++) {
    const x = isBand ? xScale(String(grid[i])) + xScale.bandwidth / 2 : xScale(grid[i]);
    const y = yScale(pdp[i]);
    const dot = EG.svg('circle', {cx: x, cy: y, r: 4.5, fill: palette[0], stroke: 'var(--bg-card)', 'stroke-width': 2, opacity: 0});
    dot.addEventListener('mouseenter', function(e) {
      dot.setAttribute('opacity', '1');
      dot.setAttribute('r', '6');
      let html = '<b>' + (data.featureName || 'Feature') + ' = ' + EG.fmt(grid[i], 3) + '</b><br>PD = ' + EG.fmt(pdp[i], 4);
      if (ice) html += '<br><span style="opacity:0.6">' + ice.length + ' ICE lines</span>';
      EG.tooltip.show(e, html);
    });
    dot.addEventListener('mouseleave', function() {
      dot.setAttribute('opacity', '0');
      dot.setAttribute('r', '4.5');
      EG.tooltip.hide();
    });
    g.appendChild(dot);
  }

  // Legend
  const items = [{label: 'PDP (mean)', color: palette[0]}];
  if (ice) items.push({label: 'ICE lines', color: palette[0]});
  EG.drawLegend(container, items);
}
"""

# ---------------------------------------------------------------------------
# 2D PDP JavaScript (heatmap-style)
# ---------------------------------------------------------------------------

_PDP2D_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const margin = {top: 20, right: 90, bottom: 55, left: 65};
  const ctx = EG.createSVG(container, config.width, config.height, margin);
  const {g, width: W, height: H} = ctx;

  const gx = data.gridX, gy = data.gridY;
  const nx = gx.length, ny = gy.length;
  const vals = data.values;

  // Flatten to find range
  let vMin = Infinity, vMax = -Infinity;
  vals.forEach(function(row) { row.forEach(function(v) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; }); });

  const cellW = W / nx;
  const cellH = H / ny;

  // Color scale (blue → white → red)
  function heatColor(v) {
    const t = (v - vMin) / (vMax - vMin + 1e-10);
    if (t <= 0.5) {
      const s = t * 2;
      return interpolate('#2166ac', '#f7f7f7', s);
    } else {
      const s = (t - 0.5) * 2;
      return interpolate('#f7f7f7', '#b2182b', s);
    }
  }
  function interpolate(c1, c2, t) {
    const r1 = parseInt(c1.slice(1,3),16), g1 = parseInt(c1.slice(3,5),16), b1 = parseInt(c1.slice(5,7),16);
    const r2 = parseInt(c2.slice(1,3),16), g2 = parseInt(c2.slice(3,5),16), b2 = parseInt(c2.slice(5,7),16);
    const r = Math.round(r1 + (r2-r1)*t), gg = Math.round(g1 + (g2-g1)*t), b = Math.round(b1 + (b2-b1)*t);
    return 'rgb('+r+','+gg+','+b+')';
  }

  // Draw cells
  for (let yi = 0; yi < ny; yi++) {
    for (let xi = 0; xi < nx; xi++) {
      const v = vals[yi][xi];
      const rect = EG.svg('rect', {
        x: xi * cellW, y: (ny - 1 - yi) * cellH,
        width: cellW + 0.5, height: cellH + 0.5,
        fill: heatColor(v), stroke: 'none'
      });
      rect.addEventListener('mouseenter', function(e) {
        rect.setAttribute('stroke', 'var(--text-primary)');
        rect.setAttribute('stroke-width', '2');
        EG.tooltip.show(e,
          '<b>' + EG.esc(data.featureX) + '</b> = ' + EG.fmt(gx[xi], 3) + '<br>' +
          '<b>' + EG.esc(data.featureY) + '</b> = ' + EG.fmt(gy[yi], 3) + '<br>' +
          'PD = ' + EG.fmt(v, 4));
      });
      rect.addEventListener('mouseleave', function() {
        rect.setAttribute('stroke', 'none');
        EG.tooltip.hide();
      });
      g.appendChild(rect);
    }
  }

  // Axes
  const xTicks = EG.niceTicks(gx[0], gx[nx-1], 5);
  const xS = EG.scaleLinear([gx[0], gx[nx-1]], [0, W]);
  const yS = EG.scaleLinear([gy[0], gy[ny-1]], [H, 0]);
  xTicks.forEach(function(v) {
    g.appendChild(EG.svg('text', {x:xS(v), y:H+18, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'10px'})).textContent = EG.fmt(v, 2);
  });
  g.appendChild(EG.svg('text', {x:W/2, y:H+42, 'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'12px', 'font-weight':'500'})).textContent = data.featureX;

  const yTicks = EG.niceTicks(gy[0], gy[ny-1], 5);
  yTicks.forEach(function(v) {
    g.appendChild(EG.svg('text', {x:-8, y:yS(v)+4, 'text-anchor':'end', fill:'var(--text-secondary)', 'font-size':'10px'})).textContent = EG.fmt(v, 2);
  });
  g.appendChild(EG.svg('text', {'text-anchor':'middle', fill:'var(--text-secondary)', 'font-size':'12px', 'font-weight':'500',
    transform:`translate(-50,${H/2}) rotate(-90)`})).textContent = data.featureY;

  // Color bar
  const barX = W + 15, barW = 18, barH = H;
  const nSteps = 50;
  for (let i = 0; i < nSteps; i++) {
    const t = i / (nSteps - 1);
    const v = vMin + t * (vMax - vMin);
    g.appendChild(EG.svg('rect', {
      x: barX, y: barH - (i+1) * barH/nSteps,
      width: barW, height: barH/nSteps + 0.5,
      fill: heatColor(v), stroke: 'none'
    }));
  }
  g.appendChild(EG.svg('text', {x:barX+barW+5, y:5, fill:'var(--text-secondary)', 'font-size':'10px', 'dominant-baseline':'middle'})).textContent = EG.fmt(vMax, 3);
  g.appendChild(EG.svg('text', {x:barX+barW+5, y:H, fill:'var(--text-secondary)', 'font-size':'10px', 'dominant-baseline':'middle'})).textContent = EG.fmt(vMin, 3);
}
"""
