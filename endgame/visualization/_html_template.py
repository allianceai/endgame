"""Shared HTML/CSS/JS template for endgame chart visualizations.

Provides:
- CSS variables for light/dark themes
- Shared JS utilities: tooltip, axis drawing, legend, color scales, formatting
- Self-contained HTML shell (no CDN dependencies)
"""

from __future__ import annotations


def render_html(
    *,
    chart_type: str,
    data_json: str,
    config_json: str,
    chart_js: str,
    title: str,
    theme: str = "dark",
    width: int = 900,
    height: int = 500,
    embedded: bool = False,
) -> str:
    """Render a complete self-contained HTML page for a chart.

    Parameters
    ----------
    chart_type : str
        Chart identifier (e.g., 'bar', 'heatmap').
    data_json : str
        JSON string of chart data.
    config_json : str
        JSON string of chart configuration.
    chart_js : str
        Chart-specific JavaScript code defining ``renderChart(data, config)``.
    title : str
        Page/chart title (already HTML-escaped).
    theme : str
        'dark' or 'light'.
    width, height : int
        Chart dimensions in pixels.
    embedded : bool
        If True, uses a fixed-height container for Jupyter embedding.

    Returns
    -------
    str
        Complete HTML document.
    """
    container_height = f"{height + 80}px" if embedded else "100vh"

    return f"""<!DOCTYPE html>
<html lang="en" data-theme="{theme}">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<style>
{_BASE_CSS}
</style>
</head>
<body style="height: {container_height};">
<div id="chart-wrapper">
  <h1 id="chart-title">{title}</h1>
  <div id="chart-container" style="width:{width}px; height:{height}px;"></div>
  <div id="tooltip" class="tooltip"></div>
</div>
<script>
// ---- Data & Config ----
const CHART_DATA = {data_json};
const CHART_CONFIG = {config_json};

// ---- Shared Utilities ----
{_SHARED_JS}

// ---- Chart-Specific Code ----
{chart_js}

// ---- Bootstrap ----
document.addEventListener('DOMContentLoaded', function() {{
  renderChart(CHART_DATA, CHART_CONFIG);
}});
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Base CSS with light/dark theme support
# ---------------------------------------------------------------------------

_BASE_CSS = """
:root {
  --bg-primary: #0f1117;
  --bg-secondary: #1a1b26;
  --bg-card: #1e1f2e;
  --text-primary: #e0e0e0;
  --text-secondary: #a0a0b0;
  --text-muted: #606070;
  --border: #2a2b3d;
  --accent: #4e79a7;
  --grid-line: rgba(255,255,255,0.08);
  --tooltip-bg: rgba(15,17,23,0.95);
  --tooltip-border: rgba(78,121,167,0.3);
}

[data-theme="light"] {
  --bg-primary: #ffffff;
  --bg-secondary: #f8f9fa;
  --bg-card: #ffffff;
  --text-primary: #1a1a2e;
  --text-secondary: #555570;
  --text-muted: #999;
  --border: #e0e0e0;
  --accent: #4e79a7;
  --grid-line: rgba(0,0,0,0.08);
  --tooltip-bg: rgba(255,255,255,0.97);
  --tooltip-border: rgba(78,121,167,0.3);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  display: flex;
  justify-content: center;
  align-items: flex-start;
  padding: 20px;
  overflow: auto;
}

#chart-wrapper {
  display: flex;
  flex-direction: column;
  align-items: center;
  width: 100%;
}

#chart-title {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 12px;
  text-align: center;
}

#chart-container {
  position: relative;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
}

#chart-container svg {
  display: block;
}

.tooltip {
  position: fixed;
  padding: 8px 12px;
  background: var(--tooltip-bg);
  border: 1px solid var(--tooltip-border);
  border-radius: 6px;
  font-size: 12px;
  color: var(--text-primary);
  pointer-events: none;
  opacity: 0;
  transition: opacity 0.15s ease;
  z-index: 1000;
  max-width: 300px;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
  line-height: 1.5;
}

.tooltip.visible {
  opacity: 1;
}

.legend {
  display: flex;
  flex-wrap: wrap;
  gap: 12px;
  justify-content: center;
  margin-top: 8px;
  font-size: 12px;
  color: var(--text-secondary);
}

.legend-item {
  display: flex;
  align-items: center;
  gap: 5px;
  cursor: default;
}

.legend-swatch {
  width: 12px;
  height: 12px;
  border-radius: 2px;
  flex-shrink: 0;
}
"""


# ---------------------------------------------------------------------------
# Shared JavaScript utilities
# ---------------------------------------------------------------------------

_SHARED_JS = r"""
const EG = {};

// ---- SVG namespace ----
EG.SVG_NS = 'http://www.w3.org/2000/svg';

// ---- Create SVG element ----
EG.svg = function(tag, attrs) {
  const el = document.createElementNS(EG.SVG_NS, tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      el.setAttribute(k, v);
    }
  }
  return el;
};

// ---- Create root SVG ----
EG.createSVG = function(container, w, h, margin) {
  margin = margin || {top: 40, right: 30, bottom: 50, left: 60};
  const svg = EG.svg('svg', {width: w, height: h});
  container.appendChild(svg);
  const plotArea = EG.svg('g', {transform: `translate(${margin.left},${margin.top})`});
  svg.appendChild(plotArea);
  const iw = w - margin.left - margin.right;
  const ih = h - margin.top - margin.bottom;
  return {svg, g: plotArea, width: iw, height: ih, margin};
};

// ---- Number formatting ----
EG.fmt = function(v, decimals) {
  if (v === null || v === undefined) return '';
  decimals = decimals !== undefined ? decimals : 2;
  if (Math.abs(v) >= 1e6) return (v/1e6).toFixed(1) + 'M';
  if (Math.abs(v) >= 1e3) return (v/1e3).toFixed(1) + 'K';
  return Number(v).toFixed(decimals);
};

EG.pct = function(v) { return (v * 100).toFixed(1) + '%'; };

// ---- HTML escaping ----
EG.esc = function(s) {
  const div = document.createElement('div');
  div.textContent = s;
  return div.innerHTML;
};

// ---- Tooltip ----
EG.tooltip = (function() {
  let el = null;
  return {
    show: function(evt, html) {
      if (!el) el = document.getElementById('tooltip');
      el.innerHTML = html;
      el.classList.add('visible');
      const x = evt.clientX + 12;
      const y = evt.clientY - 10;
      el.style.left = Math.min(x, window.innerWidth - 280) + 'px';
      el.style.top = Math.min(y, window.innerHeight - 100) + 'px';
    },
    hide: function() {
      if (!el) el = document.getElementById('tooltip');
      el.classList.remove('visible');
    }
  };
})();

// ---- Linear scale ----
EG.scaleLinear = function(domain, range) {
  const d0 = domain[0], d1 = domain[1];
  const r0 = range[0], r1 = range[1];
  const span = d1 - d0 || 1;
  function scale(v) { return r0 + (v - d0) / span * (r1 - r0); }
  scale.invert = function(v) { return d0 + (v - r0) / (r1 - r0) * span; };
  scale.domain = domain;
  scale.range = range;
  return scale;
};

// ---- Band scale (for categories) ----
EG.scaleBand = function(labels, range, padding) {
  padding = padding || 0.1;
  const r0 = range[0], r1 = range[1];
  const n = labels.length;
  const totalPadding = padding * (n + 1);
  const bandWidth = (r1 - r0 - totalPadding * (r1 - r0) / (n + n * padding)) / n;
  const step = bandWidth + padding * bandWidth;
  const offset = (r1 - r0 - n * step + padding * bandWidth) / 2;
  function scale(label) {
    const i = labels.indexOf(label);
    return r0 + offset + i * step;
  }
  scale.bandwidth = function() { return bandWidth; };
  scale.labels = labels;
  return scale;
};

// ---- Nice axis ticks ----
EG.niceTicks = function(min, max, count) {
  count = count || 6;
  if (min === max) { return [min]; }
  const range = max - min;
  const rough = range / count;
  const mag = Math.pow(10, Math.floor(Math.log10(rough)));
  const residual = rough / mag;
  let nice;
  if (residual <= 1.5) nice = 1 * mag;
  else if (residual <= 3) nice = 2 * mag;
  else if (residual <= 7) nice = 5 * mag;
  else nice = 10 * mag;
  const lo = Math.floor(min / nice) * nice;
  const hi = Math.ceil(max / nice) * nice;
  const ticks = [];
  for (let v = lo; v <= hi + nice * 0.5; v += nice) {
    ticks.push(Math.round(v * 1e10) / 1e10);
  }
  return ticks;
};

// ---- Draw X axis ----
EG.drawXAxis = function(g, scale, height, label, isBand) {
  const axisG = EG.svg('g', {transform: `translate(0,${height})`});
  g.appendChild(axisG);

  // Baseline
  axisG.appendChild(EG.svg('line', {x1: 0, y1: 0, x2: scale.range ? scale.range[1] : 0, y2: 0,
    stroke: 'var(--border)', 'stroke-width': 1}));

  if (isBand) {
    const labels = scale.labels || [];
    const bw = scale.bandwidth();
    labels.forEach(function(lbl) {
      const x = scale(lbl) + bw / 2;
      const tick = EG.svg('text', {x: x, y: 22, 'text-anchor': 'middle',
        fill: 'var(--text-secondary)', 'font-size': '11px'});
      tick.textContent = lbl.length > 12 ? lbl.slice(0,10) + '…' : lbl;
      axisG.appendChild(tick);
    });
  } else {
    const ticks = EG.niceTicks(scale.domain[0], scale.domain[1]);
    ticks.forEach(function(v) {
      const x = scale(v);
      axisG.appendChild(EG.svg('line', {x1: x, y1: 0, x2: x, y2: 5,
        stroke: 'var(--text-muted)', 'stroke-width': 1}));
      const tick = EG.svg('text', {x: x, y: 20, 'text-anchor': 'middle',
        fill: 'var(--text-secondary)', 'font-size': '11px'});
      tick.textContent = EG.fmt(v, v % 1 === 0 ? 0 : 2);
      axisG.appendChild(tick);
    });
  }

  if (label) {
    const w = scale.range ? (scale.range[1] - scale.range[0]) : 0;
    const lbl = EG.svg('text', {x: w / 2, y: 40, 'text-anchor': 'middle',
      fill: 'var(--text-secondary)', 'font-size': '12px', 'font-weight': '500'});
    lbl.textContent = label;
    axisG.appendChild(lbl);
  }

  return axisG;
};

// ---- Draw Y axis ----
EG.drawYAxis = function(g, scale, width, label, isBand) {
  const axisG = EG.svg('g');
  g.appendChild(axisG);

  axisG.appendChild(EG.svg('line', {x1: 0, y1: 0, x2: 0,
    y2: scale.range ? scale.range[0] : 0,
    stroke: 'var(--border)', 'stroke-width': 1}));

  if (isBand) {
    const labels = scale.labels || [];
    const bw = scale.bandwidth();
    labels.forEach(function(lbl) {
      const y = scale(lbl) + bw / 2;
      const tick = EG.svg('text', {x: -8, y: y + 4, 'text-anchor': 'end',
        fill: 'var(--text-secondary)', 'font-size': '11px'});
      tick.textContent = lbl.length > 15 ? lbl.slice(0,13) + '…' : lbl;
      axisG.appendChild(tick);
    });
  } else {
    const ticks = EG.niceTicks(scale.domain[0], scale.domain[1]);
    ticks.forEach(function(v) {
      const y = scale(v);
      // Grid line
      axisG.appendChild(EG.svg('line', {x1: 0, y1: y, x2: width, y2: y,
        stroke: 'var(--grid-line)', 'stroke-width': 1}));
      axisG.appendChild(EG.svg('line', {x1: -5, y1: y, x2: 0, y2: y,
        stroke: 'var(--text-muted)', 'stroke-width': 1}));
      const tick = EG.svg('text', {x: -8, y: y + 4, 'text-anchor': 'end',
        fill: 'var(--text-secondary)', 'font-size': '11px'});
      tick.textContent = EG.fmt(v, v % 1 === 0 ? 0 : 2);
      axisG.appendChild(tick);
    });
  }

  if (label) {
    const h = scale.range ? Math.abs(scale.range[0] - scale.range[1]) : 0;
    const lbl = EG.svg('text', {'text-anchor': 'middle',
      fill: 'var(--text-secondary)', 'font-size': '12px', 'font-weight': '500',
      transform: `translate(-45,${h/2}) rotate(-90)`});
    lbl.textContent = label;
    axisG.appendChild(lbl);
  }

  return axisG;
};

// ---- Draw legend ----
EG.drawLegend = function(container, items) {
  // items: [{label, color}]
  const div = document.createElement('div');
  div.className = 'legend';
  items.forEach(function(item) {
    const el = document.createElement('div');
    el.className = 'legend-item';
    const swatch = document.createElement('div');
    swatch.className = 'legend-swatch';
    swatch.style.background = item.color;
    el.appendChild(swatch);
    const text = document.createElement('span');
    text.textContent = item.label;
    el.appendChild(text);
    div.appendChild(el);
  });
  container.parentElement.appendChild(div);
};

// ---- Color interpolation ----
EG.lerpColor = function(c1, c2, t) {
  const r1 = parseInt(c1.slice(1,3), 16), g1 = parseInt(c1.slice(3,5), 16), b1 = parseInt(c1.slice(5,7), 16);
  const r2 = parseInt(c2.slice(1,3), 16), g2 = parseInt(c2.slice(3,5), 16), b2 = parseInt(c2.slice(5,7), 16);
  const r = Math.round(r1 + (r2-r1)*t), g = Math.round(g1 + (g2-g1)*t), b = Math.round(b1 + (b2-b1)*t);
  return '#' + ((1<<24)+(r<<16)+(g<<8)+b).toString(16).slice(1);
};

// ---- Map value to color from palette ----
EG.colorScale = function(palette, min, max) {
  return function(v) {
    if (palette.length === 1) return palette[0];
    const t = Math.max(0, Math.min(1, (v - min) / (max - min || 1)));
    const idx = t * (palette.length - 1);
    const lo = Math.floor(idx), hi = Math.min(lo + 1, palette.length - 1);
    return EG.lerpColor(palette[lo], palette[hi], idx - lo);
  };
};
"""
