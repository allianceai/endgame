"""Gauge (speedometer) chart visualizer.

Interactive gauge chart for single-metric dashboard display. Shows a
value on an arc with configurable zones (e.g., bad/ok/good), a needle,
and a digital readout.

Example
-------
>>> from endgame.visualization import GaugeChartVisualizer
>>> viz = GaugeChartVisualizer(value=0.92, label="Accuracy")
>>> viz.save("gauge.html")
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class GaugeChartVisualizer(BaseVisualizer):
    """Interactive gauge chart visualizer.

    Parameters
    ----------
    value : float
        Current metric value.
    min_value : float, default=0
        Minimum scale value.
    max_value : float, default=1
        Maximum scale value.
    label : str, default=''
        Metric label shown below the value.
    zones : list of (float, float, str), optional
        Colored zones as (start, end, color). If None, uses default
        red/yellow/green zones dividing the range into thirds.
    format_str : str, optional
        Python format string for the displayed value (e.g., '.1%', '.3f').
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette.
    width : int, default=450
        Chart width.
    height : int, default=350
        Chart height.
    theme : str, default='dark'
        'dark' or 'light'.
    """

    def __init__(
        self,
        value: float,
        *,
        min_value: float = 0,
        max_value: float = 1,
        label: str = "",
        zones: Sequence[tuple[float, float, str]] | None = None,
        format_str: str | None = None,
        title: str = "",
        palette: str = "tableau",
        width: int = 450,
        height: int = 350,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self.value = float(value)
        self.min_value = float(min_value)
        self.max_value = float(max_value)
        self.label = label
        self.format_str = format_str

        if zones is None:
            third = (max_value - min_value) / 3
            self.zones = [
                (min_value, min_value + third, "#d62728"),
                (min_value + third, min_value + 2 * third, "#ff7f0e"),
                (min_value + 2 * third, max_value, "#2ca02c"),
            ]
        else:
            self.zones = [(float(a), float(b), c) for a, b, c in zones]

    @classmethod
    def from_score(
        cls,
        score: float,
        metric_name: str = "Score",
        *,
        min_value: float = 0,
        max_value: float = 1,
        **kwargs,
    ) -> GaugeChartVisualizer:
        """Create from a single metric score.

        Parameters
        ----------
        score : float
            Metric value.
        metric_name : str, default='Score'
            Name of the metric.
        min_value : float, default=0
            Min scale.
        max_value : float, default=1
            Max scale.
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("label", metric_name)
        return cls(score, min_value=min_value, max_value=max_value, **kwargs)

    @classmethod
    def from_accuracy(cls, accuracy: float, **kwargs) -> GaugeChartVisualizer:
        """Create for an accuracy metric (0-1 scale).

        Parameters
        ----------
        accuracy : float
            Accuracy value (0 to 1).
        **kwargs
            Additional keyword arguments.
        """
        kwargs.setdefault("label", "Accuracy")
        kwargs.setdefault("format_str", ".1%")
        return cls(accuracy, min_value=0, max_value=1, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        # Format the display value
        if self.format_str:
            try:
                display_val = format(self.value, self.format_str)
            except (ValueError, TypeError):
                display_val = str(round(self.value, 4))
        else:
            display_val = str(round(self.value, 4))

        return {
            "value": self.value,
            "minValue": self.min_value,
            "maxValue": self.max_value,
            "label": self.label,
            "displayValue": display_val,
            "zones": [
                {"start": z[0], "end": z[1], "color": z[2]}
                for z in self.zones
            ],
        }

    def _chart_type(self) -> str:
        return "gauge"

    def _get_chart_js(self) -> str:
        return _GAUGE_JS


_GAUGE_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const W = config.width;
  const H = config.height;

  const svg = EG.svg('svg', {width: W, height: H});
  container.appendChild(svg);

  const cx = W / 2;
  const cy = H * 0.58;
  const outerR = Math.min(W, H) * 0.38;
  const innerR = outerR * 0.7;
  const needleR = outerR * 0.88;

  const vMin = data.minValue;
  const vMax = data.maxValue;
  const range = vMax - vMin || 1;

  // Gauge arc from -135° to +135° (270° sweep)
  const startAngle = -135 * Math.PI / 180;
  const endAngle = 135 * Math.PI / 180;
  const totalSweep = endAngle - startAngle;

  function valToAngle(v) {
    var t = (v - vMin) / range;
    t = Math.max(0, Math.min(1, t));
    return startAngle + t * totalSweep;
  }

  function polarX(angle, r) { return cx + r * Math.cos(angle); }
  function polarY(angle, r) { return cy + r * Math.sin(angle); }

  function arcPath(r, a1, a2) {
    const x1 = polarX(a1, r), y1 = polarY(a1, r);
    const x2 = polarX(a2, r), y2 = polarY(a2, r);
    const largeArc = Math.abs(a2 - a1) > Math.PI ? 1 : 0;
    return 'M' + x1 + ' ' + y1 + ' A' + r + ' ' + r + ' 0 ' + largeArc + ' 1 ' + x2 + ' ' + y2;
  }

  // Background arc (track)
  svg.appendChild(EG.svg('path', {
    d: arcPath(outerR, startAngle, endAngle),
    fill: 'none', stroke: 'var(--grid-line)', 'stroke-width': outerR - innerR, opacity: 0.3,
    'stroke-linecap': 'round'
  }));

  // Zone arcs
  data.zones.forEach(function(z) {
    const a1 = valToAngle(z.start);
    const a2 = valToAngle(z.end);
    const midR = (outerR + innerR) / 2;
    svg.appendChild(EG.svg('path', {
      d: arcPath(midR, a1, a2),
      fill: 'none', stroke: z.color, 'stroke-width': outerR - innerR - 4,
      'stroke-linecap': 'butt', opacity: 0.7
    }));
  });

  // Tick marks
  var nTicks = 10;
  for (var t = 0; t <= nTicks; t++) {
    var val = vMin + (t / nTicks) * range;
    var angle = valToAngle(val);
    var major = t % 2 === 0;
    var r1 = outerR + 4;
    var r2 = outerR + (major ? 14 : 9);
    svg.appendChild(EG.svg('line', {
      x1: polarX(angle, r1), y1: polarY(angle, r1),
      x2: polarX(angle, r2), y2: polarY(angle, r2),
      stroke: 'var(--text-muted)', 'stroke-width': major ? 2 : 1
    }));
    if (major) {
      svg.appendChild(EG.svg('text', {
        x: polarX(angle, outerR + 24), y: polarY(angle, outerR + 24) + 3,
        'text-anchor': 'middle', fill: 'var(--text-secondary)', 'font-size': '10px'
      })).textContent = EG.fmt(val, range >= 10 ? 0 : 2);
    }
  }

  // Needle
  var needleAngle = valToAngle(data.value);
  var nx = polarX(needleAngle, needleR);
  var ny = polarY(needleAngle, needleR);

  // Needle body (tapered)
  var perpAngle = needleAngle + Math.PI / 2;
  var baseW = 4;
  var bx1 = cx + baseW * Math.cos(perpAngle);
  var by1 = cy + baseW * Math.sin(perpAngle);
  var bx2 = cx - baseW * Math.cos(perpAngle);
  var by2 = cy - baseW * Math.sin(perpAngle);

  svg.appendChild(EG.svg('path', {
    d: 'M' + bx1 + ' ' + by1 + ' L' + nx + ' ' + ny + ' L' + bx2 + ' ' + by2 + ' Z',
    fill: 'var(--text-primary)', opacity: 0.9
  }));

  // Center dot
  svg.appendChild(EG.svg('circle', {cx: cx, cy: cy, r: 8, fill: 'var(--text-primary)'}));
  svg.appendChild(EG.svg('circle', {cx: cx, cy: cy, r: 4, fill: 'var(--bg-card)'}));

  // Digital readout
  svg.appendChild(EG.svg('text', {
    x: cx, y: cy + outerR * 0.45,
    'text-anchor': 'middle', fill: 'var(--text-primary)',
    'font-size': '28px', 'font-weight': '700',
    'font-family': '"SF Mono", "Fira Code", monospace'
  })).textContent = data.displayValue;

  // Label
  if (data.label) {
    svg.appendChild(EG.svg('text', {
      x: cx, y: cy + outerR * 0.45 + 24,
      'text-anchor': 'middle', fill: 'var(--text-muted)',
      'font-size': '13px', 'font-weight': '500'
    })).textContent = data.label;
  }
}
"""
