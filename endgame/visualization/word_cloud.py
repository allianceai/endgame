"""Word cloud visualizer.

Interactive word cloud for text data visualization, feature name
exploration, and keyword frequency display.

Example
-------
>>> from endgame.visualization import WordCloudVisualizer
>>> words = {"accuracy": 50, "precision": 35, "recall": 40, "f1": 45}
>>> viz = WordCloudVisualizer(words, title="Metrics")
>>> viz.save("word_cloud.html")
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from endgame.visualization._base import BaseVisualizer


class WordCloudVisualizer(BaseVisualizer):
    """Interactive word cloud visualizer.

    Parameters
    ----------
    words : dict of str → float
        Word/phrase → weight/frequency.
    max_words : int, default=100
        Maximum number of words to display.
    min_font : int, default=12
        Minimum font size in pixels.
    max_font : int, default=60
        Maximum font size in pixels.
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
        words: dict[str, float],
        *,
        max_words: int = 100,
        min_font: int = 12,
        max_font: int = 60,
        title: str = "",
        palette: str = "tableau",
        width: int = 800,
        height: int = 500,
        theme: str = "dark",
    ):
        super().__init__(title=title, palette=palette, width=width, height=height, theme=theme)
        self._words = dict(words)
        self.max_words = max_words
        self.min_font = min_font
        self.max_font = max_font

    @classmethod
    def from_feature_names(
        cls,
        feature_names: Sequence[str],
        importances: Sequence[float] | None = None,
        **kwargs,
    ) -> WordCloudVisualizer:
        """Create from feature names and optional importances.

        Parameters
        ----------
        feature_names : list of str
            Feature names.
        importances : list of float, optional
            Feature importances.
        **kwargs
            Additional keyword arguments.
        """
        if importances is not None:
            words = dict(zip(feature_names, importances))
        else:
            words = {name: 1.0 for name in feature_names}
        kwargs.setdefault("title", "Feature Names")
        return cls(words, **kwargs)

    def _build_data(self) -> dict[str, Any]:
        # Sort by weight, take top N
        sorted_words = sorted(self._words.items(), key=lambda x: x[1], reverse=True)
        sorted_words = sorted_words[:self.max_words]

        if not sorted_words:
            return {"words": []}

        max_w = sorted_words[0][1]
        min_w = sorted_words[-1][1] if len(sorted_words) > 1 else max_w

        words = []
        rng = random.Random(42)
        for word, weight in sorted_words:
            if max_w > min_w:
                t = (weight - min_w) / (max_w - min_w)
            else:
                t = 0.5
            font = self.min_font + t * (self.max_font - self.min_font)
            words.append({
                "text": word,
                "weight": round(float(weight), 4),
                "fontSize": round(font, 1),
                "rotation": rng.choice([0, 0, 0, -90]) if rng.random() > 0.7 else 0,
            })

        return {"words": words}

    def _chart_type(self) -> str:
        return "word_cloud"

    def _get_chart_js(self) -> str:
        return _WORDCLOUD_JS


_WORDCLOUD_JS = r"""
function renderChart(data, config) {
  const container = document.getElementById('chart-container');
  const W = config.width, H = config.height;
  const svg = EG.svg('svg', {width: W, height: H});
  container.appendChild(svg);
  const palette = config.palette;
  const words = data.words;

  // Simple spiral placement
  const placed = [];
  const cx = W / 2, cy = H / 2;

  words.forEach(function(w, i) {
    const color = palette[i % palette.length];
    let bestX = cx, bestY = cy;
    let found = false;

    // Spiral outward to find non-overlapping position
    for (let t = 0; t < 500 && !found; t++) {
      const angle = t * 0.15;
      const radius = t * 1.2;
      const tx = cx + radius * Math.cos(angle);
      const ty = cy + radius * Math.sin(angle);

      // Rough bounding box
      const estW = w.text.length * w.fontSize * 0.55;
      const estH = w.fontSize * 1.2;

      if (tx - estW / 2 < 10 || tx + estW / 2 > W - 10 ||
          ty - estH / 2 < 10 || ty + estH / 2 > H - 10) continue;

      let overlap = false;
      for (let j = 0; j < placed.length; j++) {
        const p = placed[j];
        if (Math.abs(tx - p.x) < (estW + p.w) / 2 &&
            Math.abs(ty - p.y) < (estH + p.h) / 2) {
          overlap = true;
          break;
        }
      }

      if (!overlap) {
        bestX = tx;
        bestY = ty;
        placed.push({x: tx, y: ty, w: estW, h: estH});
        found = true;
      }
    }

    const text = EG.svg('text', {
      x: bestX, y: bestY,
      'text-anchor': 'middle', 'dominant-baseline': 'middle',
      fill: color, 'font-size': w.fontSize + 'px',
      'font-weight': w.fontSize > 30 ? '700' : '500',
      transform: w.rotation ? `rotate(${w.rotation},${bestX},${bestY})` : '',
      opacity: 0.9
    });
    text.textContent = w.text;
    text.addEventListener('mouseenter', function(e) {
      text.setAttribute('opacity', '1');
      text.setAttribute('font-size', (w.fontSize * 1.1) + 'px');
      EG.tooltip.show(e, '<b>' + EG.esc(w.text) + '</b><br>Weight: ' + EG.fmt(w.weight));
    });
    text.addEventListener('mouseleave', function() {
      text.setAttribute('opacity', '0.9');
      text.setAttribute('font-size', w.fontSize + 'px');
      EG.tooltip.hide();
    });
    svg.appendChild(text);
  });
}
"""
