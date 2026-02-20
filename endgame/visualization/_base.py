"""Base visualizer ABC for all endgame chart types.

All chart visualizers inherit from BaseVisualizer, which provides:
- save(filepath) → Path: write self-contained HTML
- to_json() → str: JSON data export
- _repr_html_() → str: Jupyter inline display
- Common parameters: title, palette, width, height, theme
"""

from __future__ import annotations

import html as html_module
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from endgame.visualization._html_template import render_html
from endgame.visualization._palettes import DEFAULT_CATEGORICAL, get_palette


class BaseVisualizer(ABC):
    """Abstract base class for all endgame visualizers.

    Parameters
    ----------
    title : str, optional
        Chart title.
    palette : str, default='tableau'
        Color palette name (see ``_palettes.py``).
    width : int, default=900
        Chart width in pixels.
    height : int, default=500
        Chart height in pixels.
    theme : str, default='dark'
        Color theme ('dark' or 'light').
    """

    def __init__(
        self,
        *,
        title: str = "",
        palette: str = DEFAULT_CATEGORICAL,
        width: int = 900,
        height: int = 500,
        theme: str = "dark",
    ):
        self.title = title
        self.palette = palette
        self.width = width
        self.height = height
        if theme not in ("dark", "light"):
            raise ValueError(f"theme must be 'dark' or 'light', got '{theme}'")
        self.theme = theme

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def _build_data(self) -> dict[str, Any]:
        """Build the JSON-serializable data dict for this chart.

        Returns
        -------
        dict
            Chart data that will be serialized and passed to the JS renderer.
        """

    @abstractmethod
    def _chart_type(self) -> str:
        """Return the chart type identifier (e.g., 'bar', 'heatmap').

        This is used by the HTML template to dispatch to the correct JS renderer.
        """

    @abstractmethod
    def _get_chart_js(self) -> str:
        """Return the chart-specific JavaScript rendering code.

        This JS will be injected into the HTML template. It should define a
        ``renderChart(data, config)`` function that draws the chart into the
        ``#chart-container`` element.
        """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def to_json(self) -> str:
        """Export chart data as a JSON string.

        Returns
        -------
        str
            JSON representation of the chart data.
        """
        return json.dumps(self._build_data(), indent=2)

    def save(self, filepath: str | Path, open_browser: bool = False) -> Path:
        """Save the visualization as a self-contained HTML file.

        Parameters
        ----------
        filepath : str or Path
            Output file path (should end in .html).
        open_browser : bool, default=False
            If True, open the file in the default web browser.

        Returns
        -------
        Path
            The absolute path to the saved file.
        """
        filepath = Path(filepath)
        if not filepath.suffix:
            filepath = filepath.with_suffix(".html")

        html_content = self._render_html(embedded=False)
        filepath.write_text(html_content, encoding="utf-8")

        if open_browser:
            import webbrowser
            webbrowser.open(filepath.resolve().as_uri())

        return filepath.resolve()

    def _repr_html_(self) -> str:
        """Jupyter notebook inline display."""
        return self._render_html(embedded=True)

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_html(self, embedded: bool = False) -> str:
        """Build the full HTML page for this chart."""
        title = html_module.escape(self.title) if self.title else self._chart_type().replace("_", " ").title()
        data = self._build_data()
        colors = get_palette(self.palette)

        config = {
            "width": self.width,
            "height": self.height,
            "theme": self.theme,
            "palette": colors,
            "title": title,
        }

        return render_html(
            chart_type=self._chart_type(),
            data_json=json.dumps(data),
            config_json=json.dumps(config),
            chart_js=self._get_chart_js(),
            title=title,
            theme=self.theme,
            width=self.width,
            height=self.height,
            embedded=embedded,
        )
