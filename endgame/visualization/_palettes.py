"""Color palettes for endgame visualizations.

Provides categorical, sequential, and diverging palettes for all chart types.
Extracted from tree_visualizer.py and extended with heatmap/correlation palettes.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Categorical palettes (for discrete categories)
# ---------------------------------------------------------------------------

CATEGORICAL: dict[str, list[str]] = {
    "tableau": [
        "#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f",
        "#edc948", "#b07aa1", "#ff9da7", "#9c755f", "#bab0ac",
    ],
    "viridis": [
        "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#b5de2b", "#fde725",
    ],
    "pastel": [
        "#8dd3c7", "#ffffb3", "#bebada", "#fb8072", "#80b1d3",
        "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd",
    ],
    "dark": [
        "#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e",
        "#e6ab02", "#a6761d", "#666666", "#e41a1c", "#377eb8",
    ],
    "bold": [
        "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
        "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    ],
}

# ---------------------------------------------------------------------------
# Sequential palettes (for continuous values, low → high)
# ---------------------------------------------------------------------------

SEQUENTIAL: dict[str, list[str]] = {
    "blues": [
        "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6",
        "#4292c6", "#2171b5", "#08519c", "#08306b",
    ],
    "reds": [
        "#fff5f0", "#fee0d2", "#fcbba1", "#fc9272", "#fb6a4a",
        "#ef3b2c", "#cb181d", "#a50f15", "#67000d",
    ],
    "greens": [
        "#f7fcf5", "#e5f5e0", "#c7e9c0", "#a1d99b", "#74c476",
        "#41ab5d", "#238b45", "#006d2c", "#00441b",
    ],
    "oranges": [
        "#fff5eb", "#fee6ce", "#fdd0a2", "#fdae6b", "#fd8d3c",
        "#f16913", "#d94801", "#a63603", "#7f2704",
    ],
    "purples": [
        "#fcfbfd", "#efedf5", "#dadaeb", "#bcbddc", "#9e9ac8",
        "#807dba", "#6a51a3", "#54278f", "#3f007d",
    ],
    "viridis_seq": [
        "#440154", "#482777", "#3e4989", "#31688e", "#26828e",
        "#1f9e89", "#35b779", "#6ece58", "#fde725",
    ],
    "inferno": [
        "#000004", "#1b0c41", "#4a0c6b", "#781c6d", "#a52c60",
        "#cf4446", "#ed6925", "#fb9b06", "#fcffa4",
    ],
    "plasma": [
        "#0d0887", "#46039f", "#7201a8", "#9c179e", "#bd3786",
        "#d8576b", "#ed7953", "#fb9f3a", "#fdca26",
    ],
}

# ---------------------------------------------------------------------------
# Diverging palettes (for values around a midpoint, e.g., correlation)
# ---------------------------------------------------------------------------

DIVERGING: dict[str, list[str]] = {
    "rdbu": [
        "#67001f", "#b2182b", "#d6604d", "#f4a582", "#fddbc7",
        "#f7f7f7", "#d1e5f0", "#92c5de", "#4393c3", "#2166ac", "#053061",
    ],
    "spectral": [
        "#9e0142", "#d53e4f", "#f46d43", "#fdae61", "#fee08b",
        "#ffffbf", "#e6f598", "#abdda4", "#66c2a5", "#3288bd", "#5e4fa2",
    ],
    "rdylgn": [
        "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b",
        "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837",
    ],
    "piyg": [
        "#8e0152", "#c51b7d", "#de77ae", "#f1b6da", "#fde0ef",
        "#f7f7f7", "#e6f5d0", "#b8e186", "#7fbc41", "#4d9221", "#276419",
    ],
    "brbg": [
        "#543005", "#8c510a", "#bf812d", "#dfc27d", "#f6e8c3",
        "#f5f5f5", "#c7eae5", "#80cdc1", "#35978f", "#01665e", "#003c30",
    ],
}

# Default palette for each type
DEFAULT_CATEGORICAL = "tableau"
DEFAULT_SEQUENTIAL = "blues"
DEFAULT_DIVERGING = "rdbu"


def get_palette(
    name: str,
    n: int | None = None,
) -> list[str]:
    """Get a palette by name.

    Searches categorical, sequential, and diverging palettes.

    Parameters
    ----------
    name : str
        Palette name (e.g., 'tableau', 'blues', 'rdbu').
    n : int, optional
        Number of colors to return. If None, returns the full palette.
        If n > palette length, colors cycle.

    Returns
    -------
    list of str
        List of hex color strings.
    """
    for collection in (CATEGORICAL, SEQUENTIAL, DIVERGING):
        if name in collection:
            colors = collection[name]
            if n is None:
                return list(colors)
            if n <= len(colors):
                # Subsample evenly
                if n == 1:
                    return [colors[len(colors) // 2]]
                step = (len(colors) - 1) / (n - 1)
                return [colors[round(i * step)] for i in range(n)]
            # Cycle if more needed
            return [colors[i % len(colors)] for i in range(n)]
    raise ValueError(
        f"Unknown palette '{name}'. Available: "
        f"{sorted(list(CATEGORICAL) + list(SEQUENTIAL) + list(DIVERGING))}"
    )


def interpolate_color(color1: str, color2: str, t: float) -> str:
    """Linearly interpolate between two hex colors.

    Parameters
    ----------
    color1, color2 : str
        Hex color strings (e.g., '#ff0000').
    t : float
        Interpolation parameter in [0, 1].

    Returns
    -------
    str
        Interpolated hex color.
    """
    r1, g1, b1 = _hex_to_rgb(color1)
    r2, g2, b2 = _hex_to_rgb(color2)
    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


def make_color_scale(
    palette_name: str,
    n_steps: int = 256,
) -> list[str]:
    """Generate a smooth color scale with n_steps colors.

    Parameters
    ----------
    palette_name : str
        Name of a sequential or diverging palette.
    n_steps : int
        Number of color steps.

    Returns
    -------
    list of str
        List of hex color strings.
    """
    base = get_palette(palette_name)
    if n_steps <= len(base):
        return get_palette(palette_name, n_steps)

    result = []
    for i in range(n_steps):
        t = i / (n_steps - 1) * (len(base) - 1)
        idx = int(t)
        frac = t - idx
        if idx >= len(base) - 1:
            result.append(base[-1])
        else:
            result.append(interpolate_color(base[idx], base[idx + 1], frac))
    return result


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    h = hex_color.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def _rgb_to_hex(r: int, g: int, b: int) -> str:
    """Convert RGB tuple to hex color."""
    return f"#{r:02x}{g:02x}{b:02x}"
