"""Defuzzification methods for converting fuzzy output to crisp values.

Supported methods:
- Centroid (center of gravity)
- Bisector (bisector of area)
- Mean of Maxima (MOM)
- Weighted Average (for Sugeno/TSK systems)
- Height Method (simplified weighted average)

Example
-------
>>> from endgame.fuzzy.core.defuzzification import centroid, defuzzify
>>> x = np.linspace(0, 10, 100)
>>> mf_values = np.exp(-0.5 * ((x - 5) / 1.5) ** 2)
>>> centroid(x, mf_values)  # ~5.0
>>> defuzzify(x, mf_values, method='centroid')  # ~5.0
"""

from __future__ import annotations

import numpy as np


def centroid(x: np.ndarray, mf_values: np.ndarray) -> float:
    """Centroid (center of gravity) defuzzification.

    Parameters
    ----------
    x : ndarray of shape (n_points,)
        Universe of discourse (x-axis values).
    mf_values : ndarray of shape (n_points,)
        Aggregated membership function values.

    Returns
    -------
    float
        Crisp output value.
    """
    x = np.asarray(x, dtype=np.float64)
    mf_values = np.asarray(mf_values, dtype=np.float64)
    total = np.sum(mf_values)
    if total == 0:
        return float(np.mean(x))
    return float(np.sum(x * mf_values) / total)


def bisector(x: np.ndarray, mf_values: np.ndarray) -> float:
    """Bisector defuzzification.

    Finds the x value that divides the area under the MF into two equal halves.

    Parameters
    ----------
    x : ndarray of shape (n_points,)
        Universe of discourse.
    mf_values : ndarray of shape (n_points,)
        Aggregated membership function values.

    Returns
    -------
    float
        Crisp output value.
    """
    x = np.asarray(x, dtype=np.float64)
    mf_values = np.asarray(mf_values, dtype=np.float64)
    total_area = np.trapezoid(mf_values, x) if hasattr(np, 'trapezoid') else np.trapz(mf_values, x)
    if total_area == 0:
        return float(np.mean(x))

    half_area = total_area / 2.0
    cumulative = np.cumsum(mf_values[:-1] * np.diff(x))
    idx = np.searchsorted(cumulative, half_area)
    idx = min(idx, len(x) - 1)
    return float(x[idx])


def mean_of_maxima(x: np.ndarray, mf_values: np.ndarray) -> float:
    """Mean of Maxima (MOM) defuzzification.

    Returns the mean of all x values where the MF reaches its maximum.

    Parameters
    ----------
    x : ndarray of shape (n_points,)
        Universe of discourse.
    mf_values : ndarray of shape (n_points,)
        Aggregated membership function values.

    Returns
    -------
    float
        Crisp output value.
    """
    x = np.asarray(x, dtype=np.float64)
    mf_values = np.asarray(mf_values, dtype=np.float64)
    max_val = np.max(mf_values)
    if max_val == 0:
        return float(np.mean(x))
    mask = np.isclose(mf_values, max_val, rtol=1e-10)
    return float(np.mean(x[mask]))


def weighted_average(centers: np.ndarray, heights: np.ndarray) -> float:
    """Weighted average defuzzification (for TSK/Sugeno systems).

    Parameters
    ----------
    centers : ndarray of shape (n_rules,)
        Consequent values (rule outputs) for each rule.
    heights : ndarray of shape (n_rules,)
        Firing strengths (weights) for each rule.

    Returns
    -------
    float
        Crisp output value.
    """
    centers = np.asarray(centers, dtype=np.float64)
    heights = np.asarray(heights, dtype=np.float64)
    total = np.sum(heights)
    if total == 0:
        return float(np.mean(centers))
    return float(np.sum(centers * heights) / total)


def height_method(centers: np.ndarray, heights: np.ndarray) -> float:
    """Height method defuzzification.

    Simplified weighted average using peak heights only.
    Equivalent to weighted_average but emphasizes it uses
    membership function peaks rather than areas.

    Parameters
    ----------
    centers : ndarray of shape (n_rules,)
        Centers (peaks) of output membership functions.
    heights : ndarray of shape (n_rules,)
        Clipped heights of output membership functions.

    Returns
    -------
    float
        Crisp output value.
    """
    return weighted_average(centers, heights)


_METHOD_MAP = {
    "centroid": centroid,
    "bisector": bisector,
    "mom": mean_of_maxima,
    "mean_of_maxima": mean_of_maxima,
}

_DISCRETE_METHOD_MAP = {
    "weighted_average": weighted_average,
    "height": height_method,
}


def defuzzify(
    x: np.ndarray,
    mf_values: np.ndarray,
    method: str = "centroid",
) -> float:
    """Defuzzify using the specified method.

    Parameters
    ----------
    x : ndarray
        Universe of discourse (continuous methods) or rule centers (discrete).
    mf_values : ndarray
        Membership values (continuous) or firing strengths (discrete).
    method : str, default='centroid'
        One of 'centroid', 'bisector', 'mean_of_maxima'/'mom',
        'weighted_average', 'height'.

    Returns
    -------
    float
        Crisp output value.
    """
    if method in _METHOD_MAP:
        return _METHOD_MAP[method](x, mf_values)
    if method in _DISCRETE_METHOD_MAP:
        return _DISCRETE_METHOD_MAP[method](x, mf_values)
    raise ValueError(
        f"Unknown defuzzification method: {method}. "
        f"Choose from {list(_METHOD_MAP) + list(_DISCRETE_METHOD_MAP)}"
    )
