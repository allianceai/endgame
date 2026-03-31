"""T-norms (fuzzy AND) and T-conorms (fuzzy OR) for fuzzy logic operations.

T-norms satisfy: commutativity, associativity, monotonicity, boundary (T(a,1)=a).
T-conorms are dual: S(a,b) = 1 - T(1-a, 1-b).

Example
-------
>>> from endgame.fuzzy.core.operators import t_norm, t_conorm
>>> t_norm(0.7, 0.5, method='min')        # min(0.7, 0.5) = 0.5
>>> t_norm(0.7, 0.5, method='product')     # 0.7 * 0.5 = 0.35
>>> t_conorm(0.7, 0.5, method='max')       # max(0.7, 0.5) = 0.7
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


# --- T-Norm Implementations ---


class BaseTNorm(ABC):
    """Abstract base for t-norm operators."""

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply t-norm element-wise."""

    def reduce(self, values: np.ndarray, axis: int = -1) -> np.ndarray:
        """Reduce along an axis by applying t-norm sequentially.

        Parameters
        ----------
        values : ndarray
            Array of membership values.
        axis : int, default=-1
            Axis to reduce along.

        Returns
        -------
        ndarray
            Reduced result.
        """
        values = np.asarray(values)
        slices = np.moveaxis(values, axis, 0)
        result = slices[0].copy()
        for i in range(1, len(slices)):
            result = self(result, slices[i])
        return result


class MinTNorm(BaseTNorm):
    """Minimum (Zadeh) t-norm: T(a, b) = min(a, b)."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(a, b)


class ProductTNorm(BaseTNorm):
    """Algebraic product t-norm: T(a, b) = a * b."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a) * np.asarray(b)


class LukasiewiczTNorm(BaseTNorm):
    """Lukasiewicz t-norm: T(a, b) = max(a + b - 1, 0)."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(np.asarray(a) + np.asarray(b) - 1.0, 0.0)


class HamacherTNorm(BaseTNorm):
    """Hamacher t-norm: T(a, b) = (a * b) / (p + (1-p)(a + b - a*b)).

    Parameters
    ----------
    p : float, default=0.0
        Hamacher parameter (p >= 0). p=0 gives Hamacher product,
        p=1 gives algebraic product, p->inf approaches drastic product.
    """

    def __init__(self, p: float = 0.0):
        if p < 0:
            raise ValueError(f"p must be >= 0, got {p}")
        self.p = float(p)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ab = a * b
        denom = self.p + (1.0 - self.p) * (a + b - ab)
        # Handle edge case where both a and b are 0
        result = np.where(denom > 0, ab / denom, 0.0)
        return result


# --- T-Conorm Implementations ---


class BaseTConorm(ABC):
    """Abstract base for t-conorm (s-norm) operators."""

    @abstractmethod
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Apply t-conorm element-wise."""

    def reduce(self, values: np.ndarray, axis: int = -1) -> np.ndarray:
        """Reduce along an axis by applying t-conorm sequentially."""
        values = np.asarray(values)
        slices = np.moveaxis(values, axis, 0)
        result = slices[0].copy()
        for i in range(1, len(slices)):
            result = self(result, slices[i])
        return result


class MaxTConorm(BaseTConorm):
    """Maximum (Zadeh) t-conorm: S(a, b) = max(a, b)."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.maximum(a, b)


class ProbabilisticSumTConorm(BaseTConorm):
    """Probabilistic sum t-conorm: S(a, b) = a + b - a*b."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a), np.asarray(b)
        return a + b - a * b


class LukasiewiczTConorm(BaseTConorm):
    """Lukasiewicz t-conorm: S(a, b) = min(a + b, 1)."""

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.minimum(np.asarray(a) + np.asarray(b), 1.0)


class HamacherTConorm(BaseTConorm):
    """Hamacher t-conorm: S(a, b) = (a + b - (2-p)*a*b) / (1 - (1-p)*a*b).

    Parameters
    ----------
    p : float, default=0.0
        Hamacher parameter (p >= 0).
    """

    def __init__(self, p: float = 0.0):
        if p < 0:
            raise ValueError(f"p must be >= 0, got {p}")
        self.p = float(p)

    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
        ab = a * b
        numer = a + b - (2.0 - self.p) * ab
        denom = 1.0 - (1.0 - self.p) * ab
        result = np.where(denom > 0, numer / denom, 1.0)
        return np.clip(result, 0.0, 1.0)


# --- Convenience Functions ---

_TNORM_MAP: dict[str, type[BaseTNorm]] = {
    "min": MinTNorm,
    "product": ProductTNorm,
    "lukasiewicz": LukasiewiczTNorm,
    "hamacher": HamacherTNorm,
}

_TCONORM_MAP: dict[str, type[BaseTConorm]] = {
    "max": MaxTConorm,
    "probabilistic_sum": ProbabilisticSumTConorm,
    "lukasiewicz": LukasiewiczTConorm,
    "hamacher": HamacherTConorm,
}


def t_norm(
    a: np.ndarray | float,
    b: np.ndarray | float,
    method: str = "min",
) -> np.ndarray:
    """Apply a t-norm (fuzzy AND) operation.

    Parameters
    ----------
    a, b : array-like or float
        Membership values in [0, 1].
    method : str, default='min'
        One of 'min', 'product', 'lukasiewicz', 'hamacher'.

    Returns
    -------
    ndarray
        T-norm result.
    """
    if method not in _TNORM_MAP:
        raise ValueError(f"Unknown t-norm: {method}. Choose from {list(_TNORM_MAP)}")
    op = _TNORM_MAP[method]()
    return op(np.asarray(a), np.asarray(b))


def t_conorm(
    a: np.ndarray | float,
    b: np.ndarray | float,
    method: str = "max",
) -> np.ndarray:
    """Apply a t-conorm (fuzzy OR) operation.

    Parameters
    ----------
    a, b : array-like or float
        Membership values in [0, 1].
    method : str, default='max'
        One of 'max', 'probabilistic_sum', 'lukasiewicz', 'hamacher'.

    Returns
    -------
    ndarray
        T-conorm result.
    """
    if method not in _TCONORM_MAP:
        raise ValueError(
            f"Unknown t-conorm: {method}. Choose from {list(_TCONORM_MAP)}"
        )
    op = _TCONORM_MAP[method]()
    return op(np.asarray(a), np.asarray(b))
