"""Fuzzy membership function library.

All membership functions are vectorized (NumPy) and differentiable.
Optional PyTorch tensor support for gradient-based optimization.

Supported shapes:
- Triangular, Trapezoidal, Gaussian, Generalized Bell
- Sigmoidal, Difference of Sigmoidals, Pi-shaped
- Interval Type-2 Gaussian and Triangular (for Type-2 fuzzy systems)

Example
-------
>>> from endgame.fuzzy.core.membership import GaussianMF, TriangularMF
>>> mf = GaussianMF(center=0.0, sigma=1.0)
>>> mf(np.array([0.0, 1.0, 2.0]))
array([1.  , 0.607, 0.135])
>>> tri = TriangularMF(a=-1.0, b=0.0, c=1.0)
>>> tri(np.array([-1.0, 0.0, 1.0]))
array([0., 1., 0.])
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class BaseMembershipFunction(ABC):
    """Abstract base class for fuzzy membership functions.

    All membership functions map inputs to [0, 1] and expose
    their parameters for optimization.
    """

    @abstractmethod
    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Compute membership degrees.

        Parameters
        ----------
        x : ndarray of shape (n_samples,) or (n_samples, 1)
            Input values.

        Returns
        -------
        ndarray of shape (n_samples,)
            Membership degrees in [0, 1].
        """

    @property
    @abstractmethod
    def parameters(self) -> dict[str, float]:
        """Return membership function parameters as a dictionary."""

    def to_torch(self) -> Any:
        """Convert to a differentiable PyTorch module.

        Returns
        -------
        torch.nn.Module
            PyTorch module with learnable parameters.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for differentiable MFs.")
        return _TorchMFWrapper(self)

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v:.4g}" for k, v in self.parameters.items())
        return f"{self.__class__.__name__}({params})"


class TriangularMF(BaseMembershipFunction):
    """Triangular membership function.

    Parameters
    ----------
    a : float
        Left foot (membership = 0).
    b : float
        Peak (membership = 1).
    c : float
        Right foot (membership = 0).
    """

    def __init__(self, a: float, b: float, c: float):
        if not a <= b <= c:
            raise ValueError(f"Must have a <= b <= c, got a={a}, b={b}, c={c}")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        result = np.zeros_like(x)
        if self.b > self.a:
            mask = (x >= self.a) & (x <= self.b)
            result[mask] = (x[mask] - self.a) / (self.b - self.a)
        if self.c > self.b:
            mask = (x > self.b) & (x <= self.c)
            result[mask] = (self.c - x[mask]) / (self.c - self.b)
        result[x == self.b] = 1.0
        return result

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}


class TrapezoidalMF(BaseMembershipFunction):
    """Trapezoidal membership function.

    Parameters
    ----------
    a : float
        Left foot (membership starts rising from 0).
    b : float
        Left shoulder (membership reaches 1).
    c : float
        Right shoulder (membership starts falling from 1).
    d : float
        Right foot (membership reaches 0).
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        if not a <= b <= c <= d:
            raise ValueError(
                f"Must have a <= b <= c <= d, got a={a}, b={b}, c={c}, d={d}"
            )
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        result = np.zeros_like(x)
        # Rising edge
        if self.b > self.a:
            mask = (x >= self.a) & (x < self.b)
            result[mask] = (x[mask] - self.a) / (self.b - self.a)
        # Plateau
        mask = (x >= self.b) & (x <= self.c)
        result[mask] = 1.0
        # Falling edge
        if self.d > self.c:
            mask = (x > self.c) & (x <= self.d)
            result[mask] = (self.d - x[mask]) / (self.d - self.c)
        return result

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}


class GaussianMF(BaseMembershipFunction):
    """Gaussian membership function.

    Parameters
    ----------
    center : float
        Center (mean) of the Gaussian.
    sigma : float
        Standard deviation (width).
    """

    def __init__(self, center: float, sigma: float):
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        self.center = float(center)
        self.sigma = float(sigma)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.exp(-0.5 * ((x - self.center) / self.sigma) ** 2)

    @property
    def parameters(self) -> dict[str, float]:
        return {"center": self.center, "sigma": self.sigma}


class GeneralizedBellMF(BaseMembershipFunction):
    """Generalized bell-shaped membership function.

    f(x) = 1 / (1 + |((x - c) / a)|^(2b))

    Parameters
    ----------
    a : float
        Width parameter (must be positive).
    b : float
        Slope parameter (must be positive).
    c : float
        Center.
    """

    def __init__(self, a: float, b: float, c: float):
        if a <= 0:
            raise ValueError(f"a must be positive, got {a}")
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.abs((x - self.c) / self.a) ** (2 * self.b))

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c}


class SigmoidalMF(BaseMembershipFunction):
    """Sigmoidal membership function.

    f(x) = 1 / (1 + exp(-a * (x - c)))

    Parameters
    ----------
    a : float
        Slope (positive = rising, negative = falling).
    c : float
        Crossover point (where membership = 0.5).
    """

    def __init__(self, a: float, c: float):
        self.a = float(a)
        self.c = float(c)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        return 1.0 / (1.0 + np.exp(-self.a * (x - self.c)))

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "c": self.c}


class DifferenceSigmoidalMF(BaseMembershipFunction):
    """Difference of two sigmoidal membership functions.

    f(x) = sig(a1, c1)(x) - sig(a2, c2)(x)

    Creates a bell or band-shaped function from two sigmoids.

    Parameters
    ----------
    a1 : float
        Slope of first sigmoid (should be positive).
    c1 : float
        Crossover of first sigmoid.
    a2 : float
        Slope of second sigmoid (should be positive).
    c2 : float
        Crossover of second sigmoid (c2 > c1 for band shape).
    """

    def __init__(self, a1: float, c1: float, a2: float, c2: float):
        self.a1 = float(a1)
        self.c1 = float(c1)
        self.a2 = float(a2)
        self.c2 = float(c2)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        s1 = 1.0 / (1.0 + np.exp(-self.a1 * (x - self.c1)))
        s2 = 1.0 / (1.0 + np.exp(-self.a2 * (x - self.c2)))
        return np.clip(s1 - s2, 0.0, 1.0)

    @property
    def parameters(self) -> dict[str, float]:
        return {"a1": self.a1, "c1": self.c1, "a2": self.a2, "c2": self.c2}


class PiMF(BaseMembershipFunction):
    """Pi-shaped membership function.

    Combines an S-shaped (rising) and Z-shaped (falling) curve.

    Parameters
    ----------
    a : float
        Left foot.
    b : float
        Left peak.
    c : float
        Right peak.
    d : float
        Right foot.
    """

    def __init__(self, a: float, b: float, c: float, d: float):
        if not a <= b <= c <= d:
            raise ValueError(
                f"Must have a <= b <= c <= d, got a={a}, b={b}, c={c}, d={d}"
            )
        self.a = float(a)
        self.b = float(b)
        self.c = float(c)
        self.d = float(d)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).ravel()
        result = np.zeros_like(x)

        # S-curve (rising part)
        mid_ab = (self.a + self.b) / 2.0
        if self.b > self.a:
            mask1 = (x >= self.a) & (x < mid_ab)
            result[mask1] = 2.0 * ((x[mask1] - self.a) / (self.b - self.a)) ** 2
            mask2 = (x >= mid_ab) & (x < self.b)
            result[mask2] = 1.0 - 2.0 * ((x[mask2] - self.b) / (self.b - self.a)) ** 2

        # Plateau
        mask = (x >= self.b) & (x <= self.c)
        result[mask] = 1.0

        # Z-curve (falling part)
        mid_cd = (self.c + self.d) / 2.0
        if self.d > self.c:
            mask3 = (x > self.c) & (x <= mid_cd)
            result[mask3] = 1.0 - 2.0 * ((x[mask3] - self.c) / (self.d - self.c)) ** 2
            mask4 = (x > mid_cd) & (x <= self.d)
            result[mask4] = 2.0 * ((x[mask4] - self.d) / (self.d - self.c)) ** 2

        return result

    @property
    def parameters(self) -> dict[str, float]:
        return {"a": self.a, "b": self.b, "c": self.c, "d": self.d}


# --- Interval Type-2 Membership Functions ---


class IntervalType2GaussianMF(BaseMembershipFunction):
    """Interval Type-2 Gaussian membership function.

    Defines a Footprint of Uncertainty (FOU) between an upper (UMF)
    and lower (LMF) Gaussian membership function with uncertain sigma.

    Parameters
    ----------
    center : float
        Center of the Gaussian.
    sigma_lower : float
        Sigma for the lower membership function (narrower).
    sigma_upper : float
        Sigma for the upper membership function (wider).
    """

    def __init__(self, center: float, sigma_lower: float, sigma_upper: float):
        if sigma_lower <= 0 or sigma_upper <= 0:
            raise ValueError("Sigmas must be positive")
        if sigma_lower > sigma_upper:
            raise ValueError("sigma_lower must be <= sigma_upper")
        self.center = float(center)
        self.sigma_lower = float(sigma_lower)
        self.sigma_upper = float(sigma_upper)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return upper membership (UMF) values."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.exp(-0.5 * ((x - self.center) / self.sigma_upper) ** 2)

    def upper(self, x: np.ndarray) -> np.ndarray:
        """Upper membership function (wider Gaussian)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.exp(-0.5 * ((x - self.center) / self.sigma_upper) ** 2)

    def lower(self, x: np.ndarray) -> np.ndarray:
        """Lower membership function (narrower Gaussian)."""
        x = np.asarray(x, dtype=np.float64).ravel()
        return np.exp(-0.5 * ((x - self.center) / self.sigma_lower) ** 2)

    @property
    def parameters(self) -> dict[str, float]:
        return {
            "center": self.center,
            "sigma_lower": self.sigma_lower,
            "sigma_upper": self.sigma_upper,
        }


class IntervalType2TriangularMF(BaseMembershipFunction):
    """Interval Type-2 Triangular membership function.

    Defines a FOU between an upper and lower triangular MF.

    Parameters
    ----------
    a_lower, b_lower, c_lower : float
        Parameters for the lower (inner) triangular MF.
    a_upper, b_upper, c_upper : float
        Parameters for the upper (outer) triangular MF.
    """

    def __init__(
        self,
        a_lower: float, b_lower: float, c_lower: float,
        a_upper: float, b_upper: float, c_upper: float,
    ):
        self.lower_mf = TriangularMF(a_lower, b_lower, c_lower)
        self.upper_mf = TriangularMF(a_upper, b_upper, c_upper)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Return upper membership (UMF) values."""
        return self.upper_mf(x)

    def upper(self, x: np.ndarray) -> np.ndarray:
        """Upper membership function."""
        return self.upper_mf(x)

    def lower(self, x: np.ndarray) -> np.ndarray:
        """Lower membership function."""
        return self.lower_mf(x)

    @property
    def parameters(self) -> dict[str, float]:
        return {
            **{f"lower_{k}": v for k, v in self.lower_mf.parameters.items()},
            **{f"upper_{k}": v for k, v in self.upper_mf.parameters.items()},
        }


# --- PyTorch Differentiable Wrapper ---


if HAS_TORCH:
    class _TorchMFWrapper(torch.nn.Module):
        """Wraps a NumPy membership function as a differentiable PyTorch module."""

        def __init__(self, mf: BaseMembershipFunction):
            super().__init__()
            self._mf_class = type(mf)
            for name, value in mf.parameters.items():
                self.register_parameter(
                    name, torch.nn.Parameter(torch.tensor(value, dtype=torch.float32))
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Compute membership using PyTorch ops for gradient flow."""
            name = self._mf_class.__name__
            if name == "GaussianMF":
                center = self.center  # type: ignore[attr-defined]
                sigma = self.sigma  # type: ignore[attr-defined]
                return torch.exp(-0.5 * ((x - center) / sigma) ** 2)
            elif name == "TriangularMF":
                a, b, c = self.a, self.b, self.c  # type: ignore[attr-defined]
                left = (x - a) / (b - a + 1e-10)
                right = (c - x) / (c - b + 1e-10)
                return torch.clamp(torch.min(left, right), 0.0, 1.0)
            elif name == "GeneralizedBellMF":
                a, b, c = self.a, self.b, self.c  # type: ignore[attr-defined]
                return 1.0 / (1.0 + torch.abs((x - c) / a) ** (2 * b))
            elif name == "SigmoidalMF":
                a, c = self.a, self.c  # type: ignore[attr-defined]
                return torch.sigmoid(a * (x - c))
            elif name == "TrapezoidalMF":
                a, b, c, d = self.a, self.b, self.c, self.d  # type: ignore[attr-defined]
                left = (x - a) / (b - a + 1e-10)
                right = (d - x) / (d - c + 1e-10)
                return torch.clamp(torch.min(left, right), 0.0, 1.0)
            else:
                # Fallback: convert to numpy, compute, convert back (no grad)
                with torch.no_grad():
                    mf = self._mf_class(**{
                        name: p.item()
                        for name, p in self.named_parameters()
                    })
                    result = mf(x.detach().cpu().numpy())
                    return torch.tensor(result, dtype=x.dtype, device=x.device)
else:
    class _TorchMFWrapper:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not available."""
        def __init__(self, mf: BaseMembershipFunction):
            raise ImportError("PyTorch is required for differentiable MFs.")


def create_uniform_mfs(
    n_mfs: int,
    x_min: float,
    x_max: float,
    mf_type: str = "gaussian",
) -> list[BaseMembershipFunction]:
    """Create uniformly spaced membership functions over a range.

    Parameters
    ----------
    n_mfs : int
        Number of membership functions.
    x_min : float
        Minimum value of the input range.
    x_max : float
        Maximum value of the input range.
    mf_type : str, default='gaussian'
        Type: 'gaussian', 'triangular', or 'trapezoidal'.

    Returns
    -------
    list of BaseMembershipFunction
    """
    centers = np.linspace(x_min, x_max, n_mfs)
    step = (x_max - x_min) / max(n_mfs - 1, 1)

    mfs: list[BaseMembershipFunction] = []
    if mf_type == "gaussian":
        sigma = step / 2.0
        for c in centers:
            mfs.append(GaussianMF(center=float(c), sigma=max(sigma, 1e-6)))
    elif mf_type == "triangular":
        for i, c in enumerate(centers):
            a = float(c - step) if i > 0 else float(x_min - step)
            b = float(c)
            cc = float(c + step) if i < n_mfs - 1 else float(x_max + step)
            mfs.append(TriangularMF(a=a, b=b, c=cc))
    elif mf_type == "trapezoidal":
        half = step / 4.0
        for c in centers:
            mfs.append(TrapezoidalMF(
                a=float(c - step),
                b=float(c - half),
                c=float(c + half),
                d=float(c + step),
            ))
    else:
        raise ValueError(f"Unknown mf_type: {mf_type}")
    return mfs
