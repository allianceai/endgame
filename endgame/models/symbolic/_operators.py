"""Safe operator registries for symbolic regression.

Provides numpy, torch, and sympy operator mappings with safe evaluation
(division-by-zero protection, overflow clamping, etc.).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

# Check for optional dependencies
try:
    import sympy
    HAS_SYMPY = True
except ImportError:
    sympy = None
    HAS_SYMPY = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False


# ============================================================
# Safe numpy functions
# ============================================================

def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Division with zero-protection."""
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(np.abs(b) > 1e-10, a / b, 1.0)
    return np.clip(result, -1e10, 1e10)


def _safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.abs(x) + 1e-10)


def _safe_log2(x: np.ndarray) -> np.ndarray:
    return np.log2(np.abs(x) + 1e-10)


def _safe_log10(x: np.ndarray) -> np.ndarray:
    return np.log10(np.abs(x) + 1e-10)


def _safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.abs(x))


def _safe_pow(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.power(np.abs(a) + 1e-10, np.clip(b, -5, 5))
    return np.clip(result, -1e10, 1e10)


def _safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -500, 500))


def _safe_tan(x: np.ndarray) -> np.ndarray:
    result = np.tan(x)
    return np.clip(result, -1e10, 1e10)


# ============================================================
# Numpy operator registry
# ============================================================

# Binary operators: (numpy_func, arity=2, display_string)
BINARY_OPS_NP: dict[str, Callable] = {
    "+": np.add,
    "-": np.subtract,
    "*": np.multiply,
    "/": _safe_div,
    "^": _safe_pow,
    "max": np.maximum,
    "min": np.minimum,
}

# Unary operators: (numpy_func, arity=1, display_string)
UNARY_OPS_NP: dict[str, Callable] = {
    "sin": np.sin,
    "cos": np.cos,
    "tan": _safe_tan,
    "exp": _safe_exp,
    "log": _safe_log,
    "log2": _safe_log2,
    "log10": _safe_log10,
    "sqrt": _safe_sqrt,
    "square": np.square,
    "cube": lambda x: x ** 3,
    "abs": np.abs,
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    "neg": np.negative,
    "inv": lambda x: _safe_div(np.ones_like(x), x),
}


# ============================================================
# Sympy operator registry
# ============================================================

def _get_sympy_binary(name: str):
    """Get the sympy binary operator function."""
    if not HAS_SYMPY:
        raise ImportError("sympy is required for symbolic export")
    _map = {
        "+": lambda a, b: a + b,
        "-": lambda a, b: a - b,
        "*": lambda a, b: a * b,
        "/": lambda a, b: a / b,
        "^": lambda a, b: a ** b,
        "max": sympy.Max,
        "min": sympy.Min,
    }
    return _map.get(name)


def _get_sympy_unary(name: str):
    """Get the sympy unary operator function."""
    if not HAS_SYMPY:
        raise ImportError("sympy is required for symbolic export")
    _map = {
        "sin": sympy.sin,
        "cos": sympy.cos,
        "tan": sympy.tan,
        "exp": sympy.exp,
        "log": sympy.log,
        "log2": lambda x: sympy.log(x, 2),
        "log10": lambda x: sympy.log(x, 10),
        "sqrt": sympy.sqrt,
        "square": lambda x: x ** 2,
        "cube": lambda x: x ** 3,
        "abs": sympy.Abs,
        "sign": sympy.sign,
        "floor": sympy.floor,
        "ceil": sympy.ceiling,
        "neg": lambda x: -x,
        "inv": lambda x: 1 / x,
    }
    return _map.get(name)


# ============================================================
# PyTorch operator registry (for constant optimization)
# ============================================================

def _torch_safe_div(a, b):
    return torch.where(torch.abs(b) > 1e-10, a / b, torch.ones_like(a))


def _torch_safe_log(x):
    return torch.log(torch.abs(x) + 1e-10)


def _torch_safe_sqrt(x):
    return torch.sqrt(torch.abs(x))


def _torch_safe_pow(a, b):
    return torch.pow(torch.abs(a) + 1e-10, torch.clamp(b, -5, 5))


def _torch_safe_exp(x):
    return torch.exp(torch.clamp(x, -500, 500))


def _torch_safe_tan(x):
    return torch.clamp(torch.tan(x), -1e10, 1e10)


def get_torch_binary(name: str) -> Callable | None:
    """Get torch binary operator, or None if unknown."""
    if not HAS_TORCH:
        return None
    _map = {
        "+": torch.add,
        "-": torch.sub,
        "*": torch.mul,
        "/": _torch_safe_div,
        "^": _torch_safe_pow,
        "max": torch.maximum,
        "min": torch.minimum,
    }
    return _map.get(name)


def get_torch_unary(name: str) -> Callable | None:
    """Get torch unary operator, or None if unknown."""
    if not HAS_TORCH:
        return None
    _map = {
        "sin": torch.sin,
        "cos": torch.cos,
        "tan": _torch_safe_tan,
        "exp": _torch_safe_exp,
        "log": _torch_safe_log,
        "log2": lambda x: torch.log2(torch.abs(x) + 1e-10),
        "log10": lambda x: torch.log10(torch.abs(x) + 1e-10),
        "sqrt": _torch_safe_sqrt,
        "square": torch.square,
        "cube": lambda x: x ** 3,
        "abs": torch.abs,
        "sign": torch.sign,
        "floor": torch.floor,
        "ceil": torch.ceil,
        "neg": torch.neg,
        "inv": lambda x: _torch_safe_div(torch.ones_like(x), x),
    }
    return _map.get(name)


# ============================================================
# Operator display helpers
# ============================================================

BINARY_DISPLAY: dict[str, str] = {
    "+": "{} + {}",
    "-": "{} - {}",
    "*": "{} * {}",
    "/": "{} / {}",
    "^": "({})^({})",
    "max": "max({}, {})",
    "min": "min({}, {})",
}

UNARY_DISPLAY: dict[str, str] = {
    "sin": "sin({})",
    "cos": "cos({})",
    "tan": "tan({})",
    "exp": "exp({})",
    "log": "log({})",
    "log2": "log2({})",
    "log10": "log10({})",
    "sqrt": "sqrt({})",
    "square": "({})^2",
    "cube": "({})^3",
    "abs": "abs({})",
    "sign": "sign({})",
    "floor": "floor({})",
    "ceil": "ceil({})",
    "neg": "-({})",
    "inv": "1/({})",
}

# Operator precedence for parenthesization
PRECEDENCE: dict[str, int] = {
    "+": 1,
    "-": 1,
    "*": 2,
    "/": 2,
    "^": 3,
    "max": 10,
    "min": 10,
}


# ============================================================
# Preset operator sets
# ============================================================

OPERATOR_SETS = {
    "basic": {
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": [],
    },
    "arithmetic": {
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": ["square", "cube", "sqrt", "abs"],
    },
    "trigonometric": {
        "binary_operators": ["+", "-", "*", "/"],
        "unary_operators": ["sin", "cos", "tan"],
    },
    "scientific": {
        "binary_operators": ["+", "-", "*", "/", "^"],
        "unary_operators": ["sin", "cos", "exp", "log", "sqrt", "square", "abs"],
    },
    "full": {
        "binary_operators": ["+", "-", "*", "/", "^", "max", "min"],
        "unary_operators": [
            "sin", "cos", "tan", "exp", "log", "log10", "log2",
            "sqrt", "square", "cube", "abs", "sign", "floor", "ceil",
        ],
    },
}


def validate_operators(binary_operators, unary_operators):
    """Validate that all operators are supported."""
    for op in binary_operators:
        if op not in BINARY_OPS_NP:
            raise ValueError(
                f"Unknown binary operator: {op!r}. "
                f"Supported: {list(BINARY_OPS_NP.keys())}"
            )
    for op in unary_operators:
        if op not in UNARY_OPS_NP:
            raise ValueError(
                f"Unknown unary operator: {op!r}. "
                f"Supported: {list(UNARY_OPS_NP.keys())}"
            )
