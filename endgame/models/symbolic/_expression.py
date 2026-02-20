"""Expression tree representation for symbolic regression.

Trees are built from four node types:
- Constant: a learnable scalar
- Variable: references a feature column by index
- Unary: applies a unary operator to one child
- Binary: applies a binary operator to two children
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Union

import numpy as np

from endgame.models.symbolic._operators import (
    BINARY_OPS_NP,
    HAS_SYMPY,
    HAS_TORCH,
    PRECEDENCE,
    UNARY_DISPLAY,
    UNARY_OPS_NP,
    _get_sympy_binary,
    _get_sympy_unary,
    get_torch_binary,
    get_torch_unary,
)

# ============================================================
# Node types
# ============================================================

@dataclass
class Constant:
    """Leaf node holding a scalar constant."""
    value: float

    def clone(self) -> Constant:
        return Constant(self.value)


@dataclass
class Variable:
    """Leaf node referencing an input feature."""
    index: int
    name: str | None = None

    def clone(self) -> Variable:
        return Variable(self.index, self.name)


@dataclass
class Unary:
    """Internal node applying a unary operator."""
    op: str
    child: Node

    def clone(self) -> Unary:
        return Unary(self.op, self.child.clone())


@dataclass
class Binary:
    """Internal node applying a binary operator."""
    op: str
    left: Node
    right: Node

    def clone(self) -> Binary:
        return Binary(self.op, self.left.clone(), self.right.clone())


Node = Union[Constant, Variable, Unary, Binary]


# ============================================================
# Tree metrics
# ============================================================

def complexity(node: Node) -> int:
    """Count total number of nodes (operators + operands)."""
    if isinstance(node, (Constant, Variable)):
        return 1
    if isinstance(node, Unary):
        return 1 + complexity(node.child)
    if isinstance(node, Binary):
        return 1 + complexity(node.left) + complexity(node.right)
    raise TypeError(f"Unknown node type: {type(node)}")


def depth(node: Node) -> int:
    """Maximum depth of the tree (leaves have depth 0)."""
    if isinstance(node, (Constant, Variable)):
        return 0
    if isinstance(node, Unary):
        return 1 + depth(node.child)
    if isinstance(node, Binary):
        return 1 + max(depth(node.left), depth(node.right))
    raise TypeError(f"Unknown node type: {type(node)}")


def count_constants(node: Node) -> int:
    """Count number of constant nodes."""
    if isinstance(node, Constant):
        return 1
    if isinstance(node, Variable):
        return 0
    if isinstance(node, Unary):
        return count_constants(node.child)
    if isinstance(node, Binary):
        return count_constants(node.left) + count_constants(node.right)
    return 0


def get_constants(node: Node) -> list[float]:
    """Collect all constant values in DFS order."""
    if isinstance(node, Constant):
        return [node.value]
    if isinstance(node, Variable):
        return []
    if isinstance(node, Unary):
        return get_constants(node.child)
    if isinstance(node, Binary):
        return get_constants(node.left) + get_constants(node.right)
    return []


def set_constants(node: Node, values: list[float], idx: list[int] = None) -> None:
    """Set constant values in DFS order (mutates tree)."""
    if idx is None:
        idx = [0]
    if isinstance(node, Constant):
        if idx[0] < len(values):
            node.value = values[idx[0]]
            idx[0] += 1
    elif isinstance(node, Unary):
        set_constants(node.child, values, idx)
    elif isinstance(node, Binary):
        set_constants(node.left, values, idx)
        set_constants(node.right, values, idx)


def get_variables_used(node: Node) -> set:
    """Return set of variable indices used in the tree."""
    if isinstance(node, Constant):
        return set()
    if isinstance(node, Variable):
        return {node.index}
    if isinstance(node, Unary):
        return get_variables_used(node.child)
    if isinstance(node, Binary):
        return get_variables_used(node.left) | get_variables_used(node.right)
    return set()


# ============================================================
# Enumeration (for selecting random subtrees)
# ============================================================

def _enumerate_nodes(node: Node, acc: list[tuple[Node, Node | None, str]],
                     parent: Node | None = None, child_attr: str = "") -> None:
    """Collect all (node, parent, child_attr) triples."""
    acc.append((node, parent, child_attr))
    if isinstance(node, Unary):
        _enumerate_nodes(node.child, acc, node, "child")
    elif isinstance(node, Binary):
        _enumerate_nodes(node.left, acc, node, "left")
        _enumerate_nodes(node.right, acc, node, "right")


def enumerate_nodes(root: Node) -> list[tuple[Node, Node | None, str]]:
    """Return list of (node, parent, child_attr) for every node in the tree."""
    acc: list[tuple[Node, Node | None, str]] = []
    _enumerate_nodes(root, acc)
    return acc


# ============================================================
# Evaluation
# ============================================================

def evaluate(node: Node, X: np.ndarray) -> np.ndarray:
    """Evaluate the expression tree on data X (n_samples, n_features).

    Returns a 1-D array of shape (n_samples,).
    """
    if isinstance(node, Constant):
        return np.full(X.shape[0], node.value, dtype=np.float64)
    if isinstance(node, Variable):
        return X[:, node.index].astype(np.float64)
    if isinstance(node, Unary):
        child_val = evaluate(node.child, X)
        func = UNARY_OPS_NP.get(node.op)
        if func is None:
            raise ValueError(f"Unknown unary op: {node.op}")
        result = func(child_val)
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10), -1e10, 1e10)
    if isinstance(node, Binary):
        left_val = evaluate(node.left, X)
        right_val = evaluate(node.right, X)
        func = BINARY_OPS_NP.get(node.op)
        if func is None:
            raise ValueError(f"Unknown binary op: {node.op}")
        result = func(left_val, right_val)
        return np.clip(np.nan_to_num(result, nan=0.0, posinf=1e10, neginf=-1e10), -1e10, 1e10)
    raise TypeError(f"Unknown node type: {type(node)}")


# ============================================================
# String representation
# ============================================================

def to_string(node: Node, feature_names: list[str] | None = None,
              parent_op: str | None = None, is_right: bool = False) -> str:
    """Convert expression tree to human-readable infix string."""
    if isinstance(node, Constant):
        v = node.value
        if v == int(v) and abs(v) < 1e6:
            return str(int(v))
        return f"{v:.4g}"
    if isinstance(node, Variable):
        if feature_names and node.index < len(feature_names):
            return feature_names[node.index]
        return node.name or f"x{node.index}"
    if isinstance(node, Unary):
        child_str = to_string(node.child, feature_names)
        template = UNARY_DISPLAY.get(node.op, f"{node.op}({{}})")
        return template.format(child_str)
    if isinstance(node, Binary):
        left_str = to_string(node.left, feature_names, node.op, False)
        right_str = to_string(node.right, feature_names, node.op, True)

        # Use function notation for max/min
        if node.op in ("max", "min"):
            return f"{node.op}({left_str}, {right_str})"

        # Add parens for precedence
        op_prec = PRECEDENCE.get(node.op, 0)
        result = f"{left_str} {node.op} {right_str}"

        if parent_op:
            parent_prec = PRECEDENCE.get(parent_op, 0)
            needs_parens = (
                op_prec < parent_prec
                or (op_prec == parent_prec and is_right and parent_op in ("-", "/"))
            )
            if needs_parens:
                result = f"({result})"

        return result
    raise TypeError(f"Unknown node type: {type(node)}")


# ============================================================
# SymPy conversion
# ============================================================

def to_sympy(node: Node, feature_names: list[str] | None = None):
    """Convert expression tree to SymPy expression."""
    if not HAS_SYMPY:
        raise ImportError("sympy is required for symbolic export. Install with: pip install sympy")
    import sympy

    if isinstance(node, Constant):
        return sympy.Float(node.value)
    if isinstance(node, Variable):
        name = (feature_names[node.index] if feature_names and node.index < len(feature_names)
                else node.name or f"x{node.index}")
        return sympy.Symbol(name)
    if isinstance(node, Unary):
        child_expr = to_sympy(node.child, feature_names)
        func = _get_sympy_unary(node.op)
        if func is None:
            raise ValueError(f"No sympy mapping for unary op: {node.op}")
        return func(child_expr)
    if isinstance(node, Binary):
        left_expr = to_sympy(node.left, feature_names)
        right_expr = to_sympy(node.right, feature_names)
        func = _get_sympy_binary(node.op)
        if func is None:
            raise ValueError(f"No sympy mapping for binary op: {node.op}")
        return func(left_expr, right_expr)
    raise TypeError(f"Unknown node type: {type(node)}")


# ============================================================
# PyTorch evaluation (for constant optimisation)
# ============================================================

def evaluate_torch(node: Node, X_tensor):
    """Evaluate expression tree using torch tensors (for autograd).

    Constants are represented as torch parameters so gradients flow.
    This function expects that constants have already been replaced
    with torch.nn.Parameter objects via a separate mechanism.

    For simplicity, we use a dict mapping constant-id to parameter and
    reconstruct in _constant_optimizer.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for constant optimization")
    import torch

    if isinstance(node, Constant):
        return torch.full((X_tensor.shape[0],), node.value,
                          dtype=X_tensor.dtype, device=X_tensor.device)
    if isinstance(node, Variable):
        return X_tensor[:, node.index]
    if isinstance(node, Unary):
        child_val = evaluate_torch(node.child, X_tensor)
        func = get_torch_unary(node.op)
        if func is None:
            raise ValueError(f"No torch mapping for unary op: {node.op}")
        return func(child_val)
    if isinstance(node, Binary):
        left_val = evaluate_torch(node.left, X_tensor)
        right_val = evaluate_torch(node.right, X_tensor)
        func = get_torch_binary(node.op)
        if func is None:
            raise ValueError(f"No torch mapping for binary op: {node.op}")
        return func(left_val, right_val)
    raise TypeError(f"Unknown node type: {type(node)}")
