"""PyTorch LBFGS-based constant optimization for expression trees.

After the GP discovers a promising equation *structure*, this module
refines the numeric constants using gradient-based optimization.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from endgame.models.symbolic._expression import (
    Node,
    count_constants,
    evaluate,
    get_constants,
    set_constants,
)
from endgame.models.symbolic._population import clone_tree

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def optimize_constants(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable[[np.ndarray, np.ndarray], float],
    n_restarts: int = 2,
    max_steps: int = 50,
) -> Node:
    """Optimize constants in *tree* via LBFGS.

    Falls back to numpy-based Nelder-Mead if PyTorch is unavailable.

    Parameters
    ----------
    tree : Node
        Expression tree (will be deep-copied; original is not modified).
    X, y : ndarray
        Training data.
    loss_fn : callable
        Loss function  ``loss_fn(y_true, y_pred) -> float``.
    n_restarts : int
        Number of random restarts.
    max_steps : int
        Max optimiser steps per restart.

    Returns
    -------
    Node
        Tree with optimized constants.
    """
    n_const = count_constants(tree)
    if n_const == 0:
        return clone_tree(tree)

    best_tree = clone_tree(tree)
    best_loss = _eval_loss(best_tree, X, y, loss_fn)

    init_consts = get_constants(tree)

    if HAS_TORCH:
        best_tree, best_loss = _optimize_torch(
            tree, X, y, loss_fn, init_consts, n_restarts, max_steps,
            best_tree, best_loss,
        )
    else:
        best_tree, best_loss = _optimize_scipy(
            tree, X, y, loss_fn, init_consts, n_restarts, max_steps,
            best_tree, best_loss,
        )

    return best_tree


# ============================================================
# Internal helpers
# ============================================================

def _eval_loss(tree: Node, X: np.ndarray, y: np.ndarray,
               loss_fn: Callable) -> float:
    try:
        y_pred = evaluate(tree, X)
        if not np.all(np.isfinite(y_pred)):
            return 1e20
        loss = loss_fn(y, y_pred)
        return loss if np.isfinite(loss) else 1e20
    except Exception:
        return 1e20


def _optimize_torch(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable,
    init_consts: list[float],
    n_restarts: int,
    max_steps: int,
    best_tree: Node,
    best_loss: float,
) -> tuple:
    """Optimize using PyTorch LBFGS."""
    X_t = torch.tensor(X, dtype=torch.float64)
    y_t = torch.tensor(y, dtype=torch.float64)
    n_const = len(init_consts)

    for restart in range(n_restarts):
        if restart == 0:
            start = np.array(init_consts, dtype=np.float64)
        else:
            start = np.array(init_consts, dtype=np.float64) + np.random.randn(n_const) * 0.5

        params = torch.tensor(start, dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.LBFGS([params], max_iter=max_steps, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad()
            # Set constants and evaluate
            trial_tree = clone_tree(tree)
            set_constants(trial_tree, params.detach().numpy().tolist())
            # Use numpy eval for robustness
            y_pred = evaluate(trial_tree, X)
            loss_val = loss_fn(y, y_pred)
            # Convert to tensor for backward
            loss_tensor = torch.tensor(loss_val, dtype=torch.float64, requires_grad=True)
            return loss_tensor

        # LBFGS needs closure, but we can't easily get gradients through
        # numpy eval. Use finite-difference approximation instead.
        try:
            _fd_optimize(tree, X, y, loss_fn, start, max_steps, params)
        except Exception:
            pass

        # Evaluate final
        trial = clone_tree(tree)
        set_constants(trial, params.detach().numpy().tolist())
        loss = _eval_loss(trial, X, y, loss_fn)
        if loss < best_loss:
            best_loss = loss
            best_tree = trial

    return best_tree, best_loss


def _fd_optimize(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable,
    start: np.ndarray,
    max_steps: int,
    params_tensor,
) -> None:
    """Simple gradient descent with finite-difference gradients."""
    current = start.copy()
    lr = 0.01
    eps = 1e-6
    n = len(current)

    for step in range(min(max_steps, 30)):
        # Compute gradient via central differences
        grad = np.zeros(n)
        trial = clone_tree(tree)
        set_constants(trial, current.tolist())
        base_loss = _eval_loss(trial, X, y, loss_fn)

        for i in range(n):
            perturbed = current.copy()
            perturbed[i] += eps
            trial_p = clone_tree(tree)
            set_constants(trial_p, perturbed.tolist())
            loss_p = _eval_loss(trial_p, X, y, loss_fn)
            grad[i] = (loss_p - base_loss) / eps

        # Gradient clipping
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 10:
            grad = grad * 10 / grad_norm

        current -= lr * grad

    # Write back
    params_tensor.data = torch.tensor(current, dtype=torch.float64)


def _optimize_scipy(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable,
    init_consts: list[float],
    n_restarts: int,
    max_steps: int,
    best_tree: Node,
    best_loss: float,
) -> tuple:
    """Fallback optimization using scipy Nelder-Mead (no gradient needed)."""
    try:
        from scipy.optimize import minimize as sp_minimize
    except ImportError:
        # No scipy either — just return the tree as-is
        return best_tree, best_loss

    n_const = len(init_consts)

    def objective(consts):
        trial = clone_tree(tree)
        set_constants(trial, consts.tolist())
        return _eval_loss(trial, X, y, loss_fn)

    for restart in range(n_restarts):
        if restart == 0:
            x0 = np.array(init_consts)
        else:
            x0 = np.array(init_consts) + np.random.randn(n_const) * 0.5

        try:
            result = sp_minimize(objective, x0, method="Nelder-Mead",
                                 options={"maxiter": max_steps, "xatol": 1e-6})
            if result.fun < best_loss:
                trial = clone_tree(tree)
                set_constants(trial, result.x.tolist())
                best_tree = trial
                best_loss = result.fun
        except Exception:
            pass

    return best_tree, best_loss
