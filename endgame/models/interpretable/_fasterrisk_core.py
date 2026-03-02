"""Native FasterRisk algorithm implementation.

Implements the three-stage FasterRisk algorithm from:
Liu & Rudin, "FasterRisk: Fast and Accurate Interpretable Risk Scores",
NeurIPS 2022.

Stages:
  1. Beam search for k-sparse continuous logistic regression solution
  2. Diverse pool collection via feature swapping (Rashomon set)
  3. Star ray search with auxiliary loss rounding for integer coefficients

The algorithm produces sparse linear classifiers with integer coefficients
(risk scoring systems) that can be printed on an index card and computed
by hand, while maintaining competitive predictive accuracy.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Loss and gradient helpers
# ---------------------------------------------------------------------------

def _logistic_loss(y_scores: np.ndarray) -> float:
    """Logistic loss: sum log(1 + exp(-y_scores)), numerically stable."""
    return float(np.sum(np.logaddexp(0, -y_scores)))


def _sigma(y_scores: np.ndarray) -> np.ndarray:
    """Misclassification probability: 1 / (1 + exp(y_scores))."""
    return 1.0 / (1.0 + np.exp(np.clip(y_scores, -500, 500)))


# ---------------------------------------------------------------------------
# Stage 1 helpers: coordinate descent and beam search
# ---------------------------------------------------------------------------

def _sparse_cd(
    yX: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    beta0: float,
    y_scores: np.ndarray,
    support: list[int],
    lb: np.ndarray,
    ub: np.ndarray,
    Lj: np.ndarray,
    L0: float,
    lam2: float,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Cyclic coordinate descent on given support until convergence.

    Updates *beta* and *y_scores* in place.

    Returns
    -------
    beta0 : float
        Updated intercept.
    loss : float
        Final loss value.
    """
    prev_loss = _logistic_loss(y_scores) + lam2 * float(np.dot(beta, beta))

    for _ in range(max_iter):
        # Intercept update
        sig = _sigma(y_scores)
        delta_0 = float(np.dot(y, sig)) / L0
        beta0 += delta_0
        y_scores += y * delta_0

        # Feature updates (Gauss-Seidel: recompute sigma after each)
        for j in support:
            sig = _sigma(y_scores)
            grad_j = -float(np.dot(yX[:, j], sig)) + 2 * lam2 * beta[j]
            beta_j_new = float(np.clip(beta[j] - grad_j / Lj[j], lb[j], ub[j]))
            delta_j = beta_j_new - beta[j]
            if abs(delta_j) > 1e-15:
                beta[j] = beta_j_new
                y_scores += yX[:, j] * delta_j

        loss = _logistic_loss(y_scores) + lam2 * float(np.dot(beta, beta))
        if abs(prev_loss - loss) / (abs(prev_loss) + 1e-10) < tol:
            break
        prev_loss = loss

    return beta0, loss


def _beam_search(
    X: np.ndarray,
    y: np.ndarray,
    yX: np.ndarray,
    k: int,
    lb: np.ndarray,
    ub: np.ndarray,
    Lj: np.ndarray,
    L0: float,
    lam2: float,
    parent_size: int = 10,
    max_iter: int = 1000,
) -> tuple[float, float, np.ndarray, list[int], np.ndarray]:
    """Stage 1: Beam search for *k*-sparse continuous solution.

    Returns (loss, beta0, beta, support, y_scores).
    """
    n, p = X.shape

    # Intercept-only initialisation
    n_pos = int(np.sum(y == 1))
    n_neg = n - n_pos
    beta0_init = float(np.log(max(n_pos, 1) / max(n_neg, 1)))
    beta_init = np.zeros(p, dtype=np.float64)
    y_scores_init = y * beta0_init
    loss_init = _logistic_loss(y_scores_init)

    # beam element: (loss, beta0, beta, support_list, y_scores)
    beam = [(loss_init, beta0_init, beta_init, [], y_scores_init)]

    for _t in range(k):
        candidates: list[tuple[float, int, int, float]] = []

        for pidx, (_, _b0, beta_p, support_p, ys_p) in enumerate(beam):
            support_set = set(support_p)
            sig = _sigma(ys_p)

            # Vectorised gradient & tentative steps for all features
            grads = -(yX.T @ sig) + 2 * lam2 * beta_p
            deltas = np.clip(beta_p - grads / Lj, lb, ub) - beta_p

            # Zero-out features already in support
            for j in support_set:
                deltas[j] = 0.0

            cand_idx = np.where(np.abs(deltas) > 1e-15)[0]
            if len(cand_idx) == 0:
                continue

            # Vectorised tentative loss evaluation
            Y_tent = ys_p[:, None] + yX[:, cand_idx] * deltas[cand_idx]
            losses = np.sum(np.logaddexp(0, -Y_tent), axis=0)

            for i, j in enumerate(cand_idx):
                candidates.append(
                    (float(losses[i]), pidx, int(j), float(deltas[j]))
                )

        if not candidates:
            break

        # Keep top parent_size by tentative loss, then fine-tune with CD
        candidates.sort(key=lambda c: c[0])
        candidates = candidates[:parent_size]

        new_beam = []
        for _, pidx, j_new, delta_j in candidates:
            _, beta0_c, beta_c, support_c, ys_c = beam[pidx]
            beta_c = beta_c.copy()
            ys_c = ys_c.copy()

            # Apply tentative step
            beta_c[j_new] += delta_j
            ys_c += yX[:, j_new] * delta_j
            new_support = support_c + [j_new]

            beta0_c, loss_c = _sparse_cd(
                yX, y, beta_c, beta0_c, ys_c, new_support,
                lb, ub, Lj, L0, lam2, max_iter=max_iter,
            )
            new_beam.append((loss_c, beta0_c, beta_c, new_support, ys_c))

        new_beam.sort(key=lambda b: b[0])
        beam = new_beam[:parent_size]

    return beam[0]


# ---------------------------------------------------------------------------
# Stage 2: diverse pool
# ---------------------------------------------------------------------------

def _collect_diverse_pool(
    X: np.ndarray,
    y: np.ndarray,
    yX: np.ndarray,
    beta_star: np.ndarray,
    beta0_star: float,
    support_star: list[int],
    y_scores_star: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    Lj: np.ndarray,
    L0: float,
    lam2: float,
    pool_size: int = 50,
    gap_tol: float = 0.05,
) -> list[tuple[np.ndarray, float]]:
    """Stage 2: Collect diverse near-optimal continuous solutions.

    Returns list of ``(beta, beta0)`` tuples.
    """
    loss_star = _logistic_loss(y_scores_star) + lam2 * float(
        np.dot(beta_star, beta_star)
    )
    pool: list[tuple[np.ndarray, float, float]] = [
        (beta_star.copy(), beta0_star, loss_star)
    ]

    support_set = set(support_star)
    non_support = np.array(
        [j for j in range(X.shape[1]) if j not in support_set]
    )
    if len(non_support) == 0:
        return [(b, b0) for b, b0, _ in pool]

    for j_minus in support_star:
        # Remove feature j_minus
        ys_temp = y_scores_star - yX[:, j_minus] * beta_star[j_minus]

        # Rank non-support features by |gradient|
        sig = _sigma(ys_temp)
        grads = np.abs(yX[:, non_support].T @ sig)
        top_order = np.argsort(-grads)[: pool_size]
        top_candidates = non_support[top_order]

        for j_plus in top_candidates:
            new_support = [
                j for j in support_star if j != j_minus
            ] + [int(j_plus)]

            beta_swap = beta_star.copy()
            beta_swap[j_minus] = 0.0
            ys_swap = ys_temp.copy()

            beta0_swap, loss_swap = _sparse_cd(
                yX, y, beta_swap, beta0_star, ys_swap, new_support,
                lb, ub, Lj, L0, lam2, max_iter=10,
            )

            if (loss_swap - loss_star) / (abs(loss_star) + 1e-10) < gap_tol:
                pool.append((beta_swap.copy(), beta0_swap, loss_swap))

    pool.sort(key=lambda x: x[2])
    pool = pool[:pool_size]
    return [(b, b0) for b, b0, _ in pool]


# ---------------------------------------------------------------------------
# Stage 3: auxiliary rounding + star ray search
# ---------------------------------------------------------------------------

def _auxiliary_round(
    betas_scaled: np.ndarray,
    yX: np.ndarray,
    y: np.ndarray,
    beta0_int: int,
) -> np.ndarray:
    """Auxiliary-loss sequential rounding (Algorithm 3 in the paper).

    Greedily rounds one coefficient at a time, choosing the
    (dimension, direction) pair that minimises an auxiliary loss bound.

    Returns integer coefficient array.
    """
    p = len(betas_scaled)
    floor_vals = np.floor(betas_scaled)
    ceil_vals = np.ceil(betas_scaled)

    needs_rounding = np.where(np.abs(ceil_vals - floor_vals) > 0.5)[0]
    if len(needs_rounding) == 0:
        return np.round(betas_scaled).astype(int)

    result = betas_scaled.copy()
    already_int = np.abs(ceil_vals - floor_vals) < 0.5
    result[already_int] = np.round(betas_scaled[already_int])

    # Worst-case logistic weights (Gamma matrix, fixed throughout rounding)
    Gamma = np.where(yX > 0, floor_vals, ceil_vals)
    yXB_extreme = np.sum(yX * Gamma, axis=1) + y * beta0_int
    l_factors = _sigma(-yXB_extreme)  # 1/(1+exp(yXB_extreme))

    # Weighted features for dims needing rounding
    lyX = l_factors[:, None] * yX[:, needs_rounding]
    lyX_norm_sq = np.sum(lyX ** 2, axis=0)

    dist_floor = floor_vals[needs_rounding] - betas_scaled[needs_rounding]
    dist_ceil = ceil_vals[needs_rounding] - betas_scaled[needs_rounding]

    n = len(y)
    running_diff = np.zeros(n)
    rounded = np.zeros(len(needs_rounding), dtype=bool)

    for _ in range(len(needs_rounding)):
        active_idx = np.where(~rounded)[0]
        if len(active_idx) == 0:
            break

        # Vectorised cost computation
        dots = lyX[:, active_idx].T @ running_diff
        running_sq = float(np.dot(running_diff, running_diff))

        df = dist_floor[active_idx]
        dc = dist_ceil[active_idx]
        lnorm = lyX_norm_sq[active_idx]

        cost_floor = running_sq + 2 * df * dots + df ** 2 * lnorm
        cost_ceil = running_sq + 2 * dc * dots + dc ** 2 * lnorm

        floor_better = cost_floor <= cost_ceil
        best_costs = np.where(floor_better, cost_floor, cost_ceil)

        best_pos = int(np.argmin(best_costs))
        best_local = active_idx[best_pos]
        best_orig = needs_rounding[best_local]

        if floor_better[best_pos]:
            result[best_orig] = floor_vals[best_orig]
            running_diff += dist_floor[best_local] * lyX[:, best_local]
        else:
            result[best_orig] = ceil_vals[best_orig]
            running_diff += dist_ceil[best_local] * lyX[:, best_local]

        rounded[best_local] = True

    return result.astype(int)


def _star_ray_search(
    pool: list[tuple[np.ndarray, float]],
    X: np.ndarray,
    y: np.ndarray,
    yX: np.ndarray,
    max_coef: int,
    num_rays: int = 20,
) -> tuple[float, int, np.ndarray]:
    """Stage 3: Find best multiplier and integer coefficients.

    Returns ``(multiplier, intercept_int, coefficients_int)``.
    """
    best_loss = np.inf
    best_result = None

    for beta_cont, beta0_cont in pool:
        support = np.where(np.abs(beta_cont) > 1e-10)[0]

        if len(support) == 0:
            beta0_int = int(np.round(beta0_cont))
            loss = _logistic_loss(y * beta0_int)
            if loss < best_loss:
                best_loss = loss
                best_result = (
                    1.0, beta0_int,
                    np.zeros(len(beta_cont), dtype=int),
                )
            continue

        # Multiplier range: m * |beta[j]| <= max_coef for all j in support
        max_mult = float(np.min(max_coef / np.abs(beta_cont[support])))
        max_mult = max(max_mult, 1.0)
        multipliers = np.linspace(1.0, max_mult, max(num_rays, 1))

        for m in multipliers:
            beta_scaled = m * beta_cont
            beta0_int = int(np.round(m * beta0_cont))

            beta_int = _auxiliary_round(beta_scaled, yX, y, beta0_int)
            beta_int = np.clip(beta_int, -max_coef, max_coef)

            scores = X @ beta_int + beta0_int
            loss = _logistic_loss(y * scores / m)

            if loss < best_loss:
                best_loss = loss
                best_result = (float(m), beta0_int, beta_int.copy())

    if best_result is None:
        n_pos = int(np.sum(y == 1))
        n = len(y)
        beta0_int = int(
            np.round(np.log(max(n_pos, 1) / max(n - n_pos, 1)))
        )
        best_result = (1.0, beta0_int, np.zeros(X.shape[1], dtype=int))

    return best_result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fasterrisk_fit(
    X: np.ndarray,
    y_binary: np.ndarray,
    k: int,
    max_coef: int = 5,
    parent_size: int = 10,
    pool_size: int = 50,
    num_rays: int = 20,
    max_iter: int = 1000,
    lambda2: float = 1e-8,
) -> tuple[float, int, np.ndarray]:
    """Fit a FasterRisk model: sparse integer risk scores.

    Parameters
    ----------
    X : ndarray of shape (n, p)
        Feature matrix (typically binary 0/1 after discretisation).
    y_binary : ndarray of shape (n,)
        Labels in {-1, +1}.
    k : int
        Sparsity (maximum number of non-zero coefficients).
    max_coef : int
        Maximum absolute value for integer coefficients.
    parent_size : int
        Beam width in the beam search.
    pool_size : int
        Number of diverse solutions to collect.
    num_rays : int
        Number of multiplier candidates in the star ray search.
    max_iter : int
        Maximum coordinate-descent iterations.
    lambda2 : float
        L2 regularisation (small, for numerical stability).

    Returns
    -------
    multiplier : float
        Scaling factor for probability calibration.
    intercept : int
        Integer intercept.
    coefficients : ndarray of int
        Integer feature coefficients.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y_binary, dtype=np.float64)
    n, p = X.shape

    yX = y[:, None] * X
    Lj = 0.25 * np.sum(X ** 2, axis=0) + 2 * lambda2
    Lj = np.maximum(Lj, 1e-10)
    L0 = n * 0.25

    lb = -max_coef * np.ones(p)
    ub = max_coef * np.ones(p)
    k = min(k, p)

    # Stage 1
    loss, beta0, beta, support, y_scores = _beam_search(
        X, y, yX, k, lb, ub, Lj, L0, lambda2, parent_size, max_iter,
    )

    # Stage 2
    pool = _collect_diverse_pool(
        X, y, yX, beta, beta0, support, y_scores,
        lb, ub, Lj, L0, lambda2, pool_size,
    )

    # Stage 3
    return _star_ray_search(pool, X, y, yX, max_coef, num_rays)
