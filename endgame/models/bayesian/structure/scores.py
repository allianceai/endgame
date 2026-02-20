"""Scoring functions for Bayesian Network structure learning.

This module provides various scoring functions used to evaluate
candidate DAG structures during structure learning:

- BDeu (Bayesian Dirichlet equivalent uniform): Default choice
- BIC (Bayesian Information Criterion): Penalizes complexity
- K2: Prior-based score, requires variable ordering
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.special import gammaln

if TYPE_CHECKING:
    import networkx as nx


def bdeu_score(
    data: np.ndarray,
    parents: list[int],
    child: int,
    cardinalities: dict[int, int],
    equivalent_sample_size: float = 10.0,
) -> float:
    """
    Compute BDeu (Bayesian Dirichlet equivalent uniform) score.

    The BDeu score assumes a uniform prior over parameters and is
    score-equivalent (DAGs in the same equivalence class get the same score).

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    parents : List[int]
        Indices of parent variables.
    child : int
        Index of child variable.
    cardinalities : Dict[int, int]
        Number of states for each variable.
    equivalent_sample_size : float, default=10.0
        ESS parameter. Lower values = more aggressive pruning.

    Returns
    -------
    float
        BDeu score component for this family (higher is better).

    Notes
    -----
    BDeu score formula:

    .. math::
        \\text{BDeu}(X_i | \\text{Pa}_i) = \\sum_j \\left[
            \\log \\frac{\\Gamma(\\alpha_{ij})}{\\Gamma(\\alpha_{ij} + N_{ij})}
            + \\sum_k \\log \\frac{\\Gamma(\\alpha_{ijk} + N_{ijk})}{\\Gamma(\\alpha_{ijk})}
        \\right]

    Where:
    - j indexes parent configurations
    - k indexes child states
    - α_ijk = ESS / (q_i * r_i) with q_i = number of parent configs, r_i = child cardinality
    """
    n_samples = data.shape[0]
    child_card = cardinalities.get(child, len(np.unique(data[:, child])))

    if len(parents) == 0:
        # No parents - simple case
        counts = np.bincount(data[:, child].astype(int), minlength=child_card)
        alpha = equivalent_sample_size / child_card

        score = gammaln(equivalent_sample_size) - gammaln(n_samples + equivalent_sample_size)
        score += np.sum(gammaln(counts + alpha) - gammaln(alpha))
        return score

    # Compute parent configuration cardinality
    parent_cards = [cardinalities.get(p, len(np.unique(data[:, p]))) for p in parents]
    q_i = int(np.prod(parent_cards))  # number of parent configurations
    r_i = child_card

    # α_ijk = ESS / (q_i * r_i)
    alpha = equivalent_sample_size / (q_i * r_i)
    alpha_j = equivalent_sample_size / q_i

    # Compute parent configuration indices
    parent_data = data[:, parents].astype(int)

    # Convert parent configs to single index
    multipliers = np.ones(len(parents), dtype=int)
    for i in range(len(parents) - 1, 0, -1):
        multipliers[i-1] = multipliers[i] * parent_cards[i]

    parent_configs = np.dot(parent_data, multipliers)
    child_data = data[:, child].astype(int)

    # Count occurrences
    score = 0.0

    for j in range(q_i):
        # Samples matching this parent configuration
        mask = parent_configs == j
        n_j = mask.sum()

        if n_j == 0:
            # No samples - prior contribution only
            score += gammaln(alpha_j) - gammaln(alpha_j)
            score += r_i * (gammaln(alpha) - gammaln(alpha))
            continue

        # N_ij = total samples with parent config j
        score += gammaln(alpha_j) - gammaln(n_j + alpha_j)

        # Count child values for this parent config
        child_subset = child_data[mask]
        counts = np.bincount(child_subset, minlength=r_i)

        # Sum over child states
        score += np.sum(gammaln(counts + alpha) - gammaln(alpha))

    return score


def bic_score(
    data: np.ndarray,
    parents: list[int],
    child: int,
    cardinalities: dict[int, int],
) -> float:
    """
    Compute BIC (Bayesian Information Criterion) score.

    BIC balances model fit with complexity penalty based on sample size.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    parents : List[int]
        Indices of parent variables.
    child : int
        Index of child variable.
    cardinalities : Dict[int, int]
        Number of states for each variable.

    Returns
    -------
    float
        BIC score component (higher is better, but note BIC prefers simpler models).

    Notes
    -----
    BIC = log-likelihood - (k/2) * log(n)

    where k is the number of free parameters.
    """
    n_samples = data.shape[0]
    child_card = cardinalities.get(child, len(np.unique(data[:, child])))

    # Log-likelihood
    ll = _compute_log_likelihood(data, parents, child, cardinalities)

    # Number of free parameters
    if len(parents) == 0:
        n_params = child_card - 1
    else:
        parent_cards = [cardinalities.get(p, len(np.unique(data[:, p]))) for p in parents]
        q_i = int(np.prod(parent_cards))
        n_params = q_i * (child_card - 1)

    # BIC penalty
    penalty = (n_params / 2) * np.log(n_samples)

    return ll - penalty


def k2_score(
    data: np.ndarray,
    parents: list[int],
    child: int,
    cardinalities: dict[int, int],
) -> float:
    """
    Compute K2 score (Cooper & Herskovits, 1992).

    K2 uses a specific prior that doesn't require the ESS parameter.
    Assumes uniform prior over structures.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    parents : List[int]
        Indices of parent variables.
    child : int
        Index of child variable.
    cardinalities : Dict[int, int]
        Number of states for each variable.

    Returns
    -------
    float
        K2 score component (higher is better).
    """
    n_samples = data.shape[0]
    child_card = cardinalities.get(child, len(np.unique(data[:, child])))

    if len(parents) == 0:
        counts = np.bincount(data[:, child].astype(int), minlength=child_card)
        # K2 formula: log(Γ(r_i)) - log(Γ(N + r_i)) + sum_k log(Γ(N_k + 1))
        score = gammaln(child_card) - gammaln(n_samples + child_card)
        score += np.sum(gammaln(counts + 1))
        return score

    parent_cards = [cardinalities.get(p, len(np.unique(data[:, p]))) for p in parents]
    q_i = int(np.prod(parent_cards))
    r_i = child_card

    # Compute parent configurations
    parent_data = data[:, parents].astype(int)
    multipliers = np.ones(len(parents), dtype=int)
    for i in range(len(parents) - 1, 0, -1):
        multipliers[i-1] = multipliers[i] * parent_cards[i]
    parent_configs = np.dot(parent_data, multipliers)
    child_data = data[:, child].astype(int)

    score = 0.0

    for j in range(q_i):
        mask = parent_configs == j
        n_j = mask.sum()

        score += gammaln(r_i) - gammaln(n_j + r_i)

        if n_j > 0:
            child_subset = child_data[mask]
            counts = np.bincount(child_subset, minlength=r_i)
            score += np.sum(gammaln(counts + 1))
        else:
            # No samples - just add log(Γ(1)) = 0 for each state
            pass

    return score


def _compute_log_likelihood(
    data: np.ndarray,
    parents: list[int],
    child: int,
    cardinalities: dict[int, int],
) -> float:
    """
    Compute log-likelihood of child given parents.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    parents : List[int]
        Parent indices.
    child : int
        Child index.
    cardinalities : Dict[int, int]
        Variable cardinalities.

    Returns
    -------
    float
        Log-likelihood.
    """
    n_samples = data.shape[0]
    child_card = cardinalities.get(child, len(np.unique(data[:, child])))

    if len(parents) == 0:
        counts = np.bincount(data[:, child].astype(int), minlength=child_card)
        probs = (counts + 1e-10) / (n_samples + 1e-10 * child_card)
        ll = np.sum(counts * np.log(probs + 1e-10))
        return ll

    parent_cards = [cardinalities.get(p, len(np.unique(data[:, p]))) for p in parents]
    q_i = int(np.prod(parent_cards))

    parent_data = data[:, parents].astype(int)
    multipliers = np.ones(len(parents), dtype=int)
    for i in range(len(parents) - 1, 0, -1):
        multipliers[i-1] = multipliers[i] * parent_cards[i]
    parent_configs = np.dot(parent_data, multipliers)
    child_data = data[:, child].astype(int)

    ll = 0.0

    for j in range(q_i):
        mask = parent_configs == j
        n_j = mask.sum()

        if n_j == 0:
            continue

        child_subset = child_data[mask]
        counts = np.bincount(child_subset, minlength=child_card)
        probs = (counts + 1e-10) / (n_j + 1e-10 * child_card)
        ll += np.sum(counts * np.log(probs + 1e-10))

    return ll


def compute_structure_score(
    data: np.ndarray,
    structure: nx.DiGraph,
    cardinalities: dict[int, int],
    score_type: str = 'bdeu',
    equivalent_sample_size: float = 10.0,
) -> float:
    """
    Compute total score for a complete DAG structure.

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    structure : nx.DiGraph
        The DAG structure to score.
    cardinalities : Dict[int, int]
        Number of states for each variable.
    score_type : str, default='bdeu'
        Scoring function: 'bdeu', 'bic', or 'k2'.
    equivalent_sample_size : float, default=10.0
        ESS for BDeu score.

    Returns
    -------
    float
        Total score (sum of family scores).
    """
    score_funcs = {
        'bdeu': lambda d, p, c, card: bdeu_score(d, p, c, card, equivalent_sample_size),
        'bic': bic_score,
        'k2': k2_score,
    }

    if score_type not in score_funcs:
        raise ValueError(f"Unknown score type: {score_type}. Use 'bdeu', 'bic', or 'k2'.")

    score_func = score_funcs[score_type]
    total_score = 0.0

    # Score each node given its parents
    for node in structure.nodes():
        if node == 'Y':
            continue  # Skip target node or handle separately

        parents = list(structure.predecessors(node))
        # Filter out 'Y' from parents for scoring (handled separately)
        numeric_parents = [p for p in parents if isinstance(p, int)]

        if isinstance(node, int):
            total_score += score_func(data, numeric_parents, node, cardinalities)

    return total_score


def score_edge_addition(
    data: np.ndarray,
    current_parents: list[int],
    new_parent: int,
    child: int,
    cardinalities: dict[int, int],
    score_type: str = 'bdeu',
    equivalent_sample_size: float = 10.0,
) -> float:
    """
    Compute score delta for adding an edge.

    Parameters
    ----------
    data : np.ndarray
        Data matrix.
    current_parents : List[int]
        Current parent set.
    new_parent : int
        Proposed new parent.
    child : int
        Child node.
    cardinalities : Dict[int, int]
        Variable cardinalities.
    score_type : str, default='bdeu'
        Scoring function.
    equivalent_sample_size : float, default=10.0
        ESS for BDeu.

    Returns
    -------
    float
        Score improvement (positive = better).
    """
    score_funcs = {
        'bdeu': lambda d, p, c, card: bdeu_score(d, p, c, card, equivalent_sample_size),
        'bic': bic_score,
        'k2': k2_score,
    }

    score_func = score_funcs[score_type]

    current_score = score_func(data, current_parents, child, cardinalities)
    new_parents = current_parents + [new_parent]
    new_score = score_func(data, new_parents, child, cardinalities)

    return new_score - current_score
