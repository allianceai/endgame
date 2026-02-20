"""Multi-population genetic programming engine for symbolic regression.

Implements:
- Ramped half-and-half initialization
- Tournament selection
- Subtree crossover
- Point / subtree / hoist mutation
- Elitism + ring-topology migration between populations
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from endgame.models.symbolic._expression import (
    Binary,
    Constant,
    Node,
    Unary,
    Variable,
    complexity,
    enumerate_nodes,
    evaluate,
)

# ============================================================
# Helper: clone a node tree
# ============================================================

def clone_tree(node: Node) -> Node:
    """Deep-copy an expression tree."""
    return node.clone()


# ============================================================
# Random tree generation
# ============================================================

def random_tree(
    rng: np.random.Generator,
    n_features: int,
    binary_operators: list[str],
    unary_operators: list[str],
    max_depth: int,
    method: str = "grow",
) -> Node:
    """Generate a random expression tree.

    Parameters
    ----------
    method : "grow" or "full"
        "grow" allows early termination (mix of shapes).
        "full" always expands to max_depth.
    """
    if max_depth <= 0 or (method == "grow" and rng.random() < 0.3):
        # terminal: constant or variable
        if rng.random() < 0.5:
            return Constant(rng.standard_normal() * 2)
        else:
            return Variable(rng.integers(0, n_features))

    # Choose operator
    has_unary = len(unary_operators) > 0
    total = len(binary_operators) + (len(unary_operators) if has_unary else 0)
    if total == 0:
        return Constant(rng.standard_normal())

    idx = rng.integers(0, total)
    if idx < len(binary_operators):
        op = binary_operators[idx]
        left = random_tree(rng, n_features, binary_operators, unary_operators, max_depth - 1, method)
        right = random_tree(rng, n_features, binary_operators, unary_operators, max_depth - 1, method)
        return Binary(op, left, right)
    else:
        op = unary_operators[idx - len(binary_operators)]
        child = random_tree(rng, n_features, binary_operators, unary_operators, max_depth - 1, method)
        return Unary(op, child)


def ramped_half_and_half(
    rng: np.random.Generator,
    n_trees: int,
    n_features: int,
    binary_operators: list[str],
    unary_operators: list[str],
    min_depth: int = 1,
    max_depth: int = 4,
) -> list[Node]:
    """Generate a diverse initial population using ramped half-and-half."""
    trees = []
    depths = list(range(min_depth, max_depth + 1))
    for i in range(n_trees):
        d = depths[i % len(depths)]
        method = "full" if i % 2 == 0 else "grow"
        trees.append(random_tree(rng, n_features, binary_operators, unary_operators, d, method))
    return trees


# ============================================================
# Fitness evaluation
# ============================================================

def compute_fitness(
    tree: Node,
    X: np.ndarray,
    y: np.ndarray,
    loss_fn: Callable,
    parsimony: float = 0.0,
) -> float:
    """Evaluate fitness (lower is better).

    Returns loss + parsimony * complexity.
    """
    try:
        y_pred = evaluate(tree, X)
        if not np.all(np.isfinite(y_pred)):
            return 1e20
        loss = loss_fn(y, y_pred)
        if not np.isfinite(loss):
            return 1e20
        return loss + parsimony * complexity(tree)
    except Exception:
        return 1e20


# ============================================================
# Selection
# ============================================================

def tournament_select(
    population: list[Node],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    tournament_size: int = 5,
) -> Node:
    """Select one individual via tournament selection."""
    idx = rng.choice(len(population), size=min(tournament_size, len(population)), replace=False)
    best = idx[np.argmin(fitnesses[idx])]
    return clone_tree(population[best])


# ============================================================
# Crossover
# ============================================================

def subtree_crossover(
    parent1: Node,
    parent2: Node,
    rng: np.random.Generator,
    maxsize: int = 30,
) -> tuple[Node, Node]:
    """Swap random subtrees between two parents."""
    child1 = clone_tree(parent1)
    child2 = clone_tree(parent2)

    nodes1 = enumerate_nodes(child1)
    nodes2 = enumerate_nodes(child2)

    if len(nodes1) < 2 or len(nodes2) < 2:
        return child1, child2

    # Pick random non-root nodes
    idx1 = rng.integers(1, len(nodes1))
    idx2 = rng.integers(1, len(nodes2))

    node1, parent1_ref, attr1 = nodes1[idx1]
    node2, parent2_ref, attr2 = nodes2[idx2]

    # Swap
    if parent1_ref is not None and parent2_ref is not None:
        setattr(parent1_ref, attr1, node2)
        setattr(parent2_ref, attr2, node1)

    # Check size constraint
    if complexity(child1) > maxsize:
        child1 = clone_tree(parent1)
    if complexity(child2) > maxsize:
        child2 = clone_tree(parent2)

    return child1, child2


# ============================================================
# Mutation
# ============================================================

def point_mutation(
    tree: Node,
    rng: np.random.Generator,
    binary_operators: list[str],
    unary_operators: list[str],
    n_features: int,
) -> Node:
    """Mutate a single random node (change operator or value)."""
    tree = clone_tree(tree)
    nodes = enumerate_nodes(tree)
    idx = rng.integers(0, len(nodes))
    node, parent, attr = nodes[idx]

    if isinstance(node, Constant):
        # Perturb constant
        node.value += rng.standard_normal() * 0.5
    elif isinstance(node, Variable):
        node.index = rng.integers(0, n_features)
    elif isinstance(node, Unary) and len(unary_operators) > 0:
        node.op = unary_operators[rng.integers(0, len(unary_operators))]
    elif isinstance(node, Binary) and len(binary_operators) > 0:
        node.op = binary_operators[rng.integers(0, len(binary_operators))]

    return tree


def subtree_mutation(
    tree: Node,
    rng: np.random.Generator,
    binary_operators: list[str],
    unary_operators: list[str],
    n_features: int,
    maxsize: int = 30,
) -> Node:
    """Replace a random subtree with a new random tree."""
    tree = clone_tree(tree)
    nodes = enumerate_nodes(tree)

    if len(nodes) < 2:
        return random_tree(rng, n_features, binary_operators, unary_operators, 3, "grow")

    idx = rng.integers(1, len(nodes))
    node, parent, attr = nodes[idx]

    new_subtree = random_tree(rng, n_features, binary_operators, unary_operators, 3, "grow")
    if parent is not None:
        setattr(parent, attr, new_subtree)

    if complexity(tree) > maxsize:
        return clone_tree(tree)  # fallback: return unmutated copy
    return tree


def hoist_mutation(tree: Node, rng: np.random.Generator) -> Node:
    """Replace tree with one of its subtrees (reduces complexity)."""
    nodes = enumerate_nodes(tree)
    if len(nodes) < 2:
        return clone_tree(tree)
    # Pick a random non-leaf internal node
    internal = [(n, p, a) for n, p, a in nodes if isinstance(n, (Unary, Binary))]
    if not internal:
        return clone_tree(tree)
    idx = rng.integers(0, len(internal))
    return clone_tree(internal[idx][0])


def mutate(
    tree: Node,
    rng: np.random.Generator,
    binary_operators: list[str],
    unary_operators: list[str],
    n_features: int,
    maxsize: int = 30,
) -> Node:
    """Apply a randomly chosen mutation."""
    r = rng.random()
    if r < 0.4:
        return point_mutation(tree, rng, binary_operators, unary_operators, n_features)
    elif r < 0.75:
        return subtree_mutation(tree, rng, binary_operators, unary_operators, n_features, maxsize)
    else:
        return hoist_mutation(tree, rng)


# ============================================================
# Population evolution (single generation)
# ============================================================

def evolve_population(
    population: list[Node],
    fitnesses: np.ndarray,
    rng: np.random.Generator,
    binary_operators: list[str],
    unary_operators: list[str],
    n_features: int,
    tournament_size: int = 5,
    crossover_prob: float = 0.5,
    mutation_prob: float = 0.3,
    elite_frac: float = 0.1,
    maxsize: int = 30,
) -> list[Node]:
    """Produce next generation from current population.

    Uses elitism + tournament selection + crossover/mutation.
    """
    pop_size = len(population)
    new_pop: list[Node] = []

    # Elitism: keep top individuals
    n_elite = max(1, int(pop_size * elite_frac))
    elite_idx = np.argsort(fitnesses)[:n_elite]
    for i in elite_idx:
        new_pop.append(clone_tree(population[i]))

    # Fill rest
    while len(new_pop) < pop_size:
        r = rng.random()
        if r < crossover_prob:
            p1 = tournament_select(population, fitnesses, rng, tournament_size)
            p2 = tournament_select(population, fitnesses, rng, tournament_size)
            c1, c2 = subtree_crossover(p1, p2, rng, maxsize)
            new_pop.append(c1)
            if len(new_pop) < pop_size:
                new_pop.append(c2)
        elif r < crossover_prob + mutation_prob:
            parent = tournament_select(population, fitnesses, rng, tournament_size)
            child = mutate(parent, rng, binary_operators, unary_operators, n_features, maxsize)
            new_pop.append(child)
        else:
            # Reproduction (copy selected individual)
            parent = tournament_select(population, fitnesses, rng, tournament_size)
            new_pop.append(parent)

    return new_pop[:pop_size]


# ============================================================
# Migration (ring topology)
# ============================================================

def migrate(
    populations: list[list[Node]],
    all_fitnesses: list[np.ndarray],
    rng: np.random.Generator,
    n_migrants: int = 2,
) -> None:
    """Ring-topology migration: send best individuals to next population."""
    n_pops = len(populations)
    if n_pops < 2:
        return

    migrants = []
    for i in range(n_pops):
        fit = all_fitnesses[i]
        best_idx = np.argsort(fit)[:n_migrants]
        migrants.append([clone_tree(populations[i][j]) for j in best_idx])

    for i in range(n_pops):
        source = (i - 1) % n_pops
        # Replace worst individuals
        worst_idx = np.argsort(all_fitnesses[i])[-n_migrants:]
        for j, w in enumerate(worst_idx):
            if j < len(migrants[source]):
                populations[i][w] = migrants[source][j]
