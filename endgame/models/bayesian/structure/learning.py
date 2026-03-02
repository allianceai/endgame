from __future__ import annotations

"""Structure learning algorithms for Bayesian Network Classifiers.

This module provides:
- Mutual information computation (MI, CMI)
- Chow-Liu tree algorithm for TAN
- K-Dependence Bayes structure construction
- Hill climbing for general structure learning
"""

from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np

from endgame.models.bayesian.structure.scores import (
    bdeu_score,
    bic_score,
    k2_score,
)

# =============================================================================
# Mutual Information Computation
# =============================================================================


def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_idx: int,
) -> float:
    """
    Compute mutual information I(X_i; Y).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values.
    feature_idx : int
        Index of the feature.

    Returns
    -------
    float
        Mutual information in nats.
    """
    x = X[:, feature_idx].astype(int)
    y = y.astype(int)

    n_samples = len(x)

    # Count joint and marginal probabilities
    x_values = np.unique(x)
    y_values = np.unique(y)

    # Marginal counts
    x_counts = np.bincount(x, minlength=max(x_values) + 1)
    y_counts = np.bincount(y, minlength=max(y_values) + 1)

    mi = 0.0

    for xi in x_values:
        for yi in y_values:
            # Joint count
            joint_mask = (x == xi) & (y == yi)
            n_xy = joint_mask.sum()

            if n_xy == 0:
                continue

            # P(x,y), P(x), P(y)
            p_xy = n_xy / n_samples
            p_x = x_counts[xi] / n_samples
            p_y = y_counts[yi] / n_samples

            if p_x > 0 and p_y > 0:
                mi += p_xy * np.log(p_xy / (p_x * p_y))

    return max(0.0, mi)  # MI is non-negative


def compute_conditional_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    feature_i: int,
    feature_j: int,
) -> float:
    """
    Compute conditional mutual information I(X_i; X_j | Y).

    This measures how much information X_i and X_j share when
    the class label Y is known.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values.
    feature_i : int
        Index of first feature.
    feature_j : int
        Index of second feature.

    Returns
    -------
    float
        Conditional mutual information in nats.
    """
    xi = X[:, feature_i].astype(int)
    xj = X[:, feature_j].astype(int)
    y = y.astype(int)

    n_samples = len(xi)

    xi_values = np.unique(xi)
    xj_values = np.unique(xj)
    y_values = np.unique(y)

    cmi = 0.0

    for yv in y_values:
        y_mask = y == yv
        n_y = y_mask.sum()

        if n_y == 0:
            continue

        p_y = n_y / n_samples

        xi_y = xi[y_mask]
        xj_y = xj[y_mask]

        # Marginal counts given Y=yv
        xi_counts_y = np.bincount(xi_y, minlength=max(xi_values) + 1)
        xj_counts_y = np.bincount(xj_y, minlength=max(xj_values) + 1)

        for xiv in xi_values:
            for xjv in xj_values:
                # Joint count given Y
                joint_mask = (xi_y == xiv) & (xj_y == xjv)
                n_ij_y = joint_mask.sum()

                if n_ij_y == 0:
                    continue

                # P(xi, xj | y)
                p_ij_y = n_ij_y / n_y

                # P(xi | y), P(xj | y)
                p_i_y = xi_counts_y[xiv] / n_y if n_y > 0 else 0
                p_j_y = xj_counts_y[xjv] / n_y if n_y > 0 else 0

                if p_i_y > 0 and p_j_y > 0:
                    # Weight by P(y) for the overall CMI
                    cmi += p_y * p_ij_y * np.log(p_ij_y / (p_i_y * p_j_y))

    return max(0.0, cmi)


def compute_cmi_matrix(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute pairwise conditional mutual information matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features).
    y : np.ndarray
        Target values.
    n_jobs : int, default=1
        Number of parallel jobs. -1 uses all cores.

    Returns
    -------
    np.ndarray
        CMI matrix of shape (n_features, n_features).
        cmi_matrix[i, j] = I(X_i; X_j | Y)
    """
    n_features = X.shape[1]
    cmi_matrix = np.zeros((n_features, n_features))

    # Compute pairs
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]

    def compute_pair(pair):
        i, j = pair
        return i, j, compute_conditional_mutual_information(X, y, i, j)

    if n_jobs == 1:
        # Sequential computation
        for i, j in pairs:
            cmi = compute_conditional_mutual_information(X, y, i, j)
            cmi_matrix[i, j] = cmi
            cmi_matrix[j, i] = cmi
    else:
        # Parallel computation
        import os
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            results = list(executor.map(compute_pair, pairs))

        for i, j, cmi in results:
            cmi_matrix[i, j] = cmi
            cmi_matrix[j, i] = cmi

    return cmi_matrix


def compute_mi_scores(
    X: np.ndarray,
    y: np.ndarray,
    n_jobs: int = 1,
) -> np.ndarray:
    """
    Compute mutual information I(X_i; Y) for all features.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    n_jobs : int, default=1
        Number of parallel jobs.

    Returns
    -------
    np.ndarray
        MI scores of shape (n_features,).
    """
    n_features = X.shape[1]

    def compute_mi(idx):
        return compute_mutual_information(X, y, idx)

    if n_jobs == 1:
        mi_scores = np.array([compute_mi(i) for i in range(n_features)])
    else:
        import os
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            mi_scores = np.array(list(executor.map(compute_mi, range(n_features))))

    return mi_scores


# =============================================================================
# Chow-Liu Tree Algorithm
# =============================================================================


def chow_liu_tree(
    cmi_matrix: np.ndarray,
) -> nx.Graph:
    """
    Build maximum spanning tree using Chow-Liu algorithm.

    This creates an undirected tree structure by finding the maximum
    weight spanning tree where weights are conditional mutual information.

    Parameters
    ----------
    cmi_matrix : np.ndarray
        Pairwise CMI matrix of shape (n_features, n_features).

    Returns
    -------
    nx.Graph
        Undirected maximum spanning tree.
    """
    n_features = cmi_matrix.shape[0]

    # Create complete graph with CMI weights
    G = nx.Graph()
    for i in range(n_features):
        G.add_node(i)
        for j in range(i + 1, n_features):
            if cmi_matrix[i, j] > 0:
                G.add_edge(i, j, weight=cmi_matrix[i, j])

    # Find maximum spanning tree (Prim's algorithm on negative weights)
    # NetworkX has maximum_spanning_tree
    if len(G.edges()) > 0:
        mst = nx.maximum_spanning_tree(G, weight='weight')
    else:
        # No edges - return empty graph
        mst = nx.Graph()
        for i in range(n_features):
            mst.add_node(i)

    return mst


def orient_tree(
    tree: nx.Graph,
    root: int,
) -> nx.DiGraph:
    """
    Orient an undirected tree by directing edges away from root.

    Parameters
    ----------
    tree : nx.Graph
        Undirected tree.
    root : int
        Root node to orient away from.

    Returns
    -------
    nx.DiGraph
        Directed tree with edges pointing away from root.
    """
    directed = nx.DiGraph()

    if root not in tree:
        # Root not in tree, add all nodes
        for node in tree.nodes():
            directed.add_node(node)
        return directed

    # BFS from root to orient edges
    visited = {root}
    queue = [root]

    while queue:
        current = queue.pop(0)
        for neighbor in tree.neighbors(current):
            if neighbor not in visited:
                directed.add_edge(current, neighbor)
                visited.add(neighbor)
                queue.append(neighbor)

    # Add isolated nodes
    for node in tree.nodes():
        if node not in directed:
            directed.add_node(node)

    return directed


def build_tan_structure(
    X: np.ndarray,
    y: np.ndarray,
    root_selection: str | int = 'max_mi',
    n_jobs: int = 1,
) -> nx.DiGraph:
    """
    Build Tree Augmented Naive Bayes structure.

    TAN structure:
    1. All features have Y as parent (Naive Bayes edges)
    2. Features form a tree structure based on CMI

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    root_selection : str or int, default='max_mi'
        How to select tree root:
        - 'max_mi': Feature with highest MI with target
        - 'random': Random selection
        - int: Specific feature index
    n_jobs : int, default=1
        Parallelization for CMI computation.

    Returns
    -------
    nx.DiGraph
        TAN structure with 'Y' as class node.
    """
    n_features = X.shape[1]

    # Compute CMI matrix
    cmi_matrix = compute_cmi_matrix(X, y, n_jobs=n_jobs)

    # Build Chow-Liu tree
    tree = chow_liu_tree(cmi_matrix)

    # Select root
    if root_selection == 'max_mi':
        mi_scores = compute_mi_scores(X, y, n_jobs=n_jobs)
        root = int(np.argmax(mi_scores))
    elif root_selection == 'random':
        root = np.random.randint(0, n_features)
    elif isinstance(root_selection, int):
        root = root_selection
    else:
        raise ValueError(f"Unknown root_selection: {root_selection}")

    # Orient tree away from root
    directed_tree = orient_tree(tree, root)

    # Add edges from Y to all features (Naive Bayes structure)
    structure = nx.DiGraph()
    structure.add_node('Y')

    for node in range(n_features):
        structure.add_node(node)
        structure.add_edge('Y', node)

    # Add tree edges
    for u, v in directed_tree.edges():
        structure.add_edge(u, v)

    return structure


# =============================================================================
# K-Dependence Bayes Structure
# =============================================================================


def build_kdb_structure(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 2,
    n_jobs: int = 1,
) -> nx.DiGraph:
    """
    Build K-Dependence Bayes structure.

    KDB allows each feature to have at most K feature parents
    (plus the class Y as a parent).

    Algorithm:
    1. Rank features by MI with Y
    2. For each feature in rank order:
       - Add edges from Y and up to K higher-ranked features
       - Select parents that maximize CMI

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target values.
    k : int, default=2
        Maximum number of feature parents.
    n_jobs : int, default=1
        Parallelization.

    Returns
    -------
    nx.DiGraph
        KDB structure.
    """
    n_features = X.shape[1]

    # Compute MI and CMI
    mi_scores = compute_mi_scores(X, y, n_jobs=n_jobs)
    cmi_matrix = compute_cmi_matrix(X, y, n_jobs=n_jobs)

    # Rank features by MI (descending)
    feature_order = np.argsort(-mi_scores)

    structure = nx.DiGraph()
    structure.add_node('Y')

    for rank, feature in enumerate(feature_order):
        structure.add_node(feature)
        structure.add_edge('Y', feature)

        # Select up to K parents from higher-ranked features
        if rank > 0 and k > 0:
            candidates = feature_order[:rank]  # Higher-ranked features

            # Score each candidate by CMI with current feature
            candidate_scores = [(c, cmi_matrix[c, feature]) for c in candidates]
            candidate_scores.sort(key=lambda x: x[1], reverse=True)

            # Add top K parents
            for parent, _ in candidate_scores[:k]:
                structure.add_edge(parent, feature)

    return structure


# =============================================================================
# Hill Climbing Structure Learning
# =============================================================================


def greedy_hill_climbing(
    data: np.ndarray,
    cardinalities: dict[int, int],
    score_type: str = 'bdeu',
    equivalent_sample_size: float = 10.0,
    max_parents: int = 3,
    max_iter: int = 1000,
    convergence_threshold: float = 1e-4,
    tabu_length: int = 10,
    random_restarts: int = 0,
    random_state: int | None = None,
) -> nx.DiGraph:
    """
    Learn DAG structure using hill climbing with optional restarts.

    Operations considered:
    - Add edge
    - Remove edge
    - Reverse edge

    Parameters
    ----------
    data : np.ndarray
        Data matrix of shape (n_samples, n_variables).
    cardinalities : Dict[int, int]
        Number of states for each variable.
    score_type : str, default='bdeu'
        Scoring function: 'bdeu', 'bic', 'k2'.
    equivalent_sample_size : float, default=10.0
        ESS for BDeu.
    max_parents : int, default=3
        Maximum parents per node.
    max_iter : int, default=1000
        Maximum iterations.
    convergence_threshold : float, default=1e-4
        Stop when improvement falls below this.
    tabu_length : int, default=10
        Length of tabu list (0 to disable).
    random_restarts : int, default=0
        Number of random restarts.
    random_state : int, optional
        Random seed.

    Returns
    -------
    nx.DiGraph
        Learned DAG structure.
    """
    rng = np.random.RandomState(random_state)
    n_vars = data.shape[1]

    score_funcs = {
        'bdeu': lambda d, p, c, card: bdeu_score(d, p, c, card, equivalent_sample_size),
        'bic': bic_score,
        'k2': k2_score,
    }
    score_func = score_funcs[score_type]

    def compute_score(graph):
        """Compute total score for a graph."""
        total = 0.0
        for node in range(n_vars):
            parents = list(graph.predecessors(node))
            total += score_func(data, parents, node, cardinalities)
        return total

    def get_neighbors(graph, tabu_set):
        """Generate valid neighbor structures."""
        neighbors = []

        for i in range(n_vars):
            for j in range(n_vars):
                if i == j:
                    continue

                # Add edge i -> j
                if not graph.has_edge(i, j) and (('add', i, j) not in tabu_set):
                    if len(list(graph.predecessors(j))) < max_parents:
                        new_graph = graph.copy()
                        new_graph.add_edge(i, j)
                        if nx.is_directed_acyclic_graph(new_graph):
                            neighbors.append((new_graph, ('add', i, j)))

                # Remove edge i -> j
                if graph.has_edge(i, j) and (('remove', i, j) not in tabu_set):
                    new_graph = graph.copy()
                    new_graph.remove_edge(i, j)
                    neighbors.append((new_graph, ('remove', i, j)))

                # Reverse edge i -> j to j -> i
                if graph.has_edge(i, j) and (('reverse', i, j) not in tabu_set):
                    if len(list(graph.predecessors(i))) < max_parents:
                        new_graph = graph.copy()
                        new_graph.remove_edge(i, j)
                        new_graph.add_edge(j, i)
                        if nx.is_directed_acyclic_graph(new_graph):
                            neighbors.append((new_graph, ('reverse', i, j)))

        return neighbors

    def hill_climb(initial_graph):
        """Single hill climbing run."""
        current_graph = initial_graph.copy()
        current_score = compute_score(current_graph)

        tabu_list = []

        for iteration in range(max_iter):
            tabu_set = set(tabu_list)
            neighbors = get_neighbors(current_graph, tabu_set)

            if not neighbors:
                break

            # Find best neighbor
            best_neighbor = None
            best_score = current_score
            best_operation = None

            for neighbor_graph, operation in neighbors:
                neighbor_score = compute_score(neighbor_graph)
                if neighbor_score > best_score:
                    best_score = neighbor_score
                    best_neighbor = neighbor_graph
                    best_operation = operation

            # Check for improvement
            if best_neighbor is None or best_score - current_score < convergence_threshold:
                break

            # Update
            current_graph = best_neighbor
            current_score = best_score

            # Update tabu list
            if tabu_length > 0 and best_operation:
                # Add reverse operation to tabu
                op_type, i, j = best_operation
                if op_type == 'add':
                    tabu_list.append(('remove', i, j))
                elif op_type == 'remove':
                    tabu_list.append(('add', i, j))
                elif op_type == 'reverse':
                    tabu_list.append(('reverse', j, i))

                if len(tabu_list) > tabu_length:
                    tabu_list.pop(0)

        return current_graph, current_score

    # Initial empty graph
    best_graph = nx.DiGraph()
    for i in range(n_vars):
        best_graph.add_node(i)
    best_score = compute_score(best_graph)

    # Hill climb from empty graph
    result_graph, result_score = hill_climb(best_graph)
    if result_score > best_score:
        best_graph = result_graph
        best_score = result_score

    # Random restarts
    for _ in range(random_restarts):
        # Random starting graph
        random_graph = nx.DiGraph()
        for i in range(n_vars):
            random_graph.add_node(i)

        # Add random edges
        n_edges = rng.randint(0, n_vars * 2)
        for _ in range(n_edges):
            i, j = rng.randint(0, n_vars, size=2)
            if i != j and not random_graph.has_edge(i, j):
                if len(list(random_graph.predecessors(j))) < max_parents:
                    random_graph.add_edge(i, j)
                    if not nx.is_directed_acyclic_graph(random_graph):
                        random_graph.remove_edge(i, j)

        result_graph, result_score = hill_climb(random_graph)
        if result_score > best_score:
            best_graph = result_graph
            best_score = result_score

    return best_graph


def get_markov_blanket(
    graph: nx.DiGraph,
    target: str | int,
) -> set:
    """
    Get Markov Blanket of a node.

    MB(X) = Parents(X) ∪ Children(X) ∪ Parents(Children(X))

    Parameters
    ----------
    graph : nx.DiGraph
        DAG structure.
    target : str or int
        Target node.

    Returns
    -------
    set
        Nodes in the Markov Blanket.
    """
    if target not in graph:
        return set()

    mb = set()

    # Parents
    mb.update(graph.predecessors(target))

    # Children
    children = set(graph.successors(target))
    mb.update(children)

    # Co-parents
    for child in children:
        mb.update(graph.predecessors(child))

    mb.discard(target)
    return mb
