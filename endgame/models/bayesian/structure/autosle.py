from __future__ import annotations

"""AutoSLE: Automatic Structure Learning Ensemble.

AutoSLE provides scalable structure learning for massive datasets
(10k+ variables) using a divide-and-conquer approach with ensemble
voting on edges.

Key features:
- Variable partitioning via spectral clustering
- Multiple solver backends (PC, FGES, Hill Climbing)
- Edge voting for robustness
- Parallelized cluster solving

References
----------
Chickering, D. M. (2002). Optimal Structure Identification With
Greedy Search. JMLR, 3.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import networkx as nx
import numpy as np

from endgame.models.bayesian.structure.learning import (
    greedy_hill_climbing,
)


class AutoSLE:
    """
    Scalable structure learning for massive variable sets.

    AutoSLE works by:
    1. Partitioning variables into manageable clusters
    2. Running multiple structure learning algorithms on each cluster
    3. Combining results via edge voting
    4. Learning inter-cluster edges
    5. Fusing into a global DAG

    Parameters
    ----------
    solvers : list[str], default=['pc', 'fges', 'hc']
        Base solvers to ensemble:
        - 'pc': PC-Stable (constraint-based, via pgmpy if available)
        - 'fges': Fast Greedy Equivalence Search (via causal-learn if available)
        - 'hc': Hill Climbing with restarts (built-in)
        - 'ges': Greedy Equivalence Search

    partition_method : {'spectral', 'correlation', 'random'}, default='spectral'
        How to partition variables into clusters.

    max_cluster_size : int, default=50
        Maximum variables per cluster.

    edge_threshold : float, default=0.5
        Minimum fraction of solvers that must agree on an edge.

    n_jobs : int, default=-1
        Parallelization for cluster solving. -1 uses all cores.

    random_state : int, optional
        Random seed for reproducibility.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    structure_ : nx.DiGraph
        Learned global DAG.

    edge_confidence_ : dict[tuple, float]
        Confidence score (agreement ratio) for each edge.

    cluster_assignments_ : np.ndarray
        Which cluster each variable was assigned to.

    Examples
    --------
    >>> from endgame.models.bayesian.structure import AutoSLE
    >>> sle = AutoSLE(max_cluster_size=30)
    >>> structure = sle.learn(data, variable_names=['x1', 'x2', ...])
    """

    def __init__(
        self,
        solvers: list[str] = None,
        partition_method: str = 'spectral',
        max_cluster_size: int = 50,
        edge_threshold: float = 0.5,
        n_jobs: int = -1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.solvers = solvers or ['hc']  # Default to built-in HC
        self.partition_method = partition_method
        self.max_cluster_size = max_cluster_size
        self.edge_threshold = edge_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.structure_: nx.DiGraph | None = None
        self.edge_confidence_: dict[tuple, float] | None = None
        self.cluster_assignments_: np.ndarray | None = None

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[AutoSLE] {message}")

    def _effective_n_jobs(self) -> int:
        """Get effective number of jobs."""
        import os
        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return self.n_jobs

    def learn(
        self,
        data: np.ndarray,
        variable_names: list[str] | None = None,
        cardinalities: dict[int, int] | None = None,
    ) -> nx.DiGraph:
        """
        Learn structure from data.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_variables). Should be discrete (integer).
        variable_names : list[str], optional
            Names for variables. Defaults to indices.
        cardinalities : dict[int, int], optional
            Number of states per variable. Auto-computed if not provided.

        Returns
        -------
        nx.DiGraph
            Learned directed acyclic graph.
        """
        n_samples, n_vars = data.shape
        self._log(f"Learning structure for {n_vars} variables, {n_samples} samples")

        # Auto-compute cardinalities if not provided
        if cardinalities is None:
            cardinalities = {
                i: len(np.unique(data[:, i])) for i in range(n_vars)
            }

        # Step 1: Partition variables
        self._log(f"Partitioning variables (method={self.partition_method})")
        clusters = self._partition(data)
        self.cluster_assignments_ = self._assignments_from_clusters(clusters, n_vars)
        self._log(f"Created {len(clusters)} clusters")

        # Step 2: Learn structure within each cluster (parallelized)
        self._log("Learning intra-cluster structures...")

        cluster_args = [
            (data[:, cluster], cluster, cardinalities)
            for cluster in clusters
        ]

        if self._effective_n_jobs() > 1 and len(clusters) > 1:
            # Parallel execution
            with ThreadPoolExecutor(max_workers=self._effective_n_jobs()) as executor:
                cluster_results = list(executor.map(self._solve_cluster, cluster_args))
        else:
            # Sequential execution
            cluster_results = [self._solve_cluster(args) for args in cluster_args]

        # Step 3: Ensemble within each cluster
        self._log("Ensembling solver results...")
        subgraphs = []
        for results in cluster_results:
            subgraph = self._ensemble_edges(results)
            subgraphs.append(subgraph)

        # Step 4: Learn inter-cluster edges
        self._log("Learning inter-cluster edges...")
        cluster_representatives = self._get_representatives(data, clusters)
        inter_edges = self._learn_inter_cluster(data, clusters, cluster_representatives)

        # Step 5: Fuse into global graph
        self._log("Fusing into global structure...")
        self.structure_ = self._fuse_graphs(subgraphs, inter_edges, clusters)

        # Relabel with names if provided
        if variable_names:
            self.structure_ = nx.relabel_nodes(
                self.structure_,
                {i: name for i, name in enumerate(variable_names)}
            )
            # Also update edge confidence keys
            if self.edge_confidence_:
                new_conf = {}
                for (u, v), conf in self.edge_confidence_.items():
                    u_name = variable_names[u] if isinstance(u, int) and u < len(variable_names) else u
                    v_name = variable_names[v] if isinstance(v, int) and v < len(variable_names) else v
                    new_conf[(u_name, v_name)] = conf
                self.edge_confidence_ = new_conf

        self._log(f"Final structure: {self.structure_.number_of_nodes()} nodes, "
                 f"{self.structure_.number_of_edges()} edges")

        return self.structure_

    def _partition(self, data: np.ndarray) -> list[list[int]]:
        """Partition variables into clusters."""
        n_vars = data.shape[1]

        if n_vars <= self.max_cluster_size:
            # No need to partition
            return [list(range(n_vars))]

        if self.partition_method == 'spectral':
            return self._spectral_partition(data)
        elif self.partition_method == 'correlation':
            return self._correlation_partition(data)
        elif self.partition_method == 'random':
            return self._random_partition(n_vars)
        else:
            raise ValueError(f"Unknown partition method: {self.partition_method}")

    def _spectral_partition(self, data: np.ndarray) -> list[list[int]]:
        """Partition using spectral clustering on correlation matrix."""
        try:
            from sklearn.cluster import SpectralClustering
        except ImportError:
            warnings.warn("sklearn not available, falling back to random partition")
            return self._random_partition(data.shape[1])

        n_vars = data.shape[1]
        n_clusters = max(1, n_vars // self.max_cluster_size)

        # Compute absolute correlation as affinity
        corr = np.abs(np.corrcoef(data.T))
        # Handle NaN (constant features)
        corr = np.nan_to_num(corr, nan=0.0)
        np.fill_diagonal(corr, 1.0)

        clustering = SpectralClustering(
            n_clusters=n_clusters,
            affinity='precomputed',
            random_state=self.random_state,
        ).fit(corr)

        clusters = [
            np.where(clustering.labels_ == i)[0].tolist()
            for i in range(n_clusters)
        ]

        # Filter empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        return clusters

    def _correlation_partition(self, data: np.ndarray) -> list[list[int]]:
        """Partition using hierarchical clustering on correlation."""
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import squareform
        except ImportError:
            warnings.warn("scipy not available, falling back to random partition")
            return self._random_partition(data.shape[1])

        n_vars = data.shape[1]

        # Compute correlation-based distance
        corr = np.corrcoef(data.T)
        corr = np.nan_to_num(corr, nan=0.0)
        dist = 1 - np.abs(corr)
        dist = (dist + dist.T) / 2  # Force symmetry for squareform
        np.fill_diagonal(dist, 0)

        # Hierarchical clustering
        condensed = squareform(dist)
        Z = linkage(condensed, method='average')

        # Cut to get desired number of clusters
        n_clusters = max(1, n_vars // self.max_cluster_size)
        labels = fcluster(Z, n_clusters, criterion='maxclust')

        clusters = [
            np.where(labels == i)[0].tolist()
            for i in range(1, n_clusters + 1)
        ]

        return [c for c in clusters if len(c) > 0]

    def _random_partition(self, n_vars: int) -> list[list[int]]:
        """Random partition of variables."""
        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(n_vars)

        clusters = []
        for i in range(0, n_vars, self.max_cluster_size):
            cluster = indices[i:i + self.max_cluster_size].tolist()
            if cluster:
                clusters.append(cluster)

        return clusters

    def _assignments_from_clusters(
        self,
        clusters: list[list[int]],
        n_vars: int,
    ) -> np.ndarray:
        """Convert cluster list to assignment array."""
        assignments = np.zeros(n_vars, dtype=int)
        for i, cluster in enumerate(clusters):
            for var in cluster:
                assignments[var] = i
        return assignments

    def _solve_cluster(
        self,
        args: tuple[np.ndarray, list[int], dict[int, int]],
    ) -> dict[str, nx.DiGraph]:
        """Run all solvers on a single cluster."""
        cluster_data, cluster_indices, full_cardinalities = args

        # Map cardinalities to local indices
        local_cards = {
            i: full_cardinalities.get(cluster_indices[i], len(np.unique(cluster_data[:, i])))
            for i in range(len(cluster_indices))
        }

        results = {}

        for solver_name in self.solvers:
            try:
                graph = self._run_solver(solver_name, cluster_data, local_cards)

                # Remap node indices to global indices
                graph = nx.relabel_nodes(
                    graph,
                    {i: cluster_indices[i] for i in range(len(cluster_indices))}
                )
                results[solver_name] = graph

            except Exception as e:
                # Log but don't fail - ensemble can handle missing solvers
                if self.verbose:
                    print(f"[AutoSLE] Solver {solver_name} failed: {e}")
                results[solver_name] = nx.DiGraph()

        return results

    def _run_solver(
        self,
        solver_name: str,
        data: np.ndarray,
        cardinalities: dict[int, int],
    ) -> nx.DiGraph:
        """Run a single structure learning solver."""
        if solver_name == 'hc':
            return greedy_hill_climbing(
                data,
                cardinalities,
                score_type='bdeu',
                max_parents=3,
                random_state=self.random_state,
            )

        elif solver_name == 'pc':
            try:
                import pandas as pd
                from pgmpy.estimators import PC
                from pgmpy.models import BayesianNetwork
            except ImportError:
                raise ImportError("PC solver requires pgmpy: pip install pgmpy")

            df = pd.DataFrame(data)
            pc = PC(df)
            model = pc.estimate()
            return nx.DiGraph(model.edges())

        elif solver_name in ('fges', 'ges'):
            try:
                from causallearn.search.ScoreBased.GES import ges
            except ImportError:
                raise ImportError("GES solver requires causal-learn: pip install causal-learn")

            record = ges(data)
            G = record['G']
            # Convert causal-learn graph to networkx
            graph = nx.DiGraph()
            for i in range(data.shape[1]):
                graph.add_node(i)
            # Parse adjacency from causal-learn format
            adj = G.graph
            for i in range(adj.shape[0]):
                for j in range(adj.shape[1]):
                    if adj[i, j] == 1 and adj[j, i] == -1:
                        graph.add_edge(i, j)
                    elif adj[i, j] == -1 and adj[j, i] == 1:
                        graph.add_edge(j, i)
            return graph

        else:
            raise ValueError(f"Unknown solver: {solver_name}")

    def _ensemble_edges(
        self,
        solver_results: dict[str, nx.DiGraph],
    ) -> nx.DiGraph:
        """Combine solver outputs via edge voting."""
        edge_votes: dict[tuple, int] = {}

        for solver_name, graph in solver_results.items():
            for u, v in graph.edges():
                edge = (u, v)
                edge_votes[edge] = edge_votes.get(edge, 0) + 1

        # Keep edges that pass threshold
        n_solvers = len([g for g in solver_results.values() if g.number_of_edges() > 0])
        if n_solvers == 0:
            n_solvers = 1

        final_graph = nx.DiGraph()

        # Add all nodes
        all_nodes = set()
        for graph in solver_results.values():
            all_nodes.update(graph.nodes())
        for node in all_nodes:
            final_graph.add_node(node)

        # Initialize edge confidence dict
        if self.edge_confidence_ is None:
            self.edge_confidence_ = {}

        for edge, votes in edge_votes.items():
            confidence = votes / n_solvers
            if confidence >= self.edge_threshold:
                # Check for cycles before adding
                final_graph.add_edge(*edge)
                if not nx.is_directed_acyclic_graph(final_graph):
                    final_graph.remove_edge(*edge)
                else:
                    self.edge_confidence_[edge] = confidence

        return final_graph

    def _get_representatives(
        self,
        data: np.ndarray,
        clusters: list[list[int]],
    ) -> list[int]:
        """Get representative variable for each cluster (highest variance)."""
        representatives = []

        for cluster in clusters:
            if len(cluster) == 0:
                continue

            cluster_data = data[:, cluster]
            variances = np.var(cluster_data, axis=0)
            best_idx = np.argmax(variances)
            representatives.append(cluster[best_idx])

        return representatives

    def _learn_inter_cluster(
        self,
        data: np.ndarray,
        clusters: list[list[int]],
        representatives: list[int],
    ) -> list[tuple[int, int]]:
        """Learn edges between clusters using representatives."""
        if len(clusters) <= 1:
            return []

        # Extract representative data
        rep_data = data[:, representatives]
        rep_cards = {
            i: len(np.unique(rep_data[:, i]))
            for i in range(len(representatives))
        }

        # Run structure learning on representatives
        try:
            rep_graph = greedy_hill_climbing(
                rep_data,
                rep_cards,
                score_type='bdeu',
                max_parents=2,
                random_state=self.random_state,
            )
        except Exception:
            return []

        # Map edges back to original indices
        inter_edges = []
        for u, v in rep_graph.edges():
            orig_u = representatives[u]
            orig_v = representatives[v]
            inter_edges.append((orig_u, orig_v))

        return inter_edges

    def _fuse_graphs(
        self,
        subgraphs: list[nx.DiGraph],
        inter_edges: list[tuple[int, int]],
        clusters: list[list[int]],
    ) -> nx.DiGraph:
        """Fuse subgraphs and inter-cluster edges into global DAG."""
        global_graph = nx.DiGraph()

        # Add all nodes from subgraphs
        for subgraph in subgraphs:
            global_graph.add_nodes_from(subgraph.nodes())
            global_graph.add_edges_from(subgraph.edges())

        # Add inter-cluster edges (check for cycles)
        for u, v in inter_edges:
            if u not in global_graph:
                global_graph.add_node(u)
            if v not in global_graph:
                global_graph.add_node(v)

            global_graph.add_edge(u, v)
            if not nx.is_directed_acyclic_graph(global_graph):
                global_graph.remove_edge(u, v)
                # Try reverse direction
                global_graph.add_edge(v, u)
                if not nx.is_directed_acyclic_graph(global_graph):
                    global_graph.remove_edge(v, u)

        return global_graph

    def get_edge_confidence(self, u: int, v: int) -> float:
        """
        Get confidence score for an edge.

        Parameters
        ----------
        u, v : int
            Source and target node indices.

        Returns
        -------
        float
            Confidence score (0-1), or 0 if edge doesn't exist.
        """
        if self.edge_confidence_ is None:
            return 0.0
        return self.edge_confidence_.get((u, v), 0.0)

    def get_highly_confident_edges(
        self,
        min_confidence: float = 0.8,
    ) -> list[tuple]:
        """
        Get edges with confidence above threshold.

        Parameters
        ----------
        min_confidence : float, default=0.8
            Minimum confidence score.

        Returns
        -------
        list[tuple]
            List of (source, target) edges.
        """
        if self.edge_confidence_ is None:
            return []

        return [
            edge for edge, conf in self.edge_confidence_.items()
            if conf >= min_confidence
        ]
