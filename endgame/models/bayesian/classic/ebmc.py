from __future__ import annotations

"""Efficient Bayesian Multivariate Classifier (EBMC).

EBMC is a Bayesian Network Classifier that performs automatic
feature selection by identifying the Markov Blanket of the target.

Key features:
- Greedy forward selection optimizing BDeu score
- Equivalence transformation to escape local optima
- Automatic Markov Blanket pruning

References
----------
Jaeger, M. (2003). Probabilistic classifiers and the concepts they recognize.
ICML.
"""

import warnings
from typing import Any

import networkx as nx
import numpy as np
from sklearn.utils.validation import check_is_fitted

from endgame.models.bayesian.base import (
    BaseBayesianClassifier,
    ConvergenceWarning,
)
from endgame.models.bayesian.structure.learning import (
    compute_mi_scores,
    get_markov_blanket,
)
from endgame.models.bayesian.structure.scores import (
    bdeu_score,
    bic_score,
    k2_score,
)


class EBMCClassifier(BaseBayesianClassifier):
    """
    Efficient Bayesian Multivariate Classifier with automatic feature selection.

    EBMC learns a Bayesian Network structure and then prunes features
    to those in the Markov Blanket of the target. This provides
    built-in feature selection while maintaining interpretability.

    Parameters
    ----------
    score : {'bdeu', 'bic', 'k2'}, default='bdeu'
        Scoring function for structure learning.

    equivalent_sample_size : float, default=10.0
        ESS for BDeu score. Lower = more aggressive pruning.

    max_parents : int, default=3
        Maximum parents per node (controls complexity).

    max_features : int | None, default=None
        Maximum features to select. None = no limit.

    convergence_threshold : float, default=1e-4
        Stop when score improvement falls below this.

    max_iter : int, default=100
        Maximum iterations for structure search.

    use_equivalence_transform : bool, default=True
        Whether to apply statistical equivalence transformation.

    smoothing : float, default=1.0
        Laplace smoothing for CPT estimation.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    selected_features_ : list[int]
        Indices of selected features (Markov Blanket).

    structure_ : nx.DiGraph
        Learned DAG structure.

    cpts_ : dict
        Conditional probability tables.

    current_score_ : float
        Final structure score.

    Examples
    --------
    >>> from endgame.models.bayesian import EBMCClassifier
    >>> clf = EBMCClassifier(max_parents=2)
    >>> clf.fit(X_train, y_train)
    >>> print(f"Selected {len(clf.selected_features_)} features")
    >>> clf.predict(X_test)
    """

    def __init__(
        self,
        score: str = 'bdeu',
        equivalent_sample_size: float = 10.0,
        max_parents: int = 3,
        max_features: int | None = None,
        convergence_threshold: float = 1e-4,
        max_iter: int = 100,
        use_equivalence_transform: bool = True,
        smoothing: float = 1.0,
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            smoothing=smoothing,
            max_cardinality=max_cardinality,
            auto_discretize=auto_discretize,
            discretizer_strategy=discretizer_strategy,
            discretizer_max_bins=discretizer_max_bins,
            random_state=random_state,
            verbose=verbose,
        )
        self.score = score
        self.equivalent_sample_size = equivalent_sample_size
        self.max_parents = max_parents
        self.max_features = max_features
        self.convergence_threshold = convergence_threshold
        self.max_iter = max_iter
        self.use_equivalence_transform = use_equivalence_transform

        # Set after fit
        self.selected_features_: list[int] | None = None
        self.current_score_: float | None = None
        self.cpts_: dict | None = None
        self.class_prior_: np.ndarray | None = None

    def _get_score_func(self):
        """Get the scoring function."""
        if self.score == 'bdeu':
            return lambda data, parents, child, cards: bdeu_score(
                data, parents, child, cards, self.equivalent_sample_size
            )
        elif self.score == 'bic':
            return bic_score
        elif self.score == 'k2':
            return k2_score
        else:
            raise ValueError(f"Unknown score: {self.score}")

    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """
        Learn structure via greedy forward selection + equivalence transform.

        Phase 1: Greedy forward selection of features
        Phase 2: Equivalence transformation (if enabled)
        Phase 3: Markov Blanket pruning
        """
        n_features = X.shape[1]
        score_func = self._get_score_func()

        # Combine X and y for scoring
        data = np.column_stack([X, y])
        target_idx = n_features

        # Extended cardinalities including target
        extended_cards = dict(self.cardinalities_)
        extended_cards[target_idx] = self.n_classes_

        # Phase 1: Greedy forward selection
        self._log("Phase 1: Greedy forward selection")
        current_graph = nx.DiGraph()
        current_graph.add_node('Y')

        # Start with MI-ranked features
        mi_scores = compute_mi_scores(X, y)
        feature_order = np.argsort(-mi_scores)

        self.current_score_ = self._compute_graph_score(
            current_graph, data, extended_cards, score_func
        )

        features_added = []

        for iteration in range(min(n_features, self.max_iter)):
            best_feature = None
            best_score = self.current_score_
            best_graph = None

            # Try adding each remaining feature
            for feat in feature_order:
                if feat in features_added:
                    continue

                if self.max_features and len(features_added) >= self.max_features:
                    break

                # Create candidate graph
                candidate = current_graph.copy()
                candidate.add_node(feat)
                candidate.add_edge('Y', feat)

                # Optionally add edges from existing features
                for existing in features_added:
                    if len(list(candidate.predecessors(feat))) < self.max_parents:
                        candidate.add_edge(existing, feat)

                # Score candidate
                candidate_score = self._compute_graph_score(
                    candidate, data, extended_cards, score_func
                )

                if candidate_score > best_score + self.convergence_threshold:
                    best_score = candidate_score
                    best_feature = feat
                    best_graph = candidate

            if best_feature is None:
                break

            current_graph = best_graph
            self.current_score_ = best_score
            features_added.append(best_feature)

            self._log(f"  Added feature {best_feature}, score: {best_score:.4f}")

        # Phase 2: Equivalence transformation
        if self.use_equivalence_transform:
            self._log("Phase 2: Equivalence transformation")
            current_graph, self.current_score_ = self._apply_equivalence_transform(
                current_graph, data, extended_cards, score_func
            )

        # Phase 3: Markov Blanket pruning
        self._log("Phase 3: Markov Blanket pruning")
        mb_nodes = get_markov_blanket(current_graph, 'Y')
        mb_nodes.add('Y')

        # Keep only MB nodes
        self.selected_features_ = sorted([n for n in mb_nodes if isinstance(n, int)])

        if len(self.selected_features_) == 0:
            # No features selected - fall back to top MI features
            self.selected_features_ = feature_order[:min(3, n_features)].tolist()
            for feat in self.selected_features_:
                if feat not in current_graph:
                    current_graph.add_node(feat)
                    current_graph.add_edge('Y', feat)

        # Compute feature importances
        self.feature_importances_ = np.zeros(n_features)
        for i, feat in enumerate(self.selected_features_):
            self.feature_importances_[feat] = mi_scores[feat]
        if self.feature_importances_.sum() > 0:
            self.feature_importances_ /= self.feature_importances_.sum()

        self._log(f"Selected {len(self.selected_features_)} features: {self.selected_features_}")

        # Return subgraph with selected features
        final_nodes = set(self.selected_features_) | {'Y'}
        final_graph = current_graph.subgraph(final_nodes).copy()

        return final_graph

    def _compute_graph_score(
        self,
        graph: nx.DiGraph,
        data: np.ndarray,
        cardinalities: dict,
        score_func,
    ) -> float:
        """Compute total score for a graph."""
        total = 0.0

        for node in graph.nodes():
            if node == 'Y':
                continue

            parents = list(graph.predecessors(node))
            # Convert 'Y' to target index
            parent_indices = []
            for p in parents:
                if p == 'Y':
                    parent_indices.append(data.shape[1] - 1)
                else:
                    parent_indices.append(p)

            total += score_func(data, parent_indices, node, cardinalities)

        return total

    def _apply_equivalence_transform(
        self,
        graph: nx.DiGraph,
        data: np.ndarray,
        cardinalities: dict,
        score_func,
    ) -> tuple[nx.DiGraph, float]:
        """
        Apply statistical equivalence transformation to find better structure.

        Uses edge reversals and additions that maintain equivalence class.
        """
        current_graph = graph.copy()
        current_score = self._compute_graph_score(
            current_graph, data, cardinalities, score_func
        )

        improved = True
        iterations = 0

        while improved and iterations < self.max_iter:
            improved = False
            iterations += 1

            # Try edge reversals
            edges = list(current_graph.edges())
            for u, v in edges:
                if u == 'Y' or v == 'Y':
                    continue  # Don't reverse edges involving class

                # Check if reversal is valid
                test_graph = current_graph.copy()
                test_graph.remove_edge(u, v)
                test_graph.add_edge(v, u)

                # Check max parents constraint
                if len(list(test_graph.predecessors(u))) > self.max_parents:
                    continue

                # Check DAG property
                if not nx.is_directed_acyclic_graph(test_graph):
                    continue

                test_score = self._compute_graph_score(
                    test_graph, data, cardinalities, score_func
                )

                if test_score > current_score + self.convergence_threshold:
                    current_graph = test_graph
                    current_score = test_score
                    improved = True
                    break

            # Try edge additions (covered edges)
            if not improved:
                nodes = [n for n in current_graph.nodes() if n != 'Y']
                for u in nodes:
                    for v in nodes:
                        if u == v or current_graph.has_edge(u, v):
                            continue

                        if len(list(current_graph.predecessors(v))) >= self.max_parents:
                            continue

                        test_graph = current_graph.copy()
                        test_graph.add_edge(u, v)

                        if not nx.is_directed_acyclic_graph(test_graph):
                            continue

                        test_score = self._compute_graph_score(
                            test_graph, data, cardinalities, score_func
                        )

                        if test_score > current_score + self.convergence_threshold:
                            current_graph = test_graph
                            current_score = test_score
                            improved = True
                            break
                    if improved:
                        break

        if iterations >= self.max_iter:
            warnings.warn(
                f"Equivalence transform did not converge in {self.max_iter} iterations",
                ConvergenceWarning
            )

        return current_graph, current_score

    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph,
    ) -> None:
        """Learn CPTs for the selected features."""
        self.cpts_ = {}

        # Class prior
        y_counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        self.class_prior_ = (y_counts + self.smoothing) / (
            len(y) + self.smoothing * self.n_classes_
        )

        # CPT for each selected feature
        for node in structure.nodes():
            if node == 'Y':
                continue

            parents = list(structure.predecessors(node))
            self.cpts_[node] = self._estimate_cpt(X, y, node, parents)

    def _estimate_cpt(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: int,
        parents: list,
    ) -> np.ndarray:
        """Estimate CPT with Laplace smoothing."""
        node_card = self.cardinalities_[node]

        has_class_parent = 'Y' in parents
        feature_parents = [p for p in parents if p != 'Y']

        if not parents:
            counts = np.bincount(X[:, node].astype(int), minlength=node_card)
            return (counts + self.smoothing) / (len(X) + self.smoothing * node_card)

        # Build shape
        parent_cards = []
        if has_class_parent:
            parent_cards.append(self.n_classes_)
        for p in feature_parents:
            parent_cards.append(self.cardinalities_[p])

        shape = [node_card] + parent_cards
        counts = np.zeros(shape)

        for i in range(X.shape[0]):
            # Clamp node value to valid range
            node_val = min(max(int(X[i, node]), 0), node_card - 1)

            idx = [node_val]
            if has_class_parent:
                idx.append(int(y[i]))
            for p in feature_parents:
                # Clamp parent value to valid range
                p_card = self.cardinalities_[p]
                p_val = min(max(int(X[i, p]), 0), p_card - 1)
                idx.append(p_val)

            counts[tuple(idx)] += 1

        parent_totals = counts.sum(axis=0, keepdims=True)
        cpt = (counts + self.smoothing) / (parent_totals + self.smoothing * node_card)

        return cpt

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using learned structure."""
        check_is_fitted(self, ['structure_', 'cpts_', 'class_prior_'])

        # Preprocess input (applies discretization if needed)
        X = self._preprocess_X(X)

        X = X.astype(int)
        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))

        # Add log class prior
        log_proba += np.log(self.class_prior_ + 1e-10)

        # Add conditionals for selected features only
        for node in self.selected_features_:
            if node not in self.cpts_:
                continue

            cpt = self.cpts_[node]
            parents = list(self.structure_.predecessors(node))

            has_class_parent = 'Y' in parents
            feature_parents = [p for p in parents if p != 'Y']

            for c_idx in range(self.n_classes_):
                for i in range(n_samples):
                    # Clamp node value to valid range
                    node_val = min(max(int(X[i, node]), 0), cpt.shape[0] - 1)

                    idx = [node_val]
                    if has_class_parent:
                        idx.append(c_idx)
                    for p in feature_parents:
                        # Clamp parent value to valid range
                        p_card = self.cardinalities_.get(p, 1)
                        p_val = min(max(int(X[i, p]), 0), p_card - 1)
                        idx.append(p_val)

                    try:
                        prob = cpt[tuple(idx)]
                        log_proba[i, c_idx] += np.log(prob + 1e-10)
                    except IndexError:
                        log_proba[i, c_idx] += np.log(1.0 / cpt.shape[0])

        # Normalize
        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)

        return proba

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get fitted state for serialization."""
        state = {
            'selected_features': self.selected_features_,
            'current_score': self.current_score_,
        }

        if self.cpts_ is not None:
            state['cpts'] = {str(k): v.tolist() for k, v in self.cpts_.items()}

        if self.class_prior_ is not None:
            state['class_prior'] = self.class_prior_.tolist()

        if self.feature_importances_ is not None:
            state['feature_importances'] = self.feature_importances_.tolist()

        return state

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state."""
        self.selected_features_ = state.get('selected_features', [])
        self.current_score_ = state.get('current_score')

        if 'cpts' in state:
            self.cpts_ = {int(k): np.array(v) for k, v in state['cpts'].items()}

        if 'class_prior' in state:
            self.class_prior_ = np.array(state['class_prior'])

        if 'feature_importances' in state:
            self.feature_importances_ = np.array(state['feature_importances'])
