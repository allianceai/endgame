"""Tree Augmented Naive Bayes (TAN) Classifier.

TAN relaxes the Naive Bayes independence assumption by allowing
features to form a tree structure in addition to having the
class as a common parent.

References
----------
Friedman, N., Geiger, D., & Goldszmidt, M. (1997).
Bayesian Network Classifiers. Machine Learning, 29(2-3).
"""

from typing import Any

import networkx as nx
import numpy as np
from sklearn.utils.validation import check_is_fitted

from endgame.models.bayesian.base import (
    BaseBayesianClassifier,
)
from endgame.models.bayesian.structure.learning import (
    build_tan_structure,
    compute_mi_scores,
)


class TANClassifier(BaseBayesianClassifier):
    """
    Tree Augmented Naive Bayes classifier.

    TAN extends Naive Bayes by allowing features to have one additional
    parent from other features, forming a tree structure. This captures
    pairwise feature dependencies while remaining computationally tractable.

    Parameters
    ----------
    smoothing : float, default=1.0
        Laplace smoothing parameter (alpha).
        Use 0 for MLE, 1 for add-one smoothing.

    root_selection : {'max_mi', 'random', int}, default='max_mi'
        How to select the root of the tree structure.
        - 'max_mi': Feature with highest MI with target
        - 'random': Random selection (for ensembling)
        - int: Specific feature index

    missing_values : {'error', 'marginalize'}, default='error'
        Strategy for missing values during predict.

    auto_discretize : bool, default=True
        If True, automatically discretize continuous features.

    discretizer_strategy : str, default='mdlp'
        Discretization strategy: 'mdlp', 'equal_width', 'equal_freq', 'kmeans'.

    discretizer_max_bins : int, default=10
        Maximum bins per feature when auto-discretizing.

    n_jobs : int, default=1
        Parallelization for MI computation. -1 uses all cores.

    random_state : int, optional
        Random seed for reproducibility.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    structure_ : nx.DiGraph
        Learned TAN structure after fit().

    cpts_ : dict[int, np.ndarray]
        Conditional probability tables.
        cpts_[i] has shape that depends on parent configuration.

    class_prior_ : np.ndarray
        Prior class probabilities P(Y).

    feature_importances_ : np.ndarray
        Mutual information I(X_i; Y) normalized.

    Examples
    --------
    >>> from endgame.models.bayesian import TANClassifier
    >>> clf = TANClassifier(smoothing=1.0)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)
    """

    def __init__(
        self,
        smoothing: float = 1.0,
        root_selection: str | int = 'max_mi',
        missing_values: str = 'error',
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        n_jobs: int = 1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            smoothing=smoothing,
            missing_values=missing_values,
            max_cardinality=max_cardinality,
            auto_discretize=auto_discretize,
            discretizer_strategy=discretizer_strategy,
            discretizer_max_bins=discretizer_max_bins,
            random_state=random_state,
            verbose=verbose,
        )
        self.root_selection = root_selection
        self.n_jobs = n_jobs

        # Will be set during fit
        self.cpts_: dict[int, np.ndarray] | None = None
        self.class_prior_: np.ndarray | None = None

    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """
        Learn TAN structure using Chow-Liu algorithm on CMI.

        Steps:
        1. Compute CMI matrix I(X_i; X_j | Y) for all pairs
        2. Build maximum spanning tree using Chow-Liu
        3. Orient edges away from root
        4. Add Y -> X_i edges for all features
        """
        # Set random state for root selection
        if self.root_selection == 'random':
            np.random.seed(self.random_state)

        structure = build_tan_structure(
            X, y,
            root_selection=self.root_selection,
            n_jobs=self.n_jobs,
        )

        # Compute feature importances from MI
        mi_scores = compute_mi_scores(X, y, n_jobs=self.n_jobs)
        self.feature_importances_ = mi_scores / (mi_scores.sum() + 1e-10)

        return structure

    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph,
    ) -> None:
        """
        Learn CPTs using MLE with Laplace smoothing.

        For each feature X_i with parents Pa_i:
        P(X_i = k | Pa_i = j) = (N_ijk + alpha) / (N_ij + alpha * |X_i|)
        """
        self.cpts_ = {}

        # Class prior P(Y)
        y_counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        self.class_prior_ = (y_counts + self.smoothing) / (
            len(y) + self.smoothing * self.n_classes_
        )

        # CPT for each feature
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
        parents: list[int | str],
    ) -> np.ndarray:
        """
        Estimate CPT for a single node given its parents.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix.
        y : np.ndarray
            Target values.
        node : int
            Feature index.
        parents : List[Union[int, str]]
            Parent indices (may include 'Y').

        Returns
        -------
        np.ndarray
            CPT array.
        """
        node_card = self.cardinalities_[node]

        # Separate Y from feature parents
        has_class_parent = 'Y' in parents
        feature_parents = [p for p in parents if p != 'Y']

        if not parents:
            # No parents - just marginal
            counts = np.bincount(X[:, node].astype(int), minlength=node_card)
            cpt = (counts + self.smoothing) / (len(X) + self.smoothing * node_card)
            return cpt

        # Build parent configuration shape
        parent_cards = []
        if has_class_parent:
            parent_cards.append(self.n_classes_)
        for p in feature_parents:
            parent_cards.append(self.cardinalities_[p])

        # Initialize counts array
        # Shape: (node_card, *parent_cards)
        shape = [node_card] + parent_cards
        counts = np.zeros(shape)

        # Count occurrences
        n_samples = X.shape[0]
        for i in range(n_samples):
            # Clamp node value to valid range
            node_val = min(int(X[i, node]), node_card - 1)
            node_val = max(node_val, 0)

            # Build parent config index
            parent_idx = []
            if has_class_parent:
                parent_idx.append(int(y[i]))
            for p in feature_parents:
                # Clamp parent value to valid range
                p_card = self.cardinalities_[p]
                p_val = min(int(X[i, p]), p_card - 1)
                p_val = max(p_val, 0)
                parent_idx.append(p_val)

            idx = tuple([node_val] + parent_idx)
            counts[idx] += 1

        # Smoothed probability
        # Sum over node values for each parent config
        parent_totals = counts.sum(axis=0, keepdims=True)
        cpt = (counts + self.smoothing) / (parent_totals + self.smoothing * node_card)

        return cpt

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities using TAN structure.

        For each class c:
        P(Y=c|X) ∝ P(Y=c) * ∏_i P(X_i | Pa(X_i), Y=c)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['structure_', 'cpts_', 'class_prior_'])

        # Preprocess input (applies discretization if needed)
        X = self._preprocess_X(X)

        # Convert to int
        X = X.astype(int)

        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))

        # Add log class prior
        log_proba += np.log(self.class_prior_ + 1e-10)

        # Add log conditional probabilities for each feature
        for node in range(self.n_features_in_):
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

                    # Build index into CPT
                    idx = [node_val]
                    if has_class_parent:
                        idx.append(c_idx)
                    for p in feature_parents:
                        # Clamp parent value to valid range
                        p_card = self.cardinalities_.get(p, cpt.shape[len(idx)])
                        p_val = min(max(int(X[i, p]), 0), p_card - 1)
                        idx.append(p_val)

                    try:
                        prob = cpt[tuple(idx)]
                        log_proba[i, c_idx] += np.log(prob + 1e-10)
                    except IndexError:
                        # Fallback - use uniform
                        log_proba[i, c_idx] += np.log(1.0 / cpt.shape[0])

        # Normalize to probabilities
        log_proba -= log_proba.max(axis=1, keepdims=True)  # For numerical stability
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)

        return proba

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get fitted state for serialization."""
        state = {}

        if self.cpts_ is not None:
            state['cpts'] = {
                str(k): v.tolist() for k, v in self.cpts_.items()
            }

        if self.class_prior_ is not None:
            state['class_prior'] = self.class_prior_.tolist()

        if self.feature_importances_ is not None:
            state['feature_importances'] = self.feature_importances_.tolist()

        return state

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state from serialization."""
        if 'cpts' in state:
            self.cpts_ = {
                int(k): np.array(v) for k, v in state['cpts'].items()
            }

        if 'class_prior' in state:
            self.class_prior_ = np.array(state['class_prior'])

        if 'feature_importances' in state:
            self.feature_importances_ = np.array(state['feature_importances'])
