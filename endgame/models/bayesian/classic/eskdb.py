"""Ensemble of Selective K-Dependence Bayes (ESKDB) Classifier.

ESKDB is a state-of-the-art ensemble method for Bayesian Network
Classification. It combines multiple KDB classifiers with diversity
through Stochastic Attribute Ordering (SAO) and/or bootstrapping.

Key features:
- K-Dependence structure (generalizes NB, TAN)
- Hierarchical Dirichlet Process smoothing
- Ensemble diversity via SAO or bootstrap

References
----------
Webb, G. I., Boughton, J. R., & Wang, Z. (2005).
Not So Naive Bayes: Aggregating One-Dependence Estimators.
Machine Learning, 58(1).
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import networkx as nx
import numpy as np
from sklearn.utils.validation import check_is_fitted, check_X_y

from endgame.models.bayesian.base import (
    BaseBayesianClassifier,
)
from endgame.models.bayesian.structure.learning import (
    compute_mi_scores,
)


class KDBClassifier(BaseBayesianClassifier):
    """
    K-Dependence Bayes Classifier.

    KDB allows each feature to have at most K feature parents
    plus the class as a parent. This generalizes:
    - K=0: Naive Bayes
    - K=1: One-Dependence Estimator (AODE)
    - K>=n_features: Unrestricted (but computationally expensive)

    Parameters
    ----------
    k : int, default=2
        Maximum number of feature parents per node.

    smoothing : {'laplace', 'hdp'}, default='laplace'
        Smoothing method:
        - 'laplace': Standard add-alpha smoothing
        - 'hdp': Hierarchical Dirichlet Process

    smoothing_alpha : float, default=1.0
        Smoothing parameter (for Laplace).

    attribute_order : list[int] | None, default=None
        Custom attribute ordering. If None, uses MI ranking.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        k: int = 2,
        smoothing: str = 'laplace',
        smoothing_alpha: float = 1.0,
        attribute_order: list[int] | None = None,
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            smoothing=smoothing_alpha,
            max_cardinality=max_cardinality,
            auto_discretize=auto_discretize,
            discretizer_strategy=discretizer_strategy,
            discretizer_max_bins=discretizer_max_bins,
            random_state=random_state,
            verbose=verbose,
        )
        self.k = k
        self.smoothing_method = smoothing
        self.smoothing_alpha = smoothing_alpha
        self.attribute_order = attribute_order

        self.cpts_: dict | None = None
        self.class_prior_: np.ndarray | None = None
        self.mi_scores_: np.ndarray | None = None

    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """Build KDB structure with at most K feature parents."""
        n_features = X.shape[1]

        # Compute MI scores
        self.mi_scores_ = compute_mi_scores(X, y)

        # Get attribute order
        if self.attribute_order is not None:
            feature_order = np.array(self.attribute_order)
        else:
            feature_order = np.argsort(-self.mi_scores_)

        # Build structure
        structure = nx.DiGraph()
        structure.add_node('Y')

        for rank, feature in enumerate(feature_order):
            structure.add_node(feature)
            structure.add_edge('Y', feature)

            # Add up to K parents from higher-ranked features
            if rank > 0 and self.k > 0:
                candidates = feature_order[:rank]

                # Score by CMI (approximated by MI for efficiency)
                parent_scores = [(c, self.mi_scores_[c]) for c in candidates]
                parent_scores.sort(key=lambda x: x[1], reverse=True)

                for parent, _ in parent_scores[:self.k]:
                    structure.add_edge(parent, feature)

        # Feature importances
        self.feature_importances_ = self.mi_scores_ / (self.mi_scores_.sum() + 1e-10)

        return structure

    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph,
    ) -> None:
        """Learn CPTs with specified smoothing."""
        self.cpts_ = {}

        # Class prior
        y_counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        self.class_prior_ = (y_counts + self.smoothing) / (
            len(y) + self.smoothing * self.n_classes_
        )

        for node in structure.nodes():
            if node == 'Y':
                continue

            parents = list(structure.predecessors(node))

            if self.smoothing_method == 'hdp':
                self.cpts_[node] = self._estimate_cpt_hdp(X, y, node, parents)
            else:
                self.cpts_[node] = self._estimate_cpt_laplace(X, y, node, parents)

    def _estimate_cpt_laplace(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: int,
        parents: list,
    ) -> np.ndarray:
        """Standard Laplace smoothing."""
        node_card = self.cardinalities_[node]

        has_class_parent = 'Y' in parents
        feature_parents = [p for p in parents if p != 'Y']

        if not parents:
            counts = np.bincount(X[:, node].astype(int), minlength=node_card)
            return (counts + self.smoothing_alpha) / (
                len(X) + self.smoothing_alpha * node_card
            )

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
        cpt = (counts + self.smoothing_alpha) / (
            parent_totals + self.smoothing_alpha * node_card
        )

        return cpt

    def _estimate_cpt_hdp(
        self,
        X: np.ndarray,
        y: np.ndarray,
        node: int,
        parents: list,
    ) -> np.ndarray:
        """
        Hierarchical Dirichlet Process smoothing.

        Adapts smoothing based on:
        - Local sparsity
        - Parent configuration frequency
        - Global distribution
        """
        node_card = self.cardinalities_[node]

        has_class_parent = 'Y' in parents
        feature_parents = [p for p in parents if p != 'Y']

        if not parents:
            counts = np.bincount(X[:, node].astype(int), minlength=node_card)
            # Global distribution as base
            global_prob = (counts + 1) / (len(X) + node_card)
            return global_prob

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

        # Compute global (marginal) distribution
        X_node_clamped = np.clip(X[:, node].astype(int), 0, node_card - 1)
        global_counts = np.bincount(X_node_clamped, minlength=node_card)
        global_prob = (global_counts + 1) / (len(X) + node_card)

        # HDP smoothing: blend local with global based on sample count
        parent_totals = counts.sum(axis=0, keepdims=True)

        # Concentration parameter (adapts to sparsity)
        # More samples -> less smoothing
        concentration = self.smoothing_alpha * node_card

        # HDP formula: P = (N_local + concentration * P_global) / (N_parent + concentration)
        cpt = (counts + concentration * global_prob.reshape([-1] + [1] * len(parent_cards))) / (
            parent_totals + concentration
        )

        return cpt

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, ['structure_', 'cpts_', 'class_prior_'])

        # Preprocess input (applies discretization if needed)
        X = self._preprocess_X(X)
        X = X.astype(int)

        n_samples = X.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))

        log_proba += np.log(self.class_prior_ + 1e-10)

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

        log_proba -= log_proba.max(axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= proba.sum(axis=1, keepdims=True)

        return proba

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get fitted state."""
        state = {}
        if self.cpts_:
            state['cpts'] = {str(k): v.tolist() for k, v in self.cpts_.items()}
        if self.class_prior_ is not None:
            state['class_prior'] = self.class_prior_.tolist()
        if self.mi_scores_ is not None:
            state['mi_scores'] = self.mi_scores_.tolist()
        if self.feature_importances_ is not None:
            state['feature_importances'] = self.feature_importances_.tolist()
        return state

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state."""
        if 'cpts' in state:
            self.cpts_ = {int(k): np.array(v) for k, v in state['cpts'].items()}
        if 'class_prior' in state:
            self.class_prior_ = np.array(state['class_prior'])
        if 'mi_scores' in state:
            self.mi_scores_ = np.array(state['mi_scores'])
        if 'feature_importances' in state:
            self.feature_importances_ = np.array(state['feature_importances'])


class ESKDBClassifier(BaseBayesianClassifier):
    """
    Ensemble of Selective K-Dependence Bayes classifiers.

    ESKDB is a state-of-the-art BNC ensemble that achieves diversity
    through Stochastic Attribute Ordering (SAO) and/or bootstrapping.

    Parameters
    ----------
    n_estimators : int, default=50
        Number of KDB models in ensemble.

    k : int, default=2
        Maximum number of parent features per node (K-dependence).

    smoothing : {'laplace', 'hdp'}, default='hdp'
        - 'laplace': Standard add-alpha smoothing
        - 'hdp': Hierarchical Dirichlet Process (adapts to sparsity)

    diversity_method : {'sao', 'bootstrap', 'both'}, default='sao'
        How to generate ensemble diversity:
        - 'sao': Stochastic Attribute Ordering
        - 'bootstrap': Sample with replacement
        - 'both': Combine both methods

    aggregation : {'averaging', 'voting', 'stacking'}, default='averaging'
        How to combine predictions.

    n_jobs : int, default=-1
        Parallelization. -1 uses all cores.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    estimators_ : list[KDBClassifier]
        Fitted KDB models.

    oob_score_ : float
        Out-of-bag accuracy (when diversity_method includes 'bootstrap').

    feature_importances_ : np.ndarray
        Average feature importance across estimators.

    Examples
    --------
    >>> from endgame.models.bayesian import ESKDBClassifier
    >>> clf = ESKDBClassifier(n_estimators=50, k=2)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 50,
        k: int = 2,
        smoothing: str = 'hdp',
        diversity_method: str = 'sao',
        aggregation: str = 'averaging',
        n_jobs: int = -1,
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            max_cardinality=max_cardinality,
            auto_discretize=auto_discretize,
            discretizer_strategy=discretizer_strategy,
            discretizer_max_bins=discretizer_max_bins,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_estimators = n_estimators
        self.k = k
        self.smoothing_method = smoothing
        self.diversity_method = diversity_method
        self.aggregation = aggregation
        self.n_jobs = n_jobs

        self.estimators_: list[KDBClassifier] | None = None
        self.oob_score_: float | None = None
        self.oob_predictions_: np.ndarray | None = None

    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """
        ESKDB doesn't learn a single structure - each estimator has its own.
        Return a placeholder structure for API compatibility.
        """
        structure = nx.DiGraph()
        structure.add_node('Y')
        for i in range(X.shape[1]):
            structure.add_node(i)
            structure.add_edge('Y', i)
        return structure

    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph,
    ) -> None:
        """Not used for ESKDB - parameters learned by individual estimators."""
        pass

    def fit(self, X, y, **fit_params) -> 'ESKDBClassifier':
        """
        Fit ensemble of KDB classifiers.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be continuous (will be auto-discretized if
            auto_discretize=True) or discrete/integer-valued.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)

        # Store original n_features before any transformations
        self.n_features_in_ = X.shape[1]

        if self.auto_discretize:
            X = self._discretize_input(X, y, fit=True)
        elif self._needs_discretization(X):
            raise ValueError(
                "BayesianClassifiers require discrete (integer) input. "
                "Set auto_discretize=True or use BayesianDiscretizer to convert continuous features."
            )
        else:
            self.discretizer_ = None

        # Shift any negative integer features to start at 0
        X = self._remap_to_nonnegative(X, fit=True)

        X, y = self._validate_discrete_input(X, y, self.max_cardinality)

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([self._class_to_idx[v] for v in y])

        self.cardinalities_ = self._compute_cardinalities(X, y)

        # Compute MI scores for SAO
        mi_scores = compute_mi_scores(X, y)

        self._log(f"Fitting ESKDB with {self.n_estimators} estimators...")

        # Initialize random state
        rng = np.random.RandomState(self.random_state)

        # Generate seeds for each estimator
        seeds = rng.randint(0, 2**31, size=self.n_estimators)

        # Prepare training arguments
        train_args = []
        for i in range(self.n_estimators):
            attr_order = self._generate_attribute_order(mi_scores, seeds[i])

            if self.diversity_method in ('bootstrap', 'both'):
                boot_idx = rng.choice(len(X), len(X), replace=True)
                X_train = X[boot_idx]
                y_train = y[boot_idx]
                oob_mask = ~np.isin(np.arange(len(X)), boot_idx)
            else:
                X_train = X
                y_train = y
                oob_mask = None

            train_args.append((X_train, y_train, attr_order, seeds[i], oob_mask))

        # Train estimators
        self.estimators_ = []
        oob_predictions = np.zeros((len(X), self.n_classes_))
        oob_counts = np.zeros(len(X))

        def train_single(args):
            X_train, y_train, attr_order, seed, _ = args
            est = KDBClassifier(
                k=self.k,
                smoothing=self.smoothing_method,
                attribute_order=attr_order.tolist(),
                max_cardinality=self.max_cardinality,
                random_state=seed,
            )
            est.cardinalities_ = self.cardinalities_
            est.n_features_in_ = self.n_features_in_
            est.classes_ = self.classes_
            est.n_classes_ = self.n_classes_
            est.fit(X_train, y_train)
            return est

        # Train in parallel or sequential
        import os
        n_jobs = self.n_jobs
        if n_jobs == -1:
            n_jobs = os.cpu_count() or 1

        if n_jobs > 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                self.estimators_ = list(executor.map(train_single, train_args))
        else:
            for args in train_args:
                self.estimators_.append(train_single(args))

        # Compute OOB score if using bootstrap
        if self.diversity_method in ('bootstrap', 'both'):
            for i, (_, _, _, _, oob_mask) in enumerate(train_args):
                if oob_mask is not None and oob_mask.any():
                    X_oob = X[oob_mask]
                    proba = self.estimators_[i].predict_proba(X_oob)
                    oob_predictions[oob_mask] += proba
                    oob_counts[oob_mask] += 1

            # Normalize and compute accuracy
            valid = oob_counts > 0
            if valid.any():
                oob_predictions[valid] /= oob_counts[valid, np.newaxis]
                oob_pred = self.classes_[oob_predictions[valid].argmax(axis=1)]
                self.oob_score_ = (oob_pred == y[valid]).mean()
                self.oob_predictions_ = oob_predictions

        # Aggregate feature importances
        importances = np.zeros(self.n_features_in_)
        for est in self.estimators_:
            if hasattr(est, 'feature_importances_') and est.feature_importances_ is not None:
                importances += est.feature_importances_
        self.feature_importances_ = importances / len(self.estimators_)

        # Create placeholder structure
        self.structure_ = self._learn_structure(X, y)

        self._is_fitted = True
        return self

    def _generate_attribute_order(
        self,
        mi_scores: np.ndarray,
        seed: int,
    ) -> np.ndarray:
        """
        Stochastic Attribute Ordering (SAO).

        Sample attribute order with probability proportional to MI(X_i; Y).
        This creates structural diversity while preserving quality.
        """
        if self.diversity_method not in ('sao', 'both'):
            # Deterministic ordering
            return np.argsort(-mi_scores)

        rng = np.random.RandomState(seed)

        # Probability proportional to MI (with small epsilon for zero-MI features)
        probs = mi_scores + 1e-10
        probs = probs / probs.sum()

        # Sample without replacement
        order = rng.choice(
            len(mi_scores),
            size=len(mi_scores),
            replace=False,
            p=probs,
        )

        return order

    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities by aggregating estimators.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['estimators_'])

        # Preprocess input (applies discretization if needed)
        X = self._preprocess_X(X)
        X = X.astype(int)

        n_samples = X.shape[0]

        if self.aggregation == 'averaging':
            # Average probabilities
            proba = np.zeros((n_samples, self.n_classes_))
            for est in self.estimators_:
                proba += est.predict_proba(X)
            proba /= len(self.estimators_)

        elif self.aggregation == 'voting':
            # Majority voting on predictions
            votes = np.zeros((n_samples, self.n_classes_))
            for est in self.estimators_:
                preds = est.predict(X)
                for i, pred in enumerate(preds):
                    class_idx = np.where(self.classes_ == pred)[0][0]
                    votes[i, class_idx] += 1
            proba = votes / len(self.estimators_)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

        # Ensure valid probabilities
        proba = np.clip(proba, 1e-10, 1.0)
        proba /= proba.sum(axis=1, keepdims=True)

        return proba

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get fitted state."""
        state = {
            'n_estimators_actual': len(self.estimators_) if self.estimators_ else 0,
            'oob_score': self.oob_score_,
        }

        if self.feature_importances_ is not None:
            state['feature_importances'] = self.feature_importances_.tolist()

        # Note: Individual estimator states could be serialized but would be large
        # For now, we don't serialize the full estimators

        return state

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state."""
        self.oob_score_ = state.get('oob_score')

        if 'feature_importances' in state:
            self.feature_importances_ = np.array(state['feature_importances'])
