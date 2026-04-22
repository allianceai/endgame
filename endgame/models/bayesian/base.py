from __future__ import annotations

"""Base classes and mixins for Bayesian Network Classifiers."""

from abc import abstractmethod
from enum import Enum
from typing import Any

import networkx as nx
import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y

from endgame.core.base import EndgameEstimator
from endgame.core.glassbox import GlassboxMixin

# =============================================================================
# Custom Exceptions
# =============================================================================


class BayesianNetworkError(Exception):
    """Base exception for bayesian module."""
    pass


class StructureLearningError(BayesianNetworkError):
    """Raised when structure learning fails."""
    pass


class CyclicGraphError(StructureLearningError):
    """Raised when learned structure contains cycles."""
    pass


class CardinalityError(BayesianNetworkError):
    """Raised when feature has too many/few unique values."""
    pass


class ConvergenceWarning(UserWarning):
    """Issued when optimization doesn't converge."""
    pass


class NotFittedError(BayesianNetworkError):
    """Raised when using a model before fitting."""
    pass


# =============================================================================
# Enums
# =============================================================================


class MissingValueStrategy(Enum):
    """Strategy for handling missing values."""
    ERROR = "error"              # Raise ValueError (default for fit)
    MARGINALIZE = "marginalize"  # Sum over possible values (default for predict)
    IMPUTE_MODE = "impute_mode"  # Fill with training mode
    IMPUTE_PARENT = "impute_parent"  # Predict from parents in DAG


# =============================================================================
# Serialization Mixin
# =============================================================================


class BayesianSerializationMixin:
    """Provides to_dict/from_dict for all BNC classifiers."""

    _VERSION = "0.4.0"

    def to_dict(self) -> dict[str, Any]:
        """
        Export model state to a dictionary.

        Handles:
        - numpy arrays → lists
        - networkx graphs → edge lists
        - torch modules → state_dicts (base64 encoded)

        Returns
        -------
        dict
            Serialized model state.
        """
        state = {
            'class_name': self.__class__.__name__,
            'version': self._VERSION,
            'params': self.get_params(),
            'fitted': hasattr(self, 'structure_') and self.structure_ is not None,
        }

        if state['fitted']:
            state['structure'] = list(self.structure_.edges())
            state['classes'] = self.classes_.tolist() if hasattr(self, 'classes_') else []
            state['cardinalities'] = self.cardinalities_ if hasattr(self, 'cardinalities_') else {}
            state['n_features_in'] = self.n_features_in_ if hasattr(self, 'n_features_in_') else 0

            if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
                state['feature_names_in'] = self.feature_names_in_.tolist()

            # Subclass-specific state
            state.update(self._get_fitted_state())

        return state

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get subclass-specific fitted state. Override in subclasses."""
        return {}

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Set subclass-specific fitted state. Override in subclasses."""
        pass

    @classmethod
    def from_dict(cls, state: dict[str, Any]) -> BayesianClassifierMixin:
        """
        Reconstruct model from dictionary.

        Parameters
        ----------
        state : dict
            Serialized model state from to_dict().

        Returns
        -------
        BayesianClassifierMixin
            Reconstructed model.
        """
        if state['class_name'] != cls.__name__:
            raise ValueError(f"State is for {state['class_name']}, not {cls.__name__}")

        model = cls(**state['params'])

        if state['fitted']:
            model.structure_ = nx.DiGraph(state['structure'])
            model.classes_ = np.array(state['classes'])
            model.n_classes_ = len(model.classes_)
            model.cardinalities_ = state['cardinalities']
            model.n_features_in_ = state['n_features_in']

            if 'feature_names_in' in state:
                model.feature_names_in_ = np.array(state['feature_names_in'])

            model._set_fitted_state(state)
            model._is_fitted = True

        return model


# =============================================================================
# Base Bayesian Classifier Mixin
# =============================================================================


class BayesianClassifierMixin(BayesianSerializationMixin):
    """
    Mixin providing common Bayesian Network Classifier functionality.

    All BNC classifiers should inherit from this mixin to get:
    - Structure validation and access
    - Markov blanket computation
    - Feature importance from mutual information
    - Explanation methods
    - Export to pgmpy format

    Attributes set after fit():
        structure_: nx.DiGraph - The learned DAG
        classes_: np.ndarray - Unique class labels
        cardinalities_: dict[int, int] - {feature_idx: n_values}
        feature_importances_: np.ndarray - MI(X_i; Y) normalized
        n_features_in_: int - Number of features seen during fit
        feature_names_in_: np.ndarray - Feature names (if DataFrame input)
    """

    @property
    def markov_blanket_(self) -> set[int]:
        """
        Returns indices of features in target's Markov Blanket.

        The Markov Blanket MB(Y) includes:
        - Parents of Y
        - Children of Y
        - Parents of children of Y (co-parents)

        Returns
        -------
        set[int]
            Feature indices in the Markov Blanket.
        """
        if not hasattr(self, 'structure_') or self.structure_ is None:
            raise NotFittedError("Call fit() first")
        return self._compute_markov_blanket(self.structure_, target='Y')

    def _compute_markov_blanket(
        self,
        graph: nx.DiGraph,
        target: str | int
    ) -> set[int]:
        """
        Compute Markov Blanket for a target node.

        Parameters
        ----------
        graph : nx.DiGraph
            The DAG structure.
        target : str or int
            Target node identifier.

        Returns
        -------
        set[int]
            Nodes in the Markov Blanket.
        """
        if target not in graph:
            return set()

        mb = set()

        # Parents
        parents = set(graph.predecessors(target))
        mb.update(parents)

        # Children
        children = set(graph.successors(target))
        mb.update(children)

        # Co-parents (parents of children)
        for child in children:
            coparents = set(graph.predecessors(child))
            mb.update(coparents)

        # Remove target and filter to integer nodes (features)
        mb.discard(target)
        return {n for n in mb if isinstance(n, int)}

    def explain(self, x: np.ndarray) -> dict[str, Any]:
        """
        Explain prediction for a single instance.

        Computes the contribution of each feature to the prediction
        based on the Bayesian network structure.

        Parameters
        ----------
        x : np.ndarray
            Single instance of shape (n_features,).

        Returns
        -------
        dict
            {
                'prediction': int - Predicted class,
                'probabilities': np.ndarray - Class probabilities,
                'influential_features': list[tuple[str, float]] - (name, contribution),
                'local_structure': nx.DiGraph - Subgraph relevant to this instance
            }
        """
        if not hasattr(self, 'structure_') or self.structure_ is None:
            raise NotFittedError("Call fit() first")

        x = np.atleast_2d(x)
        if x.shape[0] != 1:
            raise ValueError("explain() requires a single instance")

        # Get prediction
        proba = self.predict_proba(x)[0]
        prediction = self.classes_[np.argmax(proba)]

        # Compute feature contributions
        contributions = self._compute_feature_contributions(x[0], proba)

        # Get feature names
        if hasattr(self, 'feature_names_in_') and self.feature_names_in_ is not None:
            names = self.feature_names_in_
        else:
            names = [f"feature_{i}" for i in range(len(contributions))]

        # Sort by absolute contribution
        influential = sorted(
            zip(names, contributions),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        # Extract local structure (features that matter for this prediction)
        local_nodes = {'Y'} | self.markov_blanket_
        local_structure = self.structure_.subgraph(local_nodes).copy()

        return {
            'prediction': prediction,
            'probabilities': proba,
            'influential_features': influential,
            'local_structure': local_structure
        }

    def _compute_feature_contributions(
        self,
        x: np.ndarray,
        proba: np.ndarray
    ) -> np.ndarray:
        """
        Compute per-feature contribution to the prediction.

        Default implementation uses the difference in log-probability
        when the feature is marginalized vs observed.

        Parameters
        ----------
        x : np.ndarray
            Single instance.
        proba : np.ndarray
            Predicted probabilities.

        Returns
        -------
        np.ndarray
            Contribution scores for each feature.
        """
        # Default: use feature importances weighted by deviation from mean
        if hasattr(self, 'feature_importances_'):
            return self.feature_importances_.copy()
        return np.zeros(len(x))

    def to_pgmpy(self):
        """
        Export to pgmpy BayesianNetwork for advanced inference/visualization.

        Returns
        -------
        pgmpy.models.BayesianNetwork
            Exported model (requires pgmpy to be installed).
        """
        try:
            from pgmpy.factors.discrete import TabularCPD
            from pgmpy.models import BayesianNetwork
        except ImportError:
            raise ImportError(
                "pgmpy is required for export. "
                "Install with: pip install endgame-ml[bayesian]"
            )

        if not hasattr(self, 'structure_') or self.structure_ is None:
            raise NotFittedError("Call fit() first")

        # Create pgmpy model
        model = BayesianNetwork(list(self.structure_.edges()))

        # Add CPTs if available
        if hasattr(self, 'cpts_'):
            for node, cpt in self.cpts_.items():
                parents = list(self.structure_.predecessors(node))
                # Convert to pgmpy format (TabularCPD expects specific shape)
                cpd = self._numpy_cpt_to_pgmpy(node, cpt, parents)
                model.add_cpds(cpd)

        return model

    def _numpy_cpt_to_pgmpy(
        self,
        node: int,
        cpt: np.ndarray,
        parents: list[int]
    ):
        """Convert numpy CPT to pgmpy TabularCPD format."""
        from pgmpy.factors.discrete import TabularCPD

        node_card = self.cardinalities_.get(node, cpt.shape[0])

        if len(parents) == 0:
            # Simple prior
            values = cpt.reshape(-1, 1)
        else:
            # Reshape for pgmpy (node_cardinality x product of parent cardinalities)
            parent_cards = [self.cardinalities_.get(p, 2) for p in parents]
            values = cpt.reshape(node_card, -1)

        evidence = [str(p) for p in parents] if parents else None
        evidence_card = [self.cardinalities_.get(p, 2) for p in parents] if parents else None

        return TabularCPD(
            variable=str(node),
            variable_card=node_card,
            values=values,
            evidence=evidence,
            evidence_card=evidence_card
        )

    @abstractmethod
    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """
        Subclass-specific structure learning.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Target values.

        Returns
        -------
        nx.DiGraph
            Learned DAG structure.
        """
        pass

    @abstractmethod
    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph
    ) -> None:
        """
        Subclass-specific parameter learning (CPTs or neural).

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Target values.
        structure : nx.DiGraph
            The DAG structure.
        """
        pass

    def _validate_discrete_input(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        max_cardinality: int = 100
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Validate that input is discrete (integer-valued).

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray, optional
            Target values.
        max_cardinality : int, default=100
            Maximum allowed unique values per feature.

        Returns
        -------
        tuple
            Validated (X, y).
        """
        # Ensure integer type
        if not np.issubdtype(X.dtype, np.integer):
            # Check if values are actually integers
            if np.allclose(X, X.astype(int)):
                X = X.astype(int)
            else:
                raise ValueError(
                    "BayesianClassifiers require discrete (integer) input. "
                    "Use BayesianDiscretizer to convert continuous features."
                )

        # Check cardinalities
        for i in range(X.shape[1]):
            unique_vals = np.unique(X[:, i])
            if len(unique_vals) > max_cardinality:
                raise CardinalityError(
                    f"Feature {i} has {len(unique_vals)} unique values, "
                    f"exceeding max_cardinality={max_cardinality}. "
                    "Consider discretizing with fewer bins."
                )
            # Ensure values are non-negative and sequential
            if unique_vals.min() < 0:
                raise ValueError(
                    f"Feature {i} contains negative values. "
                    "Values must be non-negative integers."
                )

        if y is not None:
            if not np.issubdtype(y.dtype, np.integer):
                if np.allclose(y, y.astype(int)):
                    y = y.astype(int)

        return X, y

    def _compute_cardinalities(
        self,
        X: np.ndarray,
        y: np.ndarray
    ) -> dict[int | str, int]:
        """
        Compute the number of unique values for each variable.

        Parameters
        ----------
        X : np.ndarray
            Features of shape (n_samples, n_features).
        y : np.ndarray
            Target values.

        Returns
        -------
        dict
            {feature_idx: cardinality} plus {'Y': n_classes}.
        """
        cardinalities = {}

        for i in range(X.shape[1]):
            cardinalities[i] = len(np.unique(X[:, i]))

        cardinalities['Y'] = len(np.unique(y))

        return cardinalities


# =============================================================================
# Base Bayesian Classifier
# =============================================================================


class BaseBayesianClassifier(GlassboxMixin, ClassifierMixin, EndgameEstimator, BayesianClassifierMixin):
    """
    Base class for all Bayesian Network Classifiers.

    Provides the common fit/predict/predict_proba interface that all
    BNC implementations share.

    Parameters
    ----------
    smoothing : float, default=1.0
        Laplace smoothing parameter (alpha).
        Use 0 for MLE, 1 for add-one smoothing.
    missing_values : str, default='error'
        Strategy for missing values: 'error', 'marginalize', 'impute_mode'.
    max_cardinality : int, default=100
        Maximum unique values per feature.
    auto_discretize : bool, default=True
        If True, automatically discretize continuous features using
        BayesianDiscretizer. If False, raise an error for continuous input.
    discretizer_strategy : str, default='mdlp'
        Discretization strategy when auto_discretize=True.
        Options: 'mdlp', 'equal_width', 'equal_freq', 'kmeans'.
    discretizer_max_bins : int, default=10
        Maximum number of bins per feature when auto-discretizing.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        smoothing: float = 1.0,
        missing_values: str = 'error',
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.smoothing = smoothing
        self.missing_values = missing_values
        self.max_cardinality = max_cardinality
        self.auto_discretize = auto_discretize
        self.discretizer_strategy = discretizer_strategy
        self.discretizer_max_bins = discretizer_max_bins

        # Will be set during fit
        self.structure_: nx.DiGraph | None = None
        self.classes_: np.ndarray | None = None
        self.cardinalities_: dict | None = None
        self.feature_importances_: np.ndarray | None = None
        self.n_features_in_: int | None = None
        self.feature_names_in_: np.ndarray | None = None
        self.discretizer_ = None  # Will hold BayesianDiscretizer if auto-discretizing

    def _needs_discretization(self, X: np.ndarray) -> bool:
        """Check if input needs discretization (has continuous features)."""
        if np.issubdtype(X.dtype, np.integer):
            return False
        if np.allclose(X, X.astype(int)):
            return False
        return True

    def _remap_to_nonnegative(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """Remap integer features so all values start at 0.

        Bayesian networks treat features as categorical indices, so shifting
        by a constant doesn't change the model semantics.
        """
        X = X.copy()
        if fit:
            self._feature_min_offsets = np.zeros(X.shape[1], dtype=int)
            for i in range(X.shape[1]):
                col_min = int(X[:, i].min())
                if col_min < 0:
                    self._feature_min_offsets[i] = -col_min
        if hasattr(self, '_feature_min_offsets'):
            for i in range(X.shape[1]):
                if self._feature_min_offsets[i] != 0:
                    X[:, i] += self._feature_min_offsets[i]
        return X

    def _discretize_input(self, X: np.ndarray, y: np.ndarray = None, fit: bool = True) -> np.ndarray:
        """Apply discretization to continuous input.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray, optional
            Target values (needed for supervised discretization during fit).
        fit : bool, default=True
            If True, fit the discretizer. If False, only transform.

        Returns
        -------
        np.ndarray
            Discretized features.
        """
        from endgame.preprocessing import BayesianDiscretizer

        if fit:
            self.discretizer_ = BayesianDiscretizer(
                strategy=self.discretizer_strategy,
                max_bins=self.discretizer_max_bins,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            self._log(f"Auto-discretizing continuous features using {self.discretizer_strategy}...")
            X_disc = self.discretizer_.fit_transform(X, y)
        else:
            if self.discretizer_ is None:
                raise ValueError("Discretizer not fitted. Call fit() first.")
            X_disc = self.discretizer_.transform(X)

        return X_disc

    def fit(self, X, y, **fit_params) -> BaseBayesianClassifier:
        """
        Fit the Bayesian Network Classifier.

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
            Fitted classifier.
        """
        # Input validation
        X, y = check_X_y(X, y)

        # Store original n_features before any transformations
        self.n_features_in_ = X.shape[1]

        # Always discretize when auto_discretize=True: the BayesianDiscretizer
        # handles per-feature detection (keeps low-cardinality integer features
        # as-is, bins the rest).  This avoids the global _needs_discretization
        # check which can miss high-cardinality integer features.
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

        # Validate discrete input
        X, y = self._validate_discrete_input(X, y, self.max_cardinality)

        # Store metadata
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        # Remap y to contiguous 0..n_classes_-1 indices.
        # Required because CPT dimensions are sized by n_classes_, so raw
        # label values (e.g. [0, 2, 5]) would index out of bounds.
        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([self._class_to_idx[v] for v in y])

        self.cardinalities_ = self._compute_cardinalities(X, y)

        self._log(f"Fitting {self.__class__.__name__} on {X.shape[0]} samples, "
                  f"{X.shape[1]} features, {self.n_classes_} classes")

        # Structure learning
        self._log("Learning structure...")
        self.structure_ = self._learn_structure(X, y)

        # Validate structure is DAG
        if not nx.is_directed_acyclic_graph(self.structure_):
            raise CyclicGraphError("Learned structure contains cycles")

        # Parameter learning
        self._log("Learning parameters...")
        self._learn_parameters(X, y, self.structure_)

        self._is_fitted = True
        return self

    def _preprocess_X(self, X: np.ndarray) -> np.ndarray:
        """Preprocess input for prediction (apply discretization if needed).

        Parameters
        ----------
        X : np.ndarray
            Input features.

        Returns
        -------
        np.ndarray
            Preprocessed (potentially discretized) features.
        """
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        # Apply discretization if it was used during training
        if self.discretizer_ is not None:
            X = self._discretize_input(X, fit=False)

        # Apply the same non-negative remapping used during fit
        X = self._remap_to_nonnegative(X, fit=False)

        return X

    def predict(self, X) -> np.ndarray:
        """
        Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """
        Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        pass

    def score(self, X, y) -> float:
        """
        Return mean accuracy on the given test data and labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.

        Returns
        -------
        float
            Mean accuracy.
        """
        return np.mean(self.predict(X) == y)


    _structure_type = "bayesian_network"

    def _structure_content(self) -> dict[str, Any]:
        if not hasattr(self, 'structure_') or self.structure_ is None:
            raise NotFittedError("Call fit() first")
        nodes = list(self.structure_.nodes())
        edges = [[str(u), str(v)] for u, v in self.structure_.edges()]
        cpts_payload: dict[str, Any] = {}
        if hasattr(self, 'cpts_') and self.cpts_ is not None:
            for node, cpt in self.cpts_.items():
                arr = np.asarray(cpt)
                cpts_payload[str(node)] = {
                    "shape": list(arr.shape),
                    "values": arr.tolist(),
                }
        cardinalities = {
            str(k): int(v)
            for k, v in (self.cardinalities_ or {}).items()
        } if getattr(self, "cardinalities_", None) else {}
        feature_importances = (
            self.feature_importances_.tolist()
            if getattr(self, "feature_importances_", None) is not None
            else []
        )
        mb: list = []
        try:
            mb = sorted(self.markov_blanket_)
        except Exception:
            mb = []
        return {
            "nodes": [str(n) for n in nodes],
            "edges": edges,
            "cpts": cpts_payload,
            "cardinalities": cardinalities,
            "feature_importances": feature_importances,
            "markov_blanket": mb,
        }
