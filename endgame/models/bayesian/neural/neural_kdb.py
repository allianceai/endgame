"""Neural K-Dependence Bayes Classifier.

NeuralKDB replaces the exponentially-sized Conditional Probability Tables
in classical KDB with neural networks that learn to estimate P(X_i | Pa(X_i), Y).

Key innovations:
- Handles high-cardinality features via embeddings
- Generalizes to unseen parent configurations
- GPU-accelerated training and inference
- Maintains the interpretable KDB structure

References
----------
Based on the KDB structure from:
Webb, G. I., et al. (2005). Not So Naive Bayes.
"""

import copy
from typing import Any

import networkx as nx
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from sklearn.utils.validation import check_is_fitted, check_X_y

from endgame.models.bayesian.base import (
    BaseBayesianClassifier,
)
from endgame.models.bayesian.structure.learning import (
    build_kdb_structure,
    compute_mi_scores,
)

if HAS_TORCH:
    from endgame.models.bayesian.neural.embeddings import ConditionalEmbeddingNet


class NeuralKDBClassifier(BaseBayesianClassifier):
    """
    K-Dependence Bayes with neural conditional probability estimators.

    NeuralKDB maintains the interpretable DAG structure of classical KDB
    but uses neural networks to estimate conditional probabilities. This
    enables handling of high-cardinality features and better generalization.

    Parameters
    ----------
    k : int, default=2
        Maximum parents per feature (excluding class).

    embedding_dim : int, default=16
        Dimensionality of value embeddings.

    hidden_dim : int, default=64
        Hidden layer size in conditional networks.

    n_hidden_layers : int, default=2
        Number of hidden layers per conditional network.

    epochs : int, default=20
        Training epochs.

    batch_size : int, default=256
        Mini-batch size for training.

    learning_rate : float, default=1e-3
        Adam learning rate.

    weight_decay : float, default=1e-5
        L2 regularization.

    dropout : float, default=0.1
        Dropout rate in networks.

    device : str, default='auto'
        'cuda', 'cpu', or 'auto' (detect GPU).

    early_stopping : int | None, default=5
        Stop if validation loss doesn't improve for this many epochs.
        None disables early stopping.

    validation_fraction : float, default=0.1
        Fraction of training data for validation (if X_val not provided).

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    structure_ : nx.DiGraph
        Learned KDB structure.

    conditionals_ : nn.ModuleDict
        Neural conditional estimators for each feature.

    class_prior_ : np.ndarray
        Prior class probabilities.

    Examples
    --------
    >>> from endgame.models.bayesian import NeuralKDBClassifier
    >>> clf = NeuralKDBClassifier(k=2, epochs=10)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)
    """

    def __init__(
        self,
        k: int = 2,
        embedding_dim: int = 16,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        epochs: int = 20,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        dropout: float = 0.1,
        device: str = 'auto',
        early_stopping: int | None = 5,
        validation_fraction: float = 0.1,
        max_cardinality: int = 100,
        auto_discretize: bool = True,
        discretizer_strategy: str = 'mdlp',
        discretizer_max_bins: int = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError(
                "NeuralKDBClassifier requires PyTorch. "
                "Install with: pip install torch"
            )

        super().__init__(
            max_cardinality=max_cardinality,
            auto_discretize=auto_discretize,
            discretizer_strategy=discretizer_strategy,
            discretizer_max_bins=discretizer_max_bins,
            random_state=random_state,
            verbose=verbose,
        )

        self.k = k
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.device = device
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction

        self.conditionals_: nn.ModuleDict | None = None
        self.class_prior_: np.ndarray | None = None
        self.device_: torch.device | None = None
        self._best_weights: dict | None = None
        self.training_history_: list[dict] | None = None

    def _setup_device(self) -> None:
        """Setup compute device."""
        if self.device == 'auto':
            self.device_ = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device_ = torch.device(self.device)

        self._log(f"Using device: {self.device_}")

    def _learn_structure(self, X: np.ndarray, y: np.ndarray) -> nx.DiGraph:
        """Build KDB structure."""
        structure = build_kdb_structure(X, y, k=self.k)

        # Compute feature importances
        mi_scores = compute_mi_scores(X, y)
        self.feature_importances_ = mi_scores / (mi_scores.sum() + 1e-10)

        return structure

    def _learn_parameters(
        self,
        X: np.ndarray,
        y: np.ndarray,
        structure: nx.DiGraph,
    ) -> None:
        """Create and initialize neural conditional estimators."""
        # Class prior
        y_counts = np.bincount(y.astype(int), minlength=self.n_classes_)
        self.class_prior_ = y_counts / len(y)

        # Create neural conditional for each feature
        self.conditionals_ = nn.ModuleDict()

        for node in structure.nodes():
            if node == 'Y':
                continue

            parents = list(structure.predecessors(node))

            # Build parent cardinality list
            # Y is always a parent in KDB, so include class cardinality
            parent_cards = []
            for p in parents:
                if p == 'Y':
                    parent_cards.append(self.n_classes_)
                else:
                    parent_cards.append(self.cardinalities_[p])

            if len(parent_cards) == 0:
                # Shouldn't happen in KDB, but handle gracefully
                parent_cards = [self.n_classes_]

            self.conditionals_[str(node)] = ConditionalEmbeddingNet(
                target_cardinality=self.cardinalities_[node],
                parent_cardinalities=parent_cards,
                embedding_dim=self.embedding_dim,
                hidden_dim=self.hidden_dim,
                n_hidden_layers=self.n_hidden_layers,
                dropout=self.dropout,
            )

        # Move to device
        self.conditionals_.to(self.device_)

    def fit(
        self,
        X,
        y,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        **fit_params,
    ) -> 'NeuralKDBClassifier':
        """
        Fit the Neural KDB classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Can be continuous (will be auto-discretized if
            auto_discretize=True) or discrete/integer-valued.
        y : array-like of shape (n_samples,)
            Target values.
        X_val : np.ndarray, optional
            Validation features for early stopping.
        y_val : np.ndarray, optional
            Validation targets.

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

        # Setup
        self._setup_device()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Store metadata
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        self._class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y = np.array([self._class_to_idx[v] for v in y])

        self.cardinalities_ = self._compute_cardinalities(X, y)

        # Create validation set if not provided
        if X_val is None and self.early_stopping:
            n_val = int(len(X) * self.validation_fraction)
            if n_val > 0:
                rng = np.random.RandomState(self.random_state)
                indices = rng.permutation(len(X))
                val_idx = indices[:n_val]
                train_idx = indices[n_val:]

                X_val = X[val_idx]
                y_val = y[val_idx]
                X = X[train_idx]
                y = y[train_idx]

        # Structure learning
        self._log("Learning KDB structure...")
        self.structure_ = self._learn_structure(X, y)

        # Parameter initialization
        self._log("Creating neural conditionals...")
        self._learn_parameters(X, y, self.structure_)

        # Training
        self._log("Training neural networks...")
        self._train(X, y, X_val, y_val)

        self._is_fitted = True
        return self

    def _train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Joint training of all conditional networks."""
        # Create dataset
        X_t = torch.tensor(X, dtype=torch.long, device=self.device_)
        y_t = torch.tensor(y, dtype=torch.long, device=self.device_)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Optimizer over all conditional parameters
        all_params = []
        for net in self.conditionals_.values():
            all_params.extend(net.parameters())

        optimizer = torch.optim.Adam(
            all_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float('inf')
        patience_counter = 0
        self.training_history_ = []

        for epoch in range(self.epochs):
            # Training phase
            self.conditionals_.train()
            epoch_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                loss = self._compute_batch_loss(X_batch, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)

            # Validation phase
            val_loss = None
            if X_val is not None:
                val_loss = self._compute_validation_loss(X_val, y_val)

                history_entry = {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': val_loss,
                }
            else:
                history_entry = {
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                }

            self.training_history_.append(history_entry)

            if self.verbose:
                val_str = f", val_loss: {val_loss:.4f}" if val_loss else ""
                self._log(f"Epoch {epoch + 1}/{self.epochs}: "
                         f"train_loss: {avg_train_loss:.4f}{val_str}")

            # Early stopping
            if X_val is not None and self.early_stopping:
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_best_weights()
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        self._log(f"Early stopping at epoch {epoch + 1}")
                        self._restore_best_weights()
                        break

    def _compute_batch_loss(
        self,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss for a batch.

        Loss = -sum_i log P(x_i | parents(x_i), y)
        """
        batch_size = X_batch.shape[0]
        total_loss = torch.tensor(0.0, device=self.device_)

        for node_str, net in self.conditionals_.items():
            node = int(node_str)
            parents = list(self.structure_.predecessors(node))

            # Build parent values tensor
            parent_values = self._get_parent_values(X_batch, y_batch, parents)

            # Get logits and compute cross-entropy
            logits = net(parent_values)
            target = X_batch[:, node]

            # Clamp target to valid range
            target = target.clamp(0, self.cardinalities_[node] - 1)

            loss = F.cross_entropy(logits, target)
            total_loss = total_loss + loss

        return total_loss

    def _get_parent_values(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        parents: list[int | str],
    ) -> torch.Tensor:
        """Build tensor of parent values."""
        parent_cols = []

        for p in parents:
            if p == 'Y':
                parent_cols.append(y.unsqueeze(1))
            else:
                parent_cols.append(X[:, p].unsqueeze(1))

        if len(parent_cols) == 0:
            # Shouldn't happen, but handle gracefully
            return y.unsqueeze(1)

        return torch.cat(parent_cols, dim=1)

    def _compute_validation_loss(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> float:
        """Compute validation loss."""
        self.conditionals_.eval()

        X_t = torch.tensor(X_val, dtype=torch.long, device=self.device_)
        y_t = torch.tensor(y_val, dtype=torch.long, device=self.device_)

        with torch.no_grad():
            loss = self._compute_batch_loss(X_t, y_t)

        return loss.item()

    def _save_best_weights(self) -> None:
        """Save best model weights for early stopping."""
        self._best_weights = {
            name: copy.deepcopy(net.state_dict())
            for name, net in self.conditionals_.items()
        }

    def _restore_best_weights(self) -> None:
        """Restore best model weights."""
        if self._best_weights is not None:
            for name, net in self.conditionals_.items():
                if name in self._best_weights:
                    net.load_state_dict(self._best_weights[name])

    def predict_proba(self, X) -> np.ndarray:
        """
        Compute P(Y|X) using neural conditionals.

        For each class c:
        P(Y=c|X) ∝ P(Y=c) * ∏_i P(x_i | parents(x_i), Y=c)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ['structure_', 'conditionals_', 'class_prior_'])

        # Preprocess input (applies discretization if needed)
        X = self._preprocess_X(X)

        self.conditionals_.eval()
        X_t = torch.tensor(X, dtype=torch.long, device=self.device_)
        n_samples = X.shape[0]

        with torch.no_grad():
            log_probs = torch.zeros(
                n_samples, self.n_classes_,
                device=self.device_
            )

            # Add log class prior
            log_prior = torch.log(
                torch.tensor(self.class_prior_, device=self.device_) + 1e-10
            )
            log_probs += log_prior

            # Add log P(x_i | parents) for each class
            for node_str, net in self.conditionals_.items():
                node = int(node_str)
                parents = list(self.structure_.predecessors(node))

                # For each class, compute P(x_i | parents, y=c)
                for c_idx in range(self.n_classes_):
                    # Build parent values with y=c_idx
                    y_c = torch.full(
                        (n_samples,), c_idx,
                        dtype=torch.long, device=self.device_
                    )
                    parent_values = self._get_parent_values(X_t, y_c, parents)

                    # Get log probabilities
                    log_p = net.get_log_proba(parent_values)

                    # Select probability of actual observed value
                    node_values = X_t[:, node].clamp(0, self.cardinalities_[node] - 1)
                    log_probs[:, c_idx] += log_p[
                        torch.arange(n_samples, device=self.device_),
                        node_values
                    ]

            # Normalize with softmax
            probs = F.softmax(log_probs, dim=-1)

        return probs.cpu().numpy()

    def to_onnx(self, path: str) -> None:
        """
        Export model to ONNX format for production deployment.

        Parameters
        ----------
        path : str
            Path to save ONNX model.
        """
        # This is a placeholder - full ONNX export would require
        # combining the conditional networks into a single forward pass
        raise NotImplementedError(
            "ONNX export is not yet implemented for NeuralKDB. "
            "For production, use torch.jit.script for TorchScript export."
        )

    def _get_fitted_state(self) -> dict[str, Any]:
        """Get fitted state for serialization."""
        import base64
        import io

        state = {
            'class_prior': self.class_prior_.tolist() if self.class_prior_ is not None else None,
            'training_history': self.training_history_,
        }

        if self.feature_importances_ is not None:
            state['feature_importances'] = self.feature_importances_.tolist()

        # Serialize neural network weights
        if self.conditionals_ is not None:
            conditional_states = {}
            for name, net in self.conditionals_.items():
                buffer = io.BytesIO()
                torch.save(net.state_dict(), buffer)
                conditional_states[name] = base64.b64encode(
                    buffer.getvalue()
                ).decode('ascii')
            state['conditional_states'] = conditional_states

        return state

    def _set_fitted_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state from serialization."""

        if 'class_prior' in state and state['class_prior'] is not None:
            self.class_prior_ = np.array(state['class_prior'])

        if 'training_history' in state:
            self.training_history_ = state['training_history']

        if 'feature_importances' in state:
            self.feature_importances_ = np.array(state['feature_importances'])

        # Note: Restoring neural network weights requires re-creating the
        # networks first, which needs the structure. Full deserialization
        # would need to be done in from_dict after structure is restored.
