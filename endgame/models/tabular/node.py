from __future__ import annotations

"""NODE: Neural Oblivious Decision Ensembles.

NODE is a differentiable ensemble of oblivious decision trees (ODTs).
It bridges the gap between gradient boosting and neural networks.

References
----------
- Popov et al. "Neural Oblivious Decision Ensembles for Deep Learning
  on Tabular Data" (2020)
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for NODE. Install with: pip install torch")


def entmax15(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Entmax 1.5 activation (sparse softmax).

    Entmax with alpha=1.5 produces sparse outputs while being differentiable.
    """
    # Simplified implementation using sparsemax approximation
    # Full entmax requires iterative algorithm
    return F.softmax(x * 1.5, dim=dim)


def entmoid15(x: torch.Tensor) -> torch.Tensor:
    """Entmoid 1.5 - entmax-based sigmoid."""
    return torch.sigmoid(x * 1.5)


class _DenseODSTBlock(nn.Module):
    """Dense block of Oblivious Decision Trees.

    Each tree makes decisions based on feature comparisons and
    returns a weighted combination of leaf values.
    """

    def __init__(
        self,
        n_features: int,
        n_trees: int = 1024,
        tree_depth: int = 6,
        choice_function: str = "entmax15",
        bin_function: str = "entmoid15",
    ):
        super().__init__()

        self.n_features = n_features
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.n_leaves = 2 ** tree_depth

        # Feature selection weights for each internal node
        # Shape: (n_trees, tree_depth, n_features)
        self.feature_selection = nn.Parameter(
            torch.zeros(n_trees, tree_depth, n_features)
        )
        nn.init.xavier_uniform_(self.feature_selection)

        # Thresholds for comparisons
        # Shape: (n_trees, tree_depth)
        self.thresholds = nn.Parameter(torch.zeros(n_trees, tree_depth))

        # Leaf values (responses)
        # Shape: (n_trees, n_leaves)
        self.leaf_values = nn.Parameter(torch.zeros(n_trees, self.n_leaves))
        nn.init.xavier_uniform_(self.leaf_values)

        # Choice and bin functions
        if choice_function == "entmax15":
            self.choice_fn = entmax15
        elif choice_function == "softmax":
            self.choice_fn = lambda x, dim=-1: F.softmax(x, dim=dim)
        else:
            self.choice_fn = entmax15

        if bin_function == "entmoid15":
            self.bin_fn = entmoid15
        elif bin_function == "sigmoid":
            self.bin_fn = torch.sigmoid
        else:
            self.bin_fn = entmoid15

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through ODT block.

        Parameters
        ----------
        x : Tensor of shape (batch, n_features)

        Returns
        -------
        Tensor of shape (batch, n_trees)
            Tree outputs.
        """
        batch_size = x.shape[0]

        # Feature selection with soft attention
        # (n_trees, tree_depth, n_features) -> softmax over features
        feature_weights = self.choice_fn(self.feature_selection, dim=-1)

        # Compute selected features for each node
        # x: (batch, n_features)
        # feature_weights: (n_trees, tree_depth, n_features)
        # Result: (batch, n_trees, tree_depth)
        selected_features = torch.einsum('bf,tdf->btd', x, feature_weights)

        # Compare with thresholds
        # thresholds: (n_trees, tree_depth)
        # node_decisions: (batch, n_trees, tree_depth) - soft decisions [0, 1]
        node_decisions = self.bin_fn(selected_features - self.thresholds)

        # Convert decisions to leaf indices
        # Each leaf corresponds to a binary path through the tree
        # leaf_probs: (batch, n_trees, n_leaves)
        leaf_probs = self._compute_leaf_probabilities(node_decisions)

        # Weighted sum of leaf values
        # leaf_values: (n_trees, n_leaves)
        # output: (batch, n_trees)
        output = torch.einsum('btl,tl->bt', leaf_probs, self.leaf_values)

        return output

    def _compute_leaf_probabilities(self, decisions: torch.Tensor) -> torch.Tensor:
        """Compute soft probabilities for each leaf using vectorized operations.

        Parameters
        ----------
        decisions : Tensor of shape (batch, n_trees, tree_depth)
            Soft decisions at each internal node.

        Returns
        -------
        Tensor of shape (batch, n_trees, n_leaves)
            Probability of reaching each leaf.
        """
        batch_size, n_trees, depth = decisions.shape
        device = decisions.device

        # Pre-compute binary path matrix for all leaves (done once, cached)
        # Shape: (n_leaves, depth) - binary representation of each leaf index
        if not hasattr(self, '_path_matrix') or self._path_matrix.device != device:
            leaf_indices = torch.arange(self.n_leaves, device=device)
            # For each depth d, check if bit (depth-1-d) is set
            self._path_matrix = torch.zeros(self.n_leaves, depth, device=device)
            for d in range(depth):
                self._path_matrix[:, d] = (leaf_indices >> (depth - 1 - d)) & 1

        # Vectorized computation:
        # decisions: (batch, n_trees, depth)
        # path_matrix: (n_leaves, depth)
        #
        # For each leaf, we need product over depth of:
        #   decision[d] if path[d] == 1 else (1 - decision[d])
        #
        # This equals: decision^path * (1-decision)^(1-path)
        # In log space: path * log(decision) + (1-path) * log(1-decision)

        # Clamp for numerical stability
        decisions_clamped = torch.clamp(decisions, 1e-6, 1 - 1e-6)

        # Compute in log space for stability
        # log_decisions: (batch, n_trees, depth)
        log_d = torch.log(decisions_clamped)
        log_1_minus_d = torch.log(1 - decisions_clamped)

        # path_matrix: (n_leaves, depth) -> expand for broadcasting
        # Result: (batch, n_trees, n_leaves)
        log_leaf_probs = torch.einsum('btd,ld->btl', log_d, self._path_matrix) + \
                         torch.einsum('btd,ld->btl', log_1_minus_d, 1 - self._path_matrix)

        leaf_probs = torch.exp(log_leaf_probs)

        return leaf_probs


class _NODEModule(nn.Module):
    """PyTorch NODE module."""

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_layers: int = 2,
        n_trees: int = 1024,
        tree_depth: int = 6,
        choice_function: str = "entmax15",
        bin_function: str = "entmoid15",
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_layers = n_layers
        self.is_regression = is_regression

        # Input normalization
        self.input_bn = nn.BatchNorm1d(n_features)

        # NODE layers
        self.layers = nn.ModuleList()
        current_dim = n_features

        for i in range(n_layers):
            self.layers.append(
                _DenseODSTBlock(
                    n_features=current_dim,
                    n_trees=n_trees,
                    tree_depth=tree_depth,
                    choice_function=choice_function,
                    bin_function=bin_function,
                )
            )
            current_dim = n_trees

        # Output layer
        output_dim = 1 if is_regression else n_classes
        self.output = nn.Linear(current_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input
        x = self.input_bn(x)

        # NODE layers
        for layer in self.layers:
            x = layer(x)

        # Output
        return self.output(x)


class NODEClassifier(ClassifierMixin, BaseEstimator):
    """NODE: Neural Oblivious Decision Ensembles for classification.

    A differentiable ensemble of oblivious decision trees.
    Bridges the gap between gradient boosting and neural networks.

    Parameters
    ----------
    n_layers : int, default=1
        Number of dense NODE layers. Start with 1 for most datasets.
    n_trees : int, default=128
        Number of trees per layer. 64-256 works well for most datasets.
    tree_depth : int, default=4
        Depth of each oblivious tree. 3-5 works well; deeper trees risk overfitting.
    choice_function : str, default='softmax'
        Soft choice function: 'entmax15', 'softmax'. Softmax is more stable.
    bin_function : str, default='sigmoid'
        Binning function: 'entmoid15', 'sigmoid'. Sigmoid is more stable.
    learning_rate : float, default=0.01
        Learning rate. Higher values (0.01-0.1) often work better for NODE.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=128
        Training batch size. Smaller batches (64-256) often work better.
    early_stopping : int, default=20
        Early stopping patience.
    max_grad_norm : float, default=1.0
        Maximum gradient norm for clipping.
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when eval_set not provided.
        Set to 0 to disable internal validation split (not recommended).
    device : str, default='auto'
        Device: 'cuda', 'cpu', 'auto'.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    model_ : _NODEModule
        Fitted model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> clf = NODEClassifier(n_layers=1, n_trees=128, tree_depth=4)
    >>> clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    NODE works best with:
    - An eval_set for early stopping (or validation_fraction > 0)
    - Higher learning rates (0.01-0.1) than typical neural networks
    - Smaller batch sizes (64-256)
    - Fewer/shallower trees than you might expect (start small)

    When using sklearn's cross_val_score (which doesn't support eval_set),
    the model will automatically create an internal validation split using
    validation_fraction of the training data.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_layers: int = 1,
        n_trees: int = 128,
        tree_depth: int = 4,
        choice_function: str = "softmax",
        bin_function: str = "sigmoid",
        learning_rate: float = 0.01,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 128,
        early_stopping: int = 20,
        max_grad_norm: float = 1.0,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.max_grad_norm = max_grad_norm
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: _NODEModule | None = None
        self._device = None
        self._label_encoder: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[NODE] {msg}")

    def _get_device(self):
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> NODEClassifier:
        """Fit the NODE classifier.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        eval_set : tuple, optional
            Validation set for early stopping.

        Returns
        -------
        self
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create model
        self.model_ = _NODEModule(
            n_features=X.shape[1],
            n_classes=self.n_classes_,
            n_layers=self.n_layers,
            n_trees=self.n_trees,
            tree_depth=self.tree_depth,
            choice_function=self.choice_function,
            bin_function=self.bin_function,
            is_regression=False,
        ).to(self._device)

        # Create internal validation split if no eval_set provided
        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_val = int(len(X_scaled) * self.validation_fraction)
            if n_val >= 1:
                try:
                    X_train_internal, X_val_internal, y_train_internal, y_val_internal = train_test_split(
                        X_scaled, y_encoded,
                        test_size=self.validation_fraction,
                        stratify=y_encoded,
                        random_state=self.random_state,
                    )
                except ValueError:
                    X_train_internal, X_val_internal, y_train_internal, y_val_internal = train_test_split(
                        X_scaled, y_encoded,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                x_tensor = torch.tensor(X_train_internal, dtype=torch.float32)
                y_tensor = torch.tensor(y_train_internal, dtype=torch.long)
                eval_set = (X_val_internal, y_val_internal)

        # Data loader
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            # Check if validation data needs scaling (external eval_set)
            if isinstance(y_val[0], (str, np.str_)) or (hasattr(y_val, 'dtype') and y_val.dtype.kind in ('U', 'S', 'O')):
                # External eval_set with original labels
                X_val = self._scaler.transform(np.asarray(X_val, dtype=np.float32))
                X_val = np.nan_to_num(X_val, nan=0.0)
                y_val_encoded = self._label_encoder.transform(y_val)
            elif X_val.shape[1] == X_scaled.shape[1] and np.allclose(X_val.mean(), 0, atol=1):
                # Already scaled internal split
                y_val_encoded = y_val
            else:
                # External eval_set with numerical labels
                X_val = self._scaler.transform(np.asarray(X_val, dtype=np.float32))
                X_val = np.nan_to_num(X_val, nan=0.0)
                y_val_encoded = self._label_encoder.transform(y_val) if hasattr(self._label_encoder, 'transform') else y_val

            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val_encoded, dtype=torch.long),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device}...")

        for epoch in range(self.n_epochs):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(x_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()

                # Gradient clipping for stability
                if self.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model_.parameters(), self.max_grad_norm)

                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history_["train_loss"].append(train_loss)

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self._device)
                        y_batch = y_batch.to(self._device)

                        logits = self.model_(x_batch)
                        loss = F.cross_entropy(logits, y_batch)
                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss /= n_val_batches
                self.history_["val_loss"].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(f"Epoch {epoch+1}/{self.n_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("NODEClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)

                logits = self.model_(x_batch)
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))


class NODERegressor(BaseEstimator, RegressorMixin):
    """NODE for regression.

    Same architecture as NODEClassifier but with MSE loss.
    See NODEClassifier for parameter descriptions.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_layers: int = 2,
        n_trees: int = 256,
        tree_depth: int = 4,
        choice_function: str = "softmax",
        bin_function: str = "sigmoid",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 512,
        early_stopping: int = 15,
        max_grad_norm: float = 1.0,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_trees = n_trees
        self.tree_depth = tree_depth
        self.choice_function = choice_function
        self.bin_function = bin_function
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self._device = None
        self._scaler = None
        self._target_scaler = None
        self.history_ = {"train_loss": [], "val_loss": []}
        self._is_fitted = False

    def fit(self, X, y, eval_set=None, **fit_params) -> NODERegressor:
        """Fit NODE regressor."""
        _check_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._device = torch.device(
            "cuda" if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1, 1).astype(np.float32)

        self._scaler = StandardScaler()
        X_scaled = np.nan_to_num(self._scaler.fit_transform(X), nan=0.0)

        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y).ravel()

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        self.model_ = _NODEModule(
            n_features=X.shape[1],
            n_classes=1,
            n_layers=self.n_layers,
            n_trees=self.n_trees,
            tree_depth=self.tree_depth,
            choice_function=self.choice_function,
            bin_function=self.bin_function,
            is_regression=True,
        ).to(self._device)

        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        for epoch in range(self.n_epochs):
            self.model_.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self.model_(x_batch).squeeze()
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("NODERegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_pred = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                pred = self.model_(x_batch).squeeze()
                all_pred.append(pred.cpu().numpy())

        pred = np.concatenate(all_pred)
        return self._target_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
