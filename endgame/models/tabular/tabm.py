"""TabM: Parameter-Efficient MLP Ensembling for tabular data.

TabM efficiently imitates an ensemble of MLPs using BatchEnsemble-style
parameter sharing. One model produces k predictions per sample; during
training each prediction is trained independently (mean loss across members),
during inference they are averaged.

References
----------
- Gorishniy et al. "TabM: Advancing Tabular Deep Learning with
  Parameter-Efficient Ensembling" (ICLR 2025)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch imports (lazy loaded)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    _nn_Module = nn.Module
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    TensorDataset = None
    _nn_Module = object  # Fallback base class


def _check_torch():
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for TabM. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _LinearBatchEnsemble(_nn_Module):
    """Linear layer with BatchEnsemble parameter sharing.

    Each ensemble member m has scaling vectors r_m and s_m applied to a
    shared weight matrix W so that the effective weight for member m is
    diag(s_m) @ W @ diag(r_m).

    Parameters
    ----------
    in_features : int
        Number of input features.
    out_features : int
        Number of output features.
    k : int
        Number of ensemble members.
    bias : bool, default=True
        Whether to include a per-member bias.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        k: int,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        # Shared weight matrix
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Per-member scaling vectors
        # r_m: input scaling, initialized to ones
        self.r = nn.Parameter(torch.ones(k, in_features))
        # s_m: output scaling, initialized to normal(0, 0.01) + 1
        self.s = nn.Parameter(torch.randn(k, out_features) * 0.01 + 1.0)

        # Per-member bias
        if bias:
            self.bias = nn.Parameter(torch.zeros(k, out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, k, in_features)

        Returns
        -------
        torch.Tensor of shape (batch, k, out_features)
        """
        # x: (batch, k, in_features)
        # r: (k, in_features) -> scale input per member
        x_scaled = x * self.r.unsqueeze(0)  # (batch, k, in_features)

        # Apply shared weight: (batch, k, in_features) @ (in_features, out_features)
        out = torch.matmul(x_scaled, self.weight.t())  # (batch, k, out_features)

        # Apply output scaling per member
        out = out * self.s.unsqueeze(0)  # (batch, k, out_features)

        if self.bias is not None:
            out = out + self.bias.unsqueeze(0)  # (batch, k, out_features)

        return out


class _PiecewiseLinearEmbedding(_nn_Module):
    """Piecewise linear embedding for numerical features.

    Learns piecewise-linear transformations of each feature using
    learnable bin edges and slopes.

    Parameters
    ----------
    n_features : int
        Number of numerical features.
    n_bins : int, default=16
        Number of bins for piecewise linear encoding.
    """

    def __init__(self, n_features: int, n_bins: int = 16):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins

        # Learnable bin edges for each feature: (n_features, n_bins - 1)
        # Initialize uniformly in [-1, 1] range (data is standardized)
        edges = torch.linspace(-2.0, 2.0, n_bins + 1)[1:-1]  # n_bins - 1 edges
        self.edges = nn.Parameter(
            edges.unsqueeze(0).expand(n_features, -1).clone()
        )

        # Learnable slopes/weights for each bin: (n_features, n_bins)
        self.weights = nn.Parameter(torch.ones(n_features, n_bins))
        nn.init.normal_(self.weights, mean=0.0, std=0.01)

        # Learnable bias for each feature
        self.bias = nn.Parameter(torch.zeros(n_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_features)

        Returns
        -------
        torch.Tensor of shape (batch, n_features)
            Piecewise-linear transformed features.
        """
        batch_size = x.shape[0]

        # Sort edges to maintain ordering
        sorted_edges, _ = torch.sort(self.edges, dim=1)  # (n_features, n_bins-1)

        # Compute bin membership via soft binning
        # x: (batch, n_features) -> (batch, n_features, 1)
        x_expanded = x.unsqueeze(-1)  # (batch, n_features, 1)
        edges_expanded = sorted_edges.unsqueeze(0)  # (1, n_features, n_bins-1)

        # Compute which bin each value falls into using sigmoid soft assignments
        # For each feature, compute cumulative indicator: how much of x is above each edge
        indicators = torch.sigmoid(10.0 * (x_expanded - edges_expanded))  # (batch, n_features, n_bins-1)

        # Compute bin activations: bin_i activation = indicator_i - indicator_{i+1}
        # First bin: 1 - indicator_0
        # Middle bins: indicator_{i-1} - indicator_i
        # Last bin: indicator_{n_bins-2}
        ones = torch.ones(batch_size, self.n_features, 1, device=x.device)
        zeros = torch.zeros(batch_size, self.n_features, 1, device=x.device)
        cumulative = torch.cat([ones, indicators, zeros], dim=-1)  # (batch, n_features, n_bins+1)
        bin_activations = cumulative[:, :, :-1] - cumulative[:, :, 1:]  # (batch, n_features, n_bins)

        # Weight the activations
        # weights: (n_features, n_bins) -> (1, n_features, n_bins)
        weighted = (bin_activations * self.weights.unsqueeze(0)).sum(dim=-1)  # (batch, n_features)

        return weighted + self.bias.unsqueeze(0)


class _TabMNetwork(_nn_Module):
    """Full TabM network.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int
        Number of output units (n_classes for classification, 1 for regression).
    k : int
        Number of ensemble members.
    n_blocks : int
        Number of MLP blocks.
    d_block : int
        Hidden dimension of each block.
    dropout : float
        Dropout rate.
    use_embeddings : bool
        Whether to use piecewise linear embeddings.
    n_bins : int
        Number of bins for piecewise linear embeddings.
    is_regression : bool
        Whether this is a regression task.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.0,
        use_embeddings: bool = False,
        n_bins: int = 16,
        is_regression: bool = False,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.k = k
        self.is_regression = is_regression

        # Optional piecewise linear embeddings
        self.embedding = None
        if use_embeddings:
            self.embedding = _PiecewiseLinearEmbedding(n_features, n_bins)

        # Build MLP blocks: Linear -> BatchNorm -> ReLU -> Dropout
        blocks = []
        in_dim = n_features
        for i in range(n_blocks):
            out_dim = d_block
            block = nn.ModuleDict({
                "linear": _LinearBatchEnsemble(in_dim, out_dim, k),
                "bn": nn.BatchNorm1d(out_dim),
                "dropout": nn.Dropout(dropout),
            })
            blocks.append(block)
            in_dim = out_dim

        self.blocks = nn.ModuleList(blocks)

        # Output head
        self.output = _LinearBatchEnsemble(in_dim, n_outputs, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, n_features)

        Returns
        -------
        torch.Tensor of shape (batch, k, n_outputs)
        """
        # Apply optional embeddings
        if self.embedding is not None:
            x = self.embedding(x)  # (batch, n_features)

        # Expand to k members: (batch, n_features) -> (batch, k, n_features)
        x = x.unsqueeze(1).expand(-1, self.k, -1)

        # MLP blocks
        for block in self.blocks:
            x = block["linear"](x)  # (batch, k, d_block)

            # BatchNorm: reshape to apply across batch and k dimensions
            batch_size, k, d = x.shape
            x_bn = x.reshape(batch_size * k, d)
            x_bn = block["bn"](x_bn)
            x = x_bn.reshape(batch_size, k, d)

            x = F.relu(x)
            x = block["dropout"](x)

        # Output layer
        x = self.output(x)  # (batch, k, n_outputs)

        return x


class TabMClassifier(ClassifierMixin, BaseEstimator):
    """TabM: Parameter-Efficient MLP Ensembling for classification.

    Efficiently imitates an ensemble of MLPs using BatchEnsemble-style
    parameter sharing. One model produces k predictions per sample; during
    training each prediction is trained independently (mean loss across
    ensemble members), during inference softmax probabilities are averaged
    across the k members.

    Parameters
    ----------
    k : int, default=32
        Number of ensemble members.
    n_blocks : int, default=3
        Number of MLP blocks.
    d_block : int, default=256
        Hidden dimension of each block.
    dropout : float, default=0.0
        Dropout rate.
    use_embeddings : bool, default=False
        Whether to use piecewise linear embeddings for numerical features.
    n_bins : int, default=16
        Number of bins for piecewise linear embeddings.
    learning_rate : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    patience : int, default=16
        Early stopping patience (epochs without improvement).
    val_size : float, default=0.2
        Fraction of training data to use for validation/early stopping.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : _TabMNetwork
        Fitted PyTorch model.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.
    feature_importances_ : ndarray
        Feature importances via permutation importance.

    Examples
    --------
    >>> from endgame.models.tabular import TabMClassifier
    >>> clf = TabMClassifier(k=32, n_blocks=3, d_block=256)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.0,
        use_embeddings: bool = False,
        n_bins: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 16,
        val_size: float = 0.2,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[TabM] {message}")

    def _get_device(self) -> torch.device:
        """Get computation device."""
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> TabMClassifier:
        """Fit the TabM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. If not provided, a split of
            the training data is used (controlled by val_size).

        Returns
        -------
        self
            Fitted classifier.
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

        # Preprocess features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        # Store number of features for feature_importances_
        self._n_features = X.shape[1]

        # Validation split
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val_encoded = self._label_encoder.transform(np.asarray(y_val))
            X_val_scaled = self._scaler.transform(X_val)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0).astype(np.float32)
            X_train, y_train = X_scaled, y_encoded
        else:
            try:
                X_train, X_val_scaled, y_train, y_val_encoded = train_test_split(
                    X_scaled, y_encoded,
                    test_size=self.val_size,
                    random_state=self.random_state,
                    stratify=y_encoded,
                )
            except ValueError:
                X_train, X_val_scaled, y_train, y_val_encoded = train_test_split(
                    X_scaled, y_encoded,
                    test_size=self.val_size,
                    random_state=self.random_state,
                )

        # Create tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_encoded, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

        # Create model
        self.model_ = _TabMNetwork(
            n_features=X.shape[1],
            n_outputs=self.n_classes_,
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            use_embeddings=self.use_embeddings,
            n_bins=self.n_bins,
            is_regression=False,
        ).to(self._device)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        self._log(f"Training on {self._device} with k={self.k} ensemble members...")

        for epoch in range(self.n_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                # logits: (batch, k, n_classes)
                logits = self.model_(x_batch)

                # Compute cross-entropy loss for each member independently, then average
                # logits: (batch, k, n_classes) -> (batch * k, n_classes)
                batch_size_actual = logits.shape[0]
                logits_flat = logits.reshape(batch_size_actual * self.k, self.n_classes_)
                # Repeat targets for each member
                y_repeated = y_batch.unsqueeze(1).expand(-1, self.k).reshape(-1)
                loss = F.cross_entropy(logits_flat, y_repeated)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # Validate
            self.model_.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self._device)
                    y_batch = y_batch.to(self._device)

                    logits = self.model_(x_batch)
                    batch_size_actual = logits.shape[0]
                    logits_flat = logits.reshape(batch_size_actual * self.k, self.n_classes_)
                    y_repeated = y_batch.unsqueeze(1).expand(-1, self.k).reshape(-1)
                    loss = F.cross_entropy(logits_flat, y_repeated)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k_name: v.cpu().clone()
                    for k_name, v in self.model_.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                self._log(f"Early stopping at epoch {epoch + 1}")
                break

            scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Compute feature importances via permutation importance
        self._compute_feature_importances(X_val_scaled, y_val_encoded)

        self._is_fitted = True
        return self

    def _compute_feature_importances(self, X_val: np.ndarray, y_val: np.ndarray):
        """Compute feature importances via permutation importance."""
        self.model_.eval()
        baseline_acc = self._score_numpy(X_val, y_val)

        importances = np.zeros(self._n_features)
        rng = np.random.RandomState(self.random_state)

        for j in range(self._n_features):
            X_perm = X_val.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_acc = self._score_numpy(X_perm, y_val)
            importances[j] = max(0.0, baseline_acc - perm_acc)

        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(self._n_features) / self._n_features

        self.feature_importances_ = importances

    def _score_numpy(self, X_scaled: np.ndarray, y_encoded: np.ndarray) -> float:
        """Compute accuracy on already-scaled data."""
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self.model_(X_t)  # (n, k, n_classes)
            proba = F.softmax(logits, dim=-1).mean(dim=1)  # (n, n_classes)
            preds = proba.argmax(dim=1).cpu().numpy()
        return (preds == y_encoded).mean()

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities, averaged across k ensemble members.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        self.model_.eval()
        all_proba = []
        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                x_batch = torch.tensor(
                    X_scaled[start:end], dtype=torch.float32
                ).to(self._device)

                # logits: (batch, k, n_classes)
                logits = self.model_(x_batch)
                # Average softmax probabilities across k members
                proba = F.softmax(logits, dim=-1).mean(dim=1)  # (batch, n_classes)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(class_indices)

    def _check_is_fitted(self):
        """Check if fitted."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("TabMClassifier has not been fitted.")


class TabMRegressor(RegressorMixin, BaseEstimator):
    """TabM: Parameter-Efficient MLP Ensembling for regression.

    Same architecture as TabMClassifier but with MSE loss and
    averaged predictions across k ensemble members.

    Parameters
    ----------
    k : int, default=32
        Number of ensemble members.
    n_blocks : int, default=3
        Number of MLP blocks.
    d_block : int, default=256
        Hidden dimension of each block.
    dropout : float, default=0.0
        Dropout rate.
    use_embeddings : bool, default=False
        Whether to use piecewise linear embeddings for numerical features.
    n_bins : int, default=16
        Number of bins for piecewise linear embeddings.
    learning_rate : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    patience : int, default=16
        Early stopping patience (epochs without improvement).
    val_size : float, default=0.2
        Fraction of training data to use for validation/early stopping.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    model_ : _TabMNetwork
        Fitted PyTorch model.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.
    feature_importances_ : ndarray
        Feature importances via permutation importance.

    Examples
    --------
    >>> from endgame.models.tabular import TabMRegressor
    >>> reg = TabMRegressor(k=32, n_blocks=3, d_block=256)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        k: int = 32,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.0,
        use_embeddings: bool = False,
        n_bins: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 16,
        val_size: float = 0.2,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.k = k
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[TabM] {message}")

    def _get_device(self) -> torch.device:
        """Get computation device."""
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        """Set random seeds for reproducibility."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> TabMRegressor:
        """Fit the TabM regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training target values.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. If not provided, a split of
            the training data is used (controlled by val_size).

        Returns
        -------
        self
            Fitted regressor.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        # Scale target
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y.reshape(-1, 1)).ravel().astype(np.float32)

        # Store number of features for feature_importances_
        self._n_features = X.shape[1]

        # Validation split
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val, dtype=np.float32)
            X_val_scaled = self._scaler.transform(X_val)
            X_val_scaled = np.nan_to_num(X_val_scaled, nan=0.0).astype(np.float32)
            y_val_scaled = self._target_scaler.transform(
                y_val.reshape(-1, 1)
            ).ravel().astype(np.float32)
            X_train, y_train = X_scaled, y_scaled
        else:
            X_train, X_val_scaled, y_train, y_val_scaled = train_test_split(
                X_scaled, y_scaled,
                test_size=self.val_size,
                random_state=self.random_state,
            )

        # Create tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
        y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )
        val_dataset = TensorDataset(X_val_t, y_val_t)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, num_workers=0)

        # Create model
        self.model_ = _TabMNetwork(
            n_features=X.shape[1],
            n_outputs=1,
            k=self.k,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            use_embeddings=self.use_embeddings,
            n_bins=self.n_bins,
            is_regression=True,
        ).to(self._device)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        self._log(f"Training on {self._device} with k={self.k} ensemble members...")

        for epoch in range(self.n_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                # preds: (batch, k, 1)
                preds = self.model_(x_batch)
                preds = preds.squeeze(-1)  # (batch, k)

                # Compute MSE loss for each member independently, then average
                y_expanded = y_batch.unsqueeze(1).expand(-1, self.k)  # (batch, k)
                loss = F.mse_loss(preds, y_expanded)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # Validate
            self.model_.eval()
            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(self._device)
                    y_batch = y_batch.to(self._device)

                    preds = self.model_(x_batch).squeeze(-1)  # (batch, k)
                    y_expanded = y_batch.unsqueeze(1).expand(-1, self.k)
                    loss = F.mse_loss(preds, y_expanded)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k_name: v.cpu().clone()
                    for k_name, v in self.model_.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.patience:
                self._log(f"Early stopping at epoch {epoch + 1}")
                break

            scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Compute feature importances via permutation importance
        self._compute_feature_importances(X_val_scaled, y_val_scaled)

        self._is_fitted = True
        return self

    def _compute_feature_importances(self, X_val: np.ndarray, y_val: np.ndarray):
        """Compute feature importances via permutation importance."""
        self.model_.eval()
        baseline_mse = self._mse_numpy(X_val, y_val)

        importances = np.zeros(self._n_features)
        rng = np.random.RandomState(self.random_state)

        for j in range(self._n_features):
            X_perm = X_val.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_mse = self._mse_numpy(X_perm, y_val)
            importances[j] = max(0.0, perm_mse - baseline_mse)

        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(self._n_features) / self._n_features

        self.feature_importances_ = importances

    def _mse_numpy(self, X_scaled: np.ndarray, y_scaled: np.ndarray) -> float:
        """Compute MSE on already-scaled data."""
        X_t = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            preds = self.model_(X_t).squeeze(-1)  # (n, k)
            preds_avg = preds.mean(dim=1).cpu().numpy()  # (n,)
        return float(np.mean((preds_avg - y_scaled) ** 2))

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values, averaged across k ensemble members.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        self.model_.eval()
        all_preds = []
        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                x_batch = torch.tensor(
                    X_scaled[start:end], dtype=torch.float32
                ).to(self._device)

                # preds: (batch, k, 1)
                preds = self.model_(x_batch).squeeze(-1)  # (batch, k)
                # Average predictions across k members
                preds_avg = preds.mean(dim=1)  # (batch,)
                all_preds.append(preds_avg.cpu().numpy())

        preds_scaled = np.concatenate(all_preds)
        return self._target_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

    def _check_is_fitted(self):
        """Check if fitted."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("TabMRegressor has not been fitted.")
