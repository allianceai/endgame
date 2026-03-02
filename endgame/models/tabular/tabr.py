"""TabR: Retrieval-Augmented Tabular Deep Learning.

TabR enhances deep learning predictions by retrieving similar training
examples at inference time, combining the strengths of kNN and neural
networks for tabular data.

References
----------
- Gorishniy et al. "TabR: Tabular Deep Learning Meets Nearest Neighbors" (2024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
            "PyTorch is required for TabR. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _MLP(_nn_Module):
    """Simple MLP building block."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        out_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        current_dim = in_dim
        for i in range(n_layers - 1):
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _TabRNetwork(_nn_Module):
    """Core PyTorch architecture for TabR.

    The network consists of:
    1. Feature embedding: linear projection to d_model dimension
    2. Context encoder: MLP producing keys/values for retrieval
    3. Query encoder: MLP producing queries
    4. kNN retrieval: find k nearest neighbors using cosine similarity
    5. Attention-based aggregation of retrieved neighbors
    6. Final prediction head (MLP)

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes (1 for regression).
    d_model : int, default=192
        Embedding dimension.
    n_heads : int, default=8
        Number of attention heads for neighbor aggregation.
    n_layers : int, default=2
        Number of layers in encoder MLPs.
    k_neighbors : int, default=96
        Number of nearest neighbors to retrieve.
    dropout : float, default=0.0
        Dropout rate.
    is_regression : bool, default=False
        Whether this is a regression task.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 2,
        k_neighbors: int = 96,
        dropout: float = 0.0,
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.d_model = d_model
        self.n_heads = n_heads
        self.k_neighbors = k_neighbors
        self.is_regression = is_regression

        # Feature embedding: project raw features to d_model
        self.feature_embedding = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Context encoder: produces keys and values for retrieval
        self.context_encoder = _MLP(
            in_dim=d_model,
            hidden_dim=d_model,
            out_dim=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Query encoder: produces queries for retrieval
        self.query_encoder = _MLP(
            in_dim=d_model,
            hidden_dim=d_model,
            out_dim=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Attention for aggregating retrieved neighbor information
        self.retrieval_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(d_model)

        # Final prediction head
        output_dim = 1 if is_regression else n_classes
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim),
        )

        # Stored training context (set after fitting)
        self._context_keys: torch.Tensor | None = None  # (N_train, d_model)
        self._context_labels: torch.Tensor | None = None  # (N_train,) or (N_train, n_classes)

    def set_context(
        self,
        context_keys: torch.Tensor,
        context_labels: torch.Tensor,
    ):
        """Store training set context for retrieval.

        Parameters
        ----------
        context_keys : torch.Tensor of shape (n_train, d_model)
            Encoded training set representations (keys).
        context_labels : torch.Tensor
            Training labels (encoded as integers for classification,
            float for regression).
        """
        self._context_keys = context_keys.detach()
        self._context_labels = context_labels.detach()

    def _retrieve_neighbors(
        self,
        queries: torch.Tensor,
        candidate_keys: torch.Tensor,
        candidate_labels: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Retrieve k nearest neighbors using cosine similarity.

        Parameters
        ----------
        queries : (batch_size, d_model)
        candidate_keys : (n_candidates, d_model)
        candidate_labels : (n_candidates,)
        k : int

        Returns
        -------
        neighbor_keys : (batch_size, k, d_model)
        neighbor_labels : (batch_size, k)
        similarities : (batch_size, k)
        """
        # Normalize for cosine similarity
        q_norm = F.normalize(queries, p=2, dim=-1)
        k_norm = F.normalize(candidate_keys, p=2, dim=-1)

        # Compute similarities: (batch_size, n_candidates)
        sims = torch.mm(q_norm, k_norm.t())

        # Get top-k neighbors
        actual_k = min(k, candidate_keys.shape[0])
        top_sims, top_indices = torch.topk(sims, actual_k, dim=-1)

        # Gather neighbor keys and labels
        neighbor_keys = candidate_keys[top_indices]  # (batch, k, d_model)
        neighbor_labels = candidate_labels[top_indices]  # (batch, k)

        return neighbor_keys, neighbor_labels, top_sims

    def forward(
        self,
        x: torch.Tensor,
        candidate_keys: torch.Tensor | None = None,
        candidate_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with retrieval-augmented prediction.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)
            Input features.
        candidate_keys : torch.Tensor, optional
            Override context keys for retrieval. If None, uses stored context.
        candidate_labels : torch.Tensor, optional
            Override context labels. If None, uses stored context.

        Returns
        -------
        logits : torch.Tensor of shape (batch_size, n_classes) or (batch_size, 1)
        """
        batch_size = x.shape[0]

        # Step 1: Embed features
        embeddings = self.feature_embedding(x)  # (batch, d_model)

        # Step 2: Encode query and context representations
        queries = self.query_encoder(embeddings)  # (batch, d_model)
        context_repr = self.context_encoder(embeddings)  # (batch, d_model)

        # Step 3: Determine retrieval candidates
        if candidate_keys is None:
            candidate_keys = self._context_keys
        if candidate_labels is None:
            candidate_labels = self._context_labels

        if candidate_keys is not None and candidate_labels is not None:
            # Step 4: Retrieve k nearest neighbors
            neighbor_keys, neighbor_labels, similarities = self._retrieve_neighbors(
                queries, candidate_keys, candidate_labels, self.k_neighbors
            )

            # Step 5: Attention-based aggregation of neighbors
            # queries: (batch, 1, d_model) as the query token
            # neighbor_keys: (batch, k, d_model) as key/value tokens
            query_for_attn = queries.unsqueeze(1)  # (batch, 1, d_model)
            attn_out, _ = self.retrieval_attention(
                query_for_attn, neighbor_keys, neighbor_keys
            )
            attn_out = self.attn_norm(attn_out.squeeze(1))  # (batch, d_model)

            # Step 6: Combine context representation with retrieval output
            combined = torch.cat([context_repr, attn_out], dim=-1)  # (batch, d_model*2)
        else:
            # No retrieval candidates available - use zero retrieval signal
            zero_retrieval = torch.zeros_like(context_repr)
            combined = torch.cat([context_repr, zero_retrieval], dim=-1)

        # Step 7: Prediction head
        logits = self.prediction_head(combined)

        return logits

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode inputs to context key representations.

        Used for building the retrieval index from training data.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, n_features)

        Returns
        -------
        keys : torch.Tensor of shape (batch_size, d_model)
        """
        embeddings = self.feature_embedding(x)
        return self.context_encoder(embeddings)


class TabRClassifier(ClassifierMixin, BaseEstimator):
    """TabR: Retrieval-Augmented Tabular Deep Learning Classifier.

    Enhances deep learning predictions by retrieving similar training
    examples at inference time, combining the strengths of kNN and
    neural networks.

    Parameters
    ----------
    d_model : int, default=192
        Embedding dimension.
    n_heads : int, default=8
        Number of attention heads for neighbor aggregation.
    n_layers : int, default=2
        Number of layers in encoder MLPs.
    k_neighbors : int, default=96
        Number of nearest neighbors to retrieve.
    dropout : float, default=0.0
        Dropout rate.
    lr : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    patience : int, default=16
        Early stopping patience (epochs without improvement).
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
    model_ : _TabRNetwork
        Fitted PyTorch model.
    feature_importances_ : ndarray of shape (n_features,)
        Permutation-based feature importances on the validation set.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.tabular import TabRClassifier
    >>> clf = TabRClassifier(d_model=192, k_neighbors=96)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 2,
        k_neighbors: int = 96,
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 16,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: _TabRNetwork | None = None
        self._device: torch.device | None = None
        self._label_encoder: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self._train_X_scaled: np.ndarray | None = None
        self._train_y_encoded: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[TabR] {message}")

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

    def _build_context_index(self, X_scaled: np.ndarray, y_encoded: np.ndarray):
        """Build the retrieval context index from training data.

        Encodes training data through the context encoder and stores
        the resulting keys along with labels for retrieval at
        inference time.
        """
        self.model_.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long).to(self._device)

        # Encode training data in batches to avoid memory issues
        all_keys = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                batch_keys = self.model_.encode(X_tensor[start:end])
                all_keys.append(batch_keys)

        context_keys = torch.cat(all_keys, dim=0)
        self.model_.set_context(context_keys, y_tensor)

    def _compute_feature_importances(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute permutation-based feature importances on validation set.

        Parameters
        ----------
        X_val : ndarray of shape (n_samples, n_features)
            Scaled validation features.
        y_val : ndarray of shape (n_samples,)
            Encoded validation labels.

        Returns
        -------
        importances : ndarray of shape (n_features,)
        """
        self.model_.eval()
        n_features = X_val.shape[0 + 1]

        # Compute baseline accuracy
        preds_baseline = self._predict_from_scaled(X_val)
        baseline_acc = np.mean(preds_baseline == y_val)

        importances = np.zeros(n_features)
        rng = np.random.RandomState(self.random_state)

        for feat_idx in range(n_features):
            X_permuted = X_val.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            preds_permuted = self._predict_from_scaled(X_permuted)
            permuted_acc = np.mean(preds_permuted == y_val)
            importances[feat_idx] = baseline_acc - permuted_acc

        # Normalize to sum to 1, handling negative values
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(n_features) / n_features

        return importances

    def _predict_from_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict class indices from already-scaled features."""
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        all_preds = []

        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                logits = self.model_(X_tensor[start:end])
                preds = logits.argmax(dim=-1)
                all_preds.append(preds.cpu().numpy())

        return np.concatenate(all_preds)

    def fit(self, X, y, **fit_params) -> TabRClassifier:
        """Fit the TabR classifier.

        Trains the retrieval-augmented model with early stopping on
        a 20% validation split from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Fitted classifier.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()
        self.history_ = {"train_loss": [], "val_loss": []}

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        # Train/validation split for early stopping
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2,
                random_state=self.random_state, stratify=y_encoded,
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X_scaled, y_encoded, test_size=0.2,
                random_state=self.random_state,
            )

        # Store full training data for retrieval context
        self._train_X_scaled = X_scaled
        self._train_y_encoded = y_encoded

        n_features = X.shape[1]

        # Create model
        self.model_ = _TabRNetwork(
            n_features=n_features,
            n_classes=self.n_classes_,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            k_neighbors=self.k_neighbors,
            dropout=self.dropout,
            is_regression=False,
        ).to(self._device)

        # Prepare tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        # Optimizer
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs,
        )

        # Training loop with early stopping
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device} with {X_train.shape[0]} samples...")

        for epoch in range(self.n_epochs):
            # --- Train phase ---
            self.model_.train()

            # Build context index from training split for retrieval during training
            self.model_.eval()
            with torch.no_grad():
                train_keys_list = []
                for start in range(0, X_train_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_train_t.shape[0])
                    batch = X_train_t[start:end].to(self._device)
                    train_keys_list.append(self.model_.encode(batch))
                context_keys = torch.cat(train_keys_list, dim=0)
            context_labels = y_train_t.to(self._device)
            self.model_.set_context(context_keys, context_labels)

            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # --- Validation phase ---
            self.model_.eval()

            # Rebuild context for validation (using updated encoder)
            with torch.no_grad():
                train_keys_list = []
                for start in range(0, X_train_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_train_t.shape[0])
                    batch = X_train_t[start:end].to(self._device)
                    train_keys_list.append(self.model_.encode(batch))
                context_keys = torch.cat(train_keys_list, dim=0)
            self.model_.set_context(context_keys, context_labels)

            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for start in range(0, X_val_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_val_t.shape[0])
                    X_batch = X_val_t[start:end].to(self._device)
                    y_batch = y_val_t[start:end].to(self._device)

                    logits = self.model_(X_batch)
                    loss = F.cross_entropy(logits, y_batch)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
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

        # Build final context index from ALL training data
        self._build_context_index(self._train_X_scaled, self._train_y_encoded)

        # Compute feature importances on validation set
        self.feature_importances_ = self._compute_feature_importances(
            X_val, y_val
        )

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                X_batch = X_tensor[start:end].to(self._device)
                logits = self.model_(X_batch)
                proba = F.softmax(logits, dim=1)
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
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(class_indices)

    def _check_is_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "TabRClassifier has not been fitted. Call fit() first."
            )


class TabRRegressor(RegressorMixin, BaseEstimator):
    """TabR: Retrieval-Augmented Tabular Deep Learning Regressor.

    Same retrieval-augmented architecture as TabRClassifier, adapted
    for regression with MSE loss and target standardization.

    Parameters
    ----------
    d_model : int, default=192
        Embedding dimension.
    n_heads : int, default=8
        Number of attention heads for neighbor aggregation.
    n_layers : int, default=2
        Number of layers in encoder MLPs.
    k_neighbors : int, default=96
        Number of nearest neighbors to retrieve.
    dropout : float, default=0.0
        Dropout rate.
    lr : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    patience : int, default=16
        Early stopping patience (epochs without improvement).
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    model_ : _TabRNetwork
        Fitted PyTorch model.
    feature_importances_ : ndarray of shape (n_features,)
        Permutation-based feature importances on the validation set.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.tabular import TabRRegressor
    >>> reg = TabRRegressor(d_model=192, k_neighbors=96)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        d_model: int = 192,
        n_heads: int = 8,
        n_layers: int = 2,
        k_neighbors: int = 96,
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        patience: int = 16,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.k_neighbors = k_neighbors
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: _TabRNetwork | None = None
        self._device: torch.device | None = None
        self._scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._train_X_scaled: np.ndarray | None = None
        self._train_y_scaled: np.ndarray | None = None
        self.feature_importances_: np.ndarray | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[TabR] {message}")

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

    def _build_context_index(self, X_scaled: np.ndarray, y_scaled: np.ndarray):
        """Build the retrieval context index from training data."""
        self.model_.eval()
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32).to(self._device)

        all_keys = []
        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                batch_keys = self.model_.encode(X_tensor[start:end])
                all_keys.append(batch_keys)

        context_keys = torch.cat(all_keys, dim=0)
        self.model_.set_context(context_keys, y_tensor)

    def _predict_from_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict raw (scaled) values from already-scaled features."""
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self._device)
        all_preds = []

        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                logits = self.model_(X_tensor[start:end])
                all_preds.append(logits.squeeze(-1).cpu().numpy())

        return np.concatenate(all_preds)

    def _compute_feature_importances(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> np.ndarray:
        """Compute permutation-based feature importances on validation set.

        Parameters
        ----------
        X_val : ndarray of shape (n_samples, n_features)
            Scaled validation features.
        y_val : ndarray of shape (n_samples,)
            Scaled validation targets.

        Returns
        -------
        importances : ndarray of shape (n_features,)
        """
        self.model_.eval()
        n_features = X_val.shape[1]

        # Compute baseline MSE
        preds_baseline = self._predict_from_scaled(X_val)
        baseline_mse = np.mean((preds_baseline - y_val) ** 2)

        importances = np.zeros(n_features)
        rng = np.random.RandomState(self.random_state)

        for feat_idx in range(n_features):
            X_permuted = X_val.copy()
            X_permuted[:, feat_idx] = rng.permutation(X_permuted[:, feat_idx])
            preds_permuted = self._predict_from_scaled(X_permuted)
            permuted_mse = np.mean((preds_permuted - y_val) ** 2)
            # Higher MSE after permutation => more important feature
            importances[feat_idx] = permuted_mse - baseline_mse

        # Normalize
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(n_features) / n_features

        return importances

    def fit(self, X, y, **fit_params) -> TabRRegressor:
        """Fit the TabR regressor.

        Trains the retrieval-augmented model with early stopping on
        a 20% validation split from the training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.

        Returns
        -------
        self
            Fitted regressor.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()
        self.history_ = {"train_loss": [], "val_loss": []}

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        # Standardize target
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y_scaled, test_size=0.2,
            random_state=self.random_state,
        )

        # Store full training data for retrieval context
        self._train_X_scaled = X_scaled
        self._train_y_scaled = y_scaled

        n_features = X.shape[1]

        # Create model
        self.model_ = _TabRNetwork(
            n_features=n_features,
            n_classes=1,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            k_neighbors=self.k_neighbors,
            dropout=self.dropout,
            is_regression=True,
        ).to(self._device)

        # Prepare tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
        )

        # Optimizer
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device} with {X_train.shape[0]} samples...")

        for epoch in range(self.n_epochs):
            # --- Train phase ---
            # Build context from training split
            self.model_.eval()
            with torch.no_grad():
                train_keys_list = []
                for start in range(0, X_train_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_train_t.shape[0])
                    batch = X_train_t[start:end].to(self._device)
                    train_keys_list.append(self.model_.encode(batch))
                context_keys = torch.cat(train_keys_list, dim=0)
            context_labels = y_train_t.to(self._device)
            self.model_.set_context(context_keys, context_labels)

            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self.model_(X_batch).squeeze(-1)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # --- Validation phase ---
            self.model_.eval()

            # Rebuild context with updated encoder
            with torch.no_grad():
                train_keys_list = []
                for start in range(0, X_train_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_train_t.shape[0])
                    batch = X_train_t[start:end].to(self._device)
                    train_keys_list.append(self.model_.encode(batch))
                context_keys = torch.cat(train_keys_list, dim=0)
            self.model_.set_context(context_keys, context_labels)

            val_loss = 0.0
            n_val_batches = 0

            with torch.no_grad():
                for start in range(0, X_val_t.shape[0], self.batch_size):
                    end = min(start + self.batch_size, X_val_t.shape[0])
                    X_batch = X_val_t[start:end].to(self._device)
                    y_batch = y_val_t[start:end].to(self._device)

                    pred = self.model_(X_batch).squeeze(-1)
                    loss = F.mse_loss(pred, y_batch)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
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

        # Build final context index from ALL training data
        self._build_context_index(self._train_X_scaled, self._train_y_scaled)

        # Compute feature importances on validation set
        self.feature_importances_ = self._compute_feature_importances(
            X_val, y_val
        )

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0).astype(np.float32)

        pred_scaled = self._predict_from_scaled(X_scaled)
        return self._target_scaler.inverse_transform(
            pred_scaled.reshape(-1, 1)
        ).ravel()

    def _check_is_fitted(self):
        """Check if the model has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                "TabRRegressor has not been fitted. Call fit() first."
            )
