from __future__ import annotations

"""Modern Neighborhood Component Analysis.

A kNN-based approach with learned distance metric that's surprisingly
competitive with gradient boosting on many tabular tasks.

References
----------
- Goldberger et al. "Neighbourhood Components Analysis" (2004)
- Movshovitz-Attias et al. "No Fuss Distance Metric Learning Using Proxies" (2017)
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
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
        raise ImportError(
            "PyTorch is required for ModernNCA. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _EmbeddingNetwork(nn.Module):
    """Neural network for learning feature embeddings."""

    def __init__(
        self,
        n_features: int,
        embedding_dim: int = 128,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 256]

        layers = []
        in_dim = n_features

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, embedding_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed features into metric space."""
        embeddings = self.network(x)
        # L2 normalize for cosine similarity
        return F.normalize(embeddings, p=2, dim=1)


class ModernNCAClassifier(ClassifierMixin, BaseEstimator):
    """Modern Neighborhood Component Analysis classifier.

    A kNN-based approach with learned distance metric using neural networks.
    Surprisingly competitive with gradient boosting on many tasks.

    The model learns an embedding space where samples of the same class
    are close together and samples of different classes are far apart.
    At inference, it uses kNN in this learned space.

    Parameters
    ----------
    n_neighbors : int, default=32
        Number of neighbors for kNN prediction.
    embedding_dim : int, default=128
        Dimension of learned embedding space.
    hidden_dims : List[int], default=[256, 256]
        Hidden layer dimensions for embedding network.
    temperature : float, default=0.1
        Softmax temperature for neighbor weighting.
    dropout : float, default=0.1
        Dropout rate in embedding network.
    learning_rate : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Training epochs.
    batch_size : int, default=256
        Batch size.
    early_stopping : int, default=15
        Early stopping patience.
    device : str, default='auto'
        Device.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    model_ : _EmbeddingNetwork
        Fitted embedding network.
    train_embeddings_ : ndarray
        Embeddings of training data.
    train_labels_ : ndarray
        Training labels.
    history_ : dict
        Training history.

    Examples
    --------
    >>> clf = ModernNCAClassifier(n_neighbors=32, embedding_dim=128)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    This approach is particularly effective when:
    - The decision boundary is locally smooth
    - Class separation can benefit from learned features
    - You want probabilistic predictions based on neighborhood
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_neighbors: int = 32,
        embedding_dim: int = 128,
        hidden_dims: list[int] = None,
        temperature: float = 0.1,
        dropout: float = 0.1,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 15,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_neighbors = n_neighbors
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims or [256, 256]
        self.temperature = temperature
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: _EmbeddingNetwork | None = None
        self._device = None
        self._label_encoder: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self.train_embeddings_: np.ndarray | None = None
        self.train_labels_: np.ndarray | None = None
        self._X_train: np.ndarray | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[ModernNCA] {msg}")

    def _get_device(self):
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _nca_loss(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """NCA loss: softmax over pairwise distances.

        Maximizes probability that a point is correctly classified
        by its neighbors in embedding space.
        """
        batch_size = embeddings.shape[0]

        # Compute pairwise distances (squared Euclidean)
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a.b
        # Since embeddings are L2-normalized, ||a||^2 = ||b||^2 = 1
        # Distance = 2 - 2*a.b = 2*(1 - cosine_similarity)
        similarities = torch.mm(embeddings, embeddings.t())  # Cosine similarity
        distances = 2 * (1 - similarities)  # Squared Euclidean distance

        # Apply temperature scaling
        scaled_distances = -distances / self.temperature

        # Mask self-similarity (diagonal)
        mask = torch.eye(batch_size, device=embeddings.device).bool()
        scaled_distances = scaled_distances.masked_fill(mask, float('-inf'))

        # Softmax over distances to get "probability" of selecting each neighbor
        neighbor_probs = F.softmax(scaled_distances, dim=1)

        # For each sample, compute probability of correct classification
        # = sum of probs for neighbors of same class
        labels_match = labels.unsqueeze(0) == labels.unsqueeze(1)  # (batch, batch)
        labels_match = labels_match.float()
        labels_match = labels_match.masked_fill(mask, 0)  # Exclude self

        # Probability of correct classification
        correct_probs = (neighbor_probs * labels_match).sum(dim=1)

        # Avoid log(0)
        correct_probs = correct_probs.clamp(min=1e-10)

        # NCA loss: negative log probability
        loss = -torch.log(correct_probs).mean()

        return loss

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> ModernNCAClassifier:
        """Fit the ModernNCA classifier.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Training labels.
        eval_set : tuple, optional
            Validation set.

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

        # Store for prediction
        self._X_train = X_scaled
        self.train_labels_ = y_encoded

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create model
        self.model_ = _EmbeddingNetwork(
            n_features=X.shape[1],
            embedding_dim=self.embedding_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self._device)

        # Data loader
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self._scaler.transform(np.asarray(X_val, dtype=np.float32))
            X_val = np.nan_to_num(X_val, nan=0.0)
            y_val_encoded = self._label_encoder.transform(y_val)

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
                embeddings = self.model_(x_batch)
                loss = self._nca_loss(embeddings, y_batch)
                loss.backward()
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

                        embeddings = self.model_(x_batch)
                        loss = self._nca_loss(embeddings, y_batch)
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

        # Compute training embeddings for kNN
        self._compute_train_embeddings()

        self._is_fitted = True
        return self

    def _compute_train_embeddings(self):
        """Compute embeddings for all training samples."""
        self.model_.eval()
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(self._X_train), self.batch_size):
                end = min(start + self.batch_size, len(self._X_train))
                x_batch = torch.tensor(
                    self._X_train[start:end],
                    dtype=torch.float32
                ).to(self._device)

                embeddings = self.model_(x_batch)
                all_embeddings.append(embeddings.cpu().numpy())

        self.train_embeddings_ = np.vstack(all_embeddings)

    def _embed(self, X: np.ndarray) -> np.ndarray:
        """Embed features using the learned network."""
        X_scaled = self._scaler.transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        self.model_.eval()
        all_embeddings = []

        with torch.no_grad():
            for start in range(0, len(X_scaled), self.batch_size):
                end = min(start + self.batch_size, len(X_scaled))
                x_batch = torch.tensor(
                    X_scaled[start:end],
                    dtype=torch.float32
                ).to(self._device)

                embeddings = self.model_(x_batch)
                all_embeddings.append(embeddings.cpu().numpy())

        return np.vstack(all_embeddings)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using kNN in embedding space."""
        if not self._is_fitted:
            raise RuntimeError("ModernNCAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float32)

        # Embed test samples
        test_embeddings = self._embed(X)

        # Compute distances to all training samples (cosine distance)
        # Since embeddings are normalized, cosine_dist = 1 - dot_product
        similarities = np.dot(test_embeddings, self.train_embeddings_.T)

        # Get k nearest neighbors
        k = min(self.n_neighbors, len(self.train_embeddings_))

        proba = np.zeros((len(X), self.n_classes_))

        for i in range(len(X)):
            # Get top-k most similar
            top_k_idx = np.argsort(similarities[i])[-k:]
            top_k_sim = similarities[i, top_k_idx]
            top_k_labels = self.train_labels_[top_k_idx]

            # Softmax weighting with numerical stability
            scaled_sim = top_k_sim / self.temperature
            scaled_sim = scaled_sim - np.max(scaled_sim)  # For numerical stability
            weights = np.exp(scaled_sim)
            weights = weights / weights.sum()

            # Weighted voting
            for j, (label, weight) in enumerate(zip(top_k_labels, weights)):
                proba[i, label] += weight

        # Ensure probabilities are valid (clamp to [0, 1] and normalize)
        proba = np.clip(proba, 0.0, 1.0)
        row_sums = proba.sum(axis=1, keepdims=True)
        # Avoid division by zero
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        proba = proba / row_sums

        return proba

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def transform(self, X) -> np.ndarray:
        """Transform features to embedding space.

        Parameters
        ----------
        X : array-like
            Input features.

        Returns
        -------
        ndarray
            Learned embeddings.
        """
        if not self._is_fitted:
            raise RuntimeError("ModernNCAClassifier has not been fitted.")

        return self._embed(np.asarray(X, dtype=np.float32))
