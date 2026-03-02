from __future__ import annotations

"""Denoising Autoencoder for tabular representation learning.

DAE is a key technique from Tabular Playground Series 1st place solutions.
It corrupts input with swap noise, trains to reconstruct original, and
extracts bottleneck embeddings as new features.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# PyTorch imports (lazy loaded)
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for DenoisingAutoEncoder. "
            "Install with: pip install torch"
        )


# Only define the module class if torch is available
_DAEModule = None

if HAS_TORCH:
    class _DAEModule(nn.Module):
        """PyTorch Denoising Autoencoder module.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : List[int]
            Encoder hidden layer dimensions (decoder mirrors).
        dropout : float
            Dropout rate.
        activation : str
            Activation function.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            dropout: float = 0.1,
            activation: str = "relu",
        ):
            super().__init__()

            self.activation_fn = self._get_activation(activation)

            # Build encoder
            encoder_layers = []
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                encoder_layers.append(nn.BatchNorm1d(hidden_dim))
                encoder_layers.append(self.activation_fn)
                encoder_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            self.encoder = nn.Sequential(*encoder_layers)

            # Build decoder (mirror of encoder)
            decoder_layers = []
            decoder_dims = hidden_dims[::-1][1:] + [input_dim]
            for hidden_dim in decoder_dims[:-1]:
                decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
                decoder_layers.append(nn.BatchNorm1d(hidden_dim))
                decoder_layers.append(self.activation_fn)
                decoder_layers.append(nn.Dropout(dropout))
                prev_dim = hidden_dim

            # Final layer without activation
            decoder_layers.append(nn.Linear(prev_dim, decoder_dims[-1]))
            self.decoder = nn.Sequential(*decoder_layers)

            self.bottleneck_dim = hidden_dims[-1]

        def _get_activation(self, activation: str) -> nn.Module:
            """Get activation function by name."""
            activations = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.1),
                "elu": nn.ELU(),
                "selu": nn.SELU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "tanh": nn.Tanh(),
            }
            if activation not in activations:
                raise ValueError(f"Unknown activation: {activation}")
            return activations[activation]

        def encode(self, x: torch.Tensor) -> torch.Tensor:
            """Encode input to bottleneck representation."""
            return self.encoder(x)

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            """Decode bottleneck to reconstruction."""
            return self.decoder(z)

        def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            """Forward pass returning both encoding and reconstruction."""
            z = self.encode(x)
            x_reconstructed = self.decode(z)
            return z, x_reconstructed


class DenoisingAutoEncoder(BaseEstimator, TransformerMixin):
    """Denoising Autoencoder for tabular representation learning.

    Corrupts input with swap noise (randomly swapping values between samples),
    trains to reconstruct the original input, and extracts bottleneck layer
    embeddings as new features.

    This is a key technique from Tabular Playground Series 1st place solutions.

    Parameters
    ----------
    hidden_dims : List[int], default=[256, 128, 64]
        Architecture of encoder (decoder mirrors).
        The last dimension is the bottleneck/embedding size.
    noise_fraction : float, default=0.1
        Fraction of features to corrupt with swap noise.
    dropout : float, default=0.1
        Dropout rate for regularization.
    activation : str, default='relu'
        Activation function: 'relu', 'leaky_relu', 'elu', 'selu',
        'gelu', 'swish', 'tanh'.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    early_stopping : int, default=10
        Patience for early stopping (based on reconstruction loss).
    scheduler : str, default='cosine'
        Learning rate scheduler: 'cosine', 'step', 'none'.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto' (auto-detect GPU).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    model_ : _DAEModule
        Fitted PyTorch DAE model.
    scaler_ : StandardScaler
        Feature scaler.
    n_features_in_ : int
        Number of input features.
    embedding_dim_ : int
        Dimension of the learned embeddings.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.preprocessing import DenoisingAutoEncoder
    >>> # Create DAE with 64-dimensional embeddings
    >>> dae = DenoisingAutoEncoder(hidden_dims=[256, 128, 64], n_epochs=50)
    >>> # Fit on training data
    >>> dae.fit(X_train)
    >>> # Extract embeddings as new features
    >>> X_train_embed = dae.transform(X_train)
    >>> X_test_embed = dae.transform(X_test)
    >>> # Combine with original features
    >>> X_train_enriched = np.hstack([X_train, X_train_embed])
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        noise_fraction: float = 0.1,
        dropout: float = 0.1,
        activation: str = "relu",
        n_epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping: int = 10,
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_torch()

        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.noise_fraction = noise_fraction
        self.dropout = dropout
        self.activation = activation
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Model components
        self.model_: _DAEModule | None = None
        self.scaler_: StandardScaler | None = None
        self._device: torch.device | None = None
        self.n_features_in_: int | None = None
        self.embedding_dim_: int | None = None
        self.history_: dict = {"train_loss": [], "val_loss": []}

    def _log(self, message: str):
        """Log a message if verbose."""
        if self.verbose:
            print(f"[DAE] {message}")

    def _get_device(self) -> torch.device:
        """Get computation device."""
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

    def _to_numpy(self, X) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, np.ndarray):
            return X

        try:
            import pandas as pd
            if isinstance(X, (pd.DataFrame, pd.Series)):
                return X.values
        except ImportError:
            pass

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                return X.to_numpy()
        except ImportError:
            pass

        return np.asarray(X)

    def _apply_swap_noise(
        self,
        X: torch.Tensor,
        noise_fraction: float,
    ) -> torch.Tensor:
        """Apply swap noise by randomly swapping values between samples.

        Parameters
        ----------
        X : Tensor
            Input data of shape (batch_size, n_features).
        noise_fraction : float
            Fraction of features to corrupt.

        Returns
        -------
        Tensor
            Corrupted data.
        """
        batch_size, n_features = X.shape
        n_corrupt = int(n_features * noise_fraction)

        X_corrupted = X.clone()

        for i in range(batch_size):
            # Select features to corrupt
            corrupt_features = torch.randperm(n_features)[:n_corrupt]
            # Select random other samples to swap from
            swap_indices = torch.randint(0, batch_size, (n_corrupt,))
            # Swap values
            X_corrupted[i, corrupt_features] = X[swap_indices, corrupt_features]

        return X_corrupted

    def _get_scheduler(
        self,
        optimizer: optim.Optimizer,
        n_epochs: int,
    ) -> Any | None:
        """Create learning rate scheduler."""
        if self.scheduler == "none":
            return None
        elif self.scheduler == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=1e-6
            )
        elif self.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=n_epochs // 3, gamma=0.1
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def _train_epoch(
        self,
        X: torch.Tensor,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        self.model_.train()
        n_samples = X.shape[0]
        indices = torch.randperm(n_samples)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_samples, self.batch_size):
            end = min(start + self.batch_size, n_samples)
            batch_indices = indices[start:end]
            X_batch = X[batch_indices].to(self._device)

            # Apply swap noise
            X_corrupted = self._apply_swap_noise(X_batch, self.noise_fraction)

            optimizer.zero_grad()
            _, X_reconstructed = self.model_(X_corrupted)
            loss = criterion(X_reconstructed, X_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(
        self,
        X: torch.Tensor,
        criterion: nn.Module,
    ) -> float:
        """Compute validation loss (without noise)."""
        self.model_.eval()
        n_samples = X.shape[0]
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = X[start:end].to(self._device)

                _, X_reconstructed = self.model_(X_batch)
                loss = criterion(X_reconstructed, X_batch)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def fit(self, X, y=None) -> DenoisingAutoEncoder:
        """Fit the Denoising Autoencoder.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self
            Fitted transformer.
        """
        self._set_seed()
        self._device = self._get_device()

        X_arr = self._to_numpy(X).astype(np.float32)
        self.n_features_in_ = X_arr.shape[1]

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_arr)
        X_tensor = torch.FloatTensor(X_scaled)

        # Split for validation (10%)
        n_samples = X_tensor.shape[0]
        n_val = max(1, int(n_samples * 0.1))
        indices = torch.randperm(n_samples)
        X_train = X_tensor[indices[n_val:]]
        X_val = X_tensor[indices[:n_val]]

        # Create model
        self.model_ = _DAEModule(
            input_dim=self.n_features_in_,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            activation=self.activation,
        ).to(self._device)

        self.embedding_dim_ = self.model_.bottleneck_dim

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._get_scheduler(optimizer, self.n_epochs)
        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training DAE on {self._device}...")
        self._log(f"Input dim: {self.n_features_in_}, Embedding dim: {self.embedding_dim_}")

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(X_train, optimizer, criterion)
            val_loss = self._validate_epoch(X_val, criterion)

            self.history_["train_loss"].append(train_loss)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch + 1}/{self.n_epochs}: "
                    f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                )

            if patience_counter >= self.early_stopping:
                self._log(f"Early stopping at epoch {epoch + 1}")
                break

            if scheduler is not None:
                scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def transform(self, X) -> np.ndarray:
        """Extract bottleneck embeddings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        ndarray of shape (n_samples, embedding_dim)
            Bottleneck embeddings.
        """
        if self.model_ is None:
            raise RuntimeError("DenoisingAutoEncoder has not been fitted.")

        X_arr = self._to_numpy(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_arr)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        self.model_.eval()
        embeddings = []

        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                X_batch = X_tensor[start:end]
                z = self.model_.encode(X_batch)
                embeddings.append(z.cpu().numpy())

        return np.vstack(embeddings)

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used.

        Returns
        -------
        ndarray of shape (n_samples, embedding_dim)
            Bottleneck embeddings.
        """
        return self.fit(X, y).transform(X)

    def reconstruct(self, X) -> np.ndarray:
        """Reconstruct input from embeddings.

        Useful for detecting anomalies (high reconstruction error).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to reconstruct.

        Returns
        -------
        ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        if self.model_ is None:
            raise RuntimeError("DenoisingAutoEncoder has not been fitted.")

        X_arr = self._to_numpy(X).astype(np.float32)
        X_scaled = self.scaler_.transform(X_arr)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        self.model_.eval()
        reconstructed = []

        with torch.no_grad():
            for start in range(0, X_tensor.shape[0], self.batch_size):
                end = min(start + self.batch_size, X_tensor.shape[0])
                X_batch = X_tensor[start:end]
                _, X_recon = self.model_(X_batch)
                reconstructed.append(X_recon.cpu().numpy())

        X_recon_scaled = np.vstack(reconstructed)
        return self.scaler_.inverse_transform(X_recon_scaled)

    def reconstruction_error(self, X) -> np.ndarray:
        """Compute per-sample reconstruction error.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate.

        Returns
        -------
        ndarray of shape (n_samples,)
            Mean squared reconstruction error per sample.
        """
        X_arr = self._to_numpy(X).astype(np.float32)
        X_recon = self.reconstruct(X_arr)
        return np.mean((X_arr - X_recon) ** 2, axis=1)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : ignored
            Not used.

        Returns
        -------
        List[str]
            Output feature names.
        """
        if self.embedding_dim_ is None:
            raise RuntimeError("DenoisingAutoEncoder has not been fitted.")
        return [f"dae_embed_{i}" for i in range(self.embedding_dim_)]
