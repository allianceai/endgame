"""Variational Autoencoder for Dimensionality Reduction.

This module implements a VAE-based approach to dimensionality reduction,
learning a smooth latent space that can be used for both reduction and
generation.

Classes
-------
VAEReducer : Variational Autoencoder for dimensionality reduction
"""


import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def _check_torch_installed():
    """Raise ImportError if PyTorch is not installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for VAEReducer. "
            "Install it with: pip install torch"
        )


class _VAEModule(nn.Module):
    """Variational Autoencoder neural network module.

    Architecture:
    - Encoder: Input -> Hidden layers -> (mu, log_var)
    - Reparameterization: z = mu + std * epsilon
    - Decoder: z -> Hidden layers -> Reconstruction
    """

    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        encoder_layers: list[int],
        decoder_layers: list[int],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # Activation function
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "leaky_relu":
            self.activation = nn.LeakyReLU(0.2)
        elif activation == "elu":
            self.activation = nn.ELU()
        elif activation == "tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # Build encoder
        encoder_modules = []
        prev_dim = input_dim
        for hidden_dim in encoder_layers:
            encoder_modules.append(nn.Linear(prev_dim, hidden_dim))
            encoder_modules.append(nn.BatchNorm1d(hidden_dim))
            encoder_modules.append(self.activation)
            if dropout > 0:
                encoder_modules.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*encoder_modules)

        # Latent space parameters
        self.fc_mu = nn.Linear(prev_dim, latent_dim)
        self.fc_log_var = nn.Linear(prev_dim, latent_dim)

        # Build decoder
        decoder_modules = []
        prev_dim = latent_dim
        for hidden_dim in decoder_layers:
            decoder_modules.append(nn.Linear(prev_dim, hidden_dim))
            decoder_modules.append(nn.BatchNorm1d(hidden_dim))
            decoder_modules.append(self.activation)
            if dropout > 0:
                decoder_modules.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        decoder_modules.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_modules)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        mu : Tensor of shape (batch, latent_dim)
            Mean of the latent distribution.
        log_var : Tensor of shape (batch, latent_dim)
            Log variance of the latent distribution.
        """
        h = self.encoder(x)
        mu = self.fc_mu(h)
        log_var = self.fc_log_var(h)
        return mu, log_var

    def reparameterize(
        self, mu: torch.Tensor, log_var: torch.Tensor
    ) -> torch.Tensor:
        """Reparameterization trick for backpropagation.

        Parameters
        ----------
        mu : Tensor of shape (batch, latent_dim)
        log_var : Tensor of shape (batch, latent_dim)

        Returns
        -------
        z : Tensor of shape (batch, latent_dim)
            Sampled latent vector.
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction.

        Parameters
        ----------
        z : Tensor of shape (batch, latent_dim)

        Returns
        -------
        x_recon : Tensor of shape (batch, input_dim)
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, input_dim)

        Returns
        -------
        x_recon : Tensor of shape (batch, input_dim)
            Reconstructed input.
        mu : Tensor of shape (batch, latent_dim)
            Latent mean.
        log_var : Tensor of shape (batch, latent_dim)
            Latent log variance.
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_recon = self.decode(z)
        return x_recon, mu, log_var


class VAEReducer(TransformerMixin, BaseEstimator):
    """Variational Autoencoder for Dimensionality Reduction.

    VAE learns a probabilistic mapping to a lower-dimensional latent space
    with a smooth structure, useful for both dimensionality reduction and
    generative modeling.

    Parameters
    ----------
    n_components : int, default=2
        Dimension of the latent space.

    encoder_layers : list of int, default=[256, 128]
        Hidden layer sizes for the encoder.

    decoder_layers : list of int, default=[128, 256]
        Hidden layer sizes for the decoder.

    activation : str, default='relu'
        Activation function: 'relu', 'leaky_relu', 'elu', 'tanh'.

    dropout : float, default=0.0
        Dropout rate in hidden layers.

    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.

    batch_size : int, default=128
        Mini-batch size.

    n_epochs : int, default=100
        Number of training epochs.

    beta : float, default=1.0
        Weight of KL divergence term (beta-VAE parameter).
        Higher values create more disentangled representations.

    early_stopping : int, default=10
        Stop if no improvement for this many epochs.

    validation_fraction : float, default=0.1
        Fraction of data for validation.

    scale_data : bool, default=True
        Whether to standardize input features.

    device : str, default='auto'
        Device: 'auto', 'cpu', or 'cuda'.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    model_ : _VAEModule
        Fitted VAE model.

    reconstruction_loss_ : float
        Final reconstruction loss.

    kl_loss_ : float
        Final KL divergence loss.

    Example
    -------
    >>> from endgame.dimensionality_reduction import VAEReducer
    >>> vae = VAEReducer(n_components=10, encoder_layers=[512, 256])
    >>> X_latent = vae.fit_transform(X)
    >>> # Reconstruct data
    >>> X_recon = vae.inverse_transform(X_latent)
    >>> # Generate new samples
    >>> z_random = np.random.randn(100, 10)
    >>> X_generated = vae.decode(z_random)
    """

    def __init__(
        self,
        n_components: int = 2,
        encoder_layers: list[int] | None = None,
        decoder_layers: list[int] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 128,
        n_epochs: int = 100,
        beta: float = 1.0,
        early_stopping: int = 10,
        validation_fraction: float = 0.1,
        scale_data: bool = True,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_components = n_components
        self.encoder_layers = encoder_layers or [256, 128]
        self.decoder_layers = decoder_layers or [128, 256]
        self.activation = activation
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.beta = beta
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.scale_data = scale_data
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _get_device(self):
        """Determine the device to use."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _vae_loss(
        self,
        x: torch.Tensor,
        x_recon: torch.Tensor,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute VAE loss: reconstruction + KL divergence.

        Parameters
        ----------
        x : Original input
        x_recon : Reconstructed input
        mu : Latent mean
        log_var : Latent log variance

        Returns
        -------
        total_loss : Combined loss
        recon_loss : Reconstruction loss
        kl_loss : KL divergence loss
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(x_recon, x, reduction="mean")

        # KL divergence: D_KL(q(z|x) || p(z))
        # For Gaussian: -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
        kl_loss = -0.5 * torch.mean(
            1 + log_var - mu.pow(2) - log_var.exp()
        )

        # Total loss with beta weighting
        total_loss = recon_loss + self.beta * kl_loss

        return total_loss, recon_loss, kl_loss

    def _log(self, message: str):
        """Print message if verbose."""
        if self.verbose:
            print(message)

    def fit(self, X, y=None):
        """Fit the VAE model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored

        Returns
        -------
        self : VAEReducer
        """
        _check_torch_installed()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        X = check_array(X, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        # Scale data
        if self.scale_data:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)
        else:
            self._scaler = None

        # Get device
        self._device = self._get_device()
        self._log(f"Training VAE on {self._device}")

        # Create model
        self.model_ = _VAEModule(
            input_dim=self.n_features_in_,
            latent_dim=self.n_components,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            activation=self.activation,
            dropout=self.dropout,
        ).to(self._device)

        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Train/validation split
        if self.validation_fraction > 0:
            n_val = int(len(X) * self.validation_fraction)
            indices = np.random.permutation(len(X))
            train_idx, val_idx = indices[n_val:], indices[:n_val]
            X_train = X_tensor[train_idx]
            X_val = X_tensor[val_idx]
        else:
            X_train = X_tensor
            X_val = None

        train_loader = DataLoader(
            TensorDataset(X_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        if X_val is not None:
            val_loader = DataLoader(
                TensorDataset(X_val),
                batch_size=self.batch_size,
            )
        else:
            val_loader = None

        # Optimizer
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Training
            self.model_.train()
            train_recon_loss = 0.0
            train_kl_loss = 0.0
            n_batches = 0

            for (x_batch,) in train_loader:
                x_batch = x_batch.to(self._device)

                optimizer.zero_grad()
                x_recon, mu, log_var = self.model_(x_batch)
                loss, recon_loss, kl_loss = self._vae_loss(
                    x_batch, x_recon, mu, log_var
                )
                loss.backward()
                optimizer.step()

                train_recon_loss += recon_loss.item()
                train_kl_loss += kl_loss.item()
                n_batches += 1

            train_recon_loss /= n_batches
            train_kl_loss /= n_batches

            # Validation
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for (x_batch,) in val_loader:
                        x_batch = x_batch.to(self._device)
                        x_recon, mu, log_var = self.model_(x_batch)
                        loss, _, _ = self._vae_loss(
                            x_batch, x_recon, mu, log_var
                        )
                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss /= n_val_batches

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
                        f"Epoch {epoch+1}: recon={train_recon_loss:.4f}, "
                        f"kl={train_kl_loss:.4f}, val={val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(
                        f"Epoch {epoch+1}: recon={train_recon_loss:.4f}, "
                        f"kl={train_kl_loss:.4f}"
                    )

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        self.reconstruction_loss_ = train_recon_loss
        self.kl_loss_ = train_kl_loss

        return self

    def transform(self, X) -> np.ndarray:
        """Transform data to latent space.

        Uses the mean of the latent distribution (deterministic).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_latent : ndarray of shape (n_samples, n_components)
            Latent representation.
        """
        check_is_fitted(self, "model_")

        X = check_array(X, dtype=np.float32)
        if self._scaler is not None:
            X = self._scaler.transform(X)

        self.model_.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
            mu, _ = self.model_.encode(X_tensor)
            return mu.cpu().numpy()

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_latent) -> np.ndarray:
        """Transform from latent space back to data space.

        Parameters
        ----------
        X_latent : array-like of shape (n_samples, n_components)
            Data in latent space.

        Returns
        -------
        X_recon : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        check_is_fitted(self, "model_")

        X_latent = check_array(X_latent, dtype=np.float32)

        self.model_.eval()
        with torch.no_grad():
            z_tensor = torch.tensor(X_latent, dtype=torch.float32).to(self._device)
            x_recon = self.model_.decode(z_tensor).cpu().numpy()

        if self._scaler is not None:
            x_recon = self._scaler.inverse_transform(x_recon)

        return x_recon

    def decode(self, z) -> np.ndarray:
        """Decode latent vectors to data space.

        Alias for inverse_transform, useful for generation.

        Parameters
        ----------
        z : array-like of shape (n_samples, n_components)
            Latent vectors.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Generated data.
        """
        return self.inverse_transform(z)

    def sample(self, n_samples: int = 100) -> np.ndarray:
        """Generate new samples from the prior.

        Parameters
        ----------
        n_samples : int, default=100
            Number of samples to generate.

        Returns
        -------
        X : ndarray of shape (n_samples, n_features)
            Generated samples.
        """
        check_is_fitted(self, "model_")

        z = np.random.randn(n_samples, self.n_components).astype(np.float32)
        return self.decode(z)

    def reconstruct(self, X) -> np.ndarray:
        """Reconstruct input through the VAE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to reconstruct.

        Returns
        -------
        X_recon : ndarray of shape (n_samples, n_features)
            Reconstructed data.
        """
        X_latent = self.transform(X)
        return self.inverse_transform(X_latent)

    def reconstruction_error(self, X) -> np.ndarray:
        """Compute reconstruction error for each sample.

        Useful for anomaly detection.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to evaluate.

        Returns
        -------
        errors : ndarray of shape (n_samples,)
            Reconstruction error (MSE) per sample.
        """
        X = check_array(X, dtype=np.float32)
        X_recon = self.reconstruct(X)
        return np.mean((X - X_recon) ** 2, axis=1)
