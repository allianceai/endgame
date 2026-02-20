"""Generative model-based oversamplers for class imbalance handling.

This module provides oversampling methods based on generative models:
GANs, flow matching, and diffusion models for tabular data.

Algorithms
----------
- CTGANResampler: Conditional GAN wrapper (requires ``ctgan``)
- ForestFlowResampler: XGBoost-based flow matching (requires ``xgboost``)
- TabDDPMResampler: Denoising diffusion for tabular data (requires PyTorch)
- TabSynResampler: VAE + latent diffusion for tabular data (requires PyTorch)

References
----------
- CTGAN (NeurIPS 2019)
- ForestFlow (Jolicoeur-Martineau et al., 2024)
- TabDDPM (ICML 2023)
- TabSyn (ICLR 2024)
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

# ---------------------------------------------------------------------------
# Optional dependency checks
# ---------------------------------------------------------------------------
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import ctgan as _ctgan_lib

    HAS_CTGAN = True
except ImportError:
    HAS_CTGAN = False

try:
    import xgboost as xgb

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for this sampler. Install with: pip install torch"
        )


def _check_ctgan():
    if not HAS_CTGAN:
        raise ImportError(
            "ctgan is required for CTGANResampler. Install with: pip install ctgan"
        )


def _check_xgboost():
    if not HAS_XGBOOST:
        raise ImportError(
            "XGBoost is required for ForestFlowResampler. "
            "Install with: pip install xgboost"
        )


def _get_device(device: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# =============================================================================
# CTGANResampler
# =============================================================================


class CTGANResampler(BaseEstimator):
    """Conditional Tabular GAN oversampler.

    Thin wrapper around the ``ctgan.CTGAN`` package. Trains a conditional GAN
    on minority class data and generates synthetic samples to balance.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`~imbalance_geometric._compute_sampling_targets`.
    embedding_dim : int, default=128
        Embedding dimension for the generator.
    generator_dim : tuple of int, default=(256, 256)
        Generator hidden layer sizes.
    discriminator_dim : tuple of int, default=(256, 256)
        Discriminator hidden layer sizes.
    n_epochs : int, default=300
        Training epochs.
    batch_size : int, default=500
        Training batch size.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    CTGAN (NeurIPS 2019)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        embedding_dim: int = 128,
        generator_dim: tuple[int, ...] = (256, 256),
        discriminator_dim: tuple[int, ...] = (256, 256),
        n_epochs: int = 300,
        batch_size: int = 500,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.embedding_dim = embedding_dim
        self.generator_dim = generator_dim
        self.discriminator_dim = discriminator_dim
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: ArrayLike) -> CTGANResampler:
        """Fit (validate input)."""
        _check_ctgan()
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample using CTGAN."""
        import pandas as pd
        from ctgan import CTGAN

        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue

            # CTGAN expects a DataFrame
            col_names = [f"f{i}" for i in range(X.shape[1])]
            df_cls = pd.DataFrame(X_cls, columns=col_names)

            model = CTGAN(
                embedding_dim=self.embedding_dim,
                generator_dim=self.generator_dim,
                discriminator_dim=self.discriminator_dim,
                epochs=self.n_epochs,
                batch_size=min(self.batch_size, len(X_cls)),
                verbose=self.verbose,
            )
            model.fit(df_cls)

            synthetic_df = model.sample(n_synthetic)
            synthetic_X.append(synthetic_df.values.astype(np.float64))
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# ForestFlowResampler
# =============================================================================


class ForestFlowResampler(BaseEstimator):
    """XGBoost-based flow matching oversampler (ForestFlow).

    Trains XGBoost to learn the velocity field ``v(x, t) = x_1 - x_0``
    of a conditional flow matching ODE, then integrates from noise to data
    via Euler steps. CPU-friendly — no PyTorch required.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`~imbalance_geometric._compute_sampling_targets`.
    n_estimators : int, default=100
        Number of trees per XGBoost model.
    max_depth : int, default=6
        Maximum tree depth.
    n_steps : int, default=50
        Number of Euler integration steps.
    noise_type : str, default='gaussian'
        Noise distribution for the source: 'gaussian' or 'uniform'.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    Jolicoeur-Martineau et al., "Generating and Imputing Tabular Data via
    Diffusion and Flow XGBoost Models", 2024.
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        n_estimators: int = 100,
        max_depth: int = 6,
        n_steps: int = 50,
        noise_type: str = "gaussian",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_steps = n_steps
        self.noise_type = noise_type
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: ArrayLike, y: ArrayLike) -> ForestFlowResampler:
        """Fit (validate input)."""
        _check_xgboost()
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def _train_flow(self, X_cls: np.ndarray, rng: np.random.RandomState):
        """Train XGBoost velocity field model on a single class."""
        import xgboost as xgb

        n, d = X_cls.shape

        # Sample noise x_0 and time t uniformly
        n_train = n * 10  # more training pairs
        t = rng.uniform(0, 1, size=n_train)
        idx = rng.randint(0, n, size=n_train)
        x_1 = X_cls[idx]

        if self.noise_type == "gaussian":
            x_0 = rng.randn(n_train, d)
        else:
            x_0 = rng.uniform(-1, 1, size=(n_train, d))

        # Interpolated points: x_t = (1-t)*x_0 + t*x_1
        t_col = t[:, None]
        x_t = (1 - t_col) * x_0 + t_col * x_1

        # Velocity target: v = x_1 - x_0
        v_target = x_1 - x_0

        # Features: [x_t, t]
        features = np.hstack([x_t, t[:, None]])

        # Train one multi-output XGBoost (using d separate models)
        models = []
        for j in range(d):
            model = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=0.1,
                random_state=self.random_state,
                verbosity=0,
            )
            model.fit(features, v_target[:, j])
            models.append(model)

        return models

    def _generate(
        self,
        models: list,
        n_synthetic: int,
        n_features: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate samples via Euler integration."""
        dt = 1.0 / self.n_steps

        # Start from noise
        if self.noise_type == "gaussian":
            x = rng.randn(n_synthetic, n_features)
        else:
            x = rng.uniform(-1, 1, size=(n_synthetic, n_features))

        for step in range(self.n_steps):
            t = step * dt
            t_arr = np.full((n_synthetic, 1), t)
            features = np.hstack([x, t_arr])

            # Predict velocity
            v = np.zeros((n_synthetic, n_features))
            for j, model in enumerate(models):
                v[:, j] = model.predict(features)

            x = x + v * dt

        return x

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample using ForestFlow."""
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue

            if self.verbose:
                print(f"[ForestFlow] Training flow for class {cls} "
                      f"({len(X_cls)} samples, generating {n_synthetic})...")

            # Standardize per-class
            mean = X_cls.mean(axis=0)
            std = X_cls.std(axis=0) + 1e-8
            X_norm = (X_cls - mean) / std

            models = self._train_flow(X_norm, rng)
            generated = self._generate(models, n_synthetic, X.shape[1], rng)

            # De-standardize
            generated = generated * std + mean

            synthetic_X.append(generated)
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# TabDDPM internal module (only defined when PyTorch available)
# =============================================================================

_TabDDPMModule = None

if HAS_TORCH:

    class _TabDDPMModule(nn.Module):
        """MLP denoiser for TabDDPM.

        Predicts noise given noisy input + timestep embedding.
        """

        def __init__(
            self, input_dim: int, hidden_dims: list[int], n_timesteps: int
        ):
            super().__init__()
            self.time_embed = nn.Embedding(n_timesteps, hidden_dims[0])

            layers = []
            prev_dim = input_dim + hidden_dims[0]  # input + time embedding
            for h in hidden_dims:
                layers.append(nn.Linear(prev_dim, h))
                layers.append(nn.SiLU())
                layers.append(nn.Dropout(0.1))
                prev_dim = h
            layers.append(nn.Linear(prev_dim, input_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            t_emb = self.time_embed(t)
            h = torch.cat([x, t_emb], dim=1)
            return self.net(h)

    class _VAEModule(nn.Module):
        """Simple VAE module for TabSyn."""

        def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
            super().__init__()
            # Encoder
            enc_layers = []
            prev = input_dim
            for h in hidden_dims:
                enc_layers.append(nn.Linear(prev, h))
                enc_layers.append(nn.SiLU())
                prev = h
            self.encoder = nn.Sequential(*enc_layers)
            self.fc_mu = nn.Linear(prev, latent_dim)
            self.fc_logvar = nn.Linear(prev, latent_dim)

            # Decoder
            dec_layers = []
            prev = latent_dim
            for h in reversed(hidden_dims):
                dec_layers.append(nn.Linear(prev, h))
                dec_layers.append(nn.SiLU())
                prev = h
            dec_layers.append(nn.Linear(prev, input_dim))
            self.decoder = nn.Sequential(*dec_layers)

        def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            h = self.encoder(x)
            return self.fc_mu(h), self.fc_logvar(h)

        def reparameterize(
            self, mu: torch.Tensor, logvar: torch.Tensor
        ) -> torch.Tensor:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            return self.decode(z), mu, logvar


# =============================================================================
# TabDDPMResampler
# =============================================================================


class TabDDPMResampler(BaseEstimator):
    """Tab-DDPM oversampler: denoising diffusion for tabular data.

    Uses Gaussian diffusion with an MLP denoiser that predicts noise given
    a noisy sample and timestep embedding.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`~imbalance_geometric._compute_sampling_targets`.
    n_timesteps : int, default=1000
        Number of diffusion timesteps.
    hidden_dims : list of int, default=[256, 256]
        MLP denoiser hidden layer sizes.
    n_epochs : int, default=100
        Training epochs.
    batch_size : int, default=256
        Training batch size.
    lr : float, default=1e-3
        Learning rate.
    device : str, default='auto'
        Computation device.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    TabDDPM (Kotelnikov et al., ICML 2023)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        n_timesteps: int = 1000,
        hidden_dims: list[int] | None = None,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.n_timesteps = n_timesteps
        self.hidden_dims = hidden_dims or [256, 256]
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[TabDDPM] {msg}")

    def fit(self, X: ArrayLike, y: ArrayLike) -> TabDDPMResampler:
        """Fit (validate input)."""
        _check_torch()
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def _make_noise_schedule(self) -> tuple[np.ndarray, np.ndarray]:
        """Linear noise schedule returning (betas, alphas_cumprod)."""
        betas = np.linspace(1e-4, 0.02, self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        return betas, alphas_cumprod

    def _train_diffusion(self, X_cls: np.ndarray, device: torch.device):
        """Train diffusion model on minority class data."""
        n, d = X_cls.shape
        _, alphas_cumprod = self._make_noise_schedule()
        alphas_cumprod_t = torch.FloatTensor(alphas_cumprod).to(device)

        model = _TabDDPMModule(d, self.hidden_dims, self.n_timesteps).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        X_tensor = torch.FloatTensor(X_cls).to(device)

        model.train()
        for epoch in range(self.n_epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                idx = perm[start:end]
                x_0 = X_tensor[idx]
                bs = x_0.shape[0]

                # Sample random timesteps
                t = torch.randint(0, self.n_timesteps, (bs,), device=device)

                # Forward diffusion: q(x_t | x_0) = N(sqrt(alpha_bar)*x_0, (1-alpha_bar)*I)
                alpha_bar = alphas_cumprod_t[t].unsqueeze(1)
                noise = torch.randn_like(x_0)
                x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * noise

                # Predict noise
                noise_pred = model(x_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 5) == 0:
                self._log(f"Epoch {epoch + 1}/{self.n_epochs}, loss={total_loss / n_batches:.4f}")

        return model, alphas_cumprod

    @torch.no_grad()
    def _sample(
        self,
        model: nn.Module,
        alphas_cumprod: np.ndarray,
        n_samples: int,
        n_features: int,
        device: torch.device,
    ) -> np.ndarray:
        """Reverse diffusion sampling."""
        betas = np.linspace(1e-4, 0.02, self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod_t = torch.FloatTensor(alphas_cumprod).to(device)

        model.eval()
        x = torch.randn(n_samples, n_features, device=device)

        for t_val in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
            alpha_bar = alphas_cumprod_t[t_val]
            alpha_bar_prev = alphas_cumprod_t[t_val - 1] if t_val > 0 else torch.tensor(1.0)
            beta = betas[t_val]

            noise_pred = model(x, t)

            # Predict x_0
            x_0_pred = (x - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

            # Compute posterior mean
            coeff1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
            coeff2 = (1 - alpha_bar_prev) * torch.sqrt(torch.tensor(alphas[t_val])) / (1 - alpha_bar)
            mean = coeff1 * x_0_pred + coeff2 * x

            if t_val > 0:
                sigma = np.sqrt(beta * (1 - alpha_bar_prev.item()) / (1 - alpha_bar.item()))
                x = mean + sigma * torch.randn_like(x)
            else:
                x = mean

        return x.cpu().numpy()

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample using TabDDPM."""
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        device = _get_device(self.device)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue

            self._log(f"Training diffusion for class {cls} "
                      f"({len(X_cls)} samples, generating {n_synthetic})...")

            # Standardize
            mean = X_cls.mean(axis=0)
            std = X_cls.std(axis=0) + 1e-8
            X_norm = ((X_cls - mean) / std).astype(np.float32)

            model, alphas_cumprod = self._train_diffusion(X_norm, device)
            generated = self._sample(
                model, alphas_cumprod, n_synthetic, X.shape[1], device
            )

            # De-standardize
            generated = generated * std + mean

            synthetic_X.append(generated)
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# TabSynResampler
# =============================================================================


class TabSynResampler(BaseEstimator):
    """TabSyn oversampler: VAE + latent diffusion for tabular data.

    Two-stage approach:
    1. Train a VAE on minority data to learn a smooth latent space.
    2. Train a diffusion model in the latent space.
    Generation: reverse diffusion in latent space -> decode through VAE.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`~imbalance_geometric._compute_sampling_targets`.
    latent_dim : int, default=64
        VAE latent dimension.
    vae_hidden_dims : list of int, default=[256, 128]
        VAE encoder/decoder hidden sizes.
    vae_epochs : int, default=100
        VAE training epochs.
    diffusion_hidden_dims : list of int, default=[256, 256]
        Diffusion denoiser hidden sizes.
    diffusion_epochs : int, default=100
        Diffusion training epochs.
    n_timesteps : int, default=1000
        Number of diffusion timesteps.
    batch_size : int, default=256
        Training batch size.
    lr : float, default=1e-3
        Learning rate.
    device : str, default='auto'
        Computation device.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    TabSyn (Zhang et al., ICLR 2024)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        latent_dim: int = 64,
        vae_hidden_dims: list[int] | None = None,
        vae_epochs: int = 100,
        diffusion_hidden_dims: list[int] | None = None,
        diffusion_epochs: int = 100,
        n_timesteps: int = 1000,
        batch_size: int = 256,
        lr: float = 1e-3,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.latent_dim = latent_dim
        self.vae_hidden_dims = vae_hidden_dims or [256, 128]
        self.vae_epochs = vae_epochs
        self.diffusion_hidden_dims = diffusion_hidden_dims or [256, 256]
        self.diffusion_epochs = diffusion_epochs
        self.n_timesteps = n_timesteps
        self.batch_size = batch_size
        self.lr = lr
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[TabSyn] {msg}")

    def fit(self, X: ArrayLike, y: ArrayLike) -> TabSynResampler:
        """Fit (validate input)."""
        _check_torch()
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def _train_vae(
        self, X_cls: np.ndarray, device: torch.device
    ) -> _VAEModule:
        """Train VAE on class data."""
        n, d = X_cls.shape
        vae = _VAEModule(d, self.vae_hidden_dims, self.latent_dim).to(device)
        optimizer = optim.Adam(vae.parameters(), lr=self.lr)

        X_tensor = torch.FloatTensor(X_cls).to(device)

        vae.train()
        for epoch in range(self.vae_epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                x = X_tensor[perm[start:end]]

                recon, mu, logvar = vae(x)
                recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                loss = recon_loss + kl_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % max(1, self.vae_epochs // 5) == 0:
                self._log(f"VAE Epoch {epoch + 1}/{self.vae_epochs}, "
                          f"loss={total_loss / max(1, n):.4f}")

        return vae

    def _encode_to_latent(
        self,
        vae: _VAEModule,
        X_cls: np.ndarray,
        device: torch.device,
    ) -> np.ndarray:
        """Encode data to latent space using trained VAE."""
        vae.eval()
        X_tensor = torch.FloatTensor(X_cls).to(device)
        with torch.no_grad():
            mu, _ = vae.encode(X_tensor)
        return mu.cpu().numpy()

    def _train_latent_diffusion(
        self, Z: np.ndarray, device: torch.device
    ) -> tuple:
        """Train diffusion model in latent space."""
        n, d = Z.shape
        betas = np.linspace(1e-4, 0.02, self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas)
        alphas_cumprod_t = torch.FloatTensor(alphas_cumprod).to(device)

        model = _TabDDPMModule(d, self.diffusion_hidden_dims, self.n_timesteps).to(device)
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        Z_tensor = torch.FloatTensor(Z).to(device)

        model.train()
        for epoch in range(self.diffusion_epochs):
            perm = torch.randperm(n)
            total_loss = 0.0
            n_batches = 0

            for start in range(0, n, self.batch_size):
                end = min(start + self.batch_size, n)
                z_0 = Z_tensor[perm[start:end]]
                bs = z_0.shape[0]

                t = torch.randint(0, self.n_timesteps, (bs,), device=device)
                alpha_bar = alphas_cumprod_t[t].unsqueeze(1)
                noise = torch.randn_like(z_0)
                z_t = torch.sqrt(alpha_bar) * z_0 + torch.sqrt(1 - alpha_bar) * noise

                noise_pred = model(z_t, t)
                loss = nn.functional.mse_loss(noise_pred, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % max(1, self.diffusion_epochs // 5) == 0:
                self._log(f"Diffusion Epoch {epoch + 1}/{self.diffusion_epochs}, "
                          f"loss={total_loss / n_batches:.4f}")

        return model, alphas_cumprod

    @torch.no_grad()
    def _sample_latent(
        self,
        model: nn.Module,
        alphas_cumprod: np.ndarray,
        n_samples: int,
        latent_dim: int,
        device: torch.device,
    ) -> np.ndarray:
        """Reverse diffusion sampling in latent space."""
        betas = np.linspace(1e-4, 0.02, self.n_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod_t = torch.FloatTensor(alphas_cumprod).to(device)

        model.eval()
        z = torch.randn(n_samples, latent_dim, device=device)

        for t_val in reversed(range(self.n_timesteps)):
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=device)
            alpha_bar = alphas_cumprod_t[t_val]
            alpha_bar_prev = alphas_cumprod_t[t_val - 1] if t_val > 0 else torch.tensor(1.0)
            beta = betas[t_val]

            noise_pred = model(z, t)
            z_0_pred = (z - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)

            coeff1 = beta * torch.sqrt(alpha_bar_prev) / (1 - alpha_bar)
            coeff2 = (1 - alpha_bar_prev) * torch.sqrt(torch.tensor(alphas[t_val])) / (1 - alpha_bar)
            mean = coeff1 * z_0_pred + coeff2 * z

            if t_val > 0:
                sigma = np.sqrt(beta * (1 - alpha_bar_prev.item()) / (1 - alpha_bar.item()))
                z = mean + sigma * torch.randn_like(z)
            else:
                z = mean

        return z.cpu().numpy()

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample using TabSyn."""
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        device = _get_device(self.device)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue

            self._log(f"Training TabSyn for class {cls} "
                      f"({len(X_cls)} samples, generating {n_synthetic})...")

            # Standardize
            mean = X_cls.mean(axis=0)
            std = X_cls.std(axis=0) + 1e-8
            X_norm = ((X_cls - mean) / std).astype(np.float32)

            # Clamp latent dim to input dim
            effective_latent = min(self.latent_dim, X_norm.shape[1])

            # Stage 1: Train VAE
            self._log("Stage 1: Training VAE...")
            vae = _VAEModule(
                X_norm.shape[1], self.vae_hidden_dims, effective_latent
            ).to(device)
            vae_optimizer = optim.Adam(vae.parameters(), lr=self.lr)
            X_tensor = torch.FloatTensor(X_norm).to(device)

            vae.train()
            for epoch in range(self.vae_epochs):
                perm = torch.randperm(len(X_norm))
                total_loss = 0.0
                for start in range(0, len(X_norm), self.batch_size):
                    end = min(start + self.batch_size, len(X_norm))
                    x = X_tensor[perm[start:end]]
                    recon, mu, logvar = vae(x)
                    recon_loss = nn.functional.mse_loss(recon, x, reduction="sum")
                    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
                    loss = recon_loss + kl_loss
                    vae_optimizer.zero_grad()
                    loss.backward()
                    vae_optimizer.step()
                    total_loss += loss.item()

                if self.verbose and (epoch + 1) % max(1, self.vae_epochs // 5) == 0:
                    self._log(f"VAE Epoch {epoch + 1}/{self.vae_epochs}, "
                              f"loss={total_loss / max(1, len(X_norm)):.4f}")

            # Encode to latent space
            vae.eval()
            with torch.no_grad():
                mu, _ = vae.encode(X_tensor)
                Z = mu.cpu().numpy()

            # Stage 2: Train latent diffusion
            self._log("Stage 2: Training latent diffusion...")
            diff_model, alphas_cumprod = self._train_latent_diffusion(Z, device)

            # Generate in latent space
            Z_gen = self._sample_latent(
                diff_model, alphas_cumprod, n_synthetic, effective_latent, device
            )

            # Decode through VAE
            vae.eval()
            with torch.no_grad():
                Z_tensor = torch.FloatTensor(Z_gen).to(device)
                generated = vae.decode(Z_tensor).cpu().numpy()

            # De-standardize
            generated = generated * std + mean

            synthetic_X.append(generated)
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# Category dict for registration
GENERATIVE_SAMPLERS = {
    "ctgan": CTGANResampler,
    "forest_flow": ForestFlowResampler,
    "tabddpm": TabDDPMResampler,
    "tabsyn": TabSynResampler,
}
