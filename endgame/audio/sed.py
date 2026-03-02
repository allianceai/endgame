from __future__ import annotations

"""Sound Event Detection (SED) models for audio classification.

SED models provide temporal localization of audio events, predicting
which seconds contain each target sound class.

This is a key technique for BirdCLEF and similar audio competitions.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

# PyTorch imports (lazy loaded)
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
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for SED models. "
            "Install with: pip install torch"
        )


class _FocalLoss(nn.Module):
    """Focal Loss for class-imbalanced classification.

    Applies a modulating factor (1 - p_t)^gamma to standard cross-entropy,
    down-weighting well-classified examples so the model focuses on hard cases.

    Parameters
    ----------
    weight : Tensor, optional
        Per-class weights (same as CrossEntropyLoss weight parameter).
    gamma : float, default=2.0
        Focusing parameter. Higher values increase focus on hard examples.
    label_smoothing : float, default=0.0
        Label smoothing factor.
    reduction : str, default='mean'
        Reduction mode: 'mean', 'sum', or 'none'.
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.0, reduction="mean"):
        super().__init__()
        self.register_buffer("weight", weight)
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


class _AttentionPool(nn.Module):
    """Attention pooling layer for SED."""

    def __init__(self, in_features: int):
        super().__init__()
        self.attention = nn.Linear(in_features, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply attention pooling.

        Parameters
        ----------
        x : Tensor of shape (batch, time, features)
            Input features.

        Returns
        -------
        pooled : Tensor of shape (batch, features)
            Attention-pooled features.
        weights : Tensor of shape (batch, time)
            Attention weights.
        """
        # Compute attention weights
        weights = torch.softmax(self.attention(x), dim=1)  # (batch, time, 1)
        # Weighted sum
        pooled = torch.sum(x * weights, dim=1)  # (batch, features)
        return pooled, weights.squeeze(-1)


class _SEDModule(nn.Module):
    """PyTorch SED module.

    Architecture: CNN encoder + temporal model + classifier

    Parameters
    ----------
    n_mels : int
        Number of mel frequency bins.
    n_classes : int
        Number of target classes.
    encoder : str
        CNN encoder type: 'cnn6', 'cnn10', 'cnn14', 'efficientnet'.
    temporal_model : str
        Temporal model: 'transformer', 'gru', 'lstm', 'none'.
    hidden_dim : int
        Hidden dimension for temporal model.
    n_heads : int
        Number of attention heads (for transformer).
    n_layers : int
        Number of layers in temporal model.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 10,
        encoder: str = "cnn10",
        temporal_model: str = "transformer",
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.encoder_type = encoder
        self.temporal_type = temporal_model

        # Build CNN encoder
        self.encoder, encoder_out_dim = self._build_encoder(encoder, n_mels)

        # Build temporal model
        self.temporal, temporal_out_dim = self._build_temporal(
            encoder_out_dim, hidden_dim, temporal_model, n_heads, n_layers, dropout
        )

        # Attention pooling
        self.attention_pool = _AttentionPool(temporal_out_dim)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(temporal_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

        # Frame-level classifier (for SED output)
        self.frame_classifier = nn.Linear(temporal_out_dim, n_classes)

    def _build_encoder(
        self,
        encoder: str,
        n_mels: int,
    ) -> tuple[nn.Module, int]:
        """Build CNN encoder."""
        if encoder == "cnn6":
            return self._build_cnn(n_mels, [64, 128, 256], [2, 2, 2])
        elif encoder == "cnn10":
            return self._build_cnn(n_mels, [64, 128, 256, 512], [2, 2, 2, 2])
        elif encoder == "cnn14":
            return self._build_cnn(n_mels, [64, 128, 256, 512, 1024], [2, 2, 2, 2, 2])
        elif encoder == "efficientnet":
            return self._build_efficientnet(n_mels)
        else:
            raise ValueError(f"Unknown encoder: {encoder}")

    def _build_cnn(
        self,
        n_mels: int,
        channels: list[int],
        pool_sizes: list[int],
    ) -> tuple[nn.Module, int]:
        """Build simple CNN encoder."""
        layers = []
        in_channels = 1
        freq_dim = n_mels

        for out_channels, pool_size in zip(channels, pool_sizes):
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d((pool_size, pool_size)),
            ])
            in_channels = out_channels
            freq_dim = freq_dim // pool_size

        # Global frequency pooling
        layers.append(nn.AdaptiveAvgPool2d((None, 1)))

        encoder = nn.Sequential(*layers)
        out_dim = channels[-1]  # After freq pooling: (batch, channels, time, 1)
        return encoder, out_dim

    def _build_efficientnet(self, n_mels: int) -> tuple[nn.Module, int]:
        """Build EfficientNet-based encoder."""
        try:
            import timm
        except ImportError:
            raise ImportError("timm is required for efficientnet encoder")

        # Use EfficientNet-B0
        backbone = timm.create_model(
            "tf_efficientnet_b0_ns",
            pretrained=True,
            in_chans=1,
            features_only=True,
        )

        class EfficientNetEncoder(nn.Module):
            def __init__(self, backbone):
                super().__init__()
                self.backbone = backbone
                self.pool = nn.AdaptiveAvgPool2d((None, 1))

            def forward(self, x):
                features = self.backbone(x)[-1]  # Get last feature map
                return self.pool(features)

        return EfficientNetEncoder(backbone), 1280  # EfficientNet-B0 output channels

    def _build_temporal(
        self,
        in_dim: int,
        hidden_dim: int,
        temporal_type: str,
        n_heads: int,
        n_layers: int,
        dropout: float,
    ) -> tuple[nn.Module, int]:
        """Build temporal model."""
        if temporal_type == "none":
            return nn.Identity(), in_dim

        elif temporal_type == "transformer":
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=in_dim,
                nhead=n_heads,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
            )
            transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            return transformer, in_dim

        elif temporal_type == "gru":
            gru = nn.GRU(
                in_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            return gru, hidden_dim * 2

        elif temporal_type == "lstm":
            lstm = nn.LSTM(
                in_dim,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0,
            )
            return lstm, hidden_dim * 2

        else:
            raise ValueError(f"Unknown temporal model: {temporal_type}")

    def forward(
        self,
        x: torch.Tensor,
        return_frame_level: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1, time, n_mels) or (batch, time, n_mels)
            Input mel-spectrogram.
        return_frame_level : bool
            Whether to return frame-level predictions.

        Returns
        -------
        clip_logits : Tensor of shape (batch, n_classes)
            Clip-level predictions.
        frame_logits : Tensor of shape (batch, time, n_classes), optional
            Frame-level predictions (if return_frame_level=True).
        """
        # Add channel dim if needed
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (batch, 1, time, n_mels)

        # CNN encoding
        features = self.encoder(x)  # (batch, channels, time', 1)
        features = features.squeeze(-1)  # (batch, channels, time')
        features = features.permute(0, 2, 1)  # (batch, time', channels)

        # Temporal modeling
        if self.temporal_type in ["gru", "lstm"]:
            features, _ = self.temporal(features)
        elif self.temporal_type == "transformer":
            features = self.temporal(features)

        # Frame-level predictions
        frame_logits = self.frame_classifier(features)  # (batch, time', n_classes)

        # Attention pooling for clip-level prediction
        pooled, attention_weights = self.attention_pool(features)
        clip_logits = self.classifier(pooled)

        if return_frame_level:
            return clip_logits, frame_logits

        return clip_logits

    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Get attention weights for visualization.

        Parameters
        ----------
        x : Tensor
            Input mel-spectrogram.

        Returns
        -------
        Tensor of shape (batch, time')
            Attention weights.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)

        features = self.encoder(x)
        features = features.squeeze(-1).permute(0, 2, 1)

        if self.temporal_type in ["gru", "lstm"]:
            features, _ = self.temporal(features)
        elif self.temporal_type == "transformer":
            features = self.temporal(features)

        _, attention_weights = self.attention_pool(features)
        return attention_weights


class SEDModel(BaseEstimator, ClassifierMixin):
    """Sound Event Detection model for temporal audio classification.

    Combines CNN encoder for spectrogram processing with temporal modeling
    (Transformer/RNN) to produce both clip-level and frame-level predictions.

    Parameters
    ----------
    n_mels : int, default=128
        Number of mel frequency bins in input spectrograms.
    n_classes : int, default=10
        Number of target sound classes.
    encoder : str, default='cnn10'
        CNN encoder: 'cnn6', 'cnn10', 'cnn14', 'efficientnet'.
    temporal_model : str, default='transformer'
        Temporal model: 'transformer', 'gru', 'lstm', 'none'.
    hidden_dim : int, default=256
        Hidden dimension for temporal model.
    n_heads : int, default=4
        Number of attention heads (transformer only).
    n_layers : int, default=2
        Number of temporal model layers.
    dropout : float, default=0.2
        Dropout rate.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=50
        Maximum training epochs.
    batch_size : int, default=32
        Training batch size.
    early_stopping : int, default=10
        Early stopping patience.
    mixup_alpha : float, default=0.0
        Mixup augmentation strength (0 to disable).
    label_smoothing : float, default=0.0
        Label smoothing factor.
    multi_label : bool, default=True
        If True, use BCEWithLogitsLoss (for multi-label SED tasks).
        If False, use CrossEntropyLoss (for single-label multi-class
        classification tasks like baby cry classification).
    class_weights : str, array-like, or None, default=None
        Class weights for the loss function. Only used when multi_label=False.
        - None: no weighting
        - "balanced": compute weights inversely proportional to class frequencies
        - array-like: per-class weights of shape (n_classes,)
    loss_fn : str, default='ce'
        Loss function: 'ce' for CrossEntropyLoss, 'focal' for FocalLoss.
        Focal loss down-weights well-classified examples (good for imbalanced data).
        Only used when multi_label=False.
    focal_gamma : float, default=2.0
        Focusing parameter for focal loss. Higher = more focus on hard examples.
    spec_augment : bool, default=False
        Apply SpecAugment (frequency + time masking) during training.
    spec_augment_freq_mask : int, default=10
        Maximum width of frequency masks for SpecAugment.
    spec_augment_time_mask : int, default=20
        Maximum width of time masks for SpecAugment.
    scheduler : str, default='cosine'
        Learning rate scheduler.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    model_ : _SEDModule
        Fitted PyTorch model.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.audio import SEDModel, SpectrogramTransformer
    >>> # Convert audio to spectrograms
    >>> spec_transform = SpectrogramTransformer(n_mels=128)
    >>> X_spec = spec_transform.transform(audio_samples)
    >>> # Train SED model
    >>> sed = SEDModel(n_classes=10, encoder='cnn10', temporal_model='transformer')
    >>> sed.fit(X_spec, y)
    >>> # Get predictions
    >>> clip_pred = sed.predict(X_spec_test)
    >>> clip_proba = sed.predict_proba(X_spec_test)
    >>> # Get frame-level predictions
    >>> frame_proba = sed.predict_frames(X_spec_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_mels: int = 128,
        n_classes: int = 10,
        encoder: str = "cnn10",
        temporal_model: str = "transformer",
        hidden_dim: int = 256,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 32,
        early_stopping: int = 10,
        mixup_alpha: float = 0.0,
        label_smoothing: float = 0.0,
        multi_label: bool = True,
        class_weights: str | np.ndarray | None = None,
        loss_fn: str = "ce",
        focal_gamma: float = 2.0,
        spec_augment: bool = False,
        spec_augment_freq_mask: int = 10,
        spec_augment_time_mask: int = 20,
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_torch()

        self.n_mels = n_mels
        self.n_classes = n_classes
        self.encoder = encoder
        self.temporal_model = temporal_model
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.mixup_alpha = mixup_alpha
        self.label_smoothing = label_smoothing
        self.multi_label = multi_label
        self.class_weights = class_weights
        self.loss_fn = loss_fn
        self.focal_gamma = focal_gamma
        self.spec_augment = spec_augment
        self.spec_augment_freq_mask = spec_augment_freq_mask
        self.spec_augment_time_mask = spec_augment_time_mask
        self.scheduler = scheduler
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: _SEDModule | None = None
        self._device: torch.device | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[SED] {message}")

    def _get_device(self) -> torch.device:
        """Get computation device."""
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        """Set random seeds."""
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _mixup(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        alpha: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
        """Apply mixup augmentation."""
        if alpha <= 0:
            return x, y, y, 1.0

        lam = np.random.beta(alpha, alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]

        return mixed_x, y_a, y_b, lam

    def _apply_spec_augment(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SpecAugment (frequency + time masking) to a batch.

        Parameters
        ----------
        x : Tensor of shape (batch, time, n_mels)
            Input spectrograms.

        Returns
        -------
        Tensor
            Augmented spectrograms.
        """
        batch_size, n_time, n_freq = x.shape
        x = x.clone()
        for i in range(batch_size):
            # Frequency masking (2 masks)
            for _ in range(2):
                f = torch.randint(0, self.spec_augment_freq_mask + 1, (1,)).item()
                if n_freq > f > 0:
                    f0 = torch.randint(0, n_freq - f, (1,)).item()
                    x[i, :, f0:f0 + f] = 0
            # Time masking (2 masks)
            for _ in range(2):
                t = torch.randint(0, self.spec_augment_time_mask + 1, (1,)).item()
                if n_time > t > 0:
                    t0 = torch.randint(0, n_time - t, (1,)).item()
                    x[i, t0:t0 + t, :] = 0
        return x

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

    def fit(
        self,
        X,
        y,
        val_data: tuple[Any, Any] | None = None,
    ) -> SEDModel:
        """Fit the SED model.

        Parameters
        ----------
        X : array-like of shape (n_samples, time, n_mels)
            Mel-spectrogram data.
        y : array-like of shape (n_samples,) or (n_samples, n_classes)
            Target labels (integer or one-hot).
        val_data : tuple of (X_val, y_val), optional
            Validation data for early stopping.

        Returns
        -------
        self
            Fitted model.
        """
        self._set_seed()
        self._device = self._get_device()

        # Convert to numpy
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y)

        # Handle labels
        if y_arr.ndim == 1:
            self.classes_ = np.unique(y_arr)
            self.n_classes_ = len(self.classes_)
            if self.multi_label:
                # Convert to one-hot with optional label smoothing
                y_onehot = np.zeros((len(y_arr), self.n_classes_), dtype=np.float32)
                for i, label in enumerate(y_arr):
                    y_onehot[i, label] = 1.0
                if self.label_smoothing > 0:
                    y_onehot = y_onehot * (1 - self.label_smoothing) + self.label_smoothing / self.n_classes_
                y_arr = y_onehot
            # For single-label (multi_label=False), keep integer labels
        else:
            self.n_classes_ = y_arr.shape[1]
            self.classes_ = np.arange(self.n_classes_)

        # Create model
        self.model_ = _SEDModule(
            n_mels=self.n_mels,
            n_classes=self.n_classes_,
            encoder=self.encoder,
            temporal_model=self.temporal_model,
            hidden_dim=self.hidden_dim,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=self.dropout,
        ).to(self._device)

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._get_scheduler(optimizer, self.n_epochs)

        # Compute class weights for single-label mode
        weights_tensor = None
        if not self.multi_label and self.class_weights is not None:
            if self.class_weights == "balanced":
                class_counts = np.bincount(
                    y_arr.astype(int), minlength=self.n_classes_
                )
                weights = len(y_arr) / (self.n_classes_ * class_counts.clip(min=1))
                weights_tensor = torch.FloatTensor(weights).to(self._device)
            else:
                weights_tensor = torch.FloatTensor(
                    np.asarray(self.class_weights)
                ).to(self._device)

        # Loss function
        if self.multi_label:
            criterion = nn.BCEWithLogitsLoss()
        elif self.loss_fn == "focal":
            criterion = _FocalLoss(
                weight=weights_tensor,
                gamma=self.focal_gamma,
                label_smoothing=self.label_smoothing,
            )
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weights_tensor,
                label_smoothing=self.label_smoothing,
            )

        # Create dataloaders
        if self.multi_label:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_arr),
                torch.FloatTensor(y_arr),
            )
        else:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_arr),
                torch.LongTensor(y_arr.astype(np.int64)),
            )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self._device.type == "cuda",
        )

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            X_val = np.asarray(X_val, dtype=np.float32)
            y_val = np.asarray(y_val)
            if self.multi_label:
                if y_val.ndim == 1:
                    y_val_onehot = np.zeros((len(y_val), self.n_classes_), dtype=np.float32)
                    for i, label in enumerate(y_val):
                        y_val_onehot[i, label] = 1.0
                    y_val = y_val_onehot
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.FloatTensor(y_val),
                )
            else:
                if y_val.ndim > 1:
                    y_val = y_val.argmax(axis=1)
                val_dataset = TensorDataset(
                    torch.FloatTensor(X_val),
                    torch.LongTensor(y_val.astype(np.int64)),
                )

            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training SED on {self._device}...")

        for epoch in range(self.n_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                # SpecAugment
                if self.spec_augment:
                    X_batch = self._apply_spec_augment(X_batch)

                # Mixup
                if self.mixup_alpha > 0:
                    X_batch, y_a, y_b, lam = self._mixup(X_batch, y_batch, self.mixup_alpha)

                optimizer.zero_grad()
                logits = self.model_(X_batch)

                if self.mixup_alpha > 0:
                    loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                else:
                    loss = criterion(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history_["train_loss"].append(train_loss)

            # Validate
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self._device)
                        y_batch = y_batch.to(self._device)

                        logits = self.model_(X_batch)
                        loss = criterion(logits, y_batch)

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

                if self.verbose and (epoch + 1) % 5 == 0:
                    self._log(
                        f"Epoch {epoch + 1}/{self.n_epochs}: "
                        f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}"
                    )

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if self.verbose and (epoch + 1) % 5 == 0:
                    self._log(f"Epoch {epoch + 1}/{self.n_epochs}: train_loss={train_loss:.4f}")

            if scheduler is not None:
                scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, time, n_mels)
            Mel-spectrogram data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities (clip-level).

        Parameters
        ----------
        X : array-like of shape (n_samples, time, n_mels)
            Mel-spectrogram data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("SEDModel has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_arr)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                end = min(start + self.batch_size, len(X_tensor))
                X_batch = X_tensor[start:end].to(self._device)
                logits = self.model_(X_batch)
                if self.multi_label:
                    proba = torch.sigmoid(logits)
                else:
                    proba = torch.softmax(logits, dim=-1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict_frames(self, X) -> np.ndarray:
        """Predict frame-level probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, time, n_mels)
            Mel-spectrogram data.

        Returns
        -------
        ndarray of shape (n_samples, time', n_classes)
            Frame-level class probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("SEDModel has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_arr)

        self.model_.eval()
        all_frame_proba = []

        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                end = min(start + self.batch_size, len(X_tensor))
                X_batch = X_tensor[start:end].to(self._device)
                _, frame_logits = self.model_(X_batch, return_frame_level=True)
                frame_proba = torch.sigmoid(frame_logits)
                all_frame_proba.append(frame_proba.cpu().numpy())

        return np.vstack(all_frame_proba)

    def get_attention_weights(self, X) -> np.ndarray:
        """Get attention weights for temporal localization.

        Parameters
        ----------
        X : array-like of shape (n_samples, time, n_mels)
            Mel-spectrogram data.

        Returns
        -------
        ndarray of shape (n_samples, time')
            Attention weights indicating event locations.
        """
        if self.model_ is None:
            raise RuntimeError("SEDModel has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_tensor = torch.FloatTensor(X_arr).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            weights = self.model_.get_attention_weights(X_tensor)

        return weights.cpu().numpy()
