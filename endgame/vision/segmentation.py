"""Segmentation models using segmentation_models_pytorch.

Provides U-Net and other segmentation architectures with pretrained encoders.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# SMP imports
try:
    import segmentation_models_pytorch as smp

    HAS_SMP = True
except ImportError:
    HAS_SMP = False


def _check_dependencies():
    """Check if required dependencies are available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for SegmentationModel. "
            "Install with: pip install torch"
        )
    if not HAS_SMP:
        raise ImportError(
            "segmentation_models_pytorch is required for SegmentationModel. "
            "Install with: pip install segmentation-models-pytorch"
        )


class SegmentationModel(BaseEstimator):
    """Segmentation model wrapper using segmentation_models_pytorch.

    Provides easy access to U-Net, FPN, DeepLabV3, and other architectures
    with pretrained encoders from timm/torchvision.

    Parameters
    ----------
    architecture : str, default='unet'
        Segmentation architecture:
        - 'unet': Standard U-Net
        - 'unetplusplus' or 'unet++': U-Net++
        - 'fpn': Feature Pyramid Network
        - 'pspnet': Pyramid Scene Parsing Network
        - 'deeplabv3': DeepLab V3
        - 'deeplabv3plus' or 'deeplabv3+': DeepLab V3+
        - 'pan': Pyramid Attention Network
        - 'linknet': LinkNet
        - 'manet': Multi-scale Attention Net
    encoder : str, default='efficientnet-b4'
        Encoder backbone from smp encoders.
        Common choices:
        - 'resnet34', 'resnet50', 'resnet101'
        - 'efficientnet-b0' to 'efficientnet-b7'
        - 'timm-efficientnet-b4'
        - 'se_resnext50_32x4d'
        - 'mit_b0' to 'mit_b5' (SegFormer encoders)
    encoder_weights : str, default='imagenet'
        Pretrained weights: 'imagenet', 'noisy-student', etc.
    in_channels : int, default=3
        Number of input channels.
    num_classes : int, default=1
        Number of output classes.
    activation : str, optional
        Output activation: 'sigmoid', 'softmax', 'softmax2d', None.
        Default: 'sigmoid' for 1 class, None for multi-class.
    learning_rate : float, default=1e-4
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=50
        Maximum training epochs.
    batch_size : int, default=16
        Training batch size.
    early_stopping : int, default=10
        Early stopping patience.
    loss : str, default='dice'
        Loss function: 'dice', 'bce', 'focal', 'dice_bce', 'lovasz'.
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
    model_ : nn.Module
        Fitted segmentation model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.vision import SegmentationModel
    >>> # Create U-Net with EfficientNet-B4 encoder
    >>> seg = SegmentationModel(
    ...     architecture='unet',
    ...     encoder='efficientnet-b4',
    ...     num_classes=1,
    ...     n_epochs=30
    ... )
    >>> # Train on images and masks
    >>> seg.fit(images, masks, val_data=(val_images, val_masks))
    >>> # Predict
    >>> predictions = seg.predict(test_images)
    >>> # Get probabilities
    >>> probabilities = seg.predict_proba(test_images)
    """

    ARCHITECTURES = {
        "unet": "Unet",
        "unetplusplus": "UnetPlusPlus",
        "unet++": "UnetPlusPlus",
        "fpn": "FPN",
        "pspnet": "PSPNet",
        "deeplabv3": "DeepLabV3",
        "deeplabv3plus": "DeepLabV3Plus",
        "deeplabv3+": "DeepLabV3Plus",
        "pan": "PAN",
        "linknet": "Linknet",
        "manet": "MAnet",
    }

    def __init__(
        self,
        architecture: str = "unet",
        encoder: str = "efficientnet-b4",
        encoder_weights: str = "imagenet",
        in_channels: int = 3,
        num_classes: int = 1,
        activation: str | None = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 16,
        early_stopping: int = 10,
        loss: str = "dice",
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_dependencies()

        self.architecture = architecture
        self.encoder = encoder
        self.encoder_weights = encoder_weights
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.loss = loss
        self.scheduler = scheduler
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: nn.Module | None = None
        self._device: torch.device | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[Segmentation] {message}")

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

    def _create_model(self) -> nn.Module:
        """Create segmentation model."""
        arch_name = self.ARCHITECTURES.get(self.architecture.lower())
        if arch_name is None:
            raise ValueError(
                f"Unknown architecture: {self.architecture}. "
                f"Choose from: {list(self.ARCHITECTURES.keys())}"
            )

        # Determine activation
        activation = self.activation
        if activation is None:
            activation = "sigmoid" if self.num_classes == 1 else None

        model_class = getattr(smp, arch_name)
        return model_class(
            encoder_name=self.encoder,
            encoder_weights=self.encoder_weights,
            in_channels=self.in_channels,
            classes=self.num_classes,
            activation=activation,
        )

    def _get_loss(self) -> nn.Module:
        """Get loss function."""
        if self.loss == "dice":
            return smp.losses.DiceLoss(mode="binary" if self.num_classes == 1 else "multiclass")
        elif self.loss == "bce":
            return nn.BCEWithLogitsLoss()
        elif self.loss == "focal":
            return smp.losses.FocalLoss(mode="binary" if self.num_classes == 1 else "multiclass")
        elif self.loss == "dice_bce":
            dice = smp.losses.DiceLoss(mode="binary" if self.num_classes == 1 else "multiclass")
            bce = nn.BCEWithLogitsLoss()

            class DiceBCELoss(nn.Module):
                def forward(self, pred, target):
                    return 0.5 * dice(pred, target) + 0.5 * bce(pred, target.float())

            return DiceBCELoss()
        elif self.loss == "lovasz":
            return smp.losses.LovaszLoss(mode="binary" if self.num_classes == 1 else "multiclass")
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

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
                optimizer, T_max=n_epochs, eta_min=1e-7
            )
        elif self.scheduler == "step":
            return optim.lr_scheduler.StepLR(
                optimizer, step_size=n_epochs // 3, gamma=0.1
            )
        elif self.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def fit(
        self,
        X,
        y,
        val_data: tuple[Any, Any] | None = None,
    ) -> "SegmentationModel":
        """Fit the segmentation model.

        Parameters
        ----------
        X : array-like of shape (n_samples, C, H, W) or (n_samples, H, W, C)
            Training images.
        y : array-like of shape (n_samples, H, W) or (n_samples, n_classes, H, W)
            Training masks.
        val_data : tuple of (X_val, y_val), optional
            Validation data.

        Returns
        -------
        self
            Fitted model.
        """
        self._set_seed()
        self._device = self._get_device()

        # Convert to numpy
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32)

        # Ensure channel-first format for images
        if X_arr.ndim == 4 and X_arr.shape[-1] in [1, 3, 4]:
            X_arr = np.transpose(X_arr, (0, 3, 1, 2))

        # Ensure proper mask format
        if y_arr.ndim == 3:
            y_arr = y_arr[:, np.newaxis, :, :]  # Add channel dim

        # Create model
        self.model_ = self._create_model().to(self._device)

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._get_scheduler(optimizer, self.n_epochs)
        criterion = self._get_loss()

        # Create dataloaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_arr),
            torch.FloatTensor(y_arr),
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
            y_val = np.asarray(y_val, dtype=np.float32)

            if X_val.ndim == 4 and X_val.shape[-1] in [1, 3, 4]:
                X_val = np.transpose(X_val, (0, 3, 1, 2))
            if y_val.ndim == 3:
                y_val = y_val[:, np.newaxis, :, :]

            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
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

        self._log(f"Training {self.architecture} with {self.encoder} on {self._device}...")

        for epoch in range(self.n_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
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

                        outputs = self.model_(X_batch)
                        loss = criterion(outputs, y_batch)

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
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss if val_loader is None else val_loss)
                else:
                    scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        """Predict segmentation masks.

        Parameters
        ----------
        X : array-like of shape (n_samples, C, H, W) or (n_samples, H, W, C)
            Input images.
        threshold : float, default=0.5
            Threshold for binary prediction.

        Returns
        -------
        ndarray
            Predicted masks.
        """
        proba = self.predict_proba(X)

        if self.num_classes == 1:
            return (proba > threshold).astype(np.uint8)
        else:
            return np.argmax(proba, axis=1).astype(np.uint8)

    def predict_proba(self, X) -> np.ndarray:
        """Predict segmentation probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, C, H, W) or (n_samples, H, W, C)
            Input images.

        Returns
        -------
        ndarray
            Predicted probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("SegmentationModel has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)

        # Ensure channel-first format
        if X_arr.ndim == 4 and X_arr.shape[-1] in [1, 3, 4]:
            X_arr = np.transpose(X_arr, (0, 3, 1, 2))

        X_tensor = torch.FloatTensor(X_arr)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X_tensor), self.batch_size):
                end = min(start + self.batch_size, len(X_tensor))
                X_batch = X_tensor[start:end].to(self._device)
                outputs = self.model_(X_batch)

                # Handle activation (if model has sigmoid, output is already prob)
                if self.activation == "sigmoid" or self.num_classes == 1:
                    proba = outputs
                elif self.num_classes > 1:
                    proba = torch.softmax(outputs, dim=1)
                else:
                    proba = torch.sigmoid(outputs)

                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def get_encoder_features(self, X) -> list[np.ndarray]:
        """Extract encoder features at different scales.

        Parameters
        ----------
        X : array-like
            Input images.

        Returns
        -------
        List[ndarray]
            Features at each encoder stage.
        """
        if self.model_ is None:
            raise RuntimeError("SegmentationModel has not been fitted.")

        X_arr = np.asarray(X, dtype=np.float32)
        if X_arr.ndim == 4 and X_arr.shape[-1] in [1, 3, 4]:
            X_arr = np.transpose(X_arr, (0, 3, 1, 2))

        X_tensor = torch.FloatTensor(X_arr).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            features = self.model_.encoder(X_tensor)

        return [f.cpu().numpy() for f in features]

    @staticmethod
    def available_encoders() -> list[str]:
        """Get list of available encoders.

        Returns
        -------
        List[str]
            Available encoder names.
        """
        _check_dependencies()
        return smp.encoders.get_encoder_names()
