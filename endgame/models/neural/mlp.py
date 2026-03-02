from __future__ import annotations

"""Multi-Layer Perceptron implementations for tabular data.

This module provides PyTorch-based MLP classifiers and regressors with
modern techniques like batch normalization, dropout, and learning rate scheduling.
"""

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

from endgame.core.base import EndgameEstimator

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
            "PyTorch is required for neural network models. "
            "Install with: pip install endgame-ml[tabular]"
        )


# Only define PyTorch module if torch is available
_MLPModule = None

if HAS_TORCH:
    class _MLPModule(nn.Module):
        """PyTorch MLP module.

        Parameters
        ----------
        input_dim : int
            Number of input features.
        hidden_dims : List[int]
            Hidden layer dimensions.
        output_dim : int
            Number of output units.
        dropout : float
            Dropout rate.
        batch_norm : bool
            Whether to use batch normalization.
        activation : str
            Activation function.
        """

        def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            output_dim: int,
            dropout: float = 0.3,
            batch_norm: bool = True,
            activation: str = "relu",
        ):
            super().__init__()

            self.activation_fn = self._get_activation(activation)

            layers = []
            prev_dim = input_dim

            for hidden_dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, hidden_dim))

                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden_dim))

                layers.append(self.activation_fn)

                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

                prev_dim = hidden_dim

            self.hidden_layers = nn.Sequential(*layers)
            self.output_layer = nn.Linear(prev_dim, output_dim)

        def _get_activation(self, activation: str) -> nn.Module:
            """Get activation function by name."""
            activations = {
                "relu": nn.ReLU(),
                "leaky_relu": nn.LeakyReLU(0.1),
                "elu": nn.ELU(),
                "selu": nn.SELU(),
                "gelu": nn.GELU(),
                "swish": nn.SiLU(),
                "mish": nn.Mish(),
                "tanh": nn.Tanh(),
            }
            if activation not in activations:
                raise ValueError(
                    f"Unknown activation: {activation}. "
                    f"Choose from: {list(activations.keys())}"
                )
            return activations[activation]

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass."""
            x = self.hidden_layers(x)
            return self.output_layer(x)


class _BaseMLPEstimator(EndgameEstimator):
    """Base class for MLP estimators.

    Parameters
    ----------
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for regularization.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function: 'relu', 'leaky_relu', 'elu', 'selu',
        'gelu', 'swish', 'mish', 'tanh'.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Number of epochs without improvement to stop training.
    scheduler : str, default='cosine'
        Learning rate scheduler: 'cosine', 'step', 'plateau', 'none'.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto' (auto-detect).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 10,
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        _check_torch()
        super().__init__(random_state=random_state, verbose=verbose)

        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.device = device

        # Model components (set during fit)
        self.model_: _MLPModule | None = None
        self.scaler_: StandardScaler | None = None
        self._device: torch.device | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    def _get_device(self) -> torch.device:
        """Get the computation device."""
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
        elif self.scheduler == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=5
            )
        else:
            raise ValueError(f"Unknown scheduler: {self.scheduler}")

    def _create_dataloader(
        self,
        X: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X)
        y_tensor = self._prepare_target_tensor(y)

        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=self._device.type == "cuda",
        )

    def _prepare_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Prepare target tensor (override in subclasses)."""
        raise NotImplementedError

    def _compute_loss(
        self,
        outputs: torch.Tensor,
        targets: torch.Tensor,
        criterion: nn.Module,
    ) -> torch.Tensor:
        """Compute loss (override in subclasses if needed)."""
        return criterion(outputs, targets)

    def _train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Train for one epoch."""
        self.model_.train()
        total_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(self._device)
            y_batch = y_batch.to(self._device)

            optimizer.zero_grad()
            outputs = self.model_(X_batch)
            loss = self._compute_loss(outputs, y_batch, criterion)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / n_batches

    def _validate_epoch(
        self,
        dataloader: DataLoader,
        criterion: nn.Module,
    ) -> float:
        """Validate for one epoch."""
        self.model_.eval()
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                outputs = self.model_(X_batch)
                loss = self._compute_loss(outputs, y_batch, criterion)

                total_loss += loss.item()
                n_batches += 1

        return total_loss / n_batches

    def _fit_impl(
        self,
        X: np.ndarray,
        y: np.ndarray,
        output_dim: int,
        criterion: nn.Module,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> EndgameEstimator:
        """Internal fit implementation."""
        self._set_seed()
        self._device = self._get_device()

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        # Create model
        self.model_ = _MLPModule(
            input_dim=X.shape[1],
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation=self.activation,
        ).to(self._device)

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._get_scheduler(optimizer, self.n_epochs)

        # Create dataloaders
        train_loader = self._create_dataloader(X_scaled, y, shuffle=True)

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            X_val_scaled = self.scaler_.transform(X_val)
            val_loader = self._create_dataloader(X_val_scaled, y_val, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training MLP on {self._device} for up to {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.history_["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._validate_epoch(val_loader, criterion)
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
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(f"Epoch {epoch + 1}/{self.n_epochs}: train_loss={train_loss:.4f}")

            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss if val_loader is None else val_loss)
                else:
                    scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation."""
        self._check_is_fitted()

        X_scaled = self.scaler_.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_tensor)

        return outputs.cpu().numpy()


class MLPClassifier(ClassifierMixin, _BaseMLPEstimator):
    """Multi-Layer Perceptron classifier.

    PyTorch-based MLP with modern techniques for tabular classification.

    Parameters
    ----------
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for regularization.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Patience for early stopping.
    class_weight : str or dict, optional
        Class weights: 'balanced' or dict mapping classes to weights.
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
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : _MLPModule
        Fitted PyTorch model.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.neural import MLPClassifier
    >>> clf = MLPClassifier(hidden_dims=[128, 64], n_epochs=50)
    >>> clf.fit(X_train, y_train, val_data=(X_val, y_val))
    >>> predictions = clf.predict(X_test)
    >>> probabilities = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 10,
        class_weight: str | dict | None = None,
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            scheduler=scheduler,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.class_weight = class_weight

        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self._label_encoder: LabelEncoder | None = None
        self._class_weights: torch.Tensor | None = None

    def _prepare_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Prepare target tensor for classification."""
        return torch.LongTensor(y)

    def _compute_class_weights(self, y: np.ndarray) -> torch.Tensor | None:
        """Compute class weights."""
        if self.class_weight is None:
            return None

        if self.class_weight == "balanced":
            from sklearn.utils.class_weight import compute_class_weight
            weights = compute_class_weight(
                "balanced", classes=np.unique(y), y=y
            )
            return torch.FloatTensor(weights)

        if isinstance(self.class_weight, dict):
            weights = np.array([
                self.class_weight.get(c, 1.0) for c in range(self.n_classes_)
            ])
            return torch.FloatTensor(weights)

        return None

    def fit(
        self,
        X,
        y,
        val_data: tuple[Any, Any] | None = None,
    ) -> MLPClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        val_data : tuple of (X_val, y_val), optional
            Validation data for early stopping.

        Returns
        -------
        self
            Fitted classifier.
        """
        X_arr, y_arr = self._validate_data(X, y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_arr)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Compute class weights
        self._class_weights = self._compute_class_weights(y_encoded)

        # Prepare validation data
        if val_data is not None:
            X_val, y_val = val_data
            X_val = self._to_numpy(X_val)
            y_val = self._label_encoder.transform(np.asarray(y_val))
            val_data = (X_val, y_val)

        # Create criterion
        if self._class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=self._class_weights.to(self._get_device()))
        else:
            criterion = nn.CrossEntropyLoss()

        return self._fit_impl(
            X_arr, y_encoded, self.n_classes_, criterion, val_data
        )

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        X_arr = self._to_numpy(X)
        logits = self._predict_impl(X_arr)

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class MLPRegressor(_BaseMLPEstimator, RegressorMixin):
    """Multi-Layer Perceptron regressor.

    PyTorch-based MLP with modern techniques for tabular regression.

    Parameters
    ----------
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for regularization.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Patience for early stopping.
    loss : str, default='mse'
        Loss function: 'mse', 'mae', 'huber'.
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
    model_ : _MLPModule
        Fitted PyTorch model.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.neural import MLPRegressor
    >>> reg = MLPRegressor(hidden_dims=[128, 64], n_epochs=50)
    >>> reg.fit(X_train, y_train, val_data=(X_val, y_val))
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 10,
        loss: str = "mse",
        scheduler: str = "cosine",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            hidden_dims=hidden_dims,
            dropout=dropout,
            batch_norm=batch_norm,
            activation=activation,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            n_epochs=n_epochs,
            batch_size=batch_size,
            early_stopping=early_stopping,
            scheduler=scheduler,
            device=device,
            random_state=random_state,
            verbose=verbose,
        )
        self.loss = loss

        self._target_scaler: StandardScaler | None = None

    def _prepare_target_tensor(self, y: np.ndarray) -> torch.Tensor:
        """Prepare target tensor for regression."""
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        return torch.FloatTensor(y)

    def _get_criterion(self) -> nn.Module:
        """Get loss criterion."""
        if self.loss == "mse":
            return nn.MSELoss()
        elif self.loss == "mae":
            return nn.L1Loss()
        elif self.loss == "huber":
            return nn.HuberLoss()
        else:
            raise ValueError(f"Unknown loss: {self.loss}")

    def fit(
        self,
        X,
        y,
        val_data: tuple[Any, Any] | None = None,
    ) -> MLPRegressor:
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        val_data : tuple of (X_val, y_val), optional
            Validation data for early stopping.

        Returns
        -------
        self
            Fitted regressor.
        """
        X_arr, y_arr = self._validate_data(X, y)

        # Scale targets
        self._target_scaler = StandardScaler()
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_scaled = self._target_scaler.fit_transform(y_arr)

        # Determine output dimension
        output_dim = y_scaled.shape[1]

        # Prepare validation data
        if val_data is not None:
            X_val, y_val = val_data
            X_val = self._to_numpy(X_val)
            y_val = np.asarray(y_val)
            if y_val.ndim == 1:
                y_val = y_val.reshape(-1, 1)
            y_val = self._target_scaler.transform(y_val)
            val_data = (X_val, y_val)

        criterion = self._get_criterion()

        return self._fit_impl(X_arr, y_scaled, output_dim, criterion, val_data)

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,) or (n_samples, n_targets)
            Predicted values.
        """
        X_arr = self._to_numpy(X)
        predictions = self._predict_impl(X_arr)

        # Inverse transform
        predictions = self._target_scaler.inverse_transform(predictions)

        # Squeeze if single target
        if predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions
