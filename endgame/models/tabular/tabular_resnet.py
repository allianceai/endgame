from __future__ import annotations

"""Tabular ResNet implementation.

Based on the ResNet architecture from:
"Revisiting Deep Learning Models for Tabular Data" (Gorishniy et al., 2021)

This implements a simple residual network adapted for tabular data with
feature embeddings and skip connections.
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for TabularResNet. "
            "Install with: pip install torch"
        )


class _ResidualBlock(nn.Module):
    """Residual block for tabular data.

    Architecture: Linear -> BatchNorm -> ReLU -> Dropout -> Linear -> skip connection
    """

    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(d_in)
        self.linear1 = nn.Linear(d_in, d_hidden)
        self.norm2 = nn.BatchNorm1d(d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x = F.relu(x)
        x = self.linear1(x)
        x = self.norm2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x + residual


class _TabularResNetModule(nn.Module):
    """PyTorch module for Tabular ResNet."""

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        d_embedding: int = 128,
        n_blocks: int = 4,
        d_hidden_factor: float = 2.0,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.n_features = n_features
        self.d_embedding = d_embedding

        # Input embedding
        self.input_linear = nn.Linear(n_features, d_embedding)
        self.input_norm = nn.BatchNorm1d(d_embedding)

        # Residual blocks
        d_hidden = int(d_embedding * d_hidden_factor)
        self.blocks = nn.ModuleList([
            _ResidualBlock(d_embedding, d_hidden, dropout)
            for _ in range(n_blocks)
        ])

        # Output
        self.output_norm = nn.BatchNorm1d(d_embedding)
        self.output_linear = nn.Linear(d_embedding, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input embedding
        x = self.input_linear(x)
        x = self.input_norm(x)
        x = F.relu(x)

        # Residual blocks
        for block in self.blocks:
            x = block(x)

        # Output
        x = self.output_norm(x)
        x = F.relu(x)
        x = self.output_linear(x)

        return x


class TabularResNetClassifier(BaseEstimator, ClassifierMixin):
    """Tabular ResNet Classifier.

    A residual network architecture adapted for tabular data, following
    the design from "Revisiting Deep Learning Models for Tabular Data".

    Parameters
    ----------
    d_embedding : int, default=128
        Dimension of the embedding layer.
    n_blocks : int, default=4
        Number of residual blocks.
    d_hidden_factor : float, default=2.0
        Hidden layer size = d_embedding * d_hidden_factor.
    dropout : float, default=0.1
        Dropout rate.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=256
        Batch size for training.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        Weight decay (L2 regularization).
    early_stopping : int, default=20
        Number of epochs without improvement before stopping.
    device : str, default='auto'
        Device to use ('auto', 'cuda', 'cpu').
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> from endgame.models.tabular import TabularResNetClassifier
    >>> clf = TabularResNetClassifier(n_epochs=50)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    """

    def __init__(
        self,
        d_embedding: int = 128,
        n_blocks: int = 4,
        d_hidden_factor: float = 2.0,
        dropout: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.d_embedding = d_embedding
        self.n_blocks = n_blocks
        self.d_hidden_factor = d_hidden_factor
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None = None,
    ) -> TabularResNetClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.

        Returns
        -------
        self
        """
        _check_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = self._get_device()

        # Preprocess
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(X)

        self.label_encoder_ = LabelEncoder()
        y = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        self.n_classes_ = len(self.classes_)

        # Create model
        n_features = X.shape[1]
        self.module_ = _TabularResNetModule(
            n_features=n_features,
            n_outputs=self.n_classes_,
            d_embedding=self.d_embedding,
            n_blocks=self.n_blocks,
            d_hidden_factor=self.d_hidden_factor,
            dropout=self.dropout,
        ).to(device)

        # Create data loaders
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self.scaler_.transform(X_val)
            y_val = self.label_encoder_.transform(y_val)
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.long)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.module_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.module_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                logits = self.module_(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if eval_set is not None:
                self.module_.eval()
                with torch.no_grad():
                    val_logits = self.module_(X_val_tensor.to(device))
                    val_loss = F.cross_entropy(val_logits, y_val_tensor.to(device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.n_epochs}, Train Loss: {train_loss:.4f}"
                if eval_set is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        proba : array of shape (n_samples, n_classes)
            Class probabilities.
        """
        _check_torch()

        device = self._get_device()
        X = self.scaler_.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.module_.eval()
        with torch.no_grad():
            logits = self.module_(X_tensor)
            proba = F.softmax(logits, dim=1).cpu().numpy()

        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        y_pred = np.argmax(proba, axis=1)
        return self.label_encoder_.inverse_transform(y_pred)


class TabularResNetRegressor(BaseEstimator, RegressorMixin):
    """Tabular ResNet Regressor.

    A residual network architecture adapted for tabular regression.

    Parameters
    ----------
    d_embedding : int, default=128
        Dimension of the embedding layer.
    n_blocks : int, default=4
        Number of residual blocks.
    d_hidden_factor : float, default=2.0
        Hidden layer size = d_embedding * d_hidden_factor.
    dropout : float, default=0.1
        Dropout rate.
    n_epochs : int, default=100
        Number of training epochs.
    batch_size : int, default=256
        Batch size for training.
    lr : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        Weight decay (L2 regularization).
    early_stopping : int, default=20
        Number of epochs without improvement before stopping.
    device : str, default='auto'
        Device to use ('auto', 'cuda', 'cpu').
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        d_embedding: int = 128,
        n_blocks: int = 4,
        d_hidden_factor: float = 2.0,
        dropout: float = 0.1,
        n_epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-5,
        early_stopping: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.d_embedding = d_embedding
        self.n_blocks = n_blocks
        self.d_hidden_factor = d_hidden_factor
        self.dropout = dropout
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None = None,
    ) -> TabularResNetRegressor:
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.

        Returns
        -------
        self
        """
        _check_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = self._get_device()

        # Preprocess
        self.scaler_X_ = StandardScaler()
        X = self.scaler_X_.fit_transform(X)

        self.scaler_y_ = StandardScaler()
        y = self.scaler_y_.fit_transform(y.reshape(-1, 1)).ravel()

        # Create model
        n_features = X.shape[1]
        self.module_ = _TabularResNetModule(
            n_features=n_features,
            n_outputs=1,
            d_embedding=self.d_embedding,
            n_blocks=self.n_blocks,
            d_hidden_factor=self.d_hidden_factor,
            dropout=self.dropout,
        ).to(device)

        # Create data loaders
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        train_dataset = TensorDataset(X_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = self.scaler_X_.transform(X_val)
            y_val = self.scaler_y_.transform(y_val.reshape(-1, 1)).ravel()
            X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)

        # Optimizer
        optimizer = torch.optim.AdamW(
            self.module_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.module_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                pred = self.module_(X_batch)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)

            # Validation
            if eval_set is not None:
                self.module_.eval()
                with torch.no_grad():
                    val_pred = self.module_(X_val_tensor.to(device))
                    val_loss = F.mse_loss(val_pred, y_val_tensor.to(device)).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch + 1}")
                    break

            if self.verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{self.n_epochs}, Train Loss: {train_loss:.4f}"
                if eval_set is not None:
                    msg += f", Val Loss: {val_loss:.4f}"
                print(msg)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features.

        Returns
        -------
        y_pred : array of shape (n_samples,)
            Predicted target values.
        """
        _check_torch()

        device = self._get_device()
        X = self.scaler_X_.transform(X)
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        self.module_.eval()
        with torch.no_grad():
            pred = self.module_(X_tensor).cpu().numpy().ravel()

        # Inverse transform
        pred = self.scaler_y_.inverse_transform(pred.reshape(-1, 1)).ravel()
        return pred
