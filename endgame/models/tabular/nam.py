from __future__ import annotations

"""Neural Additive Models (NAM) for interpretable deep learning.

NAM learns a separate neural network for each input feature, then combines
their outputs additively. This provides interpretability similar to GAMs
while leveraging neural network expressivity.

References
----------
- Agarwal et al. "Neural Additive Models: Interpretable Machine Learning
  with Neural Nets" (2021)
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
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
            "PyTorch is required for NAM. "
            "Install with: pip install endgame-ml[tabular]"
        )


class ExU(nn.Module):
    """Exponential Unit (ExU) activation layer.

    ExU provides learnable activation functions that can capture
    complex feature shapes while maintaining interpretability.

    f(x) = (x - bias) * exp(weights)

    Parameters
    ----------
    n_in : int
        Input dimension.
    n_out : int
        Output dimension.
    """

    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_in, n_out))
        self.bias = nn.Parameter(torch.zeros(n_in))

        # Initialize weights with truncated normal
        nn.init.trunc_normal_(self.weights, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, n_in)

        Returns
        -------
        Tensor of shape (batch, n_out)
        """
        # Clamp for numerical stability
        x = (x - self.bias).unsqueeze(-1)  # (batch, n_in, 1)
        weights_exp = torch.exp(torch.clamp(self.weights, -2, 2))  # (n_in, n_out)
        return (x * weights_exp).sum(dim=1)  # (batch, n_out)


class FeatureNN(nn.Module):
    """Neural network for a single feature.

    Each feature gets its own small network that learns the
    relationship between that feature and the target.

    Parameters
    ----------
    n_hidden : int
        Number of hidden units per layer.
    n_layers : int
        Number of hidden layers.
    activation : str
        Activation function: 'relu', 'exu'.
    dropout : float
        Dropout rate.
    """

    def __init__(
        self,
        n_hidden: int = 64,
        n_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers

        layers = []

        # Input layer
        if activation == "exu":
            layers.append(ExU(1, n_hidden))
        else:
            layers.append(nn.Linear(1, n_hidden))

        # Hidden layers
        for _ in range(n_layers - 1):
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            if activation == "exu":
                layers.append(ExU(n_hidden, n_hidden))
            else:
                layers.append(nn.Linear(n_hidden, n_hidden))
                layers.append(nn.ReLU())

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        # Output layer (single value for this feature's contribution)
        layers.append(nn.Linear(n_hidden, 1))

        self.layers = nn.ModuleList(layers)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1)
            Single feature values.

        Returns
        -------
        Tensor of shape (batch, 1)
            Feature contribution to prediction.
        """
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (nn.Linear, ExU)):
                x = layer(x)
                # Apply ReLU after Linear (except last layer)
                if self.activation == "relu" and isinstance(layer, nn.Linear) and i < len(self.layers) - 1:
                    if i < len(self.layers) - 1 and not isinstance(self.layers[i + 1], nn.ReLU):
                        x = F.relu(x)
            elif isinstance(layer, nn.ReLU) or isinstance(layer, nn.Dropout):
                x = layer(x)

        return x


class _NAMModule(nn.Module):
    """PyTorch NAM module (vectorized).

    Uses grouped Conv1d to process all features in parallel instead
    of looping through per-feature networks sequentially.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_classes : int
        Number of output classes (1 for regression).
    n_hidden : int
        Hidden units per feature network.
    n_layers : int
        Layers per feature network.
    activation : str
        Activation function.
    dropout : float
        Dropout rate.
    feature_dropout : float
        Probability of dropping entire feature networks during training.
    is_regression : bool
        Whether this is a regression task.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_hidden: int = 64,
        n_layers: int = 3,
        activation: str = "relu",
        dropout: float = 0.0,
        feature_dropout: float = 0.0,
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.is_regression = is_regression
        self.feature_dropout = feature_dropout
        self.activation_name = activation

        conv_layers = nn.ModuleList()
        in_dim = 1

        # Input + hidden layers
        for layer_idx in range(n_layers):
            conv_layers.append(nn.Conv1d(
                in_channels=n_features * in_dim,
                out_channels=n_features * n_hidden,
                kernel_size=1,
                groups=n_features,
            ))
            in_dim = n_hidden

        # Output layer: each feature -> 1 output
        conv_layers.append(nn.Conv1d(
            in_channels=n_features * n_hidden,
            out_channels=n_features,
            kernel_size=1,
            groups=n_features,
        ))

        self.conv_layers = conv_layers
        self.drop = nn.Dropout(dropout) if dropout > 0 else None

        output_dim = 1 if is_regression else n_classes
        self.bias = nn.Parameter(torch.zeros(output_dim))

        if not is_regression and n_classes > 1:
            self.output_weights = nn.Linear(n_features, n_classes, bias=False)
        else:
            self.output_weights = None

    def forward(
        self,
        x: torch.Tensor,
        return_feature_contributions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Vectorized forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, n_features)
        return_feature_contributions : bool
            If True, also return individual feature contributions.

        Returns
        -------
        output : Tensor
        contributions : Tensor, optional
        """
        # x: (batch, n_features) -> (batch, n_features, 1) for Conv1d
        h = x.unsqueeze(-1)

        for i, layer in enumerate(self.conv_layers):
            h = layer(h)
            if i < len(self.conv_layers) - 1:
                if self.activation_name == "relu":
                    h = F.relu(h)
                elif self.activation_name == "tanh":
                    h = torch.tanh(h)
                elif self.activation_name == "silu":
                    h = F.silu(h)
                if self.drop is not None:
                    h = self.drop(h)

        contributions = h.squeeze(-1)  # (batch, n_features)

        if self.training and self.feature_dropout > 0:
            mask = torch.bernoulli(
                torch.full(
                    (x.shape[0], self.n_features),
                    1 - self.feature_dropout,
                    device=x.device,
                )
            )
            contributions = contributions * mask / (1 - self.feature_dropout)

        if self.output_weights is not None:
            output = self.output_weights(contributions) + self.bias
        else:
            output = contributions.sum(dim=1, keepdim=True) + self.bias

        if return_feature_contributions:
            return output, contributions
        return output

    def get_feature_importance(self) -> np.ndarray:
        """Get feature importance based on first layer weights."""
        first_layer = self.conv_layers[0]
        w = first_layer.weight.detach().abs()
        n_groups = self.n_features
        per_group = w.shape[0] // n_groups
        importance = w.reshape(n_groups, per_group, -1).sum(dim=(1, 2)).cpu().numpy()
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return importance


class NAMClassifier(ClassifierMixin, BaseEstimator):
    """Neural Additive Model for classification.

    NAM learns a separate neural network for each input feature,
    providing interpretability similar to GAMs while leveraging
    neural network expressivity. The model is fully interpretable
    as you can visualize each feature's contribution.

    Parameters
    ----------
    n_hidden : int, default=64
        Number of hidden units per feature network.
    n_layers : int, default=3
        Number of hidden layers per feature network.
    activation : str, default='relu'
        Activation function: 'relu' or 'exu' (exponential units).
        ExU can capture more complex shapes but is less stable.
    dropout : float, default=0.0
        Dropout rate within feature networks.
    feature_dropout : float, default=0.0
        Probability of dropping entire feature networks during training.
        Acts as regularization to prevent feature co-adaptation.
    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    output_regularization : float, default=0.0
        Regularization on feature network outputs to encourage sparsity.
    n_epochs : int, default=100
        Maximum number of training epochs.
    batch_size : int, default=128
        Training batch size.
    early_stopping : int, default=20
        Early stopping patience (epochs without improvement).
    validation_fraction : float, default=0.1
        Fraction of training data for validation when eval_set not provided.
    device : str, default='auto'
        Device to use: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Whether to print training progress.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    model_ : _NAMModule
        The fitted PyTorch module.
    feature_importances_ : ndarray
        Feature importance scores.
    history_ : dict
        Training history with loss values.

    Examples
    --------
    >>> from endgame.models.tabular import NAMClassifier
    >>> clf = NAMClassifier(n_hidden=64, n_layers=3)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> # Get feature contributions for interpretability
    >>> contributions = clf.get_feature_contributions(X_test)

    Notes
    -----
    NAM provides several interpretability features:
    - `get_feature_contributions(X)`: Get each feature's contribution
    - `feature_importances_`: Overall feature importance
    - `plot_feature_effects()`: Visualize learned feature shapes (if matplotlib available)

    For best results:
    - Start with default hyperparameters
    - Use feature_dropout > 0 if features are correlated
    - Try 'exu' activation for highly non-linear relationships
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_hidden: int = 32,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        feature_dropout: float = 0.0,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-5,
        output_regularization: float = 0.0,
        n_epochs: int = 50,
        batch_size: int = 1024,
        early_stopping: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_regularization = output_regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: _NAMModule | None = None
        self.feature_importances_: np.ndarray | None = None
        self._device: torch.device | None = None
        self._label_encoder: LabelEncoder | None = None
        self._scaler: StandardScaler | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[NAM] {msg}")

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> NAMClassifier:
        """Fit the NAM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. If not provided,
            uses validation_fraction of training data.
        **fit_params : dict
            Additional parameters (ignored).

        Returns
        -------
        self : NAMClassifier
            Fitted classifier.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create model
        self.model_ = _NAMModule(
            n_features=self.n_features_in_,
            n_classes=self.n_classes_,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            feature_dropout=self.feature_dropout,
            is_regression=False,
        ).to(self._device)

        # Create internal validation split if no eval_set provided
        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_val = int(len(X_scaled) * self.validation_fraction)
            if n_val >= 1:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_scaled, y_encoded,
                        test_size=self.validation_fraction,
                        stratify=y_encoded,
                        random_state=self.random_state,
                    )
                except ValueError:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X_scaled, y_encoded,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                x_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.long)
                eval_set = (X_val, y_val)

        # Data loaders
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
        )

        # Validation loader
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if not isinstance(y_val, np.ndarray) or y_val.dtype.kind not in ('i', 'u'):
                # Need to transform labels
                X_val = self._scaler.transform(np.asarray(X_val, dtype=np.float32))
                X_val = np.nan_to_num(X_val, nan=0.0)
                y_val = self._label_encoder.transform(y_val)

            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.long),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training NAM on {self._device} with {self.n_features_in_} features...")

        for epoch in range(self.n_epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()

                # Forward pass with contributions for regularization
                if self.output_regularization > 0:
                    logits, contributions = self.model_(x_batch, return_feature_contributions=True)
                    loss = F.cross_entropy(logits, y_batch)
                    # Add output regularization (L2 on contributions)
                    loss = loss + self.output_regularization * (contributions ** 2).mean()
                else:
                    logits = self.model_(x_batch)
                    loss = F.cross_entropy(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history_["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self._device)
                        y_batch = y_batch.to(self._device)

                        logits = self.model_(x_batch)
                        loss = F.cross_entropy(logits, y_batch)
                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss /= n_val_batches
                self.history_["val_loss"].append(val_loss)

                scheduler.step(val_loss)

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
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(f"Epoch {epoch+1}/{self.n_epochs}: train={train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Compute feature importances
        self.model_.eval()
        self.feature_importances_ = self.model_.get_feature_importance()

        self._is_fitted = True
        self._log(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("NAMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)

                logits = self.model_(x_batch)
                # Always use softmax - model outputs n_classes logits
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def get_feature_contributions(self, X) -> np.ndarray:
        """Get individual feature contributions for predictions.

        This is the key interpretability feature of NAM. Each feature's
        contribution shows how it affects the prediction independently.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        contributions : ndarray of shape (n_samples, n_features)
            Each feature's contribution to the prediction.
            Positive values push toward higher class indices.
        """
        if not self._is_fitted:
            raise RuntimeError("NAMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_contributions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)

                _, contributions = self.model_(x_batch, return_feature_contributions=True)
                all_contributions.append(contributions.cpu().numpy())

        return np.vstack(all_contributions)

    def plot_feature_effects(
        self,
        feature_idx: int | None = None,
        X: np.ndarray | None = None,
        n_points: int = 100,
    ):
        """Plot learned feature effect shapes.

        Parameters
        ----------
        feature_idx : int, optional
            Index of feature to plot. If None, plots all features.
        X : array-like, optional
            Data to determine feature ranges. If None, uses standard range.
        n_points : int, default=100
            Number of points to evaluate.

        Returns
        -------
        fig : matplotlib Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib is required for plotting. "
                "Install with: pip install endgame-ml[benchmark]"
            )

        if not self._is_fitted:
            raise RuntimeError("NAMClassifier has not been fitted.")

        features_to_plot = [feature_idx] if feature_idx is not None else range(self.n_features_in_)
        n_plots = len(features_to_plot)

        # Determine layout
        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        self.model_.eval()

        for plot_idx, feat_idx in enumerate(features_to_plot):
            ax = axes[plot_idx]

            # Determine feature range
            if X is not None:
                X_arr = np.asarray(X, dtype=np.float32)
                feat_min, feat_max = X_arr[:, feat_idx].min(), X_arr[:, feat_idx].max()
            else:
                feat_min, feat_max = -3, 3  # Standard normal range

            # Create input range (in scaled space)
            if X is not None:
                X_scaled = self._scaler.transform(X_arr)
                feat_min_scaled = X_scaled[:, feat_idx].min()
                feat_max_scaled = X_scaled[:, feat_idx].max()
            else:
                feat_min_scaled, feat_max_scaled = -3, 3

            x_range = np.linspace(feat_min_scaled, feat_max_scaled, n_points).reshape(-1, 1)
            x_tensor = torch.tensor(x_range, dtype=torch.float32).to(self._device)

            # Get feature network output
            with torch.no_grad():
                fnn = self.model_.feature_nns[feat_idx]
                y_effect = fnn(x_tensor).cpu().numpy().flatten()

            # Convert x back to original scale for plotting
            x_original = np.linspace(feat_min, feat_max, n_points)

            ax.plot(x_original, y_effect, 'b-', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(f'Feature {feat_idx}')
            ax.set_ylabel('Effect')
            ax.set_title(f'Feature {feat_idx} (importance: {self.feature_importances_[feat_idx]:.3f})')
            ax.grid(True, alpha=0.3)

        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig


class NAMRegressor(RegressorMixin, BaseEstimator):
    """Neural Additive Model for regression.

    Same architecture as NAMClassifier but with MSE loss for
    continuous target prediction.

    Parameters
    ----------
    n_hidden : int, default=64
        Number of hidden units per feature network.
    n_layers : int, default=3
        Number of hidden layers per feature network.
    activation : str, default='relu'
        Activation function: 'relu' or 'exu'.
    dropout : float, default=0.0
        Dropout rate within feature networks.
    feature_dropout : float, default=0.0
        Probability of dropping entire feature networks.
    learning_rate : float, default=1e-3
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    output_regularization : float, default=0.0
        Regularization on feature network outputs.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=128
        Training batch size.
    early_stopping : int, default=20
        Early stopping patience.
    validation_fraction : float, default=0.1
        Fraction for validation when eval_set not provided.
    device : str, default='auto'
        Device to use.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    model_ : _NAMModule
        Fitted model.
    feature_importances_ : ndarray
        Feature importance scores.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.models.tabular import NAMRegressor
    >>> reg = NAMRegressor(n_hidden=64)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    >>> contributions = reg.get_feature_contributions(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_hidden: int = 32,
        n_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        feature_dropout: float = 0.0,
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-5,
        output_regularization: float = 0.0,
        n_epochs: int = 50,
        batch_size: int = 1024,
        early_stopping: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.activation = activation
        self.dropout = dropout
        self.feature_dropout = feature_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.output_regularization = output_regularization
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.n_features_in_: int = 0
        self.model_: _NAMModule | None = None
        self.feature_importances_: np.ndarray | None = None
        self._device: torch.device | None = None
        self._scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[NAM] {msg}")

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> NAMRegressor:
        """Fit the NAM regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.
        **fit_params : dict
            Additional parameters (ignored).

        Returns
        -------
        self : NAMRegressor
            Fitted regressor.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Scale target
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y).ravel()

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # Create model
        self.model_ = _NAMModule(
            n_features=self.n_features_in_,
            n_classes=1,
            n_hidden=self.n_hidden,
            n_layers=self.n_layers,
            activation=self.activation,
            dropout=self.dropout,
            feature_dropout=self.feature_dropout,
            is_regression=True,
        ).to(self._device)

        # Create internal validation split if no eval_set provided
        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_val = int(len(X_scaled) * self.validation_fraction)
            if n_val >= 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_scaled, y_scaled,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                )
                x_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32)
                eval_set = (X_val, y_val)

        # Data loaders
        train_dataset = TensorDataset(x_tensor, y_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=len(train_dataset) > self.batch_size,
        )

        # Validation loader
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training NAM on {self._device} with {self.n_features_in_} features...")

        for epoch in range(self.n_epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()

                if self.output_regularization > 0:
                    pred, contributions = self.model_(x_batch, return_feature_contributions=True)
                    loss = F.mse_loss(pred.squeeze(), y_batch)
                    loss = loss + self.output_regularization * (contributions ** 2).mean()
                else:
                    pred = self.model_(x_batch)
                    loss = F.mse_loss(pred.squeeze(), y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history_["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self._device)
                        y_batch = y_batch.to(self._device)

                        pred = self.model_(x_batch)
                        loss = F.mse_loss(pred.squeeze(), y_batch)
                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss /= n_val_batches
                self.history_["val_loss"].append(val_loss)

                scheduler.step(val_loss)

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
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(f"Epoch {epoch+1}/{self.n_epochs}: train={train_loss:.4f}")

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Compute feature importances
        self.model_.eval()
        self.feature_importances_ = self.model_.get_feature_importance()

        self._is_fitted = True
        self._log(f"Training complete. Best val loss: {best_val_loss:.4f}")

        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("NAMRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_pred = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)

                pred = self.model_(x_batch)
                all_pred.append(pred.cpu().numpy())

        pred = np.vstack(all_pred).ravel()
        return self._target_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()

    def get_feature_contributions(self, X) -> np.ndarray:
        """Get individual feature contributions for predictions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        contributions : ndarray of shape (n_samples, n_features)
            Each feature's contribution to the prediction.
        """
        if not self._is_fitted:
            raise RuntimeError("NAMRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_contributions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)

                _, contributions = self.model_(x_batch, return_feature_contributions=True)
                all_contributions.append(contributions.cpu().numpy())

        return np.vstack(all_contributions)

    def plot_feature_effects(
        self,
        feature_idx: int | None = None,
        X: np.ndarray | None = None,
        n_points: int = 100,
    ):
        """Plot learned feature effect shapes.

        Parameters
        ----------
        feature_idx : int, optional
            Index of feature to plot. If None, plots all features.
        X : array-like, optional
            Data to determine feature ranges.
        n_points : int, default=100
            Number of points to evaluate.

        Returns
        -------
        fig : matplotlib Figure
            The figure object.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting.")

        if not self._is_fitted:
            raise RuntimeError("NAMRegressor has not been fitted.")

        features_to_plot = [feature_idx] if feature_idx is not None else range(self.n_features_in_)
        n_plots = len(features_to_plot)

        n_cols = min(4, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        self.model_.eval()

        for plot_idx, feat_idx in enumerate(features_to_plot):
            ax = axes[plot_idx]

            if X is not None:
                X_arr = np.asarray(X, dtype=np.float32)
                feat_min, feat_max = X_arr[:, feat_idx].min(), X_arr[:, feat_idx].max()
                X_scaled = self._scaler.transform(X_arr)
                feat_min_scaled = X_scaled[:, feat_idx].min()
                feat_max_scaled = X_scaled[:, feat_idx].max()
            else:
                feat_min, feat_max = -3, 3
                feat_min_scaled, feat_max_scaled = -3, 3

            x_range = np.linspace(feat_min_scaled, feat_max_scaled, n_points).reshape(-1, 1)
            x_tensor = torch.tensor(x_range, dtype=torch.float32).to(self._device)

            with torch.no_grad():
                fnn = self.model_.feature_nns[feat_idx]
                y_effect = fnn(x_tensor).cpu().numpy().flatten()

            x_original = np.linspace(feat_min, feat_max, n_points)

            ax.plot(x_original, y_effect, 'b-', linewidth=2)
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(f'Feature {feat_idx}')
            ax.set_ylabel('Effect')
            ax.set_title(f'Feature {feat_idx} (importance: {self.feature_importances_[feat_idx]:.3f})')
            ax.grid(True, alpha=0.3)

        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        return fig
