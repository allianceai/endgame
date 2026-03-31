"""FNN-TSK (Deep Feedforward Fuzzy Neural Network with TSK layers).

Each layer is a complete TSK fuzzy system. Stacking multiple layers
creates a deep fuzzy architecture that is differentiable end-to-end.

References
----------
- Wu et al., "A Deep Fuzzy System" (2020)

Example
-------
>>> from endgame.fuzzy.neurofuzzy import FNNTSKRegressor
>>> model = FNNTSKRegressor(n_layers=3, n_rules_per_layer=5, n_epochs=100)
>>> model.fit(X_train, y_train)
>>> y_pred = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for FNN-TSK. Install with: pip install torch"
        )


if HAS_TORCH:
    class _TSKLayer(nn.Module):
        """Single TSK fuzzy layer.

        Input → Gaussian MFs → product t-norm → normalized weights → TSK output.
        """

        def __init__(self, n_input: int, n_rules: int, n_output: int):
            super().__init__()
            self.n_input = n_input
            self.n_rules = n_rules
            self.n_output = n_output

            # Gaussian MF parameters: centers and log-sigmas per rule per feature
            self.centers = nn.Parameter(torch.randn(n_rules, n_input))
            self.log_sigmas = nn.Parameter(torch.zeros(n_rules, n_input))

            # TSK order-1 consequent: w_r @ [x; 1] for each rule
            self.consequent = nn.Linear(n_input + 1, n_output * n_rules)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]
            sigmas = torch.exp(self.log_sigmas) + 1e-6  # (n_rules, n_input)

            # Membership: (batch, 1, n_input) vs (1, n_rules, n_input)
            x_exp = x.unsqueeze(1)
            mu = torch.exp(
                -0.5 * ((x_exp - self.centers.unsqueeze(0)) / sigmas.unsqueeze(0)) ** 2
            )  # (batch, n_rules, n_input)

            # Firing strength: product t-norm across features
            firing = mu.prod(dim=2)  # (batch, n_rules)
            firing_norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-10)

            # Consequent: linear in input
            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            cons_all = self.consequent(x_aug)  # (batch, n_output * n_rules)
            cons = cons_all.view(batch_size, self.n_rules, self.n_output)

            # Weighted sum
            output = (firing_norm.unsqueeze(2) * cons).sum(dim=1)
            return output

    class _FNNTSKModule(nn.Module):
        """Deep stacked TSK network."""

        def __init__(
            self,
            n_features: int,
            n_layers: int,
            n_rules_per_layer: int,
            hidden_dim: int,
            n_outputs: int,
        ):
            super().__init__()
            layers = []
            in_dim = n_features
            for i in range(n_layers - 1):
                layers.append(_TSKLayer(in_dim, n_rules_per_layer, hidden_dim))
                in_dim = hidden_dim
            layers.append(_TSKLayer(in_dim, n_rules_per_layer, n_outputs))
            self.layers = nn.ModuleList(layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for layer in self.layers:
                x = layer(x)
            return x

        def init_from_data(self, X: torch.Tensor):
            """Initialize centers from data statistics."""
            with torch.no_grad():
                x = X
                for layer in self.layers:
                    x_min = x.min(dim=0).values
                    x_max = x.max(dim=0).values
                    n_input = layer.n_input
                    for r in range(layer.n_rules):
                        layer.centers.data[r] = (
                            x_min + (x_max - x_min) * torch.rand(n_input)
                        )
                    x_range = (x_max - x_min).clamp(min=0.1)
                    layer.log_sigmas.data[:] = torch.log(
                        x_range / (layer.n_rules * 2) + 0.01
                    )
                    # Propagate through layer for next initialization
                    x = layer(x)


class FNNTSKRegressor(BaseEstimator, RegressorMixin):
    """Deep Feedforward TSK Fuzzy Neural Network for regression.

    Parameters
    ----------
    n_layers : int, default=3
        Number of stacked TSK layers.
    n_rules_per_layer : int, default=5
        Number of rules in each layer.
    hidden_dim : int, default=10
        Dimension of intermediate layer outputs.
    n_epochs : int, default=100
        Training epochs.
    lr : float, default=0.01
        Learning rate.
    batch_size : int, default=32
        Mini-batch size.
    early_stopping : int or None, default=10
        Patience for early stopping.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_rules_per_layer: int = 5,
        hidden_dim: int = 10,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int | None = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_rules_per_layer = n_rules_per_layer
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FNNTSKRegressor:
        """Fit the deep TSK network.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        _check_torch()
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

        self.module_ = _FNNTSKModule(
            n_features=self.n_features_in_,
            n_layers=self.n_layers,
            n_rules_per_layer=self.n_rules_per_layer,
            hidden_dim=self.hidden_dim,
            n_outputs=1,
        )
        self.module_.init_from_data(X_t)

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        best_loss = float("inf")
        patience = 0
        n = len(X_t)

        for epoch in range(self.n_epochs):
            self.module_.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = self.module_(X_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}")

            if self.early_stopping is not None:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience = 0
                    self.best_state_ = {
                        k: v.clone() for k, v in self.module_.state_dict().items()
                    }
                else:
                    patience += 1
                    if patience >= self.early_stopping:
                        break

        if self.early_stopping and hasattr(self, "best_state_"):
            self.module_.load_state_dict(self.best_state_)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["module_"])
        X = check_array(X, dtype=np.float32)
        self.module_.eval()
        with torch.no_grad():
            return self.module_(torch.tensor(X, dtype=torch.float32)).numpy().ravel()


class FNNTSKClassifier(BaseEstimator, ClassifierMixin):
    """Deep Feedforward TSK Fuzzy Neural Network for classification.

    Parameters
    ----------
    n_layers : int, default=3
        Number of stacked TSK layers.
    n_rules_per_layer : int, default=5
        Number of rules per layer.
    hidden_dim : int, default=10
        Hidden dimension.
    n_epochs : int, default=100
        Training epochs.
    lr : float, default=0.01
        Learning rate.
    batch_size : int, default=32
        Mini-batch size.
    early_stopping : int or None, default=10
        Patience for early stopping.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.
    """

    def __init__(
        self,
        n_layers: int = 3,
        n_rules_per_layer: int = 5,
        hidden_dim: int = 10,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int | None = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.n_rules_per_layer = n_rules_per_layer
        self.hidden_dim = hidden_dim
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FNNTSKClassifier:
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        _check_torch()
        X, y = check_X_y(X, y, dtype=np.float32)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)

        self.module_ = _FNNTSKModule(
            n_features=self.n_features_in_,
            n_layers=self.n_layers,
            n_rules_per_layer=self.n_rules_per_layer,
            hidden_dim=self.hidden_dim,
            n_outputs=n_classes,
        )
        self.module_.init_from_data(X_t)

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        best_loss = float("inf")
        patience = 0
        n = len(X_t)

        for epoch in range(self.n_epochs):
            self.module_.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                out = self.module_(X_t[idx])
                loss = criterion(out, y_t[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.module_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            if self.early_stopping is not None:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience = 0
                    self.best_state_ = {
                        k: v.clone() for k, v in self.module_.state_dict().items()
                    }
                else:
                    patience += 1
                    if patience >= self.early_stopping:
                        break

        if self.early_stopping and hasattr(self, "best_state_"):
            self.module_.load_state_dict(self.best_state_)

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["module_"])
        X = check_array(X, dtype=np.float32)
        self.module_.eval()
        with torch.no_grad():
            logits = self.module_(torch.tensor(X, dtype=torch.float32))
            return torch.softmax(logits, dim=1).numpy()

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
