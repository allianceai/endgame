from __future__ import annotations

"""GAMI-Net: Generalized Additive Models with Structured Interactions.

GAMI-Net is a neural network that learns GAM-style additive main effects
plus a sparse set of pairwise interactions. It provides interpretable
shape functions while automatically detecting important interactions.

References
----------
- Yang et al. "GAMI-Net: An Explainable Neural Network Based on
  Generalized Additive Models with Structured Interactions" (2021)
- https://github.com/zzzace2000/GAMI-Net

Note: This is a PyTorch implementation (the original is TensorFlow).

Example
-------
>>> from endgame.models.interpretable import GAMINetClassifier
>>> clf = GAMINetClassifier(interact_num=10, main_hidden_units=[64, 32])
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> main_effects, interactions = clf.get_effects(X_test)
"""

from itertools import combinations
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

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
            "PyTorch is required for GAMI-Net. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _MainEffectNet(nn.Module):
    """Network for a single main effect f_i(x_i).

    A small MLP that takes a single feature as input and outputs
    its contribution to the prediction.
    """

    def __init__(
        self,
        hidden_units: list[int] = [64, 32],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = 1

        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden

        # Output layer
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 1)

        Returns
        -------
        output : Tensor of shape (batch, 1)
        """
        return self.net(x)


class _BatchedMainEffectNet(nn.Module):
    """Batched main effect networks for all features at once.

    Uses grouped 1D convolutions to process all features in parallel,
    avoiding the sequential Python loop over individual feature nets.
    Each "group" in the convolution corresponds to one feature's network.
    """

    def __init__(
        self,
        n_features: int,
        hidden_units: list[int] = [32, 16],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.n_features = n_features

        layers = nn.ModuleList()
        in_dim = 1

        for hidden in hidden_units:
            layers.append(nn.Conv1d(
                in_channels=n_features * in_dim,
                out_channels=n_features * hidden,
                kernel_size=1,
                groups=n_features,
            ))
            in_dim = hidden

        layers.append(nn.Conv1d(
            in_channels=n_features * in_dim,
            out_channels=n_features,
            kernel_size=1,
            groups=n_features,
        ))

        self.layers = layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, n_features)

        Returns
        -------
        output : Tensor of shape (batch, n_features)
        """
        # x: (batch, n_features) -> (batch, n_features, 1) for Conv1d
        h = x.unsqueeze(-1)

        for i, layer in enumerate(self.layers):
            if i > 0:
                if self.activation == "relu":
                    h = F.relu(h)
                elif self.activation == "tanh":
                    h = torch.tanh(h)
                elif self.activation == "silu":
                    h = F.silu(h)
                if self.dropout is not None:
                    h = self.dropout(h)
            h = layer(h)

        return h.squeeze(-1)  # (batch, n_features)


class _InteractionNet(nn.Module):
    """Network for a pairwise interaction f_ij(x_i, x_j).

    A small MLP that takes two features as input and outputs
    their interaction contribution.
    """

    def __init__(
        self,
        hidden_units: list[int] = [32, 16],
        activation: str = "relu",
        dropout: float = 0.0,
    ):
        super().__init__()

        layers = []
        in_dim = 2  # Two input features

        for hidden in hidden_units:
            layers.append(nn.Linear(in_dim, hidden))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "silu":
                layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden

        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, 2)
            Two feature values.

        Returns
        -------
        output : Tensor of shape (batch, 1)
        """
        return self.net(x)


class _GAMINetModule(nn.Module):
    """GAMI-Net PyTorch module.

    Combines batched main effect networks and interaction networks.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        main_hidden_units: list[int] = [32, 16],
        interact_hidden_units: list[int] = [16, 8],
        interaction_pairs: list[tuple[int, int]] | None = None,
        activation: str = "relu",
        dropout: float = 0.0,
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.is_regression = is_regression

        self.main_net = _BatchedMainEffectNet(
            n_features=n_features,
            hidden_units=main_hidden_units,
            activation=activation,
            dropout=dropout,
        )

        self.interaction_pairs = interaction_pairs or []
        self.interact_nets = nn.ModuleList([
            _InteractionNet(
                hidden_units=interact_hidden_units,
                activation=activation,
                dropout=dropout,
            )
            for _ in self.interaction_pairs
        ])

        output_dim = 1 if is_regression or n_classes <= 2 else n_classes
        self.bias = nn.Parameter(torch.zeros(output_dim))

        if not is_regression and n_classes > 2:
            total_effects = n_features + len(self.interaction_pairs)
            self.output_layer = nn.Linear(total_effects, n_classes, bias=False)
        else:
            self.output_layer = None

    def forward(
        self,
        x: torch.Tensor,
        return_effects: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : Tensor of shape (batch, n_features)
        return_effects : bool
            If True, return main effects and interactions separately.

        Returns
        -------
        output : Tensor
        main_effects : Tensor, optional
        interactions : Tensor, optional
        """
        batch_size = x.shape[0]

        main_effects = self.main_net(x)  # (batch, n_features)

        n_interactions = len(self.interaction_pairs)
        interactions = torch.zeros(batch_size, max(n_interactions, 1), device=x.device)

        for idx, (i, j) in enumerate(self.interaction_pairs):
            pair_input = torch.stack([x[:, i], x[:, j]], dim=1)
            interactions[:, idx] = self.interact_nets[idx](pair_input).squeeze(-1)

        if self.output_layer is not None:
            all_effects = torch.cat([main_effects, interactions[:, :n_interactions]], dim=1)
            output = self.output_layer(all_effects) + self.bias
        else:
            total = main_effects.sum(dim=1, keepdim=True)
            if n_interactions > 0:
                total = total + interactions[:, :n_interactions].sum(dim=1, keepdim=True)
            output = total + self.bias

        if return_effects:
            return output, main_effects, interactions[:, :n_interactions] if n_interactions > 0 else torch.zeros(batch_size, 0, device=x.device)
        return output

    def get_main_importance(self) -> np.ndarray:
        """Get importance of main effects based on conv layer weights."""
        first_layer = self.main_net.layers[0]
        w = first_layer.weight.detach().abs()
        n_groups = self.n_features
        per_group = w.shape[0] // n_groups
        importance = w.reshape(n_groups, per_group, -1).sum(dim=(1, 2)).cpu().numpy()
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return importance


class GAMINetClassifier(ClassifierMixin, BaseEstimator):
    """GAMI-Net Classifier.

    Generalized Additive Model with Structured Interactions.
    Learns interpretable main effects plus sparse pairwise interactions.

    Parameters
    ----------
    main_hidden_units : list of int, default=[64, 32]
        Hidden layer sizes for main effect networks.

    interact_hidden_units : list of int, default=[32, 16]
        Hidden layer sizes for interaction networks.

    interact_num : int, default=10
        Maximum number of interactions to learn. Set to 0 for pure GAM.
        Interactions are selected based on learned importance.

    activation : str, default="relu"
        Activation function: "relu", "tanh", or "silu".

    dropout : float, default=0.0
        Dropout rate.

    heredity : bool, default=True
        If True, only allow interactions between features with significant
        main effects (strong heredity constraint).

    interaction_selection : str, default="greedy"
        How to select interactions:
        - "greedy": Greedily add most important interactions
        - "all": Consider all pairwise interactions (expensive)

    learning_rate : float, default=1e-3
        Learning rate.

    weight_decay : float, default=1e-5
        L2 regularization.

    n_epochs : int, default=100
        Maximum epochs.

    batch_size : int, default=128
        Batch size.

    early_stopping : int, default=20
        Early stopping patience.

    validation_fraction : float, default=0.1
        Validation fraction.

    device : str, default="auto"
        Device.

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Verbosity.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    n_features_in_ : int
        Number of features.

    interaction_pairs_ : list of tuple
        Selected interaction pairs (i, j).

    feature_importances_ : ndarray
        Main effect importances.

    Examples
    --------
    >>> from endgame.models.interpretable import GAMINetClassifier
    >>> clf = GAMINetClassifier(interact_num=10)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> main, interact = clf.get_effects(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        main_hidden_units: list[int] = [32, 16],
        interact_hidden_units: list[int] = [16, 8],
        interact_num: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
        heredity: bool = True,
        interaction_selection: str = "greedy",
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 1024,
        early_stopping: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.main_hidden_units = main_hidden_units
        self.interact_hidden_units = interact_hidden_units
        self.interact_num = interact_num
        self.activation = activation
        self.dropout = dropout
        self.heredity = heredity
        self.interaction_selection = interaction_selection
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GAMI-Net] {msg}")

    def _get_device(self):
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
    ) -> GAMINetClassifier:
        """Fit the GAMI-Net classifier.

        Uses a two-stage training:
        1. Train main effects only
        2. Select and train interactions

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        eval_set : tuple, optional
            Validation set.

        Returns
        -------
        self : GAMINetClassifier
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
        X_scaled = np.nan_to_num(self._scaler.fit_transform(X), nan=0.0)

        # Stage 1: Train main effects only
        self._log("Stage 1: Training main effects...")
        self.interaction_pairs_ = []

        self.model_ = _GAMINetModule(
            n_features=self.n_features_in_,
            n_classes=self.n_classes_,
            main_hidden_units=self.main_hidden_units,
            interact_hidden_units=self.interact_hidden_units,
            interaction_pairs=[],
            activation=self.activation,
            dropout=self.dropout,
            is_regression=False,
        ).to(self._device)

        self._train_model(X_scaled, y_encoded, eval_set, stage="main")

        # Stage 2: Select and train interactions
        if self.interact_num > 0:
            self._log("Stage 2: Selecting interactions...")
            self.interaction_pairs_ = self._select_interactions(X_scaled, y_encoded)

            if len(self.interaction_pairs_) > 0:
                self._log(f"Training with {len(self.interaction_pairs_)} interactions...")

                # Create new model with interactions
                self.model_ = _GAMINetModule(
                    n_features=self.n_features_in_,
                    n_classes=self.n_classes_,
                    main_hidden_units=self.main_hidden_units,
                    interact_hidden_units=self.interact_hidden_units,
                    interaction_pairs=self.interaction_pairs_,
                    activation=self.activation,
                    dropout=self.dropout,
                    is_regression=False,
                ).to(self._device)

                self._train_model(X_scaled, y_encoded, eval_set, stage="full")

        self.model_.eval()
        self.feature_importances_ = self.model_.get_main_importance()
        self._is_fitted = True

        return self

    def _train_model(self, X, y, eval_set, stage: str = "main"):
        """Train the model."""
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)

        # Split validation
        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_val = int(len(X) * self.validation_fraction)
            if n_val >= 1:
                try:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y,
                        test_size=self.validation_fraction,
                        stratify=y,
                        random_state=self.random_state,
                    )
                except ValueError:
                    X_train, X_val, y_train, y_val = train_test_split(
                        X, y,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )
                x_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.long)
                eval_set = (X_val, y_val)

        train_loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                batch_size=self.batch_size,
            )

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        n_epochs = self.n_epochs if stage == "full" else self.n_epochs // 2

        for epoch in range(n_epochs):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(x_batch)

                if self.n_classes_ == 2:
                    loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_batch.float())
                else:
                    loss = F.cross_entropy(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for x_batch, y_batch in val_loader:
                        x_batch = x_batch.to(self._device)
                        y_batch = y_batch.to(self._device)
                        logits = self.model_(x_batch)

                        if self.n_classes_ == 2:
                            loss = F.binary_cross_entropy_with_logits(logits.squeeze(), y_batch.float())
                        else:
                            loss = F.cross_entropy(logits, y_batch)

                        val_loss += loss.item()
                        n_val_batches += 1

                val_loss /= n_val_batches

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

    def _select_interactions(self, X, y) -> list[tuple[int, int]]:
        """Select important interactions using greedy search."""
        # Get main effect importances
        main_importance = self.model_.get_main_importance()

        # Candidate pairs
        if self.heredity:
            # Only consider pairs where both features have importance above median
            threshold = np.median(main_importance)
            important_features = np.where(main_importance >= threshold)[0]
            candidates = list(combinations(important_features, 2))
        else:
            candidates = list(combinations(range(self.n_features_in_), 2))

        if len(candidates) == 0:
            return []

        # Greedy selection based on residual improvement
        selected = []
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(x_tensor)

        # Score each candidate (simplified: correlation with residual)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self._device)
        residuals = y_tensor - torch.sigmoid(logits.squeeze()) if self.n_classes_ == 2 else None

        pair_scores = []
        for i, j in candidates:
            # Simple heuristic: product of features
            interaction_term = X[:, i] * X[:, j]
            if residuals is not None:
                score = abs(np.corrcoef(interaction_term, residuals.cpu().numpy())[0, 1])
            else:
                score = np.var(interaction_term)

            if not np.isnan(score):
                pair_scores.append(((i, j), score))

        # Sort by score and select top
        pair_scores.sort(key=lambda x: x[1], reverse=True)
        selected = [pair for pair, _ in pair_scores[:self.interact_num]]

        return selected

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                logits = self.model_(x_batch)

                if self.n_classes_ == 2:
                    proba = torch.sigmoid(logits)
                    proba = torch.cat([1 - proba, proba], dim=1)
                else:
                    proba = F.softmax(logits, dim=1)

                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def get_effects(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Get main effects and interactions separately.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        main_effects : ndarray of shape (n_samples, n_features)
        interactions : ndarray of shape (n_samples, n_interactions)
        """
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_main = []
        all_interact = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                _, main, interact = self.model_(x_batch, return_effects=True)
                all_main.append(main.cpu().numpy())
                all_interact.append(interact.cpu().numpy())

        return np.vstack(all_main), np.vstack(all_interact)


class GAMINetRegressor(RegressorMixin, BaseEstimator):
    """GAMI-Net Regressor.

    Same as GAMINetClassifier but for regression with MSE loss.

    Parameters
    ----------
    Same as GAMINetClassifier.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        main_hidden_units: list[int] = [32, 16],
        interact_hidden_units: list[int] = [16, 8],
        interact_num: int = 10,
        activation: str = "relu",
        dropout: float = 0.0,
        heredity: bool = True,
        interaction_selection: str = "greedy",
        learning_rate: float = 5e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 50,
        batch_size: int = 1024,
        early_stopping: int = 10,
        validation_fraction: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.main_hidden_units = main_hidden_units
        self.interact_hidden_units = interact_hidden_units
        self.interact_num = interact_num
        self.activation = activation
        self.dropout = dropout
        self.heredity = heredity
        self.interaction_selection = interaction_selection
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GAMI-Net] {msg}")

    def _get_device(self):
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, X, y, eval_set=None, **fit_params) -> GAMINetRegressor:
        """Fit the GAMI-Net regressor."""
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        self.n_features_in_ = X.shape[1]

        # Scale
        self._scaler = StandardScaler()
        X_scaled = np.nan_to_num(self._scaler.fit_transform(X), nan=0.0)

        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y).ravel()

        # Stage 1: Main effects
        self._log("Stage 1: Training main effects...")
        self.interaction_pairs_ = []

        self.model_ = _GAMINetModule(
            n_features=self.n_features_in_,
            n_classes=1,
            main_hidden_units=self.main_hidden_units,
            interact_hidden_units=self.interact_hidden_units,
            interaction_pairs=[],
            activation=self.activation,
            dropout=self.dropout,
            is_regression=True,
        ).to(self._device)

        self._train_model(X_scaled, y_scaled, eval_set, stage="main")

        # Stage 2: Interactions
        if self.interact_num > 0:
            self._log("Stage 2: Selecting interactions...")
            self.interaction_pairs_ = self._select_interactions(X_scaled, y_scaled)

            if len(self.interaction_pairs_) > 0:
                self._log(f"Training with {len(self.interaction_pairs_)} interactions...")

                self.model_ = _GAMINetModule(
                    n_features=self.n_features_in_,
                    n_classes=1,
                    main_hidden_units=self.main_hidden_units,
                    interact_hidden_units=self.interact_hidden_units,
                    interaction_pairs=self.interaction_pairs_,
                    activation=self.activation,
                    dropout=self.dropout,
                    is_regression=True,
                ).to(self._device)

                self._train_model(X_scaled, y_scaled, eval_set, stage="full")

        self.model_.eval()
        self.feature_importances_ = self.model_.get_main_importance()
        self._is_fitted = True

        return self

    def _train_model(self, X, y, eval_set, stage: str = "main"):
        """Train the model."""
        x_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_val = int(len(X) * self.validation_fraction)
            if n_val >= 1:
                X_train, X_val, y_train, y_val = train_test_split(
                    X, y,
                    test_size=self.validation_fraction,
                    random_state=self.random_state,
                )
                x_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_tensor = torch.tensor(y_train, dtype=torch.float32)
                eval_set = (X_val, y_val)

        train_loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.float32),
                ),
                batch_size=self.batch_size,
            )

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        n_epochs = self.n_epochs if stage == "full" else self.n_epochs // 2

        for epoch in range(n_epochs):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self.model_(x_batch)
                loss = F.mse_loss(pred.squeeze(), y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

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

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping:
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

    def _select_interactions(self, X, y) -> list[tuple[int, int]]:
        """Select important interactions."""
        main_importance = self.model_.get_main_importance()

        if self.heredity:
            threshold = np.median(main_importance)
            important_features = np.where(main_importance >= threshold)[0]
            candidates = list(combinations(important_features, 2))
        else:
            candidates = list(combinations(range(self.n_features_in_), 2))

        if len(candidates) == 0:
            return []

        # Get residuals
        x_tensor = torch.tensor(X, dtype=torch.float32).to(self._device)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(x_tensor).squeeze().cpu().numpy()

        residuals = y - pred

        pair_scores = []
        for i, j in candidates:
            interaction_term = X[:, i] * X[:, j]
            score = abs(np.corrcoef(interaction_term, residuals)[0, 1])
            if not np.isnan(score):
                pair_scores.append(((i, j), score))

        pair_scores.sort(key=lambda x: x[1], reverse=True)
        return [pair for pair, _ in pair_scores[:self.interact_num]]

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        check_is_fitted(self, "model_")

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

    def get_effects(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Get main effects and interactions."""
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_main = []
        all_interact = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                _, main, interact = self.model_(x_batch, return_effects=True)
                all_main.append(main.cpu().numpy())
                all_interact.append(interact.cpu().numpy())

        return np.vstack(all_main), np.vstack(all_interact)
