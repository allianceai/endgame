from __future__ import annotations

"""MLP with Entity Embeddings for categorical features.

Entity embeddings learn dense representations for categorical variables,
enabling neural networks to effectively handle high-cardinality features.

Reference: https://arxiv.org/abs/1604.06737 (Entity Embeddings of Categorical Variables)
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
            "Install with: pip install torch"
        )


class _EmbeddingMLPModule(nn.Module):
    """PyTorch module for MLP with entity embeddings.

    Parameters
    ----------
    n_continuous : int
        Number of continuous features.
    categorical_dims : List[Tuple[int, int]]
        List of (cardinality, embedding_dim) for each categorical feature.
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
    embedding_dropout : float
        Dropout rate for embeddings.
    """

    def __init__(
        self,
        n_continuous: int,
        categorical_dims: list[tuple[int, int]],
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.3,
        batch_norm: bool = True,
        activation: str = "relu",
        embedding_dropout: float = 0.1,
    ):
        super().__init__()

        self.n_continuous = n_continuous
        self.categorical_dims = categorical_dims

        # Create embeddings for each categorical feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, embed_dim)
            for cardinality, embed_dim in categorical_dims
        ])

        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Calculate total input dimension
        total_embed_dim = sum(embed_dim for _, embed_dim in categorical_dims)
        input_dim = n_continuous + total_embed_dim

        # Activation function
        self.activation_fn = self._get_activation(activation)

        # Build hidden layers
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

        # Initialize embeddings
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight)

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

    def forward(
        self,
        x_continuous: torch.Tensor,
        x_categorical: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x_continuous : Tensor of shape (batch_size, n_continuous)
            Continuous features.
        x_categorical : Tensor of shape (batch_size, n_categorical)
            Categorical features (integer encoded).

        Returns
        -------
        Tensor
            Output predictions.
        """
        # Get embeddings for each categorical feature
        embedded = []
        for i, embedding in enumerate(self.embeddings):
            embedded.append(embedding(x_categorical[:, i]))

        # Concatenate embeddings
        if embedded:
            x_embed = torch.cat(embedded, dim=1)
            x_embed = self.embedding_dropout(x_embed)

            # Concatenate with continuous features
            if self.n_continuous > 0:
                x = torch.cat([x_continuous, x_embed], dim=1)
            else:
                x = x_embed
        else:
            x = x_continuous

        # Forward through hidden layers
        x = self.hidden_layers(x)
        return self.output_layer(x)

    def get_embeddings(self, feature_idx: int) -> np.ndarray:
        """Get embedding weights for a categorical feature.

        Parameters
        ----------
        feature_idx : int
            Index of the categorical feature.

        Returns
        -------
        ndarray
            Embedding weight matrix.
        """
        return self.embeddings[feature_idx].weight.detach().cpu().numpy()


class _BaseEmbeddingMLP(EndgameEstimator):
    """Base class for EmbeddingMLP estimators.

    Parameters
    ----------
    categorical_features : List[str] or List[int], optional
        Names or indices of categorical features.
    embedding_dims : Dict[str, int] or int, optional
        Embedding dimensions: dict mapping feature names to dims,
        or int for default dimension (uses rule: min(50, (cardinality+1)//2)).
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for hidden layers.
    embedding_dropout : float, default=0.1
        Dropout rate for embeddings.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Early stopping patience.
    scheduler : str, default='cosine'
        Learning rate scheduler.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        categorical_features: list[str] | list[int] | None = None,
        embedding_dims: dict[str, int] | int | None = None,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        embedding_dropout: float = 0.1,
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

        self.categorical_features = categorical_features
        self.embedding_dims = embedding_dims
        self.hidden_dims = hidden_dims or [256, 128]
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.batch_norm = batch_norm
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.scheduler = scheduler
        self.device = device

        # Model components
        self.model_: _EmbeddingMLPModule | None = None
        self.scaler_: StandardScaler | None = None
        self._device: torch.device | None = None
        self._cat_encoders: dict[int, LabelEncoder] = {}
        self._cat_indices: list[int] = []
        self._cont_indices: list[int] = []
        self._categorical_dims: list[tuple[int, int]] = []
        self._feature_names: list[str] | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

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

    def _infer_feature_indices(self, X: np.ndarray, feature_names: list[str]):
        """Infer categorical and continuous feature indices."""
        n_features = X.shape[1]

        if self.categorical_features is None:
            # Auto-detect: features with few unique values or non-float dtype
            self._cat_indices = []
            for i in range(n_features):
                unique = np.unique(X[:, i])
                # Consider categorical if <= 50 unique values and looks like integers
                if len(unique) <= 50 and np.allclose(unique, unique.astype(int)):
                    self._cat_indices.append(i)
        elif isinstance(self.categorical_features[0], str):
            # Feature names provided
            self._cat_indices = [
                feature_names.index(name)
                for name in self.categorical_features
                if name in feature_names
            ]
        else:
            # Indices provided
            self._cat_indices = list(self.categorical_features)

        self._cont_indices = [
            i for i in range(n_features) if i not in self._cat_indices
        ]

    def _compute_embedding_dim(self, cardinality: int, feature_name: str) -> int:
        """Compute embedding dimension for a categorical feature."""
        if isinstance(self.embedding_dims, dict):
            if feature_name in self.embedding_dims:
                return self.embedding_dims[feature_name]

        if isinstance(self.embedding_dims, int):
            return self.embedding_dims

        # Default rule: min(50, (cardinality + 1) // 2)
        return min(50, (cardinality + 1) // 2)

    def _prepare_data(
        self,
        X: np.ndarray,
        fit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare continuous and categorical data.

        Parameters
        ----------
        X : ndarray
            Input features.
        fit : bool
            Whether to fit encoders/scalers.

        Returns
        -------
        X_cont : ndarray
            Scaled continuous features.
        X_cat : ndarray
            Encoded categorical features.
        """
        # Continuous features
        if self._cont_indices:
            X_cont = X[:, self._cont_indices].astype(np.float32)
            if fit:
                self.scaler_ = StandardScaler()
                X_cont = self.scaler_.fit_transform(X_cont)
            else:
                X_cont = self.scaler_.transform(X_cont)
        else:
            X_cont = np.zeros((X.shape[0], 0), dtype=np.float32)

        # Categorical features
        if self._cat_indices:
            X_cat = np.zeros((X.shape[0], len(self._cat_indices)), dtype=np.int64)
            for j, i in enumerate(self._cat_indices):
                if fit:
                    self._cat_encoders[i] = LabelEncoder()
                    X_cat[:, j] = self._cat_encoders[i].fit_transform(X[:, i].astype(str))
                else:
                    # Handle unseen categories
                    col = X[:, i].astype(str)
                    known = set(self._cat_encoders[i].classes_)
                    col = np.array([c if c in known else self._cat_encoders[i].classes_[0] for c in col])
                    X_cat[:, j] = self._cat_encoders[i].transform(col)
        else:
            X_cat = np.zeros((X.shape[0], 0), dtype=np.int64)

        return X_cont, X_cat

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
        X_cont: np.ndarray,
        X_cat: np.ndarray,
        y: np.ndarray,
        shuffle: bool = True,
    ) -> DataLoader:
        """Create a DataLoader."""
        X_cont_tensor = torch.FloatTensor(X_cont)
        X_cat_tensor = torch.LongTensor(X_cat)
        y_tensor = self._prepare_target_tensor(y)

        dataset = TensorDataset(X_cont_tensor, X_cat_tensor, y_tensor)
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

        for X_cont, X_cat, y_batch in dataloader:
            X_cont = X_cont.to(self._device)
            X_cat = X_cat.to(self._device)
            y_batch = y_batch.to(self._device)

            optimizer.zero_grad()
            outputs = self.model_(X_cont, X_cat)
            loss = criterion(outputs, y_batch)
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
            for X_cont, X_cat, y_batch in dataloader:
                X_cont = X_cont.to(self._device)
                X_cat = X_cat.to(self._device)
                y_batch = y_batch.to(self._device)

                outputs = self.model_(X_cont, X_cat)
                loss = criterion(outputs, y_batch)

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

        # Get feature names
        self._feature_names = [f"f{i}" for i in range(X.shape[1])]

        # Infer feature indices
        self._infer_feature_indices(X, self._feature_names)

        # Prepare data
        X_cont, X_cat = self._prepare_data(X, fit=True)

        # Compute categorical dimensions
        self._categorical_dims = []
        for j, i in enumerate(self._cat_indices):
            cardinality = len(self._cat_encoders[i].classes_)
            embed_dim = self._compute_embedding_dim(cardinality, self._feature_names[i])
            self._categorical_dims.append((cardinality, embed_dim))

        # Create model
        self.model_ = _EmbeddingMLPModule(
            n_continuous=len(self._cont_indices),
            categorical_dims=self._categorical_dims,
            hidden_dims=self.hidden_dims,
            output_dim=output_dim,
            dropout=self.dropout,
            batch_norm=self.batch_norm,
            activation=self.activation,
            embedding_dropout=self.embedding_dropout,
        ).to(self._device)

        # Create optimizer and scheduler
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = self._get_scheduler(optimizer, self.n_epochs)

        # Create dataloaders
        train_loader = self._create_dataloader(X_cont, X_cat, y, shuffle=True)

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            X_val_cont, X_val_cat = self._prepare_data(X_val, fit=False)
            val_loader = self._create_dataloader(X_val_cont, X_val_cat, y_val, shuffle=False)

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training EmbeddingMLP on {self._device}...")
        self._log(f"Continuous features: {len(self._cont_indices)}, Categorical features: {len(self._cat_indices)}")

        for epoch in range(self.n_epochs):
            train_loss = self._train_epoch(train_loader, optimizer, criterion)
            self.history_["train_loss"].append(train_loss)

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

            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(train_loss if val_loader is None else val_loss)
                else:
                    scheduler.step()

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def _predict_impl(self, X: np.ndarray) -> np.ndarray:
        """Internal predict implementation."""
        self._check_is_fitted()

        X_cont, X_cat = self._prepare_data(X, fit=False)

        X_cont_tensor = torch.FloatTensor(X_cont).to(self._device)
        X_cat_tensor = torch.LongTensor(X_cat).to(self._device)

        self.model_.eval()
        with torch.no_grad():
            outputs = self.model_(X_cont_tensor, X_cat_tensor)

        return outputs.cpu().numpy()

    def get_embeddings(self, feature: str | int) -> np.ndarray:
        """Get learned embeddings for a categorical feature.

        Parameters
        ----------
        feature : str or int
            Feature name or index.

        Returns
        -------
        ndarray of shape (cardinality, embedding_dim)
            Embedding weights.
        """
        self._check_is_fitted()

        if isinstance(feature, str):
            if feature not in self._feature_names:
                raise ValueError(f"Unknown feature: {feature}")
            feature_idx = self._feature_names.index(feature)
        else:
            feature_idx = feature

        if feature_idx not in self._cat_indices:
            raise ValueError(f"Feature {feature} is not categorical")

        cat_pos = self._cat_indices.index(feature_idx)
        return self.model_.get_embeddings(cat_pos)


class EmbeddingMLPClassifier(ClassifierMixin, _BaseEmbeddingMLP):
    """MLP classifier with entity embeddings for categorical features.

    Learns dense representations for categorical variables, enabling
    effective handling of high-cardinality features.

    Parameters
    ----------
    categorical_features : List[str] or List[int], optional
        Names or indices of categorical features.
        If None, auto-detects based on unique values.
    embedding_dims : Dict[str, int] or int, optional
        Embedding dimensions per feature or default dimension.
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for hidden layers.
    embedding_dropout : float, default=0.1
        Dropout rate for embeddings.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Early stopping patience.
    class_weight : str or dict, optional
        Class weights: 'balanced' or dict.
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
    model_ : _EmbeddingMLPModule
        Fitted PyTorch model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.models.neural import EmbeddingMLPClassifier
    >>> clf = EmbeddingMLPClassifier(
    ...     categorical_features=['category', 'brand'],
    ...     embedding_dims={'category': 10, 'brand': 8},
    ...     hidden_dims=[128, 64]
    ... )
    >>> clf.fit(X_train, y_train, val_data=(X_val, y_val))
    >>> predictions = clf.predict(X_test)
    >>> # Get learned embeddings
    >>> category_embeddings = clf.get_embeddings('category')
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        categorical_features: list[str] | list[int] | None = None,
        embedding_dims: dict[str, int] | int | None = None,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        embedding_dropout: float = 0.1,
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
            categorical_features=categorical_features,
            embedding_dims=embedding_dims,
            hidden_dims=hidden_dims,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
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
    ) -> EmbeddingMLPClassifier:
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
        """Predict class labels."""
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        X_arr = self._to_numpy(X)
        logits = self._predict_impl(X_arr)

        # Apply softmax
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


class EmbeddingMLPRegressor(_BaseEmbeddingMLP, RegressorMixin):
    """MLP regressor with entity embeddings for categorical features.

    Learns dense representations for categorical variables, enabling
    effective handling of high-cardinality features.

    Parameters
    ----------
    categorical_features : List[str] or List[int], optional
        Names or indices of categorical features.
    embedding_dims : Dict[str, int] or int, optional
        Embedding dimensions per feature or default dimension.
    hidden_dims : List[int], default=[256, 128]
        Hidden layer dimensions.
    dropout : float, default=0.3
        Dropout rate for hidden layers.
    embedding_dropout : float, default=0.1
        Dropout rate for embeddings.
    batch_norm : bool, default=True
        Whether to use batch normalization.
    activation : str, default='relu'
        Activation function.
    learning_rate : float, default=1e-3
        Initial learning rate.
    weight_decay : float, default=1e-5
        L2 regularization strength.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=10
        Early stopping patience.
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
    model_ : _EmbeddingMLPModule
        Fitted PyTorch model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.models.neural import EmbeddingMLPRegressor
    >>> reg = EmbeddingMLPRegressor(
    ...     categorical_features=['store_id', 'product_id'],
    ...     embedding_dims=16
    ... )
    >>> reg.fit(X_train, y_train, val_data=(X_val, y_val))
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        categorical_features: list[str] | list[int] | None = None,
        embedding_dims: dict[str, int] | int | None = None,
        hidden_dims: list[int] = None,
        dropout: float = 0.3,
        embedding_dropout: float = 0.1,
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
            categorical_features=categorical_features,
            embedding_dims=embedding_dims,
            hidden_dims=hidden_dims,
            dropout=dropout,
            embedding_dropout=embedding_dropout,
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
    ) -> EmbeddingMLPRegressor:
        """Fit the regressor."""
        X_arr, y_arr = self._validate_data(X, y)

        # Scale targets
        self._target_scaler = StandardScaler()
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)
        y_scaled = self._target_scaler.fit_transform(y_arr)

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
        """Predict target values."""
        X_arr = self._to_numpy(X)
        predictions = self._predict_impl(X_arr)

        predictions = self._target_scaler.inverse_transform(predictions)

        if predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions
