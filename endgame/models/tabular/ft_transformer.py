"""FT-Transformer: Feature Tokenizer Transformer for tabular data.

FT-Transformer transforms each feature into an embedding and applies
transformer layers, achieving state-of-the-art results on tabular data.

References
----------
- Gorishniy et al. "Revisiting Deep Learning Models for Tabular Data" (2021)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch imports (lazy loaded)
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
    _nn_Module = nn.Module
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    F = None
    optim = None
    DataLoader = None
    TensorDataset = None
    _nn_Module = object  # Fallback base class


def _check_torch():
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for FT-Transformer. "
            "Install with: pip install torch"
        )


class _NumericalEmbedding(_nn_Module):
    """Embedding layer for numerical features."""

    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(n_features, d_token))
        self.bias = nn.Parameter(torch.empty(n_features, d_token))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features)
        # output: (batch, n_features, d_token)
        return x.unsqueeze(-1) * self.weight + self.bias


class _CategoricalEmbedding(_nn_Module):
    """Embedding layer for categorical features."""

    def __init__(self, cardinalities: list[int], d_token: int):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality + 1, d_token)  # +1 for unknown
            for cardinality in cardinalities
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_cat_features) - integer encoded
        # output: (batch, n_cat_features, d_token)
        return torch.stack([
            emb(x[:, i]) for i, emb in enumerate(self.embeddings)
        ], dim=1)


class _FTTransformerBlock(_nn_Module):
    """Single transformer block for FT-Transformer."""

    def __init__(
        self,
        d_token: int,
        n_heads: int,
        d_ffn: int,
        attention_dropout: float,
        ffn_dropout: float,
        residual_dropout: float,
    ):
        super().__init__()

        self.attention = nn.MultiheadAttention(
            d_token, n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)

        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(residual_dropout),
        )

        self.residual_dropout = nn.Dropout(residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + self.residual_dropout(attn_out))

        # Feed-forward
        x = self.norm2(x + self.ffn(x))

        return x


class _FTTransformerModule(_nn_Module):
    """PyTorch module for FT-Transformer."""

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: int,
        d_token: int = 192,
        n_blocks: int = 3,
        n_heads: int = 8,
        d_ffn_factor: float = 4 / 3,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.is_regression = is_regression

        # Feature embeddings
        if n_num_features > 0:
            self.num_embedding = _NumericalEmbedding(n_num_features, d_token)
        else:
            self.num_embedding = None

        if self.n_cat_features > 0:
            self.cat_embedding = _CategoricalEmbedding(cat_cardinalities, d_token)
        else:
            self.cat_embedding = None

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_token))
        nn.init.normal_(self.cls_token, std=0.02)

        # Transformer blocks
        d_ffn = int(d_token * d_ffn_factor)
        self.blocks = nn.ModuleList([
            _FTTransformerBlock(
                d_token, n_heads, d_ffn,
                attention_dropout, ffn_dropout, residual_dropout,
            )
            for _ in range(n_blocks)
        ])

        self.norm = nn.LayerNorm(d_token)

        # Output head
        output_dim = 1 if is_regression else n_classes
        self.head = nn.Linear(d_token, output_dim)

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]

        # Embed features
        embeddings = []

        if x_num is not None and self.num_embedding is not None:
            embeddings.append(self.num_embedding(x_num))

        if x_cat is not None and self.cat_embedding is not None:
            embeddings.append(self.cat_embedding(x_cat))

        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)  # (batch, n_features, d_token)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, 1 + n_features, d_token)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Take CLS token output
        x = self.norm(x[:, 0])

        # Head
        return self.head(x)


class FTTransformerClassifier(ClassifierMixin, BaseEstimator):
    """Feature Tokenizer Transformer for tabular classification.

    Transforms each feature into an embedding and applies transformer
    layers. Currently state-of-the-art for deep learning on tabular data.

    Parameters
    ----------
    n_blocks : int, default=3
        Number of transformer blocks.
    d_token : int, default=192
        Embedding dimension for each feature token.
    n_heads : int, default=8
        Number of attention heads.
    attention_dropout : float, default=0.2
        Attention dropout rate.
    ffn_dropout : float, default=0.1
        Feed-forward dropout rate.
    residual_dropout : float, default=0.0
        Residual connection dropout.
    d_ffn_factor : float, default=4/3
        FFN hidden dimension factor (d_ffn = d_token * d_ffn_factor).
    learning_rate : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    early_stopping : int, default=15
        Early stopping patience.
    cat_cardinality_threshold : int, default=20
        Treat features with <= this many unique values as categorical.
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
    model_ : _FTTransformerModule
        Fitted PyTorch model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> from endgame.models.tabular import FTTransformerClassifier
    >>> clf = FTTransformerClassifier(n_blocks=3, d_token=192)
    >>> clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_blocks: int = 3,
        d_token: int = 192,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        d_ffn_factor: float = 4 / 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 15,
        cat_cardinality_threshold: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_blocks = n_blocks
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.d_ffn_factor = d_ffn_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.cat_cardinality_threshold = cat_cardinality_threshold
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: _FTTransformerModule | None = None
        self._device: torch.device | None = None
        self._label_encoder: LabelEncoder | None = None
        self._num_scaler: StandardScaler | None = None
        self._cat_encoders: dict[int, LabelEncoder] = {}
        self._num_feature_indices: list[int] = []
        self._cat_feature_indices: list[int] = []
        self._cat_cardinalities: list[int] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str):
        """Log if verbose."""
        if self.verbose:
            print(f"[FT-Transformer] {message}")

    def _get_device(self) -> torch.device:
        """Get computation device."""
        _check_torch()
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

    def _identify_feature_types(self, X: np.ndarray):
        """Identify numerical vs categorical features."""
        n_features = X.shape[1]

        self._num_feature_indices = []
        self._cat_feature_indices = []
        self._cat_cardinalities = []

        for i in range(n_features):
            n_unique = len(np.unique(X[:, i][~np.isnan(X[:, i])]))

            if n_unique <= self.cat_cardinality_threshold:
                self._cat_feature_indices.append(i)
                self._cat_cardinalities.append(n_unique)
            else:
                self._num_feature_indices.append(i)

    def _preprocess_features(
        self,
        X: np.ndarray,
        fit: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Preprocess features into numerical and categorical tensors."""
        # Numerical features
        x_num = None
        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices].astype(np.float32)

            if fit:
                self._num_scaler = StandardScaler()
                X_num = self._num_scaler.fit_transform(X_num)
            else:
                X_num = self._num_scaler.transform(X_num)

            # Handle NaN
            X_num = np.nan_to_num(X_num, nan=0.0)
            x_num = torch.tensor(X_num, dtype=torch.float32)

        # Categorical features
        x_cat = None
        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            X_cat_encoded = np.zeros(X_cat.shape, dtype=np.int64)

            for i, col_idx in enumerate(self._cat_feature_indices):
                if fit:
                    le = LabelEncoder()
                    # Handle unseen categories
                    col_data = X_cat[:, i].astype(str)
                    le.fit(col_data)
                    self._cat_encoders[col_idx] = le
                    X_cat_encoded[:, i] = le.transform(col_data)
                else:
                    le = self._cat_encoders[col_idx]
                    col_data = X_cat[:, i].astype(str)
                    # Handle unknown categories
                    known_mask = np.isin(col_data, le.classes_)
                    X_cat_encoded[known_mask, i] = le.transform(col_data[known_mask])
                    X_cat_encoded[~known_mask, i] = len(le.classes_)  # Unknown token

            x_cat = torch.tensor(X_cat_encoded, dtype=torch.long)

        return x_num, x_cat

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> FTTransformerClassifier:
        """Fit the FT-Transformer classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.

        Returns
        -------
        self
            Fitted classifier.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Identify feature types
        self._identify_feature_types(X)

        # Preprocess features
        x_num, x_cat = self._preprocess_features(X, fit=True)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create model
        self.model_ = _FTTransformerModule(
            n_num_features=len(self._num_feature_indices),
            cat_cardinalities=self._cat_cardinalities,
            n_classes=self.n_classes_,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            is_regression=False,
        ).to(self._device)

        # Create data loaders
        if x_num is not None and x_cat is not None:
            train_dataset = TensorDataset(x_num, x_cat, y_tensor)
        elif x_num is not None:
            train_dataset = TensorDataset(x_num, y_tensor)
        else:
            train_dataset = TensorDataset(x_cat, y_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        # Validation data
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val)
            y_val_encoded = self._label_encoder.transform(y_val)

            x_num_val, x_cat_val = self._preprocess_features(X_val, fit=False)
            y_val_tensor = torch.tensor(y_val_encoded, dtype=torch.long)

            if x_num_val is not None and x_cat_val is not None:
                val_dataset = TensorDataset(x_num_val, x_cat_val, y_val_tensor)
            elif x_num_val is not None:
                val_dataset = TensorDataset(x_num_val, y_val_tensor)
            else:
                val_dataset = TensorDataset(x_cat_val, y_val_tensor)

            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Optimizer
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device}...")

        for epoch in range(self.n_epochs):
            # Train
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if len(batch) == 3:
                    x_num_batch, x_cat_batch, y_batch = batch
                    x_num_batch = x_num_batch.to(self._device)
                    x_cat_batch = x_cat_batch.to(self._device)
                elif x_num is not None:
                    x_num_batch, y_batch = batch
                    x_num_batch = x_num_batch.to(self._device)
                    x_cat_batch = None
                else:
                    x_cat_batch, y_batch = batch
                    x_cat_batch = x_cat_batch.to(self._device)
                    x_num_batch = None

                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(x_num_batch, x_cat_batch)
                loss = F.cross_entropy(logits, y_batch)
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
                    for batch in val_loader:
                        if len(batch) == 3:
                            x_num_batch, x_cat_batch, y_batch = batch
                            x_num_batch = x_num_batch.to(self._device)
                            x_cat_batch = x_cat_batch.to(self._device)
                        elif x_num is not None:
                            x_num_batch, y_batch = batch
                            x_num_batch = x_num_batch.to(self._device)
                            x_cat_batch = None
                        else:
                            x_cat_batch, y_batch = batch
                            x_cat_batch = x_cat_batch.to(self._device)
                            x_num_batch = None

                        y_batch = y_batch.to(self._device)

                        logits = self.model_(x_num_batch, x_cat_batch)
                        loss = F.cross_entropy(logits, y_batch)
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

            scheduler.step()

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()

        X = np.asarray(X)
        x_num, x_cat = self._preprocess_features(X, fit=False)

        self.model_.eval()
        all_proba = []

        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)

                x_num_batch = x_num[start:end].to(self._device) if x_num is not None else None
                x_cat_batch = x_cat[start:end].to(self._device) if x_cat is not None else None

                logits = self.model_(x_num_batch, x_cat_batch)
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(class_indices)

    def _check_is_fitted(self):
        """Check if fitted."""
        if not self._is_fitted:
            raise RuntimeError("FTTransformerClassifier has not been fitted.")


class FTTransformerRegressor(BaseEstimator, RegressorMixin):
    """Feature Tokenizer Transformer for regression.

    Same architecture as FTTransformerClassifier but with regression head.

    Parameters
    ----------
    n_blocks : int, default=3
        Number of transformer blocks.
    d_token : int, default=192
        Embedding dimension.
    n_heads : int, default=8
        Number of attention heads.
    attention_dropout : float, default=0.2
        Attention dropout.
    ffn_dropout : float, default=0.1
        Feed-forward dropout.
    learning_rate : float, default=1e-4
        Learning rate.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum epochs.
    batch_size : int, default=256
        Batch size.
    early_stopping : int, default=15
        Early stopping patience.
    device : str, default='auto'
        Device.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Examples
    --------
    >>> reg = FTTransformerRegressor()
    >>> reg.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_blocks: int = 3,
        d_token: int = 192,
        n_heads: int = 8,
        attention_dropout: float = 0.2,
        ffn_dropout: float = 0.1,
        residual_dropout: float = 0.0,
        d_ffn_factor: float = 4 / 3,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 15,
        cat_cardinality_threshold: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_blocks = n_blocks
        self.d_token = d_token
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.residual_dropout = residual_dropout
        self.d_ffn_factor = d_ffn_factor
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.cat_cardinality_threshold = cat_cardinality_threshold
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_: _FTTransformerModule | None = None
        self._device: torch.device | None = None
        self._num_scaler: StandardScaler | None = None
        self._target_scaler: StandardScaler | None = None
        self._cat_encoders: dict[int, LabelEncoder] = {}
        self._num_feature_indices: list[int] = []
        self._cat_feature_indices: list[int] = []
        self._cat_cardinalities: list[int] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str):
        if self.verbose:
            print(f"[FT-Transformer] {message}")

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _identify_feature_types(self, X: np.ndarray):
        n_features = X.shape[1]
        self._num_feature_indices = []
        self._cat_feature_indices = []
        self._cat_cardinalities = []

        for i in range(n_features):
            n_unique = len(np.unique(X[:, i][~np.isnan(X[:, i])]))
            if n_unique <= self.cat_cardinality_threshold:
                self._cat_feature_indices.append(i)
                self._cat_cardinalities.append(n_unique)
            else:
                self._num_feature_indices.append(i)

    def _preprocess_features(self, X: np.ndarray, fit: bool = False):
        x_num = None
        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices].astype(np.float32)
            if fit:
                self._num_scaler = StandardScaler()
                X_num = self._num_scaler.fit_transform(X_num)
            else:
                X_num = self._num_scaler.transform(X_num)
            X_num = np.nan_to_num(X_num, nan=0.0)
            x_num = torch.tensor(X_num, dtype=torch.float32)

        x_cat = None
        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            X_cat_encoded = np.zeros(X_cat.shape, dtype=np.int64)

            for i, col_idx in enumerate(self._cat_feature_indices):
                if fit:
                    le = LabelEncoder()
                    col_data = X_cat[:, i].astype(str)
                    le.fit(col_data)
                    self._cat_encoders[col_idx] = le
                    X_cat_encoded[:, i] = le.transform(col_data)
                else:
                    le = self._cat_encoders[col_idx]
                    col_data = X_cat[:, i].astype(str)
                    known_mask = np.isin(col_data, le.classes_)
                    X_cat_encoded[known_mask, i] = le.transform(col_data[known_mask])
                    X_cat_encoded[~known_mask, i] = len(le.classes_)

            x_cat = torch.tensor(X_cat_encoded, dtype=torch.long)

        return x_num, x_cat

    def fit(self, X, y, eval_set=None, **fit_params) -> FTTransformerRegressor:
        """Fit the FT-Transformer regressor."""
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1).astype(np.float32)

        # Scale target
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y).ravel()

        self._identify_feature_types(X)
        x_num, x_cat = self._preprocess_features(X, fit=True)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # Create model
        self.model_ = _FTTransformerModule(
            n_num_features=len(self._num_feature_indices),
            cat_cardinalities=self._cat_cardinalities,
            n_classes=1,
            d_token=self.d_token,
            n_blocks=self.n_blocks,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            residual_dropout=self.residual_dropout,
            is_regression=True,
        ).to(self._device)

        # Create data loader
        if x_num is not None and x_cat is not None:
            train_dataset = TensorDataset(x_num, x_cat, y_tensor)
        elif x_num is not None:
            train_dataset = TensorDataset(x_num, y_tensor)
        else:
            train_dataset = TensorDataset(x_cat, y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation
        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val = np.asarray(X_val)
            y_val = self._target_scaler.transform(
                np.asarray(y_val).reshape(-1, 1)
            ).ravel()

            x_num_val, x_cat_val = self._preprocess_features(X_val, fit=False)
            y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

            if x_num_val is not None and x_cat_val is not None:
                val_dataset = TensorDataset(x_num_val, x_cat_val, y_val_tensor)
            elif x_num_val is not None:
                val_dataset = TensorDataset(x_num_val, y_val_tensor)
            else:
                val_dataset = TensorDataset(x_cat_val, y_val_tensor)

            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        for epoch in range(self.n_epochs):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                if len(batch) == 3:
                    x_num_batch, x_cat_batch, y_batch = batch
                    x_num_batch = x_num_batch.to(self._device)
                    x_cat_batch = x_cat_batch.to(self._device)
                elif x_num is not None:
                    x_num_batch, y_batch = batch
                    x_num_batch = x_num_batch.to(self._device)
                    x_cat_batch = None
                else:
                    x_cat_batch, y_batch = batch
                    x_cat_batch = x_cat_batch.to(self._device)
                    x_num_batch = None

                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self.model_(x_num_batch, x_cat_batch).squeeze()
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            self.history_["train_loss"].append(train_loss)

            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0
                n_val_batches = 0

                with torch.no_grad():
                    for batch in val_loader:
                        if len(batch) == 3:
                            x_num_batch, x_cat_batch, y_batch = batch
                            x_num_batch = x_num_batch.to(self._device)
                            x_cat_batch = x_cat_batch.to(self._device)
                        elif x_num is not None:
                            x_num_batch, y_batch = batch
                            x_num_batch = x_num_batch.to(self._device)
                            x_cat_batch = None
                        else:
                            x_cat_batch, y_batch = batch
                            x_cat_batch = x_cat_batch.to(self._device)
                            x_num_batch = None

                        y_batch = y_batch.to(self._device)
                        pred = self.model_(x_num_batch, x_cat_batch).squeeze()
                        loss = F.mse_loss(pred, y_batch)
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

                if patience_counter >= self.early_stopping:
                    break

            scheduler.step()

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("FTTransformerRegressor has not been fitted.")

        X = np.asarray(X)
        x_num, x_cat = self._preprocess_features(X, fit=False)

        self.model_.eval()
        all_pred = []
        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)

                x_num_batch = x_num[start:end].to(self._device) if x_num is not None else None
                x_cat_batch = x_cat[start:end].to(self._device) if x_cat is not None else None

                pred = self.model_(x_num_batch, x_cat_batch).squeeze()
                all_pred.append(pred.cpu().numpy())

        pred_scaled = np.concatenate(all_pred)
        return self._target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
