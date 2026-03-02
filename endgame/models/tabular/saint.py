from __future__ import annotations

"""SAINT: Self-Attention and Intersample Attention Transformer.

SAINT combines column-wise self-attention with row-wise (intersample) attention
to capture both feature interactions and sample similarities.

References
----------
- Somepalli et al. "SAINT: Improved Neural Networks for Tabular Data
  via Row Attention and Contrastive Pre-Training" (2021)
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
            "PyTorch is required for SAINT. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _SAINTEmbedding(nn.Module):
    """Embedding layer for SAINT that handles both numerical and categorical.

    Uses efficient batched operations instead of per-feature loops.
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        d_model: int,
    ):
        super().__init__()

        self.n_num_features = n_num_features
        self.n_cat_features = len(cat_cardinalities)
        self.d_model = d_model

        # Efficient numerical embeddings using a single weight matrix
        # Instead of n separate Linear(1, d_model), use one Linear(n, n*d_model)
        # then reshape. This is much faster.
        if n_num_features > 0:
            # Each numerical feature gets its own embedding weights
            # Shape: (n_num_features, d_model)
            self.num_weight = nn.Parameter(torch.empty(n_num_features, d_model))
            self.num_bias = nn.Parameter(torch.zeros(n_num_features, d_model))
            nn.init.kaiming_uniform_(self.num_weight, a=np.sqrt(5))

        # Categorical embeddings (still need per-feature due to varying cardinalities)
        if self.n_cat_features > 0:
            self.cat_embeddings = nn.ModuleList([
                nn.Embedding(card + 1, d_model)  # +1 for unknown
                for card in cat_cardinalities
            ])

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = []

        # Vectorized numerical embedding
        if x_num is not None and self.n_num_features > 0:
            # x_num: (batch, n_num_features)
            # Expand to (batch, n_num_features, 1) for broadcasting
            x_expanded = x_num.unsqueeze(-1)  # (batch, n_num_features, 1)
            # Multiply by weights and add bias
            # Result: (batch, n_num_features, d_model)
            num_embedded = x_expanded * self.num_weight + self.num_bias
            embeddings.append(num_embedded)

        # Categorical embeddings (batch all lookups together)
        if x_cat is not None and self.n_cat_features > 0:
            cat_embedded = []
            for i, emb in enumerate(self.cat_embeddings):
                cat_embedded.append(emb(x_cat[:, i]))  # (batch, d_model)
            # Stack to (batch, n_cat_features, d_model)
            cat_embedded = torch.stack(cat_embedded, dim=1)
            embeddings.append(cat_embedded)

        # Concatenate along feature dimension
        if len(embeddings) == 2:
            return torch.cat(embeddings, dim=1)  # (batch, n_features, d_model)
        else:
            return embeddings[0]


class _SAINTBlock(nn.Module):
    """SAINT block with self-attention and intersample attention."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ffn: int,
        attention_dropout: float,
        ffn_dropout: float,
        use_intersample: bool = True,
    ):
        super().__init__()

        self.use_intersample = use_intersample

        # Self-attention (column-wise)
        self.self_attention = nn.MultiheadAttention(
            d_model, n_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)

        # Intersample attention (row-wise) - attention across samples
        if use_intersample:
            self.intersample_attention = nn.MultiheadAttention(
                d_model, n_heads,
                dropout=attention_dropout,
                batch_first=True,
            )
            self.norm2 = nn.LayerNorm(d_model)

        # Feed-forward
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(d_ffn, d_model),
            nn.Dropout(ffn_dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, n_features, d_model)
        batch_size, n_features, d_model = x.shape

        # Self-attention (across features for each sample)
        attn_out, _ = self.self_attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Intersample attention (across samples for each feature)
        if self.use_intersample:
            # Reshape: (batch, n_features, d_model) -> (n_features, batch, d_model)
            x_t = x.transpose(0, 1)

            # Attention across batch dimension
            inter_out, _ = self.intersample_attention(x_t, x_t, x_t)

            # Reshape back
            inter_out = inter_out.transpose(0, 1)
            x = self.norm2(x + inter_out)

        # Feed-forward
        x = self.norm3(x + self.ffn(x))

        return x


class _SAINTModule(nn.Module):
    """PyTorch SAINT module."""

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: list[int],
        n_classes: int,
        d_model: int = 32,
        n_layers: int = 6,
        n_heads: int = 8,
        d_ffn_factor: float = 4.0,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        use_intersample: bool = True,
        is_regression: bool = False,
    ):
        super().__init__()

        self.is_regression = is_regression

        # Embedding
        self.embedding = _SAINTEmbedding(
            n_num_features, cat_cardinalities, d_model
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls_token, std=0.02)

        # SAINT blocks
        d_ffn = int(d_model * d_ffn_factor)
        self.blocks = nn.ModuleList([
            _SAINTBlock(
                d_model, n_heads, d_ffn,
                attention_dropout, ffn_dropout,
                use_intersample=use_intersample,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)

        # Output
        output_dim = 1 if is_regression else n_classes
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_dim),
        )

    def forward(
        self,
        x_num: torch.Tensor | None = None,
        x_cat: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = x_num.shape[0] if x_num is not None else x_cat.shape[0]

        # Embed features
        x = self.embedding(x_num, x_cat)  # (batch, n_features, d_model)

        # Add CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls, x], dim=1)

        # SAINT blocks
        for block in self.blocks:
            x = block(x)

        # Take CLS output
        x = self.norm(x[:, 0])

        return self.head(x)


class SAINTClassifier(ClassifierMixin, BaseEstimator):
    """SAINT: Self-Attention and Intersample Attention Transformer.

    Combines column-wise self-attention with row-wise (intersample) attention
    to capture both feature interactions and sample similarities.

    Parameters
    ----------
    n_layers : int, default=3
        Number of SAINT layers. 2-4 works well for most datasets.
    d_model : int, default=32
        Model dimension.
    n_heads : int, default=4
        Number of attention heads.
    attention_dropout : float, default=0.1
        Attention dropout.
    ffn_dropout : float, default=0.1
        Feed-forward dropout.
    d_ffn_factor : float, default=4.0
        FFN hidden dimension factor.
    use_intersample : bool, default=True
        Whether to use intersample attention (unique to SAINT).
    learning_rate : float, default=1e-3
        Learning rate. Higher rates (1e-3) often work better than 1e-4.
    weight_decay : float, default=1e-5
        L2 regularization.
    n_epochs : int, default=100
        Maximum epochs.
    batch_size : int, default=256
        Batch size.
    early_stopping : int, default=15
        Early stopping patience.
    validation_fraction : float, default=0.1
        Fraction of training data to use for validation when eval_set not provided.
    cat_cardinality_threshold : int, default=20
        Threshold for categorical detection.
    device : str, default='auto'
        Device.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.
    model_ : _SAINTModule
        Fitted model.
    history_ : dict
        Training history.

    Examples
    --------
    >>> clf = SAINTClassifier(n_layers=3, d_model=32)
    >>> clf.fit(X_train, y_train, eval_set=(X_val, y_val))
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    SAINT's intersample attention allows it to consider relationships
    between different samples, which can be powerful for learning patterns
    that span across the dataset.

    For best performance:
    - Use an eval_set for early stopping (or validation_fraction > 0)
    - Start with n_layers=3 and increase if underfitting
    - Higher learning rates (1e-3) often work better than typical transformer LR
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_layers: int = 3,
        d_model: int = 32,
        n_heads: int = 4,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        d_ffn_factor: float = 4.0,
        use_intersample: bool = True,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping: int = 15,
        validation_fraction: float = 0.1,
        cat_cardinality_threshold: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.d_ffn_factor = d_ffn_factor
        self.use_intersample = use_intersample
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.cat_cardinality_threshold = cat_cardinality_threshold
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: _SAINTModule | None = None
        self._device = None
        self._label_encoder: LabelEncoder | None = None
        self._num_scaler: StandardScaler | None = None
        self._cat_encoders: dict[int, LabelEncoder] = {}
        self._num_feature_indices: list[int] = []
        self._cat_feature_indices: list[int] = []
        self._cat_cardinalities: list[int] = []
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, msg: str):
        if self.verbose:
            print(f"[SAINT] {msg}")

    def _get_device(self):
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def _identify_feature_types(self, X: np.ndarray):
        self._num_feature_indices = []
        self._cat_feature_indices = []
        self._cat_cardinalities = []

        for i in range(X.shape[1]):
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

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> SAINTClassifier:
        """Fit the SAINT classifier."""
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X)
        y = np.asarray(y)

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        self._identify_feature_types(X)
        x_num, x_cat = self._preprocess_features(X, fit=True)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create model
        self.model_ = _SAINTModule(
            n_num_features=len(self._num_feature_indices),
            cat_cardinalities=self._cat_cardinalities,
            n_classes=self.n_classes_,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            use_intersample=self.use_intersample,
            is_regression=False,
        ).to(self._device)

        # Create internal validation split if no eval_set provided
        if eval_set is None and self.validation_fraction > 0:
            from sklearn.model_selection import train_test_split
            n_samples = X.shape[0]
            n_val = int(n_samples * self.validation_fraction)
            if n_val >= 1:
                indices = np.arange(n_samples)
                try:
                    train_idx, val_idx = train_test_split(
                        indices,
                        test_size=self.validation_fraction,
                        stratify=y_encoded,
                        random_state=self.random_state,
                    )
                except ValueError:
                    train_idx, val_idx = train_test_split(
                        indices,
                        test_size=self.validation_fraction,
                        random_state=self.random_state,
                    )

                # Split tensors
                x_num_train = x_num[train_idx] if x_num is not None else None
                x_num_val = x_num[val_idx] if x_num is not None else None
                x_cat_train = x_cat[train_idx] if x_cat is not None else None
                x_cat_val = x_cat[val_idx] if x_cat is not None else None
                y_train = y_tensor[train_idx]
                y_val = y_tensor[val_idx]

                # Update tensors for training
                x_num = x_num_train
                x_cat = x_cat_train
                y_tensor = y_train

                # Set up internal eval_set
                eval_set = (x_num_val, x_cat_val, y_val, True)  # True flag indicates internal split

        # Data loader
        if x_num is not None and x_cat is not None:
            train_dataset = TensorDataset(x_num, x_cat, y_tensor)
        elif x_num is not None:
            train_dataset = TensorDataset(x_num, y_tensor)
        else:
            train_dataset = TensorDataset(x_cat, y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation
        val_loader = None
        if eval_set is not None and len(eval_set) == 4 and eval_set[3] is True:
            # Internal split - tensors already prepared
            x_num_val, x_cat_val, y_val_tensor = eval_set[0], eval_set[1], eval_set[2]

            if x_num_val is not None and x_cat_val is not None:
                val_dataset = TensorDataset(x_num_val, x_cat_val, y_val_tensor)
            elif x_num_val is not None:
                val_dataset = TensorDataset(x_num_val, y_val_tensor)
            else:
                val_dataset = TensorDataset(x_cat_val, y_val_tensor)

            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        elif eval_set is not None:
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

        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.n_epochs)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device}...")

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
                logits = self.model_(x_num_batch, x_cat_batch)
                loss = F.cross_entropy(logits, y_batch)
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
                    self._log(f"Epoch {epoch+1}/{self.n_epochs}: train={train_loss:.4f}, val={val_loss:.4f}")

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                    break

            scheduler.step()

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("SAINTClassifier has not been fitted.")

        X = np.asarray(X)
        x_num, x_cat = self._preprocess_features(X, fit=False)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                end = min(start + self.batch_size, X.shape[0])

                x_num_batch = x_num[start:end].to(self._device) if x_num is not None else None
                x_cat_batch = x_cat[start:end].to(self._device) if x_cat is not None else None

                logits = self.model_(x_num_batch, x_cat_batch)
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))


class SAINTRegressor(BaseEstimator, RegressorMixin):
    """SAINT for regression.

    Same architecture as SAINTClassifier but with MSE loss.

    Parameters are the same as SAINTClassifier except no n_classes.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_layers: int = 6,
        d_model: int = 32,
        n_heads: int = 8,
        attention_dropout: float = 0.1,
        ffn_dropout: float = 0.1,
        d_ffn_factor: float = 4.0,
        use_intersample: bool = True,
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
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.attention_dropout = attention_dropout
        self.ffn_dropout = ffn_dropout
        self.d_ffn_factor = d_ffn_factor
        self.use_intersample = use_intersample
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.cat_cardinality_threshold = cat_cardinality_threshold
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self._device = None
        self._num_scaler = None
        self._target_scaler = None
        self._cat_encoders = {}
        self._num_feature_indices = []
        self._cat_feature_indices = []
        self._cat_cardinalities = []
        self.history_ = {"train_loss": [], "val_loss": []}
        self._is_fitted = False

    def fit(self, X, y, eval_set=None, **fit_params) -> SAINTRegressor:
        """Fit SAINT regressor."""
        _check_torch()

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        self._device = torch.device(
            "cuda" if self.device == "auto" and torch.cuda.is_available()
            else "cpu" if self.device == "auto" else self.device
        )

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1).astype(np.float32)

        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(y).ravel()

        # Feature processing (same as classifier)
        self._num_feature_indices = []
        self._cat_feature_indices = []
        self._cat_cardinalities = []

        for i in range(X.shape[1]):
            n_unique = len(np.unique(X[:, i][~np.isnan(X[:, i])]))
            if n_unique <= self.cat_cardinality_threshold:
                self._cat_feature_indices.append(i)
                self._cat_cardinalities.append(n_unique)
            else:
                self._num_feature_indices.append(i)

        # Preprocess
        x_num = None
        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices].astype(np.float32)
            self._num_scaler = StandardScaler()
            X_num = np.nan_to_num(self._num_scaler.fit_transform(X_num), nan=0.0)
            x_num = torch.tensor(X_num, dtype=torch.float32)

        x_cat = None
        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            X_cat_encoded = np.zeros(X_cat.shape, dtype=np.int64)
            for i, col_idx in enumerate(self._cat_feature_indices):
                le = LabelEncoder()
                X_cat_encoded[:, i] = le.fit_transform(X_cat[:, i].astype(str))
                self._cat_encoders[col_idx] = le
            x_cat = torch.tensor(X_cat_encoded, dtype=torch.long)

        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

        # Model
        self.model_ = _SAINTModule(
            n_num_features=len(self._num_feature_indices),
            cat_cardinalities=self._cat_cardinalities,
            n_classes=1,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            d_ffn_factor=self.d_ffn_factor,
            attention_dropout=self.attention_dropout,
            ffn_dropout=self.ffn_dropout,
            use_intersample=self.use_intersample,
            is_regression=True,
        ).to(self._device)

        # Training (simplified)
        if x_num is not None and x_cat is not None:
            train_dataset = TensorDataset(x_num, x_cat, y_tensor)
        elif x_num is not None:
            train_dataset = TensorDataset(x_num, y_tensor)
        else:
            train_dataset = TensorDataset(x_cat, y_tensor)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        optimizer = optim.AdamW(self.model_.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        for epoch in range(self.n_epochs):
            self.model_.train()
            for batch in train_loader:
                if len(batch) == 3:
                    xn, xc, yb = batch
                    xn, xc = xn.to(self._device), xc.to(self._device)
                elif x_num is not None:
                    xn, yb = batch
                    xn, xc = xn.to(self._device), None
                else:
                    xc, yb = batch
                    xn, xc = None, xc.to(self._device)

                yb = yb.to(self._device)
                optimizer.zero_grad()
                pred = self.model_(xn, xc).squeeze()
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("SAINTRegressor has not been fitted.")

        X = np.asarray(X)

        # Preprocess
        x_num = None
        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices].astype(np.float32)
            X_num = np.nan_to_num(self._num_scaler.transform(X_num), nan=0.0)
            x_num = torch.tensor(X_num, dtype=torch.float32)

        x_cat = None
        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            X_cat_encoded = np.zeros(X_cat.shape, dtype=np.int64)
            for i, col_idx in enumerate(self._cat_feature_indices):
                le = self._cat_encoders[col_idx]
                col_data = X_cat[:, i].astype(str)
                known = np.isin(col_data, le.classes_)
                X_cat_encoded[known, i] = le.transform(col_data[known])
                X_cat_encoded[~known, i] = len(le.classes_)
            x_cat = torch.tensor(X_cat_encoded, dtype=torch.long)

        self.model_.eval()
        all_pred = []

        with torch.no_grad():
            for start in range(0, X.shape[0], self.batch_size):
                end = min(start + self.batch_size, X.shape[0])
                xn = x_num[start:end].to(self._device) if x_num is not None else None
                xc = x_cat[start:end].to(self._device) if x_cat is not None else None
                pred = self.model_(xn, xc).squeeze()
                all_pred.append(pred.cpu().numpy())

        pred = np.concatenate(all_pred)
        return self._target_scaler.inverse_transform(pred.reshape(-1, 1)).ravel()
