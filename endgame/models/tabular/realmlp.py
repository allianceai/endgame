"""RealMLP: Better by Default MLPs for Tabular Data.

Implements RealMLP from Holzmuller et al. (NeurIPS 2024), an improved MLP
with carefully meta-tuned defaults and a bag of tricks including robust
preprocessing, piecewise-linear numerical embeddings, diagonal weight
layers, and cosine-with-warmup learning rate scheduling.

References
----------
- Holzmuller et al. "Better by Default: Strong Pre-Tuned MLPs and Boosted
  Trees on Tabular Data" (NeurIPS 2024)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

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
    _nn_Module = object


def _check_torch():
    """Check if PyTorch is available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for RealMLP. "
            "Install with: pip install torch"
        )


# =============================================================================
# Preprocessing
# =============================================================================


class _SmoothClip:
    """Smooth clipping of outliers using tanh at +/- c standard deviations.

    After robust scaling (subtract median, divide by IQR), values beyond
    ``+/- c`` are smoothly compressed via ``c * tanh(x / c)``.  This avoids
    hard truncation artefacts while still bounding the feature range.

    Parameters
    ----------
    c : float, default=3.0
        Number of (IQR-scaled) standard deviations at which clipping begins.
    """

    def __init__(self, c: float = 3.0):
        self.c = c

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """Apply smooth clipping element-wise.

        Parameters
        ----------
        x : ndarray
            Already robustly-scaled features.

        Returns
        -------
        ndarray
            Clipped features.
        """
        if self.c <= 0:
            return x
        return self.c * np.tanh(x / self.c)


class _RobustPreprocessor:
    """Robust feature preprocessor: median/IQR scaling + smooth clipping.

    For each feature the preprocessor computes the median and
    inter-quartile range (IQR) during ``fit`` and then applies the
    transformation ``(x - median) / IQR`` followed by smooth clipping
    during ``transform``.  Features with zero IQR are kept centred but
    not scaled.

    Parameters
    ----------
    smooth_clip_c : float, default=3.0
        Clipping parameter forwarded to :class:`_SmoothClip`.
    """

    def __init__(self, smooth_clip_c: float = 3.0):
        self.smooth_clip_c = smooth_clip_c
        self._clip = _SmoothClip(c=smooth_clip_c)
        self.median_: np.ndarray | None = None
        self.iqr_: np.ndarray | None = None
        self._is_fitted = False

    def fit(self, X: np.ndarray) -> _RobustPreprocessor:
        """Fit median and IQR on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        self
        """
        self.median_ = np.nanmedian(X, axis=0)
        q75 = np.nanpercentile(X, 75, axis=0)
        q25 = np.nanpercentile(X, 25, axis=0)
        self.iqr_ = q75 - q25
        # Avoid division by zero: replace zero IQR with 1.0
        self.iqr_[self.iqr_ == 0] = 1.0
        self._is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features using fitted statistics.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Transformed features with the same shape.
        """
        assert self._is_fitted, "_RobustPreprocessor has not been fitted."
        X_scaled = (X - self.median_) / self.iqr_
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        return self._clip(X_scaled)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


# =============================================================================
# PyTorch Modules
# =============================================================================


class _DiagonalLayer(_nn_Module):
    """Element-wise learned scaling layer (one weight per feature dimension).

    After (optional) embeddings the feature vector is scaled element-wise
    by a learnable weight vector, allowing the network to emphasise or
    suppress individual features before the MLP blocks.

    Parameters
    ----------
    d_in : int
        Dimensionality of the input (total embedding width).
    """

    def __init__(self, d_in: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_in))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weight


class _PiecewiseLinearEmbedding(_nn_Module):
    """Piecewise-linear numerical embedding for each input feature.

    Each scalar feature is transformed into an ``n_bins``-dimensional
    representation using learned bin edges and slopes, producing a
    piecewise-linear mapping from R -> R^n_bins.  Bin boundaries are
    initialised to evenly-spaced quantiles of the training data (when
    available) or a uniform grid.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_bins : int, default=16
        Number of bins (output embedding dimension per feature).
    """

    def __init__(self, n_features: int, n_bins: int = 16):
        super().__init__()
        self.n_features = n_features
        self.n_bins = n_bins

        # Bin boundaries: shape (n_features, n_bins + 1)
        # Initialised to uniform grid [-1, 1]
        boundaries = torch.linspace(-1.0, 1.0, n_bins + 1).unsqueeze(0)
        boundaries = boundaries.expand(n_features, -1).clone()
        self.boundaries = nn.Parameter(boundaries)

        # Slopes for each bin: shape (n_features, n_bins)
        self.slopes = nn.Parameter(torch.ones(n_features, n_bins))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute piecewise-linear embeddings.

        Parameters
        ----------
        x : Tensor of shape (batch_size, n_features)

        Returns
        -------
        Tensor of shape (batch_size, n_features * n_bins)
        """
        batch_size = x.shape[0]

        # Sort boundaries to maintain monotonicity
        boundaries, _ = torch.sort(self.boundaries, dim=-1)

        # x: (batch, n_features) -> (batch, n_features, 1)
        x_expanded = x.unsqueeze(-1)

        # Left edges: (n_features, n_bins)
        left = boundaries[:, :-1]
        # Right edges: (n_features, n_bins)
        right = boundaries[:, 1:]

        # Width of each bin
        widths = right - left
        widths = widths.clamp(min=1e-8)

        # Position within each bin, clamped to [0, 1]
        t = (x_expanded - left) / widths
        t = t.clamp(0.0, 1.0)

        # Piecewise-linear output: slope * position
        out = self.slopes * t  # (batch, n_features, n_bins)

        # Flatten: (batch, n_features * n_bins)
        return out.reshape(batch_size, -1)


class _RealMLPBlock(_nn_Module):
    """Single MLP block: Linear -> LayerNorm -> ReLU -> Dropout.

    Parameters
    ----------
    d_in : int
        Input dimension.
    d_out : int
        Output dimension.
    dropout : float, default=0.15
        Dropout rate applied after ReLU.
    """

    def __init__(self, d_in: int, d_out: int, dropout: float = 0.15):
        super().__init__()
        self.linear = nn.Linear(d_in, d_out)
        self.norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.norm(x)
        x = F.relu(x)
        x = self.dropout(x)
        return x


class _RealMLPNetwork(_nn_Module):
    """Full RealMLP network.

    Architecture:
        [optional] PiecewiseLinearEmbedding -> DiagonalLayer
        -> MLP blocks (Linear -> LayerNorm -> ReLU -> Dropout) x n_blocks
        -> Linear output head

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int
        Number of output units (n_classes for classification, 1 for regression).
    n_blocks : int, default=3
        Number of MLP blocks.
    d_block : int, default=256
        Hidden dimension of each block.
    dropout : float, default=0.15
        Dropout rate inside blocks.
    use_embeddings : bool, default=True
        Whether to use piecewise-linear embeddings.
    n_bins : int, default=16
        Number of bins for embeddings.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.15,
        use_embeddings: bool = True,
        n_bins: int = 16,
    ):
        super().__init__()
        self.n_features = n_features
        self.use_embeddings = use_embeddings

        # Determine input dimension after optional embedding
        if use_embeddings:
            self.embedding = _PiecewiseLinearEmbedding(n_features, n_bins)
            d_in = n_features * n_bins
        else:
            self.embedding = None
            d_in = n_features

        # Diagonal layer
        self.diagonal = _DiagonalLayer(d_in)

        # MLP blocks
        blocks = []
        current_d = d_in
        for _ in range(n_blocks):
            blocks.append(_RealMLPBlock(current_d, d_block, dropout))
            current_d = d_block
        self.blocks = nn.ModuleList(blocks)

        # Output head
        self.head = nn.Linear(current_d, n_outputs)

        # Initialise weights
        self._init_weights()

    def _init_weights(self):
        """Improved initialisation: Kaiming for Linear, ones for LayerNorm."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.embedding is not None:
            x = self.embedding(x)
        x = self.diagonal(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


# =============================================================================
# Sklearn-Compatible Estimators
# =============================================================================


class RealMLPClassifier(ClassifierMixin, BaseEstimator):
    """RealMLP classifier with meta-tuned defaults for tabular data.

    Implements the RealMLP architecture from Holzmuller et al. (NeurIPS 2024)
    with robust preprocessing, piecewise-linear numerical embeddings, a
    diagonal scaling layer, and a cosine-with-warmup learning rate schedule.

    Parameters
    ----------
    n_blocks : int, default=3
        Number of MLP blocks (Linear -> LayerNorm -> ReLU -> Dropout).
    d_block : int, default=256
        Hidden dimension of each block.
    dropout : float, default=0.15
        Dropout rate inside blocks.
    learning_rate : float, default=0.04
        Peak learning rate (after warmup).
    weight_decay : float, default=0.0
        L2 regularisation coefficient.
    n_epochs : int, default=256
        Maximum number of training epochs.
    batch_size : int, default=256
        Training mini-batch size.
    smooth_clip_c : float, default=3.0
        Smooth-clipping parameter for outlier compression.
    use_embeddings : bool, default=True
        Whether to use piecewise-linear embeddings.
    n_bins : int, default=16
        Number of bins in piecewise-linear embeddings.
    warmup_fraction : float, default=0.1
        Fraction of total steps spent in linear warmup.
    early_stopping : int, default=20
        Early stopping patience (epochs without improvement).
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, print training progress.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : _RealMLPNetwork
        Fitted PyTorch model.
    history_ : dict
        Training history with ``'train_loss'`` and ``'val_loss'`` keys.
    feature_importances_ : ndarray of shape (n_features,)
        Importance of each input feature derived from the diagonal layer.

    Examples
    --------
    >>> from endgame.models.tabular import RealMLPClassifier
    >>> clf = RealMLPClassifier(n_blocks=3, d_block=256)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.15,
        learning_rate: float = 0.04,
        weight_decay: float = 0.0,
        n_epochs: int = 256,
        batch_size: int = 256,
        smooth_clip_c: float = 3.0,
        use_embeddings: bool = True,
        n_bins: int = 16,
        warmup_fraction: float = 0.1,
        early_stopping: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.smooth_clip_c = smooth_clip_c
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.warmup_fraction = warmup_fraction
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[RealMLP] {message}")

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _warmup_cosine_schedule(
        self, optimizer, warmup_steps: int, total_steps: int
    ):
        """Create a cosine-with-warmup LR scheduler."""

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> RealMLPClassifier:
        """Fit the RealMLP classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. If not provided, 10%
            of the training data is held out automatically.

        Returns
        -------
        self
            Fitted classifier.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Robust preprocessing
        self._preprocessor = _RobustPreprocessor(
            smooth_clip_c=self.smooth_clip_c
        )
        X_processed = self._preprocessor.fit_transform(X).astype(np.float32)
        self.n_features_in_ = X.shape[1]

        # Split for early stopping if no eval_set provided
        if eval_set is None:
            from sklearn.model_selection import train_test_split

            n_val = max(1, int(0.1 * len(X_processed)))
            if n_val < len(X_processed):
                try:
                    (
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                    ) = train_test_split(
                        X_processed,
                        y_encoded,
                        test_size=n_val,
                        random_state=self.random_state,
                        stratify=y_encoded,
                    )
                except ValueError:
                    (
                        X_train,
                        X_val,
                        y_train,
                        y_val,
                    ) = train_test_split(
                        X_processed,
                        y_encoded,
                        test_size=n_val,
                        random_state=self.random_state,
                    )
            else:
                X_train, y_train = X_processed, y_encoded
                X_val, y_val = X_processed, y_encoded
        else:
            X_train, y_train = X_processed, y_encoded
            X_val_raw, y_val_raw = eval_set
            X_val = self._preprocessor.transform(
                np.asarray(X_val_raw, dtype=np.float32)
            ).astype(np.float32)
            y_val = self._label_encoder.transform(np.asarray(y_val_raw))

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.long)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.long)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Build network
        self.model_ = _RealMLPNetwork(
            n_features=X.shape[1],
            n_outputs=self.n_classes_,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            use_embeddings=self.use_embeddings,
            n_bins=self.n_bins,
        ).to(self._device)

        # Optimiser
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine-with-warmup scheduler
        total_steps = self.n_epochs * len(train_loader)
        warmup_steps = int(self.warmup_fraction * total_steps)
        scheduler = self._warmup_cosine_schedule(
            optimizer, warmup_steps, total_steps
        )

        # Training loop
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(
            f"Training on {self._device} | "
            f"{sum(p.numel() for p in self.model_.parameters())} params"
        )

        for epoch in range(self.n_epochs):
            # --- Train ---
            self.model_.train()
            train_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = F.cross_entropy(logits, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # --- Validate ---
            self.model_.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    logits = self.model_(X_batch)
                    loss = F.cross_entropy(logits, y_batch)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
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

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities summing to 1 for each sample.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_processed = self._preprocessor.transform(X).astype(np.float32)

        self.model_.eval()
        all_proba = []
        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = torch.tensor(
                    X_processed[start:end], dtype=torch.float32
                ).to(self._device)
                logits = self.model_(X_batch)
                proba = F.softmax(logits, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray
            Predicted class labels (in the original label space).
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(class_indices)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the diagonal layer weights.

        Returns a non-negative array of shape ``(n_features_in_,)`` that
        sums to 1.  When embeddings are used the per-bin diagonal weights
        are aggregated per feature.
        """
        self._check_is_fitted()
        diag_weights = (
            self.model_.diagonal.weight.detach().cpu().numpy()
        )
        diag_abs = np.abs(diag_weights)

        if self.use_embeddings:
            # Aggregate per feature (n_features * n_bins -> n_features)
            n_features = self.n_features_in_
            diag_abs = diag_abs.reshape(n_features, -1).sum(axis=1)

        total = diag_abs.sum()
        if total > 0:
            return diag_abs / total
        return diag_abs

    def _check_is_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("RealMLPClassifier has not been fitted.")


class RealMLPRegressor(RegressorMixin, BaseEstimator):
    """RealMLP regressor with meta-tuned defaults for tabular data.

    Same architecture as :class:`RealMLPClassifier` but with a single
    linear output and MSE loss.

    Parameters
    ----------
    n_blocks : int, default=3
        Number of MLP blocks.
    d_block : int, default=256
        Hidden dimension of each block.
    dropout : float, default=0.15
        Dropout rate inside blocks.
    learning_rate : float, default=0.04
        Peak learning rate.
    weight_decay : float, default=0.0
        L2 regularisation coefficient.
    n_epochs : int, default=256
        Maximum training epochs.
    batch_size : int, default=256
        Training batch size.
    smooth_clip_c : float, default=3.0
        Smooth-clipping parameter.
    use_embeddings : bool, default=True
        Whether to use piecewise-linear embeddings.
    n_bins : int, default=16
        Bins in piecewise-linear embeddings.
    warmup_fraction : float, default=0.1
        Fraction of steps for linear warmup.
    early_stopping : int, default=20
        Early stopping patience.
    device : str, default='auto'
        Computation device.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable training output.

    Attributes
    ----------
    model_ : _RealMLPNetwork
        Fitted PyTorch model.
    history_ : dict
        Training history.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances from the diagonal layer.

    Examples
    --------
    >>> from endgame.models.tabular import RealMLPRegressor
    >>> reg = RealMLPRegressor(n_blocks=3, d_block=256)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_blocks: int = 3,
        d_block: int = 256,
        dropout: float = 0.15,
        learning_rate: float = 0.04,
        weight_decay: float = 0.0,
        n_epochs: int = 256,
        batch_size: int = 256,
        smooth_clip_c: float = 3.0,
        use_embeddings: bool = True,
        n_bins: int = 16,
        warmup_fraction: float = 0.1,
        early_stopping: int = 20,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_blocks = n_blocks
        self.d_block = d_block
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.smooth_clip_c = smooth_clip_c
        self.use_embeddings = use_embeddings
        self.n_bins = n_bins
        self.warmup_fraction = warmup_fraction
        self.early_stopping = early_stopping
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str):
        if self.verbose:
            print(f"[RealMLP] {message}")

    def _get_device(self) -> torch.device:
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _warmup_cosine_schedule(
        self, optimizer, warmup_steps: int, total_steps: int
    ):
        """Create a cosine-with-warmup LR scheduler."""

        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(
                max(1, total_steps - warmup_steps)
            )
            return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    def fit(
        self,
        X,
        y,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> RealMLPRegressor:
        """Fit the RealMLP regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping. If not provided, 10%
            of the training data is held out automatically.

        Returns
        -------
        self
            Fitted regressor.
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).ravel()

        # Target scaling (standard scaling)
        self._target_mean = y.mean()
        self._target_std = y.std()
        if self._target_std == 0:
            self._target_std = 1.0
        y_scaled = (y - self._target_mean) / self._target_std

        # Robust preprocessing
        self._preprocessor = _RobustPreprocessor(
            smooth_clip_c=self.smooth_clip_c
        )
        X_processed = self._preprocessor.fit_transform(X).astype(np.float32)
        self.n_features_in_ = X.shape[1]

        # Split for early stopping if no eval_set provided
        if eval_set is None:
            from sklearn.model_selection import train_test_split

            n_val = max(1, int(0.1 * len(X_processed)))
            if n_val < len(X_processed):
                (
                    X_train,
                    X_val,
                    y_train,
                    y_val,
                ) = train_test_split(
                    X_processed,
                    y_scaled,
                    test_size=n_val,
                    random_state=self.random_state,
                )
            else:
                X_train, y_train = X_processed, y_scaled
                X_val, y_val = X_processed, y_scaled
        else:
            X_train, y_train = X_processed, y_scaled
            X_val_raw, y_val_raw = eval_set
            X_val = self._preprocessor.transform(
                np.asarray(X_val_raw, dtype=np.float32)
            ).astype(np.float32)
            y_val_np = np.asarray(y_val_raw, dtype=np.float32).ravel()
            y_val = (y_val_np - self._target_mean) / self._target_std

        # Tensors
        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32)
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = DataLoader(
            TensorDataset(X_val_t, y_val_t),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        # Build network
        self.model_ = _RealMLPNetwork(
            n_features=X.shape[1],
            n_outputs=1,
            n_blocks=self.n_blocks,
            d_block=self.d_block,
            dropout=self.dropout,
            use_embeddings=self.use_embeddings,
            n_bins=self.n_bins,
        ).to(self._device)

        # Optimiser
        optimizer = optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine-with-warmup scheduler
        total_steps = self.n_epochs * len(train_loader)
        warmup_steps = int(self.warmup_fraction * total_steps)
        scheduler = self._warmup_cosine_schedule(
            optimizer, warmup_steps, total_steps
        )

        # Training loop
        self.history_: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
        }
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(
            f"Training on {self._device} | "
            f"{sum(p.numel() for p in self.model_.parameters())} params"
        )

        for epoch in range(self.n_epochs):
            # --- Train ---
            self.model_.train()
            train_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                pred = self.model_(X_batch).squeeze(-1)
                loss = F.mse_loss(pred, y_batch)
                loss.backward()
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # --- Validate ---
            self.model_.eval()
            val_loss = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(self._device)
                    y_batch = y_batch.to(self._device)
                    pred = self.model_(X_batch).squeeze(-1)
                    loss = F.mse_loss(pred, y_batch)
                    val_loss += loss.item()
                    n_val_batches += 1

            val_loss /= max(n_val_batches, 1)
            self.history_["val_loss"].append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
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

        # Restore best model
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted targets in the original scale.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=np.float32)
        X_processed = self._preprocessor.transform(X).astype(np.float32)

        self.model_.eval()
        all_pred = []
        n_samples = X.shape[0]

        with torch.no_grad():
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                X_batch = torch.tensor(
                    X_processed[start:end], dtype=torch.float32
                ).to(self._device)
                pred = self.model_(X_batch).squeeze(-1)
                all_pred.append(pred.cpu().numpy())

        pred_scaled = np.concatenate(all_pred)
        # Inverse target scaling
        return pred_scaled * self._target_std + self._target_mean

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the diagonal layer weights."""
        self._check_is_fitted()
        diag_weights = (
            self.model_.diagonal.weight.detach().cpu().numpy()
        )
        diag_abs = np.abs(diag_weights)

        if self.use_embeddings:
            n_features = self.n_features_in_
            diag_abs = diag_abs.reshape(n_features, -1).sum(axis=1)

        total = diag_abs.sum()
        if total > 0:
            return diag_abs / total
        return diag_abs

    def _check_is_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("RealMLPRegressor has not been fitted.")
