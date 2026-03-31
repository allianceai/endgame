"""Fuzzy membership-based attention mechanism.

Replaces standard softmax attention with fuzzy membership-based
weighting using Gaussian membership functions and t-norm aggregation.

Requires PyTorch.

References
----------
Zheng, Y., et al. (2022). Fuzzy attention neural network to tackle
discontinuity in airway segmentation. NeurIPS.

Example
-------
>>> from endgame.fuzzy.modern.fuzzy_attention import FuzzyAttentionClassifier
>>> clf = FuzzyAttentionClassifier(n_heads=4, d_model=32, n_epochs=20)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for FuzzyAttentionLayer. "
            "Install it with: pip install torch"
        )


if HAS_TORCH:

    class FuzzyAttentionLayer(nn.Module):
        """Fuzzy membership-based attention layer.

        Replaces standard softmax attention with Gaussian fuzzy similarity
        between queries and keys, applying t-norm for multi-head aggregation.

        Parameters
        ----------
        d_model : int, default=64
            Model dimension.
        n_heads : int, default=4
            Number of attention heads.
        sigma_init : float, default=1.0
            Initial sigma for Gaussian membership functions.

        Examples
        --------
        >>> layer = FuzzyAttentionLayer(d_model=32, n_heads=4)
        >>> Q = torch.randn(8, 10, 32)  # (batch, seq, d_model)
        >>> K = torch.randn(8, 10, 32)
        >>> V = torch.randn(8, 10, 32)
        >>> out = layer(Q, K, V)  # (batch, seq, d_model)
        """

        def __init__(
            self,
            d_model: int = 64,
            n_heads: int = 4,
            sigma_init: float = 1.0,
        ):
            super().__init__()
            self.d_model = d_model
            self.n_heads = n_heads
            self.d_head = d_model // n_heads

            assert d_model % n_heads == 0, (
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )

            self.W_q = nn.Linear(d_model, d_model)
            self.W_k = nn.Linear(d_model, d_model)
            self.W_v = nn.Linear(d_model, d_model)
            self.W_out = nn.Linear(d_model, d_model)

            # Learnable sigma for Gaussian fuzzy similarity per head
            self.sigma = nn.Parameter(
                torch.full((n_heads, 1, 1), sigma_init)
            )

        def forward(
            self,
            Q: torch.Tensor,
            K: torch.Tensor,
            V: torch.Tensor,
        ) -> torch.Tensor:
            """Compute fuzzy attention.

            Parameters
            ----------
            Q : Tensor of shape (batch, seq_q, d_model)
                Query tensor.
            K : Tensor of shape (batch, seq_k, d_model)
                Key tensor.
            V : Tensor of shape (batch, seq_k, d_model)
                Value tensor.

            Returns
            -------
            Tensor of shape (batch, seq_q, d_model)
                Attention output.
            """
            batch_size = Q.size(0)

            # Project and reshape for multi-head
            q = self.W_q(Q).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            k = self.W_k(K).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            v = self.W_v(V).view(batch_size, -1, self.n_heads, self.d_head).transpose(1, 2)
            # q, k, v: (batch, n_heads, seq, d_head)

            # Compute fuzzy similarity using Gaussian kernel
            # distance^2 between Q and K
            diff = q.unsqueeze(-2) - k.unsqueeze(-3)  # (batch, heads, seq_q, seq_k, d_head)
            dist_sq = (diff ** 2).sum(dim=-1)  # (batch, heads, seq_q, seq_k)

            # Gaussian membership: exp(-dist^2 / (2 * sigma^2))
            sigma_sq = self.sigma ** 2 + 1e-8
            fuzzy_sim = torch.exp(-dist_sq / (2.0 * sigma_sq))

            # T-norm aggregation across heads (product t-norm)
            # For attention weights per head, normalize
            attn_sum = fuzzy_sim.sum(dim=-1, keepdim=True)
            attn_sum = torch.where(attn_sum == 0, torch.ones_like(attn_sum), attn_sum)
            attn_weights = fuzzy_sim / attn_sum

            # Weighted values
            out = torch.matmul(attn_weights, v)  # (batch, heads, seq_q, d_head)

            # Reshape and project
            out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
            return self.W_out(out)

else:

    class FuzzyAttentionLayer:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not available."""

        def __init__(self, *args: Any, **kwargs: Any):
            _check_torch()


class FuzzyAttentionClassifier(BaseEstimator, ClassifierMixin):
    """Sklearn-compatible classifier using fuzzy attention.

    Wraps FuzzyAttentionLayer in a classification model that treats
    each sample as a single-token sequence and applies fuzzy attention
    followed by a linear classification head.

    Parameters
    ----------
    d_model : int, default=64
        Model dimension.
    n_heads : int, default=4
        Number of fuzzy attention heads.
    n_epochs : int, default=100
        Number of training epochs.
    lr : float, default=0.001
        Learning rate.
    batch_size : int, default=32
        Mini-batch size.
    sigma_init : float, default=1.0
        Initial sigma for fuzzy similarity.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.fuzzy_attention import FuzzyAttentionClassifier
    >>> X = np.random.randn(100, 10)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = FuzzyAttentionClassifier(d_model=16, n_heads=2, n_epochs=10)
    >>> clf.fit(X, y)
    FuzzyAttentionClassifier(d_model=16, n_epochs=10, n_heads=2)
    >>> clf.predict(X[:3])
    array([...])
    """

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        n_epochs: int = 100,
        lr: float = 0.001,
        batch_size: int = 32,
        sigma_init: float = 1.0,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.sigma_init = sigma_init
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FuzzyAttentionClassifier:
        """Fit the fuzzy attention classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        _check_torch()
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        # Build model
        self.model_ = _FuzzyAttentionNet(
            n_features=X.shape[1],
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_classes=n_classes,
            sigma_init=self.sigma_init,
        )

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y_enc, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                avg = epoch_loss / max(n_batches, 1)
                print(
                    f"[FuzzyAttn] Epoch {epoch+1}/{self.n_epochs}, "
                    f"Loss: {avg:.4f}"
                )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        _check_torch()
        check_is_fitted(self, ["model_"])
        X = check_array(X)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.model_(X_t)
            proba = torch.softmax(logits, dim=-1).numpy()
        return proba


if HAS_TORCH:

    class _FuzzyAttentionNet(nn.Module):
        """Internal network using FuzzyAttentionLayer for classification."""

        def __init__(
            self,
            n_features: int,
            d_model: int,
            n_heads: int,
            n_classes: int,
            sigma_init: float = 1.0,
        ):
            super().__init__()
            self.input_proj = nn.Linear(n_features, d_model)
            self.attention = FuzzyAttentionLayer(
                d_model=d_model,
                n_heads=n_heads,
                sigma_init=sigma_init,
            )
            self.norm = nn.LayerNorm(d_model)
            self.classifier = nn.Linear(d_model, n_classes)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass.

            Parameters
            ----------
            x : Tensor of shape (batch, n_features)

            Returns
            -------
            Tensor of shape (batch, n_classes)
            """
            # Project to d_model and add sequence dim
            h = self.input_proj(x).unsqueeze(1)  # (batch, 1, d_model)
            h = self.attention(h, h, h)  # self-attention
            h = self.norm(h)
            h = h.squeeze(1)  # (batch, d_model)
            return self.classifier(h)
