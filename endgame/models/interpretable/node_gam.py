from __future__ import annotations

"""NODE-GAM: Neural Oblivious Decision Ensembles as GAMs.

NODE-GAM combines the differentiable decision trees from NODE with the
interpretability constraints of GAMs. It learns a separate ensemble of
oblivious decision trees for each feature, producing smooth shape functions.

References
----------
- Chang et al. "NODE-GAM: Neural Generalized Additive Model for
  Interpretable Deep Learning" (ICLR 2022)
- https://github.com/zzzace2000/node-gam

Example
-------
>>> from endgame.models.interpretable import NodeGAMClassifier
>>> clf = NodeGAMClassifier(n_trees_per_feature=32, depth=4)
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> contributions = clf.get_feature_contributions(X_test)
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

from endgame.core.glassbox import GlassboxMixin

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
            "PyTorch is required for NODE-GAM. "
            "Install with: pip install endgame-ml[tabular]"
        )


class _EntmaxBisect(torch.autograd.Function):
    """Entmax 1.5 activation using bisection algorithm.

    Entmax is a sparse alternative to softmax that produces exact zeros.
    1.5-entmax is between softmax (alpha=1) and sparsemax (alpha=2).
    """

    @staticmethod
    def forward(ctx, input, alpha=1.5, dim=-1, n_iter=20):
        ctx.alpha = alpha
        ctx.dim = dim

        # Bisection to find threshold
        input_max = input.max(dim=dim, keepdim=True)[0]
        input_shifted = input - input_max

        # Initialize bounds
        tau_lo = input_shifted.min(dim=dim, keepdim=True)[0] - 1
        tau_hi = input_max.clone().fill_(0)

        for _ in range(n_iter):
            tau_mid = (tau_lo + tau_hi) / 2
            p = torch.clamp(input_shifted - tau_mid, min=0) ** (1 / (alpha - 1))
            p_sum = p.sum(dim=dim, keepdim=True)

            # Update bounds
            tau_lo = torch.where(p_sum < 1, tau_lo, tau_mid)
            tau_hi = torch.where(p_sum < 1, tau_mid, tau_hi)

        # Final computation
        tau = (tau_lo + tau_hi) / 2
        output = torch.clamp(input_shifted - tau, min=0) ** (1 / (alpha - 1))

        # Normalize
        output = output / output.sum(dim=dim, keepdim=True).clamp(min=1e-10)

        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        alpha = ctx.alpha
        dim = ctx.dim

        # Gradient computation
        gppr = output ** (2 - alpha)
        gppr_sum = gppr.sum(dim=dim, keepdim=True)

        grad = grad_output * gppr
        grad_sum = grad.sum(dim=dim, keepdim=True)

        grad_input = grad - gppr * (grad_sum / gppr_sum.clamp(min=1e-10))

        return grad_input, None, None, None


def entmax15(input, dim=-1):
    """Apply 1.5-entmax activation."""
    return _EntmaxBisect.apply(input, 1.5, dim)


class _BatchedObliviousTreeEnsemble(nn.Module):
    """Batched ensemble of oblivious decision trees for a single feature.

    Processes all trees in parallel using batched tensor operations
    instead of sequential Python loops.
    """

    def __init__(self, n_trees: int, depth: int, temperature: float = 1.0):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.temperature = temperature

        self.thresholds = nn.Parameter(torch.zeros(n_trees, depth))
        self.responses = nn.Parameter(torch.zeros(n_trees, self.n_leaves))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for all trees at once.

        Parameters
        ----------
        x : Tensor of shape (batch, 1)

        Returns
        -------
        output : Tensor of shape (batch,)
        """
        # x: (batch, 1) -> expand to (batch, n_trees, depth)
        x_expanded = x.expand(-1, self.n_trees).unsqueeze(-1).expand(-1, -1, self.depth)

        split_decisions = torch.sigmoid(
            (x_expanded - self.thresholds.unsqueeze(0)) / self.temperature
        )  # (batch, n_trees, depth)

        # Pre-compute leaf path masks: (n_leaves, depth)
        leaf_indices = torch.arange(self.n_leaves, device=x.device)
        goes_right = torch.stack([
            ((leaf_indices >> d) & 1).float() for d in range(self.depth)
        ], dim=1)  # (n_leaves, depth)

        # Compute leaf probs: (batch, n_trees, n_leaves)
        p_right = split_decisions.unsqueeze(2)  # (batch, n_trees, 1, depth)
        goes_right_exp = goes_right.unsqueeze(0).unsqueeze(0)  # (1, 1, n_leaves, depth)
        p_path = goes_right_exp * p_right + (1 - goes_right_exp) * (1 - p_right)
        leaf_probs = p_path.prod(dim=-1)  # (batch, n_trees, n_leaves)

        tree_outputs = (leaf_probs * self.responses.unsqueeze(0)).sum(dim=-1)  # (batch, n_trees)
        return tree_outputs.sum(dim=-1)  # (batch,)


_FEATURE_CHUNK = 200

_MAX_LEAF_COMPLEXITY = 20_000


def _adapt_complexity(n_features, n_trees, depth):
    """Scale down tree count / depth for high-dimensional inputs.

    Keeps ``n_features * n_trees * 2**depth`` below ``_MAX_LEAF_COMPLEXITY``
    to bound per-forward-pass memory.  Depth is reduced first (cheapest
    quality loss), then tree count.
    """
    eff_trees = n_trees
    eff_depth = depth
    while n_features * eff_trees * (2 ** eff_depth) > _MAX_LEAF_COMPLEXITY:
        if eff_depth > 2:
            eff_depth -= 1
        elif eff_trees > 4:
            eff_trees = max(4, eff_trees // 2)
        else:
            break
    return eff_trees, eff_depth


class _NodeGAMModule(nn.Module):
    """NODE-GAM PyTorch module (vectorized).

    Uses batched tree ensembles per feature with all features
    processed via a single stacked operation.  For high-feature-count
    inputs (>_FEATURE_CHUNK) the forward pass processes features in
    chunks to keep intermediate tensors cache-friendly and avoids
    allocating a massive 5-D tensor.
    """

    def __init__(
        self,
        n_features: int,
        n_classes: int,
        n_trees_per_feature: int = 32,
        depth: int = 4,
        temperature: float = 1.0,
        is_regression: bool = False,
    ):
        super().__init__()

        self.n_features = n_features
        self.n_classes = n_classes
        self.is_regression = is_regression
        self.n_trees = n_trees_per_feature
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.temperature = temperature

        self.thresholds = nn.Parameter(
            torch.zeros(n_features, n_trees_per_feature, depth)
        )
        self.responses = nn.Parameter(
            torch.zeros(n_features, n_trees_per_feature, self.n_leaves)
        )

        leaf_indices = torch.arange(self.n_leaves)
        goes_right = torch.stack([
            ((leaf_indices >> d) & 1).float() for d in range(depth)
        ])  # (depth, n_leaves)
        self.register_buffer("_goes_right", goes_right)

        output_dim = 1 if is_regression or n_classes <= 2 else n_classes
        self.bias = nn.Parameter(torch.zeros(output_dim))

        if not is_regression and n_classes > 2:
            self.output_layer = nn.Linear(n_features, n_classes, bias=False)
        else:
            self.output_layer = None

    def _leaf_probs_chunk(
        self,
        x_chunk: torch.Tensor,
        thresh_chunk: torch.Tensor,
    ) -> torch.Tensor:
        """Compute leaf probabilities for a contiguous chunk of features.

        Iterates over depth levels instead of materialising a 5-D tensor,
        reducing peak memory from O(B*F*T*L*D) to O(B*F*T*L).
        """
        x_exp = x_chunk.unsqueeze(2).unsqueeze(3)  # (B, Fc, 1, 1) — view
        split_decisions = torch.sigmoid(
            (x_exp - thresh_chunk.unsqueeze(0)) / self.temperature
        )  # (B, Fc, T, D)

        leaf_probs = torch.ones(
            x_chunk.shape[0], x_chunk.shape[1], self.n_trees, self.n_leaves,
            device=x_chunk.device,
        )
        for d in range(self.depth):
            p_d = split_decisions[:, :, :, d].unsqueeze(-1)  # (B,Fc,T,1)
            gr_d = self._goes_right[d]  # (n_leaves,)
            leaf_probs = leaf_probs * (gr_d * p_d + (1.0 - gr_d) * (1.0 - p_d))
        return leaf_probs  # (B, Fc, T, n_leaves)

    def forward(
        self,
        x: torch.Tensor,
        return_contributions: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        For feature counts above ``_FEATURE_CHUNK`` the computation is
        split into chunks so that intermediate tensors stay cache-friendly.
        """
        F = self.n_features
        chunk = _FEATURE_CHUNK

        if chunk >= F or return_contributions or self.output_layer is not None:
            leaf_probs = self._leaf_probs_chunk(x, self.thresholds)
            contributions = torch.einsum(
                "bftl,ftl->bf", leaf_probs, self.responses,
            )
        else:
            contrib_sums: list[torch.Tensor] = []
            for start in range(0, F, chunk):
                end = min(start + chunk, F)
                lp = self._leaf_probs_chunk(
                    x[:, start:end], self.thresholds[start:end],
                )
                cs = torch.einsum(
                    "bftl,ftl->b", lp, self.responses[start:end],
                )
                contrib_sums.append(cs)
            output = torch.stack(contrib_sums).sum(dim=0).unsqueeze(1) + self.bias
            return output

        if self.output_layer is not None:
            output = self.output_layer(contributions) + self.bias
        else:
            output = contributions.sum(dim=1, keepdim=True) + self.bias

        if return_contributions:
            return output, contributions
        return output

    def get_feature_importance(self) -> np.ndarray:
        """Compute feature importance based on tree responses."""
        importance = self.responses.abs().sum(dim=(1, 2)).detach().cpu().numpy()
        total = importance.sum()
        if total > 0:
            importance = importance / total
        return importance


class NodeGAMClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """NODE-GAM Classifier.

    Neural Oblivious Decision Ensembles combined with GAM structure.
    Learns differentiable decision tree ensembles for each feature,
    producing interpretable shape functions.

    Parameters
    ----------
    n_trees_per_feature : int, default=32
        Number of oblivious trees per feature.

    depth : int, default=4
        Depth of each oblivious tree (2^depth leaves per tree).

    temperature : float, default=1.0
        Temperature for soft split decisions. Lower values make
        decisions more crisp (closer to hard trees).

    learning_rate : float, default=1e-3
        Learning rate for Adam optimizer.

    weight_decay : float, default=1e-5
        L2 regularization.

    n_epochs : int, default=100
        Maximum training epochs.

    batch_size : int, default=128
        Training batch size.

    early_stopping : int, default=20
        Early stopping patience.

    validation_fraction : float, default=0.1
        Fraction for validation when eval_set not provided.

    device : str, default="auto"
        Device to use: "cuda", "cpu", or "auto".

    random_state : int, optional
        Random seed.

    verbose : bool, default=False
        Verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    n_features_in_ : int
        Number of features.

    model_ : _NodeGAMModule
        Fitted model.

    feature_importances_ : ndarray
        Feature importance scores.

    Examples
    --------
    >>> from endgame.models.interpretable import NodeGAMClassifier
    >>> clf = NodeGAMClassifier(n_trees_per_feature=32, depth=4)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> contributions = clf.get_feature_contributions(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_trees_per_feature: int = 16,
        depth: int = 3,
        temperature: float = 1.0,
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
        self.n_trees_per_feature = n_trees_per_feature
        self.depth = depth
        self.temperature = temperature
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
            print(f"[NODE-GAM] {msg}")

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
    ) -> NodeGAMClassifier:
        """Fit the NODE-GAM classifier.

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
        self : NodeGAMClassifier
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

        eff_trees, eff_depth = _adapt_complexity(
            self.n_features_in_, self.n_trees_per_feature, self.depth,
        )

        self.model_ = _NodeGAMModule(
            n_features=self.n_features_in_,
            n_classes=self.n_classes_,
            n_trees_per_feature=eff_trees,
            depth=eff_depth,
            temperature=self.temperature,
            is_regression=False,
        ).to(self._device)

        # Training setup
        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_encoded, dtype=torch.long)

        # Create validation split if needed
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

        train_loader = DataLoader(
            TensorDataset(x_tensor, y_tensor),
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if eval_set is not None:
            X_val, y_val = eval_set
            if not isinstance(y_val, np.ndarray) or y_val.dtype.kind not in ('i', 'u'):
                X_val = self._scaler.transform(np.asarray(X_val, dtype=np.float32))
                X_val = np.nan_to_num(X_val, nan=0.0)
                y_val = self._label_encoder.transform(y_val)
            val_loader = DataLoader(
                TensorDataset(
                    torch.tensor(X_val, dtype=torch.float32),
                    torch.tensor(y_val, dtype=torch.long),
                ),
                batch_size=self.batch_size,
            )

        # Optimizer
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device} with {self.n_features_in_} features...")

        # Determine loss function based on number of classes
        is_binary = self.n_classes_ == 2

        for epoch in range(self.n_epochs):
            # Training
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                logits = self.model_(x_batch)

                if is_binary:
                    # Binary: use BCE with logits, target must be float
                    loss = F.binary_cross_entropy_with_logits(
                        logits.squeeze(-1), y_batch.float()
                    )
                else:
                    # Multiclass: use cross entropy
                    loss = F.cross_entropy(logits, y_batch)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches

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

                        if is_binary:
                            loss = F.binary_cross_entropy_with_logits(
                                logits.squeeze(-1), y_batch.float()
                            )
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

                if self.verbose and (epoch + 1) % 10 == 0:
                    self._log(f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}")

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch + 1}")
                    break

        # Restore best
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.model_.eval()
        self.feature_importances_ = self.model_.get_feature_importance()
        self._is_fitted = True

        return self

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
                    # Binary: logits is (batch, 1)
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

    def get_feature_contributions(self, X) -> np.ndarray:
        """Get per-feature contributions for predictions.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        contributions : ndarray of shape (n_samples, n_features)
        """
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_contributions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                _, contributions = self.model_(x_batch, return_contributions=True)
                all_contributions.append(contributions.cpu().numpy())

        return np.vstack(all_contributions)


class NodeGAMRegressor(GlassboxMixin, RegressorMixin, BaseEstimator):
    """NODE-GAM Regressor.

    Same architecture as NodeGAMClassifier but with MSE loss.

    Parameters
    ----------
    n_trees_per_feature : int, default=32
        Number of trees per feature.

    depth : int, default=4
        Tree depth.

    temperature : float, default=1.0
        Softness of decisions.

    learning_rate : float, default=1e-3
        Learning rate.

    weight_decay : float, default=1e-5
        L2 regularization.

    n_epochs : int, default=100
        Max epochs.

    batch_size : int, default=128
        Batch size.

    early_stopping : int, default=20
        Patience.

    validation_fraction : float, default=0.1
        Validation split.

    device : str, default="auto"
        Device.

    random_state : int, optional
        Seed.

    verbose : bool, default=False
        Verbosity.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_trees_per_feature: int = 16,
        depth: int = 3,
        temperature: float = 1.0,
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
        self.n_trees_per_feature = n_trees_per_feature
        self.depth = depth
        self.temperature = temperature
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
            print(f"[NODE-GAM] {msg}")

    def _get_device(self):
        _check_torch()
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

    def fit(self, X, y, eval_set=None, **fit_params) -> NodeGAMRegressor:
        """Fit the regressor."""
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

        eff_trees, eff_depth = _adapt_complexity(
            self.n_features_in_, self.n_trees_per_feature, self.depth,
        )

        self.model_ = _NodeGAMModule(
            n_features=self.n_features_in_,
            n_classes=1,
            n_trees_per_feature=eff_trees,
            depth=eff_depth,
            temperature=self.temperature,
            is_regression=True,
        ).to(self._device)

        x_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(y_scaled, dtype=torch.float32)

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

        for epoch in range(self.n_epochs):
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

        self.model_.eval()
        self.feature_importances_ = self.model_.get_feature_importance()
        self._is_fitted = True

        return self

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

    def get_feature_contributions(self, X) -> np.ndarray:
        """Get per-feature contributions."""
        check_is_fitted(self, "model_")

        X = np.asarray(X, dtype=np.float32)
        X_scaled = np.nan_to_num(self._scaler.transform(X), nan=0.0)

        self.model_.eval()
        all_contributions = []

        with torch.no_grad():
            for start in range(0, len(X), self.batch_size):
                end = min(start + self.batch_size, len(X))
                x_batch = torch.tensor(X_scaled[start:end], dtype=torch.float32).to(self._device)
                _, contributions = self.model_(x_batch, return_contributions=True)
                all_contributions.append(contributions.cpu().numpy())

        return np.vstack(all_contributions)


def _nodegam_structure(self, *, link: str) -> dict[str, Any]:
    check_is_fitted(self, "model_")
    feature_names = self._structure_feature_names(self.n_features_in_)
    return {
        "link": link,
        "terms": [
            {
                "name": feature_names[i],
                "feature_index": i,
                "type": "main",
                "importance": float(self.feature_importances_[i]),
            }
            for i in range(self.n_features_in_)
        ],
        "n_trees_per_feature": int(getattr(self, "n_trees_per_feature", 0)),
        "tree_depth": int(getattr(self, "depth", 0)),
        "feature_importances": self.feature_importances_.tolist(),
        "note": "Shape functions are learned neural oblivious trees; use get_feature_contributions(X) for per-sample effects.",
    }


NodeGAMClassifier._structure_type = "additive"
NodeGAMClassifier._structure_content = lambda self: _nodegam_structure(self, link="logit")
NodeGAMRegressor._structure_type = "additive"
NodeGAMRegressor._structure_content = lambda self: _nodegam_structure(self, link="identity")
