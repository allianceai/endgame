"""GRANDE: GRadient-based Neural Decision Ensemble for tabular data.

GRANDE learns hard axis-aligned decision tree ensembles using
backpropagation. It combines the inductive bias of decision trees
with the optimization power of gradient descent by using differentiable
soft routing during training that anneals to hard splits at inference.

References
----------
- Marton et al. "GRANDE: Gradient-Based Decision Tree Ensembles
  for Tabular Data" (2024)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

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
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GRANDE. "
            "Install with: pip install torch"
        )


class _BatchedGRANDENetwork(_nn_Module):
    """Vectorized ensemble of differentiable soft decision trees.

    All tree parameters are stored as single batched tensors so the
    entire ensemble is evaluated in one fused forward pass — no Python
    loop over individual trees.

    Parameters
    ----------
    n_features : int
        Number of input features.
    n_outputs : int
        Number of outputs (n_classes or 1).
    n_trees : int
        Number of trees.
    depth : int
        Depth of each tree.
    temperature : float
        Initial soft-routing temperature.
    """

    def __init__(
        self,
        n_features: int,
        n_outputs: int,
        n_trees: int = 128,
        depth: int = 5,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_trees = n_trees
        self.depth = depth
        self.temperature = temperature

        n_internal = 2 ** depth - 1
        n_leaves = 2 ** depth

        self.n_internal = n_internal
        self.n_leaves = n_leaves

        # Batched parameters — one tensor per parameter type across ALL trees.
        self.feature_weights = nn.Parameter(
            torch.randn(n_trees, n_internal, n_features) * 0.01
        )
        self.thresholds = nn.Parameter(torch.zeros(n_trees, n_internal))
        self.leaf_values = nn.Parameter(
            torch.randn(n_trees, n_leaves, n_outputs) * 0.01
        )

        # Pre-compute level start/size pairs (tiny, done once)
        self._levels = [
            (2 ** lvl - 1, 2 ** lvl) for lvl in range(depth)
        ]

    def forward(
        self,
        x: torch.Tensor,
        temperature: float | None = None,
    ) -> torch.Tensor:
        """Batched forward pass over all trees.

        Parameters
        ----------
        x : (batch, n_features)
        temperature : optional override

        Returns
        -------
        (batch, n_outputs) — averaged ensemble prediction.
        """
        temp = temperature if temperature is not None else self.temperature
        batch = x.shape[0]

        # Soft feature selection: (n_trees, n_internal, n_features)
        feature_probs = F.softmax(self.feature_weights / temp, dim=2)

        # Selected feature values per internal node per tree:
        # einsum 'bf,tif->tbi' = for each (tree, batch, internal):
        #   sum_f x[b,f] * feature_probs[t,i,f]
        selected = torch.einsum("bf,tif->tbi", x, feature_probs)

        # Routing probabilities: (n_trees, batch, n_internal)
        route = torch.sigmoid(
            (selected - self.thresholds.unsqueeze(1)) / temp
        )

        # Level-by-level leaf probability computation.
        # The Python loop is over *depth* (typically 4-6), not n_trees.
        node_probs = x.new_ones(self.n_trees, batch, 1)

        for level_start, level_size in self._levels:
            lp = node_probs[:, :, level_start : level_start + level_size]
            lr = route[:, :, level_start : level_start + level_size]
            # (n_trees, batch, level_size, 2) → (n_trees, batch, 2*level_size)
            children = torch.stack([lp * (1.0 - lr), lp * lr], dim=3)
            children = children.reshape(self.n_trees, batch, -1)
            node_probs = torch.cat([node_probs, children], dim=2)

        leaf_probs = node_probs[:, :, self.n_internal :]  # (T, B, L)

        # Weighted sum of leaf values: (T, B, O) = (T, B, L) @ (T, L, O)
        tree_out = torch.bmm(leaf_probs, self.leaf_values)

        return tree_out.mean(dim=0)  # (B, O)

    def feature_importances(self) -> np.ndarray:
        """Derive importances from softmax feature-selection weights."""
        with torch.no_grad():
            # (n_trees, n_internal, n_features) → softmax → sum over trees & nodes
            w = F.softmax(self.feature_weights, dim=2)
            imp = w.sum(dim=(0, 1)).cpu().numpy()
        total = imp.sum()
        if total > 0:
            imp /= total
        return imp


class GRANDEClassifier(ClassifierMixin, BaseEstimator):
    """GRANDE classifier: Gradient-based Neural Decision Ensemble.

    Learns an ensemble of differentiable decision trees using
    backpropagation. During training, soft routing (sigmoid-based)
    allows gradient flow through tree structure. Temperature
    annealing transitions from soft to hard splits.

    Parameters
    ----------
    n_trees : int, default=128
        Number of trees in the ensemble.
    depth : int, default=5
        Depth of each decision tree.
    lr : float, default=0.005
        Learning rate for Adam optimizer.
    weight_decay : float, default=0.0
        L2 regularization strength.
    n_epochs : int, default=50
        Maximum number of training epochs.
    batch_size : int, default=512
        Training batch size.
    patience : int, default=10
        Early stopping patience (epochs without improvement).
    temperature_init : float, default=1.0
        Initial temperature for soft routing.
    temperature_final : float, default=0.1
        Final temperature after annealing.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during training.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : _BatchedGRANDENetwork
        Fitted PyTorch network.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores derived from learned feature
        selection weights across all trees and internal nodes.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.tabular import GRANDEClassifier
    >>> clf = GRANDEClassifier(n_trees=64, depth=4, n_epochs=50)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_trees: int = 128,
        depth: int = 5,
        lr: float = 0.005,
        weight_decay: float = 0.0,
        n_epochs: int = 50,
        batch_size: int = 512,
        patience: int = 10,
        temperature_init: float = 1.0,
        temperature_final: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_trees = n_trees
        self.depth = depth
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GRANDE] {msg}")

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

    def _temperature(self, epoch: int, n_epochs: int) -> float:
        if n_epochs <= 1:
            return self.temperature_final
        t = epoch / (n_epochs - 1)
        log_init = np.log(max(self.temperature_init, 1e-8))
        log_fin = np.log(max(self.temperature_final, 1e-8))
        return float(np.exp(log_init + t * (log_fin - log_init)))

    def fit(self, X, y, **fit_params) -> GRANDEClassifier:
        """Fit the GRANDE classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        self._label_encoder = LabelEncoder()
        y_enc = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        n_features = Xs.shape[1]
        self._n_features = n_features

        # Train / val split
        n = Xs.shape[0]
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        nv = max(1, int(0.2 * n))
        vi, ti = idx[:nv], idx[nv:]

        use_pin = self._device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(Xs[ti]),
                torch.from_numpy(y_enc[ti].astype(np.int64)),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin,
        )
        X_val = torch.from_numpy(Xs[vi]).to(self._device)
        y_val = torch.from_numpy(y_enc[vi].astype(np.int64)).to(self._device)

        self.model_ = _BatchedGRANDENetwork(
            n_features=n_features,
            n_outputs=self.n_classes_,
            n_trees=self.n_trees,
            depth=self.depth,
            temperature=self.temperature_init,
        ).to(self._device)

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        self._log(
            f"Training on {self._device} — "
            f"{self.n_trees} trees, depth={self.depth}"
        )

        for epoch in range(self.n_epochs):
            temp = self._temperature(epoch, self.n_epochs)

            self.model_.train()
            loss_sum, nb = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(self._device, non_blocking=use_pin)
                yb = yb.to(self._device, non_blocking=use_pin)
                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(xb, temperature=temp)
                loss = F.cross_entropy(logits, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                nb += 1

            tl = loss_sum / max(nb, 1)
            self.history_["train_loss"].append(tl)

            self.model_.eval()
            with torch.no_grad():
                vl = F.cross_entropy(
                    self.model_(X_val, temperature=temp), y_val
                ).item()
            self.history_["val_loss"].append(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch+1}/{self.n_epochs}: "
                    f"train={tl:.4f} val={vl:.4f} temp={temp:.4f}"
                )
            if patience_ctr >= self.patience:
                self._log(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.feature_importances_ = self.model_.feature_importances()
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
        """
        self._check_is_fitted()
        Xs = self._scaler.transform(
            np.asarray(X, dtype=np.float32)
        ).astype(np.float32)

        self.model_.eval()
        parts = []
        with torch.no_grad():
            for s in range(0, Xs.shape[0], self.batch_size):
                xb = torch.from_numpy(Xs[s : s + self.batch_size]).to(
                    self._device
                )
                logits = self.model_(xb, temperature=self.temperature_final)
                parts.append(F.softmax(logits, dim=1).cpu().numpy())
        return np.vstack(parts)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray
        """
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def _check_is_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("GRANDEClassifier has not been fitted.")


class GRANDERegressor(RegressorMixin, BaseEstimator):
    """GRANDE regressor: Gradient-based Neural Decision Ensemble.

    Same architecture as GRANDEClassifier but adapted for regression
    with MSE loss and a single output per leaf.

    Parameters
    ----------
    n_trees : int, default=128
        Number of trees in the ensemble.
    depth : int, default=5
        Depth of each decision tree.
    lr : float, default=0.005
        Learning rate for Adam optimizer.
    weight_decay : float, default=0.0
        L2 regularization strength.
    n_epochs : int, default=50
        Maximum number of training epochs.
    batch_size : int, default=512
        Training batch size.
    patience : int, default=10
        Early stopping patience.
    temperature_init : float, default=1.0
        Initial temperature for soft routing.
    temperature_final : float, default=0.1
        Final temperature after annealing.
    device : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during training.

    Attributes
    ----------
    model_ : _BatchedGRANDENetwork
        Fitted PyTorch network.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importance scores.
    history_ : dict
        Training history with 'train_loss' and 'val_loss'.

    Examples
    --------
    >>> from endgame.models.tabular import GRANDERegressor
    >>> reg = GRANDERegressor(n_trees=64, depth=4, n_epochs=50)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_trees: int = 128,
        depth: int = 5,
        lr: float = 0.005,
        weight_decay: float = 0.0,
        n_epochs: int = 50,
        batch_size: int = 512,
        patience: int = 10,
        temperature_init: float = 1.0,
        temperature_final: float = 0.1,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_trees = n_trees
        self.depth = depth
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.temperature_init = temperature_init
        self.temperature_final = temperature_final
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GRANDE] {msg}")

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

    def _temperature(self, epoch: int, n_epochs: int) -> float:
        if n_epochs <= 1:
            return self.temperature_final
        t = epoch / (n_epochs - 1)
        log_init = np.log(max(self.temperature_init, 1e-8))
        log_fin = np.log(max(self.temperature_final, 1e-8))
        return float(np.exp(log_init + t * (log_fin - log_init)))

    def fit(self, X, y, **fit_params) -> GRANDERegressor:
        """Fit the GRANDE regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        _check_torch()
        self._set_seed()
        self._device = self._get_device()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self._scaler = StandardScaler()
        Xs = self._scaler.fit_transform(X).astype(np.float32)

        self._target_scaler = StandardScaler()
        ys = self._target_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel().astype(np.float32)

        n_features = Xs.shape[1]
        self._n_features = n_features

        n = Xs.shape[0]
        rng = np.random.RandomState(self.random_state)
        idx = rng.permutation(n)
        nv = max(1, int(0.2 * n))
        vi, ti = idx[:nv], idx[nv:]

        use_pin = self._device.type == "cuda"
        train_loader = DataLoader(
            TensorDataset(
                torch.from_numpy(Xs[ti]),
                torch.from_numpy(ys[ti]),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=use_pin,
        )
        X_val = torch.from_numpy(Xs[vi]).to(self._device)
        y_val = torch.from_numpy(ys[vi]).to(self._device)

        self.model_ = _BatchedGRANDENetwork(
            n_features=n_features,
            n_outputs=1,
            n_trees=self.n_trees,
            depth=self.depth,
            temperature=self.temperature_init,
        ).to(self._device)

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        best_val_loss = float("inf")
        best_state = None
        patience_ctr = 0
        self.history_ = {"train_loss": [], "val_loss": []}

        self._log(
            f"Training on {self._device} — "
            f"{self.n_trees} trees, depth={self.depth}"
        )

        for epoch in range(self.n_epochs):
            temp = self._temperature(epoch, self.n_epochs)

            self.model_.train()
            loss_sum, nb = 0.0, 0
            for xb, yb in train_loader:
                xb = xb.to(self._device, non_blocking=use_pin)
                yb = yb.to(self._device, non_blocking=use_pin)
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb, temperature=temp).squeeze(-1)
                loss = F.mse_loss(pred, yb)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                nb += 1

            tl = loss_sum / max(nb, 1)
            self.history_["train_loss"].append(tl)

            self.model_.eval()
            with torch.no_grad():
                vp = self.model_(X_val, temperature=temp).squeeze(-1)
                vl = F.mse_loss(vp, y_val).item()
            self.history_["val_loss"].append(vl)

            if vl < best_val_loss:
                best_val_loss = vl
                best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model_.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1

            if self.verbose and (epoch + 1) % 10 == 0:
                self._log(
                    f"Epoch {epoch+1}/{self.n_epochs}: "
                    f"train={tl:.4f} val={vl:.4f} temp={temp:.4f}"
                )
            if patience_ctr >= self.patience:
                self._log(f"Early stopping at epoch {epoch+1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.feature_importances_ = self.model_.feature_importances()
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
        """
        self._check_is_fitted()
        Xs = self._scaler.transform(
            np.asarray(X, dtype=np.float32)
        ).astype(np.float32)

        self.model_.eval()
        parts = []
        with torch.no_grad():
            for s in range(0, Xs.shape[0], self.batch_size):
                xb = torch.from_numpy(Xs[s : s + self.batch_size]).to(
                    self._device
                )
                pred = self.model_(
                    xb, temperature=self.temperature_final
                ).squeeze(-1)
                parts.append(pred.cpu().numpy())

        scaled = np.concatenate(parts)
        return self._target_scaler.inverse_transform(
            scaled.reshape(-1, 1)
        ).ravel()

    def _check_is_fitted(self):
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("GRANDERegressor has not been fitted.")
