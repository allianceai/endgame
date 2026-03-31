"""FALCON (Fuzzy Adaptive Learning Control Network).

Five-layer architecture mapping inputs through fuzzification, rules,
and defuzzification. All parameters are trained via backpropagation.

References
----------
- Lin & Lee, "A Neural Fuzzy System with Linguistic Teaching Signals" (1995)

Example
-------
>>> from endgame.fuzzy.neurofuzzy import FALCONRegressor
>>> model = FALCONRegressor(n_rules=10, n_epochs=100)
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
            "PyTorch is required for FALCON. Install with: pip install torch"
        )


if HAS_TORCH:
    class _FALCONModule(nn.Module):
        """Five-layer FALCON network.

        Layer 1: Input layer
        Layer 2: Fuzzification (Gaussian MFs with learnable params)
        Layer 3: Rule layer (t-norm of antecedent memberships)
        Layer 4: Consequent layer (linear functions per rule)
        Layer 5: Defuzzification (weighted average)
        """

        def __init__(
            self,
            n_features: int,
            n_mfs: int,
            n_rules: int,
            n_outputs: int = 1,
        ):
            super().__init__()
            self.n_features = n_features
            self.n_mfs = n_mfs
            self.n_rules = n_rules
            self.n_outputs = n_outputs

            # Layer 2: Gaussian MF parameters (centers and widths)
            self.centers = nn.Parameter(
                torch.randn(n_features, n_mfs)
            )
            self.widths = nn.Parameter(
                torch.ones(n_features, n_mfs) * 0.5
            )

            # Rule-antecedent connection: which MF per feature per rule
            # Use soft assignment via attention-like weights
            self.rule_weights = nn.Parameter(
                torch.randn(n_rules, n_features, n_mfs)
            )

            # Layer 4: Consequent parameters (order-1 TSK)
            self.consequent_weights = nn.Parameter(
                torch.randn(n_rules, n_features + 1, n_outputs)
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch_size = x.shape[0]

            # Layer 2: Fuzzification
            # x: (batch, n_features) -> memberships: (batch, n_features, n_mfs)
            x_expanded = x.unsqueeze(2)  # (batch, n_features, 1)
            sigma = torch.abs(self.widths) + 1e-6  # Ensure positive
            memberships = torch.exp(
                -0.5 * ((x_expanded - self.centers) / sigma) ** 2
            )  # (batch, n_features, n_mfs)

            # Layer 3: Rule firing strengths
            # Soft rule-antecedent selection via softmax
            rule_attn = torch.softmax(self.rule_weights, dim=2)  # (n_rules, n_features, n_mfs)
            # Weighted membership per rule per feature
            # (batch, 1, n_features, n_mfs) * (1, n_rules, n_features, n_mfs)
            weighted_mu = memberships.unsqueeze(1) * rule_attn.unsqueeze(0)
            # Sum over MFs to get per-feature per-rule activation
            rule_feature_mu = weighted_mu.sum(dim=3)  # (batch, n_rules, n_features)
            # Product t-norm across features
            firing = rule_feature_mu.prod(dim=2)  # (batch, n_rules)
            firing = firing + 1e-10

            # Normalize firing strengths
            firing_norm = firing / (firing.sum(dim=1, keepdim=True) + 1e-10)

            # Layer 4: Consequent computation (order-1 TSK)
            x_aug = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)
            # (batch, n_features+1) @ (n_rules, n_features+1, n_outputs)
            # -> (batch, n_rules, n_outputs)
            consequents = torch.einsum("bf,rfk->brk", x_aug, self.consequent_weights)

            # Layer 5: Weighted average
            output = (firing_norm.unsqueeze(2) * consequents).sum(dim=1)
            return output

        def init_from_data(self, X: torch.Tensor, y: torch.Tensor = None):
            """Initialize MF parameters from data statistics."""
            with torch.no_grad():
                for j in range(self.n_features):
                    x_min = X[:, j].min()
                    x_max = X[:, j].max()
                    self.centers.data[j] = torch.linspace(x_min, x_max, self.n_mfs)
                    self.widths.data[j] = (x_max - x_min) / (self.n_mfs * 2) + 0.01


class FALCONRegressor(BaseEstimator, RegressorMixin):
    """FALCON regression network.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input.
    n_epochs : int, default=100
        Training epochs.
    lr : float, default=0.01
        Learning rate.
    batch_size : int, default=32
        Mini-batch size.
    early_stopping : int or None, default=10
        Stop after this many epochs without improvement. None disables.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int | None = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FALCONRegressor:
        """Fit the FALCON network.

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
            np.random.seed(self.random_state)

        device = torch.device("cpu")
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(1)

        self.module_ = _FALCONModule(
            n_features=self.n_features_in_,
            n_mfs=self.n_mfs,
            n_rules=self.n_rules,
            n_outputs=1,
        ).to(device)
        self.module_.init_from_data(X_t)

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_loss = float("inf")
        patience_counter = 0
        n = len(X_t)

        for epoch in range(self.n_epochs):
            self.module_.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_batch = X_t[idx]
                y_batch = y_t[idx]

                optimizer.zero_grad()
                output = self.module_(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.n_epochs}, Loss: {avg_loss:.6f}")

            if self.early_stopping is not None:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.best_state_ = {
                        k: v.clone() for k, v in self.module_.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        if self.verbose:
                            print(f"Early stopping at epoch {epoch+1}")
                        break

        if self.early_stopping is not None and hasattr(self, "best_state_"):
            self.module_.load_state_dict(self.best_state_)

        self.n_iter_ = epoch + 1
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
            X_t = torch.tensor(X, dtype=torch.float32)
            output = self.module_(X_t)
        return output.numpy().ravel()


class FALCONClassifier(BaseEstimator, ClassifierMixin):
    """FALCON classification network.

    Uses a multi-output FALCON with softmax for class probabilities.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input.
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
        n_rules: int = 10,
        n_mfs: int = 3,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int | None = 10,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FALCONClassifier:
        """Fit the FALCON classifier.

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
        self.n_classes_ = len(self.classes_)

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        device = torch.device("cpu")
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        y_t = torch.tensor(y_enc, dtype=torch.long, device=device)

        self.module_ = _FALCONModule(
            n_features=self.n_features_in_,
            n_mfs=self.n_mfs,
            n_rules=self.n_rules,
            n_outputs=self.n_classes_,
        ).to(device)
        self.module_.init_from_data(X_t)

        optimizer = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()

        best_loss = float("inf")
        patience_counter = 0
        n = len(X_t)

        for epoch in range(self.n_epochs):
            self.module_.train()
            perm = torch.randperm(n)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, self.batch_size):
                idx = perm[i:i + self.batch_size]
                optimizer.zero_grad()
                output = self.module_(X_t[idx])
                loss = criterion(output, y_t[idx])
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if self.early_stopping is not None:
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                    self.best_state_ = {
                        k: v.clone() for k, v in self.module_.state_dict().items()
                    }
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        break

        if self.early_stopping is not None and hasattr(self, "best_state_"):
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
            X_t = torch.tensor(X, dtype=torch.float32)
            logits = self.module_(X_t)
            proba = torch.softmax(logits, dim=1)
        return proba.numpy()

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
