"""Fully differentiable fuzzy inference system using PyTorch.

All membership function parameters, rule antecedents, and consequent
parameters are learned end-to-end via backpropagation.

Requires PyTorch.

References
----------
Deng, Z., et al. (2019). Deep neural network-based classification method
with Takagi-Sugeno-Kang fuzzy inference system. IEEE Access.

Example
-------
>>> from endgame.fuzzy.modern.differentiable import DifferentiableFuzzySystem
>>> model = DifferentiableFuzzySystem(n_rules=10, task='classification')
>>> model.fit(X_train, y_train)
>>> predictions = model.predict(X_test)
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
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for DifferentiableFuzzySystem. "
            "Install it with: pip install torch"
        )


if HAS_TORCH:

    class _DifferentiableTSK(nn.Module):
        """Differentiable TSK fuzzy system as a PyTorch module.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_rules : int
            Number of fuzzy rules.
        n_outputs : int
            Number of outputs.
        order : int
            TSK order (0 or 1).
        mf_type : str
            Membership function type: 'gaussian' or 'bell'.
        """

        def __init__(
            self,
            n_features: int,
            n_rules: int,
            n_outputs: int = 1,
            order: int = 1,
            mf_type: str = "gaussian",
        ):
            super().__init__()
            self.n_features = n_features
            self.n_rules = n_rules
            self.n_outputs = n_outputs
            self.order = order
            self.mf_type = mf_type

            # Antecedent parameters: centers and widths per rule per feature
            self.centers = nn.Parameter(torch.randn(n_rules, n_features) * 0.1)
            self.log_sigmas = nn.Parameter(torch.zeros(n_rules, n_features))

            if mf_type == "bell":
                self.log_b = nn.Parameter(torch.zeros(n_rules, n_features))

            # Consequent parameters
            if order == 0:
                self.consequents = nn.Parameter(
                    torch.randn(n_rules, n_outputs) * 0.01
                )
            else:
                self.consequents = nn.Parameter(
                    torch.randn(n_rules, n_features + 1, n_outputs) * 0.01
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through the fuzzy system.

            Parameters
            ----------
            x : Tensor of shape (batch, n_features)

            Returns
            -------
            Tensor of shape (batch, n_outputs)
            """
            # Compute firing strengths
            firing = self._compute_firing(x)  # (batch, n_rules)

            # Normalize
            fire_sum = firing.sum(dim=1, keepdim=True).clamp(min=1e-10)
            normalized = firing / fire_sum

            # Compute consequents
            if self.order == 0:
                # consequents: (n_rules, n_outputs)
                output = torch.matmul(
                    normalized, self.consequents
                )  # (batch, n_outputs)
            else:
                # consequents: (n_rules, n_features+1, n_outputs)
                batch_size = x.size(0)
                x_aug = torch.cat(
                    [x, torch.ones(batch_size, 1, device=x.device)], dim=1
                )  # (batch, n_features+1)

                # Per-rule consequent output
                rule_outputs = torch.einsum(
                    "bf,rfo->bro", x_aug, self.consequents
                )  # (batch, n_rules, n_outputs)

                output = torch.einsum(
                    "br,bro->bo", normalized, rule_outputs
                )  # (batch, n_outputs)

            return output

        def _compute_firing(self, x: torch.Tensor) -> torch.Tensor:
            """Compute firing strengths for all rules.

            Parameters
            ----------
            x : Tensor of shape (batch, n_features)

            Returns
            -------
            Tensor of shape (batch, n_rules)
            """
            sigmas = torch.exp(self.log_sigmas).clamp(min=1e-4)

            if self.mf_type == "gaussian":
                # x: (batch, features), centers: (rules, features)
                diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
                exponent = -0.5 * (diff / sigmas.unsqueeze(0)) ** 2
                memberships = torch.exp(exponent)  # (batch, rules, features)
            elif self.mf_type == "bell":
                b = torch.exp(self.log_b).clamp(min=0.1)
                diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
                memberships = 1.0 / (
                    1.0 + (torch.abs(diff) / sigmas.unsqueeze(0)) ** (2 * b.unsqueeze(0))
                )
            else:
                raise ValueError(f"Unknown mf_type: {self.mf_type}")

            # Product t-norm across features
            firing = memberships.prod(dim=2)  # (batch, n_rules)
            return firing


class DifferentiableFuzzySystem(BaseEstimator):
    """Fully end-to-end trainable fuzzy inference system.

    All membership function parameters are learned via backpropagation.
    Supports both regression and classification tasks.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    mf_type : str, default='gaussian'
        Membership function type: 'gaussian' or 'bell'.
    order : int, default=1
        TSK order (0=constant, 1=linear consequents).
    n_epochs : int, default=100
        Number of training epochs.
    lr : float, default=0.01
        Learning rate.
    batch_size : int, default=32
        Mini-batch size.
    task : str, default='regression'
        Task type: 'regression' or 'classification'.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    classes_ : ndarray (classification only)
        Unique class labels.
    model_ : _DifferentiableTSK
        Trained PyTorch model.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.differentiable import DifferentiableFuzzySystem
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.1
    >>> model = DifferentiableFuzzySystem(n_rules=5, n_epochs=50, task='regression')
    >>> model.fit(X, y)
    DifferentiableFuzzySystem(n_epochs=50, n_rules=5)
    >>> preds = model.predict(X[:5])
    """

    def __init__(
        self,
        n_rules: int = 10,
        mf_type: str = "gaussian",
        order: int = 1,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        task: str = "regression",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.mf_type = mf_type
        self.order = order
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.task = task
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> DifferentiableFuzzySystem:
        """Fit the differentiable fuzzy system.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values or class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        _check_torch()
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)

        if self.task == "classification":
            self.label_encoder_ = LabelEncoder()
            y_enc = self.label_encoder_.fit_transform(y)
            self.classes_ = self.label_encoder_.classes_
            n_outputs = len(self.classes_)
            y_t = torch.tensor(y_enc, dtype=torch.long)
            criterion = nn.CrossEntropyLoss()
        else:
            n_outputs = 1
            y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            criterion = nn.MSELoss()

        self.model_ = _DifferentiableTSK(
            n_features=X.shape[1],
            n_rules=self.n_rules,
            n_outputs=n_outputs,
            order=self.order,
            mf_type=self.mf_type,
        )

        # Initialize centers from data statistics
        with torch.no_grad():
            X_t_init = torch.tensor(X, dtype=torch.float32)
            x_mean = X_t_init.mean(dim=0)
            x_std = X_t_init.std(dim=0).clamp(min=1e-4)
            self.model_.centers.data = (
                x_mean.unsqueeze(0)
                + torch.randn_like(self.model_.centers) * x_std.unsqueeze(0)
            )
            self.model_.log_sigmas.data = torch.log(
                x_std.unsqueeze(0).expand_as(self.model_.log_sigmas)
            )

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.model_.train()
        self.loss_history_ = []

        for epoch in range(self.n_epochs):
            epoch_loss = 0.0
            n_batches = 0
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self.model_(X_batch)

                if self.task == "classification":
                    loss = criterion(output, y_batch)
                else:
                    loss = criterion(output, y_batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.loss_history_.append(avg_loss)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[DiffFuzzy] Epoch {epoch+1}/{self.n_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values or class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predictions.
        """
        _check_torch()
        check_is_fitted(self, ["model_"])
        X = check_array(X)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            output = self.model_(X_t)

        if self.task == "classification":
            proba = torch.softmax(output, dim=-1).numpy()
            return self.classes_[np.argmax(proba, axis=1)]
        else:
            return output.squeeze(-1).numpy()

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities (classification only).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.

        Raises
        ------
        ValueError
            If task is not 'classification'.
        """
        if self.task != "classification":
            raise ValueError("predict_proba is only available for classification task")

        _check_torch()
        check_is_fitted(self, ["model_"])
        X = check_array(X)

        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            output = self.model_(X_t)
            return torch.softmax(output, dim=-1).numpy()
