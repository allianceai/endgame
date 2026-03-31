"""Interval Type-2 ANFIS (Adaptive Neuro-Fuzzy Inference System).

ANFIS with IT2 membership functions in Layer 1 producing dual
(upper/lower) firing strengths that propagate through the network.

Requires PyTorch for gradient-based optimization of MF parameters.

Example
-------
>>> from endgame.fuzzy.type2.it2_anfis import IT2ANFISRegressor
>>> import numpy as np
>>> X = np.random.randn(200, 3)
>>> y = X @ [1, -0.5, 2] + 0.1 * np.random.randn(200)
>>> model = IT2ANFISRegressor(n_rules=5, n_mfs=3, n_epochs=50, lr=0.01)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch() -> None:
    """Raise ImportError if PyTorch is not available."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for IT2ANFISRegressor. "
            "Install it with: pip install torch"
        )


if HAS_TORCH:

    class _IT2MFLayer(nn.Module):
        """Layer 1: IT2 Gaussian membership functions.

        Each MF has a center and two sigma parameters (lower and upper)
        defining the Footprint of Uncertainty.

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_mfs : int
            Number of MFs per feature.
        """

        def __init__(self, n_features: int, n_mfs: int):
            super().__init__()
            self.n_features = n_features
            self.n_mfs = n_mfs

            # Learnable parameters
            self.centers = nn.Parameter(torch.zeros(n_features, n_mfs))
            self.sigma_lower = nn.Parameter(torch.ones(n_features, n_mfs))
            self.sigma_upper = nn.Parameter(torch.ones(n_features, n_mfs))

        def initialize(self, X: np.ndarray, fou_factor: float = 0.3) -> None:
            """Initialize parameters from data statistics.

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
            fou_factor : float
                Controls FOU width.
            """
            with torch.no_grad():
                for j in range(self.n_features):
                    x_min = float(np.min(X[:, j]))
                    x_max = float(np.max(X[:, j]))
                    padding = (x_max - x_min) * 0.1 if x_max > x_min else 1.0
                    centers = np.linspace(
                        x_min - padding, x_max + padding, self.n_mfs
                    )
                    step = (x_max + padding - x_min + padding) / max(
                        self.n_mfs - 1, 1
                    )
                    sigma_base = max(step / 2.0, 1e-4)

                    self.centers.data[j] = torch.tensor(
                        centers, dtype=torch.float32
                    )
                    self.sigma_lower.data[j] = torch.full(
                        (self.n_mfs,),
                        sigma_base * (1.0 - fou_factor * 0.5),
                    )
                    self.sigma_upper.data[j] = torch.full(
                        (self.n_mfs,),
                        sigma_base * (1.0 + fou_factor * 0.5),
                    )

        def forward(
            self, x: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            """Compute upper and lower membership values.

            Parameters
            ----------
            x : Tensor of shape (batch, n_features)

            Returns
            -------
            tuple of Tensor
                upper_mu: (batch, n_features, n_mfs)
                lower_mu: (batch, n_features, n_mfs)
            """
            # Ensure positive sigmas
            sigma_lo = torch.abs(self.sigma_lower) + 1e-6
            sigma_up = torch.abs(self.sigma_upper) + 1e-6

            # x: (batch, n_features) -> (batch, n_features, 1)
            x_expanded = x.unsqueeze(-1)
            # centers: (n_features, n_mfs) -> (1, n_features, n_mfs)
            c = self.centers.unsqueeze(0)

            diff = x_expanded - c  # (batch, n_features, n_mfs)

            upper_mu = torch.exp(
                -0.5 * (diff / sigma_up.unsqueeze(0)) ** 2
            )
            lower_mu = torch.exp(
                -0.5 * (diff / sigma_lo.unsqueeze(0)) ** 2
            )

            return upper_mu, lower_mu

    class _IT2ANFISNetwork(nn.Module):
        """5-layer IT2 ANFIS network.

        Layer 1: IT2 MFs -> upper/lower membership values
        Layer 2: Product -> dual firing strengths (upper/lower)
        Layer 3: Normalization of firing strengths
        Layer 4: Consequent computation (linear TSK)
        Layer 5: Type reduction and defuzzification

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_rules : int
            Number of fuzzy rules.
        n_mfs : int
            Number of MFs per feature.
        """

        def __init__(self, n_features: int, n_rules: int, n_mfs: int):
            super().__init__()
            self.n_features = n_features
            self.n_rules = n_rules
            self.n_mfs = n_mfs

            # Layer 1: IT2 MFs
            self.mf_layer = _IT2MFLayer(n_features, n_mfs)

            # Rule-to-MF mapping: which MF each rule uses for each feature
            # (n_rules, n_features) index into n_mfs
            self.register_buffer(
                "rule_mf_indices",
                torch.zeros(n_rules, n_features, dtype=torch.long),
            )

            # Layer 4: Consequent parameters (linear TSK)
            # For each rule: [a1, ..., an, b] -> n_features + 1 params
            self.consequent_params = nn.Parameter(
                torch.zeros(n_rules, n_features + 1)
            )

        def initialize(
            self,
            X: np.ndarray,
            y: np.ndarray,
            fou_factor: float = 0.3,
            random_state: int | None = None,
        ) -> None:
            """Initialize all layers from data.

            Parameters
            ----------
            X : ndarray of shape (n_samples, n_features)
            y : ndarray of shape (n_samples,)
            fou_factor : float
            random_state : int or None
            """
            rng = np.random.RandomState(random_state)

            # Initialize MF parameters
            self.mf_layer.initialize(X, fou_factor)

            # Assign random MF indices to rules
            indices = rng.randint(0, self.n_mfs, size=(self.n_rules, self.n_features))
            self.rule_mf_indices.copy_(torch.tensor(indices, dtype=torch.long))

            # Initialize consequent params near zero with small bias toward mean
            with torch.no_grad():
                self.consequent_params.data.zero_()
                self.consequent_params.data[:, -1] = float(np.mean(y))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through all 5 layers.

            Parameters
            ----------
            x : Tensor of shape (batch, n_features)

            Returns
            -------
            Tensor of shape (batch,)
                Predicted values.
            """
            batch_size = x.shape[0]

            # Layer 1: Compute IT2 memberships
            upper_mu, lower_mu = self.mf_layer(x)
            # upper_mu, lower_mu: (batch, n_features, n_mfs)

            # Layer 2: Compute dual firing strengths via product
            # For each rule, multiply the selected MF values across features
            firing_upper = torch.ones(batch_size, self.n_rules, device=x.device)
            firing_lower = torch.ones(batch_size, self.n_rules, device=x.device)

            for j in range(self.n_features):
                mf_indices = self.rule_mf_indices[:, j]  # (n_rules,)
                # Gather the MF values for each rule
                # upper_mu[:, j, :] is (batch, n_mfs)
                u_j = upper_mu[:, j, :]  # (batch, n_mfs)
                l_j = lower_mu[:, j, :]  # (batch, n_mfs)

                # Index into the correct MF for each rule
                idx = mf_indices.unsqueeze(0).expand(batch_size, -1)  # (batch, n_rules)
                u_selected = torch.gather(u_j, 1, idx)  # (batch, n_rules)
                l_selected = torch.gather(l_j, 1, idx)  # (batch, n_rules)

                firing_upper = firing_upper * u_selected
                firing_lower = firing_lower * l_selected

            # Layer 3: Normalize firing strengths
            sum_upper = firing_upper.sum(dim=1, keepdim=True) + 1e-12
            sum_lower = firing_lower.sum(dim=1, keepdim=True) + 1e-12
            norm_upper = firing_upper / sum_upper
            norm_lower = firing_lower / sum_lower

            # Layer 4: Compute consequent outputs
            # x_ext: (batch, n_features + 1)
            x_ext = torch.cat(
                [x, torch.ones(batch_size, 1, device=x.device)], dim=1
            )
            # consequent_outputs: (batch, n_rules)
            consequent_outputs = x_ext @ self.consequent_params.T

            # Layer 5: Type reduction (Nie-Tan approximation for differentiability)
            # Use average of normalized upper and lower weighted outputs
            output_upper = (norm_upper * consequent_outputs).sum(dim=1)
            output_lower = (norm_lower * consequent_outputs).sum(dim=1)
            output = (output_upper + output_lower) / 2.0

            return output


class IT2ANFISRegressor(BaseEstimator, RegressorMixin):
    """ANFIS with Interval Type-2 Membership Functions.

    Implements the 5-layer ANFIS architecture with IT2 Gaussian MFs
    in Layer 1, producing dual (upper/lower) firing strengths.

    Uses hybrid learning:
    - Backpropagation for MF parameters (centers, sigmas)
    - LSE-integrated through the gradient (consequent params are also
      learned via backprop for simplicity and differentiability)

    Type reduction in Layer 5 uses Nie-Tan (differentiable) during
    training for gradient flow.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_epochs : int, default=100
        Number of training epochs.
    lr : float, default=0.01
        Learning rate for Adam optimizer.
    batch_size : int, default=32
        Mini-batch size for training.
    fou_factor : float, default=0.3
        Footprint of Uncertainty size.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    network_ : _IT2ANFISNetwork
        Trained PyTorch network.
    train_losses_ : list of float
        Training loss per epoch.

    Examples
    --------
    >>> from endgame.fuzzy.type2.it2_anfis import IT2ANFISRegressor
    >>> import numpy as np
    >>> X = np.random.randn(200, 3)
    >>> y = X @ [1, -0.5, 2] + 0.1 * np.random.randn(200)
    >>> model = IT2ANFISRegressor(n_rules=5, n_mfs=3, n_epochs=50)
    >>> model.fit(X, y)
    IT2ANFISRegressor(n_epochs=50, n_mfs=3, n_rules=5)
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        n_rules: int = 10,
        n_mfs: int = 3,
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        fou_factor: float = 0.3,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.fou_factor = fou_factor
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> IT2ANFISRegressor:
        """Fit the IT2 ANFIS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        _check_torch()
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Build network
        self.network_ = _IT2ANFISNetwork(
            n_features=self.n_features_in_,
            n_rules=self.n_rules,
            n_mfs=self.n_mfs,
        )
        self.network_.initialize(
            X, y, fou_factor=self.fou_factor, random_state=self.random_state
        )

        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Optimizer
        optimizer = torch.optim.Adam(self.network_.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        # Training loop
        self.train_losses_ = []
        self.network_.train()

        for epoch in range(self.n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = perm[start:end]

                X_batch = X_t[idx]
                y_batch = y_t[idx]

                optimizer.zero_grad()
                pred = self.network_(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            self.train_losses_.append(avg_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.6f}")

        self.network_.eval()
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.

        Raises
        ------
        ImportError
            If PyTorch is not installed.
        """
        _check_torch()
        check_is_fitted(self, ["network_"])
        X = check_array(X, dtype=np.float64)

        X_t = torch.tensor(X, dtype=torch.float32)
        self.network_.eval()
        with torch.no_grad():
            predictions = self.network_(X_t).numpy()

        return predictions.astype(np.float64)
