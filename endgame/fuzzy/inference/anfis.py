"""Adaptive Neuro-Fuzzy Inference System (ANFIS).

Implements the ANFIS architecture (Jang, 1993) with hybrid learning:
forward pass uses least-squares estimation for consequent parameters,
backward pass uses gradient descent for premise (MF) parameters.

Requires PyTorch for gradient-based optimization of premise parameters.

Example
-------
>>> from endgame.fuzzy.inference.anfis import ANFISRegressor
>>> model = ANFISRegressor(n_mfs=3, n_epochs=50, lr=0.01)
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyClassifier, BaseFuzzyRegressor
from endgame.fuzzy.core.membership import GaussianMF, create_uniform_mfs

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
            "PyTorch is required for ANFIS. "
            "Install it with: pip install torch"
        )


if HAS_TORCH:

    class _ANFISModule(nn.Module):
        """PyTorch module implementing the 5-layer ANFIS architecture.

        Layer 1: Fuzzification (parameterized Gaussian MFs)
        Layer 2: Rule firing strengths via t-norm (product)
        Layer 3: Normalized firing strengths
        Layer 4: Consequent computation (linear functions)
        Layer 5: Summation (weighted average)

        Parameters
        ----------
        n_features : int
            Number of input features.
        n_mfs : int
            Number of MFs per feature.
        n_rules : int
            Number of rules.
        order : int
            TSK order (0 or 1).
        antecedent_indices : ndarray of shape (n_rules, n_features)
            MF index per feature per rule.
        centers : list of ndarray
            Initial MF centers per feature.
        sigmas : list of ndarray
            Initial MF sigmas per feature.
        """

        def __init__(
            self,
            n_features: int,
            n_mfs: int,
            n_rules: int,
            order: int,
            antecedent_indices: np.ndarray,
            centers: list[np.ndarray],
            sigmas: list[np.ndarray],
        ):
            super().__init__()
            self.n_features = n_features
            self.n_mfs = n_mfs
            self.n_rules = n_rules
            self.order = order

            # Register antecedent indices as a buffer (non-learnable)
            self.register_buffer(
                "antecedent_indices",
                torch.tensor(antecedent_indices, dtype=torch.long),
            )

            # Layer 1: Premise parameters (learnable)
            # centers_param[j] has shape (n_mfs,)
            self.centers_param = nn.ParameterList([
                nn.Parameter(torch.tensor(c, dtype=torch.float32))
                for c in centers
            ])
            self.sigmas_param = nn.ParameterList([
                nn.Parameter(torch.tensor(s, dtype=torch.float32))
                for s in sigmas
            ])

            # Layer 4: Consequent parameters (learnable)
            if order == 0:
                self.consequent = nn.Parameter(
                    torch.zeros(n_rules, 1, dtype=torch.float32)
                )
            else:
                # n_features + 1 for bias
                self.consequent = nn.Parameter(
                    torch.zeros(n_rules, n_features + 1, dtype=torch.float32)
                )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """Forward pass through all 5 ANFIS layers.

            Parameters
            ----------
            x : Tensor of shape (batch_size, n_features)

            Returns
            -------
            Tensor of shape (batch_size,)
                Predicted output.
            """
            batch_size = x.shape[0]

            # Layer 1: Fuzzification - compute membership values
            # memberships[j][k] = Gaussian MF value for feature j, term k
            memberships = []
            for j in range(self.n_features):
                center = self.centers_param[j]  # (n_mfs,)
                sigma = torch.abs(self.sigmas_param[j]) + 1e-6  # (n_mfs,)
                # x[:, j] has shape (batch_size,)
                xj = x[:, j].unsqueeze(1)  # (batch_size, 1)
                # Gaussian: exp(-0.5 * ((x - c) / s)^2)
                mu = torch.exp(-0.5 * ((xj - center.unsqueeze(0)) / sigma.unsqueeze(0)) ** 2)
                # mu has shape (batch_size, n_mfs)
                memberships.append(mu)

            # Layer 2: Rule firing strengths (product t-norm)
            # For each rule r, multiply memberships across features
            firing = torch.ones(batch_size, self.n_rules, device=x.device)
            for j in range(self.n_features):
                mf_indices = self.antecedent_indices[:, j]  # (n_rules,)
                # Gather the right MF for each rule
                mu_j = memberships[j][:, mf_indices]  # (batch_size, n_rules)
                firing = firing * mu_j

            # Layer 3: Normalization
            firing_sum = firing.sum(dim=1, keepdim=True).clamp(min=1e-12)
            norm_firing = firing / firing_sum  # (batch_size, n_rules)

            # Layer 4: Consequent computation
            if self.order == 0:
                # rule_output = constant per rule
                rule_out = self.consequent.squeeze(-1).unsqueeze(0).expand(
                    batch_size, -1
                )  # (batch_size, n_rules)
            else:
                # rule_output = x @ w + bias for each rule
                # consequent has shape (n_rules, n_features + 1)
                weights = self.consequent[:, :-1]  # (n_rules, n_features)
                bias = self.consequent[:, -1]  # (n_rules,)
                # x @ weights^T + bias
                rule_out = x @ weights.T + bias.unsqueeze(0)  # (batch_size, n_rules)

            # Layer 5: Weighted sum
            output = (norm_firing * rule_out).sum(dim=1)  # (batch_size,)
            return output

        def get_premise_params(self) -> list[nn.Parameter]:
            """Return all premise (MF) parameters."""
            params = []
            for p in self.centers_param:
                params.append(p)
            for p in self.sigmas_param:
                params.append(p)
            return params

        def get_consequent_params(self) -> list[nn.Parameter]:
            """Return all consequent parameters."""
            return [self.consequent]


class ANFISRegressor(BaseFuzzyRegressor):
    """Adaptive Neuro-Fuzzy Inference System for regression.

    Implements Jang's ANFIS (1993) with a 5-layer architecture and
    hybrid learning algorithm that alternates between:

    - Forward pass: fix premise parameters, estimate consequent
      parameters via least squares.
    - Backward pass: fix consequent parameters, update premise
      parameters via gradient descent.

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_rules : int or None, default=None
        Number of rules. If None, uses n_mfs ** n_features (capped at 200).
    order : int, default=1
        TSK order for consequents (0=constant, 1=linear).
    mf_type : str, default='gaussian'
        Only 'gaussian' is supported for gradient-based learning.
    t_norm : str, default='product'
        T-norm for rule firing (only 'product' for differentiable ANFIS).
    n_epochs : int, default=100
        Maximum number of training epochs.
    lr : float, default=0.01
        Learning rate for premise parameter updates.
    batch_size : int, default=32
        Mini-batch size. Set to 0 or None for full-batch.
    early_stopping : int, default=10
        Number of epochs without improvement before stopping.
        Set to 0 to disable.
    optimizer : str, default='hybrid'
        Training strategy: 'hybrid' (Jang's original), 'sgd', or 'adam'.
    reg_lambda : float, default=1e-6
        Regularization for LSE in hybrid mode.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    module_ : _ANFISModule
        Fitted PyTorch module (available only with PyTorch).
    mfs_ : list of list of BaseMembershipFunction
        Extracted antecedent MFs after fitting.
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        MF assignments per rule and feature.
    consequent_params_ : ndarray of shape (n_rules, n_params)
        Consequent parameters.
    train_losses_ : list of float
        Training loss per epoch.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.anfis import ANFISRegressor
    >>> X = np.random.rand(200, 2)
    >>> y = np.sin(X[:, 0]) + np.cos(X[:, 1])
    >>> model = ANFISRegressor(n_mfs=3, n_epochs=50, lr=0.01)
    >>> model.fit(X, y)
    ANFISRegressor(...)
    >>> preds = model.predict(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        n_rules: int | None = None,
        order: int = 1,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int = 10,
        optimizer: str = "hybrid",
        reg_lambda: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=n_rules or 0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.order = order
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.reg_lambda = reg_lambda
        self._n_rules_user = n_rules

    def _build_antecedent_indices(self, n_features: int) -> np.ndarray:
        """Build antecedent index array."""
        from itertools import product as iterproduct

        if self._n_rules_user is None:
            total_grid = self.n_mfs ** n_features
            if total_grid <= 200:
                return np.array(
                    list(iterproduct(range(self.n_mfs), repeat=n_features)),
                    dtype=int,
                )
            else:
                rng = np.random.RandomState(self.random_state)
                return rng.randint(0, self.n_mfs, size=(200, n_features))
        else:
            rng = np.random.RandomState(self.random_state)
            return rng.randint(
                0, self.n_mfs, size=(self._n_rules_user, n_features)
            )

    def fit(self, X: Any, y: Any) -> ANFISRegressor:
        """Fit the ANFIS model using hybrid or gradient-based learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        _check_torch()
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features
        self.feature_names_ = [f"x{j}" for j in range(n_features)]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Initialize antecedent MFs from data
        self.mfs_ = self._init_membership_functions(X)
        self.antecedent_indices_ = self._build_antecedent_indices(n_features)
        actual_n_rules = self.antecedent_indices_.shape[0]
        self.n_rules = actual_n_rules

        # Extract initial centers and sigmas for the torch module
        centers = []
        sigmas = []
        for j in range(n_features):
            c = np.array([mf.center for mf in self.mfs_[j]])
            s = np.array([mf.sigma for mf in self.mfs_[j]])
            centers.append(c.astype(np.float32))
            sigmas.append(s.astype(np.float32))

        # Build PyTorch module
        self.module_ = _ANFISModule(
            n_features=n_features,
            n_mfs=self.n_mfs,
            n_rules=actual_n_rules,
            order=self.order,
            antecedent_indices=self.antecedent_indices_,
            centers=centers,
            sigmas=sigmas,
        )

        # Convert data to tensors
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        # Train
        if self.optimizer == "hybrid":
            self._train_hybrid(X_t, y_t)
        elif self.optimizer in ("sgd", "adam"):
            self._train_gradient(X_t, y_t)
        else:
            raise ValueError(
                f"Unknown optimizer: {self.optimizer}. "
                "Choose from 'hybrid', 'sgd', 'adam'."
            )

        # Extract fitted parameters back to numpy
        self._extract_params()
        return self

    def _train_hybrid(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Hybrid learning: LSE for consequents, GD for premises.

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
        y : Tensor of shape (n_samples,)
        """
        premise_optimizer = torch.optim.Adam(
            self.module_.get_premise_params(), lr=self.lr
        )

        n_samples = X.shape[0]
        batch_size = self.batch_size if self.batch_size and self.batch_size > 0 else n_samples
        self.train_losses_ = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            # Shuffle
            perm = torch.randperm(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                # Forward pass: compute output with current premise params
                # but optimize consequent params via LSE
                self.module_.eval()
                with torch.no_grad():
                    # Compute normalized firing strengths
                    norm_firing = self._compute_norm_firing_torch(X_batch)

                # LSE for consequent parameters
                self._update_consequents_lse(X_batch, y_batch, norm_firing)

                # Backward pass: optimize premise params via GD
                self.module_.train()
                premise_optimizer.zero_grad()
                y_pred = self.module_(X_batch)
                loss = torch.mean((y_pred - y_batch) ** 2)
                loss.backward()
                premise_optimizer.step()

                epoch_loss += loss.item() * (end - start)
                n_batches += 1

            epoch_loss /= n_samples
            self.train_losses_.append(epoch_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                self._log(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

            # Early stopping
            if self.early_stopping > 0:
                if epoch_loss < best_loss - 1e-8:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        self._log(f"Early stopping at epoch {epoch + 1}")
                        break

    def _compute_norm_firing_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Compute normalized firing strengths (no grad).

        Parameters
        ----------
        x : Tensor of shape (batch_size, n_features)

        Returns
        -------
        Tensor of shape (batch_size, n_rules)
        """
        batch_size = x.shape[0]
        module = self.module_

        memberships = []
        for j in range(module.n_features):
            center = module.centers_param[j]
            sigma = torch.abs(module.sigmas_param[j]) + 1e-6
            xj = x[:, j].unsqueeze(1)
            mu = torch.exp(-0.5 * ((xj - center.unsqueeze(0)) / sigma.unsqueeze(0)) ** 2)
            memberships.append(mu)

        firing = torch.ones(batch_size, module.n_rules, device=x.device)
        for j in range(module.n_features):
            mf_indices = module.antecedent_indices[:, j]
            mu_j = memberships[j][:, mf_indices]
            firing = firing * mu_j

        firing_sum = firing.sum(dim=1, keepdim=True).clamp(min=1e-12)
        return firing / firing_sum

    def _update_consequents_lse(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        norm_firing: torch.Tensor,
    ) -> None:
        """Update consequent parameters via weighted LSE.

        Parameters
        ----------
        X : Tensor of shape (batch_size, n_features)
        y : Tensor of shape (batch_size,)
        norm_firing : Tensor of shape (batch_size, n_rules)
        """
        module = self.module_
        n_rules = module.n_rules
        n_features = module.n_features

        nf = norm_firing.detach().cpu().numpy()
        X_np = X.detach().cpu().numpy()
        y_np = y.detach().cpu().numpy()

        if module.order == 0:
            # y = nf @ c
            A = nf
            ATA = A.T @ A + self.reg_lambda * np.eye(n_rules)
            ATy = A.T @ y_np
            try:
                theta = np.linalg.solve(ATA, ATy)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, y_np, rcond=None)[0]
            with torch.no_grad():
                module.consequent.copy_(
                    torch.tensor(theta.reshape(n_rules, 1), dtype=torch.float32)
                )
        else:
            n_params = n_features + 1
            A = np.zeros((X_np.shape[0], n_rules * n_params))
            for r in range(n_rules):
                w_r = nf[:, r]
                for j in range(n_features):
                    A[:, r * n_params + j] = w_r * X_np[:, j]
                A[:, r * n_params + n_features] = w_r

            ATA = A.T @ A + self.reg_lambda * np.eye(A.shape[1])
            ATy = A.T @ y_np
            try:
                theta = np.linalg.solve(ATA, ATy)
            except np.linalg.LinAlgError:
                theta = np.linalg.lstsq(A, y_np, rcond=None)[0]
            with torch.no_grad():
                module.consequent.copy_(
                    torch.tensor(
                        theta.reshape(n_rules, n_params), dtype=torch.float32
                    )
                )

    def _train_gradient(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """Pure gradient-based training (SGD or Adam).

        Parameters
        ----------
        X : Tensor of shape (n_samples, n_features)
        y : Tensor of shape (n_samples,)
        """
        if self.optimizer == "adam":
            opt = torch.optim.Adam(self.module_.parameters(), lr=self.lr)
        else:
            opt = torch.optim.SGD(self.module_.parameters(), lr=self.lr)

        n_samples = X.shape[0]
        batch_size = self.batch_size if self.batch_size and self.batch_size > 0 else n_samples
        self.train_losses_ = []
        best_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.n_epochs):
            perm = torch.randperm(n_samples)
            X_shuf = X[perm]
            y_shuf = y[perm]

            epoch_loss = 0.0
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)
                X_batch = X_shuf[start:end]
                y_batch = y_shuf[start:end]

                opt.zero_grad()
                y_pred = self.module_(X_batch)
                loss = torch.mean((y_pred - y_batch) ** 2)
                loss.backward()
                opt.step()

                epoch_loss += loss.item() * (end - start)

            epoch_loss /= n_samples
            self.train_losses_.append(epoch_loss)

            if self.verbose and (epoch + 1) % max(1, self.n_epochs // 10) == 0:
                self._log(f"Epoch {epoch + 1}/{self.n_epochs}, Loss: {epoch_loss:.6f}")

            if self.early_stopping > 0:
                if epoch_loss < best_loss - 1e-8:
                    best_loss = epoch_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.early_stopping:
                        self._log(f"Early stopping at epoch {epoch + 1}")
                        break

    def _extract_params(self) -> None:
        """Extract fitted parameters from the PyTorch module back to numpy."""
        module = self.module_
        module.eval()

        # Rebuild MFs with updated parameters
        self.mfs_ = []
        for j in range(module.n_features):
            centers = module.centers_param[j].detach().cpu().numpy()
            sigmas = np.abs(module.sigmas_param[j].detach().cpu().numpy()) + 1e-6
            feature_mfs = []
            for k in range(module.n_mfs):
                feature_mfs.append(
                    GaussianMF(center=float(centers[k]), sigma=float(sigmas[k]))
                )
            self.mfs_.append(feature_mfs)

        # Extract consequent parameters
        self.consequent_params_ = module.consequent.detach().cpu().numpy()

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using the fitted ANFIS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        _check_torch()
        check_is_fitted(self, ["module_", "mfs_", "antecedent_indices_"])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        self.module_.eval()
        X_t = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            predictions = self.module_(X_t).cpu().numpy()
        return predictions


class ANFISClassifier(BaseFuzzyClassifier):
    """Adaptive Neuro-Fuzzy Inference System for classification.

    Wraps ANFISRegressor in a one-vs-rest strategy with softmax
    normalization for multi-class probability estimates.

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    n_rules : int or None, default=None
        Number of rules.
    order : int, default=1
        TSK order for consequents (0=constant, 1=linear).
    mf_type : str, default='gaussian'
        Membership function type.
    t_norm : str, default='product'
        T-norm for rule firing.
    n_epochs : int, default=100
        Maximum training epochs.
    lr : float, default=0.01
        Learning rate for premise parameters.
    batch_size : int, default=32
        Mini-batch size.
    early_stopping : int, default=10
        Patience for early stopping.
    optimizer : str, default='hybrid'
        Training strategy: 'hybrid', 'sgd', or 'adam'.
    reg_lambda : float, default=1e-6
        Regularization for LSE.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    estimators_ : list of ANFISRegressor
        One regressor per class.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.anfis import ANFISClassifier
    >>> X = np.random.rand(200, 2)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> clf = ANFISClassifier(n_mfs=2, n_epochs=30)
    >>> clf.fit(X, y)
    ANFISClassifier(...)
    >>> proba = clf.predict_proba(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        n_rules: int | None = None,
        order: int = 1,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        n_epochs: int = 100,
        lr: float = 0.01,
        batch_size: int = 32,
        early_stopping: int = 10,
        optimizer: str = "hybrid",
        reg_lambda: float = 1e-6,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=n_rules or 0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.order = order
        self.n_epochs = n_epochs
        self.lr = lr
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.optimizer = optimizer
        self.reg_lambda = reg_lambda
        self._n_rules_user = n_rules

    def fit(self, X: Any, y: Any) -> ANFISClassifier:
        """Fit one ANFIS regressor per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        _check_torch()
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        y_encoded = self._encode_labels(y)

        self.estimators_ = []
        for c in range(self.n_classes_):
            y_binary = (y_encoded == c).astype(np.float64)
            reg = ANFISRegressor(
                n_mfs=self.n_mfs,
                n_rules=self._n_rules_user,
                order=self.order,
                mf_type=self.mf_type,
                t_norm=self.t_norm,
                n_epochs=self.n_epochs,
                lr=self.lr,
                batch_size=self.batch_size,
                early_stopping=self.early_stopping,
                optimizer=self.optimizer,
                reg_lambda=self.reg_lambda,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary)
            self.estimators_.append(reg)

        # Store for base class compat
        self.antecedent_indices_ = self.estimators_[0].antecedent_indices_
        self.mfs_ = self.estimators_[0].mfs_
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities via softmax over per-class outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X, dtype=np.float64)

        raw_outputs = np.column_stack([
            reg.predict(X) for reg in self.estimators_
        ])

        # Softmax
        exp_vals = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
        proba = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return proba
