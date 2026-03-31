"""Mini-Batch Gradient Descent with Regularization, DropRule, and AdaBound.

Training utility for TSK-based fuzzy systems with modern optimization
techniques including rule dropout and adaptive learning rates.

References
----------
Wu, D., et al. (2020). Optimize TSK Fuzzy Systems for Regression Problems:
Mini-Batch Gradient Descent With Regularization, DropRule, and AdaBound
(MBGD-RDA). IEEE Transactions on Fuzzy Systems, 28(5), 1003-1015.

Example
-------
>>> from endgame.fuzzy.modern.mbgd_rda import MBGDRDARegressor
>>> reg = MBGDRDARegressor(n_rules=10, n_epochs=100)
>>> reg.fit(X_train, y_train)
>>> predictions = reg.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class MBGDRDATrainer:
    """Mini-Batch Gradient Descent with Regularization, DropRule, and AdaBound.

    Training utility for any TSK-based system with ``centers_``,
    ``sigmas_``, and ``consequent_params_`` attributes.

    Parameters
    ----------
    lr : float, default=0.01
        Initial learning rate.
    droprule_rate : float, default=0.5
        Fraction of rules to drop during each training step (like dropout).
    l2_reg : float, default=0.01
        L2 regularization strength for consequent parameters.
    batch_size : int, default=32
        Mini-batch size.
    n_epochs : int, default=100
        Number of training epochs.
    lr_bound_final : float, default=0.1
        Final learning rate bound for AdaBound.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Examples
    --------
    >>> trainer = MBGDRDATrainer(lr=0.01, droprule_rate=0.3, n_epochs=50)
    >>> # model must have centers_, sigmas_, consequent_params_ attributes
    >>> trainer.train(model, X_train, y_train)
    """

    def __init__(
        self,
        lr: float = 0.01,
        droprule_rate: float = 0.5,
        l2_reg: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        lr_bound_final: float = 0.1,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.lr = lr
        self.droprule_rate = droprule_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.lr_bound_final = lr_bound_final
        self.random_state = random_state
        self.verbose = verbose

    def train(self, model: Any, X: np.ndarray, y: np.ndarray) -> list[float]:
        """Train a TSK model in-place using MBGD-RDA.

        The model must have the following attributes:
        - ``centers_``: ndarray of shape (n_rules, n_features)
        - ``sigmas_``: ndarray of shape (n_rules, n_features)
        - ``consequent_params_``: ndarray of shape (n_rules, n_features + 1)

        Parameters
        ----------
        model : object
            TSK model with centers_, sigmas_, consequent_params_.
        X : ndarray of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        list of float
            Loss history (one value per epoch).
        """
        rng = np.random.RandomState(self.random_state)
        n_samples = X.shape[0]
        n_rules = model.centers_.shape[0]

        # AdaBound state: first and second moment estimates
        m_centers = np.zeros_like(model.centers_)
        v_centers = np.zeros_like(model.centers_)
        m_sigmas = np.zeros_like(model.sigmas_)
        v_sigmas = np.zeros_like(model.sigmas_)
        m_conseq = np.zeros_like(model.consequent_params_)
        v_conseq = np.zeros_like(model.consequent_params_)

        beta1, beta2, eps = 0.9, 0.999, 1e-8
        loss_history = []

        for epoch in range(self.n_epochs):
            # Shuffle data
            perm = rng.permutation(n_samples)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                idx = perm[start:end]
                X_batch = X[idx]
                y_batch = y[idx]

                # DropRule: randomly mask rules
                if self.droprule_rate > 0 and n_rules > 1:
                    keep_mask = rng.rand(n_rules) > self.droprule_rate
                    # Ensure at least one rule is kept
                    if not keep_mask.any():
                        keep_mask[rng.randint(n_rules)] = True
                else:
                    keep_mask = np.ones(n_rules, dtype=bool)

                # Forward pass with masked rules
                firing = self._compute_firing(X_batch, model.centers_, model.sigmas_)
                firing[:, ~keep_mask] = 0.0

                fire_sum = firing.sum(axis=1, keepdims=True)
                fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
                normalized = firing / fire_sum

                X_aug = np.column_stack([X_batch, np.ones(len(X_batch))])
                consequents = np.zeros((len(X_batch), n_rules))
                for r in range(n_rules):
                    if keep_mask[r]:
                        consequents[:, r] = X_aug @ model.consequent_params_[r]

                output = np.sum(normalized * consequents, axis=1)
                error = output - y_batch
                batch_loss = float(np.mean(error ** 2))

                # L2 regularization on consequent params
                reg_loss = self.l2_reg * float(np.sum(model.consequent_params_ ** 2))
                batch_loss += reg_loss

                epoch_loss += batch_loss
                n_batches += 1

                # Compute gradients
                grad_centers, grad_sigmas, grad_conseq = self._compute_gradients(
                    X_batch, y_batch, error, firing, normalized, consequents,
                    model.centers_, model.sigmas_, model.consequent_params_,
                    keep_mask,
                )

                # Add L2 regularization gradient
                grad_conseq += 2.0 * self.l2_reg * model.consequent_params_

                # AdaBound update
                t = epoch * max(n_samples // self.batch_size, 1) + n_batches
                lr_t = self._adabound_lr(t)

                model.centers_ = self._adabound_step(
                    model.centers_, grad_centers,
                    m_centers, v_centers, t, lr_t, beta1, beta2, eps,
                )
                model.sigmas_ = self._adabound_step(
                    model.sigmas_, grad_sigmas,
                    m_sigmas, v_sigmas, t, lr_t, beta1, beta2, eps,
                )
                model.consequent_params_ = self._adabound_step(
                    model.consequent_params_, grad_conseq,
                    m_conseq, v_conseq, t, lr_t, beta1, beta2, eps,
                )

                # Ensure sigmas stay positive
                model.sigmas_ = np.maximum(model.sigmas_, 1e-4)

            avg_loss = epoch_loss / max(n_batches, 1)
            loss_history.append(avg_loss)

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[MBGD-RDA] Epoch {epoch+1}/{self.n_epochs}, "
                    f"Loss: {avg_loss:.6f}"
                )

        return loss_history

    @staticmethod
    def _compute_firing(
        X: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
    ) -> np.ndarray:
        """Compute Gaussian firing strengths."""
        n_rules = centers.shape[0]
        firing = np.zeros((X.shape[0], n_rules))
        for r in range(n_rules):
            diff = X - centers[r]
            sigma_sq = sigmas[r] ** 2 + 1e-10
            firing[:, r] = np.exp(-0.5 * np.sum(diff ** 2 / sigma_sq, axis=1))
        return firing

    @staticmethod
    def _compute_gradients(
        X: np.ndarray,
        y: np.ndarray,
        error: np.ndarray,
        firing: np.ndarray,
        normalized: np.ndarray,
        consequents: np.ndarray,
        centers: np.ndarray,
        sigmas: np.ndarray,
        conseq_params: np.ndarray,
        keep_mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients for centers, sigmas, and consequent params."""
        n_rules = centers.shape[0]
        n_samples = X.shape[0]
        X_aug = np.column_stack([X, np.ones(n_samples)])

        grad_centers = np.zeros_like(centers)
        grad_sigmas = np.zeros_like(sigmas)
        grad_conseq = np.zeros_like(conseq_params)

        fire_sum = firing.sum(axis=1) + 1e-10
        total_output = np.sum(normalized * consequents, axis=1)

        for r in range(n_rules):
            if not keep_mask[r]:
                continue

            diff = X - centers[r]
            sigma_sq = sigmas[r] ** 2 + 1e-10

            # Gradient for consequent params
            grad_conseq[r] = (2.0 / n_samples) * (
                (error * normalized[:, r])[:, None] * X_aug
            ).sum(axis=0)

            # Gradient of normalized firing w.r.t. firing strength
            d_norm = (consequents[:, r] - total_output) / fire_sum

            # d_firing / d_center
            d_fire_center = firing[:, r:r+1] * diff / sigma_sq
            grad_centers[r] = (2.0 / n_samples) * (
                (error * d_norm)[:, None] * d_fire_center
            ).sum(axis=0)

            # d_firing / d_sigma
            d_fire_sigma = firing[:, r:r+1] * diff ** 2 / (sigmas[r] ** 3 + 1e-10)
            grad_sigmas[r] = (2.0 / n_samples) * (
                (error * d_norm)[:, None] * d_fire_sigma
            ).sum(axis=0)

        return grad_centers, grad_sigmas, grad_conseq

    def _adabound_lr(self, t: int) -> float:
        """Compute AdaBound learning rate for step t."""
        # Dynamic bounds that converge to final_lr
        final_lr = self.lr_bound_final
        gamma = 1.0 - 1.0 / (1.0 + t * 0.001)
        lb = final_lr * (1.0 - 1.0 / (gamma * t + 1.0 + 1e-10))
        ub = final_lr * (1.0 + 1.0 / (gamma * t + 1e-10))
        return float(np.clip(self.lr, lb, ub))

    @staticmethod
    def _adabound_step(
        param: np.ndarray,
        grad: np.ndarray,
        m: np.ndarray,
        v: np.ndarray,
        t: int,
        lr: float,
        beta1: float,
        beta2: float,
        eps: float,
    ) -> np.ndarray:
        """One AdaBound optimization step (in-place moment update)."""
        m[:] = beta1 * m + (1.0 - beta1) * grad
        v[:] = beta2 * v + (1.0 - beta2) * grad ** 2

        m_hat = m / (1.0 - beta1 ** max(t, 1))
        v_hat = v / (1.0 - beta2 ** max(t, 1))

        step = lr * m_hat / (np.sqrt(v_hat) + eps)
        return param - np.clip(step, -1.0, 1.0)


class MBGDRDARegressor(BaseEstimator, RegressorMixin):
    """TSK fuzzy regressor trained with MBGD-RDA.

    A standalone first-order TSK system with Gaussian antecedents
    and linear consequents, trained using mini-batch gradient descent
    with DropRule regularization and AdaBound optimizer.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    lr : float, default=0.01
        Initial learning rate.
    droprule_rate : float, default=0.5
        Fraction of rules to randomly drop during training.
    l2_reg : float, default=0.01
        L2 regularization strength.
    batch_size : int, default=32
        Mini-batch size.
    n_epochs : int, default=100
        Number of training epochs.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Rule antecedent centers.
    sigmas_ : ndarray of shape (n_rules, n_features)
        Rule antecedent widths.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        Rule consequent parameters (weights + bias).
    n_features_in_ : int
        Number of features seen during fit.
    loss_history_ : list of float
        Training loss per epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.mbgd_rda import MBGDRDARegressor
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(200) * 0.1
    >>> reg = MBGDRDARegressor(n_rules=5, n_epochs=50)
    >>> reg.fit(X, y)
    MBGDRDARegressor(n_epochs=50, n_rules=5)
    >>> preds = reg.predict(X[:5])
    """

    def __init__(
        self,
        n_rules: int = 10,
        lr: float = 0.01,
        droprule_rate: float = 0.5,
        l2_reg: float = 0.01,
        batch_size: int = 32,
        n_epochs: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.lr = lr
        self.droprule_rate = droprule_rate
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> MBGDRDARegressor:
        """Fit the TSK model using MBGD-RDA training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        n_rules = min(self.n_rules, n_samples)

        # Initialize parameters
        indices = rng.choice(n_samples, size=n_rules, replace=False)
        self.centers_ = X[indices].copy()

        stds = np.std(X, axis=0)
        stds = np.where(stds < 1e-6, 1.0, stds)
        self.sigmas_ = np.tile(stds, (n_rules, 1))

        self.consequent_params_ = rng.randn(n_rules, n_features + 1) * 0.01

        # Train with MBGD-RDA
        trainer = MBGDRDATrainer(
            lr=self.lr,
            droprule_rate=self.droprule_rate,
            l2_reg=self.l2_reg,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.loss_history_ = trainer.train(self, X, y)
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
        """
        check_is_fitted(self, ["centers_", "sigmas_", "consequent_params_"])
        X = check_array(X)

        firing = MBGDRDATrainer._compute_firing(X, self.centers_, self.sigmas_)
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum

        X_aug = np.column_stack([X, np.ones(len(X))])
        n_rules = self.centers_.shape[0]
        consequents = np.zeros((len(X), n_rules))
        for r in range(n_rules):
            consequents[:, r] = X_aug @ self.consequent_params_[r]

        return np.sum(normalized * consequents, axis=1)
