"""Deep learning survival models.

Provides PyTorch-based neural network models for survival analysis:
DeepSurv, DeepHit, LogisticHazard, MTLR, and CoxTime.

All models require PyTorch. Import will succeed even without PyTorch,
but fitting will raise ImportError.

Example
-------
>>> from endgame.survival.neural import DeepSurvRegressor
>>> model = DeepSurvRegressor(hidden_dims=(128, 64), n_epochs=50)
>>> model.fit(X_train, y_train)  # doctest: +SKIP
>>> risk = model.predict(X_test)  # doctest: +SKIP
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from endgame.survival.base import (
    BaseSurvivalEstimator,
    _check_survival_y,
    _get_time_event,
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _check_torch():
    if not HAS_TORCH:
        raise ImportError(
            "Neural survival models require PyTorch. "
            "Install with: pip install torch"
        )


def _get_device(device: str = "auto") -> Any:
    """Resolve device string to torch.device."""
    _check_torch()
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# ---------------------------------------------------------------------------
# PyTorch modules
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class _DeepSurvModule(nn.Module):
        """MLP for Cox partial hazard."""

        def __init__(
            self,
            n_features: int,
            hidden_dims: tuple[int, ...] = (256, 128),
            dropout: float = 0.1,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = n_features
            for h in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_dim, h),
                        nn.ReLU(),
                        nn.BatchNorm1d(h),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x).squeeze(-1)

    class _DiscreteTimeModule(nn.Module):
        """MLP outputting probabilities over discrete time bins."""

        def __init__(
            self,
            n_features: int,
            n_bins: int,
            hidden_dims: tuple[int, ...] = (256, 128),
            dropout: float = 0.1,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = n_features
            for h in hidden_dims:
                layers.extend(
                    [
                        nn.Linear(in_dim, h),
                        nn.ReLU(),
                        nn.BatchNorm1d(h),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = h
            layers.append(nn.Linear(in_dim, n_bins))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    def _cox_ph_loss(
        risk_pred: torch.Tensor,
        time: torch.Tensor,
        event: torch.Tensor,
    ) -> torch.Tensor:
        """Negative Cox partial log-likelihood loss.

        Parameters
        ----------
        risk_pred : Tensor (n,)
            Predicted log-hazard ratios.
        time : Tensor (n,)
            Observed times.
        event : Tensor (n,)
            Event indicators (1 = event, 0 = censored).
        """
        # Sort by time descending
        order = torch.argsort(time, descending=True)
        risk_pred = risk_pred[order]
        event = event[order]

        # Log-sum-exp of cumulative risk (from longest to shortest time)
        log_cumsum_exp = torch.logcumsumexp(risk_pred, dim=0)

        # Reverse back to original order for events
        # Actually we compute on sorted data:
        # Loss = -sum_{events} [h_i - log(sum_{j in R_i} exp(h_j))]
        loss = -torch.sum((risk_pred - log_cumsum_exp) * event)

        n_events = event.sum()
        if n_events > 0:
            loss = loss / n_events
        return loss


# ---------------------------------------------------------------------------
# DeepSurv
# ---------------------------------------------------------------------------


class DeepSurvRegressor(BaseSurvivalEstimator):
    """DeepSurv: Neural Cox Proportional Hazards (Katzman et al., 2018).

    A multi-layer perceptron that outputs log-hazard ratios, trained
    with the Cox partial likelihood loss.

    Parameters
    ----------
    hidden_dims : tuple of int, default=(256, 128)
        Hidden layer dimensions.
    dropout : float, default=0.1
        Dropout rate.
    lr : float, default=1e-3
        Learning rate.
    n_epochs : int, default=100
        Maximum training epochs.
    batch_size : int, default=256
        Mini-batch size.
    early_stopping_rounds : int, default=10
        Stop if validation loss doesn't improve for this many epochs.
    val_size : float, default=0.2
        Fraction of training data for validation (early stopping).
    device : str, default="auto"
        Device ("auto", "cpu", "cuda").
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    model_ : nn.Module
        Trained PyTorch module.
    train_losses_ : list of float
        Training loss per epoch.
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        early_stopping_rounds: int = 10,
        val_size: float = 0.2,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping_rounds = early_stopping_rounds
        self.val_size = val_size
        self.device = device

    def fit(self, X: Any, y: Any, **fit_params) -> DeepSurvRegressor:
        """Fit DeepSurv model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : structured survival array
        """
        _check_torch()
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        # Scale features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Train/val split
        rs = self.random_state
        X_tr, X_val, t_tr, t_val, e_tr, e_val = train_test_split(
            X_scaled, time, event, test_size=self.val_size, random_state=rs,
            stratify=event,
        )

        device = _get_device(self.device)
        self.model_ = _DeepSurvModule(
            X.shape[1], self.hidden_dims, self.dropout
        ).to(device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        # Convert to tensors
        X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
        t_tr_t = torch.tensor(t_tr, dtype=torch.float32, device=device)
        e_tr_t = torch.tensor(e_tr, dtype=torch.float32, device=device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
        t_val_t = torch.tensor(t_val, dtype=torch.float32, device=device)
        e_val_t = torch.tensor(e_val, dtype=torch.float32, device=device)

        self.train_losses_ = []
        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        for epoch in range(self.n_epochs):
            self.model_.train()
            # Full batch for Cox loss (needs risk set)
            optimizer.zero_grad()
            risk_pred = self.model_(X_tr_t)
            loss = _cox_ph_loss(risk_pred, t_tr_t, e_tr_t)
            loss.backward()
            optimizer.step()

            self.train_losses_.append(loss.item())

            # Validation
            self.model_.eval()
            with torch.no_grad():
                val_risk = self.model_(X_val_t)
                val_loss = _cox_ph_loss(val_risk, t_val_t, e_val_t).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {
                    k: v.clone() for k, v in self.model_.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= self.early_stopping_rounds:
                self._log(f"Early stopping at epoch {epoch + 1}")
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # Compute Breslow baseline
        self.model_.eval()
        with torch.no_grad():
            all_X = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            all_risk = self.model_(all_X).cpu().numpy()

        self._baseline_cumhaz, self._baseline_times = _breslow_estimator(
            time, event, all_risk
        )

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict risk scores (higher = more risk)."""
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            risk = self.model_(X_t).cpu().numpy()
        return risk

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict S(t|X) via Breslow baseline."""
        self._check_is_fitted()
        risk = self.predict(X)
        if times is None:
            times = self._baseline_times

        # S(t|X) = S_0(t)^exp(risk)
        # H_0(t) is baseline cumulative hazard
        S0 = np.exp(-self._baseline_cumhaz)

        # Interpolate baseline at requested times
        S0_at_times = np.interp(times, self._baseline_times, S0)

        n = len(risk)
        surv = np.zeros((n, len(times)))
        for i in range(n):
            surv[i] = S0_at_times ** np.exp(risk[i])

        return surv

    def predict_median_survival_time(self, X: Any) -> np.ndarray:
        """Predict median survival time."""
        self._check_is_fitted()
        surv = self.predict_survival_function(X)
        times = self._baseline_times
        n = surv.shape[0]
        medians = np.full(n, np.inf)
        for i in range(n):
            idx = np.searchsorted(-surv[i], -0.5)
            if idx < len(times):
                medians[i] = times[idx]
        return medians


# ---------------------------------------------------------------------------
# DeepHit
# ---------------------------------------------------------------------------


class DeepHitRegressor(BaseSurvivalEstimator):
    """DeepHit: discrete-time survival model (Lee et al., 2018).

    Discretizes time into bins and outputs a probability mass function
    over time bins.

    Parameters
    ----------
    n_bins : int, default=100
        Number of discrete time bins.
    alpha : float, default=0.5
        Weight balancing log-likelihood vs ranking loss.
    hidden_dims : tuple of int, default=(256, 128)
    dropout : float, default=0.1
    lr : float, default=1e-3
    n_epochs : int, default=100
    batch_size : int, default=256
    device : str, default="auto"
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_bins: int = 100,
        alpha: float = 0.5,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_bins = n_bins
        self.alpha = alpha
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: Any, y: Any, **fit_params) -> DeepHitRegressor:
        _check_torch()
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Create time bins
        self._bin_edges = np.linspace(0, time.max() * 1.01, self.n_bins + 1)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2

        # Discretize times
        bin_idx = np.digitize(time, self._bin_edges[1:])
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        device = _get_device(self.device)
        self.model_ = _DiscreteTimeModule(
            X.shape[1], self.n_bins, self.hidden_dims, self.dropout
        ).to(device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        bin_t = torch.tensor(bin_idx, dtype=torch.long, device=device)
        event_t = torch.tensor(event, dtype=torch.float32, device=device)

        self.train_losses_ = []

        for epoch in range(self.n_epochs):
            self.model_.train()
            optimizer.zero_grad()

            logits = self.model_(X_t)
            # Softmax to get PMF
            pmf = torch.softmax(logits, dim=-1)

            # Log-likelihood: log(f(t_i)) for events, log(S(t_i)) for censored
            # f(t_i) = pmf[bin_idx_i], S(t_i) = sum_{k > bin_idx_i} pmf[k]
            log_pmf = torch.log(pmf + 1e-10)

            # Event: log f(t_i)
            event_ll = torch.sum(
                log_pmf[torch.arange(len(bin_t)), bin_t] * event_t
            )

            # Censored: log S(t_i) = log(sum_{k >= bin_i} pmf[k])
            # Create mask for k >= bin_idx
            censored_mask = ~event.astype(bool)
            if censored_mask.any():
                surv_probs = torch.zeros(len(bin_t), device=device)
                for i in range(len(bin_t)):
                    if not event[i]:
                        surv_probs[i] = pmf[i, bin_t[i] :].sum()
                censored_t = torch.tensor(
                    censored_mask, dtype=torch.float32, device=device
                )
                censored_ll = torch.sum(
                    torch.log(surv_probs + 1e-10) * censored_t
                )
            else:
                censored_ll = torch.tensor(0.0, device=device)

            loss = -(event_ll + censored_ll) / len(bin_t)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict risk scores (expected time, negated for C-index)."""
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            logits = self.model_(X_t)
            pmf = torch.softmax(logits, dim=-1).cpu().numpy()

        # Expected time = sum(pmf * bin_centers)
        expected_time = pmf @ self._bin_centers
        # Negate so higher = more risk
        return -expected_time

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None,
    ) -> np.ndarray:
        """Predict S(t|X)."""
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            logits = self.model_(X_t)
            pmf = torch.softmax(logits, dim=-1).cpu().numpy()

        # S(t) = 1 - CDF(t) = sum_{k: bin_center_k > t} pmf[k]
        if times is None:
            times = self._bin_centers

        n_samples = pmf.shape[0]
        surv = np.zeros((n_samples, len(times)))
        cdf = np.cumsum(pmf, axis=1)

        for j, t in enumerate(times):
            bin_idx = np.searchsorted(self._bin_edges[1:], t)
            bin_idx = min(bin_idx, self.n_bins - 1)
            surv[:, j] = 1.0 - cdf[:, bin_idx]

        return surv


# ---------------------------------------------------------------------------
# Logistic Hazard
# ---------------------------------------------------------------------------


class LogisticHazardRegressor(BaseSurvivalEstimator):
    """Discrete-time logistic hazard model (Kvamme & Borgan, 2019).

    Models hazard at each discrete time bin as a logistic function.
    h(t_k|X) = sigmoid(MLP_k(X)).
    S(t|X) = prod_{k: t_k <= t} (1 - h(t_k|X)).

    Parameters
    ----------
    n_bins : int, default=100
    hidden_dims : tuple of int, default=(256, 128)
    dropout : float, default=0.1
    lr : float, default=1e-3
    n_epochs : int, default=100
    batch_size : int, default=256
    device : str, default="auto"
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_bins: int = 100,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_bins = n_bins
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: Any, y: Any, **fit_params) -> LogisticHazardRegressor:
        _check_torch()
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._bin_edges = np.linspace(0, time.max() * 1.01, self.n_bins + 1)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        bin_idx = np.digitize(time, self._bin_edges[1:])
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        # Create targets: for each sample, binary label at each time bin
        # y_k = 1 if event at bin k, 0 otherwise
        # Censoring: only observe up to bin_idx
        n = len(time)
        targets = np.zeros((n, self.n_bins), dtype=np.float32)
        masks = np.zeros((n, self.n_bins), dtype=np.float32)
        for i in range(n):
            masks[i, : bin_idx[i] + 1] = 1.0
            if event[i]:
                targets[i, bin_idx[i]] = 1.0

        device = _get_device(self.device)
        self.model_ = _DiscreteTimeModule(
            X.shape[1], self.n_bins, self.hidden_dims, self.dropout
        ).to(device)
        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        targets_t = torch.tensor(targets, device=device)
        masks_t = torch.tensor(masks, device=device)

        self.train_losses_ = []
        for epoch in range(self.n_epochs):
            self.model_.train()
            optimizer.zero_grad()
            logits = self.model_(X_t)
            # Binary cross-entropy at each observed bin
            bce = nn.functional.binary_cross_entropy_with_logits(
                logits, targets_t, reduction="none"
            )
            loss = (bce * masks_t).sum() / masks_t.sum()
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Risk score: negative expected survival time."""
        surv = self.predict_survival_function(X)
        expected = np.trapz(surv, self._bin_centers, axis=1)
        return -expected

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            logits = self.model_(X_t)
            hazard = torch.sigmoid(logits).cpu().numpy()

        # S(t) = prod_{k: t_k <= t} (1 - h_k)
        survival_at_bins = np.cumprod(1.0 - hazard, axis=1)

        if times is None:
            return survival_at_bins

        # Interpolate to requested times
        n = X_scaled.shape[0]
        surv = np.zeros((n, len(times)))
        for j, t in enumerate(times):
            bin_idx = np.searchsorted(self._bin_centers, t)
            bin_idx = min(bin_idx, self.n_bins - 1)
            surv[:, j] = survival_at_bins[:, bin_idx]
        return surv


# ---------------------------------------------------------------------------
# MTLR
# ---------------------------------------------------------------------------


class MTLRRegressor(BaseSurvivalEstimator):
    """Multi-Task Logistic Regression (Yu et al., 2011).

    Linear model with monotonicity constraint on survival function.
    Each time bin has its own logistic regression.

    Parameters
    ----------
    n_bins : int, default=100
    C : float, default=1.0
        Inverse regularization strength.
    lr : float, default=1e-3
    n_epochs : int, default=100
    device : str, default="auto"
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        n_bins: int = 100,
        C: float = 1.0,
        lr: float = 1e-3,
        n_epochs: int = 100,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_bins = n_bins
        self.C = C
        self.lr = lr
        self.n_epochs = n_epochs
        self.device = device

    def fit(self, X: Any, y: Any, **fit_params) -> MTLRRegressor:
        _check_torch()
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._bin_edges = np.linspace(0, time.max() * 1.01, self.n_bins + 1)
        self._bin_centers = (self._bin_edges[:-1] + self._bin_edges[1:]) / 2
        bin_idx = np.digitize(time, self._bin_edges[1:])
        bin_idx = np.clip(bin_idx, 0, self.n_bins - 1)

        n, d = X_scaled.shape

        device = _get_device(self.device)

        # MTLR: weight matrix W (d, n_bins) and bias b (n_bins,)
        self._W = torch.randn(d, self.n_bins, device=device, requires_grad=True) * 0.01
        self._b = torch.zeros(self.n_bins, device=device, requires_grad=True)

        optimizer = optim.Adam([self._W, self._b], lr=self.lr)

        X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
        bin_t = torch.tensor(bin_idx, dtype=torch.long, device=device)
        event_t = torch.tensor(event, dtype=torch.float32, device=device)

        self.train_losses_ = []
        for epoch in range(self.n_epochs):
            optimizer.zero_grad()

            # Compute logits and cumulative softmax
            logits = X_t @ self._W + self._b  # (n, n_bins)
            # Use cumulative sum to enforce monotonicity
            cum_logits = torch.cumsum(logits, dim=1)
            pmf = torch.softmax(cum_logits, dim=1)

            # Log-likelihood
            log_pmf = torch.log(pmf + 1e-10)
            event_ll = torch.sum(
                log_pmf[torch.arange(n, device=device), bin_t] * event_t
            )
            # Censored: log S(t_i)
            surv_ll = torch.tensor(0.0, device=device)
            censored_mask = ~event
            if censored_mask.any():
                for i in range(n):
                    if not event[i]:
                        surv_ll += torch.log(pmf[i, bin_t[i]:].sum() + 1e-10)

            # Regularization
            reg = (1.0 / self.C) * torch.sum(self._W ** 2)

            loss = -(event_ll + surv_ll) / n + reg
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        surv = self.predict_survival_function(X)
        expected = np.trapz(surv, self._bin_centers, axis=1)
        return -expected

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        device = self._W.device
        with torch.no_grad():
            X_t = torch.tensor(X_scaled, dtype=torch.float32, device=device)
            logits = X_t @ self._W + self._b
            cum_logits = torch.cumsum(logits, dim=1)
            pmf = torch.softmax(cum_logits, dim=1).cpu().numpy()

        # S(t) = sum_{k > bin(t)} pmf[k]
        cdf = np.cumsum(pmf, axis=1)
        survival_at_bins = 1.0 - cdf

        if times is None:
            return survival_at_bins

        n = X_scaled.shape[0]
        surv = np.zeros((n, len(times)))
        for j, t in enumerate(times):
            bin_idx = np.searchsorted(self._bin_centers, t)
            bin_idx = min(bin_idx, self.n_bins - 1)
            surv[:, j] = survival_at_bins[:, bin_idx]
        return surv


# ---------------------------------------------------------------------------
# CoxTime
# ---------------------------------------------------------------------------


class CoxTimeRegressor(BaseSurvivalEstimator):
    """CoxTime: continuous-time Cox model with neural network.

    More flexible than DeepSurv: the network takes both features X
    and time t as input, allowing time-dependent effects.

    Parameters
    ----------
    hidden_dims : tuple of int, default=(256, 128)
    dropout : float, default=0.1
    lr : float, default=1e-3
    n_epochs : int, default=100
    batch_size : int, default=256
    device : str, default="auto"
    random_state : int or None, default=None
    verbose : bool, default=False
    """

    def __init__(
        self,
        hidden_dims: tuple[int, ...] = (256, 128),
        dropout: float = 0.1,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X: Any, y: Any, **fit_params) -> CoxTimeRegressor:
        _check_torch()
        X, y = self._validate_survival_data(X, y)
        time, event = _get_time_event(y)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._time_mean = time.mean()
        self._time_std = time.std() + 1e-8

        device = _get_device(self.device)

        # Network takes (X, t) as input → n_features + 1
        self.model_ = _DeepSurvModule(
            X.shape[1] + 1, self.hidden_dims, self.dropout
        ).to(device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.lr)

        time_scaled = (time - self._time_mean) / self._time_std
        X_aug = np.column_stack([X_scaled, time_scaled])

        X_t = torch.tensor(X_aug, dtype=torch.float32, device=device)
        t_t = torch.tensor(time, dtype=torch.float32, device=device)
        e_t = torch.tensor(event, dtype=torch.float32, device=device)

        self.train_losses_ = []
        for epoch in range(self.n_epochs):
            self.model_.train()
            optimizer.zero_grad()
            risk = self.model_(X_t)
            loss = _cox_ph_loss(risk, t_t, e_t)
            loss.backward()
            optimizer.step()
            self.train_losses_.append(loss.item())

        # Breslow baseline
        self.model_.eval()
        with torch.no_grad():
            all_risk = self.model_(X_t).cpu().numpy()
        self._baseline_cumhaz, self._baseline_times = _breslow_estimator(
            time, event, all_risk
        )

        self.is_fitted_ = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Risk scores at median time."""
        self._check_is_fitted()
        _check_torch()
        X = self._to_numpy(X)
        X_scaled = self._scaler.transform(X)
        median_t = np.median(self._baseline_times)
        t_scaled = (median_t - self._time_mean) / self._time_std
        X_aug = np.column_stack([X_scaled, np.full(len(X), t_scaled)])

        device = next(self.model_.parameters()).device
        self.model_.eval()
        with torch.no_grad():
            X_t = torch.tensor(X_aug, dtype=torch.float32, device=device)
            risk = self.model_(X_t).cpu().numpy()
        return risk

    def predict_survival_function(
        self, X: Any, times: np.ndarray | None = None,
    ) -> np.ndarray:
        self._check_is_fitted()
        risk = self.predict(X)
        if times is None:
            times = self._baseline_times

        S0 = np.exp(-self._baseline_cumhaz)
        S0_at_times = np.interp(times, self._baseline_times, S0)

        n = len(risk)
        surv = np.zeros((n, len(times)))
        for i in range(n):
            surv[i] = S0_at_times ** np.exp(risk[i])
        return surv


# ---------------------------------------------------------------------------
# Breslow estimator utility
# ---------------------------------------------------------------------------


def _breslow_estimator(
    time: np.ndarray,
    event: np.ndarray,
    risk_scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Breslow baseline cumulative hazard.

    Parameters
    ----------
    time : array (n,)
    event : array (n,) bool
    risk_scores : array (n,) log-hazard ratios

    Returns
    -------
    cumulative_hazard : array (n_events,)
    event_times : array (n_events,) sorted unique event times
    """
    exp_risk = np.exp(risk_scores)
    unique_times = np.sort(np.unique(time[event]))
    cumhaz = np.zeros(len(unique_times))

    running_sum = 0.0
    for idx, t in enumerate(unique_times):
        at_risk = time >= t
        risk_sum = exp_risk[at_risk].sum()
        n_events_at_t = ((time == t) & event).sum()
        if risk_sum > 0:
            running_sum += n_events_at_t / risk_sum
        cumhaz[idx] = running_sum

    return cumhaz, unique_times
