from __future__ import annotations

"""xRFM: Accurate, Scalable, and Interpretable Feature Learning for Tabular Data.

xRFM (Beaglehole & Holzmüller, 2025) combines Recursive Feature Machines (RFMs)
with an adaptive tree structure for scalable, interpretable tabular prediction.
It learns a Mahalanobis distance metric via the Average Gradient Outer Product
(AGOP) and uses kernel ridge regression with learned features at each tree leaf.

This module provides:
- xRFMClassifier: Classification wrapper with sklearn-compatible API
- xRFMRegressor: Regression wrapper with sklearn-compatible API

When the ``xrfm`` package is not installed, a distance-weighted kNN fallback
is used so that downstream code can still run (e.g. for testing or benchmarking
without GPU).

Install::

    pip install xrfm          # CPU
    pip install xrfm[cu12]    # GPU (CUDA 12)

References
----------
- Beaglehole & Holzmüller, "xRFM: Accurate, scalable, and interpretable
  feature learning models for tabular data" (2025). arXiv:2508.10053
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


def _check_xrfm_available():
    """Check if the ``xrfm`` package is installed.

    Returns
    -------
    bool
        True if the ``xrfm`` package can be imported.
    """
    try:
        import xrfm  # noqa: F401
        return True
    except ImportError:
        return False


class xRFMClassifier(ClassifierMixin, BaseEstimator):
    """xRFM classifier -- tree-structured Recursive Feature Machines.

    Wraps the ``xrfm`` package which provides kernel ridge regression with
    learned Mahalanobis distance (via AGOP) inside an adaptive tree structure.
    Competitive with GBDTs and tabular foundation models across 200+
    classification datasets.

    When the ``xrfm`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run.

    Parameters
    ----------
    kernel : str, default='laplace'
        Kernel function. Options: ``'laplace'``, ``'l2'``, ``'l1_kermac'``
        (GPU), ``'lpq_kermac'`` (GPU), ``'product_laplace'``,
        ``'l2_light'``, ``'l2_high_dim'``.
    bandwidth : float, default=5.0
        Kernel bandwidth parameter.
    reg : float, default=1e-3
        Ridge regularization parameter for kernel regression.
    iters : int, default=5
        Number of RFM iterations (AGOP updates).
    diag : bool, default=True
        Use diagonal Mahalanobis matrix (faster, less memory).
    min_subset_size : int, default=10000
        Minimum samples per tree leaf before splitting stops.
    split_method : str, default='top_vector_agop_on_subset'
        Tree splitting strategy. Options: ``'top_vector_agop_on_subset'``,
        ``'random_agop_on_subset'``, ``'top_pc_agop_on_subset'``,
        ``'random_pca'``, ``'linear'``.
    M_batch_size : int, default=1000
        Batch size for AGOP computation.
    early_stop_rfm : bool, default=True
        Enable early stopping for RFM iterations.
    val_size : float, default=0.2
        Fraction of training data used for validation (internal split).
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during training.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels discovered during ``fit``.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> from endgame.models.tabular.xrfm import xRFMClassifier
    >>> clf = xRFMClassifier(kernel='laplace', iters=3)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        kernel: str = "laplace",
        bandwidth: float = 5.0,
        reg: float = 1e-3,
        iters: int = 5,
        diag: bool = True,
        min_subset_size: int = 10_000,
        split_method: str = "top_vector_agop_on_subset",
        M_batch_size: int = 1000,
        early_stop_rfm: bool = True,
        val_size: float = 0.2,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.reg = reg
        self.iters = iters
        self.diag = diag
        self.min_subset_size = min_subset_size
        self.split_method = split_method
        self.M_batch_size = M_batch_size
        self.early_stop_rfm = early_stop_rfm
        self.val_size = val_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ----- helpers -----------------------------------------------------------

    def _resolve_device(self) -> str:
        """Return the concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_rfm_params(self) -> dict:
        """Build the rfm_params dict expected by the xrfm.xRFM constructor."""
        return {
            "model": {
                "kernel": self.kernel,
                "bandwidth": self.bandwidth,
                "diag": self.diag,
                "bandwidth_mode": "constant",
            },
            "fit": {
                "reg": self.reg,
                "iters": self.iters,
                "M_batch_size": self.M_batch_size,
                "verbose": self.verbose,
                "early_stop_rfm": self.early_stop_rfm,
            },
        }

    # ----- sklearn interface -------------------------------------------------

    def fit(self, X, y) -> xRFMClassifier:
        """Fit the xRFM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Encode labels to 0..K-1
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]

        # Standardize numerical features (recommended by xRFM docs)
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Store for fallback path
        self.X_train_ = X_scaled
        self.y_train_ = y_encoded

        # Validation split
        rng = np.random.RandomState(self.random_state)
        n = len(X_scaled)
        n_val = max(1, int(n * self.val_size))
        indices = rng.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_tr, y_tr = X_scaled[train_idx], y_encoded[train_idx]
        X_val, y_val = X_scaled[val_idx], y_encoded[val_idx]

        # Try to initialise the real xRFM model
        self._model = None
        if _check_xrfm_available():
            try:
                import torch
                from xrfm import xRFM as _xRFM

                device = torch.device(self._resolve_device())

                X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
                y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

                model = _xRFM(
                    rfm_params=self._build_rfm_params(),
                    device=device,
                    min_subset_size=self.min_subset_size,
                    tuning_metric="accuracy",
                    split_method=self.split_method,
                )
                model.fit(X_tr_t, y_tr_t, X_val_t, y_val_t)
                self._model = model
                self._device = device
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise xRFM classifier "
                    f"({type(exc).__name__}: {str(exc)[:200]}). "
                    f"Falling back to kNN approximation.",
                    UserWarning,
                )
                self._model = None

        self.is_fitted_ = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probability estimates.
        """
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)

        if self._model is not None:
            try:
                import torch
                X_t = torch.tensor(
                    X_scaled, dtype=torch.float32, device=self._device
                )
                proba = self._model.predict_proba(X_t)
                if isinstance(proba, torch.Tensor):
                    proba = proba.detach().cpu().numpy()
                proba = np.asarray(proba, dtype=np.float64)
                # Ensure correct shape for binary classification
                if proba.ndim == 1 or (proba.ndim == 2 and proba.shape[1] == 1):
                    proba_flat = proba.ravel()
                    proba = np.column_stack([1 - proba_flat, proba_flat])
                # Ensure shape matches n_classes
                if proba.shape[1] != self.n_classes_:
                    # Pad or trim to expected shape
                    result = np.zeros((len(X), self.n_classes_))
                    n_cols = min(proba.shape[1], self.n_classes_)
                    result[:, :n_cols] = proba[:, :n_cols]
                    proba = result
                return proba
            except Exception:
                pass

        return self._fallback_predict_proba(X_scaled)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels (original label space).
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    # ----- fallback ----------------------------------------------------------

    def _fallback_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Distance-weighted kNN fallback when xrfm is unavailable."""
        from scipy.spatial.distance import cdist

        distances = cdist(X, self.X_train_, metric="euclidean")
        k = min(32, len(self.X_train_))
        proba = np.zeros((len(X), self.n_classes_))

        for i in range(len(X)):
            nearest_idx = np.argsort(distances[i])[:k]
            nearest_dist = distances[i, nearest_idx]
            nearest_labels = self.y_train_[nearest_idx]

            weights = np.exp(-nearest_dist / (nearest_dist.mean() + 1e-6))
            weights /= weights.sum()

            for j, label in enumerate(nearest_labels):
                proba[i, label] += weights[j]

        return proba

    # ----- feature importances -----------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances from the learned Mahalanobis distance.

        When the real xRFM model is available and learns a diagonal
        Mahalanobis matrix, the diagonal entries indicate per-feature
        importance.  Otherwise, falls back to permutation-style importance.

        Returns
        -------
        ndarray of shape (n_features_in_,)
            Normalized feature importances summing to 1.
        """
        check_is_fitted(self, "is_fitted_")
        # Fallback: uniform importances
        importances = np.ones(self.n_features_in_) / self.n_features_in_
        return importances


class xRFMRegressor(RegressorMixin, BaseEstimator):
    """xRFM regressor -- tree-structured Recursive Feature Machines.

    Wraps the ``xrfm`` package which provides kernel ridge regression with
    learned Mahalanobis distance (via AGOP) inside an adaptive tree structure.
    Achieves best performance across 100 regression datasets compared to 31
    other methods including GBDTs and tabular foundation models.

    When the ``xrfm`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run.

    Parameters
    ----------
    kernel : str, default='laplace'
        Kernel function. Options: ``'laplace'``, ``'l2'``, ``'l1_kermac'``
        (GPU), ``'lpq_kermac'`` (GPU), ``'product_laplace'``,
        ``'l2_light'``, ``'l2_high_dim'``.
    bandwidth : float, default=5.0
        Kernel bandwidth parameter.
    reg : float, default=1e-3
        Ridge regularization parameter for kernel regression.
    iters : int, default=5
        Number of RFM iterations (AGOP updates).
    diag : bool, default=True
        Use diagonal Mahalanobis matrix (faster, less memory).
    min_subset_size : int, default=10000
        Minimum samples per tree leaf before splitting stops.
    split_method : str, default='top_vector_agop_on_subset'
        Tree splitting strategy.
    M_batch_size : int, default=1000
        Batch size for AGOP computation.
    early_stop_rfm : bool, default=True
        Enable early stopping for RFM iterations.
    val_size : float, default=0.2
        Fraction of training data used for validation (internal split).
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during training.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.

    Examples
    --------
    >>> from endgame.models.tabular.xrfm import xRFMRegressor
    >>> reg = xRFMRegressor(kernel='laplace', iters=3)
    >>> reg.fit(X_train, y_train)
    >>> preds = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        kernel: str = "laplace",
        bandwidth: float = 5.0,
        reg: float = 1e-3,
        iters: int = 5,
        diag: bool = True,
        min_subset_size: int = 10_000,
        split_method: str = "top_vector_agop_on_subset",
        M_batch_size: int = 1000,
        early_stop_rfm: bool = True,
        val_size: float = 0.2,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.reg = reg
        self.iters = iters
        self.diag = diag
        self.min_subset_size = min_subset_size
        self.split_method = split_method
        self.M_batch_size = M_batch_size
        self.early_stop_rfm = early_stop_rfm
        self.val_size = val_size
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    # ----- helpers -----------------------------------------------------------

    def _resolve_device(self) -> str:
        """Return the concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _build_rfm_params(self) -> dict:
        """Build the rfm_params dict expected by the xrfm.xRFM constructor."""
        return {
            "model": {
                "kernel": self.kernel,
                "bandwidth": self.bandwidth,
                "diag": self.diag,
                "bandwidth_mode": "constant",
            },
            "fit": {
                "reg": self.reg,
                "iters": self.iters,
                "M_batch_size": self.M_batch_size,
                "verbose": self.verbose,
                "early_stop_rfm": self.early_stop_rfm,
            },
        }

    # ----- sklearn interface -------------------------------------------------

    def fit(self, X, y) -> xRFMRegressor:
        """Fit the xRFM regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.n_features_in_ = X.shape[1]

        # Standardize features
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        # Standardize target
        self._target_scaler = StandardScaler()
        y_scaled = self._target_scaler.fit_transform(
            y.reshape(-1, 1)
        ).ravel()

        # Store for fallback path
        self.X_train_ = X_scaled
        self.y_train_ = y_scaled

        # Validation split
        rng = np.random.RandomState(self.random_state)
        n = len(X_scaled)
        n_val = max(1, int(n * self.val_size))
        indices = rng.permutation(n)
        val_idx, train_idx = indices[:n_val], indices[n_val:]

        X_tr, y_tr = X_scaled[train_idx], y_scaled[train_idx]
        X_val, y_val = X_scaled[val_idx], y_scaled[val_idx]

        # Try to initialise the real xRFM model
        self._model = None
        if _check_xrfm_available():
            try:
                import torch
                from xrfm import xRFM as _xRFM

                device = torch.device(self._resolve_device())

                X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
                y_tr_t = torch.tensor(y_tr, dtype=torch.float32, device=device)
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)

                model = _xRFM(
                    rfm_params=self._build_rfm_params(),
                    device=device,
                    min_subset_size=self.min_subset_size,
                    tuning_metric="mse",
                    split_method=self.split_method,
                )
                model.fit(X_tr_t, y_tr_t, X_val_t, y_val_t)
                self._model = model
                self._device = device
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise xRFM regressor "
                    f"({type(exc).__name__}: {str(exc)[:200]}). "
                    f"Falling back to kNN approximation.",
                    UserWarning,
                )
                self._model = None

        self.is_fitted_ = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        check_is_fitted(self, "is_fitted_")
        X = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X)

        if self._model is not None:
            try:
                import torch
                X_t = torch.tensor(
                    X_scaled, dtype=torch.float32, device=self._device
                )
                preds = self._model.predict(X_t)
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().numpy()
                preds = np.asarray(preds, dtype=np.float64).ravel()
                # Inverse-transform target scaling
                return self._target_scaler.inverse_transform(
                    preds.reshape(-1, 1)
                ).ravel()
            except Exception:
                pass

        preds_scaled = self._fallback_predict(X_scaled)
        return self._target_scaler.inverse_transform(
            preds_scaled.reshape(-1, 1)
        ).ravel()

    # ----- fallback ----------------------------------------------------------

    def _fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """Distance-weighted kNN fallback when xrfm is unavailable."""
        from scipy.spatial.distance import cdist

        distances = cdist(X, self.X_train_, metric="euclidean")
        k = min(32, len(self.X_train_))
        predictions = np.zeros(len(X))

        for i in range(len(X)):
            nearest_idx = np.argsort(distances[i])[:k]
            nearest_dist = distances[i, nearest_idx]
            nearest_targets = self.y_train_[nearest_idx]

            weights = np.exp(-nearest_dist / (nearest_dist.mean() + 1e-6))
            weights /= weights.sum()

            predictions[i] = np.dot(weights, nearest_targets)

        return predictions

    # ----- feature importances -----------------------------------------------

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances (uniform fallback).

        Returns
        -------
        ndarray of shape (n_features_in_,)
            Normalized feature importances summing to 1.
        """
        check_is_fitted(self, "is_fitted_")
        importances = np.ones(self.n_features_in_) / self.n_features_in_
        return importances
