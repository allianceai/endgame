"""Ordinal Regression models for ordered categorical targets.

Ordinal regression is appropriate when the target variable has a natural
ordering (e.g., 'bad' < 'average' < 'good') but the distances between
categories are unknown or not meaningful.

Key models:
- All-Threshold (AT): Each class boundary has its own threshold
- Immediate-Threshold (IT): Adjacent classes share boundaries
- SE: Same as AT but using absolute errors
- LAD: Least Absolute Deviation regression

References
----------
- Rennie & Srebro, "Loss Functions for Preference Levels" (2005)
- Pedregosa et al., "mord: A Python Package for Ordinal Regression" (2015)
- https://pythonhosted.org/mord/
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Try importing mord
_HAS_MORD = False
try:
    import mord
    _HAS_MORD = True
except ImportError:
    pass


class OrdinalClassifier(ClassifierMixin, BaseEstimator):
    """Unified Ordinal Regression Classifier with auto-variant selection.

    Wraps mord library ordinal regression methods with automatic model
    selection based on data characteristics.

    Ordinal regression is critical for ordered categorical targets where
    standard classification ignores the ordering (e.g., rating prediction,
    grade classification, severity levels).

    Parameters
    ----------
    variant : str, default='auto'
        Ordinal regression variant:
        - 'auto': Automatically select based on data
        - 'at': All-Threshold (LogisticAT) - most common
        - 'it': Immediate-Threshold (LogisticIT)
        - 'se': All-Threshold with absolute errors
        - 'lad': Least Absolute Deviation
        - 'ridge': Ordinal Ridge regression
    alpha : float, default=1.0
        Regularization strength (inverse of C for logistic models,
        regularization strength for Ridge/LAD).
    max_iter : int, default=1000
        Maximum iterations for optimization.
    auto_scale : bool, default=True
        Whether to standardize features before fitting.
    random_state : int, optional
        Random seed (not used by all variants).

    Attributes
    ----------
    classes_ : ndarray
        Ordered class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features.
    variant_ : str
        The actual variant used.
    model_ : mord estimator
        Fitted ordinal regression model.
    coef_ : ndarray
        Feature coefficients.
    theta_ : ndarray
        Class thresholds (boundaries).

    Examples
    --------
    >>> from endgame.models.ordinal import OrdinalClassifier
    >>> clf = OrdinalClassifier(variant='at', alpha=1.0)
    >>> clf.fit(X_train, y_train)  # y_train has ordered labels
    >>> y_pred = clf.predict(X_test)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    Ordinal regression assumes:
    1. Target classes have a meaningful order
    2. A latent continuous variable underlies the ordered categories
    3. Thresholds partition this latent space into ordered categories

    The cumulative model is:
        P(Y <= j) = g(theta_j - X @ beta)
    where g is a link function (logistic, probit, etc.).
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        variant: str = "auto",
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.variant = variant
        self.alpha = alpha
        self.max_iter = max_iter
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.variant_: str | None = None
        self.model_: Any | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def _detect_variant(self, X: np.ndarray, y: np.ndarray) -> str:
        """Auto-detect the best ordinal regression variant.

        Decision logic:
        - Small dataset (n < 1000) -> ridge (faster, more stable)
        - Large dataset -> at (more flexible)
        - Many classes (> 10) -> ridge (fewer parameters)
        """
        n_samples = X.shape[0]
        n_classes = len(np.unique(y))

        if n_samples < 1000 or n_classes > 10:
            return "ridge"
        else:
            return "at"

    def _create_model(self):
        """Create the appropriate ordinal regression model."""
        if not _HAS_MORD:
            raise ImportError(
                "Ordinal regression requires mord. "
                "Install with: pip install mord"
            )

        if self.variant_ == "at":
            return mord.LogisticAT(alpha=self.alpha, max_iter=self.max_iter)
        elif self.variant_ == "it":
            return mord.LogisticIT(alpha=self.alpha, max_iter=self.max_iter)
        elif self.variant_ == "se":
            return mord.LogisticSE(alpha=self.alpha, max_iter=self.max_iter)
        elif self.variant_ == "lad":
            return mord.LAD(C=1.0 / self.alpha, max_iter=self.max_iter)
        elif self.variant_ == "ridge":
            return mord.OrdinalRidge(alpha=self.alpha, max_iter=self.max_iter)
        else:
            raise ValueError(
                f"Unknown variant: {self.variant_}. "
                "Options: 'auto', 'at', 'it', 'se', 'lad', 'ridge'"
            )

    def fit(self, X, y, sample_weight=None, **fit_params) -> "OrdinalClassifier":
        """Fit the ordinal regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Ordered target labels. Labels should be integers 0, 1, 2, ...
            or will be encoded to integers preserving order.
        sample_weight : array-like, optional
            Not supported by mord, ignored.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Encode labels to consecutive integers
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X.copy()

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Determine variant
        if self.variant == "auto":
            self.variant_ = self._detect_variant(X_scaled, y_encoded)
        else:
            self.variant_ = self.variant

        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X_scaled, y_encoded)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict ordinal class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("OrdinalClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X.copy()

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        y_pred_encoded = self.model_.predict(X_scaled)
        return self._label_encoder.inverse_transform(y_pred_encoded.astype(int))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        For ordinal regression, probabilities are derived from the
        cumulative model:
            P(Y = j) = P(Y <= j) - P(Y <= j-1)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("OrdinalClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X.copy()

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # mord models typically don't have predict_proba
        # We compute it from the cumulative probabilities
        if hasattr(self.model_, 'predict_proba'):
            return self.model_.predict_proba(X_scaled)
        else:
            # Compute from linear predictions and thresholds
            return self._compute_proba(X_scaled)

    def _compute_proba(self, X: np.ndarray) -> np.ndarray:
        """Compute class probabilities from cumulative model.

        P(Y = j) = sigmoid(theta_j - X@beta) - sigmoid(theta_{j-1} - X@beta)
        """
        from scipy.special import expit

        # Linear predictions
        linear = X @ self.coef_.ravel()

        # Thresholds
        theta = self.theta_

        # Cumulative probabilities
        n_samples = X.shape[0]
        n_classes = len(theta) + 1
        proba = np.zeros((n_samples, n_classes))

        # P(Y <= j) for each threshold
        cumprob = np.zeros((n_samples, n_classes))
        cumprob[:, -1] = 1.0  # P(Y <= K-1) = 1

        for j in range(n_classes - 1):
            cumprob[:, j] = expit(theta[j] - linear)

        # P(Y = j) = P(Y <= j) - P(Y <= j-1)
        proba[:, 0] = cumprob[:, 0]
        for j in range(1, n_classes):
            proba[:, j] = cumprob[:, j] - cumprob[:, j - 1]

        # Clip for numerical stability
        proba = np.clip(proba, 1e-10, 1.0)
        proba = proba / proba.sum(axis=1, keepdims=True)

        return proba

    @property
    def coef_(self) -> np.ndarray:
        """Feature coefficients."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        return self.model_.coef_

    @property
    def theta_(self) -> np.ndarray:
        """Class thresholds (boundaries)."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted.")
        if hasattr(self.model_, 'theta_'):
            return self.model_.theta_
        elif hasattr(self.model_, 'classes_'):
            # Ridge uses different attribute name
            return getattr(self.model_, 'theta_', np.arange(self.n_classes_ - 1))
        return np.arange(self.n_classes_ - 1)


# Convenience wrappers for specific variants

class OrdinalRidge(OrdinalClassifier):
    """Ordinal Ridge Regression.

    Ridge regression for ordinal targets. Uses L2 regularization.
    Good for smaller datasets and many ordinal classes.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum iterations.
    auto_scale : bool, default=True
        Whether to standardize features.

    Examples
    --------
    >>> from endgame.models.ordinal import OrdinalRidge
    >>> clf = OrdinalRidge(alpha=1.0)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            variant="ridge",
            alpha=alpha,
            max_iter=max_iter,
            auto_scale=auto_scale,
            random_state=random_state,
        )


class LogisticAT(OrdinalClassifier):
    """All-Threshold Ordinal Logistic Regression.

    The most common ordinal regression model. Each class boundary has
    its own threshold parameter.

    Also known as: Proportional Odds Model, Cumulative Logit Model.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (inverse of C).
    max_iter : int, default=1000
        Maximum iterations.
    auto_scale : bool, default=True
        Whether to standardize features.

    Examples
    --------
    >>> from endgame.models.ordinal import LogisticAT
    >>> clf = LogisticAT(alpha=1.0)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            variant="at",
            alpha=alpha,
            max_iter=max_iter,
            auto_scale=auto_scale,
            random_state=random_state,
        )


class LogisticIT(OrdinalClassifier):
    """Immediate-Threshold Ordinal Logistic Regression.

    Adjacent classes share threshold boundaries. More constrained
    than All-Threshold, which can help with small datasets.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum iterations.
    auto_scale : bool, default=True
        Whether to standardize features.

    Examples
    --------
    >>> from endgame.models.ordinal import LogisticIT
    >>> clf = LogisticIT(alpha=1.0)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            variant="it",
            alpha=alpha,
            max_iter=max_iter,
            auto_scale=auto_scale,
            random_state=random_state,
        )


class LogisticSE(OrdinalClassifier):
    """Squared-Error Ordinal Logistic Regression.

    All-Threshold variant but using squared errors in optimization.
    Can be more robust to outliers.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength.
    max_iter : int, default=1000
        Maximum iterations.
    auto_scale : bool, default=True
        Whether to standardize features.

    Examples
    --------
    >>> from endgame.models.ordinal import LogisticSE
    >>> clf = LogisticSE(alpha=1.0)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            variant="se",
            alpha=alpha,
            max_iter=max_iter,
            auto_scale=auto_scale,
            random_state=random_state,
        )


class LAD(OrdinalClassifier):
    """Least Absolute Deviation Ordinal Regression.

    Uses L1 loss (absolute errors) instead of L2. More robust
    to outliers in the target variable.

    Parameters
    ----------
    alpha : float, default=1.0
        Regularization strength (inverse of C parameter).
    max_iter : int, default=1000
        Maximum iterations.
    auto_scale : bool, default=True
        Whether to standardize features.

    Examples
    --------
    >>> from endgame.models.ordinal import LAD
    >>> clf = LAD(alpha=1.0)
    >>> clf.fit(X_train, y_train)
    >>> y_pred = clf.predict(X_test)
    """

    def __init__(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        super().__init__(
            variant="lad",
            alpha=alpha,
            max_iter=max_iter,
            auto_scale=auto_scale,
            random_state=random_state,
        )
