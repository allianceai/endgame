from __future__ import annotations

"""TabDPT: Tabular Discriminative Pre-trained Transformer.

TabDPT (Ma et al., 2024) is a foundation model for tabular data that uses
in-context learning.  It is pre-trained on 123 real-world OpenML datasets
using self-supervised column prediction.  At inference time it performs
zero-shot prediction without fine-tuning -- you simply pass training and
test data together and it predicts via in-context learning (analogous to
TabPFN).

This module provides:
- TabDPTClassifier: Classification wrapper (max 10,000 samples, 500 features)
- TabDPTRegressor: Regression wrapper (max 10,000 samples, 500 features)

When the ``tabdpt`` package is not installed, a distance-weighted kNN
fallback is used so that downstream code can still run (e.g. for testing
or benchmarking without GPU).

References
----------
- Ma et al. "TabDPT: Scaling Tabular Foundation Models" (2024)

Limitations
-----------
- Maximum 10,000 training samples (can override with ignore_pretraining_limits)
- Maximum 500 features
- Maximum 10 classes (classification)
- Supports both classification and regression
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_is_fitted


def _check_tabdpt_available():
    """Check if the ``tabdpt`` package is installed.

    Returns
    -------
    bool
        True if the ``tabdpt`` package can be imported.
    """
    try:
        import tabdpt  # noqa: F401
        return True
    except ImportError:
        return False


class TabDPTClassifier(ClassifierMixin, BaseEstimator):
    """TabDPT classifier -- foundation model for tabular classification.

    Wraps the ``tabdpt`` package which provides a discriminative pre-trained
    transformer for tabular data.  The model uses in-context learning: at
    inference, training examples are fed as context and predictions are made
    without gradient updates.

    When the ``tabdpt`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run (e.g. for testing
    or benchmarking without GPU).

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.  More estimators improve accuracy at
        the cost of inference time.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    max_samples : int, default=10000
        Maximum number of training samples.  Datasets exceeding this limit
        will raise ``ValueError`` unless ``ignore_pretraining_limits=True``.
    max_features : int, default=500
        Maximum number of features.
    max_classes : int, default=10
        Maximum number of classes.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature / class limits.
        Results may be less reliable outside the pre-training distribution.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels discovered during ``fit``.
    n_classes_ : int
        Number of classes.

    Limitations
    -----------
    - Max 10,000 training samples (unless ``ignore_pretraining_limits=True``)
    - Max 500 features
    - Max 10 classes

    Examples
    --------
    >>> from endgame.models.tabular.tabdpt import TabDPTClassifier
    >>> clf = TabDPTClassifier(n_estimators=16)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    MAX_SAMPLES = 10_000
    MAX_FEATURES = 500
    MAX_CLASSES = 10

    def __init__(
        self,
        n_estimators: int = 8,
        device: str = "auto",
        random_state: int = 0,
        max_samples: int = 10_000,
        max_features: int = 500,
        max_classes: int = 10,
        ignore_pretraining_limits: bool = False,
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_features = max_features
        self.max_classes = max_classes
        self.ignore_pretraining_limits = ignore_pretraining_limits

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray, y: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if not self.ignore_pretraining_limits and n_samples > self.max_samples:
            raise ValueError(
                f"TabDPT supports a maximum of {self.max_samples} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if not self.ignore_pretraining_limits and n_features > self.max_features:
            raise ValueError(
                f"TabDPT supports a maximum of {self.max_features} features, "
                f"got {n_features}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if not self.ignore_pretraining_limits and n_classes > self.max_classes:
            raise ValueError(
                f"TabDPT supports a maximum of {self.max_classes} classes, "
                f"got {n_classes}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

    def _resolve_device(self) -> str:
        """Return the concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ----- sklearn interface -------------------------------------------------

    def fit(self, X, y) -> TabDPTClassifier:
        """Store training data and initialise the underlying TabDPT model.

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

        self._validate_data_limits(X, y_encoded)

        # Store for fallback path
        self.X_train_ = X
        self.y_train_ = y_encoded

        # Try to initialise the real TabDPT model
        self._model = None
        if _check_tabdpt_available():
            import tabdpt

            device = self._resolve_device()

            try:
                self._model = tabdpt.TabDPTClassifier(
                    n_estimators=self.n_estimators,
                    device=device,
                    random_state=self.random_state,
                )
                self._model.fit(X, y_encoded)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabDPT classifier "
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

        if self._model is not None:
            return self._model.predict_proba(X)

        return self._fallback_predict_proba(X)

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
        """Distance-weighted kNN fallback when tabdpt is unavailable."""
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

    # ----- static helpers ----------------------------------------------------

    @staticmethod
    def check_constraints(X, y) -> dict:
        """Check whether data satisfies TabDPT constraints.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        y = np.asarray(y)
        issues: list[str] = []

        if X.shape[0] > TabDPTClassifier.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabDPTClassifier.MAX_SAMPLES}"
            )
        if X.shape[1] > TabDPTClassifier.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabDPTClassifier.MAX_FEATURES}"
            )
        n_classes = len(np.unique(y))
        if n_classes > TabDPTClassifier.MAX_CLASSES:
            issues.append(
                f"Too many classes: {n_classes} > "
                f"{TabDPTClassifier.MAX_CLASSES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}


class TabDPTRegressor(RegressorMixin, BaseEstimator):
    """TabDPT regressor -- foundation model for tabular regression.

    Wraps the ``tabdpt`` package which provides a discriminative pre-trained
    transformer for tabular data.  The model uses in-context learning for
    zero-shot regression.

    When the ``tabdpt`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    max_samples : int, default=10000
        Maximum number of training samples.
    max_features : int, default=500
        Maximum number of features.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature limits.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.

    Limitations
    -----------
    - Max 10,000 training samples (unless ``ignore_pretraining_limits=True``)
    - Max 500 features

    Examples
    --------
    >>> from endgame.models.tabular.tabdpt import TabDPTRegressor
    >>> reg = TabDPTRegressor(n_estimators=16)
    >>> reg.fit(X_train, y_train)
    >>> preds = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    MAX_SAMPLES = 10_000
    MAX_FEATURES = 500

    def __init__(
        self,
        n_estimators: int = 8,
        device: str = "auto",
        random_state: int = 0,
        max_samples: int = 10_000,
        max_features: int = 500,
        ignore_pretraining_limits: bool = False,
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.max_samples = max_samples
        self.max_features = max_features
        self.ignore_pretraining_limits = ignore_pretraining_limits

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape

        if not self.ignore_pretraining_limits and n_samples > self.max_samples:
            raise ValueError(
                f"TabDPT supports a maximum of {self.max_samples} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if not self.ignore_pretraining_limits and n_features > self.max_features:
            raise ValueError(
                f"TabDPT supports a maximum of {self.max_features} features, "
                f"got {n_features}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

    def _resolve_device(self) -> str:
        """Return the concrete device string."""
        if self.device != "auto":
            return self.device
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    # ----- sklearn interface -------------------------------------------------

    def fit(self, X, y) -> TabDPTRegressor:
        """Store training data and initialise the underlying TabDPT model.

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

        self._validate_data_limits(X)

        self.n_features_in_ = X.shape[1]
        self.X_train_ = X
        self.y_train_ = y

        self._model = None
        if _check_tabdpt_available():
            import tabdpt

            device = self._resolve_device()

            try:
                self._model = tabdpt.TabDPTRegressor(
                    n_estimators=self.n_estimators,
                    device=device,
                    random_state=self.random_state,
                )
                self._model.fit(X, y)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabDPT regressor "
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

        if self._model is not None:
            return self._model.predict(X)

        return self._fallback_predict(X)

    # ----- fallback ----------------------------------------------------------

    def _fallback_predict(self, X: np.ndarray) -> np.ndarray:
        """Distance-weighted kNN fallback when tabdpt is unavailable."""
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

    # ----- static helpers ----------------------------------------------------

    @staticmethod
    def check_constraints(X) -> dict:
        """Check whether data satisfies TabDPT regression constraints.

        Parameters
        ----------
        X : array-like
            Features.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        issues: list[str] = []

        if X.shape[0] > TabDPTRegressor.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabDPTRegressor.MAX_SAMPLES}"
            )
        if X.shape[1] > TabDPTRegressor.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabDPTRegressor.MAX_FEATURES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}
