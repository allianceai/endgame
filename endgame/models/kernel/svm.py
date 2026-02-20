"""Support Vector Machine wrappers with competition-tuned defaults.

SVMs use max-margin optimization which is fundamentally different from
probabilistic models, making them valuable for ensemble diversity.

References
----------
- Cortes & Vapnik, "Support-Vector Networks" (1995)
- sklearn.svm documentation
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC, SVR


class SVMClassifier(ClassifierMixin, BaseEstimator):
    """Support Vector Machine Classifier with competition-tuned defaults.

    A max-margin kernel classifier that finds the optimal separating
    hyperplane. Different optimization objective from probabilistic models,
    making it valuable for ensemble diversity.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'.
    C : float, default=1.0
        Regularization parameter. Lower = more regularization.
    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly', 'sigmoid'.
    degree : int, default=3
        Degree for polynomial kernel.
    probability : bool, default=True
        Enable probability estimates (uses Platt scaling).
    class_weight : str or dict, default='balanced'
        Class weights: 'balanced', None, or dict.
    auto_scale : bool, default=True
        Automatically scale features before fitting.
    max_iter : int, default=10000
        Maximum iterations for solver.
    cache_size : float, default=500
        Kernel cache size in MB.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    model_ : SVC
        Fitted sklearn SVC.
    support_vectors_ : ndarray
        Support vectors from training.

    Examples
    --------
    >>> from endgame.models.kernel import SVMClassifier
    >>> clf = SVMClassifier(kernel='rbf', C=1.0, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    SVMs work best when:
    - Features are scaled (auto_scale=True handles this)
    - Dataset is small-medium sized (scales O(n^2) to O(n^3))
    - Clear margin separation exists

    The max-margin objective is fundamentally different from log-loss
    (logistic regression) or GBDT objectives, providing ensemble diversity.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str | float = "scale",
        degree: int = 3,
        probability: bool = True,
        class_weight: str | dict | None = "balanced",
        auto_scale: bool = True,
        max_iter: int = 10000,
        cache_size: float = 500,
        random_state: int | None = None,
    ):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.probability = probability
        self.class_weight = class_weight
        self.auto_scale = auto_scale
        self.max_iter = max_iter
        self.cache_size = cache_size
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: SVC | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, sample_weight=None, **fit_params) -> "SVMClassifier":
        """Fit the SVM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        # Create and fit model
        self.model_ = SVC(
            kernel=self.kernel,
            C=self.C,
            gamma=self.gamma,
            degree=self.degree,
            probability=self.probability,
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            cache_size=self.cache_size,
            random_state=self.random_state,
        )

        self.model_.fit(X_scaled, y_encoded, sample_weight=sample_weight)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

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
            raise RuntimeError("SVMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        y_pred = self.model_.predict(X_scaled)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Uses Platt scaling for probability calibration.

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
            raise RuntimeError("SVMClassifier has not been fitted.")

        if not self.probability:
            raise RuntimeError("Set probability=True to use predict_proba.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model_.predict_proba(X_scaled)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        decision : ndarray
            Decision function values.
        """
        if not self._is_fitted:
            raise RuntimeError("SVMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        return self.model_.decision_function(X_scaled)

    @property
    def support_vectors_(self):
        """Support vectors from training."""
        if not self._is_fitted:
            raise RuntimeError("SVMClassifier has not been fitted.")
        return self.model_.support_vectors_

    @property
    def n_support_(self):
        """Number of support vectors for each class."""
        if not self._is_fitted:
            raise RuntimeError("SVMClassifier has not been fitted.")
        return self.model_.n_support_


class SVMRegressor(RegressorMixin, BaseEstimator):
    """Support Vector Machine Regressor with competition-tuned defaults.

    Epsilon-SVR that finds a tube around the data where deviations
    smaller than epsilon are ignored.

    Parameters
    ----------
    kernel : str, default='rbf'
        Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'.
    C : float, default=1.0
        Regularization parameter.
    epsilon : float, default=0.1
        Epsilon in the epsilon-SVR model.
    gamma : str or float, default='scale'
        Kernel coefficient.
    degree : int, default=3
        Degree for polynomial kernel.
    auto_scale : bool, default=True
        Automatically scale features before fitting.
    max_iter : int, default=10000
        Maximum iterations for solver.
    cache_size : float, default=500
        Kernel cache size in MB.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    model_ : SVR
        Fitted sklearn SVR.

    Examples
    --------
    >>> from endgame.models.kernel import SVMRegressor
    >>> reg = SVMRegressor(kernel='rbf', C=1.0)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        epsilon: float = 0.1,
        gamma: str | float = "scale",
        degree: int = 3,
        auto_scale: bool = True,
        max_iter: int = 10000,
        cache_size: float = 500,
    ):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.auto_scale = auto_scale
        self.max_iter = max_iter
        self.cache_size = cache_size

        self.n_features_in_: int = 0
        self.model_: SVR | None = None
        self._scaler: StandardScaler | None = None
        self._y_scaler: StandardScaler | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, sample_weight=None, **fit_params) -> "SVMRegressor":
        """Fit the SVM regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        self.n_features_in_ = X.shape[1]

        # Scale features
        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            # Also scale target for SVR
            self._y_scaler = StandardScaler()
            y_scaled = self._y_scaler.fit_transform(y.reshape(-1, 1)).ravel()
        else:
            X_scaled = X
            y_scaled = y

        # Handle NaN
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0)

        # Create and fit model
        self.model_ = SVR(
            kernel=self.kernel,
            C=self.C,
            epsilon=self.epsilon,
            gamma=self.gamma,
            degree=self.degree,
            max_iter=self.max_iter,
            cache_size=self.cache_size,
        )

        self.model_.fit(X_scaled, y_scaled, sample_weight=sample_weight)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        if not self._is_fitted:
            raise RuntimeError("SVMRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)

        if self.auto_scale:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        y_pred = self.model_.predict(X_scaled)

        if self.auto_scale:
            y_pred = self._y_scaler.inverse_transform(y_pred.reshape(-1, 1)).ravel()

        return y_pred

    @property
    def support_vectors_(self):
        """Support vectors from training."""
        if not self._is_fitted:
            raise RuntimeError("SVMRegressor has not been fitted.")
        return self.model_.support_vectors_
