"""Discriminant Analysis classifiers.

Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA),
and Regularized Discriminant Analysis (RDA) that interpolates between them.

These methods model class-conditional distributions and make different
assumptions from logistic regression, tree-based methods, and neural
networks, providing ensemble diversity.

References
----------
- Fisher, "The Use of Multiple Measurements in Taxonomic Problems" (1936)
- Friedman, "Regularized Discriminant Analysis" (1989)
- sklearn.discriminant_analysis documentation
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
from sklearn.preprocessing import LabelEncoder


class LDAClassifier(ClassifierMixin, BaseEstimator):
    """Linear Discriminant Analysis Classifier.

    LDA assumes that all classes share the same covariance matrix.
    This leads to linear decision boundaries between classes.

    Parameters
    ----------
    solver : str, default='svd'
        Solver: 'svd', 'lsqr', 'eigen'.
    shrinkage : str, float, or None, default='auto'
        Shrinkage parameter: 'auto' (Ledoit-Wolf), float in [0,1], or None.
    n_components : int, optional
        Number of components for dimensionality reduction.
    store_covariance : bool, default=False
        Store the covariance matrix.
    tol : float, default=1e-4
        Tolerance for singular value decomposition.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    coef_ : ndarray
        Weights of the features.
    intercept_ : ndarray
        Intercept term.

    Examples
    --------
    >>> from endgame.models.baselines import LDAClassifier
    >>> clf = LDAClassifier(shrinkage='auto')
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    LDA is different from logistic regression because:
    1. LDA is generative (models P(X|y)), LR is discriminative (models P(y|X))
    2. LDA assumes Gaussian class-conditional distributions
    3. LDA can be more efficient with limited data

    The shrinkage='auto' option uses Ledoit-Wolf estimation which
    improves performance when n_features > n_samples.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        solver: str = "svd",
        shrinkage: str | float | None = "auto",
        n_components: int | None = None,
        store_covariance: bool = False,
        tol: float = 1e-4,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.n_components = n_components
        self.store_covariance = store_covariance
        self.tol = tol

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: LinearDiscriminantAnalysis | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> "LDAClassifier":
        """Fit the LDA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.

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

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Determine solver compatibility with shrinkage
        solver = self.solver
        shrinkage = self.shrinkage

        # SVD solver doesn't support shrinkage
        if solver == "svd" and shrinkage is not None:
            solver = "lsqr"  # Switch to lsqr which supports shrinkage

        # Create and fit model
        self.model_ = LinearDiscriminantAnalysis(
            solver=solver,
            shrinkage=shrinkage if solver != "svd" else None,
            n_components=self.n_components,
            store_covariance=self.store_covariance,
            tol=self.tol,
        )

        self.model_.fit(X_clean, y_encoded)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        y_pred = self.model_.predict(X_clean)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.predict_proba(X_clean)

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.predict_log_proba(X_clean)

    def transform(self, X) -> np.ndarray:
        """Project data to maximize class separation."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.transform(X_clean)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.decision_function(X_clean)

    @property
    def coef_(self):
        """Feature weights."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")
        return self.model_.coef_

    @property
    def intercept_(self):
        """Intercept term."""
        if not self._is_fitted:
            raise RuntimeError("LDAClassifier has not been fitted.")
        return self.model_.intercept_


class QDAClassifier(ClassifierMixin, BaseEstimator):
    """Quadratic Discriminant Analysis Classifier.

    QDA allows each class to have its own covariance matrix,
    leading to quadratic decision boundaries between classes.

    Parameters
    ----------
    reg_param : float, default=0.0
        Regularization parameter: covariance = (1-reg_param)*cov + reg_param*I
    store_covariance : bool, default=False
        Store the covariance matrices.
    tol : float, default=1e-4
        Tolerance for rank estimation.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.models.baselines import QDAClassifier
    >>> clf = QDAClassifier(reg_param=0.1)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    QDA is more flexible than LDA because it allows different class
    covariances. However, this requires estimating more parameters:
    - LDA: O(d^2) for shared covariance
    - QDA: O(K * d^2) for K classes

    Use reg_param > 0 when you have few samples per class to
    regularize the covariance estimates toward the identity matrix.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        reg_param: float = 0.0,
        store_covariance: bool = False,
        tol: float = 1e-4,
    ):
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.model_: QuadraticDiscriminantAnalysis | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> "QDAClassifier":
        """Fit the QDA classifier."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Create and fit model
        self.model_ = QuadraticDiscriminantAnalysis(
            reg_param=self.reg_param,
            store_covariance=self.store_covariance,
            tol=self.tol,
        )

        self.model_.fit(X_clean, y_encoded)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("QDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        y_pred = self.model_.predict(X_clean)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("QDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.predict_proba(X_clean)

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("QDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.predict_log_proba(X_clean)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function."""
        if not self._is_fitted:
            raise RuntimeError("QDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        return self.model_.decision_function(X_clean)


class RDAClassifier(ClassifierMixin, BaseEstimator):
    """Regularized Discriminant Analysis Classifier.

    RDA interpolates between LDA and QDA using a regularization parameter.
    This allows finding the optimal trade-off between the bias of LDA
    and the variance of QDA.

    Parameters
    ----------
    alpha : float, default=0.5
        Interpolation parameter between LDA (alpha=1) and QDA (alpha=0).
        alpha=0.5 is a common middle ground.
    shrinkage : float, default=0.0
        Shrinkage toward scaled identity: cov = (1-shrinkage)*cov + shrinkage*trace(cov)/d*I
    store_covariance : bool, default=False
        Store the covariance matrices.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.models.baselines import RDAClassifier
    >>> clf = RDAClassifier(alpha=0.5, shrinkage=0.1)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    RDA was proposed by Friedman (1989) to handle the bias-variance
    trade-off between LDA and QDA. The regularized covariance is:

        Sigma_k(alpha, gamma) = alpha * Sigma_pooled + (1-alpha) * Sigma_k

    followed by shrinkage toward scaled identity.

    This provides a continuous family of classifiers that can adapt
    to the complexity supported by the data.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        alpha: float = 0.5,
        shrinkage: float = 0.0,
        store_covariance: bool = False,
    ):
        self.alpha = alpha
        self.shrinkage = shrinkage
        self.store_covariance = store_covariance

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self._label_encoder: LabelEncoder | None = None
        self._means: np.ndarray | None = None
        self._priors: np.ndarray | None = None
        self._covariances: list[np.ndarray] | None = None
        self._pooled_cov: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> "RDAClassifier":
        """Fit the RDA classifier."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Handle NaN
        X_clean = np.nan_to_num(X, nan=0.0)

        # Compute class statistics
        self._means = np.zeros((self.n_classes_, n_features))
        self._priors = np.zeros(self.n_classes_)
        class_covs = []

        for k in range(self.n_classes_):
            mask = y_encoded == k
            X_k = X_clean[mask]
            self._means[k] = np.mean(X_k, axis=0)
            self._priors[k] = np.sum(mask) / n_samples

            if len(X_k) > 1:
                cov_k = np.cov(X_k, rowvar=False)
                if cov_k.ndim == 0:
                    cov_k = np.array([[cov_k]])
            else:
                cov_k = np.eye(n_features) * 1e-6

            class_covs.append(cov_k)

        # Compute pooled covariance
        self._pooled_cov = np.zeros((n_features, n_features))
        for k in range(self.n_classes_):
            mask = y_encoded == k
            n_k = np.sum(mask)
            self._pooled_cov += (n_k - 1) * class_covs[k]
        self._pooled_cov /= (n_samples - self.n_classes_)

        # Regularize covariances (RDA formula)
        self._covariances = []
        for k in range(self.n_classes_):
            # Interpolate between pooled and class-specific
            cov_k = self.alpha * self._pooled_cov + (1 - self.alpha) * class_covs[k]

            # Apply shrinkage toward scaled identity
            if self.shrinkage > 0:
                trace = np.trace(cov_k)
                scaled_identity = (trace / n_features) * np.eye(n_features)
                cov_k = (1 - self.shrinkage) * cov_k + self.shrinkage * scaled_identity

            # Ensure positive definiteness
            cov_k += 1e-6 * np.eye(n_features)

            self._covariances.append(cov_k)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        if not self._is_fitted:
            raise RuntimeError("RDAClassifier has not been fitted.")

        log_proba = self.predict_log_proba(X)
        y_pred = np.argmax(log_proba, axis=1)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        log_proba = self.predict_log_proba(X)
        # Softmax normalization
        log_proba -= np.max(log_proba, axis=1, keepdims=True)
        proba = np.exp(log_proba)
        proba /= np.sum(proba, axis=1, keepdims=True)
        return proba

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("RDAClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        n_samples = X_clean.shape[0]
        log_proba = np.zeros((n_samples, self.n_classes_))

        for k in range(self.n_classes_):
            # Log prior
            log_prior = np.log(self._priors[k])

            # Mahalanobis distance
            diff = X_clean - self._means[k]
            cov_inv = np.linalg.inv(self._covariances[k])
            _, log_det = np.linalg.slogdet(self._covariances[k])

            mahal = np.sum(diff @ cov_inv * diff, axis=1)

            # Log likelihood (up to constant)
            log_likelihood = -0.5 * (log_det + mahal)

            log_proba[:, k] = log_prior + log_likelihood

        return log_proba

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function (log posteriors)."""
        return self.predict_log_proba(X)
