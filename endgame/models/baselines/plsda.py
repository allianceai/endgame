from __future__ import annotations

"""PLS-DA: Partial Least Squares Discriminant Analysis.

The workhorse classifier of chemometrics, metabolomics and neuroimaging when there are
many correlated features and few samples: features are projected onto a handful of latent
components that maximize covariance with the class indicator, and a logistic model on
those scores gives probabilities requiring independent calibration (the classic PLS-DA thresholds the
regression output instead; the logistic head is the standard probabilistic variant).

References
----------
- Barker & Rayens (2003) "Partial least squares for discrimination", J. Chemometrics.
- Lee et al. (2018) "Partial least squares-discriminant analysis (PLS-DA) for classification
  of high-dimensional (HD) data", Analyst.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.validation import check_array, check_is_fitted

from endgame.validation._study import aligned_vector


class PLSDAClassifier(ClassifierMixin, BaseEstimator):
    """PLS-DA with a logistic head on the latent scores.

    Parameters
    ----------
    n_components : int, default=2
        Number of PLS latent components (capped at the centered feature rank).
    scale : bool, default=True
        Standardize features before PLS.
    class_weight : str or dict or None, default='balanced'
        Class weighting of the logistic head.
    C : float, default=1.0
        Inverse regularization of the logistic head.

    Attributes
    ----------
    pls_ : PLSRegression
    head_ : LogisticRegression
    classes_ : ndarray

    Examples
    --------
    >>> from endgame.models import PLSDAClassifier
    >>> clf = PLSDAClassifier(n_components=3).fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> scores = clf.transform(X_test)   # latent components, for plotting / VIP-style inspection
    """

    def __init__(self, n_components: int = 2, scale: bool = True, class_weight="balanced", C: float = 1.0):
        self.n_components = n_components
        self.scale = scale
        self.class_weight = class_weight
        self.C = C

    def _matrix(self, X, fitting=False):
        if hasattr(X, "columns"):
            if not X.columns.is_unique:
                raise ValueError("duplicate feature names")
            if fitting:
                self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            elif hasattr(self, "feature_names_in_"):
                if set(X.columns) != set(self.feature_names_in_):
                    raise ValueError("prediction feature schema differs from training")
                X = X.loc[:, list(self.feature_names_in_)]
        arr = check_array(X, dtype=float)
        if not fitting and arr.shape[1] != self.n_features_in_:
            raise ValueError("prediction feature count differs from training")
        return arr

    def fit(self, X, y):
        for attr in ("head_", "pls_", "feature_names_in_"):
            if hasattr(self, attr):
                delattr(self, attr)
        if (isinstance(self.n_components, bool) or not isinstance(self.n_components, (int, np.integer))
                or self.n_components < 1):
            raise ValueError("n_components must be a positive integer")
        X = self._matrix(X, fitting=True)
        y = aligned_vector(y, len(X), "y")
        self.n_features_in_ = X.shape[1]
        self._le = LabelEncoder().fit(y)
        self.classes_ = self._le.classes_
        if len(self.classes_) < 2:
            raise ValueError("PLS-DA requires at least two classes")
        yi = self._le.transform(y)
        Y = np.eye(len(self.classes_))[yi] if len(self.classes_) > 2 else yi.astype(float)
        self.scaler_ = StandardScaler().fit(X) if self.scale else None
        Xs = self.scaler_.transform(X) if self.scale else X
        rank = np.linalg.matrix_rank(Xs - Xs.mean(axis=0))
        if rank < 1:
            raise ValueError("PLS-DA requires nonconstant features")
        k = int(min(self.n_components, rank))
        self.n_components_ = k
        self.pls_ = PLSRegression(n_components=k, scale=False).fit(Xs, Y)
        weights = self.class_weight
        if isinstance(weights, dict):
            if set(weights) - set(self.classes_):
                raise ValueError("class_weight contains unknown class labels")
            weights = {i: weights.get(label, 1.) for i, label in enumerate(self.classes_)}
        self.head_ = LogisticRegression(C=self.C, class_weight=weights, max_iter=5000)
        self.head_.fit(self.pls_.transform(Xs), yi)
        return self

    def transform(self, X):
        check_is_fitted(self, "pls_")
        X = self._matrix(X)
        return self.pls_.transform(self.scaler_.transform(X) if self.scale else X)

    def predict_proba(self, X):
        check_is_fitted(self, "head_")
        return self.head_.predict_proba(self.transform(X))

    def predict(self, X):
        return self._le.inverse_transform(np.argmax(self.predict_proba(X), axis=1))

    @property
    def feature_importances_(self):
        """Absolute standardized loading-weighted head coefficients per input feature."""
        check_is_fitted(self, "pls_")
        w = self.head_.coef_ @ self.pls_.x_rotations_.T  # (n_classes_or_1, n_features)
        return np.abs(w).mean(axis=0)
