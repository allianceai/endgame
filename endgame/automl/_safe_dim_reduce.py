"""Wrappers for dimensionality reduction that gracefully handle
n_components > n_features by capping at runtime."""

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD


class SafePCA(BaseEstimator, TransformerMixin):
    """PCA that caps n_components at min(n_samples, n_features) - 1."""

    def __init__(self, n_components=10, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        n = min(X.shape[0], X.shape[1])
        k = min(self.n_components, max(1, n - 1))
        self._pca = PCA(n_components=k, random_state=self.random_state)
        self._pca.fit(X)
        return self

    def transform(self, X):
        return self._pca.transform(X)


class SafeTruncatedSVD(BaseEstimator, TransformerMixin):
    """TruncatedSVD that caps n_components at n_features - 1."""

    def __init__(self, n_components=10, random_state=None):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        k = min(self.n_components, max(1, X.shape[1] - 1))
        self._svd = TruncatedSVD(n_components=k, random_state=self.random_state)
        self._svd.fit(X)
        return self

    def transform(self, X):
        return self._svd.transform(X)
