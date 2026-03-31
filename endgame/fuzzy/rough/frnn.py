"""Fuzzy-Rough Nearest Neighbor (FRNN) classifier.

Combines fuzzy similarity with rough set lower/upper approximations
to classify test samples based on their fuzzy-rough neighborhood.

References
----------
Jensen, R., & Cornelis, C. (2008). A new approach to fuzzy-rough
nearest neighbour classification. LNCS 5306, 310-317.

Example
-------
>>> from endgame.fuzzy.rough.frnn import FuzzyRoughNNClassifier
>>> clf = FuzzyRoughNNClassifier(n_neighbors=5)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class FuzzyRoughNNClassifier(BaseEstimator, ClassifierMixin):
    """Fuzzy-Rough Nearest Neighbor classifier.

    Uses fuzzy similarity and rough set approximations to classify
    test samples. Lower approximations capture certainty while
    upper approximations capture possibility of class membership.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to consider.
    owa_weights : str, default='additive'
        OWA weight generation strategy. One of 'additive', 'exponential',
        'inverse_additive', or 'strict'.
    similarity_metric : str, default='gaussian'
        Similarity metric: 'gaussian' or 'linear'.
    sigma : float, default=1.0
        Width parameter for the Gaussian similarity kernel.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    X_ : ndarray of shape (n_samples, n_features)
        Training data stored for lazy prediction.
    y_ : ndarray of shape (n_samples,)
        Training labels.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.rough.frnn import FuzzyRoughNNClassifier
    >>> X = np.array([[0, 0], [1, 1], [2, 2], [3, 3]])
    >>> y = np.array([0, 0, 1, 1])
    >>> clf = FuzzyRoughNNClassifier(n_neighbors=2, sigma=1.0)
    >>> clf.fit(X, y)
    FuzzyRoughNNClassifier(n_neighbors=2)
    >>> clf.predict(np.array([[1.5, 1.5]]))
    array([0])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        owa_weights: str = "additive",
        similarity_metric: str = "gaussian",
        sigma: float = 1.0,
    ):
        self.n_neighbors = n_neighbors
        self.owa_weights = owa_weights
        self.similarity_metric = similarity_metric
        self.sigma = sigma

    def fit(self, X: Any, y: Any) -> FuzzyRoughNNClassifier:
        """Fit the FRNN classifier by storing training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.X_ = X.copy()
        self.y_ = y.copy()
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels for test samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities using fuzzy-rough approximations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probability estimates. Scores are normalized to sum to 1.
        """
        check_is_fitted(self, ["X_", "y_", "classes_"])
        X = check_array(X)

        n_samples = X.shape[0]
        scores = np.zeros((n_samples, self.n_classes_))

        for i in range(n_samples):
            scores[i] = self._classify_sample(X[i])

        # Normalize to probabilities
        row_sums = scores.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1.0, row_sums)
        return scores / row_sums

    def _classify_sample(self, x: np.ndarray) -> np.ndarray:
        """Classify a single sample using FRNN.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single test sample.

        Returns
        -------
        ndarray of shape (n_classes,)
            Combined scores per class.
        """
        # Compute distances to all training samples
        dists = np.linalg.norm(self.X_ - x, axis=1)

        # Find K nearest neighbors
        k = min(self.n_neighbors, len(self.X_))
        nn_indices = np.argsort(dists)[:k]

        # Compute fuzzy similarity for neighbors
        similarities = self._compute_similarity(dists[nn_indices])

        nn_labels = self.y_[nn_indices]

        # Compute OWA weights
        w = self._owa_weights(k)

        scores = np.zeros(self.n_classes_)

        for c_idx, c in enumerate(self.classes_):
            c_mask = nn_labels == c
            not_c_mask = ~c_mask

            # Lower approximation: certainty that x belongs to class c
            # For non-c neighbors, compute 1 - similarity, then take OWA-min
            if np.any(not_c_mask):
                neg_sims = 1.0 - similarities[not_c_mask]
                neg_sorted = np.sort(neg_sims)  # ascending for OWA-min
                w_lower = w[:len(neg_sorted)]
                w_lower = w_lower / (w_lower.sum() + 1e-12)
                lower = np.dot(w_lower, neg_sorted)
            else:
                lower = 1.0

            # Upper approximation: possibility that x belongs to class c
            # For c-neighbors, take OWA-max of similarities
            if np.any(c_mask):
                pos_sims = similarities[c_mask]
                pos_sorted = np.sort(pos_sims)[::-1]  # descending for OWA-max
                w_upper = w[:len(pos_sorted)]
                w_upper = w_upper / (w_upper.sum() + 1e-12)
                upper = np.dot(w_upper, pos_sorted)
            else:
                upper = 0.0

            # Combine lower and upper approximations
            scores[c_idx] = (lower + upper) / 2.0

        return scores

    def _compute_similarity(self, distances: np.ndarray) -> np.ndarray:
        """Compute fuzzy similarity from distances.

        Parameters
        ----------
        distances : ndarray of shape (n,)
            Euclidean distances.

        Returns
        -------
        ndarray of shape (n,)
            Similarity values in [0, 1].
        """
        if self.similarity_metric == "gaussian":
            return np.exp(-0.5 * (distances / self.sigma) ** 2)
        elif self.similarity_metric == "linear":
            max_d = distances.max()
            if max_d == 0:
                return np.ones_like(distances)
            return 1.0 - distances / max_d
        else:
            raise ValueError(
                f"Unknown similarity_metric: {self.similarity_metric}. "
                "Choose 'gaussian' or 'linear'."
            )

    def _owa_weights(self, n: int) -> np.ndarray:
        """Generate Ordered Weighted Average weights.

        Parameters
        ----------
        n : int
            Number of weights to generate.

        Returns
        -------
        ndarray of shape (n,)
            OWA weights summing to 1.
        """
        if n == 0:
            return np.array([])

        if self.owa_weights == "additive":
            w = np.arange(n, 0, -1, dtype=np.float64)
        elif self.owa_weights == "exponential":
            w = 2.0 ** np.arange(n - 1, -1, -1, dtype=np.float64)
        elif self.owa_weights == "inverse_additive":
            w = 1.0 / np.arange(1, n + 1, dtype=np.float64)
        elif self.owa_weights == "strict":
            w = np.zeros(n)
            w[0] = 1.0
            return w
        else:
            raise ValueError(
                f"Unknown owa_weights: {self.owa_weights}. "
                "Choose 'additive', 'exponential', 'inverse_additive', or 'strict'."
            )

        return w / w.sum()
