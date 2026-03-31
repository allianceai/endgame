"""AutoCloud — Data cloud-based evolving classifier.

Implements an evolving classifier based on the AutoCloud concept where
data clouds are non-parametric representations of clusters formed via
recursive density estimation. No predefined structure is assumed.

References
----------
- Bezerra, C. G., Costa, B. S. J., Guedes, L. A., & Angelov, P. P. (2020).
  An evolving approach to data streams clustering based on typicality and
  eccentricity data analytics. Information Sciences, 518, 13-28.
- Angelov, P. P. & Gu, X. (2019). Empirical Approach to Machine Learning.
  Springer.

Example
-------
>>> from endgame.fuzzy.evolving.autocloud import AutoCloudClassifier
>>> import numpy as np
>>> X = np.random.randn(100, 3)
>>> y = (X[:, 0] > 0).astype(int)
>>> model = AutoCloudClassifier(significance_level=0.05)
>>> model.fit(X, y)
>>> preds = model.predict(X[:5])
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class _DataCloud:
    """Internal representation of a single data cloud.

    A data cloud is a non-parametric cluster characterized by its
    mean, variance, and sample count for recursive density estimation.

    Parameters
    ----------
    center : ndarray of shape (n_features,)
        Initial center of the cloud.
    label : int
        Class label associated with this cloud.
    """

    def __init__(self, center: np.ndarray, label: int):
        self.center = center.copy()
        self.label = label
        self.n_samples = 1
        self._sum = center.copy()
        self._sum_sq = center ** 2
        self.variance = np.zeros_like(center)

    def update(self, x: np.ndarray) -> None:
        """Update cloud statistics with a new sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            New data point.
        """
        self.n_samples += 1
        self._sum += x
        self._sum_sq += x ** 2
        self.center = self._sum / self.n_samples
        self.variance = (
            self._sum_sq / self.n_samples - self.center ** 2
        )
        self.variance = np.maximum(self.variance, 0.0)

    def eccentricity(self, x: np.ndarray) -> float:
        """Compute eccentricity of a point relative to this cloud.

        Eccentricity measures how far a point is from the cloud center
        relative to the cloud spread. Higher = more eccentric (outlier).

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Data point.

        Returns
        -------
        float
            Eccentricity value.
        """
        if self.n_samples < 2:
            return 0.0

        diff = x - self.center
        var_sum = np.sum(self.variance) + 1e-10
        ecc = np.sum(diff ** 2) / (var_sum * self.n_samples)
        return float(ecc)

    def typicality(self, x: np.ndarray) -> float:
        """Compute typicality (inverse eccentricity) of a point.

        Typicality measures how representative a point is of the cloud.
        Higher = more typical (closer to center).

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Data point.

        Returns
        -------
        float
            Typicality value in (0, 1].
        """
        ecc = self.eccentricity(x)
        return 1.0 / (1.0 + ecc)

    def density(self, x: np.ndarray) -> float:
        """Compute recursive density estimate for a point.

        Uses Cauchy-type density based on distance to center and spread.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Data point.

        Returns
        -------
        float
            Density value.
        """
        diff = x - self.center
        spread = np.sum(self.variance) + 1e-10
        dist_sq = np.sum(diff ** 2)
        return 1.0 / (1.0 + dist_sq / spread)


class AutoCloudClassifier(BaseEstimator, ClassifierMixin):
    """Data cloud-based evolving classifier.

    Forms fuzzy rules from data density without predefined structure.
    Each data cloud is a non-parametric representation of a cluster
    associated with a class label. Classification is based on the
    typicality/density of new samples to existing clouds.

    Parameters
    ----------
    significance_level : float, default=0.05
        Threshold for deciding when to create a new cloud. A new cloud
        is created when the eccentricity of a sample exceeds
        ``2 / (n_samples * significance_level)`` for all existing clouds
        of its class.
    max_clouds : int, default=100
        Maximum number of data clouds.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    clouds_ : list of _DataCloud
        Evolved data clouds.
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    n_clouds_ : int
        Current number of data clouds.
    n_samples_seen_ : int
        Total number of samples processed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.evolving.autocloud import AutoCloudClassifier
    >>> X = np.vstack([np.random.randn(50, 2) + [2, 2],
    ...               np.random.randn(50, 2) - [2, 2]])
    >>> y = np.array([0]*50 + [1]*50)
    >>> model = AutoCloudClassifier(significance_level=0.05)
    >>> model.fit(X, y)
    AutoCloudClassifier()
    >>> model.predict(X[:3]).shape
    (3,)
    """

    def __init__(
        self,
        significance_level: float = 0.05,
        max_clouds: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.significance_level = significance_level
        self.max_clouds = max_clouds
        self.random_state = random_state
        self.verbose = verbose

    def _should_create_cloud(
        self, x: np.ndarray, label: int
    ) -> bool:
        """Determine if a new cloud should be created for this sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input sample.
        label : int
            Encoded class label.

        Returns
        -------
        bool
            True if a new cloud should be created.
        """
        if len(self.clouds_) >= self.max_clouds:
            return False

        # Find clouds of same class
        class_clouds = [c for c in self.clouds_ if c.label == label]
        if not class_clouds:
            return True

        # Check eccentricity against all same-class clouds
        threshold = 2.0 / max(
            self.n_samples_seen_ * self.significance_level, 1.0
        )

        for cloud in class_clouds:
            ecc = cloud.eccentricity(x)
            if ecc <= threshold:
                return False

        return True

    def _find_closest_cloud(
        self, x: np.ndarray, label: int | None = None
    ) -> int | None:
        """Find the closest cloud to a data point.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input sample.
        label : int or None
            If provided, only consider clouds of this class.

        Returns
        -------
        int or None
            Index of the closest cloud, or None if no matching clouds.
        """
        best_idx = None
        best_density = -np.inf

        for i, cloud in enumerate(self.clouds_):
            if label is not None and cloud.label != label:
                continue
            d = cloud.density(x)
            if d > best_density:
                best_density = d
                best_idx = i

        return best_idx

    def _process_sample(self, x: np.ndarray, label: int) -> None:
        """Process a single labeled sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Input features.
        label : int
            Encoded class label.
        """
        self.n_samples_seen_ += 1

        if not self.clouds_:
            self.clouds_.append(_DataCloud(x, label))
            if self.verbose:
                print(f"[AutoCloud] Created initial cloud. Label={label}")
            return

        if self._should_create_cloud(x, label):
            self.clouds_.append(_DataCloud(x, label))
            if self.verbose:
                print(
                    f"[AutoCloud] Created cloud {len(self.clouds_)}. "
                    f"Label={label}"
                )
        else:
            # Update closest cloud of same class
            idx = self._find_closest_cloud(x, label=label)
            if idx is not None:
                self.clouds_[idx].update(x)

    def fit(self, X: Any, y: Any) -> AutoCloudClassifier:
        """Build data clouds from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        self.n_samples_seen_ = 0

        self._label_encoder_ = LabelEncoder()
        y_enc = self._label_encoder_.fit_transform(y)
        self.classes_ = self._label_encoder_.classes_

        self.clouds_ = []

        rng = np.random.RandomState(self.random_state)
        indices = rng.permutation(X.shape[0])

        for idx in indices:
            self._process_sample(X[idx], y_enc[idx])

        return self

    def partial_fit(
        self, X: Any, y: Any, classes: Any = None
    ) -> AutoCloudClassifier:
        """Incrementally update with new data, potentially creating new clouds.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New input data.
        y : array-like of shape (n_samples,)
            New class labels.
        classes : array-like, optional
            All possible class labels. Required on the first call if
            not all classes are present.

        Returns
        -------
        self
            Updated estimator.
        """
        X, y = check_X_y(X, y)

        if not hasattr(self, "n_features_in_"):
            self.n_features_in_ = X.shape[1]
            self.n_samples_seen_ = 0
            self.clouds_ = []
            if classes is not None:
                self.classes_ = np.asarray(classes)
            else:
                self.classes_ = np.unique(y)
            self._label_encoder_ = LabelEncoder()
            self._label_encoder_.fit(self.classes_)

        y_enc = self._label_encoder_.transform(y)

        for i in range(X.shape[0]):
            self._process_sample(X[i], y_enc[i])

        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities based on cloud densities.

        For each sample, the probability of each class is proportional
        to the maximum density from clouds of that class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["clouds_", "classes_"])
        X = check_array(X)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            for j, cloud in enumerate(self.clouds_):
                class_idx = cloud.label
                density = cloud.density(X[i])
                proba[i, class_idx] = max(proba[i, class_idx], density)

        # Normalize to probabilities
        row_sums = np.sum(proba, axis=1, keepdims=True)
        row_sums = np.maximum(row_sums, 1e-10)
        proba = proba / row_sums

        return proba

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    @property
    def n_clouds_(self) -> int:
        """Number of data clouds in the system."""
        check_is_fitted(self, ["clouds_"])
        return len(self.clouds_)
