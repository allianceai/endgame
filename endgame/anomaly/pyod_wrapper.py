"""PyOD integration wrapper for unified anomaly detection.

This module provides a universal wrapper around PyOD's 40+ anomaly detection
algorithms with a consistent sklearn-compatible interface. Supports algorithm
selection by name and automatic hyperparameter configuration.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

# Map algorithm names to PyOD classes
PYOD_ALGORITHMS = {
    # Probabilistic
    "ECOD": ("pyod.models.ecod", "ECOD"),
    "COPOD": ("pyod.models.copod", "COPOD"),
    "ABOD": ("pyod.models.abod", "ABOD"),
    "FastABOD": ("pyod.models.abod", "ABOD"),  # with method='fast'

    # Linear Models
    "PCA": ("pyod.models.pca", "PCA"),
    "MCD": ("pyod.models.mcd", "MCD"),
    "OCSVM": ("pyod.models.ocsvm", "OCSVM"),
    "LMDD": ("pyod.models.lmdd", "LMDD"),

    # Proximity-Based
    "LOF": ("pyod.models.lof", "LOF"),
    "COF": ("pyod.models.cof", "COF"),
    "CBLOF": ("pyod.models.cblof", "CBLOF"),
    "LOCI": ("pyod.models.loci", "LOCI"),
    "HBOS": ("pyod.models.hbos", "HBOS"),
    "KNN": ("pyod.models.knn", "KNN"),
    "AvgKNN": ("pyod.models.knn", "KNN"),  # with method='mean'
    "MedKNN": ("pyod.models.knn", "KNN"),  # with method='median'
    "SOD": ("pyod.models.sod", "SOD"),
    "ROD": ("pyod.models.rod", "ROD"),

    # Outlier Ensembles
    "IForest": ("pyod.models.iforest", "IForest"),
    "INNE": ("pyod.models.inne", "INNE"),
    "FB": ("pyod.models.feature_bagging", "FeatureBagging"),
    "LSCP": ("pyod.models.lscp", "LSCP"),
    "XGBOD": ("pyod.models.xgbod", "XGBOD"),
    "LODA": ("pyod.models.loda", "LODA"),
    "SUOD": ("pyod.models.suod", "SUOD"),

    # Neural Networks
    "AutoEncoder": ("pyod.models.auto_encoder", "AutoEncoder"),
    "VAE": ("pyod.models.vae", "VAE"),
    "SO_GAAL": ("pyod.models.so_gaal", "SO_GAAL"),
    "MO_GAAL": ("pyod.models.mo_gaal", "MO_GAAL"),
    "DeepSVDD": ("pyod.models.deep_svdd", "DeepSVDD"),
    "AnoGAN": ("pyod.models.anogan", "AnoGAN"),
    "ALAD": ("pyod.models.alad", "ALAD"),

    # Graph-Based
    "LUNAR": ("pyod.models.lunar", "LUNAR"),

    # Statistical
    "MAD": ("pyod.models.mad", "MAD"),
    "SOS": ("pyod.models.sos", "SOS"),
    "QMCD": ("pyod.models.qmcd", "QMCD"),
    "KDE": ("pyod.models.kde", "KDE"),
    "Sampling": ("pyod.models.sampling", "Sampling"),
    "GMM": ("pyod.models.gmm", "GMM"),
}


# Default hyperparameters for common algorithms
_DEFAULT_PARAMS = {
    "ECOD": {},  # parameter-free
    "COPOD": {},  # parameter-free
    "IForest": {"n_estimators": 200, "max_samples": "auto"},
    "LOF": {"n_neighbors": 20},
    "KNN": {"n_neighbors": 10},
    "HBOS": {"n_bins": 20},
    "PCA": {"n_components": None},  # auto
    "OCSVM": {"kernel": "rbf", "nu": 0.5},
    "CBLOF": {"n_clusters": 8, "alpha": 0.9, "beta": 5},
    "AutoEncoder": {"hidden_neurons": [64, 32, 32, 64], "epochs": 100},
    "VAE": {"encoder_neurons": [64, 32], "decoder_neurons": [32, 64], "epochs": 100},
}


def _import_pyod_class(algorithm: str):
    """Dynamically import a PyOD algorithm class."""
    if algorithm not in PYOD_ALGORITHMS:
        raise ValueError(
            f"Unknown algorithm: {algorithm}. "
            f"Available: {list(PYOD_ALGORITHMS.keys())}"
        )

    module_path, class_name = PYOD_ALGORITHMS[algorithm]

    try:
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"Could not import {algorithm}. "
            f"Make sure PyOD is installed: pip install endgame-ml[anomaly]\n"
            f"Original error: {e}"
        )


class PyODDetector(BaseEstimator, OutlierMixin):
    """Universal wrapper for PyOD anomaly detection algorithms.

    This wrapper provides a unified sklearn-compatible interface to all
    PyOD algorithms, with consistent scoring conventions and automatic
    hyperparameter defaults.

    Parameters
    ----------
    algorithm : str, default='ECOD'
        Name of the PyOD algorithm. See PYOD_ALGORITHMS for available options.
        Popular choices:
        - 'ECOD': Empirical Cumulative Distribution (fast, parameter-free)
        - 'COPOD': Copula-Based (fast, parameter-free)
        - 'IForest': Isolation Forest
        - 'LOF': Local Outlier Factor
        - 'KNN': K-Nearest Neighbors
        - 'HBOS': Histogram-Based (very fast)
        - 'PCA': Principal Component Analysis
        - 'AutoEncoder': Deep learning autoencoder
    contamination : float, default=0.1
        Expected proportion of anomalies.
    random_state : int or None, default=None
        Random seed for reproducibility.
    **kwargs : dict
        Additional algorithm-specific parameters passed to the PyOD model.

    Attributes
    ----------
    model_ : PyOD model
        Fitted PyOD detector instance.
    threshold_ : float
        Decision threshold for binary classification.

    Examples
    --------
    >>> from endgame.anomaly import PyODDetector, PYOD_ALGORITHMS
    >>>
    >>> # List available algorithms
    >>> print(list(PYOD_ALGORITHMS.keys()))
    >>>
    >>> # Fast parameter-free detection
    >>> detector = PyODDetector(algorithm='ECOD')
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)
    >>>
    >>> # KNN-based detection
    >>> detector = PyODDetector(algorithm='KNN', n_neighbors=15)
    >>> detector.fit(X_train)
    >>> labels = detector.predict(X_test)
    >>>
    >>> # Deep learning detector
    >>> detector = PyODDetector(
    ...     algorithm='AutoEncoder',
    ...     hidden_neurons=[128, 64, 64, 128],
    ...     epochs=50
    ... )
    >>> detector.fit(X_train)
    """

    def __init__(
        self,
        algorithm: str = "ECOD",
        contamination: float = 0.1,
        random_state: int | None = None,
        **kwargs: Any,
    ):
        self.algorithm = algorithm
        self.contamination = contamination
        self.random_state = random_state
        self.kwargs = kwargs

    def _get_model_params(self) -> dict:
        """Get parameters for the underlying PyOD model."""
        # Start with algorithm defaults
        params = _DEFAULT_PARAMS.get(self.algorithm, {}).copy()

        # Override with user kwargs
        params.update(self.kwargs)

        # Always set contamination
        params["contamination"] = self.contamination

        # Set random state if applicable
        if self.random_state is not None:
            # Different algorithms use different parameter names
            if self.algorithm in ["IForest", "LODA", "FB"]:
                params["random_state"] = self.random_state

        # Handle special algorithm variants
        if self.algorithm == "FastABOD":
            params["method"] = "fast"
        elif self.algorithm == "AvgKNN":
            params["method"] = "mean"
        elif self.algorithm == "MedKNN":
            params["method"] = "median"

        return params

    def fit(self, X: ArrayLike, y=None) -> PyODDetector:
        """Fit the PyOD detector on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used, present for API consistency.

        Returns
        -------
        self : PyODDetector
            Fitted detector.
        """
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # Import and instantiate the PyOD model
        model_class = _import_pyod_class(self.algorithm)
        params = self._get_model_params()

        # Filter params to only those accepted by the model
        import inspect
        valid_params = inspect.signature(model_class.__init__).parameters.keys()
        filtered_params = {k: v for k, v in params.items() if k in valid_params}

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.model_ = model_class(**filtered_params)
            self.model_.fit(X)

        self.n_features_in_ = X.shape[1]
        self.threshold_ = self.model_.threshold_

        return self

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores. Higher = more anomalous.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # PyOD already uses higher = more anomalous
        return self.model_.decision_function(X)

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for anomalies, 0 for normal samples.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        # PyOD uses 1 for anomaly, 0 for normal (our convention)
        return self.model_.predict(X)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and predict anomaly labels."""
        self.fit(X)
        return self.predict(X)

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict anomaly probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Probabilities for [normal, anomaly] classes.
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if hasattr(self.model_, "predict_proba"):
            return self.model_.predict_proba(X)
        else:
            # Fallback: convert scores to pseudo-probabilities via sigmoid
            scores = self.decision_function(X)
            # Normalize scores
            scores_normalized = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
            return np.column_stack([1 - scores_normalized, scores_normalized])

    def predict_confidence(self, X: ArrayLike) -> np.ndarray:
        """Return prediction confidence scores.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        confidence : ndarray of shape (n_samples,)
            Confidence scores (higher = more confident prediction).
        """
        check_is_fitted(self, ["model_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64)

        if hasattr(self.model_, "predict_confidence"):
            return self.model_.predict_confidence(X)
        else:
            # Fallback: use distance from threshold
            scores = self.decision_function(X)
            return np.abs(scores - self.threshold_)

    @property
    def available_algorithms(self) -> list[str]:
        """List of available PyOD algorithms."""
        return list(PYOD_ALGORITHMS.keys())


def create_detector_ensemble(
    algorithms: list[str] | None = None,
    contamination: float = 0.1,
    random_state: int | None = None,
) -> list[PyODDetector]:
    """Create an ensemble of diverse PyOD detectors.

    Parameters
    ----------
    algorithms : list of str or None, default=None
        Algorithms to include. None uses a default diverse set:
        ['ECOD', 'COPOD', 'IForest', 'LOF', 'KNN', 'HBOS']
    contamination : float, default=0.1
        Expected proportion of anomalies.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Returns
    -------
    detectors : list of PyODDetector
        List of configured detectors ready for fitting.

    Examples
    --------
    >>> from endgame.anomaly import create_detector_ensemble
    >>> detectors = create_detector_ensemble(contamination=0.05)
    >>> for det in detectors:
    ...     det.fit(X_train)
    >>> # Combine scores
    >>> scores = np.mean([d.decision_function(X_test) for d in detectors], axis=0)
    """
    if algorithms is None:
        algorithms = ["ECOD", "COPOD", "IForest", "LOF", "KNN", "HBOS"]

    return [
        PyODDetector(
            algorithm=algo,
            contamination=contamination,
            random_state=random_state,
        )
        for algo in algorithms
    ]
