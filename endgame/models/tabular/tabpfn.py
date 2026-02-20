"""TabPFN: Tabular Prior-Data Fitted Network.

TabPFN is a transformer that performs in-context learning for tabular data.
No training required - just inference. Pre-trained on millions of synthetic
datasets, it achieves competitive performance on small tabular datasets.

This module provides:
- TabPFNClassifier: v1 wrapper (classification only, max 100 features)
- TabPFNv2Classifier: v2 wrapper (classification, max 500 features, post-hoc ensembling)
- TabPFNv2Regressor: v2 wrapper (regression, max 500 features, post-hoc ensembling)
- TabPFN25Classifier: v2.5 wrapper (classification, max 2000 features, 50K samples)
- TabPFN25Regressor: v2.5 wrapper (regression, max 2000 features, 50K samples)

References
----------
- Hollmann et al. "TabPFN: A Transformer That Solves Small Tabular Classification
  Problems in a Second" (2023)
- Hollmann et al. "Accurate Predictions on Small Data with a Tabular Foundation
  Model" (2025) - TabPFN v2
- Prior Labs "TabPFN v2.5 / RealTabPFN-2.5" (2025) - 5x sample and 4x feature
  scaling, fine-tuned on real data

Limitations (v1)
----------------
- Maximum 10,000 training samples
- Maximum 100 features (after encoding)
- Maximum 10 classes
- Classification only (no regression variant)
- Works best with numerical features (categorical features are auto-encoded)

Limitations (v2)
----------------
- Maximum 10,000 training samples (can override with ignore_pretraining_limits)
- Maximum 500 features
- Maximum 10 classes (classification)
- Supports both classification and regression
- Native categorical feature handling

Limitations (v2.5)
------------------
- Maximum 50,000 training samples (can override with ignore_pretraining_limits)
- Maximum 2,000 features
- Maximum 10 classes (classification)
- Supports both classification and regression
- Native categorical feature handling
- model_version='2.5_real' uses RealTabPFN-2.5 (fine-tuned on real data)
- model_version='2.5' uses the synthetic-only pre-trained variant
"""

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted


class TabPFNClassifier(ClassifierMixin, BaseEstimator):
    """TabPFN: Tabular Prior-Data Fitted Network classifier.

    A transformer that performs in-context learning for tabular classification.
    No training required - just stores data and performs inference.
    Extremely fast and competitive on small datasets.

    Parameters
    ----------
    n_ensemble_configurations : int, default=16
        Number of ensemble configurations. More configurations increase
        accuracy but slow down inference. Values: 1, 4, 8, 16, 32.
    device : str, default='auto'
        Computation device: 'cuda', 'cpu', or 'auto'.
    batch_size : int, default=1
        Batch size for inference.
    seed : int, default=0
        Random seed for ensemble configurations.
    subsample_features : bool, default=True
        Whether to subsample features if > 100 after encoding.
    categorical_features : List[int] or 'auto', default='auto'
        Indices of categorical features. If 'auto', detects features with
        few unique values (< 20) as categorical. Set to None to disable
        categorical encoding.
    cat_cardinality_threshold : int, default=20
        When categorical_features='auto', features with <= this many unique
        values are treated as categorical and one-hot encoded.
    scale_numerical : bool, default=True
        Whether to standardize numerical features.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    X_train_ : ndarray
        Stored training features (after encoding).
    y_train_ : ndarray
        Stored training labels.

    Limitations
    -----------
    - Max 10,000 training samples (hard limit of the model)
    - Max 100 features after encoding (hard limit, will subsample if exceeded)
    - Max 10 classes (hard limit)
    - Classification only (no regression)

    Examples
    --------
    >>> from endgame.models.tabular import TabPFNClassifier
    >>> clf = TabPFNClassifier(n_ensemble_configurations=32)
    >>> clf.fit(X_train, y_train)  # Just stores data, no training
    >>> proba = clf.predict_proba(X_test)  # In-context learning inference

    Notes
    -----
    TabPFN performs remarkably well on small datasets (< 3000 samples)
    and is often competitive with well-tuned gradient boosting methods.
    Categorical features are automatically one-hot encoded, which is the
    recommended approach for TabPFN as it expects continuous features.
    """

    _estimator_type = "classifier"

    # Model constraints
    MAX_SAMPLES = 10000
    MAX_FEATURES = 100
    MAX_CLASSES = 10

    def __init__(
        self,
        n_ensemble_configurations: int = 16,
        device: str = "auto",
        batch_size: int = 1,
        seed: int = 0,
        subsample_features: bool = True,
        categorical_features: list[int] | str | None = "auto",
        cat_cardinality_threshold: int = 20,
        scale_numerical: bool = True,
    ):
        self.n_ensemble_configurations = n_ensemble_configurations
        self.device = device
        self.batch_size = batch_size
        self.seed = seed
        self.subsample_features = subsample_features
        self.categorical_features = categorical_features
        self.cat_cardinality_threshold = cat_cardinality_threshold
        self.scale_numerical = scale_numerical

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.X_train_: np.ndarray | None = None
        self.y_train_: np.ndarray | None = None
        self._label_encoder: LabelEncoder | None = None
        self._model = None
        self._device = None
        self._feature_indices: np.ndarray | None = None
        self._is_fitted: bool = False

        # Preprocessing state
        self._cat_feature_indices: list[int] | None = None
        self._num_feature_indices: list[int] | None = None
        self._one_hot_encoder: OneHotEncoder | None = None
        self._scaler: StandardScaler | None = None

    def _check_tabpfn_available(self):
        """Check if TabPFN is installed."""
        try:
            from tabpfn import TabPFNClassifier as _TabPFN
            return True
        except ImportError:
            return False

    def _get_device(self):
        """Determine computation device."""
        import torch

        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate inputs against TabPFN constraints."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if n_samples > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN supports maximum {self.MAX_SAMPLES} training samples, "
                f"got {n_samples}. Consider subsampling your data."
            )

        if n_classes > self.MAX_CLASSES:
            raise ValueError(
                f"TabPFN supports maximum {self.MAX_CLASSES} classes, "
                f"got {n_classes}."
            )

        if n_features > self.MAX_FEATURES and not self.subsample_features:
            raise ValueError(
                f"TabPFN supports maximum {self.MAX_FEATURES} features, "
                f"got {n_features}. Set subsample_features=True to subsample."
            )

    def _detect_categorical_features(self, X: np.ndarray) -> list[int]:
        """Detect categorical features based on cardinality and dtype.

        Features are considered categorical if:
        - They have low cardinality (unique values <= threshold)
        - OR all values are integers with small range
        """
        cat_indices = []
        n_samples = X.shape[0]

        for i in range(X.shape[1]):
            col = X[:, i]
            n_unique = len(np.unique(col[~np.isnan(col)]))

            # Consider categorical if:
            # 1. Low cardinality
            # 2. All integer values
            is_integer = np.allclose(col[~np.isnan(col)], col[~np.isnan(col)].astype(int))

            if n_unique <= self.cat_cardinality_threshold and (is_integer or n_unique <= 10):
                cat_indices.append(i)

        return cat_indices

    def _preprocess_fit(self, X: np.ndarray) -> np.ndarray:
        """Fit preprocessing and transform training data.

        Handles categorical feature encoding and numerical scaling.
        """
        n_features = X.shape[1]

        # Determine categorical features
        if self.categorical_features == "auto":
            self._cat_feature_indices = self._detect_categorical_features(X)
        elif self.categorical_features is None:
            self._cat_feature_indices = []
        else:
            self._cat_feature_indices = list(self.categorical_features)

        self._num_feature_indices = [i for i in range(n_features) if i not in self._cat_feature_indices]

        transformed_parts = []

        # One-hot encode categorical features
        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            # Handle NaN in categorical by replacing with a special value
            X_cat = np.nan_to_num(X_cat, nan=-999)

            self._one_hot_encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown='ignore',
                drop=None,  # Keep all categories for TabPFN
            )
            X_cat_encoded = self._one_hot_encoder.fit_transform(X_cat)
            transformed_parts.append(X_cat_encoded)

        # Scale numerical features
        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices]
            X_num = np.nan_to_num(X_num, nan=0.0)

            if self.scale_numerical:
                self._scaler = StandardScaler()
                X_num_scaled = self._scaler.fit_transform(X_num)
            else:
                X_num_scaled = X_num
            transformed_parts.append(X_num_scaled)

        if transformed_parts:
            X_transformed = np.hstack(transformed_parts)
        else:
            X_transformed = X

        return X_transformed.astype(np.float32)

    def _preprocess_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted preprocessing."""
        transformed_parts = []

        if self._cat_feature_indices:
            X_cat = X[:, self._cat_feature_indices]
            X_cat = np.nan_to_num(X_cat, nan=-999)
            X_cat_encoded = self._one_hot_encoder.transform(X_cat)
            transformed_parts.append(X_cat_encoded)

        if self._num_feature_indices:
            X_num = X[:, self._num_feature_indices]
            X_num = np.nan_to_num(X_num, nan=0.0)

            if self._scaler is not None:
                X_num_scaled = self._scaler.transform(X_num)
            else:
                X_num_scaled = X_num
            transformed_parts.append(X_num_scaled)

        if transformed_parts:
            X_transformed = np.hstack(transformed_parts)
        else:
            X_transformed = X

        return X_transformed.astype(np.float32)

    def fit(self, X, y) -> "TabPFNClassifier":
        """Store training data for in-context learning.

        Note: TabPFN doesn't actually train - it performs in-context learning
        at inference time. This method preprocesses, validates, and stores data.

        Categorical features are automatically detected and one-hot encoded,
        as TabPFN works best with continuous numerical features.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.

        Returns
        -------
        self
            Fitted estimator.

        Raises
        ------
        ValueError
            If data exceeds TabPFN's limitations.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Validate basic constraints first
        if X.shape[0] > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN supports maximum {self.MAX_SAMPLES} training samples, "
                f"got {X.shape[0]}. Consider subsampling your data."
            )

        if self.n_classes_ > self.MAX_CLASSES:
            raise ValueError(
                f"TabPFN supports maximum {self.MAX_CLASSES} classes, "
                f"got {self.n_classes_}."
            )

        # Preprocess: encode categorical features and scale numerical
        X_processed = self._preprocess_fit(X)

        # Handle feature subsampling if needed (after encoding)
        if X_processed.shape[1] > self.MAX_FEATURES:
            if self.subsample_features:
                # Select most important features using variance
                rng = np.random.RandomState(self.seed)
                variances = np.var(X_processed, axis=0)
                # Keep features with highest variance
                self._feature_indices = np.argsort(variances)[-self.MAX_FEATURES:]
                X_processed = X_processed[:, self._feature_indices]
            else:
                raise ValueError(
                    f"Too many features after encoding: {X_processed.shape[1]} > {self.MAX_FEATURES}. "
                    f"Set subsample_features=True to subsample."
                )
        else:
            self._feature_indices = None

        # Store data
        self.X_train_ = X_processed
        self.y_train_ = y_encoded

        # Initialize model if available
        if self._check_tabpfn_available():
            from tabpfn import TabPFNClassifier as _TabPFN

            self._device = self._get_device()

            try:
                # New TabPFN API (v2.0+): n_estimators replaces N_ensemble_configurations
                # Also use native categorical feature support instead of our preprocessing
                try:
                    self._model = _TabPFN(
                        n_estimators=self.n_ensemble_configurations,
                        device=self._device,
                        random_state=self.seed,
                        # Use TabPFN's native categorical handling if we detected categorical features
                        categorical_features_indices=self._cat_feature_indices if self._cat_feature_indices else None,
                    )
                except TypeError:
                    # Fallback for older TabPFN versions
                    self._model = _TabPFN(
                        N_ensemble_configurations=self.n_ensemble_configurations,
                        device=self._device,
                        seed=self.seed,
                    )

                # When using native categorical support, pass original preprocessed data
                # (the new TabPFN handles categorical encoding internally)
                self._model.fit(self.X_train_, self.y_train_)

            except RuntimeError as e:
                # TabPFN v2.5+ requires HuggingFace authentication for model download
                # Fall back to kNN approximation
                import warnings
                warnings.warn(
                    f"Could not initialize TabPFN model (likely requires HuggingFace authentication). "
                    f"Using kNN fallback. Error: {str(e)[:100]}",
                    UserWarning
                )
                self._model = None

        self._is_fitted = True
        return self

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities using in-context learning.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        self._check_is_fitted()

        X = np.asarray(X, dtype=np.float32)

        # Apply same preprocessing (categorical encoding + scaling)
        X_processed = self._preprocess_transform(X)

        # Apply same feature subsampling
        if self._feature_indices is not None:
            X_processed = X_processed[:, self._feature_indices]

        if self._model is not None:
            # Use actual TabPFN
            proba = self._model.predict_proba(X_processed)
        else:
            # Fallback: simple kNN-based approximation
            proba = self._fallback_predict_proba(X_processed)

        return proba

    def _fallback_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fallback prediction when TabPFN is not installed.

        Uses a simple distance-weighted kNN as approximation.
        """
        from scipy.spatial.distance import cdist

        # Compute distances
        distances = cdist(X, self.X_train_, metric='euclidean')

        # Softmax-weighted voting
        k = min(32, len(self.X_train_))
        proba = np.zeros((len(X), self.n_classes_))

        for i in range(len(X)):
            # Get k nearest neighbors
            nearest_idx = np.argsort(distances[i])[:k]
            nearest_dist = distances[i, nearest_idx]
            nearest_labels = self.y_train_[nearest_idx]

            # Distance-weighted voting
            weights = np.exp(-nearest_dist / (nearest_dist.mean() + 1e-6))
            weights /= weights.sum()

            for j, label in enumerate(nearest_labels):
                proba[i, label] += weights[j]

        return proba

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        class_indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(class_indices)

    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("TabPFNClassifier has not been fitted.")

    @staticmethod
    def check_constraints(X, y) -> dict:
        """Check if data meets TabPFN constraints.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        dict
            Dictionary with 'valid' bool and 'issues' list.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        issues = []

        if X.shape[0] > TabPFNClassifier.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > {TabPFNClassifier.MAX_SAMPLES}"
            )

        if X.shape[1] > TabPFNClassifier.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > {TabPFNClassifier.MAX_FEATURES}"
            )

        n_classes = len(np.unique(y))
        if n_classes > TabPFNClassifier.MAX_CLASSES:
            issues.append(
                f"Too many classes: {n_classes} > {TabPFNClassifier.MAX_CLASSES}"
            )

        return {
            "valid": len(issues) == 0,
            "issues": issues,
        }


# ---------------------------------------------------------------------------
# TabPFN v2 wrappers
# ---------------------------------------------------------------------------

def _check_tabpfn_v2_available():
    """Check if TabPFN v2 (tabpfn >= 2.0) is installed.

    Returns
    -------
    bool
        True if the tabpfn package exposes both TabPFNClassifier and
        TabPFNRegressor (v2 API).
    """
    try:
        from tabpfn import TabPFNClassifier as _Clf  # noqa: F401
        from tabpfn import TabPFNRegressor as _Reg
        return True
    except ImportError:
        return False


class TabPFNv2Classifier(ClassifierMixin, BaseEstimator):
    """TabPFN v2 classifier -- foundation model for tabular classification.

    Wraps the ``tabpfn`` v2 package (``tabpfn >= 2.0``) which supports native
    categorical handling, post-hoc ensembling, and relaxed dataset limits
    compared to the original TabPFN.

    When the ``tabpfn`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run (e.g. for testing
    or benchmarking without GPU).

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.  More estimators improve accuracy at the
        cost of inference time.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    categorical_features_indices : list of int or None, default=None
        Indices of categorical features.  ``None`` lets TabPFN v2 auto-detect.
    post_hoc_ensembling : bool, default=False
        If ``True``, enable post-hoc ensembling for improved accuracy.
    memory_saving_mode : bool, default=False
        If ``True``, reduce peak memory at the cost of slower inference.
        Useful for large datasets.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature limits.  Results
        may be less reliable outside the pre-training distribution.
    fit_mode : str or None, default=None
        Fit mode passed to the underlying TabPFN model.  For example
        ``'fit_with_cache'`` enables the KV cache for faster repeated
        predictions at the cost of extra memory.

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
    >>> from endgame.models.tabular.tabpfn import TabPFNv2Classifier
    >>> clf = TabPFNv2Classifier(n_estimators=16)
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
        categorical_features_indices: list[int] | None = None,
        post_hoc_ensembling: bool = False,
        memory_saving_mode: bool = False,
        ignore_pretraining_limits: bool = False,
        fit_mode: str | None = None,
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices
        self.post_hoc_ensembling = post_hoc_ensembling
        self.memory_saving_mode = memory_saving_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.fit_mode = fit_mode

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray, y: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if not self.ignore_pretraining_limits and n_samples > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN v2 supports a maximum of {self.MAX_SAMPLES} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if n_features > self.MAX_FEATURES:
            raise ValueError(
                f"TabPFN v2 supports a maximum of {self.MAX_FEATURES} features, "
                f"got {n_features}."
            )

        if n_classes > self.MAX_CLASSES:
            raise ValueError(
                f"TabPFN v2 supports a maximum of {self.MAX_CLASSES} classes, "
                f"got {n_classes}."
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

    def fit(self, X, y) -> "TabPFNv2Classifier":
        """Store training data and initialise the underlying TabPFN v2 model.

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

        self._validate_data_limits(X, y_encoded)

        # Store for fallback path
        self.X_train_ = X
        self.y_train_ = y_encoded

        # Try to initialise the real TabPFN v2 model
        self._model = None
        if _check_tabpfn_v2_available():
            from tabpfn import TabPFNClassifier as _TabPFNClf

            device = self._resolve_device()

            init_kwargs = dict(
                n_estimators=self.n_estimators,
                device=device,
                random_state=self.random_state,
            )
            if self.categorical_features_indices is not None:
                init_kwargs["categorical_features_indices"] = (
                    self.categorical_features_indices
                )
            if self.ignore_pretraining_limits:
                init_kwargs["ignore_pretraining_limits"] = True
            if self.memory_saving_mode:
                init_kwargs["memory_saving_mode"] = True
            if self.post_hoc_ensembling:
                init_kwargs["post_hoc_ensembling"] = True

            try:
                self._model = _TabPFNClf(**init_kwargs)
                fit_kwargs = {}
                if self.fit_mode is not None:
                    import inspect
                    _fit_accepted = set(inspect.signature(self._model.fit).parameters)
                    if "fit_mode" in _fit_accepted:
                        fit_kwargs["fit_mode"] = self.fit_mode
                self._model.fit(X, y_encoded, **fit_kwargs)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabPFN v2 classifier "
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
        """Distance-weighted kNN fallback when tabpfn is unavailable."""
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
        """Check whether data satisfies TabPFN v2 constraints.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        y = np.asarray(y)
        issues: list[str] = []

        if X.shape[0] > TabPFNv2Classifier.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabPFNv2Classifier.MAX_SAMPLES}"
            )
        if X.shape[1] > TabPFNv2Classifier.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabPFNv2Classifier.MAX_FEATURES}"
            )
        n_classes = len(np.unique(y))
        if n_classes > TabPFNv2Classifier.MAX_CLASSES:
            issues.append(
                f"Too many classes: {n_classes} > "
                f"{TabPFNv2Classifier.MAX_CLASSES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}


class TabPFNv2Regressor(RegressorMixin, BaseEstimator):
    """TabPFN v2 regressor -- foundation model for tabular regression.

    Wraps the ``tabpfn`` v2 package (``tabpfn >= 2.0``) which adds native
    regression support alongside classification.

    When the ``tabpfn`` package is not installed, a distance-weighted kNN
    fallback is used so that downstream code can still run.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    categorical_features_indices : list of int or None, default=None
        Indices of categorical features.  ``None`` lets TabPFN v2 auto-detect.
    post_hoc_ensembling : bool, default=False
        If ``True``, enable post-hoc ensembling for improved accuracy.
    memory_saving_mode : bool, default=False
        If ``True``, reduce peak memory at the cost of slower inference.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature limits.
    fit_mode : str or None, default=None
        Fit mode passed to the underlying TabPFN model (e.g.
        ``'fit_with_cache'``).

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
    >>> from endgame.models.tabular.tabpfn import TabPFNv2Regressor
    >>> reg = TabPFNv2Regressor(n_estimators=16)
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
        categorical_features_indices: list[int] | None = None,
        post_hoc_ensembling: bool = False,
        memory_saving_mode: bool = False,
        ignore_pretraining_limits: bool = False,
        fit_mode: str | None = None,
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices
        self.post_hoc_ensembling = post_hoc_ensembling
        self.memory_saving_mode = memory_saving_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.fit_mode = fit_mode

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape

        if not self.ignore_pretraining_limits and n_samples > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN v2 supports a maximum of {self.MAX_SAMPLES} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if n_features > self.MAX_FEATURES:
            raise ValueError(
                f"TabPFN v2 supports a maximum of {self.MAX_FEATURES} features, "
                f"got {n_features}."
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

    def fit(self, X, y) -> "TabPFNv2Regressor":
        """Store training data and initialise the underlying TabPFN v2 model.

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
        if _check_tabpfn_v2_available():
            from tabpfn import TabPFNRegressor as _TabPFNReg

            device = self._resolve_device()

            init_kwargs = dict(
                n_estimators=self.n_estimators,
                device=device,
                random_state=self.random_state,
            )
            if self.categorical_features_indices is not None:
                init_kwargs["categorical_features_indices"] = (
                    self.categorical_features_indices
                )
            if self.ignore_pretraining_limits:
                init_kwargs["ignore_pretraining_limits"] = True
            if self.memory_saving_mode:
                init_kwargs["memory_saving_mode"] = True
            if self.post_hoc_ensembling:
                init_kwargs["post_hoc_ensembling"] = True

            try:
                self._model = _TabPFNReg(**init_kwargs)
                fit_kwargs = {}
                if self.fit_mode is not None:
                    import inspect
                    _fit_accepted = set(inspect.signature(self._model.fit).parameters)
                    if "fit_mode" in _fit_accepted:
                        fit_kwargs["fit_mode"] = self.fit_mode
                self._model.fit(X, y, **fit_kwargs)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabPFN v2 regressor "
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
        """Distance-weighted kNN fallback when tabpfn is unavailable."""
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
        """Check whether data satisfies TabPFN v2 regression constraints.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        issues: list[str] = []

        if X.shape[0] > TabPFNv2Regressor.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabPFNv2Regressor.MAX_SAMPLES}"
            )
        if X.shape[1] > TabPFNv2Regressor.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabPFNv2Regressor.MAX_FEATURES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
# TabPFN v2.5 wrappers
# ---------------------------------------------------------------------------

def _check_tabpfn_25_available():
    """Check if TabPFN v2.5 (tabpfn >= 2.5) is installed.

    Returns
    -------
    bool
        True if the tabpfn package is installed at version >= 2.5.
    """
    try:
        import tabpfn
        from tabpfn import TabPFNClassifier as _Clf  # noqa: F401
        from tabpfn import TabPFNRegressor as _Reg
        version = getattr(tabpfn, "__version__", "0.0.0")
        major, minor = (int(x) for x in version.split(".")[:2])
        return (major, minor) >= (2, 5)
    except (ImportError, ValueError, AttributeError):
        return False


class TabPFN25Classifier(ClassifierMixin, BaseEstimator):
    """TabPFN v2.5 classifier -- foundation model for tabular classification.

    Wraps the ``tabpfn`` v2.5 package which scales to 50,000 training samples
    and 2,000 features, a 5x and 4x improvement over TabPFN v2 respectively.
    The default ``model_version='2.5_real'`` selects RealTabPFN-2.5, which is
    fine-tuned on real-world data for improved practical performance.

    When the ``tabpfn`` package (>= 2.5) is not installed, a distance-weighted
    kNN fallback is used so that downstream code can still run (e.g. for
    testing or benchmarking without GPU).

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.  More estimators improve accuracy at the
        cost of inference time.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    categorical_features_indices : list of int or None, default=None
        Indices of categorical features.  ``None`` lets TabPFN auto-detect.
    fit_mode : str, default='fit_with_cache'
        Fit mode passed to the underlying TabPFN model.  ``'fit_with_cache'``
        enables the KV cache for faster repeated predictions at the cost of
        extra memory.
    memory_saving_mode : bool, default=False
        If ``True``, reduce peak memory at the cost of slower inference.
        Useful for large datasets.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature limits.  Results
        may be less reliable outside the pre-training distribution.
    model_version : str, default='2.5_real'
        Which TabPFN 2.5 variant to use:
        - ``'2.5_real'``: RealTabPFN-2.5 fine-tuned on real data (recommended)
        - ``'2.5'``: Synthetic-only pre-trained variant

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels discovered during ``fit``.
    n_classes_ : int
        Number of classes.

    Limitations
    -----------
    - Max 50,000 training samples (unless ``ignore_pretraining_limits=True``)
    - Max 2,000 features
    - Max 10 classes

    Examples
    --------
    >>> from endgame.models.tabular.tabpfn import TabPFN25Classifier
    >>> clf = TabPFN25Classifier(n_estimators=16)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    """

    _estimator_type = "classifier"

    MAX_SAMPLES = 50_000
    MAX_FEATURES = 2_000
    MAX_CLASSES = 10

    def __init__(
        self,
        n_estimators: int = 8,
        device: str = "auto",
        random_state: int = 0,
        categorical_features_indices: list[int] | None = None,
        fit_mode: str = "fit_with_cache",
        memory_saving_mode: bool = False,
        ignore_pretraining_limits: bool = False,
        model_version: str = "2.5_real",
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.model_version = model_version

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray, y: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        if not self.ignore_pretraining_limits and n_samples > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN 2.5 supports a maximum of {self.MAX_SAMPLES} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if n_features > self.MAX_FEATURES:
            raise ValueError(
                f"TabPFN 2.5 supports a maximum of {self.MAX_FEATURES} features, "
                f"got {n_features}."
            )

        if n_classes > self.MAX_CLASSES:
            raise ValueError(
                f"TabPFN 2.5 supports a maximum of {self.MAX_CLASSES} classes, "
                f"got {n_classes}."
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

    def fit(self, X, y) -> "TabPFN25Classifier":
        """Store training data and initialise the underlying TabPFN 2.5 model.

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

        self._validate_data_limits(X, y_encoded)

        # Store for fallback path
        self.X_train_ = X
        self.y_train_ = y_encoded

        # Try to initialise the real TabPFN 2.5 model
        self._model = None
        if _check_tabpfn_25_available():
            import inspect

            from tabpfn import TabPFNClassifier as _TabPFNClf

            device = self._resolve_device()

            init_kwargs = dict(
                n_estimators=self.n_estimators,
                device=device,
                random_state=self.random_state,
            )

            # Only pass parameters the installed TabPFN version accepts
            _accepted = set(inspect.signature(_TabPFNClf.__init__).parameters)
            if "model_version" in _accepted:
                init_kwargs["model_version"] = self.model_version
            if self.categorical_features_indices is not None:
                init_kwargs["categorical_features_indices"] = (
                    self.categorical_features_indices
                )
            if self.ignore_pretraining_limits and "ignore_pretraining_limits" in _accepted:
                init_kwargs["ignore_pretraining_limits"] = True
            if self.memory_saving_mode and "memory_saving_mode" in _accepted:
                init_kwargs["memory_saving_mode"] = True

            try:
                self._model = _TabPFNClf(**init_kwargs)
                fit_kwargs = {}
                if self.fit_mode is not None:
                    _fit_accepted = set(inspect.signature(self._model.fit).parameters)
                    if "fit_mode" in _fit_accepted:
                        fit_kwargs["fit_mode"] = self.fit_mode
                self._model.fit(X, y_encoded, **fit_kwargs)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabPFN 2.5 classifier "
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
        """Distance-weighted kNN fallback when tabpfn is unavailable."""
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
        """Check whether data satisfies TabPFN 2.5 constraints.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        y = np.asarray(y)
        issues: list[str] = []

        if X.shape[0] > TabPFN25Classifier.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabPFN25Classifier.MAX_SAMPLES}"
            )
        if X.shape[1] > TabPFN25Classifier.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabPFN25Classifier.MAX_FEATURES}"
            )
        n_classes = len(np.unique(y))
        if n_classes > TabPFN25Classifier.MAX_CLASSES:
            issues.append(
                f"Too many classes: {n_classes} > "
                f"{TabPFN25Classifier.MAX_CLASSES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}


class TabPFN25Regressor(RegressorMixin, BaseEstimator):
    """TabPFN v2.5 regressor -- foundation model for tabular regression.

    Wraps the ``tabpfn`` v2.5 package which scales to 50,000 training samples
    and 2,000 features.  The default ``model_version='2.5_real'`` selects
    RealTabPFN-2.5, fine-tuned on real-world data.

    When the ``tabpfn`` package (>= 2.5) is not installed, a distance-weighted
    kNN fallback is used so that downstream code can still run.

    Parameters
    ----------
    n_estimators : int, default=8
        Number of ensemble members.
    device : str, default='auto'
        Computation device: ``'cuda'``, ``'cpu'``, or ``'auto'``.
    random_state : int, default=0
        Random seed for reproducibility.
    categorical_features_indices : list of int or None, default=None
        Indices of categorical features.  ``None`` lets TabPFN auto-detect.
    fit_mode : str, default='fit_with_cache'
        Fit mode passed to the underlying TabPFN model (e.g.
        ``'fit_with_cache'``).
    memory_saving_mode : bool, default=False
        If ``True``, reduce peak memory at the cost of slower inference.
    ignore_pretraining_limits : bool, default=False
        If ``True``, bypass the built-in sample / feature limits.
    model_version : str, default='2.5_real'
        Which TabPFN 2.5 variant to use:
        - ``'2.5_real'``: RealTabPFN-2.5 fine-tuned on real data (recommended)
        - ``'2.5'``: Synthetic-only pre-trained variant

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during ``fit``.

    Limitations
    -----------
    - Max 50,000 training samples (unless ``ignore_pretraining_limits=True``)
    - Max 2,000 features

    Examples
    --------
    >>> from endgame.models.tabular.tabpfn import TabPFN25Regressor
    >>> reg = TabPFN25Regressor(n_estimators=16)
    >>> reg.fit(X_train, y_train)
    >>> preds = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    MAX_SAMPLES = 50_000
    MAX_FEATURES = 2_000

    def __init__(
        self,
        n_estimators: int = 8,
        device: str = "auto",
        random_state: int = 0,
        categorical_features_indices: list[int] | None = None,
        fit_mode: str = "fit_with_cache",
        memory_saving_mode: bool = False,
        ignore_pretraining_limits: bool = False,
        model_version: str = "2.5_real",
    ):
        self.n_estimators = n_estimators
        self.device = device
        self.random_state = random_state
        self.categorical_features_indices = categorical_features_indices
        self.fit_mode = fit_mode
        self.memory_saving_mode = memory_saving_mode
        self.ignore_pretraining_limits = ignore_pretraining_limits
        self.model_version = model_version

    # ----- validation helpers ------------------------------------------------

    def _validate_data_limits(self, X: np.ndarray):
        """Raise ``ValueError`` when hard limits are exceeded."""
        n_samples, n_features = X.shape

        if not self.ignore_pretraining_limits and n_samples > self.MAX_SAMPLES:
            raise ValueError(
                f"TabPFN 2.5 supports a maximum of {self.MAX_SAMPLES} training "
                f"samples, got {n_samples}.  Set ignore_pretraining_limits=True "
                f"to override."
            )

        if n_features > self.MAX_FEATURES:
            raise ValueError(
                f"TabPFN 2.5 supports a maximum of {self.MAX_FEATURES} features, "
                f"got {n_features}."
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

    def fit(self, X, y) -> "TabPFN25Regressor":
        """Store training data and initialise the underlying TabPFN 2.5 model.

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
        if _check_tabpfn_25_available():
            import inspect

            from tabpfn import TabPFNRegressor as _TabPFNReg

            device = self._resolve_device()

            init_kwargs = dict(
                n_estimators=self.n_estimators,
                device=device,
                random_state=self.random_state,
            )

            _accepted = set(inspect.signature(_TabPFNReg.__init__).parameters)
            if "model_version" in _accepted:
                init_kwargs["model_version"] = self.model_version
            if self.categorical_features_indices is not None:
                init_kwargs["categorical_features_indices"] = (
                    self.categorical_features_indices
                )
            if self.ignore_pretraining_limits and "ignore_pretraining_limits" in _accepted:
                init_kwargs["ignore_pretraining_limits"] = True
            if self.memory_saving_mode and "memory_saving_mode" in _accepted:
                init_kwargs["memory_saving_mode"] = True

            try:
                self._model = _TabPFNReg(**init_kwargs)
                fit_kwargs = {}
                if self.fit_mode is not None:
                    _fit_accepted = set(inspect.signature(self._model.fit).parameters)
                    if "fit_mode" in _fit_accepted:
                        fit_kwargs["fit_mode"] = self.fit_mode
                self._model.fit(X, y, **fit_kwargs)
            except Exception as exc:
                warnings.warn(
                    f"Could not initialise TabPFN 2.5 regressor "
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
        """Distance-weighted kNN fallback when tabpfn is unavailable."""
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
        """Check whether data satisfies TabPFN 2.5 regression constraints.

        Returns
        -------
        dict
            ``{'valid': bool, 'issues': list[str]}``
        """
        X = np.asarray(X)
        issues: list[str] = []

        if X.shape[0] > TabPFN25Regressor.MAX_SAMPLES:
            issues.append(
                f"Too many samples: {X.shape[0]} > "
                f"{TabPFN25Regressor.MAX_SAMPLES}"
            )
        if X.shape[1] > TabPFN25Regressor.MAX_FEATURES:
            issues.append(
                f"Too many features: {X.shape[1]} > "
                f"{TabPFN25Regressor.MAX_FEATURES}"
            )

        return {"valid": len(issues) == 0, "issues": issues}
