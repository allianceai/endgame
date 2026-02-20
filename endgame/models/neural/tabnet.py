"""TabNet wrapper for tabular data.

TabNet is an attention-based architecture for tabular data that provides
interpretability through feature attention masks.

Reference: https://arxiv.org/abs/1908.07442
"""

from typing import Any

import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from endgame.core.base import EndgameEstimator

# TabNet imports (lazy loaded)
try:
    from pytorch_tabnet.tab_model import TabNetClassifier as _TabNetClassifier
    from pytorch_tabnet.tab_model import TabNetRegressor as _TabNetRegressor

    HAS_TABNET = True
except ImportError:
    HAS_TABNET = False


def _check_tabnet():
    """Check if pytorch-tabnet is available."""
    if not HAS_TABNET:
        raise ImportError(
            "pytorch-tabnet is required for TabNet models. "
            "Install with: pip install pytorch-tabnet"
        )


class _BaseTabNetWrapper(EndgameEstimator):
    """Base class for TabNet wrappers.

    Parameters
    ----------
    n_d : int, default=64
        Width of the decision prediction layer.
    n_a : int, default=64
        Width of the attention embedding for each mask.
    n_steps : int, default=5
        Number of steps in the architecture.
    gamma : float, default=1.5
        Coefficient for feature reusage in the masks.
    n_independent : int, default=2
        Number of independent GLU layers at each step.
    n_shared : int, default=2
        Number of shared GLU layers at each step.
    momentum : float, default=0.3
        Momentum for batch normalization.
    clip_value : float, optional
        Gradient clipping value.
    lambda_sparse : float, default=1e-4
        Coefficient for sparsity regularization.
    optimizer_fn : callable, optional
        Optimizer function (default: torch.optim.Adam).
    optimizer_params : dict, optional
        Parameters for optimizer (default: {"lr": 2e-2}).
    scheduler_fn : callable, optional
        Learning rate scheduler function.
    scheduler_params : dict, optional
        Parameters for scheduler.
    mask_type : str, default='sparsemax'
        Attention mask type: 'sparsemax' or 'entmax'.
    n_epochs : int, default=100
        Maximum number of training epochs.
    patience : int, default=15
        Early stopping patience.
    batch_size : int, default=1024
        Training batch size.
    virtual_batch_size : int, default=256
        Virtual batch size for Ghost Batch Normalization.
    device_name : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level (0, 1, or 2).
    """

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.3,
        clip_value: float | None = None,
        lambda_sparse: float = 1e-4,
        optimizer_fn: Any | None = None,
        optimizer_params: dict | None = None,
        scheduler_fn: Any | None = None,
        scheduler_params: dict | None = None,
        mask_type: str = "sparsemax",
        n_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 256,
        device_name: str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
    ):
        _check_tabnet()
        super().__init__(random_state=random_state, verbose=verbose > 0)

        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.n_independent = n_independent
        self.n_shared = n_shared
        self.momentum = momentum
        self.clip_value = clip_value
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        # Store None as-is to ensure sklearn clone works correctly
        self.optimizer_params = optimizer_params
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params
        self.mask_type = mask_type
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.virtual_batch_size = virtual_batch_size
        self.device_name = device_name
        self._verbose = verbose

        self.model_: Any | None = None
        self._feature_names: list[str] | None = None

    def _get_device_name(self) -> str:
        """Get device name."""
        if self.device_name == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device_name

    def _get_model_params(self) -> dict[str, Any]:
        """Get parameters for TabNet model."""
        # Use defaults for None values (but store None to ensure sklearn clone works)
        optimizer_params = self.optimizer_params if self.optimizer_params is not None else {"lr": 2e-2}
        scheduler_params = self.scheduler_params if self.scheduler_params is not None else {}

        params = {
            "n_d": self.n_d,
            "n_a": self.n_a,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "n_independent": self.n_independent,
            "n_shared": self.n_shared,
            "momentum": self.momentum,
            "lambda_sparse": self.lambda_sparse,
            "optimizer_params": optimizer_params,
            "mask_type": self.mask_type,
            "device_name": self._get_device_name(),
            "verbose": self._verbose,
            "seed": self.random_state or 0,
        }

        if self.clip_value is not None:
            params["clip_value"] = self.clip_value

        if self.optimizer_fn is not None:
            params["optimizer_fn"] = self.optimizer_fn

        if self.scheduler_fn is not None:
            params["scheduler_fn"] = self.scheduler_fn
            params["scheduler_params"] = scheduler_params

        if self.scheduler_fn is not None:
            params["scheduler_fn"] = self.scheduler_fn
            params["scheduler_params"] = self.scheduler_params

        return params

    def _store_feature_names(self, X):
        """Store feature names from input."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
                return
        except ImportError:
            pass

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                self._feature_names = list(X.columns)
                return
        except ImportError:
            pass

        X_arr = self._to_numpy(X)
        self._feature_names = [f"f{i}" for i in range(X_arr.shape[1])]

    def explain(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Get feature importance masks.

        Parameters
        ----------
        X : array-like
            Input data.

        Returns
        -------
        explain_matrix : ndarray of shape (n_samples, n_features)
            Aggregated feature importance for each sample.
        masks : ndarray of shape (n_steps, n_samples, n_features)
            Attention masks for each step.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.explain(X_arr)

    @property
    def feature_importances_(self) -> dict[str, float]:
        """Feature importance dictionary."""
        self._check_is_fitted()

        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
            return dict(zip(self._feature_names, importances))

        return {}


class TabNetClassifier(_BaseTabNetWrapper, ClassifierMixin):
    """TabNet classifier wrapper.

    Attention-based deep learning architecture for tabular classification
    with built-in feature selection and interpretability.

    Parameters
    ----------
    n_d : int, default=64
        Width of the decision prediction layer.
    n_a : int, default=64
        Width of the attention embedding.
    n_steps : int, default=5
        Number of decision steps.
    gamma : float, default=1.5
        Coefficient for feature reusage.
    n_independent : int, default=2
        Number of independent GLU layers.
    n_shared : int, default=2
        Number of shared GLU layers.
    momentum : float, default=0.3
        Batch normalization momentum.
    clip_value : float, optional
        Gradient clipping value.
    lambda_sparse : float, default=1e-4
        Sparsity regularization coefficient.
    optimizer_params : dict, optional
        Optimizer parameters.
    mask_type : str, default='sparsemax'
        Attention type: 'sparsemax' or 'entmax'.
    n_epochs : int, default=100
        Maximum training epochs.
    patience : int, default=15
        Early stopping patience.
    batch_size : int, default=1024
        Training batch size.
    virtual_batch_size : int, default=256
        Ghost Batch Normalization batch size.
    device_name : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : TabNetClassifier
        Fitted TabNet model.
    feature_importances_ : dict
        Feature importance dictionary.

    Examples
    --------
    >>> from endgame.models.neural import TabNetClassifier
    >>> clf = TabNetClassifier(n_steps=3, n_epochs=50)
    >>> clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> predictions = clf.predict(X_test)
    >>> proba = clf.predict_proba(X_test)
    >>> # Get feature importance masks
    >>> explain_matrix, masks = clf.explain(X_test)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 5,
        gamma: float = 1.5,
        n_independent: int = 2,
        n_shared: int = 2,
        momentum: float = 0.3,
        clip_value: float | None = None,
        lambda_sparse: float = 1e-4,
        optimizer_fn: Any | None = None,
        optimizer_params: dict | None = None,
        scheduler_fn: Any | None = None,
        scheduler_params: dict | None = None,
        mask_type: str = "sparsemax",
        n_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 1024,
        virtual_batch_size: int = 256,
        device_name: str = "auto",
        random_state: int | None = None,
        verbose: int = 0,
    ):
        super().__init__(
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum,
            clip_value=clip_value,
            lambda_sparse=lambda_sparse,
            optimizer_fn=optimizer_fn,
            optimizer_params=optimizer_params,
            scheduler_fn=scheduler_fn,
            scheduler_params=scheduler_params,
            mask_type=mask_type,
            n_epochs=n_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            device_name=device_name,
            random_state=random_state,
            verbose=verbose,
        )

        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self._label_encoder: LabelEncoder | None = None

    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator."""
        from sklearn.utils._tags import ClassifierTags, InputTags, Tags, TargetTags

        return Tags(
            estimator_type="classifier",
            target_tags=TargetTags(
                required=True, one_d_labels=False, two_d_labels=False,
                positive_only=False, multi_output=False, single_output=True,
            ),
            transformer_tags=None,
            classifier_tags=ClassifierTags(poor_score=False, multi_class=True, multi_label=False),
            regressor_tags=None,
            array_api_support=False,
            no_validation=False,
            non_deterministic=True,  # Neural networks are non-deterministic
            requires_fit=True,
            _skip_test=False,
            input_tags=InputTags(
                one_d_array=False, two_d_array=True, three_d_array=False,
                sparse=False, categorical=False, string=False, dict=False,
                positive_only=False, allow_nan=False, pairwise=False,
            ),
        )

    def fit(
        self,
        X,
        y,
        eval_set: list[tuple[Any, Any]] | None = None,
        eval_name: list[str] | None = None,
        eval_metric: list[str] | None = None,
        weights: int | np.ndarray | None = None,
        **fit_params,
    ) -> "TabNetClassifier":
        """Fit the classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
        eval_name : list of str, optional
            Names for evaluation sets.
        eval_metric : list of str, optional
            Evaluation metrics.
        weights : int or ndarray, optional
            Sample weights (0 for unweighted, 1 for balanced, or array).
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self
            Fitted classifier.
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y)

        self._store_feature_names(X)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y_arr)

        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Create model
        self.model_ = _TabNetClassifier(**self._get_model_params())

        # Prepare eval set
        if eval_set is not None:
            eval_set_processed = []
            for X_val, y_val in eval_set:
                X_val_arr = self._to_numpy(X_val)
                y_val_arr = self._label_encoder.transform(np.asarray(y_val))
                eval_set_processed.append((X_val_arr, y_val_arr))
            eval_set = eval_set_processed

        # Fit model
        fit_kwargs = {
            "X_train": X_arr,
            "y_train": y_encoded,
            "eval_set": eval_set,
            "eval_name": eval_name,
            "eval_metric": eval_metric,
            "max_epochs": self.n_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
        }
        # Only pass weights if not None (pytorch-tabnet doesn't handle None well)
        if weights is not None:
            fit_kwargs["weights"] = weights

        fit_kwargs.update(fit_params)
        self.model_.fit(**fit_kwargs)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like
            Input samples.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        predictions = self.model_.predict(X_arr)
        return self._label_encoder.inverse_transform(predictions)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Input samples.

        Returns
        -------
        ndarray
            Class probabilities.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.predict_proba(X_arr)


class TabNetRegressor(_BaseTabNetWrapper, RegressorMixin):
    """TabNet regressor wrapper.

    Attention-based deep learning architecture for tabular regression
    with built-in feature selection and interpretability.

    Parameters
    ----------
    n_d : int, default=64
        Width of the decision prediction layer.
    n_a : int, default=64
        Width of the attention embedding.
    n_steps : int, default=5
        Number of decision steps.
    gamma : float, default=1.5
        Coefficient for feature reusage.
    n_independent : int, default=2
        Number of independent GLU layers.
    n_shared : int, default=2
        Number of shared GLU layers.
    momentum : float, default=0.3
        Batch normalization momentum.
    clip_value : float, optional
        Gradient clipping value.
    lambda_sparse : float, default=1e-4
        Sparsity regularization coefficient.
    optimizer_params : dict, optional
        Optimizer parameters.
    mask_type : str, default='sparsemax'
        Attention type: 'sparsemax' or 'entmax'.
    n_epochs : int, default=100
        Maximum training epochs.
    patience : int, default=15
        Early stopping patience.
    batch_size : int, default=1024
        Training batch size.
    virtual_batch_size : int, default=256
        Ghost Batch Normalization batch size.
    device_name : str, default='auto'
        Device: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    model_ : TabNetRegressor
        Fitted TabNet model.
    feature_importances_ : dict
        Feature importance dictionary.

    Examples
    --------
    >>> from endgame.models.neural import TabNetRegressor
    >>> reg = TabNetRegressor(n_steps=3, n_epochs=50)
    >>> reg.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> predictions = reg.predict(X_test)
    >>> # Get feature importance masks
    >>> explain_matrix, masks = reg.explain(X_test)
    """

    _estimator_type = "regressor"

    def fit(
        self,
        X,
        y,
        eval_set: list[tuple[Any, Any]] | None = None,
        eval_name: list[str] | None = None,
        eval_metric: list[str] | None = None,
        weights: int | np.ndarray | None = None,
        **fit_params,
    ) -> "TabNetRegressor":
        """Fit the regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,) or (n_samples, n_targets)
            Target values.
        eval_set : list of (X, y) tuples, optional
            Validation sets for early stopping.
        eval_name : list of str, optional
            Names for evaluation sets.
        eval_metric : list of str, optional
            Evaluation metrics.
        weights : int or ndarray, optional
            Sample weights.
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self
            Fitted regressor.
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y)

        self._store_feature_names(X)

        # Ensure y is 2D
        if y_arr.ndim == 1:
            y_arr = y_arr.reshape(-1, 1)

        # Create model
        self.model_ = _TabNetRegressor(**self._get_model_params())

        # Prepare eval set
        if eval_set is not None:
            eval_set_processed = []
            for X_val, y_val in eval_set:
                X_val_arr = self._to_numpy(X_val)
                y_val_arr = np.asarray(y_val)
                if y_val_arr.ndim == 1:
                    y_val_arr = y_val_arr.reshape(-1, 1)
                eval_set_processed.append((X_val_arr, y_val_arr))
            eval_set = eval_set_processed

        # Fit model
        fit_kwargs = {
            "X_train": X_arr,
            "y_train": y_arr,
            "eval_set": eval_set,
            "eval_name": eval_name,
            "eval_metric": eval_metric,
            "max_epochs": self.n_epochs,
            "patience": self.patience,
            "batch_size": self.batch_size,
            "virtual_batch_size": self.virtual_batch_size,
        }
        # Only pass weights if not None (pytorch-tabnet doesn't handle None well)
        if weights is not None:
            fit_kwargs["weights"] = weights

        fit_kwargs.update(fit_params)
        self.model_.fit(**fit_kwargs)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like
            Input samples.

        Returns
        -------
        ndarray
            Predicted values.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        predictions = self.model_.predict(X_arr)

        # Squeeze if single target
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.ravel()

        return predictions
