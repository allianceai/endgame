from __future__ import annotations

"""GANDALF: Gated Adaptive Network for Deep Automated Learning of Features.

GANDALF is a high-performance, interpretable deep learning architecture for
tabular data. It uses Gated Feature Learning Units (GFLUs) with a gating
mechanism and built-in feature selection.

References
----------
- Joseph & Raj, "GANDALF: Gated Adaptive Network for Deep Automated Learning
  of Features" (2022)
- pytorch-tabular documentation: https://pytorch-tabular.readthedocs.io/

Notes
-----
This module wraps pytorch-tabular's GANDALF implementation with an sklearn-
compatible API for seamless integration with endgame pipelines.
"""

from typing import Any, Union

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    from pytorch_tabular import TabularModel
    from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig
    from pytorch_tabular.models import GANDALFConfig
    HAS_PYTORCH_TABULAR = True

    # Fix for PyTorch 2.6+ which defaults torch.load to weights_only=True.
    # pytorch-tabular checkpoints contain omegaconf objects that require
    # weights_only=False. Patch torch.load's default for compatibility.
    import functools as _functools
    _original_torch_load = torch.load

    @_functools.wraps(_original_torch_load)
    def _patched_torch_load(*args, **kwargs):
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_torch_load(*args, **kwargs)

    torch.load = _patched_torch_load

except ImportError:
    HAS_PYTORCH_TABULAR = False


def _suppress_lightning_logging():
    """Silence all pytorch-tabular / Lightning console output."""
    import logging
    for name in ("pytorch_tabular", "lightning", "lightning.pytorch",
                 "lightning.fabric", "lightning.fabric.utilities.seed",
                 "lightning.pytorch.utilities.rank_zero",
                 "lightning.pytorch.accelerators.cuda"):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        import lightning.pytorch.utilities.rank_zero as _rz_mod
        from lightning.pytorch.utilities.rank_zero import rank_zero_only as _rzo
        _rz_mod.rank_zero_info = lambda *a, **kw: None
        import lightning.fabric.utilities.rank_zero as _rz_fab
        _rz_fab.rank_zero_info = lambda *a, **kw: None
    except (ImportError, AttributeError):
        pass


def _check_dependencies():
    """Check that required dependencies are installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for GANDALF. Install with: pip install torch"
        )
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for GANDALF. Install with: pip install pandas"
        )
    if not HAS_PYTORCH_TABULAR:
        raise ImportError(
            "pytorch-tabular is required for GANDALF. "
            "Install with: pip install pytorch-tabular"
        )


class GANDALFClassifier(ClassifierMixin, BaseEstimator):
    """GANDALF for classification.

    Gated Adaptive Network for Deep Automated Learning of Features (GANDALF)
    is a neural network architecture that uses Gated Feature Learning Units
    (GFLUs) for representation learning with built-in feature selection.

    This is a wrapper around pytorch-tabular's GANDALF implementation with
    an sklearn-compatible interface.

    Parameters
    ----------
    gflu_stages : int, default=6
        Number of layers in the feature abstraction layer. More stages
        allow for more complex feature interactions but increase risk
        of overfitting.
    gflu_dropout : float, default=0.0
        Dropout rate for the feature abstraction layer. Use 0.1-0.3
        for regularization on smaller datasets.
    gflu_feature_init_sparsity : float, default=0.3
        Initial percentage of features to select in each GFLU stage.
        Lower values encourage sparsity/interpretability.
    learnable_sparsity : bool, default=True
        If True, sparsity parameters will be learned during training.
        If False, fixed to initial values.
    embedding_dropout : float, default=0.0
        Dropout applied to categorical embeddings.
    learning_rate : float, default=1e-3
        Initial learning rate. GANDALF often works well with 1e-3 to 1e-2.
    batch_size : int, default=256
        Training batch size.
    n_epochs : int, default=100
        Maximum number of training epochs.
    early_stopping_patience : int, default=10
        Number of epochs to wait for improvement before stopping.
    validation_fraction : float, default=0.2
        Fraction of training data for validation when eval_set not provided.
    cat_features : List[str], optional
        List of categorical feature column names.
    continuous_features : List[str], optional
        List of continuous feature column names. If None, inferred.
    device : str, default='auto'
        Device for training: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose training output.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    model_ : TabularModel
        Fitted pytorch-tabular model.
    feature_importances_ : ndarray
        Feature importance scores (if available).
    history_ : dict
        Training history with losses.

    Examples
    --------
    >>> from endgame.models.tabular import GANDALFClassifier
    >>> clf = GANDALFClassifier(gflu_stages=6, n_epochs=50)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)
    >>> preds = clf.predict(X_test)

    Notes
    -----
    GANDALF typically works well out-of-the-box for most tabular datasets.
    Key tuning considerations:

    - Start with default gflu_stages=6, reduce if overfitting
    - Use dropout (0.1-0.3) for small datasets
    - Learning rates 1e-3 to 1e-2 often work well
    - Lower feature_init_sparsity for more interpretability
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        gflu_stages: int = 6,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        embedding_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_fraction: float = 0.2,
        cat_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.embedding_dropout = embedding_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.cat_features = cat_features
        self.continuous_features = continuous_features
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.model_: TabularModel | None = None
        self._label_encoder: LabelEncoder | None = None
        self._feature_names: list[str] | None = None
        self._cat_features: list[str] | None = None
        self._cont_features: list[str] | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[GANDALF] {message}")

    def _get_device(self) -> str:
        """Get accelerator string for pytorch-tabular."""
        if self.device == "auto":
            return "auto"
        elif self.device == "cuda":
            return "gpu"
        else:
            return "cpu"

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        """Prepare data for pytorch-tabular.

        Parameters
        ----------
        X : array-like or DataFrame
            Feature data.
        y : array-like, optional
            Target labels.
        fit : bool
            Whether this is for fitting (determines feature detection).

        Returns
        -------
        pd.DataFrame
            DataFrame ready for pytorch-tabular.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if fit:
                self._feature_names = list(df.columns)
        else:
            X = np.asarray(X)
            if fit:
                self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self._feature_names)

        if y is not None:
            df["target"] = y

        return df

    def _infer_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Infer categorical and continuous feature types.

        Parameters
        ----------
        df : pd.DataFrame
            Training data.

        Returns
        -------
        Tuple[List[str], List[str]]
            (categorical_features, continuous_features)
        """
        if self.cat_features is not None and self.continuous_features is not None:
            return self.cat_features, self.continuous_features

        cat_cols = []
        cont_cols = []

        for col in self._feature_names:
            if col == "target":
                continue

            if self.cat_features is not None and col in self.cat_features:
                cat_cols.append(col)
            elif self.continuous_features is not None and col in self.continuous_features:
                cont_cols.append(col)
            elif df[col].dtype == "object" or df[col].dtype.name == "category":
                cat_cols.append(col)
            elif df[col].nunique() <= 10 and df[col].dtype in ["int64", "int32"]:
                # Treat low-cardinality integers as categorical
                cat_cols.append(col)
            else:
                cont_cols.append(col)

        return cat_cols, cont_cols

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> GANDALFClassifier:
        """Fit the GANDALF classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training labels.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.
        **fit_params : dict
            Additional parameters (unused, for API compatibility).

        Returns
        -------
        self
            Fitted classifier.
        """
        _check_dependencies()

        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        # Encode labels
        y = np.asarray(y)
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Prepare training data
        train_df = self._prepare_data(X, y_encoded, fit=True)

        # Infer feature types
        self._cat_features, self._cont_features = self._infer_feature_types(train_df)

        self._log(f"Categorical features: {len(self._cat_features)}")
        self._log(f"Continuous features: {len(self._cont_features)}")

        # Prepare validation data with stratified split to avoid unseen labels.
        # Falls back to random split when classes have too few samples for
        # stratification (e.g. a class with only 1 sample).
        val_df = None
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val_encoded = self._label_encoder.transform(np.asarray(y_val))
            val_df = self._prepare_data(X_val, y_val_encoded, fit=False)
        elif self.validation_fraction and self.validation_fraction > 0:
            from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
            rng_seed = self.random_state or 42
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=self.validation_fraction,
                    random_state=rng_seed,
                )
                train_idx, val_idx = next(sss.split(train_df, y_encoded))
            except ValueError:
                ss = ShuffleSplit(
                    n_splits=1, test_size=self.validation_fraction,
                    random_state=rng_seed,
                )
                train_idx, val_idx = next(ss.split(train_df))
            val_df = train_df.iloc[val_idx].reset_index(drop=True)
            train_df = train_df.iloc[train_idx].reset_index(drop=True)

        if not self.verbose:
            _suppress_lightning_logging()

        # Configure pytorch-tabular
        data_config = DataConfig(
            target=["target"],
            continuous_cols=self._cont_features if self._cont_features else [],
            categorical_cols=self._cat_features if self._cat_features else [],
            validation_split=None,
            num_workers=0,
        )

        model_config = GANDALFConfig(
            task="classification",
            gflu_stages=self.gflu_stages,
            gflu_dropout=self.gflu_dropout,
            gflu_feature_init_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
            embedding_dropout=self.embedding_dropout,
            learning_rate=self.learning_rate,
            seed=self.random_state or 42,
        )

        _trainer_kw = {}
        if not self.verbose:
            _trainer_kw["enable_model_summary"] = False

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            early_stopping="valid_loss",
            early_stopping_patience=self.early_stopping_patience,
            accelerator=self._get_device(),
            devices=1,
            auto_select_gpus=True,
            progress_bar="none" if not self.verbose else "rich",
            load_best=True,
            trainer_kwargs=_trainer_kw,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr_scheduler="CosineAnnealingWarmRestarts",
            lr_scheduler_params={"T_0": 10, "eta_min": 1e-6},
        )

        # Create and train model
        self.model_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            verbose=self.verbose,
        )

        self._log(f"Training on {self._get_device()}...")
        self._log(f"Training samples: {len(train_df)}")

        self.model_.fit(
            train=train_df,
            validation=val_df,
            seed=self.random_state or 42,
        )

        self._is_fitted = True
        return self

    def predict_proba(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("GANDALFClassifier has not been fitted.")

        test_df = self._prepare_data(X, fit=False)

        # Get predictions from pytorch-tabular
        preds = self.model_.predict(test_df)

        # Extract probability columns
        prob_cols = [col for col in preds.columns if col.endswith("_probability")]

        if prob_cols:
            proba = preds[prob_cols].values
        else:
            # Fallback: use prediction column with softmax
            pred_col = [col for col in preds.columns if "prediction" in col.lower()]
            if pred_col:
                # For binary classification, might need to construct probabilities
                raw_preds = preds[pred_col[0]].values
                if self.n_classes_ == 2:
                    proba = np.column_stack([1 - raw_preds, raw_preds])
                else:
                    proba = np.zeros((len(raw_preds), self.n_classes_))
                    proba[np.arange(len(raw_preds)), raw_preds.astype(int)] = 1.0
            else:
                raise ValueError("Could not extract probabilities from model output")

        return proba

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Feature importance scores (if available).

        Returns
        -------
        ndarray or None
            Feature importances from the fitted model.
        """
        if not self._is_fitted or self.model_ is None:
            return None

        # GANDALF doesn't have direct feature importances in the same way
        # as tree models, but we could potentially extract attention weights
        # For now, return None
        return None


class GANDALFRegressor(BaseEstimator, RegressorMixin):
    """GANDALF for regression.

    Gated Adaptive Network for Deep Automated Learning of Features (GANDALF)
    for regression tasks. Uses Gated Feature Learning Units (GFLUs) for
    representation learning with built-in feature selection.

    Parameters
    ----------
    gflu_stages : int, default=6
        Number of layers in the feature abstraction layer.
    gflu_dropout : float, default=0.0
        Dropout rate for the feature abstraction layer.
    gflu_feature_init_sparsity : float, default=0.3
        Initial percentage of features to select in each GFLU stage.
    learnable_sparsity : bool, default=True
        If True, sparsity parameters will be learned during training.
    embedding_dropout : float, default=0.0
        Dropout applied to categorical embeddings.
    learning_rate : float, default=1e-3
        Initial learning rate.
    batch_size : int, default=256
        Training batch size.
    n_epochs : int, default=100
        Maximum number of training epochs.
    early_stopping_patience : int, default=10
        Number of epochs to wait for improvement before stopping.
    validation_fraction : float, default=0.2
        Fraction of training data for validation when eval_set not provided.
    target_range : Tuple[float, float], optional
        Target range (min, max) for output scaling. Helps model learn
        bounded outputs.
    cat_features : List[str], optional
        List of categorical feature column names.
    continuous_features : List[str], optional
        List of continuous feature column names.
    device : str, default='auto'
        Device for training: 'cuda', 'cpu', or 'auto'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose training output.

    Attributes
    ----------
    model_ : TabularModel
        Fitted pytorch-tabular model.
    history_ : dict
        Training history with losses.

    Examples
    --------
    >>> from endgame.models.tabular import GANDALFRegressor
    >>> reg = GANDALFRegressor(gflu_stages=6, n_epochs=50)
    >>> reg.fit(X_train, y_train)
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        gflu_stages: int = 6,
        gflu_dropout: float = 0.0,
        gflu_feature_init_sparsity: float = 0.3,
        learnable_sparsity: bool = True,
        embedding_dropout: float = 0.0,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        n_epochs: int = 100,
        early_stopping_patience: int = 10,
        validation_fraction: float = 0.2,
        target_range: tuple[float, float] | None = None,
        cat_features: list[str] | None = None,
        continuous_features: list[str] | None = None,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.gflu_stages = gflu_stages
        self.gflu_dropout = gflu_dropout
        self.gflu_feature_init_sparsity = gflu_feature_init_sparsity
        self.learnable_sparsity = learnable_sparsity
        self.embedding_dropout = embedding_dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.validation_fraction = validation_fraction
        self.target_range = target_range
        self.cat_features = cat_features
        self.continuous_features = continuous_features
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        # Fitted attributes
        self.model_: TabularModel | None = None
        self._feature_names: list[str] | None = None
        self._cat_features: list[str] | None = None
        self._cont_features: list[str] | None = None
        self.history_: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        self._is_fitted: bool = False

    def _log(self, message: str) -> None:
        """Log message if verbose."""
        if self.verbose:
            print(f"[GANDALF] {message}")

    def _get_device(self) -> str:
        """Get accelerator string for pytorch-tabular."""
        if self.device == "auto":
            return "auto"
        elif self.device == "cuda":
            return "gpu"
        else:
            return "cpu"

    def _prepare_data(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray | None = None,
        fit: bool = False,
    ) -> pd.DataFrame:
        """Prepare data for pytorch-tabular."""
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if fit:
                self._feature_names = list(df.columns)
        else:
            X = np.asarray(X)
            if fit:
                self._feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=self._feature_names)

        if y is not None:
            df["target"] = y

        return df

    def _infer_feature_types(self, df: pd.DataFrame) -> tuple[list[str], list[str]]:
        """Infer categorical and continuous feature types."""
        if self.cat_features is not None and self.continuous_features is not None:
            return self.cat_features, self.continuous_features

        cat_cols = []
        cont_cols = []

        for col in self._feature_names:
            if col == "target":
                continue

            if self.cat_features is not None and col in self.cat_features:
                cat_cols.append(col)
            elif self.continuous_features is not None and col in self.continuous_features:
                cont_cols.append(col)
            elif df[col].dtype == "object" or df[col].dtype.name == "category" or df[col].nunique() <= 10 and df[col].dtype in ["int64", "int32"]:
                cat_cols.append(col)
            else:
                cont_cols.append(col)

        return cat_cols, cont_cols

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        eval_set: tuple[Any, Any] | None = None,
        **fit_params,
    ) -> GANDALFRegressor:
        """Fit the GANDALF regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Training targets.
        eval_set : tuple, optional
            Validation set (X_val, y_val) for early stopping.
        **fit_params : dict
            Additional parameters (unused).

        Returns
        -------
        self
            Fitted regressor.
        """
        _check_dependencies()

        # Set random seed
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)

        y = np.asarray(y).astype(np.float32)

        # Normalize regression targets for stable training
        self._y_mean = float(y.mean())
        self._y_std = float(y.std()) or 1.0
        y = (y - self._y_mean) / self._y_std

        # Prepare training data
        train_df = self._prepare_data(X, y, fit=True)

        # Infer feature types
        self._cat_features, self._cont_features = self._infer_feature_types(train_df)

        self._log(f"Categorical features: {len(self._cat_features)}")
        self._log(f"Continuous features: {len(self._cont_features)}")

        # Prepare validation data
        val_df = None
        if eval_set is not None:
            X_val, y_val = eval_set
            y_val = np.asarray(y_val).astype(np.float32)
            y_val = (y_val - self._y_mean) / self._y_std
            val_df = self._prepare_data(X_val, y_val, fit=False)

        # Configure target range if specified (normalize to match targets)
        target_range_config = None
        if self.target_range is not None:
            lo = (self.target_range[0] - self._y_mean) / self._y_std
            hi = (self.target_range[1] - self._y_mean) / self._y_std
            target_range_config = [[lo, hi]]

        # Suppress Lightning logging when not verbose
        if not self.verbose:
            _suppress_lightning_logging()

        # Configure pytorch-tabular
        data_config = DataConfig(
            target=["target"],
            continuous_cols=self._cont_features if self._cont_features else [],
            categorical_cols=self._cat_features if self._cat_features else [],
            validation_split=self.validation_fraction if eval_set is None else None,
            num_workers=0,
        )

        model_config = GANDALFConfig(
            task="regression",
            gflu_stages=self.gflu_stages,
            gflu_dropout=self.gflu_dropout,
            gflu_feature_init_sparsity=self.gflu_feature_init_sparsity,
            learnable_sparsity=self.learnable_sparsity,
            embedding_dropout=self.embedding_dropout,
            learning_rate=self.learning_rate,
            target_range=target_range_config,
            seed=self.random_state or 42,
        )

        _trainer_kw = {}
        if not self.verbose:
            _trainer_kw["enable_model_summary"] = False

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.n_epochs,
            early_stopping="valid_loss",
            early_stopping_patience=self.early_stopping_patience,
            accelerator=self._get_device(),
            devices=1,
            auto_select_gpus=True,
            progress_bar="none" if not self.verbose else "rich",
            load_best=True,
            trainer_kwargs=_trainer_kw,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            lr_scheduler="CosineAnnealingWarmRestarts",
            lr_scheduler_params={"T_0": 10, "eta_min": 1e-6},
        )

        # Create and train model
        self.model_ = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            verbose=self.verbose,
        )

        self._log(f"Training on {self._get_device()}...")
        self._log(f"Training samples: {len(train_df)}")

        self.model_.fit(
            train=train_df,
            validation=val_df,
            seed=self.random_state or 42,
        )

        self._is_fitted = True
        return self

    def predict(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted target values.
        """
        if not self._is_fitted:
            raise RuntimeError("GANDALFRegressor has not been fitted.")

        test_df = self._prepare_data(X, fit=False)

        # Get predictions from pytorch-tabular
        preds = self.model_.predict(test_df)

        # Extract prediction column — explicit name first, then fallback
        if "__target___prediction" in preds.columns:
            raw = preds["__target___prediction"].values
        elif "target_prediction" in preds.columns:
            raw = preds["target_prediction"].values
        else:
            # Fallback: find column ending with "_prediction"
            pred_cols = [col for col in preds.columns if col.endswith("_prediction")]
            if pred_cols:
                raw = preds[pred_cols[0]].values
            else:
                # Last resort: first numeric column
                raw = None
                for col in preds.columns:
                    if preds[col].dtype in [np.float32, np.float64, np.int32, np.int64]:
                        raw = preds[col].values
                        break
                if raw is None:
                    raise ValueError("Could not extract predictions from model output")

        # Denormalize predictions
        return raw * self._y_std + self._y_mean

    @property
    def feature_importances_(self) -> np.ndarray | None:
        """Feature importance scores (if available)."""
        if not self._is_fitted or self.model_ is None:
            return None
        return None
