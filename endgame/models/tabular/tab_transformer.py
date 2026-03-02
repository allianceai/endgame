from __future__ import annotations

"""TabTransformer: Tabular Data Modeling Using Contextual Embeddings.

TabTransformer uses Transformer-based attention mechanism to model
categorical features as contextual embeddings, while continuous features
are processed through a simple linear layer.

Key differences from FT-Transformer:
- TabTransformer: Only categorical features use Transformer (original paper)
- FT-Transformer: Both categorical AND continuous features use Transformer

References
----------
- Huang et al., "TabTransformer: Tabular Data Modeling Using Contextual
  Embeddings" (2020), arXiv:2012.06678
- pytorch-tabular documentation: https://pytorch-tabular.readthedocs.io/

Notes
-----
This module wraps pytorch-tabular's TabTransformer implementation with an
sklearn-compatible API for seamless integration with endgame pipelines.
"""

from typing import Any

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
    from pytorch_tabular.models import TabTransformerConfig
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


def _check_dependencies():
    """Check that required dependencies are installed."""
    if not HAS_TORCH:
        raise ImportError(
            "PyTorch is required for TabTransformer. "
            "Install with: pip install endgame-ml[tabular]"
        )
    if not HAS_PANDAS:
        raise ImportError(
            "pandas is required for TabTransformer. "
            "Install with: pip install endgame-ml[tabular]"
        )
    if not HAS_PYTORCH_TABULAR:
        raise ImportError(
            "pytorch-tabular is required for TabTransformer. "
            "Install with: pip install endgame-ml[tabular]"
        )


class TabTransformerClassifier(ClassifierMixin, BaseEstimator):
    """TabTransformer Classifier for tabular data.

    TabTransformer uses self-attention on categorical feature embeddings
    to create contextual representations, while continuous features are
    passed through a simple normalization layer.

    Parameters
    ----------
    input_embed_dim : int, default=32
        Embedding dimension for categorical features.
    num_heads : int, default=8
        Number of attention heads.
    num_attn_blocks : int, default=6
        Number of Transformer attention blocks.
    attn_dropout : float, default=0.1
        Dropout rate for attention layers.
    ff_dropout : float, default=0.1
        Dropout rate for feed-forward layers.
    share_embedding : bool, default=True
        Whether to add shared embedding across all categorical features.
    shared_embedding_fraction : float, default=0.25
        Fraction of embedding shared across categories.
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    batch_size : int, default=256
        Training batch size.
    max_epochs : int, default=100
        Maximum training epochs.
    early_stopping_patience : int, default=10
        Early stopping patience.
    gpu : int or None, default=None
        GPU index to use. None for CPU.
    verbose : bool, default=False
        Whether to show training progress.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of input features.
    feature_names_in_ : ndarray
        Names of input features.

    Examples
    --------
    >>> from endgame.models.tabular import TabTransformerClassifier
    >>> clf = TabTransformerClassifier(num_attn_blocks=4, max_epochs=50)
    >>> clf.fit(X_train, y_train, cat_cols=['category1', 'category2'])
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    TabTransformer is particularly effective when:
    1. You have many categorical features
    2. Categories have semantic relationships
    3. Feature interactions are important

    For datasets with mostly continuous features, consider FT-Transformer
    which applies attention to all features.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        input_embed_dim: int = 32,
        num_heads: int = 8,
        num_attn_blocks: int = 6,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        share_embedding: bool = True,
        shared_embedding_fraction: float = 0.25,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        gpu: int | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.input_embed_dim = input_embed_dim
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.share_embedding = share_embedding
        self.shared_embedding_fraction = shared_embedding_fraction
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gpu = gpu
        self.verbose = verbose
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.feature_names_in_: np.ndarray | None = None
        self._model: Any | None = None
        self._label_encoder: LabelEncoder | None = None
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X,
        y,
        cat_cols: list[str] | None = None,
        eval_set: tuple | None = None,
        **fit_params,
    ) -> TabTransformerClassifier:
        """Fit the TabTransformer classifier.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        cat_cols : list of str, optional
            Names of categorical columns. If None, will auto-detect.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.

        Returns
        -------
        self
        """
        _check_dependencies()

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"f{i}" for i in range(X.shape[1])]

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns.tolist())

        # Encode target
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Create target column
        target_col = "__target__"
        df = X.copy()
        df[target_col] = y_encoded

        # Detect categorical columns
        if cat_cols is not None:
            self._cat_cols = list(cat_cols)
        else:
            self._cat_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if target_col in self._cat_cols:
                self._cat_cols.remove(target_col)

        # Remaining columns are numeric
        self._num_cols = [
            c for c in X.columns if c not in self._cat_cols
        ]

        # Ensure categorical columns are string type
        for col in self._cat_cols:
            df[col] = df[col].astype(str)

        # Handle validation set with stratified split to avoid unseen labels
        if eval_set is not None:
            X_val, y_val = eval_set
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
                X_val.columns = [f"f{i}" for i in range(X_val.shape[1])]
            y_val_encoded = self._label_encoder.transform(y_val)
            df_val = X_val.copy()
            df_val[target_col] = y_val_encoded
            for col in self._cat_cols:
                df_val[col] = df_val[col].astype(str)
        else:
            from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
            rng_seed = self.random_state or 42
            try:
                sss = StratifiedShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=rng_seed,
                )
                train_idx, val_idx = next(sss.split(df, y_encoded))
            except ValueError:
                ss = ShuffleSplit(
                    n_splits=1, test_size=0.2, random_state=rng_seed,
                )
                train_idx, val_idx = next(ss.split(df))
            df_val = df.iloc[val_idx].reset_index(drop=True)
            df = df.iloc[train_idx].reset_index(drop=True)

        if not self.verbose:
            import logging
            logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)
            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
            logging.getLogger("lightning").setLevel(logging.ERROR)

        # Build configs
        data_config = DataConfig(
            target=[target_col],
            continuous_cols=self._num_cols or [],
            categorical_cols=self._cat_cols or [],
        )

        model_config = TabTransformerConfig(
            task="classification",
            input_embed_dim=self.input_embed_dim,
            num_heads=self.num_heads,
            num_attn_blocks=self.num_attn_blocks,
            attn_dropout=self.attn_dropout,
            share_embedding=self.share_embedding,
            shared_embedding_fraction=self.shared_embedding_fraction,
            learning_rate=self.learning_rate,
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            accelerator="gpu" if self.gpu is not None else "cpu",
            devices=1,
            progress_bar="rich" if self.verbose else "none",
            load_best=True,
            seed=self.random_state or 42,
        )

        optimizer_config = OptimizerConfig()

        self._model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            verbose=self.verbose,
        )

        self._model.fit(train=df, validation=df_val)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        if not self._is_fitted:
            raise RuntimeError("TabTransformerClassifier has not been fitted.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"f{i}" for i in range(X.shape[1])]

        for col in self._cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)

        preds = self._model.predict(X)
        # Get probability columns
        proba_cols = [c for c in preds.columns if c.endswith("_probability")]
        if proba_cols:
            return preds[proba_cols].values
        else:
            # Fallback: convert predictions to one-hot
            pred_col = preds.columns[0]
            y_pred = preds[pred_col].values.astype(int)
            proba = np.zeros((len(y_pred), self.n_classes_))
            proba[np.arange(len(y_pred)), y_pred] = 1.0
            return proba


class TabTransformerRegressor(RegressorMixin, BaseEstimator):
    """TabTransformer Regressor for tabular data.

    Parameters
    ----------
    input_embed_dim : int, default=32
        Embedding dimension for categorical features.
    num_heads : int, default=8
        Number of attention heads.
    num_attn_blocks : int, default=6
        Number of Transformer attention blocks.
    attn_dropout : float, default=0.1
        Dropout rate for attention layers.
    ff_dropout : float, default=0.1
        Dropout rate for feed-forward layers.
    share_embedding : bool, default=True
        Whether to add shared embedding across all categorical features.
    shared_embedding_fraction : float, default=0.25
        Fraction of embedding shared across categories.
    learning_rate : float, default=1e-3
        Learning rate for optimizer.
    batch_size : int, default=256
        Training batch size.
    max_epochs : int, default=100
        Maximum training epochs.
    early_stopping_patience : int, default=10
        Early stopping patience.
    gpu : int or None, default=None
        GPU index to use. None for CPU.
    verbose : bool, default=False
        Whether to show training progress.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Examples
    --------
    >>> from endgame.models.tabular import TabTransformerRegressor
    >>> reg = TabTransformerRegressor(num_attn_blocks=4, max_epochs=50)
    >>> reg.fit(X_train, y_train, cat_cols=['category1', 'category2'])
    >>> predictions = reg.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        input_embed_dim: int = 32,
        num_heads: int = 8,
        num_attn_blocks: int = 6,
        attn_dropout: float = 0.1,
        ff_dropout: float = 0.1,
        share_embedding: bool = True,
        shared_embedding_fraction: float = 0.25,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        max_epochs: int = 100,
        early_stopping_patience: int = 10,
        gpu: int | None = None,
        verbose: bool = False,
        random_state: int | None = None,
    ):
        self.input_embed_dim = input_embed_dim
        self.num_heads = num_heads
        self.num_attn_blocks = num_attn_blocks
        self.attn_dropout = attn_dropout
        self.ff_dropout = ff_dropout
        self.share_embedding = share_embedding
        self.shared_embedding_fraction = shared_embedding_fraction
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.gpu = gpu
        self.verbose = verbose
        self.random_state = random_state

        self.n_features_in_: int = 0
        self.feature_names_in_: np.ndarray | None = None
        self._model: Any | None = None
        self._cat_cols: list[str] = []
        self._num_cols: list[str] = []
        self._is_fitted: bool = False

    def fit(
        self,
        X,
        y,
        cat_cols: list[str] | None = None,
        eval_set: tuple | None = None,
        **fit_params,
    ) -> TabTransformerRegressor:
        """Fit the TabTransformer regressor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.
        cat_cols : list of str, optional
            Names of categorical columns. If None, will auto-detect.
        eval_set : tuple of (X_val, y_val), optional
            Validation set for early stopping.

        Returns
        -------
        self
        """
        _check_dependencies()

        # Convert to DataFrame if needed
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"f{i}" for i in range(X.shape[1])]

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.array(X.columns.tolist())

        # Normalize regression targets for stable training
        y_arr = np.asarray(y, dtype=np.float32)
        self._y_mean = float(y_arr.mean())
        self._y_std = float(y_arr.std()) or 1.0
        y_norm = (y_arr - self._y_mean) / self._y_std

        # Create target column
        target_col = "__target__"
        df = X.copy()
        df[target_col] = y_norm

        # Detect categorical columns
        if cat_cols is not None:
            self._cat_cols = list(cat_cols)
        else:
            self._cat_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()
            if target_col in self._cat_cols:
                self._cat_cols.remove(target_col)

        # Remaining columns are numeric
        self._num_cols = [
            c for c in X.columns if c not in self._cat_cols
        ]

        # Ensure categorical columns are string type
        for col in self._cat_cols:
            df[col] = df[col].astype(str)

        # Handle validation set
        if eval_set is not None:
            X_val, y_val = eval_set
            if not isinstance(X_val, pd.DataFrame):
                X_val = pd.DataFrame(X_val)
                X_val.columns = [f"f{i}" for i in range(X_val.shape[1])]
            df_val = X_val.copy()
            y_val_arr = np.asarray(y_val, dtype=np.float32)
            df_val[target_col] = (y_val_arr - self._y_mean) / self._y_std
            for col in self._cat_cols:
                df_val[col] = df_val[col].astype(str)
        else:
            from sklearn.model_selection import ShuffleSplit
            ss = ShuffleSplit(
                n_splits=1, test_size=0.2,
                random_state=self.random_state or 42,
            )
            train_idx, val_idx = next(ss.split(df))
            df_val = df.iloc[val_idx].reset_index(drop=True)
            df = df.iloc[train_idx].reset_index(drop=True)

        if not self.verbose:
            import logging
            logging.getLogger("pytorch_tabular").setLevel(logging.ERROR)
            logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
            logging.getLogger("lightning").setLevel(logging.ERROR)

        # Build configs
        data_config = DataConfig(
            target=[target_col],
            continuous_cols=self._num_cols or [],
            categorical_cols=self._cat_cols or [],
        )

        model_config = TabTransformerConfig(
            task="regression",
            input_embed_dim=self.input_embed_dim,
            num_heads=self.num_heads,
            num_attn_blocks=self.num_attn_blocks,
            attn_dropout=self.attn_dropout,
            share_embedding=self.share_embedding,
            shared_embedding_fraction=self.shared_embedding_fraction,
            learning_rate=self.learning_rate,
        )

        trainer_config = TrainerConfig(
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            early_stopping_patience=self.early_stopping_patience,
            accelerator="gpu" if self.gpu is not None else "cpu",
            devices=1,
            progress_bar="rich" if self.verbose else "none",
            load_best=True,
            seed=self.random_state or 42,
        )

        optimizer_config = OptimizerConfig()

        self._model = TabularModel(
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            optimizer_config=optimizer_config,
            verbose=self.verbose,
        )

        self._model.fit(train=df, validation=df_val)
        self._is_fitted = True

        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        if not self._is_fitted:
            raise RuntimeError("TabTransformerRegressor has not been fitted.")

        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            X.columns = [f"f{i}" for i in range(X.shape[1])]

        for col in self._cat_cols:
            if col in X.columns:
                X[col] = X[col].astype(str)

        preds = self._model.predict(X)
        pred_col = preds.columns[0]
        raw = preds[pred_col].values
        # Denormalize predictions
        return raw * self._y_std + self._y_mean
