from __future__ import annotations

"""Unified wrappers for gradient boosting libraries."""

from typing import Any, Literal

import numpy as np

from endgame.core.base import EndgameEstimator
from endgame.core.config import get_preset

PresetName = Literal["endgame", "fast", "overfit", "custom"]


class GBDTWrapper(EndgameEstimator):
    """Unified interface for XGBoost, LightGBM, and CatBoost.

    Provides consistent API across gradient boosting frameworks with
    competition-tuned default parameters.

    Parameters
    ----------
    backend : str, default='lightgbm'
        Boosting library: 'xgboost', 'lightgbm', 'catboost'.
    task : str, default='auto'
        Task type: 'auto', 'classification', 'regression'.
    preset : str, default='endgame'
        Hyperparameter preset: 'endgame', 'fast', 'overfit', 'custom'.
    use_gpu : bool or str, default='auto'
        Enable GPU: True, False, or 'auto' (auto-detect).
    categorical_features : List[str], optional
        Columns to treat as categorical.
    early_stopping_rounds : int, default=100
        Early stopping patience.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    **kwargs
        Override preset parameters.

    Attributes
    ----------
    model_ : estimator
        Fitted underlying model.
    feature_importances_ : Dict[str, float]
        Feature importance dictionary.
    best_iteration_ : int
        Best iteration (with early stopping).

    Examples
    --------
    >>> from endgame.models import GBDTWrapper
    >>> model = GBDTWrapper(backend='lightgbm', preset='endgame')
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    >>> predictions = model.predict(X_test)
    """

    def __init__(
        self,
        backend: str = "lightgbm",
        task: str = "auto",
        preset: PresetName = "endgame",
        use_gpu: bool | str = "auto",
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.backend = backend
        self.task = task
        self.preset = preset
        self.use_gpu = use_gpu
        self.categorical_features = categorical_features
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs

        self.model_: Any | None = None
        self._feature_names: list[str] | None = None
        self._n_classes: int | None = None
        self._task_type: str | None = None
        self.classes_: np.ndarray | None = None  # For sklearn classifier compatibility

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters including kwargs for sklearn clone compatibility."""
        # Start with standard sklearn get_params
        params = super().get_params(deep=deep)
        # Add kwargs so they're preserved during clone
        params.update(self.kwargs)
        return params

    def set_params(self, **params) -> GBDTWrapper:
        """Set parameters including kwargs."""
        # Separate known params from kwargs
        known_params = {'backend', 'task', 'preset', 'use_gpu',
                       'categorical_features', 'early_stopping_rounds',
                       'random_state', 'verbose'}

        new_kwargs = {}
        for key, value in params.items():
            if key in known_params:
                setattr(self, key, value)
            else:
                new_kwargs[key] = value

        # Update kwargs
        self.kwargs.update(new_kwargs)
        return self

    def __sklearn_tags__(self):
        """Return sklearn tags for this estimator.

        This is required for sklearn 1.6+ compatibility to properly identify
        the estimator type (classifier vs regressor).
        """
        from sklearn.utils._tags import ClassifierTags, InputTags, RegressorTags, Tags, TargetTags

        # Determine estimator type
        if self.task == "classification":
            estimator_type = "classifier"
        elif self.task == "regression":
            estimator_type = "regressor"
        else:
            # Default to classifier for auto (most common use case)
            estimator_type = "classifier"

        # Build tags
        target_tags = TargetTags(
            required=True,
            one_d_labels=False,
            two_d_labels=False,
            positive_only=False,
            multi_output=False,
            single_output=True,
        )

        input_tags = InputTags(
            one_d_array=False,
            two_d_array=True,
            three_d_array=False,
            sparse=False,
            categorical=True,  # CatBoost supports categorical
            string=False,
            dict=False,
            positive_only=False,
            allow_nan=True,  # Most GBDT models handle NaN
            pairwise=False,
        )

        classifier_tags = ClassifierTags(
            poor_score=False,
            multi_class=True,
            multi_label=False,
        ) if estimator_type == "classifier" else None

        regressor_tags = RegressorTags(
            poor_score=False,
            multi_target=False,
        ) if estimator_type == "regressor" else None

        return Tags(
            estimator_type=estimator_type,
            target_tags=target_tags,
            transformer_tags=None,
            classifier_tags=classifier_tags,
            regressor_tags=regressor_tags,
            array_api_support=False,
            no_validation=False,
            non_deterministic=False,
            requires_fit=True,
            _skip_test=False,
            input_tags=input_tags,
        )

    def _detect_gpu(self) -> bool:
        """Detect if GPU is available."""
        if self.use_gpu is False:
            return False
        if self.use_gpu is True:
            return True

        # Auto-detect
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            pass

        try:
            import cupy
            return True
        except ImportError:
            pass

        return False

    def _get_params(self) -> dict[str, Any]:
        """Get merged parameters from preset and overrides."""
        params = get_preset(self.backend, self.preset)
        params.update(self.kwargs)

        # Set random state
        if self.random_state is not None:
            if self.backend == "lightgbm" or self.backend == "xgboost":
                params["random_state"] = self.random_state
            elif self.backend == "catboost":
                params["random_seed"] = self.random_state

        # Set verbose (ensure it's passed to underlying model)
        if self.backend == "lightgbm":
            params["verbosity"] = -1 if not self.verbose else 1
        elif self.backend == "xgboost":
            params["verbosity"] = 0 if not self.verbose else 1
        elif self.backend == "catboost":
            params["verbose"] = self.verbose

        # Handle GPU
        use_gpu = self._detect_gpu()
        if use_gpu:
            if self.backend == "lightgbm":
                params["device"] = "gpu"
            elif self.backend == "xgboost":
                # Check if XGBoost was built with GPU support
                try:
                    import xgboost as xgb
                    # Try to detect if GPU support is available
                    # XGBoost 2.0+ uses 'cuda' device, older versions use 'gpu_hist'
                    if hasattr(xgb, 'build_info') and callable(xgb.build_info):
                        build_info = xgb.build_info()
                        if build_info.get('USE_CUDA', False):
                            params["tree_method"] = "hist"
                            params["device"] = "cuda"
                    else:
                        # Older XGBoost - try gpu_hist but it may fail
                        # Better to fall back to hist to avoid runtime errors
                        params["tree_method"] = "hist"
                except Exception:
                    params["tree_method"] = "hist"
            elif self.backend == "catboost":
                params["task_type"] = "GPU"

        return params

    def _infer_task(self, y: np.ndarray) -> str:
        """Infer task type from target.

        Uses heuristics to determine if the target represents classification or regression:
        - If string/object dtype -> classification
        - If integer dtype with reasonable number of unique values -> classification
        - If float dtype with many unique values -> regression
        """
        if self.task != "auto":
            return self.task

        unique = np.unique(y)
        n_unique = len(unique)
        n_samples = len(y)

        # String or object type -> classification
        if y.dtype.kind in ('U', 'S', 'O'):
            return "classification"

        # Integer type with reasonable cardinality -> classification
        # Allow up to 100 classes or 10% of samples (whichever is larger)
        max_classes = max(100, int(n_samples * 0.1))
        if y.dtype.kind in ('i', 'u', 'b'):  # Integer, unsigned int, bool
            if n_unique <= max_classes:
                return "classification"

        # Float type but all values are actually integers -> classification
        if y.dtype.kind == 'f':
            if np.all(unique == unique.astype(int)) and n_unique <= max_classes:
                return "classification"

        return "regression"

    def _create_model(self, task: str) -> Any:
        """Create the underlying model."""
        params = self._get_params()

        if self.backend == "lightgbm":
            import lightgbm as lgb
            if task == "classification":
                return lgb.LGBMClassifier(**params)
            return lgb.LGBMRegressor(**params)

        elif self.backend == "xgboost":
            import xgboost as xgb
            if task == "classification":
                return xgb.XGBClassifier(**params)
            return xgb.XGBRegressor(**params)

        elif self.backend == "catboost":
            from catboost import CatBoostClassifier, CatBoostRegressor
            if task == "classification":
                return CatBoostClassifier(**params)
            return CatBoostRegressor(**params)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def fit(
        self,
        X,
        y,
        eval_set: list[tuple[Any, Any]] | None = None,
        sample_weight: np.ndarray | None = None,
        **fit_params,
    ) -> GBDTWrapper:
        """Fit the model.

        Parameters
        ----------
        X : array-like
            Training features.
        y : array-like
            Target values.
        eval_set : List[Tuple], optional
            Validation set(s) for early stopping.
        sample_weight : array-like, optional
            Sample weights.
        **fit_params
            Additional fit parameters.

        Returns
        -------
        self
        """
        y_arr = np.asarray(y)

        # Store feature names and prepare training data.
        # Keep DataFrames intact — all three backends handle them natively
        # and preserving column names prevents sklearn feature-name warnings.
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                self._feature_names = list(X.columns)
                X_fit = X
            else:
                X_fit = self._to_numpy(X)
        except ImportError:
            X_fit = self._to_numpy(X)

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                self._feature_names = list(X.columns)
                X_fit = self._to_numpy(X)  # polars not supported by backends
        except ImportError:
            pass

        if self._feature_names is None:
            n_cols = X_fit.shape[1] if hasattr(X_fit, "shape") else np.asarray(X_fit).shape[1]
            self._feature_names = [f"f{i}" for i in range(n_cols)]

        # Infer task
        self._task_type = self._infer_task(y_arr)
        self._label_remap = None
        if self._task_type == "classification":
            self.classes_ = np.unique(y_arr)
            self._n_classes = len(self.classes_)
            # Remap labels to contiguous 0..n-1 (required by XGBoost)
            if not np.array_equal(self.classes_, np.arange(self._n_classes)):
                self._label_remap = {c: i for i, c in enumerate(self.classes_)}
                y_arr = np.array([self._label_remap[v] for v in y_arr])

        # Create model
        self.model_ = self._create_model(self._task_type)

        # Prepare fit arguments
        fit_args = fit_params.copy()

        if sample_weight is not None:
            fit_args["sample_weight"] = sample_weight

        # Handle early stopping
        if eval_set is not None:
            eval_pairs = []
            for X_e, y_e in eval_set:
                y_e_arr = np.asarray(y_e)
                if self._label_remap is not None:
                    y_e_arr = np.array([self._label_remap.get(v, v) for v in y_e_arr])
                eval_pairs.append((self._match_format(X_e, X_fit), y_e_arr))
            prepared_eval_set = eval_pairs
            if self.backend == "lightgbm":
                fit_args["eval_set"] = prepared_eval_set
                fit_args["callbacks"] = [
                    self._get_lgb_early_stopping_callback()
                ]
            elif self.backend == "xgboost":
                fit_args["eval_set"] = prepared_eval_set
                fit_args["verbose"] = self.verbose
                # XGBoost 2.0+ requires early_stopping_rounds at construction
                # time; set it on the model directly before fitting
                self.model_.set_params(
                    early_stopping_rounds=self.early_stopping_rounds
                )
            elif self.backend == "catboost":
                fit_args["eval_set"] = prepared_eval_set
                fit_args["early_stopping_rounds"] = self.early_stopping_rounds

        # Handle categorical features
        if self.categorical_features and self.backend == "catboost":
            cat_indices = [
                self._feature_names.index(c)
                for c in self.categorical_features
                if c in self._feature_names
            ]
            fit_args["cat_features"] = cat_indices

        self._log(f"Training {self.backend} model with {len(X_fit)} samples...")
        self.model_.fit(X_fit, y_arr, **fit_args)

        self._is_fitted = True
        return self

    def _get_lgb_early_stopping_callback(self):
        """Get LightGBM early stopping callback."""
        import lightgbm as lgb
        return lgb.early_stopping(
            stopping_rounds=self.early_stopping_rounds,
            verbose=self.verbose,
        )

    def _to_model_input(self, X):
        """Convert X to the format the fitted model expects.

        If the model was fitted on a DataFrame (has ``feature_names_in_``),
        convert X to a DataFrame with matching column names so sklearn's
        feature-name validation is satisfied.  Otherwise return numpy.
        """
        import pandas as pd
        if isinstance(X, pd.DataFrame):
            return X
        fitted_names = getattr(self.model_, "feature_names_in_", None)
        if fitted_names is not None and not isinstance(X, pd.DataFrame):
            return pd.DataFrame(self._to_numpy(X), columns=fitted_names)
        return self._to_numpy(X)

    @staticmethod
    def _match_format(source, reference):
        """Ensure *source* has the same type/columns as *reference*."""
        import pandas as pd
        if isinstance(reference, pd.DataFrame):
            if isinstance(source, pd.DataFrame):
                return source
            return pd.DataFrame(
                np.asarray(source) if not isinstance(source, np.ndarray) else source,
                columns=reference.columns,
            )
        # reference is numpy — convert source to numpy too
        if isinstance(source, pd.DataFrame):
            return source.values
        return np.asarray(source) if not isinstance(source, np.ndarray) else source

    def predict(self, X) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like
            Features to predict.

        Returns
        -------
        ndarray
            Predictions.
        """
        self._check_is_fitted()
        X_in = self._to_model_input(X)
        preds = self.model_.predict(X_in)
        if self._label_remap is not None and self._task_type == "classification":
            preds = self.classes_[preds.astype(int)]
        return preds

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Features to predict.

        Returns
        -------
        ndarray
            Class probabilities.
        """
        self._check_is_fitted()

        if self._task_type != "classification":
            raise ValueError("predict_proba only available for classification")

        X_in = self._to_model_input(X)
        return self.model_.predict_proba(X_in)

    @property
    def feature_importances_(self) -> dict[str, float]:
        """Feature importance dictionary."""
        self._check_is_fitted()

        if hasattr(self.model_, "feature_importances_"):
            importances = self.model_.feature_importances_
            return dict(zip(self._feature_names, importances))

        return {}

    @property
    def best_iteration_(self) -> int | None:
        """Best iteration from early stopping."""
        self._check_is_fitted()

        if hasattr(self.model_, "best_iteration_"):
            return self.model_.best_iteration_
        if hasattr(self.model_, "best_iteration"):
            return self.model_.best_iteration

        return None

    def score(self, X, y, sample_weight=None) -> float:
        """Return the score on the given data.

        For classification, returns accuracy.
        For regression, returns R² score.

        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True labels or target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Score.
        """
        self._check_is_fitted()

        if self._task_type == "classification":
            from sklearn.metrics import accuracy_score
            y_pred = self.predict(X)
            return accuracy_score(y, y_pred, sample_weight=sample_weight)
        else:
            from sklearn.metrics import r2_score
            y_pred = self.predict(X)
            return r2_score(y, y_pred, sample_weight=sample_weight)


class LGBMWrapper(GBDTWrapper):
    """LightGBM-specific wrapper with additional features.

    Parameters
    ----------
    preset : str, default='endgame'
        Hyperparameter preset.
    task : str, default='auto'
        Task type: 'auto', 'classification', 'regression'.
    use_goss : bool, default=False
        Use Gradient-based One-Side Sampling.
    **kwargs
        Additional parameters.

    Examples
    --------
    >>> model = LGBMWrapper(preset='endgame')
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    """

    def __init__(
        self,
        preset: PresetName = "endgame",
        task: str = "auto",
        use_goss: bool = False,
        use_gpu: bool | str = "auto",
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if use_goss:
            kwargs["boosting_type"] = "goss"

        super().__init__(
            backend="lightgbm",
            task=task,
            preset=preset,
            use_gpu=use_gpu,
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.use_goss = use_goss


class XGBWrapper(GBDTWrapper):
    """XGBoost-specific wrapper with additional features.

    Parameters
    ----------
    preset : str, default='endgame'
        Hyperparameter preset.
    task : str, default='auto'
        Task type: 'auto', 'classification', 'regression'.
    use_dart : bool, default=False
        Use DART boosting.
    **kwargs
        Additional parameters.

    Examples
    --------
    >>> model = XGBWrapper(preset='endgame')
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    """

    def __init__(
        self,
        preset: PresetName = "endgame",
        task: str = "auto",
        use_dart: bool = False,
        use_gpu: bool | str = "auto",
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if use_dart:
            kwargs["booster"] = "dart"

        super().__init__(
            backend="xgboost",
            task=task,
            preset=preset,
            use_gpu=use_gpu,
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.use_dart = use_dart


class CatBoostWrapper(GBDTWrapper):
    """CatBoost-specific wrapper with native categorical handling.

    Parameters
    ----------
    preset : str, default='endgame'
        Hyperparameter preset.
    task : str, default='auto'
        Task type: 'auto', 'classification', 'regression'.
    auto_class_weights : str, optional
        Auto class weighting: 'Balanced', 'SqrtBalanced'.
    **kwargs
        Additional parameters.

    Examples
    --------
    >>> model = CatBoostWrapper(preset='endgame', categorical_features=['category'])
    >>> model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    """

    def __init__(
        self,
        preset: PresetName = "endgame",
        task: str = "auto",
        auto_class_weights: str | None = None,
        use_gpu: bool | str = "auto",
        categorical_features: list[str] | None = None,
        early_stopping_rounds: int = 100,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        if auto_class_weights:
            kwargs["auto_class_weights"] = auto_class_weights

        super().__init__(
            backend="catboost",
            task=task,
            preset=preset,
            use_gpu=use_gpu,
            categorical_features=categorical_features,
            early_stopping_rounds=early_stopping_rounds,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.auto_class_weights = auto_class_weights
