from __future__ import annotations

"""Automatic preprocessing selection and pipeline building.

This module provides intelligent preprocessing selection based on
data characteristics and meta-features.
"""

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
)

logger = logging.getLogger(__name__)


@dataclass
class PreprocessorStep:
    """A preprocessing step configuration.

    Attributes
    ----------
    name : str
        Step name.
    step_type : str
        Type of preprocessing (imputer, encoder, scaler, etc.).
    params : dict
        Parameters for the step.
    columns : list, optional
        Columns to apply this step to.
    priority : int
        Execution priority (lower = earlier).
    """

    name: str
    step_type: str
    params: dict[str, Any] = field(default_factory=dict)
    columns: list[str] | None = None
    priority: int = 0


class AutoPreprocessor:
    """Automatic preprocessing pipeline builder.

    Analyzes data characteristics and builds an appropriate preprocessing
    pipeline based on meta-features and model requirements.

    Parameters
    ----------
    handle_missing : bool, default=True
        Whether to handle missing values.
    handle_categorical : bool, default=True
        Whether to encode categorical features.
    scale_features : bool, default=True
        Whether to scale numeric features.
    handle_outliers : bool, default=False
        Whether to handle outliers.
    feature_selection : bool, default=False
        Whether to perform feature selection.
    max_cardinality : int, default=50
        Maximum cardinality for one-hot encoding.
    verbose : int, default=0
        Verbosity level.

    Attributes
    ----------
    pipeline_ : Pipeline
        Fitted preprocessing pipeline.
    column_types_ : dict
        Detected column types.
    steps_ : list
        Applied preprocessing steps.

    Examples
    --------
    >>> preprocessor = AutoPreprocessor()
    >>> pipeline = preprocessor.build_pipeline(X, y, meta_features)
    >>> X_processed = pipeline.fit_transform(X, y)
    """

    def __init__(
        self,
        handle_missing: bool = True,
        handle_categorical: bool = True,
        scale_features: bool = True,
        handle_outliers: bool = False,
        feature_selection: bool = False,
        max_cardinality: int = 50,
        verbose: int = 0,
    ):
        self.handle_missing = handle_missing
        self.handle_categorical = handle_categorical
        self.scale_features = scale_features
        self.handle_outliers = handle_outliers
        self.feature_selection = feature_selection
        self.max_cardinality = max_cardinality
        self.verbose = verbose

        # State
        self.pipeline_: Pipeline | None = None
        self.column_types_: dict[str, list[str]] = {}
        self.steps_: list[PreprocessorStep] = []

    def build_pipeline(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
        meta_features: dict[str, float] | None = None,
        model_name: str | None = None,
    ) -> Pipeline:
        """Build a preprocessing pipeline for the data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : array-like, optional
            Target values.
        meta_features : dict, optional
            Dataset meta-features.
        model_name : str, optional
            Target model name (affects preprocessing choices).

        Returns
        -------
        Pipeline
            Preprocessing pipeline.
        """
        if meta_features is None:
            meta_features = {}

        # Detect column types
        self.column_types_ = self._detect_column_types(X)

        # Determine preprocessing steps
        self.steps_ = self._determine_steps(X, y, meta_features, model_name)

        # Build pipeline
        self.pipeline_ = self._build_sklearn_pipeline(X)

        if self.verbose > 0:
            print(f"Built preprocessing pipeline with {len(self.steps_)} steps")
            for step in self.steps_:
                print(f"  - {step.name}: {step.step_type}")

        return self.pipeline_

    def _detect_column_types(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> dict[str, list[str]]:
        """Detect column types in the data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.

        Returns
        -------
        dict
            Mapping of type to column names.
        """
        types = {
            "numeric": [],
            "categorical": [],
            "boolean": [],
            "datetime": [],
            "text": [],
        }

        if isinstance(X, np.ndarray):
            # For numpy arrays, assume all numeric
            types["numeric"] = list(range(X.shape[1]))
            return types

        for col in X.columns:
            dtype = X[col].dtype

            if pd.api.types.is_bool_dtype(dtype):
                types["boolean"].append(col)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                types["datetime"].append(col)
            elif pd.api.types.is_numeric_dtype(dtype):
                # Check if it's actually categorical (few unique values)
                n_unique = X[col].nunique()
                if n_unique <= 10 and n_unique < len(X) * 0.05:
                    types["categorical"].append(col)
                else:
                    types["numeric"].append(col)
            elif dtype == object:  # noqa: E721
                # Check if it's text or categorical
                sample = X[col].dropna().head(100)
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    n_unique = X[col].nunique()

                    if avg_len > 50 or n_unique > self.max_cardinality:
                        types["text"].append(col)
                    else:
                        types["categorical"].append(col)
            else:
                types["categorical"].append(col)

        return types

    def _determine_steps(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None,
        meta_features: dict[str, float],
        model_name: str | None,
    ) -> list[PreprocessorStep]:
        """Determine preprocessing steps based on data characteristics.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : array-like, optional
            Target values.
        meta_features : dict
            Dataset meta-features.
        model_name : str, optional
            Target model name.

        Returns
        -------
        list of PreprocessorStep
            Preprocessing steps to apply.
        """
        steps = []

        # Check model requirements
        needs_encoding = self._model_needs_encoding(model_name)
        needs_scaling = self._model_needs_scaling(model_name)
        handles_missing = self._model_handles_missing(model_name)

        # 1. Handle missing values
        if self.handle_missing and not handles_missing:
            pct_missing = meta_features.get("pct_missing", 0)
            if isinstance(X, pd.DataFrame):
                pct_missing = X.isna().mean().mean()

            if pct_missing > 0:
                # Choose imputation strategy
                if pct_missing < 0.05 or pct_missing < 0.2:
                    strategy = "median"
                else:
                    strategy = "constant"

                steps.append(PreprocessorStep(
                    name="numeric_imputer",
                    step_type="imputer",
                    params={"strategy": strategy, "fill_value": -999 if strategy == "constant" else None},
                    columns=self.column_types_["numeric"],
                    priority=0,
                ))

                if self.column_types_["categorical"]:
                    steps.append(PreprocessorStep(
                        name="categorical_imputer",
                        step_type="imputer",
                        params={"strategy": "constant", "fill_value": "missing"},
                        columns=self.column_types_["categorical"],
                        priority=0,
                    ))

        # 2. Handle categorical encoding
        if self.handle_categorical and needs_encoding and self.column_types_["categorical"]:
            for col in self.column_types_["categorical"]:
                n_unique = X[col].nunique() if isinstance(X, pd.DataFrame) else 10

                if n_unique <= 10:
                    # One-hot for low cardinality
                    encoding = "onehot"
                elif n_unique <= self.max_cardinality:
                    # Ordinal for medium cardinality
                    encoding = "ordinal"
                else:
                    # Target encoding for high cardinality
                    encoding = "target"

                steps.append(PreprocessorStep(
                    name=f"encoder_{col}",
                    step_type="encoder",
                    params={"method": encoding},
                    columns=[col],
                    priority=1,
                ))

        # 3. Scale numeric features
        if self.scale_features and needs_scaling and self.column_types_["numeric"]:
            # Choose scaler based on data characteristics
            if self.handle_outliers:
                scaler = "robust"
            else:
                scaler = "standard"

            steps.append(PreprocessorStep(
                name="scaler",
                step_type="scaler",
                params={"method": scaler},
                columns=self.column_types_["numeric"],
                priority=2,
            ))

        # 4. Handle datetime features
        if self.column_types_["datetime"]:
            steps.append(PreprocessorStep(
                name="datetime_encoder",
                step_type="datetime",
                params={"extract": ["year", "month", "day", "dayofweek"]},
                columns=self.column_types_["datetime"],
                priority=1,
            ))

        # Sort by priority
        steps.sort(key=lambda s: s.priority)

        return steps

    def _model_needs_encoding(self, model_name: str | None) -> bool:
        """Check if model needs categorical encoding.

        Parameters
        ----------
        model_name : str, optional
            Model name.

        Returns
        -------
        bool
            Whether encoding is needed.
        """
        if model_name is None:
            return True

        # Models that handle categoricals natively
        native_categorical = {"catboost", "lgbm", "xgb", "c50", "ebm"}
        return model_name not in native_categorical

    def _model_needs_scaling(self, model_name: str | None) -> bool:
        """Check if model needs feature scaling.

        Parameters
        ----------
        model_name : str, optional
            Model name.

        Returns
        -------
        bool
            Whether scaling is needed.
        """
        if model_name is None:
            return True

        # Models that need scaling
        needs_scaling = {
            "linear", "svm", "knn", "mlp", "ft_transformer",
            "saint", "tabnet", "node", "gp",
        }
        return model_name in needs_scaling

    def _model_handles_missing(self, model_name: str | None) -> bool:
        """Check if model handles missing values natively.

        Parameters
        ----------
        model_name : str, optional
            Model name.

        Returns
        -------
        bool
            Whether model handles missing values.
        """
        if model_name is None:
            return False

        # Models that handle missing natively
        handles_missing = {"lgbm", "xgb", "catboost"}
        return model_name in handles_missing

    def _build_sklearn_pipeline(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> Pipeline:
        """Build sklearn Pipeline from steps.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.

        Returns
        -------
        Pipeline
            Sklearn pipeline.
        """
        if not self.steps_:
            # Return identity pipeline
            return Pipeline([("passthrough", "passthrough")])

        # Group steps by column transformer or regular transformer
        transformers = []
        pipeline_steps = []

        # Process imputers
        imputer_steps = [s for s in self.steps_ if s.step_type == "imputer"]
        if imputer_steps:
            imputer_transformers = []
            for step in imputer_steps:
                transformer = SimpleImputer(
                    strategy=step.params.get("strategy", "median"),
                    fill_value=step.params.get("fill_value"),
                )
                if step.columns:
                    imputer_transformers.append((step.name, transformer, step.columns))

            if imputer_transformers:
                ct = ColumnTransformer(
                    imputer_transformers,
                    remainder="passthrough",
                    sparse_threshold=0,
                )
                pipeline_steps.append(("imputation", ct))

        # Process encoders
        encoder_steps = [s for s in self.steps_ if s.step_type == "encoder"]
        if encoder_steps:
            encoder_transformers = []
            for step in encoder_steps:
                method = step.params.get("method", "onehot")

                if method == "onehot":
                    transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                elif method == "ordinal":
                    transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                else:
                    # Default to ordinal for now (target encoding would need custom impl)
                    transformer = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

                if step.columns:
                    encoder_transformers.append((step.name, transformer, step.columns))

            if encoder_transformers:
                # Get remaining columns
                encoded_cols = set()
                for _, _, cols in encoder_transformers:
                    encoded_cols.update(cols)

                ct = ColumnTransformer(
                    encoder_transformers,
                    remainder="passthrough",
                    sparse_threshold=0,
                )
                pipeline_steps.append(("encoding", ct))

        # Process scalers
        scaler_steps = [s for s in self.steps_ if s.step_type == "scaler"]
        if scaler_steps:
            for step in scaler_steps:
                method = step.params.get("method", "standard")

                if method == "standard":
                    scaler = StandardScaler()
                elif method == "minmax":
                    scaler = MinMaxScaler()
                elif method == "robust":
                    scaler = RobustScaler()
                else:
                    scaler = StandardScaler()

                pipeline_steps.append(("scaling", scaler))

        if not pipeline_steps:
            return Pipeline([("passthrough", "passthrough")])

        return Pipeline(pipeline_steps)

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> AutoPreprocessor:
        """Fit the preprocessor.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : array-like, optional
            Target values.

        Returns
        -------
        self
        """
        if self.pipeline_ is None:
            self.build_pipeline(X, y)

        self.pipeline_.fit(X, y)
        return self

    def transform(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        """Transform data using the fitted pipeline.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.

        Returns
        -------
        ndarray
            Transformed features.
        """
        if self.pipeline_ is None:
            raise ValueError("Preprocessor not fitted. Call fit() first.")

        return self.pipeline_.transform(X)

    def fit_transform(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray | None = None,
    ) -> np.ndarray:
        """Fit and transform data.

        Parameters
        ----------
        X : DataFrame or ndarray
            Feature matrix.
        y : array-like, optional
            Target values.

        Returns
        -------
        ndarray
            Transformed features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_feature_names(self) -> list[str]:
        """Get output feature names.

        Returns
        -------
        list of str
            Feature names after transformation.
        """
        if self.pipeline_ is None:
            return []

        # Try to get feature names from the pipeline
        try:
            return list(self.pipeline_.get_feature_names_out())
        except Exception:
            return []
