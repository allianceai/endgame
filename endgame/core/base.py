from __future__ import annotations

"""Base classes for all Endgame estimators."""

from abc import abstractmethod
from typing import Any, Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

from endgame.core.polars_ops import (
    HAS_PANDAS,
    HAS_POLARS,
    detect_input_format,
    from_lazyframe,
    to_lazyframe,
)

if HAS_POLARS:
    import polars as pl
if HAS_PANDAS:
    import pandas as pd


class EndgameEstimator(BaseEstimator):
    """Base class for all Endgame estimators.

    Provides common functionality including:
    - Sklearn-compatible interface
    - Automatic input format handling
    - Consistent random state management
    - Verbose logging support

    Parameters
    ----------
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.random_state = random_state
        self.verbose = verbose
        self._is_fitted = False

    def _log(self, message: str, level: str = "info") -> None:
        """Log a message if verbose mode is enabled."""
        if self.verbose:
            prefix = {"info": "[INFO]", "warn": "[WARN]", "error": "[ERROR]"}.get(
                level, "[INFO]"
            )
            print(f"{prefix} {message}")

    def _check_is_fitted(self) -> None:
        """Check if the estimator has been fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )

    def _validate_data(
        self,
        X: Any,
        y: Any | None = None,
        reset: bool = True,
    ) -> tuple:
        """Validate and preprocess input data.

        Parameters
        ----------
        X : array-like
            Input features.
        y : array-like, optional
            Target variable.
        reset : bool, default=True
            Whether to reset internal state (during fit).

        Returns
        -------
        tuple
            Validated (X, y) or just X if y is None.
        """
        # Convert X to numpy if needed for sklearn compatibility
        X_validated = self._to_numpy(X)

        if reset:
            self._n_features_in = X_validated.shape[1] if X_validated.ndim > 1 else 1

        if y is not None:
            y_validated = np.asarray(y)
            return X_validated, y_validated

        return X_validated

    def _to_numpy(self, X: Any) -> np.ndarray:
        """Convert input to numpy array."""
        if isinstance(X, np.ndarray):
            return X

        if HAS_PANDAS and isinstance(X, (pd.DataFrame, pd.Series)):
            return X.values

        if HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
            if isinstance(X, pl.LazyFrame):
                X = X.collect()
            return X.to_numpy()

        return np.asarray(X)

    @property
    def is_fitted(self) -> bool:
        """Whether the estimator has been fitted."""
        return self._is_fitted

    def save(self, path: str, **kwargs) -> str:
        """Save this estimator to disk.

        Parameters
        ----------
        path : str
            Destination path.
        **kwargs
            Passed to :func:`endgame.persistence.save`.

        Returns
        -------
        str
            The actual path where the model was saved.
        """
        from endgame.persistence import save
        return save(self, path, **kwargs)

    @classmethod
    def load(cls, path: str, **kwargs):
        """Load an estimator from disk.

        Parameters
        ----------
        path : str
            Path to the saved model.
        **kwargs
            Passed to :func:`endgame.persistence.load`.

        Returns
        -------
        estimator
            The loaded estimator.
        """
        from endgame.persistence import load
        return load(path, **kwargs)


class EndgameClassifierMixin(ClassifierMixin):
    """Mixin class for Endgame classifiers."""

    _estimator_type = "classifier"

    def predict_proba(self, X) -> np.ndarray:
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
        raise NotImplementedError("predict_proba must be implemented by subclass")

    def predict(self, X) -> np.ndarray:
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
        return self.classes_[np.argmax(proba, axis=1)]


class EndgameRegressorMixin(RegressorMixin):
    """Mixin class for Endgame regressors."""

    _estimator_type = "regressor"


class PolarsTransformer(EndgameEstimator, TransformerMixin):
    """Base class for Polars-optimized transformers.

    Accepts: Polars DataFrame, Pandas DataFrame, or NumPy array
    Returns: Same format as input (configurable)
    Internal: Converts to pl.LazyFrame for optimized execution

    Parameters
    ----------
    output_format : str, default='auto'
        Output format: 'auto' (match input), 'polars', 'pandas', 'numpy'.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        output_format: Literal["auto", "polars", "pandas", "numpy"] = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.output_format = output_format
        self._input_format: str | None = None
        self._feature_names_in: list[str] | None = None

    def _to_lazyframe(
        self,
        X: Any,
        store_metadata: bool = False,
    ) -> pl.LazyFrame:
        """Convert input to Polars LazyFrame.

        Parameters
        ----------
        X : array-like or DataFrame
            Input data.
        store_metadata : bool, default=False
            Whether to store input format and column names.

        Returns
        -------
        pl.LazyFrame
            Polars LazyFrame.
        """
        if store_metadata:
            self._input_format = detect_input_format(X)

            # Store column names
            if HAS_PANDAS and isinstance(X, pd.DataFrame):
                self._feature_names_in = list(X.columns)
            elif HAS_POLARS and isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    self._feature_names_in = X.collect_schema().names()
                else:
                    self._feature_names_in = list(X.columns)

        return to_lazyframe(X, column_names=self._feature_names_in)

    def _from_lazyframe(
        self,
        lf: pl.LazyFrame,
        output_format: str | None = None,
    ) -> Any:
        """Convert LazyFrame back to requested format.

        Parameters
        ----------
        lf : pl.LazyFrame
            Polars LazyFrame.
        output_format : str, optional
            Override output format.

        Returns
        -------
        DataFrame or array
            Data in requested format.
        """
        if output_format is None:
            if self.output_format == "auto":
                output_format = self._input_format or "numpy"
            else:
                output_format = self.output_format

        return from_lazyframe(lf, output_format)

    @abstractmethod
    def fit(self, X, y=None, **fit_params):
        """Fit the transformer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        self
            Fitted transformer.
        """
        pass

    @abstractmethod
    def transform(self, X) -> Any:
        """Transform the data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        array-like
            Transformed data.
        """
        pass

    def fit_transform(self, X, y=None, **fit_params) -> Any:
        """Fit and transform in one step.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,), optional
            Target values.
        **fit_params : dict
            Additional fit parameters.

        Returns
        -------
        array-like
            Transformed data.
        """
        return self.fit(X, y, **fit_params).transform(X)

    @property
    def feature_names_in_(self) -> list[str] | None:
        """Input feature names."""
        return self._feature_names_in

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names.

        Parameters
        ----------
        input_features : List[str], optional
            Input feature names. If None, uses stored names.

        Returns
        -------
        List[str]
            Output feature names.
        """
        self._check_is_fitted()

        if input_features is None:
            input_features = self._feature_names_in

        # Default: return input features (subclasses override for new features)
        return input_features or []


class BaseEnsemble(EndgameEstimator):
    """Base class for ensemble methods.

    Parameters
    ----------
    estimators : List[estimator]
        List of base estimators.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.
    """

    def __init__(
        self,
        estimators: list[Any] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.estimators = estimators or []
        self.weights_: dict[int, float] | None = None

    def _validate_predictions(
        self,
        predictions: list[np.ndarray],
        y_true: np.ndarray | None = None,
    ) -> list[np.ndarray]:
        """Validate prediction arrays for ensemble.

        Parameters
        ----------
        predictions : List[np.ndarray]
            Predictions from each model.
        y_true : np.ndarray, optional
            True labels for validation.

        Returns
        -------
        List[np.ndarray]
            Validated predictions.
        """
        if len(predictions) == 0:
            raise ValueError("predictions list cannot be empty")

        # Convert to numpy arrays
        predictions = [np.asarray(p) for p in predictions]

        # Check shapes match
        n_samples = predictions[0].shape[0]
        for i, p in enumerate(predictions[1:], 1):
            if p.shape[0] != n_samples:
                raise ValueError(
                    f"Prediction {i} has {p.shape[0]} samples, "
                    f"expected {n_samples}"
                )

        # Validate against y_true if provided
        if y_true is not None:
            y_true = np.asarray(y_true)
            if y_true.shape[0] != n_samples:
                raise ValueError(
                    f"y_true has {y_true.shape[0]} samples, "
                    f"predictions have {n_samples}"
                )

        return predictions

    def _weighted_average(
        self,
        predictions: list[np.ndarray],
        weights: dict[int, float] | None = None,
    ) -> np.ndarray:
        """Compute weighted average of predictions.

        Parameters
        ----------
        predictions : List[np.ndarray]
            Predictions from each model.
        weights : Dict[int, float], optional
            Weight for each model. If None, uses uniform weights.

        Returns
        -------
        np.ndarray
            Weighted average predictions.
        """
        if weights is None:
            weights = {i: 1.0 / len(predictions) for i in range(len(predictions))}

        result = np.zeros_like(predictions[0], dtype=np.float64)
        total_weight = 0.0

        for idx, weight in weights.items():
            if weight > 0 and idx < len(predictions):
                result += weight * predictions[idx]
                total_weight += weight

        if total_weight > 0:
            result /= total_weight

        return result
