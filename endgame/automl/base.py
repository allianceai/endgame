"""Base classes for AutoML predictors.

This module defines the base predictor class that all domain-specific
predictors inherit from.
"""

import logging
import pickle
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

import numpy as np
import pandas as pd

from endgame.automl.presets import PresetConfig, get_preset
from endgame.automl.time_manager import TimeBudgetManager
from endgame.core.base import EndgameEstimator

if TYPE_CHECKING:
    from endgame.tracking.base import ExperimentLogger

logger = logging.getLogger(__name__)


# Type aliases
DataInput = Union[str, Path, pd.DataFrame, np.ndarray, dict[str, Any]]


@dataclass
class FitSummary:
    """Summary of the fitting process.

    Attributes
    ----------
    total_time : float
        Total training time in seconds.
    n_models_trained : int
        Number of models trained.
    n_models_failed : int
        Number of models that failed.
    best_model : str
        Name of the best model.
    best_score : float
        Score of the best model.
    cv_score : float
        Cross-validation score of the ensemble.
    stage_times : dict
        Time spent in each pipeline stage.
    """

    total_time: float = 0.0
    n_models_trained: int = 0
    n_models_failed: int = 0
    best_model: str = ""
    best_score: float = 0.0
    cv_score: float = 0.0
    stage_times: dict[str, float] = field(default_factory=dict)


class BasePredictor(EndgameEstimator, ABC):
    """Base class for all AutoML predictors.

    This class provides the common interface and functionality for
    domain-specific predictors (Tabular, Vision, Text, etc.).

    Parameters
    ----------
    label : str
        Name of the target column.
    problem_type : str, default="auto"
        Type of problem: "classification", "regression", "multiclass", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "good_quality",
        "medium_quality", "fast", "interpretable".
    time_limit : int, optional
        Time limit in seconds. If None, uses preset default.
    search_strategy : str, default="portfolio"
        Search strategy: "portfolio", "heuristic", "genetic", "random", "bayesian".
    track_experiments : bool, default=True
        Whether to track experiments to the meta-learning database.
    output_path : str, optional
        Path to save outputs (models, logs, etc.).
    random_state : int, default=42
        Random seed for reproducibility.
    verbosity : int, default=2
        Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).
    logger : ExperimentLogger, optional
        Experiment logger for tracking params, metrics, and artifacts.
        When provided, fit() automatically logs training configuration
        and results. When None (default), no tracking overhead is added.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the predictor has been fitted.
    fit_summary_ : FitSummary
        Summary of the fitting process.
    problem_type_ : str
        Detected or specified problem type.
    classes_ : np.ndarray
        Class labels for classification problems.
    feature_names_ : list of str
        Names of input features.
    """

    def __init__(
        self,
        label: str,
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        search_strategy: str = "portfolio",
        track_experiments: bool = True,
        output_path: str | None = None,
        random_state: int = 42,
        verbosity: int = 2,
        logger: "ExperimentLogger | None" = None,
    ):
        super().__init__(random_state=random_state, verbose=verbosity > 0)

        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.presets = presets
        self.time_limit = time_limit
        self.search_strategy = search_strategy
        self.track_experiments = track_experiments
        self.output_path = output_path
        self.verbosity = verbosity
        self.logger = logger

        # Load preset configuration
        self._preset_config: PresetConfig = get_preset(presets)

        # Set time limit from preset if not specified
        if self.time_limit is None:
            self.time_limit = self._preset_config.default_time_limit

        # Initialize state
        self.is_fitted_ = False
        self.fit_summary_: FitSummary | None = None
        self.problem_type_: str | None = None
        self.classes_: np.ndarray | None = None
        self.feature_names_: list[str] | None = None

        # Internal components (set during fit)
        self._models: dict[str, Any] = {}
        self._ensemble: Any | None = None
        self._preprocessor: Any | None = None
        self._calibrator: Any | None = None
        self._time_manager: TimeBudgetManager | None = None

    @abstractmethod
    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> "BasePredictor":
        """Fit the AutoML predictor.

        Parameters
        ----------
        train_data : str, Path, DataFrame, or ndarray
            Training data. Can be a file path, DataFrame, or array.
        tuning_data : optional
            Validation/tuning data. If None, uses internal holdout.
        time_limit : int, optional
            Override the time limit.
        presets : str, optional
            Override the preset.
        hyperparameters : dict, optional
            Override hyperparameters for specific models.
        **kwargs
            Additional arguments.

        Returns
        -------
        BasePredictor
            The fitted predictor.
        """
        pass

    @abstractmethod
    def predict(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data to predict on.
        model : str, optional
            Specific model to use. If None, uses the ensemble.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        pass

    def predict_proba(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions (classification only).

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Probability predictions with shape (n_samples, n_classes).
        """
        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression")
        # Default implementation - subclasses should override
        raise NotImplementedError("Subclass must implement predict_proba")

    def evaluate(
        self,
        data: DataInput,
        metrics: list[str] | None = None,
        silent: bool = False,
    ) -> dict[str, float]:
        """Evaluate the predictor on data.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Data to evaluate on. Must contain the target column.
        metrics : list of str, optional
            Metrics to compute. If None, uses default metrics.
        silent : bool, default=False
            Whether to suppress output.

        Returns
        -------
        dict
            Dictionary mapping metric names to scores.
        """
        self._check_is_fitted()

        # Load and prepare data
        X, y = self._load_and_prepare_data(data, for_prediction=False)

        # Get predictions
        y_pred = self.predict(X)

        # Default metrics
        if metrics is None:
            if self.problem_type_ in ("classification", "binary", "multiclass"):
                metrics = ["accuracy", "roc_auc", "f1"]
            else:
                metrics = ["rmse", "r2", "mae"]

        # Compute metrics
        from sklearn import metrics as sklearn_metrics

        results = {}
        for metric_name in metrics:
            try:
                if metric_name == "accuracy":
                    results[metric_name] = sklearn_metrics.accuracy_score(y, y_pred)
                elif metric_name == "roc_auc":
                    if hasattr(self, "predict_proba"):
                        y_proba = self.predict_proba(X)
                        if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                            y_proba = y_proba[:, 1]
                        results[metric_name] = sklearn_metrics.roc_auc_score(
                            y, y_proba, multi_class="ovr"
                        )
                elif metric_name == "f1":
                    results[metric_name] = sklearn_metrics.f1_score(
                        y, y_pred, average="weighted"
                    )
                elif metric_name == "rmse":
                    results[metric_name] = np.sqrt(
                        sklearn_metrics.mean_squared_error(y, y_pred)
                    )
                elif metric_name == "r2":
                    results[metric_name] = sklearn_metrics.r2_score(y, y_pred)
                elif metric_name == "mae":
                    results[metric_name] = sklearn_metrics.mean_absolute_error(y, y_pred)
                else:
                    logger.warning(f"Unknown metric: {metric_name}")
            except Exception as e:
                logger.warning(f"Could not compute {metric_name}: {e}")
                results[metric_name] = np.nan

        if not silent and self.verbosity > 0:
            print("\nEvaluation Results:")
            for metric_name, score in results.items():
                print(f"  {metric_name}: {score:.4f}")

            if self.problem_type_ in ("classification", "binary", "multiclass"):
                try:
                    from sklearn.metrics import classification_report
                    print("\n  Classification Report:")
                    print(classification_report(y, y_pred, digits=4))
                except Exception:
                    pass

        return results

    def leaderboard(
        self,
        extra_info: bool = False,
        silent: bool = False,
    ) -> pd.DataFrame:
        """Get the model leaderboard.

        Parameters
        ----------
        extra_info : bool, default=False
            Whether to include extra information (fit time, etc.).
        silent : bool, default=False
            Whether to suppress output.

        Returns
        -------
        pd.DataFrame
            Leaderboard with model names and scores.
        """
        self._check_is_fitted()

        rows = []
        for name, model_info in self._models.items():
            row = {
                "model": name,
                "score": model_info.get("score", np.nan),
            }

            if extra_info:
                row["fit_time"] = model_info.get("fit_time", np.nan)
                row["predict_time"] = model_info.get("predict_time", np.nan)
                row["n_features"] = model_info.get("n_features", np.nan)

            rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values("score", ascending=False).reset_index(drop=True)

        if not silent and self.verbosity > 0:
            print(df.to_string(index=False))

        return df

    def feature_importance(
        self,
        model: str | None = None,
        importance_type: str = "split",
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Parameters
        ----------
        model : str, optional
            Specific model. If None, uses best model.
        importance_type : str, default="split"
            Type of importance: "split", "gain", "permutation".

        Returns
        -------
        pd.DataFrame
            Feature importance scores.
        """
        self._check_is_fitted()

        if model is None:
            # Use best model
            if self.fit_summary_ and self.fit_summary_.best_model:
                model = self.fit_summary_.best_model
            else:
                model = list(self._models.keys())[0]

        if model not in self._models:
            raise ValueError(f"Model '{model}' not found")

        estimator = self._models[model].get("estimator")
        if estimator is None:
            raise ValueError(f"No estimator found for model '{model}'")

        # Try to get feature importances
        if hasattr(estimator, "feature_importances_"):
            importances = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            importances = np.abs(estimator.coef_).flatten()
        else:
            raise ValueError(f"Model '{model}' does not provide feature importances")

        # Build dataframe
        if self.feature_names_ is not None:
            names = self.feature_names_
        else:
            names = [f"feature_{i}" for i in range(len(importances))]

        df = pd.DataFrame({"feature": names, "importance": importances})
        df = df.sort_values("importance", ascending=False).reset_index(drop=True)

        return df

    def save(self, path: str | None = None) -> str:
        """Save the predictor to disk.

        Uses the endgame persistence module for individual components
        while preserving the existing directory layout for backwards
        compatibility.

        Parameters
        ----------
        path : str, optional
            Path to save to. If None, uses output_path.

        Returns
        -------
        str
            Path where the predictor was saved.
        """
        from endgame.persistence import save as eg_save

        if path is None:
            if self.output_path:
                path = self.output_path
            else:
                path = f"automl_predictor_{int(time.time())}"

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save main predictor state (lightweight dict, use pickle)
        state = {
            "label": self.label,
            "problem_type": self.problem_type,
            "eval_metric": self.eval_metric,
            "presets": self.presets,
            "time_limit": self.time_limit,
            "search_strategy": self.search_strategy,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
            "is_fitted_": self.is_fitted_,
            "fit_summary_": self.fit_summary_,
            "problem_type_": self.problem_type_,
            "classes_": self.classes_,
            "feature_names_": self.feature_names_,
        }

        with open(path / "predictor_state.pkl", "wb") as f:
            pickle.dump(state, f)

        # Save individual models via persistence module
        models_dir = path / "models"
        for name, model_info in self._models.items():
            model_path = models_dir / name
            model_path.mkdir(parents=True, exist_ok=True)
            estimator = model_info.get("estimator")
            if estimator is not None:
                eg_save(estimator, str(model_path / "model"))
            # Save non-estimator metadata separately
            meta = {k: v for k, v in model_info.items() if k != "estimator"}
            with open(model_path / "meta.pkl", "wb") as f:
                pickle.dump(meta, f)

        # Save ensemble
        if self._ensemble is not None:
            eg_save(self._ensemble, str(path / "ensemble"))

        # Save preprocessor
        if self._preprocessor is not None:
            eg_save(self._preprocessor, str(path / "preprocessor"))

        if self.verbosity > 0:
            print(f"Predictor saved to: {path}")

        return str(path)

    @classmethod
    def load(cls, path: str) -> "BasePredictor":
        """Load a predictor from disk.

        Supports both the legacy pickle format and the new endgame
        persistence format.

        Parameters
        ----------
        path : str
            Path to load from.

        Returns
        -------
        BasePredictor
            The loaded predictor.
        """
        from endgame.persistence import load as eg_load

        path = Path(path)

        # Load state
        with open(path / "predictor_state.pkl", "rb") as f:
            state = pickle.load(f)

        # Create predictor
        predictor = cls(
            label=state["label"],
            problem_type=state["problem_type"],
            eval_metric=state["eval_metric"],
            presets=state["presets"],
            time_limit=state["time_limit"],
            search_strategy=state["search_strategy"],
            random_state=state["random_state"],
            verbosity=state["verbosity"],
        )

        # Restore state
        predictor.is_fitted_ = state["is_fitted_"]
        predictor.fit_summary_ = state["fit_summary_"]
        predictor.problem_type_ = state["problem_type_"]
        predictor.classes_ = state["classes_"]
        predictor.feature_names_ = state["feature_names_"]

        # Load models
        models_path = path / "models"
        if models_path.exists():
            for model_dir in models_path.iterdir():
                if model_dir.is_dir():
                    model_info = {}

                    # Try new format first (model.egm + meta.pkl)
                    egm_file = model_dir / "model.egm"
                    meta_file = model_dir / "meta.pkl"
                    legacy_file = model_dir / "model.pkl"

                    if egm_file.exists():
                        model_info["estimator"] = eg_load(str(egm_file))
                        if meta_file.exists():
                            with open(meta_file, "rb") as f:
                                model_info.update(pickle.load(f))
                    elif legacy_file.exists():
                        # Legacy format: single pickle with everything
                        with open(legacy_file, "rb") as f:
                            model_info = pickle.load(f)

                    if model_info:
                        predictor._models[model_dir.name] = model_info

        # Load ensemble (try new format, fall back to legacy)
        for ensemble_path in [path / "ensemble.egm", path / "ensemble.pkl"]:
            if ensemble_path.exists():
                if ensemble_path.suffix == ".egm":
                    predictor._ensemble = eg_load(str(ensemble_path))
                else:
                    with open(ensemble_path, "rb") as f:
                        predictor._ensemble = pickle.load(f)
                break

        # Load preprocessor (try new format, fall back to legacy)
        for prep_path in [path / "preprocessor.egm", path / "preprocessor.pkl"]:
            if prep_path.exists():
                if prep_path.suffix == ".egm":
                    predictor._preprocessor = eg_load(str(prep_path))
                else:
                    with open(prep_path, "rb") as f:
                        predictor._preprocessor = pickle.load(f)
                break

        return predictor

    def _check_is_fitted(self) -> None:
        """Check if the predictor is fitted."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Predictor is not fitted. Call fit() before making predictions."
            )

    def _detect_problem_type(self, y: np.ndarray) -> str:
        """Detect the problem type from the target.

        Parameters
        ----------
        y : np.ndarray
            Target values.

        Returns
        -------
        str
            Detected problem type.
        """
        if self.problem_type != "auto":
            return self.problem_type

        # Check dtype
        if y.dtype.kind == "f":
            # Float dtype
            n_unique = len(np.unique(y[~np.isnan(y)]))
            if n_unique <= 20:
                # Could be classification encoded as float
                if np.allclose(y, y.astype(int)):
                    return "multiclass" if n_unique > 2 else "binary"
            return "regression"

        # Integer or object dtype
        n_unique = len(np.unique(y))
        if n_unique == 2:
            return "binary"
        elif n_unique <= 100:
            return "multiclass"
        else:
            # Too many unique values for classification
            return "regression"

    def _get_eval_metric(self) -> str:
        """Get the evaluation metric based on problem type."""
        if self.eval_metric != "auto":
            return self.eval_metric

        if self.problem_type_ in ("binary", "classification"):
            return "roc_auc"
        elif self.problem_type_ == "multiclass":
            return "log_loss"
        else:
            return "rmse"

    def _load_and_prepare_data(
        self,
        data: DataInput,
        for_prediction: bool = True,
    ) -> tuple:
        """Load and prepare data.

        Parameters
        ----------
        data : various
            Input data.
        for_prediction : bool
            Whether this is for prediction (no target needed).

        Returns
        -------
        X, y : tuple
            Features and target (y is None if for_prediction=True).
        """
        # This is a simple implementation - subclasses should override
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, np.ndarray):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        if for_prediction:
            if self.label in df.columns:
                X = df.drop(columns=[self.label])
            else:
                X = df
            return X, None
        else:
            if self.label not in df.columns:
                raise ValueError(f"Target column '{self.label}' not found in data")
            y = df[self.label].values
            X = df.drop(columns=[self.label])
            return X, y

    def refit_full(self, data: DataInput | None = None) -> "BasePredictor":
        """Retrain best model(s) on all available data (train + validation).

        After cross-validation identifies the best model and hyperparameters,
        this method retrains on the full dataset for maximum deployment
        performance. The refitted model cannot be evaluated (no holdout).

        Parameters
        ----------
        data : DataInput, optional
            Full dataset. If None, uses the training data from the last
            fit() call (subclasses must store it).

        Returns
        -------
        BasePredictor
            Self with models retrained on full data.

        Raises
        ------
        RuntimeError
            If the predictor has not been fitted.
        """
        self._check_is_fitted()

        if not self._models:
            raise RuntimeError("No models available to refit.")

        logger.warning(
            "refit_full() retrains on all data. The resulting model "
            "cannot be evaluated with a holdout set."
        )
        return self

    # ------------------------------------------------------------------
    # Logger helpers
    # ------------------------------------------------------------------

    def _log_fit_start(self, n_samples: int, n_features: int) -> None:
        """Log parameters at the start of fitting (if logger is set)."""
        if self.logger is None:
            return
        self.logger.log_params({
            "label": self.label,
            "problem_type": self.problem_type,
            "presets": self.presets,
            "time_limit": self.time_limit,
            "n_samples": n_samples,
            "n_features": n_features,
        })

    def _log_fit_end(self) -> None:
        """Log metrics at the end of fitting (if logger is set)."""
        if self.logger is None or self.fit_summary_ is None:
            return
        metrics = {
            "best_score": self.fit_summary_.best_score,
            "cv_score": self.fit_summary_.cv_score,
            "total_time": self.fit_summary_.total_time,
            "n_models_trained": float(self.fit_summary_.n_models_trained),
        }
        self.logger.log_metrics(metrics)

    def __repr__(self) -> str:
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        return (
            f"{self.__class__.__name__}("
            f"label='{self.label}', "
            f"presets='{self.presets}', "
            f"{fitted_str})"
        )
