from __future__ import annotations

"""Quick API implementation for rapid prototyping.

Provides one-line model training with automatic preprocessing,
cross-validation, and model selection.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from endgame.tracking.base import ExperimentLogger
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

PresetName = Literal["fast", "default", "competition", "interpretable"]
TaskType = Literal["classification", "regression"]


# =============================================================================
# Preset Configurations
# =============================================================================

PRESETS: dict[str, dict[str, Any]] = {
    "fast": {
        "description": "Quick iteration with minimal models (~1 min)",
        "models": ["lgbm_fast", "linear"],
        "cv_folds": 3,
        "early_stopping": 50,
        "n_estimators": 500,
    },
    "default": {
        "description": "Balanced speed and accuracy (~5 min)",
        "models": ["lgbm", "xgb", "catboost", "linear"],
        "cv_folds": 5,
        "early_stopping": 100,
        "n_estimators": 2000,
    },
    "competition": {
        "description": "Full competitive pipeline (~30 min)",
        "models": [
            "lgbm",
            "xgb",
            "catboost",
            "linear",
            "knn",
            "elm",
            "rotation_forest",
        ],
        "cv_folds": 5,
        "early_stopping": 100,
        "n_estimators": 5000,
    },
    "interpretable": {
        "description": "Only interpretable models",
        "models": ["linear", "ebm", "nam"],
        "cv_folds": 5,
        "early_stopping": 100,
        "n_estimators": 2000,
    },
}


# =============================================================================
# Model Registry
# =============================================================================


def _extract_feature_names(X) -> list[str] | None:
    """Extract feature names from X if available."""
    if isinstance(X, pd.DataFrame):
        return X.columns.tolist()
    if hasattr(X, "columns"):
        return list(X.columns)
    return None


def _build_feature_importances(
    model, feature_names: list[str] | None
) -> dict[str, float]:
    """Build feature importance dict using real feature names when available."""
    if not hasattr(model, "feature_importances_"):
        return {}
    fi = model.feature_importances_
    if isinstance(fi, dict):
        return fi
    names = feature_names or [f"f{i}" for i in range(len(fi))]
    return {names[i]: v for i, v in enumerate(fi)}


_TABPFN25_MAX_SAMPLES = 50_000


def _supports_eval_set(model) -> bool:
    """Check if model.fit() accepts an eval_set keyword argument."""
    import inspect
    try:
        sig = inspect.signature(model.fit)
        return "eval_set" in sig.parameters
    except (ValueError, TypeError):
        return False


def _has_non_numeric(X) -> bool:
    """Check if X contains non-numeric columns (object/category dtypes)."""
    if isinstance(X, pd.DataFrame):
        return any(X[c].dtype.kind in ("O", "S", "U") or
                   isinstance(X[c].dtype, pd.CategoricalDtype)
                   for c in X.columns)
    return False


def _maybe_prepend_tabpfn25(
    models: list[str], n_samples: int, X=None
) -> list[str]:
    """Prepend TabPFN v2.5 to the model list when the dataset fits."""
    if n_samples <= _TABPFN25_MAX_SAMPLES and "tabpfn25" not in models:
        if X is not None and _has_non_numeric(X):
            return models
        return ["tabpfn25"] + models
    return models


def _select_model_for_data(
    X, y, task: TaskType = "classification"
) -> str:
    """Pick a default model based on dataset characteristics.

    Lightweight heuristic — no data-dependent fitting, just shape/type checks.
    """
    n_samples, n_features = X.shape
    has_cats = _has_non_numeric(X)

    if task == "classification":
        if n_samples <= 3000 and n_features <= 100 and not has_cats:
            return "tabpfn25"
        if n_samples <= 5000 and n_features <= 30:
            return "ebm"
        if has_cats or n_samples > 10000:
            return "lgbm"
        if n_features > 500:
            return "lgbm"
        return "lgbm"
    else:  # regression
        if n_features <= 30 and n_samples <= 5000:
            return "ebm"
        return "lgbm"


def _get_model(model_key: str, task: TaskType, preset_config: dict[str, Any]):
    """Get a model instance by key."""
    # Import here to avoid circular imports
    from endgame.models.baselines import (
        ELMClassifier,
        ELMRegressor,
        KNNClassifier,
        KNNRegressor,
        LinearClassifier,
        LinearRegressor,
    )
    from endgame.models.wrappers import CatBoostWrapper, LGBMWrapper, XGBWrapper

    n_estimators = preset_config.get("n_estimators", 2000)
    early_stopping = preset_config.get("early_stopping", 100)

    if model_key == "lgbm":
        return LGBMWrapper(
            preset="endgame",
            early_stopping_rounds=early_stopping,
            n_estimators=n_estimators,
        )
    elif model_key == "lgbm_fast":
        return LGBMWrapper(
            preset="fast",
            early_stopping_rounds=early_stopping,
        )
    elif model_key == "xgb":
        return XGBWrapper(
            preset="endgame",
            early_stopping_rounds=early_stopping,
            n_estimators=n_estimators,
        )
    elif model_key == "catboost":
        return CatBoostWrapper(
            preset="endgame",
            early_stopping_rounds=early_stopping,
            iterations=n_estimators,
        )
    elif model_key == "linear":
        if task == "classification":
            return LinearClassifier()
        return LinearRegressor()
    elif model_key == "knn":
        if task == "classification":
            return KNNClassifier()
        return KNNRegressor()
    elif model_key == "elm":
        if task == "classification":
            return ELMClassifier()
        return ELMRegressor()
    elif model_key == "rotation_forest":
        from endgame.models.trees import RotationForestClassifier, RotationForestRegressor

        if task == "classification":
            return RotationForestClassifier(n_estimators=100)
        return RotationForestRegressor(n_estimators=100)
    elif model_key == "ebm":
        from endgame.models.ebm import EBMClassifier, EBMRegressor

        if task == "classification":
            return EBMClassifier()
        return EBMRegressor()
    elif model_key == "nam":
        # NAM is classification-only for now
        from endgame.models.tabular.nam import NAMClassifier, NAMRegressor

        if task == "classification":
            return NAMClassifier()
        return NAMRegressor()
    elif model_key == "tabpfn25":
        from endgame.models.tabular.tabpfn import TabPFN25Classifier, TabPFN25Regressor

        if task == "classification":
            return TabPFN25Classifier()
        return TabPFN25Regressor()
    else:
        raise ValueError(f"Unknown model key: {model_key}")


# =============================================================================
# Result Classes
# =============================================================================


@dataclass
class QuickResult:
    """Result from quick.classify() or quick.regress().

    Attributes
    ----------
    model : Any
        The fitted model.
    oof_predictions : ndarray
        Out-of-fold predictions.
    cv_score : float
        Cross-validation score.
    metric : str
        Metric used for scoring.
    feature_importances : Dict[str, float]
        Feature importance dictionary (if available).
    feature_names : list[str] | None
        Feature names from training data (if available).
    """

    model: Any
    oof_predictions: np.ndarray
    cv_score: float
    metric: str
    feature_importances: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] | None = None
    _label_encoder: LabelEncoder | None = field(default=None, repr=False)

    def __repr__(self) -> str:
        return f"QuickResult(cv_score={self.cv_score:.4f}, metric='{self.metric}')"

    def report(
        self,
        X_test,
        y_test,
        *,
        save_path: str | None = None,
        model_name: str | None = None,
        dataset_name: str | None = None,
        theme: str = "dark",
    ):
        """Generate an interactive classification report.

        Parameters
        ----------
        X_test : array-like
            Test features.
        y_test : array-like
            True test labels.
        save_path : str, optional
            Path to save the HTML report.
        model_name : str, optional
            Display name for the model.
        dataset_name : str, optional
            Display name for the dataset.
        theme : str, default='dark'
            Report theme: 'dark' or 'light'.

        Returns
        -------
        ClassificationReport
            Interactive report object.
        """
        from endgame.visualization import ClassificationReport

        y_test_enc = self._label_encoder.transform(y_test) if self._label_encoder else y_test
        # Encode categorical/object columns to numeric so that
        # np.asarray (used by ClassificationReport) produces a float array
        if isinstance(X_test, pd.DataFrame):
            X_report = X_test.copy()
            for col in X_report.columns:
                if hasattr(X_report[col], "cat"):
                    X_report[col] = X_report[col].cat.codes.replace(-1, np.nan).astype("float")
                elif X_report[col].dtype == object:
                    X_report[col] = pd.Categorical(X_report[col]).codes
                    X_report[col] = X_report[col].replace(-1, np.nan).astype("float")
            X_test = X_report
        report = ClassificationReport(
            self.model,
            X_test,
            y_test_enc,
            feature_names=self.feature_names,
            model_name=model_name,
            dataset_name=dataset_name,
            theme=theme,
        )
        if save_path:
            report.save(save_path)
        return report


@dataclass
class ModelResult:
    """Result for a single model in comparison."""

    name: str
    model: Any
    oof_predictions: np.ndarray
    cv_score: float
    fit_time: float


@dataclass
class ComparisonResult:
    """Result from quick.compare().

    Attributes
    ----------
    results : List[ModelResult]
        Results for each model, sorted by score.
    best_model : Any
        The best performing model.
    leaderboard : List[Dict]
        Leaderboard with model names and scores.
    metric : str
        Metric used for scoring.
    """

    results: list[ModelResult]
    best_model: Any
    leaderboard: list[dict[str, Any]]
    metric: str

    def __repr__(self) -> str:
        lines = ["ComparisonResult:"]
        for i, entry in enumerate(self.leaderboard[:5]):
            lines.append(f"  {i+1}. {entry['model']}: {entry['score']:.4f}")
        if len(self.leaderboard) > 5:
            lines.append(f"  ... and {len(self.leaderboard) - 5} more")
        return "\n".join(lines)


# =============================================================================
# Main API Functions
# =============================================================================


def classify(
    X,
    y,
    preset: PresetName = "default",
    metric: str = "roc_auc",
    cv_folds: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
    explainable: bool = False,
    logger: ExperimentLogger | None = None,
) -> QuickResult:
    """Quick classification with automatic model selection.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Target labels.
    preset : str, default='default'
        Preset configuration: 'fast', 'default', 'competition', 'interpretable'.
    metric : str, default='roc_auc'
        Scoring metric: 'roc_auc', 'accuracy', 'f1'.
    cv_folds : int, optional
        Number of CV folds. If None, uses preset default.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Whether to print progress.
    explainable : bool, default=False
        If True, only use inherently interpretable models (EBM, linear, NAM).
        EBM is the default when explainable=True.
    logger : ExperimentLogger, optional
        Experiment logger for tracking params and metrics.

    Returns
    -------
    QuickResult
        Result containing model, OOF predictions, and CV score.

    Examples
    --------
    >>> import endgame as eg
    >>> result = eg.quick.classify(X, y)
    >>> print(f"CV Score: {result.cv_score:.4f}")
    >>> predictions = result.model.predict(X_test)
    """
    feature_names = _extract_feature_names(X)
    if isinstance(X, pd.DataFrame):
        X = X.copy()
    else:
        X = np.asarray(X)
    y = np.asarray(y)

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    n_classes = len(le.classes_)

    # Get preset config
    preset_config = PRESETS[preset].copy()
    n_folds = cv_folds or preset_config["cv_folds"]

    # Pick model
    if explainable:
        model_key = "ebm"
    else:
        model_key = _select_model_for_data(X, y_encoded, task="classification")

    if verbose:
        print(f"Training {model_key} with {preset} preset...")

    # Log params if logger provided
    if logger is not None:
        logger.log_params({
            "task": "classification",
            "model": model_key,
            "preset": preset,
            "metric": metric,
            "cv_folds": n_folds,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "n_classes": n_classes,
        })

    # Get model
    model = _get_model(model_key, "classification", preset_config)

    # Cross-validation
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Generate OOF predictions
    if n_classes == 2:
        oof_preds = np.zeros(len(y))
    else:
        oof_preds = np.zeros((len(y), n_classes))

    # Use iloc for DataFrames, integer indexing for arrays
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y_encoded)):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

        # Fit with early stopping if available
        if _supports_eval_set(model):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)

        # Predict
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_val)
            if n_classes == 2:
                oof_preds[val_idx] = proba[:, 1]
            else:
                oof_preds[val_idx] = proba
        else:
            oof_preds[val_idx] = model.predict(X_val)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds} complete")

    # Calculate score
    if metric == "roc_auc":
        if n_classes == 2:
            cv_score = roc_auc_score(y_encoded, oof_preds)
        else:
            cv_score = roc_auc_score(y_encoded, oof_preds, multi_class="ovr")
    elif metric == "accuracy":
        if n_classes == 2:
            cv_score = accuracy_score(y_encoded, (oof_preds > 0.5).astype(int))
        else:
            cv_score = accuracy_score(y_encoded, np.argmax(oof_preds, axis=1))
    elif metric == "f1":
        if n_classes == 2:
            cv_score = f1_score(y_encoded, (oof_preds > 0.5).astype(int))
        else:
            cv_score = f1_score(
                y_encoded, np.argmax(oof_preds, axis=1), average="weighted"
            )
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Fit final model on all data
    model.fit(X, y_encoded)

    # Get feature importances if available
    feature_importances = _build_feature_importances(model, feature_names)

    if verbose:
        print(f"CV {metric}: {cv_score:.4f}")

    # Log metrics if logger provided
    if logger is not None:
        logger.log_metrics({metric: cv_score})

    return QuickResult(
        model=model,
        oof_predictions=oof_preds,
        cv_score=cv_score,
        metric=metric,
        feature_importances=feature_importances,
        feature_names=feature_names,
        _label_encoder=le,
    )


def regress(
    X,
    y,
    preset: PresetName = "default",
    metric: str = "rmse",
    cv_folds: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
    explainable: bool = False,
    logger: ExperimentLogger | None = None,
) -> QuickResult:
    """Quick regression with automatic model selection.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Target values.
    preset : str, default='default'
        Preset configuration: 'fast', 'default', 'competition', 'interpretable'.
    metric : str, default='rmse'
        Scoring metric: 'rmse', 'r2', 'mae'.
    cv_folds : int, optional
        Number of CV folds. If None, uses preset default.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Whether to print progress.
    explainable : bool, default=False
        If True, only use inherently interpretable models (EBM, linear, NAM).
        EBM is the default when explainable=True.
    logger : ExperimentLogger, optional
        Experiment logger for tracking params and metrics.

    Returns
    -------
    QuickResult
        Result containing model, OOF predictions, and CV score.

    Examples
    --------
    >>> import endgame as eg
    >>> result = eg.quick.regress(X, y)
    >>> print(f"CV RMSE: {result.cv_score:.4f}")
    >>> predictions = result.model.predict(X_test)
    """
    feature_names = _extract_feature_names(X)
    if isinstance(X, pd.DataFrame):
        X = X.copy()
    else:
        X = np.asarray(X)
    y = np.asarray(y, dtype=np.float64)

    # Get preset config
    preset_config = PRESETS[preset].copy()
    n_folds = cv_folds or preset_config["cv_folds"]

    # Pick model
    if explainable:
        model_key = "ebm"
    else:
        model_key = _select_model_for_data(X, y, task="regression")

    if verbose:
        print(f"Training {model_key} with {preset} preset...")

    # Log params if logger provided
    if logger is not None:
        logger.log_params({
            "task": "regression",
            "model": model_key,
            "preset": preset,
            "metric": metric,
            "cv_folds": n_folds,
            "n_samples": len(X),
            "n_features": X.shape[1],
        })

    # Get model
    model = _get_model(model_key, "regression", preset_config)

    # Cross-validation
    cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Generate OOF predictions
    oof_preds = np.zeros(len(y))

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        if isinstance(X, pd.DataFrame):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        else:
            X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit with early stopping if available
        if _supports_eval_set(model):
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        else:
            model.fit(X_train, y_train)

        # Predict
        oof_preds[val_idx] = model.predict(X_val)

        if verbose:
            print(f"  Fold {fold_idx + 1}/{n_folds} complete")

    # Calculate score
    if metric == "rmse":
        cv_score = np.sqrt(mean_squared_error(y, oof_preds))
    elif metric == "r2":
        cv_score = r2_score(y, oof_preds)
    elif metric == "mae":
        cv_score = np.mean(np.abs(y - oof_preds))
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Fit final model on all data
    model.fit(X, y)

    # Get feature importances if available
    feature_importances = _build_feature_importances(model, feature_names)

    if verbose:
        print(f"CV {metric}: {cv_score:.4f}")

    # Log metrics if logger provided
    if logger is not None:
        logger.log_metrics({metric: cv_score})

    return QuickResult(
        model=model,
        oof_predictions=oof_preds,
        cv_score=cv_score,
        metric=metric,
        feature_importances=feature_importances,
        feature_names=feature_names,
    )


def compare(
    X,
    y,
    task: TaskType = "classification",
    preset: PresetName = "default",
    metric: str | None = None,
    cv_folds: int | None = None,
    random_state: int = 42,
    verbose: bool = True,
    logger: ExperimentLogger | None = None,
) -> ComparisonResult:
    """Compare multiple models quickly.

    Parameters
    ----------
    X : array-like
        Training features.
    y : array-like
        Target values/labels.
    task : str, default='classification'
        Task type: 'classification' or 'regression'.
    preset : str, default='default'
        Preset configuration.
    metric : str, optional
        Scoring metric. If None, uses default for task.
    cv_folds : int, optional
        Number of CV folds.
    random_state : int, default=42
        Random seed.
    verbose : bool, default=True
        Whether to print progress.
    logger : ExperimentLogger, optional
        Experiment logger for tracking params and metrics.

    Returns
    -------
    ComparisonResult
        Comparison results with leaderboard.

    Examples
    --------
    >>> import endgame as eg
    >>> comparison = eg.quick.compare(X, y, task='classification')
    >>> print(comparison)  # Shows leaderboard
    >>> best_model = comparison.best_model
    """
    import time

    if isinstance(X, pd.DataFrame):
        X = X.copy()
    else:
        X = np.asarray(X)
    y = np.asarray(y)

    # Get preset config
    preset_config = PRESETS[preset].copy()
    n_folds = cv_folds or preset_config["cv_folds"]

    # Default metrics
    if metric is None:
        metric = "roc_auc" if task == "classification" else "rmse"

    # Prepend TabPFN v2.5 for small numeric datasets
    models = _maybe_prepend_tabpfn25(preset_config["models"], len(X), X)

    # Prepare data
    if task == "classification":
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        n_classes = len(le.classes_)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    else:
        y_encoded = y.astype(np.float64)
        n_classes = 0
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    # Log params if logger provided
    if logger is not None:
        logger.log_params({
            "task": task,
            "preset": preset,
            "metric": metric,
            "cv_folds": n_folds,
            "n_samples": len(X),
            "n_features": X.shape[1],
            "models": ",".join(models),
        })

    results = []

    for model_key in models:
        if verbose:
            print(f"Training {model_key}...")

        try:
            start_time = time.time()

            # Get model
            model = _get_model(model_key, task, preset_config)

            # Generate OOF predictions
            if task == "classification" and n_classes == 2:
                oof_preds = np.zeros(len(y))
            elif task == "classification":
                oof_preds = np.zeros((len(y), n_classes))
            else:
                oof_preds = np.zeros(len(y))

            for train_idx, val_idx in cv.split(X, y_encoded if task == "classification" else None):
                if isinstance(X, pd.DataFrame):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                else:
                    X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y_encoded[train_idx], y_encoded[val_idx]

                if _supports_eval_set(model):
                    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
                else:
                    model.fit(X_train, y_train)

                if task == "classification" and hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_val)
                    if n_classes == 2:
                        oof_preds[val_idx] = proba[:, 1]
                    else:
                        oof_preds[val_idx] = proba
                else:
                    oof_preds[val_idx] = model.predict(X_val)

            fit_time = time.time() - start_time

            # Calculate score
            if task == "classification":
                if metric == "roc_auc":
                    if n_classes == 2:
                        cv_score = roc_auc_score(y_encoded, oof_preds)
                    else:
                        cv_score = roc_auc_score(y_encoded, oof_preds, multi_class="ovr")
                elif metric == "accuracy":
                    if n_classes == 2:
                        cv_score = accuracy_score(y_encoded, (oof_preds > 0.5).astype(int))
                    else:
                        cv_score = accuracy_score(y_encoded, np.argmax(oof_preds, axis=1))
                elif metric == "f1":
                    if n_classes == 2:
                        cv_score = f1_score(y_encoded, (oof_preds > 0.5).astype(int))
                    else:
                        cv_score = f1_score(
                            y_encoded, np.argmax(oof_preds, axis=1), average="weighted"
                        )
                else:
                    raise ValueError(f"Unknown metric: {metric}")
            else:
                if metric == "rmse":
                    cv_score = np.sqrt(mean_squared_error(y_encoded, oof_preds))
                elif metric == "r2":
                    cv_score = r2_score(y_encoded, oof_preds)
                elif metric == "mae":
                    cv_score = np.mean(np.abs(y_encoded - oof_preds))
                else:
                    raise ValueError(f"Unknown metric: {metric}")

            # Fit final model
            model.fit(X, y_encoded)

            results.append(
                ModelResult(
                    name=model_key,
                    model=model,
                    oof_predictions=oof_preds,
                    cv_score=cv_score,
                    fit_time=fit_time,
                )
            )

            if verbose:
                print(f"  {metric}: {cv_score:.4f} ({fit_time:.1f}s)")

        except Exception as e:
            if verbose:
                print(f"  Failed: {e}")
            continue

    # Sort results (higher is better for most metrics, except rmse/mae)
    reverse = metric not in ("rmse", "mae")
    results.sort(key=lambda r: r.cv_score, reverse=reverse)

    # Create leaderboard
    leaderboard = [
        {"model": r.name, "score": r.cv_score, "fit_time": r.fit_time} for r in results
    ]

    # Log metrics if logger provided
    if logger is not None and results:
        logger.log_metrics({
            f"best_{metric}": results[0].cv_score,
            "n_models_compared": float(len(results)),
        })

    return ComparisonResult(
        results=results,
        best_model=results[0].model if results else None,
        leaderboard=leaderboard,
        metric=metric,
    )
