"""Type definitions and result dataclasses for Endgame."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Union

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False
    pl = None

# Type aliases for flexible input handling
ArrayLike = Union[np.ndarray, Sequence, "pd.Series"] if HAS_PANDAS else Union[np.ndarray, Sequence]
FrameLike = Union[np.ndarray, "pd.DataFrame", "pl.DataFrame", "pl.LazyFrame"]

DriftSeverity = Literal["none", "mild", "moderate", "severe"]
OutputFormat = Literal["polars", "pandas", "numpy"]


@dataclass
class AdversarialValidationResult:
    """Result from adversarial validation drift detection.

    Attributes
    ----------
    auc_score : float
        ROC-AUC score of classifier distinguishing train from test.
        Score > 0.5 indicates distribution drift.
    drifted_features : List[str]
        Features ranked by contribution to drift (highest first).
    feature_importances : Dict[str, float]
        Importance score for each feature in distinguishing train/test.
    drift_severity : DriftSeverity
        Categorical assessment: 'none', 'mild', 'moderate', 'severe'.
    """
    auc_score: float
    drifted_features: list[str]
    feature_importances: dict[str, float]
    drift_severity: DriftSeverity

    @classmethod
    def from_auc(cls, auc: float, importances: dict[str, float]) -> "AdversarialValidationResult":
        """Create result from AUC score and feature importances."""
        sorted_features = sorted(importances.keys(), key=lambda x: importances[x], reverse=True)

        if auc < 0.55:
            severity = "none"
        elif auc < 0.65:
            severity = "mild"
        elif auc < 0.75:
            severity = "moderate"
        else:
            severity = "severe"

        return cls(
            auc_score=auc,
            drifted_features=sorted_features,
            feature_importances=importances,
            drift_severity=severity,
        )


@dataclass
class OOFResult:
    """Result from out-of-fold cross-validation.

    Attributes
    ----------
    oof_predictions : np.ndarray
        Out-of-fold predictions for all training samples.
    fold_scores : List[float]
        Validation score for each fold.
    mean_score : float
        Mean validation score across folds.
    std_score : float
        Standard deviation of validation scores.
    models : List[Any]
        Trained model for each fold (if return_models=True).
    fold_indices : List[tuple]
        (train_idx, val_idx) for each fold.
    """
    oof_predictions: np.ndarray
    fold_scores: list[float]
    mean_score: float
    std_score: float
    models: list[Any] = field(default_factory=list)
    fold_indices: list[tuple] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"OOFResult(mean_score={self.mean_score:.4f}, "
            f"std_score={self.std_score:.4f}, n_folds={len(self.fold_scores)})"
        )


@dataclass
class OptimizationResult:
    """Result from hyperparameter optimization.

    Attributes
    ----------
    best_params : Dict[str, Any]
        Best hyperparameters found.
    best_score : float
        Best validation score achieved.
    study : Any
        Optuna study object for further analysis.
    all_trials : List[Dict]
        Summary of all optimization trials.
    n_trials : int
        Number of trials completed.
    """
    best_params: dict[str, Any]
    best_score: float
    study: Any
    all_trials: list[dict]
    n_trials: int

    def __repr__(self) -> str:
        return (
            f"OptimizationResult(best_score={self.best_score:.4f}, "
            f"n_trials={self.n_trials})"
        )


@dataclass
class EnsembleResult:
    """Result from ensemble weight optimization.

    Attributes
    ----------
    weights : Dict[int, float]
        Optimal weight for each model (by index).
    best_score : float
        Best ensemble score achieved.
    improvement : float
        Score improvement over best single model.
    selected_models : List[int]
        Indices of models with non-zero weight.
    """
    weights: dict[int, float]
    best_score: float
    improvement: float
    selected_models: list[int]

    def __repr__(self) -> str:
        return (
            f"EnsembleResult(best_score={self.best_score:.4f}, "
            f"n_models={len(self.selected_models)}, improvement={self.improvement:.4f})"
        )


@dataclass
class TokenizationReport:
    """Result from tokenizer analysis.

    Attributes
    ----------
    inefficient_tokens : Dict[str, int]
        Tokens that expand to multiple subtokens with counts.
    oov_frequency : float
        Fraction of tokens that are out-of-vocabulary.
    avg_sequence_length : float
        Average tokenized sequence length.
    suggested_replacements : Dict[str, str]
        Suggested text replacements to improve tokenization.
    """
    inefficient_tokens: dict[str, int]
    oov_frequency: float
    avg_sequence_length: float
    suggested_replacements: dict[str, str]
