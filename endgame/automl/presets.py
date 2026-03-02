from __future__ import annotations

"""Preset configurations for AutoML.

This module defines preset configurations that control the behavior of
AutoML predictors, including time limits, model selection, ensembling,
and calibration settings.

Presets
-------
- best_quality: Maximum accuracy, no time limit
- high_quality: High accuracy with 4 hour default
- good_quality: Good accuracy with 1 hour default
- medium_quality: Balanced speed/accuracy with 15 min default
- fast: Quick results with 5 min default
- exhaustive: Evolutionary search over ALL models + preprocessing + ensembles
- interpretable: Only interpretable models
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PresetConfig:
    """Configuration for an AutoML preset.

    Attributes
    ----------
    name : str
        Name of the preset.
    description : str
        Human-readable description.
    default_time_limit : int or None
        Default time limit in seconds. None means no limit.
    cv_folds : int
        Number of cross-validation folds.
    num_bag_folds : int
        Number of bagging folds (0 to disable).
    num_stack_levels : int
        Number of stacking levels (0 to disable).
    hyperparameter_tune : bool
        Whether to tune hyperparameters.
    tune_trials : int
        Number of tuning trials per model.
    ensemble_method : str
        Ensemble method: "none", "hill_climbing", "stacking".
    calibrate : bool
        Whether to calibrate probabilities.
    use_holdout : bool
        Whether to use a holdout set.
    holdout_frac : float
        Fraction of data for holdout (if use_holdout=True).
    feature_engineering : str
        Feature engineering level: "none", "light", "moderate", "aggressive".
    model_pool : list of str
        List of model names to consider.
    time_allocations : dict
        Time allocation fractions for each stage.
    """

    name: str
    description: str
    default_time_limit: int | None
    cv_folds: int
    num_bag_folds: int
    num_stack_levels: int
    hyperparameter_tune: bool
    tune_trials: int
    ensemble_method: str
    calibrate: bool
    use_holdout: bool
    holdout_frac: float
    feature_engineering: str
    model_pool: list[str]
    time_allocations: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        """Set default time allocations if not provided."""
        if not self.time_allocations:
            self.time_allocations = DEFAULT_TIME_ALLOCATIONS.copy()


# Default time allocation across pipeline stages
DEFAULT_TIME_ALLOCATIONS = {
    "profiling": 0.01,
    "quality_guardrails": 0.02,
    "data_cleaning": 0.02,
    "preprocessing": 0.05,
    "feature_engineering": 0.03,
    "data_augmentation": 0.02,
    "model_selection": 0.04,
    "model_training": 0.40,
    "constraint_check": 0.01,
    "hyperparameter_tuning": 0.20,
    "ensembling": 0.06,
    "threshold_opt": 0.02,
    "calibration": 0.03,
    "post_training": 0.02,
    "explainability": 0.02,
}

# Model pools for different presets
MODEL_POOLS = {
    "all": [
        # GBDTs (always strong)
        "lgbm", "xgb", "catboost", "ngboost",
        # Deep tabular (modern)
        "ft_transformer", "saint", "tabnet", "node", "nam",
        "tabular_resnet", "tabm", "realmlp", "grande",
        "gandalf", "tabr", "tabdpt", "modern_nca",
        "tab_transformer", "mlp", "embedding_mlp",
        # Custom trees
        "rotation_forest", "c50", "c50_ensemble", "oblique_forest",
        "extra_oblique_forest", "patch_oblique_forest", "honest_forest",
        "evolutionary_tree", "rf", "extra_trees",
        "adtree", "model_tree", "cubist",
        # Linear
        "linear", "elm", "mars",
        # Kernel
        "svm", "gp",
        # Rules
        "rulefit", "furia", "prim",
        # Bayesian
        "tan", "eskdb", "kdb", "bart", "naive_bayes",
        "neural_kdb", "auto_sle", "ebmc_classifier",
        # Interpretable
        "ebm", "ebmc", "ebmr",
        "gam", "node_gam", "gami_net",
        "corels", "slim", "fasterrisk", "gosdt",
        # Ordinal
        "ordinal", "logistic_at", "logistic_it", "logistic_se",
        "ordinal_ridge", "ordinal_lad",
        # Discriminant
        "lda", "qda", "knn",
        # Foundation
        "tabpfn", "tabpfn_v2", "tabpfn_25", "xrfm",
        # Symbolic
        "symbolic_regression", "symbolic_regressor",
        # Quantile
        "quantile_forest",
    ],
    "best_quality": [
        # GBDTs (always strong)
        "lgbm", "xgb", "catboost", "ngboost",
        # Deep tabular (modern)
        "ft_transformer", "saint", "tabnet", "node", "nam",
        "tabular_resnet", "tabm", "realmlp", "grande",
        "gandalf", "tabr", "tabdpt", "modern_nca",
        "tab_transformer", "mlp", "embedding_mlp",
        # Custom trees
        "rotation_forest", "c50", "c50_ensemble", "oblique_forest",
        "extra_oblique_forest", "patch_oblique_forest", "honest_forest",
        "evolutionary_tree", "rf", "extra_trees",
        "adtree", "model_tree", "cubist",
        # Linear
        "linear", "elm", "mars",
        # Kernel
        "svm", "gp",
        # Rules
        "rulefit", "furia", "prim",
        # Bayesian
        "tan", "eskdb", "kdb", "bart", "naive_bayes",
        "neural_kdb", "auto_sle", "ebmc_classifier",
        # Interpretable
        "ebm", "ebmc", "ebmr",
        "gam", "node_gam", "gami_net",
        "corels", "slim", "fasterrisk", "gosdt",
        # Ordinal
        "ordinal", "logistic_at", "logistic_it", "logistic_se",
        "ordinal_ridge", "ordinal_lad",
        # Discriminant
        "lda", "qda", "knn",
        # Foundation
        "tabpfn", "tabpfn_v2", "tabpfn_25", "xrfm",
        # Symbolic
        "symbolic_regression", "symbolic_regressor",
        # Quantile
        "quantile_forest",
    ],
    "high_quality": [
        "lgbm", "xgb", "catboost",
        "ft_transformer", "tabm", "tabnet",
        "tabpfn_v2", "tabpfn_25",
        "realmlp", "tabr", "gandalf",
        "rotation_forest", "rf",
        "extra_oblique_forest", "honest_forest",
        "linear",
    ],
    "good_quality": [
        "lgbm", "xgb", "catboost",
        "tabm", "realmlp", "rf",
        "ebm",
        "linear", "elm",
    ],
    "medium_quality": [
        "lgbm", "xgb", "catboost",
        "rf",
        "linear",
    ],
    "fast": [
        "lgbm",
    ],
    "interpretable": [
        # GAM-style models
        "ebm", "gam", "nam", "node_gam", "gami_net",
        # Rule-based models
        "rulefit", "furia", "corels", "prim",
        # Sparse linear / scorecards
        "linear", "mars", "slim", "fasterrisk",
        # Interpretable trees
        "c50", "gosdt", "cubist",
        # Symbolic
        "symbolic_regression",
        # Probabilistic (feature importances + uncertainty)
        "ngboost",
        # Bayesian (interpretable)
        "naive_bayes", "tan", "lda",
        # Foundation (interpretable)
        "xrfm",
        # Ordinal
        "ordinal",
    ],
}


# Preset configurations
PRESETS: dict[str, PresetConfig] = {
    "best_quality": PresetConfig(
        name="best_quality",
        description="Maximum accuracy, no time limit",
        default_time_limit=None,
        cv_folds=8,
        num_bag_folds=5,
        num_stack_levels=2,
        hyperparameter_tune=True,
        tune_trials=100,
        ensemble_method="auto",
        calibrate=True,
        use_holdout=True,
        holdout_frac=0.1,
        feature_engineering="aggressive",
        model_pool=MODEL_POOLS["best_quality"],
        time_allocations={
            "profiling": 0.01,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.02,
            "preprocessing": 0.08,
            "feature_engineering": 0.04,
            "data_augmentation": 0.02,
            "model_selection": 0.04,
            "model_training": 0.35,
            "constraint_check": 0.01,
            "hyperparameter_tuning": 0.25,
            "ensembling": 0.05,
            "threshold_opt": 0.02,
            "calibration": 0.03,
            "post_training": 0.03,
            "explainability": 0.03,
        },
    ),
    "high_quality": PresetConfig(
        name="high_quality",
        description="High accuracy with 4 hour default",
        default_time_limit=14400,  # 4 hours
        cv_folds=5,
        num_bag_folds=5,
        num_stack_levels=1,
        hyperparameter_tune=True,
        tune_trials=50,
        ensemble_method="auto",
        calibrate=True,
        use_holdout=True,
        holdout_frac=0.1,
        feature_engineering="moderate",
        model_pool=MODEL_POOLS["high_quality"],
        time_allocations={
            "profiling": 0.02,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.02,
            "preprocessing": 0.08,
            "feature_engineering": 0.03,
            "data_augmentation": 0.02,
            "model_selection": 0.04,
            "model_training": 0.38,
            "constraint_check": 0.02,
            "hyperparameter_tuning": 0.20,
            "ensembling": 0.05,
            "threshold_opt": 0.03,
            "calibration": 0.03,
            "post_training": 0.03,
            "explainability": 0.03,
        },
    ),
    "good_quality": PresetConfig(
        name="good_quality",
        description="Good accuracy with 1 hour default",
        default_time_limit=3600,  # 1 hour
        cv_folds=5,
        num_bag_folds=3,
        num_stack_levels=0,
        hyperparameter_tune=True,
        tune_trials=25,
        ensemble_method="auto",
        calibrate=True,
        use_holdout=False,
        holdout_frac=0.0,
        feature_engineering="moderate",
        model_pool=MODEL_POOLS["good_quality"],
        time_allocations={
            "profiling": 0.02,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.02,
            "preprocessing": 0.08,
            "feature_engineering": 0.03,
            "data_augmentation": 0.02,
            "model_selection": 0.04,
            "model_training": 0.42,
            "constraint_check": 0.02,
            "hyperparameter_tuning": 0.15,
            "ensembling": 0.05,
            "threshold_opt": 0.03,
            "calibration": 0.03,
            "post_training": 0.03,
            "explainability": 0.04,
        },
    ),
    "medium_quality": PresetConfig(
        name="medium_quality",
        description="Balanced speed/accuracy with 15 min default",
        default_time_limit=900,  # 15 minutes
        cv_folds=5,
        num_bag_folds=0,
        num_stack_levels=0,
        hyperparameter_tune=True,
        tune_trials=10,
        ensemble_method="auto",
        calibrate=False,
        use_holdout=False,
        holdout_frac=0.0,
        feature_engineering="light",
        model_pool=MODEL_POOLS["medium_quality"],
        time_allocations={
            "profiling": 0.02,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.01,
            "preprocessing": 0.05,
            "feature_engineering": 0.02,
            "data_augmentation": 0.01,
            "model_selection": 0.04,
            "model_training": 0.55,
            "constraint_check": 0.01,
            "hyperparameter_tuning": 0.12,
            "ensembling": 0.05,
            "threshold_opt": 0.03,
            "calibration": 0.03,
            "post_training": 0.01,
            "explainability": 0.03,
        },
    ),
    "fast": PresetConfig(
        name="fast",
        description="Quick results with 5 min default",
        default_time_limit=300,  # 5 minutes
        cv_folds=3,
        num_bag_folds=0,
        num_stack_levels=0,
        hyperparameter_tune=False,
        tune_trials=0,
        ensemble_method="none",
        calibrate=False,
        use_holdout=False,
        holdout_frac=0.0,
        feature_engineering="none",
        model_pool=MODEL_POOLS["fast"],
        time_allocations={
            "profiling": 0.02,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.0,
            "preprocessing": 0.10,
            "feature_engineering": 0.0,
            "data_augmentation": 0.0,
            "model_selection": 0.05,
            "model_training": 0.76,
            "constraint_check": 0.0,
            "hyperparameter_tuning": 0.0,
            "ensembling": 0.0,
            "calibration": 0.0,
            "post_training": 0.0,
            "threshold_opt": 0.05,
            "explainability": 0.0,
        },
    ),
    "exhaustive": PresetConfig(
        name="exhaustive",
        description="Evolutionary search over ALL models, preprocessing, and ensembles",
        default_time_limit=0,  # unlimited by default — use patience / keyboard interrupt
        cv_folds=3,
        num_bag_folds=3,
        num_stack_levels=1,
        hyperparameter_tune=True,
        tune_trials=30,
        ensemble_method="auto",
        calibrate=True,
        use_holdout=False,
        holdout_frac=0.0,
        feature_engineering="moderate",
        model_pool=MODEL_POOLS["all"],
        time_allocations={
            "profiling": 0.01,
            "quality_guardrails": 0.01,
            "data_cleaning": 0.01,
            "preprocessing": 0.02,
            "feature_engineering": 0.02,
            "data_augmentation": 0.01,
            "model_selection": 0.01,
            "model_training": 0.80,
            "constraint_check": 0.0,
            "hyperparameter_tuning": 0.0,
            "ensembling": 0.05,
            "threshold_opt": 0.01,
            "calibration": 0.02,
            "post_training": 0.01,
            "explainability": 0.02,
        },
    ),
    "interpretable": PresetConfig(
        name="interpretable",
        description="Only interpretable models",
        default_time_limit=900,  # 15 minutes
        cv_folds=3,
        num_bag_folds=0,
        num_stack_levels=0,
        hyperparameter_tune=True,
        tune_trials=25,
        ensemble_method="none",  # No ensemble for interpretability
        calibrate=True,
        use_holdout=False,
        holdout_frac=0.0,
        feature_engineering="light",
        model_pool=MODEL_POOLS["interpretable"],
        time_allocations={
            "profiling": 0.02,
            "quality_guardrails": 0.02,
            "data_cleaning": 0.0,
            "preprocessing": 0.05,
            "feature_engineering": 0.0,
            "data_augmentation": 0.0,
            "model_selection": 0.05,
            "model_training": 0.60,
            "constraint_check": 0.02,
            "hyperparameter_tuning": 0.10,
            "ensembling": 0.0,
            "calibration": 0.05,
            "post_training": 0.0,
            "threshold_opt": 0.03,
            "explainability": 0.06,
        },
    ),
}


def get_preset(name: str) -> PresetConfig:
    """Get a preset configuration by name.

    Parameters
    ----------
    name : str
        Name of the preset.

    Returns
    -------
    PresetConfig
        The preset configuration.

    Raises
    ------
    ValueError
        If preset name is not recognized.
    """
    if name not in PRESETS:
        available = ", ".join(PRESETS.keys())
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    return PRESETS[name]


def list_presets() -> list[str]:
    """List available preset names.

    Returns
    -------
    list of str
        Available preset names.
    """
    return list(PRESETS.keys())


def get_preset_summary() -> dict[str, dict[str, Any]]:
    """Get a summary of all presets.

    Returns
    -------
    dict
        Summary of each preset.
    """
    return {
        name: {
            "description": preset.description,
            "time_limit": preset.default_time_limit,
            "cv_folds": preset.cv_folds,
            "n_models": len(preset.model_pool),
            "ensemble": preset.ensemble_method,
            "calibrate": preset.calibrate,
        }
        for name, preset in PRESETS.items()
    }
