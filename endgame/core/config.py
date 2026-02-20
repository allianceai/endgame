"""Default configurations and hyperparameter presets for Endgame."""

from typing import Any, Literal

# =============================================================================
# LightGBM Presets
# =============================================================================

LGBM_ENDGAME_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.01,
    "n_estimators": 10000,
    "num_leaves": 31,
    "max_depth": 7,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_samples": 20,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_FAST_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "num_leaves": 63,
    "max_depth": 8,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

LGBM_OVERFIT_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.1,
    "n_estimators": 500,
    "num_leaves": 127,
    "max_depth": -1,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "min_child_samples": 1,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": 42,
}

# =============================================================================
# XGBoost Presets
# =============================================================================

XGB_ENDGAME_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.01,
    "n_estimators": 10000,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "colsample_bylevel": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "min_child_weight": 1,
    "tree_method": "hist",
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

XGB_FAST_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.05,
    "n_estimators": 1000,
    "max_depth": 8,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

XGB_OVERFIT_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.1,
    "n_estimators": 500,
    "max_depth": 10,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "verbosity": 0,
    "n_jobs": -1,
    "random_state": 42,
}

# =============================================================================
# CatBoost Presets
# =============================================================================

CATBOOST_ENDGAME_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.01,
    "iterations": 10000,
    "depth": 7,
    "l2_leaf_reg": 3.0,
    "bagging_temperature": 1.0,
    "random_strength": 1.0,
    "border_count": 254,
    "grow_policy": "SymmetricTree",
    "verbose": False,
    "thread_count": -1,
    "random_seed": 42,
}

CATBOOST_FAST_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.1,
    "iterations": 200,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "verbose": False,
    "thread_count": -1,
    "random_seed": 42,
}

CATBOOST_OVERFIT_DEFAULTS: dict[str, Any] = {
    "learning_rate": 0.1,
    "iterations": 500,
    "depth": 10,
    "l2_leaf_reg": 0.0,
    "verbose": False,
    "thread_count": -1,
    "random_seed": 42,
}

# =============================================================================
# Neural Network Presets
# =============================================================================

MLP_ENDGAME_DEFAULTS: dict[str, Any] = {
    "hidden_dims": [256, 128],
    "dropout": 0.3,
    "batch_norm": True,
    "activation": "relu",
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "n_epochs": 100,
    "batch_size": 256,
    "early_stopping_patience": 10,
}

TABNET_ENDGAME_DEFAULTS: dict[str, Any] = {
    "n_d": 64,
    "n_a": 64,
    "n_steps": 5,
    "gamma": 1.5,
    "n_independent": 2,
    "n_shared": 2,
    "momentum": 0.3,
    "mask_type": "entmax",
}

# =============================================================================
# Transformer Presets
# =============================================================================

DEBERTA_ENDGAME_DEFAULTS: dict[str, Any] = {
    "model_name": "microsoft/deberta-v3-large",
    "max_length": 512,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "num_epochs": 5,
    "batch_size": 16,
    "gradient_accumulation_steps": 1,
    "fp16": True,
    "weight_decay": 0.01,
}

# =============================================================================
# Preset Registry
# =============================================================================

PresetName = Literal["endgame", "fast", "overfit", "custom"]
BackendName = Literal["lightgbm", "xgboost", "catboost", "mlp", "tabnet", "deberta"]

_PRESET_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "lightgbm": {
        "endgame": LGBM_ENDGAME_DEFAULTS,
        "fast": LGBM_FAST_DEFAULTS,
        "overfit": LGBM_OVERFIT_DEFAULTS,
    },
    "xgboost": {
        "endgame": XGB_ENDGAME_DEFAULTS,
        "fast": XGB_FAST_DEFAULTS,
        "overfit": XGB_OVERFIT_DEFAULTS,
    },
    "catboost": {
        "endgame": CATBOOST_ENDGAME_DEFAULTS,
        "fast": CATBOOST_FAST_DEFAULTS,
        "overfit": CATBOOST_OVERFIT_DEFAULTS,
    },
    "mlp": {
        "endgame": MLP_ENDGAME_DEFAULTS,
    },
    "tabnet": {
        "endgame": TABNET_ENDGAME_DEFAULTS,
    },
    "deberta": {
        "endgame": DEBERTA_ENDGAME_DEFAULTS,
    },
}


def get_preset(backend: BackendName, preset: PresetName = "endgame") -> dict[str, Any]:
    """Get hyperparameter preset for a given backend.

    Parameters
    ----------
    backend : str
        Model backend: 'lightgbm', 'xgboost', 'catboost', 'mlp', 'tabnet', 'deberta'
    preset : str, default='endgame'
        Preset name: 'endgame' (competition-tuned), 'fast', 'overfit', 'custom'

    Returns
    -------
    Dict[str, Any]
        Hyperparameter dictionary.

    Raises
    ------
    ValueError
        If backend or preset is not recognized.

    Examples
    --------
    >>> params = get_preset('lightgbm', 'endgame')
    >>> params['learning_rate']
    0.01
    """
    if backend not in _PRESET_REGISTRY:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            f"Available: {list(_PRESET_REGISTRY.keys())}"
        )

    presets = _PRESET_REGISTRY[backend]
    if preset == "custom":
        return {}
    if preset not in presets:
        raise ValueError(
            f"Unknown preset '{preset}' for backend '{backend}'. "
            f"Available: {list(presets.keys())}"
        )

    return presets[preset].copy()


# =============================================================================
# Early Stopping Defaults
# =============================================================================

EARLY_STOPPING_DEFAULTS: dict[str, int] = {
    "lightgbm": 100,
    "xgboost": 100,
    "catboost": 100,
    "neural": 10,
}


# =============================================================================
# Cross-Validation Defaults
# =============================================================================

CV_DEFAULTS: dict[str, Any] = {
    "n_splits": 5,
    "shuffle": True,
    "random_state": 42,
}

PURGED_CV_DEFAULTS: dict[str, Any] = {
    "n_splits": 5,
    "purge_gap": 0,
    "embargo_pct": 0.01,
}
