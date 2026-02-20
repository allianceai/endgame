"""Predefined hyperparameter search spaces for common models."""

from typing import Any

# =============================================================================
# LightGBM Search Spaces
# =============================================================================

# Rationale for n_estimators=100-1000 range:
# - With early_stopping_rounds=50, models converge well before 1000 trees
# - Gorishniy et al. (2021) "Revisiting Deep Learning for Tabular Data" uses 2000
#   but with early stopping, achieving similar effective ensemble sizes
# - Reduces worst-case training time by 2x while early stopping handles convergence
# - Upper bound of 1000 still allows complex patterns; early stopping prevents waste
LGBM_STANDARD_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.2},
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "num_leaves": {"type": "int", "low": 16, "high": 128},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "min_child_samples": {"type": "int", "low": 5, "high": 100},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "reg_alpha": {"type": "loguniform", "low": 1e-8, "high": 10.0},
    "reg_lambda": {"type": "loguniform", "low": 1e-8, "high": 10.0},
}

LGBM_LARGE_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.001, "high": 0.3},
    "n_estimators": {"type": "int", "low": 50, "high": 5000},
    "num_leaves": {"type": "int", "low": 8, "high": 256},
    "max_depth": {"type": "int", "low": 2, "high": 16},
    "min_child_samples": {"type": "int", "low": 1, "high": 200},
    "subsample": {"type": "float", "low": 0.3, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
    "reg_alpha": {"type": "loguniform", "low": 1e-10, "high": 100.0},
    "reg_lambda": {"type": "loguniform", "low": 1e-10, "high": 100.0},
    "min_split_gain": {"type": "loguniform", "low": 1e-8, "high": 1.0},
    "path_smooth": {"type": "float", "low": 0.0, "high": 10.0},
}

LGBM_FAST_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "num_leaves": {"type": "int", "low": 16, "high": 64},
    "max_depth": {"type": "int", "low": 4, "high": 8},
}

# =============================================================================
# XGBoost Search Spaces
# =============================================================================

# Same rationale as LGBM: early stopping makes high n_estimators wasteful
# max_depth capped at 10 per Grinsztajn et al. finding that deeper trees rarely help
XGB_STANDARD_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.2},
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "min_child_weight": {"type": "int", "low": 1, "high": 20},
    "subsample": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
    "colsample_bylevel": {"type": "float", "low": 0.5, "high": 1.0},
    "reg_alpha": {"type": "loguniform", "low": 1e-8, "high": 10.0},
    "reg_lambda": {"type": "loguniform", "low": 1e-8, "high": 10.0},
    "gamma": {"type": "loguniform", "low": 1e-8, "high": 5.0},
}

XGB_LARGE_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.001, "high": 0.3},
    "n_estimators": {"type": "int", "low": 50, "high": 5000},
    "max_depth": {"type": "int", "low": 2, "high": 16},
    "min_child_weight": {"type": "int", "low": 1, "high": 100},
    "subsample": {"type": "float", "low": 0.3, "high": 1.0},
    "colsample_bytree": {"type": "float", "low": 0.3, "high": 1.0},
    "colsample_bylevel": {"type": "float", "low": 0.3, "high": 1.0},
    "colsample_bynode": {"type": "float", "low": 0.3, "high": 1.0},
    "reg_alpha": {"type": "loguniform", "low": 1e-10, "high": 100.0},
    "reg_lambda": {"type": "loguniform", "low": 1e-10, "high": 100.0},
    "gamma": {"type": "loguniform", "low": 1e-10, "high": 10.0},
    "max_delta_step": {"type": "int", "low": 0, "high": 10},
}

# =============================================================================
# CatBoost Search Spaces
# =============================================================================

# CatBoost: depth has outsized impact on training time, cap at 8
# CatBoost's ordered boosting is inherently slower, so tighter bounds help
CATBOOST_STANDARD_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.005, "high": 0.2},
    "iterations": {"type": "int", "low": 100, "high": 1000},
    "depth": {"type": "int", "low": 4, "high": 8},
    "l2_leaf_reg": {"type": "loguniform", "low": 1e-8, "high": 10.0},
    "bagging_temperature": {"type": "float", "low": 0.0, "high": 1.0},
    "random_strength": {"type": "float", "low": 0.0, "high": 10.0},
    "border_count": {"type": "int", "low": 32, "high": 255},
}

CATBOOST_LARGE_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 0.001, "high": 0.3},
    "iterations": {"type": "int", "low": 50, "high": 5000},
    "depth": {"type": "int", "low": 2, "high": 12},
    "l2_leaf_reg": {"type": "loguniform", "low": 1e-10, "high": 100.0},
    "bagging_temperature": {"type": "float", "low": 0.0, "high": 5.0},
    "random_strength": {"type": "float", "low": 0.0, "high": 20.0},
    "border_count": {"type": "int", "low": 16, "high": 255},
    "min_data_in_leaf": {"type": "int", "low": 1, "high": 100},
}

# =============================================================================
# Neural Network Search Spaces
# =============================================================================

MLP_STANDARD_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 1e-5, "high": 1e-2},
    "weight_decay": {"type": "loguniform", "low": 1e-8, "high": 1e-2},
    "dropout": {"type": "float", "low": 0.0, "high": 0.5},
    "hidden_dim": {"type": "categorical", "choices": [64, 128, 256, 512]},
    "n_layers": {"type": "int", "low": 1, "high": 4},
    "batch_size": {"type": "categorical", "choices": [32, 64, 128, 256]},
}

TRANSFORMER_STANDARD_SPACE: dict[str, dict[str, Any]] = {
    "learning_rate": {"type": "loguniform", "low": 1e-6, "high": 5e-4},
    "weight_decay": {"type": "loguniform", "low": 1e-8, "high": 1e-1},
    "warmup_ratio": {"type": "float", "low": 0.0, "high": 0.2},
    "num_epochs": {"type": "int", "low": 2, "high": 10},
    "batch_size": {"type": "categorical", "choices": [8, 16, 32]},
    "max_length": {"type": "categorical", "choices": [256, 384, 512]},
}

# Rotation Forest search space
# Note: RotationForest uses base_estimator for tree params, only tune ensemble-level params
# n_estimators capped at 500 since RF is much slower than GBDT per tree
ROTATION_FOREST_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "low": 50, "high": 500},
    "n_subsets": {"type": "int", "low": 2, "high": 4},
    "max_features": {"type": "float", "low": 0.5, "high": 1.0},
}

# Random Forest search space - matched to GBDT ranges for fair comparison
# RF benefits less from very high n_estimators due to lack of boosting
RANDOM_FOREST_SPACE: dict[str, dict[str, Any]] = {
    "n_estimators": {"type": "int", "low": 100, "high": 1000},
    "max_depth": {"type": "int", "low": 3, "high": 10},
    "min_samples_split": {"type": "int", "low": 2, "high": 20},
    "min_samples_leaf": {"type": "int", "low": 1, "high": 10},
    "max_features": {"type": "categorical", "choices": ["sqrt", "log2", 0.5, 0.8, 1.0]},
}

# =============================================================================
# Registry
# =============================================================================

SPACE_REGISTRY: dict[str, dict[str, dict[str, Any]]] = {
    "lgbm_standard": LGBM_STANDARD_SPACE,
    "lgbm_large": LGBM_LARGE_SPACE,
    "lgbm_fast": LGBM_FAST_SPACE,
    "xgb_standard": XGB_STANDARD_SPACE,
    "xgb_large": XGB_LARGE_SPACE,
    "catboost_standard": CATBOOST_STANDARD_SPACE,
    "catboost_large": CATBOOST_LARGE_SPACE,
    "mlp_standard": MLP_STANDARD_SPACE,
    "transformer_standard": TRANSFORMER_STANDARD_SPACE,
    "rotation_forest": ROTATION_FOREST_SPACE,
    "random_forest": RANDOM_FOREST_SPACE,
}


def get_space(name: str) -> dict[str, dict[str, Any]]:
    """Get a predefined search space by name.

    Parameters
    ----------
    name : str
        Space name: 'lgbm_standard', 'xgb_standard', etc.

    Returns
    -------
    Dict
        Search space configuration.

    Raises
    ------
    ValueError
        If space name is not found.
    """
    if name not in SPACE_REGISTRY:
        raise ValueError(
            f"Unknown space '{name}'. "
            f"Available: {list(SPACE_REGISTRY.keys())}"
        )
    return SPACE_REGISTRY[name].copy()


def get_lgbm_space(size: str = "standard") -> dict[str, dict[str, Any]]:
    """Get LightGBM search space.

    Parameters
    ----------
    size : str, default='standard'
        Space size: 'standard', 'large', 'fast'.

    Returns
    -------
    Dict
        Search space configuration.
    """
    return get_space(f"lgbm_{size}")


def get_xgb_space(size: str = "standard") -> dict[str, dict[str, Any]]:
    """Get XGBoost search space.

    Parameters
    ----------
    size : str, default='standard'
        Space size: 'standard', 'large'.

    Returns
    -------
    Dict
        Search space configuration.
    """
    return get_space(f"xgb_{size}")


def get_catboost_space(size: str = "standard") -> dict[str, dict[str, Any]]:
    """Get CatBoost search space.

    Parameters
    ----------
    size : str, default='standard'
        Space size: 'standard', 'large'.

    Returns
    -------
    Dict
        Search space configuration.
    """
    return get_space(f"catboost_{size}")
