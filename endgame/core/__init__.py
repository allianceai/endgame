from __future__ import annotations

"""Core module: Base classes, configuration, and utilities."""

from endgame.core.base import (
    EndgameClassifierMixin,
    EndgameEstimator,
    EndgameRegressorMixin,
    PolarsTransformer,
)
from endgame.core.config import (
    CATBOOST_ENDGAME_DEFAULTS,
    LGBM_ENDGAME_DEFAULTS,
    LGBM_FAST_DEFAULTS,
    XGB_ENDGAME_DEFAULTS,
    XGB_FAST_DEFAULTS,
    get_preset,
)
from endgame.core.polars_ops import (
    from_lazyframe,
    infer_categorical_columns,
    infer_numeric_columns,
    to_lazyframe,
)
from endgame.core.types import (
    AdversarialValidationResult,
    ArrayLike,
    FrameLike,
    OOFResult,
    OptimizationResult,
)

__all__ = [
    # Base classes
    "EndgameEstimator",
    "PolarsTransformer",
    "EndgameClassifierMixin",
    "EndgameRegressorMixin",
    # Config
    "LGBM_ENDGAME_DEFAULTS",
    "XGB_ENDGAME_DEFAULTS",
    "CATBOOST_ENDGAME_DEFAULTS",
    "LGBM_FAST_DEFAULTS",
    "XGB_FAST_DEFAULTS",
    "get_preset",
    # Types
    "AdversarialValidationResult",
    "OOFResult",
    "OptimizationResult",
    "ArrayLike",
    "FrameLike",
    # Polars ops
    "to_lazyframe",
    "from_lazyframe",
    "infer_categorical_columns",
    "infer_numeric_columns",
]
