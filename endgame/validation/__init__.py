"""Validation module: CV strategies, drift detection, and trust-cv utilities."""

from endgame.validation.adversarial import AdversarialValidator
from endgame.validation.cv_utils import (
    check_cv_lb_correlation,
    cross_validate_oof,
)
from endgame.validation.nested_cv import NestedCV, NestedCVResult
from endgame.validation.splitters import (
    AdversarialKFold,
    CombinatorialPurgedKFold,
    MultilabelStratifiedKFold,
    PurgedTimeSeriesSplit,
    RepeatedStratifiedGroupKFold,
    StratifiedGroupKFold,
)

__all__ = [
    # Adversarial validation
    "AdversarialValidator",
    # CV splitters
    "PurgedTimeSeriesSplit",
    "StratifiedGroupKFold",
    "MultilabelStratifiedKFold",
    "AdversarialKFold",
    "RepeatedStratifiedGroupKFold",
    "CombinatorialPurgedKFold",
    # CV utilities
    "cross_validate_oof",
    "check_cv_lb_correlation",
    # Nested CV
    "NestedCV",
    "NestedCVResult",
]
