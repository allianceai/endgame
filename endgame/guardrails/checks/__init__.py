from __future__ import annotations

"""Check registry for guardrails module.

Each check is registered as (function, default_enabled, cost_tier)
where cost_tier is one of "cheap", "moderate", "expensive".
"""

from typing import Callable

from endgame.guardrails.checks.data_health import (
    check_class_imbalance,
    check_constant_columns,
    check_feature_sample_ratio,
    check_id_columns,
    check_missing_columns,
    check_sample_count,
    check_suspect_features,
    check_variance,
)
from endgame.guardrails.checks.leakage import (
    check_categorical_leakage,
    check_correlation_leakage,
    check_mutual_info_leakage,
)
from endgame.guardrails.checks.redundancy import (
    check_near_duplicate_columns,
    check_pairwise_correlation,
    check_vif,
)
from endgame.guardrails.checks.train_test import (
    check_adversarial_drift,
    check_duplicate_rows,
    check_temporal_leakage,
)

# (function, default_enabled, cost_tier, category)
CHECK_REGISTRY: dict[str, tuple[Callable, bool, str, str]] = {
    # Data health — cheap
    "constant_columns": (check_constant_columns, True, "cheap", "data_health"),
    "missing_columns": (check_missing_columns, True, "cheap", "data_health"),
    "sample_count": (check_sample_count, True, "cheap", "data_health"),
    "feature_sample_ratio": (check_feature_sample_ratio, True, "cheap", "data_health"),
    "id_columns": (check_id_columns, True, "cheap", "data_health"),
    "variance": (check_variance, True, "cheap", "data_health"),
    "class_imbalance": (check_class_imbalance, True, "cheap", "data_health"),
    "suspect_features": (check_suspect_features, False, "cheap", "data_health"),
    # Leakage — moderate
    "correlation_leakage": (check_correlation_leakage, True, "moderate", "leakage"),
    "mutual_info_leakage": (check_mutual_info_leakage, True, "moderate", "leakage"),
    "categorical_leakage": (check_categorical_leakage, True, "moderate", "leakage"),
    # Redundancy — moderate/expensive
    "pairwise_correlation": (check_pairwise_correlation, True, "moderate", "redundancy"),
    "near_duplicate_columns": (check_near_duplicate_columns, True, "moderate", "redundancy"),
    "vif": (check_vif, False, "expensive", "redundancy"),
    # Train/test — moderate/expensive
    "duplicate_rows": (check_duplicate_rows, True, "moderate", "train_test"),
    "adversarial_drift": (check_adversarial_drift, False, "expensive", "train_test"),
    "temporal_leakage": (check_temporal_leakage, False, "cheap", "train_test"),
}

# Cost tier ordering for execution priority
COST_ORDER = {"cheap": 0, "moderate": 1, "expensive": 2}
