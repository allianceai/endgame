from __future__ import annotations

"""Utils module: Metrics, submission helpers, and reproducibility."""

from endgame.utils.metrics import (
    competition_metric,
    map_at_k,
    ndcg_at_k,
    quadratic_weighted_kappa,
)
from endgame.utils.reproducibility import SeedEverything, seed_everything
from endgame.utils.sharpe import (
    SharpeAnalysis,
    analyze_sharpe,
    deflated_sharpe_ratio,
    estimate_n_independent_trials,
    expected_max_sharpe,
    haircut_sharpe_ratio,
    minimum_track_record_length,
    multiple_testing_summary,
    probabilistic_sharpe_ratio,
    sharpe_ratio,
    sharpe_ratio_std,
)
from endgame.utils.submission import SubmissionHelper

__all__ = [
    # Metrics
    "quadratic_weighted_kappa",
    "map_at_k",
    "ndcg_at_k",
    "competition_metric",
    # Submission
    "SubmissionHelper",
    # Reproducibility
    "SeedEverything",
    "seed_everything",
    # Sharpe Ratio & Multiple Testing
    "sharpe_ratio",
    "sharpe_ratio_std",
    "probabilistic_sharpe_ratio",
    "expected_max_sharpe",
    "deflated_sharpe_ratio",
    "analyze_sharpe",
    "minimum_track_record_length",
    "haircut_sharpe_ratio",
    "estimate_n_independent_trials",
    "multiple_testing_summary",
    "SharpeAnalysis",
]
