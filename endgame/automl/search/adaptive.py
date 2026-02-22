"""Adaptive strategy switching for AutoML search.

Wraps multiple search strategies and switches between them based on
performance feedback.  The default schedule is:

1. **Portfolio** phase — diverse model sweep for initial coverage.
2. **Bayesian** phase — focused HPO on the top models.
3. **Genetic** phase — pipeline co-evolution for long runs.

The switch happens when a strategy stagnates (no improvement for
``switch_patience`` rounds) or when the phase budget is exhausted.
"""

from __future__ import annotations

import logging
from typing import Any

from endgame.automl.search.base import (
    BaseSearchStrategy,
    PipelineConfig,
    SearchResult,
)

logger = logging.getLogger(__name__)


class AdaptiveSearch(BaseSearchStrategy):
    """Meta-strategy that switches between sub-strategies adaptively.

    Parameters
    ----------
    strategies : list of (BaseSearchStrategy, int)
        Ordered list of ``(strategy, max_rounds)`` pairs.  Each
        strategy runs for up to ``max_rounds`` suggest/update cycles
        before the next one takes over.  A ``max_rounds`` of ``0``
        means unlimited (last strategy runs until the outer loop stops).
    switch_patience : int, default=3
        Switch to the next strategy early if this many consecutive
        rounds produce no improvement.
    verbose : int, default=0
        Verbosity level.

    Examples
    --------
    >>> from endgame.automl.search.portfolio import PortfolioSearch
    >>> from endgame.automl.search.bayesian import BayesianSearch
    >>> adaptive = AdaptiveSearch(
    ...     strategies=[
    ...         (PortfolioSearch(task_type="classification"), 10),
    ...         (BayesianSearch(task_type="classification"), 0),
    ...     ],
    ...     switch_patience=3,
    ... )
    """

    def __init__(
        self,
        strategies: list[tuple[BaseSearchStrategy, int]],
        switch_patience: int = 3,
        verbose: int = 0,
        **kwargs,
    ):
        # Extract common params from first strategy
        first = strategies[0][0] if strategies else None
        super().__init__(
            task_type=getattr(first, "task_type", "classification"),
            eval_metric=getattr(first, "eval_metric", "auto"),
            random_state=getattr(first, "random_state", None),
            verbose=verbose,
        )
        self._strategies = strategies
        self.switch_patience = switch_patience

        self._phase: int = 0
        self._phase_rounds: int = 0
        self._phase_best_score: float = -float("inf")
        self._phase_stale: int = 0

    # ── Properties ──────────────────────────────────────────────────

    @property
    def current_strategy(self) -> BaseSearchStrategy:
        """The currently active sub-strategy."""
        return self._strategies[self._phase][0]

    @property
    def current_phase(self) -> int:
        return self._phase

    @property
    def phase_name(self) -> str:
        return type(self.current_strategy).__name__

    # ── Public API ──────────────────────────────────────────────────

    def suggest(
        self,
        meta_features: dict[str, float] | None = None,
        n_suggestions: int = 1,
    ) -> list[PipelineConfig]:
        """Suggest configs from the current strategy, switching if needed."""
        # Check if we should advance to the next phase
        self._maybe_switch()

        strategy = self.current_strategy
        configs = strategy.suggest(meta_features, n_suggestions)

        # If current strategy is exhausted, try next
        if not configs and self._phase < len(self._strategies) - 1:
            self._advance_phase("exhausted")
            configs = self.current_strategy.suggest(meta_features, n_suggestions)

        # Tag configs with phase info
        for c in configs:
            c.metadata["adaptive_phase"] = self._phase
            c.metadata["adaptive_strategy"] = self.phase_name

        return configs

    def update(self, result: SearchResult) -> None:
        """Route result to the current strategy and track improvement."""
        super().update(result)

        # Forward to current strategy
        try:
            self.current_strategy.update(result)
        except Exception as e:
            logger.debug(f"Strategy update failed: {e}")

        self._phase_rounds += 1

        # Track improvement within this phase
        if result.success and result.score > self._phase_best_score + 1e-5:
            self._phase_best_score = result.score
            self._phase_stale = 0
        else:
            self._phase_stale += 1

    def should_stop(self, max_iterations: int | None = None) -> bool:
        """Stop only if all strategies are exhausted."""
        if max_iterations and self.n_evaluated_ >= max_iterations:
            return True
        # If we're on the last strategy, delegate
        if self._phase >= len(self._strategies) - 1:
            return self.current_strategy.should_stop(max_iterations)
        return False

    def set_feature_importance_feedback(
        self, mask: Any, scores: Any = None,
    ) -> None:
        """Forward feedback to all strategies that support it."""
        for strategy, _ in self._strategies:
            if hasattr(strategy, "set_feature_importance_feedback"):
                strategy.set_feature_importance_feedback(mask, scores)

    # ── Phase management ────────────────────────────────────────────

    def _maybe_switch(self) -> None:
        """Check if we should advance to the next strategy phase."""
        if self._phase >= len(self._strategies) - 1:
            return  # Already on last strategy

        _, max_rounds = self._strategies[self._phase]

        # Switch if max_rounds reached (0 = unlimited)
        if max_rounds > 0 and self._phase_rounds >= max_rounds:
            self._advance_phase("budget")
            return

        # Switch if stagnating
        if self._phase_stale >= self.switch_patience:
            self._advance_phase("stagnation")
            return

    def _advance_phase(self, reason: str) -> None:
        """Move to the next strategy phase."""
        old_name = self.phase_name
        self._phase = min(self._phase + 1, len(self._strategies) - 1)
        new_name = self.phase_name

        # Reset phase counters
        self._phase_rounds = 0
        self._phase_stale = 0
        # Carry over best score so improvement tracking is global
        # (don't reset _phase_best_score)

        # Sync the new strategy with all results collected so far
        new_strategy = self.current_strategy
        synced = {r.config.config_id for r in new_strategy.results_} if new_strategy.results_ else set()
        for r in self.results_:
            if r.config.config_id not in synced:
                try:
                    new_strategy.update(r)
                except Exception:
                    pass

        if self.verbose > 0:
            print(
                f"  [Adaptive] Switching {old_name} -> {new_name} "
                f"(reason: {reason}, after {self._phase_rounds} rounds)"
            )
        logger.info(
            f"Adaptive strategy switch: {old_name} -> {new_name} ({reason})"
        )
