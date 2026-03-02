from __future__ import annotations

"""Time budget management for AutoML pipelines.

This module provides utilities for managing time allocation across
different stages of the AutoML pipeline, with support for redistribution
and graceful timeout handling.
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class StageResult:
    """Result from executing a pipeline stage.

    Attributes
    ----------
    stage_name : str
        Name of the stage.
    success : bool
        Whether the stage completed successfully.
    duration : float
        Time taken in seconds.
    output : Any
        Output from the stage.
    timed_out : bool
        Whether the stage timed out.
    error : str or None
        Error message if the stage failed.
    metadata : dict
        Additional metadata about the stage execution.
    """

    stage_name: str
    success: bool
    duration: float
    output: Any = None
    timed_out: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TimeBudgetManager:
    """Manages time allocation across AutoML pipeline stages.

    This class tracks time spent in each stage and allows for dynamic
    reallocation of unused time to other stages.

    Parameters
    ----------
    total_budget : float
        Total time budget in seconds.
    allocations : dict
        Dictionary mapping stage names to allocation fractions (0.0 to 1.0).
        Fractions should sum to 1.0.
    min_stage_time : float, default=5.0
        Minimum time (in seconds) to allocate to any stage.

    Examples
    --------
    >>> allocations = {
    ...     "profiling": 0.05,
    ...     "training": 0.70,
    ...     "ensemble": 0.25,
    ... }
    >>> mgr = TimeBudgetManager(total_budget=600, allocations=allocations)
    >>> mgr.start()
    >>> mgr.begin_stage("profiling")
    >>> # ... do profiling ...
    >>> mgr.end_stage()
    >>> print(mgr.remaining_budget("training"))  # Training time remaining
    """

    def __init__(
        self,
        total_budget: float,
        allocations: dict[str, float],
        min_stage_time: float = 5.0,
    ):
        self.total_budget = total_budget
        self._original_allocations = allocations.copy()
        self.allocations = allocations.copy()
        self.min_stage_time = min_stage_time

        # Validate allocations
        total_alloc = sum(allocations.values())
        if abs(total_alloc - 1.0) > 0.01:
            logger.warning(
                f"Time allocations sum to {total_alloc:.2f}, not 1.0. Normalizing."
            )
            for key in self.allocations:
                self.allocations[key] /= total_alloc

        # State tracking
        self._start_time: float | None = None
        self._stage_times: dict[str, float] = {}
        self._stage_budgets: dict[str, float] = {}
        self._current_stage: str | None = None
        self._stage_start: float | None = None
        self._completed_stages: set = set()

        # Pre-compute stage budgets.
        # When total_budget == 0 (unlimited), each stage gets a generous
        # default so the pipeline doesn't skip stages.
        if total_budget <= 0:
            for stage, frac in self.allocations.items():
                self._stage_budgets[stage] = float("inf")
        else:
            for stage, frac in self.allocations.items():
                self._stage_budgets[stage] = max(
                    total_budget * frac, min_stage_time
                )

    def start(self) -> TimeBudgetManager:
        """Start the overall timer.

        Returns
        -------
        TimeBudgetManager
            Self for chaining.
        """
        self._start_time = time.time()
        logger.debug(f"TimeBudgetManager started with {self.total_budget}s budget")
        return self

    def begin_stage(self, stage_name: str) -> float:
        """Begin a new pipeline stage.

        Parameters
        ----------
        stage_name : str
            Name of the stage to begin.

        Returns
        -------
        float
            Budget available for this stage in seconds.

        Raises
        ------
        ValueError
            If a stage is already in progress.
        """
        if self._current_stage is not None:
            raise ValueError(
                f"Stage '{self._current_stage}' is already in progress. "
                f"Call end_stage() before beginning '{stage_name}'."
            )

        self._current_stage = stage_name
        self._stage_start = time.time()

        budget = self.remaining_budget(stage_name)
        logger.debug(f"Beginning stage '{stage_name}' with {budget:.1f}s budget")
        return budget

    def end_stage(self) -> float:
        """End the current stage.

        Returns
        -------
        float
            Duration of the stage in seconds.

        Raises
        ------
        ValueError
            If no stage is in progress.
        """
        if self._current_stage is None:
            raise ValueError("No stage is currently in progress.")

        duration = time.time() - self._stage_start
        self._stage_times[self._current_stage] = duration
        self._completed_stages.add(self._current_stage)

        logger.debug(
            f"Ended stage '{self._current_stage}' after {duration:.1f}s "
            f"(budget was {self._stage_budgets.get(self._current_stage, 0):.1f}s)"
        )

        stage_name = self._current_stage
        self._current_stage = None
        self._stage_start = None

        return duration

    def remaining_budget(self, stage_name: str | None = None) -> float:
        """Get remaining time budget.

        Parameters
        ----------
        stage_name : str, optional
            If provided, return remaining budget for that specific stage.
            Otherwise, return overall remaining budget.

        Returns
        -------
        float
            Remaining time in seconds.
        """
        if self._start_time is None:
            if stage_name:
                return self._stage_budgets.get(stage_name, 0)
            return self.total_budget if self.total_budget > 0 else float("inf")

        elapsed = self.elapsed()

        # total_budget == 0 means unlimited
        if self.total_budget <= 0:
            overall_remaining = float("inf")
        else:
            overall_remaining = max(0, self.total_budget - elapsed)

        if stage_name is None:
            return overall_remaining

        # Stage-specific remaining
        stage_budget = self._stage_budgets.get(stage_name, 0)
        stage_used = self._stage_times.get(stage_name, 0)

        # If stage is currently running, include current time
        if self._current_stage == stage_name and self._stage_start is not None:
            stage_used += time.time() - self._stage_start

        stage_remaining = max(0, stage_budget - stage_used)

        # Don't exceed overall remaining
        return min(stage_remaining, overall_remaining)

    def elapsed(self) -> float:
        """Get total elapsed time since start.

        Returns
        -------
        float
            Elapsed time in seconds.
        """
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def is_overtime(self) -> bool:
        """Check if overall time budget has been exceeded.

        Returns
        -------
        bool
            True if overtime.
        """
        if self.total_budget is None:
            return False
        return self.elapsed() > self.total_budget

    def is_stage_overtime(self, stage_name: str | None = None) -> bool:
        """Check if a specific stage is overtime.

        Parameters
        ----------
        stage_name : str, optional
            Stage to check. If None, checks current stage.

        Returns
        -------
        bool
            True if stage is overtime.
        """
        stage = stage_name or self._current_stage
        if stage is None:
            return False
        return self.remaining_budget(stage) <= 0

    def redistribute(
        self,
        from_stage: str,
        to_stage: str,
        fraction: float = 1.0,
    ) -> float:
        """Redistribute unused time from one stage to another.

        Parameters
        ----------
        from_stage : str
            Stage to take time from.
        to_stage : str
            Stage to give time to.
        fraction : float, default=1.0
            Fraction of unused time to redistribute.

        Returns
        -------
        float
            Amount of time redistributed in seconds.
        """
        unused = self.remaining_budget(from_stage)
        transfer = unused * fraction

        if transfer > 0:
            # Update budgets
            old_to_budget = self._stage_budgets.get(to_stage, 0)
            self._stage_budgets[to_stage] = old_to_budget + transfer
            self._stage_budgets[from_stage] = (
                self._stage_budgets.get(from_stage, 0) - transfer
            )

            logger.debug(
                f"Redistributed {transfer:.1f}s from '{from_stage}' to '{to_stage}'"
            )

        return transfer

    def redistribute_remaining(self, to_stage: str) -> float:
        """Redistribute all unused time from completed stages to a target stage.

        Parameters
        ----------
        to_stage : str
            Stage to give time to.

        Returns
        -------
        float
            Total amount redistributed in seconds.
        """
        total_redistributed = 0

        for stage in self._completed_stages:
            if stage != to_stage:
                redistributed = self.redistribute(stage, to_stage, fraction=1.0)
                total_redistributed += redistributed

        return total_redistributed

    def estimate_remaining_iterations(
        self,
        per_iteration_time: float,
        stage_name: str | None = None,
    ) -> int:
        """Estimate how many more iterations can be completed.

        Parameters
        ----------
        per_iteration_time : float
            Estimated time per iteration in seconds.
        stage_name : str, optional
            Stage to check. If None, uses overall remaining.

        Returns
        -------
        int
            Estimated number of remaining iterations.
        """
        remaining = self.remaining_budget(stage_name)
        if per_iteration_time <= 0:
            return 0
        return int(remaining / per_iteration_time)

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of time usage.

        Returns
        -------
        dict
            Summary with elapsed, remaining, and per-stage times.
        """
        return {
            "total_budget": self.total_budget,
            "elapsed": self.elapsed(),
            "remaining": self.remaining_budget(),
            "overtime": self.is_overtime(),
            "current_stage": self._current_stage,
            "completed_stages": list(self._completed_stages),
            "stage_times": self._stage_times.copy(),
            "stage_budgets": self._stage_budgets.copy(),
        }

    def __repr__(self) -> str:
        elapsed = self.elapsed()
        remaining = self.remaining_budget()
        return (
            f"TimeBudgetManager("
            f"elapsed={elapsed:.1f}s, "
            f"remaining={remaining:.1f}s, "
            f"stages={len(self._completed_stages)}/{len(self.allocations)})"
        )


class TimeoutContext:
    """Context manager for executing code with a soft timeout.

    This provides a soft timeout that can be checked periodically,
    rather than a hard interrupt-based timeout.

    Parameters
    ----------
    timeout : float
        Timeout in seconds.
    check_interval : float, default=1.0
        How often to update the timeout flag (not used directly,
        but useful for documentation).

    Examples
    --------
    >>> with TimeoutContext(60) as ctx:
    ...     for i in range(1000):
    ...         if ctx.is_timed_out():
    ...             break
    ...         # ... do work ...
    """

    def __init__(self, timeout: float, check_interval: float = 1.0):
        self.timeout = timeout
        self.check_interval = check_interval
        self._start_time: float | None = None
        self._timed_out = False

    def __enter__(self) -> TimeoutContext:
        self._start_time = time.time()
        self._timed_out = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._timed_out = self.elapsed() > self.timeout
        return False

    def elapsed(self) -> float:
        """Get elapsed time since entering context."""
        if self._start_time is None:
            return 0.0
        return time.time() - self._start_time

    def remaining(self) -> float:
        """Get remaining time."""
        return max(0, self.timeout - self.elapsed())

    def is_timed_out(self) -> bool:
        """Check if timeout has been exceeded."""
        if self._start_time is None:
            return False
        return self.elapsed() > self.timeout


def run_with_timeout(
    func: Callable[..., T],
    timeout: float,
    *args,
    default: T | None = None,
    **kwargs,
) -> tuple[T, bool, float]:
    """Run a function with a timeout, returning early if possible.

    Note: This is a soft timeout. The function must check for timeout
    periodically if it's a long-running operation. For hard timeouts,
    consider using multiprocessing with a timeout.

    Parameters
    ----------
    func : callable
        Function to run.
    timeout : float
        Timeout in seconds.
    *args
        Positional arguments to pass to func.
    default : any, optional
        Default value to return if timeout occurs.
    **kwargs
        Keyword arguments to pass to func.

    Returns
    -------
    result : any
        Result from func, or default if timed out.
    timed_out : bool
        Whether the function timed out.
    duration : float
        Actual duration in seconds.
    """
    start = time.time()

    try:
        result = func(*args, **kwargs)
        duration = time.time() - start
        timed_out = duration > timeout
        return result, timed_out, duration
    except Exception as e:
        duration = time.time() - start
        logger.warning(f"Function {func.__name__} failed after {duration:.1f}s: {e}")
        return default, False, duration
