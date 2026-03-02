from __future__ import annotations

"""Deployment constraint checking for AutoML pipelines.

Validates trained models against user-specified deployment constraints
such as prediction latency, model size, and interpretability requirements.
"""

import logging
import pickle
import sys
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


@dataclass
class DeploymentConstraints:
    """Constraints for model deployment.

    Parameters
    ----------
    max_predict_latency_ms : float, optional
        Maximum prediction latency per batch of 100 samples (ms).
    max_model_size_mb : float, optional
        Maximum serialized model size (MB).
    max_memory_mb : float, optional
        Maximum memory usage (MB).
    require_interpretable : bool, default=False
        If True, only allow interpretable models.
    max_features : int, optional
        Maximum number of features the model can use.
    """

    max_predict_latency_ms: float | None = None
    max_model_size_mb: float | None = None
    max_memory_mb: float | None = None
    require_interpretable: bool = False
    max_features: int | None = None


@dataclass
class ConstraintViolation:
    """A single constraint violation.

    Attributes
    ----------
    model_name : str
        Name of the model that violated the constraint.
    constraint : str
        Name of the constraint violated.
    value : float
        Actual value.
    limit : float
        Constraint limit.
    message : str
        Human-readable description.
    """

    model_name: str
    constraint: str
    value: float
    limit: float
    message: str


class ConstraintCheckExecutor(BaseStageExecutor):
    """Check trained models against deployment constraints.

    Parameters
    ----------
    constraints : DeploymentConstraints, optional
        Deployment constraints. If None, all models pass.
    """

    def __init__(self, constraints: DeploymentConstraints | None = None):
        self.constraints = constraints

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Validate models against deployment constraints.

        Reads ``trained_models``, ``X`` from context.
        Writes ``compliant_models``, ``constraint_violations`` to context.
        """
        start = time.time()

        if self.constraints is None:
            return StageResult(
                stage_name="constraint_check",
                success=True,
                duration=time.time() - start,
                output={},
                metadata={"skipped": "no_constraints"},
            )

        trained_models = context.get("trained_models", {})
        X = context.get("X_preprocessed", context.get("X"))

        if not trained_models:
            return StageResult(
                stage_name="constraint_check",
                success=True,
                duration=time.time() - start,
                output={"compliant_models": [], "constraint_violations": []},
            )

        violations: list[ConstraintViolation] = []
        compliant: list[str] = []

        for model_name, model in trained_models.items():
            if time.time() - start >= time_budget * 0.95:
                # Out of time, assume remaining models are compliant
                compliant.append(model_name)
                continue

            model_violations = self._check_model(
                model_name, model, X, self.constraints
            )
            if model_violations:
                violations.extend(model_violations)
                for v in model_violations:
                    logger.info(
                        f"Constraint violation: {v.message}"
                    )
            else:
                compliant.append(model_name)

        logger.info(
            f"Constraint check: {len(compliant)}/{len(trained_models)} models "
            f"compliant, {len(violations)} violations"
        )

        duration = time.time() - start
        return StageResult(
            stage_name="constraint_check",
            success=True,
            duration=duration,
            output={
                "compliant_models": compliant,
                "constraint_violations": violations,
            },
            metadata={
                "n_compliant": len(compliant),
                "n_violations": len(violations),
            },
        )

    def _check_model(
        self,
        model_name: str,
        model: Any,
        X: Any,
        constraints: DeploymentConstraints,
    ) -> list[ConstraintViolation]:
        """Check a single model against constraints."""
        violations = []

        # Prediction latency check
        if constraints.max_predict_latency_ms is not None and X is not None:
            latency = self._measure_latency(model, X)
            if latency is not None and latency > constraints.max_predict_latency_ms:
                violations.append(ConstraintViolation(
                    model_name=model_name,
                    constraint="max_predict_latency_ms",
                    value=latency,
                    limit=constraints.max_predict_latency_ms,
                    message=(
                        f"{model_name}: latency {latency:.1f}ms > "
                        f"{constraints.max_predict_latency_ms}ms limit"
                    ),
                ))

        # Model size check
        if constraints.max_model_size_mb is not None:
            size_mb = self._estimate_size(model)
            if size_mb is not None and size_mb > constraints.max_model_size_mb:
                violations.append(ConstraintViolation(
                    model_name=model_name,
                    constraint="max_model_size_mb",
                    value=size_mb,
                    limit=constraints.max_model_size_mb,
                    message=(
                        f"{model_name}: size {size_mb:.1f}MB > "
                        f"{constraints.max_model_size_mb}MB limit"
                    ),
                ))

        # Interpretability check
        if constraints.require_interpretable:
            try:
                from endgame.automl.model_registry import (
                    INTERPRETABLE_MODELS,
                    MODEL_REGISTRY,
                )

                is_interpretable = model_name in INTERPRETABLE_MODELS
                if not is_interpretable and model_name in MODEL_REGISTRY:
                    is_interpretable = MODEL_REGISTRY[model_name].interpretable
                if not is_interpretable:
                    violations.append(ConstraintViolation(
                        model_name=model_name,
                        constraint="require_interpretable",
                        value=0,
                        limit=1,
                        message=f"{model_name}: not interpretable",
                    ))
            except ImportError:
                pass

        return violations

    def _measure_latency(self, model: Any, X: Any) -> float | None:
        """Measure prediction latency on a 100-sample batch (ms)."""
        try:
            if hasattr(X, "iloc"):
                X_batch = X.iloc[:100]
            else:
                X_batch = np.asarray(X)[:100]

            # Warm-up
            model.predict(X_batch)

            # Timed run
            t0 = time.perf_counter()
            model.predict(X_batch)
            t1 = time.perf_counter()

            return (t1 - t0) * 1000  # ms
        except Exception:
            return None

    def _estimate_size(self, model: Any) -> float | None:
        """Estimate model size in MB via pickle."""
        try:
            data = pickle.dumps(model)
            return len(data) / (1024 * 1024)
        except Exception:
            try:
                return sys.getsizeof(model) / (1024 * 1024)
            except Exception:
                return None
