"""Threshold optimization executor for AutoML pipelines.

Optimizes classification decision thresholds on OOF predictions to
improve metrics like F1, balanced accuracy, etc.
"""

import logging
import time
from typing import Any

import numpy as np

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


class ThresholdOptimizationExecutor(BaseStageExecutor):
    """Optimize classification thresholds on OOF predictions.

    Parameters
    ----------
    metric : str, default="f1_weighted"
        Metric to optimize thresholds for.
    """

    def __init__(self, metric: str = "f1_weighted"):
        self.metric = metric

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Find optimal thresholds using OOF predictions.

        Reads ``task_type``, ``oof_predictions``, ``y`` from context.
        Writes ``threshold_optimizer`` to context.
        """
        start = time.time()

        task_type = context.get("task_type", "classification")
        if task_type == "regression":
            return StageResult(
                stage_name="threshold_opt",
                success=True,
                duration=time.time() - start,
                output={},
                metadata={"skipped": "regression"},
            )

        y = context.get("y")
        oof_predictions = context.get("oof_predictions", {})
        trained_models = context.get("trained_models", {})
        ensemble = context.get("ensemble")

        if y is None:
            return StageResult(
                stage_name="threshold_opt",
                success=True,
                duration=time.time() - start,
                output={},
            )

        # Get OOF probabilities from ensemble or best model
        oof_proba = None

        # Try ensemble OOF first
        if ensemble is not None and hasattr(ensemble, "predict_proba"):
            # Ensemble doesn't have OOF by default, try individual models
            pass

        # Fall back to best model OOF
        if oof_proba is None and oof_predictions:
            # Use the first available OOF predictions
            for model_name, oof in oof_predictions.items():
                if oof is not None and hasattr(oof, "shape"):
                    oof_proba = oof
                    break

        if oof_proba is None:
            return StageResult(
                stage_name="threshold_opt",
                success=True,
                duration=time.time() - start,
                output={},
                metadata={"skipped": "no_oof_predictions"},
            )

        try:
            from endgame.ensemble.threshold import ThresholdOptimizer

            n_classes = len(np.unique(y))
            optimizer = ThresholdOptimizer(
                metric=self.metric,
                multiclass=(n_classes > 2),
            )
            optimizer.fit(y, oof_proba)

            logger.info(
                f"Threshold optimization: threshold={optimizer.threshold_}, "
                f"score={optimizer.best_score_:.4f}"
            )

            duration = time.time() - start
            return StageResult(
                stage_name="threshold_opt",
                success=True,
                duration=duration,
                output={"threshold_optimizer": optimizer},
                metadata={
                    "threshold": optimizer.threshold_,
                    "best_score": optimizer.best_score_,
                },
            )

        except ImportError:
            logger.debug("ThresholdOptimizer not available")
            return StageResult(
                stage_name="threshold_opt",
                success=True,
                duration=time.time() - start,
                output={},
            )
        except Exception as e:
            logger.warning(f"Threshold optimization failed: {e}")
            return StageResult(
                stage_name="threshold_opt",
                success=False,
                duration=time.time() - start,
                error=str(e),
            )
