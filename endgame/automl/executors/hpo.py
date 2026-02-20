"""Hyperparameter tuning executor for AutoML pipelines.

Selects the top-N models from initial training and tunes them with
Optuna, replacing the original models if tuning improves scores.
"""

import logging
import time
from typing import Any

import numpy as np

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


class HyperparameterTuningExecutor(BaseStageExecutor):
    """Tunes hyperparameters of top-performing models via Optuna.

    Parameters
    ----------
    top_n : int, default=3
        Number of top models to tune.
    cv_folds : int, default=5
        Cross-validation folds for tuning.
    """

    def __init__(self, top_n: int = 3, cv_folds: int = 5):
        self.top_n = top_n
        self.cv_folds = cv_folds

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Tune top-N models using Optuna.

        Reads ``results``, ``trained_models``, ``oof_predictions`` from
        context. Writes updated versions back plus ``tuning_results``.
        """
        start = time.time()

        results = context.get("results", [])
        trained_models = context.get("trained_models", {})
        oof_predictions = context.get("oof_predictions", {})
        task_type = context.get("task_type", "classification")

        if not results or not trained_models:
            return StageResult(
                stage_name="hyperparameter_tuning",
                success=True,
                duration=time.time() - start,
                output={"tuning_results": []},
            )

        # Pick best data available (must match what models were trained on)
        X = context.get(
            "X_augmented",
            context.get(
                "X_engineered",
                context.get("X_processed", context.get("X")),
            ),
        )
        y = context.get("y")
        if X is None or y is None:
            return StageResult(
                stage_name="hyperparameter_tuning",
                success=True,
                duration=time.time() - start,
                output={"tuning_results": []},
            )

        # Select top-N models by score
        successful = [r for r in results if r.success]
        successful.sort(key=lambda r: r.score, reverse=True)
        top_models = successful[: self.top_n]

        if not top_models:
            return StageResult(
                stage_name="hyperparameter_tuning",
                success=True,
                duration=time.time() - start,
                output={"tuning_results": []},
            )

        time_per_model = max(10.0, (time_budget - (time.time() - start)) / len(top_models))
        tuning_results = []

        for result in top_models:
            model_name = result.config.model_name
            original_score = result.score

            if time.time() - start >= time_budget * 0.95:
                break

            try:
                tuned_score, tuned_model = self._tune_model(
                    model_name=model_name,
                    trained_models=trained_models,
                    X=X,
                    y=y,
                    task_type=task_type,
                    timeout=time_per_model,
                )

                improved = tuned_score is not None and tuned_score > original_score
                if improved:
                    trained_models[model_name] = tuned_model
                    logger.info(
                        f"HPO improved {model_name}: "
                        f"{original_score:.4f} -> {tuned_score:.4f}"
                    )

                tuning_results.append({
                    "model": model_name,
                    "original_score": original_score,
                    "tuned_score": tuned_score,
                    "improved": improved,
                })

            except Exception as e:
                logger.warning(f"HPO failed for {model_name}: {e}")
                tuning_results.append({
                    "model": model_name,
                    "original_score": original_score,
                    "tuned_score": None,
                    "improved": False,
                    "error": str(e),
                })

        duration = time.time() - start
        return StageResult(
            stage_name="hyperparameter_tuning",
            success=True,
            duration=duration,
            output={
                "trained_models": trained_models,
                "oof_predictions": oof_predictions,
                "tuning_results": tuning_results,
            },
        )

    def _tune_model(
        self,
        model_name: str,
        trained_models: dict,
        X: Any,
        y: Any,
        task_type: str,
        timeout: float,
    ) -> tuple[float | None, Any]:
        """Tune a single model. Returns (score, model) or (None, None)."""
        try:
            from endgame.automl.model_registry import get_model_info
            from endgame.tune.optuna import OptunaOptimizer
            from endgame.tune.spaces import get_space
        except ImportError:
            logger.debug("Optuna or tune module not available, skipping HPO")
            return None, None

        info = get_model_info(model_name)
        if not info.tuning_space:
            logger.debug(f"No tuning space for {model_name}, skipping")
            return None, None

        space = get_space(info.tuning_space)

        # Get the existing estimator to clone.  If the model is wrapped
        # in a Pipeline (preprocessing + model), extract the inner model
        # for tuning so that HP names don't need a ``model__`` prefix.
        from sklearn.base import clone
        from sklearn.pipeline import Pipeline

        existing = trained_models.get(model_name)
        if existing is None:
            return None, None

        pipeline_prefix = None
        if isinstance(existing, Pipeline):
            inner = existing.named_steps.get("model", existing.steps[-1][1])
            pipeline_prefix = existing
            try:
                estimator = clone(inner)
            except Exception:
                estimator = inner.__class__(**inner.get_params())
        else:
            try:
                estimator = clone(existing)
            except Exception:
                estimator = existing.__class__(**existing.get_params())

        metric = "roc_auc" if task_type == "classification" else "neg_root_mean_squared_error"

        optimizer = OptunaOptimizer(
            estimator=estimator,
            param_space=space,
            metric=metric,
            cv=self.cv_folds,
            n_trials=50,
            timeout=int(timeout),
            verbose=False,
        )

        result = optimizer.optimize(X, y)
        if optimizer.best_estimator_ is not None:
            best = optimizer.best_estimator_
            if pipeline_prefix is not None:
                rebuilt = clone(pipeline_prefix)
                rebuilt.steps[-1] = (rebuilt.steps[-1][0], best)
                best = rebuilt
            return optimizer.best_score_, best

        return None, None
