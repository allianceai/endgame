"""Explainability executor for AutoML pipelines.

Computes SHAP feature importances and optionally feature interactions
for the best model after training.
"""

import logging
import time
from typing import Any

import numpy as np
import pandas as pd

from endgame.automl.orchestrator import BaseStageExecutor, StageResult

logger = logging.getLogger(__name__)


class ExplainabilityExecutor(BaseStageExecutor):
    """Generate model explanations using SHAP.

    Parameters
    ----------
    max_samples : int, default=1000
        Maximum number of training samples to use for explanations.
    top_features : int, default=10
        Number of top features to report.
    compute_interactions : bool, default=False
        Whether to compute feature interactions (requires more time).
    """

    def __init__(
        self,
        max_samples: int = 1000,
        top_features: int = 10,
        compute_interactions: bool = False,
    ):
        self.max_samples = max_samples
        self.top_features = top_features
        self.compute_interactions = compute_interactions

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Compute explanations for the best model.

        Reads ``trained_models``, ``results``, ``X``, ``feature_names``
        from context. Writes ``explanations`` dict to context.
        """
        start = time.time()

        trained_models = context.get("trained_models", {})
        results = context.get("results", [])
        X = context.get(
            "X_augmented",
            context.get(
                "X_engineered",
                context.get("X_processed", context.get("X")),
            ),
        )

        if not trained_models or X is None:
            return StageResult(
                stage_name="explainability",
                success=True,
                duration=time.time() - start,
                output={},
            )

        # Find best model
        successful = [r for r in results if r.success]
        if not successful:
            return StageResult(
                stage_name="explainability",
                success=True,
                duration=time.time() - start,
                output={},
            )

        successful.sort(key=lambda r: r.score, reverse=True)
        best_name = successful[0].config.model_name
        best_model = trained_models.get(best_name)

        if best_model is None:
            return StageResult(
                stage_name="explainability",
                success=True,
                duration=time.time() - start,
                output={},
            )

        # Subsample data
        if isinstance(X, pd.DataFrame):
            n = min(len(X), self.max_samples)
            X_sample = X.iloc[:n]
            feature_names = X.columns.tolist()
        else:
            X_arr = np.asarray(X)
            n = min(X_arr.shape[0], self.max_samples)
            X_sample = X_arr[:n]
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        explanations: dict[str, Any] = {"model": best_name}

        # Compute SHAP values
        try:
            from endgame.explain import SHAPExplainer

            explainer = SHAPExplainer(model=best_model, verbose=False)
            explanation = explainer.explain(X_sample)

            # Extract feature importance from SHAP values
            shap_values = explanation.values
            if shap_values.ndim == 3:
                # Multiclass: mean absolute across classes
                importance = np.mean(np.abs(shap_values), axis=(0, 2))
            else:
                importance = np.mean(np.abs(shap_values), axis=0)

            # Build feature importance dataframe
            importance_df = pd.DataFrame({
                "feature": feature_names[:len(importance)],
                "importance": importance,
            }).sort_values("importance", ascending=False).reset_index(drop=True)

            explanations["feature_importance_df"] = importance_df
            explanations["top_features"] = importance_df["feature"].head(
                self.top_features
            ).tolist()
            explanations["shap_explanation"] = explanation

            logger.info(
                f"SHAP explanations computed for {best_name}: "
                f"top feature = {explanations['top_features'][0]}"
            )

        except ImportError:
            logger.debug("SHAP explainer not available")
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")

        # Compute feature interactions if time permits
        if (
            self.compute_interactions
            and "top_features" in explanations
            and time.time() - start < time_budget * 0.7
        ):
            try:
                from endgame.explain.interaction import FeatureInteraction

                interaction = FeatureInteraction(model=best_model)
                interaction_result = interaction.explain(
                    X_sample, top_k=5
                )
                explanations["feature_interactions"] = interaction_result

            except ImportError:
                logger.debug("FeatureInteraction not available")
            except Exception as e:
                logger.warning(f"Feature interaction computation failed: {e}")

        duration = time.time() - start
        return StageResult(
            stage_name="explainability",
            success=True,
            duration=duration,
            output={"explanations": explanations},
        )
