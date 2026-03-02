from __future__ import annotations

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

        # Fit an interpretable surrogate model to approximate the
        # complex model's predictions, providing global interpretability.
        if time.time() - start < time_budget * 0.85:
            surrogate = self._fit_surrogate(
                best_model, X_sample, feature_names,
            )
            if surrogate is not None:
                explanations["surrogate"] = surrogate

        duration = time.time() - start
        return StageResult(
            stage_name="explainability",
            success=True,
            duration=duration,
            output={"explanations": explanations},
        )

    @staticmethod
    def _fit_surrogate(
        model: Any,
        X_sample: np.ndarray | pd.DataFrame,
        feature_names: list[str],
    ) -> dict[str, Any] | None:
        """Fit an interpretable surrogate to approximate a complex model.

        Trains a shallow decision tree on the complex model's
        predictions.  If available, also fits an EBM for higher
        fidelity.  Returns a dict with the surrogate model, its
        fidelity score (R^2 or accuracy vs the complex model), and
        a human-readable rule summary.

        Returns ``None`` if surrogate fitting fails entirely.
        """
        try:
            from sklearn.metrics import accuracy_score, r2_score
            from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

            X_arr = np.asarray(X_sample)

            # Get complex model predictions as surrogate targets
            if hasattr(model, "predict_proba"):
                y_pred = model.predict(X_arr)
                is_clf = True
            elif hasattr(model, "predict"):
                y_pred = model.predict(X_arr)
                is_clf = False
                # Heuristic: if predictions are integer-like, treat as clf
                if np.array_equal(y_pred, y_pred.astype(int)):
                    is_clf = True
            else:
                return None

            result: dict[str, Any] = {}

            # Fit a shallow decision tree
            if is_clf:
                surrogate = DecisionTreeClassifier(
                    max_depth=5, min_samples_leaf=10, random_state=42,
                )
                surrogate.fit(X_arr, y_pred)
                surrogate_pred = surrogate.predict(X_arr)
                fidelity = float(accuracy_score(y_pred, surrogate_pred))
                score_name = "accuracy"
            else:
                surrogate = DecisionTreeRegressor(
                    max_depth=5, min_samples_leaf=10, random_state=42,
                )
                surrogate.fit(X_arr, y_pred)
                surrogate_pred = surrogate.predict(X_arr)
                fidelity = float(r2_score(y_pred, surrogate_pred))
                score_name = "r2"

            result["tree_surrogate"] = surrogate
            result["tree_fidelity"] = fidelity
            result["tree_fidelity_metric"] = score_name

            # Extract human-readable rules from the tree
            try:
                from sklearn.tree import export_text
                rules = export_text(
                    surrogate,
                    feature_names=feature_names[:X_arr.shape[1]],
                    max_depth=4,
                )
                result["tree_rules"] = rules
            except Exception:
                pass

            # Feature importance from surrogate
            fi = surrogate.feature_importances_
            top_idx = np.argsort(fi)[::-1][:10]
            result["surrogate_top_features"] = [
                (feature_names[i] if i < len(feature_names) else f"feature_{i}", float(fi[i]))
                for i in top_idx if fi[i] > 0
            ]

            # Try EBM surrogate for higher fidelity (if available)
            try:
                from interpret.glassbox import (
                    ExplainableBoostingClassifier,
                    ExplainableBoostingRegressor,
                )

                if is_clf:
                    ebm = ExplainableBoostingClassifier(
                        max_rounds=100, interactions=0,
                    )
                    ebm.fit(X_arr, y_pred)
                    ebm_pred = ebm.predict(X_arr)
                    ebm_fidelity = float(accuracy_score(y_pred, ebm_pred))
                else:
                    ebm = ExplainableBoostingRegressor(
                        max_rounds=100, interactions=0,
                    )
                    ebm.fit(X_arr, y_pred)
                    ebm_pred = ebm.predict(X_arr)
                    ebm_fidelity = float(r2_score(y_pred, ebm_pred))

                result["ebm_surrogate"] = ebm
                result["ebm_fidelity"] = ebm_fidelity
            except ImportError:
                pass
            except Exception as e:
                logger.debug(f"EBM surrogate failed: {e}")

            logger.info(
                f"Surrogate tree fidelity: {fidelity:.3f} ({score_name})"
            )
            return result

        except Exception as e:
            logger.debug(f"Surrogate model fitting failed: {e}")
            return None
