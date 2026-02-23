"""Discovery tools: list_models, recommend_models, describe_model."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def list_models(
        task_type: str | None = None,
        family: str | None = None,
        interpretable_only: bool = False,
        fast_only: bool = False,
        max_samples: int | None = None,
    ) -> str:
        """List available models, optionally filtered by task_type (classification/regression), family (gbdt/neural/tree/linear/kernel/rules/bayesian/foundation/ensemble), interpretable_only, fast_only, or max_samples."""
        try:
            with capture_stdout():
                from endgame.automl.model_registry import (
                    MODEL_REGISTRY,
                )
                from endgame.automl.model_registry import (
                    list_models as _list_models,
                )

                names = _list_models(
                    family=family,
                    task_type=task_type,
                    interpretable_only=interpretable_only,
                    exclude_slow=fast_only,
                    max_samples=max_samples,
                )

                models = []
                for n in names:
                    info = MODEL_REGISTRY[n]
                    models.append({
                        "name": n,
                        "display_name": info.display_name,
                        "family": info.family,
                        "fit_time": info.typical_fit_time,
                        "interpretable": info.interpretable,
                        "notes": info.notes,
                    })

                return ok_response({
                    "count": len(models),
                    "models": models,
                })

        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def recommend_models(
        dataset_id: str,
        time_budget: str = "medium",
        interpretable_only: bool = False,
        top_n: int = 5,
    ) -> str:
        """Recommend models for a loaded dataset based on its meta-features. time_budget: fast, medium, high, unlimited."""
        try:
            ds = session.get_dataset(dataset_id)

            with capture_stdout():
                from endgame.automl.model_registry import (
                    MODEL_REGISTRY,
                    get_default_portfolio,
                    get_interpretable_portfolio,
                )

                raw_task = ds.task_type or "classification"
                # Normalize: "binary"/"multiclass" → "classification" for registry lookup
                task_type = "classification" if raw_task in ("binary", "multiclass") else raw_task
                n_samples = len(ds.df)

                if interpretable_only:
                    portfolio = get_interpretable_portfolio(
                        task_type=task_type,
                        n_samples=n_samples,
                        time_budget=time_budget,
                    )
                else:
                    portfolio = get_default_portfolio(
                        task_type=task_type,
                        n_samples=n_samples,
                        time_budget=time_budget,
                    )

                recommendations = []
                for name in portfolio[:top_n]:
                    info = MODEL_REGISTRY[name]
                    recommendations.append({
                        "name": name,
                        "display_name": info.display_name,
                        "family": info.family,
                        "fit_time": info.typical_fit_time,
                        "interpretable": info.interpretable,
                        "notes": info.notes,
                    })

                return ok_response({
                    "dataset": ds.name,
                    "task_type": task_type,
                    "n_samples": n_samples,
                    "time_budget": time_budget,
                    "recommendations": recommendations,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def describe_model(model_name: str) -> str:
        """Get detailed information about a specific model (parameters, capabilities, speed, notes)."""
        try:
            with capture_stdout():
                from endgame.automl.model_registry import get_model_info

                info = get_model_info(model_name)

                return ok_response({
                    "name": info.name,
                    "display_name": info.display_name,
                    "family": info.family,
                    "class_path": info.class_path,
                    "task_types": info.task_types,
                    "supports_sample_weight": info.supports_sample_weight,
                    "supports_feature_importance": info.supports_feature_importance,
                    "supports_gpu": info.supports_gpu,
                    "requires_torch": info.requires_torch,
                    "requires_julia": info.requires_julia,
                    "typical_fit_time": info.typical_fit_time,
                    "memory_usage": info.memory_usage,
                    "interpretable": info.interpretable,
                    "handles_categorical": info.handles_categorical,
                    "handles_missing": info.handles_missing,
                    "max_samples": info.max_samples,
                    "min_samples": info.min_samples,
                    "default_params": info.default_params,
                    "notes": info.notes,
                })

        except KeyError:
            return error_response(
                "not_found",
                f"Model '{model_name}' not found",
                hint="Use list_models() to see available models",
            )
        except Exception as e:
            return error_response("internal", str(e))
