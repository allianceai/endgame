"""Discovery tools: list_modules, describe_api, guide, list_models, recommend_models, describe_model."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def list_modules(module: str | None = None, search: str | None = None) -> str:
        """Discover Endgame's 31 modules. No arguments: every module, what it is for and which tools reach it.
        module="feature_selection": that module's classes and functions with one-line summaries.
        search="entropy": matching classes/functions across all modules.
        Then describe_api(name) for parameters; use them via transform_data, train_model or run_python."""
        try:
            with capture_stdout():
                from endgame.mcp import catalog

                if search:
                    return ok_response({"query": search, "matches": catalog.search(search)})
                if module:
                    return ok_response({"module": module, "members": catalog.members(module)})
                return ok_response({
                    "modules": catalog.overview(),
                    "how_to_use": "Dedicated tools cover common steps; for anything else: describe_api(name), then "
                                  "transform_data (transformers), train_model(model_name='endgame.x.Class') "
                                  "(estimators) or run_python. guide() gives the experiment workflow.",
                })
        except ModuleNotFoundError as e:
            return error_response("not_found", str(e), hint="list_modules() shows the module names")
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def describe_api(name: str) -> str:
        """Signature, parameters, docstring and methods of any Endgame class or function,
        e.g. "endgame.feature_selection.BorutaSelector" or "sample_entropy"."""
        try:
            with capture_stdout():
                from endgame.mcp import catalog

                return ok_response(catalog.describe(name))
        except (KeyError, AttributeError, ModuleNotFoundError) as e:
            return error_response("not_found", str(e), hint="list_modules(search=...) finds names")
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def guide(topic: str = "workflow") -> str:
        """How to run a strong, honest ML experiment with Endgame's tools and modules. Read before starting.
        Topics: workflow, validation, features, selection, models, ensembling, small_data, time_series,
        beyond_tabular, code."""
        from endgame.mcp.guide import TOPICS

        if topic not in TOPICS:
            return error_response("not_found", f"Unknown topic '{topic}'", hint=f"One of: {', '.join(TOPICS)}")
        return ok_response({"topic": topic, "guide": TOPICS[topic]})

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
        top_n: int = 12,
    ) -> str:
        """Models worth comparing on a loaded dataset, each with why: GBDTs, tabular foundation models ranked by
        TabArena (on tables up to 50k rows), neural and interpretable models with more time, a linear baseline.
        Also lists models that need a package or licence. time_budget: fast, medium, high, unlimited.
        Pass the names to compare_models."""
        try:
            ds = session.get_dataset(dataset_id)

            with capture_stdout():
                from endgame.automl.model_registry import (
                    MODEL_REGISTRY,
                    get_interpretable_portfolio,
                    recommend_portfolio,
                )

                raw_task = ds.task_type or "classification"
                task_type = "classification" if raw_task in ("binary", "multiclass") else raw_task
                n_samples = len(ds.df)
                n_features = ds.df.shape[1] - (1 if ds.target_column in ds.df.columns else 0)

                if interpretable_only:
                    names = get_interpretable_portfolio(task_type=task_type, n_samples=n_samples,
                                                        time_budget=time_budget)
                    recs = {"recommended": [{"name": n, "display_name": MODEL_REGISTRY[n].display_name,
                                             "family": MODEL_REGISTRY[n].family,
                                             "fit_time": MODEL_REGISTRY[n].typical_fit_time,
                                             "why": MODEL_REGISTRY[n].notes} for n in names],
                            "unavailable": []}
                else:
                    recs = recommend_portfolio(task_type=task_type, n_samples=n_samples, n_features=n_features,
                                               time_budget=time_budget)

                return ok_response({
                    "dataset": ds.name,
                    "task_type": task_type,
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "time_budget": time_budget,
                    "recommendations": recs["recommended"][:top_n],
                    "unavailable_here": recs["unavailable"],
                    "next": "compare_models(dataset_id, models=[...names]) trains them on the same folds; "
                            "ensemble() combines the best.",
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
