"""Training tools: train_model, automl, quick_compare."""

from __future__ import annotations

import json
import time as _time

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager
from endgame.mcp.tools._timeout import MCPTimeoutError, timeout_guard


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def train_model(
        dataset_id: str,
        model_name: str,
        params: str | dict | None = None,
        cv_folds: int = 5,
        metric: str = "auto",
    ) -> str:
        """Train a single model on a dataset with cross-validation.

        params: hyperparameter overrides as a dict or JSON string, e.g. {"n_estimators": 500}.
        Returns a model ID with CV metrics.
        """
        try:
            ds = session.get_dataset(dataset_id)
            if ds.target_column is None or ds.target_column not in ds.df.columns:
                return error_response("validation", "Dataset has no target column set")

            with capture_stdout():
                import numpy as np
                import pandas as pd
                from sklearn import metrics as sklearn_metrics
                from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict

                from endgame.automl.model_registry import get_model_info, instantiate_model

                # Parse params
                if isinstance(params, dict):
                    override_params = params
                elif isinstance(params, str):
                    override_params = json.loads(params)
                else:
                    override_params = {}

                task_type = ds.task_type or "classification"
                info = get_model_info(model_name)

                # Prepare data
                X = ds.df.drop(columns=[ds.target_column])
                y = ds.df[ds.target_column]

                # Always label-encode categorical features for MCP consistency
                # (ensures eval/predict/visualize use the same encoders)
                from endgame.mcp.tools._encoding import fit_feature_encoders
                X, feature_encoders = fit_feature_encoders(X)

                # Handle missing values for models that don't support them
                if not info.handles_missing and X.isna().any().any():
                    X = X.fillna(X.median(numeric_only=True))
                    for col in X.select_dtypes(include=["object", "category"]).columns:
                        X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")

                # Encode target for classification if needed
                label_encoder = None
                if task_type != "regression" and y.dtype in ("object", "category"):
                    from sklearn.preprocessing import LabelEncoder
                    label_encoder = LabelEncoder()
                    y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=y.name)

                # Instantiate model
                estimator = instantiate_model(model_name, task_type=task_type, **override_params)

                # Cross-validation
                if task_type == "regression":
                    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                else:
                    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

                start = _time.time()
                with timeout_guard():
                    oof_preds = cross_val_predict(estimator, X, y, cv=cv, method="predict")
                fit_time = _time.time() - start

                # Compute metrics
                computed_metrics = {}
                if task_type == "regression":
                    computed_metrics["rmse"] = float(np.sqrt(sklearn_metrics.mean_squared_error(y, oof_preds)))
                    computed_metrics["r2"] = float(sklearn_metrics.r2_score(y, oof_preds))
                    computed_metrics["mae"] = float(sklearn_metrics.mean_absolute_error(y, oof_preds))
                else:
                    computed_metrics["accuracy"] = float(sklearn_metrics.accuracy_score(y, oof_preds))
                    computed_metrics["f1"] = float(sklearn_metrics.f1_score(y, oof_preds, average="weighted"))
                    try:
                        if len(np.unique(y)) == 2:
                            oof_proba = cross_val_predict(estimator, X, y, cv=cv, method="predict_proba")
                            computed_metrics["roc_auc"] = float(
                                sklearn_metrics.roc_auc_score(y, oof_proba[:, 1])
                            )
                    except Exception:
                        pass

                # Final fit on full data
                with timeout_guard():
                    estimator.fit(X, y)

                # Get model params
                model_params = estimator.get_params() if hasattr(estimator, "get_params") else override_params

                art = session.add_model(
                    estimator=estimator,
                    name=f"{model_name}_1",
                    model_type=model_name,
                    dataset_id=dataset_id,
                    task_type=task_type,
                    metrics=computed_metrics,
                    params=model_params,
                    fit_time=fit_time,
                    feature_names=list(X.columns),
                    oof_predictions=oof_preds,
                    label_encoders=feature_encoders if feature_encoders else None,
                    target_encoder=label_encoder,
                )

                return ok_response({
                    "model_id": art.id,
                    "model_name": model_name,
                    "display_name": info.display_name,
                    "task_type": task_type,
                    "cv_folds": cv_folds,
                    "metrics": {k: round(v, 4) for k, v in computed_metrics.items()},
                    "fit_time": round(fit_time, 2),
                    "n_features": X.shape[1],
                })

        except MCPTimeoutError as e:
            return error_response("timeout", str(e), hint="Try a simpler model or smaller dataset.")
        except KeyError as e:
            return error_response("not_found", str(e), hint="Use list_models() to see available models")
        except ImportError as e:
            return error_response("missing_dependency", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def automl(
        dataset_id: str,
        preset: str = "medium_quality",
        time_limit: int | None = None,
        interpretable_only: bool = False,
    ) -> str:
        """Run a full AutoML pipeline (preprocessing, model selection, training, ensembling).

        Presets: best_quality, high_quality, good_quality, medium_quality, fast, interpretable.
        """
        try:
            ds = session.get_dataset(dataset_id)
            if ds.target_column is None:
                return error_response("validation", "Dataset has no target column set")

            with capture_stdout():
                from endgame.automl.tabular import TabularPredictor

                predictor = TabularPredictor(
                    label=ds.target_column,
                    presets=preset,
                    time_limit=time_limit,
                    verbosity=0,
                )

                start = _time.time()
                predictor.fit(ds.df, interpretable_only=interpretable_only)
                total_time = _time.time() - start

                # Store predictor
                pred_id = f"automl_{dataset_id}"
                session.automl_predictors[pred_id] = predictor

                # Extract best model as ModelArtifact
                summary = predictor.fit_summary_
                best_model_name = summary.best_model if summary else "unknown"
                best_estimator = predictor.get_model(best_model_name) if best_model_name != "unknown" else None

                metrics = {}
                if summary:
                    metrics["best_score"] = summary.best_score

                model_art = session.add_model(
                    estimator=best_estimator or predictor,
                    name=f"automl_{best_model_name}",
                    model_type=f"automl_{preset}",
                    dataset_id=dataset_id,
                    task_type=ds.task_type or "classification",
                    metrics=metrics,
                    fit_time=total_time,
                    feature_names=predictor.feature_names_ or [],
                )

                # Build leaderboard
                leaderboard = []
                if predictor.leaderboard_ is not None and len(predictor.leaderboard_) > 0:
                    for _, row in predictor.leaderboard_.iterrows():
                        leaderboard.append({
                            "model": row.get("model", ""),
                            "score": round(float(row.get("score", 0)), 4),
                            "fit_time": round(float(row.get("fit_time", 0)), 2),
                        })

                return ok_response({
                    "model_id": model_art.id,
                    "predictor_id": pred_id,
                    "preset": preset,
                    "best_model": best_model_name,
                    "best_score": round(summary.best_score, 4) if summary else None,
                    "n_models_trained": summary.n_models_trained if summary else 0,
                    "total_time": round(total_time, 2),
                    "leaderboard": leaderboard,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def quick_compare(
        dataset_id: str,
        preset: str = "default",
        metric: str = "auto",
    ) -> str:
        """Quickly compare multiple models on a dataset and return a leaderboard.

        Presets: fast, default, competition, interpretable.
        """
        try:
            ds = session.get_dataset(dataset_id)
            if ds.target_column is None:
                return error_response("validation", "Dataset has no target column set")

            with capture_stdout():
                import pandas as pd

                from endgame.quick import compare

                X = ds.df.drop(columns=[ds.target_column])
                y = ds.df[ds.target_column]

                task = "regression" if ds.task_type == "regression" else "classification"

                result = compare(X, y, task=task, preset=preset)

                leaderboard = []
                if hasattr(result, "leaderboard") and result.leaderboard is not None:
                    for _, row in result.leaderboard.iterrows():
                        entry = row.to_dict()
                        leaderboard.append({
                            k: round(v, 4) if isinstance(v, float) else v
                            for k, v in entry.items()
                        })

                return ok_response({
                    "dataset": ds.name,
                    "task_type": task,
                    "preset": preset,
                    "leaderboard": leaderboard,
                    "n_models": len(leaderboard),
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except ImportError as e:
            return error_response("missing_dependency", str(e))
        except Exception as e:
            return error_response("internal", str(e))
