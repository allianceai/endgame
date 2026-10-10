"""Training tools: train_model, compare_models, automl, quick_compare."""

from __future__ import annotations

import json
import time as _time
from types import SimpleNamespace

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager
from endgame.mcp.tools._timeout import MCPTimeoutError, timeout_guard


def _model_info(model_name):
    """Registry info for a model key, or a stand-in for an estimator given by class path."""
    from endgame.automl.model_registry import MODEL_REGISTRY

    if model_name in MODEL_REGISTRY or "." not in model_name:
        from endgame.automl.model_registry import get_model_info
        return get_model_info(model_name)
    return SimpleNamespace(family="custom", handles_missing=False, display_name=model_name.rsplit(".", 1)[-1])


def _make_estimator(model_name, task_type, params):
    """A fresh estimator from a registry key ("lgbm") or a class path ("endgame.models.trees.RotationForestClassifier")."""
    from endgame.automl.model_registry import MODEL_REGISTRY, instantiate_model

    if model_name in MODEL_REGISTRY:
        return instantiate_model(model_name, task_type=task_type, **params)
    from endgame.mcp.catalog import resolve
    return resolve(model_name)(**params)


def _splitter(task_type, cv_folds, time_ordered, groups):
    from sklearn.model_selection import (
        GroupKFold,
        KFold,
        StratifiedGroupKFold,
        StratifiedKFold,
        TimeSeriesSplit,
    )

    if time_ordered:
        return TimeSeriesSplit(n_splits=cv_folds)
    if groups is not None:
        if task_type == "regression":
            return GroupKFold(n_splits=cv_folds)
        return StratifiedGroupKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    if task_type == "regression":
        return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    return StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)


def _cross_validate(model_name, task_type, override_params, X, y, cv_folds, time_ordered, groups=None):
    """Out-of-fold predictions of a fresh ``model_name`` (plus out-of-fold class probabilities), its CV time and
    parameters. Module level so a foundation model can run it in a child process (``_isolated``)."""
    import numpy as np
    from sklearn.base import clone
    from sklearn.model_selection import cross_val_predict

    estimator = _make_estimator(model_name, task_type, override_params)
    cv = _splitter(task_type, cv_folds, time_ordered, groups)

    binary = task_type != "regression" and len(np.unique(y)) == 2
    start = _time.time()
    oof_proba = None
    if time_ordered:
        # Oct 1: shuffled CV said 65 min on Ezra's nights; later nights said ~95.
        scored, preds, probas = [], [], []
        for train_idx, test_idx in cv.split(X):
            fold = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
            preds.append(fold.predict(X.iloc[test_idx]))
            if binary and hasattr(fold, "predict_proba"):
                probas.append(fold.predict_proba(X.iloc[test_idx]))
            scored.append(test_idx)
        scored = np.concatenate(scored)
        # Concatenated, the predictions keep their dtype; an object array made sklearn's classification metrics fail
        # with "a mix of binary and unknown targets" (Oct 9, Ezra's nights as yes/no questions).
        oof_preds = np.concatenate(preds)
        oof_proba = np.concatenate(probas) if probas else None
        y_scored = y.iloc[scored]
        oof_rows = np.full(len(y), np.nan, dtype=float if task_type == "regression" else object)
        oof_rows[scored] = oof_preds  # earliest rows were never scored
    else:
        oof_preds = cross_val_predict(estimator, X, y, cv=cv, groups=groups, method="predict")
        y_scored = y
        oof_rows = oof_preds
    fit_time = _time.time() - start

    if task_type != "regression" and not time_ordered and hasattr(estimator, "predict_proba"):
        try:
            oof_proba = cross_val_predict(estimator, X, y, cv=cv, groups=groups, method="predict_proba")
        except Exception:
            pass
    params = estimator.get_params() if hasattr(estimator, "get_params") else override_params
    return oof_preds, y_scored, oof_rows, oof_proba, fit_time, params


def _train_one(session, dataset_id, model_name, override_params, cv_folds, time_ordered, group_column=None):
    """Cross-validate and fit one model on a session dataset; store it. Returns (artifact, response dict)."""
    import numpy as np
    import pandas as pd
    from sklearn import metrics as sk

    from endgame.mcp.tools._encoding import fit_feature_encoders, identifier_columns
    from endgame.mcp.tools._isolated import (
        IsolatedClassifier,
        IsolatedRegressor,
        release,
        run_isolated,
        should_isolate,
    )

    ds = session.get_dataset(dataset_id)
    if ds.target_column is None or ds.target_column not in ds.df.columns:
        raise ValueError("Dataset has no target column set")
    task_type = ds.task_type or "classification"
    info = _model_info(model_name)

    X = ds.df.drop(columns=[ds.target_column])
    groups = None
    if group_column:
        groups = X.pop(group_column).to_numpy()
    dropped = identifier_columns(X)
    X = X.drop(columns=dropped)
    y = ds.df[ds.target_column]

    # Always label-encode categorical features for MCP consistency (eval/predict/visualize use the same encoders)
    X, feature_encoders = fit_feature_encoders(X)

    # Handle missing values for models that don't support them
    if not info.handles_missing and X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "missing")

    label_encoder = None
    if task_type != "regression" and y.dtype in ("object", "category"):
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        y = pd.Series(label_encoder.fit_transform(y.astype(str)), name=y.name)

    # A GPU foundation model cross-validates in a child process, so its GPU memory is freed after
    isolate = should_isolate(info)
    release()   # a model worker left by an earlier call would hold GPU memory
    args = (model_name, task_type, override_params, X, y, cv_folds, time_ordered, groups)
    with timeout_guard():
        cv_out = run_isolated(_cross_validate, *args) if isolate else _cross_validate(*args)
    oof_preds, y_scored, oof_rows, oof_proba, fit_time, model_params = cv_out

    metrics = {}
    if task_type == "regression":
        metrics["rmse"] = float(np.sqrt(sk.mean_squared_error(y_scored, oof_preds)))
        metrics["r2"] = float(sk.r2_score(y_scored, oof_preds))
        metrics["mae"] = float(sk.mean_absolute_error(y_scored, oof_preds))
    else:
        metrics["accuracy"] = float(sk.accuracy_score(y_scored, oof_preds))
        metrics["f1"] = float(sk.f1_score(y_scored, oof_preds, average="weighted"))
        if oof_proba is not None:
            if oof_proba.shape[1] == 2:
                metrics["roc_auc"] = float(sk.roc_auc_score(y_scored, oof_proba[:, 1]))
            else:
                metrics["roc_auc_ovr"] = float(sk.roc_auc_score(y_scored, oof_proba, multi_class="ovr"))
            metrics["log_loss"] = float(sk.log_loss(y_scored, oof_proba))

    # Final fit on full data; an isolated model fits in its worker on first use
    if isolate:
        estimator = (IsolatedRegressor if task_type == "regression" else IsolatedClassifier)(
            model_name, task_type, override_params, X, y)
    else:
        estimator = _make_estimator(model_name, task_type, override_params)
        with timeout_guard():
            estimator.fit(X, y)

    cv = {"folds": cv_folds, "time_ordered": time_ordered, "group_column": group_column}
    art = session.add_model(
        estimator=estimator, name=f"{model_name.rsplit('.', 1)[-1]}_1", model_type=model_name,
        dataset_id=dataset_id, task_type=task_type, metrics=metrics, params=model_params, fit_time=fit_time,
        feature_names=list(X.columns), oof_predictions=oof_rows,
        label_encoders=feature_encoders if feature_encoders else None, target_encoder=label_encoder,
        oof_proba=oof_proba, cv=cv,
    )
    return art, {
        "model_id": art.id,
        "model_name": model_name,
        "display_name": info.display_name,
        "task_type": task_type,
        "cv_folds": cv_folds,
        "cv": "time_ordered" if time_ordered else (f"grouped by {group_column}" if group_column else "shuffled"),
        "metrics": {k: round(v, 4) for k, v in metrics.items()},
        "fit_time": round(fit_time, 2),
        "n_features": X.shape[1],
        "dropped_columns": dropped,
    }


def _parse_params(params):
    if isinstance(params, dict):
        return params
    return json.loads(params) if params else {}


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def train_model(
        dataset_id: str,
        model_name: str,
        params: str | dict | None = None,
        cv_folds: int = 5,
        metric: str = "auto",
        time_ordered: bool = False,
        group_column: str | None = None,
    ) -> str:
        """Train one model with cross-validation; stores its out-of-fold predictions (for ensemble) and a final fit.

        model_name: a registry key (list_models / recommend_models: "lgbm", "catboost", "kumo_tabular",
        "tabpfn_35", "ebm", "realmlp", ...) or the class path of any estimator in any module, e.g.
        "endgame.models.trees.RotationForestClassifier" or "endgame.fuzzy.ANFISRegressor".
        params: hyperparameter overrides as a dict or JSON string, e.g. {"n_estimators": 500}.
        time_ordered: rows are in time order; each fold trains on earlier rows and is scored on later ones.
        group_column: rows sharing this column's value (a player, patient, site) stay in the same fold; the column
        is not used as a feature. Use it whenever an entity appears in more than one row.
        compare_models trains several on the same folds.
        """
        try:
            with capture_stdout():
                _, response = _train_one(session, dataset_id, model_name, _parse_params(params), cv_folds,
                                         time_ordered, group_column)
                return ok_response(response)
        except MCPTimeoutError as e:
            return error_response("timeout", str(e), hint="Try a simpler model or smaller dataset.")
        except ValueError as e:
            return error_response("validation", str(e))
        except KeyError as e:
            return error_response("not_found", str(e), hint="Use list_models() to see available models")
        except ImportError as e:
            return error_response("missing_dependency", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def compare_models(
        dataset_id: str,
        models: list[str] | None = None,
        time_budget: str = "medium",
        cv_folds: int = 5,
        time_ordered: bool = False,
        group_column: str | None = None,
        params: str | dict | None = None,
        metric: str = "auto",
        time_limit: int = 1500,
    ) -> str:
        """Train several models on the same folds and rank them by metric (auto: r2, or roc_auc; also rmse, mae,
        accuracy, f1, log_loss, roc_auc_ovr; lower-is-better metrics rank ascending); each is stored with its out-of-fold predictions,
        so ensemble(model_ids) can combine them. models: registry keys or class paths; default: what
        recommend_models picks for this table and time_budget (GBDTs, tabular foundation models such as
        Kumo-Tabular / LimiX / TabPFN, more families with a bigger budget, and a linear baseline).
        params: {"model_name": {overrides}}. A model that fails is reported, not fatal. time_limit (seconds, default
        1500): no new model starts after it, so one call stays inside an MCP client's call timeout; models not
        started are listed in not_run to pass to another call."""
        try:
            ds = session.get_dataset(dataset_id)
            per_model = _parse_params(params)
            with capture_stdout():
                if not models:
                    from endgame.automl.model_registry import recommend_portfolio
                    task = "regression" if ds.task_type == "regression" else "classification"
                    n_features = ds.df.shape[1] - 1
                    models = [r["name"] for r in recommend_portfolio(task, len(ds.df), n_features,
                                                                     time_budget)["recommended"]]
                results, failures, not_run = [], [], []
                started = _time.time()
                for name in models:
                    if (results or failures) and _time.time() - started > time_limit:  # always try one
                        not_run.append(name)
                        continue
                    try:
                        _, response = _train_one(session, dataset_id, name, per_model.get(name, {}), cv_folds,
                                                 time_ordered, group_column)
                        results.append(response)
                    except Exception as e:  # one model's failure must not lose the others
                        failures.append({"model_name": name, "error": f"{type(e).__name__}: {str(e)[:300]}"})

            regression = ds.task_type == "regression"
            available = results[0]["metrics"] if results else {}
            if metric != "auto" and results and metric not in available:
                return error_response("validation", f"metric '{metric}' is not computed for this task",
                                      hint=f"One of: {', '.join(available)}")
            key = metric if metric != "auto" else ("r2" if regression else next(
                (k for k in ("roc_auc", "roc_auc_ovr", "accuracy") if k in available), "accuracy"))
            lower_better = key in ("rmse", "mae", "log_loss")
            results.sort(key=lambda r: r["metrics"].get(key, float("inf") if lower_better else float("-inf")),
                         reverse=not lower_better)
            leaderboard = [{"rank": i + 1, "model_id": r["model_id"], "model_name": r["model_name"],
                            "display_name": r["display_name"], **r["metrics"], "fit_time": r["fit_time"]}
                           for i, r in enumerate(results)]
            return ok_response({
                "dataset": ds.name, "ranked_by": key, "cv": results[0]["cv"] if results else None,
                "cv_folds": cv_folds, "leaderboard": leaderboard, "failed": failures, "not_run": not_run,
                "next": "ensemble(model_ids=[top model_ids]) combines their out-of-fold predictions"
                        if len(results) > 1 else "",
            })
        except KeyError as e:
            return error_response("not_found", str(e))
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

            from endgame.mcp.tools._isolated import release
            release()   # free the GPU memory of an idle model worker before training in-process

            with capture_stdout():
                import pandas as pd

                from endgame.automl.tabular import TabularPredictor
                from endgame.mcp.tools._encoding import identifier_columns
                dropped = identifier_columns(ds.df.drop(columns=[ds.target_column]))
                predictor = TabularPredictor(
                    label=ds.target_column,
                    presets=preset,
                    time_limit=time_limit,
                    verbosity=0,
                    problem_type=ds.task_type if ds.task_type in ("regression", "binary", "multiclass") else "auto",
                )

                start = _time.time()
                predictor.fit(ds.df.drop(columns=dropped), interpretable_only=interpretable_only)
                total_time = _time.time() - start

                # Store predictor
                pred_id = f"automl_{dataset_id}"
                session.automl_predictors[pred_id] = predictor

                # Extract best model as ModelArtifact
                summary = predictor.fit_summary_
                best_model_name = summary.best_model if summary else "unknown"
                if best_model_name == "fallback_hgb":
                    # Oct 1: this came back as status ok with score 0.0.
                    return error_response(
                        "training_failed",
                        "Every candidate model failed or ran out of time; AutoML returned its untuned "
                        "fallback (HistGradientBoosting) with no score.",
                        hint="Raise time_limit, try preset='fast', or train_model one model to see its error.")
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
                        score = row.get("score")
                        leaderboard.append({
                            "model": row.get("model", ""),
                            "score": None if pd.isna(score) else round(float(score), 4),  # None = never scored
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
                    "dropped_columns": dropped,
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
        """Quick leaderboard from endgame.quick presets (fast, default, competition, interpretable); models are
        not kept. To compare chosen models (incl. foundation models) on the same folds and ensemble them, use
        compare_models."""
        try:
            ds = session.get_dataset(dataset_id)
            if ds.target_column is None:
                return error_response("validation", "Dataset has no target column set")

            from endgame.mcp.tools._isolated import release
            release()   # free the GPU memory of an idle model worker before training in-process

            with capture_stdout():
                import pandas as pd

                from endgame.quick import compare

                X = ds.df.drop(columns=[ds.target_column])
                y = ds.df[ds.target_column]
                # compare() fits the frame as given, so a text column (a date,
                # a weekday) failed every model. Encode like train_model does.
                from endgame.mcp.tools._encoding import fit_feature_encoders, identifier_columns
                X, _ = fit_feature_encoders(X.drop(columns=identifier_columns(X)))

                task = "regression" if ds.task_type == "regression" else "classification"

                result = compare(X, y, task=task, preset=preset,
                                 metric=None if metric == "auto" else metric)

                # compare() returns the leaderboard as a list of dicts.
                leaderboard = [
                    {k: round(v, 4) if isinstance(v, float) else v for k, v in entry.items()}
                    for entry in (result.leaderboard or [])
                ]

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
