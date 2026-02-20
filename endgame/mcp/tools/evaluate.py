"""Evaluation tools: evaluate_model, explain_model."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def evaluate_model(
        model_id: str,
        dataset_id: str | None = None,
        metrics: str | None = None,
    ) -> str:
        """Evaluate a trained model. If dataset_id is given, evaluates on that dataset (test set). Otherwise uses OOF predictions.

        metrics is a comma-separated list, e.g. 'accuracy,roc_auc,f1'. If omitted, uses defaults for the task type.
        """
        try:
            model_art = session.get_model(model_id)

            with capture_stdout():
                import numpy as np
                from sklearn import metrics as sklearn_metrics

                metric_list = None
                if metrics:
                    metric_list = [m.strip() for m in metrics.split(",")]

                # Determine what data to evaluate on
                if dataset_id:
                    ds = session.get_dataset(dataset_id)
                    if ds.target_column is None or ds.target_column not in ds.df.columns:
                        return error_response("validation", "Dataset has no target column")

                    from endgame.mcp.tools._encoding import apply_encoders
                    X = ds.df.drop(columns=[ds.target_column])
                    y_raw = ds.df[ds.target_column]
                    X, y_true = apply_encoders(X, y_raw, model_art)

                    y_pred = model_art.estimator.predict(X)
                    y_proba = None
                    if hasattr(model_art.estimator, "predict_proba"):
                        try:
                            y_proba = model_art.estimator.predict_proba(X)
                        except Exception:
                            pass
                else:
                    # Use OOF predictions from training
                    if model_art.oof_predictions is None:
                        return error_response(
                            "validation",
                            "No OOF predictions available. Provide a dataset_id for evaluation.",
                        )
                    ds = session.get_dataset(model_art.dataset_id)
                    from endgame.mcp.tools._encoding import encode_target
                    y_true, _ = encode_target(
                        ds.df[ds.target_column],
                        target_encoder=model_art.target_encoder,
                    )
                    y_pred = model_art.oof_predictions
                    y_proba = None

                # Default metrics
                if metric_list is None:
                    if model_art.task_type == "regression":
                        metric_list = ["rmse", "r2", "mae"]
                    else:
                        metric_list = ["accuracy", "f1", "roc_auc"]

                computed = {}
                for m in metric_list:
                    try:
                        if m == "accuracy":
                            computed[m] = float(sklearn_metrics.accuracy_score(y_true, y_pred))
                        elif m == "f1":
                            computed[m] = float(sklearn_metrics.f1_score(y_true, y_pred, average="weighted"))
                        elif m == "precision":
                            computed[m] = float(sklearn_metrics.precision_score(y_true, y_pred, average="weighted"))
                        elif m == "recall":
                            computed[m] = float(sklearn_metrics.recall_score(y_true, y_pred, average="weighted"))
                        elif m == "balanced_accuracy":
                            computed[m] = float(sklearn_metrics.balanced_accuracy_score(y_true, y_pred))
                        elif m == "matthews_corrcoef":
                            computed[m] = float(sklearn_metrics.matthews_corrcoef(y_true, y_pred))
                        elif m == "cohen_kappa":
                            computed[m] = float(sklearn_metrics.cohen_kappa_score(y_true, y_pred))
                        elif m == "log_loss" and y_proba is not None:
                            computed[m] = float(sklearn_metrics.log_loss(y_true, y_proba))
                        elif m == "roc_auc":
                            if y_proba is not None and y_proba.ndim == 2:
                                if y_proba.shape[1] == 2:
                                    computed[m] = float(sklearn_metrics.roc_auc_score(y_true, y_proba[:, 1]))
                                else:
                                    computed[m] = float(sklearn_metrics.roc_auc_score(y_true, y_proba, multi_class="ovr"))
                        elif m == "rmse":
                            computed[m] = float(np.sqrt(sklearn_metrics.mean_squared_error(y_true, y_pred)))
                        elif m == "r2":
                            computed[m] = float(sklearn_metrics.r2_score(y_true, y_pred))
                        elif m == "mae":
                            computed[m] = float(sklearn_metrics.mean_absolute_error(y_true, y_pred))
                        elif m == "mape":
                            computed[m] = float(sklearn_metrics.mean_absolute_percentage_error(y_true, y_pred))
                        elif m == "median_ae":
                            computed[m] = float(sklearn_metrics.median_absolute_error(y_true, y_pred))
                        elif m == "max_error":
                            computed[m] = float(sklearn_metrics.max_error(y_true, y_pred))
                        elif m == "explained_variance":
                            computed[m] = float(sklearn_metrics.explained_variance_score(y_true, y_pred))
                    except Exception:
                        pass

                return ok_response({
                    "model_id": model_id,
                    "model_name": model_art.name,
                    "evaluated_on": dataset_id or f"OOF ({model_art.dataset_id})",
                    "n_samples": len(y_true),
                    "metrics": {k: round(v, 4) for k, v in computed.items()},
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def explain_model(
        model_id: str,
        method: str = "importance",
        top_n: int = 15,
    ) -> str:
        """Get feature importance or model explanations. Methods: importance, permutation."""
        try:
            model_art = session.get_model(model_id)

            with capture_stdout():
                import numpy as np

                estimator = model_art.estimator
                feature_names = model_art.feature_names

                if method == "importance":
                    if hasattr(estimator, "feature_importances_"):
                        importances = estimator.feature_importances_
                    elif hasattr(estimator, "coef_"):
                        importances = np.abs(estimator.coef_).flatten()
                    else:
                        return error_response(
                            "validation",
                            f"Model '{model_art.name}' doesn't provide feature_importances_",
                            hint="Try method='permutation' instead",
                        )

                    # Build sorted list — handle dict (e.g. LGBMWrapper) or array
                    if isinstance(importances, dict):
                        pairs = sorted(
                            importances.items(),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[:top_n]
                    else:
                        imp_list = importances.tolist() if hasattr(importances, "tolist") else list(importances)
                        if len(feature_names) != len(imp_list):
                            feature_names = [f"feature_{i}" for i in range(len(imp_list))]
                        pairs = sorted(
                            zip(feature_names, imp_list),
                            key=lambda x: abs(x[1]),
                            reverse=True,
                        )[:top_n]

                    features = [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]

                    return ok_response({
                        "model_id": model_id,
                        "method": "importance",
                        "features": features,
                    })

                elif method == "permutation":
                    ds = session.get_dataset(model_art.dataset_id)
                    X = ds.df.drop(columns=[ds.target_column])
                    y_raw = ds.df[ds.target_column]

                    from endgame.mcp.tools._encoding import apply_encoders
                    X, y = apply_encoders(X, y_raw, model_art)

                    from sklearn.inspection import permutation_importance

                    result = permutation_importance(
                        estimator, X, y, n_repeats=10, random_state=42, n_jobs=-1
                    )

                    pairs = sorted(
                        zip(X.columns.tolist(), result.importances_mean.tolist()),
                        key=lambda x: abs(x[1]),
                        reverse=True,
                    )[:top_n]

                    features = [{"feature": f, "importance": round(float(v), 4)} for f, v in pairs]

                    return ok_response({
                        "model_id": model_id,
                        "method": "permutation",
                        "features": features,
                    })

                else:
                    return error_response(
                        "validation",
                        f"Unknown method: {method}",
                        hint="Use 'importance' or 'permutation'",
                    )

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
