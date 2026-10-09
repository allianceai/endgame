"""MCP resource definitions (catalog, presets, metrics, session state)."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.resource("endgame://catalog/models")
    def catalog_models() -> str:
        """All available models grouped by family with name, display_name, fit_time, and description."""
        from endgame.automl.model_registry import MODEL_FAMILIES, MODEL_REGISTRY

        by_family: dict[str, list] = {}
        for name, info in MODEL_REGISTRY.items():
            fam = info.family
            if fam not in by_family:
                by_family[fam] = []
            by_family[fam].append({
                "name": name,
                "display_name": info.display_name,
                "fit_time": info.typical_fit_time,
                "interpretable": info.interpretable,
                "task_types": info.task_types,
                "description": info.notes,
            })

        catalog = {
            "total_models": len(MODEL_REGISTRY),
            "families": {
                fam: {
                    "label": MODEL_FAMILIES.get(fam, fam),
                    "models": models,
                }
                for fam, models in sorted(by_family.items())
            },
        }
        return json.dumps(catalog, indent=2)

    @mcp.resource("endgame://catalog/presets")
    def catalog_presets() -> str:
        """AutoML preset configurations with descriptions, time limits, and model pools."""
        from endgame.automl.presets import PRESETS

        presets = {}
        for name, p in PRESETS.items():
            presets[name] = {
                "description": p.description,
                "default_time_limit": p.default_time_limit,
                "cv_folds": p.cv_folds,
                "n_models": len(p.model_pool),
                "model_pool": p.model_pool,
                "ensemble_method": p.ensemble_method,
                "hyperparameter_tune": p.hyperparameter_tune,
                "calibrate": p.calibrate,
                "feature_engineering": p.feature_engineering,
            }
        return json.dumps(presets, indent=2)

    @mcp.resource("endgame://catalog/visualizers")
    def catalog_visualizers() -> str:
        """Available chart types with required inputs."""
        visualizers = {
            "ml_evaluation": {
                "roc_curve": {"requires": ["model_id", "dataset_id"], "description": "ROC curve with AUC"},
                "pr_curve": {"requires": ["model_id", "dataset_id"], "description": "Precision-Recall curve"},
                "confusion_matrix": {"requires": ["model_id", "dataset_id"], "description": "Confusion matrix heatmap"},
                "calibration_plot": {"requires": ["model_id", "dataset_id"], "description": "Calibration (reliability) diagram"},
                "lift_chart": {"requires": ["model_id", "dataset_id"], "description": "Lift/gain chart"},
                "feature_importance": {"requires": ["model_id"], "description": "Feature importance bar chart"},
                "pdp": {"requires": ["model_id", "dataset_id"], "description": "Partial dependence plot"},
                "waterfall": {"requires": ["model_id", "dataset_id"], "description": "SHAP waterfall for single prediction"},
            },
            "data_exploration": {
                "histogram": {"requires": ["dataset_id"], "description": "Distribution histogram"},
                "scatterplot": {"requires": ["dataset_id"], "description": "2D scatter plot"},
                "heatmap": {"requires": ["dataset_id"], "description": "Correlation heatmap"},
                "box_plot": {"requires": ["dataset_id"], "description": "Box plot for distributions"},
                "violin_plot": {"requires": ["dataset_id"], "description": "Violin plot"},
                "bar_chart": {"requires": ["dataset_id"], "description": "Bar chart"},
                "line_chart": {"requires": ["dataset_id"], "description": "Line chart"},
                "parallel_coordinates": {"requires": ["dataset_id"], "description": "Parallel coordinates plot"},
            },
            "reports": {
                "classification_report": {"requires": ["model_id", "dataset_id"], "description": "Full classification evaluation report"},
                "regression_report": {"requires": ["model_id", "dataset_id"], "description": "Full regression evaluation report"},
            },
        }
        return json.dumps(visualizers, indent=2)

    @mcp.resource("endgame://catalog/metrics")
    def catalog_metrics() -> str:
        """Available evaluation metrics for classification and regression."""
        metrics = {
            "classification": {
                "accuracy": "Fraction of correct predictions",
                "roc_auc": "Area under the ROC curve (binary/OVR)",
                "f1": "Weighted F1 score",
                "log_loss": "Logarithmic loss (cross-entropy)",
                "precision": "Weighted precision",
                "recall": "Weighted recall",
                "balanced_accuracy": "Balanced accuracy (macro recall)",
                "matthews_corrcoef": "Matthews correlation coefficient",
                "cohen_kappa": "Cohen's kappa agreement",
            },
            "regression": {
                "rmse": "Root mean squared error",
                "r2": "Coefficient of determination",
                "mae": "Mean absolute error",
                "mape": "Mean absolute percentage error",
                "median_ae": "Median absolute error",
                "max_error": "Maximum residual error",
                "explained_variance": "Explained variance score",
            },
        }
        return json.dumps(metrics, indent=2)

    @mcp.resource("endgame://session/state")
    def session_state() -> str:
        """Current session state: loaded datasets, trained models, and visualizations."""
        return json.dumps(session.get_state_summary(), indent=2, default=str)

    @mcp.resource("endgame://catalog/modules")
    def catalog_modules() -> str:
        """Every Endgame module: what it is for and which MCP tools reach it."""
        from endgame.mcp.catalog import overview

        return json.dumps(overview(), indent=2)

    @mcp.resource("endgame://guide/workflow")
    def guide_workflow() -> str:
        """How to run a strong, honest ML experiment with Endgame (same text as the guide tool)."""
        from endgame.mcp.guide import TOPICS

        return "\n\n".join(TOPICS.values())

    @mcp.resource("endgame://guide/examples")
    def guide_examples() -> str:
        """Example tool-call workflows for common ML tasks."""
        examples = {
            "tabular_experiment": {
                "description": "A full experiment on one table",
                "steps": [
                    'guide()',
                    'load_data(source="train.csv", target_column="target")',
                    'check_data_quality(dataset_id="ds_a")',
                    'split_data(dataset_id="ds_a", test_size=0.2)  # -> ds_train, ds_test',
                    'compare_models(dataset_id="ds_train", models=["linear", "lgbm"])  # baselines',
                    'engineer_features(dataset_id="ds_train", operations=[{"type": "interactions"}, '
                    '{"type": "target_encode"}])',
                    'select_features(dataset_id="ds_train_features", method="mrmr", n_features=30, '
                    'apply_to=["ds_test"])',
                    'recommend_models(dataset_id="ds_sel")',
                    'compare_models(dataset_id="ds_sel")  # GBDTs + foundation models + baseline, same folds',
                    'ensemble(model_ids=["model_1", "model_2", "model_3"], method="hill_climbing")',
                    'evaluate_model(model_id="model_ens", dataset_id="ds_test_sel")',
                    'explain_model(model_id="model_1")',
                ],
            },
            "entity_level_from_sensor_data": {
                "description": "Per-entity prediction from a long table of sensor/tracking frames",
                "steps": [
                    'load_data(source="players.csv", target_column="outcome")  # one row per player',
                    'load_data(source="tracking.csv")  # many rows per player',
                    'engineer_features(dataset_id="ds_players", operations=[{"type": "aggregate", '
                    '"source": "ds_tracking", "by": ["player_id"], "columns": ["speed", "accel"], '
                    '"aggs": ["mean", "max", "q90", "sample_entropy", "higuchi_fd"], "order_by": "time"}, '
                    '{"type": "group_normalize", "by": "position", "method": "zscore"}])',
                    'compare_models(dataset_id="ds_players_features", group_column="team")',
                ],
            },
            "time_ordered": {
                "description": "Rows in time order (one per day/night/week)",
                "steps": [
                    'engineer_features(dataset_id="ds_a", operations=[{"type": "lags", "columns": ["y_prev"], '
                    '"order_by": "date"}, {"type": "rolling", "windows": [3, 7], "order_by": "date"}])',
                    'compare_models(dataset_id="ds_a_features", time_ordered=True)',
                ],
            },
            "any_module": {
                "description": "Use a module without a dedicated tool",
                "steps": [
                    'list_modules(search="conformal")',
                    'describe_api(name="endgame.calibration.ConformalRegressor")',
                    'run_python(code="X = dataset(\'ds_a\'); ...")',
                    'transform_data(dataset_id="ds_a", transformer="endgame.dimensionality_reduction.UMAPReducer", '
                    'params={"n_components": 5})',
                    'train_model(dataset_id="ds_a", model_name="endgame.models.trees.RotationForestClassifier")',
                ],
            },
            "automl": {
                "description": "Full AutoML pipeline in one call",
                "steps": [
                    'load_data(source="data.csv", target_column="label")',
                    'automl(dataset_id="ds_a", preset="high_quality")',
                ],
            },
        }
        return json.dumps(examples, indent=2)
