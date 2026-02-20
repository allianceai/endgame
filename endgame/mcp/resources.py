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

    @mcp.resource("endgame://guide/examples")
    def guide_examples() -> str:
        """Example tool-call workflows for common ML tasks."""
        examples = {
            "classification": {
                "description": "Train a classifier from CSV",
                "steps": [
                    'load_data(source="data.csv", target_column="target")',
                    'recommend_models(dataset_id="ds_...", time_budget="medium")',
                    'train_model(dataset_id="ds_...", model_name="lgbm")',
                    'evaluate_model(model_id="model_...", dataset_id="ds_...")',
                    'create_visualization(chart_type="roc_curve", model_id="model_...", dataset_id="ds_...")',
                    'export_script(model_id="model_...")',
                ],
            },
            "regression": {
                "description": "Train a regressor",
                "steps": [
                    'load_data(source="house_prices.csv", target_column="price")',
                    'inspect_data(dataset_id="ds_...", operation="summary")',
                    'train_model(dataset_id="ds_...", model_name="xgb")',
                    'evaluate_model(model_id="model_...", dataset_id="ds_...")',
                ],
            },
            "automl": {
                "description": "Full AutoML pipeline",
                "steps": [
                    'load_data(source="data.csv", target_column="label")',
                    'automl(dataset_id="ds_...", preset="medium_quality")',
                    'evaluate_model(model_id="model_...")',
                ],
            },
            "compare_models": {
                "description": "Compare multiple models quickly",
                "steps": [
                    'load_data(source="data.csv", target_column="target")',
                    'quick_compare(dataset_id="ds_...", preset="medium_quality")',
                ],
            },
            "explore_data": {
                "description": "Explore and understand a dataset",
                "steps": [
                    'load_data(source="data.csv", target_column="target")',
                    'inspect_data(dataset_id="ds_...", operation="summary")',
                    'inspect_data(dataset_id="ds_...", operation="correlations")',
                    'inspect_data(dataset_id="ds_...", operation="missing")',
                    'create_visualization(chart_type="histogram", dataset_id="ds_...", params={"column": "age"})',
                ],
            },
        }
        return json.dumps(examples, indent=2)
