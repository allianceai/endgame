"""Visualization tools: create_visualization, create_report."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def create_visualization(
        chart_type: str,
        model_id: str | None = None,
        dataset_id: str | None = None,
        params: str | None = None,
        title: str | None = None,
    ) -> str:
        """Create a visualization and save as self-contained HTML.

        chart_type: roc_curve, pr_curve, confusion_matrix, calibration_plot, lift_chart,
                    feature_importance, histogram, scatterplot, heatmap, box_plot,
                    bar_chart, line_chart.
        params: optional JSON string with extra parameters (e.g. '{"column": "age"}' for histogram).
        """
        try:
            extra = json.loads(params) if params else {}

            with capture_stdout():

                import numpy as np

                output_dir = session.working_dir / "visualizations"
                output_dir.mkdir(exist_ok=True)
                out_path = str(output_dir / f"{chart_type}_{len(session.visualizations)}.html")

                # Helper: get model predictions
                def _get_model_data():
                    if model_id is None:
                        raise ValueError("model_id required for this chart type")
                    m = session.get_model(model_id)
                    ds_id = dataset_id or m.dataset_id
                    ds = session.get_dataset(ds_id)
                    X = ds.df.drop(columns=[ds.target_column])
                    y_raw = ds.df[ds.target_column]
                    from endgame.mcp.tools._encoding import apply_encoders
                    X, y = apply_encoders(X, y_raw, m)
                    return m, ds, X, y

                # Helper: safely get probability scores for binary classification
                def _get_binary_proba(m, X, y, chart_name):
                    n_classes = len(np.unique(y))
                    if n_classes > 2:
                        return None, error_response(
                            "validation",
                            f"{chart_name} requires binary classification (found {n_classes} classes).",
                            hint="Use confusion_matrix for multiclass problems.",
                        )
                    if not hasattr(m.estimator, "predict_proba"):
                        return None, error_response(
                            "validation",
                            f"Model '{m.name}' does not support predict_proba, required for {chart_name}.",
                            hint="Try confusion_matrix instead.",
                        )
                    try:
                        y_proba = m.estimator.predict_proba(X)
                    except Exception as exc:
                        return None, error_response(
                            "internal",
                            f"predict_proba failed: {exc}",
                        )
                    scores = y_proba[:, 1] if y_proba.ndim == 2 else y_proba
                    return scores, None

                # ML evaluation charts
                if chart_type == "roc_curve":
                    m, ds, X, y = _get_model_data()
                    scores, err = _get_binary_proba(m, X, y, "roc_curve")
                    if err is not None:
                        return err
                    from endgame.visualization import ROCCurveVisualizer
                    viz = ROCCurveVisualizer(y_true=y, y_score=scores)
                    viz.save(out_path)

                elif chart_type == "pr_curve":
                    m, ds, X, y = _get_model_data()
                    scores, err = _get_binary_proba(m, X, y, "pr_curve")
                    if err is not None:
                        return err
                    from endgame.visualization import PRCurveVisualizer
                    viz = PRCurveVisualizer(y_true=y, y_score=scores)
                    viz.save(out_path)

                elif chart_type == "confusion_matrix":
                    m, ds, X, y = _get_model_data()
                    from endgame.visualization import ConfusionMatrixVisualizer
                    y_pred = m.estimator.predict(X)
                    viz = ConfusionMatrixVisualizer(y_true=y, y_pred=y_pred)
                    viz.save(out_path)

                elif chart_type == "calibration_plot":
                    m, ds, X, y = _get_model_data()
                    scores, err = _get_binary_proba(m, X, y, "calibration_plot")
                    if err is not None:
                        return err
                    from endgame.visualization import CalibrationPlotVisualizer
                    viz = CalibrationPlotVisualizer(y_true=y, y_prob=scores)
                    viz.save(out_path)

                elif chart_type == "lift_chart":
                    m, ds, X, y = _get_model_data()
                    scores, err = _get_binary_proba(m, X, y, "lift_chart")
                    if err is not None:
                        return err
                    from endgame.visualization import LiftChartVisualizer
                    viz = LiftChartVisualizer(y_true=y, y_score=scores)
                    viz.save(out_path)

                elif chart_type == "feature_importance":
                    if model_id is None:
                        return error_response("validation", "model_id required")
                    m = session.get_model(model_id)
                    from endgame.visualization import BarChartVisualizer
                    if hasattr(m.estimator, "feature_importances_"):
                        imp = m.estimator.feature_importances_
                        names = m.feature_names or [f"f_{i}" for i in range(len(imp))]
                        top_n = extra.get("top_n", 20)
                        idx = np.argsort(imp)[-top_n:][::-1]
                        viz = BarChartVisualizer(
                            labels=[names[i] for i in idx],
                            values=[float(imp[i]) for i in idx],
                            title=title or "Feature Importance",
                        )
                        viz.save(out_path)
                    else:
                        return error_response("validation", "Model has no feature_importances_")

                # Data exploration charts
                elif chart_type == "histogram":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    col = extra.get("column", ds.target_column)
                    if col is None:
                        return error_response("validation", "Specify column in params")
                    from endgame.visualization import HistogramVisualizer
                    viz = HistogramVisualizer(
                        data=ds.df[col].dropna().tolist(),
                        title=title or f"Distribution of {col}",
                    )
                    viz.save(out_path)

                elif chart_type == "scatterplot":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    x_col = extra.get("x")
                    y_col = extra.get("y")
                    if not x_col or not y_col:
                        num_cols = ds.df.select_dtypes(include="number").columns.tolist()
                        if len(num_cols) < 2:
                            return error_response("validation", "Need at least 2 numeric columns")
                        x_col = x_col or num_cols[0]
                        y_col = y_col or num_cols[1]
                    from endgame.visualization import ScatterplotVisualizer
                    viz = ScatterplotVisualizer(
                        x=ds.df[x_col].tolist(),
                        y=ds.df[y_col].tolist(),
                        x_label=x_col,
                        y_label=y_col,
                        title=title or f"{x_col} vs {y_col}",
                    )
                    viz.save(out_path)

                elif chart_type == "heatmap":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    from endgame.visualization import HeatmapVisualizer
                    num_df = ds.df.select_dtypes(include="number")
                    if num_df.shape[1] < 2:
                        return error_response(
                            "validation",
                            f"Heatmap requires at least 2 numeric columns (found {num_df.shape[1]}).",
                            hint="Use bar_chart or histogram for categorical data.",
                        )
                    corr = num_df.corr()
                    viz = HeatmapVisualizer(
                        data=corr.values.tolist(),
                        x_labels=corr.columns.tolist(),
                        y_labels=corr.index.tolist(),
                        title=title or "Correlation Heatmap",
                    )
                    viz.save(out_path)

                elif chart_type == "box_plot":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    col = extra.get("column")
                    from endgame.visualization import BoxPlotVisualizer
                    if col:
                        viz = BoxPlotVisualizer(
                            data=[ds.df[col].dropna().tolist()],
                            labels=[col],
                            title=title or f"Box Plot of {col}",
                        )
                    else:
                        num_cols = ds.df.select_dtypes(include="number").columns.tolist()[:10]
                        viz = BoxPlotVisualizer(
                            data=[ds.df[c].dropna().tolist() for c in num_cols],
                            labels=num_cols,
                            title=title or "Box Plots",
                        )
                    viz.save(out_path)

                elif chart_type == "bar_chart":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    col = extra.get("column", ds.target_column)
                    if col is None:
                        return error_response("validation", "Specify column in params")
                    from endgame.visualization import BarChartVisualizer
                    vc = ds.df[col].value_counts().head(20)
                    viz = BarChartVisualizer(
                        labels=vc.index.astype(str).tolist(),
                        values=vc.values.tolist(),
                        title=title or f"Value Counts: {col}",
                    )
                    viz.save(out_path)

                elif chart_type == "line_chart":
                    if dataset_id is None:
                        return error_response("validation", "dataset_id required")
                    ds = session.get_dataset(dataset_id)
                    col = extra.get("column")
                    if col is None:
                        return error_response("validation", "Specify column in params")
                    from endgame.visualization import LineChartVisualizer
                    viz = LineChartVisualizer(
                        x=list(range(len(ds.df[col]))),
                        y=ds.df[col].tolist(),
                        title=title or f"Line Chart: {col}",
                    )
                    viz.save(out_path)

                else:
                    return error_response(
                        "validation",
                        f"Unknown chart type: {chart_type}",
                        hint="Read endgame://catalog/visualizers for available chart types",
                    )

                art = session.add_visualization(
                    chart_type=chart_type,
                    html_path=out_path,
                    model_id=model_id,
                    dataset_id=dataset_id,
                )

                return ok_response({
                    "visualization_id": art.id,
                    "chart_type": chart_type,
                    "html_path": out_path,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except json.JSONDecodeError:
            return error_response("validation", "Invalid JSON for params")
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def create_report(
        model_id: str,
        dataset_id: str | None = None,
        report_type: str = "auto",
    ) -> str:
        """Generate a comprehensive HTML evaluation report. report_type: auto, classification, regression."""
        try:
            model_art = session.get_model(model_id)
            ds_id = dataset_id or model_art.dataset_id
            ds = session.get_dataset(ds_id)

            with capture_stdout():

                import numpy as np

                output_dir = session.working_dir / "reports"
                output_dir.mkdir(exist_ok=True)
                out_path = str(output_dir / f"report_{model_art.name}.html")

                X = ds.df.drop(columns=[ds.target_column])
                y_raw = ds.df[ds.target_column]

                from endgame.mcp.tools._encoding import apply_encoders
                X, y = apply_encoders(X, y_raw, model_art)

                rtype = report_type
                if rtype == "auto":
                    rtype = "regression" if model_art.task_type == "regression" else "classification"

                if rtype == "classification":
                    from endgame.visualization import ClassificationReport
                    y_pred = model_art.estimator.predict(X)
                    y_proba = None
                    if hasattr(model_art.estimator, "predict_proba"):
                        try:
                            y_proba = model_art.estimator.predict_proba(X)
                        except Exception:
                            pass
                    report = ClassificationReport(
                        y_true=y if isinstance(y, np.ndarray) else y.values,
                        y_pred=y_pred,
                        y_prob=y_proba,
                        title=f"Classification Report: {model_art.name}",
                    )
                    report.save(out_path)
                else:
                    from endgame.visualization import RegressionReport
                    y_pred = model_art.estimator.predict(X)
                    report = RegressionReport(
                        y_true=y if isinstance(y, np.ndarray) else y.values,
                        y_pred=y_pred,
                        title=f"Regression Report: {model_art.name}",
                    )
                    report.save(out_path)

                art = session.add_visualization(
                    chart_type=f"{rtype}_report",
                    html_path=out_path,
                    model_id=model_id,
                    dataset_id=ds_id,
                )

                return ok_response({
                    "visualization_id": art.id,
                    "report_type": rtype,
                    "html_path": out_path,
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
