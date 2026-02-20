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
        params: str | dict | None = None,
        title: str | None = None,
    ) -> str:
        """Create a visualization and save as self-contained HTML.

        chart_type: roc_curve, pr_curve, confusion_matrix, calibration_plot, lift_chart,
                    feature_importance, histogram, scatterplot, heatmap, box_plot,
                    bar_chart, line_chart.
        params: extra parameters as a dict or JSON string (e.g. {"column": "age"} for histogram).
        """
        try:
            if isinstance(params, dict):
                extra = params
            elif isinstance(params, str):
                extra = json.loads(params)
            else:
                extra = {}

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

                # ML evaluation charts — use from_estimator() classmethods
                if chart_type == "roc_curve":
                    m, ds, X, y = _get_model_data()
                    from endgame.visualization import ROCCurveVisualizer
                    viz = ROCCurveVisualizer.from_estimator(
                        m.estimator, X, y, title=title or "ROC Curve",
                    )
                    viz.save(out_path)

                elif chart_type == "pr_curve":
                    m, ds, X, y = _get_model_data()
                    from endgame.visualization import PRCurveVisualizer
                    viz = PRCurveVisualizer.from_estimator(
                        m.estimator, X, y, title=title or "Precision-Recall Curve",
                    )
                    viz.save(out_path)

                elif chart_type == "confusion_matrix":
                    m, ds, X, y = _get_model_data()
                    from sklearn.metrics import confusion_matrix as _confusion_matrix

                    from endgame.visualization import ConfusionMatrixVisualizer
                    y_pred = m.estimator.predict(X)
                    cm = _confusion_matrix(y, y_pred)
                    viz = ConfusionMatrixVisualizer(
                        matrix=cm, title=title or "Confusion Matrix",
                    )
                    viz.save(out_path)

                elif chart_type == "calibration_plot":
                    m, ds, X, y = _get_model_data()
                    from endgame.visualization import CalibrationPlotVisualizer
                    viz = CalibrationPlotVisualizer.from_estimator(
                        m.estimator, X, y, title=title or "Calibration Plot",
                    )
                    viz.save(out_path)

                elif chart_type == "lift_chart":
                    m, ds, X, y = _get_model_data()
                    from endgame.visualization import LiftChartVisualizer
                    viz = LiftChartVisualizer.from_estimator(
                        m.estimator, X, y, title=title or "Lift Chart",
                    )
                    viz.save(out_path)

                elif chart_type == "feature_importance":
                    if model_id is None:
                        return error_response("validation", "model_id required")
                    m = session.get_model(model_id)
                    from endgame.visualization import BarChartVisualizer
                    if hasattr(m.estimator, "feature_importances_"):
                        imp = m.estimator.feature_importances_
                        top_n = extra.get("top_n", 20)
                        # Handle dict (e.g. LGBMWrapper) or array
                        if isinstance(imp, dict):
                            pairs = sorted(imp.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
                            labels = [p[0] for p in pairs]
                            values = [float(p[1]) for p in pairs]
                        else:
                            names = m.feature_names or [f"f_{i}" for i in range(len(imp))]
                            idx = np.argsort(imp)[-top_n:][::-1]
                            labels = [names[i] for i in idx]
                            values = [float(imp[i]) for i in idx]
                        viz = BarChartVisualizer(
                            labels=labels,
                            values=values,
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
                        box_data = {col: ds.df[col].dropna().tolist()}
                    else:
                        num_cols = ds.df.select_dtypes(include="number").columns.tolist()[:10]
                        box_data = {c: ds.df[c].dropna().tolist() for c in num_cols}
                    viz = BoxPlotVisualizer(
                        data=box_data,
                        title=title or ("Box Plots" if not col else f"Box Plot of {col}"),
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
                        series={col: ds.df[col].tolist()},
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
                    report = ClassificationReport(
                        model_art.estimator, X, y,
                        model_name=model_art.name,
                        dataset_name=ds.name,
                    )
                    report.save(out_path)
                else:
                    from endgame.visualization import RegressionReport
                    report = RegressionReport(
                        model_art.estimator, X, y,
                        model_name=model_art.name,
                        dataset_name=ds.name,
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
