"""Data tools: load_data, inspect_data, split_data."""

from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def load_data(
        source: str,
        target_column: str | None = None,
        name: str | None = None,
        sample_n: int | None = None,
    ) -> str:
        """Load a dataset from a CSV/Parquet file path, URL, or OpenML (e.g. 'openml:31' or 'openml:credit-g').

        Returns a dataset ID for use with other tools.
        """
        try:
            with capture_stdout():
                import pandas as pd

                ds_name = name or source.split("/")[-1].split(".")[0]

                # OpenML loading
                if source.startswith("openml:"):
                    ref = source[len("openml:"):]
                    try:
                        import openml
                        if ref.isdigit():
                            dataset = openml.datasets.get_dataset(int(ref))
                        else:
                            datasets = openml.datasets.list_datasets(output_format="dataframe")
                            match = datasets[datasets["name"].str.lower() == ref.lower()]
                            if match.empty:
                                return error_response("not_found", f"OpenML dataset '{ref}' not found")
                            dataset = openml.datasets.get_dataset(int(match.iloc[0]["did"]))
                        X, y, cat_ind, attr_names = dataset.get_data(
                            target=dataset.default_target_attribute
                        )
                        df = X.copy()
                        tc = target_column or dataset.default_target_attribute or "target"
                        df[tc] = y
                        ds_name = name or dataset.name
                    except ImportError:
                        return error_response(
                            "missing_dependency",
                            "openml package required. Install with: pip install openml",
                            hint="pip install openml",
                        )
                # URL loading
                elif source.startswith("http://") or source.startswith("https://"):
                    df = pd.read_csv(source)
                    tc = target_column
                # Local file loading
                else:
                    from pathlib import Path
                    p = Path(source)
                    if not p.exists():
                        return error_response("not_found", f"File not found: {source}")
                    suffix = p.suffix.lower()
                    if suffix in (".parquet", ".pq"):
                        df = pd.read_parquet(source)
                    elif suffix in (".xlsx", ".xls"):
                        df = pd.read_excel(source)
                    elif suffix == ".json":
                        df = pd.read_json(source)
                    else:
                        df = pd.read_csv(source)
                    tc = target_column

                # Sample if requested
                if sample_n and len(df) > sample_n:
                    df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)

                # Infer task type
                task_type = None
                if tc and tc in df.columns:
                    from endgame.automl.utils.data_loader import infer_task_type
                    task_type = infer_task_type(df[tc])

                # Compute basic meta-features
                meta = {
                    "n_samples": len(df),
                    "n_features": len(df.columns) - (1 if tc and tc in df.columns else 0),
                    "n_numeric": len(df.select_dtypes(include="number").columns),
                    "n_categorical": len(df.select_dtypes(include=["object", "category"]).columns),
                    "missing_pct": round(float(df.isna().mean().mean()) * 100, 2),
                }
                if task_type and task_type != "regression" and tc and tc in df.columns:
                    meta["n_classes"] = int(df[tc].nunique())

                art = session.add_dataset(
                    df=df,
                    name=ds_name,
                    source=source,
                    target_column=tc,
                    task_type=task_type,
                    meta_features=meta,
                )

                return ok_response({
                    "dataset_id": art.id,
                    "name": art.name,
                    "shape": list(df.shape),
                    "target_column": tc,
                    "task_type": task_type,
                    "meta_features": meta,
                    "columns": list(df.columns[:30]),
                    "head": df.head(5).to_dict(orient="records"),
                })

        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def inspect_data(
        dataset_id: str,
        operation: str = "summary",
        column: str | None = None,
    ) -> str:
        """Inspect a loaded dataset. Operations: summary, describe, correlations, missing, distribution, head, dtypes."""
        try:
            ds = session.get_dataset(dataset_id)
            df = ds.df

            with capture_stdout():
                if operation == "summary":
                    result = {
                        "name": ds.name,
                        "shape": list(df.shape),
                        "target_column": ds.target_column,
                        "task_type": ds.task_type,
                        "dtypes": {c: str(df[c].dtype) for c in df.columns},
                        "missing": {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().any()},
                        "meta_features": ds.meta_features,
                    }

                elif operation == "describe":
                    if column:
                        desc = df[column].describe().to_dict()
                    else:
                        desc = df.describe(include="all").to_dict()
                    result = {"describe": desc}

                elif operation == "correlations":
                    num_df = df.select_dtypes(include="number")
                    if num_df.shape[1] < 2:
                        return ok_response({"correlations": "Not enough numeric columns"})
                    corr = num_df.corr().round(3)
                    # Only top correlations to keep output manageable
                    import numpy as np
                    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
                    pairs = []
                    for i in range(len(corr)):
                        for j in range(i + 1, len(corr.columns)):
                            pairs.append({
                                "col1": corr.index[i],
                                "col2": corr.columns[j],
                                "correlation": float(corr.iloc[i, j]),
                            })
                    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
                    result = {"top_correlations": pairs[:20]}

                elif operation == "missing":
                    missing = df.isna().sum()
                    result = {
                        "missing_counts": {c: int(v) for c, v in missing.items() if v > 0},
                        "total_missing": int(missing.sum()),
                        "pct_missing": round(float(df.isna().mean().mean()) * 100, 2),
                    }

                elif operation == "distribution":
                    col = column or ds.target_column
                    if col is None or col not in df.columns:
                        return error_response("validation", f"Column '{col}' not found")
                    series = df[col]
                    if series.dtype == "object" or series.dtype.name == "category":
                        result = {"value_counts": series.value_counts().head(20).to_dict()}
                    else:
                        result = {
                            "mean": float(series.mean()),
                            "std": float(series.std()),
                            "min": float(series.min()),
                            "max": float(series.max()),
                            "median": float(series.median()),
                            "skew": float(series.skew()),
                            "quantiles": {
                                str(q): float(series.quantile(q))
                                for q in [0.01, 0.25, 0.5, 0.75, 0.99]
                            },
                        }

                elif operation == "head":
                    result = {"head": df.head(10).to_dict(orient="records")}

                elif operation == "dtypes":
                    result = {"dtypes": {c: str(df[c].dtype) for c in df.columns}}

                else:
                    return error_response(
                        "validation",
                        f"Unknown operation: {operation}",
                        hint="Use one of: summary, describe, correlations, missing, distribution, head, dtypes",
                    )

                return ok_response(result)

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))

    @mcp.tool()
    def split_data(
        dataset_id: str,
        test_size: float = 0.2,
        stratify: bool = True,
    ) -> str:
        """Split a dataset into train and test sets. Returns two new dataset IDs."""
        try:
            ds = session.get_dataset(dataset_id)
            df = ds.df

            with capture_stdout():
                from sklearn.model_selection import train_test_split

                stratify_col = None
                if stratify and ds.target_column and ds.task_type != "regression":
                    stratify_col = df[ds.target_column]

                try:
                    train_df, test_df = train_test_split(
                        df, test_size=test_size, random_state=42,
                        stratify=stratify_col,
                    )
                except ValueError:
                    train_df, test_df = train_test_split(
                        df, test_size=test_size, random_state=42,
                    )

                train_df = train_df.reset_index(drop=True)
                test_df = test_df.reset_index(drop=True)

                train_art = session.add_dataset(
                    df=train_df,
                    name=f"{ds.name}_train",
                    source="derived",
                    target_column=ds.target_column,
                    task_type=ds.task_type,
                )
                test_art = session.add_dataset(
                    df=test_df,
                    name=f"{ds.name}_test",
                    source="derived",
                    target_column=ds.target_column,
                    task_type=ds.task_type,
                )

                return ok_response({
                    "train_dataset_id": train_art.id,
                    "test_dataset_id": test_art.id,
                    "train_shape": list(train_df.shape),
                    "test_shape": list(test_df.shape),
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except Exception as e:
            return error_response("internal", str(e))
