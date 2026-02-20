"""Preprocessing tool: preprocess."""

from __future__ import annotations

import json

from mcp.server.fastmcp import FastMCP

from endgame.mcp.server import capture_stdout, error_response, ok_response
from endgame.mcp.session import SessionManager


def register(mcp: FastMCP, session: SessionManager) -> None:

    @mcp.tool()
    def preprocess(
        dataset_id: str,
        operations: str | list,
    ) -> str:
        """Apply preprocessing operations to a dataset. Returns a new dataset ID.

        operations: array of operations, each with a 'type' and optional parameters.
        Supported types:
        - impute: Fill missing values. params: {strategy: 'mean'|'median'|'most_frequent'|'constant'}
        - scale: Scale numeric features. params: {method: 'standard'|'minmax'|'robust'}
        - encode: Encode categorical features. params: {method: 'onehot'|'label'|'target'}
        - balance: Handle class imbalance. params: {method: 'smote'|'random_over'|'random_under'}
        - select_features: Feature selection. params: {method: 'variance'|'mutual_info', top_k: 20}
        - drop_columns: Drop specific columns. params: {columns: ['col1', 'col2']}

        Example: [{"type": "impute", "strategy": "median"}, {"type": "scale", "method": "standard"}]
        """
        try:
            ds = session.get_dataset(dataset_id)
            if isinstance(operations, list):
                ops = operations
            else:
                ops = json.loads(operations)

            with capture_stdout():
                import numpy as np
                import pandas as pd

                df = ds.df.copy()
                target_col = ds.target_column
                applied = []

                for op in ops:
                    op_type = op.get("type", op.get("name", ""))

                    if op_type == "impute":
                        strategy = op.get("strategy", "median")
                        from sklearn.impute import SimpleImputer
                        num_cols = df.select_dtypes(include="number").columns
                        cat_cols = df.select_dtypes(include=["object", "category"]).columns
                        if target_col:
                            num_cols = [c for c in num_cols if c != target_col]
                            cat_cols = [c for c in cat_cols if c != target_col]

                        if len(num_cols) > 0:
                            imp = SimpleImputer(strategy=strategy if strategy != "most_frequent" else "median")
                            df[num_cols] = imp.fit_transform(df[num_cols])
                        if len(cat_cols) > 0:
                            imp = SimpleImputer(strategy="most_frequent")
                            df[cat_cols] = imp.fit_transform(df[cat_cols])
                        applied.append(f"impute({strategy})")

                    elif op_type == "scale":
                        method = op.get("method", "standard")
                        num_cols = df.select_dtypes(include="number").columns
                        if target_col:
                            num_cols = [c for c in num_cols if c != target_col]
                        if len(num_cols) > 0:
                            if method == "standard":
                                from sklearn.preprocessing import StandardScaler
                                scaler = StandardScaler()
                            elif method == "minmax":
                                from sklearn.preprocessing import MinMaxScaler
                                scaler = MinMaxScaler()
                            elif method == "robust":
                                from sklearn.preprocessing import RobustScaler
                                scaler = RobustScaler()
                            else:
                                return error_response("validation", f"Unknown scale method: {method}")
                            df[num_cols] = scaler.fit_transform(df[num_cols])
                        applied.append(f"scale({method})")

                    elif op_type == "encode":
                        method = op.get("method", "label")
                        cat_cols = df.select_dtypes(include=["object", "category"]).columns
                        if target_col:
                            cat_cols = [c for c in cat_cols if c != target_col]
                        if len(cat_cols) > 0:
                            if method == "label":
                                from sklearn.preprocessing import LabelEncoder
                                for col in cat_cols:
                                    le = LabelEncoder()
                                    df[col] = le.fit_transform(df[col].astype(str))
                            elif method == "onehot":
                                df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
                            elif method == "target":
                                if target_col and target_col in df.columns:
                                    for col in cat_cols:
                                        means = df.groupby(col)[target_col].mean()
                                        df[col] = df[col].map(means)
                            else:
                                return error_response("validation", f"Unknown encode method: {method}")
                        applied.append(f"encode({method})")

                    elif op_type == "balance":
                        method = op.get("method", "smote")
                        if target_col is None:
                            return error_response("validation", "Cannot balance without target column")
                        try:
                            X = df.drop(columns=[target_col])
                            y = df[target_col]
                            # Handle non-numeric for SMOTE
                            cat_cols = X.select_dtypes(include=["object", "category"]).columns
                            if len(cat_cols) > 0:
                                from sklearn.preprocessing import LabelEncoder
                                for col in cat_cols:
                                    X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                            if X.isna().any().any():
                                X = X.fillna(X.median(numeric_only=True))

                            if method == "smote":
                                from imblearn.over_sampling import SMOTE
                                sampler = SMOTE(random_state=42)
                            elif method == "random_over":
                                from imblearn.over_sampling import RandomOverSampler
                                sampler = RandomOverSampler(random_state=42)
                            elif method == "random_under":
                                from imblearn.under_sampling import RandomUnderSampler
                                sampler = RandomUnderSampler(random_state=42)
                            else:
                                return error_response("validation", f"Unknown balance method: {method}")

                            X_res, y_res = sampler.fit_resample(X, y)
                            df = pd.concat([X_res, y_res], axis=1)
                        except ImportError:
                            return error_response(
                                "missing_dependency",
                                "imbalanced-learn required for balancing",
                                hint="pip install imbalanced-learn",
                            )
                        applied.append(f"balance({method})")

                    elif op_type == "select_features":
                        method = op.get("method", "variance")
                        top_k = op.get("top_k", 20)
                        if target_col is None:
                            return error_response("validation", "Feature selection requires target column")
                        X = df.drop(columns=[target_col])
                        y = df[target_col]
                        num_cols = X.select_dtypes(include="number").columns.tolist()
                        top_k = min(top_k, len(num_cols))

                        if method == "variance":
                            from sklearn.feature_selection import VarianceThreshold
                            sel = VarianceThreshold()
                            sel.fit(X[num_cols])
                            variances = sel.variances_
                            top_idx = np.argsort(variances)[-top_k:]
                            keep = [num_cols[i] for i in top_idx]
                        elif method == "mutual_info":
                            from sklearn.feature_selection import (
                                mutual_info_classif,
                                mutual_info_regression,
                            )
                            X_num = X[num_cols].fillna(0)
                            if ds.task_type == "regression":
                                mi = mutual_info_regression(X_num, y, random_state=42)
                            else:
                                mi = mutual_info_classif(X_num, y, random_state=42)
                            top_idx = np.argsort(mi)[-top_k:]
                            keep = [num_cols[i] for i in top_idx]
                        else:
                            return error_response("validation", f"Unknown selection method: {method}")

                        non_num = [c for c in X.columns if c not in num_cols]
                        df = pd.concat([X[keep + non_num], y], axis=1)
                        applied.append(f"select_features({method}, top_k={top_k})")

                    elif op_type == "drop_columns":
                        columns = op.get("columns", [])
                        df = df.drop(columns=[c for c in columns if c in df.columns])
                        applied.append(f"drop_columns({columns})")

                    else:
                        return error_response(
                            "validation",
                            f"Unknown operation type: {op_type}",
                            hint="Supported: impute, scale, encode, balance, select_features, drop_columns",
                        )

                art = session.add_dataset(
                    df=df,
                    name=f"{ds.name}_preprocessed",
                    source="derived",
                    target_column=target_col if target_col and target_col in df.columns else None,
                    task_type=ds.task_type,
                )

                return ok_response({
                    "dataset_id": art.id,
                    "shape": list(df.shape),
                    "operations_applied": applied,
                    "columns": list(df.columns[:30]),
                })

        except KeyError as e:
            return error_response("not_found", str(e))
        except json.JSONDecodeError:
            return error_response("validation", "Invalid JSON for operations parameter")
        except Exception as e:
            return error_response("internal", str(e))
