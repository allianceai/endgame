"""Standalone Python script generator.

Given a ModelArtifact and DatasetArtifact, produces a self-contained script
that reproduces the trained pipeline.
"""

from __future__ import annotations

from endgame.mcp.session import DatasetArtifact, ModelArtifact


def generate_script(
    model_art: ModelArtifact,
    dataset_art: DatasetArtifact,
    dataset_path: str | None = None,
    include_preprocessing: bool = True,
) -> str:
    """Generate a standalone Python script reproducing the pipeline."""

    lines: list[str] = []
    lines.append('"""')
    lines.append(f"Auto-generated pipeline script for model: {model_art.name}")
    lines.append(f"Model type: {model_art.model_type}")
    lines.append(f"Task type: {model_art.task_type}")
    lines.append(f"Dataset: {dataset_art.name}")
    if model_art.metrics:
        lines.append(f"Metrics: {model_art.metrics}")
    lines.append('"""')
    lines.append("")

    # Imports
    lines.append("import numpy as np")
    lines.append("import pandas as pd")
    lines.append("from sklearn.model_selection import train_test_split")
    lines.append("")

    # Data loading
    data_path = dataset_path or dataset_art.source
    if data_path.startswith("openml:"):
        lines.append("import openml")
        ref = data_path[len("openml:"):]
        if ref.isdigit():
            lines.append(f'dataset = openml.datasets.get_dataset({ref})')
        else:
            lines.append(f'# Load OpenML dataset: {ref}')
            lines.append(f'dataset = openml.datasets.get_dataset("{ref}")')
        lines.append("X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)")
        lines.append("df = X.copy()")
        lines.append(f'df["{dataset_art.target_column}"] = y')
    elif data_path == "derived":
        lines.append("# Dataset was derived from another dataset")
        lines.append('# Replace with your data loading code')
        lines.append('df = pd.read_csv("your_data.csv")')
    else:
        if data_path.endswith((".parquet", ".pq")):
            lines.append(f'df = pd.read_parquet("{data_path}")')
        else:
            lines.append(f'df = pd.read_csv("{data_path}")')
    lines.append("")

    # Target column
    tc = dataset_art.target_column
    if tc:
        lines.append(f'TARGET = "{tc}"')
        lines.append("X = df.drop(columns=[TARGET])")
        lines.append("y = df[TARGET]")
        lines.append("")

    # Preprocessing
    if include_preprocessing:
        lines.append("# --- Preprocessing ---")

        # Handle categoricals
        lines.append("cat_cols = X.select_dtypes(include=['object', 'category']).columns")
        lines.append("if len(cat_cols) > 0:")
        lines.append("    from sklearn.preprocessing import LabelEncoder")
        lines.append("    for col in cat_cols:")
        lines.append("        le = LabelEncoder()")
        lines.append("        X[col] = le.fit_transform(X[col].astype(str))")
        lines.append("")

        # Handle missing
        lines.append("if X.isna().any().any():")
        lines.append("    X = X.fillna(X.median(numeric_only=True))")
        lines.append("")

        # Encode target if classification
        if model_art.task_type != "regression":
            lines.append("if y.dtype == 'object':")
            lines.append("    from sklearn.preprocessing import LabelEncoder")
            lines.append("    le_target = LabelEncoder()")
            lines.append("    y = pd.Series(le_target.fit_transform(y), name=y.name)")
            lines.append("")

    # Train/test split
    lines.append("# --- Train/Test Split ---")
    if model_art.task_type != "regression":
        lines.append("X_train, X_test, y_train, y_test = train_test_split(")
        lines.append("    X, y, test_size=0.2, random_state=42, stratify=y")
        lines.append(")")
    else:
        lines.append("X_train, X_test, y_train, y_test = train_test_split(")
        lines.append("    X, y, test_size=0.2, random_state=42")
        lines.append(")")
    lines.append("")

    # Model instantiation
    lines.append("# --- Model ---")
    _add_model_import(lines, model_art)
    lines.append("")

    # Fit and evaluate
    lines.append("# --- Train ---")
    lines.append("model.fit(X_train, y_train)")
    lines.append("")

    lines.append("# --- Evaluate ---")
    if model_art.task_type == "regression":
        lines.append("from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error")
        lines.append("y_pred = model.predict(X_test)")
        lines.append('print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")')
        lines.append('print(f"R2:   {r2_score(y_test, y_pred):.4f}")')
        lines.append('print(f"MAE:  {mean_absolute_error(y_test, y_pred):.4f}")')
    else:
        lines.append("from sklearn.metrics import accuracy_score, classification_report")
        lines.append("y_pred = model.predict(X_test)")
        lines.append('print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")')
        lines.append("print(classification_report(y_test, y_pred))")
    lines.append("")

    # Save model
    lines.append("# --- Save ---")
    lines.append("import endgame as eg")
    lines.append(f'eg.save(model, "trained_{model_art.name}")')
    lines.append("")

    return "\n".join(lines)


def _add_model_import(lines: list[str], model_art: ModelArtifact) -> None:
    """Add model import and instantiation code."""

    model_type = model_art.model_type

    # If it came from automl, generate automl code
    if model_type.startswith("automl_"):
        preset = model_type.replace("automl_", "")
        lines.append("from endgame.automl.tabular import TabularPredictor")
        lines.append(f'predictor = TabularPredictor(label=TARGET, presets="{preset}")')
        lines.append("predictor.fit(pd.concat([X_train, y_train], axis=1))")
        lines.append("model = predictor")
        return

    # Try to get from registry
    try:
        from endgame.automl.model_registry import get_model_info
        info = get_model_info(model_type)

        class_path = info.class_path
        if model_art.task_type == "regression" and "Classifier" in class_path:
            class_path = class_path.replace("Classifier", "Regressor")

        module, cls = class_path.rsplit(".", 1)
        lines.append(f"from {module} import {cls}")

        # Build params string
        params = model_art.params
        # Filter out non-serializable params
        clean_params = {}
        for k, v in params.items():
            if isinstance(v, (int, float, str, bool, type(None), list, tuple)):
                clean_params[k] = v

        if clean_params:
            param_strs = []
            for k, v in clean_params.items():
                param_strs.append(f"    {k}={v!r},")
            lines.append(f"model = {cls}(")
            lines.extend(param_strs)
            lines.append(")")
        else:
            lines.append(f"model = {cls}()")

    except (KeyError, ImportError):
        lines.append(f"# Model type: {model_type}")
        lines.append("# Reconstruct your model here")
        lines.append("from sklearn.ensemble import GradientBoostingClassifier")
        lines.append("model = GradientBoostingClassifier()")
