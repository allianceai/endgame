"""Standalone Python script generator.

Given a ModelArtifact and DatasetArtifact, produces a self-contained script
that reproduces the trained pipeline: the same rows, features, encoding,
model and cross-validation the MCP server used, so it prints the same metrics.
"""

from __future__ import annotations

import ast
from pathlib import Path

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
    lines.append("")

    # Data loading: the named file, else the dataset's own source file, else the server's saved copy of a derived one
    src = dataset_art.source or ""
    is_file = src.startswith(("openml:", "http://", "https://")) or Path(src).is_file()
    data_path = dataset_path or (src if is_file else dataset_art.path)
    if data_path is None:
        lines.append(f"# Dataset {dataset_art.name!r} was built in the MCP session and not saved to disk")
        lines.append('# Replace with your data loading code')
        lines.append('df = pd.read_csv("your_data.csv")')
    elif data_path.startswith("openml:"):
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
    elif data_path.endswith((".parquet", ".pq")):
        lines.append(f'df = pd.read_parquet("{data_path}")')
    else:
        lines.append(f'df = pd.read_csv("{data_path}")')
    lines.append("")

    # Target and features: the columns the model was trained on
    tc = dataset_art.target_column
    cv = model_art.cv
    group = (cv or {}).get("group_column")
    if tc:
        lines.append(f'TARGET = "{tc}"')
        if group:
            lines.append(f'groups = df["{group}"].to_numpy()  # folds keep each {group} together')
        names = set(model_art.feature_names or [])
        left_out = [c for c in dataset_art.df.columns if names and c != tc and c not in names]
        lines.append(f"X = df.drop(columns=[TARGET] + {left_out!r})")
        lines.append("y = df[TARGET]")
        lines.append("")

    # Preprocessing, as the MCP server did it
    if include_preprocessing:
        lines.append("# --- Preprocessing ---")
        lines.append("from sklearn.preprocessing import LabelEncoder")
        lines.append("for col in X.select_dtypes(include=['object', 'category', 'string']).columns:")
        lines.append("    X[col] = LabelEncoder().fit_transform(X[col].astype(str))")
        if not _handles_missing(model_art.model_type):
            lines.append("X = X.fillna(X.median(numeric_only=True))  # this model does not take missing values")
        if model_art.task_type != "regression":
            lines.append("if y.dtype == 'object' or isinstance(y.dtype, pd.CategoricalDtype):")
            lines.append("    y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), name=y.name)")
        lines.append("")

    if model_art.model_type.startswith("automl_"):
        _add_holdout(lines, model_art, automl=True)
    else:
        lines.append("# --- Model ---")
        _add_model_import(lines, model_art)
        lines.append("")
        if cv:
            _add_cross_validation(lines, model_art)
        else:
            _add_holdout(lines, model_art)

    # Save model
    lines.append("# --- Save ---")
    lines.append("import endgame as eg")
    lines.append(f'eg.save(model, "trained_{model_art.name}")')
    lines.append("")

    return "\n".join(lines)


def _handles_missing(model_type: str) -> bool:
    try:
        from endgame.automl.model_registry import get_model_info
        return bool(get_model_info(model_type).handles_missing)
    except (KeyError, ImportError):
        return False


def _literal(value):
    """``value`` as something whose repr is valid Python, or None."""
    if hasattr(value, "item") and not hasattr(value, "__len__"):
        value = value.item()  # numpy scalar
    try:
        return value if ast.literal_eval(repr(value)) == value else None
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        return None


def _add_model_import(lines: list[str], model_art: ModelArtifact) -> None:
    """Build the model the way the MCP server did: a registry key, or the trained estimator's own class."""
    params = {}
    for k, v in (model_art.params or {}).items():
        lit = _literal(v)
        if "__" not in k and (lit is not None or v is None):
            params[k] = lit
    lines.append("PARAMS = {")
    lines.extend(f"    {k!r}: {v!r}," for k, v in params.items())
    lines.append("}")

    from endgame.automl.model_registry import MODEL_REGISTRY
    if model_art.model_type in MODEL_REGISTRY:
        lines.append("from endgame.automl.model_registry import instantiate_model")
        lines.append(
            f"model = instantiate_model({model_art.model_type!r}, task_type={model_art.task_type!r}, **PARAMS)")
        return

    cls = type(model_art.estimator)
    if "." in model_art.model_type:
        module, name = model_art.model_type.rsplit(".", 1)
    else:  # an estimator handed over from run_python
        module, name = cls.__module__, cls.__qualname__
    lines.append(f"from {module} import {name}")
    lines.append(f"model = {name}(**PARAMS)")


def _add_cross_validation(lines: list[str], model_art: ModelArtifact) -> None:
    """The same folds as train_model/compare_models, so the printed metrics match the reported ones."""
    from endgame.mcp.tools.train import _splitter

    cv = model_art.cv
    group = cv.get("group_column")
    splitter = _splitter(model_art.task_type, cv["folds"], cv.get("time_ordered"), True if group else None)
    lines.append("# --- Cross-validation (how the reported metrics were measured) ---")
    lines.append(f"from sklearn.model_selection import {type(splitter).__name__}")
    lines.append(f"cv = {splitter!r}")
    if cv.get("time_ordered"):
        lines.append("from sklearn.base import clone")
        lines.append("scored, oof = [], []")
        lines.append("for train_idx, test_idx in cv.split(X):")
        lines.append("    fold = clone(model).fit(X.iloc[train_idx], y.iloc[train_idx])")
        lines.append("    oof.append(fold.predict(X.iloc[test_idx]))")
        lines.append("    scored.append(test_idx)")
        lines.append("y_scored, oof = y.iloc[np.concatenate(scored)], np.concatenate(oof)  # earliest rows: never scored")
    else:
        lines.append("from sklearn.model_selection import cross_val_predict")
        groups = ", groups=groups" if group else ""
        lines.append(f"oof = cross_val_predict(model, X, y, cv=cv{groups})")
        lines.append("y_scored = y")
    _add_metrics(lines, model_art, "y_scored", "oof")
    lines.append("")
    lines.append("# --- Final model on every row ---")
    lines.append("model.fit(X, y)")
    lines.append("")


def _add_holdout(lines: list[str], model_art: ModelArtifact, automl: bool = False) -> None:
    """An 80/20 split, for models the server did not cross-validate (AutoML, ensembles, run_python)."""
    lines.append("# --- Train/Test Split ---")
    lines.append("from sklearn.model_selection import train_test_split")
    stratify = ", stratify=y" if model_art.task_type != "regression" else ""
    lines.append("X_train, X_test, y_train, y_test = train_test_split(")
    lines.append(f"    X, y, test_size=0.2, random_state=42{stratify}")
    lines.append(")")
    lines.append("")
    if automl:
        preset = model_art.model_type.replace("automl_", "")
        lines.append("from endgame.automl.tabular import TabularPredictor")
        lines.append(f'predictor = TabularPredictor(label=TARGET, presets="{preset}")')
        lines.append("predictor.fit(pd.concat([X_train, y_train], axis=1))")
        lines.append("model = predictor")
    else:
        lines.append("# --- Train ---")
        lines.append("model.fit(X_train, y_train)")
    lines.append("")
    lines.append("# --- Evaluate ---")
    lines.append("y_pred = model.predict(X_test)")
    _add_metrics(lines, model_art, "y_test", "y_pred")
    lines.append("")


def _add_metrics(lines: list[str], model_art: ModelArtifact, y_true: str, y_pred: str) -> None:
    if model_art.task_type == "regression":
        lines.append("from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score")
        lines.append(f'print(f"RMSE: {{np.sqrt(mean_squared_error({y_true}, {y_pred})):.4f}}")')
        lines.append(f'print(f"R2:   {{r2_score({y_true}, {y_pred}):.4f}}")')
        lines.append(f'print(f"MAE:  {{mean_absolute_error({y_true}, {y_pred}):.4f}}")')
    else:
        lines.append("from sklearn.metrics import accuracy_score, classification_report, f1_score")
        lines.append(f'print(f"Accuracy: {{accuracy_score({y_true}, {y_pred}):.4f}}")')
        lines.append(f'print(f"F1 (weighted): {{f1_score({y_true}, {y_pred}, average=\'weighted\'):.4f}}")')
        lines.append(f"print(classification_report({y_true}, {y_pred}))")
