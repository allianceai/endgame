"""Shared encoding utilities for MCP tools.

Ensures that categorical features and target columns are encoded consistently
across train, evaluate, predict, and visualize operations using the encoders
fitted during training.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def encode_features(
    X: pd.DataFrame,
    label_encoders: dict | None = None,
) -> pd.DataFrame:
    """Encode categorical features using stored encoders from training.

    If ``label_encoders`` is provided (from a trained ModelArtifact), uses
    those fitted encoders for consistent mapping.  If ``None``, the model
    handles categoricals natively — features are returned unchanged.

    Unseen categories are mapped to ``-1``.
    """
    if label_encoders is None:
        # No encoders stored — model handles categoricals natively (e.g. LightGBM, CatBoost)
        return X

    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return X

    X = X.copy()

    for col in cat_cols:
        if col in label_encoders:
            le = label_encoders[col]
            known = set(le.classes_)
            X[col] = X[col].astype(str).map(
                lambda v, _known=known, _le=le: (
                    int(_le.transform([v])[0]) if v in _known else -1
                )
            )
        else:
            # Column wasn't encoded during training — fit a new one
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    return X


def encode_target(
    y: pd.Series | np.ndarray,
    target_encoder: Any | None = None,
) -> tuple[np.ndarray, Any]:
    """Encode target column using a stored encoder from training.

    Returns ``(encoded_y, encoder)``.  If the target is already numeric,
    returns it unchanged with ``encoder=None``.
    """
    if isinstance(y, pd.Series):
        vals = y.values
    else:
        vals = np.asarray(y)

    # Handle pandas Categorical — convert to object array for encoding
    if hasattr(vals, "categories"):
        vals = np.array(vals, dtype=object)

    if vals.dtype == object:
        if target_encoder is not None:
            known = set(target_encoder.classes_)
            encoded = np.array([
                int(target_encoder.transform([v])[0]) if v in known else -1
                for v in vals
            ])
            return encoded, target_encoder
        else:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            return le.fit_transform(vals), le

    return vals, target_encoder


def fit_feature_encoders(X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fit new label encoders on categorical features during training.

    Returns ``(encoded_X, encoders_dict)`` where ``encoders_dict`` maps
    column names to fitted ``LabelEncoder`` instances.
    """
    cat_cols = X.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        return X, {}

    from sklearn.preprocessing import LabelEncoder

    X = X.copy()
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        encoders[col] = le

    return X, encoders


def apply_encoders(
    X: pd.DataFrame,
    y: pd.Series | np.ndarray | None,
    model_art: Any,
    fill_missing: bool = True,
) -> tuple[pd.DataFrame, np.ndarray | None]:
    """Apply stored encoders from a ModelArtifact to features and target.

    Convenience function that combines ``encode_features``, ``encode_target``,
    and missing value imputation in a single call.
    """
    X = encode_features(X, label_encoders=model_art.label_encoders)

    if fill_missing and X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))
        for col in X.select_dtypes(include=["object", "category"]).columns:
            mode = X[col].mode()
            X[col] = X[col].fillna(mode.iloc[0] if not mode.empty else "missing")

    y_encoded = None
    if y is not None:
        if model_art.task_type != "regression":
            y_encoded, _ = encode_target(y, target_encoder=model_art.target_encoder)
        else:
            y_encoded = y.values if isinstance(y, pd.Series) else np.asarray(y)
    return X, y_encoded
