"""Stateful session management for the MCP server.

Tracks loaded datasets, trained models, and visualizations across tool
calls using short IDs (e.g., ``ds_a1b2c3d4``, ``model_e5f6g7h8``).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _short_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


@dataclass
class DatasetArtifact:
    id: str
    name: str
    source: str
    df: pd.DataFrame
    target_column: str | None = None
    task_type: str | None = None
    meta_features: dict | None = None
    feature_names: list[str] | None = None
    path: str | None = None  # parquet copy on disk (derived datasets), so the id survives a server restart


@dataclass
class ModelArtifact:
    id: str
    name: str
    model_type: str
    estimator: Any
    dataset_id: str
    task_type: str
    metrics: dict = field(default_factory=dict)
    params: dict = field(default_factory=dict)
    fit_time: float = 0.0
    feature_names: list[str] = field(default_factory=list)
    oof_predictions: np.ndarray | None = None
    label_encoders: dict | None = None
    target_encoder: Any | None = None
    oof_proba: np.ndarray | None = None  # classifiers: out-of-fold class probabilities (shuffled CV)
    cv: dict | None = None               # how the out-of-fold predictions were made (folds, time order)


@dataclass
class VisualizationArtifact:
    id: str
    chart_type: str
    html_path: str
    model_id: str | None = None
    dataset_id: str | None = None


class SessionManager:
    """Manages all artifacts for a single MCP session."""

    def __init__(self) -> None:
        self.datasets: dict[str, DatasetArtifact] = {}
        self.models: dict[str, ModelArtifact] = {}
        self.visualizations: dict[str, VisualizationArtifact] = {}
        self.automl_predictors: dict[str, Any] = {}
        self.code_namespace: dict[str, Any] | None = None  # run_python's variables

        wd = os.environ.get("ENDGAME_MCP_WORKDIR", "/tmp/endgame_mcp")
        self.working_dir = Path(wd)
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.working_dir / "datasets"
        self.data_dir.mkdir(exist_ok=True)
        week_ago = time.time() - 7 * 86400
        for f in self.data_dir.iterdir():
            try:
                if f.stat().st_mtime < week_ago:
                    f.unlink()
            except FileNotFoundError:  # another server pruned it first
                pass

    # -- datasets ---------------------------------------------------------

    def add_dataset(
        self,
        df: pd.DataFrame,
        name: str,
        source: str,
        target_column: str | None = None,
        task_type: str | None = None,
        meta_features: dict | None = None,
        save: bool = True,
    ) -> DatasetArtifact:
        """Register a dataset. ``save=False`` for one read straight from a file: its source is its record."""
        ds_id = _short_id("ds")
        feature_names = [c for c in df.columns if c != target_column]
        art = DatasetArtifact(
            id=ds_id,
            name=name,
            source=source,
            df=df,
            target_column=target_column,
            task_type=task_type,
            meta_features=meta_features,
            feature_names=feature_names,
        )
        self.datasets[ds_id] = art
        if save:
            self.save_dataset(art)
        return art

    # ponytail: tables over 2 GB in memory stay in memory only; the disk here is nearly full
    MAX_SAVE_BYTES = 2_000_000_000

    def save_dataset(self, art: DatasetArtifact) -> None:
        """Write a parquet copy so the dataset id still resolves after the server restarts.

        Oct 10: HALIE's player-filtered tracking table existed only in a server that was restarted mid-goal, so
        later steps and the runbook could not find it.
        """
        path = self.data_dir / f"{art.id}.parquet"
        try:
            if art.df.memory_usage(deep=False).sum() > self.MAX_SAVE_BYTES:
                return
            meta = {k: getattr(art, k) for k in ("name", "source", "target_column", "task_type", "meta_features")}
            path.with_suffix(".json").write_text(json.dumps(meta, default=str))
            tmp = path.with_suffix(".tmp")
            art.df.to_parquet(tmp)
            tmp.replace(path)
            art.path = str(path)
        except Exception as e:  # saving is a convenience; the dataset is still usable in memory
            print(f"endgame: dataset {art.id} not saved to disk: {e}", file=sys.stderr)

    def get_dataset(self, dataset_id: str) -> DatasetArtifact:
        if dataset_id not in self.datasets:
            path = self.data_dir / f"{dataset_id}.parquet"
            if not (re.fullmatch(r"ds_[0-9a-f]{8}", dataset_id) and path.exists()):
                raise KeyError(
                    f"Dataset '{dataset_id}' not found. Available: {list(self.datasets)}. "
                    "Datasets read straight from a file are not kept across server restarts: load_data it again."
                )
            meta = json.loads(path.with_suffix(".json").read_text())
            df = pd.read_parquet(path)
            self.datasets[dataset_id] = DatasetArtifact(
                id=dataset_id, df=df, path=str(path),
                feature_names=[c for c in df.columns if c != meta["target_column"]], **meta,
            )
        return self.datasets[dataset_id]

    # -- models -----------------------------------------------------------

    def add_model(
        self,
        estimator: Any,
        name: str,
        model_type: str,
        dataset_id: str,
        task_type: str,
        metrics: dict | None = None,
        params: dict | None = None,
        fit_time: float = 0.0,
        feature_names: list[str] | None = None,
        oof_predictions: np.ndarray | None = None,
        label_encoders: dict | None = None,
        target_encoder: Any | None = None,
        oof_proba: np.ndarray | None = None,
        cv: dict | None = None,
    ) -> ModelArtifact:
        model_id = _short_id("model")
        art = ModelArtifact(
            id=model_id,
            name=name,
            model_type=model_type,
            estimator=estimator,
            dataset_id=dataset_id,
            task_type=task_type,
            metrics=metrics or {},
            params=params or {},
            fit_time=fit_time,
            feature_names=feature_names or [],
            oof_predictions=oof_predictions,
            label_encoders=label_encoders,
            target_encoder=target_encoder,
            oof_proba=oof_proba,
            cv=cv,
        )
        self.models[model_id] = art
        return art

    def get_model(self, model_id: str) -> ModelArtifact:
        if model_id not in self.models:
            available = list(self.models.keys())
            raise KeyError(
                f"Model '{model_id}' not found. Available: {available}"
            )
        return self.models[model_id]

    # -- visualizations ---------------------------------------------------

    def add_visualization(
        self,
        chart_type: str,
        html_path: str,
        model_id: str | None = None,
        dataset_id: str | None = None,
    ) -> VisualizationArtifact:
        viz_id = _short_id("viz")
        art = VisualizationArtifact(
            id=viz_id,
            chart_type=chart_type,
            html_path=html_path,
            model_id=model_id,
            dataset_id=dataset_id,
        )
        self.visualizations[viz_id] = art
        return art

    # -- summary ----------------------------------------------------------

    def get_state_summary(self) -> dict:
        """Return a lightweight summary of all session artifacts."""
        return {
            "datasets": {
                ds_id: {
                    "name": ds.name,
                    "source": ds.source,
                    "shape": list(ds.df.shape),
                    "target_column": ds.target_column,
                    "task_type": ds.task_type,
                    "path": ds.path,
                }
                for ds_id, ds in self.datasets.items()
            },
            "models": {
                m_id: {
                    "name": m.name,
                    "model_type": m.model_type,
                    "dataset_id": m.dataset_id,
                    "task_type": m.task_type,
                    "metrics": m.metrics,
                    "fit_time": round(m.fit_time, 2),
                }
                for m_id, m in self.models.items()
            },
            "visualizations": {
                v_id: {
                    "chart_type": v.chart_type,
                    "html_path": v.html_path,
                    "model_id": v.model_id,
                    "dataset_id": v.dataset_id,
                }
                for v_id, v in self.visualizations.items()
            },
            "automl_predictors": list(self.automl_predictors.keys()),
        }
