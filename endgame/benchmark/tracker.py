from __future__ import annotations

"""Experiment tracking for benchmark runs.

Records pipeline configurations, hyperparameters, metrics, and results
for systematic analysis. Supports master database functionality for
meta-learning dataset accumulation.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


# Default master database location
DEFAULT_MASTER_DB_PATH = Path.home() / ".endgame" / "meta_learning_db.parquet"


def get_experiment_hash(
    dataset_name: str,
    model_name: str,
    hyperparameters: dict[str, Any],
    task_type: str = "classification",
) -> str:
    """Generate a unique hash for an experiment configuration.

    This hash is used to detect duplicate experiments in the master database.
    Two experiments are considered duplicates if they have the same:
    - dataset name
    - model name
    - hyperparameters
    - task type

    Parameters
    ----------
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model/pipeline.
    hyperparameters : Dict[str, Any]
        Model hyperparameters.
    task_type : str
        Task type (classification/regression).

    Returns
    -------
    str
        SHA256 hash (first 16 characters) uniquely identifying this config.
    """
    # Create a canonical representation
    config = {
        "dataset": dataset_name,
        "model": model_name,
        "task_type": task_type,
        "hyperparameters": _canonicalize_params(hyperparameters),
    }

    # Convert to JSON string with sorted keys for consistency
    config_str = json.dumps(config, sort_keys=True, default=str)

    # Return first 16 chars of SHA256 hash
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _canonicalize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert parameters to a canonical form for hashing."""
    result = {}
    for key, value in sorted(params.items()):
        if value is None:
            result[key] = None
        elif isinstance(value, (int, float, str, bool)):
            result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = list(value)
        elif isinstance(value, dict):
            result[key] = _canonicalize_params(value)
        else:
            # For objects, use their string representation
            result[key] = str(type(value).__name__)
    return result


@dataclass
class ExperimentRecord:
    """Single experiment record.

    Attributes
    ----------
    experiment_id : str
        Unique identifier for this experiment.
    timestamp : str
        ISO timestamp of when the experiment was run.
    dataset_name : str
        Name of the dataset.
    dataset_id : Optional[str]
        External ID (e.g., OpenML ID).
    model_name : str
        Name/identifier of the model or pipeline.
    pipeline_config : Dict
        Serialized pipeline configuration.
    hyperparameters : Dict
        Model hyperparameters.
    metrics : Dict[str, float]
        Performance metrics.
    meta_features : Dict[str, float]
        Dataset meta-features.
    cv_scores : Optional[List[float]]
        Per-fold CV scores.
    fit_time : float
        Training time in seconds.
    predict_time : float
        Prediction time in seconds.
    memory_mb : float
        Peak memory usage in MB.
    n_samples : int
        Number of training samples.
    n_features : int
        Number of features.
    task_type : str
        Type of task.
    status : str
        Experiment status: "success", "failed", "timeout".
    error_message : Optional[str]
        Error message if failed.
    tags : List[str]
        User-defined tags.
    notes : str
        Additional notes.
    """
    experiment_id: str = ""
    timestamp: str = ""
    dataset_name: str = ""
    dataset_id: str | None = None
    model_name: str = ""
    pipeline_config: dict[str, Any] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    meta_features: dict[str, float] = field(default_factory=dict)
    cv_scores: list[float] | None = None
    fit_time: float = 0.0
    predict_time: float = 0.0
    memory_mb: float = 0.0
    n_samples: int = 0
    n_features: int = 0
    task_type: str = "classification"
    status: str = "pending"
    error_message: str | None = None
    tags: list[str] = field(default_factory=list)
    notes: str = ""
    model_structure: str | None = None  # Human-readable model structure (tree, rules, etc.)
    config_hash: str = ""  # Hash for deduplication in master database

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = str(uuid.uuid4())[:8]
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if not self.config_hash and self.dataset_name and self.model_name:
            self.config_hash = get_experiment_hash(
                self.dataset_name,
                self.model_name,
                self.hyperparameters,
                self.task_type,
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "experiment_id": self.experiment_id,
            "timestamp": self.timestamp,
            "dataset_name": self.dataset_name,
            "dataset_id": self.dataset_id,
            "model_name": self.model_name,
            "pipeline_config": json.dumps(self.pipeline_config),
            "hyperparameters": json.dumps(self.hyperparameters),
            "metrics": self.metrics.copy(),
            "meta_features": self.meta_features.copy(),
            "cv_scores": self.cv_scores,
            "fit_time": self.fit_time,
            "predict_time": self.predict_time,
            "memory_mb": self.memory_mb,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "task_type": self.task_type,
            "status": self.status,
            "error_message": self.error_message,
            "tags": self.tags,
            "notes": self.notes,
            "model_structure": self.model_structure,
            "config_hash": self.config_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentRecord:
        """Create from dictionary."""
        # Parse JSON fields if needed
        if isinstance(data.get("pipeline_config"), str):
            data["pipeline_config"] = json.loads(data["pipeline_config"])
        if isinstance(data.get("hyperparameters"), str):
            data["hyperparameters"] = json.loads(data["hyperparameters"])
        return cls(**data)


class ExperimentTracker:
    """Track and store experiment results.

    Provides methods for logging experiments, querying results, and
    exporting to various formats.

    Parameters
    ----------
    name : str, default="benchmark"
        Name for this tracking session.
    auto_save : bool, default=False
        Automatically save after each experiment.
    save_path : str, optional
        Path for auto-saving results.

    Examples
    --------
    >>> tracker = ExperimentTracker(name="my_benchmark")
    >>> tracker.log_experiment(
    ...     dataset_name="iris",
    ...     model_name="RandomForest",
    ...     metrics={"accuracy": 0.95, "f1": 0.94},
    ...     hyperparameters={"n_estimators": 100},
    ... )
    >>> df = tracker.to_dataframe()
    """

    def __init__(
        self,
        name: str = "benchmark",
        auto_save: bool = False,
        save_path: str | None = None,
    ):
        self.name = name
        self.auto_save = auto_save
        self.save_path = save_path

        self._records: list[ExperimentRecord] = []
        self._start_time = datetime.now()

    def log_experiment(
        self,
        dataset_name: str,
        model_name: str,
        metrics: dict[str, float],
        hyperparameters: dict[str, Any] | None = None,
        pipeline_config: dict[str, Any] | None = None,
        meta_features: dict[str, float] | None = None,
        cv_scores: list[float] | None = None,
        fit_time: float = 0.0,
        predict_time: float = 0.0,
        memory_mb: float = 0.0,
        n_samples: int = 0,
        n_features: int = 0,
        task_type: str = "classification",
        dataset_id: str | None = None,
        status: str = "success",
        error_message: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
        model_structure: str | None = None,
    ) -> ExperimentRecord:
        """Log a single experiment.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        model_name : str
            Name of the model/pipeline.
        metrics : Dict[str, float]
            Performance metrics.
        hyperparameters : Dict, optional
            Model hyperparameters.
        pipeline_config : Dict, optional
            Full pipeline configuration.
        meta_features : Dict, optional
            Dataset meta-features.
        cv_scores : List[float], optional
            Per-fold CV scores.
        fit_time : float
            Training time in seconds.
        predict_time : float
            Prediction time in seconds.
        memory_mb : float
            Peak memory usage in MB.
        n_samples : int
            Number of samples.
        n_features : int
            Number of features.
        task_type : str
            Task type.
        dataset_id : str, optional
            External dataset ID.
        status : str
            Experiment status.
        error_message : str, optional
            Error message if failed.
        tags : List[str], optional
            Tags for filtering.
        notes : str
            Additional notes.

        Returns
        -------
        ExperimentRecord
            The logged experiment record.
        """
        record = ExperimentRecord(
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            model_name=model_name,
            pipeline_config=pipeline_config or {},
            hyperparameters=hyperparameters or {},
            metrics=metrics,
            meta_features=meta_features or {},
            cv_scores=cv_scores,
            fit_time=fit_time,
            predict_time=predict_time,
            memory_mb=memory_mb,
            n_samples=n_samples,
            n_features=n_features,
            task_type=task_type,
            status=status,
            error_message=error_message,
            tags=tags or [],
            notes=notes,
            model_structure=model_structure,
        )

        self._records.append(record)

        if self.auto_save and self.save_path:
            self.save(self.save_path)

        return record

    def log_failure(
        self,
        dataset_name: str,
        model_name: str,
        error_message: str,
        **kwargs,
    ) -> ExperimentRecord:
        """Log a failed experiment."""
        return self.log_experiment(
            dataset_name=dataset_name,
            model_name=model_name,
            metrics={},
            status="failed",
            error_message=error_message,
            **kwargs,
        )

    @property
    def records(self) -> list[ExperimentRecord]:
        """Get all experiment records."""
        return self._records

    def __len__(self) -> int:
        return len(self._records)

    def __iter__(self):
        return iter(self._records)

    def get_by_dataset(self, dataset_name: str) -> list[ExperimentRecord]:
        """Get records for a specific dataset."""
        return [r for r in self._records if r.dataset_name == dataset_name]

    def get_by_model(self, model_name: str) -> list[ExperimentRecord]:
        """Get records for a specific model."""
        return [r for r in self._records if r.model_name == model_name]

    def get_by_tag(self, tag: str) -> list[ExperimentRecord]:
        """Get records with a specific tag."""
        return [r for r in self._records if tag in r.tags]

    def get_successful(self) -> list[ExperimentRecord]:
        """Get successful experiments only."""
        return [r for r in self._records if r.status == "success"]

    def to_dataframe(self, include_meta_features: bool = True):
        """Convert to DataFrame.

        Parameters
        ----------
        include_meta_features : bool, default=True
            Include meta-features as columns.

        Returns
        -------
        DataFrame
            Polars DataFrame (or Pandas if Polars unavailable).
        """
        rows = []

        for record in self._records:
            row = {
                "experiment_id": record.experiment_id,
                "timestamp": record.timestamp,
                "dataset_name": record.dataset_name,
                "dataset_id": record.dataset_id,
                "model_name": record.model_name,
                "pipeline_config": json.dumps(record.pipeline_config),
                "hyperparameters": json.dumps(record.hyperparameters),
                "fit_time": record.fit_time,
                "predict_time": record.predict_time,
                "memory_mb": record.memory_mb,
                "n_samples": record.n_samples,
                "n_features": record.n_features,
                "task_type": record.task_type,
                "status": record.status,
                "error_message": record.error_message,
                "tags": ",".join(record.tags),
                "notes": record.notes,
                "config_hash": record.config_hash,
            }

            # Add metrics as separate columns (ensure float type for consistency)
            for metric_name, value in record.metrics.items():
                # Convert to float to ensure consistent types across records
                try:
                    row[f"metric_{metric_name}"] = float(value) if value is not None else None
                except (TypeError, ValueError):
                    row[f"metric_{metric_name}"] = None

            # Add CV scores summary
            if record.cv_scores:
                row["cv_mean"] = np.mean(record.cv_scores)
                row["cv_std"] = np.std(record.cv_scores)

            # Add meta-features (ensure float type for consistency)
            if include_meta_features:
                for mf_name, value in record.meta_features.items():
                    try:
                        row[f"mf_{mf_name}"] = float(value) if value is not None else None
                    except (TypeError, ValueError):
                        row[f"mf_{mf_name}"] = None

            rows.append(row)

        if not rows:
            # Return empty DataFrame with minimal schema
            if HAS_POLARS:
                return pl.DataFrame()
            elif HAS_PANDAS:
                return pd.DataFrame()
            else:
                raise ImportError("Either polars or pandas is required")

        # Collect all unique columns across all rows to ensure consistent schema
        all_columns = set()
        for row in rows:
            all_columns.update(row.keys())

        # Fill missing columns with None for each row
        for row in rows:
            for col in all_columns:
                if col not in row:
                    row[col] = None

        if HAS_POLARS:
            # Use infer_schema_length=None to check all rows for schema inference
            return pl.DataFrame(rows, infer_schema_length=None)
        elif HAS_PANDAS:
            return pd.DataFrame(rows)
        else:
            raise ImportError("Either polars or pandas is required")

    def to_dict_list(self) -> list[dict[str, Any]]:
        """Convert to list of dictionaries."""
        return [r.to_dict() for r in self._records]

    def save(self, path: str, append: bool = False, deduplicate: bool = True) -> None:
        """Save results to file.

        Parameters
        ----------
        path : str
            Output path. Supports: .parquet, .csv, .json
        append : bool, default=False
            If True and file exists, append new records to existing file.
            If False, overwrite existing file.
        deduplicate : bool, default=True
            When appending, skip records with duplicate config_hash.
        """
        # Handle append mode
        if append and Path(path).exists():
            # Load existing records
            existing = ExperimentTracker(name="existing")
            try:
                existing.load(path)

                # Merge new records into existing
                existing.merge(self, deduplicate=deduplicate)

                # Use merged tracker for saving
                tracker_to_save = existing
            except Exception as e:
                # If loading fails, fall back to overwrite
                print(f"Warning: Could not load existing file for append: {e}. Overwriting instead.")
                tracker_to_save = self
        else:
            tracker_to_save = self

        df = tracker_to_save.to_dataframe()

        if path.endswith(".parquet"):
            if HAS_POLARS:
                df.write_parquet(path)
            else:
                df.to_parquet(path, index=False)
        elif path.endswith(".csv"):
            if HAS_POLARS:
                df.write_csv(path)
            else:
                df.to_csv(path, index=False)
        elif path.endswith(".json"):
            import json
            with open(path, "w") as f:
                json.dump(tracker_to_save.to_dict_list(), f, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def load(self, path: str) -> ExperimentTracker:
        """Load results from file.

        Parameters
        ----------
        path : str
            Input path.

        Returns
        -------
        self
        """
        if path.endswith(".json"):
            import json
            with open(path) as f:
                data = json.load(f)
            self._records = [ExperimentRecord.from_dict(d) for d in data]
        else:
            if HAS_POLARS:
                if path.endswith(".parquet"):
                    df = pl.read_parquet(path)
                else:
                    df = pl.read_csv(path)
                data = df.to_dicts()
            else:
                if path.endswith(".parquet"):
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_csv(path)
                data = df.to_dict(orient="records")

            # Reconstruct records from DataFrame format
            self._records = []
            for row in data:
                record = self._row_to_record(row)
                self._records.append(record)

        return self

    def _row_to_record(self, row: dict[str, Any]) -> ExperimentRecord:
        """Convert DataFrame row to ExperimentRecord."""
        # Extract metrics
        metrics = {}
        meta_features = {}

        for key, value in row.items():
            if key.startswith("metric_") and value is not None:
                metrics[key[7:]] = float(value)
            elif key.startswith("mf_") and value is not None:
                meta_features[key[3:]] = float(value)

        # Parse JSON fields
        pipeline_config = {}
        hyperparameters = {}

        if row.get("pipeline_config"):
            try:
                pipeline_config = json.loads(row["pipeline_config"])
            except (json.JSONDecodeError, TypeError):
                pass

        if row.get("hyperparameters"):
            try:
                hyperparameters = json.loads(row["hyperparameters"])
            except (json.JSONDecodeError, TypeError):
                pass

        # Parse tags
        tags = []
        if row.get("tags"):
            tags = row["tags"].split(",") if isinstance(row["tags"], str) else []

        return ExperimentRecord(
            experiment_id=row.get("experiment_id", ""),
            timestamp=row.get("timestamp", ""),
            dataset_name=row.get("dataset_name", ""),
            dataset_id=row.get("dataset_id"),
            model_name=row.get("model_name", ""),
            pipeline_config=pipeline_config,
            hyperparameters=hyperparameters,
            metrics=metrics,
            meta_features=meta_features,
            fit_time=float(row.get("fit_time", 0)),
            predict_time=float(row.get("predict_time", 0)),
            memory_mb=float(row.get("memory_mb", 0)),
            n_samples=int(row.get("n_samples", 0)),
            n_features=int(row.get("n_features", 0)),
            task_type=row.get("task_type", "classification"),
            status=row.get("status", "unknown"),
            error_message=row.get("error_message"),
            tags=tags,
            notes=row.get("notes", ""),
            config_hash=row.get("config_hash", ""),
        )

    def summary(self) -> str:
        """Get summary of tracked experiments."""
        successful = self.get_successful()

        datasets = set(r.dataset_name for r in self._records)
        models = set(r.model_name for r in self._records)

        lines = [
            f"Experiment Tracker: {self.name}",
            "=" * 50,
            f"Total experiments: {len(self._records)}",
            f"Successful: {len(successful)}",
            f"Failed: {len(self._records) - len(successful)}",
            f"Datasets: {len(datasets)}",
            f"Models: {len(models)}",
        ]

        if successful:
            # Get available metrics
            metrics = set()
            for r in successful:
                metrics.update(r.metrics.keys())

            lines.append(f"\nMetrics tracked: {', '.join(sorted(metrics))}")

            # Best scores per metric
            for metric in sorted(metrics):
                scores = [(r.model_name, r.dataset_name, r.metrics.get(metric, float('-inf')))
                          for r in successful if metric in r.metrics]
                if scores:
                    best = max(scores, key=lambda x: x[2])
                    lines.append(f"Best {metric}: {best[2]:.4f} ({best[0]} on {best[1]})")

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all records."""
        self._records = []

    def get_config_hashes(self) -> set:
        """Get set of all config hashes in the tracker."""
        return {r.config_hash for r in self._records if r.config_hash}

    def merge(
        self,
        other: ExperimentTracker,
        deduplicate: bool = True,
    ) -> ExperimentTracker:
        """Merge another tracker into this one.

        Parameters
        ----------
        other : ExperimentTracker
            Tracker to merge.
        deduplicate : bool, default=True
            Skip records with duplicate config_hash.

        Returns
        -------
        self
        """
        if deduplicate:
            existing_hashes = self.get_config_hashes()
            for record in other._records:
                if record.config_hash not in existing_hashes:
                    self._records.append(record)
                    existing_hashes.add(record.config_hash)
        else:
            self._records.extend(other._records)

        return self

    def save_to_master(
        self,
        path: str | Path | None = None,
        deduplicate: bool = True,
    ) -> int:
        """Save results to master database, appending to existing records.

        This is the primary method for building a meta-learning dataset.
        New experiments are appended to the master database, with duplicate
        configurations (same dataset + model + hyperparameters) skipped.

        Parameters
        ----------
        path : str or Path, optional
            Path to master database. Defaults to ~/.endgame/meta_learning_db.parquet
        deduplicate : bool, default=True
            Skip records with duplicate config_hash.

        Returns
        -------
        int
            Number of new records added.

        Examples
        --------
        >>> tracker = ExperimentTracker()
        >>> # ... run experiments ...
        >>> n_added = tracker.save_to_master()
        >>> print(f"Added {n_added} new experiments to master database")
        """
        if path is None:
            path = DEFAULT_MASTER_DB_PATH
        path = Path(path)

        # Create directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing master database
        existing = ExperimentTracker(name="master_db")
        if path.exists():
            try:
                existing.load(str(path))
            except Exception as e:
                print(f"Warning: Could not load existing master DB: {e}")

        initial_count = len(existing)

        # Merge new records
        existing.merge(self, deduplicate=deduplicate)

        n_added = len(existing) - initial_count

        # Save back to master
        if n_added > 0 or not path.exists():
            existing.save(str(path))

        return n_added

    @classmethod
    def load_master(
        cls,
        path: str | Path | None = None,
    ) -> ExperimentTracker:
        """Load the master meta-learning database.

        Parameters
        ----------
        path : str or Path, optional
            Path to master database. Defaults to ~/.endgame/meta_learning_db.parquet

        Returns
        -------
        ExperimentTracker
            Tracker with all historical experiments.

        Examples
        --------
        >>> tracker = ExperimentTracker.load_master()
        >>> print(f"Master database has {len(tracker)} experiments")
        """
        if path is None:
            path = DEFAULT_MASTER_DB_PATH
        path = Path(path)

        tracker = cls(name="master_db")

        if path.exists():
            tracker.load(str(path))

        return tracker

    @staticmethod
    def get_master_db_path() -> Path:
        """Get the default master database path.

        Returns
        -------
        Path
            Default path: ~/.endgame/meta_learning_db.parquet
        """
        return DEFAULT_MASTER_DB_PATH

    def filter_existing(
        self,
        master_path: str | Path | None = None,
    ) -> list[tuple[str, str]]:
        """Find which (dataset, model) pairs already exist in master DB.

        Useful for skipping already-benchmarked combinations.

        Parameters
        ----------
        master_path : str or Path, optional
            Path to master database.

        Returns
        -------
        List[Tuple[str, str]]
            List of (dataset_name, model_name) pairs that exist.
        """
        master = self.load_master(master_path)
        existing_hashes = master.get_config_hashes()

        existing_pairs = []
        for record in master._records:
            if record.config_hash in existing_hashes:
                existing_pairs.append((record.dataset_name, record.model_name))

        return existing_pairs


def serialize_pipeline(pipeline) -> dict[str, Any]:
    """Serialize a sklearn pipeline to a dictionary.

    Parameters
    ----------
    pipeline : estimator
        Sklearn estimator or pipeline.

    Returns
    -------
    Dict[str, Any]
        Serialized configuration.
    """
    config = {
        "type": type(pipeline).__name__,
        "module": type(pipeline).__module__,
    }

    # Handle sklearn Pipeline
    if hasattr(pipeline, "named_steps"):
        config["steps"] = []
        for name, estimator in pipeline.named_steps.items():
            step_config = {
                "name": name,
                "type": type(estimator).__name__,
                "params": _serialize_params(estimator.get_params()),
            }
            config["steps"].append(step_config)
    else:
        # Single estimator
        config["params"] = _serialize_params(pipeline.get_params())

    return config


def _serialize_params(params: dict[str, Any]) -> dict[str, Any]:
    """Serialize parameters to JSON-compatible format."""
    result = {}
    for key, value in params.items():
        if value is None or isinstance(value, (int, float, str, bool)):
            result[key] = value
        elif isinstance(value, np.ndarray):
            result[key] = value.tolist()
        elif isinstance(value, (list, tuple)):
            result[key] = [_serialize_value(v) for v in value]
        elif isinstance(value, dict):
            result[key] = _serialize_params(value)
        elif hasattr(value, "get_params"):
            # Nested estimator
            result[key] = {
                "type": type(value).__name__,
                "params": _serialize_params(value.get_params()),
            }
        else:
            result[key] = str(value)
    return result


def _serialize_value(value: Any) -> Any:
    """Serialize a single value."""
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif hasattr(value, "get_params"):
        return {"type": type(value).__name__}
    else:
        return str(value)
