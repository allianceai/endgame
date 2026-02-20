"""Lightweight console/file logger with no external dependencies."""

import json
import logging
import time
from pathlib import Path
from typing import Any

from endgame.tracking.base import ExperimentLogger

logger = logging.getLogger(__name__)


class ConsoleLogger(ExperimentLogger):
    """Simple console/file logger with no external dependencies.

    Prints experiment tracking information to the console and optionally
    writes a JSON log file. Useful for lightweight tracking without MLflow.

    Parameters
    ----------
    log_file : str or Path, optional
        Path to a JSON log file. If provided, all events are appended.
    verbose : bool, default=True
        Whether to print to console.

    Examples
    --------
    >>> logger = ConsoleLogger()
    >>> with logger:
    ...     logger.log_params({"lr": 0.01, "epochs": 10})
    ...     logger.log_metrics({"accuracy": 0.95})

    With file logging:

    >>> logger = ConsoleLogger(log_file="experiment_log.json")
    >>> logger.start_run("my_experiment")
    >>> logger.log_metrics({"f1": 0.92})
    >>> logger.end_run()
    """

    def __init__(
        self,
        log_file: str | Path | None = None,
        verbose: bool = True,
    ):
        self.log_file = Path(log_file) if log_file else None
        self.verbose = verbose

        self._run_id: str | None = None
        self._run_name: str | None = None
        self._start_time: float | None = None
        self._params: dict[str, Any] = {}
        self._metrics: list[dict[str, Any]] = []
        self._artifacts: list[str] = []
        self._experiment_name: str = "default"

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a new run."""
        self._run_id = f"console-{int(time.time() * 1000)}"
        self._run_name = run_name or self._run_id
        self._start_time = time.time()
        self._params = {}
        self._metrics = []
        self._artifacts = []

        if self.verbose:
            print(f"[Tracking] Run started: {self._run_name}")
            if tags:
                print(f"[Tracking] Tags: {tags}")

        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        duration = time.time() - self._start_time if self._start_time else 0

        if self.verbose:
            print(f"[Tracking] Run ended: {self._run_name} ({status}, {duration:.1f}s)")

        # Write to log file if configured
        if self.log_file:
            entry = {
                "run_id": self._run_id,
                "run_name": self._run_name,
                "experiment": self._experiment_name,
                "status": status,
                "duration": round(duration, 2),
                "params": self._params,
                "metrics": self._metrics,
                "artifacts": self._artifacts,
            }
            self.log_file.parent.mkdir(parents=True, exist_ok=True)

            # Append to JSON lines file
            with open(self.log_file, "a") as f:
                f.write(json.dumps(entry) + "\n")

        self._run_id = None
        self._start_time = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters."""
        self._params.update(params)

        if self.verbose:
            for k, v in params.items():
                print(f"[Tracking]   param {k}={v}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics."""
        entry = {**metrics}
        if step is not None:
            entry["_step"] = step
        self._metrics.append(entry)

        if self.verbose:
            step_str = f" (step={step})" if step is not None else ""
            for k, v in metrics.items():
                print(f"[Tracking]   metric {k}={v:.4f}{step_str}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact path."""
        self._artifacts.append(local_path)

        if self.verbose:
            dest = f" -> {artifact_path}" if artifact_path else ""
            print(f"[Tracking]   artifact {local_path}{dest}")

    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        """Log a model (records type name only)."""
        model_type = type(model).__name__
        self._artifacts.append(f"{artifact_path}/{model_type}")

        if self.verbose:
            print(f"[Tracking]   model {model_type} -> {artifact_path}")

    def set_experiment(self, name: str) -> None:
        """Set the active experiment name."""
        self._experiment_name = name

        if self.verbose:
            print(f"[Tracking] Experiment: {name}")

    def __repr__(self) -> str:
        return f"ConsoleLogger(log_file={self.log_file!r}, verbose={self.verbose})"
