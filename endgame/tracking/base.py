from __future__ import annotations

"""Abstract base class for experiment loggers."""

from abc import ABC, abstractmethod
from typing import Any


class ExperimentLogger(ABC):
    """Abstract base for experiment logging backends.

    All loggers implement this interface, enabling consistent
    tracking across MLflow, console, or custom backends.

    Supports context manager usage::

        with MyLogger() as logger:
            logger.log_params({"lr": 0.01})
            logger.log_metrics({"acc": 0.95})
    """

    @abstractmethod
    def start_run(self, run_name: str | None = None, tags: dict[str, str] | None = None) -> str:
        """Start a new tracking run.

        Parameters
        ----------
        run_name : str, optional
            Human-readable name for this run.
        tags : dict, optional
            Tags to associate with the run.

        Returns
        -------
        str
            Run identifier.
        """

    @abstractmethod
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run.

        Parameters
        ----------
        status : str, default="FINISHED"
            Final status: "FINISHED", "FAILED", "KILLED".
        """

    @abstractmethod
    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters for the current run.

        Parameters
        ----------
        params : dict
            Parameter key-value pairs. Nested dicts are flattened
            with dot notation (e.g., ``{"model.lr": 0.01}``).
        """

    @abstractmethod
    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics for the current run.

        Parameters
        ----------
        metrics : dict
            Metric key-value pairs.
        step : int, optional
            Step number for time-series metrics (e.g., epoch).
        """

    @abstractmethod
    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact.

        Parameters
        ----------
        local_path : str
            Path to the local file.
        artifact_path : str, optional
            Subdirectory within the artifact store.
        """

    @abstractmethod
    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        """Log a trained model as an artifact.

        Parameters
        ----------
        model : Any
            Trained model (sklearn, pytorch, etc.).
        artifact_path : str, default="model"
            Subdirectory for the model artifact.
        **kwargs
            Backend-specific options.
        """

    @abstractmethod
    def set_experiment(self, name: str) -> None:
        """Set the active experiment.

        Parameters
        ----------
        name : str
            Experiment name.
        """

    def __enter__(self) -> ExperimentLogger:
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        status = "FAILED" if exc_type is not None else "FINISHED"
        self.end_run(status=status)
