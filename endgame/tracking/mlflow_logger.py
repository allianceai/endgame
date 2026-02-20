"""MLflow-backed experiment logger."""

import logging
from typing import Any

from endgame.tracking.base import ExperimentLogger

logger = logging.getLogger(__name__)


def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten a nested dict with dot-separated keys."""
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _get_mlflow():
    """Lazily import mlflow, raising a clear error if missing."""
    try:
        import mlflow

        return mlflow
    except ImportError:
        raise ImportError(
            "mlflow is required for MLflowLogger. "
            "Install it with: pip install endgame-ml[tracking]"
        )


class MLflowLogger(ExperimentLogger):
    """MLflow-backed experiment logger.

    Parameters
    ----------
    tracking_uri : str, default="mlruns"
        MLflow tracking server URI or local directory.
    experiment_name : str, default="endgame"
        Default experiment name.
    auto_log : bool, default=False
        Whether to enable MLflow autologging for sklearn.

    Examples
    --------
    >>> logger = MLflowLogger(experiment_name="my_project")
    >>> with logger:
    ...     logger.log_params({"model": "lgbm", "n_estimators": 2000})
    ...     logger.log_metrics({"roc_auc": 0.934})
    """

    def __init__(
        self,
        tracking_uri: str = "mlruns",
        experiment_name: str = "endgame",
        auto_log: bool = False,
    ):
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        self.auto_log = auto_log
        self._run_id: str | None = None

        # Initialize MLflow
        mlflow = _get_mlflow()
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

        if auto_log:
            mlflow.sklearn.autolog(log_models=False)

    def start_run(
        self,
        run_name: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> str:
        """Start a new MLflow run."""
        mlflow = _get_mlflow()

        run = mlflow.start_run(run_name=run_name, tags=tags)
        self._run_id = run.info.run_id
        return self._run_id

    def end_run(self, status: str = "FINISHED") -> None:
        """End the current MLflow run."""
        mlflow = _get_mlflow()
        mlflow.end_run(status=status)
        self._run_id = None

    def log_params(self, params: dict[str, Any]) -> None:
        """Log parameters, flattening nested dicts."""
        mlflow = _get_mlflow()
        flat = _flatten_dict(params)

        # MLflow has a 500-char limit on param values
        sanitized = {}
        for k, v in flat.items():
            v_str = str(v)
            if len(v_str) > 500:
                v_str = v_str[:497] + "..."
            sanitized[k] = v_str

        mlflow.log_params(sanitized)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics with optional step."""
        mlflow = _get_mlflow()
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log a local file as an artifact."""
        mlflow = _get_mlflow()
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def log_model(self, model: Any, artifact_path: str = "model", **kwargs) -> None:
        """Log a model artifact.

        Automatically selects the appropriate MLflow model flavor:
        - ``mlflow.sklearn`` for sklearn-compatible models
        - ``mlflow.pytorch`` for PyTorch modules
        - Falls back to ``mlflow.pyfunc`` otherwise
        """
        mlflow = _get_mlflow()

        # Try sklearn flavor first (covers most endgame models)
        try:
            from sklearn.base import BaseEstimator

            if isinstance(model, BaseEstimator):
                mlflow.sklearn.log_model(model, artifact_path=artifact_path, **kwargs)
                return
        except ImportError:
            pass

        # Try PyTorch flavor
        try:
            import torch

            if isinstance(model, torch.nn.Module):
                mlflow.pytorch.log_model(model, artifact_path=artifact_path, **kwargs)
                return
        except ImportError:
            pass

        # Fallback: pickle-based pyfunc
        logger.warning(
            f"No specific MLflow flavor for {type(model).__name__}, "
            "falling back to pickle artifact."
        )
        import pickle
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmpdir:
            model_path = Path(tmpdir) / "model.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            mlflow.log_artifact(str(model_path), artifact_path=artifact_path)

    def set_experiment(self, name: str) -> None:
        """Set the active MLflow experiment."""
        mlflow = _get_mlflow()
        self.experiment_name = name
        mlflow.set_experiment(name)

    def __repr__(self) -> str:
        return (
            f"MLflowLogger(tracking_uri={self.tracking_uri!r}, "
            f"experiment_name={self.experiment_name!r})"
        )
