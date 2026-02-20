"""Experiment tracking for Endgame.

Provides pluggable experiment logging backends for tracking
model training params, metrics, and artifacts.

Example
-------
>>> from endgame.tracking import ConsoleLogger
>>> with ConsoleLogger() as logger:
...     logger.log_params({"n_estimators": 100, "lr": 0.01})
...     logger.log_metrics({"accuracy": 0.95, "f1": 0.93})

With MLflow (requires ``pip install endgame-ml[tracking]``):

>>> from endgame.tracking import MLflowLogger
>>> with MLflowLogger(experiment_name="my_experiment") as logger:
...     logger.log_params({"model": "lgbm", "n_estimators": 2000})
...     logger.log_metrics({"roc_auc": 0.934})
"""

from endgame.tracking.base import ExperimentLogger
from endgame.tracking.console_logger import ConsoleLogger


def get_logger(backend: str = "console", **kwargs) -> ExperimentLogger:
    """Factory function to get a logger instance.

    Parameters
    ----------
    backend : str, default="console"
        Logging backend: "console", "mlflow".
    **kwargs
        Arguments passed to the logger constructor.

    Returns
    -------
    ExperimentLogger
        Logger instance.
    """
    if backend == "console":
        return ConsoleLogger(**kwargs)
    elif backend == "mlflow":
        from endgame.tracking.mlflow_logger import MLflowLogger

        return MLflowLogger(**kwargs)
    else:
        raise ValueError(
            f"Unknown backend: {backend!r}. Supported: 'console', 'mlflow'."
        )


__all__ = [
    "ExperimentLogger",
    "ConsoleLogger",
    "MLflowLogger",
    "get_logger",
]


def __getattr__(name: str):
    if name == "MLflowLogger":
        from endgame.tracking.mlflow_logger import MLflowLogger

        return MLflowLogger
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
