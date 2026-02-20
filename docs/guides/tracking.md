# Experiment Tracking

Endgame provides pluggable experiment tracking through the `endgame.tracking`
module. Track model parameters, metrics, and artifacts using MLflow or a
lightweight console logger.

---

## Quick Start

```python
from endgame.tracking import ConsoleLogger

with ConsoleLogger() as logger:
    logger.log_params({"model": "lgbm", "n_estimators": 2000})
    logger.log_metrics({"roc_auc": 0.934, "accuracy": 0.91})
```

---

## MLflow Integration

Install the tracking extra:

```bash
pip install endgame-ml[tracking]
```

```python
from endgame.tracking import MLflowLogger

with MLflowLogger(experiment_name="my_project") as logger:
    logger.log_params({"model": "lgbm", "lr": 0.05})
    logger.log_metrics({"roc_auc": 0.934})
    logger.log_model(fitted_model, "best_model")
```

View results in the MLflow UI:

```bash
mlflow ui
# Open http://localhost:5000
```

---

## AutoML Integration

Pass a logger to `TabularPredictor` to automatically track training:

```python
from endgame.automl import TabularPredictor
from endgame.tracking import MLflowLogger

logger = MLflowLogger(experiment_name="automl_runs")

with logger:
    predictor = TabularPredictor(label="target", logger=logger)
    predictor.fit(train_df)
```

Parameters (preset, time limit, data shape) and metrics (best score, CV
score, training time) are logged automatically.

---

## Quick API Integration

The `classify()`, `regress()`, and `compare()` functions also accept a logger:

```python
from endgame.quick import classify
from endgame.tracking import ConsoleLogger

with ConsoleLogger() as logger:
    result = classify(X, y, logger=logger)
```

---

## Console Logger

For lightweight tracking without external dependencies:

```python
from endgame.tracking import ConsoleLogger

# Print to console
logger = ConsoleLogger()

# Log to file (JSON lines format)
logger = ConsoleLogger(log_file="experiments.jsonl")

# Silent (file only)
logger = ConsoleLogger(log_file="experiments.jsonl", verbose=False)
```

---

## Factory Function

Use `get_logger()` for backend-agnostic code:

```python
from endgame.tracking import get_logger

# Defaults to console
logger = get_logger("console")

# Switch to MLflow
logger = get_logger("mlflow", experiment_name="my_project")
```

---

## Custom Logger

Implement the `ExperimentLogger` interface for custom backends
(e.g., Weights & Biases, Neptune):

```python
from endgame.tracking.base import ExperimentLogger

class WandbLogger(ExperimentLogger):
    def start_run(self, run_name=None, tags=None):
        import wandb
        self._run = wandb.init(project="endgame", name=run_name, tags=tags)
        return self._run.id

    def end_run(self, status="FINISHED"):
        self._run.finish()

    def log_params(self, params):
        import wandb
        wandb.config.update(params)

    def log_metrics(self, metrics, step=None):
        import wandb
        wandb.log(metrics, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        import wandb
        wandb.save(local_path)

    def log_model(self, model, artifact_path="model", **kwargs):
        pass  # W&B handles models via artifacts

    def set_experiment(self, name):
        pass  # W&B uses project names
```

---

## API Reference

| Class | Description |
|---|---|
| `ExperimentLogger` | Abstract base class |
| `ConsoleLogger` | Print/file logger (no dependencies) |
| `MLflowLogger` | MLflow backend |
| `get_logger()` | Factory function |
