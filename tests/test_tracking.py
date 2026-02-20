"""Tests for the tracking module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from endgame.tracking.base import ExperimentLogger
from endgame.tracking.console_logger import ConsoleLogger


class TestExperimentLogger:
    """Tests for the abstract ExperimentLogger interface."""

    def test_abstract_cannot_instantiate(self):
        """Test that ExperimentLogger cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ExperimentLogger()

    def test_subclass_must_implement_methods(self):
        """Test that subclass must implement all abstract methods."""
        class IncompleteLogger(ExperimentLogger):
            pass

        with pytest.raises(TypeError):
            IncompleteLogger()


class TestConsoleLogger:
    """Tests for ConsoleLogger."""

    def test_start_end_run(self, capsys):
        """Test basic run lifecycle."""
        logger = ConsoleLogger()
        run_id = logger.start_run("test_run")
        assert run_id is not None
        assert "test_run" in capsys.readouterr().out

        logger.end_run()
        output = capsys.readouterr().out
        assert "FINISHED" in output

    def test_log_params(self, capsys):
        """Test parameter logging."""
        logger = ConsoleLogger()
        logger.start_run()
        logger.log_params({"lr": 0.01, "epochs": 10})
        output = capsys.readouterr().out
        assert "lr" in output
        assert "0.01" in output
        logger.end_run()

    def test_log_metrics(self, capsys):
        """Test metric logging."""
        logger = ConsoleLogger()
        logger.start_run()
        logger.log_metrics({"accuracy": 0.95, "f1": 0.93})
        output = capsys.readouterr().out
        assert "accuracy" in output
        assert "0.9500" in output
        logger.end_run()

    def test_log_metrics_with_step(self, capsys):
        """Test metric logging with step number."""
        logger = ConsoleLogger()
        logger.start_run()
        logger.log_metrics({"loss": 0.5}, step=1)
        output = capsys.readouterr().out
        assert "step=1" in output
        logger.end_run()

    def test_log_artifact(self, capsys):
        """Test artifact logging."""
        logger = ConsoleLogger()
        logger.start_run()
        logger.log_artifact("/tmp/model.pkl", "models")
        output = capsys.readouterr().out
        assert "/tmp/model.pkl" in output
        logger.end_run()

    def test_log_model(self, capsys):
        """Test model logging."""
        logger = ConsoleLogger()
        logger.start_run()
        model = LinearRegression()
        logger.log_model(model, "my_model")
        output = capsys.readouterr().out
        assert "LinearRegression" in output
        logger.end_run()

    def test_set_experiment(self, capsys):
        """Test experiment name setting."""
        logger = ConsoleLogger()
        logger.set_experiment("my_experiment")
        output = capsys.readouterr().out
        assert "my_experiment" in output

    def test_context_manager(self, capsys):
        """Test context manager usage."""
        with ConsoleLogger() as logger:
            logger.log_params({"key": "value"})
            logger.log_metrics({"score": 0.9})
        output = capsys.readouterr().out
        assert "FINISHED" in output

    def test_context_manager_on_exception(self, capsys):
        """Test context manager marks run as FAILED on exception."""
        try:
            with ConsoleLogger() as logger:
                logger.log_params({"key": "value"})
                raise ValueError("test error")
        except ValueError:
            pass
        output = capsys.readouterr().out
        assert "FAILED" in output

    def test_file_logging(self):
        """Test logging to a JSON file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "log.jsonl"
            logger = ConsoleLogger(log_file=str(log_file), verbose=False)

            logger.start_run("file_test")
            logger.log_params({"lr": 0.01})
            logger.log_metrics({"acc": 0.95})
            logger.end_run()

            assert log_file.exists()
            with open(log_file) as f:
                entry = json.loads(f.readline())

            assert entry["run_name"] == "file_test"
            assert entry["params"]["lr"] == 0.01
            assert entry["metrics"][0]["acc"] == 0.95
            assert entry["status"] == "FINISHED"

    def test_silent_mode(self, capsys):
        """Test that verbose=False suppresses output."""
        logger = ConsoleLogger(verbose=False)
        logger.start_run("silent_run")
        logger.log_params({"key": "value"})
        logger.log_metrics({"score": 0.5})
        logger.end_run()
        output = capsys.readouterr().out
        assert output == ""

    def test_repr(self):
        """Test __repr__."""
        logger = ConsoleLogger(verbose=False)
        assert "ConsoleLogger" in repr(logger)


class TestMLflowLogger:
    """Tests for MLflowLogger (skipped if mlflow not installed)."""

    @pytest.fixture(autouse=True)
    def check_mlflow(self):
        """Skip all tests if mlflow is not installed."""
        pytest.importorskip("mlflow")

    def test_import(self):
        """Test that MLflowLogger can be imported."""
        from endgame.tracking.mlflow_logger import MLflowLogger
        assert MLflowLogger is not None

    def test_start_end_run(self, tmp_path):
        """Test MLflow run lifecycle."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        run_id = logger.start_run("test_run")
        assert run_id is not None
        logger.end_run()

    def test_log_params_flat(self, tmp_path):
        """Test flat parameter logging."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        logger.start_run()
        logger.log_params({"lr": 0.01, "epochs": 10})
        logger.end_run()

    def test_log_params_nested(self, tmp_path):
        """Test nested parameter logging (flattened with dots)."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        logger.start_run()
        logger.log_params({"model": {"n_estimators": 100, "lr": 0.01}})
        logger.end_run()

    def test_log_metrics_with_step(self, tmp_path):
        """Test metric logging with step."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        logger.start_run()
        logger.log_metrics({"loss": 0.5}, step=1)
        logger.log_metrics({"loss": 0.3}, step=2)
        logger.end_run()

    def test_log_model_sklearn(self, tmp_path):
        """Test logging an sklearn model."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        logger.start_run()
        model = LinearRegression()
        model.fit(np.array([[1], [2], [3]]), np.array([1, 2, 3]))
        logger.log_model(model, "my_model")
        logger.end_run()

    def test_set_experiment(self, tmp_path):
        """Test experiment name setting."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="initial",
        )
        logger.set_experiment("updated")
        assert logger.experiment_name == "updated"

    def test_context_manager(self, tmp_path):
        """Test context manager usage."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        with MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        ) as logger:
            logger.log_params({"key": "value"})

    def test_repr(self, tmp_path):
        """Test __repr__."""
        from endgame.tracking.mlflow_logger import MLflowLogger

        logger = MLflowLogger(
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        assert "MLflowLogger" in repr(logger)


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def test_get_console_logger(self):
        """Test getting a console logger."""
        from endgame.tracking import get_logger

        logger = get_logger("console")
        assert isinstance(logger, ConsoleLogger)

    def test_get_console_logger_with_params(self):
        """Test getting a console logger with parameters."""
        from endgame.tracking import get_logger

        logger = get_logger("console", verbose=False)
        assert isinstance(logger, ConsoleLogger)
        assert not logger.verbose

    def test_get_mlflow_logger(self, tmp_path):
        """Test getting an MLflow logger (skip if mlflow not installed)."""
        pytest.importorskip("mlflow")
        from endgame.tracking import get_logger

        logger = get_logger(
            "mlflow",
            tracking_uri=str(tmp_path / "mlruns"),
            experiment_name="test",
        )
        from endgame.tracking.mlflow_logger import MLflowLogger
        assert isinstance(logger, MLflowLogger)

    def test_unknown_backend(self):
        """Test that unknown backend raises ValueError."""
        from endgame.tracking import get_logger

        with pytest.raises(ValueError, match="Unknown backend"):
            get_logger("unknown")


class TestAutoMLIntegration:
    """Tests for logger integration with AutoML predictors."""

    def test_tabular_predictor_accepts_logger(self):
        """Test that TabularPredictor accepts a logger parameter."""
        from endgame.automl import TabularPredictor

        logger = ConsoleLogger(verbose=False)
        predictor = TabularPredictor(label="target", logger=logger, verbosity=0)
        assert predictor.logger is logger

    def test_tabular_predictor_no_logger_default(self):
        """Test that TabularPredictor defaults to no logger."""
        from endgame.automl import TabularPredictor

        predictor = TabularPredictor(label="target", verbosity=0)
        assert predictor.logger is None

    def test_base_predictor_logger_param(self):
        """Test that BasePredictor stores logger."""
        from endgame.automl import BasePredictor

        logger = ConsoleLogger(verbose=False)

        # BasePredictor is abstract, but we can check via TabularPredictor
        from endgame.automl import TabularPredictor
        predictor = TabularPredictor(label="y", logger=logger, verbosity=0)
        assert predictor.logger is logger

    def test_multimodal_predictor_accepts_logger(self):
        """Test that MultiModalPredictor accepts a logger parameter."""
        from endgame.automl import MultiModalPredictor

        logger = ConsoleLogger(verbose=False)
        predictor = MultiModalPredictor(label="target", logger=logger, verbosity=0)
        assert predictor.logger is logger


class TestRefitFull:
    """Tests for refit_full() method."""

    def test_refit_full_exists_on_tabular(self):
        """Test that refit_full() exists on TabularPredictor."""
        from endgame.automl import TabularPredictor

        predictor = TabularPredictor(label="target", verbosity=0)
        assert hasattr(predictor, "refit_full")

    def test_refit_full_exists_on_multimodal(self):
        """Test that refit_full() exists on MultiModalPredictor."""
        from endgame.automl import MultiModalPredictor

        predictor = MultiModalPredictor(label="target", verbosity=0)
        assert hasattr(predictor, "refit_full")

    def test_refit_full_requires_fitted(self):
        """Test that refit_full() raises if not fitted."""
        from endgame.automl import TabularPredictor

        predictor = TabularPredictor(label="target", verbosity=0)
        with pytest.raises(RuntimeError, match="not fitted"):
            predictor.refit_full()

    def test_refit_full_requires_data(self):
        """Test that refit_full() raises if no data available."""
        from endgame.automl import TabularPredictor

        predictor = TabularPredictor(label="target", verbosity=0)
        predictor.is_fitted_ = True
        predictor._models = {"test": {"estimator": LinearRegression()}}
        predictor._train_data_ref = None

        with pytest.raises(ValueError, match="No data available"):
            predictor.refit_full()
