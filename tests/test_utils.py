"""Tests for utils module."""

import numpy as np
import pytest
import tempfile
import os


class TestMetrics:
    """Tests for custom metrics."""

    def test_competition_metric_returns_callable(self):
        """Test competition_metric returns a callable metric function."""
        from endgame.utils.metrics import competition_metric

        metric_fn = competition_metric("map")
        assert callable(metric_fn)

    def test_competition_metric_map(self):
        """Test mAP metric via competition_metric lookup."""
        from endgame.utils.metrics import competition_metric

        y_true = np.array([[1, 0, 1], [0, 1, 0]])
        y_pred = np.array([[0.9, 0.1, 0.8], [0.2, 0.9, 0.1]])

        metric_fn = competition_metric("map")
        score = metric_fn(y_true, y_pred)
        assert score > 0.5

    def test_competition_metric_mcrmse(self):
        """Test MCRMSE metric via competition_metric lookup."""
        from endgame.utils.metrics import competition_metric

        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

        metric_fn = competition_metric("mcrmse")
        score = metric_fn(y_true, y_pred)
        assert score == 0.0

    def test_competition_metric_unknown_raises(self):
        """Test that unknown metric name raises ValueError."""
        from endgame.utils.metrics import competition_metric

        with pytest.raises(ValueError, match="Unknown metric"):
            competition_metric("nonexistent_metric_xyz")

    def test_quadratic_weighted_kappa(self):
        """Test QWK metric."""
        from endgame.utils.metrics import quadratic_weighted_kappa

        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1, 2, 3, 4, 5])

        score = quadratic_weighted_kappa(y_true, y_pred)
        assert score == 1.0

    def test_mcrmse(self):
        """Test MCRMSE metric."""
        from endgame.utils.metrics import mcrmse

        y_true = np.array([[1.0, 2.0], [3.0, 4.0]])
        y_pred = np.array([[1.0, 2.0], [3.0, 4.0]])

        score = mcrmse(y_true, y_pred)
        assert score == 0.0


class TestSubmission:
    """Tests for submission utilities."""

    def test_create_submission(self):
        """Test submission file creation."""
        pl = pytest.importorskip("polars")
        from endgame.utils.submission import SubmissionHelper

        ids = ["id1", "id2", "id3"]
        predictions = np.array([0.1, 0.5, 0.9])

        helper = SubmissionHelper(id_col="id", target_col="target")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "submission.csv")
            helper.to_csv(
                predictions=predictions,
                ids=ids,
                filepath=filepath,
            )

            # Verify file was created
            assert os.path.exists(filepath)

            # Verify contents
            df = pl.read_csv(filepath)
            assert len(df) == 3
            assert list(df.columns) == ["id", "target"]

    def test_create_multilabel_submission(self):
        """Test multilabel submission."""
        pl = pytest.importorskip("polars")
        from endgame.utils.submission import SubmissionHelper

        ids = ["id1", "id2"]
        predictions = np.array([[0.1, 0.9], [0.8, 0.2]])
        label_cols = ["label_a", "label_b"]

        helper = SubmissionHelper(id_col="id", target_col=label_cols)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "submission.csv")
            helper.to_csv(
                predictions=predictions,
                ids=ids,
                filepath=filepath,
            )

            df = pl.read_csv(filepath)
            assert len(df) == 2
            assert list(df.columns) == ["id", "label_a", "label_b"]

    def test_validate_submission(self):
        """Test submission validation."""
        pl = pytest.importorskip("polars")
        from endgame.utils.submission import SubmissionHelper

        helper = SubmissionHelper(id_col="id", target_col="target")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create sample submission
            sample_path = os.path.join(tmpdir, "sample.csv")
            pl.DataFrame({
                "id": ["a", "b", "c"],
                "target": [0.0, 0.0, 0.0],
            }).write_csv(sample_path)

            # Create test submission
            submission_path = os.path.join(tmpdir, "submission.csv")
            pl.DataFrame({
                "id": ["a", "b", "c"],
                "target": [0.1, 0.5, 0.9],
            }).write_csv(submission_path)

            # Should validate successfully
            result = helper.validate(submission_path, sample_path)
            assert result["valid"]

    def test_validate_submission_wrong_rows(self):
        """Test validation catches wrong number of rows."""
        pl = pytest.importorskip("polars")
        from endgame.utils.submission import SubmissionHelper

        helper = SubmissionHelper(id_col="id", target_col="target")

        with tempfile.TemporaryDirectory() as tmpdir:
            sample_path = os.path.join(tmpdir, "sample.csv")
            pl.DataFrame({
                "id": ["a", "b", "c"],
                "target": [0.0, 0.0, 0.0],
            }).write_csv(sample_path)

            submission_path = os.path.join(tmpdir, "submission.csv")
            pl.DataFrame({
                "id": ["a", "b"],  # Missing one row
                "target": [0.1, 0.5],
            }).write_csv(submission_path)

            result = helper.validate(submission_path, sample_path)
            assert not result["valid"]
            assert any("row" in err.lower() for err in result["errors"])


class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_seed_everything(self):
        """Test seeding all random generators."""
        from endgame.utils.reproducibility import seed_everything

        seed_everything(42)

        # Numpy should be deterministic
        a = np.random.rand(5)
        seed_everything(42)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_reproducible_run_environment_info(self):
        """Test ReproducibleRun captures environment info."""
        from endgame.utils.reproducibility import ReproducibleRun

        with ReproducibleRun(seed=42) as run:
            env_info = run.environment_info

        assert "python_version" in env_info
        assert "platform" in env_info
        assert "numpy_version" in env_info
        assert "seed" in env_info
        assert env_info["seed"] == 42

    def test_seed_everything_context_manager(self):
        """Test SeedEverything context manager."""
        from endgame.utils.reproducibility import SeedEverything

        with SeedEverything(42):
            a = np.random.rand(5)

        with SeedEverything(42):
            b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)
