from __future__ import annotations

"""High-level competition management interface.

Provides a convenient abstraction for working with a single Kaggle competition.
"""

import json
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from endgame.kaggle.client import (
    CompetitionInfo,
    KaggleClient,
    SubmissionInfo,
    SubmissionResult,
)
from endgame.utils.submission import SubmissionHelper

# Optional pandas import
HAS_PANDAS = False
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass

# Optional polars import
HAS_POLARS = False
try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    pass


# Default cache directory
DEFAULT_CACHE_DIR = Path.home() / ".endgame" / "competitions"


@dataclass
class ValidationResult:
    """Result of submission validation.

    Attributes
    ----------
    valid : bool
        Whether the submission is valid.
    errors : List[str]
        List of validation errors.
    warnings : List[str]
        List of warnings (non-fatal issues).
    n_rows : int
        Number of rows in submission.
    expected_rows : int
        Expected number of rows.
    columns : List[str]
        Columns in submission file.
    expected_columns : List[str]
        Expected columns from sample submission.
    """
    valid: bool
    errors: list[str]
    warnings: list[str]
    n_rows: int = 0
    expected_rows: int = 0
    columns: list[str] = None
    expected_columns: list[str] = None

    def __post_init__(self):
        if self.columns is None:
            self.columns = []
        if self.expected_columns is None:
            self.expected_columns = []

    def __str__(self) -> str:
        if self.valid:
            msg = f"Valid submission ({self.n_rows} rows)"
            if self.warnings:
                msg += f"\nWarnings: {', '.join(self.warnings)}"
            return msg
        return "Invalid submission:\n" + "\n".join(f"  - {e}" for e in self.errors)


class Competition:
    """High-level interface for working with a Kaggle competition.

    Manages data download, loading, submission creation, and validation
    for a single competition.

    Parameters
    ----------
    slug : str
        Competition URL slug (e.g., 'titanic', 'house-prices-advanced-regression-techniques').
    data_dir : str or Path, optional
        Directory for competition data. Defaults to ~/.endgame/competitions/{slug}/
    client : KaggleClient, optional
        Kaggle client instance. Created automatically if not provided.

    Attributes
    ----------
    slug : str
        Competition slug.
    data_dir : Path
        Directory containing competition data.
    raw_dir : Path
        Directory for raw downloaded files.
    submissions_dir : Path
        Directory for local submission files.

    Examples
    --------
    >>> comp = Competition("titanic")
    >>> comp.download_data()

    >>> train = comp.load_train()
    >>> test = comp.load_test()

    >>> # Make predictions...
    >>> predictions = model.predict(test)

    >>> # Create and submit
    >>> submission_path = comp.create_submission(predictions)
    >>> result = comp.submit(submission_path, "First attempt")
    >>> print(f"Score: {result.public_score}")
    """

    def __init__(
        self,
        slug: str,
        data_dir: str | Path | None = None,
        client: KaggleClient | None = None,
    ):
        self.slug = slug
        self._client = client

        # Set up directories
        if data_dir is None:
            self.data_dir = DEFAULT_CACHE_DIR / slug
        else:
            self.data_dir = Path(data_dir)

        self.raw_dir = self.data_dir / "raw"
        self.submissions_dir = self.data_dir / "submissions"

        # Cached info
        self._info: CompetitionInfo | None = None
        self._submission_format: dict[str, Any] | None = None

    @property
    def client(self) -> KaggleClient:
        """Get or create Kaggle client."""
        if self._client is None:
            self._client = KaggleClient()
        return self._client

    def info(self, refresh: bool = False) -> CompetitionInfo:
        """Get competition information.

        Parameters
        ----------
        refresh : bool, default=False
            Force refresh from API.

        Returns
        -------
        CompetitionInfo
            Competition details.
        """
        if self._info is None or refresh:
            self._info = self.client.get_competition(self.slug)
        return self._info

    def download_data(
        self,
        force: bool = False,
        quiet: bool = False,
    ) -> dict[str, Path]:
        """Download competition data files.

        Parameters
        ----------
        force : bool, default=False
            Force re-download even if files exist.
        quiet : bool, default=False
            Suppress download progress output.

        Returns
        -------
        Dict[str, Path]
            Mapping of file names to local paths.

        Notes
        -----
        You must accept the competition rules on Kaggle's website
        before downloading data.
        """
        self.raw_dir.mkdir(parents=True, exist_ok=True)

        # Check if data already exists
        existing_files = list(self.raw_dir.glob("*"))
        csv_files = [f for f in existing_files if f.suffix in ('.csv', '.parquet', '.feather')]

        if csv_files and not force:
            if not quiet:
                print(f"Data already downloaded in {self.raw_dir}")
                print("Use force=True to re-download")
            return self._list_data_files()

        if not quiet:
            print(f"Downloading data for competition: {self.slug}")

        self.client.download_competition(
            self.slug,
            path=self.raw_dir,
            force=force,
            quiet=quiet,
            unzip=True,
        )

        return self._list_data_files()

    def _list_data_files(self) -> dict[str, Path]:
        """List available data files."""
        files = {}
        for f in self.raw_dir.iterdir():
            if f.is_file() and not f.name.startswith('.'):
                files[f.name] = f
        return files

    def _find_file(self, patterns: list[str]) -> Path | None:
        """Find a file matching one of the patterns."""
        for pattern in patterns:
            matches = list(self.raw_dir.glob(pattern))
            if matches:
                return matches[0]
        return None

    def _load_file(self, filepath: Path) -> Any:
        """Load a data file based on its extension."""
        suffix = filepath.suffix.lower()

        if suffix == '.csv':
            if HAS_PANDAS:
                return pd.read_csv(filepath)
            elif HAS_POLARS:
                return pl.read_csv(filepath)
            else:
                raise ImportError("pandas or polars required to load CSV files")

        elif suffix == '.parquet':
            if HAS_PANDAS:
                return pd.read_parquet(filepath)
            elif HAS_POLARS:
                return pl.read_parquet(filepath)
            else:
                raise ImportError("pandas or polars required to load Parquet files")

        elif suffix == '.feather':
            if HAS_PANDAS:
                return pd.read_feather(filepath)
            elif HAS_POLARS:
                return pl.read_ipc(filepath)
            else:
                raise ImportError("pandas or polars required to load Feather files")

        elif suffix == '.json':
            with open(filepath) as f:
                return json.load(f)

        else:
            raise ValueError(f"Unsupported file format: {suffix}")

    def load_train(self) -> Any:
        """Load the training dataset.

        Returns
        -------
        DataFrame
            Training data as pandas or polars DataFrame.

        Raises
        ------
        FileNotFoundError
            If training file not found. Call download_data() first.
        """
        filepath = self._find_file([
            "train.csv", "train.parquet", "train.feather",
            "training.csv", "train_data.csv", "*train*.csv"
        ])

        if filepath is None:
            raise FileNotFoundError(
                f"Training file not found in {self.raw_dir}. "
                "Call download_data() first."
            )

        return self._load_file(filepath)

    def load_test(self) -> Any:
        """Load the test dataset.

        Returns
        -------
        DataFrame
            Test data as pandas or polars DataFrame.
        """
        filepath = self._find_file([
            "test.csv", "test.parquet", "test.feather",
            "testing.csv", "test_data.csv", "*test*.csv"
        ])

        if filepath is None:
            raise FileNotFoundError(
                f"Test file not found in {self.raw_dir}. "
                "Call download_data() first."
            )

        return self._load_file(filepath)

    def load_sample_submission(self) -> Any:
        """Load the sample submission file.

        Returns
        -------
        DataFrame
            Sample submission as pandas or polars DataFrame.
        """
        filepath = self._find_file([
            "sample_submission.csv", "sampleSubmission.csv",
            "sample_sub.csv", "*sample*.csv", "*submission*.csv"
        ])

        if filepath is None:
            raise FileNotFoundError(
                f"Sample submission not found in {self.raw_dir}. "
                "Call download_data() first."
            )

        return self._load_file(filepath)

    def load_file(self, filename: str) -> Any:
        """Load a specific file from the competition data.

        Parameters
        ----------
        filename : str
            Name of the file to load.

        Returns
        -------
        DataFrame or dict
            Loaded data.
        """
        filepath = self.raw_dir / filename

        if not filepath.exists():
            # Try glob matching
            matches = list(self.raw_dir.glob(f"*{filename}*"))
            if matches:
                filepath = matches[0]
            else:
                raise FileNotFoundError(f"File not found: {filename}")

        return self._load_file(filepath)

    def get_submission_format(self) -> dict[str, Any]:
        """Detect the expected submission format from sample submission.

        Returns
        -------
        Dict[str, Any]
            Dictionary with keys:
            - 'id_col': Name of ID column
            - 'target_cols': List of target column names
            - 'n_rows': Expected number of rows
            - 'dtypes': Column data types
        """
        if self._submission_format is not None:
            return self._submission_format

        sample = self.load_sample_submission()

        if HAS_PANDAS and isinstance(sample, pd.DataFrame):
            columns = list(sample.columns)
            n_rows = len(sample)
            dtypes = {col: str(dtype) for col, dtype in sample.dtypes.items()}
            id_values = sample.iloc[:, 0].tolist()
        elif HAS_POLARS:
            columns = sample.columns
            n_rows = len(sample)
            dtypes = {col: str(dtype) for col, dtype in zip(sample.columns, sample.dtypes)}
            id_values = sample[:, 0].to_list()
        else:
            raise ImportError("pandas or polars required")

        # First column is typically ID
        id_col = columns[0]
        target_cols = columns[1:] if len(columns) > 1 else columns

        self._submission_format = {
            'id_col': id_col,
            'target_cols': list(target_cols),
            'n_rows': n_rows,
            'dtypes': dtypes,
            'columns': columns,
            'id_values': id_values,
        }

        return self._submission_format

    def create_submission(
        self,
        predictions: np.ndarray | list | Any,
        filepath: str | Path | None = None,
        ids: np.ndarray | list | None = None,
        validate: bool = True,
    ) -> Path:
        """Create a submission file from predictions.

        Automatically formats the submission to match the expected format
        based on the sample submission file.

        Parameters
        ----------
        predictions : array-like or DataFrame
            Predicted values. Can be:
            - 1D array for single target
            - 2D array for multiple targets
            - DataFrame with predictions
        filepath : str or Path, optional
            Output file path. If None, generates timestamped filename.
        ids : array-like, optional
            ID values for each prediction. If None, uses IDs from sample submission.
        validate : bool, default=True
            Validate submission before saving.

        Returns
        -------
        Path
            Path to created submission file.
        """
        self.submissions_dir.mkdir(parents=True, exist_ok=True)

        # Get expected format
        fmt = self.get_submission_format()

        # Generate filepath if not provided
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = self.submissions_dir / f"submission_{timestamp}.csv"
        else:
            filepath = Path(filepath)

        # Get IDs
        if ids is None:
            ids = fmt['id_values']

        # Handle different prediction types
        if HAS_PANDAS and isinstance(predictions, pd.DataFrame):
            # DataFrame with predictions - extract values
            if len(predictions.columns) == len(fmt['target_cols']):
                predictions = predictions.values
            else:
                # Assume first column(s) are targets
                predictions = predictions.iloc[:, :len(fmt['target_cols'])].values
        elif HAS_POLARS and hasattr(predictions, 'to_numpy'):
            predictions = predictions.to_numpy()

        predictions = np.asarray(predictions)

        # Ensure correct shape
        if predictions.ndim == 1:
            if len(fmt['target_cols']) == 1:
                pass  # Correct shape
            else:
                predictions = predictions.reshape(-1, 1)

        # Use SubmissionHelper for consistent formatting
        helper = SubmissionHelper(
            id_col=fmt['id_col'],
            target_col=fmt['target_cols'] if len(fmt['target_cols']) > 1 else fmt['target_cols'][0],
        )

        helper.to_csv(
            predictions=predictions,
            ids=np.array(ids),
            filepath=str(filepath),
        )

        # Validate if requested
        if validate:
            result = self.validate_submission(filepath)
            if not result.valid:
                warnings.warn(f"Submission validation failed:\n{result}")

        return filepath

    def validate_submission(
        self,
        filepath: str | Path,
    ) -> ValidationResult:
        """Validate a submission file against expected format.

        Parameters
        ----------
        filepath : str or Path
            Path to submission file to validate.

        Returns
        -------
        ValidationResult
            Validation results with errors and warnings.
        """
        filepath = Path(filepath)
        errors = []
        warnings_list = []

        if not filepath.exists():
            return ValidationResult(
                valid=False,
                errors=[f"File not found: {filepath}"],
                warnings=[],
            )

        # Load submission
        try:
            if HAS_PANDAS:
                submission = pd.read_csv(filepath)
                sub_columns = list(submission.columns)
                sub_n_rows = len(submission)
                has_nulls = submission.isna().any().any()
            elif HAS_POLARS:
                submission = pl.read_csv(filepath)
                sub_columns = submission.columns
                sub_n_rows = len(submission)
                has_nulls = submission.null_count().sum_horizontal()[0] > 0
            else:
                return ValidationResult(
                    valid=False,
                    errors=["pandas or polars required for validation"],
                    warnings=[],
                )
        except Exception as e:
            return ValidationResult(
                valid=False,
                errors=[f"Failed to read submission: {e}"],
                warnings=[],
            )

        # Get expected format
        try:
            fmt = self.get_submission_format()
            expected_columns = fmt['columns']
            expected_n_rows = fmt['n_rows']
        except FileNotFoundError:
            # Can't validate without sample submission
            return ValidationResult(
                valid=True,
                errors=[],
                warnings=["Could not validate: sample submission not found"],
                n_rows=sub_n_rows,
                columns=sub_columns,
            )

        # Check columns
        if sub_columns != expected_columns:
            errors.append(
                f"Column mismatch: got {sub_columns}, expected {expected_columns}"
            )

        # Check row count
        if sub_n_rows != expected_n_rows:
            errors.append(
                f"Row count mismatch: got {sub_n_rows}, expected {expected_n_rows}"
            )

        # Check for nulls
        if has_nulls:
            warnings_list.append("Submission contains null/NaN values")

        # Check ID values match
        if HAS_PANDAS and fmt['id_col'] in sub_columns:
            sub_ids = submission[fmt['id_col']].tolist()
            if sub_ids != fmt['id_values']:
                errors.append("ID values don't match expected values from sample submission")

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings_list,
            n_rows=sub_n_rows,
            expected_rows=expected_n_rows,
            columns=sub_columns,
            expected_columns=expected_columns,
        )

    def submit(
        self,
        filepath: str | Path,
        message: str,
        validate: bool = True,
    ) -> SubmissionResult:
        """Submit a file to the competition.

        Parameters
        ----------
        filepath : str or Path
            Path to submission file.
        message : str
            Submission description/message.
        validate : bool, default=True
            Validate submission before uploading.

        Returns
        -------
        SubmissionResult
            Result of submission attempt.
        """
        filepath = Path(filepath)

        # Validate first
        if validate:
            validation = self.validate_submission(filepath)
            if not validation.valid:
                return SubmissionResult(
                    success=False,
                    error=f"Validation failed: {', '.join(validation.errors)}"
                )

        return self.client.submit(self.slug, filepath, message)

    def submissions(self, limit: int | None = None) -> list[SubmissionInfo]:
        """Get submission history for this competition.

        Parameters
        ----------
        limit : int, optional
            Maximum number of submissions to return.

        Returns
        -------
        List[SubmissionInfo]
            List of submissions, most recent first.
        """
        return self.client.get_submissions(self.slug, limit=limit)

    def leaderboard(self, n: int = 20) -> list[dict[str, Any]]:
        """Get competition leaderboard.

        Parameters
        ----------
        n : int, default=20
            Number of entries to return.

        Returns
        -------
        List[Dict[str, Any]]
            Leaderboard entries.
        """
        lb = self.client.get_leaderboard(self.slug)
        return lb[:n]

    def __repr__(self) -> str:
        return f"Competition('{self.slug}')"

    def __str__(self) -> str:
        try:
            info = self.info()
            return str(info)
        except Exception:
            return f"Competition: {self.slug}"
