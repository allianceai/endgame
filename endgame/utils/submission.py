from __future__ import annotations

"""Submission file generation and validation."""

from typing import Any

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class SubmissionHelper:
    """Helper for generating properly formatted submission files.

    Handles common submission formats for Kaggle competitions.

    Parameters
    ----------
    id_col : str, default='id'
        Name of the ID column.
    target_col : str or List[str], default='target'
        Name(s) of the target column(s).
    float_precision : int, default=6
        Decimal places for float values.

    Examples
    --------
    >>> helper = SubmissionHelper(id_col='Id', target_col='Prediction')
    >>> helper.to_csv(predictions, ids, 'submission.csv')
    >>> helper.validate('submission.csv', 'sample_submission.csv')
    """

    def __init__(
        self,
        id_col: str = "id",
        target_col: str | list[str] = "target",
        float_precision: int = 6,
    ):
        self.id_col = id_col
        self.target_col = target_col
        self.float_precision = float_precision

    def to_csv(
        self,
        predictions: np.ndarray,
        ids: np.ndarray | None = None,
        filepath: str = "submission.csv",
        sample_submission: str | None = None,
    ) -> str:
        """Generate submission CSV file.

        Parameters
        ----------
        predictions : array-like
            Predicted values.
        ids : array-like, optional
            Sample IDs. If None, uses 0, 1, 2, ...
        filepath : str, default='submission.csv'
            Output file path.
        sample_submission : str, optional
            Path to sample submission for ID extraction.

        Returns
        -------
        str
            Path to generated submission file.
        """
        predictions = np.asarray(predictions)

        # Get IDs
        if ids is None:
            if sample_submission is not None:
                ids = self._load_ids_from_sample(sample_submission)
            else:
                ids = np.arange(len(predictions))

        ids = np.asarray(ids)

        if len(ids) != len(predictions):
            raise ValueError(
                f"Length mismatch: {len(ids)} IDs vs {len(predictions)} predictions"
            )

        # Build submission dataframe
        if HAS_PANDAS:
            df = pd.DataFrame({self.id_col: ids})

            if isinstance(self.target_col, list):
                # Multi-target
                if predictions.ndim == 1:
                    raise ValueError("Multi-target specified but predictions is 1D")
                for i, col in enumerate(self.target_col):
                    df[col] = predictions[:, i]
            else:
                # Single target
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    predictions = predictions.ravel()
                df[self.target_col] = predictions

            # Format floats
            float_cols = df.select_dtypes(include=[np.float64, np.float32]).columns
            for col in float_cols:
                df[col] = df[col].round(self.float_precision)

            df.to_csv(filepath, index=False)

        elif HAS_POLARS:
            data = {self.id_col: ids}

            if isinstance(self.target_col, list):
                for i, col in enumerate(self.target_col):
                    data[col] = np.round(predictions[:, i], self.float_precision)
            else:
                if predictions.ndim == 2 and predictions.shape[1] == 1:
                    predictions = predictions.ravel()
                data[self.target_col] = np.round(predictions, self.float_precision)

            df = pl.DataFrame(data)
            df.write_csv(filepath)

        else:
            # Pure numpy fallback
            with open(filepath, 'w') as f:
                # Header
                if isinstance(self.target_col, list):
                    header = [self.id_col] + self.target_col
                else:
                    header = [self.id_col, self.target_col]
                f.write(','.join(header) + '\n')

                # Data
                for i in range(len(ids)):
                    row = [str(ids[i])]
                    if isinstance(self.target_col, list):
                        for j in range(len(self.target_col)):
                            row.append(f"{predictions[i, j]:.{self.float_precision}f}")
                    else:
                        val = predictions[i] if predictions.ndim == 1 else predictions[i, 0]
                        row.append(f"{val:.{self.float_precision}f}")
                    f.write(','.join(row) + '\n')

        return filepath

    def _load_ids_from_sample(self, sample_path: str) -> np.ndarray:
        """Load IDs from sample submission file."""
        if HAS_PANDAS:
            df = pd.read_csv(sample_path)
            return df[self.id_col].values
        elif HAS_POLARS:
            df = pl.read_csv(sample_path)
            return df[self.id_col].to_numpy()
        else:
            ids = []
            with open(sample_path) as f:
                header = f.readline().strip().split(',')
                id_idx = header.index(self.id_col)
                for line in f:
                    parts = line.strip().split(',')
                    ids.append(parts[id_idx])
            return np.array(ids)

    def validate(
        self,
        submission_path: str,
        sample_submission_path: str,
    ) -> dict[str, Any]:
        """Validate submission against sample submission.

        Parameters
        ----------
        submission_path : str
            Path to submission file.
        sample_submission_path : str
            Path to sample submission file.

        Returns
        -------
        Dict[str, Any]
            Validation results with keys:
            - valid: bool
            - errors: List[str]
            - warnings: List[str]
            - n_rows: int
        """
        result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "n_rows": 0,
        }

        if HAS_PANDAS:
            try:
                sub = pd.read_csv(submission_path)
                sample = pd.read_csv(sample_submission_path)
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Failed to read CSV: {e}")
                return result

            result["n_rows"] = len(sub)

            # Check row count
            if len(sub) != len(sample):
                result["valid"] = False
                result["errors"].append(
                    f"Row count mismatch: {len(sub)} vs {len(sample)}"
                )

            # Check columns
            if list(sub.columns) != list(sample.columns):
                result["valid"] = False
                result["errors"].append(
                    f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}"
                )

            # Check IDs
            if self.id_col in sub.columns and self.id_col in sample.columns:
                if not np.array_equal(sub[self.id_col].values, sample[self.id_col].values):
                    result["valid"] = False
                    result["errors"].append("ID values don't match sample submission")

            # Check for NaN
            if sub.isna().any().any():
                result["warnings"].append("Submission contains NaN values")

        elif HAS_POLARS:
            try:
                sub = pl.read_csv(submission_path)
                sample = pl.read_csv(sample_submission_path)
            except Exception as e:
                result["valid"] = False
                result["errors"].append(f"Failed to read CSV: {e}")
                return result

            result["n_rows"] = len(sub)

            if len(sub) != len(sample):
                result["valid"] = False
                result["errors"].append(
                    f"Row count mismatch: {len(sub)} vs {len(sample)}"
                )

            if sub.columns != sample.columns:
                result["valid"] = False
                result["errors"].append(
                    f"Column mismatch: {sub.columns} vs {sample.columns}"
                )

            if sub.null_count().sum_horizontal()[0] > 0:
                result["warnings"].append("Submission contains null values")

        return result

    def from_oof_predictions(
        self,
        oof_models: list[Any],
        X_test: np.ndarray,
        weights: dict[int, float] | None = None,
        ids: np.ndarray | None = None,
        filepath: str = "submission.csv",
    ) -> str:
        """Generate submission from OOF models.

        Parameters
        ----------
        oof_models : List
            List of trained models (from cross-validation).
        X_test : array-like
            Test features.
        weights : Dict[int, float], optional
            Model weights. If None, uses uniform weights.
        ids : array-like, optional
            Test sample IDs.
        filepath : str
            Output file path.

        Returns
        -------
        str
            Path to submission file.
        """
        X_test = np.asarray(X_test)
        n_models = len(oof_models)

        if weights is None:
            weights = {i: 1.0 / n_models for i in range(n_models)}

        # Collect predictions
        predictions = np.zeros(len(X_test))

        for i, model in enumerate(oof_models):
            weight = weights.get(i, 0.0)
            if weight > 0:
                if hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_test)
                    if pred.ndim == 2:
                        pred = pred[:, 1]
                else:
                    pred = model.predict(X_test)
                predictions += weight * pred

        # Normalize
        total_weight = sum(weights.values())
        if total_weight > 0:
            predictions /= total_weight

        return self.to_csv(predictions, ids, filepath)
