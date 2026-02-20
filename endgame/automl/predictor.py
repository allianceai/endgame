"""Unified AutoML predictor entry point.

This module provides the AutoMLPredictor class - the primary interface
for automated machine learning in Endgame.
"""

import logging
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from endgame.automl.base import BasePredictor, DataInput
from endgame.automl.tabular import TabularPredictor

logger = logging.getLogger(__name__)


class AutoMLPredictor:
    """Unified AutoML predictor that automatically selects the right domain.

    This is the main entry point for AutoML in Endgame. It provides a simple
    3-line interface that matches AutoGluon's simplicity while leveraging
    Endgame's full capabilities.

    Parameters
    ----------
    label : str
        Name of the target column.
    problem_type : str, default="auto"
        Type of problem: "classification", "regression", "multiclass", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "good_quality",
        "medium_quality", "fast", "interpretable".
    time_limit : int, optional
        Time limit in seconds. If None, uses preset default.
    search_strategy : str, default="portfolio"
        Search strategy: "portfolio", "heuristic", "genetic", "random", "bayesian".
    track_experiments : bool, default=True
        Whether to track experiments to the meta-learning database.
    output_path : str, optional
        Path to save outputs (models, logs, etc.).
    random_state : int, default=42
        Random seed for reproducibility.
    verbosity : int, default=2
        Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).

    Attributes
    ----------
    predictor_ : BasePredictor
        The underlying domain-specific predictor.
    domain_ : str
        The detected or specified data domain.

    Examples
    --------
    3-line usage (matches AutoGluon):

    >>> from endgame.automl import AutoMLPredictor
    >>> predictor = AutoMLPredictor(label="target").fit("train.csv")
    >>> predictions = predictor.predict("test.csv")

    With more options:

    >>> predictor = AutoMLPredictor(
    ...     label="price",
    ...     presets="best_quality",
    ...     time_limit=3600,
    ... )
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    Using different presets:

    >>> # Fast training for prototyping
    >>> predictor = AutoMLPredictor(label="target", presets="fast")
    >>>
    >>> # High quality for production
    >>> predictor = AutoMLPredictor(label="target", presets="high_quality")
    >>>
    >>> # Best quality for competitions
    >>> predictor = AutoMLPredictor(label="target", presets="best_quality")

    Different search strategies:

    >>> # Default portfolio search
    >>> predictor = AutoMLPredictor(label="target", search_strategy="portfolio")
    >>>
    >>> # Genetic algorithm search
    >>> predictor = AutoMLPredictor(label="target", search_strategy="genetic")
    >>>
    >>> # Bayesian optimization
    >>> predictor = AutoMLPredictor(label="target", search_strategy="bayesian")
    """

    def __init__(
        self,
        label: str,
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        search_strategy: str = "portfolio",
        track_experiments: bool = True,
        output_path: str | None = None,
        random_state: int = 42,
        verbosity: int = 2,
    ):
        self.label = label
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.presets = presets
        self.time_limit = time_limit
        self.search_strategy = search_strategy
        self.track_experiments = track_experiments
        self.output_path = output_path
        self.random_state = random_state
        self.verbosity = verbosity

        # State
        self.predictor_: BasePredictor | None = None
        self.domain_: str | None = None

    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        domain: str | None = None,
        **kwargs,
    ) -> "AutoMLPredictor":
        """Fit the AutoML predictor.

        Parameters
        ----------
        train_data : str, Path, DataFrame, or ndarray
            Training data. Can be a file path, DataFrame, or array.
        tuning_data : optional
            Validation/tuning data. If None, uses internal holdout.
        time_limit : int, optional
            Override the time limit.
        presets : str, optional
            Override the preset.
        hyperparameters : dict, optional
            Override hyperparameters for specific models.
        domain : str, optional
            Data domain: "tabular", "text", "vision", "timeseries", "audio".
            If None, auto-detects from data.
        **kwargs
            Additional arguments passed to the domain-specific predictor.

        Returns
        -------
        AutoMLPredictor
            The fitted predictor.
        """
        # Auto-detect domain if not specified
        if domain is None:
            domain = self._detect_domain(train_data)

        self.domain_ = domain

        # Create domain-specific predictor
        self.predictor_ = self._create_predictor(domain)

        # Fit the predictor
        self.predictor_.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            time_limit=time_limit or self.time_limit,
            presets=presets or self.presets,
            hyperparameters=hyperparameters,
            **kwargs,
        )

        return self

    def predict(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data to predict on.
        model : str, optional
            Specific model to use. If None, uses the ensemble.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._check_is_fitted()
        return self.predictor_.predict(data, model=model)

    def predict_proba(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions (classification only).

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Probability predictions with shape (n_samples, n_classes).
        """
        self._check_is_fitted()
        return self.predictor_.predict_proba(data, model=model)

    def evaluate(
        self,
        data: DataInput,
        metrics: list[str] | None = None,
        silent: bool = False,
    ) -> dict[str, float]:
        """Evaluate the predictor on data.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Data to evaluate on. Must contain the target column.
        metrics : list of str, optional
            Metrics to compute. If None, uses default metrics.
        silent : bool, default=False
            Whether to suppress output.

        Returns
        -------
        dict
            Dictionary mapping metric names to scores.
        """
        self._check_is_fitted()
        return self.predictor_.evaluate(data, metrics=metrics, silent=silent)

    def leaderboard(
        self,
        extra_info: bool = False,
        silent: bool = False,
    ) -> pd.DataFrame:
        """Get the model leaderboard.

        Parameters
        ----------
        extra_info : bool, default=False
            Whether to include extra information (fit time, etc.).
        silent : bool, default=False
            Whether to suppress output.

        Returns
        -------
        pd.DataFrame
            Leaderboard with model names and scores.
        """
        self._check_is_fitted()
        return self.predictor_.leaderboard(extra_info=extra_info, silent=silent)

    def feature_importance(
        self,
        model: str | None = None,
        importance_type: str = "split",
    ) -> pd.DataFrame:
        """Get feature importance scores.

        Parameters
        ----------
        model : str, optional
            Specific model. If None, uses best model.
        importance_type : str, default="split"
            Type of importance: "split", "gain", "permutation".

        Returns
        -------
        pd.DataFrame
            Feature importance scores.
        """
        self._check_is_fitted()
        return self.predictor_.feature_importance(
            model=model, importance_type=importance_type
        )

    def save(self, path: str | None = None) -> str:
        """Save the predictor to disk.

        Parameters
        ----------
        path : str, optional
            Path to save to. If None, uses output_path.

        Returns
        -------
        str
            Path where the predictor was saved.
        """
        self._check_is_fitted()
        return self.predictor_.save(path)

    @classmethod
    def load(cls, path: str) -> "AutoMLPredictor":
        """Load a predictor from disk.

        Parameters
        ----------
        path : str
            Path to load from.

        Returns
        -------
        AutoMLPredictor
            The loaded predictor.
        """
        path = Path(path)

        # Determine domain from saved predictor
        # For now, assume tabular
        predictor = TabularPredictor.load(str(path))

        # Create wrapper
        wrapper = cls(
            label=predictor.label,
            problem_type=predictor.problem_type,
            eval_metric=predictor.eval_metric,
            presets=predictor.presets,
            time_limit=predictor.time_limit,
            search_strategy=predictor.search_strategy,
            random_state=predictor.random_state,
            verbosity=predictor.verbosity,
        )
        wrapper.predictor_ = predictor
        wrapper.domain_ = "tabular"

        return wrapper

    def _detect_domain(self, data: DataInput) -> str:
        """Detect the data domain from the input.

        Parameters
        ----------
        data : various
            Input data.

        Returns
        -------
        str
            Detected domain.
        """
        # For now, default to tabular
        # Future: implement detection for text, vision, etc.

        if isinstance(data, (str, Path)):
            path = Path(data)
            suffix = path.suffix.lower()

            # Image formats
            if suffix in (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"):
                return "vision"

            # Audio formats
            if suffix in (".wav", ".mp3", ".flac", ".ogg", ".m4a"):
                return "audio"

            # Check file contents for text vs tabular
            if suffix == ".csv":
                # Could be tabular or text - check first rows
                try:
                    df = pd.read_csv(data, nrows=5)
                    if self._looks_like_text_data(df):
                        return "text"
                    return "tabular"
                except Exception:
                    return "tabular"

            return "tabular"

        elif isinstance(data, pd.DataFrame):
            if self._looks_like_text_data(data):
                return "text"
            return "tabular"

        elif isinstance(data, np.ndarray):
            # Check shape for images
            if data.ndim == 4:
                # Likely image data (N, H, W, C) or (N, C, H, W)
                return "vision"
            return "tabular"

        return "tabular"

    def _looks_like_text_data(self, df: pd.DataFrame) -> bool:
        """Check if DataFrame looks like text data.

        Parameters
        ----------
        df : DataFrame
            Data to check.

        Returns
        -------
        bool
            True if data appears to be text.
        """
        # Check for text-like columns (long strings)
        for col in df.columns:
            if col == self.label:
                continue

            if df[col].dtype == object:
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    avg_len = sample.astype(str).str.len().mean()
                    # Text columns typically have long strings
                    if avg_len > 100:
                        return True

        return False

    def _create_predictor(self, domain: str) -> BasePredictor:
        """Create a domain-specific predictor.

        Parameters
        ----------
        domain : str
            Data domain.

        Returns
        -------
        BasePredictor
            Domain-specific predictor.
        """
        if domain == "tabular":
            return TabularPredictor(
                label=self.label,
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                presets=self.presets,
                time_limit=self.time_limit,
                search_strategy=self.search_strategy,
                track_experiments=self.track_experiments,
                output_path=self.output_path,
                random_state=self.random_state,
                verbosity=self.verbosity,
            )

        elif domain == "text":
            try:
                from endgame.automl.text import TextPredictor

                return TextPredictor(
                    label=self.label,
                    problem_type=self.problem_type,
                    eval_metric=self.eval_metric,
                    presets=self.presets,
                    time_limit=self.time_limit,
                    track_experiments=self.track_experiments,
                    output_path=self.output_path,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                )
            except ImportError:
                warnings.warn(
                    "TextPredictor not available. Using TabularPredictor with text features.",
                    UserWarning,
                )
                return TabularPredictor(
                    label=self.label,
                    problem_type=self.problem_type,
                    eval_metric=self.eval_metric,
                    presets=self.presets,
                    time_limit=self.time_limit,
                    search_strategy=self.search_strategy,
                    track_experiments=self.track_experiments,
                    output_path=self.output_path,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                )

        elif domain == "vision":
            try:
                from endgame.automl.vision import VisionPredictor

                return VisionPredictor(
                    label=self.label,
                    problem_type=self.problem_type,
                    eval_metric=self.eval_metric,
                    presets=self.presets,
                    time_limit=self.time_limit,
                    track_experiments=self.track_experiments,
                    output_path=self.output_path,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                )
            except ImportError:
                raise NotImplementedError(
                    "VisionPredictor is not yet implemented. "
                    "Please use TabularPredictor with image features."
                )

        elif domain == "timeseries":
            try:
                from endgame.automl.timeseries import TimeSeriesPredictor

                return TimeSeriesPredictor(
                    label=self.label,
                    problem_type=self.problem_type,
                    eval_metric=self.eval_metric,
                    presets=self.presets,
                    time_limit=self.time_limit,
                    track_experiments=self.track_experiments,
                    output_path=self.output_path,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                )
            except ImportError:
                raise NotImplementedError(
                    "TimeSeriesPredictor is not yet implemented."
                )

        elif domain == "audio":
            try:
                from endgame.automl.audio import AudioPredictor

                return AudioPredictor(
                    label=self.label,
                    problem_type=self.problem_type,
                    eval_metric=self.eval_metric,
                    presets=self.presets,
                    time_limit=self.time_limit,
                    track_experiments=self.track_experiments,
                    output_path=self.output_path,
                    random_state=self.random_state,
                    verbosity=self.verbosity,
                )
            except ImportError:
                raise NotImplementedError(
                    "AudioPredictor is not yet implemented."
                )

        else:
            raise ValueError(f"Unknown domain: {domain}")

    def _check_is_fitted(self) -> None:
        """Check if the predictor is fitted."""
        if self.predictor_ is None or not self.predictor_.is_fitted_:
            raise RuntimeError(
                "Predictor is not fitted. Call fit() before making predictions."
            )

    @property
    def is_fitted(self) -> bool:
        """Check if the predictor is fitted."""
        return self.predictor_ is not None and self.predictor_.is_fitted_

    @property
    def problem_type_(self) -> str | None:
        """Get the detected problem type."""
        if self.predictor_ is None:
            return None
        return self.predictor_.problem_type_

    @property
    def classes_(self) -> np.ndarray | None:
        """Get the class labels for classification."""
        if self.predictor_ is None:
            return None
        return self.predictor_.classes_

    @property
    def feature_names_(self) -> list[str] | None:
        """Get the feature names."""
        if self.predictor_ is None:
            return None
        return self.predictor_.feature_names_

    @property
    def fit_summary_(self):
        """Get the fit summary."""
        if self.predictor_ is None:
            return None
        return self.predictor_.fit_summary_

    def __repr__(self) -> str:
        if self.predictor_ is None:
            fitted_str = "not fitted"
        else:
            fitted_str = "fitted" if self.predictor_.is_fitted_ else "not fitted"

        return (
            f"AutoMLPredictor("
            f"label='{self.label}', "
            f"presets='{self.presets}', "
            f"domain='{self.domain_}', "
            f"{fitted_str})"
        )
