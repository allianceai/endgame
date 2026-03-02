from __future__ import annotations

"""Time series data AutoML predictor.

This module provides the TimeSeriesPredictor class for automated machine learning
on time series classification tasks using ROCKET-family classifiers and feature
extraction.
"""

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from endgame.automl.base import BasePredictor, DataInput, FitSummary
from endgame.automl.presets import get_preset

logger = logging.getLogger(__name__)


# Default time series classifiers for different scenarios
DEFAULT_CLASSIFIERS = {
    "fast": "minirocket",
    "medium": "minirocket",
    "best": "hydra_minirocket",
    "rocket": "rocket",
    "multirocket": "multirocket",
}

# Classifier configurations
CLASSIFIER_CONFIGS = {
    "minirocket": {
        "n_kernels": 10000,
        "classifier": "ridge",
    },
    "rocket": {
        "n_kernels": 10000,
        "classifier": "ridge",
    },
    "multirocket": {
        "n_kernels": 10000,
        "classifier": "ridge",
    },
    "hydra": {
        "n_groups": 64,
        "n_kernels_per_group": 8,
        "classifier": "ridge",
    },
    "hydra_minirocket": {
        "n_kernels": 10000,
        "n_groups": 64,
        "classifier": "ridge",
    },
}


class TimeSeriesPredictor(BasePredictor):
    """AutoML predictor for time series classification.

    This predictor automates the complete time series classification pipeline
    using state-of-the-art ROCKET-family classifiers (MiniROCKET, HYDRA,
    MultiROCKET) which are the fastest and most accurate methods for time
    series classification.

    Parameters
    ----------
    label : str
        Name of the target column.
    time_column : str, optional
        Name of the column containing time values (for wide-format data).
    problem_type : str, default="auto"
        Type of problem: "classification" or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "medium_quality", "fast".
    time_limit : int, optional
        Time limit in seconds.
    classifier_presets : list of str, optional
        List of classifiers to try. Options: "minirocket", "rocket",
        "multirocket", "hydra", "hydra_minirocket". If None, auto-selects.
    use_feature_extraction : bool, default=False
        Whether to also extract statistical features.
    feature_preset : str, default="efficient"
        Feature extraction preset: "minimal", "efficient", "comprehensive".
    track_experiments : bool, default=True
        Whether to track experiments.
    output_path : str, optional
        Path to save outputs.
    random_state : int, default=42
        Random seed.
    verbosity : int, default=2
        Verbosity level.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the predictor has been fitted.
    fit_summary_ : FitSummary
        Summary of the fitting process.
    problem_type_ : str
        Detected or specified problem type.
    classes_ : np.ndarray
        Class labels for classification.

    Examples
    --------
    Basic usage with 3D numpy array (n_samples, n_channels, n_timesteps):

    >>> predictor = TimeSeriesPredictor(label="activity")
    >>> predictor.fit(X_train, y_train)
    >>> predictions = predictor.predict(X_test)

    With DataFrame (wide format - each row is a time series):

    >>> predictor = TimeSeriesPredictor(label="class")
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    With multiple classifiers:

    >>> predictor = TimeSeriesPredictor(
    ...     label="class",
    ...     classifier_presets=["minirocket", "hydra_minirocket"],
    ...     presets="best_quality",
    ... )
    >>> predictor.fit(X_train, y_train)
    >>> predictions = predictor.predict(X_test)

    Notes
    -----
    The ROCKET-family classifiers are state-of-the-art for time series
    classification, achieving top accuracy on the UCR archive while being
    orders of magnitude faster than other methods.

    - MiniROCKET: Recommended default. 75x faster than ROCKET with similar accuracy.
    - HYDRA: Dictionary-based approach, complements ROCKET.
    - HYDRA+MiniROCKET: Best accuracy by combining both approaches.
    """

    def __init__(
        self,
        label: str,
        time_column: str | None = None,
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        classifier_presets: list[str] | None = None,
        use_feature_extraction: bool = False,
        feature_preset: str = "efficient",
        track_experiments: bool = True,
        output_path: str | None = None,
        random_state: int = 42,
        verbosity: int = 2,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            presets=presets,
            time_limit=time_limit,
            search_strategy="portfolio",
            track_experiments=track_experiments,
            output_path=output_path,
            random_state=random_state,
            verbosity=verbosity,
        )

        self.time_column = time_column
        self.classifier_presets = classifier_presets
        self.use_feature_extraction = use_feature_extraction
        self.feature_preset = feature_preset

        # Time series-specific state
        self._feature_extractor = None
        self._input_shape = None

    def fit(
        self,
        train_data: DataInput | np.ndarray,
        y: np.ndarray | None = None,
        tuning_data: DataInput | tuple[np.ndarray, np.ndarray] | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> TimeSeriesPredictor:
        """Fit the TimeSeriesPredictor on time series data.

        Parameters
        ----------
        train_data : str, Path, DataFrame, or ndarray
            Training data. Can be:
            - DataFrame with label column (wide format)
            - 3D numpy array (n_samples, n_channels, n_timesteps)
            - 2D numpy array (n_samples, n_timesteps) for univariate
        y : ndarray, optional
            Target labels (required if train_data is numpy array).
        tuning_data : optional
            Validation data in same format as train_data.
        time_limit : int, optional
            Override time limit.
        presets : str, optional
            Override preset.
        hyperparameters : dict, optional
            Override hyperparameters.
        **kwargs
            Additional arguments.

        Returns
        -------
        TimeSeriesPredictor
            The fitted predictor.
        """
        start_time = time.time()

        # Override settings if specified
        if time_limit is not None:
            self.time_limit = time_limit
        if presets is not None:
            self.presets = presets
            self._preset_config = get_preset(presets)

        if self.verbosity > 0:
            print("Beginning TimeSeriesPredictor training")
            print(f"  Presets: {self.presets}")

        # 1. Load and validate data
        X_train, y_train = self._load_ts_data(train_data, y)
        self._input_shape = X_train.shape

        if self.verbosity > 0:
            print(f"  Input shape: {X_train.shape}")
            if len(X_train.shape) == 3:
                print(f"    Samples: {X_train.shape[0]}, Channels: {X_train.shape[1]}, Timesteps: {X_train.shape[2]}")
            else:
                print(f"    Samples: {X_train.shape[0]}, Timesteps: {X_train.shape[1]}")

        # 2. Load tuning data if provided
        X_val, y_val = None, None
        if tuning_data is not None:
            if isinstance(tuning_data, tuple):
                X_val, y_val = self._load_ts_data(tuning_data[0], tuning_data[1])
            else:
                X_val, y_val = self._load_ts_data(tuning_data, None)

        # 3. Detect problem type
        self.problem_type_ = self._detect_problem_type(y_train)

        if self.verbosity > 0:
            print(f"  Problem type: {self.problem_type_}")
            print(f"  Training samples: {len(X_train)}")

        # Store classes for classification
        if self.problem_type_ in ("binary", "multiclass"):
            self.classes_ = np.unique(y_train)
            if self.verbosity > 0:
                print(f"  Classes: {len(self.classes_)}")

        # 4. Optional feature extraction
        X_features_train = None
        X_features_val = None
        if self.use_feature_extraction:
            X_features_train = self._extract_features(X_train)
            if X_val is not None:
                X_features_val = self._extract_features(X_val)

        # 5. Select and train classifiers
        classifiers = self._select_classifiers()
        trained_models = self._train_classifiers(
            classifiers, X_train, y_train, X_val, y_val,
            X_features_train, X_features_val, hyperparameters
        )

        # 6. Build ensemble (if multiple models)
        if len(trained_models) > 1:
            self._build_ensemble(trained_models, X_val, y_val)
        else:
            best_name = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            self._ensemble = trained_models[best_name]["estimator"]

        # 7. Build fit summary
        total_time = time.time() - start_time
        self._build_fit_summary(trained_models, total_time)

        self.is_fitted_ = True

        if self.verbosity > 0:
            print("\nTraining complete!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Best score: {self.fit_summary_.best_score:.4f}")
            print(f"  Best model: {self.fit_summary_.best_model}")

        return self

    def predict(
        self,
        data: DataInput | np.ndarray,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data in same format as training.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._check_is_fitted()

        # Load data
        X, _ = self._load_ts_data(data, None)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
            predictions = estimator.predict(X)
        elif self._ensemble is not None:
            if hasattr(self._ensemble, "predict"):
                predictions = self._ensemble.predict(X)
            else:
                predictions = self._ensemble.predict(X)
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]
            predictions = estimator.predict(X)

        return predictions

    def predict_proba(
        self,
        data: DataInput | np.ndarray,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data in same format as training.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Probability predictions.
        """
        self._check_is_fitted()

        if self.problem_type_ == "regression":
            raise ValueError("predict_proba not available for regression")

        # Load data
        X, _ = self._load_ts_data(data, None)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            estimator = self._ensemble
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X)
        else:
            raise ValueError("Model does not support predict_proba")

    def _load_ts_data(
        self,
        data: DataInput | np.ndarray,
        y: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Load and validate time series data.

        Parameters
        ----------
        data : various
            Input data.
        y : ndarray, optional
            Target labels.

        Returns
        -------
        tuple
            (X, y) as numpy arrays.
        """
        # Handle numpy arrays directly
        if isinstance(data, np.ndarray):
            X = data
            # Ensure 3D shape (n_samples, n_channels, n_timesteps)
            if len(X.shape) == 2:
                X = X[:, np.newaxis, :]  # Add channel dimension
            return X, y

        # Handle DataFrames
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Extract labels
        if self.label in df.columns:
            y = df[self.label].values
            df = df.drop(columns=[self.label])
        elif y is None:
            raise ValueError(f"Label column '{self.label}' not found and y not provided")

        # Convert to numpy array (wide format - each row is a time series)
        X = df.values.astype(np.float32)

        # Add channel dimension if needed
        if len(X.shape) == 2:
            X = X[:, np.newaxis, :]

        return X, y

    def _extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract statistical features from time series.

        Parameters
        ----------
        X : ndarray
            Time series data (n_samples, n_channels, n_timesteps).

        Returns
        -------
        ndarray
            Extracted features.
        """
        if self.verbosity > 0:
            print(f"  Extracting features ({self.feature_preset} preset)...")

        try:
            from endgame.timeseries import TimeSeriesFeatureExtractor

            if self._feature_extractor is None:
                self._feature_extractor = TimeSeriesFeatureExtractor()

            # Reshape to 2D for feature extraction (n_samples, n_timesteps)
            # Concatenate channels if multivariate
            if len(X.shape) == 3:
                X_2d = X.reshape(X.shape[0], -1)
            else:
                X_2d = X

            features = self._feature_extractor.fit_transform(X_2d)

            if self.verbosity > 0:
                print(f"    Extracted {features.shape[1]} features")

            return features

        except ImportError:
            logger.warning("TimeSeriesFeatureExtractor not available")
            return None

    def _select_classifiers(self) -> list[str]:
        """Select time series classifiers to train.

        Returns
        -------
        list of str
            Classifier names to train.
        """
        if self.classifier_presets is not None:
            return self.classifier_presets

        # Select based on preset
        if self.presets == "fast":
            return [DEFAULT_CLASSIFIERS["fast"]]
        elif self.presets in ("medium_quality", "good_quality"):
            return [DEFAULT_CLASSIFIERS["medium"]]
        elif self.presets in ("high_quality", "best_quality"):
            return [DEFAULT_CLASSIFIERS["medium"], DEFAULT_CLASSIFIERS["best"]]
        else:
            return [DEFAULT_CLASSIFIERS["medium"]]

    def _train_classifiers(
        self,
        classifier_names: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        X_features_train: np.ndarray | None,
        X_features_val: np.ndarray | None,
        hyperparameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Train time series classifiers.

        Parameters
        ----------
        classifier_names : list
            Classifiers to train.
        X_train : ndarray
            Training time series.
        y_train : ndarray
            Training labels.
        X_val : ndarray, optional
            Validation time series.
        y_val : ndarray, optional
            Validation labels.
        X_features_train : ndarray, optional
            Training features.
        X_features_val : ndarray, optional
            Validation features.
        hyperparameters : dict, optional
            Override hyperparameters.

        Returns
        -------
        dict
            Trained classifiers with metadata.
        """
        trained = {}

        for clf_name in classifier_names:
            if self.verbosity > 0:
                print(f"Training {clf_name}...")

            start_time = time.time()

            try:
                # Get config
                config = CLASSIFIER_CONFIGS.get(clf_name, CLASSIFIER_CONFIGS["minirocket"]).copy()
                if hyperparameters:
                    config.update(hyperparameters)

                # Create classifier
                estimator = self._create_ts_classifier(clf_name, config)

                # Train
                estimator.fit(X_train, y_train)

                # Get validation score
                if X_val is not None and y_val is not None:
                    y_pred = estimator.predict(X_val)
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, y_pred)
                else:
                    # Use training accuracy as fallback
                    y_pred = estimator.predict(X_train)
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_train, y_pred)

                fit_time = time.time() - start_time

                trained[clf_name] = {
                    "estimator": estimator,
                    "score": score,
                    "fit_time": fit_time,
                }

                self._models[clf_name] = trained[clf_name]

                if self.verbosity > 0:
                    print(f"  {clf_name}: score={score:.4f}, time={fit_time:.1f}s")

            except Exception as e:
                logger.warning(f"Failed to train {clf_name}: {e}")
                if self.verbosity > 0:
                    print(f"  {clf_name}: FAILED - {e}")

        return trained

    def _create_ts_classifier(
        self,
        clf_name: str,
        config: dict[str, Any],
    ):
        """Create a time series classifier.

        Parameters
        ----------
        clf_name : str
            Classifier name.
        config : dict
            Configuration.

        Returns
        -------
        estimator
            Time series classifier.
        """
        try:
            from endgame.timeseries import (
                HydraClassifier,
                HydraMiniRocketClassifier,
                MiniRocketClassifier,
                MultiRocketClassifier,
                RocketClassifier,
            )

            if clf_name == "minirocket":
                return MiniRocketClassifier(
                    n_kernels=config.get("n_kernels", 10000),
                    random_state=self.random_state,
                )
            elif clf_name == "rocket":
                return RocketClassifier(
                    n_kernels=config.get("n_kernels", 10000),
                    random_state=self.random_state,
                )
            elif clf_name == "multirocket":
                return MultiRocketClassifier(
                    n_kernels=config.get("n_kernels", 10000),
                    random_state=self.random_state,
                )
            elif clf_name == "hydra":
                return HydraClassifier(
                    n_groups=config.get("n_groups", 64),
                    n_kernels_per_group=config.get("n_kernels_per_group", 8),
                    random_state=self.random_state,
                )
            elif clf_name == "hydra_minirocket":
                return HydraMiniRocketClassifier(
                    n_kernels=config.get("n_kernels", 10000),
                    n_groups=config.get("n_groups", 64),
                    random_state=self.random_state,
                )
            else:
                # Default to MiniRocket
                return MiniRocketClassifier(
                    n_kernels=config.get("n_kernels", 10000),
                    random_state=self.random_state,
                )

        except ImportError as e:
            raise ImportError(
                f"Time series classifier '{clf_name}' requires additional dependencies. "
                f"Install with: pip install endgame-ml[timeseries]. Error: {e}"
            )

    def _build_ensemble(
        self,
        trained_models: dict[str, Any],
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Build ensemble from trained classifiers.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        X_val : ndarray, optional
            Validation data.
        y_val : ndarray, optional
            Validation labels.
        """
        if len(trained_models) < 2:
            return

        # Simple weighted averaging based on validation scores
        total_score = sum(m["score"] for m in trained_models.values())
        if total_score > 0:
            weights = {name: m["score"] / total_score for name, m in trained_models.items()}
        else:
            weights = {name: 1.0 / len(trained_models) for name in trained_models}

        # Create ensemble wrapper
        self._ensemble = _TSEnsemble(
            models={name: m["estimator"] for name, m in trained_models.items()},
            weights=weights,
            classes=self.classes_,
        )

    def _build_fit_summary(
        self,
        trained_models: dict[str, Any],
        total_time: float,
    ) -> None:
        """Build the fit summary.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        total_time : float
            Total training time.
        """
        n_trained = len(trained_models)

        best_model = ""
        best_score = 0.0
        if trained_models:
            best_model = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            best_score = trained_models[best_model].get("score", 0.0)

        stage_times = {name: m.get("fit_time", 0) for name, m in trained_models.items()}

        self.fit_summary_ = FitSummary(
            total_time=total_time,
            n_models_trained=n_trained,
            n_models_failed=0,
            best_model=best_model,
            best_score=best_score,
            cv_score=best_score,
            stage_times=stage_times,
        )


class _TSEnsemble:
    """Simple weighted ensemble for time series classifiers."""

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float],
        classes: np.ndarray,
    ):
        self.models = models
        self.weights = weights
        self.classes = classes

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using majority voting."""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted averaging."""
        proba = None
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            weight = self.weights.get(name, 0) / total_weight

            if weight > 0:
                if hasattr(model, "predict_proba"):
                    model_proba = model.predict_proba(X)
                else:
                    # Convert predictions to one-hot
                    preds = model.predict(X)
                    model_proba = np.zeros((len(preds), len(self.classes)))
                    for i, p in enumerate(preds):
                        idx = np.where(self.classes == p)[0]
                        if len(idx) > 0:
                            model_proba[i, idx[0]] = 1.0

                if proba is None:
                    proba = weight * model_proba
                else:
                    proba += weight * model_proba

        return proba if proba is not None else np.zeros((len(X), len(self.classes)))
