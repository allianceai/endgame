"""Audio data AutoML predictor.

This module provides the AudioPredictor class for automated machine learning
on audio classification and sound event detection tasks.
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


# Default audio models for different scenarios
DEFAULT_MODELS = {
    "fast": "cnn6",
    "medium": "cnn10",
    "best": "cnn14",
    "efficientnet": "efficientnet",
}

# Model configurations
MODEL_CONFIGS = {
    "cnn6": {
        "encoder": "cnn6",
        "hidden_dim": 256,
        "n_epochs": 30,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
    "cnn10": {
        "encoder": "cnn10",
        "hidden_dim": 256,
        "n_epochs": 50,
        "batch_size": 32,
        "learning_rate": 1e-3,
    },
    "cnn14": {
        "encoder": "cnn14",
        "hidden_dim": 512,
        "n_epochs": 50,
        "batch_size": 16,
        "learning_rate": 5e-4,
    },
    "efficientnet": {
        "encoder": "efficientnet",
        "hidden_dim": 256,
        "n_epochs": 50,
        "batch_size": 16,
        "learning_rate": 5e-4,
    },
}


class AudioPredictor(BasePredictor):
    """AutoML predictor for audio classification and sound event detection.

    This predictor automates the complete audio classification pipeline
    including spectrogram conversion, data augmentation, model training,
    and sound event detection.

    Parameters
    ----------
    label : str
        Name of the target column.
    audio_column : str, default="audio"
        Name of the column containing audio paths or waveforms.
    problem_type : str, default="auto"
        Type of problem: "classification", "multilabel", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "medium_quality", "fast".
    time_limit : int, optional
        Time limit in seconds.
    model_presets : list of str, optional
        List of encoder models to try. Options: "cnn6", "cnn10", "cnn14",
        "efficientnet". If None, auto-selects.
    sample_rate : int, default=32000
        Audio sample rate in Hz.
    n_mels : int, default=128
        Number of mel frequency bins.
    spectrogram_type : str, default="mel"
        Type of spectrogram: "mel" or "pcen".
    use_augmentation : bool, default=True
        Whether to use audio augmentation during training.
    augmentations : list of str, optional
        Augmentations to apply. Default is ["noise", "gain", "timeshift"].
    use_mixup : bool, default=False
        Whether to use mixup augmentation.
    mixup_alpha : float, default=0.4
        Mixup interpolation strength.
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
    audio_column_ : str
        Detected or specified audio column.

    Examples
    --------
    Basic usage with audio file paths:

    >>> predictor = AudioPredictor(label="species", audio_column="filepath")
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    With waveform arrays and custom settings:

    >>> predictor = AudioPredictor(
    ...     label="class",
    ...     sample_rate=22050,
    ...     n_mels=64,
    ...     model_presets=["cnn14", "efficientnet"],
    ...     use_mixup=True,
    ...     presets="best_quality",
    ... )
    >>> predictor.fit(X_train, y_train)  # X_train is array of waveforms
    >>> predictions = predictor.predict(X_test)

    Sound event detection with frame-level predictions:

    >>> predictor = AudioPredictor(label="event", presets="best_quality")
    >>> predictor.fit(train_df)
    >>> frame_probs = predictor.predict_frames(test_df)  # (n_samples, n_frames, n_classes)
    """

    def __init__(
        self,
        label: str,
        audio_column: str = "audio",
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        model_presets: list[str] | None = None,
        sample_rate: int = 32000,
        n_mels: int = 128,
        spectrogram_type: str = "mel",
        use_augmentation: bool = True,
        augmentations: list[str] | None = None,
        use_mixup: bool = False,
        mixup_alpha: float = 0.4,
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

        self.audio_column = audio_column
        self.model_presets = model_presets
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.spectrogram_type = spectrogram_type
        self.use_augmentation = use_augmentation
        self.augmentations = augmentations or ["noise", "gain", "timeshift"]
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

        # Audio-specific state
        self.audio_column_: str | None = None
        self._spectrogram_transformer = None
        self._audio_augmentation = None

    def fit(
        self,
        train_data: DataInput | np.ndarray,
        y: np.ndarray | None = None,
        tuning_data: DataInput | tuple[np.ndarray, np.ndarray] | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> "AudioPredictor":
        """Fit the AudioPredictor on audio data.

        Parameters
        ----------
        train_data : str, Path, DataFrame, or ndarray
            Training data. Can be:
            - DataFrame with audio paths and label columns
            - 2D numpy array of waveforms (n_samples, n_samples_audio)
            - 3D numpy array of spectrograms (n_samples, n_mels, n_frames)
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
        AudioPredictor
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
            print("Beginning AudioPredictor training")
            print(f"  Presets: {self.presets}")
            print(f"  Sample rate: {self.sample_rate} Hz")
            print(f"  Mel bins: {self.n_mels}")

        # 1. Load and validate data
        X_train, y_train = self._load_audio_data(train_data, y)

        if self.verbosity > 0:
            print(f"  Training samples: {len(X_train)}")

        # 2. Load tuning data if provided
        X_val, y_val = None, None
        if tuning_data is not None:
            if isinstance(tuning_data, tuple):
                X_val, y_val = self._load_audio_data(tuning_data[0], tuning_data[1])
            else:
                X_val, y_val = self._load_audio_data(tuning_data, None)

        # 3. Detect problem type
        self.problem_type_ = self._detect_problem_type(y_train)

        if self.verbosity > 0:
            print(f"  Problem type: {self.problem_type_}")

        # Store classes for classification
        if self.problem_type_ in ("binary", "multiclass"):
            self.classes_ = np.unique(y_train)
            if self.verbosity > 0:
                print(f"  Classes: {len(self.classes_)}")

        # 4. Setup spectrogram transformer
        self._setup_spectrogram_transformer()

        # 5. Convert to spectrograms
        X_spec_train = self._convert_to_spectrograms(X_train)
        X_spec_val = None
        if X_val is not None:
            X_spec_val = self._convert_to_spectrograms(X_val)

        if self.verbosity > 0:
            print(f"  Spectrogram shape: {X_spec_train[0].shape}")

        # 6. Setup augmentation
        if self.use_augmentation:
            self._setup_augmentation()

        # 7. Select and train models
        models = self._select_models()
        trained_models = self._train_models(
            models, X_spec_train, y_train, X_spec_val, y_val, hyperparameters
        )

        # 8. Build ensemble (if multiple models)
        if len(trained_models) > 1:
            self._build_ensemble(trained_models, X_spec_val, y_val)
        else:
            best_name = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            self._ensemble = trained_models[best_name]["estimator"]

        # 9. Build fit summary
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
        """Generate predictions (clip-level).

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

        # Load and convert data
        X, _ = self._load_audio_data(data, None)
        X_spec = self._convert_to_spectrograms(X)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
            predictions = estimator.predict(X_spec)
        elif self._ensemble is not None:
            predictions = self._ensemble.predict(X_spec)
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]
            predictions = estimator.predict(X_spec)

        return predictions

    def predict_proba(
        self,
        data: DataInput | np.ndarray,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions (clip-level).

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

        # Load and convert data
        X, _ = self._load_audio_data(data, None)
        X_spec = self._convert_to_spectrograms(X)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            estimator = self._ensemble
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(X_spec)
        else:
            raise ValueError("Model does not support predict_proba")

    def predict_frames(
        self,
        data: DataInput | np.ndarray,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate frame-level predictions for sound event detection.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data in same format as training.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Frame-level probabilities (n_samples, n_frames, n_classes).
        """
        self._check_is_fitted()

        # Load and convert data
        X, _ = self._load_audio_data(data, None)
        X_spec = self._convert_to_spectrograms(X)

        # Get frame-level predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            # Use best model for frame predictions
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        if hasattr(estimator, "predict_frames"):
            return estimator.predict_frames(X_spec)
        else:
            raise ValueError("Model does not support frame-level predictions")

    def _load_audio_data(
        self,
        data: DataInput | np.ndarray,
        y: np.ndarray | None,
    ) -> tuple[list, np.ndarray | None]:
        """Load and validate audio data.

        Parameters
        ----------
        data : various
            Input data.
        y : ndarray, optional
            Target labels.

        Returns
        -------
        tuple
            (audio_data, y) where audio_data is list of waveforms or paths.
        """
        # Handle numpy arrays directly (waveforms or spectrograms)
        if isinstance(data, np.ndarray):
            if len(data.shape) == 2:
                # Could be waveforms (n_samples, n_audio_samples) or spectrograms
                X = list(data)
            elif len(data.shape) == 3:
                # Spectrograms (n_samples, n_mels, n_frames)
                X = list(data)
            else:
                X = list(data)
            return X, y

        # Handle DataFrames
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Detect audio column if not specified
        if self.audio_column_ is None:
            self.audio_column_ = self._detect_audio_column(df)

        if self.audio_column_ not in df.columns:
            raise ValueError(f"Audio column '{self.audio_column_}' not found")

        audio_data = df[self.audio_column_].tolist()

        # Get labels
        if self.label in df.columns:
            y = df[self.label].values
        elif y is None:
            pass  # y may be None for prediction

        return audio_data, y

    def _detect_audio_column(self, df: pd.DataFrame) -> str:
        """Detect the audio column in the dataframe.

        Parameters
        ----------
        df : DataFrame
            Input dataframe.

        Returns
        -------
        str
            Detected audio column name.
        """
        # First check if specified audio_column exists
        if self.audio_column in df.columns:
            return self.audio_column

        # Look for common audio column names
        common_names = [
            "audio", "audio_path", "filepath", "file_path", "path",
            "filename", "file", "wav", "wav_path", "sound", "clip",
        ]
        for name in common_names:
            if name in df.columns:
                return name

        # Look for columns ending with common audio extensions
        for col in df.columns:
            if col == self.label:
                continue
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                sample_str = sample.astype(str)
                if sample_str.str.contains(r"\.(?:wav|mp3|flac|ogg|m4a)$", case=False).any():
                    if self.verbosity > 0:
                        print(f"  Auto-detected audio column: {col}")
                    return col

        raise ValueError("Could not detect audio column. Please specify audio_column parameter.")

    def _setup_spectrogram_transformer(self) -> None:
        """Setup spectrogram transformer."""
        try:
            if self.spectrogram_type == "pcen":
                from endgame.audio import PCENTransformer
                self._spectrogram_transformer = PCENTransformer(
                    sample_rate=self.sample_rate,
                    n_mels=self.n_mels,
                )
            else:
                from endgame.audio import SpectrogramTransformer
                self._spectrogram_transformer = SpectrogramTransformer(
                    sample_rate=self.sample_rate,
                    n_mels=self.n_mels,
                )

            self._spectrogram_transformer.fit(None)

            if self.verbosity > 0:
                print(f"  Spectrogram type: {self.spectrogram_type}")

        except ImportError:
            logger.warning("SpectrogramTransformer not available, using raw waveforms")

    def _setup_augmentation(self) -> None:
        """Setup audio augmentation."""
        try:
            from endgame.audio import AudioAugmentation

            self._audio_augmentation = AudioAugmentation(
                augmentations=self.augmentations,
                sample_rate=self.sample_rate,
                p=0.5,
            )

            if self.verbosity > 0:
                print(f"  Augmentations: {self.augmentations}")

        except ImportError:
            logger.debug("AudioAugmentation not available")

    def _convert_to_spectrograms(self, audio_data: list) -> np.ndarray:
        """Convert audio data to spectrograms.

        Parameters
        ----------
        audio_data : list
            List of waveforms or file paths.

        Returns
        -------
        ndarray
            Spectrograms (n_samples, n_mels, n_frames).
        """
        if self._spectrogram_transformer is None:
            # Assume data is already spectrograms
            return np.array(audio_data)

        spectrograms = []
        for audio in audio_data:
            # Load audio if it's a path
            if isinstance(audio, (str, Path)):
                try:
                    import librosa
                    waveform, _ = librosa.load(audio, sr=self.sample_rate)
                except ImportError:
                    raise ImportError("librosa is required for loading audio files")
            else:
                waveform = np.array(audio)

            # Convert to spectrogram
            spec = self._spectrogram_transformer.transform(waveform.reshape(1, -1))
            spectrograms.append(spec[0])

        return np.array(spectrograms)

    def _select_models(self) -> list[str]:
        """Select audio models to train.

        Returns
        -------
        list of str
            Model names to train.
        """
        if self.model_presets is not None:
            return self.model_presets

        # Select based on preset
        if self.presets == "fast":
            return [DEFAULT_MODELS["fast"]]
        elif self.presets in ("medium_quality", "good_quality"):
            return [DEFAULT_MODELS["medium"]]
        elif self.presets in ("high_quality", "best_quality"):
            return [DEFAULT_MODELS["medium"], DEFAULT_MODELS["best"]]
        else:
            return [DEFAULT_MODELS["medium"]]

    def _train_models(
        self,
        model_names: list[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
        hyperparameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Train audio models.

        Parameters
        ----------
        model_names : list
            Models to train.
        X_train : ndarray
            Training spectrograms.
        y_train : ndarray
            Training labels.
        X_val : ndarray, optional
            Validation spectrograms.
        y_val : ndarray, optional
            Validation labels.
        hyperparameters : dict, optional
            Override hyperparameters.

        Returns
        -------
        dict
            Trained models with metadata.
        """
        trained = {}

        for model_name in model_names:
            if self.verbosity > 0:
                print(f"Training {model_name}...")

            start_time = time.time()

            try:
                # Get config
                config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["cnn10"]).copy()
                if hyperparameters:
                    config.update(hyperparameters)

                # Add mixup if enabled
                if self.use_mixup:
                    config["mixup_alpha"] = self.mixup_alpha

                # Create and train model
                estimator = self._create_audio_classifier(model_name, config)

                # Prepare eval set
                eval_set = None
                if X_val is not None and y_val is not None:
                    eval_set = (X_val, y_val)

                estimator.fit(X_train, y_train, eval_set=eval_set)

                # Get validation score
                if X_val is not None and y_val is not None:
                    y_pred = estimator.predict(X_val)
                    from sklearn.metrics import accuracy_score
                    score = accuracy_score(y_val, y_pred)
                else:
                    score = 0.0

                fit_time = time.time() - start_time

                trained[model_name] = {
                    "estimator": estimator,
                    "score": score,
                    "fit_time": fit_time,
                }

                self._models[model_name] = trained[model_name]

                if self.verbosity > 0:
                    print(f"  {model_name}: score={score:.4f}, time={fit_time:.1f}s")

            except Exception as e:
                logger.warning(f"Failed to train {model_name}: {e}")
                if self.verbosity > 0:
                    print(f"  {model_name}: FAILED - {e}")

        return trained

    def _create_audio_classifier(
        self,
        model_name: str,
        config: dict[str, Any],
    ):
        """Create an audio classifier.

        Parameters
        ----------
        model_name : str
            Model/encoder name.
        config : dict
            Configuration.

        Returns
        -------
        estimator
            Audio classifier.
        """
        try:
            from endgame.audio import SEDModel

            num_classes = len(self.classes_) if self.classes_ is not None else 2

            return SEDModel(
                n_mels=self.n_mels,
                n_classes=num_classes,
                encoder=config.get("encoder", "cnn10"),
                temporal_model="transformer",
                hidden_dim=config.get("hidden_dim", 256),
                n_heads=4,
                n_layers=2,
                dropout=0.2,
                learning_rate=config.get("learning_rate", 1e-3),
                weight_decay=1e-5,
                n_epochs=config.get("n_epochs", 50),
                batch_size=config.get("batch_size", 32),
                early_stopping=10,
                mixup_alpha=config.get("mixup_alpha", 0.0),
                scheduler="cosine",
            )

        except ImportError:
            raise ImportError(
                "SEDModel requires torch and librosa. "
                "Install with: pip install torch librosa"
            )

    def _build_ensemble(
        self,
        trained_models: dict[str, Any],
        X_val: np.ndarray | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Build ensemble from trained models.

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
        self._ensemble = _AudioEnsemble(
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


class _AudioEnsemble:
    """Simple weighted ensemble for audio models."""

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
        """Make predictions."""
        proba = self.predict_proba(X)
        return self.classes[np.argmax(proba, axis=1)]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        proba = None
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            weight = self.weights.get(name, 0) / total_weight

            if weight > 0 and hasattr(model, "predict_proba"):
                model_proba = model.predict_proba(X)

                if proba is None:
                    proba = weight * model_proba
                else:
                    proba += weight * model_proba

        return proba if proba is not None else np.zeros((len(X), len(self.classes)))
