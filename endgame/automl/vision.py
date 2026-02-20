"""Vision/Image data AutoML predictor.

This module provides the VisionPredictor class for automated machine learning
on image classification tasks using timm backbones.
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


# Default vision models for different scenarios
DEFAULT_MODELS = {
    "fast": "efficientnet_b0",
    "medium": "efficientnet_b3",
    "best": "eva02_base_patch14_224",
    "efficient": "convnext_tiny",
    "transformer": "swin_tiny_patch4_window7_224",
}

# Model configurations with training params
MODEL_CONFIGS = {
    "efficientnet_b0": {
        "image_size": 224,
        "learning_rate": 1e-3,
        "batch_size": 64,
        "n_epochs": 20,
    },
    "efficientnet_b3": {
        "image_size": 300,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "n_epochs": 30,
    },
    "eva02_base_patch14_224": {
        "image_size": 224,
        "learning_rate": 1e-4,
        "batch_size": 32,
        "n_epochs": 30,
    },
    "convnext_tiny": {
        "image_size": 224,
        "learning_rate": 5e-4,
        "batch_size": 48,
        "n_epochs": 25,
    },
    "swin_tiny_patch4_window7_224": {
        "image_size": 224,
        "learning_rate": 5e-4,
        "batch_size": 32,
        "n_epochs": 30,
    },
}


class VisionPredictor(BasePredictor):
    """AutoML predictor for image/vision classification.

    This predictor automates the complete vision pipeline including
    data augmentation, backbone selection, training, and test-time
    augmentation.

    Parameters
    ----------
    label : str
        Name of the target column.
    image_column : str, default="image"
        Name of the column containing image paths or arrays.
    problem_type : str, default="auto"
        Type of problem: "classification", "multilabel", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "medium_quality", "fast".
    time_limit : int, optional
        Time limit in seconds.
    model_presets : list of str, optional
        List of backbone models to try. If None, auto-selects.
    augmentation : str, default="standard"
        Augmentation preset: "light", "standard", "heavy".
    use_tta : bool, default=True
        Whether to use test-time augmentation for predictions.
    tta_augmentations : list of str, optional
        TTA augmentations to use. Default is ["identity", "hflip"].
    track_experiments : bool, default=True
        Whether to track experiments.
    output_path : str, optional
        Path to save outputs and checkpoints.
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
    image_column_ : str
        Detected or specified image column.

    Examples
    --------
    Basic usage with image paths:

    >>> predictor = VisionPredictor(label="species", image_column="image_path")
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    With custom models and TTA:

    >>> predictor = VisionPredictor(
    ...     label="category",
    ...     image_column="filepath",
    ...     model_presets=["efficientnet_b3", "swin_tiny_patch4_window7_224"],
    ...     use_tta=True,
    ...     presets="best_quality",
    ... )
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)
    """

    def __init__(
        self,
        label: str,
        image_column: str = "image",
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        model_presets: list[str] | None = None,
        augmentation: str = "standard",
        use_tta: bool = True,
        tta_augmentations: list[str] | None = None,
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

        self.image_column = image_column
        self.model_presets = model_presets
        self.augmentation = augmentation
        self.use_tta = use_tta
        self.tta_augmentations = tta_augmentations or ["identity", "hflip"]

        # Vision-specific state
        self.image_column_: str | None = None
        self._augmentation_pipeline = None
        self._tta = None
        self._image_size: int = 224

    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> "VisionPredictor":
        """Fit the VisionPredictor on image data.

        Parameters
        ----------
        train_data : str, Path, DataFrame
            Training data with image paths/arrays and label columns.
        tuning_data : optional
            Validation data.
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
        VisionPredictor
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
            print("Beginning VisionPredictor training")
            print(f"  Label: {self.label}")
            print(f"  Image column: {self.image_column}")
            print(f"  Presets: {self.presets}")

        # 1. Load and validate data
        X_train, y_train, images_train = self._load_image_data(train_data)

        # 2. Load tuning data if provided
        X_val, y_val, images_val = None, None, None
        if tuning_data is not None:
            X_val, y_val, images_val = self._load_image_data(tuning_data)

        # 3. Detect problem type
        self.problem_type_ = self._detect_problem_type(y_train)

        if self.verbosity > 0:
            print(f"  Problem type: {self.problem_type_}")
            print(f"  Training samples: {len(images_train)}")

        # Store classes for classification
        if self.problem_type_ in ("binary", "multiclass"):
            self.classes_ = np.unique(y_train)
            if self.verbosity > 0:
                print(f"  Classes: {len(self.classes_)}")

        # 4. Setup augmentation
        self._setup_augmentation()

        # 5. Select and train models
        models = self._select_models()
        trained_models = self._train_models(
            models, images_train, y_train, images_val, y_val, hyperparameters
        )

        # 6. Setup TTA
        if self.use_tta:
            self._setup_tta()

        # 7. Build ensemble (if multiple models)
        if len(trained_models) > 1:
            self._build_ensemble(trained_models, images_val, y_val)
        else:
            best_name = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            self._ensemble = trained_models[best_name]["estimator"]

        # 8. Build fit summary
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
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, DataFrame
            Input data with image column.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._check_is_fitted()

        # Load images
        _, _, images = self._load_image_data(data, require_label=False)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            estimator = self._ensemble
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        # Apply TTA if enabled
        if self.use_tta and self._tta is not None:
            predictions = self._predict_with_tta(estimator, images)
        else:
            predictions = estimator.predict(images)

        return predictions

    def predict_proba(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions.

        Parameters
        ----------
        data : str, Path, DataFrame
            Input data with image column.
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

        # Load images
        _, _, images = self._load_image_data(data, require_label=False)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            estimator = self._ensemble
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        # Apply TTA if enabled
        if self.use_tta and self._tta is not None:
            proba = self._predict_proba_with_tta(estimator, images)
        elif hasattr(estimator, "predict_proba"):
            proba = estimator.predict_proba(images)
        else:
            raise ValueError("Model does not support predict_proba")

        return proba

    def _load_image_data(
        self,
        data: DataInput,
        require_label: bool = True,
    ) -> tuple[Any, np.ndarray | None, list]:
        """Load and validate image data.

        Parameters
        ----------
        data : various
            Input data.
        require_label : bool
            Whether label column is required.

        Returns
        -------
        tuple
            (X, y, images) where images is the image paths or arrays.
        """
        # Load data
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Detect image column if not specified
        if self.image_column_ is None:
            self.image_column_ = self._detect_image_column(df)

        if self.image_column_ not in df.columns:
            raise ValueError(f"Image column '{self.image_column_}' not found")

        images = df[self.image_column_].tolist()

        # Get labels
        if require_label:
            if self.label not in df.columns:
                raise ValueError(f"Label column '{self.label}' not found")
            y = df[self.label].values
            X = df.drop(columns=[self.label])
        else:
            y = None
            X = df

        return X, y, images

    def _detect_image_column(self, df: pd.DataFrame) -> str:
        """Detect the image column in the dataframe.

        Parameters
        ----------
        df : DataFrame
            Input dataframe.

        Returns
        -------
        str
            Detected image column name.
        """
        # First check if specified image_column exists
        if self.image_column in df.columns:
            return self.image_column

        # Look for common image column names
        common_names = [
            "image", "image_path", "filepath", "file_path", "path",
            "filename", "file", "img", "img_path", "image_file",
        ]
        for name in common_names:
            if name in df.columns:
                return name

        # Look for columns ending with common image extensions
        for col in df.columns:
            if col == self.label:
                continue
            sample = df[col].dropna().head(10)
            if len(sample) > 0:
                # Check if values look like file paths
                sample_str = sample.astype(str)
                if sample_str.str.contains(r"\.(?:jpg|jpeg|png|bmp|gif|tiff)$", case=False).any():
                    if self.verbosity > 0:
                        print(f"  Auto-detected image column: {col}")
                    return col

        raise ValueError("Could not detect image column. Please specify image_column parameter.")

    def _setup_augmentation(self) -> None:
        """Setup data augmentation pipeline."""
        try:
            from endgame.vision import AugmentationPipeline

            self._augmentation_pipeline = AugmentationPipeline(
                preset=self.augmentation,
                image_size=self._image_size,
            )

            if self.verbosity > 0:
                print(f"  Augmentation: {self.augmentation}")

        except ImportError:
            logger.debug("AugmentationPipeline not available")

    def _setup_tta(self) -> None:
        """Setup test-time augmentation."""
        try:
            from endgame.vision import TestTimeAugmentation

            self._tta = TestTimeAugmentation(
                augmentations=self.tta_augmentations,
                merge_mode="mean",
            )

            if self.verbosity > 0:
                print(f"  TTA enabled: {self.tta_augmentations}")

        except ImportError:
            logger.debug("TestTimeAugmentation not available")

    def _select_models(self) -> list[str]:
        """Select vision backbone models to train.

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
        images_train: list,
        y_train: np.ndarray,
        images_val: list | None,
        y_val: np.ndarray | None,
        hyperparameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Train vision models.

        Parameters
        ----------
        model_names : list
            Models to train.
        images_train : list
            Training image paths/arrays.
        y_train : ndarray
            Training labels.
        images_val : list, optional
            Validation images.
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
                config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["efficientnet_b0"]).copy()
                if hyperparameters:
                    config.update(hyperparameters)

                # Update image size
                self._image_size = config.get("image_size", 224)

                # Create and train classifier
                estimator = self._create_vision_classifier(model_name, config)

                # Prepare validation set
                eval_set = None
                if images_val is not None and y_val is not None:
                    eval_set = (images_val, y_val)

                estimator.fit(images_train, y_train, eval_set=eval_set)

                # Get validation score
                if images_val is not None and y_val is not None:
                    y_pred = estimator.predict(images_val)
                    if self.problem_type_ in ("binary", "multiclass"):
                        from sklearn.metrics import accuracy_score
                        score = accuracy_score(y_val, y_pred)
                    else:
                        from sklearn.metrics import r2_score
                        score = r2_score(y_val, y_pred)
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

        return trained

    def _create_vision_classifier(
        self,
        model_name: str,
        config: dict[str, Any],
    ):
        """Create a vision classifier.

        Parameters
        ----------
        model_name : str
            Backbone model name.
        config : dict
            Configuration.

        Returns
        -------
        estimator
            Vision classifier.
        """
        try:
            from endgame.vision import VisionBackbone

            num_classes = len(self.classes_) if self.classes_ is not None else 2

            return _VisionClassifier(
                backbone=model_name,
                num_classes=num_classes,
                image_size=config.get("image_size", 224),
                learning_rate=config.get("learning_rate", 1e-3),
                n_epochs=config.get("n_epochs", 20),
                batch_size=config.get("batch_size", 32),
                augmentation_pipeline=self._augmentation_pipeline,
                random_state=self.random_state,
                verbose=self.verbosity > 1,
            )

        except ImportError:
            raise ImportError(
                "VisionClassifier requires torch and timm. "
                "Install with: pip install torch timm"
            )

    def _predict_with_tta(
        self,
        estimator,
        images: list,
    ) -> np.ndarray:
        """Make predictions with TTA.

        Parameters
        ----------
        estimator : object
            Trained estimator.
        images : list
            Images to predict.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if hasattr(self._tta, "predict"):
            return self._tta.predict(estimator, images)
        else:
            return estimator.predict(images)

    def _predict_proba_with_tta(
        self,
        estimator,
        images: list,
    ) -> np.ndarray:
        """Make probability predictions with TTA.

        Parameters
        ----------
        estimator : object
            Trained estimator.
        images : list
            Images to predict.

        Returns
        -------
        np.ndarray
            Probability predictions.
        """
        if hasattr(self._tta, "predict_proba"):
            return self._tta.predict_proba(estimator, images)
        elif hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(images)
        else:
            raise ValueError("Model does not support predict_proba")

    def _build_ensemble(
        self,
        trained_models: dict[str, Any],
        images_val: list | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Build ensemble from trained models.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        images_val : list, optional
            Validation images.
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
        self._ensemble = _VisionEnsemble(
            models={name: m["estimator"] for name, m in trained_models.items()},
            weights=weights,
            task_type=self.problem_type_,
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


class _VisionClassifier:
    """Internal vision classifier using timm backbones."""

    def __init__(
        self,
        backbone: str,
        num_classes: int,
        image_size: int = 224,
        learning_rate: float = 1e-3,
        n_epochs: int = 20,
        batch_size: int = 32,
        augmentation_pipeline=None,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.backbone = backbone
        self.num_classes = num_classes
        self.image_size = image_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.augmentation_pipeline = augmentation_pipeline
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.classes_ = None

    def fit(self, images, y, eval_set=None):
        """Train the classifier."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        # Store classes
        self.classes_ = np.unique(y)

        # Create model
        try:
            from endgame.vision import VisionBackbone

            backbone = VisionBackbone(
                architecture=self.backbone,
                pretrained=True,
                num_classes=self.num_classes,
            )
            self.model_ = backbone.get_model()
        except ImportError:
            import timm
            self.model_ = timm.create_model(
                self.backbone,
                pretrained=True,
                num_classes=self.num_classes,
            )

        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_ = self.model_.to(device)

        # Create datasets
        train_dataset = _ImageDataset(
            images, y, self.image_size,
            augmentation=self.augmentation_pipeline,
            is_training=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )

        val_loader = None
        if eval_set is not None:
            val_images, val_y = eval_set
            val_dataset = _ImageDataset(
                val_images, val_y, self.image_size,
                is_training=False,
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=0,
            )

        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(
            self.model_.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.n_epochs
        )

        # Training loop
        for epoch in range(self.n_epochs):
            self.model_.train()
            total_loss = 0

            for batch_images, batch_labels in train_loader:
                batch_images = batch_images.to(device)
                batch_labels = batch_labels.to(device)

                optimizer.zero_grad()
                outputs = self.model_(batch_images)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            scheduler.step()

            if self.verbose and (epoch + 1) % 5 == 0:
                avg_loss = total_loss / len(train_loader)
                print(f"    Epoch {epoch + 1}/{self.n_epochs}, Loss: {avg_loss:.4f}")

        return self

    def predict(self, images):
        """Predict class labels."""
        proba = self.predict_proba(images)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, images):
        """Predict class probabilities."""
        import torch
        from torch.utils.data import DataLoader

        device = next(self.model_.parameters()).device

        dataset = _ImageDataset(
            images, None, self.image_size,
            is_training=False,
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for batch_images in loader:
                if isinstance(batch_images, tuple):
                    batch_images = batch_images[0]
                batch_images = batch_images.to(device)
                outputs = self.model_(batch_images)
                proba = torch.softmax(outputs, dim=1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)


class _ImageDataset:
    """Simple image dataset for torch DataLoader."""

    def __init__(
        self,
        images,
        labels,
        image_size,
        augmentation=None,
        is_training=True,
    ):
        self.images = images
        self.labels = labels
        self.image_size = image_size
        self.augmentation = augmentation
        self.is_training = is_training

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        import numpy as np
        import torch
        from PIL import Image

        # Load image
        img = self.images[idx]
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
            img = img.resize((self.image_size, self.image_size))
            img = np.array(img)
        elif isinstance(img, np.ndarray):
            if img.shape[-1] != 3:
                img = np.stack([img] * 3, axis=-1)
        else:
            img = np.array(img)

        # Apply augmentation
        if self.is_training and self.augmentation is not None:
            try:
                result = self.augmentation(image=img)
                img = result["image"]
            except Exception:
                pass

        # Convert to tensor (CHW format)
        if isinstance(img, np.ndarray):
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = torch.from_numpy(img)

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.long)
            return img, label
        else:
            return img


class _VisionEnsemble:
    """Simple weighted ensemble for vision models."""

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float],
        task_type: str,
    ):
        self.models = models
        self.weights = weights
        self.task_type = task_type

    def predict(self, images) -> np.ndarray:
        """Make predictions."""
        if self.task_type in ("binary", "multiclass"):
            proba = self.predict_proba(images)
            return np.argmax(proba, axis=1)
        else:
            # Regression: weighted average
            preds = []
            total_weight = sum(self.weights.values())

            for name, model in self.models.items():
                weight = self.weights.get(name, 0) / total_weight
                if weight > 0:
                    preds.append(weight * model.predict(images))

            return sum(preds)

    def predict_proba(self, images) -> np.ndarray:
        """Predict probabilities."""
        proba = None
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            weight = self.weights.get(name, 0) / total_weight

            if weight > 0 and hasattr(model, "predict_proba"):
                model_proba = model.predict_proba(images)

                if proba is None:
                    proba = weight * model_proba
                else:
                    proba += weight * model_proba

        return proba if proba is not None else np.zeros((len(images), 2))
