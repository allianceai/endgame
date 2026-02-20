"""MultiModal data AutoML predictor.

This module provides the MultiModalPredictor class for automated machine learning
on datasets containing multiple modalities (tabular, text, image, audio).
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd

from endgame.automl.base import BasePredictor, DataInput, FitSummary
from endgame.automl.presets import get_preset
from endgame.automl.utils.data_loader import infer_task_type, load_data

if TYPE_CHECKING:
    from endgame.tracking.base import ExperimentLogger

logger = logging.getLogger(__name__)


# Fusion strategy types
FusionStrategy = Literal["late", "weighted", "stacking", "attention", "embedding"]


# Default weights for different modalities (based on typical importance)
DEFAULT_MODALITY_WEIGHTS = {
    "tabular": 0.4,
    "text": 0.3,
    "image": 0.2,
    "audio": 0.1,
}


class MultiModalPredictor(BasePredictor):
    """AutoML predictor for multi-modal data.

    This predictor handles datasets containing multiple data modalities
    (tabular features, text columns, image paths, audio files) and
    combines predictions using various fusion strategies.

    Parameters
    ----------
    label : str
        Name of the target column.
    problem_type : str, default="auto"
        Type of problem: "classification", "regression", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "medium_quality", "fast".
    time_limit : int, optional
        Time limit in seconds. Automatically allocated across modalities.
    fusion_strategy : str, default="late"
        How to combine modality predictions:
        - "late": Simple averaging of predictions
        - "weighted": Weighted averaging (auto-tuned or manual)
        - "stacking": Train a meta-learner on modality predictions
        - "attention": Learned per-sample attention weights via MLP
        - "embedding": Mid-level feature concatenation with tabular model on top
    modality_weights : dict, optional
        Manual weights for each modality (for "weighted" fusion).
        Keys: "tabular", "text", "image", "audio". Values sum to 1.
    tabular_columns : list of str, optional
        Columns to treat as tabular features. Auto-detected if None.
    text_columns : list of str, optional
        Columns containing text data. Auto-detected if None.
    image_columns : list of str, optional
        Columns containing image paths. Auto-detected if None.
    audio_columns : list of str, optional
        Columns containing audio paths. Auto-detected if None.
    enable_tabular : bool, default=True
        Whether to use tabular predictor.
    enable_text : bool, default=True
        Whether to use text predictor.
    enable_image : bool, default=True
        Whether to use image predictor.
    enable_audio : bool, default=True
        Whether to use audio predictor.
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
    modalities_ : list of str
        Active modalities after fitting.
    modality_predictors_ : dict
        Individual predictors for each modality.
    modality_weights_ : dict
        Final weights for each modality.
    modality_scores_ : dict
        Validation scores for each modality.

    Examples
    --------
    Basic multi-modal usage (auto-detects modalities):

    >>> predictor = MultiModalPredictor(label="category")
    >>> predictor.fit(train_df)  # df has text, image paths, and numeric columns
    >>> predictions = predictor.predict(test_df)

    With explicit column specification:

    >>> predictor = MultiModalPredictor(
    ...     label="sentiment",
    ...     text_columns=["review", "title"],
    ...     image_columns=["product_image"],
    ...     tabular_columns=["price", "rating", "num_reviews"],
    ...     fusion_strategy="stacking",
    ... )
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    With manual weights:

    >>> predictor = MultiModalPredictor(
    ...     label="target",
    ...     fusion_strategy="weighted",
    ...     modality_weights={"text": 0.6, "tabular": 0.4},
    ... )
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)
    """

    def __init__(
        self,
        label: str,
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        fusion_strategy: FusionStrategy = "late",
        modality_weights: dict[str, float] | None = None,
        tabular_columns: list[str] | None = None,
        text_columns: list[str] | None = None,
        image_columns: list[str] | None = None,
        audio_columns: list[str] | None = None,
        enable_tabular: bool = True,
        enable_text: bool = True,
        enable_image: bool = True,
        enable_audio: bool = True,
        track_experiments: bool = True,
        output_path: str | None = None,
        random_state: int = 42,
        verbosity: int = 2,
        logger: "ExperimentLogger | None" = None,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            presets=presets,
            time_limit=time_limit,
            search_strategy="portfolio",  # Default for multimodal
            track_experiments=track_experiments,
            output_path=output_path,
            random_state=random_state,
            verbosity=verbosity,
            logger=logger,
        )

        self.fusion_strategy = fusion_strategy
        self.modality_weights = modality_weights
        self.tabular_columns = tabular_columns
        self.text_columns = text_columns
        self.image_columns = image_columns
        self.audio_columns = audio_columns
        self.enable_tabular = enable_tabular
        self.enable_text = enable_text
        self.enable_image = enable_image
        self.enable_audio = enable_audio

        # State after fitting
        self.modalities_: list[str] = []
        self.modality_predictors_: dict[str, BasePredictor] = {}
        self.modality_weights_: dict[str, float] = {}
        self.modality_scores_: dict[str, float] = {}
        self._meta_learner: Any | None = None
        self._attention_model: Any | None = None
        self._embedding_model: Any | None = None
        self._train_data_ref: DataInput | None = None

    def _detect_modalities(
        self,
        df: pd.DataFrame,
    ) -> dict[str, list[str]]:
        """Auto-detect columns for each modality.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.

        Returns
        -------
        dict
            Mapping from modality name to list of columns.
        """
        modality_columns = {
            "tabular": [],
            "text": [],
            "image": [],
            "audio": [],
        }

        # Exclude label column
        columns = [c for c in df.columns if c != self.label]

        for col in columns:
            # Skip if column type already specified
            if self.text_columns and col in self.text_columns:
                modality_columns["text"].append(col)
                continue
            if self.image_columns and col in self.image_columns:
                modality_columns["image"].append(col)
                continue
            if self.audio_columns and col in self.audio_columns:
                modality_columns["audio"].append(col)
                continue
            if self.tabular_columns and col in self.tabular_columns:
                modality_columns["tabular"].append(col)
                continue

            # Auto-detect based on content
            dtype = df[col].dtype

            # Check for image paths
            if dtype == object:  # noqa: E721
                sample = df[col].dropna().head(10)
                if len(sample) > 0:
                    sample_str = str(sample.iloc[0]).lower()

                    # Image detection
                    if any(ext in sample_str for ext in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"]):
                        modality_columns["image"].append(col)
                        continue

                    # Audio detection
                    if any(ext in sample_str for ext in [".wav", ".mp3", ".flac", ".ogg", ".m4a"]):
                        modality_columns["audio"].append(col)
                        continue

                    # Text detection: long strings or high cardinality
                    avg_len = sample.astype(str).str.len().mean()
                    n_unique = df[col].nunique()

                    if avg_len > 50 or (n_unique > 100 and avg_len > 20):
                        modality_columns["text"].append(col)
                        continue

            # Default to tabular
            modality_columns["tabular"].append(col)

        # Apply explicit column specifications (override auto-detection)
        if self.tabular_columns:
            modality_columns["tabular"] = list(self.tabular_columns)
        if self.text_columns:
            modality_columns["text"] = list(self.text_columns)
        if self.image_columns:
            modality_columns["image"] = list(self.image_columns)
        if self.audio_columns:
            modality_columns["audio"] = list(self.audio_columns)

        return modality_columns

    def _allocate_time(
        self,
        modalities: list[str],
        total_time: int,
    ) -> dict[str, int]:
        """Allocate time budget across modalities.

        Parameters
        ----------
        modalities : list of str
            Active modalities.
        total_time : int
            Total time budget in seconds.

        Returns
        -------
        dict
            Time allocation per modality.
        """
        # Reserve time for fusion (10%)
        fusion_time = int(total_time * 0.1)
        modality_time = total_time - fusion_time

        # Allocate based on typical training time
        time_weights = {
            "tabular": 1.0,
            "text": 3.0,  # Transformers are slow
            "image": 2.0,  # CNNs are medium
            "audio": 2.0,  # Similar to images
        }

        active_weights = {m: time_weights[m] for m in modalities}
        total_weight = sum(active_weights.values())

        allocation = {}
        for modality in modalities:
            weight = active_weights[modality] / total_weight
            allocation[modality] = max(60, int(modality_time * weight))  # Min 60s

        allocation["_fusion"] = fusion_time

        return allocation

    def _create_predictors(
        self,
        modality_columns: dict[str, list[str]],
        time_allocation: dict[str, int],
    ) -> dict[str, BasePredictor]:
        """Create predictors for each modality.

        Parameters
        ----------
        modality_columns : dict
            Columns for each modality.
        time_allocation : dict
            Time budget for each modality.

        Returns
        -------
        dict
            Predictor instances for each modality.
        """
        predictors = {}

        # Import here to avoid circular imports
        from endgame.automl.audio import AudioPredictor
        from endgame.automl.tabular import TabularPredictor
        from endgame.automl.text import TextPredictor
        from endgame.automl.vision import VisionPredictor

        # Tabular predictor
        if "tabular" in modality_columns and modality_columns["tabular"] and self.enable_tabular:
            predictors["tabular"] = TabularPredictor(
                label=self.label,
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                presets=self.presets,
                time_limit=time_allocation.get("tabular"),
                random_state=self.random_state,
                verbosity=max(0, self.verbosity - 1),
            )

        # Text predictor
        if "text" in modality_columns and modality_columns["text"] and self.enable_text:
            # Use first text column as primary
            text_col = modality_columns["text"][0]
            predictors["text"] = TextPredictor(
                label=self.label,
                text_column=text_col,
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                presets=self.presets,
                time_limit=time_allocation.get("text"),
                random_state=self.random_state,
                verbosity=max(0, self.verbosity - 1),
            )

        # Image predictor
        if "image" in modality_columns and modality_columns["image"] and self.enable_image:
            # Use first image column
            image_col = modality_columns["image"][0]
            predictors["image"] = VisionPredictor(
                label=self.label,
                image_column=image_col,
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                presets=self.presets,
                time_limit=time_allocation.get("image"),
                random_state=self.random_state,
                verbosity=max(0, self.verbosity - 1),
            )

        # Audio predictor
        if "audio" in modality_columns and modality_columns["audio"] and self.enable_audio:
            # Use first audio column
            audio_col = modality_columns["audio"][0]
            predictors["audio"] = AudioPredictor(
                label=self.label,
                audio_column=audio_col,
                problem_type=self.problem_type,
                eval_metric=self.eval_metric,
                presets=self.presets,
                time_limit=time_allocation.get("audio"),
                random_state=self.random_state,
                verbosity=max(0, self.verbosity - 1),
            )

        return predictors

    def _prepare_modality_data(
        self,
        df: pd.DataFrame,
        modality: str,
        modality_columns: dict[str, list[str]],
    ) -> pd.DataFrame:
        """Prepare data for a specific modality.

        Parameters
        ----------
        df : pd.DataFrame
            Full input data.
        modality : str
            Modality name.
        modality_columns : dict
            Column mapping.

        Returns
        -------
        pd.DataFrame
            Data subset for the modality.
        """
        cols = modality_columns.get(modality, [])

        if modality == "tabular":
            # Include all tabular columns
            return df[cols + [self.label]].copy() if self.label in df.columns else df[cols].copy()
        elif modality in ("text", "image", "audio"):
            # Include primary column for these modalities
            if cols:
                col = cols[0]
                if self.label in df.columns:
                    return df[[col, self.label]].copy()
                else:
                    return df[[col]].copy()

        return df.copy()

    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        **kwargs,
    ) -> "MultiModalPredictor":
        """Fit the multi-modal predictor.

        Parameters
        ----------
        train_data : str, Path, or DataFrame
            Training data with multiple modalities.
        tuning_data : optional
            Validation data.
        time_limit : int, optional
            Override time limit.
        presets : str, optional
            Override preset.
        **kwargs
            Additional arguments passed to modality predictors.

        Returns
        -------
        MultiModalPredictor
            Fitted predictor.
        """
        start_time = time.time()

        # Store data reference for refit_full()
        self._train_data_ref = train_data

        # Load data
        if isinstance(train_data, (str, Path)):
            df = load_data(train_data)
        else:
            df = train_data.copy() if isinstance(train_data, pd.DataFrame) else pd.DataFrame(train_data)

        # Override settings if provided
        if time_limit is not None:
            self.time_limit = time_limit
        if presets is not None:
            self.presets = presets

        # Get preset config
        preset_config = get_preset(self.presets)
        effective_time_limit = self.time_limit or preset_config.time_limit or 900  # 15 min default

        if self.verbosity > 0:
            print(f"MultiModalPredictor: Fitting with {self.presets} preset")
            print(f"Time limit: {effective_time_limit}s")

        # Detect problem type
        if self.problem_type == "auto":
            self.problem_type_ = infer_task_type(df, self.label)
        else:
            self.problem_type_ = self.problem_type

        # Store classes for classification
        if self.problem_type_ in ("classification", "binary", "multiclass"):
            self.classes_ = np.unique(df[self.label].dropna())

        # Detect modalities
        modality_columns = self._detect_modalities(df)

        # Filter to active modalities
        active_modalities = [
            m for m, cols in modality_columns.items()
            if cols and getattr(self, f"enable_{m}", True)
        ]

        if not active_modalities:
            raise ValueError("No active modalities detected. Check your data and settings.")

        self.modalities_ = active_modalities

        if self.verbosity > 0:
            print(f"Detected modalities: {active_modalities}")
            for m in active_modalities:
                print(f"  - {m}: {len(modality_columns[m])} columns")

        # Allocate time
        time_allocation = self._allocate_time(active_modalities, effective_time_limit)

        if self.verbosity > 1:
            print(f"Time allocation: {time_allocation}")

        # Create predictors
        self.modality_predictors_ = self._create_predictors(modality_columns, time_allocation)
        self._modality_columns = modality_columns

        # Fit each modality predictor
        modality_predictions = {}
        for modality, predictor in self.modality_predictors_.items():
            if self.verbosity > 0:
                print(f"\nFitting {modality} predictor...")

            try:
                # Prepare data for this modality
                modality_df = self._prepare_modality_data(df, modality, modality_columns)

                # Fit predictor
                predictor.fit(modality_df, **kwargs)

                # Get OOF predictions if available for fusion training
                if hasattr(predictor, "predict_proba") and self.problem_type_ in ("classification", "binary", "multiclass"):
                    try:
                        preds = predictor.predict_proba(modality_df)
                        modality_predictions[modality] = preds
                    except Exception:
                        preds = predictor.predict(modality_df)
                        modality_predictions[modality] = preds
                else:
                    preds = predictor.predict(modality_df)
                    modality_predictions[modality] = preds

                # Store validation score if available
                if hasattr(predictor, "fit_summary_") and predictor.fit_summary_:
                    self.modality_scores_[modality] = predictor.fit_summary_.best_score

                if self.verbosity > 0:
                    score = self.modality_scores_.get(modality, "N/A")
                    print(f"  {modality} predictor fitted. Score: {score}")

            except Exception as e:
                logger.warning(f"Failed to fit {modality} predictor: {e}")
                if self.verbosity > 0:
                    print(f"  WARNING: {modality} predictor failed: {e}")

        # Remove failed modalities
        self.modalities_ = list(self.modality_predictors_.keys())

        if not self.modalities_:
            raise RuntimeError("All modality predictors failed to fit.")

        # Compute fusion weights
        self._compute_fusion_weights(modality_predictions, df[self.label].values)

        # Train meta-learner for stacking fusion
        if self.fusion_strategy == "stacking" and len(modality_predictions) > 1:
            self._train_meta_learner(modality_predictions, df[self.label].values)

        # Train attention model for attention fusion
        if self.fusion_strategy == "attention" and len(modality_predictions) > 1:
            self._train_attention(modality_predictions, df[self.label].values)

        # Train embedding fusion model
        if self.fusion_strategy == "embedding" and len(modality_predictions) > 1:
            self._train_embedding_fusion(modality_predictions, df[self.label].values)

        # Mark as fitted
        self.is_fitted_ = True

        # Create fit summary
        total_time = time.time() - start_time
        self.fit_summary_ = FitSummary(
            total_time=total_time,
            n_models_trained=len(self.modalities_),
            n_models_failed=len(active_modalities) - len(self.modalities_),
            best_model=max(self.modality_scores_, key=self.modality_scores_.get) if self.modality_scores_ else "",
            best_score=max(self.modality_scores_.values()) if self.modality_scores_ else 0.0,
            stage_times={m: time_allocation.get(m, 0) for m in self.modalities_},
        )

        if self.verbosity > 0:
            print(f"\nMultiModalPredictor fitted in {total_time:.1f}s")
            print(f"Active modalities: {self.modalities_}")
            print(f"Modality weights: {self.modality_weights_}")

        return self

    def _compute_fusion_weights(
        self,
        modality_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Compute weights for fusion.

        Parameters
        ----------
        modality_predictions : dict
            Predictions from each modality.
        y_true : np.ndarray
            True labels.
        """
        if self.fusion_strategy == "late":
            # Equal weights
            n_modalities = len(self.modalities_)
            self.modality_weights_ = {m: 1.0 / n_modalities for m in self.modalities_}

        elif self.fusion_strategy == "weighted":
            if self.modality_weights:
                # Use provided weights
                total = sum(self.modality_weights.get(m, 0) for m in self.modalities_)
                self.modality_weights_ = {
                    m: self.modality_weights.get(m, 0) / total
                    for m in self.modalities_
                }
            elif self.modality_scores_:
                # Auto-tune based on validation scores
                total = sum(self.modality_scores_.values())
                if total > 0:
                    self.modality_weights_ = {
                        m: self.modality_scores_.get(m, 0) / total
                        for m in self.modalities_
                    }
                else:
                    # Fallback to equal weights
                    n = len(self.modalities_)
                    self.modality_weights_ = {m: 1.0 / n for m in self.modalities_}
            else:
                # Use default weights
                active = {m: DEFAULT_MODALITY_WEIGHTS.get(m, 0.25) for m in self.modalities_}
                total = sum(active.values())
                self.modality_weights_ = {m: w / total for m, w in active.items()}

        elif self.fusion_strategy in ("stacking", "attention"):
            # Weights will be learned by meta-learner
            self.modality_weights_ = {m: 1.0 / len(self.modalities_) for m in self.modalities_}

    def _train_meta_learner(
        self,
        modality_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Train meta-learner for stacking fusion.

        Parameters
        ----------
        modality_predictions : dict
            Predictions from each modality.
        y_true : np.ndarray
            True labels.
        """
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.preprocessing import LabelEncoder

        # Stack predictions as features
        X_meta = np.column_stack([
            modality_predictions[m] if modality_predictions[m].ndim == 1
            else modality_predictions[m]
            for m in self.modalities_
            if m in modality_predictions
        ])

        # Handle NaN
        X_meta = np.nan_to_num(X_meta)

        if self.problem_type_ in ("classification", "binary", "multiclass"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_true)
            self._meta_learner = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=self.random_state,
            )
            self._meta_learner.fit(X_meta, y_encoded)
            self._label_encoder = le
        else:
            self._meta_learner = Ridge(alpha=1.0)
            self._meta_learner.fit(X_meta, y_true)

    def predict(
        self,
        data: DataInput,
        **kwargs,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, or DataFrame
            Data to predict on.
        **kwargs
            Additional arguments.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        # Load data
        if isinstance(data, (str, Path)):
            df = load_data(data)
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        # Get predictions from each modality
        modality_preds = {}
        for modality, predictor in self.modality_predictors_.items():
            try:
                modality_df = self._prepare_modality_data(df, modality, self._modality_columns)
                modality_preds[modality] = predictor.predict(modality_df)
            except Exception as e:
                logger.warning(f"Failed to predict with {modality}: {e}")

        if not modality_preds:
            raise RuntimeError("All modality predictions failed.")

        # Fuse predictions
        return self._fuse_predictions(modality_preds, is_proba=False)

    def predict_proba(
        self,
        data: DataInput,
        **kwargs,
    ) -> np.ndarray:
        """Generate probability predictions.

        Parameters
        ----------
        data : str, Path, or DataFrame
            Data to predict on.
        **kwargs
            Additional arguments.

        Returns
        -------
        np.ndarray
            Probability predictions.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        if self.problem_type_ not in ("classification", "binary", "multiclass"):
            raise ValueError("predict_proba only available for classification.")

        # Load data
        if isinstance(data, (str, Path)):
            df = load_data(data)
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        # Get probability predictions from each modality
        modality_preds = {}
        for modality, predictor in self.modality_predictors_.items():
            try:
                modality_df = self._prepare_modality_data(df, modality, self._modality_columns)
                if hasattr(predictor, "predict_proba"):
                    modality_preds[modality] = predictor.predict_proba(modality_df)
                else:
                    # Fallback to hard predictions
                    preds = predictor.predict(modality_df)
                    # Convert to one-hot
                    n_classes = len(self.classes_)
                    proba = np.zeros((len(preds), n_classes))
                    for i, p in enumerate(preds):
                        idx = np.where(self.classes_ == p)[0]
                        if len(idx) > 0:
                            proba[i, idx[0]] = 1.0
                    modality_preds[modality] = proba
            except Exception as e:
                logger.warning(f"Failed to predict_proba with {modality}: {e}")

        if not modality_preds:
            raise RuntimeError("All modality predictions failed.")

        # Fuse probability predictions
        return self._fuse_predictions(modality_preds, is_proba=True)

    def _fuse_predictions(
        self,
        modality_preds: dict[str, np.ndarray],
        is_proba: bool = False,
    ) -> np.ndarray:
        """Fuse predictions from multiple modalities.

        Parameters
        ----------
        modality_preds : dict
            Predictions from each modality.
        is_proba : bool
            Whether these are probability predictions.

        Returns
        -------
        np.ndarray
            Fused predictions.
        """
        if self.fusion_strategy in ("late", "weighted"):
            return self._weighted_fusion(modality_preds, is_proba)
        elif self.fusion_strategy == "stacking":
            return self._stacking_fusion(modality_preds, is_proba)
        elif self.fusion_strategy == "attention":
            return self._attention_fusion(modality_preds, is_proba)
        elif self.fusion_strategy == "embedding":
            return self._embedding_fusion(modality_preds, is_proba)
        else:
            return self._weighted_fusion(modality_preds, is_proba)

    def _weighted_fusion(
        self,
        modality_preds: dict[str, np.ndarray],
        is_proba: bool,
    ) -> np.ndarray:
        """Weighted average fusion.

        Parameters
        ----------
        modality_preds : dict
            Predictions from each modality.
        is_proba : bool
            Whether these are probability predictions.

        Returns
        -------
        np.ndarray
            Fused predictions.
        """
        # Get sample size from first prediction
        first_pred = next(iter(modality_preds.values()))
        n_samples = len(first_pred)

        if is_proba:
            # Average probabilities
            n_classes = first_pred.shape[1] if first_pred.ndim > 1 else len(self.classes_)
            fused = np.zeros((n_samples, n_classes))

            for modality, preds in modality_preds.items():
                weight = self.modality_weights_.get(modality, 0)
                if preds.ndim == 1:
                    # Convert to one-hot
                    for i, p in enumerate(preds):
                        idx = np.where(self.classes_ == p)[0]
                        if len(idx) > 0:
                            fused[i, idx[0]] += weight
                else:
                    fused += weight * preds

            # Normalize
            fused = fused / fused.sum(axis=1, keepdims=True)
            return fused
        else:
            # For hard predictions, use voting or average for regression
            if self.problem_type_ in ("classification", "binary", "multiclass"):
                # Weighted voting
                from collections import Counter
                predictions = []
                for i in range(n_samples):
                    votes = Counter()
                    for modality, preds in modality_preds.items():
                        weight = self.modality_weights_.get(modality, 0)
                        votes[preds[i]] += weight
                    predictions.append(votes.most_common(1)[0][0])
                return np.array(predictions)
            else:
                # Weighted average for regression
                fused = np.zeros(n_samples)
                for modality, preds in modality_preds.items():
                    weight = self.modality_weights_.get(modality, 0)
                    fused += weight * preds.flatten()
                return fused

    def _stacking_fusion(
        self,
        modality_preds: dict[str, np.ndarray],
        is_proba: bool,
    ) -> np.ndarray:
        """Stacking fusion using meta-learner.

        Parameters
        ----------
        modality_preds : dict
            Predictions from each modality.
        is_proba : bool
            Whether to return probabilities.

        Returns
        -------
        np.ndarray
            Fused predictions.
        """
        if self._meta_learner is None:
            # Fallback to weighted fusion
            return self._weighted_fusion(modality_preds, is_proba)

        # Stack predictions
        X_meta = np.column_stack([
            modality_preds[m] if modality_preds[m].ndim == 1
            else modality_preds[m]
            for m in self.modalities_
            if m in modality_preds
        ])
        X_meta = np.nan_to_num(X_meta)

        if is_proba and hasattr(self._meta_learner, "predict_proba"):
            return self._meta_learner.predict_proba(X_meta)
        else:
            preds = self._meta_learner.predict(X_meta)
            if hasattr(self, "_label_encoder"):
                return self._label_encoder.inverse_transform(preds.astype(int))
            return preds

    def _train_attention(
        self,
        modality_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Learn per-sample attention weights via a small MLP.

        The MLP takes concatenated modality predictions as input and
        outputs softmax weights over modalities for each sample.

        Parameters
        ----------
        modality_predictions : dict
            Predictions from each modality.
        y_true : np.ndarray
            True labels.
        """
        from sklearn.neural_network import MLPClassifier, MLPRegressor
        from sklearn.preprocessing import LabelEncoder

        # Build input: concatenated modality predictions
        pred_list = []
        for m in self.modalities_:
            if m in modality_predictions:
                p = modality_predictions[m]
                if p.ndim == 1:
                    pred_list.append(p.reshape(-1, 1))
                else:
                    pred_list.append(p)
        X_attention = np.column_stack(pred_list)
        X_attention = np.nan_to_num(X_attention)

        # Train a model that learns instance-dependent modality weighting
        if self.problem_type_ in ("classification", "binary", "multiclass"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_true)
            self._attention_model = MLPClassifier(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.15,
            )
            self._attention_model.fit(X_attention, y_encoded)
            self._attention_label_encoder = le
        else:
            self._attention_model = MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                random_state=self.random_state,
                early_stopping=True,
                validation_fraction=0.15,
            )
            self._attention_model.fit(X_attention, y_true)

    def _attention_fusion(
        self,
        modality_preds: dict[str, np.ndarray],
        is_proba: bool,
    ) -> np.ndarray:
        """Attention-based fusion using learned MLP.

        Falls back to weighted fusion if the attention model is unavailable.
        """
        if self._attention_model is None:
            return self._weighted_fusion(modality_preds, is_proba)

        # Build input features from modality predictions
        pred_list = []
        for m in self.modalities_:
            if m in modality_preds:
                p = modality_preds[m]
                if p.ndim == 1:
                    pred_list.append(p.reshape(-1, 1))
                else:
                    pred_list.append(p)
        X_attention = np.column_stack(pred_list)
        X_attention = np.nan_to_num(X_attention)

        if is_proba and hasattr(self._attention_model, "predict_proba"):
            return self._attention_model.predict_proba(X_attention)
        else:
            preds = self._attention_model.predict(X_attention)
            if hasattr(self, "_attention_label_encoder"):
                return self._attention_label_encoder.inverse_transform(preds.astype(int))
            return preds

    def _train_embedding_fusion(
        self,
        modality_predictions: dict[str, np.ndarray],
        y_true: np.ndarray,
    ) -> None:
        """Train an embedding fusion model.

        Extracts feature representations from each modality and concatenates
        them, then trains a tabular model (GradientBoosting) on the combined
        feature space.

        Parameters
        ----------
        modality_predictions : dict
            Predictions (used as feature embeddings) from each modality.
        y_true : np.ndarray
            True labels.
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.preprocessing import LabelEncoder

        # Concatenate all modality predictions as embedding features
        embedding_parts = []
        for m in self.modalities_:
            if m in modality_predictions:
                p = modality_predictions[m]
                if p.ndim == 1:
                    embedding_parts.append(p.reshape(-1, 1))
                else:
                    embedding_parts.append(p)

        X_embed = np.column_stack(embedding_parts)
        X_embed = np.nan_to_num(X_embed)

        if self.problem_type_ in ("classification", "binary", "multiclass"):
            le = LabelEncoder()
            y_encoded = le.fit_transform(y_true)
            self._embedding_model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            self._embedding_model.fit(X_embed, y_encoded)
            self._embedding_label_encoder = le
        else:
            self._embedding_model = GradientBoostingRegressor(
                n_estimators=200,
                max_depth=4,
                learning_rate=0.1,
                random_state=self.random_state,
            )
            self._embedding_model.fit(X_embed, y_true)

    def _embedding_fusion(
        self,
        modality_preds: dict[str, np.ndarray],
        is_proba: bool,
    ) -> np.ndarray:
        """Embedding fusion using trained tabular model on concatenated features.

        Falls back to weighted fusion if the embedding model is unavailable.
        """
        if self._embedding_model is None:
            return self._weighted_fusion(modality_preds, is_proba)

        # Concatenate predictions as features
        embedding_parts = []
        for m in self.modalities_:
            if m in modality_preds:
                p = modality_preds[m]
                if p.ndim == 1:
                    embedding_parts.append(p.reshape(-1, 1))
                else:
                    embedding_parts.append(p)

        X_embed = np.column_stack(embedding_parts)
        X_embed = np.nan_to_num(X_embed)

        if is_proba and hasattr(self._embedding_model, "predict_proba"):
            return self._embedding_model.predict_proba(X_embed)
        else:
            preds = self._embedding_model.predict(X_embed)
            if hasattr(self, "_embedding_label_encoder"):
                return self._embedding_label_encoder.inverse_transform(preds.astype(int))
            return preds

    def evaluate(
        self,
        data: DataInput,
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate the predictor.

        Parameters
        ----------
        data : str, Path, or DataFrame
            Data with true labels.
        metrics : list of str, optional
            Metrics to compute.

        Returns
        -------
        dict
            Metric scores.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        # Load data
        if isinstance(data, (str, Path)):
            df = load_data(data)
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        y_true = df[self.label].values
        y_pred = self.predict(df)

        # Compute metrics
        results = {}

        if self.problem_type_ in ("classification", "binary", "multiclass"):
            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            results["accuracy"] = accuracy_score(y_true, y_pred)
            try:
                results["f1"] = f1_score(y_true, y_pred, average="weighted")
            except Exception:
                pass

            if hasattr(self, "predict_proba"):
                try:
                    y_proba = self.predict_proba(df)
                    if len(self.classes_) == 2:
                        results["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
                    else:
                        results["roc_auc"] = roc_auc_score(y_true, y_proba, multi_class="ovr")
                except Exception:
                    pass
        else:
            from sklearn.metrics import mean_squared_error, r2_score

            results["mse"] = mean_squared_error(y_true, y_pred)
            results["rmse"] = np.sqrt(results["mse"])
            results["r2"] = r2_score(y_true, y_pred)

        return results

    def get_modality_contributions(
        self,
        data: DataInput | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Get contribution information for each modality.

        Parameters
        ----------
        data : optional
            Data to analyze contributions on.

        Returns
        -------
        dict
            Contribution info per modality.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        contributions = {}

        for modality in self.modalities_:
            contributions[modality] = {
                "weight": self.modality_weights_.get(modality, 0),
                "score": self.modality_scores_.get(modality, None),
                "predictor_type": type(self.modality_predictors_[modality]).__name__,
            }

        return contributions

    def leaderboard(self) -> pd.DataFrame:
        """Get modality leaderboard.

        Returns
        -------
        pd.DataFrame
            Leaderboard with modality scores and weights.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        data = []
        for modality in self.modalities_:
            data.append({
                "modality": modality,
                "score": self.modality_scores_.get(modality, np.nan),
                "weight": self.modality_weights_.get(modality, 0),
                "predictor": type(self.modality_predictors_[modality]).__name__,
            })

        df = pd.DataFrame(data)
        df = df.sort_values("score", ascending=False)
        return df.reset_index(drop=True)

    def refit_full(self, data: DataInput | None = None) -> "MultiModalPredictor":
        """Retrain all modality models on the full dataset.

        Calls ``refit_full()`` on each modality predictor and re-computes
        fusion weights if using stacking fusion.

        Parameters
        ----------
        data : DataInput, optional
            Full dataset. If None, uses the training data from the last
            ``fit()`` call.

        Returns
        -------
        MultiModalPredictor
            Self with models retrained on full data.
        """
        self._check_is_fitted()

        # Resolve data source
        if data is None:
            data = self._train_data_ref
        if data is None:
            raise ValueError(
                "No data available for refitting. Pass data explicitly "
                "or ensure fit() was called with a reusable data reference."
            )

        # Load data
        if isinstance(data, (str, Path)):
            df = load_data(data)
        else:
            df = data.copy() if isinstance(data, pd.DataFrame) else pd.DataFrame(data)

        logger.warning(
            "refit_full(): Retraining all modality models on full data. "
            "The resulting models cannot be evaluated with a holdout set."
        )

        # Refit each modality predictor
        for modality, predictor in self.modality_predictors_.items():
            try:
                modality_df = self._prepare_modality_data(
                    df, modality, self._modality_columns
                )
                if hasattr(predictor, "refit_full"):
                    predictor.refit_full(modality_df)
                else:
                    predictor.fit(modality_df)

                if self.verbosity > 0:
                    print(f"  Refitted {modality} predictor")
            except Exception as e:
                logger.warning(f"Failed to refit {modality} predictor: {e}")

        return self

    def feature_importance(
        self,
        modality: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """Get feature importance for each modality.

        Parameters
        ----------
        modality : str, optional
            Specific modality to get importance for.

        Returns
        -------
        dict
            Feature importance DataFrames per modality.
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        importance = {}

        modalities = [modality] if modality else self.modalities_

        for m in modalities:
            if m in self.modality_predictors_:
                predictor = self.modality_predictors_[m]
                if hasattr(predictor, "feature_importance"):
                    try:
                        importance[m] = predictor.feature_importance()
                    except Exception:
                        pass

        return importance
