from __future__ import annotations

"""Text data AutoML predictor.

This module provides the TextPredictor class for automated machine learning
on text/NLP data using transformer models.
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


# Default transformer models for different scenarios
DEFAULT_MODELS = {
    "fast": "distilbert-base-uncased",
    "medium": "microsoft/deberta-v3-base",
    "best": "microsoft/deberta-v3-large",
    "multilingual": "xlm-roberta-base",
}

# Model configurations
MODEL_CONFIGS = {
    "distilbert-base-uncased": {
        "max_length": 512,
        "learning_rate": 3e-5,
        "batch_size": 32,
        "num_epochs": 3,
    },
    "microsoft/deberta-v3-base": {
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 5,
    },
    "microsoft/deberta-v3-large": {
        "max_length": 512,
        "learning_rate": 1e-5,
        "batch_size": 8,
        "num_epochs": 5,
    },
    "xlm-roberta-base": {
        "max_length": 512,
        "learning_rate": 2e-5,
        "batch_size": 16,
        "num_epochs": 5,
    },
}


class TextPredictor(BasePredictor):
    """AutoML predictor for text/NLP data.

    This predictor automates the complete NLP pipeline including
    text preprocessing, transformer model selection, domain-adaptive
    pretraining, and pseudo-labeling.

    Parameters
    ----------
    label : str
        Name of the target column.
    text_column : str, default="text"
        Name of the column containing text data.
    problem_type : str, default="auto"
        Type of problem: "classification", "regression", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "medium_quality", "fast".
    time_limit : int, optional
        Time limit in seconds.
    model_presets : list of str, optional
        List of transformer models to try. If None, auto-selects.
    use_dapt : bool, default=False
        Whether to use domain-adaptive pretraining.
    use_pseudo_labeling : bool, default=False
        Whether to use pseudo-labeling with test data.
    pseudo_threshold : float, default=0.95
        Confidence threshold for pseudo-labeling.
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
    text_column_ : str
        Detected or specified text column.

    Examples
    --------
    Basic usage:

    >>> predictor = TextPredictor(label="sentiment", text_column="review")
    >>> predictor.fit(train_df)
    >>> predictions = predictor.predict(test_df)

    With domain-adaptive pretraining:

    >>> predictor = TextPredictor(
    ...     label="category",
    ...     text_column="description",
    ...     use_dapt=True,
    ...     presets="best_quality",
    ... )
    >>> predictor.fit(train_df, unlabeled_data=test_df)
    >>> predictions = predictor.predict(test_df)
    """

    def __init__(
        self,
        label: str,
        text_column: str = "text",
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        model_presets: list[str] | None = None,
        use_dapt: bool = False,
        use_pseudo_labeling: bool = False,
        pseudo_threshold: float = 0.95,
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
            search_strategy="portfolio",  # Text uses model portfolio
            track_experiments=track_experiments,
            output_path=output_path,
            random_state=random_state,
            verbosity=verbosity,
        )

        self.text_column = text_column
        self.model_presets = model_presets
        self.use_dapt = use_dapt
        self.use_pseudo_labeling = use_pseudo_labeling
        self.pseudo_threshold = pseudo_threshold

        # Text-specific state
        self.text_column_: str | None = None
        self._text_preprocessor = None
        self._tokenizer_optimizer = None
        self._dapt_model_path: str | None = None

    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        unlabeled_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        **kwargs,
    ) -> TextPredictor:
        """Fit the TextPredictor on text data.

        Parameters
        ----------
        train_data : str, Path, DataFrame
            Training data with text and label columns.
        tuning_data : optional
            Validation data.
        unlabeled_data : optional
            Unlabeled data for DAPT and pseudo-labeling.
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
        TextPredictor
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
            print("Beginning TextPredictor training")
            print(f"  Label: {self.label}")
            print(f"  Text column: {self.text_column}")
            print(f"  Presets: {self.presets}")

        # 1. Load and validate data
        X_train, y_train, texts_train = self._load_text_data(train_data)

        # 2. Load tuning data if provided
        X_val, y_val, texts_val = None, None, None
        if tuning_data is not None:
            X_val, y_val, texts_val = self._load_text_data(tuning_data)

        # 3. Load unlabeled data for DAPT/pseudo-labeling
        texts_unlabeled = None
        if unlabeled_data is not None:
            _, _, texts_unlabeled = self._load_text_data(unlabeled_data, require_label=False)

        # 4. Detect problem type
        self.problem_type_ = self._detect_problem_type(y_train)

        if self.verbosity > 0:
            print(f"  Problem type: {self.problem_type_}")
            print(f"  Training samples: {len(texts_train)}")

        # Store classes for classification
        if self.problem_type_ in ("binary", "multiclass"):
            self.classes_ = np.unique(y_train)
            if self.verbosity > 0:
                print(f"  Classes: {len(self.classes_)}")

        # 5. Preprocess texts
        texts_train = self._preprocess_texts(texts_train)
        if texts_val is not None:
            texts_val = self._preprocess_texts(texts_val)
        if texts_unlabeled is not None:
            texts_unlabeled = self._preprocess_texts(texts_unlabeled)

        # 6. Domain-adaptive pretraining (if enabled)
        if self.use_dapt and texts_unlabeled is not None:
            self._run_dapt(texts_train, texts_unlabeled)

        # 7. Select and train models
        models = self._select_models()
        trained_models = self._train_models(
            models, texts_train, y_train, texts_val, y_val, hyperparameters
        )

        # 8. Pseudo-labeling (if enabled)
        if self.use_pseudo_labeling and texts_unlabeled is not None:
            self._run_pseudo_labeling(
                trained_models, texts_train, y_train, texts_unlabeled
            )

        # 9. Build ensemble (if multiple models)
        if len(trained_models) > 1:
            self._build_ensemble(trained_models, texts_val, y_val)
        else:
            # Use single best model
            best_name = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            self._ensemble = trained_models[best_name]["estimator"]

        # 10. Build fit summary
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
            Input data with text column.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._check_is_fitted()

        # Load and preprocess text
        _, _, texts = self._load_text_data(data, require_label=False)
        texts = self._preprocess_texts(texts)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
            predictions = estimator.predict(texts)
        elif self._ensemble is not None:
            predictions = self._ensemble.predict(texts)
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]
            predictions = estimator.predict(texts)

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
            Input data with text column.
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

        # Load and preprocess text
        _, _, texts = self._load_text_data(data, require_label=False)
        texts = self._preprocess_texts(texts)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model]["estimator"]
        elif self._ensemble is not None:
            estimator = self._ensemble
        else:
            best_name = self.fit_summary_.best_model
            estimator = self._models[best_name]["estimator"]

        if hasattr(estimator, "predict_proba"):
            return estimator.predict_proba(texts)
        else:
            raise ValueError("Model does not support predict_proba")

    def _load_text_data(
        self,
        data: DataInput,
        require_label: bool = True,
    ) -> tuple:
        """Load and validate text data.

        Parameters
        ----------
        data : various
            Input data.
        require_label : bool
            Whether label column is required.

        Returns
        -------
        tuple
            (X, y, texts) where texts is the text column values.
        """
        # Load data
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        # Detect text column if not specified
        if self.text_column_ is None:
            self.text_column_ = self._detect_text_column(df)

        if self.text_column_ not in df.columns:
            raise ValueError(f"Text column '{self.text_column_}' not found")

        texts = df[self.text_column_].values

        # Get labels
        if require_label:
            if self.label not in df.columns:
                raise ValueError(f"Label column '{self.label}' not found")
            y = df[self.label].values
            X = df.drop(columns=[self.label])
        else:
            y = None
            X = df

        return X, y, texts

    def _detect_text_column(self, df: pd.DataFrame) -> str:
        """Detect the text column in the dataframe.

        Parameters
        ----------
        df : DataFrame
            Input dataframe.

        Returns
        -------
        str
            Detected text column name.
        """
        # First check if specified text_column exists
        if self.text_column in df.columns:
            return self.text_column

        # Look for common text column names
        common_names = ["text", "content", "body", "message", "description", "review", "comment"]
        for name in common_names:
            if name in df.columns:
                return name

        # Find column with longest average string length
        text_col = None
        max_avg_len = 0

        for col in df.columns:
            if col == self.label:
                continue
            if df[col].dtype == object:
                avg_len = df[col].astype(str).str.len().mean()
                if avg_len > max_avg_len:
                    max_avg_len = avg_len
                    text_col = col

        if text_col is not None:
            if self.verbosity > 0:
                print(f"  Auto-detected text column: {text_col}")
            return text_col

        raise ValueError("Could not detect text column. Please specify text_column parameter.")

    def _preprocess_texts(self, texts: np.ndarray) -> list[str]:
        """Preprocess text data.

        Parameters
        ----------
        texts : ndarray
            Raw text data.

        Returns
        -------
        list of str
            Preprocessed texts.
        """
        texts = [str(t) if t is not None else "" for t in texts]

        try:
            from endgame.nlp import TextPreprocessor

            if self._text_preprocessor is None:
                self._text_preprocessor = TextPreprocessor.kaggle_preset()

            texts = self._text_preprocessor.batch(texts)
        except ImportError:
            logger.debug("TextPreprocessor not available, using raw texts")

        return texts

    def _run_dapt(
        self,
        texts_train: list[str],
        texts_unlabeled: list[str],
    ) -> None:
        """Run domain-adaptive pretraining.

        Parameters
        ----------
        texts_train : list
            Training texts.
        texts_unlabeled : list
            Unlabeled texts.
        """
        if self.verbosity > 0:
            print("Running domain-adaptive pretraining...")

        try:
            from endgame.nlp import DomainAdaptivePretrainer

            # Combine all texts for DAPT
            all_texts = texts_train + texts_unlabeled

            # Select base model
            base_model = self._get_base_model_name()

            dapt = DomainAdaptivePretrainer(
                model_name=base_model,
                num_epochs=2,
                batch_size=8,
                verbose=self.verbosity > 1,
            )

            output_dir = self.output_path or "./dapt_model"
            self._dapt_model_path = dapt.pretrain(all_texts, output_dir=output_dir)

            if self.verbosity > 0:
                print(f"  DAPT complete: {self._dapt_model_path}")

        except ImportError:
            logger.warning("DomainAdaptivePretrainer not available, skipping DAPT")
        except Exception as e:
            logger.warning(f"DAPT failed: {e}")

    def _select_models(self) -> list[str]:
        """Select transformer models to train.

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
        texts_train: list[str],
        y_train: np.ndarray,
        texts_val: list[str] | None,
        y_val: np.ndarray | None,
        hyperparameters: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Train transformer models.

        Parameters
        ----------
        model_names : list
            Models to train.
        texts_train : list
            Training texts.
        y_train : ndarray
            Training labels.
        texts_val : list, optional
            Validation texts.
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
                # Use DAPT model if available
                actual_model = self._dapt_model_path or model_name

                # Get config
                config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["microsoft/deberta-v3-base"])
                if hyperparameters:
                    config.update(hyperparameters)

                # Create and train classifier
                estimator = self._create_transformer_classifier(actual_model, config)

                # Prepare validation set
                eval_set = None
                if texts_val is not None and y_val is not None:
                    eval_set = (texts_val, y_val)

                estimator.fit(texts_train, y_train, eval_set=eval_set)

                # Get validation score
                if texts_val is not None and y_val is not None:
                    y_pred = estimator.predict(texts_val)
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

    def _create_transformer_classifier(
        self,
        model_name: str,
        config: dict[str, Any],
    ):
        """Create a transformer classifier.

        Parameters
        ----------
        model_name : str
            Model name or path.
        config : dict
            Configuration.

        Returns
        -------
        estimator
            Transformer classifier.
        """
        try:
            from endgame.nlp import TransformerClassifier

            num_labels = len(self.classes_) if self.classes_ is not None else 2

            return TransformerClassifier(
                model_name=model_name,
                num_labels=num_labels,
                max_length=config.get("max_length", 512),
                learning_rate=config.get("learning_rate", 2e-5),
                num_epochs=config.get("num_epochs", 5),
                batch_size=config.get("batch_size", 16),
                random_state=self.random_state,
                verbose=self.verbosity > 1,
            )

        except ImportError:
            raise ImportError(
                "TransformerClassifier requires the transformers library. "
                "Install with: pip install transformers"
            )

    def _run_pseudo_labeling(
        self,
        trained_models: dict[str, Any],
        texts_train: list[str],
        y_train: np.ndarray,
        texts_unlabeled: list[str],
    ) -> None:
        """Run pseudo-labeling on unlabeled data.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        texts_train : list
            Training texts.
        y_train : ndarray
            Training labels.
        texts_unlabeled : list
            Unlabeled texts.
        """
        if self.verbosity > 0:
            print("Running pseudo-labeling...")

        try:
            from endgame.nlp import PseudoLabelTrainer

            # Use best model for pseudo-labeling
            best_name = max(trained_models.keys(), key=lambda k: trained_models[k].get("score", 0))
            base_estimator = trained_models[best_name]["estimator"]

            pseudo = PseudoLabelTrainer(
                base_estimator=base_estimator,
                confidence_threshold=self.pseudo_threshold,
                n_iterations=2,
                verbose=self.verbosity > 1,
            )

            pseudo.fit(
                estimator=None,
                X_labeled=texts_train,
                y_labeled=y_train,
                X_unlabeled=texts_unlabeled,
            )

            # Update the model
            trained_models[best_name]["estimator"] = pseudo
            self._models[best_name]["estimator"] = pseudo

            if self.verbosity > 0:
                n_pseudo = len(pseudo.pseudo_labels_) if pseudo.pseudo_labels_ is not None else 0
                print(f"  Added {n_pseudo} pseudo-labels")

        except ImportError:
            logger.warning("PseudoLabelTrainer not available, skipping pseudo-labeling")
        except Exception as e:
            logger.warning(f"Pseudo-labeling failed: {e}")

    def _build_ensemble(
        self,
        trained_models: dict[str, Any],
        texts_val: list[str] | None,
        y_val: np.ndarray | None,
    ) -> None:
        """Build ensemble from trained models.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        texts_val : list, optional
            Validation texts.
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
        self._ensemble = _TextEnsemble(
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

    def _get_base_model_name(self) -> str:
        """Get the base model name for DAPT.

        Returns
        -------
        str
            Model name.
        """
        if self.model_presets:
            return self.model_presets[0]

        if self.presets == "fast":
            return DEFAULT_MODELS["fast"]
        else:
            return DEFAULT_MODELS["medium"]


class _TextEnsemble:
    """Simple weighted ensemble for text models."""

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float],
        task_type: str,
    ):
        self.models = models
        self.weights = weights
        self.task_type = task_type

    def predict(self, texts: list[str]) -> np.ndarray:
        """Make predictions."""
        if self.task_type in ("binary", "multiclass"):
            proba = self.predict_proba(texts)
            return np.argmax(proba, axis=1)
        else:
            # Regression: weighted average
            preds = []
            total_weight = sum(self.weights.values())

            for name, model in self.models.items():
                weight = self.weights.get(name, 0) / total_weight
                if weight > 0:
                    preds.append(weight * model.predict(texts))

            return sum(preds)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Predict probabilities."""
        proba = None
        total_weight = sum(self.weights.values())

        for name, model in self.models.items():
            weight = self.weights.get(name, 0) / total_weight

            if weight > 0 and hasattr(model, "predict_proba"):
                model_proba = model.predict_proba(texts)

                if proba is None:
                    proba = weight * model_proba
                else:
                    proba += weight * model_proba

        return proba if proba is not None else np.zeros((len(texts), 2))
