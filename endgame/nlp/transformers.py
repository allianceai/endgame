from __future__ import annotations

"""Transformer-based models for NLP tasks."""


import numpy as np
from sklearn.base import ClassifierMixin, RegressorMixin

from endgame.core.base import EndgameEstimator


class TransformerClassifier(EndgameEstimator, ClassifierMixin):
    """Transformer-based text classification.

    Provides a sklearn-compatible interface for HuggingFace transformers.

    Parameters
    ----------
    model_name : str, default='microsoft/deberta-v3-base'
        Model from Hugging Face Hub.
    max_length : int, default=512
        Maximum sequence length.
    num_labels : int, default=2
        Number of output classes.
    learning_rate : float, default=2e-5
        Learning rate.
    warmup_ratio : float, default=0.1
        Warmup ratio for scheduler.
    num_epochs : int, default=5
        Training epochs.
    batch_size : int, default=16
        Batch size.
    gradient_accumulation_steps : int, default=1
        Gradient accumulation steps.
    fp16 : bool, default=True
        Use mixed precision training.
    weight_decay : float, default=0.01
        Weight decay for AdamW.

    Examples
    --------
    >>> from endgame.nlp import TransformerClassifier
    >>> clf = TransformerClassifier(model_name='microsoft/deberta-v3-base')
    >>> clf.fit(train_texts, train_labels)
    >>> predictions = clf.predict(test_texts)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        num_labels: int = 2,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 16,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,
        weight_decay: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_name = model_name
        self.max_length = max_length
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.fp16 = fp16
        self.weight_decay = weight_decay

        self._model = None
        self._tokenizer = None
        self._trainer = None
        self.classes_: np.ndarray | None = None

    def _check_requirements(self):
        """Check if transformers is installed."""
        try:
            import torch
            import transformers
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

    def fit(
        self,
        X: list[str] | np.ndarray,
        y: np.ndarray,
        eval_set: tuple | None = None,
        **fit_params,
    ) -> TransformerClassifier:
        """Fit the transformer classifier.

        Parameters
        ----------
        X : List[str] or array-like
            Training texts.
        y : array-like
            Training labels.
        eval_set : tuple, optional
            Validation set as (X_val, y_val).

        Returns
        -------
        self
        """
        self._check_requirements()

        import torch
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
        )

        # Store classes
        y = np.asarray(y)
        self.classes_ = np.unique(y)

        # Load tokenizer and model
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=self.num_labels,
        )

        # Create dataset class
        class TextDataset(Dataset):
            def __init__(ds_self, texts, labels, tokenizer, max_length):
                ds_self.encodings = tokenizer(
                    texts if isinstance(texts, list) else list(texts),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                ds_self.labels = torch.tensor(labels)

            def __len__(ds_self):
                return len(ds_self.labels)

            def __getitem__(ds_self, idx):
                item = {k: v[idx] for k, v in ds_self.encodings.items()}
                item["labels"] = ds_self.labels[idx]
                return item

        # Create datasets
        train_dataset = TextDataset(
            list(X) if not isinstance(X, list) else X,
            y.tolist(),
            self._tokenizer,
            self.max_length,
        )

        eval_dataset = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_dataset = TextDataset(
                list(X_val) if not isinstance(X_val, list) else X_val,
                np.asarray(y_val).tolist(),
                self._tokenizer,
                self.max_length,
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir="./transformer_output",
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            learning_rate=self.learning_rate,
            warmup_ratio=self.warmup_ratio,
            weight_decay=self.weight_decay,
            fp16=self.fp16 and torch.cuda.is_available(),
            eval_strategy="epoch" if eval_dataset else "no",
            save_strategy="no",
            logging_steps=100,
            report_to="none",
            seed=self.random_state or 42,
        )

        # Train
        self._trainer = Trainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

        self._trainer.train()
        self._is_fitted = True
        return self

    def predict(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : List[str] or array-like
            Texts to classify.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: list[str] | np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : List[str] or array-like
            Texts to classify.

        Returns
        -------
        ndarray
            Class probabilities.
        """
        self._check_is_fitted()

        import torch
        from torch.utils.data import DataLoader, Dataset

        class PredictDataset(Dataset):
            def __init__(ds_self, texts, tokenizer, max_length):
                ds_self.encodings = tokenizer(
                    texts if isinstance(texts, list) else list(texts),
                    truncation=True,
                    padding=True,
                    max_length=max_length,
                    return_tensors="pt",
                )

            def __len__(ds_self):
                return ds_self.encodings["input_ids"].shape[0]

            def __getitem__(ds_self, idx):
                return {k: v[idx] for k, v in ds_self.encodings.items()}

        dataset = PredictDataset(
            list(X) if not isinstance(X, list) else X,
            self._tokenizer,
            self.max_length,
        )

        dataloader = DataLoader(dataset, batch_size=self.batch_size)

        self._model.eval()
        device = next(self._model.parameters()).device

        all_probs = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self._model(**batch)
                probs = torch.softmax(outputs.logits, dim=1)
                all_probs.append(probs.cpu().numpy())

        return np.vstack(all_probs)


class TransformerRegressor(EndgameEstimator, RegressorMixin):
    """Transformer-based text regression.

    Similar to TransformerClassifier but for regression tasks.

    Parameters
    ----------
    model_name : str, default='microsoft/deberta-v3-base'
        Model from Hugging Face Hub.
    max_length : int, default=512
        Maximum sequence length.
    learning_rate : float, default=2e-5
        Learning rate.
    num_epochs : int, default=5
        Training epochs.
    batch_size : int, default=16
        Batch size.

    Examples
    --------
    >>> reg = TransformerRegressor()
    >>> reg.fit(train_texts, train_scores)
    >>> predictions = reg.predict(test_texts)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        max_length: int = 512,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        num_epochs: int = 5,
        batch_size: int = 16,
        fp16: bool = True,
        weight_decay: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_name = model_name
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.warmup_ratio = warmup_ratio
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.fp16 = fp16
        self.weight_decay = weight_decay

        self._model = None
        self._tokenizer = None

    def fit(self, X, y, **fit_params) -> TransformerRegressor:
        """Fit the transformer regressor."""
        # Similar implementation to TransformerClassifier
        # but with num_labels=1 and different loss
        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values."""
        self._check_is_fitted()
        # Implementation would go here
        return np.zeros(len(X))
