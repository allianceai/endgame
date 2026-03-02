"""LLM-based oversampler for class imbalance handling.

This module provides tabular data generation via language model fine-tuning,
following the GReaT (Generation of Realistic Tabular data) approach.

Algorithms
----------
- GReaTResampler: Fine-tune a causal LM on serialized tabular rows

References
----------
- GReaT (Borisov et al., 2023)
- ImbLLM improvements (2025)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_X_y

from endgame.preprocessing.imbalance_geometric import _compute_sampling_targets

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def _check_transformers():
    if not HAS_TRANSFORMERS:
        raise ImportError(
            "transformers and PyTorch are required for GReaTResampler. "
            "Install with: pip install endgame-ml[nlp]"
        )


def _serialize_row(
    row: np.ndarray, feature_names: list[str], label_name: str, label_value: Any
) -> str:
    """Serialize a tabular row to natural language.

    Example output: "Feature1 is 3.14, Feature2 is -1.00, Class is 1"
    """
    parts = [f"{name} is {val:.4g}" for name, val in zip(feature_names, row)]
    parts.append(f"{label_name} is {label_value}")
    return ", ".join(parts)


def _parse_generated_text(
    text: str, feature_names: list[str], label_name: str, n_features: int
) -> np.ndarray | None:
    """Parse a generated text string back to a feature vector.

    Returns None if parsing fails.
    """
    values = {}
    for part in text.split(","):
        part = part.strip()
        if " is " not in part:
            continue
        name, val_str = part.split(" is ", 1)
        name = name.strip()
        val_str = val_str.strip()
        if name in feature_names:
            try:
                values[name] = float(val_str)
            except ValueError:
                continue

    if len(values) < n_features // 2:
        return None

    row = np.zeros(n_features)
    for i, name in enumerate(feature_names):
        if name in values:
            row[i] = values[name]
    return row


# Internal dataset for fine-tuning
_GreatDataset = None

if HAS_TRANSFORMERS:

    class _GreatDataset(TorchDataset):
        """Dataset of serialized tabular rows for LM fine-tuning."""

        def __init__(self, texts: list[str], tokenizer, max_length: int):
            self.encodings = tokenizer(
                texts,
                truncation=True,
                padding="max_length",
                max_length=max_length,
                return_tensors="pt",
            )

        def __len__(self):
            return self.encodings["input_ids"].shape[0]

        def __getitem__(self, idx):
            item = {k: v[idx] for k, v in self.encodings.items()}
            item["labels"] = item["input_ids"].clone()
            return item


class GReaTResampler(BaseEstimator):
    """GReaT oversampler: LLM-based tabular data generation.

    Serializes tabular rows as natural language strings, fine-tunes a small
    causal language model (e.g. distilgpt2), and generates new minority
    samples by prompting with the minority class label prefix.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`~imbalance_geometric._compute_sampling_targets`.
    model_name : str, default='distilgpt2'
        HuggingFace model name for the causal LM backbone.
    n_epochs : int, default=5
        Fine-tuning epochs.
    batch_size : int, default=8
        Training batch size.
    max_length : int, default=256
        Maximum token length for serialized rows.
    temperature : float, default=0.7
        Sampling temperature for generation.
    feature_names : list of str or None, default=None
        Feature names. If None, uses ``f0, f1, ...``.
    label_name : str, default='Class'
        Name for the target column in serialization.
    device : str, default='auto'
        Computation device.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    References
    ----------
    GReaT (Borisov et al., 2023), ImbLLM (2025)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        model_name: str = "distilgpt2",
        n_epochs: int = 5,
        batch_size: int = 8,
        max_length: int = 256,
        temperature: float = 0.7,
        feature_names: list[str] | None = None,
        label_name: str = "Class",
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.sampling_strategy = sampling_strategy
        self.model_name = model_name
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.max_length = max_length
        self.temperature = temperature
        self.feature_names = feature_names
        self.label_name = label_name
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, msg: str):
        if self.verbose:
            print(f"[GReaT] {msg}")

    def _get_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def fit(self, X: ArrayLike, y: ArrayLike) -> GReaTResampler:
        """Fit (validate input)."""
        _check_transformers()
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample using GReaT LLM generation."""
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_features = X.shape[1]
        feature_names = self.feature_names or [f"f{i}" for i in range(n_features)]
        device = self._get_device()

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            if len(X_cls) == 0:
                continue

            self._log(f"Fine-tuning {self.model_name} on class {cls} "
                      f"({len(X_cls)} samples)...")

            # Serialize training data
            texts = [
                _serialize_row(row, feature_names, self.label_name, cls)
                for row in X_cls
            ]

            # Load model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self.model_name)

            # Fine-tune
            dataset = _GreatDataset(texts, tokenizer, self.max_length)

            training_args = TrainingArguments(
                output_dir="/tmp/great_training",
                num_train_epochs=self.n_epochs,
                per_device_train_batch_size=self.batch_size,
                logging_steps=max(1, len(texts) // self.batch_size),
                save_strategy="no",
                report_to="none",
                seed=self.random_state or 42,
                no_cuda=(device == "cpu"),
            )

            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
            trainer.train()

            # Generate samples
            self._log(f"Generating {n_synthetic} samples for class {cls}...")
            prompt = f"{self.label_name} is {cls}"
            prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(
                model.device
            )

            generated_rows = []
            attempts = 0
            max_attempts = n_synthetic * 5

            model.eval()
            while len(generated_rows) < n_synthetic and attempts < max_attempts:
                batch_n = min(self.batch_size, n_synthetic - len(generated_rows))
                input_ids = prompt_ids.repeat(batch_n, 1)

                with torch.no_grad():
                    outputs = model.generate(
                        input_ids,
                        max_new_tokens=self.max_length,
                        temperature=self.temperature,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )

                for output in outputs:
                    text = tokenizer.decode(output, skip_special_tokens=True)
                    row = _parse_generated_text(
                        text, feature_names, self.label_name, n_features
                    )
                    if row is not None:
                        generated_rows.append(row)
                    attempts += 1

            # Fill shortfall with random duplicates
            if len(generated_rows) < n_synthetic:
                rng = np.random.RandomState(self.random_state)
                shortfall = n_synthetic - len(generated_rows)
                extra_idx = rng.choice(len(X_cls), size=shortfall, replace=True)
                generated_rows.extend(X_cls[extra_idx])

            synthetic_X.append(np.array(generated_rows[:n_synthetic]))
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# Category dict for registration
LLM_SAMPLERS = {
    "great": GReaTResampler,
}
