from __future__ import annotations

"""Pre-trained audio model wrappers for transfer learning.

Provides sklearn-compatible wrappers around HuggingFace audio models
(AST, wav2vec2, HuBERT, Whisper) for audio classification via fine-tuning
or feature extraction.
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, Dataset

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class _AudioDataset(Dataset):
    """Simple dataset for waveforms + labels."""

    def __init__(self, waveforms, labels, processor, max_length, sample_rate):
        self.waveforms = waveforms
        self.labels = labels
        self.processor = processor
        self.max_length = max_length
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.waveforms)

    def __getitem__(self, idx):
        waveform = self.waveforms[idx]
        label = self.labels[idx]

        # Pad or truncate
        if len(waveform) > self.max_length:
            waveform = waveform[:self.max_length]
        elif len(waveform) < self.max_length:
            waveform = np.pad(waveform, (0, self.max_length - len(waveform)))

        # Process through model's feature extractor
        inputs = self.processor(
            waveform,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=False,
        )
        input_values = inputs["input_values"].squeeze(0)

        return input_values, label


class _SpectrogramDataset(Dataset):
    """Dataset for AST-style models that expect spectrograms.

    Pre-computes all spectrograms in __init__ to avoid repeated
    feature extraction every epoch.
    """

    def __init__(self, waveforms, labels, feature_extractor, max_length, sample_rate):
        self.labels = labels
        # Pre-compute all spectrograms once
        processed = []
        for waveform in waveforms:
            if len(waveform) > max_length:
                waveform = waveform[:max_length]
            elif len(waveform) < max_length:
                waveform = np.pad(waveform, (0, max_length - len(waveform)))
            inputs = feature_extractor(
                waveform,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            processed.append(inputs["input_values"].squeeze(0))
        self.input_values = torch.stack(processed)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.input_values[idx], self.labels[idx]


class PretrainedAudioClassifier(BaseEstimator, ClassifierMixin):
    """Fine-tune pre-trained audio models for classification.

    Wraps HuggingFace audio models with an sklearn-compatible interface.
    Supports AST, wav2vec2, HuBERT, and Whisper encoder.

    Parameters
    ----------
    model_name : str, default='MIT/ast-finetuned-audioset-10-10-0.4593'
        HuggingFace model identifier. Common choices:
        - 'MIT/ast-finetuned-audioset-10-10-0.4593' (AST, AudioSet pretrained)
        - 'facebook/wav2vec2-base-960h' (wav2vec2)
        - 'facebook/hubert-base-ls960' (HuBERT)
        - 'openai/whisper-base' (Whisper encoder)
    sample_rate : int, default=16000
        Expected audio sample rate. Most HuggingFace models expect 16kHz.
    max_length_seconds : float, default=5.0
        Maximum audio length in seconds.
    freeze_encoder : bool, default=False
        Freeze the pre-trained encoder weights (only train classifier head).
    unfreeze_last_n : int, default=0
        Number of encoder layers to unfreeze from the top. Only used when
        freeze_encoder=True. Set >0 for gradual unfreezing.
    learning_rate : float, default=1e-5
        Learning rate for fine-tuning.
    head_learning_rate : float, default=1e-3
        Learning rate for the classification head (higher than encoder).
    weight_decay : float, default=0.01
        L2 regularization.
    n_epochs : int, default=20
        Maximum training epochs.
    batch_size : int, default=8
        Training batch size.
    early_stopping : int, default=5
        Early stopping patience (epochs without improvement).
    warmup_ratio : float, default=0.1
        Fraction of training steps for LR warmup.
    label_smoothing : float, default=0.0
        Label smoothing for cross-entropy loss.
    device : str, default='auto'
        Computation device.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Examples
    --------
    >>> clf = PretrainedAudioClassifier(
    ...     model_name='MIT/ast-finetuned-audioset-10-10-0.4593',
    ...     freeze_encoder=True,
    ...     n_epochs=10,
    ... )
    >>> clf.fit(X_train_waveforms, y_train)
    >>> predictions = clf.predict(X_test_waveforms)
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        model_name: str = "MIT/ast-finetuned-audioset-10-10-0.4593",
        sample_rate: int = 16000,
        max_length_seconds: float = 5.0,
        freeze_encoder: bool = False,
        unfreeze_last_n: int = 0,
        learning_rate: float = 1e-5,
        head_learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
        n_epochs: int = 20,
        batch_size: int = 8,
        early_stopping: int = 5,
        warmup_ratio: float = 0.1,
        label_smoothing: float = 0.0,
        device: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        if not HAS_TORCH:
            raise ImportError("PyTorch is required. Install with: pip install torch")

        self.model_name = model_name
        self.sample_rate = sample_rate
        self.max_length_seconds = max_length_seconds
        self.freeze_encoder = freeze_encoder
        self.unfreeze_last_n = unfreeze_last_n
        self.learning_rate = learning_rate
        self.head_learning_rate = head_learning_rate
        self.weight_decay = weight_decay
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.early_stopping = early_stopping
        self.warmup_ratio = warmup_ratio
        self.label_smoothing = label_smoothing
        self.device = device
        self.random_state = random_state
        self.verbose = verbose

        self.model_ = None
        self.processor_ = None
        self.classes_ = None
        self.n_classes_ = None
        self._device = None
        self.history_ = {"train_loss": [], "val_loss": [], "val_acc": []}

    def _log(self, msg: str):
        if self.verbose:
            print(f"[PretrainedAudio] {msg}")

    def _get_device(self):
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def _set_seed(self):
        if self.random_state is not None:
            torch.manual_seed(self.random_state)
            np.random.seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def _load_model(self, n_classes: int):
        """Load model and processor from HuggingFace."""
        try:
            from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        except ImportError:
            raise ImportError(
                "transformers is required for PretrainedAudioClassifier. "
                "Install with: pip install transformers"
            )

        self._log(f"Loading model: {self.model_name}")

        self.processor_ = AutoFeatureExtractor.from_pretrained(self.model_name)
        self.model_ = AutoModelForAudioClassification.from_pretrained(
            self.model_name,
            num_labels=n_classes,
            ignore_mismatched_sizes=True,
        )

        # Freeze encoder if requested
        if self.freeze_encoder:
            for name, param in self.model_.named_parameters():
                if "classifier" not in name and "projector" not in name:
                    param.requires_grad = False

            # Optionally unfreeze last N layers
            if self.unfreeze_last_n > 0:
                encoder_layers = []
                for name, param in self.model_.named_parameters():
                    if "layer" in name or "layers" in name:
                        encoder_layers.append((name, param))
                # Unfreeze last N unique layer groups
                layer_indices = set()
                for name, _ in encoder_layers:
                    for part in name.split("."):
                        if part.isdigit():
                            layer_indices.add(int(part))
                            break
                if layer_indices:
                    top_layers = sorted(layer_indices)[-self.unfreeze_last_n:]
                    for name, param in encoder_layers:
                        for part in name.split("."):
                            if part.isdigit() and int(part) in top_layers:
                                param.requires_grad = True
                                break

        self.model_.to(self._device)

        trainable = sum(p.numel() for p in self.model_.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model_.parameters())
        self._log(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def fit(
        self,
        X,
        y,
        val_data: tuple | None = None,
    ) -> PretrainedAudioClassifier:
        """Fine-tune the pre-trained model.

        Parameters
        ----------
        X : list of ndarray or ndarray
            Audio waveforms at self.sample_rate.
        y : array-like of shape (n_samples,)
            Integer class labels.
        val_data : tuple of (X_val, y_val), optional
            Validation data for early stopping.

        Returns
        -------
        self
        """
        self._set_seed()
        self._device = self._get_device()

        y_arr = np.asarray(y)
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        # Remap labels to 0..n_classes-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y_arr])

        # Load model
        self._load_model(self.n_classes_)

        max_length = int(self.max_length_seconds * self.sample_rate)

        # Prepare waveforms
        waveforms = self._prepare_waveforms(X)

        # Create dataset
        train_dataset = _SpectrogramDataset(
            waveforms, y_mapped, self.processor_, max_length, self.sample_rate
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=self._device.type == "cuda",
        )

        val_loader = None
        if val_data is not None:
            X_val, y_val = val_data
            y_val_arr = np.array([label_map[c] for c in np.asarray(y_val)])
            waveforms_val = self._prepare_waveforms(X_val)
            val_dataset = _SpectrogramDataset(
                waveforms_val, y_val_arr, self.processor_, max_length, self.sample_rate
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

        # Optimizer with differential learning rates
        param_groups = [
            {
                "params": [
                    p for n, p in self.model_.named_parameters()
                    if p.requires_grad and ("classifier" in n or "projector" in n)
                ],
                "lr": self.head_learning_rate,
            },
            {
                "params": [
                    p for n, p in self.model_.named_parameters()
                    if p.requires_grad and "classifier" not in n and "projector" not in n
                ],
                "lr": self.learning_rate,
            },
        ]
        # Filter empty groups
        param_groups = [g for g in param_groups if g["params"]]

        optimizer = optim.AdamW(param_groups, weight_decay=self.weight_decay)

        # Linear warmup + cosine decay
        total_steps = len(train_loader) * self.n_epochs
        warmup_steps = int(total_steps * self.warmup_ratio)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        criterion = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
        )

        # Training loop
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._log(f"Training on {self._device} for {self.n_epochs} epochs...")

        for epoch in range(self.n_epochs):
            self.model_.train()
            train_loss = 0.0
            n_batches = 0

            for batch_inputs, batch_labels in train_loader:
                batch_inputs = batch_inputs.to(self._device)
                batch_labels = batch_labels.long().to(self._device)

                optimizer.zero_grad()
                outputs = self.model_(batch_inputs)
                logits = outputs.logits
                loss = criterion(logits, batch_labels)
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= max(n_batches, 1)
            self.history_["train_loss"].append(train_loss)

            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                self.history_["val_loss"].append(val_loss)
                self.history_["val_acc"].append(val_acc)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone()
                        for k, v in self.model_.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if self.verbose:
                    self._log(
                        f"Epoch {epoch+1}/{self.n_epochs}: "
                        f"train_loss={train_loss:.4f}, "
                        f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
                    )

                if patience_counter >= self.early_stopping:
                    self._log(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if self.verbose:
                    self._log(f"Epoch {epoch+1}/{self.n_epochs}: train_loss={train_loss:.4f}")

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        return self

    def _evaluate(self, loader, criterion):
        """Evaluate on a data loader."""
        self.model_.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_inputs, batch_labels in loader:
                batch_inputs = batch_inputs.to(self._device)
                batch_labels = batch_labels.long().to(self._device)
                outputs = self.model_(batch_inputs)
                logits = outputs.logits
                loss = criterion(logits, batch_labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1)
                correct += (preds == batch_labels).sum().item()
                total += len(batch_labels)

        avg_loss = total_loss / max(len(loader), 1)
        accuracy = correct / max(total, 1)
        return avg_loss, accuracy

    def _prepare_waveforms(self, X):
        """Convert input to list of numpy arrays."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            return [X[i].astype(np.float32) for i in range(len(X))]
        elif isinstance(X, (list, tuple)):
            return [np.asarray(w, dtype=np.float32) for w in X]
        else:
            return [np.asarray(X, dtype=np.float32)]

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : list of ndarray or ndarray
            Audio waveforms.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : list of ndarray or ndarray
            Audio waveforms.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        waveforms = self._prepare_waveforms(X)
        max_length = int(self.max_length_seconds * self.sample_rate)

        dataset = _SpectrogramDataset(
            waveforms,
            np.zeros(len(waveforms)),  # dummy labels
            self.processor_,
            max_length,
            self.sample_rate,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        all_proba = []

        with torch.no_grad():
            for batch_inputs, _ in loader:
                batch_inputs = batch_inputs.to(self._device)
                outputs = self.model_(batch_inputs)
                proba = torch.softmax(outputs.logits, dim=-1)
                all_proba.append(proba.cpu().numpy())

        return np.vstack(all_proba)

    def extract_features(self, X) -> np.ndarray:
        """Extract embeddings from the pre-trained encoder.

        Useful for downstream classifiers (XGBoost, SVM, etc.) on top of
        learned representations.

        Parameters
        ----------
        X : list of ndarray or ndarray
            Audio waveforms.

        Returns
        -------
        ndarray of shape (n_samples, embedding_dim)
            Feature embeddings.
        """
        if self.model_ is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        waveforms = self._prepare_waveforms(X)
        max_length = int(self.max_length_seconds * self.sample_rate)

        dataset = _SpectrogramDataset(
            waveforms,
            np.zeros(len(waveforms)),
            self.processor_,
            max_length,
            self.sample_rate,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model_.eval()
        all_features = []

        # Hook into the model to get encoder output
        with torch.no_grad():
            for batch_inputs, _ in loader:
                batch_inputs = batch_inputs.to(self._device)
                outputs = self.model_(batch_inputs, output_hidden_states=True)
                # Use last hidden state, averaged over time
                hidden = outputs.hidden_states[-1]  # (batch, seq_len, hidden_dim)
                pooled = hidden.mean(dim=1)  # (batch, hidden_dim)
                all_features.append(pooled.cpu().numpy())

        return np.vstack(all_features)
