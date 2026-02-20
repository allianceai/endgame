"""Sequence-to-sequence translation models with PEFT support."""

import warnings
from dataclasses import dataclass

from endgame.core.base import EndgameEstimator


@dataclass
class TranslationConfig:
    """Configuration for translation models.

    Parameters
    ----------
    model_name : str
        HuggingFace model name (e.g., 'facebook/nllb-200-distilled-600M')
    source_lang : str, optional
        Source language code (e.g., 'eng_Latn' for NLLB)
    target_lang : str, optional
        Target language code
    max_source_length : int
        Maximum source sequence length
    max_target_length : int
        Maximum target sequence length
    num_beams : int
        Beam search width for generation
    use_lora : bool
        Whether to use LoRA for parameter-efficient fine-tuning
    lora_r : int
        LoRA rank
    lora_alpha : int
        LoRA alpha scaling
    lora_dropout : float
        LoRA dropout
    learning_rate : float
        Learning rate for training
    batch_size : int
        Training batch size
    gradient_accumulation_steps : int
        Gradient accumulation steps
    num_epochs : int
        Number of training epochs
    warmup_ratio : float
        Warmup ratio for scheduler
    fp16 : bool
        Use mixed precision training
    early_stopping_patience : int
        Early stopping patience (0 to disable)
    """
    model_name: str = "facebook/nllb-200-distilled-600M"
    source_lang: str | None = None
    target_lang: str | None = "eng_Latn"
    max_source_length: int = 256
    max_target_length: int = 512
    num_beams: int = 5
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = None
    learning_rate: float = 2e-4
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    fp16: bool = True
    early_stopping_patience: int = 3
    output_dir: str = "./translation_output"
    seed: int = 42


class TransformerTranslator(EndgameEstimator):
    """Transformer-based sequence-to-sequence translation.

    Provides a sklearn-compatible interface for neural machine translation
    using HuggingFace transformers with optional LoRA/PEFT support.

    Supports models like:
    - NLLB (facebook/nllb-200-distilled-600M, facebook/nllb-200-1.3B)
    - mBART (facebook/mbart-large-50-many-to-many-mmt)
    - mT5 (google/mt5-base, google/mt5-large)
    - MarianMT (Helsinki-NLP/opus-mt-*)

    Parameters
    ----------
    config : TranslationConfig, optional
        Configuration object. If None, uses defaults.
    model_name : str, optional
        Override model name from config.
    use_lora : bool, optional
        Override LoRA setting from config.
    **kwargs
        Additional config overrides.

    Examples
    --------
    >>> from endgame.nlp import TransformerTranslator, TranslationConfig
    >>>
    >>> # Quick start with defaults
    >>> translator = TransformerTranslator(model_name='facebook/nllb-200-distilled-600M')
    >>> translator.fit(source_texts, target_texts)
    >>> translations = translator.predict(test_texts)
    >>>
    >>> # With custom config
    >>> config = TranslationConfig(
    ...     model_name='google/mt5-base',
    ...     use_lora=True,
    ...     num_epochs=10,
    ... )
    >>> translator = TransformerTranslator(config=config)
    """

    _MODEL_TYPES = {
        'nllb': ['facebook/nllb'],
        'mbart': ['facebook/mbart'],
        't5': ['t5-', 'mt5-', 'google/flan-t5'],
        'marian': ['Helsinki-NLP/opus-mt'],
    }

    def __init__(
        self,
        config: TranslationConfig | None = None,
        model_name: str | None = None,
        use_lora: bool | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        super().__init__(random_state=random_state, verbose=verbose)

        # Build config
        self.config = config or TranslationConfig()

        # Override from kwargs
        if model_name is not None:
            self.config.model_name = model_name
        if use_lora is not None:
            self.config.use_lora = use_lora
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)

        if random_state is not None:
            self.config.seed = random_state

        # Will be set during fit
        self._model = None
        self._tokenizer = None
        self._trainer = None
        self._model_type = None

    def _check_requirements(self):
        """Check for required packages."""
        try:
            import torch
            import transformers
        except ImportError:
            raise ImportError(
                "PyTorch and Transformers are required. "
                "Install with: pip install torch transformers"
            )

        if self.config.use_lora:
            try:
                import peft
            except ImportError:
                warnings.warn(
                    "PEFT not installed. Disabling LoRA. "
                    "Install with: pip install peft"
                )
                self.config.use_lora = False

    def _detect_model_type(self) -> str:
        """Detect the type of model from model_name."""
        model_name = self.config.model_name.lower()

        for model_type, prefixes in self._MODEL_TYPES.items():
            if any(prefix.lower() in model_name for prefix in prefixes):
                return model_type

        return 'seq2seq'  # Generic fallback

    def _load_model_and_tokenizer(self, for_training: bool = True):
        """Load model and tokenizer based on model type."""
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if self.verbose:
            print(f"Loading model: {self.config.model_name}")

        self._model_type = self._detect_model_type()

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # Load model
        self._model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32,
        )

        # Apply LoRA if requested
        if for_training and self.config.use_lora:
            self._apply_lora()

        return self._model, self._tokenizer

    def _apply_lora(self):
        """Apply LoRA adapters to the model."""
        from peft import LoraConfig, TaskType, get_peft_model

        # Default target modules based on model type
        target_modules = self.config.lora_target_modules
        if target_modules is None:
            if self._model_type in ['nllb', 'mbart']:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif self._model_type == 't5':
                target_modules = ["q", "v", "k", "o"]
            else:
                target_modules = ["q_proj", "v_proj"]

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
        )

        self._model = get_peft_model(self._model, lora_config)

        if self.verbose:
            self._model.print_trainable_parameters()

    def fit(
        self,
        X: list[str],
        y: list[str],
        eval_set: tuple[list[str], list[str]] | None = None,
        **fit_params,
    ) -> "TransformerTranslator":
        """Fine-tune the translation model.

        Parameters
        ----------
        X : List[str]
            Source language texts.
        y : List[str]
            Target language translations.
        eval_set : tuple, optional
            Validation set as (X_val, y_val).
        **fit_params
            Additional parameters passed to Trainer.

        Returns
        -------
        self
        """
        self._check_requirements()

        import torch
        from torch.utils.data import Dataset
        from transformers import (
            DataCollatorForSeq2Seq,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )

        # Load model
        self._load_model_and_tokenizer(for_training=True)

        # Create dataset class
        class TranslationDataset(Dataset):
            def __init__(
                ds_self,
                sources: list[str],
                targets: list[str],
                tokenizer,
                max_source_length: int,
                max_target_length: int,
                source_lang: str | None = None,
                target_lang: str | None = None,
            ):
                ds_self.sources = sources
                ds_self.targets = targets
                ds_self.tokenizer = tokenizer
                ds_self.max_source_length = max_source_length
                ds_self.max_target_length = max_target_length
                ds_self.source_lang = source_lang
                ds_self.target_lang = target_lang

            def __len__(ds_self):
                return len(ds_self.sources)

            def __getitem__(ds_self, idx):
                source = ds_self.sources[idx]
                target = ds_self.targets[idx]

                # Set source language for NLLB-style models
                if ds_self.source_lang and hasattr(ds_self.tokenizer, 'src_lang'):
                    ds_self.tokenizer.src_lang = ds_self.source_lang

                # Tokenize source
                model_inputs = ds_self.tokenizer(
                    source,
                    max_length=ds_self.max_source_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                model_inputs = {k: v.squeeze(0) for k, v in model_inputs.items()}

                # Tokenize target
                if hasattr(ds_self.tokenizer, 'as_target_tokenizer'):
                    with ds_self.tokenizer.as_target_tokenizer():
                        labels = ds_self.tokenizer(
                            target,
                            max_length=ds_self.max_target_length,
                            padding="max_length",
                            truncation=True,
                            return_tensors="pt",
                        )
                else:
                    labels = ds_self.tokenizer(
                        target,
                        max_length=ds_self.max_target_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    )

                model_inputs["labels"] = labels["input_ids"].squeeze(0)
                # Replace padding with -100 for loss calculation
                model_inputs["labels"][
                    model_inputs["labels"] == ds_self.tokenizer.pad_token_id
                ] = -100

                return model_inputs

        # Create datasets
        train_dataset = TranslationDataset(
            X, y, self._tokenizer,
            self.config.max_source_length,
            self.config.max_target_length,
            self.config.source_lang,
            self.config.target_lang,
        )

        eval_dataset = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_dataset = TranslationDataset(
                X_val, y_val, self._tokenizer,
                self.config.max_source_length,
                self.config.max_target_length,
                self.config.source_lang,
                self.config.target_lang,
            )

        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=self._tokenizer,
            model=self._model,
            padding=True,
        )

        # Training arguments
        training_args = Seq2SeqTrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            fp16=self.config.fp16 and torch.cuda.is_available(),
            evaluation_strategy="epoch" if eval_dataset else "no",
            save_strategy="epoch" if eval_dataset else "no",
            save_total_limit=2,
            load_best_model_at_end=bool(eval_dataset),
            logging_steps=100,
            report_to="none",
            seed=self.config.seed,
            predict_with_generate=True,
            generation_max_length=self.config.max_target_length,
        )

        # Callbacks
        callbacks = []
        if eval_dataset and self.config.early_stopping_patience > 0:
            callbacks.append(
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.early_stopping_patience
                )
            )

        # Trainer
        self._trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            callbacks=callbacks if callbacks else None,
        )

        # Train
        if self.verbose:
            print(f"Training with {len(X)} samples...")

        self._trainer.train()
        self._is_fitted = True

        return self

    def predict(
        self,
        X: list[str],
        batch_size: int | None = None,
        **generate_kwargs,
    ) -> list[str]:
        """Generate translations for source texts.

        Parameters
        ----------
        X : List[str]
            Source language texts to translate.
        batch_size : int, optional
            Batch size for inference.
        **generate_kwargs
            Additional arguments for model.generate().

        Returns
        -------
        List[str]
            Translated texts.
        """
        self._check_is_fitted()

        import torch

        batch_size = batch_size or self.config.batch_size
        device = next(self._model.parameters()).device

        self._model.eval()
        translations = []

        # Set default generation kwargs
        gen_kwargs = {
            'num_beams': self.config.num_beams,
            'max_new_tokens': self.config.max_target_length,
            'early_stopping': True,
        }

        # Add forced BOS token for NLLB-style models
        if self.config.target_lang and self._model_type == 'nllb':
            try:
                forced_bos = self._tokenizer.convert_tokens_to_ids(
                    self.config.target_lang
                )
                gen_kwargs['forced_bos_token_id'] = forced_bos
            except Exception:
                pass

        gen_kwargs.update(generate_kwargs)

        for i in range(0, len(X), batch_size):
            batch_texts = X[i:i + batch_size]

            # Set source language
            if self.config.source_lang and hasattr(self._tokenizer, 'src_lang'):
                self._tokenizer.src_lang = self.config.source_lang

            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.config.max_source_length,
            ).to(device)

            with torch.no_grad():
                generated = self._model.generate(**inputs, **gen_kwargs)

            decoded = self._tokenizer.batch_decode(
                generated, skip_special_tokens=True
            )
            translations.extend(decoded)

            if self.verbose and (i // batch_size) % 10 == 0:
                print(f"Translated {len(translations)}/{len(X)}")

        return translations

    def save(self, path: str):
        """Save the model and tokenizer.

        Parameters
        ----------
        path : str
            Directory to save to.
        """
        self._check_is_fitted()

        from pathlib import Path
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._model.save_pretrained(str(save_path))
        self._tokenizer.save_pretrained(str(save_path))

        if self.verbose:
            print(f"Model saved to {save_path}")

    @classmethod
    def load(
        cls,
        path: str,
        config: TranslationConfig | None = None,
        **kwargs,
    ) -> "TransformerTranslator":
        """Load a saved model.

        Parameters
        ----------
        path : str
            Directory containing saved model.
        config : TranslationConfig, optional
            Configuration (uses defaults if not provided).

        Returns
        -------
        TransformerTranslator
            Loaded translator.
        """
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        translator = cls(config=config, model_name=path, **kwargs)
        translator._tokenizer = AutoTokenizer.from_pretrained(path)
        translator._model = AutoModelForSeq2SeqLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if translator.config.fp16 else torch.float32,
        )
        translator._model_type = translator._detect_model_type()
        translator._is_fitted = True

        return translator


class BackTranslator:
    """Generate synthetic parallel data via back-translation.

    Back-translation is a key technique for low-resource NMT:
    1. Train/use a model to translate target→source (reverse direction)
    2. Translate monolingual target data to source
    3. Use (synthetic_source, original_target) as additional training data

    Parameters
    ----------
    forward_model : TransformerTranslator
        Model for source→target translation.
    backward_model : TransformerTranslator, optional
        Model for target→source translation. If None, will use forward_model
        with swapped languages if supported.

    Examples
    --------
    >>> # Augment training data with back-translation
    >>> bt = BackTranslator(forward_model, backward_model)
    >>> synthetic_sources = bt.generate(monolingual_targets)
    >>> augmented_X = train_X + synthetic_sources
    >>> augmented_y = train_y + monolingual_targets
    """

    def __init__(
        self,
        forward_model: TransformerTranslator | None = None,
        backward_model: TransformerTranslator | None = None,
    ):
        self.forward_model = forward_model
        self.backward_model = backward_model

    def generate(
        self,
        target_texts: list[str],
        batch_size: int = 8,
        **generate_kwargs,
    ) -> list[str]:
        """Generate synthetic source texts from target texts.

        Parameters
        ----------
        target_texts : List[str]
            Target language texts (monolingual).
        batch_size : int
            Batch size for translation.
        **generate_kwargs
            Additional arguments for generation.

        Returns
        -------
        List[str]
            Synthetic source texts.
        """
        if self.backward_model is None:
            raise ValueError(
                "backward_model required for back-translation. "
                "Initialize with a target→source translator."
            )

        return self.backward_model.predict(
            target_texts,
            batch_size=batch_size,
            **generate_kwargs,
        )

    def augment_dataset(
        self,
        X: list[str],
        y: list[str],
        monolingual_targets: list[str],
        batch_size: int = 8,
    ) -> tuple[list[str], list[str]]:
        """Augment parallel data with back-translated synthetic pairs.

        Parameters
        ----------
        X : List[str]
            Original source texts.
        y : List[str]
            Original target texts.
        monolingual_targets : List[str]
            Monolingual target texts to back-translate.
        batch_size : int
            Batch size for translation.

        Returns
        -------
        Tuple[List[str], List[str]]
            Augmented (sources, targets).
        """
        synthetic_sources = self.generate(monolingual_targets, batch_size)

        augmented_X = list(X) + synthetic_sources
        augmented_y = list(y) + list(monolingual_targets)

        return augmented_X, augmented_y
