"""Domain-Adaptive Pretraining for transformers."""


from endgame.core.base import EndgameEstimator


class DomainAdaptivePretrainer(EndgameEstimator):
    """Domain-Adaptive Pretraining: Continue MLM on competition corpus.

    Adapts pretrained model to domain vocabulary before supervised fine-tuning.
    Key technique for NLP competitions.

    Parameters
    ----------
    model_name : str, default='microsoft/deberta-v3-base'
        Base model to continue pretraining.
    mlm_probability : float, default=0.15
        Masking probability for MLM.
    num_epochs : int, default=3
        Training epochs for MLM.
    batch_size : int, default=8
        Batch size.
    learning_rate : float, default=5e-5
        Learning rate.
    max_length : int, default=512
        Maximum sequence length.

    Examples
    --------
    >>> dapt = DomainAdaptivePretrainer(model_name='microsoft/deberta-v3-base')
    >>> adapted_path = dapt.pretrain(
    ...     texts=train_texts + test_texts,
    ...     output_dir='./adapted_model'
    ... )
    >>> # Use adapted_path for downstream fine-tuning
    """

    def __init__(
        self,
        model_name: str = "microsoft/deberta-v3-base",
        mlm_probability: float = 0.15,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        max_length: int = 512,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_name = model_name
        self.mlm_probability = mlm_probability
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_length = max_length

    def pretrain(
        self,
        texts: list[str],
        output_dir: str = "./dapt_model",
    ) -> str:
        """Continue MLM pretraining on provided texts.

        Parameters
        ----------
        texts : List[str]
            Texts for continued pretraining.
        output_dir : str
            Directory to save adapted model.

        Returns
        -------
        str
            Path to adapted model.
        """
        try:
            from datasets import Dataset
            from transformers import (
                AutoModelForMaskedLM,
                AutoTokenizer,
                DataCollatorForLanguageModeling,
                Trainer,
                TrainingArguments,
            )
        except ImportError:
            raise ImportError(
                "transformers and datasets are required. "
                "Install with: pip install transformers datasets"
            )

        self._log(f"Loading model: {self.model_name}")

        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForMaskedLM.from_pretrained(self.model_name)

        # Create dataset
        dataset = Dataset.from_dict({"text": texts})

        def tokenize_function(examples):
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )

        self._log("Tokenizing texts...")
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
        )

        # Data collator for MLM
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=True,
            mlm_probability=self.mlm_probability,
        )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.num_epochs,
            per_device_train_batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            save_steps=500,
            save_total_limit=1,
            logging_steps=100,
            report_to="none",
            seed=self.random_state or 42,
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        self._log(f"Starting DAPT for {self.num_epochs} epochs...")
        trainer.train()

        # Save
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        self._log(f"Model saved to {output_dir}")
        self._is_fitted = True

        return output_dir
