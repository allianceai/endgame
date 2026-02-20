"""Tests for NLP module."""

import numpy as np
import pytest


class TestTransformerClassifier:
    """Tests for TransformerClassifier."""

    def test_init(self):
        """Test initialization."""
        transformers = pytest.importorskip("transformers")
        from endgame.nlp import TransformerClassifier

        model = TransformerClassifier(
            model_name="prajjwal1/bert-tiny",
            num_labels=2,
            max_length=128,
        )

        assert model.model_name == "prajjwal1/bert-tiny"
        assert model.num_labels == 2

    def test_tokenize(self):
        """Test tokenization via the underlying tokenizer."""
        transformers = pytest.importorskip("transformers")
        from transformers import AutoTokenizer
        from endgame.nlp import TransformerClassifier

        model = TransformerClassifier(
            model_name="prajjwal1/bert-tiny",
            num_labels=2,
            max_length=64,
        )

        # The tokenizer is loaded during fit or can be loaded directly
        # TransformerClassifier stores it as _tokenizer after fit
        tokenizer = AutoTokenizer.from_pretrained(model.model_name)
        texts = ["Hello world", "Test text"]
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=64,
            return_tensors="pt",
        )

        assert "input_ids" in encodings
        assert encodings["input_ids"].shape[0] == 2


class TestTransformerRegressor:
    """Tests for TransformerRegressor."""

    def test_init(self):
        """Test initialization."""
        transformers = pytest.importorskip("transformers")
        from endgame.nlp import TransformerRegressor

        model = TransformerRegressor(
            model_name="prajjwal1/bert-tiny",
            max_length=128,
        )

        assert model.model_name == "prajjwal1/bert-tiny"


class TestDAPT:
    """Tests for Domain-Adaptive Pretraining."""

    def test_init(self):
        """Test DAPT initialization."""
        transformers = pytest.importorskip("transformers")
        from endgame.nlp import DomainAdaptivePretrainer

        dapt = DomainAdaptivePretrainer(
            model_name="prajjwal1/bert-tiny",
            mlm_probability=0.15,
        )

        assert dapt.model_name == "prajjwal1/bert-tiny"
        assert dapt.mlm_probability == 0.15

    def test_dapt_attributes(self):
        """Test DAPT configuration attributes."""
        transformers = pytest.importorskip("transformers")
        from endgame.nlp import DomainAdaptivePretrainer

        dapt = DomainAdaptivePretrainer(
            model_name="prajjwal1/bert-tiny",
            mlm_probability=0.15,
            max_length=64,
            num_epochs=1,
            batch_size=4,
        )

        # Verify configuration is stored correctly
        assert dapt.max_length == 64
        assert dapt.num_epochs == 1
        assert dapt.batch_size == 4
        # The pretrain() method is the main API (not prepare_dataset)
        assert hasattr(dapt, "pretrain")


class TestPseudoLabeling:
    """Tests for pseudo-labeling via PseudoLabelTrainer."""

    def test_pseudo_label_trainer_init(self):
        """Test PseudoLabelTrainer initialization."""
        from endgame.nlp.pseudo_label import PseudoLabelTrainer

        trainer = PseudoLabelTrainer(
            confidence_threshold=0.9,
            soft_labels=True,
            n_iterations=2,
            sample_weight_decay=0.5,
        )

        assert trainer.confidence_threshold == 0.9
        assert trainer.soft_labels is True
        assert trainer.n_iterations == 2
        assert trainer.sample_weight_decay == 0.5

    def test_pseudo_label_trainer_no_unlabeled(self):
        """Test PseudoLabelTrainer fit with no unlabeled data."""
        from endgame.nlp.pseudo_label import PseudoLabelTrainer
        from sklearn.linear_model import LogisticRegression

        X_labeled = np.random.randn(50, 5)
        y_labeled = (X_labeled[:, 0] > 0).astype(int)

        base = LogisticRegression(random_state=42)
        trainer = PseudoLabelTrainer(
            base_estimator=base,
            confidence_threshold=0.9,
        )

        # Fit without unlabeled data should just train normally
        trainer.fit(None, X_labeled, y_labeled)
        preds = trainer.predict(X_labeled)
        assert len(preds) == 50

    def test_pseudo_label_trainer_with_unlabeled(self):
        """Test PseudoLabelTrainer with unlabeled data."""
        from endgame.nlp.pseudo_label import PseudoLabelTrainer
        from sklearn.linear_model import LogisticRegression

        np.random.seed(42)
        X_labeled = np.random.randn(50, 5)
        y_labeled = (X_labeled[:, 0] > 0).astype(int)
        X_unlabeled = np.random.randn(30, 5)

        base = LogisticRegression(random_state=42)
        trainer = PseudoLabelTrainer(
            base_estimator=base,
            confidence_threshold=0.7,
            n_iterations=2,
        )

        # First train the base model, then do pseudo-labeling
        base_fitted = LogisticRegression(random_state=42).fit(X_labeled, y_labeled)
        trainer.fit(base_fitted, X_labeled, y_labeled, X_unlabeled=X_unlabeled)

        preds = trainer.predict(X_labeled)
        assert len(preds) == 50

        # Check pseudo labels were generated
        assert trainer.pseudo_labels_ is not None or trainer.pseudo_indices_ is not None
