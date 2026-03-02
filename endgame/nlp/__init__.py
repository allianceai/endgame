from __future__ import annotations

"""NLP module: Transformers, translation, DAPT, pseudo-labeling, tokenizer utilities, LLM."""

from endgame.nlp.dapt import DomainAdaptivePretrainer
from endgame.nlp.llm import (
    ChainOfThoughtPrompt,
    GenerationResult,
    LLMWrapper,
    SyntheticDataGenerator,
)
from endgame.nlp.metrics import (
    bert_score,
    bleu_score,
    chrf_score,
    comet_score,
    geometric_mean,
    meteor_score,
    nlp_metric,
    rouge_score,
    translation_metrics,
)
from endgame.nlp.postprocessing import (
    LLMPostEditor,
    SafeEditConfig,
    TranslationPostProcessor,
    clean_llm_output,
    is_safe_edit,
    needs_polish,
    normalize_punctuation,
)
from endgame.nlp.preprocessing import (
    TextPreprocessor,
    clean_email,
    clean_html,
    clean_urls,
    normalize_determinatives,
    normalize_subscripts,
    normalize_unicode,
    normalize_whitespace,
    remove_accents,
    truncate_text,
)
from endgame.nlp.pseudo_label import PseudoLabelTrainer
from endgame.nlp.tokenizer import TokenizationReport, TokenizerOptimizer
from endgame.nlp.transformers import TransformerClassifier, TransformerRegressor
from endgame.nlp.translation import (
    BackTranslator,
    TransformerTranslator,
    TranslationConfig,
)

__all__ = [
    # Classification/Regression
    "TransformerClassifier",
    "TransformerRegressor",
    # Translation
    "TransformerTranslator",
    "TranslationConfig",
    "BackTranslator",
    # Pre-training
    "DomainAdaptivePretrainer",
    "PseudoLabelTrainer",
    # Tokenization
    "TokenizerOptimizer",
    "TokenizationReport",
    # LLM
    "LLMWrapper",
    "ChainOfThoughtPrompt",
    "SyntheticDataGenerator",
    "GenerationResult",
    # Metrics
    "bleu_score",
    "chrf_score",
    "comet_score",
    "bert_score",
    "rouge_score",
    "meteor_score",
    "geometric_mean",
    "translation_metrics",
    "nlp_metric",
    # Preprocessing
    "TextPreprocessor",
    "normalize_unicode",
    "normalize_whitespace",
    "remove_accents",
    "normalize_subscripts",
    "normalize_determinatives",
    "clean_html",
    "clean_urls",
    "clean_email",
    "truncate_text",
    # Post-processing
    "normalize_punctuation",
    "needs_polish",
    "is_safe_edit",
    "clean_llm_output",
    "SafeEditConfig",
    "LLMPostEditor",
    "TranslationPostProcessor",
]
