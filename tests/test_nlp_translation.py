"""Tests for NLP translation module."""

import pytest
import numpy as np


class TestTextPreprocessor:
    """Test TextPreprocessor functionality."""
    
    def test_default_preset(self):
        """Test default preprocessing preset."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = TextPreprocessor.default_preset()
        
        # Test whitespace normalization
        result = preprocessor("  hello   world  ")
        assert result == "hello world"
    
    def test_transliteration_preset(self):
        """Test transliteration preset for ancient languages."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = TextPreprocessor.transliteration_preset()
        
        # Test subscript normalization
        result = preprocessor("a-na₂ šu-mi₃")
        assert "2" in result  # ₂ should become 2
        assert "3" in result  # ₃ should become 3
        
        # Test determinative normalization
        result = preprocessor("before {d}ashur after")
        assert "[d]" in result  # {d} should become [d]
    
    def test_web_text_preset(self):
        """Test web text preprocessing preset."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = TextPreprocessor.web_text_preset()
        
        # Test URL removal
        result = preprocessor("Visit https://example.com for more")
        assert "https://" not in result
        
        # Test HTML removal
        result = preprocessor("<p>Hello</p> <b>World</b>")
        assert "<p>" not in result
        assert "Hello" in result
    
    def test_chaining(self):
        """Test method chaining."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = (
            TextPreprocessor()
            .normalize_unicode()
            .normalize_whitespace()
            .lowercase()
        )
        
        result = preprocessor("  HELLO   World  ")
        assert result == "hello world"
    
    def test_custom_function(self):
        """Test adding custom preprocessing function."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = (
            TextPreprocessor()
            .custom(lambda t: t.replace("foo", "bar"))
        )
        
        result = preprocessor("foo baz foo")
        assert result == "bar baz bar"
    
    def test_batch_processing(self):
        """Test batch text processing."""
        from endgame.nlp import TextPreprocessor
        
        preprocessor = TextPreprocessor.default_preset()
        
        texts = ["  hello  ", "  world  "]
        results = preprocessor.batch(texts)
        
        assert results == ["hello", "world"]


class TestNormalizeFunctions:
    """Test individual normalization functions."""
    
    def test_normalize_subscripts(self):
        """Test subscript normalization."""
        from endgame.nlp import normalize_subscripts
        
        result = normalize_subscripts("text₀₁₂₃₄₅₆₇₈₉")
        assert result == "text0123456789"
        
        # Test superscripts
        result = normalize_subscripts("text⁰¹²³⁴⁵⁶⁷⁸⁹")
        assert result == "text0123456789"
    
    def test_normalize_determinatives(self):
        """Test determinative bracket normalization."""
        from endgame.nlp import normalize_determinatives
        
        result = normalize_determinatives("{d}ashur {lu2}merchant")
        assert result == "[d]ashur [lu2]merchant"
        
        # Test custom brackets
        result = normalize_determinatives(
            "(d)ashur",
            input_brackets="()",
            output_brackets="<>",
        )
        assert result == "<d>ashur"
    
    def test_normalize_whitespace(self):
        """Test whitespace normalization."""
        from endgame.nlp import normalize_whitespace
        
        result = normalize_whitespace("  hello\n\tworld  ")
        assert result == "hello world"
    
    def test_remove_accents(self):
        """Test accent removal."""
        from endgame.nlp import remove_accents
        
        result = remove_accents("café résumé naïve")
        assert result == "cafe resume naive"
    
    def test_truncate_text(self):
        """Test text truncation."""
        from endgame.nlp import truncate_text
        
        text = "Hello World!"
        
        # End truncation
        result = truncate_text(text, 8, "end")
        assert result == "Hello..."
        assert len(result) == 8
        
        # Start truncation
        result = truncate_text(text, 8, "start")
        assert result == "...orld!"
        
        # Middle truncation
        result = truncate_text(text, 10, "middle")
        assert "..." in result
    
    def test_clean_html(self):
        """Test HTML cleaning."""
        from endgame.nlp import clean_html
        
        result = clean_html("<p>Hello</p> <script>bad</script> World")
        assert "<p>" not in result
        assert "<script>" not in result
        assert "Hello" in result
        assert "World" in result
    
    def test_clean_urls(self):
        """Test URL cleaning."""
        from endgame.nlp import clean_urls
        
        result = clean_urls("Visit https://example.com and www.test.org")
        assert "https://" not in result
        assert "www." not in result


class TestNLPMetrics:
    """Test NLP metrics functions."""
    
    def test_geometric_mean(self):
        """Test geometric mean calculation."""
        from endgame.nlp import geometric_mean
        
        # Simple case
        result = geometric_mean(4, 9)
        assert result == 6.0
        
        # Multiple values
        result = geometric_mean(2, 8, 4)
        assert abs(result - 4.0) < 0.001
        
        # Empty case
        result = geometric_mean()
        assert result == 0.0
        
        # Zero case - zeros are filtered out, so only 5 remains
        result = geometric_mean(0, 5)
        assert result == 5.0  # Filters out 0, returns single value
        
        # All zeros
        result = geometric_mean(0, 0)
        assert result == 0.0  # All filtered, returns 0


class TestTranslationConfig:
    """Test TranslationConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        from endgame.nlp import TranslationConfig
        
        config = TranslationConfig()
        
        assert config.model_name == "facebook/nllb-200-distilled-600M"
        assert config.target_lang == "eng_Latn"
        assert config.use_lora is True
        assert config.lora_r == 16
        assert config.batch_size == 8
    
    def test_custom_config(self):
        """Test custom configuration."""
        from endgame.nlp import TranslationConfig
        
        config = TranslationConfig(
            model_name="google/mt5-base",
            use_lora=False,
            batch_size=16,
        )
        
        assert config.model_name == "google/mt5-base"
        assert config.use_lora is False
        assert config.batch_size == 16


class TestTransformerTranslator:
    """Test TransformerTranslator class (without actual model loading)."""
    
    def test_initialization(self):
        """Test translator initialization."""
        from endgame.nlp import TransformerTranslator, TranslationConfig
        
        # Default initialization
        translator = TransformerTranslator()
        assert translator.config.model_name == "facebook/nllb-200-distilled-600M"
        
        # With custom config
        config = TranslationConfig(model_name="google/mt5-base")
        translator = TransformerTranslator(config=config)
        assert translator.config.model_name == "google/mt5-base"
        
        # With kwargs override
        translator = TransformerTranslator(model_name="custom/model")
        assert translator.config.model_name == "custom/model"
    
    def test_model_type_detection(self):
        """Test model type detection."""
        from endgame.nlp import TransformerTranslator
        
        # NLLB
        translator = TransformerTranslator(model_name="facebook/nllb-200-1.3B")
        assert translator._detect_model_type() == "nllb"
        
        # mBART
        translator = TransformerTranslator(model_name="facebook/mbart-large-50")
        assert translator._detect_model_type() == "mbart"
        
        # T5/mT5
        translator = TransformerTranslator(model_name="google/mt5-base")
        assert translator._detect_model_type() == "t5"
        
        # MarianMT
        translator = TransformerTranslator(model_name="Helsinki-NLP/opus-mt-en-de")
        assert translator._detect_model_type() == "marian"


class TestBackTranslator:
    """Test BackTranslator class."""
    
    def test_initialization(self):
        """Test back-translator initialization."""
        from endgame.nlp import BackTranslator
        
        bt = BackTranslator()
        assert bt.forward_model is None
        assert bt.backward_model is None
    
    def test_requires_backward_model(self):
        """Test that generate requires backward model."""
        from endgame.nlp import BackTranslator
        
        bt = BackTranslator()
        
        with pytest.raises(ValueError, match="backward_model required"):
            bt.generate(["test"])


class TestNLPMetricFunction:
    """Test nlp_metric function."""
    
    def test_get_known_metrics(self):
        """Test retrieving known metrics."""
        from endgame.nlp import nlp_metric
        
        # These should return functions
        bleu_fn = nlp_metric("bleu")
        assert callable(bleu_fn)
        
        chrf_fn = nlp_metric("chrf")
        assert callable(chrf_fn)
    
    def test_unknown_metric_raises(self):
        """Test that unknown metric raises error."""
        from endgame.nlp import nlp_metric
        
        with pytest.raises(ValueError, match="Unknown metric"):
            nlp_metric("not_a_real_metric")
