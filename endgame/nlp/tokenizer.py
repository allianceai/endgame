"""Tokenizer analysis and optimization utilities.

Identifies tokenization artifacts that hurt performance and suggests fixes.
Key insight from Linking Writing Processes competition: replacing certain
characters improved scores by fixing tokenization issues.
"""

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TokenizationReport:
    """Report from tokenization analysis.

    Attributes
    ----------
    inefficient_tokens : Dict[str, int]
        Strings that tokenize poorly (single char → multiple tokens).
    oov_tokens : Dict[str, int]
        Out-of-vocabulary tokens and their frequencies.
    avg_sequence_length : float
        Average number of tokens per text.
    max_sequence_length : int
        Maximum number of tokens in any text.
    vocab_coverage : float
        Fraction of unique words in vocabulary.
    suggested_replacements : Dict[str, str]
        Suggested text replacements to improve tokenization.
    token_frequency : Dict[str, int]
        Most common tokens and their counts.
    """

    inefficient_tokens: dict[str, int] = field(default_factory=dict)
    oov_tokens: dict[str, int] = field(default_factory=dict)
    avg_sequence_length: float = 0.0
    max_sequence_length: int = 0
    vocab_coverage: float = 0.0
    suggested_replacements: dict[str, str] = field(default_factory=dict)
    token_frequency: dict[str, int] = field(default_factory=dict)


class TokenizerOptimizer:
    """Tokenizer analysis and optimization utilities.

    Analyzes tokenization efficiency and suggests improvements.
    Particularly useful for domain-specific text where tokenizers
    may split common terms suboptimally.

    Examples
    --------
    >>> from endgame.nlp import TokenizerOptimizer
    >>> from transformers import AutoTokenizer
    >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    >>> optimizer = TokenizerOptimizer()
    >>> # Analyze tokenization
    >>> report = optimizer.analyze_tokenization(texts, tokenizer)
    >>> print(f"Average sequence length: {report.avg_sequence_length:.1f}")
    >>> print(f"Suggested replacements: {report.suggested_replacements}")
    >>> # Apply optimizations
    >>> optimized_texts = optimizer.optimize_special_tokens(
    ...     texts, patterns=report.suggested_replacements
    ... )
    """

    def __init__(self):
        """Initialize TokenizerOptimizer."""
        self._common_inefficient_patterns = {
            # Common mathematical/scientific symbols
            "≤": "<=",
            "≥": ">=",
            "≠": "!=",
            "×": "*",
            "÷": "/",
            "±": "+/-",
            "√": "sqrt",
            "∞": "infinity",
            "π": "pi",
            "∑": "sum",
            "∏": "product",
            "∫": "integral",
            # Common typography
            "'": "'",
            """: '"',
            """: '"',
            "…": "...",
            "—": "-",
            "–": "-",
            "•": "*",
            # Currency
            "€": "EUR",
            "£": "GBP",
            "¥": "JPY",
            # Common accented characters (if causing issues)
            "é": "e",
            "è": "e",
            "ê": "e",
            "ë": "e",
            "à": "a",
            "â": "a",
            "ô": "o",
            "ù": "u",
            "ç": "c",
        }

    def analyze_tokenization(
        self,
        texts: list[str],
        tokenizer: Any,
        max_samples: int = 10000,
        check_efficiency: bool = True,
    ) -> TokenizationReport:
        """Analyze tokenization efficiency.

        Parameters
        ----------
        texts : List[str]
            Text samples to analyze.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer to analyze.
        max_samples : int, default=10000
            Maximum number of samples to analyze.
        check_efficiency : bool, default=True
            Whether to check for inefficient tokenizations.

        Returns
        -------
        TokenizationReport
            Analysis results including inefficiencies and suggestions.
        """
        # Sample if too many texts
        if len(texts) > max_samples:
            indices = np.random.choice(len(texts), max_samples, replace=False)
            texts = [texts[i] for i in indices]

        report = TokenizationReport()

        # Collect statistics
        sequence_lengths = []
        token_counter = Counter()
        word_counter = Counter()
        inefficient_counter = Counter()
        oov_counter = Counter()

        # Get special tokens
        special_tokens = set()
        if hasattr(tokenizer, "all_special_tokens"):
            special_tokens = set(tokenizer.all_special_tokens)

        # Get vocabulary
        vocab = set()
        if hasattr(tokenizer, "vocab"):
            vocab = set(tokenizer.vocab.keys())
        elif hasattr(tokenizer, "get_vocab"):
            vocab = set(tokenizer.get_vocab().keys())

        # Analyze each text
        for text in texts:
            # Tokenize
            encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])
            sequence_lengths.append(len(tokens))

            # Count tokens
            for token in tokens:
                if token not in special_tokens:
                    token_counter[token] += 1

            # Check for OOV
            if hasattr(tokenizer, "unk_token"):
                unk_token = tokenizer.unk_token
                for token in tokens:
                    if token == unk_token:
                        # Try to get original word
                        oov_counter["[UNK]"] += 1

            # Collect unique words
            words = text.split()
            for word in words:
                word_counter[word] += 1

            # Check tokenization efficiency
            if check_efficiency:
                self._check_efficiency(text, tokenizer, inefficient_counter)

        # Compute statistics
        report.avg_sequence_length = np.mean(sequence_lengths)
        report.max_sequence_length = max(sequence_lengths) if sequence_lengths else 0

        # Vocabulary coverage
        unique_words = set(word_counter.keys())
        words_in_vocab = sum(1 for w in unique_words if w.lower() in vocab or w in vocab)
        report.vocab_coverage = words_in_vocab / len(unique_words) if unique_words else 1.0

        # Most common tokens
        report.token_frequency = dict(token_counter.most_common(100))

        # Inefficient tokens
        report.inefficient_tokens = dict(inefficient_counter.most_common(50))

        # OOV tokens
        report.oov_tokens = dict(oov_counter.most_common(50))

        # Generate suggestions
        report.suggested_replacements = self._generate_suggestions(
            inefficient_counter, texts, tokenizer
        )

        return report

    def _check_efficiency(
        self,
        text: str,
        tokenizer: Any,
        counter: Counter,
    ):
        """Check for inefficient tokenizations."""
        # Check single characters that become multiple tokens
        for char in set(text):
            if len(char) == 1 and not char.isalnum() and not char.isspace():
                tokens = tokenizer.tokenize(char)
                if len(tokens) > 1:
                    counter[char] += text.count(char)

        # Check common patterns
        for pattern in self._common_inefficient_patterns:
            if pattern in text:
                tokens = tokenizer.tokenize(pattern)
                if len(tokens) > 1:
                    counter[pattern] += text.count(pattern)

    def _generate_suggestions(
        self,
        inefficient_counter: Counter,
        texts: list[str],
        tokenizer: Any,
    ) -> dict[str, str]:
        """Generate replacement suggestions."""
        suggestions = {}

        for pattern, count in inefficient_counter.most_common(20):
            if count < 10:  # Skip rare patterns
                continue

            # Check if we have a known replacement
            if pattern in self._common_inefficient_patterns:
                replacement = self._common_inefficient_patterns[pattern]
                # Verify replacement is better
                orig_tokens = len(tokenizer.tokenize(pattern))
                new_tokens = len(tokenizer.tokenize(replacement))
                if new_tokens < orig_tokens:
                    suggestions[pattern] = replacement

        return suggestions

    def optimize_special_tokens(
        self,
        texts: list[str],
        patterns: dict[str, str],
    ) -> list[str]:
        """Apply replacement patterns to texts.

        Parameters
        ----------
        texts : List[str]
            Input texts.
        patterns : Dict[str, str]
            Mapping from original patterns to replacements.

        Returns
        -------
        List[str]
            Optimized texts with replacements applied.
        """
        optimized = []
        for text in texts:
            for old, new in patterns.items():
                text = text.replace(old, new)
            optimized.append(text)
        return optimized

    def find_repeated_ngrams(
        self,
        texts: list[str],
        tokenizer: Any,
        min_n: int = 2,
        max_n: int = 5,
        min_count: int = 100,
    ) -> dict[str, int]:
        """Find frequently repeated n-grams that could be added to vocabulary.

        Parameters
        ----------
        texts : List[str]
            Text samples.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer to analyze.
        min_n : int, default=2
            Minimum n-gram length (in tokens).
        max_n : int, default=5
            Maximum n-gram length (in tokens).
        min_count : int, default=100
            Minimum count to include.

        Returns
        -------
        Dict[str, int]
            N-grams and their counts, sorted by count.
        """
        ngram_counter = Counter()

        for text in texts:
            tokens = tokenizer.tokenize(text)

            for n in range(min_n, min(max_n + 1, len(tokens) + 1)):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i + n])
                    # Only count if not starting/ending with subword markers
                    if not any(t.startswith("##") for t in ngram[1:]):
                        ngram_counter[" ".join(ngram)] += 1

        # Filter by count
        filtered = {k: v for k, v in ngram_counter.items() if v >= min_count}

        # Sort by count
        return dict(sorted(filtered.items(), key=lambda x: -x[1]))

    def compute_tokenization_stats(
        self,
        texts: list[str],
        tokenizer: Any,
    ) -> dict[str, Any]:
        """Compute detailed tokenization statistics.

        Parameters
        ----------
        texts : List[str]
            Text samples.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer to analyze.

        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - avg_tokens_per_word: average tokens per whitespace word
            - avg_chars_per_token: average characters per token
            - truncation_rate: fraction of texts exceeding max_length
            - percentiles: sequence length percentiles
        """
        tokens_per_word = []
        chars_per_token = []
        sequence_lengths = []

        max_length = getattr(tokenizer, "model_max_length", 512)
        truncated = 0

        for text in texts:
            words = text.split()
            encoding = tokenizer(text, add_special_tokens=False)
            tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

            n_tokens = len(tokens)
            sequence_lengths.append(n_tokens)

            if n_tokens > max_length:
                truncated += 1

            if words:
                tokens_per_word.append(n_tokens / len(words))

            total_chars = sum(len(t.replace("##", "").replace("Ġ", "")) for t in tokens)
            if n_tokens > 0:
                chars_per_token.append(total_chars / n_tokens)

        return {
            "avg_tokens_per_word": np.mean(tokens_per_word) if tokens_per_word else 0,
            "avg_chars_per_token": np.mean(chars_per_token) if chars_per_token else 0,
            "truncation_rate": truncated / len(texts) if texts else 0,
            "sequence_length_percentiles": {
                "p50": np.percentile(sequence_lengths, 50),
                "p75": np.percentile(sequence_lengths, 75),
                "p90": np.percentile(sequence_lengths, 90),
                "p95": np.percentile(sequence_lengths, 95),
                "p99": np.percentile(sequence_lengths, 99),
            } if sequence_lengths else {},
        }

    def suggest_max_length(
        self,
        texts: list[str],
        tokenizer: Any,
        coverage: float = 0.95,
    ) -> int:
        """Suggest optimal max_length based on sequence length distribution.

        Parameters
        ----------
        texts : List[str]
            Text samples.
        tokenizer : transformers.PreTrainedTokenizer
            Tokenizer to analyze.
        coverage : float, default=0.95
            Desired coverage (fraction of texts that won't be truncated).

        Returns
        -------
        int
            Suggested max_length.
        """
        sequence_lengths = []

        for text in texts:
            encoding = tokenizer(text, add_special_tokens=True)
            sequence_lengths.append(len(encoding["input_ids"]))

        if not sequence_lengths:
            return 512

        suggested = int(np.percentile(sequence_lengths, coverage * 100))

        # Round up to nearest power of 2 or multiple of 64
        candidates = [64, 128, 256, 384, 512, 768, 1024, 2048]
        for candidate in candidates:
            if candidate >= suggested:
                return candidate

        return suggested
