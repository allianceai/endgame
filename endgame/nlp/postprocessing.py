from __future__ import annotations

"""Post-processing utilities for NLP/translation outputs.

Based on techniques from winning Kaggle solutions for translation competitions.
Key insight: LLM post-editing can improve BLEU/chrF++ without changing meaning.
"""

import difflib
import re
from dataclasses import dataclass, field

from endgame.core.base import EndgameEstimator

# =============================================================================
# Text Normalization
# =============================================================================

# Unicode normalization maps
_DASH_MAP = str.maketrans({
    0x2013: "-",  # en-dash
    0x2014: "-",  # em-dash
    0x2212: "-",  # minus sign
})
_QUOTE_MAP = str.maketrans({
    0x201c: '"',  # left double quote "
    0x201d: '"',  # right double quote "
    0x2018: "'",  # left single quote '
    0x2019: "'",  # right single quote '
})


def normalize_punctuation(text: str) -> str:
    """Normalize punctuation spacing and unicode characters.

    - Converts fancy quotes/dashes to ASCII
    - Fixes spacing around punctuation
    - Removes duplicate spaces

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Normalized text.

    Examples
    --------
    >>> normalize_punctuation("Hello ,  world!")
    'Hello, world!'
    >>> normalize_punctuation("He said "hello"")
    'He said "hello"'
    """
    s = text.translate(_DASH_MAP).translate(_QUOTE_MAP)
    s = re.sub(r"[ \t]+", " ", s)  # Multiple spaces -> single
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)  # Remove space before punctuation
    s = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", s)  # Add space after if missing
    return s.strip()


def needs_polish(text: str) -> bool:
    """Check if text has issues that might benefit from polishing.

    Parameters
    ----------
    text : str
        Text to check.

    Returns
    -------
    bool
        True if text has potential issues.
    """
    # Unmatched brackets
    if text.count("[") != text.count("]"):
        return True
    if text.count("(") != text.count(")"):
        return True
    # Spacing issues around punctuation
    if re.search(r"\s+([,.;:!?])", text):
        return True
    # Repeated punctuation
    if re.search(r"([,.;:!?])\1{1,}", text):
        return True
    return False


# =============================================================================
# Safe Edit Validation
# =============================================================================

@dataclass
class SafeEditConfig:
    """Configuration for safe edit validation.

    Parameters
    ----------
    min_similarity : float
        Minimum character-level similarity (0-1).
    max_len_delta : int
        Maximum absolute length change allowed.
    preserve_terms : Set[str]
        Terms that must be preserved exactly.
    """
    min_similarity: float = 0.985
    max_len_delta: int = 12
    preserve_terms: set[str] = field(default_factory=lambda: {
        "Seal of", "son of", "gin", "mina", "shekel"
    })


def _extract_alpha_tokens(text: str) -> list[str]:
    """Extract alphabetic tokens (lowercased) from text."""
    return [t.lower() for t in re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", text)]


def _extract_numbers(text: str) -> list[str]:
    """Extract numeric sequences from text."""
    return re.findall(r"\d+", text)


def is_safe_edit(
    original: str,
    edited: str,
    config: SafeEditConfig | None = None,
) -> bool:
    """Check if an edit is safe (preserves meaning).

    This is crucial for LLM post-editing to prevent hallucination.

    Parameters
    ----------
    original : str
        Original text.
    edited : str
        Edited text.
    config : SafeEditConfig, optional
        Validation configuration.

    Returns
    -------
    bool
        True if edit is safe.

    Examples
    --------
    >>> is_safe_edit("Hello, world", "Hello , world")
    True
    >>> is_safe_edit("Hello, world", "Goodbye, world")
    False
    """
    if config is None:
        config = SafeEditConfig()

    # Empty or whitespace-only edits are invalid
    if not edited or edited.strip() == "":
        return False

    # Words must match (order and content)
    if _extract_alpha_tokens(original) != _extract_alpha_tokens(edited):
        return False

    # Numbers must match
    if _extract_numbers(original) != _extract_numbers(edited):
        return False

    # Bracket counts must match (prevents adding content)
    if original.count("[") != edited.count("["):
        return False
    if original.count("(") != edited.count("("):
        return False

    # Preserve important terms
    for term in config.preserve_terms:
        if term in original and term not in edited:
            return False

    # Character-level similarity check
    similarity = difflib.SequenceMatcher(None, original, edited).ratio()
    if similarity < config.min_similarity:
        return False

    # Length change check
    if abs(len(original) - len(edited)) > config.max_len_delta:
        return False

    return True


# =============================================================================
# LLM Post-Editor
# =============================================================================

def clean_llm_output(text: str) -> str:
    """Clean LLM response to extract just the corrected text.

    Removes common LLM response patterns like "Sure, here's..." etc.

    Parameters
    ----------
    text : str
        Raw LLM output.

    Returns
    -------
    str
        Cleaned text.
    """
    s = text.strip()

    # Remove common prefixes
    s = re.sub(
        r"^(Sure|Here(?:'s| is)|Corrected(?: text)?):\s*",
        "",
        s,
        flags=re.IGNORECASE
    ).strip()

    # Remove code blocks
    if "```" in s:
        s = re.sub(r"```.*?\n", "", s, flags=re.DOTALL)
        s = s.replace("```", "").strip()

    # Handle "model\n" artifacts
    if "model\n" in s:
        s = s.split("model\n")[-1].strip()

    # Remove surrounding quotes
    if len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    # Flatten to single line
    s = " ".join(s.splitlines()).strip()

    return s


DEFAULT_SYSTEM_PROMPT = """You are a deterministic post-editor for MT outputs.
Goal: maximize exact-match metrics (BLEU/chrF). Therefore NEVER paraphrase.

ALLOWED edits (ONLY):
- whitespace normalization (remove double spaces)
- spacing around punctuation , . ; : ! ?
- normalize quotes/dashes to ASCII (' " -)
- if there is an unmatched opening '[' or '(' then ONLY add the missing closing bracket ']' or ')' at the END of the text
- capitalize the first character ONLY if it is a letter AND you do not change any other characters

FORBIDDEN:
- changing, adding, deleting, or reordering ANY words
- changing numbers
- changing proper nouns or names
- adding explanations

Output: the corrected text only (single line). If no edits needed, output the input EXACTLY."""


class LLMPostEditor(EndgameEstimator):
    """LLM-based post-editor for translation outputs.

    Uses an instruction-tuned LLM to fix formatting issues while
    preserving meaning. Includes safety checks to prevent hallucination.

    Parameters
    ----------
    model_name : str
        HuggingFace model name or path. Default is gemma-3-4b-it which
        was used in winning Kaggle solutions.
    system_prompt : str, optional
        System prompt for the LLM.
    max_new_tokens : int
        Maximum tokens to generate.
    safe_edit_config : SafeEditConfig, optional
        Configuration for edit validation.
    device : str, optional
        Device to use ('cuda', 'cpu', or None for auto).
    dtype : str
        Data type ('float16', 'bfloat16', 'float32').

    Examples
    --------
    >>> editor = LLMPostEditor("google/gemma-3-4b-it")
    >>> edited = editor.edit("Hello ,  world !")
    >>> print(edited)
    'Hello, world!'
    """

    def __init__(
        self,
        model_name: str = "google/gemma-3-4b-it",  # Winning solution used this
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_new_tokens: int = 128,
        safe_edit_config: SafeEditConfig | None = None,
        device: str | None = None,
        dtype: str = "bfloat16",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_new_tokens = max_new_tokens
        self.safe_edit_config = safe_edit_config or SafeEditConfig()
        self.device = device
        self.dtype = dtype

        self._model = None
        self._tokenizer = None
        self._cache = {}

    def _load_model(self):
        """Lazy load the model."""
        if self._model is not None:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers and torch required. "
                "Install with: pip install transformers torch"
            )

        self._log(f"Loading model: {self.model_name}")

        # Determine device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(self.dtype, torch.bfloat16)

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map={"": 0} if self.device == "cuda" else None,
            torch_dtype=torch_dtype,
        )

        if self.device == "cpu":
            self._model = self._model.to(self.device)

    def _make_prompt(self, text: str) -> list[dict]:
        """Create chat messages for the LLM."""
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": text},
        ]

    def edit(self, text: str) -> str:
        """Edit a single text.

        Parameters
        ----------
        text : str
            Text to edit.

        Returns
        -------
        str
            Edited text (or original if edit is unsafe).
        """
        import torch

        self._load_model()

        text = str(text)

        # Skip very short or broken texts
        if len(text) < 5 or text == "broken text":
            return text

        # Basic normalization first
        normalized = normalize_punctuation(text)

        # Check if polishing is needed
        if not needs_polish(normalized):
            return normalized

        # Check cache
        if normalized in self._cache:
            return self._cache[normalized]

        # Generate with LLM
        messages = self._make_prompt(normalized)
        prompt_text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._tokenizer(prompt_text, return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.05,
                eos_token_id=self._tokenizer.eos_token_id,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        generated_tokens = outputs[0][input_len:]
        response = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
        response = clean_llm_output(response)

        # Validate edit safety
        if not is_safe_edit(normalized, response, self.safe_edit_config):
            response = normalized

        self._cache[normalized] = response
        return response

    def edit_batch(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[str]:
        """Edit multiple texts.

        Parameters
        ----------
        texts : List[str]
            Texts to edit.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        List[str]
            Edited texts.
        """
        try:
            from tqdm import tqdm
            iterator = tqdm(texts, desc="Polishing") if show_progress else texts
        except ImportError:
            iterator = texts

        return [self.edit(text) for text in iterator]


# =============================================================================
# Translation Post-Processor (Full Pipeline)
# =============================================================================

class TranslationPostProcessor(EndgameEstimator):
    """Complete post-processing pipeline for translation outputs.

    Combines normalization and optional LLM post-editing.

    Parameters
    ----------
    use_llm : bool
        Whether to use LLM for post-editing.
    llm_model : str, optional
        LLM model name (if use_llm=True).
    normalize_only : bool
        If True, only do basic normalization (no LLM).

    Examples
    --------
    >>> pp = TranslationPostProcessor(use_llm=False)
    >>> texts = pp.process(["Hello ,  world !", "Test."])
    """

    def __init__(
        self,
        use_llm: bool = False,
        llm_model: str = "google/gemma-3-4b-it",  # Winning solution model
        normalize_only: bool = False,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.use_llm = use_llm
        self.llm_model = llm_model
        self.normalize_only = normalize_only

        self._llm_editor = None

    def process(
        self,
        texts: list[str],
        show_progress: bool = True,
    ) -> list[str]:
        """Process a list of texts.

        Parameters
        ----------
        texts : List[str]
            Texts to process.
        show_progress : bool
            Whether to show progress bar.

        Returns
        -------
        List[str]
            Processed texts.
        """
        if self.normalize_only or not self.use_llm:
            # Just normalize
            return [normalize_punctuation(str(t)) for t in texts]

        # Use LLM post-editing
        if self._llm_editor is None:
            self._llm_editor = LLMPostEditor(
                model_name=self.llm_model,
                verbose=self.verbose,
            )

        return self._llm_editor.edit_batch(texts, show_progress=show_progress)


__all__ = [
    "normalize_punctuation",
    "needs_polish",
    "is_safe_edit",
    "clean_llm_output",
    "SafeEditConfig",
    "LLMPostEditor",
    "TranslationPostProcessor",
    "DEFAULT_SYSTEM_PROMPT",
]
