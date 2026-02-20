"""Text preprocessing utilities for NLP tasks."""

import re
import unicodedata
from collections.abc import Callable


def normalize_unicode(text: str, form: str = "NFC") -> str:
    """Normalize Unicode text.

    Parameters
    ----------
    text : str
        Input text.
    form : str, default='NFC'
        Unicode normalization form: 'NFC', 'NFKC', 'NFD', 'NFKD'.

    Returns
    -------
    str
        Normalized text.
    """
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with normalized whitespace.
    """
    return re.sub(r'\s+', ' ', text).strip()


def remove_accents(text: str) -> str:
    """Remove accents/diacritics from text.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with accents removed.
    """
    nfkd = unicodedata.normalize('NFKD', text)
    return ''.join(c for c in nfkd if not unicodedata.combining(c))


def normalize_subscripts(text: str) -> str:
    """Convert subscript/superscript numbers to regular digits.

    Useful for transliterated ancient languages where subscript numbers
    are common (e.g., Akkadian, Sumerian).

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    str
        Text with normalized numbers.
    """
    subscript_map = {
        '₀': '0', '₁': '1', '₂': '2', '₃': '3', '₄': '4',
        '₅': '5', '₆': '6', '₇': '7', '₈': '8', '₉': '9',
        '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
        '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9',
    }
    for sub, num in subscript_map.items():
        text = text.replace(sub, num)
    return text


def normalize_determinatives(
    text: str,
    input_brackets: str = "{}",
    output_brackets: str = "[]",
) -> str:
    """Normalize determinative markers in transliterated text.

    Determinatives are semantic markers in ancient language transliterations
    (e.g., {d} for divine, {lu₂} for person in Akkadian).

    Parameters
    ----------
    text : str
        Input text.
    input_brackets : str, default='{}'
        Opening and closing brackets to match.
    output_brackets : str, default='[]'
        Opening and closing brackets for output.

    Returns
    -------
    str
        Text with normalized determinatives.
    """
    open_in, close_in = input_brackets[0], input_brackets[1]
    open_out, close_out = output_brackets[0], output_brackets[1]

    pattern = re.escape(open_in) + r'([^' + re.escape(close_in) + r']+)' + re.escape(close_in)
    return re.sub(pattern, f'{open_out}\\1{close_out}', text)


def lowercase_except_markers(
    text: str,
    marker_pattern: str = r'\[[^\]]+\]',
) -> str:
    """Lowercase text except for content in markers.

    Parameters
    ----------
    text : str
        Input text.
    marker_pattern : str
        Regex pattern for markers to preserve.

    Returns
    -------
    str
        Lowercased text with preserved markers.
    """
    markers = []

    def save_marker(match):
        markers.append(match.group(0))
        return f'\x00{len(markers) - 1}\x00'

    text = re.sub(marker_pattern, save_marker, text)
    text = text.lower()

    for i, marker in enumerate(markers):
        text = text.replace(f'\x00{i}\x00', marker)

    return text


def clean_html(text: str) -> str:
    """Remove HTML tags from text.

    Parameters
    ----------
    text : str
        Input text with HTML.

    Returns
    -------
    str
        Text without HTML tags.
    """
    return re.sub(r'<[^>]+>', '', text)


def clean_urls(text: str, replacement: str = " ") -> str:
    """Remove URLs from text.

    Parameters
    ----------
    text : str
        Input text.
    replacement : str, default=' '
        Replacement for URLs.

    Returns
    -------
    str
        Text without URLs.
    """
    url_pattern = r'https?://\S+|www\.\S+'
    return re.sub(url_pattern, replacement, text)


def clean_email(text: str, replacement: str = " ") -> str:
    """Remove email addresses from text.

    Parameters
    ----------
    text : str
        Input text.
    replacement : str, default=' '
        Replacement for emails.

    Returns
    -------
    str
        Text without emails.
    """
    email_pattern = r'\S+@\S+\.\S+'
    return re.sub(email_pattern, replacement, text)


def clean_special_chars(
    text: str,
    keep: str = "",
    replacement: str = " ",
) -> str:
    """Remove special characters except specified ones.

    Parameters
    ----------
    text : str
        Input text.
    keep : str
        Characters to keep besides alphanumeric.
    replacement : str, default=' '
        Replacement for removed characters.

    Returns
    -------
    str
        Cleaned text.
    """
    pattern = f'[^a-zA-Z0-9{re.escape(keep)}\\s]'
    return re.sub(pattern, replacement, text)


def truncate_text(
    text: str,
    max_length: int,
    truncation_strategy: str = "end",
    ellipsis: str = "...",
) -> str:
    """Truncate text to maximum length.

    Parameters
    ----------
    text : str
        Input text.
    max_length : int
        Maximum length.
    truncation_strategy : str, default='end'
        Where to truncate: 'end', 'start', 'middle'.
    ellipsis : str, default='...'
        Ellipsis marker.

    Returns
    -------
    str
        Truncated text.
    """
    if len(text) <= max_length:
        return text

    if truncation_strategy == "end":
        return text[:max_length - len(ellipsis)] + ellipsis
    elif truncation_strategy == "start":
        return ellipsis + text[-(max_length - len(ellipsis)):]
    elif truncation_strategy == "middle":
        half = (max_length - len(ellipsis)) // 2
        return text[:half] + ellipsis + text[-half:]
    else:
        raise ValueError(f"Unknown truncation strategy: {truncation_strategy}")


class TextPreprocessor:
    """Composable text preprocessing pipeline.

    Build preprocessing pipelines by chaining operations.

    Parameters
    ----------
    steps : List[Callable], optional
        List of preprocessing functions.

    Examples
    --------
    >>> # Build a preprocessor for Akkadian transliteration
    >>> preprocessor = (
    ...     TextPreprocessor()
    ...     .normalize_unicode()
    ...     .normalize_whitespace()
    ...     .normalize_subscripts()
    ...     .normalize_determinatives()
    ... )
    >>> clean_text = preprocessor("a-na {d}a-šur₂ qi₂-bi-ma")
    >>>
    >>> # Or use presets
    >>> preprocessor = TextPreprocessor.transliteration_preset()
    """

    def __init__(self, steps: list[Callable] | None = None):
        self.steps: list[Callable] = steps or []

    def __call__(self, text: str) -> str:
        """Apply all preprocessing steps.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        str
            Preprocessed text.
        """
        if not text:
            return ""

        for step in self.steps:
            text = step(text)

        return text

    def batch(self, texts: list[str]) -> list[str]:
        """Apply preprocessing to a batch of texts.

        Parameters
        ----------
        texts : List[str]
            Input texts.

        Returns
        -------
        List[str]
            Preprocessed texts.
        """
        return [self(text) for text in texts]

    def add(self, func: Callable) -> "TextPreprocessor":
        """Add a preprocessing function.

        Parameters
        ----------
        func : Callable
            Function that takes str and returns str.

        Returns
        -------
        TextPreprocessor
            Self for chaining.
        """
        self.steps.append(func)
        return self

    def normalize_unicode(self, form: str = "NFC") -> "TextPreprocessor":
        """Add Unicode normalization."""
        return self.add(lambda t: normalize_unicode(t, form))

    def normalize_whitespace(self) -> "TextPreprocessor":
        """Add whitespace normalization."""
        return self.add(normalize_whitespace)

    def lowercase(self) -> "TextPreprocessor":
        """Add lowercasing."""
        return self.add(str.lower)

    def uppercase(self) -> "TextPreprocessor":
        """Add uppercasing."""
        return self.add(str.upper)

    def strip(self) -> "TextPreprocessor":
        """Add stripping."""
        return self.add(str.strip)

    def remove_accents(self) -> "TextPreprocessor":
        """Add accent removal."""
        return self.add(remove_accents)

    def normalize_subscripts(self) -> "TextPreprocessor":
        """Add subscript normalization."""
        return self.add(normalize_subscripts)

    def normalize_determinatives(
        self,
        input_brackets: str = "{}",
        output_brackets: str = "[]",
    ) -> "TextPreprocessor":
        """Add determinative normalization."""
        return self.add(
            lambda t: normalize_determinatives(t, input_brackets, output_brackets)
        )

    def clean_html(self) -> "TextPreprocessor":
        """Add HTML cleaning."""
        return self.add(clean_html)

    def clean_urls(self, replacement: str = " ") -> "TextPreprocessor":
        """Add URL cleaning."""
        return self.add(lambda t: clean_urls(t, replacement))

    def clean_email(self, replacement: str = " ") -> "TextPreprocessor":
        """Add email cleaning."""
        return self.add(lambda t: clean_email(t, replacement))

    def regex_replace(
        self,
        pattern: str,
        replacement: str,
        flags: int = 0,
    ) -> "TextPreprocessor":
        """Add regex replacement."""
        compiled = re.compile(pattern, flags)
        return self.add(lambda t: compiled.sub(replacement, t))

    def truncate(
        self,
        max_length: int,
        strategy: str = "end",
        ellipsis: str = "...",
    ) -> "TextPreprocessor":
        """Add truncation."""
        return self.add(
            lambda t: truncate_text(t, max_length, strategy, ellipsis)
        )

    def custom(self, func: Callable[[str], str]) -> "TextPreprocessor":
        """Add a custom preprocessing function.

        Parameters
        ----------
        func : Callable
            Custom function.

        Returns
        -------
        TextPreprocessor
            Self for chaining.
        """
        return self.add(func)

    @classmethod
    def default_preset(cls) -> "TextPreprocessor":
        """Create preprocessor with sensible defaults.

        Returns
        -------
        TextPreprocessor
            Preprocessor with: unicode normalization, whitespace normalization.
        """
        return (
            cls()
            .normalize_unicode()
            .normalize_whitespace()
        )

    @classmethod
    def transliteration_preset(cls) -> "TextPreprocessor":
        """Create preprocessor for transliterated ancient languages.

        Suitable for Akkadian, Sumerian, Hittite, etc.

        Returns
        -------
        TextPreprocessor
            Preprocessor with: unicode, whitespace, subscripts, determinatives.
        """
        return (
            cls()
            .normalize_unicode()
            .normalize_whitespace()
            .normalize_subscripts()
            .normalize_determinatives()
        )

    @classmethod
    def web_text_preset(cls) -> "TextPreprocessor":
        """Create preprocessor for web-scraped text.

        Returns
        -------
        TextPreprocessor
            Preprocessor with: HTML, URL, email cleaning.
        """
        return (
            cls()
            .clean_html()
            .clean_urls()
            .clean_email()
            .normalize_whitespace()
        )

    @classmethod
    def kaggle_preset(cls) -> "TextPreprocessor":
        """Create preprocessor for typical Kaggle NLP competitions.

        Returns
        -------
        TextPreprocessor
            Preprocessor with common cleaning steps.
        """
        return (
            cls()
            .normalize_unicode()
            .clean_html()
            .clean_urls()
            .clean_email()
            .normalize_whitespace()
        )
