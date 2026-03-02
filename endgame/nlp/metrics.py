from __future__ import annotations

"""NLP-specific metrics for text generation and translation."""

import warnings
from collections.abc import Callable

import numpy as np


def _check_evaluate():
    """Check if evaluate library is available."""
    try:
        import evaluate
        return True
    except ImportError:
        warnings.warn(
            "evaluate library not installed. "
            "Install with: pip install endgame-ml[nlp]"
        )
        return False


def bleu_score(
    predictions: list[str],
    references: list[str] | list[list[str]],
    lowercase: bool = False,
    tokenize: str | None = None,
) -> dict[str, float]:
    """Compute BLEU score using SacreBLEU.

    BLEU (Bilingual Evaluation Understudy) measures n-gram overlap
    between predictions and references.

    Parameters
    ----------
    predictions : List[str]
        Generated texts.
    references : List[str] or List[List[str]]
        Reference texts. Can be single reference per prediction or
        multiple references.
    lowercase : bool, default=False
        Lowercase before computing.
    tokenize : str, optional
        Tokenizer: '13a' (default), 'intl', 'zh', 'ja-mecab', etc.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'score' (0-100), 'precisions', 'bp', 'ratio', 'counts'.

    Examples
    --------
    >>> preds = ["The cat sat on the mat.", "Hello world"]
    >>> refs = ["The cat is on the mat.", "Hello there world"]
    >>> result = bleu_score(preds, refs)
    >>> print(f"BLEU: {result['score']:.2f}")
    """
    if not _check_evaluate():
        return {'score': 0.0}

    import evaluate

    # Ensure references are list of lists
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    bleu = evaluate.load("sacrebleu")

    compute_kwargs = {}
    if lowercase:
        compute_kwargs['lowercase'] = True
    if tokenize:
        compute_kwargs['tokenize'] = tokenize

    result = bleu.compute(
        predictions=predictions,
        references=references,
        **compute_kwargs,
    )

    return result


def chrf_score(
    predictions: list[str],
    references: list[str] | list[list[str]],
    word_order: int = 2,
    char_order: int = 6,
    beta: float = 2.0,
) -> dict[str, float]:
    """Compute chrF/chrF++ score.

    chrF measures character n-gram overlap. chrF++ (word_order=2) also
    includes word n-grams, making it more robust than BLEU for
    morphologically rich languages.

    Parameters
    ----------
    predictions : List[str]
        Generated texts.
    references : List[str] or List[List[str]]
        Reference texts.
    word_order : int, default=2
        Word n-gram order. 0 for chrF, 2 for chrF++.
    char_order : int, default=6
        Character n-gram order.
    beta : float, default=2.0
        Weight of recall vs precision.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'score' (0-100).

    Examples
    --------
    >>> result = chrf_score(predictions, references, word_order=2)  # chrF++
    >>> print(f"chrF++: {result['score']:.2f}")
    """
    if not _check_evaluate():
        return {'score': 0.0}

    import evaluate

    # Ensure references are list of lists
    if references and isinstance(references[0], str):
        references = [[ref] for ref in references]

    chrf = evaluate.load("chrf")

    result = chrf.compute(
        predictions=predictions,
        references=references,
        word_order=word_order,
        char_order=char_order,
        beta=beta,
    )

    return result


def comet_score(
    sources: list[str],
    predictions: list[str],
    references: list[str],
    model_name: str = "Unbabel/wmt22-comet-da",
    batch_size: int = 8,
    gpus: int = 1,
) -> dict[str, float]:
    """Compute COMET score for translation quality.

    COMET (Crosslingual Optimized Metric for Evaluation of Translation)
    uses neural models to assess translation quality, often correlating
    better with human judgment than BLEU.

    Parameters
    ----------
    sources : List[str]
        Source texts.
    predictions : List[str]
        Generated translations.
    references : List[str]
        Reference translations.
    model_name : str
        COMET model from Hugging Face.
    batch_size : int
        Batch size for inference.
    gpus : int
        Number of GPUs to use.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'score' and optionally 'scores' per sample.

    Examples
    --------
    >>> result = comet_score(sources, predictions, references)
    >>> print(f"COMET: {result['score']:.4f}")
    """
    try:
        import evaluate
        comet = evaluate.load("comet", model_name)
    except Exception as e:
        warnings.warn(f"COMET not available: {e}")
        return {'score': 0.0}

    result = comet.compute(
        sources=sources,
        predictions=predictions,
        references=references,
        batch_size=batch_size,
        gpus=gpus,
    )

    return {
        'score': result['mean_score'],
        'scores': result.get('scores', []),
    }


def bert_score(
    predictions: list[str],
    references: list[str],
    lang: str = "en",
    model_type: str | None = None,
    rescale_with_baseline: bool = True,
) -> dict[str, float]:
    """Compute BERTScore for semantic similarity.

    BERTScore uses BERT embeddings to measure semantic similarity
    between predictions and references.

    Parameters
    ----------
    predictions : List[str]
        Generated texts.
    references : List[str]
        Reference texts.
    lang : str, default='en'
        Language code.
    model_type : str, optional
        Specific BERT model to use.
    rescale_with_baseline : bool, default=True
        Rescale scores using pre-computed baselines.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'precision', 'recall', 'f1'.
    """
    if not _check_evaluate():
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}

    import evaluate
    bertscore = evaluate.load("bertscore")

    result = bertscore.compute(
        predictions=predictions,
        references=references,
        lang=lang,
        model_type=model_type,
        rescale_with_baseline=rescale_with_baseline,
    )

    return {
        'precision': np.mean(result['precision']),
        'recall': np.mean(result['recall']),
        'f1': np.mean(result['f1']),
    }


def rouge_score(
    predictions: list[str],
    references: list[str],
    rouge_types: list[str] | None = None,
    use_stemmer: bool = True,
) -> dict[str, float]:
    """Compute ROUGE scores for summarization.

    ROUGE (Recall-Oriented Understudy for Gisting Evaluation) measures
    overlap between generated and reference summaries.

    Parameters
    ----------
    predictions : List[str]
        Generated summaries.
    references : List[str]
        Reference summaries.
    rouge_types : List[str], optional
        ROUGE types: 'rouge1', 'rouge2', 'rougeL', 'rougeLsum'.
    use_stemmer : bool, default=True
        Use Porter stemmer.

    Returns
    -------
    Dict[str, float]
        Dictionary with scores for each ROUGE type.
    """
    if not _check_evaluate():
        return {}

    import evaluate
    rouge = evaluate.load("rouge")

    result = rouge.compute(
        predictions=predictions,
        references=references,
        rouge_types=rouge_types,
        use_stemmer=use_stemmer,
    )

    return result


def meteor_score(
    predictions: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute METEOR score.

    METEOR (Metric for Evaluation of Translation with Explicit ORdering)
    considers synonyms, stemming, and word order.

    Parameters
    ----------
    predictions : List[str]
        Generated texts.
    references : List[str]
        Reference texts.

    Returns
    -------
    Dict[str, float]
        Dictionary with 'score'.
    """
    if not _check_evaluate():
        return {'score': 0.0}

    import evaluate
    meteor = evaluate.load("meteor")

    result = meteor.compute(predictions=predictions, references=references)

    return result


def geometric_mean(*scores: float) -> float:
    """Compute geometric mean of scores.

    Used in competitions that combine multiple metrics (e.g., BLEU and chrF++).

    Parameters
    ----------
    *scores : float
        Scores to combine (should be positive).

    Returns
    -------
    float
        Geometric mean.

    Examples
    --------
    >>> bleu = 45.2
    >>> chrf = 58.7
    >>> combined = geometric_mean(bleu, chrf)
    >>> print(f"Combined: {combined:.2f}")
    """
    scores = [s for s in scores if s is not None and s > 0]
    if not scores:
        return 0.0
    return float(np.prod(scores) ** (1.0 / len(scores)))


def translation_metrics(
    predictions: list[str],
    references: list[str] | list[list[str]],
    sources: list[str] | None = None,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple translation metrics at once.

    Parameters
    ----------
    predictions : List[str]
        Generated translations.
    references : List[str] or List[List[str]]
        Reference translations.
    sources : List[str], optional
        Source texts (required for COMET).
    metrics : List[str], optional
        Metrics to compute. Default: ['bleu', 'chrf'].
        Options: 'bleu', 'chrf', 'comet', 'bertscore', 'meteor'.

    Returns
    -------
    Dict[str, float]
        Dictionary with all computed metrics.

    Examples
    --------
    >>> results = translation_metrics(preds, refs, metrics=['bleu', 'chrf'])
    >>> print(f"BLEU: {results['bleu']:.2f}")
    >>> print(f"chrF++: {results['chrf']:.2f}")
    >>> print(f"Combined: {results['geometric_mean']:.2f}")
    """
    if metrics is None:
        metrics = ['bleu', 'chrf']

    results = {}

    if 'bleu' in metrics:
        bleu_result = bleu_score(predictions, references)
        results['bleu'] = bleu_result.get('score', 0.0)

    if 'chrf' in metrics:
        chrf_result = chrf_score(predictions, references, word_order=2)
        results['chrf'] = chrf_result.get('score', 0.0)

    if 'comet' in metrics and sources is not None:
        # Flatten references for COMET
        refs_flat = references if isinstance(references[0], str) else [r[0] for r in references]
        comet_result = comet_score(sources, predictions, refs_flat)
        results['comet'] = comet_result.get('score', 0.0)

    if 'bertscore' in metrics:
        refs_flat = references if isinstance(references[0], str) else [r[0] for r in references]
        bert_result = bert_score(predictions, refs_flat)
        results['bertscore_f1'] = bert_result.get('f1', 0.0)

    if 'meteor' in metrics:
        refs_flat = references if isinstance(references[0], str) else [r[0] for r in references]
        meteor_result = meteor_score(predictions, refs_flat)
        results['meteor'] = meteor_result.get('meteor', 0.0)

    # Compute geometric mean of main metrics if we have BLEU and chrF
    if 'bleu' in results and 'chrf' in results:
        results['geometric_mean'] = geometric_mean(results['bleu'], results['chrf'])

    return results


def nlp_metric(metric_name: str) -> Callable:
    """Get NLP metric function by name.

    Parameters
    ----------
    metric_name : str
        Metric name: 'bleu', 'chrf', 'comet', 'bertscore', 'rouge', 'meteor'.

    Returns
    -------
    Callable
        Metric function.

    Examples
    --------
    >>> bleu_fn = nlp_metric('bleu')
    >>> score = bleu_fn(predictions, references)
    """
    metrics_map = {
        'bleu': bleu_score,
        'chrf': chrf_score,
        'chrf++': lambda p, r: chrf_score(p, r, word_order=2),
        'comet': comet_score,
        'bertscore': bert_score,
        'bert_score': bert_score,
        'rouge': rouge_score,
        'meteor': meteor_score,
    }

    metric_lower = metric_name.lower()
    if metric_lower in metrics_map:
        return metrics_map[metric_lower]

    raise ValueError(
        f"Unknown metric: {metric_name}. "
        f"Available: {list(metrics_map.keys())}"
    )
