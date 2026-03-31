"""Fuzzy rule extraction from black-box models.

Provides tools for extracting interpretable fuzzy rules from
arbitrary trained models, enabling model interpretability and
knowledge distillation.
"""

from endgame.fuzzy.extraction.rule_extraction import FuzzyRuleExtractor

__all__ = [
    "FuzzyRuleExtractor",
]
