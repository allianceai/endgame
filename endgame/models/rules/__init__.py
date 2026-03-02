from __future__ import annotations

"""Rule-based models: RuleFit and FURIA for interpretable machine learning.

This module provides RuleFit implementations that combine the predictive power
of tree ensembles with the interpretability of linear models. Also includes
FURIA, a fuzzy rule-based classifier.
"""

from endgame.models.rules.extraction import (
    extract_rules_from_ensemble,
    extract_rules_from_tree,
)
from endgame.models.rules.furia import FURIAClassifier, FuzzyCondition, FuzzyRule
from endgame.models.rules.rule import Condition, Operator, Rule, RuleEnsemble
from endgame.models.rules.rulefit import RuleFitClassifier, RuleFitRegressor

__all__ = [
    # Main estimators
    "RuleFitRegressor",
    "RuleFitClassifier",
    "FURIAClassifier",
    # Data structures
    "Condition",
    "Operator",
    "Rule",
    "RuleEnsemble",
    "FuzzyRule",
    "FuzzyCondition",
    # Extraction utilities
    "extract_rules_from_tree",
    "extract_rules_from_ensemble",
]
