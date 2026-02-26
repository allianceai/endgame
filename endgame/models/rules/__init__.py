"""Rule-based models: RuleFit and RuleFit++ for interpretable machine learning.

This module provides RuleFit implementations that combine the predictive power
of tree ensembles with the interpretability of linear models. Also includes
RuleFit++ (enhanced with soft rules, multi-source generation, elastic net)
and FURIA, a fuzzy rule-based classifier.
"""

from endgame.models.rules.extraction import (
    extract_rules_from_ensemble,
    extract_rules_from_tree,
)
from endgame.models.rules.furia import FURIAClassifier, FuzzyCondition, FuzzyRule
from endgame.models.rules.rule import Condition, Operator, Rule, RuleEnsemble
from endgame.models.rules.rulefit import RuleFitClassifier, RuleFitRegressor
from endgame.models.rules.rulefit_plus import (
    RuleFitPlusClassifier,
    RuleFitPlusRegressor,
)

__all__ = [
    # Main estimators
    "RuleFitRegressor",
    "RuleFitClassifier",
    "RuleFitPlusRegressor",
    "RuleFitPlusClassifier",
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
