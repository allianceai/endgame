"""Feature Selection Module.

This module provides a comprehensive suite of feature selection methods
organized into three categories:

1. **Filter Methods** - Model-agnostic, fast, based on statistical tests
2. **Wrapper Methods** - Model-dependent, use a model to evaluate subsets
3. **Importance Methods** - Based on feature importance from fitted models

Classes
-------
Filter Methods:
    UnivariateSelector : F-test, mutual information, chi-squared
    MRMRSelector : Minimum Redundancy Maximum Relevance
    ReliefFSelector : ReliefF and MultiSURF algorithms
    CorrelationSelector : Remove highly correlated features

Wrapper Methods:
    RFESelector : Recursive Feature Elimination
    BorutaSelector : Boruta algorithm (shadow features)
    SequentialSelector : Forward/backward selection
    GeneticSelector : Evolutionary feature selection

Importance Methods:
    PermutationImportanceSelector : Permutation importance-based
    SHAPSelector : SHAP value-based selection
    TreeImportanceSelector : Tree-based importance

Advanced:
    StabilitySelector : Stability selection wrapper
    KnockoffSelector : Knockoff filter with FDR control
    NullImportanceSelector : Null importance testing

Example
-------
>>> from endgame.feature_selection import MRMRSelector, BorutaSelector
>>> # Fast filter
>>> mrmr = MRMRSelector(n_features=50)
>>> X_filtered = mrmr.fit_transform(X, y)
>>> # Thorough wrapper
>>> boruta = BorutaSelector()
>>> X_selected = boruta.fit_transform(X_filtered, y)
"""

# Filter methods
from endgame.feature_selection.filter.correlation import CorrelationSelector
from endgame.feature_selection.filter.mrmr import MRMRSelector
from endgame.feature_selection.filter.relief import ReliefFSelector
from endgame.feature_selection.filter.univariate import (
    Chi2Selector,
    FTestSelector,
    MutualInfoSelector,
    UnivariateSelector,
)

# Importance methods
from endgame.feature_selection.importance.permutation import PermutationSelector
from endgame.feature_selection.importance.shap_importance import SHAPSelector
from endgame.feature_selection.importance.tree import TreeImportanceSelector
from endgame.feature_selection.knockoff import KnockoffSelector

# Advanced methods
from endgame.feature_selection.stability import StabilitySelector
from endgame.feature_selection.wrapper.boruta import BorutaSelector
from endgame.feature_selection.wrapper.genetic import GeneticSelector

# Wrapper methods
from endgame.feature_selection.wrapper.rfe import RFESelector
from endgame.feature_selection.wrapper.sequential import SequentialSelector

# Re-export from preprocessing (existing implementations)
from endgame.preprocessing.selection import (
    AdversarialFeatureSelector,
    NullImportanceSelector,
)

__all__ = [
    # Filter
    "UnivariateSelector",
    "MutualInfoSelector",
    "FTestSelector",
    "Chi2Selector",
    "MRMRSelector",
    "ReliefFSelector",
    "CorrelationSelector",
    # Wrapper
    "RFESelector",
    "BorutaSelector",
    "SequentialSelector",
    "GeneticSelector",
    # Importance
    "PermutationSelector",
    "SHAPSelector",
    "TreeImportanceSelector",
    # Advanced
    "StabilitySelector",
    "KnockoffSelector",
    "AdversarialFeatureSelector",
    "NullImportanceSelector",
]
