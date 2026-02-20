"""Rule and RuleEnsemble data structures for RuleFit.

This module contains the core data structures for representing rules extracted
from decision trees: Condition, Rule, and RuleEnsemble.
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class Operator(Enum):
    """Comparison operators for rule conditions."""

    LE = "<="  # Less than or equal
    GT = ">"  # Greater than
    EQ = "=="  # Equal (for categorical)
    NE = "!="  # Not equal (for categorical)
    IN = "in"  # In set (for categorical)
    NOT_IN = "not in"  # Not in set


@dataclass
class Condition:
    """
    A single condition in a rule.

    Represents: feature_name <operator> threshold

    Attributes
    ----------
    feature_idx : int
        Index of the feature in the input array.
    feature_name : str
        Human-readable feature name.
    operator : Operator
        Comparison operator.
    threshold : float or set
        Threshold value (or set of values for categorical).
    """

    feature_idx: int
    feature_name: str
    operator: Operator
    threshold: float | set

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate condition on data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        satisfied : ndarray of shape (n_samples,) dtype=bool
            True where condition is satisfied.
        """
        feature_values = X[:, self.feature_idx]

        if self.operator == Operator.LE:
            return feature_values <= self.threshold
        elif self.operator == Operator.GT:
            return feature_values > self.threshold
        elif self.operator == Operator.EQ:
            return feature_values == self.threshold
        elif self.operator == Operator.NE:
            return feature_values != self.threshold
        elif self.operator == Operator.IN:
            return np.isin(feature_values, list(self.threshold))
        elif self.operator == Operator.NOT_IN:
            return ~np.isin(feature_values, list(self.threshold))
        else:
            raise ValueError(f"Unknown operator: {self.operator}")

    def __str__(self) -> str:
        if self.operator == Operator.LE:
            return f"{self.feature_name} <= {self.threshold:.4g}"
        elif self.operator == Operator.GT:
            return f"{self.feature_name} > {self.threshold:.4g}"
        elif self.operator == Operator.EQ:
            return f"{self.feature_name} == {self.threshold}"
        elif self.operator == Operator.NE:
            return f"{self.feature_name} != {self.threshold}"
        elif self.operator == Operator.IN:
            return f"{self.feature_name} in {self.threshold}"
        elif self.operator == Operator.NOT_IN:
            return f"{self.feature_name} not in {self.threshold}"
        return f"{self.feature_name} {self.operator.value} {self.threshold}"

    def __hash__(self):
        if isinstance(self.threshold, set):
            thresh = frozenset(self.threshold)
        else:
            thresh = self.threshold
        return hash((self.feature_idx, self.operator, thresh))

    def __eq__(self, other):
        if not isinstance(other, Condition):
            return False
        return (
            self.feature_idx == other.feature_idx
            and self.operator == other.operator
            and self.threshold == other.threshold
        )


@dataclass
class Rule:
    """
    A rule consisting of multiple conditions (conjunctions).

    A rule is satisfied if ALL conditions are satisfied (AND logic).

    Attributes
    ----------
    conditions : list of Condition
        The conditions that make up this rule.
    support : float
        Fraction of training samples satisfying this rule.
    coefficient : float
        Lasso coefficient for this rule (set after fitting).
    tree_idx : int
        Index of the tree this rule came from.
    node_idx : int
        Index of the node in the tree.
    """

    conditions: list[Condition] = field(default_factory=list)
    support: float = 0.0
    coefficient: float = 0.0
    tree_idx: int = -1
    node_idx: int = -1

    @property
    def length(self) -> int:
        """Number of conditions in the rule."""
        return len(self.conditions)

    @property
    def feature_indices(self) -> list[int]:
        """List of feature indices involved in this rule."""
        return list(set(c.feature_idx for c in self.conditions))

    @property
    def importance(self) -> float:
        """Rule importance: |coefficient| * support."""
        return abs(self.coefficient) * self.support

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate rule on data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        satisfied : ndarray of shape (n_samples,) dtype=bool
            True where all conditions are satisfied.
        """
        if len(self.conditions) == 0:
            return np.ones(X.shape[0], dtype=bool)

        result = np.ones(X.shape[0], dtype=bool)
        for condition in self.conditions:
            result &= condition.evaluate(X)
        return result

    def __str__(self) -> str:
        if len(self.conditions) == 0:
            return "(intercept)"
        return " AND ".join(str(c) for c in self.conditions)

    def __hash__(self):
        # Sort conditions for consistent hashing
        return hash(
            tuple(sorted(self.conditions, key=lambda c: (c.feature_idx, c.operator.value)))
        )

    def __eq__(self, other):
        if not isinstance(other, Rule):
            return False
        return set(self.conditions) == set(other.conditions)

    def to_dict(self) -> dict:
        """Convert rule to dictionary representation."""
        return {
            "rule": str(self),
            "coefficient": self.coefficient,
            "support": self.support,
            "importance": self.importance,
            "length": self.length,
            "conditions": self.conditions,
            "feature_indices": self.feature_indices,
        }


@dataclass
class RuleEnsemble:
    """
    Collection of rules extracted from a tree ensemble.

    Attributes
    ----------
    rules : list of Rule
        All extracted rules.
    n_features : int
        Number of input features.
    feature_names : list of str
        Names of input features.
    """

    rules: list[Rule] = field(default_factory=list)
    n_features: int = 0
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.rules)

    def _compile(self):
        """Precompile rules into numpy arrays for fast vectorized evaluation.

        Only rules using LE/GT operators (i.e. all tree-extracted rules) are
        handled by the fast path.  Rules with other operators (EQ, IN, etc.)
        fall back to per-rule Python evaluation.
        """
        n_rules = len(self.rules)
        if n_rules == 0:
            self._compiled = True
            self._max_conds = 0
            self._fast_mask = np.zeros(0, dtype=bool)
            return

        max_conds = max(len(r.conditions) for r in self.rules)

        feat_idx = np.zeros((n_rules, max_conds), dtype=np.intp)
        thresholds = np.zeros((n_rules, max_conds), dtype=np.float64)
        is_le = np.zeros((n_rules, max_conds), dtype=bool)
        n_conds = np.zeros(n_rules, dtype=np.int32)
        fast_mask = np.ones(n_rules, dtype=bool)

        for j, rule in enumerate(self.rules):
            nc = len(rule.conditions)
            n_conds[j] = nc
            for k, c in enumerate(rule.conditions):
                if c.operator not in (Operator.LE, Operator.GT):
                    fast_mask[j] = False
                    break
                feat_idx[j, k] = c.feature_idx
                thresholds[j, k] = c.threshold
                is_le[j, k] = c.operator == Operator.LE

        self._c_feat_idx = feat_idx
        self._c_thresholds = thresholds
        self._c_is_le = is_le
        self._c_n_conds = n_conds
        self._max_conds = max_conds
        self._fast_mask = fast_mask
        self._compiled = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform X into binary rule features.

        Uses a vectorized fast path for tree-extracted rules (LE/GT only),
        falling back to per-rule evaluation for rules with other operators.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_rules : ndarray of shape (n_samples, n_rules)
            Binary matrix where X_rules[i, j] = 1 if sample i
            satisfies rule j.
        """
        n_samples = X.shape[0]
        n_rules = len(self.rules)

        if n_rules == 0:
            return np.zeros((n_samples, 0), dtype=np.float32)

        if not getattr(self, "_compiled", False):
            self._compile()

        result = np.ones((n_samples, n_rules), dtype=np.float32)

        fast = self._fast_mask
        if fast.any():
            fast_idx = np.where(fast)[0]
            for k in range(self._max_conds):
                active = fast_idx[self._c_n_conds[fast_idx] > k]
                if len(active) == 0:
                    break
                fi = self._c_feat_idx[active, k]
                th = self._c_thresholds[active, k]
                le = self._c_is_le[active, k]
                vals = X[:, fi]
                cond = np.where(le, vals <= th, vals > th)
                result[:, active] *= cond.astype(np.float32)

        slow_idx = np.where(~fast)[0]
        for j in slow_idx:
            result[:, j] = self.rules[j].evaluate(X).astype(np.float32)

        return result

    def filter_by_support(
        self, min_support: float = 0.01, max_support: float = 0.99
    ) -> "RuleEnsemble":
        """
        Filter rules by support thresholds.

        Parameters
        ----------
        min_support : float
            Minimum support (removes too-specific rules).
        max_support : float
            Maximum support (removes too-general rules).

        Returns
        -------
        filtered : RuleEnsemble
            New RuleEnsemble with filtered rules.
        """
        filtered_rules = [
            r for r in self.rules if min_support <= r.support <= max_support
        ]

        return RuleEnsemble(
            rules=filtered_rules,
            n_features=self.n_features,
            feature_names=self.feature_names,
        )

    def deduplicate(self) -> "RuleEnsemble":
        """
        Remove duplicate rules.

        Returns
        -------
        deduped : RuleEnsemble
            New RuleEnsemble with unique rules only.
        """
        seen = set()
        unique_rules = []

        for rule in self.rules:
            rule_hash = hash(rule)
            if rule_hash not in seen:
                seen.add(rule_hash)
                unique_rules.append(rule)

        return RuleEnsemble(
            rules=unique_rules,
            n_features=self.n_features,
            feature_names=self.feature_names,
        )

    def limit_rules(self, max_rules: int) -> "RuleEnsemble":
        """
        Limit number of rules, keeping those with highest support.

        Parameters
        ----------
        max_rules : int
            Maximum number of rules to keep.

        Returns
        -------
        limited : RuleEnsemble
            New RuleEnsemble with at most max_rules rules.
        """
        if len(self.rules) <= max_rules:
            return self

        # Sort by support (descending) and keep top max_rules
        sorted_rules = sorted(self.rules, key=lambda r: r.support, reverse=True)

        return RuleEnsemble(
            rules=sorted_rules[:max_rules],
            n_features=self.n_features,
            feature_names=self.feature_names,
        )
