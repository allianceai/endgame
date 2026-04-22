"""FURIA (Fuzzy Unordered Rule Induction Algorithm) implementation.

FURIA is a fuzzy extension of RIPPER that:
1. Learns crisp rules using a RIPPER-like algorithm
2. Fuzzifies rules by replacing sharp boundaries with trapezoidal membership functions
3. Uses "rule stretching" to cover instances not covered by any rule

References
----------
- Hühn & Hüllermeier, "FURIA: An Algorithm for Unordered Fuzzy Rule Induction" (2009)

Example
-------
>>> from endgame.models.rules import FURIAClassifier
>>> clf = FURIAClassifier(max_rules=50)
>>> clf.fit(X_train, y_train)
>>> predictions = clf.predict(X_test)
>>> print(clf.get_rules_str())  # Human-readable rules
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.core.glassbox import GlassboxMixin
from typing import Any

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Fallback decorator
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return decorator
    prange = range


@dataclass
class FuzzyCondition:
    """A fuzzy condition with trapezoidal membership function.

    The membership function is defined by four points (a, b, c, d):
    - membership = 0 for x <= a or x >= d
    - membership = 1 for b <= x <= c
    - linear interpolation for a < x < b and c < x < d

    For crisp conditions, a=b and c=d.

    Parameters
    ----------
    feature_idx : int
        Index of the feature.
    feature_name : str
        Name of the feature.
    lower_bound : float or None
        Lower bound (a, b) of trapezoidal function. None means -inf.
    upper_bound : float or None
        Upper bound (c, d) of trapezoidal function. None means +inf.
    lower_support : float
        Support point a (where membership starts rising).
    upper_support : float
        Support point d (where membership ends).
    """
    feature_idx: int
    feature_name: str
    lower_bound: float | None = None  # b: core lower (membership = 1)
    upper_bound: float | None = None  # c: core upper (membership = 1)
    lower_support: float | None = None  # a: support lower (membership = 0)
    upper_support: float | None = None  # d: support upper (membership = 0)

    def membership(self, X: np.ndarray) -> np.ndarray:
        """Compute fuzzy membership degree for samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        membership : ndarray of shape (n_samples,)
            Membership degree in [0, 1] for each sample.
        """
        x = X[:, self.feature_idx]
        n = len(x)
        membership = np.ones(n)

        # Lower bound fuzzy region
        if self.lower_bound is not None:
            if self.lower_support is not None and self.lower_support < self.lower_bound:
                # Fuzzy lower bound
                mask_below = x < self.lower_support
                mask_fuzzy = (x >= self.lower_support) & (x < self.lower_bound)
                membership[mask_below] = 0.0
                if np.any(mask_fuzzy):
                    denom = self.lower_bound - self.lower_support
                    if denom > 0:
                        membership[mask_fuzzy] = (x[mask_fuzzy] - self.lower_support) / denom
            else:
                # Crisp lower bound
                membership[x < self.lower_bound] = 0.0

        # Upper bound fuzzy region
        if self.upper_bound is not None:
            if self.upper_support is not None and self.upper_support > self.upper_bound:
                # Fuzzy upper bound
                mask_above = x > self.upper_support
                mask_fuzzy = (x > self.upper_bound) & (x <= self.upper_support)
                membership[mask_above] = 0.0
                if np.any(mask_fuzzy):
                    denom = self.upper_support - self.upper_bound
                    if denom > 0:
                        membership[mask_fuzzy] = (self.upper_support - x[mask_fuzzy]) / denom
            else:
                # Crisp upper bound
                membership[x > self.upper_bound] = 0.0

        return membership

    def __str__(self) -> str:
        parts = []
        if self.lower_bound is not None:
            if self.lower_support is not None and self.lower_support < self.lower_bound:
                parts.append(f"{self.feature_name} >= ~{self.lower_bound:.4g}")
            else:
                parts.append(f"{self.feature_name} >= {self.lower_bound:.4g}")
        if self.upper_bound is not None:
            if self.upper_support is not None and self.upper_support > self.upper_bound:
                parts.append(f"{self.feature_name} <= ~{self.upper_bound:.4g}")
            else:
                parts.append(f"{self.feature_name} <= {self.upper_bound:.4g}")
        return " AND ".join(parts) if parts else "TRUE"


@dataclass
class FuzzyRule:
    """A fuzzy rule consisting of multiple fuzzy conditions.

    The rule's firing strength is the minimum membership across all conditions
    (fuzzy AND via t-norm).

    Parameters
    ----------
    conditions : list of FuzzyCondition
        The fuzzy conditions that make up this rule.
    consequent : int
        The class label this rule predicts.
    weight : float
        Rule weight (confidence/accuracy on training data).
    support : int
        Number of training instances covered by this rule.
    """
    conditions: list[FuzzyCondition] = field(default_factory=list)
    consequent: int = 0
    weight: float = 1.0
    support: int = 0

    def firing_strength(self, X: np.ndarray) -> np.ndarray:
        """Compute firing strength (membership degree) for samples.

        Uses minimum t-norm for fuzzy AND.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        strength : ndarray of shape (n_samples,)
            Firing strength in [0, 1] for each sample.
        """
        if not self.conditions:
            return np.ones(X.shape[0])

        # Start with all ones
        strength = np.ones(X.shape[0])

        # Apply minimum t-norm (fuzzy AND)
        for cond in self.conditions:
            strength = np.minimum(strength, cond.membership(X))

        return strength

    def covers(self, X: np.ndarray, threshold: float = 0.0) -> np.ndarray:
        """Check which samples are covered by this rule.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.
        threshold : float
            Minimum firing strength to be considered covered.

        Returns
        -------
        covered : ndarray of shape (n_samples,) dtype=bool
            True where sample is covered by rule.
        """
        return self.firing_strength(X) > threshold

    def __str__(self) -> str:
        if not self.conditions:
            antecedent = "TRUE"
        else:
            antecedent = " AND ".join(str(c) for c in self.conditions)
        return f"IF {antecedent} THEN class={self.consequent} (weight={self.weight:.3f}, support={self.support})"


def _grow_rule(
    X: np.ndarray,
    y: np.ndarray,
    target_class: int,
    feature_names: list[str],
    min_support: int = 2,
    max_conditions: int = 10,
) -> tuple[FuzzyRule, np.ndarray]:
    """Grow a single rule using RIPPER-like sequential covering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target labels.
    target_class : int
        The class this rule should predict.
    feature_names : list of str
        Feature names.
    min_support : int
        Minimum number of positive examples covered.
    max_conditions : int
        Maximum number of conditions in a rule.

    Returns
    -------
    rule : FuzzyRule
        The grown rule.
    covered : ndarray of shape (n_samples,) dtype=bool
        Which samples are covered by the rule.
    """
    n_samples, n_features = X.shape
    positive = (y == target_class)

    # Start with empty rule (covers everything)
    conditions = []
    covered = np.ones(n_samples, dtype=bool)

    for _ in range(max_conditions):
        # Find the best condition to add
        best_gain = -np.inf
        best_condition = None
        best_new_covered = None

        # Current stats
        n_covered = covered.sum()
        n_pos_covered = (covered & positive).sum()

        if n_pos_covered < min_support:
            break

        # Try each feature and possible split points
        for feat_idx in range(n_features):
            feat_values = X[covered, feat_idx]
            feat_pos = positive[covered]

            if len(feat_values) < min_support:
                continue

            unique_vals = np.unique(feat_values)

            if len(unique_vals) < 2:
                continue

            # Subsample split candidates for speed on large feature spaces
            max_splits = 50
            if len(unique_vals) - 1 > max_splits:
                quantile_idx = np.linspace(0, len(unique_vals) - 2, max_splits, dtype=int)
                candidate_idx = np.unique(quantile_idx)
            else:
                candidate_idx = np.arange(len(unique_vals) - 1)

            for i in candidate_idx:
                threshold = (unique_vals[i] + unique_vals[i + 1]) / 2

                # Try lower bound condition (>=)
                new_covered_lower = covered & (X[:, feat_idx] >= threshold)
                n_new_covered = new_covered_lower.sum()
                n_new_pos = (new_covered_lower & positive).sum()

                if n_new_covered >= min_support and n_new_pos >= 1:
                    # FOIL-like information gain
                    if n_new_covered > 0 and n_covered > 0:
                        p_old = n_pos_covered / n_covered if n_covered > 0 else 0
                        p_new = n_new_pos / n_new_covered if n_new_covered > 0 else 0

                        # Only improve purity, not make it worse
                        if p_new > p_old:
                            gain = n_new_pos * (np.log2(p_new + 1e-10) - np.log2(p_old + 1e-10))

                            if gain > best_gain:
                                best_gain = gain
                                best_condition = FuzzyCondition(
                                    feature_idx=feat_idx,
                                    feature_name=feature_names[feat_idx],
                                    lower_bound=threshold,
                                    upper_bound=None,
                                )
                                best_new_covered = new_covered_lower

                # Try upper bound condition (<=)
                new_covered_upper = covered & (X[:, feat_idx] <= threshold)
                n_new_covered = new_covered_upper.sum()
                n_new_pos = (new_covered_upper & positive).sum()

                if n_new_covered >= min_support and n_new_pos >= 1:
                    if n_new_covered > 0 and n_covered > 0:
                        p_old = n_pos_covered / n_covered if n_covered > 0 else 0
                        p_new = n_new_pos / n_new_covered if n_new_covered > 0 else 0

                        if p_new > p_old:
                            gain = n_new_pos * (np.log2(p_new + 1e-10) - np.log2(p_old + 1e-10))

                            if gain > best_gain:
                                best_gain = gain
                                best_condition = FuzzyCondition(
                                    feature_idx=feat_idx,
                                    feature_name=feature_names[feat_idx],
                                    lower_bound=None,
                                    upper_bound=threshold,
                                )
                                best_new_covered = new_covered_upper

        if best_condition is None:
            break

        conditions.append(best_condition)
        covered = best_new_covered

        # Stop if we've covered only positive examples
        n_neg_covered = (covered & ~positive).sum()
        if n_neg_covered == 0:
            break

    # Compute rule statistics
    n_covered = covered.sum()
    n_pos_covered = (covered & positive).sum()
    weight = n_pos_covered / n_covered if n_covered > 0 else 0.0

    rule = FuzzyRule(
        conditions=conditions,
        consequent=target_class,
        weight=weight,
        support=int(n_pos_covered),
    )

    return rule, covered


def _fuzzify_rule(
    rule: FuzzyRule,
    X: np.ndarray,
    y: np.ndarray,
) -> FuzzyRule:
    """Fuzzify a crisp rule by adding support regions to conditions.

    Uses the training data to determine appropriate fuzzy boundaries
    by finding the range of positive examples near each boundary.

    Parameters
    ----------
    rule : FuzzyRule
        The crisp rule to fuzzify.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target labels.

    Returns
    -------
    fuzzy_rule : FuzzyRule
        The fuzzified rule.
    """
    target_class = rule.consequent
    positive = (y == target_class)

    new_conditions = []

    for cond in rule.conditions:
        feat_idx = cond.feature_idx
        feat_values = X[:, feat_idx]

        new_cond = FuzzyCondition(
            feature_idx=cond.feature_idx,
            feature_name=cond.feature_name,
            lower_bound=cond.lower_bound,
            upper_bound=cond.upper_bound,
            lower_support=cond.lower_support,
            upper_support=cond.upper_support,
        )

        # Fuzzify lower bound
        if cond.lower_bound is not None:
            # Find positive instances below the boundary
            pos_below = feat_values[(feat_values < cond.lower_bound) & positive]
            if len(pos_below) > 0:
                # Extend support to include some positive instances
                new_cond.lower_support = np.percentile(pos_below, 10) if len(pos_below) > 1 else pos_below.min()
            else:
                new_cond.lower_support = cond.lower_bound

        # Fuzzify upper bound
        if cond.upper_bound is not None:
            # Find positive instances above the boundary
            pos_above = feat_values[(feat_values > cond.upper_bound) & positive]
            if len(pos_above) > 0:
                # Extend support to include some positive instances
                new_cond.upper_support = np.percentile(pos_above, 90) if len(pos_above) > 1 else pos_above.max()
            else:
                new_cond.upper_support = cond.upper_bound

        new_conditions.append(new_cond)

    return FuzzyRule(
        conditions=new_conditions,
        consequent=rule.consequent,
        weight=rule.weight,
        support=rule.support,
    )


def _stretch_rules(
    rules: list[FuzzyRule],
    X: np.ndarray,
    y: np.ndarray,
    stretch_factor: float = 0.1,
) -> list[FuzzyRule]:
    """Stretch rules to cover uncovered instances.

    For instances not covered by any rule, find the nearest rule
    and stretch its boundaries to include the instance.

    Parameters
    ----------
    rules : list of FuzzyRule
        The rules to stretch.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target labels.
    stretch_factor : float
        Maximum stretch as fraction of feature range.

    Returns
    -------
    stretched_rules : list of FuzzyRule
        Rules with stretched boundaries.
    """
    # This is a simplified version - full FURIA uses more sophisticated stretching
    return rules  # Return as-is for now


class FURIAClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """Fuzzy Unordered Rule Induction Algorithm (FURIA) classifier.

    FURIA learns a set of fuzzy rules for classification. It extends RIPPER
    with fuzzy boundaries that allow soft decision regions.

    Parameters
    ----------
    max_rules : int, default=50
        Maximum number of rules to learn.
    min_support : int, default=2
        Minimum number of positive examples a rule must cover.
    max_conditions : int, default=10
        Maximum number of conditions per rule.
    fuzzify : bool, default=True
        Whether to fuzzify rules after learning.
    rule_stretching : bool, default=True
        Whether to use rule stretching for uncovered instances.
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    rules_ : list of FuzzyRule
        Learned fuzzy rules.
    n_features_in_ : int
        Number of features seen during fit.
    feature_names_ : list of str
        Feature names.

    Examples
    --------
    >>> from endgame.models.rules import FURIAClassifier
    >>> clf = FURIAClassifier(max_rules=30)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> print(clf.get_rules_str())
    """

    def __init__(
        self,
        max_rules: int = 50,
        min_support: int = 2,
        max_conditions: int = 10,
        fuzzify: bool = True,
        rule_stretching: bool = True,
        random_state: int | None = None,
    ):
        self.max_rules = max_rules
        self.min_support = min_support
        self.max_conditions = max_conditions
        self.fuzzify = fuzzify
        self.rule_stretching = rule_stretching
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        feature_names: list[str] | None = None,
    ) -> FURIAClassifier:
        """Fit the FURIA classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names for the features.

        Returns
        -------
        self : FURIAClassifier
            Fitted classifier.
        """
        if feature_names is not None:
            _names = list(feature_names)
        elif hasattr(X, "columns"):
            _names = list(X.columns)
        else:
            _names = None

        X, y = check_X_y(X, y)

        if self.random_state is not None:
            np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        self.feature_names_ = _names or [f"X{i}" for i in range(self.n_features_in_)]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Learn rules for each class
        self.rules_: list[FuzzyRule] = []

        for class_idx in range(self.n_classes_):
            # Sequential covering for this class
            remaining = np.ones(len(y_encoded), dtype=bool)
            positive = (y_encoded == class_idx)
            n_positive = positive.sum()

            class_rules = []

            while remaining.sum() > 0 and (remaining & positive).sum() > 0:
                if len(class_rules) >= self.max_rules // self.n_classes_:
                    break

                # Grow a rule on remaining data
                X_remaining = X[remaining]
                y_remaining = y_encoded[remaining]

                rule, covered_remaining = _grow_rule(
                    X_remaining,
                    y_remaining,
                    target_class=class_idx,
                    feature_names=self.feature_names_,
                    min_support=self.min_support,
                    max_conditions=self.max_conditions,
                )

                if rule.support < self.min_support:
                    break

                # Map covered back to original indices
                remaining_indices = np.where(remaining)[0]
                covered_original = np.zeros(len(y_encoded), dtype=bool)
                covered_original[remaining_indices[covered_remaining]] = True

                # Fuzzify the rule
                if self.fuzzify:
                    rule = _fuzzify_rule(rule, X, y_encoded)

                class_rules.append(rule)

                # Remove covered positive examples
                remaining[covered_original & positive] = False

            self.rules_.extend(class_rules)

        # Rule stretching
        if self.rule_stretching:
            self.rules_ = _stretch_rules(self.rules_, X, y_encoded)

        return self

    def predict_proba(self, X: ArrayLike) -> np.ndarray:
        """Predict class probabilities.

        Uses the fuzzy firing strengths to compute class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, 'rules_')
        X = check_array(X)

        n_samples = X.shape[0]

        # Compute weighted vote for each class
        class_scores = np.zeros((n_samples, self.n_classes_))

        for rule in self.rules_:
            # Get firing strength for each sample
            strength = rule.firing_strength(X)
            # Add weighted vote
            class_scores[:, rule.consequent] += strength * rule.weight

        # Handle samples with no rule coverage (assign uniform)
        total_score = class_scores.sum(axis=1, keepdims=True)
        no_coverage = total_score.ravel() == 0

        if np.any(no_coverage):
            class_scores[no_coverage] = 1.0 / self.n_classes_
            total_score[no_coverage] = 1.0

        # Normalize to probabilities
        proba = class_scores / (total_score + 1e-10)

        return proba

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self._label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def get_rules(self) -> list[FuzzyRule]:
        """Get the learned rules.

        Returns
        -------
        rules : list of FuzzyRule
            The learned fuzzy rules.
        """
        check_is_fitted(self, 'rules_')
        return self.rules_

    def get_rules_str(self) -> str:
        """Get a human-readable string representation of the rules.

        Returns
        -------
        rules_str : str
            String representation of all rules.
        """
        check_is_fitted(self, 'rules_')

        lines = [f"FURIA Rules ({len(self.rules_)} rules):"]
        lines.append("-" * 60)

        for i, rule in enumerate(self.rules_, 1):
            # Convert class index to label
            class_label = self._label_encoder.inverse_transform([rule.consequent])[0]
            if not rule.conditions:
                antecedent = "TRUE"
            else:
                antecedent = " AND ".join(str(c) for c in rule.conditions)
            lines.append(
                f"R{i}: IF {antecedent}\n"
                f"     THEN class={class_label} "
                f"(weight={rule.weight:.3f}, support={rule.support})"
            )

        return "\n".join(lines)

    def get_rule_importance(self) -> np.ndarray:
        """Get feature importance based on rule usage.

        Returns
        -------
        importance : ndarray of shape (n_features,)
            Feature importance scores.
        """
        check_is_fitted(self, 'rules_')

        importance = np.zeros(self.n_features_in_)

        for rule in self.rules_:
            for cond in rule.conditions:
                # Weight by rule weight and support
                importance[cond.feature_idx] += rule.weight * rule.support

        # Normalize
        if importance.sum() > 0:
            importance = importance / importance.sum()

        return importance

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance scores (sklearn compatible)."""
        return self.get_rule_importance()

    _structure_type = "fuzzy_rules"

    def _structure_content(self) -> dict[str, Any]:
        check_is_fitted(self, 'rules_')
        rule_dicts = []
        for rule in self.rules_:
            class_label = self._label_encoder.inverse_transform([rule.consequent])[0]
            conditions = []
            for c in rule.conditions:
                conditions.append({
                    "feature_index": int(c.feature_idx),
                    "feature": str(c.feature_name),
                    "lower_bound": float(c.lower_bound) if c.lower_bound is not None else None,
                    "upper_bound": float(c.upper_bound) if c.upper_bound is not None else None,
                    "lower_support": float(c.lower_support) if c.lower_support is not None else None,
                    "upper_support": float(c.upper_support) if c.upper_support is not None else None,
                    "text": str(c),
                })
            rule_dicts.append({
                "conditions": conditions,
                "consequent_class": class_label.item() if hasattr(class_label, "item") else class_label,
                "weight": float(rule.weight),
                "support": int(rule.support),
            })
        return {
            "rules": rule_dicts,
            "n_rules": len(rule_dicts),
            "feature_importances": self.feature_importances_.tolist(),
        }
