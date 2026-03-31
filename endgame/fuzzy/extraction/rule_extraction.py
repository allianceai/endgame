"""Fuzzy rule extraction from black-box models.

Extracts interpretable fuzzy IF-THEN rules that approximate
the behavior of any trained model, enabling model interpretability
and knowledge distillation into a transparent fuzzy system.

References
----------
Setiono, R., & Liu, H. (1997). NeuroLinear: From neural networks to
oblique decision rules. Neurocomputing, 17(1), 1-24.

Example
-------
>>> from endgame.fuzzy.extraction.rule_extraction import FuzzyRuleExtractor
>>> extractor = FuzzyRuleExtractor(n_rules=10)
>>> extractor.fit(black_box_model, X_train, y_train)
>>> print(extractor.get_rules_str())
>>> fidelity = extractor.score_fidelity(black_box_model, X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class FuzzyRuleExtractor(BaseEstimator, TransformerMixin):
    """Extract interpretable fuzzy rules from black-box models.

    Creates a transparent fuzzy rule base that approximates a
    black-box model's behavior. Rules use Gaussian membership
    functions and are constructed by partitioning the input space
    and identifying dominant predictions per partition.

    Parameters
    ----------
    n_rules : int, default=20
        Maximum number of fuzzy rules to extract.
    n_mfs : int, default=3
        Number of membership functions per input feature.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian' or 'triangular'.
    merge_threshold : float, default=0.9
        Similarity threshold for merging similar rules.
    verbose : bool, default=False
        Print extraction progress.

    Attributes
    ----------
    rules_ : list of dict
        Extracted rules with antecedent and consequent info.
    n_features_in_ : int
        Number of features seen during fit.
    feature_ranges_ : ndarray of shape (n_features, 2)
        Min and max per feature.
    mf_params_ : list of list of dict
        MF parameters per feature per term.
    is_classifier_ : bool
        Whether the original model was a classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from endgame.fuzzy.extraction.rule_extraction import FuzzyRuleExtractor
    >>> X = np.random.randn(200, 5)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> rf = RandomForestClassifier(n_estimators=10, random_state=42)
    >>> rf.fit(X, y)
    RandomForestClassifier(n_estimators=10, random_state=42)
    >>> extractor = FuzzyRuleExtractor(n_rules=10, n_mfs=3)
    >>> extractor.fit(rf, X, y)
    FuzzyRuleExtractor(n_rules=10)
    >>> print(extractor.get_rules_str()[:100])
    Rule 1: ...
    >>> fidelity = extractor.score_fidelity(rf, X)
    """

    def __init__(
        self,
        n_rules: int = 20,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        merge_threshold: float = 0.9,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.n_mfs = n_mfs
        self.mf_type = mf_type
        self.merge_threshold = merge_threshold
        self.verbose = verbose

    def fit(
        self,
        model: Any,
        X: Any,
        y: Any = None,
    ) -> FuzzyRuleExtractor:
        """Extract fuzzy rules from a trained model.

        Parameters
        ----------
        model : estimator
            Trained model with a ``predict`` method.
        X : array-like of shape (n_samples, n_features)
            Training data used for extraction.
        y : array-like of shape (n_samples,) or None
            Original labels (used for fidelity tracking, not required).

        Returns
        -------
        self
            Fitted extractor with extracted rules.
        """
        X = check_array(X)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

        # Determine if model is a classifier
        self.is_classifier_ = hasattr(model, "predict_proba")

        # Get model predictions
        predictions = model.predict(X)

        # Compute feature ranges
        self.feature_ranges_ = np.column_stack([X.min(axis=0), X.max(axis=0)])

        # Create fuzzy partitions
        self.mf_params_ = self._create_partitions(X)

        # Compute memberships for all samples
        memberships = self._compute_memberships(X)  # (n_samples, n_features, n_mfs)

        # Generate candidate rules by finding dominant partitions
        rules = self._generate_rules(X, memberships, predictions)

        # Merge similar rules
        rules = self._merge_rules(rules)

        # Limit number of rules
        if len(rules) > self.n_rules:
            # Keep rules with highest support
            supports = [r["support"] for r in rules]
            top_idx = np.argsort(supports)[-self.n_rules:]
            rules = [rules[i] for i in top_idx]

        self.rules_ = rules

        if self.verbose:
            print(f"[RuleExtractor] Extracted {len(self.rules_)} rules")

        return self

    def transform(self, X: Any) -> np.ndarray:
        """Apply extracted rules as a simplified prediction model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predictions from the extracted fuzzy rule system.
        """
        check_is_fitted(self, ["rules_"])
        X = check_array(X)

        memberships = self._compute_memberships(X)
        n_samples = X.shape[0]

        if self.is_classifier_:
            # Collect unique classes
            classes = sorted(set(r["consequent"] for r in self.rules_))
            class_map = {c: i for i, c in enumerate(classes)}
            scores = np.zeros((n_samples, len(classes)))

            for rule in self.rules_:
                firing = self._compute_rule_firing(memberships, rule)
                c_idx = class_map[rule["consequent"]]
                scores[:, c_idx] += firing * rule["confidence"]

            return np.array([classes[i] for i in np.argmax(scores, axis=1)])
        else:
            # Weighted average for regression
            weighted_sum = np.zeros(n_samples)
            weight_total = np.zeros(n_samples)

            for rule in self.rules_:
                firing = self._compute_rule_firing(memberships, rule)
                weighted_sum += firing * rule["consequent"] * rule["confidence"]
                weight_total += firing * rule["confidence"]

            weight_total = np.where(weight_total == 0, 1.0, weight_total)
            return weighted_sum / weight_total

    def get_rules_str(self) -> str:
        """Get human-readable string of extracted rules.

        Returns
        -------
        str
            Multi-line string with one rule per line.
        """
        check_is_fitted(self, ["rules_"])
        term_names = ["low", "medium", "high", "very_low", "low_medium",
                      "medium_high", "very_high", "extra1", "extra2", "extra3"]

        lines = []
        for i, rule in enumerate(self.rules_):
            antecedents = []
            for j, mf_idx in enumerate(rule["antecedent"]):
                if mf_idx < 0:
                    continue  # Don't-care condition
                term = term_names[mf_idx] if mf_idx < len(term_names) else f"term_{mf_idx}"
                antecedents.append(f"x{j} IS {term}")

            ant_str = " AND ".join(antecedents) if antecedents else "TRUE"
            conf = rule["confidence"]
            cons = rule["consequent"]

            if self.is_classifier_:
                cons_str = f"class={cons}"
            else:
                cons_str = f"{cons:.4f}"

            lines.append(
                f"Rule {i+1}: IF {ant_str} THEN {cons_str} "
                f"[confidence={conf:.3f}, support={rule['support']}]"
            )

        return "\n".join(lines)

    def score_fidelity(self, model: Any, X: Any) -> float:
        """Measure how well extracted rules match the original model.

        Parameters
        ----------
        model : estimator
            Original trained model.
        X : array-like of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        float
            Fidelity score (accuracy for classifiers, 1 - NRMSE for regressors).
        """
        check_is_fitted(self, ["rules_"])
        X = check_array(X)

        model_pred = model.predict(X)
        rule_pred = self.transform(X)

        if self.is_classifier_:
            return float(np.mean(model_pred == rule_pred))
        else:
            mse = np.mean((model_pred - rule_pred) ** 2)
            var = np.var(model_pred)
            if var == 0:
                return 1.0 if mse == 0 else 0.0
            return float(1.0 - np.sqrt(mse / var))

    def _create_partitions(self, X: np.ndarray) -> list[list[dict]]:
        """Create fuzzy partitions for each feature.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        list of list of dict
            MF parameters per feature per term.
        """
        n_features = X.shape[1]
        all_params = []

        for j in range(n_features):
            x_min = float(X[:, j].min())
            x_max = float(X[:, j].max())
            span = x_max - x_min
            if span < 1e-10:
                span = 1.0

            centers = np.linspace(x_min, x_max, self.n_mfs)
            sigma = span / max(self.n_mfs - 1, 1) / 2.0

            feature_params = []
            for c in centers:
                feature_params.append({
                    "center": float(c),
                    "sigma": max(sigma, 1e-6),
                })
            all_params.append(feature_params)

        return all_params

    def _compute_memberships(self, X: np.ndarray) -> np.ndarray:
        """Compute membership degrees for all samples and partitions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_features, n_mfs)
        """
        n_samples, n_features = X.shape
        memberships = np.zeros((n_samples, n_features, self.n_mfs))

        for j in range(n_features):
            for k, params in enumerate(self.mf_params_[j]):
                diff = X[:, j] - params["center"]
                memberships[:, j, k] = np.exp(
                    -0.5 * (diff / params["sigma"]) ** 2
                )

        return memberships

    def _compute_rule_firing(
        self,
        memberships: np.ndarray,
        rule: dict,
    ) -> np.ndarray:
        """Compute firing strength for a single rule.

        Parameters
        ----------
        memberships : ndarray of shape (n_samples, n_features, n_mfs)
        rule : dict
            Rule with 'antecedent' key.

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        n_samples = memberships.shape[0]
        firing = np.ones(n_samples)

        for j, mf_idx in enumerate(rule["antecedent"]):
            if mf_idx < 0:
                continue
            firing *= memberships[:, j, mf_idx]

        return firing

    def _generate_rules(
        self,
        X: np.ndarray,
        memberships: np.ndarray,
        predictions: np.ndarray,
    ) -> list[dict]:
        """Generate candidate rules from data partitions.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        memberships : ndarray of shape (n_samples, n_features, n_mfs)
        predictions : ndarray of shape (n_samples,)

        Returns
        -------
        list of dict
            Candidate rules.
        """
        n_samples, n_features = X.shape

        # Assign each sample to its dominant partition per feature
        dominant = np.argmax(memberships, axis=2)  # (n_samples, n_features)

        # Find unique partition combinations
        seen = {}
        for i in range(n_samples):
            key = tuple(dominant[i])
            if key not in seen:
                seen[key] = []
            seen[key].append(i)

        rules = []
        for antecedent, sample_indices in seen.items():
            preds = predictions[sample_indices]

            if self.is_classifier_:
                # Most common prediction
                unique, counts = np.unique(preds, return_counts=True)
                best_idx = np.argmax(counts)
                consequent = unique[best_idx]
                confidence = float(counts[best_idx] / len(preds))
            else:
                consequent = float(np.mean(preds))
                # Confidence as inverse of relative std
                std = np.std(preds)
                mean_abs = np.abs(np.mean(preds))
                if mean_abs < 1e-10:
                    confidence = 1.0
                else:
                    confidence = float(1.0 / (1.0 + std / mean_abs))

            rules.append({
                "antecedent": list(antecedent),
                "consequent": consequent,
                "confidence": confidence,
                "support": len(sample_indices),
            })

        return rules

    def _merge_rules(self, rules: list[dict]) -> list[dict]:
        """Merge similar rules to reduce complexity.

        Two rules are merged if they have the same consequent and
        their antecedents differ by at most one term.

        Parameters
        ----------
        rules : list of dict
            Candidate rules.

        Returns
        -------
        list of dict
            Merged rules.
        """
        if len(rules) <= 1:
            return rules

        merged = True
        while merged:
            merged = False
            new_rules = []
            used = set()

            for i in range(len(rules)):
                if i in used:
                    continue

                best_j = -1
                best_sim = -1

                for j in range(i + 1, len(rules)):
                    if j in used:
                        continue

                    # Check same consequent
                    if self.is_classifier_:
                        same_cons = rules[i]["consequent"] == rules[j]["consequent"]
                    else:
                        diff = abs(rules[i]["consequent"] - rules[j]["consequent"])
                        range_val = abs(rules[i]["consequent"]) + 1e-10
                        same_cons = diff / range_val < 0.1

                    if not same_cons:
                        continue

                    # Count differing antecedent terms
                    ant_i = rules[i]["antecedent"]
                    ant_j = rules[j]["antecedent"]
                    n_diff = sum(1 for a, b in zip(ant_i, ant_j) if a != b)

                    if n_diff <= 1:
                        sim = 1.0 - n_diff / max(len(ant_i), 1)
                        if sim > best_sim and sim >= self.merge_threshold:
                            best_sim = sim
                            best_j = j

                if best_j >= 0:
                    # Merge rules i and best_j
                    ant_merged = []
                    for a, b in zip(
                        rules[i]["antecedent"], rules[best_j]["antecedent"]
                    ):
                        if a == b:
                            ant_merged.append(a)
                        else:
                            ant_merged.append(-1)  # Don't-care

                    total_support = rules[i]["support"] + rules[best_j]["support"]

                    if self.is_classifier_:
                        consequent = rules[i]["consequent"]
                    else:
                        w_i = rules[i]["support"]
                        w_j = rules[best_j]["support"]
                        consequent = (
                            rules[i]["consequent"] * w_i
                            + rules[best_j]["consequent"] * w_j
                        ) / total_support

                    confidence = (
                        rules[i]["confidence"] * rules[i]["support"]
                        + rules[best_j]["confidence"] * rules[best_j]["support"]
                    ) / total_support

                    new_rules.append({
                        "antecedent": ant_merged,
                        "consequent": consequent,
                        "confidence": confidence,
                        "support": total_support,
                    })
                    used.add(i)
                    used.add(best_j)
                    merged = True
                else:
                    new_rules.append(rules[i])
                    used.add(i)

            rules = new_rules

        return rules
