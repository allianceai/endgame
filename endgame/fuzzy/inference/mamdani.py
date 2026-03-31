"""Mamdani Fuzzy Inference System.

Implements the classic Mamdani FIS with fuzzy sets for both antecedents
and consequents, using implication (clipping) and aggregation (max)
followed by defuzzification (centroid by default).

Example
-------
>>> from endgame.fuzzy.inference.mamdani import MamdaniFIS
>>> fis = MamdaniFIS(n_mfs=3, defuzz_method='centroid')
>>> fis.fit(X_train, y_train)
>>> preds = fis.predict(X_test)
"""

from __future__ import annotations

from itertools import product as iterproduct
from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyClassifier, BaseFuzzyRegressor
from endgame.fuzzy.core.defuzzification import centroid, defuzzify
from endgame.fuzzy.core.membership import BaseMembershipFunction, create_uniform_mfs


class MamdaniFIS(BaseFuzzyRegressor):
    """Mamdani Fuzzy Inference System for regression.

    Uses linguistic fuzzy rules with fuzzy sets for both antecedents
    and consequents. Rules are auto-generated from training data by
    grid partitioning the input and output space.

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination: 'min', 'product', 'lukasiewicz'.
    defuzz_method : str, default='centroid'
        Defuzzification method: 'centroid', 'bisector', 'mean_of_maxima'.
    n_output_mfs : int or None, default=None
        Number of membership functions for the output variable. Defaults
        to ``n_mfs`` if not specified.
    n_output_points : int, default=200
        Number of discretization points for the output universe of discourse.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    mfs_ : list of list of BaseMembershipFunction
        Antecedent membership functions, indexed [feature][mf_index].
    output_mfs_ : list of BaseMembershipFunction
        Consequent (output) membership functions.
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        For each rule and feature, the index of the antecedent MF.
    consequent_indices_ : ndarray of shape (n_rules,)
        For each rule, the index of the consequent output MF.
    rule_weights_ : ndarray of shape (n_rules,)
        Weight (degree) of each rule.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.mamdani import MamdaniFIS
    >>> X = np.random.rand(100, 2)
    >>> y = X[:, 0] + 2 * X[:, 1] + np.random.randn(100) * 0.1
    >>> fis = MamdaniFIS(n_mfs=3, defuzz_method='centroid')
    >>> fis.fit(X, y)
    MamdaniFIS(...)
    >>> preds = fis.predict(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        defuzz_method: str = "centroid",
        n_output_mfs: int | None = None,
        n_output_points: int = 200,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,  # determined by grid
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.defuzz_method = defuzz_method
        self.n_output_mfs = n_output_mfs
        self.n_output_points = n_output_points

    def fit(self, X: Any, y: Any) -> MamdaniFIS:
        """Fit the Mamdani FIS by generating rules from training data.

        For each data point, determines which antecedent MF has the
        highest membership for each input feature and which output MF
        has the highest membership for the target value. Conflicting
        rules (same antecedent, different consequent) are resolved by
        keeping the rule with the highest firing degree.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        # Store feature names if available
        self.feature_names_ = [f"x{j}" for j in range(n_features)]

        # Initialize antecedent MFs
        self.mfs_ = self._init_membership_functions(X)

        # Initialize output MFs
        n_out_mfs = self.n_output_mfs if self.n_output_mfs is not None else self.n_mfs
        y_min, y_max = float(np.min(y)), float(np.max(y))
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        self.output_mfs_ = create_uniform_mfs(
            n_mfs=n_out_mfs,
            x_min=y_min - padding,
            x_max=y_max + padding,
            mf_type=self.mf_type,
        )
        self.y_min_ = y_min - padding
        self.y_max_ = y_max + padding

        # For each sample, find the best antecedent MF per feature
        # and the best output MF
        sample_antecedents = np.zeros((n_samples, n_features), dtype=int)
        sample_consequents = np.zeros(n_samples, dtype=int)
        sample_degrees = np.zeros(n_samples, dtype=np.float64)

        for i in range(n_samples):
            degree = 1.0
            for j in range(n_features):
                memberships = np.array(
                    [self.mfs_[j][k](np.array([X[i, j]]))[0] for k in range(self.n_mfs)]
                )
                best_k = int(np.argmax(memberships))
                sample_antecedents[i, j] = best_k
                degree *= memberships[best_k]

            # Best output MF
            out_memberships = np.array(
                [mf(np.array([y[i]]))[0] for mf in self.output_mfs_]
            )
            best_out = int(np.argmax(out_memberships))
            sample_consequents[i] = best_out
            sample_degrees[i] = degree * out_memberships[best_out]

        # Build rule base: resolve conflicts by keeping highest degree
        rule_dict: dict[tuple[int, ...], tuple[int, float]] = {}
        for i in range(n_samples):
            ant_key = tuple(sample_antecedents[i].tolist())
            deg = sample_degrees[i]
            if ant_key not in rule_dict or deg > rule_dict[ant_key][1]:
                rule_dict[ant_key] = (sample_consequents[i], deg)

        # Convert to arrays
        n_rules = len(rule_dict)
        if n_rules == 0:
            # Fallback: single default rule
            rule_dict[tuple([0] * n_features)] = (0, 1.0)
            n_rules = 1

        self.antecedent_indices_ = np.zeros((n_rules, n_features), dtype=int)
        self.consequent_indices_ = np.zeros(n_rules, dtype=int)
        self.rule_weights_ = np.zeros(n_rules, dtype=np.float64)

        for idx, (ant_key, (cons, deg)) in enumerate(rule_dict.items()):
            self.antecedent_indices_[idx] = list(ant_key)
            self.consequent_indices_[idx] = cons
            self.rule_weights_[idx] = deg

        self.n_rules = n_rules
        self._log(f"Generated {n_rules} rules from {n_samples} samples")
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using Mamdani inference.

        For each sample:
        1. Compute firing strength of each rule.
        2. Clip (implicate) each consequent MF by its firing strength.
        3. Aggregate all clipped MFs using max operator.
        4. Defuzzify the aggregated output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["mfs_", "antecedent_indices_", "consequent_indices_"])
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        # Compute firing strengths
        strengths = self._compute_firing_strengths(X, self.antecedent_indices_)

        # Apply rule weights
        strengths = strengths * self.rule_weights_[np.newaxis, :]

        # Discretize output universe
        y_universe = np.linspace(self.y_min_, self.y_max_, self.n_output_points)

        # Pre-compute output MF values on the universe
        out_mf_values = np.zeros((len(self.output_mfs_), self.n_output_points))
        for k, mf in enumerate(self.output_mfs_):
            out_mf_values[k] = mf(y_universe)

        predictions = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            # Aggregate: for each rule, clip its consequent MF, then take max
            aggregated = np.zeros(self.n_output_points)
            for r in range(self.antecedent_indices_.shape[0]):
                firing = strengths[i, r]
                if firing < 1e-12:
                    continue
                cons_idx = self.consequent_indices_[r]
                # Implication: clip the consequent MF at the firing strength
                clipped = np.minimum(out_mf_values[cons_idx], firing)
                # Aggregation: max
                aggregated = np.maximum(aggregated, clipped)

            # Defuzzify
            if self.defuzz_method in ("weighted_average", "height"):
                # For these methods, use centers and heights
                centers = np.array([
                    float(np.sum(y_universe * out_mf_values[k]) / max(np.sum(out_mf_values[k]), 1e-12))
                    for k in range(len(self.output_mfs_))
                ])
                heights = np.array([strengths[i, r] for r in range(self.antecedent_indices_.shape[0])])
                predictions[i] = defuzzify(centers, heights, method=self.defuzz_method)
            else:
                predictions[i] = defuzzify(y_universe, aggregated, method=self.defuzz_method)

        return predictions

    def get_rules_str(self) -> str:
        """Get human-readable representation of the Mamdani rules.

        Returns
        -------
        str
            Multi-line string with one rule per line.
        """
        check_is_fitted(self, ["antecedent_indices_", "consequent_indices_"])
        term_names = [
            "low", "medium", "high", "very_low", "low_med",
            "med_high", "very_high", "extra1", "extra2", "extra3",
        ]
        lines = []
        n_rules = self.antecedent_indices_.shape[0]
        n_features = self.antecedent_indices_.shape[1]

        for r in range(n_rules):
            antecedents = []
            for j in range(n_features):
                mf_idx = self.antecedent_indices_[r, j]
                term = term_names[mf_idx] if mf_idx < len(term_names) else f"term_{mf_idx}"
                fname = self.feature_names_[j] if hasattr(self, "feature_names_") else f"x{j}"
                antecedents.append(f"{fname} IS {term}")

            cons_idx = self.consequent_indices_[r]
            cons_term = term_names[cons_idx] if cons_idx < len(term_names) else f"term_{cons_idx}"
            ant_str = " AND ".join(antecedents)
            lines.append(
                f"Rule {r + 1}: IF {ant_str} THEN y IS {cons_term} "
                f"[w={self.rule_weights_[r]:.4f}]"
            )
        return "\n".join(lines)


class MamdaniClassifier(BaseFuzzyClassifier):
    """Mamdani Fuzzy Inference System for classification.

    Creates one Mamdani FIS per class (one-vs-rest strategy). The class
    with the highest defuzzified output is predicted.

    Parameters
    ----------
    n_mfs : int, default=3
        Number of membership functions per input variable.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination.
    defuzz_method : str, default='centroid'
        Defuzzification method for each per-class FIS.
    n_output_mfs : int or None, default=None
        Number of output MFs per per-class FIS.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    estimators_ : list of MamdaniFIS
        One Mamdani FIS per class.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.mamdani import MamdaniClassifier
    >>> X = np.random.rand(100, 2)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> clf = MamdaniClassifier(n_mfs=3)
    >>> clf.fit(X, y)
    MamdaniClassifier(...)
    >>> preds = clf.predict(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 3,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        defuzz_method: str = "centroid",
        n_output_mfs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.defuzz_method = defuzz_method
        self.n_output_mfs = n_output_mfs

    def fit(self, X: Any, y: Any) -> MamdaniClassifier:
        """Fit one Mamdani FIS per class using one-vs-rest encoding.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Class labels.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_features_in_ = X.shape[1]
        y_encoded = self._encode_labels(y)

        self.estimators_ = []
        for c in range(self.n_classes_):
            # Binary target: 1.0 for this class, 0.0 otherwise
            y_binary = (y_encoded == c).astype(np.float64)
            fis = MamdaniFIS(
                n_mfs=self.n_mfs,
                mf_type=self.mf_type,
                t_norm=self.t_norm,
                defuzz_method=self.defuzz_method,
                n_output_mfs=self.n_output_mfs,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            fis.fit(X, y_binary)
            self.estimators_.append(fis)

        # Store antecedent_indices_ from first estimator for base class compat
        self.antecedent_indices_ = self.estimators_[0].antecedent_indices_
        self.mfs_ = self.estimators_[0].mfs_
        self.feature_names_ = [f"x{j}" for j in range(self.n_features_in_)]
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Each per-class FIS produces a raw output value. These are
        normalized via softmax to produce probability estimates.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X, dtype=np.float64)

        raw_outputs = np.column_stack([
            fis.predict(X) for fis in self.estimators_
        ])

        # Softmax normalization
        exp_vals = np.exp(raw_outputs - np.max(raw_outputs, axis=1, keepdims=True))
        proba = exp_vals / np.sum(exp_vals, axis=1, keepdims=True)
        return proba
