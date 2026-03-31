"""Wang-Mendel automatic rule generation method.

Implements the Wang-Mendel (1992) algorithm for automatically generating
fuzzy rules from numerical data. Rules are extracted by determining the
fuzzy set with maximum membership for each variable per data point,
with conflict resolution keeping the highest-degree rule.

Example
-------
>>> from endgame.fuzzy.inference.wang_mendel import WangMendelRegressor
>>> model = WangMendelRegressor(n_mfs=5, mf_type='triangular')
>>> model.fit(X_train, y_train)
>>> preds = model.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyRegressor
from endgame.fuzzy.core.defuzzification import defuzzify, weighted_average
from endgame.fuzzy.core.membership import BaseMembershipFunction, create_uniform_mfs


class WangMendelRegressor(BaseFuzzyRegressor):
    """Wang-Mendel automatic rule generation for regression.

    Algorithm:
    1. Partition each input and output variable into fuzzy sets
       using uniformly spaced membership functions.
    2. For each data point, determine which fuzzy set has the
       maximum membership for each variable (inputs and output).
    3. Generate one candidate fuzzy rule per data point.
    4. Resolve conflicts: when multiple rules share the same
       antecedent but have different consequents, keep only the
       rule with the highest degree (product of memberships).
    5. Use the resulting rule base for inference via weighted average.

    Parameters
    ----------
    n_mfs : int, default=5
        Number of membership functions per variable (input and output).
    mf_type : str, default='triangular'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination.
    defuzz_method : str, default='weighted_average'
        Defuzzification method. 'weighted_average' uses the center of
        the consequent MF weighted by firing strength.
        Also supports 'centroid', 'bisector', 'mean_of_maxima'.
    n_output_points : int, default=200
        Number of discretization points for Mamdani-style defuzzification
        (used when defuzz_method is not 'weighted_average' or 'height').
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    mfs_ : list of list of BaseMembershipFunction
        Input membership functions, indexed [feature][mf_index].
    output_mfs_ : list of BaseMembershipFunction
        Output membership functions.
    rule_base_ : dict
        Mapping from antecedent tuple to (consequent_index, degree) tuple.
        Keys are tuples of ints (MF indices per input feature).
    antecedent_indices_ : ndarray of shape (n_rules, n_features)
        Antecedent MF indices for each rule.
    consequent_indices_ : ndarray of shape (n_rules,)
        Consequent MF index for each rule.
    rule_degrees_ : ndarray of shape (n_rules,)
        Degree (confidence) of each rule.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.inference.wang_mendel import WangMendelRegressor
    >>> X = np.random.rand(200, 2)
    >>> y = X[:, 0] ** 2 + X[:, 1]
    >>> model = WangMendelRegressor(n_mfs=5, mf_type='triangular')
    >>> model.fit(X, y)
    WangMendelRegressor(...)
    >>> preds = model.predict(X[:5])
    """

    def __init__(
        self,
        n_mfs: int = 5,
        mf_type: str = "triangular",
        t_norm: str = "product",
        defuzz_method: str = "weighted_average",
        n_output_points: int = 200,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,  # determined by algorithm
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.defuzz_method = defuzz_method
        self.n_output_points = n_output_points

    def fit(self, X: Any, y: Any) -> WangMendelRegressor:
        """Fit the Wang-Mendel fuzzy system.

        Generates rules from training data using the Wang-Mendel algorithm:
        each data point produces a candidate rule, and conflicts are
        resolved by keeping the rule with the highest degree.

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
        self.feature_names_ = [f"x{j}" for j in range(n_features)]

        # Step 1: Partition input and output space into fuzzy sets
        self.mfs_ = self._init_membership_functions(X)

        y_min, y_max = float(np.min(y)), float(np.max(y))
        padding = (y_max - y_min) * 0.1 if y_max > y_min else 1.0
        self.output_mfs_ = create_uniform_mfs(
            n_mfs=self.n_mfs,
            x_min=y_min - padding,
            x_max=y_max + padding,
            mf_type=self.mf_type,
        )
        self.y_min_ = y_min - padding
        self.y_max_ = y_max + padding

        # Steps 2-4: Generate rules and resolve conflicts
        self.rule_base_: dict[tuple[int, ...], tuple[int, float]] = {}

        for i in range(n_samples):
            # Determine best MF for each input feature
            antecedent = []
            degree = 1.0
            for j in range(n_features):
                memberships = np.array(
                    [self.mfs_[j][k](np.array([X[i, j]]))[0] for k in range(self.n_mfs)]
                )
                best_k = int(np.argmax(memberships))
                antecedent.append(best_k)
                degree *= memberships[best_k]

            # Determine best output MF
            out_memberships = np.array(
                [mf(np.array([y[i]]))[0] for mf in self.output_mfs_]
            )
            best_out = int(np.argmax(out_memberships))
            degree *= out_memberships[best_out]

            ant_key = tuple(antecedent)

            # Conflict resolution: keep rule with highest degree
            if ant_key not in self.rule_base_ or degree > self.rule_base_[ant_key][1]:
                self.rule_base_[ant_key] = (best_out, degree)

        # Convert rule base to arrays for efficient inference
        n_rules = len(self.rule_base_)
        if n_rules == 0:
            # Fallback: add a default rule
            self.rule_base_[tuple([0] * n_features)] = (0, 1.0)
            n_rules = 1

        self.antecedent_indices_ = np.zeros((n_rules, n_features), dtype=int)
        self.consequent_indices_ = np.zeros(n_rules, dtype=int)
        self.rule_degrees_ = np.zeros(n_rules, dtype=np.float64)

        for idx, (ant_key, (cons, deg)) in enumerate(self.rule_base_.items()):
            self.antecedent_indices_[idx] = list(ant_key)
            self.consequent_indices_[idx] = cons
            self.rule_degrees_[idx] = deg

        self.n_rules = n_rules
        self._log(
            f"Wang-Mendel generated {n_rules} rules from {n_samples} samples"
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using the Wang-Mendel rule base.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(
            self, ["mfs_", "antecedent_indices_", "consequent_indices_", "rule_base_"]
        )
        X = check_array(X, dtype=np.float64)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} features, got {X.shape[1]}"
            )

        # Compute firing strengths
        strengths = self._compute_firing_strengths(X, self.antecedent_indices_)

        # Compute output MF centers for weighted average
        y_universe = np.linspace(self.y_min_, self.y_max_, self.n_output_points)
        out_mf_values = np.zeros((len(self.output_mfs_), self.n_output_points))
        for k, mf in enumerate(self.output_mfs_):
            out_mf_values[k] = mf(y_universe)

        # Pre-compute output MF centers (centroid of each MF)
        out_centers = np.zeros(len(self.output_mfs_))
        for k in range(len(self.output_mfs_)):
            total = np.sum(out_mf_values[k])
            if total > 0:
                out_centers[k] = np.sum(y_universe * out_mf_values[k]) / total
            else:
                out_centers[k] = y_universe[len(y_universe) // 2]

        if self.defuzz_method in ("weighted_average", "height"):
            # Weighted average using output MF centers
            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                # Map each rule's firing strength to its consequent center
                rule_centers = out_centers[self.consequent_indices_]
                fire = strengths[i]
                total_fire = np.sum(fire)
                if total_fire > 0:
                    predictions[i] = np.sum(fire * rule_centers) / total_fire
                else:
                    predictions[i] = np.mean(rule_centers)
            return predictions
        else:
            # Mamdani-style: clip-and-aggregate, then defuzzify
            predictions = np.zeros(X.shape[0])
            for i in range(X.shape[0]):
                aggregated = np.zeros(self.n_output_points)
                for r in range(self.antecedent_indices_.shape[0]):
                    firing = strengths[i, r]
                    if firing < 1e-12:
                        continue
                    cons_idx = self.consequent_indices_[r]
                    clipped = np.minimum(out_mf_values[cons_idx], firing)
                    aggregated = np.maximum(aggregated, clipped)

                predictions[i] = defuzzify(
                    y_universe, aggregated, method=self.defuzz_method
                )
            return predictions

    def get_rules_str(self) -> str:
        """Get human-readable representation of the Wang-Mendel rules.

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
                f"[deg={self.rule_degrees_[r]:.4f}]"
            )
        return "\n".join(lines)
