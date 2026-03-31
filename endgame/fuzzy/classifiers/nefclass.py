"""NEFCLASS: Neuro-Fuzzy Classification (Nauck & Kruse).

A 3-layer neuro-fuzzy architecture that learns interpretable fuzzy rules
from data. Layer 1 computes input membership degrees, Layer 2 activates
fuzzy rules via t-norm, and Layer 3 aggregates rule outputs per class.

Example
-------
>>> from endgame.fuzzy.classifiers import NEFCLASSClassifier
>>> clf = NEFCLASSClassifier(n_rules_per_class=5, n_mfs=3, n_epochs=50)
>>> clf.fit(X_train, y_train)
>>> proba = clf.predict_proba(X_test)
>>> rules = clf.get_rules_text()
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.core.base import BaseFuzzyClassifier
from endgame.fuzzy.core.membership import BaseMembershipFunction, create_uniform_mfs


class NEFCLASSClassifier(BaseFuzzyClassifier):
    """Neuro-Fuzzy Classification (NEFCLASS) classifier.

    Implements a 3-layer neuro-fuzzy network that learns interpretable
    IF-THEN fuzzy rules. The architecture consists of:

    - **Layer 1 (Input)**: Neurons with fuzzy membership functions that
      fuzzify each input feature.
    - **Layer 2 (Rule)**: Rule neurons that combine antecedent memberships
      using a t-norm. Each rule maps a combination of input MF indices
      to a class.
    - **Layer 3 (Output)**: Class neurons that aggregate rule activations
      for each class.

    Parameters
    ----------
    n_rules_per_class : int, default=5
        Number of fuzzy rules to learn per class.
    n_mfs : int, default=3
        Number of membership functions per input feature.
    n_epochs : int, default=50
        Number of training epochs for MF optimization.
    lr : float, default=0.01
        Learning rate for MF parameter adjustment.
    mf_type : str, default='gaussian'
        Type of membership function: 'gaussian', 'triangular', 'trapezoidal'.
    t_norm : str, default='product'
        T-norm for antecedent combination: 'min', 'product'.
    random_state : int or None, default=None
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output during training.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_classes_ : int
        Number of classes.
    n_features_in_ : int
        Number of features seen during fit.
    mfs_ : list of list of BaseMembershipFunction
        mfs_[feature_idx][mf_idx] — learned membership functions.
    rules_ : list of tuple(ndarray, int)
        Each rule is (antecedent_mf_indices, class_label) where
        antecedent_mf_indices[j] is the MF index for feature j.
    rule_weights_ : ndarray of shape (n_rules,)
        Learned weight for each rule.

    References
    ----------
    D. Nauck, R. Kruse, "NEFCLASS - A Neuro-Fuzzy Approach for the
    Classification of Data", Proc. ACM Symposium on Applied Computing, 1995.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.classifiers import NEFCLASSClassifier
    >>> X = np.random.randn(100, 4)
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> clf = NEFCLASSClassifier(n_rules_per_class=3, n_mfs=3, n_epochs=30)
    >>> clf.fit(X, y)
    NEFCLASSClassifier(n_epochs=30, n_rules_per_class=3)
    >>> clf.predict_proba(X[:5]).shape
    (5, 2)
    >>> len(clf.rules_) <= 6  # at most 3 per class * 2 classes
    True
    """

    def __init__(
        self,
        n_rules_per_class: int = 5,
        n_mfs: int = 3,
        n_epochs: int = 50,
        lr: float = 0.01,
        mf_type: str = "gaussian",
        t_norm: str = "product",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            n_rules=0,  # Will be set during fit
            n_mfs=n_mfs,
            mf_type=mf_type,
            t_norm=t_norm,
            random_state=random_state,
            verbose=verbose,
        )
        self.n_rules_per_class = n_rules_per_class
        self.n_epochs = n_epochs
        self.lr = lr

    def fit(self, X: Any, y: Any) -> NEFCLASSClassifier:
        """Fit the NEFCLASS model.

        Performs three stages:
        1. Initialize membership functions uniformly for each feature.
        2. Learn fuzzy rules by finding best-matching MFs for training samples.
        3. Optimize MF parameters to improve classification accuracy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape

        rng = np.random.RandomState(self.random_state)

        # Encode labels
        y_encoded = self._encode_labels(y)

        # Stage 1: Initialize membership functions
        self.mfs_ = self._init_membership_functions(X)

        # Stage 2: Rule learning
        self._learn_rules(X, y_encoded, rng)

        # Stage 3: MF optimization
        self._optimize_mfs(X, y_encoded)

        return self

    def _learn_rules(
        self,
        X: np.ndarray,
        y: np.ndarray,
        rng: np.random.RandomState,
    ) -> None:
        """Learn fuzzy rules from training data.

        For each training sample, find the best-matching MF for each
        input feature and create a candidate rule. Keep the top rules
        per class based on activation strength.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Encoded class labels.
        rng : RandomState
            Random number generator.
        """
        n_samples, n_features = X.shape

        # Precompute membership degrees: (n_features, n_mfs, n_samples)
        all_memberships = np.zeros((n_features, self.n_mfs, n_samples))
        for j in range(n_features):
            for k in range(self.n_mfs):
                all_memberships[j, k] = self.mfs_[j][k](X[:, j])

        # For each sample, find best-matching MF per feature
        best_mf_indices = np.argmax(all_memberships, axis=1).T
        # best_mf_indices: (n_samples, n_features)

        # Collect candidate rules per class
        # rule_key -> (antecedent_indices, class, total_activation, count)
        class_candidates: dict[int, dict[tuple, float]] = {
            c: {} for c in range(self.n_classes_)
        }

        tnorm_op = self._get_t_norm()

        for i in range(n_samples):
            antecedent = tuple(best_mf_indices[i])
            c = y[i]

            # Compute rule activation (firing strength)
            activation = np.ones(1)
            for j in range(n_features):
                mf_idx = antecedent[j]
                activation = tnorm_op(
                    activation,
                    np.array([all_memberships[j, mf_idx, i]]),
                )

            act_val = float(activation[0])

            if antecedent in class_candidates[c]:
                class_candidates[c][antecedent] += act_val
            else:
                class_candidates[c][antecedent] = act_val

        # Select top rules per class
        self.rules_ = []
        self.rule_weights_ = []

        for c in range(self.n_classes_):
            candidates = class_candidates[c]
            if not candidates:
                # Fallback: create a rule with the most common MF indices
                # for samples of this class
                class_mask = y == c
                if np.any(class_mask):
                    class_best = best_mf_indices[class_mask]
                    # Mode per feature
                    mode_indices = np.zeros(n_features, dtype=int)
                    for j in range(n_features):
                        counts = np.bincount(class_best[:, j], minlength=self.n_mfs)
                        mode_indices[j] = np.argmax(counts)
                    self.rules_.append((mode_indices.copy(), c))
                    self.rule_weights_.append(1.0)
                continue

            # Sort by cumulative activation (descending)
            sorted_rules = sorted(
                candidates.items(), key=lambda item: item[1], reverse=True
            )

            n_keep = min(self.n_rules_per_class, len(sorted_rules))
            for antecedent, act_sum in sorted_rules[:n_keep]:
                self.rules_.append((np.array(antecedent, dtype=int), c))
                self.rule_weights_.append(act_sum)

        self.rule_weights_ = np.array(self.rule_weights_, dtype=np.float64)

        # Normalize weights
        if len(self.rule_weights_) > 0 and np.max(self.rule_weights_) > 0:
            self.rule_weights_ /= np.max(self.rule_weights_)

        self._log(f"Learned {len(self.rules_)} rules")

    def _optimize_mfs(self, X: np.ndarray, y: np.ndarray) -> None:
        """Optimize membership function parameters using heuristic learning.

        Adjusts MF centers/widths to improve classification accuracy
        by shifting MFs toward correctly classified samples and away
        from misclassified ones.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        y : ndarray of shape (n_samples,)
            Encoded class labels.
        """
        if len(self.rules_) == 0:
            return

        n_samples, n_features = X.shape

        for epoch in range(self.n_epochs):
            # Shuffle sample order
            order = np.arange(n_samples)
            np.random.shuffle(order)

            n_correct = 0

            for i in order:
                x_i = X[i]
                true_class = y[i]

                # Compute rule activations
                activations = self._compute_rule_activations(x_i)

                # Determine predicted class
                class_scores = np.zeros(self.n_classes_)
                for r, (antecedent, rule_class) in enumerate(self.rules_):
                    class_scores[rule_class] += (
                        activations[r] * self.rule_weights_[r]
                    )

                pred_class = np.argmax(class_scores)

                if pred_class == true_class:
                    n_correct += 1
                    continue

                # Misclassified: adjust MFs
                # Find the best-matching rule for the true class
                best_true_rule = -1
                best_true_act = -1.0
                for r, (antecedent, rule_class) in enumerate(self.rules_):
                    if rule_class == true_class and activations[r] > best_true_act:
                        best_true_act = activations[r]
                        best_true_rule = r

                if best_true_rule < 0:
                    continue

                # Shift MFs of the true-class rule toward the sample
                true_antecedent = self.rules_[best_true_rule][0]
                for j in range(n_features):
                    mf = self.mfs_[j][true_antecedent[j]]
                    self._shift_mf_toward(mf, x_i[j], self.lr)

                # Find the best-matching rule for the predicted (wrong) class
                best_wrong_rule = -1
                best_wrong_act = -1.0
                for r, (antecedent, rule_class) in enumerate(self.rules_):
                    if rule_class == pred_class and activations[r] > best_wrong_act:
                        best_wrong_act = activations[r]
                        best_wrong_rule = r

                if best_wrong_rule >= 0:
                    wrong_antecedent = self.rules_[best_wrong_rule][0]
                    for j in range(n_features):
                        mf = self.mfs_[j][wrong_antecedent[j]]
                        self._shift_mf_away(mf, x_i[j], self.lr)

            accuracy = n_correct / n_samples if n_samples > 0 else 0.0
            self._log(f"Epoch {epoch + 1}/{self.n_epochs}, accuracy={accuracy:.4f}")

    @staticmethod
    def _shift_mf_toward(mf: BaseMembershipFunction, x: float, lr: float) -> None:
        """Shift a membership function's center toward a value.

        Parameters
        ----------
        mf : BaseMembershipFunction
            The membership function to adjust.
        x : float
            Target value to shift toward.
        lr : float
            Learning rate.
        """
        if hasattr(mf, "center"):
            mf.center += lr * (x - mf.center)
        elif hasattr(mf, "b"):
            # Triangular or trapezoidal: shift peak toward x
            mf.b += lr * (x - mf.b)
            # Maintain ordering constraints
            if hasattr(mf, "a") and hasattr(mf, "c"):
                if mf.a > mf.b:
                    mf.a = mf.b
                if mf.c < mf.b:
                    mf.c = mf.b
                if hasattr(mf, "d") and mf.d < mf.c:
                    mf.d = mf.c

    @staticmethod
    def _shift_mf_away(mf: BaseMembershipFunction, x: float, lr: float) -> None:
        """Shift a membership function's center away from a value.

        Parameters
        ----------
        mf : BaseMembershipFunction
            The membership function to adjust.
        x : float
            Value to move away from.
        lr : float
            Learning rate.
        """
        if hasattr(mf, "center"):
            mf.center -= lr * (x - mf.center)
        elif hasattr(mf, "b"):
            mf.b -= lr * (x - mf.b)
            if hasattr(mf, "a") and hasattr(mf, "c"):
                if mf.a > mf.b:
                    mf.a = mf.b
                if mf.c < mf.b:
                    mf.c = mf.b
                if hasattr(mf, "d") and mf.d < mf.c:
                    mf.d = mf.c

    def _compute_rule_activations(self, x: np.ndarray) -> np.ndarray:
        """Compute firing strength of all rules for a single sample.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Single input sample.

        Returns
        -------
        ndarray of shape (n_rules,)
            Rule activation (firing) strengths.
        """
        n_rules = len(self.rules_)
        n_features = len(x)
        tnorm_op = self._get_t_norm()

        activations = np.ones(n_rules)
        for r, (antecedent, _) in enumerate(self.rules_):
            for j in range(n_features):
                mf_idx = antecedent[j]
                membership = float(self.mfs_[j][mf_idx](np.array([x[j]]))[0])
                activations[r] = float(
                    tnorm_op(
                        np.array([activations[r]]),
                        np.array([membership]),
                    )[0]
                )
        return activations

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Computes weighted rule activations and aggregates them by class.
        The result is normalized to produce valid probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["rules_", "mfs_"])
        X = check_array(X)

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {X.shape[1]} features, expected {self.n_features_in_}"
            )

        n_samples = X.shape[0]
        proba = np.zeros((n_samples, self.n_classes_))

        for i in range(n_samples):
            activations = self._compute_rule_activations(X[i])

            for r, (antecedent, rule_class) in enumerate(self.rules_):
                proba[i, rule_class] += (
                    activations[r] * self.rule_weights_[r]
                )

        # Normalize to probabilities
        row_sums = proba.sum(axis=1, keepdims=True)
        # Avoid division by zero: assign uniform probability
        zero_rows = (row_sums < 1e-10).ravel()
        row_sums[zero_rows] = 1.0
        proba[zero_rows] = 1.0 / self.n_classes_
        proba = proba / row_sums

        return proba

    def get_rules_text(self) -> str:
        """Get human-readable text representation of learned rules.

        Returns
        -------
        str
            Multi-line string with one rule per line in
            IF-THEN format with linguistic terms.
        """
        check_is_fitted(self, ["rules_", "mfs_"])

        term_names = [
            "low", "medium", "high", "very_low", "low_med",
            "med_high", "very_high", "extra1", "extra2", "extra3",
        ]

        lines = []
        for r, (antecedent, rule_class) in enumerate(self.rules_):
            conditions = []
            for j, mf_idx in enumerate(antecedent):
                term = (
                    term_names[mf_idx]
                    if mf_idx < len(term_names)
                    else f"term_{mf_idx}"
                )
                fname = (
                    self.feature_names_[j]
                    if hasattr(self, "feature_names_")
                    else f"x{j}"
                )
                conditions.append(f"{fname} IS {term}")

            class_label = self.classes_[rule_class]
            weight = self.rule_weights_[r]
            ant_str = " AND ".join(conditions)
            lines.append(
                f"Rule {r + 1}: IF {ant_str} THEN class={class_label} "
                f"[w={weight:.4f}]"
            )

        return "\n".join(lines)
