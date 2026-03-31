"""Self-Evolving Interval Type-2 Fuzzy Neural Network (SEIT2FNN).

An online learning IT2 fuzzy system with automatic structure
(rule addition/pruning) and parameter learning capabilities.

References
----------
Juang, C.F., & Tsao, Y.W. (2008). A self-evolving interval type-2
fuzzy neural network with online structure and parameter learning.
IEEE Transactions on Fuzzy Systems, 16(6), 1411-1424.

Example
-------
>>> from endgame.fuzzy.modern.seit2fnn import SEIT2FNNClassifier
>>> clf = SEIT2FNNClassifier(max_rules=20, n_epochs=30)
>>> clf.fit(X_train, y_train)
>>> # Online update:
>>> clf.partial_fit(X_new, y_new)
>>> predictions = clf.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


class SEIT2FNNClassifier(BaseEstimator, ClassifierMixin):
    """Self-Evolving Interval Type-2 Fuzzy Neural Network for classification.

    Combines online structure learning (rule creation based on novelty)
    with parameter learning using gradient descent. Uses IT2 Gaussian
    membership functions with uncertain sigma (footprint of uncertainty).

    Parameters
    ----------
    threshold_add : float, default=0.3
        Novelty threshold for adding new rules. A new rule is created
        when the maximum firing strength for a sample is below this value.
    max_rules : int, default=50
        Maximum number of fuzzy rules.
    n_epochs : int, default=50
        Number of training epochs for batch fitting.
    lr : float, default=0.01
        Learning rate for parameter updates.
    sigma_ratio : float, default=0.3
        Ratio for initializing the lower sigma relative to upper sigma.
        Lower sigma = upper sigma * sigma_ratio.
    prune_threshold : float, default=0.01
        Rules with average firing below this threshold are pruned.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.
    n_rules_ : int
        Current number of rules.
    centers_ : ndarray of shape (n_rules, n_features)
        Rule antecedent centers.
    sigmas_upper_ : ndarray of shape (n_rules, n_features)
        Upper MF widths.
    sigmas_lower_ : ndarray of shape (n_rules, n_features)
        Lower MF widths.
    consequent_weights_ : ndarray of shape (n_rules, n_classes)
        Consequent class weights.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.seit2fnn import SEIT2FNNClassifier
    >>> X = np.random.randn(100, 5)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = SEIT2FNNClassifier(max_rules=10, n_epochs=20)
    >>> clf.fit(X, y)
    SEIT2FNNClassifier(max_rules=10, n_epochs=20)
    >>> clf.partial_fit(X[:10], y[:10])  # online update
    SEIT2FNNClassifier(max_rules=10, n_epochs=20)
    >>> clf.predict(X[:3])
    array([...])
    """

    def __init__(
        self,
        threshold_add: float = 0.3,
        max_rules: int = 50,
        n_epochs: int = 50,
        lr: float = 0.01,
        sigma_ratio: float = 0.3,
        prune_threshold: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.threshold_add = threshold_add
        self.max_rules = max_rules
        self.n_epochs = n_epochs
        self.lr = lr
        self.sigma_ratio = sigma_ratio
        self.prune_threshold = prune_threshold
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> SEIT2FNNClassifier:
        """Fit the SEIT2FNN model from scratch.

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

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_
        n_classes = len(self.classes_)

        rng = np.random.RandomState(self.random_state)

        # Initialize with no rules
        n_features = X.shape[1]
        self.centers_ = np.empty((0, n_features))
        self.sigmas_upper_ = np.empty((0, n_features))
        self.sigmas_lower_ = np.empty((0, n_features))
        self.consequent_weights_ = np.empty((0, n_classes))

        # Global data statistics for new rule initialization
        self._feature_stds = np.std(X, axis=0)
        self._feature_stds = np.where(self._feature_stds < 1e-6, 1.0, self._feature_stds)

        # Structure and parameter learning over multiple epochs
        for epoch in range(self.n_epochs):
            perm = rng.permutation(len(X))
            epoch_loss = 0.0

            for i in perm:
                x_i = X[i:i+1]
                y_i = y_enc[i]

                # Structure learning: check if new rule is needed
                if self.centers_.shape[0] == 0:
                    self._add_rule(x_i[0], y_i, n_classes)
                else:
                    firing = self._compute_it2_firing(x_i)[0]
                    max_firing = np.max(firing)

                    if max_firing < self.threshold_add and self.centers_.shape[0] < self.max_rules:
                        self._add_rule(x_i[0], y_i, n_classes)

                # Parameter learning
                loss = self._update_params(x_i, y_i, n_classes)
                epoch_loss += loss

            # Prune low-activity rules
            if self.centers_.shape[0] > 1:
                self._prune_rules(X)

            avg_loss = epoch_loss / len(X)
            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[SEIT2FNN] Epoch {epoch+1}/{self.n_epochs}, "
                    f"Loss: {avg_loss:.6f}, Rules: {self.centers_.shape[0]}"
                )

        self.n_rules_ = self.centers_.shape[0]
        return self

    def partial_fit(self, X: Any, y: Any) -> SEIT2FNNClassifier:
        """Online update with new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            New data.
        y : array-like of shape (n_samples,)
            New labels.

        Returns
        -------
        self
            Updated estimator.
        """
        X, y = check_X_y(X, y)

        if not hasattr(self, "classes_"):
            return self.fit(X, y)

        y_enc = self.label_encoder_.transform(y)
        n_classes = len(self.classes_)

        for i in range(len(X)):
            x_i = X[i:i+1]
            y_i = y_enc[i]

            if self.centers_.shape[0] == 0:
                self._add_rule(x_i[0], y_i, n_classes)
            else:
                firing = self._compute_it2_firing(x_i)[0]
                max_firing = np.max(firing)

                if max_firing < self.threshold_add and self.centers_.shape[0] < self.max_rules:
                    self._add_rule(x_i[0], y_i, n_classes)

            self._update_params(x_i, y_i, n_classes)

        self.n_rules_ = self.centers_.shape[0]
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["centers_", "classes_"])
        X = check_array(X)

        if self.centers_.shape[0] == 0:
            # No rules: uniform prediction
            n_classes = len(self.classes_)
            return np.full((len(X), n_classes), 1.0 / n_classes)

        firing = self._compute_it2_firing(X)  # (n_samples, n_rules)
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum

        # Weighted class scores
        scores = normalized @ self.consequent_weights_  # (n_samples, n_classes)

        # Softmax normalization
        scores_shifted = scores - scores.max(axis=1, keepdims=True)
        exp_scores = np.exp(scores_shifted)
        return exp_scores / exp_scores.sum(axis=1, keepdims=True)

    def _compute_it2_firing(self, X: np.ndarray) -> np.ndarray:
        """Compute IT2 firing strengths (average of upper and lower).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_rules)
            Averaged firing strengths.
        """
        n_samples = X.shape[0]
        n_rules = self.centers_.shape[0]

        if n_rules == 0:
            return np.empty((n_samples, 0))

        upper_firing = np.ones((n_samples, n_rules))
        lower_firing = np.ones((n_samples, n_rules))

        for r in range(n_rules):
            diff = X - self.centers_[r]

            # Upper MF (wider)
            sigma_u_sq = self.sigmas_upper_[r] ** 2 + 1e-10
            upper_mf = np.exp(-0.5 * diff ** 2 / sigma_u_sq)
            upper_firing[:, r] = np.prod(upper_mf, axis=1)

            # Lower MF (narrower)
            sigma_l_sq = self.sigmas_lower_[r] ** 2 + 1e-10
            lower_mf = np.exp(-0.5 * diff ** 2 / sigma_l_sq)
            lower_firing[:, r] = np.prod(lower_mf, axis=1)

        # Type-reduction: average of upper and lower
        return (upper_firing + lower_firing) / 2.0

    def _add_rule(self, x: np.ndarray, y_class: int, n_classes: int) -> None:
        """Add a new rule centered at sample x.

        Parameters
        ----------
        x : ndarray of shape (n_features,)
            Sample to center the new rule on.
        y_class : int
            Encoded class label.
        n_classes : int
            Total number of classes.
        """
        center = x.reshape(1, -1)
        sigma_upper = self._feature_stds.reshape(1, -1) * 1.0
        sigma_lower = sigma_upper * self.sigma_ratio

        # Initialize consequent to favor the sample's class
        weights = np.full((1, n_classes), -0.1)
        weights[0, y_class] = 1.0

        self.centers_ = np.vstack([self.centers_, center])
        self.sigmas_upper_ = np.vstack([self.sigmas_upper_, sigma_upper])
        self.sigmas_lower_ = np.vstack([self.sigmas_lower_, sigma_lower])
        self.consequent_weights_ = np.vstack([self.consequent_weights_, weights])

    def _update_params(
        self,
        x: np.ndarray,
        y_class: int,
        n_classes: int,
    ) -> float:
        """Update parameters for a single sample via gradient descent.

        Parameters
        ----------
        x : ndarray of shape (1, n_features)
        y_class : int
            Encoded class label.
        n_classes : int
            Total number of classes.

        Returns
        -------
        float
            Cross-entropy loss for this sample.
        """
        n_rules = self.centers_.shape[0]
        if n_rules == 0:
            return 0.0

        firing = self._compute_it2_firing(x)[0]  # (n_rules,)
        fire_sum = firing.sum()
        if fire_sum < 1e-10:
            return 0.0
        normalized = firing / fire_sum

        # Compute scores and probabilities
        scores = normalized @ self.consequent_weights_  # (n_classes,)
        scores_shifted = scores - scores.max()
        exp_scores = np.exp(scores_shifted)
        proba = exp_scores / exp_scores.sum()

        # Cross-entropy loss
        loss = -np.log(proba[y_class] + 1e-10)

        # Gradient: d_loss/d_scores = proba - one_hot
        target = np.zeros(n_classes)
        target[y_class] = 1.0
        d_scores = proba - target  # (n_classes,)

        # Update consequent weights
        for r in range(n_rules):
            grad_w = normalized[r] * d_scores
            self.consequent_weights_[r] -= self.lr * grad_w

        return float(loss)

    def _prune_rules(self, X: np.ndarray) -> None:
        """Prune rules with low average firing strength.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data for evaluating rule activity.
        """
        if self.centers_.shape[0] <= 1:
            return

        firing = self._compute_it2_firing(X)
        avg_firing = firing.mean(axis=0)
        keep = avg_firing >= self.prune_threshold

        if keep.sum() == 0:
            # Keep at least the best rule
            keep[np.argmax(avg_firing)] = True

        self.centers_ = self.centers_[keep]
        self.sigmas_upper_ = self.sigmas_upper_[keep]
        self.sigmas_lower_ = self.sigmas_lower_[keep]
        self.consequent_weights_ = self.consequent_weights_[keep]
