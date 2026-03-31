"""Fuzzy Boosted Trees -- gradient boosting with soft split nodes.

The key innovation is that each node uses a parameterised sigmoid
``w(x) = sigmoid(a * (x_j - threshold))`` to route samples softly
through both children.  The sharpness parameter *a* controls the
fuzziness: large values recover crisp splits, small values produce
smooth, interpolative decision surfaces.

Example
-------
>>> from endgame.fuzzy.ensemble import FuzzyBoostedTreesRegressor
>>> reg = FuzzyBoostedTreesRegressor(n_estimators=100, learning_rate=0.05)
>>> reg.fit(X_train, y_train)
>>> reg.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


# ---------------------------------------------------------------------------
# Internal fuzzy boosting tree (single weak learner)
# ---------------------------------------------------------------------------


class _FuzzyBoostNode:
    """Node of a fuzzy boosting tree."""

    __slots__ = (
        "feature", "threshold", "sharpness",
        "left", "right",
        "value", "is_leaf",
    )

    def __init__(self):
        self.feature: int = -1
        self.threshold: float = 0.0
        self.sharpness: float = 5.0
        self.left: _FuzzyBoostNode | None = None
        self.right: _FuzzyBoostNode | None = None
        self.value: float | np.ndarray = 0.0
        self.is_leaf: bool = False


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(
        x >= 0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


class _FuzzyBoostingTree:
    """A single fuzzy tree fitted to residuals.

    Each internal node applies a soft split via sigmoid:
    ``mu_right = sigmoid(sharpness * (x_j - threshold))``, and
    ``mu_left = 1 - mu_right``.  Leaf values are fitted to minimise
    weighted squared residuals.

    Parameters
    ----------
    max_depth : int
        Maximum tree depth.
    min_samples_leaf : int
        Minimum effective sample count per child.
    sharpness : float
        Default sharpness (steepness) of the sigmoid splits.
    rng : np.random.Generator
        Random number generator.
    """

    def __init__(
        self,
        max_depth: int,
        min_samples_leaf: int,
        sharpness: float,
        rng: np.random.Generator,
    ):
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sharpness = sharpness
        self.rng = rng
        self.root: _FuzzyBoostNode | None = None
        self.feature_importances: np.ndarray | None = None

    # -- fit ---------------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        sample_weight: np.ndarray,
    ) -> None:
        """Build the tree from residuals."""
        self.n_features_ = X.shape[1]
        self.feature_importances = np.zeros(X.shape[1], dtype=np.float64)
        self.root = self._build(X, residuals, sample_weight, depth=0)

    def _build(
        self,
        X: np.ndarray,
        residuals: np.ndarray,
        w: np.ndarray,
        depth: int,
    ) -> _FuzzyBoostNode:
        node = _FuzzyBoostNode()
        total_w = w.sum()

        # -- leaf value: weighted mean of residuals --
        if total_w < 1e-12:
            node.is_leaf = True
            node.value = 0.0
            return node

        leaf_val = np.average(residuals, weights=w)

        if depth >= self.max_depth or total_w < 2.0 * self.min_samples_leaf:
            node.is_leaf = True
            node.value = leaf_val
            return node

        best_gain = 0.0
        best_feat = -1
        best_thresh = 0.0
        best_sharp = self.sharpness

        parent_loss = np.average((residuals - leaf_val) ** 2, weights=w)

        # Evaluate candidate splits
        for feat in range(X.shape[1]):
            col = X[:, feat]
            col_min, col_max = col.min(), col.max()
            if col_max - col_min < 1e-12:
                continue

            # Candidate thresholds: 5 quantiles
            quantiles = np.quantile(col, [0.2, 0.35, 0.5, 0.65, 0.8])
            for thresh in quantiles:
                mu_right = _sigmoid(self.sharpness * (col - thresh))
                mu_left = 1.0 - mu_right

                w_left = w * mu_left
                w_right = w * mu_right
                sw_left = w_left.sum()
                sw_right = w_right.sum()

                if sw_left < self.min_samples_leaf or sw_right < self.min_samples_leaf:
                    continue

                mean_left = np.average(residuals, weights=w_left)
                mean_right = np.average(residuals, weights=w_right)

                loss_left = np.average((residuals - mean_left) ** 2, weights=w_left)
                loss_right = np.average((residuals - mean_right) ** 2, weights=w_right)

                child_loss = (sw_left * loss_left + sw_right * loss_right) / total_w
                gain = parent_loss - child_loss

                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
                    best_sharp = self.sharpness

        if best_feat < 0:
            node.is_leaf = True
            node.value = leaf_val
            return node

        node.feature = best_feat
        node.threshold = best_thresh
        node.sharpness = best_sharp
        self.feature_importances[best_feat] += best_gain * total_w  # type: ignore[index]

        col = X[:, best_feat]
        mu_right = _sigmoid(best_sharp * (col - best_thresh))
        mu_left = 1.0 - mu_right

        node.left = self._build(X, residuals, w * mu_left, depth + 1)
        node.right = self._build(X, residuals, w * mu_right, depth + 1)
        return node

    # -- predict -----------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        accum = np.zeros(n, dtype=np.float64)
        weights = np.zeros(n, dtype=np.float64)
        self._predict_node(X, self.root, np.ones(n), accum, weights)
        safe = np.where(weights > 0, weights, 1.0)
        return accum / safe

    def _predict_node(
        self,
        X: np.ndarray,
        node: _FuzzyBoostNode,
        w: np.ndarray,
        accum: np.ndarray,
        weights: np.ndarray,
    ) -> None:
        if node.is_leaf:
            accum += w * node.value
            weights += w
            return

        col = X[:, node.feature]
        mu_right = _sigmoid(node.sharpness * (col - node.threshold))
        mu_left = 1.0 - mu_right

        self._predict_node(X, node.left, w * mu_left, accum, weights)
        self._predict_node(X, node.right, w * mu_right, accum, weights)


# ---------------------------------------------------------------------------
# FuzzyBoostedTreesRegressor
# ---------------------------------------------------------------------------


class FuzzyBoostedTreesRegressor(BaseEstimator, RegressorMixin):
    """Gradient boosting with fuzzy (soft) split nodes for regression.

    At each boosting iteration a fuzzy tree is fitted to the negative
    gradient (residuals for squared-error loss).  Internal nodes use a
    sigmoid soft split ``sigmoid(sharpness * (x_j - threshold))`` so
    that every sample contributes to both children weighted by its
    membership degree.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Shrinkage applied to each tree's contribution.
    max_depth : int, default=3
        Maximum depth of each fuzzy tree.
    min_samples_leaf : int, default=10
        Minimum effective sample count per leaf.
    sharpness : float, default=5.0
        Sigmoid steepness for soft splits.  Higher values approach
        crisp (hard) splits; lower values give smoother boundaries.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    n_features_in_ : int
        Number of features.
    feature_importances_ : ndarray of shape (n_features,)
        Normalised gain-based feature importance.
    train_loss_ : list of float
        Training MSE after each boosting iteration.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyBoostedTreesRegressor
    >>> reg = FuzzyBoostedTreesRegressor(n_estimators=50, learning_rate=0.1)
    >>> reg.fit(X_train, y_train)
    >>> preds = reg.predict(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        sharpness: float = 5.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sharpness = sharpness
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> FuzzyBoostedTreesRegressor:
        """Fit the fuzzy gradient boosting regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        self.n_features_in_ = X.shape[1]

        rng = np.random.default_rng(self.random_state)

        # Initialise with mean
        self.init_value_ = float(np.mean(y))
        current_pred = np.full(X.shape[0], self.init_value_, dtype=np.float64)

        self.estimators_: list[_FuzzyBoostingTree] = []
        self.train_loss_: list[float] = []
        raw_importances = np.zeros(X.shape[1], dtype=np.float64)
        sample_weight = np.ones(X.shape[0], dtype=np.float64)

        for i in range(self.n_estimators):
            residuals = y - current_pred

            tree = _FuzzyBoostingTree(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                sharpness=self.sharpness,
                rng=rng,
            )
            tree.fit(X, residuals, sample_weight)
            update = tree.predict(X)
            current_pred += self.learning_rate * update

            self.estimators_.append(tree)
            raw_importances += tree.feature_importances  # type: ignore[operator]

            mse = float(np.mean((y - current_pred) ** 2))
            self.train_loss_.append(mse)

        total = raw_importances.sum()
        self.feature_importances_ = (
            raw_importances / total if total > 0 else raw_importances
        )
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)
        pred = np.full(X.shape[0], self.init_value_, dtype=np.float64)
        for tree in self.estimators_:
            pred += self.learning_rate * tree.predict(X)
        return pred


# ---------------------------------------------------------------------------
# FuzzyBoostedTreesClassifier
# ---------------------------------------------------------------------------


class FuzzyBoostedTreesClassifier(BaseEstimator, ClassifierMixin):
    """Gradient boosting with fuzzy (soft) split nodes for classification.

    Uses log-loss for binary targets and one-vs-rest softmax for
    multi-class.  Internal tree nodes employ a sigmoid soft split
    controlled by the ``sharpness`` parameter.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of boosting iterations.
    learning_rate : float, default=0.1
        Shrinkage factor.
    max_depth : int, default=3
        Maximum depth of each fuzzy tree.
    min_samples_leaf : int, default=10
        Minimum effective sample count per leaf.
    sharpness : float, default=5.0
        Sigmoid steepness for soft splits.
    random_state : int or None, default=None
        Random seed.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features.
    feature_importances_ : ndarray of shape (n_features,)
        Normalised gain-based feature importance.
    train_loss_ : list of float
        Training log-loss after each iteration.

    Examples
    --------
    >>> from endgame.fuzzy.ensemble import FuzzyBoostedTreesClassifier
    >>> clf = FuzzyBoostedTreesClassifier(n_estimators=50)
    >>> clf.fit(X_train, y_train)
    >>> clf.predict_proba(X_test)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 10,
        sharpness: float = 5.0,
        random_state: int | None = None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.sharpness = sharpness
        self.random_state = random_state

    def fit(self, X: Any, y: Any) -> FuzzyBoostedTreesClassifier:
        """Fit the fuzzy gradient boosting classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        self.classes_ = le.classes_
        self.n_classes_ = len(self.classes_)
        self.label_encoder_ = le

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        sample_weight = np.ones(n_samples, dtype=np.float64)

        raw_importances = np.zeros(X.shape[1], dtype=np.float64)
        self.train_loss_: list[float] = []

        if self.n_classes_ == 2:
            # Binary classification -- single set of trees
            self.estimators_: list[list[_FuzzyBoostingTree]] = [[]]

            # Initialise with log-odds
            p_pos = np.clip(np.mean(y_enc), 1e-7, 1 - 1e-7)
            self.init_raw_ = np.log(p_pos / (1 - p_pos))
            raw_pred = np.full(n_samples, self.init_raw_, dtype=np.float64)

            for i in range(self.n_estimators):
                prob = _sigmoid(raw_pred)
                residuals = y_enc.astype(np.float64) - prob

                tree = _FuzzyBoostingTree(
                    max_depth=self.max_depth,
                    min_samples_leaf=self.min_samples_leaf,
                    sharpness=self.sharpness,
                    rng=rng,
                )
                tree.fit(X, residuals, sample_weight)
                raw_pred += self.learning_rate * tree.predict(X)

                self.estimators_[0].append(tree)
                raw_importances += tree.feature_importances  # type: ignore[operator]

                # Log loss
                prob = np.clip(_sigmoid(raw_pred), 1e-15, 1 - 1e-15)
                logloss = -float(np.mean(
                    y_enc * np.log(prob) + (1 - y_enc) * np.log(1 - prob)
                ))
                self.train_loss_.append(logloss)
        else:
            # Multi-class: one-vs-rest
            self.estimators_ = [[] for _ in range(self.n_classes_)]

            # Initialise: log(class_prior)
            class_counts = np.bincount(y_enc, minlength=self.n_classes_).astype(np.float64)
            class_priors = np.clip(class_counts / n_samples, 1e-7, 1 - 1e-7)
            self.init_raw_ = np.log(class_priors)
            raw_pred = np.tile(self.init_raw_, (n_samples, 1))  # (n, K)

            for i in range(self.n_estimators):
                # Softmax probabilities
                exp_raw = np.exp(raw_pred - raw_pred.max(axis=1, keepdims=True))
                probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)

                for k in range(self.n_classes_):
                    target_k = (y_enc == k).astype(np.float64)
                    residuals_k = target_k - probs[:, k]

                    tree = _FuzzyBoostingTree(
                        max_depth=self.max_depth,
                        min_samples_leaf=self.min_samples_leaf,
                        sharpness=self.sharpness,
                        rng=rng,
                    )
                    tree.fit(X, residuals_k, sample_weight)
                    raw_pred[:, k] += self.learning_rate * tree.predict(X)

                    self.estimators_[k].append(tree)
                    raw_importances += tree.feature_importances  # type: ignore[operator]

                # Cross-entropy loss
                exp_raw = np.exp(raw_pred - raw_pred.max(axis=1, keepdims=True))
                probs = exp_raw / exp_raw.sum(axis=1, keepdims=True)
                probs = np.clip(probs, 1e-15, 1 - 1e-15)
                logloss = -float(np.mean(np.log(probs[np.arange(n_samples), y_enc])))
                self.train_loss_.append(logloss)

        total = raw_importances.sum()
        self.feature_importances_ = (
            raw_importances / total if total > 0 else raw_importances
        )
        return self

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)
        n_samples = X.shape[0]

        if self.n_classes_ == 2:
            raw = np.full(n_samples, self.init_raw_, dtype=np.float64)
            for tree in self.estimators_[0]:
                raw += self.learning_rate * tree.predict(X)
            prob_pos = _sigmoid(raw)
            return np.column_stack([1.0 - prob_pos, prob_pos])
        else:
            raw = np.tile(self.init_raw_, (n_samples, 1))
            for k in range(self.n_classes_):
                for tree in self.estimators_[k]:
                    raw[:, k] += self.learning_rate * tree.predict(X)
            exp_raw = np.exp(raw - raw.max(axis=1, keepdims=True))
            return exp_raw / exp_raw.sum(axis=1, keepdims=True)

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
