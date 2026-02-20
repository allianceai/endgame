"""Cubist Regression Tree Implementation.

A Python implementation of the Cubist algorithm for rule-based regression
with support for the Rust accelerated backend when available.

Cubist combines decision trees with linear regression models, producing
rules that predict continuous values using piecewise linear models.

Features:
- Rule-based regression
- Linear models at leaf nodes
- Committee models (ensembles)
- Instance-based (k-NN) correction
- Missing value handling
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

# Try to import Rust backend
HAS_RUST = False
CubistRust = None

try:
    from endgame.models.trees.cubist_rust import CubistRust
    HAS_RUST = True
except ImportError:
    try:
        # Try direct import (when installed via maturin develop)
        from cubist_rust import CubistRust
        HAS_RUST = True
    except ImportError:
        pass


class CubistRegressor(BaseEstimator, RegressorMixin):
    """Cubist Regression Model.

    A high-performance implementation of the Cubist algorithm that combines
    decision trees with linear regression models. The resulting model consists
    of a set of rules, where each rule has conditions and a linear model for
    prediction.

    Parameters
    ----------
    committees : int, default=1
        Number of committee members (trees) to build. Using multiple committees
        creates a boosted ensemble where each subsequent model focuses on the
        residuals from previous models.
    neighbors : int, default=0
        Number of nearest neighbors to use for instance-based correction.
        Set to 0 to disable instance-based correction. When enabled, predictions
        are adjusted based on the residuals of nearby training instances.
    min_cases : int, default=2
        Minimum number of cases in a node before splitting is considered.
    max_rules : int, default=0
        Maximum number of rules to generate (0 = unlimited).
    sample : float, default=1.0
        Fraction of training data to use in each committee member.
    extrapolation : float, default=0.05
        Amount of extrapolation allowed beyond training range (as fraction).
    unbiased : bool, default=False
        If True, use unbiased splitting criterion.
    use_rust : bool, default=True
        Use the Rust backend if available for better performance.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_rules_ : int
        Number of rules in the final model.
    feature_importances_ : ndarray of shape (n_features,)
        Feature importances based on usage in splits.

    Examples
    --------
    >>> from endgame.models.trees import CubistRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(100) * 0.1
    >>> reg = CubistRegressor(committees=5, neighbors=5)
    >>> reg.fit(X, y)
    >>> predictions = reg.predict(X[:10])

    Notes
    -----
    Cubist was developed by Ross Quinlan as a commercial product. This
    implementation is based on the algorithm described in the open-source
    C code released under GPL.

    The algorithm works as follows:
    1. Build a regression tree by recursively splitting data to minimize
       variance in the target variable.
    2. At each leaf node, fit a linear model using the cases at that node.
    3. Extract rules from the tree paths.
    4. Prune rules to remove redundant conditions.
    5. Optionally build multiple trees (committees) using boosting.
    6. Optionally apply instance-based correction using k-NN.
    """

    def __init__(
        self,
        committees: int = 1,
        neighbors: int = 0,
        min_cases: int = 10,  # Higher default to prevent overfitting
        max_rules: int = 0,
        sample: float = 1.0,
        extrapolation: float = 0.05,
        unbiased: bool = False,
        use_rust: bool = True,
        random_state: int | None = None,
    ):
        self.committees = committees
        self.neighbors = neighbors
        self.min_cases = min_cases
        self.max_rules = max_rules
        self.sample = sample
        self.extrapolation = extrapolation
        self.unbiased = unbiased
        self.use_rust = use_rust
        self.random_state = random_state

    def fit(
        self,
        X: ArrayLike,
        y: ArrayLike,
        sample_weight: ArrayLike | None = None,
    ) -> CubistRegressor:
        """Fit the Cubist regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights. Currently not fully supported.

        Returns
        -------
        self : CubistRegressor
            Fitted regressor.
        """
        X, y = check_X_y(X, y, dtype=np.float64, y_numeric=True)
        self.n_features_in_ = X.shape[1]

        # Store training stats
        self._y_mean = np.mean(y)
        self._y_std = np.std(y)
        self._y_min = np.min(y)
        self._y_max = np.max(y)

        # Try Rust backend
        if self.use_rust and HAS_RUST:
            seed = self.random_state if self.random_state is not None else 42
            self._rust_model = CubistRust(
                min_cases=self.min_cases,
                max_rules=self.max_rules,
                sample=self.sample,
                seed=seed,
                use_instance=(self.neighbors > 0),
                neighbors=max(1, self.neighbors),
                committees=self.committees,
                extrapolation=self.extrapolation,
                unbiased=self.unbiased,
            )
            self._rust_model.fit(X, y)
            self._use_rust_backend = True
            return self

        self._use_rust_backend = False

        # Pure Python fallback - simplified implementation
        # For full functionality, use the Rust backend
        from sklearn.linear_model import LinearRegression
        from sklearn.tree import DecisionTreeRegressor

        self._trees = []
        self._linear_models = []
        self._residuals = y.copy()

        for i in range(self.committees):
            # Build tree on current residuals
            tree = DecisionTreeRegressor(
                min_samples_leaf=self.min_cases,
                random_state=self.random_state,
            )
            tree.fit(X, self._residuals)

            # Get leaf assignments
            leaf_ids = tree.apply(X)
            unique_leaves = np.unique(leaf_ids)

            # Fit linear model for each leaf
            leaf_models = {}
            for leaf in unique_leaves:
                mask = leaf_ids == leaf
                if mask.sum() >= 2:
                    lr = LinearRegression()
                    lr.fit(X[mask], self._residuals[mask])
                    leaf_models[leaf] = lr
                else:
                    leaf_models[leaf] = None

            self._trees.append(tree)
            self._linear_models.append(leaf_models)

            # Update residuals for next committee member
            if i < self.committees - 1:
                pred = self._predict_single_tree(X, tree, leaf_models)
                self._residuals = self._residuals - pred

        # Store original y for instance-based correction
        if self.neighbors > 0:
            self._train_X = X.copy()
            self._train_y = y.copy()

        return self

    def _predict_single_tree(
        self,
        X: NDArray,
        tree: Any,
        leaf_models: dict[int, Any],
    ) -> NDArray:
        """Predict using a single tree and its linear models."""
        leaf_ids = tree.apply(X)
        predictions = np.zeros(len(X))

        for i, (sample, leaf) in enumerate(zip(X, leaf_ids)):
            if leaf in leaf_models and leaf_models[leaf] is not None:
                predictions[i] = leaf_models[leaf].predict(sample.reshape(1, -1))[0]
            else:
                # Use tree's leaf value
                predictions[i] = tree.predict(sample.reshape(1, -1))[0]

        return predictions

    def predict(self, X: ArrayLike) -> NDArray:
        """Predict target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self)
        X = check_array(X, dtype=np.float64)

        if self._use_rust_backend:
            return self._rust_model.predict(X)

        # Pure Python fallback
        predictions = np.zeros(len(X))

        for tree, leaf_models in zip(self._trees, self._linear_models):
            predictions += self._predict_single_tree(X, tree, leaf_models)

        # Apply instance-based correction
        if self.neighbors > 0:
            predictions = self._instance_correct(X, predictions)

        # Bound predictions
        range_val = self._y_max - self._y_min
        extra = range_val * self.extrapolation
        predictions = np.clip(predictions, self._y_min - extra, self._y_max + extra)

        return predictions

    def _instance_correct(self, X: NDArray, predictions: NDArray) -> NDArray:
        """Apply instance-based (k-NN) correction."""
        if not hasattr(self, '_train_X'):
            return predictions

        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(self.neighbors, len(self._train_X)))
        nn.fit(self._train_X)

        distances, indices = nn.kneighbors(X)

        corrected = predictions.copy()
        for i in range(len(X)):
            neighbor_idx = indices[i]
            neighbor_dist = distances[i]

            # Get training predictions for neighbors
            train_preds = np.zeros(len(neighbor_idx))
            for tree, leaf_models in zip(self._trees, self._linear_models):
                train_preds += self._predict_single_tree(
                    self._train_X[neighbor_idx], tree, leaf_models
                )

            # Compute corrections
            corrections = self._train_y[neighbor_idx] - train_preds

            # Weighted average by inverse distance
            if np.min(neighbor_dist) < 1e-10:
                # Exact match
                exact_idx = np.argmin(neighbor_dist)
                corrected[i] = self._train_y[neighbor_idx[exact_idx]]
            else:
                weights = 1.0 / neighbor_dist
                weights /= weights.sum()
                correction = np.sum(weights * corrections)
                corrected[i] = predictions[i] + correction

        return corrected

    @property
    def feature_importances_(self) -> NDArray:
        """Feature importances based on split usage."""
        check_is_fitted(self)

        if self._use_rust_backend:
            return self._rust_model.feature_importances()

        # Aggregate from trees
        importances = np.zeros(self.n_features_in_)
        for tree in self._trees:
            importances += tree.feature_importances_

        total = importances.sum()
        if total > 0:
            importances /= total
        else:
            importances = np.ones(self.n_features_in_) / self.n_features_in_

        return importances

    @property
    def n_rules_(self) -> int:
        """Number of rules in the model."""
        check_is_fitted(self)

        if self._use_rust_backend:
            return self._rust_model.n_rules()

        # Approximate by counting leaves
        return sum(tree.get_n_leaves() for tree in self._trees)

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            "committees": self.committees,
            "neighbors": self.neighbors,
            "min_cases": self.min_cases,
            "max_rules": self.max_rules,
            "sample": self.sample,
            "extrapolation": self.extrapolation,
            "unbiased": self.unbiased,
            "use_rust": self.use_rust,
            "random_state": self.random_state,
        }

    def set_params(self, **params) -> CubistRegressor:
        """Set parameters for this estimator."""
        for key, value in params.items():
            setattr(self, key, value)
        return self
