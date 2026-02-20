"""Quantile Regression Forest: Random Forest for conditional quantile estimation.

Quantile Regression Forests extend Random Forests to estimate any conditional
quantile of the response variable, not just the conditional mean. This enables:
- Prediction intervals (e.g., [10th, 90th] percentile)
- Uncertainty quantification
- Asymmetric loss optimization (pinball/quantile loss)

The key insight is that Random Forests implicitly estimate the full conditional
distribution P(Y|X) through the training samples that fall into each leaf.
By storing these samples (or weighted samples across trees), we can compute
arbitrary quantiles.

Algorithm:
1. Train a Random Forest as usual
2. For each tree, track which training samples fall into each leaf
3. At prediction time:
   a. For each test sample x, find its leaf in each tree
   b. Collect all training samples across all trees' leaves
   c. Compute the empirical quantile from this collection

References
----------
Meinshausen, N. (2006). "Quantile Regression Forests."
Journal of Machine Learning Research, 7, 983-999.
"""


import numpy as np
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _build_leaf_samples(tree: DecisionTreeRegressor, X: np.ndarray, y: np.ndarray) -> dict[int, np.ndarray]:
    """Build a mapping from leaf node IDs to the y values in that leaf.

    Parameters
    ----------
    tree : DecisionTreeRegressor
        A fitted decision tree.
    X : ndarray of shape (n_samples, n_features)
        Training data.
    y : ndarray of shape (n_samples,)
        Target values.

    Returns
    -------
    Dict[int, ndarray]
        Mapping from leaf node ID to array of y values in that leaf.
    """
    # Get leaf node for each training sample
    leaf_ids = tree.apply(X)

    # Build mapping from leaf ID to y values
    leaf_samples: dict[int, list[float]] = {}
    for leaf_id, y_val in zip(leaf_ids, y):
        if leaf_id not in leaf_samples:
            leaf_samples[leaf_id] = []
        leaf_samples[leaf_id].append(y_val)

    # Convert lists to arrays
    return {k: np.array(v) for k, v in leaf_samples.items()}


def _fit_single_tree(
    tree_idx: int,
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    bootstrap: bool,
    max_samples: int | float | None,
    base_seed: int,
    tree_params: dict,
    n_samples: int,
) -> tuple[DecisionTreeRegressor, dict[int, np.ndarray], np.ndarray]:
    """Fit a single tree and record leaf samples.

    Returns the fitted tree, leaf samples mapping, and bootstrap indices.
    """
    rng = np.random.RandomState(base_seed + tree_idx)

    # Determine bootstrap sample size
    if max_samples is None:
        n_samples_bootstrap = n_samples
    elif isinstance(max_samples, float):
        n_samples_bootstrap = max(1, int(n_samples * max_samples))
    else:
        n_samples_bootstrap = min(max_samples, n_samples)

    # Bootstrap sampling
    if bootstrap:
        indices = rng.choice(n_samples, size=n_samples_bootstrap, replace=True)
        X_sample = X[indices]
        y_sample = y[indices]
        if sample_weight is not None:
            weight_sample = sample_weight[indices]
        else:
            weight_sample = None
    else:
        indices = np.arange(n_samples)
        X_sample = X
        y_sample = y
        weight_sample = sample_weight

    # Create and fit tree
    tree = DecisionTreeRegressor(
        random_state=rng.randint(2**31),
        **tree_params
    )
    tree.fit(X_sample, y_sample, sample_weight=weight_sample)

    # Build leaf samples mapping using bootstrapped data
    leaf_samples = _build_leaf_samples(tree, X_sample, y_sample)

    return tree, leaf_samples, indices


class QuantileRegressorForest(BaseEstimator, RegressorMixin):
    """Random Forest for conditional quantile estimation.

    Quantile Regression Forests (QRF) estimate the full conditional distribution
    P(Y|X), allowing prediction of any quantile, not just the mean. This is
    essential for:
    - Prediction intervals with coverage guarantees
    - Uncertainty quantification in regression
    - Asymmetric loss functions (e.g., inventory optimization)

    The forest works by tracking which training samples end up in each leaf
    of each tree. At prediction time, for a test point x, we collect all
    training samples from the leaves that x falls into across all trees,
    then compute empirical quantiles from this collection.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the forest.

    quantiles : float or array-like of floats, default=0.5
        Quantile(s) to predict in [0, 1].
        - Single float: predict that quantile
        - Array: predict multiple quantiles simultaneously
        Default is 0.5 (median), which is more robust than mean.
        Common choices: [0.1, 0.5, 0.9] for prediction intervals.

    criterion : str, default='squared_error'
        Splitting criterion for trees:
        - 'squared_error': Mean squared error (standard)
        - 'absolute_error': Mean absolute error
        - 'friedman_mse': Improved MSE for gradient boosting
        - 'poisson': Poisson deviance

    max_depth : int, default=None
        Maximum depth of each tree. None means unlimited depth
        (nodes expand until all leaves are pure or contain
        fewer than min_samples_split samples).

    min_samples_split : int or float, default=2
        Minimum samples required to split an internal node.
        If float, fraction of n_samples.

    min_samples_leaf : int or float, default=1
        Minimum samples required at a leaf node.
        If float, fraction of n_samples.

    max_features : int, float, str, or None, default=1.0
        Number of features to consider for each split:
        - int: Use exactly max_features
        - float: Use max_features * n_features
        - 'sqrt': Use sqrt(n_features)
        - 'log2': Use log2(n_features)
        - None or 1.0: Use all features
        Note: For QRF, using all features is often preferred to
        get better leaf distributions.

    max_leaf_nodes : int, default=None
        Maximum number of leaf nodes per tree.

    min_impurity_decrease : float, default=0.0
        Minimum impurity decrease required for a split.

    bootstrap : bool, default=True
        Whether to use bootstrap sampling for each tree.

    max_samples : int or float, default=None
        Number of samples to draw for each tree (with replacement):
        - None: Draw n_samples samples
        - int: Draw max_samples samples
        - float: Draw max_samples * n_samples samples

    oob_score : bool, default=False
        Whether to compute out-of-bag score.
        Note: OOB for QRF uses median prediction for scoring.

    n_jobs : int, default=None
        Number of parallel jobs for fitting trees.
        None means 1, -1 means all processors.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    verbose : int, default=0
        Verbosity level for fitting progress.

    warm_start : bool, default=False
        If True, reuse previous fit and add more trees.

    Attributes
    ----------
    estimators_ : list of DecisionTreeRegressor
        The fitted tree estimators.

    leaf_samples_ : list of dict
        For each tree, mapping from leaf node ID to y values.

    n_features_in_ : int
        Number of features seen during fit.

    feature_importances_ : ndarray of shape (n_features_in_,)
        Impurity-based feature importances.

    oob_score_ : float
        Out-of-bag R² score (if oob_score=True).

    oob_prediction_ : ndarray of shape (n_samples,)
        OOB predictions (if oob_score=True).

    Examples
    --------
    >>> from endgame.models.trees import QuantileRegressorForest
    >>> from sklearn.datasets import make_regression
    >>> X, y = make_regression(n_samples=1000, n_features=10, random_state=42)
    >>>
    >>> # Predict median (more robust than mean)
    >>> qrf = QuantileRegressorForest(n_estimators=100, quantiles=0.5, random_state=42)
    >>> qrf.fit(X, y)
    >>> y_median = qrf.predict(X[:5])
    >>>
    >>> # Prediction intervals
    >>> qrf = QuantileRegressorForest(n_estimators=100, quantiles=[0.1, 0.5, 0.9])
    >>> qrf.fit(X, y)
    >>> intervals = qrf.predict(X[:5])  # Shape: (5, 3)
    >>> lower, median, upper = intervals[:, 0], intervals[:, 1], intervals[:, 2]
    >>>
    >>> # Change quantiles after fitting (no retraining needed!)
    >>> qrf.quantiles = [0.25, 0.75]
    >>> iqr_bounds = qrf.predict(X[:5])  # Shape: (5, 2)

    Notes
    -----
    QRF is particularly useful for:

    1. **Prediction Intervals**: Unlike standard RF which only gives point
       predictions, QRF can give valid prediction intervals by predicting
       e.g., [0.05, 0.95] quantiles for 90% coverage.

    2. **Heteroscedastic Data**: When variance of Y varies with X, QRF
       naturally captures this through different interval widths.

    3. **Conformal Prediction**: QRF quantiles can be calibrated using
       conformal methods for guaranteed coverage.

    4. **Asymmetric Loss**: For problems where over/under-prediction have
       different costs (inventory, load forecasting), predict the
       appropriate quantile that minimizes expected loss.

    Memory Usage: QRF stores all training y values at leaves, which uses
    more memory than standard RF. For very large datasets, consider
    using subsampling via max_samples parameter.

    References
    ----------
    Meinshausen, N. (2006). "Quantile Regression Forests."
    Journal of Machine Learning Research, 7, 983-999.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_estimators: int = 100,
        quantiles: float | list[float] = 0.5,
        criterion: str = "squared_error",
        max_depth: int | None = None,
        min_samples_split: int | float = 2,
        min_samples_leaf: int | float = 1,
        max_features: int | float | str | None = 1.0,
        max_leaf_nodes: int | None = None,
        min_impurity_decrease: float = 0.0,
        bootstrap: bool = True,
        max_samples: int | float | None = None,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | np.random.RandomState | None = None,
        verbose: int = 0,
        warm_start: bool = False,
    ):
        self.n_estimators = n_estimators
        self.quantiles = quantiles
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.bootstrap = bootstrap
        self.max_samples = max_samples
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.warm_start = warm_start

    def _get_quantiles_array(self) -> np.ndarray:
        """Convert quantiles parameter to array."""
        if isinstance(self.quantiles, (int, float)):
            return np.array([float(self.quantiles)])
        return np.array(self.quantiles, dtype=np.float64)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> "QuantileRegressorForest":
        """Build a quantile regression forest from training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights for fitting.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        y = y.astype(np.float64)
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Validate quantiles
        quantiles_arr = self._get_quantiles_array()
        if np.any(quantiles_arr < 0) or np.any(quantiles_arr > 1):
            raise ValueError("quantiles must be in [0, 1]")

        # Tree parameters
        tree_params = {
            "criterion": self.criterion,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_leaf_nodes": self.max_leaf_nodes,
            "min_impurity_decrease": self.min_impurity_decrease,
        }

        # Random state
        random_state = check_random_state(self.random_state)
        base_seed = random_state.randint(2**31)

        # Handle warm start
        if self.warm_start and hasattr(self, "estimators_"):
            n_existing = len(self.estimators_)
            if n_existing >= self.n_estimators:
                return self
            n_new = self.n_estimators - n_existing
        else:
            self.estimators_ = []
            self.leaf_samples_ = []
            self._bootstrap_indices = []
            n_existing = 0
            n_new = self.n_estimators

        # Store training data reference for prediction
        self._y_train = y.copy()

        # Fit trees in parallel
        if self.verbose > 0:
            print(f"Fitting {n_new} trees for Quantile Regression Forest...")

        results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_single_tree)(
                i + n_existing, X, y, sample_weight,
                self.bootstrap, self.max_samples, base_seed, tree_params, n_samples
            )
            for i in range(n_new)
        )

        # Collect results
        for tree, leaf_samples, indices in results:
            self.estimators_.append(tree)
            self.leaf_samples_.append(leaf_samples)
            self._bootstrap_indices.append(indices)

        # Compute OOB score if requested
        if self.oob_score:
            self._compute_oob_score(X, y)

        # Compute feature importances
        self._compute_feature_importances()

        return self

    def _compute_oob_score(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute out-of-bag R² score using median prediction."""
        n_samples = X.shape[0]

        # For each sample, collect y values from trees where it was OOB
        oob_collections: list[list[float]] = [[] for _ in range(n_samples)]

        for tree, leaf_samples, indices in zip(
            self.estimators_, self.leaf_samples_, self._bootstrap_indices
        ):
            # Find OOB samples
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[indices] = False
            oob_indices = np.where(oob_mask)[0]

            if len(oob_indices) == 0:
                continue

            # Get leaf IDs for OOB samples
            leaf_ids = tree.apply(X[oob_indices])

            # Collect y values for each OOB sample
            for i, leaf_id in enumerate(leaf_ids):
                if leaf_id in leaf_samples:
                    oob_collections[oob_indices[i]].extend(leaf_samples[leaf_id].tolist())

        # Compute median predictions
        oob_prediction = np.full(n_samples, np.nan)
        for i, collection in enumerate(oob_collections):
            if len(collection) > 0:
                oob_prediction[i] = np.median(collection)

        self.oob_prediction_ = oob_prediction

        # Compute R² only for samples with predictions
        valid_mask = ~np.isnan(oob_prediction)
        if np.sum(valid_mask) == 0:
            self.oob_score_ = 0.0
            return

        y_valid = y[valid_mask]
        pred_valid = oob_prediction[valid_mask]
        ss_res = np.sum((y_valid - pred_valid) ** 2)
        ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
        self.oob_score_ = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    def _compute_feature_importances(self) -> None:
        """Compute feature importances from all trees."""
        importances = np.zeros(self.n_features_in_)

        for tree in self.estimators_:
            importances += tree.feature_importances_

        importances = importances / len(self.estimators_)
        total = np.sum(importances)
        if total > 0:
            importances = importances / total

        self.feature_importances_ = importances

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict quantile(s) for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray
            Predicted quantile values.
            - If single quantile: shape (n_samples,)
            - If multiple quantiles: shape (n_samples, n_quantiles)
        """
        check_is_fitted(self, ["estimators_", "leaf_samples_"])
        X = check_array(X)

        quantiles_arr = self._get_quantiles_array()
        n_samples = X.shape[0]
        n_quantiles = len(quantiles_arr)

        # Collect y values for each sample across all trees
        predictions = np.zeros((n_samples, n_quantiles))

        for i in range(n_samples):
            # Collect all y values from leaves across trees
            y_collection = []

            for tree, leaf_samples in zip(self.estimators_, self.leaf_samples_):
                leaf_id = tree.apply(X[i:i+1])[0]
                if leaf_id in leaf_samples:
                    y_collection.extend(leaf_samples[leaf_id].tolist())

            # Compute quantiles
            if len(y_collection) > 0:
                y_arr = np.array(y_collection)
                predictions[i] = np.percentile(y_arr, quantiles_arr * 100)
            else:
                # Fallback: use training mean
                predictions[i] = np.mean(self._y_train)

        # Return 1D if single quantile
        if n_quantiles == 1:
            return predictions[:, 0]
        return predictions

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: float | list[float],
    ) -> np.ndarray:
        """Predict specific quantiles without changing the estimator.

        This allows predicting different quantiles from what was specified
        at construction, without modifying the estimator's state.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        quantiles : float or array-like of floats
            Quantile(s) to predict in [0, 1].

        Returns
        -------
        y_pred : ndarray
            Predicted quantile values.
            - If single quantile: shape (n_samples,)
            - If multiple quantiles: shape (n_samples, n_quantiles)
        """
        check_is_fitted(self, ["estimators_", "leaf_samples_"])
        X = check_array(X)

        if isinstance(quantiles, (int, float)):
            quantiles_arr = np.array([float(quantiles)])
        else:
            quantiles_arr = np.array(quantiles, dtype=np.float64)

        if np.any(quantiles_arr < 0) or np.any(quantiles_arr > 1):
            raise ValueError("quantiles must be in [0, 1]")

        n_samples = X.shape[0]
        n_quantiles = len(quantiles_arr)
        predictions = np.zeros((n_samples, n_quantiles))

        for i in range(n_samples):
            y_collection = []

            for tree, leaf_samples in zip(self.estimators_, self.leaf_samples_):
                leaf_id = tree.apply(X[i:i+1])[0]
                if leaf_id in leaf_samples:
                    y_collection.extend(leaf_samples[leaf_id].tolist())

            if len(y_collection) > 0:
                y_arr = np.array(y_collection)
                predictions[i] = np.percentile(y_arr, quantiles_arr * 100)
            else:
                predictions[i] = np.mean(self._y_train)

        if n_quantiles == 1:
            return predictions[:, 0]
        return predictions

    def predict_interval(
        self,
        X: np.ndarray,
        coverage: float = 0.9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict symmetric prediction interval.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        coverage : float, default=0.9
            Desired coverage probability in (0, 1).
            E.g., 0.9 gives [5th, 95th] percentile interval.

        Returns
        -------
        lower : ndarray of shape (n_samples,)
            Lower bound of prediction interval.
        upper : ndarray of shape (n_samples,)
            Upper bound of prediction interval.
        """
        if not 0 < coverage < 1:
            raise ValueError("coverage must be in (0, 1)")

        alpha = (1 - coverage) / 2
        quantiles = [alpha, 1 - alpha]
        predictions = self.predict_quantiles(X, quantiles)

        return predictions[:, 0], predictions[:, 1]

    def predict_mean(self, X: np.ndarray) -> np.ndarray:
        """Predict conditional mean (like standard Random Forest).

        This collects all y values from relevant leaves and returns
        their mean, equivalent to standard RF prediction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean values.
        """
        check_is_fitted(self, ["estimators_", "leaf_samples_"])
        X = check_array(X)

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            y_collection = []

            for tree, leaf_samples in zip(self.estimators_, self.leaf_samples_):
                leaf_id = tree.apply(X[i:i+1])[0]
                if leaf_id in leaf_samples:
                    y_collection.extend(leaf_samples[leaf_id].tolist())

            if len(y_collection) > 0:
                predictions[i] = np.mean(y_collection)
            else:
                predictions[i] = np.mean(self._y_train)

        return predictions

    def predict_std(self, X: np.ndarray) -> np.ndarray:
        """Predict conditional standard deviation (uncertainty).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_std : ndarray of shape (n_samples,)
            Predicted standard deviation for each sample.
        """
        check_is_fitted(self, ["estimators_", "leaf_samples_"])
        X = check_array(X)

        n_samples = X.shape[0]
        stds = np.zeros(n_samples)

        for i in range(n_samples):
            y_collection = []

            for tree, leaf_samples in zip(self.estimators_, self.leaf_samples_):
                leaf_id = tree.apply(X[i:i+1])[0]
                if leaf_id in leaf_samples:
                    y_collection.extend(leaf_samples[leaf_id].tolist())

            if len(y_collection) > 1:
                stds[i] = np.std(y_collection, ddof=1)
            elif len(y_collection) == 1:
                stds[i] = 0.0
            else:
                stds[i] = np.std(self._y_train)

        return stds

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Apply trees to X, return leaf indices.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        X_leaves : ndarray of shape (n_samples, n_estimators)
            Leaf indices for each sample in each tree.
        """
        check_is_fitted(self, ["estimators_"])
        X = check_array(X)

        n_samples = X.shape[0]
        leaves = np.zeros((n_samples, len(self.estimators_)), dtype=np.int64)

        for i, tree in enumerate(self.estimators_):
            leaves[:, i] = tree.apply(X)

        return leaves

    @property
    def n_estimators_(self) -> int:
        """Number of fitted estimators."""
        return len(self.estimators_) if hasattr(self, "estimators_") else 0


def pinball_loss(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    quantile: float,
) -> float:
    """Compute pinball (quantile) loss.

    The pinball loss is the proper scoring rule for quantile regression:
    L(y, q) = (1-alpha) * max(q-y, 0) + alpha * max(y-q, 0)

    where alpha is the quantile.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True values.
    y_pred : ndarray of shape (n_samples,)
        Predicted quantile values.
    quantile : float
        Quantile being predicted, in [0, 1].

    Returns
    -------
    float
        Mean pinball loss.

    Examples
    --------
    >>> y_true = np.array([1, 2, 3, 4, 5])
    >>> y_pred = np.array([1.5, 2.0, 2.5, 4.0, 4.5])
    >>> pinball_loss(y_true, y_pred, 0.5)  # Median loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    errors = y_true - y_pred
    loss = np.where(
        errors >= 0,
        quantile * errors,
        (quantile - 1) * errors
    )

    return float(np.mean(loss))


def interval_coverage(
    y_true: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute empirical coverage of prediction intervals.

    Parameters
    ----------
    y_true : ndarray of shape (n_samples,)
        True values.
    lower : ndarray of shape (n_samples,)
        Lower bounds of intervals.
    upper : ndarray of shape (n_samples,)
        Upper bounds of intervals.

    Returns
    -------
    float
        Fraction of y_true values within [lower, upper].
    """
    y_true = np.asarray(y_true)
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    within = (y_true >= lower) & (y_true <= upper)
    return float(np.mean(within))


def interval_width(
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    """Compute mean width of prediction intervals.

    Parameters
    ----------
    lower : ndarray of shape (n_samples,)
        Lower bounds of intervals.
    upper : ndarray of shape (n_samples,)
        Upper bounds of intervals.

    Returns
    -------
    float
        Mean interval width.
    """
    lower = np.asarray(lower)
    upper = np.asarray(upper)

    return float(np.mean(upper - lower))
