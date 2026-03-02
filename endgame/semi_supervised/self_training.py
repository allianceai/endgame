from __future__ import annotations

"""Self-Training: Semi-supervised learning via iterative pseudo-labeling.

Self-training is a simple but effective semi-supervised learning technique
that iteratively:
1. Trains a model on labeled data
2. Predicts on unlabeled data
3. Adds high-confidence predictions as pseudo-labels
4. Retrains on the expanded dataset

This wrapper works with any sklearn-compatible estimator and provides
flexible selection strategies for choosing which predictions to trust.

References
----------
Yarowsky, D. (1995). "Unsupervised Word Sense Disambiguation Rivaling
Supervised Methods." ACL.

Zhu, X. & Goldberg, A.B. (2009). "Introduction to Semi-Supervised Learning."
Synthesis Lectures on AI and ML.

sklearn.semi_supervised.SelfTrainingClassifier (scikit-learn reference)
"""

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, MetaEstimatorMixin, RegressorMixin, clone
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

SelectionCriterion = Literal["threshold", "k_best"]


class SelfTrainingClassifier(BaseEstimator, ClassifierMixin, MetaEstimatorMixin):
    """Self-training classifier for semi-supervised learning.

    Wraps any sklearn-compatible classifier to perform iterative pseudo-labeling.
    The algorithm repeatedly:
    1. Trains on labeled + pseudo-labeled data
    2. Predicts on remaining unlabeled data
    3. Selects high-confidence predictions as new pseudo-labels
    4. Repeats until convergence or max iterations

    Parameters
    ----------
    base_estimator : estimator object
        Any sklearn-compatible classifier with `fit`, `predict`, and
        `predict_proba` methods. Will be cloned for each iteration.

    criterion : {'threshold', 'k_best'}, default='threshold'
        Selection strategy for pseudo-labels:
        - 'threshold': Select samples with confidence >= threshold
        - 'k_best': Select top k most confident samples per iteration

    threshold : float, default=0.75
        Minimum confidence (max probability) required to add a pseudo-label.
        Only used when criterion='threshold'.

    k_best : int, default=10
        Number of samples to pseudo-label per iteration.
        Only used when criterion='k_best'.

    max_iter : int, default=10
        Maximum number of self-training iterations. Set to None for
        unlimited iterations (until no more samples meet the criterion).

    sample_weight_decay : float, default=1.0
        Weight multiplier for pseudo-labeled samples relative to true labels.
        - 1.0: Equal weight to pseudo-labels and true labels
        - < 1.0: Lower weight for pseudo-labels (more conservative)
        Values < 1 recommended when noise in pseudo-labels is a concern.

    progressive_weight : bool, default=False
        If True, weight pseudo-labels by their confidence score.
        Overrides sample_weight_decay for pseudo-labeled samples.

    min_confidence_increase : float, default=0.0
        Minimum increase in average confidence required to continue.
        Helps detect when self-training has converged.

    verbose : bool, default=False
        Print progress information during training.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility (used in k_best tie-breaking).

    Attributes
    ----------
    base_estimator_ : estimator
        The fitted base estimator.

    classes_ : ndarray of shape (n_classes,)
        Class labels.

    n_classes_ : int
        Number of classes.

    n_features_in_ : int
        Number of features seen during fit.

    n_iter_ : int
        Number of self-training iterations performed.

    labeled_iter_ : ndarray of shape (n_samples,)
        Iteration when each sample was labeled:
        - 0: Originally labeled
        - i > 0: Pseudo-labeled in iteration i
        - -1: Never labeled (still unlabeled)

    pseudo_labels_ : ndarray of shape (n_samples,)
        Final labels for all samples (true labels + pseudo-labels).

    transduction_ : ndarray of shape (n_samples,)
        Same as pseudo_labels_ (sklearn compatibility).

    termination_condition_ : str
        Reason for stopping: 'max_iter', 'no_change', 'all_labeled',
        or 'confidence_plateau'.

    history_ : dict
        Training history with keys:
        - 'n_pseudo_labeled': List of cumulative pseudo-labeled counts
        - 'mean_confidence': List of mean confidence per iteration
        - 'selected_per_iter': List of samples selected per iteration

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from endgame.semi_supervised import SelfTrainingClassifier
    >>>
    >>> # Prepare data: -1 indicates unlabeled samples
    >>> y_train = np.array([0, 1, 0, -1, -1, -1, 1, -1])
    >>>
    >>> # Create self-training classifier
    >>> st = SelfTrainingClassifier(
    ...     base_estimator=RandomForestClassifier(n_estimators=100),
    ...     threshold=0.8,
    ...     max_iter=10,
    ... )
    >>> st.fit(X_train, y_train)
    >>>
    >>> # Predict on new data
    >>> predictions = st.predict(X_test)
    >>> probabilities = st.predict_proba(X_test)
    >>>
    >>> # Check which samples were pseudo-labeled
    >>> print(f"Pseudo-labeled in iter 1: {np.sum(st.labeled_iter_ == 1)}")

    Notes
    -----
    **Choosing threshold vs k_best:**

    - `threshold` is preferred when you have a good sense of model calibration.
      It naturally adapts the number of samples based on confidence.

    - `k_best` is preferred for controlled expansion. It guarantees progress
      each iteration but may add low-confidence samples if k is too large.

    **Avoiding confirmation bias:**

    Self-training can reinforce the model's mistakes (confirmation bias).
    To mitigate this:
    - Use a high threshold (0.9+)
    - Use sample_weight_decay < 1.0 to trust pseudo-labels less
    - Set min_confidence_increase > 0 to detect plateaus
    - Consider using progressive_weight=True

    **Memory efficiency:**

    The wrapper stores labeled_iter_ for all samples. For very large
    unlabeled sets, consider batching the unlabeled data.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        base_estimator: BaseEstimator,
        criterion: SelectionCriterion = "threshold",
        threshold: float = 0.75,
        k_best: int = 10,
        max_iter: int | None = 10,
        sample_weight_decay: float = 1.0,
        progressive_weight: bool = False,
        min_confidence_increase: float = 0.0,
        verbose: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.max_iter = max_iter
        self.sample_weight_decay = sample_weight_decay
        self.progressive_weight = progressive_weight
        self.min_confidence_increase = min_confidence_increase
        self.verbose = verbose
        self.random_state = random_state

    def fit(self, X, y, **fit_params) -> SelfTrainingClassifier:
        """Fit the self-training classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (labeled + unlabeled).
        y : array-like of shape (n_samples,)
            Target values. Use -1 to indicate unlabeled samples.
        **fit_params : dict
            Additional parameters passed to base_estimator.fit().
            Note: sample_weight is handled internally.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Validate inputs
        X, y = check_X_y(X, y)

        if self.criterion not in ("threshold", "k_best"):
            raise ValueError(f"criterion must be 'threshold' or 'k_best', got {self.criterion}")

        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(f"threshold must be in [0, 1], got {self.threshold}")

        if self.k_best < 1:
            raise ValueError(f"k_best must be >= 1, got {self.k_best}")

        self.n_features_in_ = X.shape[1]

        # Initialize random state
        rng = check_random_state(self.random_state)

        # Identify labeled and unlabeled samples
        has_label = y != -1

        if not np.any(has_label):
            raise ValueError("y must contain at least one labeled sample (not -1)")

        # Get classes from labeled samples
        self.classes_ = np.unique(y[has_label])
        self.n_classes_ = len(self.classes_)

        # Initialize tracking arrays
        n_samples = len(y)
        self.labeled_iter_ = np.full(n_samples, -1, dtype=int)
        self.labeled_iter_[has_label] = 0  # Original labels marked as iteration 0

        # Copy y to track pseudo-labels
        self.pseudo_labels_ = y.copy()

        # Sample weights
        sample_weights = np.ones(n_samples)
        confidence_scores = np.zeros(n_samples)  # For progressive weighting

        # History tracking
        self.history_ = {
            "n_pseudo_labeled": [],
            "mean_confidence": [],
            "selected_per_iter": [],
        }

        prev_mean_confidence = 0.0
        self.n_iter_ = 0
        self.termination_condition_ = "max_iter"

        # Self-training loop
        while True:
            self.n_iter_ += 1

            if self.verbose:
                n_labeled = np.sum(self.labeled_iter_ >= 0)
                n_unlabeled = np.sum(self.labeled_iter_ == -1)
                print(f"Iteration {self.n_iter_}: {n_labeled} labeled, {n_unlabeled} unlabeled")

            # Get currently labeled samples
            labeled_mask = self.labeled_iter_ >= 0
            X_labeled = X[labeled_mask]
            y_labeled = self.pseudo_labels_[labeled_mask]

            # Compute sample weights for training
            if self.progressive_weight:
                weights_labeled = np.where(
                    self.labeled_iter_[labeled_mask] == 0,
                    1.0,  # Original labels get full weight
                    confidence_scores[labeled_mask]  # Pseudo-labels weighted by confidence
                )
            else:
                weights_labeled = np.where(
                    self.labeled_iter_[labeled_mask] == 0,
                    1.0,
                    self.sample_weight_decay
                )

            # Train base estimator
            self.base_estimator_ = clone(self.base_estimator)

            # Try to pass sample_weight if supported
            try:
                self.base_estimator_.fit(X_labeled, y_labeled, sample_weight=weights_labeled, **fit_params)
            except TypeError:
                # Estimator doesn't support sample_weight
                self.base_estimator_.fit(X_labeled, y_labeled, **fit_params)

            # Check stopping conditions
            unlabeled_mask = self.labeled_iter_ == -1
            n_unlabeled = np.sum(unlabeled_mask)

            if n_unlabeled == 0:
                self.termination_condition_ = "all_labeled"
                if self.verbose:
                    print("Stopping: All samples labeled")
                break

            if self.max_iter is not None and self.n_iter_ >= self.max_iter:
                self.termination_condition_ = "max_iter"
                if self.verbose:
                    print(f"Stopping: Reached max_iter={self.max_iter}")
                break

            # Predict on unlabeled samples
            X_unlabeled = X[unlabeled_mask]
            proba = self.base_estimator_.predict_proba(X_unlabeled)

            # Get max confidence and predicted class for each unlabeled sample
            max_proba = proba.max(axis=1)
            pred_classes = self.classes_[proba.argmax(axis=1)]

            # Select samples to pseudo-label
            if self.criterion == "threshold":
                selected_mask = max_proba >= self.threshold
            else:  # k_best
                n_select = min(self.k_best, n_unlabeled)
                if n_select == 0:
                    selected_mask = np.zeros(n_unlabeled, dtype=bool)
                else:
                    # Get indices of top k
                    # Add small random noise for tie-breaking
                    max_proba_noisy = max_proba + rng.uniform(0, 1e-10, size=len(max_proba))
                    top_k_idx = np.argpartition(-max_proba_noisy, n_select - 1)[:n_select]
                    selected_mask = np.zeros(n_unlabeled, dtype=bool)
                    selected_mask[top_k_idx] = True

            n_selected = np.sum(selected_mask)

            if n_selected == 0:
                self.termination_condition_ = "no_change"
                if self.verbose:
                    print("Stopping: No samples met selection criterion")
                break

            # Check confidence plateau
            mean_confidence = np.mean(max_proba[selected_mask])
            self.history_["mean_confidence"].append(mean_confidence)
            self.history_["selected_per_iter"].append(n_selected)

            if self.min_confidence_increase > 0:
                if mean_confidence - prev_mean_confidence < self.min_confidence_increase:
                    self.termination_condition_ = "confidence_plateau"
                    if self.verbose:
                        print("Stopping: Confidence increase below threshold")
                    break
            prev_mean_confidence = mean_confidence

            # Update labels
            unlabeled_indices = np.where(unlabeled_mask)[0]
            selected_indices = unlabeled_indices[selected_mask]

            self.pseudo_labels_[selected_indices] = pred_classes[selected_mask]
            self.labeled_iter_[selected_indices] = self.n_iter_
            confidence_scores[selected_indices] = max_proba[selected_mask]

            self.history_["n_pseudo_labeled"].append(np.sum(self.labeled_iter_ > 0))

            if self.verbose:
                print(f"  Selected {n_selected} samples (mean conf: {mean_confidence:.3f})")

        # Final model is already trained
        self.transduction_ = self.pseudo_labels_.copy()

        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self, ["base_estimator_", "classes_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        check_is_fitted(self, ["base_estimator_", "classes_"])
        X = check_array(X)
        return self.base_estimator_.predict_proba(X)

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict class log-probabilities for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Class log-probabilities.
        """
        check_is_fitted(self, ["base_estimator_", "classes_"])
        X = check_array(X)

        if hasattr(self.base_estimator_, "predict_log_proba"):
            return self.base_estimator_.predict_log_proba(X)
        return np.log(self.predict_proba(X) + 1e-10)

    def decision_function(self, X) -> np.ndarray:
        """Compute decision function for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples.

        Returns
        -------
        decision : ndarray
            Decision function values.
        """
        check_is_fitted(self, ["base_estimator_"])
        X = check_array(X)

        if hasattr(self.base_estimator_, "decision_function"):
            return self.base_estimator_.decision_function(X)

        # Fall back to log-probability difference for binary classification
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return np.log(proba[:, 1] + 1e-10) - np.log(proba[:, 0] + 1e-10)
        return proba

    def get_pseudo_labeled_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get indices, labels, and iterations of pseudo-labeled samples.

        Returns
        -------
        indices : ndarray
            Indices of pseudo-labeled samples.
        labels : ndarray
            Pseudo-labels assigned.
        iterations : ndarray
            Iteration when each sample was pseudo-labeled.
        """
        check_is_fitted(self, ["labeled_iter_", "pseudo_labels_"])

        pseudo_mask = self.labeled_iter_ > 0
        indices = np.where(pseudo_mask)[0]
        labels = self.pseudo_labels_[pseudo_mask]
        iterations = self.labeled_iter_[pseudo_mask]

        return indices, labels, iterations


class SelfTrainingRegressor(BaseEstimator, RegressorMixin, MetaEstimatorMixin):
    """Self-training regressor for semi-supervised learning.

    Extends self-training to regression by using prediction uncertainty
    instead of class probabilities for sample selection.

    The uncertainty can be estimated via:
    - Ensemble variance (if base_estimator is an ensemble)
    - Quantile predictions (if supported)
    - Residual-based heuristics

    Parameters
    ----------
    base_estimator : estimator object
        Any sklearn-compatible regressor with `fit` and `predict` methods.
        For best results, use an estimator that can provide uncertainty
        estimates (e.g., RandomForestRegressor, GradientBoostingRegressor,
        QuantileRegressorForest).

    criterion : {'threshold', 'k_best'}, default='threshold'
        Selection strategy:
        - 'threshold': Select samples with uncertainty <= threshold
        - 'k_best': Select k samples with lowest uncertainty

    threshold : float, default=1.0
        Maximum uncertainty (std dev) allowed for pseudo-labeling.
        Only used when criterion='threshold'.

    k_best : int, default=10
        Number of samples to pseudo-label per iteration.
        Only used when criterion='k_best'.

    uncertainty_method : {'ensemble', 'knn', 'residual'}, default='ensemble'
        Method for estimating prediction uncertainty:
        - 'ensemble': Use variance across ensemble members (requires
          ensemble with estimators_ attribute, e.g., RandomForest)
        - 'knn': Use variance among k nearest labeled neighbors
        - 'residual': Use cross-validated residual magnitude

    max_iter : int, default=10
        Maximum number of self-training iterations.

    sample_weight_decay : float, default=1.0
        Weight multiplier for pseudo-labeled samples.

    verbose : bool, default=False
        Print progress information.

    random_state : int, RandomState, or None, default=None
        Random seed.

    Attributes
    ----------
    base_estimator_ : estimator
        The fitted base estimator.

    n_features_in_ : int
        Number of features.

    n_iter_ : int
        Number of iterations performed.

    labeled_iter_ : ndarray of shape (n_samples,)
        Iteration when each sample was labeled (0=original, -1=unlabeled).

    pseudo_labels_ : ndarray of shape (n_samples,)
        Final labels including pseudo-labels.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from endgame.semi_supervised import SelfTrainingRegressor
    >>>
    >>> # Prepare data: np.nan indicates unlabeled samples
    >>> y_train = np.array([1.0, 2.5, 3.0, np.nan, np.nan, np.nan])
    >>>
    >>> st = SelfTrainingRegressor(
    ...     base_estimator=RandomForestRegressor(n_estimators=100),
    ...     threshold=0.5,  # Max std dev for pseudo-labeling
    ... )
    >>> st.fit(X_train, y_train)
    >>> predictions = st.predict(X_test)
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        base_estimator: BaseEstimator,
        criterion: SelectionCriterion = "threshold",
        threshold: float = 1.0,
        k_best: int = 10,
        uncertainty_method: Literal["ensemble", "knn", "residual"] = "ensemble",
        max_iter: int | None = 10,
        sample_weight_decay: float = 1.0,
        verbose: bool = False,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.base_estimator = base_estimator
        self.criterion = criterion
        self.threshold = threshold
        self.k_best = k_best
        self.uncertainty_method = uncertainty_method
        self.max_iter = max_iter
        self.sample_weight_decay = sample_weight_decay
        self.verbose = verbose
        self.random_state = random_state

    def _estimate_uncertainty_ensemble(
        self,
        X: np.ndarray,
        estimator: BaseEstimator,
    ) -> np.ndarray:
        """Estimate uncertainty using ensemble variance."""
        if not hasattr(estimator, "estimators_"):
            raise ValueError(
                "uncertainty_method='ensemble' requires an ensemble estimator "
                "with estimators_ attribute (e.g., RandomForestRegressor)"
            )

        # Get predictions from each tree/estimator
        predictions = np.array([
            tree.predict(X) for tree in estimator.estimators_
        ])

        # Return std across estimators
        return np.std(predictions, axis=0)

    def _estimate_uncertainty_knn(
        self,
        X_unlabeled: np.ndarray,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
        k: int = 5,
    ) -> np.ndarray:
        """Estimate uncertainty using k-nearest neighbor variance."""
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(k, len(X_labeled)))
        nn.fit(X_labeled)

        distances, indices = nn.kneighbors(X_unlabeled)

        # Variance of y values among nearest neighbors
        neighbor_y = y_labeled[indices]
        return np.std(neighbor_y, axis=1)

    def _estimate_uncertainty_residual(
        self,
        X: np.ndarray,
        estimator: BaseEstimator,
        X_labeled: np.ndarray,
        y_labeled: np.ndarray,
    ) -> np.ndarray:
        """Estimate uncertainty using training residuals."""
        # Get predictions on labeled data
        y_pred_labeled = estimator.predict(X_labeled)
        residuals = np.abs(y_labeled - y_pred_labeled)

        # Use median absolute residual as base uncertainty
        base_uncertainty = np.median(residuals)

        # Scale by distance from training data (simple heuristic)
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(5, len(X_labeled)))
        nn.fit(X_labeled)

        distances, _ = nn.kneighbors(X)
        mean_distances = np.mean(distances, axis=1)

        # Normalize distances
        max_dist = np.max(mean_distances) + 1e-10
        normalized_dist = mean_distances / max_dist

        # Uncertainty increases with distance
        return base_uncertainty * (1 + normalized_dist)

    def fit(self, X, y, **fit_params) -> SelfTrainingRegressor:
        """Fit the self-training regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data (labeled + unlabeled).
        y : array-like of shape (n_samples,)
            Target values. Use np.nan to indicate unlabeled samples.
        **fit_params : dict
            Additional parameters passed to base_estimator.fit().

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        y = np.asarray(y, dtype=np.float64)

        if len(y) != X.shape[0]:
            raise ValueError(f"X and y have inconsistent lengths: {X.shape[0]} vs {len(y)}")

        self.n_features_in_ = X.shape[1]

        rng = check_random_state(self.random_state)

        # Identify labeled and unlabeled
        has_label = ~np.isnan(y)

        if not np.any(has_label):
            raise ValueError("y must contain at least one labeled sample (not nan)")

        # Initialize tracking
        n_samples = len(y)
        self.labeled_iter_ = np.full(n_samples, -1, dtype=int)
        self.labeled_iter_[has_label] = 0

        self.pseudo_labels_ = y.copy()

        sample_weights = np.ones(n_samples)

        self.n_iter_ = 0
        self.termination_condition_ = "max_iter"

        while True:
            self.n_iter_ += 1

            if self.verbose:
                n_labeled = np.sum(self.labeled_iter_ >= 0)
                n_unlabeled = np.sum(self.labeled_iter_ == -1)
                print(f"Iteration {self.n_iter_}: {n_labeled} labeled, {n_unlabeled} unlabeled")

            # Get labeled data
            labeled_mask = self.labeled_iter_ >= 0
            X_labeled = X[labeled_mask]
            y_labeled = self.pseudo_labels_[labeled_mask]

            # Weights
            weights_labeled = np.where(
                self.labeled_iter_[labeled_mask] == 0,
                1.0,
                self.sample_weight_decay
            )

            # Train
            self.base_estimator_ = clone(self.base_estimator)
            try:
                self.base_estimator_.fit(X_labeled, y_labeled, sample_weight=weights_labeled, **fit_params)
            except TypeError:
                self.base_estimator_.fit(X_labeled, y_labeled, **fit_params)

            # Check stopping
            unlabeled_mask = self.labeled_iter_ == -1
            n_unlabeled = np.sum(unlabeled_mask)

            if n_unlabeled == 0:
                self.termination_condition_ = "all_labeled"
                if self.verbose:
                    print("Stopping: All samples labeled")
                break

            if self.max_iter is not None and self.n_iter_ >= self.max_iter:
                self.termination_condition_ = "max_iter"
                if self.verbose:
                    print(f"Stopping: Reached max_iter={self.max_iter}")
                break

            # Predict and estimate uncertainty
            X_unlabeled = X[unlabeled_mask]
            predictions = self.base_estimator_.predict(X_unlabeled)

            # Estimate uncertainty
            if self.uncertainty_method == "ensemble":
                uncertainty = self._estimate_uncertainty_ensemble(X_unlabeled, self.base_estimator_)
            elif self.uncertainty_method == "knn":
                uncertainty = self._estimate_uncertainty_knn(X_unlabeled, X_labeled, y_labeled)
            else:  # residual
                uncertainty = self._estimate_uncertainty_residual(
                    X_unlabeled, self.base_estimator_, X_labeled, y_labeled
                )

            # Select samples (low uncertainty = high confidence)
            if self.criterion == "threshold":
                selected_mask = uncertainty <= self.threshold
            else:  # k_best
                n_select = min(self.k_best, n_unlabeled)
                if n_select == 0:
                    selected_mask = np.zeros(n_unlabeled, dtype=bool)
                else:
                    # Select lowest uncertainty
                    uncertainty_noisy = uncertainty + rng.uniform(0, 1e-10, size=len(uncertainty))
                    top_k_idx = np.argpartition(uncertainty_noisy, n_select - 1)[:n_select]
                    selected_mask = np.zeros(n_unlabeled, dtype=bool)
                    selected_mask[top_k_idx] = True

            n_selected = np.sum(selected_mask)

            if n_selected == 0:
                self.termination_condition_ = "no_change"
                if self.verbose:
                    print("Stopping: No samples met selection criterion")
                break

            # Update labels
            unlabeled_indices = np.where(unlabeled_mask)[0]
            selected_indices = unlabeled_indices[selected_mask]

            self.pseudo_labels_[selected_indices] = predictions[selected_mask]
            self.labeled_iter_[selected_indices] = self.n_iter_

            if self.verbose:
                mean_unc = np.mean(uncertainty[selected_mask])
                print(f"  Selected {n_selected} samples (mean uncertainty: {mean_unc:.3f})")

        return self

    def predict(self, X) -> np.ndarray:
        """Predict target values for samples in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["base_estimator_"])
        X = check_array(X)
        return self.base_estimator_.predict(X)

    def get_pseudo_labeled_samples(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get indices, labels, and iterations of pseudo-labeled samples.

        Returns
        -------
        indices : ndarray
            Indices of pseudo-labeled samples.
        labels : ndarray
            Pseudo-labels assigned.
        iterations : ndarray
            Iteration when each sample was pseudo-labeled.
        """
        check_is_fitted(self, ["labeled_iter_", "pseudo_labels_"])

        pseudo_mask = self.labeled_iter_ > 0
        indices = np.where(pseudo_mask)[0]
        labels = self.pseudo_labels_[pseudo_mask]
        iterations = self.labeled_iter_[pseudo_mask]

        return indices, labels, iterations
