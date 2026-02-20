"""Label Noise Detection for data cleaning.

Implements Confident Learning and ensemble-based approaches to identify
mislabeled training examples.

References
----------
- Northcutt et al., 2021 - "Confident Learning: Estimating Uncertainty in
  Dataset Labels" (JAIR)

Example
-------
>>> from endgame.preprocessing import ConfidentLearningFilter
>>> clf = ConfidentLearningFilter(base_estimator='xgboost')
>>> noise_mask = clf.fit_detect(X, y)
>>> X_clean, y_clean = X[~noise_mask], y[~noise_mask]
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict


class ConfidentLearningFilter(BaseEstimator):
    """Identify mislabeled examples using Confident Learning.

    Uses cross-validated predicted probabilities to estimate the joint
    distribution of noisy and true labels, then identifies examples
    that are likely mislabeled.

    Parameters
    ----------
    base_estimator : estimator or str, default='rf'
        Classifier to use for cross-validated probability estimation.
        Can be 'rf' (RandomForest), 'xgboost', 'lgbm', or any
        sklearn-compatible classifier with predict_proba.
    cv : int, default=5
        Number of cross-validation folds for probability estimation.
    threshold : float or str, default='auto'
        Confidence threshold for identifying noise. If 'auto', uses
        per-class average predicted probability as threshold.
        If float, uses the same threshold for all classes.
    method : str, default='prune_by_class'
        Method for identifying noisy labels:
        - 'prune_by_class': Remove examples with low self-confidence
        - 'prune_by_noise_rate': Remove based on estimated noise rates
        - 'both': Intersection of both methods (most conservative)
    n_jobs : int, default=1
        Number of parallel jobs for cross-validation.
    random_state : int or None, default=None
        Random state for reproducibility.

    Attributes
    ----------
    noise_mask_ : ndarray of shape (n_samples,)
        Boolean mask where True indicates suspected noisy labels.
    noise_indices_ : ndarray
        Indices of suspected noisy examples.
    confident_joint_ : ndarray of shape (n_classes, n_classes)
        Estimated joint distribution of noisy vs. true labels.
    noise_rate_ : float
        Estimated overall noise rate.
    per_class_noise_rate_ : ndarray
        Estimated noise rate per class.
    pred_proba_ : ndarray of shape (n_samples, n_classes)
        Cross-validated predicted probabilities.

    Example
    -------
    >>> clf = ConfidentLearningFilter(base_estimator='rf', cv=5)
    >>> noise_mask = clf.fit_detect(X, y)
    >>> print(f"Found {noise_mask.sum()} noisy labels ({noise_mask.mean():.1%})")
    >>> X_clean, y_clean = X[~noise_mask], y[~noise_mask]
    """

    def __init__(
        self,
        base_estimator: str | Any = "rf",
        cv: int = 5,
        threshold: float | str = "auto",
        method: str = "prune_by_class",
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.cv = cv
        self.threshold = threshold
        self.method = method
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_estimator(self):
        """Resolve the base estimator."""
        if isinstance(self.base_estimator, str):
            if self.base_estimator == "rf":
                return RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    n_jobs=self.n_jobs,
                    random_state=self.random_state,
                )
            elif self.base_estimator == "xgboost":
                try:
                    from xgboost import XGBClassifier
                    return XGBClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.random_state,
                        verbosity=0,
                    )
                except ImportError:
                    raise ImportError("xgboost is required for base_estimator='xgboost'")
            elif self.base_estimator == "lgbm":
                try:
                    from lightgbm import LGBMClassifier
                    return LGBMClassifier(
                        n_estimators=100,
                        max_depth=6,
                        random_state=self.random_state,
                        verbose=-1,
                    )
                except ImportError:
                    raise ImportError("lightgbm is required for base_estimator='lgbm'")
            else:
                raise ValueError(f"Unknown estimator string: {self.base_estimator}")
        return clone(self.base_estimator)

    def fit(self, X, y) -> ConfidentLearningFilter:
        """Fit the noise detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Noisy training labels.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        n_samples = len(y)

        # Map labels to 0..n_classes-1
        label_map = {c: i for i, c in enumerate(self.classes_)}
        y_mapped = np.array([label_map[c] for c in y])

        # Cross-validated predicted probabilities
        estimator = self._get_estimator()
        cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )
        self.pred_proba_ = cross_val_predict(
            estimator, X, y_mapped,
            cv=cv, method="predict_proba",
            n_jobs=self.n_jobs,
        )

        # Compute per-class thresholds
        if self.threshold == "auto":
            thresholds = np.zeros(n_classes)
            for k in range(n_classes):
                class_mask = y_mapped == k
                if class_mask.sum() > 0:
                    thresholds[k] = self.pred_proba_[class_mask, k].mean()
                else:
                    thresholds[k] = 0.5
        else:
            thresholds = np.full(n_classes, float(self.threshold))

        # Compute confident joint matrix
        self.confident_joint_ = np.zeros((n_classes, n_classes), dtype=int)
        for i in range(n_samples):
            given_label = y_mapped[i]
            proba = self.pred_proba_[i]
            # Find classes where predicted probability exceeds threshold
            confident_classes = np.where(proba >= thresholds)[0]
            if len(confident_classes) == 0:
                # Use argmax as fallback
                confident_classes = [np.argmax(proba)]
            for pred_label in confident_classes:
                self.confident_joint_[given_label, pred_label] += 1

        # Estimate noise rates
        self.per_class_noise_rate_ = np.zeros(n_classes)
        for k in range(n_classes):
            total_k = self.confident_joint_[k].sum()
            if total_k > 0:
                self.per_class_noise_rate_[k] = 1.0 - self.confident_joint_[k, k] / total_k

        self.noise_rate_ = float(np.average(
            self.per_class_noise_rate_,
            weights=np.bincount(y_mapped, minlength=n_classes),
        ))

        # Identify noisy examples
        self.noise_mask_ = self._identify_noise(y_mapped, thresholds)
        self.noise_indices_ = np.where(self.noise_mask_)[0]

        return self

    def _identify_noise(self, y_mapped, thresholds):
        """Identify noisy examples based on the chosen method."""
        n_samples = len(y_mapped)
        n_classes = len(self.classes_)

        if self.method == "prune_by_class":
            return self._prune_by_class(y_mapped, thresholds)
        elif self.method == "prune_by_noise_rate":
            return self._prune_by_noise_rate(y_mapped)
        elif self.method == "both":
            mask1 = self._prune_by_class(y_mapped, thresholds)
            mask2 = self._prune_by_noise_rate(y_mapped)
            return mask1 & mask2
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _prune_by_class(self, y_mapped, thresholds):
        """Prune examples with low self-confidence."""
        n_samples = len(y_mapped)
        noise_mask = np.zeros(n_samples, dtype=bool)
        for i in range(n_samples):
            given_label = y_mapped[i]
            self_confidence = self.pred_proba_[i, given_label]
            pred_label = np.argmax(self.pred_proba_[i])
            # Mark as noisy if self-confidence is below threshold
            # AND predicted class differs from given label
            if self_confidence < thresholds[given_label] and pred_label != given_label:
                noise_mask[i] = True
        return noise_mask

    def _prune_by_noise_rate(self, y_mapped):
        """Prune based on estimated noise rates per class."""
        n_samples = len(y_mapped)
        n_classes = len(self.classes_)
        noise_mask = np.zeros(n_samples, dtype=bool)

        for k in range(n_classes):
            class_mask = y_mapped == k
            class_indices = np.where(class_mask)[0]
            if len(class_indices) == 0:
                continue

            # Sort by self-confidence (ascending = most likely noisy first)
            self_confidences = self.pred_proba_[class_indices, k]
            sorted_idx = np.argsort(self_confidences)

            # Remove the estimated number of noisy examples
            n_noisy = int(self.per_class_noise_rate_[k] * len(class_indices))
            noisy_in_class = class_indices[sorted_idx[:n_noisy]]
            noise_mask[noisy_in_class] = True

        return noise_mask

    def fit_detect(self, X, y) -> np.ndarray:
        """Fit and return the noise mask.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Noisy training labels.

        Returns
        -------
        noise_mask : ndarray of shape (n_samples,)
            Boolean mask where True indicates suspected noisy label.
        """
        self.fit(X, y)
        return self.noise_mask_

    def clean(self, X, y):
        """Fit and return cleaned data.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Labels.

        Returns
        -------
        X_clean : ndarray
            Features with noisy examples removed.
        y_clean : ndarray
            Labels with noisy examples removed.
        """
        noise_mask = self.fit_detect(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        return X[~noise_mask], y[~noise_mask]


class ConsensusFilter(BaseEstimator):
    """Identify noisy labels via consensus of multiple classifiers.

    Trains multiple diverse classifiers and identifies examples where
    the majority disagree with the given label.

    Parameters
    ----------
    estimators : list of estimators, optional
        List of classifiers to use. If None, uses a default diverse set.
    cv : int, default=5
        Cross-validation folds for prediction.
    consensus_threshold : float, default=0.5
        Fraction of classifiers that must disagree with the given label
        for it to be flagged as noisy.
    n_jobs : int, default=1
        Number of parallel jobs.
    random_state : int or None, default=None
        Random state for reproducibility.

    Example
    -------
    >>> from endgame.preprocessing import ConsensusFilter
    >>> cf = ConsensusFilter(consensus_threshold=0.7)
    >>> noise_mask = cf.fit_detect(X, y)
    """

    def __init__(
        self,
        estimators=None,
        cv: int = 5,
        consensus_threshold: float = 0.5,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.estimators = estimators
        self.cv = cv
        self.consensus_threshold = consensus_threshold
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_default_estimators(self):
        """Get a diverse set of default classifiers."""
        estimators = [
            RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=1
            ),
        ]
        try:
            from sklearn.linear_model import LogisticRegression
            estimators.append(
                LogisticRegression(max_iter=1000, random_state=self.random_state)
            )
        except ImportError:
            pass

        try:
            from sklearn.neighbors import KNeighborsClassifier
            estimators.append(KNeighborsClassifier(n_neighbors=10))
        except ImportError:
            pass

        try:
            from sklearn.ensemble import GradientBoostingClassifier
            estimators.append(
                GradientBoostingClassifier(
                    n_estimators=50, max_depth=4, random_state=self.random_state
                )
            )
        except ImportError:
            pass

        return estimators

    def fit(self, X, y) -> ConsensusFilter:
        """Fit the consensus noise detector.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        estimators = self.estimators or self._get_default_estimators()
        cv = StratifiedKFold(
            n_splits=self.cv,
            shuffle=True,
            random_state=self.random_state,
        )

        n_samples = len(y)
        disagreement_counts = np.zeros(n_samples, dtype=int)

        for est in estimators:
            try:
                oof_preds = cross_val_predict(est, X, y, cv=cv, n_jobs=self.n_jobs)
                disagreement_counts += (oof_preds != y).astype(int)
            except Exception:
                continue

        n_estimators_used = max(len(estimators), 1)
        self.disagreement_ratio_ = disagreement_counts / n_estimators_used
        self.noise_mask_ = self.disagreement_ratio_ >= self.consensus_threshold
        self.noise_indices_ = np.where(self.noise_mask_)[0]
        self.noise_rate_ = float(self.noise_mask_.mean())

        return self

    def fit_detect(self, X, y) -> np.ndarray:
        """Fit and return noise mask."""
        self.fit(X, y)
        return self.noise_mask_

    def clean(self, X, y):
        """Fit and return cleaned data."""
        noise_mask = self.fit_detect(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        return X[~noise_mask], y[~noise_mask]


class CrossValNoiseDetector(BaseEstimator):
    """Simple cross-validated noise detection.

    Flags examples that are consistently misclassified across CV folds
    as potentially noisy.

    Parameters
    ----------
    base_estimator : estimator, default=None
        Classifier to use. If None, uses RandomForestClassifier.
    cv : int, default=5
        Number of CV folds.
    n_repeats : int, default=3
        Number of repetitions with different random seeds.
    misclassification_threshold : float, default=0.5
        Fraction of times an example must be misclassified across all
        folds and repeats to be flagged as noisy.
    random_state : int or None, default=None
        Random state.

    Example
    -------
    >>> detector = CrossValNoiseDetector(n_repeats=5)
    >>> noise_mask = detector.fit_detect(X, y)
    """

    def __init__(
        self,
        base_estimator=None,
        cv: int = 5,
        n_repeats: int = 3,
        misclassification_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.cv = cv
        self.n_repeats = n_repeats
        self.misclassification_threshold = misclassification_threshold
        self.random_state = random_state

    def fit(self, X, y) -> CrossValNoiseDetector:
        """Fit the noise detector."""
        X = np.asarray(X)
        y = np.asarray(y)

        estimator = self.base_estimator or RandomForestClassifier(
            n_estimators=100, random_state=self.random_state
        )

        n_samples = len(y)
        misclassification_counts = np.zeros(n_samples, dtype=int)
        total_evaluations = np.zeros(n_samples, dtype=int)

        rng = np.random.RandomState(self.random_state)

        for rep in range(self.n_repeats):
            seed = rng.randint(0, 2**31)
            cv = StratifiedKFold(
                n_splits=self.cv, shuffle=True, random_state=seed
            )
            try:
                oof_preds = cross_val_predict(clone(estimator), X, y, cv=cv)
                misclassification_counts += (oof_preds != y).astype(int)
                total_evaluations += 1
            except Exception:
                continue

        total_evaluations = np.maximum(total_evaluations, 1)
        self.misclassification_rate_ = misclassification_counts / total_evaluations
        self.noise_mask_ = self.misclassification_rate_ >= self.misclassification_threshold
        self.noise_indices_ = np.where(self.noise_mask_)[0]
        self.noise_rate_ = float(self.noise_mask_.mean())

        return self

    def fit_detect(self, X, y) -> np.ndarray:
        """Fit and return noise mask."""
        self.fit(X, y)
        return self.noise_mask_

    def clean(self, X, y):
        """Fit and return cleaned data."""
        noise_mask = self.fit_detect(X, y)
        X = np.asarray(X)
        y = np.asarray(y)
        return X[~noise_mask], y[~noise_mask]
