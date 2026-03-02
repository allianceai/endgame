from __future__ import annotations

"""Venn-ABERS predictors for well-calibrated probability intervals.

Venn-ABERS provides probability intervals rather than point estimates,
with theoretical guarantees of calibration.

References
----------
- Vovk & Petej "Venn-ABERS Predictors" (2012)
- Vovk et al. "Large-scale probabilistic predictors with and without
  guarantees of validity" (2015)
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.isotonic import IsotonicRegression


class VennABERS(BaseEstimator, ClassifierMixin):
    """Venn-ABERS predictors for well-calibrated probabilities.

    Provides probability intervals [p0, p1] rather than point estimates.
    The intervals have theoretical validity guarantees under exchangeability.

    For a new test point, computes:
    - p1: calibrated probability assuming the true label is 1
    - p0: calibrated probability assuming the true label is 0

    The final probability estimate can be taken as the geometric mean
    or other combination of p0 and p1.

    Parameters
    ----------
    estimator : sklearn-compatible classifier, optional
        Base classifier with predict_proba. If None, only transform
        methods are available.
    inductive : bool, default=True
        Use inductive (split) Venn-ABERS. If False, uses full
        (computationally expensive) Venn-ABERS.
    precision : float, default=0.001
        Precision for isotonic regression calibration points.

    Attributes
    ----------
    estimator_ : estimator
        Fitted base classifier.
    p0_calibrator_ : IsotonicRegression
        Calibrator for label=0 assumption.
    p1_calibrator_ : IsotonicRegression
        Calibrator for label=1 assumption.
    cal_scores_ : ndarray
        Calibration set scores.
    cal_labels_ : ndarray
        Calibration set labels.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from endgame.calibration import VennABERS
    >>>
    >>> va = VennABERS(RandomForestClassifier(n_estimators=100))
    >>> va.fit(X_train, y_train, X_cal, y_cal)
    >>>
    >>> # Get probability intervals
    >>> p0, p1 = va.predict_proba_interval(X_test)
    >>>
    >>> # Get point estimate (geometric mean)
    >>> proba = va.predict_proba(X_test)[:, 1]
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        estimator: BaseEstimator | None = None,
        inductive: bool = True,
        precision: float = 0.001,
    ):
        self.estimator = estimator
        self.inductive = inductive
        self.precision = precision

        self.estimator_: BaseEstimator | None = None
        self.p0_calibrator_: IsotonicRegression | None = None
        self.p1_calibrator_: IsotonicRegression | None = None
        self.cal_scores_: np.ndarray | None = None
        self.cal_labels_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_train,
        y_train,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        cal_size: float = 0.2,
    ) -> VennABERS:
        """Fit Venn-ABERS predictor.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training labels.
        X_cal : array-like, optional
            Calibration features.
        y_cal : array-like, optional
            Calibration labels.
        cal_size : float, default=0.2
            Fraction for calibration if not provided separately.

        Returns
        -------
        self
        """
        from sklearn.model_selection import train_test_split

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        self.classes_ = np.unique(y_train)
        if len(self.classes_) != 2:
            raise ValueError("Venn-ABERS only supports binary classification")

        # Split calibration set if not provided
        if X_cal is None or y_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_size,
                stratify=y_train,
            )
        else:
            X_cal = np.asarray(X_cal)
            y_cal = np.asarray(y_cal)

        # Fit base estimator
        if self.estimator is not None:
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X_train, y_train)

            # Get calibration scores (probability of positive class)
            self.cal_scores_ = self.estimator_.predict_proba(X_cal)[:, 1]
        else:
            # Assume scores will be provided directly
            self.cal_scores_ = X_cal.ravel() if X_cal.ndim == 1 else X_cal[:, 0]

        self.cal_labels_ = y_cal

        # Fit isotonic calibrators
        self._fit_calibrators()

        self._is_fitted = True
        return self

    def _fit_calibrators(self):
        """Pre-fit isotonic regression calibrator on calibration data.

        This provides a baseline calibration. The full Venn-ABERS interval
        computation in _compute_venn_abers_interval() augments per test point.
        """
        scores = self.cal_scores_
        labels = self.cal_labels_

        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_labels = labels[sorted_indices]

        # Single isotonic fit for fast predict_proba fallback
        self.p1_calibrator_ = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        self.p1_calibrator_.fit(sorted_scores, sorted_labels)
        self.p0_calibrator_ = self.p1_calibrator_  # Same baseline

    def predict_proba_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict probability intervals.

        Parameters
        ----------
        X : array-like
            Test samples or scores.

        Returns
        -------
        p0 : ndarray
            Lower probability bounds (assuming label=0).
        p1 : ndarray
            Upper probability bounds (assuming label=1).
        """
        self._check_is_fitted()

        # Get test scores
        if self.estimator_ is not None:
            X = np.asarray(X)
            test_scores = self.estimator_.predict_proba(X)[:, 1]
        else:
            test_scores = np.asarray(X).ravel()

        n_test = len(test_scores)
        p0 = np.zeros(n_test)
        p1 = np.zeros(n_test)

        # For each test point, compute Venn-ABERS interval
        for i, score in enumerate(test_scores):
            p0[i], p1[i] = self._compute_venn_abers_interval(score)

        return p0, p1

    def _compute_venn_abers_interval(self, score: float) -> tuple[float, float]:
        """Compute Venn-ABERS probability interval for a single score."""
        cal_scores = self.cal_scores_
        cal_labels = self.cal_labels_
        n_cal = len(cal_labels)

        # Create augmented calibration sets
        # For p1: add (score, 1) to calibration
        scores_with_1 = np.append(cal_scores, score)
        labels_with_1 = np.append(cal_labels, 1)

        # For p0: add (score, 0) to calibration
        scores_with_0 = np.append(cal_scores, score)
        labels_with_0 = np.append(cal_labels, 0)

        # Fit isotonic regression and get calibrated probability
        iso_1 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        sorted_idx_1 = np.argsort(scores_with_1)
        iso_1.fit(scores_with_1[sorted_idx_1], labels_with_1[sorted_idx_1])
        p1 = iso_1.predict([score])[0]

        iso_0 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
        sorted_idx_0 = np.argsort(scores_with_0)
        iso_0.fit(scores_with_0[sorted_idx_0], labels_with_0[sorted_idx_0])
        p0 = iso_0.predict([score])[0]

        # Ensure p0 <= p1
        return min(p0, p1), max(p0, p1)

    def predict_proba(self, X) -> np.ndarray:
        """Predict calibrated probabilities.

        Uses geometric mean of interval endpoints as point estimate.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, 2)
            Calibrated class probabilities.
        """
        p0, p1 = self.predict_proba_interval(X)

        # Arithmetic mean of interval endpoints
        proba_pos = (p0 + p1) / 2

        # Handle edge cases
        proba_pos = np.clip(proba_pos, 0, 1)

        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def interval_width(self, X) -> np.ndarray:
        """Compute uncertainty (interval width) for each prediction.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray
            Interval widths (p1 - p0).
        """
        p0, p1 = self.predict_proba_interval(X)
        return p1 - p0

    def _check_is_fitted(self):
        """Check if fitted."""
        if not self._is_fitted:
            raise RuntimeError("VennABERS has not been fitted.")


class IVAPCalibrator(BaseEstimator, ClassifierMixin):
    """Inductive Venn-ABERS Predictor with faster inference.

    A more efficient version of Venn-ABERS that pre-computes
    calibration mappings for faster test-time inference.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        Base classifier.
    n_bins : int, default=100
        Number of bins for precomputed calibration.

    Examples
    --------
    >>> ivap = IVAPCalibrator(LogisticRegression())
    >>> ivap.fit(X_train, y_train, X_cal, y_cal)
    >>> p0, p1 = ivap.predict_proba_interval(X_test)
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        n_bins: int = 100,
    ):
        self.estimator = estimator
        self.n_bins = n_bins

        self.estimator_: BaseEstimator | None = None
        self.p0_mapping_: np.ndarray | None = None
        self.p1_mapping_: np.ndarray | None = None
        self.bin_edges_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_train,
        y_train,
        X_cal,
        y_cal,
    ) -> IVAPCalibrator:
        """Fit IVAP calibrator."""
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_cal = np.asarray(X_cal)
        y_cal = np.asarray(y_cal)

        self.classes_ = np.unique(y_train)

        # Fit base estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)

        # Get calibration scores
        cal_scores = self.estimator_.predict_proba(X_cal)[:, 1]

        # Create bin edges
        self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)

        # Precompute p0 and p1 for each bin midpoint
        bin_midpoints = (self.bin_edges_[:-1] + self.bin_edges_[1:]) / 2
        self.p0_mapping_ = np.zeros(self.n_bins)
        self.p1_mapping_ = np.zeros(self.n_bins)

        for i, score in enumerate(bin_midpoints):
            # Augment calibration set
            scores_0 = np.append(cal_scores, score)
            labels_0 = np.append(y_cal, 0)
            scores_1 = np.append(cal_scores, score)
            labels_1 = np.append(y_cal, 1)

            # Fit isotonic and get predictions
            iso_0 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_0.fit(scores_0, labels_0)
            self.p0_mapping_[i] = iso_0.predict([score])[0]

            iso_1 = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
            iso_1.fit(scores_1, labels_1)
            self.p1_mapping_[i] = iso_1.predict([score])[0]

        # Ensure p0 <= p1
        for i in range(self.n_bins):
            if self.p0_mapping_[i] > self.p1_mapping_[i]:
                self.p0_mapping_[i], self.p1_mapping_[i] = self.p1_mapping_[i], self.p0_mapping_[i]

        self._is_fitted = True
        return self

    def predict_proba_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Predict probability intervals using precomputed mappings."""
        if not self._is_fitted:
            raise RuntimeError("IVAPCalibrator has not been fitted.")

        X = np.asarray(X)
        test_scores = self.estimator_.predict_proba(X)[:, 1]

        # Find bins
        bin_indices = np.digitize(test_scores, self.bin_edges_) - 1
        bin_indices = np.clip(bin_indices, 0, self.n_bins - 1)

        p0 = self.p0_mapping_[bin_indices]
        p1 = self.p1_mapping_[bin_indices]

        return p0, p1

    def predict_proba(self, X) -> np.ndarray:
        """Predict calibrated probabilities."""
        p0, p1 = self.predict_proba_interval(X)
        proba_pos = np.sqrt(p0 * p1)
        proba_pos = np.clip(proba_pos, 0, 1)
        return np.column_stack([1 - proba_pos, proba_pos])

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
