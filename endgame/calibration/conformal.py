"""Conformal prediction for classification and regression.

Conformal prediction provides prediction sets (classification) or intervals
(regression) with guaranteed coverage probability under exchangeability.

References
----------
- Vovk et al. "Algorithmic Learning in a Random World" (2005)
- Romano et al. "Conformalized Quantile Regression" (2019)
- Angelopoulos & Bates "A Gentle Introduction to Conformal Prediction" (2022)
"""

from dataclasses import dataclass

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.model_selection import train_test_split


@dataclass
class ConformalResult:
    """Result container for conformal predictions."""

    prediction_sets: list[set[int]] | None = None  # For classification
    intervals: tuple[np.ndarray, np.ndarray] | None = None  # For regression
    point_predictions: np.ndarray | None = None
    conformity_scores: np.ndarray | None = None
    quantile: float | None = None


class ConformalClassifier(BaseEstimator, ClassifierMixin):
    """Conformal prediction wrapper for classification.

    Provides prediction sets with guaranteed coverage probability.
    Under exchangeability, P(y ∈ C(X)) >= 1 - alpha for a new test point.

    Parameters
    ----------
    estimator : sklearn-compatible classifier
        Base classifier (must have predict_proba).
    method : str, default='lac'
        Conformal method:
        - 'lac': Least Ambiguous set-valued Classifier (adaptive)
        - 'aps': Adaptive Prediction Sets (randomized, exact coverage)
        - 'raps': Regularized APS (controls set size with penalty)
        - 'naive': Simple threshold on class probabilities
    alpha : float, default=0.1
        Miscoverage rate (1 - alpha = coverage probability).
    k_reg : int, default=1
        RAPS regularization: penalize sets larger than k_reg.
    lambda_reg : float, default=0.01
        RAPS regularization strength.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    estimator_ : estimator
        Fitted base classifier.
    classes_ : ndarray
        Class labels.
    n_classes_ : int
        Number of classes.
    conformity_scores_ : ndarray
        Calibration conformity scores.
    quantile_ : float
        Calibrated quantile threshold.

    Examples
    --------
    >>> from sklearn.linear_model import LogisticRegression
    >>> from endgame.calibration import ConformalClassifier
    >>>
    >>> # Split data: train, calibration, test
    >>> X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4)
    >>> X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    >>>
    >>> cp = ConformalClassifier(LogisticRegression(), method='aps', alpha=0.1)
    >>> cp.fit(X_train, y_train, X_cal, y_cal)
    >>> prediction_sets = cp.predict(X_test)  # Returns sets with ~90% coverage
    >>> print(f"Coverage: {cp.coverage_score(X_test, y_test):.3f}")
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        estimator: BaseEstimator,
        method: str = "lac",
        alpha: float = 0.1,
        k_reg: int = 1,
        lambda_reg: float = 0.01,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.method = method
        self.alpha = alpha
        self.k_reg = k_reg
        self.lambda_reg = lambda_reg
        self.random_state = random_state

        self.estimator_: BaseEstimator | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.conformity_scores_: np.ndarray | None = None
        self.quantile_: float | None = None
        self._is_fitted: bool = False

    def fit(
        self,
        X_train,
        y_train,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        cal_size: float = 0.2,
    ) -> "ConformalClassifier":
        """Fit base model and calibrate conformal scores.

        Parameters
        ----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training features.
        y_train : array-like of shape (n_train_samples,)
            Training labels.
        X_cal : array-like of shape (n_cal_samples, n_features), optional
            Calibration features. If None, splits from training data.
        y_cal : array-like of shape (n_cal_samples,), optional
            Calibration labels. If None, splits from training data.
        cal_size : float, default=0.2
            Fraction of training data to use for calibration if X_cal not provided.

        Returns
        -------
        self
            Fitted conformal classifier.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)

        # Split calibration set if not provided
        if X_cal is None or y_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_size,
                stratify=y_train,
                random_state=self.random_state,
            )
        else:
            X_cal = np.asarray(X_cal)
            y_cal = np.asarray(y_cal)

        # Store classes
        self.classes_ = np.unique(y_train)
        self.n_classes_ = len(self.classes_)

        # Fit base estimator
        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(X_train, y_train)

        # Get calibration probabilities
        cal_proba = self.estimator_.predict_proba(X_cal)

        # Compute conformity scores on calibration set
        self.conformity_scores_ = self._compute_conformity_scores(cal_proba, y_cal)

        # Compute quantile for coverage guarantee
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)
        self.quantile_ = np.quantile(self.conformity_scores_, q_level, method='higher')

        self._is_fitted = True
        return self

    def _compute_conformity_scores(
        self,
        proba: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Compute conformity scores based on method."""
        n_samples = len(y)

        if self.method in ("naive", "lac"):
            # Score = 1 - P(true class)
            # LAC (Least Ambiguous Classifier, Sadinle et al. 2019):
            # includes class y if P(y|x) >= 1 - q_hat
            scores = 1 - proba[np.arange(n_samples), y]

        elif self.method == "aps":
            # Adaptive Prediction Sets (randomized)
            rng = np.random.RandomState(self.random_state)
            U = rng.uniform(0, 1, n_samples)

            sorted_indices = np.argsort(-proba, axis=1)
            sorted_proba = np.take_along_axis(proba, sorted_indices, axis=1)
            cumsum_proba = np.cumsum(sorted_proba, axis=1)

            scores = np.zeros(n_samples)
            for i in range(n_samples):
                true_class = y[i]
                rank = np.where(sorted_indices[i] == true_class)[0][0]
                if rank == 0:
                    scores[i] = U[i] * sorted_proba[i, 0]
                else:
                    scores[i] = cumsum_proba[i, rank - 1] + U[i] * sorted_proba[i, rank]

        elif self.method == "raps":
            # Regularized APS
            rng = np.random.RandomState(self.random_state)
            U = rng.uniform(0, 1, n_samples)

            sorted_indices = np.argsort(-proba, axis=1)
            sorted_proba = np.take_along_axis(proba, sorted_indices, axis=1)

            # Add regularization penalty
            reg_penalty = self.lambda_reg * np.maximum(
                0,
                np.arange(1, self.n_classes_ + 1) - self.k_reg
            )
            sorted_proba_reg = sorted_proba + reg_penalty
            cumsum_proba = np.cumsum(sorted_proba_reg, axis=1)

            scores = np.zeros(n_samples)
            for i in range(n_samples):
                true_class = y[i]
                rank = np.where(sorted_indices[i] == true_class)[0][0]
                if rank == 0:
                    scores[i] = U[i] * sorted_proba_reg[i, 0]
                else:
                    scores[i] = cumsum_proba[i, rank - 1] + U[i] * sorted_proba_reg[i, rank]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return scores

    def predict(self, X) -> list[set[int]]:
        """Return prediction sets for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        List[Set[int]]
            Prediction set for each sample (set of class indices).
        """
        self._check_is_fitted()
        X = np.asarray(X)

        proba = self.estimator_.predict_proba(X)
        prediction_sets = self._construct_prediction_sets(proba)

        return prediction_sets

    def _construct_prediction_sets(self, proba: np.ndarray) -> list[set[int]]:
        """Construct prediction sets based on calibrated threshold."""
        n_samples = proba.shape[0]
        prediction_sets = []

        if self.method in ("naive", "lac"):
            # Include class y if P(y|x) >= 1 - quantile
            for i in range(n_samples):
                pred_set = set(np.where(proba[i] >= 1 - self.quantile_)[0])
                if len(pred_set) == 0:
                    pred_set = {np.argmax(proba[i])}
                prediction_sets.append(pred_set)

        elif self.method in ["aps", "raps"]:
            sorted_indices = np.argsort(-proba, axis=1)
            sorted_proba = np.take_along_axis(proba, sorted_indices, axis=1)

            if self.method == "raps":
                reg_penalty = self.lambda_reg * np.maximum(
                    0,
                    np.arange(1, self.n_classes_ + 1) - self.k_reg
                )
                sorted_proba = sorted_proba + reg_penalty

            cumsum_proba = np.cumsum(sorted_proba, axis=1)

            for i in range(n_samples):
                # Include classes in descending probability order until
                # cumulative probability exceeds the quantile threshold
                exceeds = np.where(cumsum_proba[i] >= self.quantile_)[0]
                if len(exceeds) == 0:
                    # Include all classes if threshold never reached
                    pred_set = set(sorted_indices[i])
                else:
                    # Include up to and including the first class that
                    # pushes cumulative probability past the threshold
                    max_idx = exceeds[0]
                    pred_set = set(sorted_indices[i, :max_idx + 1])
                prediction_sets.append(pred_set)

        return prediction_sets

    def predict_proba(self, X) -> np.ndarray:
        """Return base classifier probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
            Class probabilities from base classifier.
        """
        self._check_is_fitted()
        return self.estimator_.predict_proba(X)

    def predict_point(self, X) -> np.ndarray:
        """Return point predictions (most likely class).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def coverage_score(self, X, y) -> float:
        """Compute empirical coverage on test data.

        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True labels.

        Returns
        -------
        float
            Fraction of samples where true label is in prediction set.
        """
        prediction_sets = self.predict(X)
        y = np.asarray(y)

        covered = sum(y[i] in pred_set for i, pred_set in enumerate(prediction_sets))
        return covered / len(y)

    def average_set_size(self, X) -> float:
        """Average size of prediction sets (efficiency metric).

        Smaller sets = more informative predictions.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        float
            Average prediction set size.
        """
        prediction_sets = self.predict(X)
        return np.mean([len(s) for s in prediction_sets])

    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("ConformalClassifier has not been fitted.")


class ConformalRegressor(BaseEstimator, RegressorMixin):
    """Conformal prediction wrapper for regression.

    Provides prediction intervals with guaranteed coverage probability.
    Under exchangeability, P(y ∈ [lower, upper]) >= 1 - alpha.

    Parameters
    ----------
    estimator : sklearn-compatible regressor
        Base regressor.
    method : str, default='absolute'
        Conformal method:
        - 'absolute': Absolute residual method (symmetric intervals)
        - 'normalized': Normalized residuals using variance estimate
        - 'cqr': Conformalized Quantile Regression (asymmetric intervals)
        - 'cqr_asymmetric': CQR with separate lower/upper scores
    alpha : float, default=0.1
        Miscoverage rate (1 - alpha = coverage probability).
    quantile_estimator : estimator, optional
        For CQR methods, estimator for quantiles. If None, uses
        GradientBoostingRegressor with quantile loss.
    symmetry : bool, default=True
        For CQR, whether to use symmetric or asymmetric intervals.
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    estimator_ : estimator
        Fitted base regressor.
    lower_estimator_ : estimator
        For CQR, fitted lower quantile estimator.
    upper_estimator_ : estimator
        For CQR, fitted upper quantile estimator.
    conformity_scores_ : ndarray
        Calibration conformity scores.
    quantile_ : float
        Calibrated quantile threshold.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from endgame.calibration import ConformalRegressor
    >>>
    >>> cr = ConformalRegressor(RandomForestRegressor(), method='cqr', alpha=0.1)
    >>> cr.fit(X_train, y_train, X_cal, y_cal)
    >>> lower, upper = cr.predict_interval(X_test)
    >>> print(f"Coverage: {cr.coverage_score(X_test, y_test):.3f}")
    >>> print(f"Avg width: {cr.average_interval_width(X_test):.3f}")
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        estimator: BaseEstimator,
        method: str = "absolute",
        alpha: float = 0.1,
        quantile_estimator: BaseEstimator | None = None,
        symmetry: bool = True,
        random_state: int | None = None,
    ):
        self.estimator = estimator
        self.method = method
        self.alpha = alpha
        self.quantile_estimator = quantile_estimator
        self.symmetry = symmetry
        self.random_state = random_state

        self.estimator_: BaseEstimator | None = None
        self.lower_estimator_: BaseEstimator | None = None
        self.upper_estimator_: BaseEstimator | None = None
        self.variance_estimator_: BaseEstimator | None = None
        self.conformity_scores_: np.ndarray | None = None
        self.quantile_: float | None = None
        self.lower_quantile_: float | None = None
        self.upper_quantile_: float | None = None
        self._is_fitted: bool = False

    def _get_quantile_estimator(self, quantile: float):
        """Get quantile regressor."""
        if self.quantile_estimator is not None:
            est = clone(self.quantile_estimator)
            if hasattr(est, 'alpha'):
                est.alpha = quantile
            elif hasattr(est, 'quantile'):
                est.quantile = quantile
            return est

        # Default to GradientBoostingRegressor with quantile loss
        try:
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                loss='quantile',
                alpha=quantile,
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state,
            )
        except ImportError:
            raise ValueError(
                "For CQR methods, please provide a quantile_estimator or "
                "ensure sklearn.ensemble.GradientBoostingRegressor is available."
            )

    def fit(
        self,
        X_train,
        y_train,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        cal_size: float = 0.2,
    ) -> "ConformalRegressor":
        """Fit base model and calibrate conformal scores.

        Parameters
        ----------
        X_train : array-like
            Training features.
        y_train : array-like
            Training targets.
        X_cal : array-like, optional
            Calibration features.
        y_cal : array-like, optional
            Calibration targets.
        cal_size : float, default=0.2
            Fraction for calibration if not provided.

        Returns
        -------
        self
            Fitted conformal regressor.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()

        # Split calibration set if not provided
        if X_cal is None or y_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_size,
                random_state=self.random_state,
            )
        else:
            X_cal = np.asarray(X_cal)
            y_cal = np.asarray(y_cal).ravel()

        if self.method in ["absolute", "normalized"]:
            # Fit point estimator
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X_train, y_train)

            # Get calibration predictions
            cal_pred = self.estimator_.predict(X_cal)
            residuals = y_cal - cal_pred

            if self.method == "absolute":
                self.conformity_scores_ = np.abs(residuals)
            else:
                # Normalized: need variance estimate
                self.variance_estimator_ = clone(self.estimator)
                self.variance_estimator_.fit(X_train, (y_train - self.estimator_.predict(X_train))**2)
                cal_var = np.maximum(self.variance_estimator_.predict(X_cal), 1e-6)
                self.conformity_scores_ = np.abs(residuals) / np.sqrt(cal_var)

        elif self.method in ["cqr", "cqr_asymmetric"]:
            # Conformalized Quantile Regression
            lower_q = self.alpha / 2
            upper_q = 1 - self.alpha / 2

            # Fit point estimator (for predict method)
            self.estimator_ = clone(self.estimator)
            self.estimator_.fit(X_train, y_train)

            # Fit quantile estimators
            self.lower_estimator_ = self._get_quantile_estimator(lower_q)
            self.upper_estimator_ = self._get_quantile_estimator(upper_q)

            self.lower_estimator_.fit(X_train, y_train)
            self.upper_estimator_.fit(X_train, y_train)

            # Get calibration quantile predictions
            cal_lower = self.lower_estimator_.predict(X_cal)
            cal_upper = self.upper_estimator_.predict(X_cal)

            if self.method == "cqr" or self.symmetry:
                # Symmetric CQR: take max of both sides
                self.conformity_scores_ = np.maximum(
                    cal_lower - y_cal,
                    y_cal - cal_upper
                )
            else:
                # Asymmetric CQR: separate scores
                self.lower_scores_ = cal_lower - y_cal
                self.upper_scores_ = y_cal - cal_upper
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Compute quantile(s)
        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)

        if self.method == "cqr_asymmetric" and not self.symmetry:
            q_level_half = np.ceil((n_cal + 1) * (1 - self.alpha / 2)) / n_cal
            q_level_half = min(q_level_half, 1.0)
            self.lower_quantile_ = np.quantile(self.lower_scores_, q_level_half, method='higher')
            self.upper_quantile_ = np.quantile(self.upper_scores_, q_level_half, method='higher')
        else:
            self.quantile_ = np.quantile(self.conformity_scores_, q_level, method='higher')

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Return point predictions.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        ndarray
            Point predictions.
        """
        self._check_is_fitted()
        return self.estimator_.predict(X)

    def predict_interval(self, X) -> tuple[np.ndarray, np.ndarray]:
        """Return prediction intervals.

        Parameters
        ----------
        X : array-like
            Test samples.

        Returns
        -------
        lower : ndarray
            Lower bounds of prediction intervals.
        upper : ndarray
            Upper bounds of prediction intervals.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if self.method == "absolute":
            pred = self.estimator_.predict(X)
            lower = pred - self.quantile_
            upper = pred + self.quantile_

        elif self.method == "normalized":
            pred = self.estimator_.predict(X)
            var = np.maximum(self.variance_estimator_.predict(X), 1e-6)
            width = self.quantile_ * np.sqrt(var)
            lower = pred - width
            upper = pred + width

        elif self.method in ["cqr", "cqr_asymmetric"]:
            pred_lower = self.lower_estimator_.predict(X)
            pred_upper = self.upper_estimator_.predict(X)

            if self.method == "cqr" or self.symmetry:
                lower = pred_lower - self.quantile_
                upper = pred_upper + self.quantile_
            else:
                lower = pred_lower - self.lower_quantile_
                upper = pred_upper + self.upper_quantile_

        return lower, upper

    def coverage_score(self, X, y) -> float:
        """Compute empirical coverage.

        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            True targets.

        Returns
        -------
        float
            Fraction of samples where true value is within interval.
        """
        lower, upper = self.predict_interval(X)
        y = np.asarray(y).ravel()

        covered = np.sum((y >= lower) & (y <= upper))
        return covered / len(y)

    def average_interval_width(self, X) -> float:
        """Average width of prediction intervals.

        Parameters
        ----------
        X : array-like
            Test features.

        Returns
        -------
        float
            Average interval width.
        """
        lower, upper = self.predict_interval(X)
        return np.mean(upper - lower)

    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not self._is_fitted:
            raise RuntimeError("ConformalRegressor has not been fitted.")


class ConformizedQuantileRegressor(BaseEstimator, RegressorMixin):
    """Conformalized Quantile Regression for adaptive prediction intervals.

    CQR combines quantile regression with conformal calibration to produce
    prediction intervals that:
    1. Adapt to heteroscedasticity (wider where uncertainty is higher)
    2. Have guaranteed coverage under exchangeability
    3. Are typically narrower than standard conformal methods

    The algorithm:
    1. Train quantile regressors for lower (α/2) and upper (1-α/2) quantiles
    2. On calibration data, compute conformity scores:
       E_i = max(q_lower(x_i) - y_i, y_i - q_upper(x_i))
    3. Compute the (1-α)(1 + 1/n) quantile of scores
    4. At prediction time:
       [q_lower(x) - Q, q_upper(x) + Q]

    This implementation integrates with QuantileRegressorForest but works
    with any regressor that can predict quantiles.

    Parameters
    ----------
    quantile_estimator : estimator, optional
        A regressor capable of predicting quantiles. Options:
        - QuantileRegressorForest (recommended): Native quantile support
        - GradientBoostingRegressor: With loss='quantile'
        - Any estimator with predict_quantiles(X, quantiles) method
        If None, uses QuantileRegressorForest with default settings.

    alpha : float, default=0.1
        Miscoverage rate. Target coverage is 1 - alpha.
        E.g., alpha=0.1 targets 90% coverage.

    cv : int or None, default=None
        If int, use cross-conformal with cv folds for more efficient
        data usage. Each fold serves as calibration for models trained
        on other folds. If None, requires separate calibration set.

    symmetric : bool, default=True
        If True, use symmetric conformity scores:
            E = max(q_lower - y, y - q_upper)
        If False, use asymmetric scores (separate lower/upper adjustments):
            E_lower = q_lower - y
            E_upper = y - q_upper
        Asymmetric can give tighter intervals but requires more calibration data.

    random_state : int, RandomState, or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    quantile_estimator_ : estimator
        Fitted quantile estimator (single model predicting both quantiles).

    conformity_scores_ : ndarray
        Calibration conformity scores.

    quantile_ : float
        Calibrated quantile threshold for symmetric CQR.

    lower_quantile_ : float
        Lower threshold for asymmetric CQR.

    upper_quantile_ : float
        Upper threshold for asymmetric CQR.

    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> from endgame.models.trees import QuantileRegressorForest
    >>> from endgame.calibration import ConformizedQuantileRegressor
    >>> from sklearn.model_selection import train_test_split
    >>>
    >>> # Split data into train and calibration
    >>> X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2)
    >>>
    >>> # Create CQR with QRF (default)
    >>> cqr = ConformizedQuantileRegressor(alpha=0.1)
    >>> cqr.fit(X_train, y_train, X_cal, y_cal)
    >>>
    >>> # Get prediction intervals
    >>> lower, upper = cqr.predict_interval(X_test)
    >>> print(f"Coverage: {cqr.coverage_score(X_test, y_test):.3f}")
    >>> print(f"Avg width: {np.mean(upper - lower):.3f}")
    >>>
    >>> # Using cross-conformal (no separate calibration set needed)
    >>> cqr_cv = ConformizedQuantileRegressor(alpha=0.1, cv=5)
    >>> cqr_cv.fit(X_train, y_train)
    >>> lower, upper = cqr_cv.predict_interval(X_test)

    Notes
    -----
    **When to use CQR vs standard conformal regression:**

    - CQR produces adaptive intervals that are wider in high-uncertainty
      regions and narrower in low-uncertainty regions
    - Standard conformal produces constant-width intervals
    - CQR is preferred for heteroscedastic data (varying noise)
    - Standard conformal is simpler and may suffice for homoscedastic data

    **Calibration set size:**

    For reliable coverage, use at least 100-500 calibration samples.
    With cv > 0, each fold acts as calibration, improving data efficiency.

    **Integration with QuantileRegressorForest:**

    QRF naturally estimates quantiles by storing leaf distributions.
    This makes it ideal for CQR as it provides well-calibrated quantile
    estimates out of the box without separate quantile loss training.

    References
    ----------
    Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized Quantile
    Regression." NeurIPS.

    Sesia, M., & Candès, E. (2020). "A comparison of some conformal quantile
    regression methods." Stat, 9(1), e261.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        quantile_estimator: BaseEstimator | None = None,
        alpha: float = 0.1,
        cv: int | None = None,
        symmetric: bool = True,
        random_state: int | np.random.RandomState | None = None,
    ):
        self.quantile_estimator = quantile_estimator
        self.alpha = alpha
        self.cv = cv
        self.symmetric = symmetric
        self.random_state = random_state

    def _get_default_quantile_estimator(self):
        """Get default QuantileRegressorForest."""
        try:
            from endgame.models.trees import QuantileRegressorForest
            return QuantileRegressorForest(
                n_estimators=100,
                random_state=self.random_state,
            )
        except ImportError:
            # Fallback to GradientBoostingRegressor
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=3,
                random_state=self.random_state,
            )

    def _predict_quantiles(
        self,
        estimator: BaseEstimator,
        X: np.ndarray,
        quantiles: list[float],
    ) -> np.ndarray:
        """Predict quantiles using the estimator.

        Returns array of shape (n_samples, n_quantiles).
        """
        # Check if estimator has native quantile support
        if hasattr(estimator, 'predict_quantiles'):
            result = estimator.predict_quantiles(X, quantiles)
            # Ensure 2D output
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            return result

        # For QRF with quantiles parameter
        if hasattr(estimator, 'quantiles'):
            original_quantiles = estimator.quantiles
            estimator.quantiles = quantiles
            result = estimator.predict(X)
            estimator.quantiles = original_quantiles
            if result.ndim == 1:
                result = result.reshape(-1, 1)
            return result

        # Fallback: assume it's a GradientBoostingRegressor-style estimator
        # Need to fit separate models for each quantile
        raise ValueError(
            "Estimator must have predict_quantiles method or be a "
            "QuantileRegressorForest. For other estimators, use "
            "ConformalRegressor with method='cqr' instead."
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray | None = None,
        y_cal: np.ndarray | None = None,
        cal_size: float = 0.2,
    ) -> "ConformizedQuantileRegressor":
        """Fit the CQR model.

        Parameters
        ----------
        X_train : array-like of shape (n_train_samples, n_features)
            Training features.
        y_train : array-like of shape (n_train_samples,)
            Training targets.
        X_cal : array-like of shape (n_cal_samples, n_features), optional
            Calibration features. Required if cv is None.
            If None and cv is None, splits from training data.
        y_cal : array-like of shape (n_cal_samples,), optional
            Calibration targets.
        cal_size : float, default=0.2
            Fraction of training data for calibration if X_cal not provided.

        Returns
        -------
        self : object
            Fitted CQR model.
        """
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()

        self.n_features_in_ = X_train.shape[1]

        # Quantiles to predict
        lower_q = self.alpha / 2
        upper_q = 1 - self.alpha / 2
        self._quantiles = [lower_q, upper_q]

        if self.cv is not None:
            # Cross-conformal: use CV to get conformity scores
            return self._fit_cross_conformal(X_train, y_train)

        # Standard split-conformal
        if X_cal is None or y_cal is None:
            X_train, X_cal, y_train, y_cal = train_test_split(
                X_train, y_train,
                test_size=cal_size,
                random_state=self.random_state,
            )
        else:
            X_cal = np.asarray(X_cal)
            y_cal = np.asarray(y_cal).ravel()

        # Initialize and fit quantile estimator
        if self.quantile_estimator is None:
            self.quantile_estimator_ = self._get_default_quantile_estimator()
        else:
            self.quantile_estimator_ = clone(self.quantile_estimator)

        self.quantile_estimator_.fit(X_train, y_train)

        # Compute conformity scores on calibration set
        self._calibrate(X_cal, y_cal)

        self._is_fitted = True
        return self

    def _fit_cross_conformal(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> "ConformizedQuantileRegressor":
        """Fit using cross-conformal for better data efficiency."""
        from sklearn.model_selection import KFold

        n_samples = len(y)
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Store conformity scores from each fold
        all_scores = []
        all_lower_scores = []
        all_upper_scores = []

        # For final model, fit on all data
        if self.quantile_estimator is None:
            self.quantile_estimator_ = self._get_default_quantile_estimator()
        else:
            self.quantile_estimator_ = clone(self.quantile_estimator)

        for train_idx, cal_idx in kf.split(X):
            X_train_fold = X[train_idx]
            y_train_fold = y[train_idx]
            X_cal_fold = X[cal_idx]
            y_cal_fold = y[cal_idx]

            # Fit on training fold
            fold_estimator = clone(self.quantile_estimator_)
            fold_estimator.fit(X_train_fold, y_train_fold)

            # Get quantile predictions on calibration fold
            q_pred = self._predict_quantiles(fold_estimator, X_cal_fold, self._quantiles)
            q_lower = q_pred[:, 0]
            q_upper = q_pred[:, 1]

            # Compute conformity scores
            if self.symmetric:
                scores = np.maximum(q_lower - y_cal_fold, y_cal_fold - q_upper)
                all_scores.extend(scores.tolist())
            else:
                all_lower_scores.extend((q_lower - y_cal_fold).tolist())
                all_upper_scores.extend((y_cal_fold - q_upper).tolist())

        # Compute calibrated quantile(s)
        n_cal = len(all_scores) if self.symmetric else len(all_lower_scores)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)

        if self.symmetric:
            self.conformity_scores_ = np.array(all_scores)
            self.quantile_ = np.quantile(self.conformity_scores_, q_level, method='higher')
        else:
            q_level_half = np.ceil((n_cal + 1) * (1 - self.alpha / 2)) / n_cal
            q_level_half = min(q_level_half, 1.0)
            self.lower_scores_ = np.array(all_lower_scores)
            self.upper_scores_ = np.array(all_upper_scores)
            self.lower_quantile_ = np.quantile(self.lower_scores_, q_level_half, method='higher')
            self.upper_quantile_ = np.quantile(self.upper_scores_, q_level_half, method='higher')

        # Fit final model on all data
        self.quantile_estimator_.fit(X, y)

        self._is_fitted = True
        return self

    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> None:
        """Compute conformity scores and calibrated quantile."""
        # Get quantile predictions
        q_pred = self._predict_quantiles(self.quantile_estimator_, X_cal, self._quantiles)
        q_lower = q_pred[:, 0]
        q_upper = q_pred[:, 1]

        n_cal = len(y_cal)
        q_level = np.ceil((n_cal + 1) * (1 - self.alpha)) / n_cal
        q_level = min(q_level, 1.0)

        if self.symmetric:
            # Symmetric conformity scores
            self.conformity_scores_ = np.maximum(
                q_lower - y_cal,
                y_cal - q_upper
            )
            self.quantile_ = np.quantile(
                self.conformity_scores_, q_level, method='higher'
            )
        else:
            # Asymmetric conformity scores
            self.lower_scores_ = q_lower - y_cal
            self.upper_scores_ = y_cal - q_upper

            q_level_half = np.ceil((n_cal + 1) * (1 - self.alpha / 2)) / n_cal
            q_level_half = min(q_level_half, 1.0)

            self.lower_quantile_ = np.quantile(
                self.lower_scores_, q_level_half, method='higher'
            )
            self.upper_quantile_ = np.quantile(
                self.upper_scores_, q_level_half, method='higher'
            )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return point predictions (median).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Point predictions.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Predict median
        q_pred = self._predict_quantiles(self.quantile_estimator_, X, [0.5])
        return q_pred[:, 0]

    def predict_interval(
        self,
        X: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return conformalized prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.

        Returns
        -------
        lower : ndarray of shape (n_samples,)
            Lower bounds of prediction intervals.
        upper : ndarray of shape (n_samples,)
            Upper bounds of prediction intervals.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        # Get quantile predictions
        q_pred = self._predict_quantiles(self.quantile_estimator_, X, self._quantiles)
        q_lower = q_pred[:, 0]
        q_upper = q_pred[:, 1]

        # Apply conformal correction
        if self.symmetric:
            lower = q_lower - self.quantile_
            upper = q_upper + self.quantile_
        else:
            lower = q_lower - self.lower_quantile_
            upper = q_upper + self.upper_quantile_

        return lower, upper

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: list[float] | None = None,
    ) -> np.ndarray:
        """Predict arbitrary quantiles (uncalibrated).

        Note: These are raw quantile predictions without conformal
        calibration. For calibrated intervals, use predict_interval().

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        quantiles : list of float, optional
            Quantiles to predict. Default: [0.1, 0.5, 0.9]

        Returns
        -------
        ndarray of shape (n_samples, n_quantiles)
            Quantile predictions.
        """
        self._check_is_fitted()
        X = np.asarray(X)

        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]

        return self._predict_quantiles(self.quantile_estimator_, X, quantiles)

    def coverage_score(self, X, y) -> float:
        """Compute empirical coverage.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.
        y : array-like of shape (n_samples,)
            True targets.

        Returns
        -------
        float
            Fraction of samples where true value is within interval.
        """
        lower, upper = self.predict_interval(X)
        y = np.asarray(y).ravel()

        covered = np.sum((y >= lower) & (y <= upper))
        return covered / len(y)

    def average_interval_width(self, X) -> float:
        """Compute average prediction interval width.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        float
            Average interval width.
        """
        lower, upper = self.predict_interval(X)
        return np.mean(upper - lower)

    def interval_width(self, X) -> np.ndarray:
        """Compute prediction interval widths for each sample.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        ndarray of shape (n_samples,)
            Interval width for each sample.
        """
        lower, upper = self.predict_interval(X)
        return upper - lower

    def score(self, X, y) -> float:
        """Return negative interval width (for sklearn compatibility).

        Higher is better (narrower intervals).

        Parameters
        ----------
        X : array-like
            Test features.
        y : array-like
            Test targets (unused, for API compatibility).

        Returns
        -------
        float
            Negative average interval width.
        """
        return -self.average_interval_width(X)

    def _check_is_fitted(self):
        """Check if the model is fitted."""
        if not hasattr(self, '_is_fitted') or not self._is_fitted:
            raise RuntimeError("ConformizedQuantileRegressor has not been fitted.")
