from __future__ import annotations

"""Probability calibration methods.

Methods for calibrating classifier probabilities to be more reliable.
Well-calibrated probabilities satisfy: P(Y=1 | P_pred = p) ≈ p

References
----------
- Platt "Probabilistic Outputs for SVMs" (1999)
- Guo et al. "On Calibration of Modern Neural Networks" (2017)
- Kull et al. "Beta calibration" (2017)
"""


import numpy as np
from scipy import optimize
from scipy.special import expit, logit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.isotonic import IsotonicRegression


class TemperatureScaling(BaseEstimator, TransformerMixin):
    """Temperature scaling for neural network calibration.

    Learns a single temperature parameter T to scale logits:
    calibrated_proba = softmax(logits / T)

    This is a simple but effective method for calibrating neural networks,
    particularly when the model is already reasonably calibrated.

    Parameters
    ----------
    method : str, default='nll'
        Optimization objective:
        - 'nll': Negative log-likelihood (cross-entropy)
        - 'ece': Expected Calibration Error
    max_iter : int, default=100
        Maximum optimization iterations.

    Attributes
    ----------
    temperature_ : float
        Learned temperature parameter.

    Examples
    --------
    >>> ts = TemperatureScaling()
    >>> ts.fit(logits_val, y_val)
    >>> calibrated = ts.transform(logits_test)
    """

    def __init__(
        self,
        method: str = "nll",
        max_iter: int = 100,
    ):
        self.method = method
        self.max_iter = max_iter
        self.temperature_: float = 1.0

    def fit(self, logits, y) -> TemperatureScaling:
        """Fit temperature parameter on validation data.

        Parameters
        ----------
        logits : array-like of shape (n_samples, n_classes)
            Raw logits (pre-softmax outputs).
        y : array-like of shape (n_samples,)
            True class labels.

        Returns
        -------
        self
        """
        logits = np.asarray(logits)
        y = np.asarray(y)

        if logits.ndim == 1:
            # Binary classification - convert to 2-class
            logits = np.column_stack([-logits / 2, logits / 2])

        if self.method == "nll":
            # Optimize negative log-likelihood
            def nll(T):
                scaled = logits / T[0]
                log_proba = scaled - np.log(np.sum(np.exp(scaled), axis=1, keepdims=True))
                return -np.mean(log_proba[np.arange(len(y)), y])

            result = optimize.minimize(
                nll,
                x0=[1.0],
                bounds=[(0.01, 10.0)],
                method='L-BFGS-B',
                options={'maxiter': self.max_iter},
            )
            self.temperature_ = result.x[0]

        elif self.method == "ece":
            # Optimize ECE directly
            from endgame.calibration.analysis import expected_calibration_error

            def ece_loss(T):
                scaled = logits / T[0]
                proba = np.exp(scaled) / np.sum(np.exp(scaled), axis=1, keepdims=True)
                return expected_calibration_error(y, proba[:, 1] if proba.shape[1] == 2 else proba)

            result = optimize.minimize(
                ece_loss,
                x0=[1.0],
                bounds=[(0.01, 10.0)],
                method='L-BFGS-B',
                options={'maxiter': self.max_iter},
            )
            self.temperature_ = result.x[0]
        else:
            raise ValueError(f"Unknown method: {self.method}")

        return self

    def transform(self, logits) -> np.ndarray:
        """Apply temperature scaling to logits.

        Parameters
        ----------
        logits : array-like
            Raw logits.

        Returns
        -------
        ndarray
            Calibrated probabilities.
        """
        logits = np.asarray(logits)

        if logits.ndim == 1:
            logits = np.column_stack([-logits / 2, logits / 2])

        scaled = logits / self.temperature_
        proba = np.exp(scaled) / np.sum(np.exp(scaled), axis=1, keepdims=True)

        return proba

    def fit_transform(self, logits, y) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(logits, y).transform(logits)


class PlattScaling(BaseEstimator, TransformerMixin):
    """Platt scaling (sigmoid calibration) for binary classification.

    Fits logistic regression: P(y=1|f) = 1 / (1 + exp(A*f + B))

    Parameters
    ----------
    prior_correction : bool, default=True
        Apply prior correction for imbalanced datasets.
        Uses Platt's method with adjusted target probabilities.
    max_iter : int, default=100
        Maximum optimization iterations.

    Attributes
    ----------
    A_ : float
        Learned slope parameter.
    B_ : float
        Learned intercept parameter.

    Examples
    --------
    >>> platt = PlattScaling()
    >>> platt.fit(scores_val, y_val)
    >>> calibrated = platt.transform(scores_test)
    """

    def __init__(
        self,
        prior_correction: bool = True,
        max_iter: int = 100,
    ):
        self.prior_correction = prior_correction
        self.max_iter = max_iter
        self.A_: float = 0.0
        self.B_: float = 0.0

    def fit(self, scores, y) -> PlattScaling:
        """Fit Platt scaling parameters.

        Parameters
        ----------
        scores : array-like of shape (n_samples,)
            Raw scores or decision function values.
        y : array-like of shape (n_samples,)
            Binary labels (0 or 1).

        Returns
        -------
        self
        """
        scores = np.asarray(scores).ravel()
        y = np.asarray(y).ravel()

        # Prior correction (Platt's algorithm)
        if self.prior_correction:
            n_pos = np.sum(y == 1)
            n_neg = np.sum(y == 0)
            # Target probabilities with prior correction
            t_pos = (n_pos + 1) / (n_pos + 2)
            t_neg = 1 / (n_neg + 2)
            targets = np.where(y == 1, t_pos, t_neg)
        else:
            targets = y.astype(float)

        # Optimize cross-entropy loss
        def neg_log_likelihood(params):
            A, B = params
            p = expit(A * scores + B)
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return -np.mean(targets * np.log(p) + (1 - targets) * np.log(1 - p))

        result = optimize.minimize(
            neg_log_likelihood,
            x0=[0.0, 0.0],
            method='L-BFGS-B',
            options={'maxiter': self.max_iter},
        )

        self.A_, self.B_ = result.x
        return self

    def transform(self, scores) -> np.ndarray:
        """Apply Platt scaling.

        Parameters
        ----------
        scores : array-like
            Raw scores.

        Returns
        -------
        ndarray
            Calibrated probabilities.
        """
        scores = np.asarray(scores).ravel()
        return expit(self.A_ * scores + self.B_)

    def fit_transform(self, scores, y) -> np.ndarray:
        """Fit and transform."""
        return self.fit(scores, y).transform(scores)


class BetaCalibration(BaseEstimator, TransformerMixin):
    """Beta calibration for improved probability estimates.

    More flexible than Platt scaling, handles different miscalibration patterns.

    Fits: calibrated = 1 / (1 + 1/exp(c*log(p/(1-p)) + d*log(p) + e*log(1-p)))

    Can be simplified to three-parameter form:
    calibrated = 1 / (1 + exp(-(a*logit(p) + b)))

    Parameters
    ----------
    parameters : str, default='abm'
        Parameterization:
        - 'abm': Three parameters (a, b, m) - most common
        - 'full': Five parameters (more flexible, may overfit)

    Attributes
    ----------
    a_, b_, m_ : float
        Learned parameters (abm mode).

    References
    ----------
    Kull et al. "Beta calibration: a well-founded and easily implemented
    improvement on logistic calibration for binary classifiers" (2017)

    Examples
    --------
    >>> beta_cal = BetaCalibration()
    >>> beta_cal.fit(proba_val, y_val)
    >>> calibrated = beta_cal.transform(proba_test)
    """

    def __init__(self, parameters: str = "abm"):
        self.parameters = parameters
        self.a_: float = 1.0
        self.b_: float = 0.0
        self.m_: float = 0.5

    def fit(self, proba, y) -> BetaCalibration:
        """Fit beta calibration parameters.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities for positive class.
        y : array-like
            Binary labels.

        Returns
        -------
        self
        """
        proba = np.asarray(proba).ravel()
        y = np.asarray(y).ravel()

        # Clip probabilities to avoid numerical issues
        proba = np.clip(proba, 1e-10, 1 - 1e-10)

        # Log-odds transformation
        log_odds = logit(proba)

        # ABM parameterization: sigmoid(a * logit(p) + b) where logit uses base m
        if self.parameters == "abm":
            def neg_log_likelihood(params):
                a, b, m = params
                # Transform to logit with base m
                # logit_m(p) = log(p^m / (1-p)^(1-m)) = m*log(p) - (1-m)*log(1-p)
                logit_m = m * np.log(proba) - (1 - m) * np.log(1 - proba)
                cal = expit(a * logit_m + b)
                cal = np.clip(cal, 1e-10, 1 - 1e-10)
                return -np.mean(y * np.log(cal) + (1 - y) * np.log(1 - cal))

            result = optimize.minimize(
                neg_log_likelihood,
                x0=[1.0, 0.0, 0.5],
                bounds=[(0.01, 100), (-10, 10), (0.01, 0.99)],
                method='L-BFGS-B',
            )
            self.a_, self.b_, self.m_ = result.x

        return self

    def transform(self, proba) -> np.ndarray:
        """Apply beta calibration.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities.

        Returns
        -------
        ndarray
            Calibrated probabilities.
        """
        proba = np.asarray(proba).ravel()
        proba = np.clip(proba, 1e-10, 1 - 1e-10)

        logit_m = self.m_ * np.log(proba) - (1 - self.m_) * np.log(1 - proba)
        return expit(self.a_ * logit_m + self.b_)

    def fit_transform(self, proba, y) -> np.ndarray:
        """Fit and transform."""
        return self.fit(proba, y).transform(proba)


class IsotonicCalibration(BaseEstimator, TransformerMixin):
    """Isotonic regression calibration.

    Non-parametric calibration that preserves ranking.
    Fits a monotonically increasing step function mapping
    predicted probabilities to calibrated probabilities.

    Best for large calibration sets (>1000 samples) where
    the flexibility doesn't lead to overfitting.

    Parameters
    ----------
    out_of_bounds : str, default='clip'
        How to handle predictions outside training range:
        - 'clip': Clip to [min, max] of training range
        - 'nan': Return NaN for out-of-bounds

    Attributes
    ----------
    isotonic_ : IsotonicRegression
        Fitted isotonic regression model.

    Examples
    --------
    >>> iso = IsotonicCalibration()
    >>> iso.fit(proba_val, y_val)
    >>> calibrated = iso.transform(proba_test)
    """

    def __init__(self, out_of_bounds: str = "clip"):
        self.out_of_bounds = out_of_bounds
        self.isotonic_: IsotonicRegression | None = None

    def fit(self, proba, y) -> IsotonicCalibration:
        """Fit isotonic regression.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities.
        y : array-like
            Binary labels.

        Returns
        -------
        self
        """
        proba = np.asarray(proba).ravel()
        y = np.asarray(y).ravel()

        self.isotonic_ = IsotonicRegression(
            y_min=0,
            y_max=1,
            out_of_bounds=self.out_of_bounds,
        )
        self.isotonic_.fit(proba, y)

        return self

    def transform(self, proba) -> np.ndarray:
        """Apply isotonic calibration.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities.

        Returns
        -------
        ndarray
            Calibrated probabilities.
        """
        if self.isotonic_ is None:
            raise RuntimeError("IsotonicCalibration has not been fitted.")

        proba = np.asarray(proba).ravel()
        return self.isotonic_.transform(proba)

    def fit_transform(self, proba, y) -> np.ndarray:
        """Fit and transform."""
        return self.fit(proba, y).transform(proba)


class HistogramBinning(BaseEstimator, TransformerMixin):
    """Histogram binning calibration.

    Divides probability space into bins and maps each bin to
    the empirical frequency of positives within that bin.

    Simple and interpretable, but can be unreliable with few samples.

    Parameters
    ----------
    n_bins : int, default=10
        Number of bins.
    strategy : str, default='uniform'
        Binning strategy:
        - 'uniform': Equal-width bins
        - 'quantile': Equal-frequency bins

    Attributes
    ----------
    bin_edges_ : ndarray
        Edges of calibration bins.
    bin_calibrations_ : ndarray
        Calibrated probability for each bin.

    Examples
    --------
    >>> hb = HistogramBinning(n_bins=15)
    >>> hb.fit(proba_val, y_val)
    >>> calibrated = hb.transform(proba_test)
    """

    def __init__(
        self,
        n_bins: int = 10,
        strategy: str = "uniform",
    ):
        self.n_bins = n_bins
        self.strategy = strategy
        self.bin_edges_: np.ndarray | None = None
        self.bin_calibrations_: np.ndarray | None = None

    def fit(self, proba, y) -> HistogramBinning:
        """Fit histogram binning.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities.
        y : array-like
            Binary labels.

        Returns
        -------
        self
        """
        proba = np.asarray(proba).ravel()
        y = np.asarray(y).ravel()

        if self.strategy == "uniform":
            self.bin_edges_ = np.linspace(0, 1, self.n_bins + 1)
        elif self.strategy == "quantile":
            self.bin_edges_ = np.percentile(
                proba,
                np.linspace(0, 100, self.n_bins + 1)
            )
            # Ensure edges are unique and sorted
            self.bin_edges_ = np.unique(self.bin_edges_)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        # Compute calibrated probability for each bin
        self.bin_calibrations_ = np.zeros(len(self.bin_edges_) - 1)

        for i in range(len(self.bin_edges_) - 1):
            mask = (proba >= self.bin_edges_[i]) & (proba < self.bin_edges_[i + 1])
            if i == len(self.bin_edges_) - 2:
                # Include right edge for last bin
                mask = mask | (proba == self.bin_edges_[i + 1])

            if np.sum(mask) > 0:
                self.bin_calibrations_[i] = np.mean(y[mask])
            else:
                # No samples in bin - use bin midpoint
                self.bin_calibrations_[i] = (self.bin_edges_[i] + self.bin_edges_[i + 1]) / 2

        return self

    def transform(self, proba) -> np.ndarray:
        """Apply histogram binning calibration.

        Parameters
        ----------
        proba : array-like
            Predicted probabilities.

        Returns
        -------
        ndarray
            Calibrated probabilities.
        """
        if self.bin_edges_ is None:
            raise RuntimeError("HistogramBinning has not been fitted.")

        proba = np.asarray(proba).ravel()

        # Find bin for each probability
        bin_indices = np.digitize(proba, self.bin_edges_) - 1
        bin_indices = np.clip(bin_indices, 0, len(self.bin_calibrations_) - 1)

        return self.bin_calibrations_[bin_indices]

    def fit_transform(self, proba, y) -> np.ndarray:
        """Fit and transform."""
        return self.fit(proba, y).transform(proba)
