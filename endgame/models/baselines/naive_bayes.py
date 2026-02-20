"""Naive Bayes classifier with automatic variant selection.

Naive Bayes assumes feature independence - a strong assumption that's
usually wrong, but often works surprisingly well. This different
inductive bias makes it valuable for ensemble diversity.

References
----------
- McCallum & Nigam, "A Comparison of Event Models for Naive Bayes Text Classification" (1998)
- sklearn.naive_bayes documentation
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder


class NaiveBayesClassifier(ClassifierMixin, BaseEstimator):
    """Naive Bayes Classifier with automatic variant selection.

    Automatically selects the appropriate Naive Bayes variant based on
    feature characteristics, or uses a specified variant.

    The feature independence assumption is fundamentally different from
    tree-based models (which capture interactions) and neural networks
    (which learn complex dependencies), making this valuable for
    ensemble diversity.

    Parameters
    ----------
    variant : str, default='auto'
        Naive Bayes variant:
        - 'auto': Automatically select based on features
        - 'gaussian': For continuous features
        - 'bernoulli': For binary features
        - 'multinomial': For count/frequency features
        - 'complement': For imbalanced text classification
    var_smoothing : float, default=1e-9
        Portion of the largest variance of all features added to
        variances for stability (Gaussian only).
    alpha : float, default=1.0
        Additive smoothing parameter (Bernoulli, Multinomial, Complement).
    binarize : float or None, default=0.0
        Threshold for binarizing features (Bernoulli only).
        None means features are already binary.
    fit_prior : bool, default=True
        Whether to learn class prior probabilities.
    class_prior : array-like, optional
        Prior probabilities of the classes.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    variant_ : str
        The actual variant used (resolved from 'auto').
    model_ : sklearn NB estimator
        Fitted Naive Bayes model.

    Examples
    --------
    >>> from endgame.models.baselines import NaiveBayesClassifier
    >>> clf = NaiveBayesClassifier(variant='auto')
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    Despite the naive independence assumption, Naive Bayes often works
    surprisingly well because:
    1. Classification only requires correct ordering, not accurate probabilities
    2. Dependencies often "cancel out" when aggregated
    3. Regularization effect from the strong prior

    For ensembles, NB provides diversity because it makes fundamentally
    different errors from models that capture feature interactions.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        variant: str = "auto",
        var_smoothing: float = 1e-9,
        alpha: float = 1.0,
        binarize: float | None = 0.0,
        fit_prior: bool = True,
        class_prior: np.ndarray | None = None,
    ):
        self.variant = variant
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.binarize = binarize
        self.fit_prior = fit_prior
        self.class_prior = class_prior

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.variant_: str | None = None
        self.model_: Any | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def _detect_variant(self, X: np.ndarray) -> str:
        """Automatically detect the best NB variant based on features.

        Decision logic:
        - If all values are 0/1 -> Bernoulli
        - If all values are non-negative integers -> Multinomial
        - Otherwise -> Gaussian
        """
        # Check if binary (0/1 only)
        unique_vals = np.unique(X[~np.isnan(X)])
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1}):
            return "bernoulli"

        # Check if non-negative integers (count data)
        if np.all(X >= 0) and np.allclose(X, X.astype(int)):
            return "multinomial"

        # Default to Gaussian for continuous features
        return "gaussian"

    def _create_model(self):
        """Create the appropriate Naive Bayes model."""
        if self.variant_ == "gaussian":
            return GaussianNB(
                var_smoothing=self.var_smoothing,
                priors=self.class_prior,
            )
        elif self.variant_ == "bernoulli":
            return BernoulliNB(
                alpha=self.alpha,
                binarize=self.binarize,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
            )
        elif self.variant_ == "multinomial":
            return MultinomialNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
            )
        elif self.variant_ == "complement":
            return ComplementNB(
                alpha=self.alpha,
                fit_prior=self.fit_prior,
                class_prior=self.class_prior,
                norm=True,
            )
        else:
            raise ValueError(f"Unknown variant: {self.variant_}. "
                           "Options: 'auto', 'gaussian', 'bernoulli', "
                           "'multinomial', 'complement'")

    def fit(self, X, y, sample_weight=None, **fit_params) -> "NaiveBayesClassifier":
        """Fit the Naive Bayes classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        # Handle NaN - replace with column means for Gaussian
        X_clean = X.copy()
        if np.any(np.isnan(X_clean)):
            col_means = np.nanmean(X_clean, axis=0)
            for i in range(X_clean.shape[1]):
                mask = np.isnan(X_clean[:, i])
                X_clean[mask, i] = col_means[i]

        # Determine variant
        if self.variant == "auto":
            self.variant_ = self._detect_variant(X_clean)
        else:
            self.variant_ = self.variant

        # Handle negative values for multinomial
        if self.variant_ == "multinomial" and np.any(X_clean < 0):
            # Shift to non-negative
            X_clean = X_clean - X_clean.min(axis=0) + 1e-10

        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X_clean, y_encoded, sample_weight=sample_weight)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        if not self._is_fitted:
            raise RuntimeError("NaiveBayesClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        # Handle multinomial negative values
        if self.variant_ == "multinomial" and np.any(X_clean < 0):
            X_clean = X_clean - X_clean.min(axis=0) + 1e-10

        y_pred = self.model_.predict(X_clean)
        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("NaiveBayesClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        # Handle multinomial negative values
        if self.variant_ == "multinomial" and np.any(X_clean < 0):
            X_clean = X_clean - X_clean.min(axis=0) + 1e-10

        return self.model_.predict_proba(X_clean)

    def predict_log_proba(self, X) -> np.ndarray:
        """Predict log class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        log_proba : ndarray of shape (n_samples, n_classes)
            Log class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("NaiveBayesClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_clean = np.nan_to_num(X, nan=0.0)

        if self.variant_ == "multinomial" and np.any(X_clean < 0):
            X_clean = X_clean - X_clean.min(axis=0) + 1e-10

        return self.model_.predict_log_proba(X_clean)

    @property
    def feature_log_prob_(self):
        """Log probability of features given a class (for discrete NB)."""
        if not self._is_fitted:
            raise RuntimeError("NaiveBayesClassifier has not been fitted.")
        if hasattr(self.model_, 'feature_log_prob_'):
            return self.model_.feature_log_prob_
        return None

    @property
    def class_log_prior_(self):
        """Log probability of each class."""
        if not self._is_fitted:
            raise RuntimeError("NaiveBayesClassifier has not been fitted.")
        if hasattr(self.model_, 'class_log_prior_'):
            return self.model_.class_log_prior_
        elif hasattr(self.model_, 'class_prior_'):
            return np.log(self.model_.class_prior_)
        return None
