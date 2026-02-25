"""NGBoost wrapper for probabilistic prediction.

Wraps the Stanford NGBoost library (https://github.com/stanfordmlgroup/ngboost)
with competition-tuned defaults and additional utilities.

NGBoost uses Natural Gradient Boosting to produce full probability distributions
for predictions, enabling uncertainty quantification and probabilistic scoring.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.tree import DecisionTreeRegressor

from endgame.core.base import EndgameEstimator

# Check for ngboost availability
try:
    from ngboost import NGBClassifier, NGBRegressor
    from ngboost.distns import (
        Bernoulli,
        Cauchy,
        ClassificationDistn,
        Exponential,
        Laplace,
        LogNormal,
        MultivariateNormal,
        Normal,
        Poisson,
        T,
        TFixedDf,
        k_categorical,
    )
    from ngboost.scores import MLE, CRPScore, LogScore
    HAS_NGBOOST = True
except ImportError:
    HAS_NGBOOST = False


def _check_ngboost():
    """Check if ngboost is available."""
    if not HAS_NGBOOST:
        raise ImportError(
            "ngboost is required for NGBoost models. "
            "Install with: pip install ngboost"
        )


# Distribution name mappings
REGRESSION_DISTRIBUTIONS = {
    "normal": "Normal",
    "lognormal": "LogNormal",
    "exponential": "Exponential",
    "laplace": "Laplace",
    "t": "T",
    "cauchy": "Cauchy",
    "poisson": "Poisson",
}

CLASSIFICATION_DISTRIBUTIONS = {
    "bernoulli": "Bernoulli",
    "categorical": "k_categorical",
}

# Scoring rule mappings
SCORES = {
    "crps": "CRPScore",
    "mle": "MLE",
    "log": "LogScore",
    "nll": "LogScore",  # Alias
}

# Presets for different use cases
NGBOOST_PRESETS = {
    "endgame": {
        "n_estimators": 500,
        "learning_rate": 0.01,
        "minibatch_frac": 1.0,
        "col_sample": 1.0,
        "tol": 1e-4,
        "natural_gradient": True,
    },
    "fast": {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "minibatch_frac": 0.5,
        "col_sample": 0.8,
        "tol": 1e-3,
        "natural_gradient": True,
    },
    "accurate": {
        "n_estimators": 1000,
        "learning_rate": 0.005,
        "minibatch_frac": 1.0,
        "col_sample": 1.0,
        "tol": 1e-5,
        "natural_gradient": True,
    },
    "competition": {
        "n_estimators": 2000,
        "learning_rate": 0.01,
        "minibatch_frac": 1.0,
        "col_sample": 1.0,
        "tol": 1e-5,
        "natural_gradient": True,
    },
}


class NGBoostRegressor(EndgameEstimator, RegressorMixin):
    """NGBoost Regressor for probabilistic regression.

    Produces full probability distributions for predictions, enabling
    uncertainty quantification and scoring with proper scoring rules.

    Parameters
    ----------
    preset : str, default='endgame'
        Hyperparameter preset: 'endgame', 'fast', 'accurate', 'competition'.
    distribution : str, default='normal'
        Output distribution: 'normal', 'lognormal', 'exponential',
        'laplace', 't', 'cauchy', 'poisson'.
    score : str, default='crps'
        Scoring rule: 'crps' (Continuous Ranked Probability Score),
        'mle'/'nll' (Maximum Likelihood / Negative Log Likelihood).
    n_estimators : int, optional
        Number of boosting iterations. Overrides preset.
    learning_rate : float, optional
        Learning rate. Overrides preset.
    minibatch_frac : float, optional
        Fraction of data to use in each iteration.
    col_sample : float, optional
        Fraction of features to use in each iteration.
    base_learner : estimator, optional
        Base learner for boosting. Default is DecisionTreeRegressor(max_depth=3).
    natural_gradient : bool, default=True
        Use natural gradient (recommended).
    early_stopping_rounds : int, optional
        Early stopping patience. If None, no early stopping.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    **kwargs
        Additional parameters passed to NGBRegressor.

    Attributes
    ----------
    model_ : NGBRegressor
        Fitted NGBoost model.
    feature_importances_ : ndarray
        Feature importances from the base learners.

    Examples
    --------
    >>> from endgame.models import NGBoostRegressor
    >>> model = NGBoostRegressor(distribution='normal', score='crps')
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Point predictions
    >>> y_pred = model.predict(X_test)
    >>>
    >>> # Full distribution predictions
    >>> y_dist = model.pred_dist(X_test)
    >>> mean = y_dist.mean()
    >>> std = y_dist.std()
    >>>
    >>> # Prediction intervals
    >>> lower, upper = model.predict_interval(X_test, alpha=0.1)  # 90% CI
    >>>
    >>> # Negative log-likelihood
    >>> nll = -y_dist.logpdf(y_test).mean()

    References
    ----------
    Duan et al., 2020. "NGBoost: Natural Gradient Boosting for
    Probabilistic Prediction." https://arxiv.org/abs/1910.03225
    """

    def __init__(
        self,
        preset: str = "endgame",
        distribution: str = "normal",
        score: str = "crps",
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        minibatch_frac: float | None = None,
        col_sample: float | None = None,
        base_learner: BaseEstimator | None = None,
        natural_gradient: bool = True,
        early_stopping_rounds: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        _check_ngboost()
        super().__init__(random_state=random_state, verbose=verbose)

        self.preset = preset
        self.distribution = distribution
        self.score = score
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.base_learner = base_learner
        self.natural_gradient = natural_gradient
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs

        self.model_: NGBRegressor | None = None
        self._feature_names: list[str] | None = None

    def _get_distribution(self):
        """Get the distribution class."""
        dist_name = self.distribution.lower()

        if dist_name == "normal":
            return Normal
        elif dist_name == "lognormal":
            return LogNormal
        elif dist_name == "exponential":
            return Exponential
        elif dist_name == "laplace":
            return Laplace
        elif dist_name == "t":
            return T
        elif dist_name == "cauchy":
            return Cauchy
        elif dist_name == "poisson":
            return Poisson
        else:
            raise ValueError(
                f"Unknown distribution: {self.distribution}. "
                f"Choose from: {list(REGRESSION_DISTRIBUTIONS.keys())}"
            )

    def _get_score(self):
        """Get the scoring rule class."""
        score_name = self.score.lower()

        if score_name in ("crps", "crps_score"):
            return CRPScore
        elif score_name in ("mle", "nll", "log", "logscore"):
            return LogScore
        else:
            raise ValueError(
                f"Unknown score: {self.score}. "
                f"Choose from: {list(SCORES.keys())}"
            )

    def _get_params(self) -> dict[str, Any]:
        """Get merged parameters from preset and overrides."""
        # Start with preset
        params = NGBOOST_PRESETS.get(self.preset, NGBOOST_PRESETS["endgame"]).copy()

        # Apply overrides
        if self.n_estimators is not None:
            params["n_estimators"] = self.n_estimators
        if self.learning_rate is not None:
            params["learning_rate"] = self.learning_rate
        if self.minibatch_frac is not None:
            params["minibatch_frac"] = self.minibatch_frac
        if self.col_sample is not None:
            params["col_sample"] = self.col_sample

        params["natural_gradient"] = self.natural_gradient

        # Add distribution and score
        params["Dist"] = self._get_distribution()
        params["Score"] = self._get_score()

        # Add base learner
        if self.base_learner is not None:
            params["Base"] = self.base_learner
        else:
            # Default base learner with reasonable depth
            params["Base"] = DecisionTreeRegressor(
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
            )

        # Random state
        if self.random_state is not None:
            params["random_state"] = self.random_state

        # Verbose
        params["verbose"] = self.verbose

        # Additional kwargs
        params.update(self.kwargs)

        return params

    def fit(
        self,
        X,
        y,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ) -> "NGBoostRegressor":
        """Fit the NGBoost regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.
        X_val : array-like, optional
            Validation features for early stopping.
        y_val : array-like, optional
            Validation targets for early stopping.
        sample_weight : array-like, optional
            Training sample weights.
        val_sample_weight : array-like, optional
            Validation sample weights.

        Returns
        -------
        self
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y).ravel()

        # Store feature names
        self._feature_names = self._get_feature_names(X, X_arr.shape[1])

        # Get parameters
        params = self._get_params()

        # Create model
        self.model_ = NGBRegressor(**params)

        # Prepare fit arguments
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        # Early stopping
        if X_val is not None and y_val is not None:
            X_val_arr = self._to_numpy(X_val)
            y_val_arr = np.asarray(y_val).ravel()
            fit_kwargs["X_val"] = X_val_arr
            fit_kwargs["Y_val"] = y_val_arr
            if val_sample_weight is not None:
                fit_kwargs["val_sample_weight"] = val_sample_weight
            if self.early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._log(f"Training NGBoost regressor with {len(X_arr)} samples...")
        self.model_.fit(X_arr, y_arr, **fit_kwargs)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict the mean of the distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted means.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.predict(X_arr)

    def pred_dist(self, X):
        """Predict the full distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        dist : ngboost distribution
            Predicted distributions with methods:
            - mean(): Expected value
            - std(): Standard deviation
            - var(): Variance
            - logpdf(y): Log probability density
            - pdf(y): Probability density
            - cdf(y): Cumulative distribution function
            - ppf(q): Percent point function (inverse CDF)
            - sample(n): Draw n samples
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.pred_dist(X_arr)

    def predict_interval(
        self,
        X,
        alpha: float = 0.1,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict prediction intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        alpha : float, default=0.1
            Significance level. Returns (1-alpha) prediction interval.
            E.g., alpha=0.1 returns 90% prediction interval.

        Returns
        -------
        lower : ndarray of shape (n_samples,)
            Lower bound of prediction interval.
        upper : ndarray of shape (n_samples,)
            Upper bound of prediction interval.
        """
        dist = self.pred_dist(X)
        lower = dist.ppf(alpha / 2)
        upper = dist.ppf(1 - alpha / 2)
        return lower, upper

    def predict_std(self, X) -> np.ndarray:
        """Predict the standard deviation (uncertainty).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        std : ndarray of shape (n_samples,)
            Predicted standard deviations.
        """
        dist = self.pred_dist(X)
        return dist.std()

    def score(self, X, y, sample_weight=None) -> float:
        """Return the negative log-likelihood on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True target values.
        sample_weight : array-like, optional
            Sample weights (not used, for API compatibility).

        Returns
        -------
        score : float
            Mean negative log-likelihood (lower is better).
        """
        self._check_is_fitted()
        dist = self.pred_dist(X)
        y_arr = np.asarray(y).ravel()
        # Return negative NLL so higher is better (sklearn convention)
        return dist.logpdf(y_arr).mean()

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on base learner splits."""
        self._check_is_fitted()
        return self.model_.feature_importances_

    def _get_feature_names(self, X, n_features: int) -> list[str]:
        """Extract feature names from input."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return list(X.columns)
        except ImportError:
            pass

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                return list(X.columns)
        except ImportError:
            pass

        return [f"f{i}" for i in range(n_features)]


class NGBoostClassifier(ClassifierMixin, EndgameEstimator):
    """NGBoost Classifier for probabilistic classification.

    Produces calibrated probability distributions over classes,
    with proper uncertainty quantification.

    Parameters
    ----------
    preset : str, default='endgame'
        Hyperparameter preset: 'endgame', 'fast', 'accurate', 'competition'.
    n_estimators : int, optional
        Number of boosting iterations. Overrides preset.
    learning_rate : float, optional
        Learning rate. Overrides preset.
    minibatch_frac : float, optional
        Fraction of data to use in each iteration.
    col_sample : float, optional
        Fraction of features to use in each iteration.
    base_learner : estimator, optional
        Base learner for boosting. Default is DecisionTreeRegressor(max_depth=3).
    natural_gradient : bool, default=True
        Use natural gradient (recommended).
    early_stopping_rounds : int, optional
        Early stopping patience. If None, no early stopping.
    random_state : int, optional
        Random seed.
    verbose : bool, default=False
        Enable verbose output.
    **kwargs
        Additional parameters passed to NGBClassifier.

    Attributes
    ----------
    model_ : NGBClassifier
        Fitted NGBoost model.
    classes_ : ndarray
        Unique class labels.
    n_classes_ : int
        Number of classes.
    feature_importances_ : ndarray
        Feature importances from the base learners.

    Examples
    --------
    >>> from endgame.models import NGBoostClassifier
    >>> model = NGBoostClassifier(preset='endgame')
    >>> model.fit(X_train, y_train)
    >>>
    >>> # Class predictions
    >>> y_pred = model.predict(X_test)
    >>>
    >>> # Probability predictions
    >>> y_proba = model.predict_proba(X_test)
    >>>
    >>> # Distribution predictions
    >>> y_dist = model.pred_dist(X_test)
    >>>
    >>> # Log-loss
    >>> from sklearn.metrics import log_loss
    >>> loss = log_loss(y_test, y_proba)

    References
    ----------
    Duan et al., 2020. "NGBoost: Natural Gradient Boosting for
    Probabilistic Prediction." https://arxiv.org/abs/1910.03225
    """

    def __init__(
        self,
        preset: str = "endgame",
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        minibatch_frac: float | None = None,
        col_sample: float | None = None,
        base_learner: BaseEstimator | None = None,
        natural_gradient: bool = True,
        early_stopping_rounds: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        _check_ngboost()
        super().__init__(random_state=random_state, verbose=verbose)

        self.preset = preset
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.minibatch_frac = minibatch_frac
        self.col_sample = col_sample
        self.base_learner = base_learner
        self.natural_gradient = natural_gradient
        self.early_stopping_rounds = early_stopping_rounds
        self.kwargs = kwargs

        self.model_: NGBClassifier | None = None
        self.classes_: np.ndarray | None = None
        self.n_classes_: int | None = None
        self._feature_names: list[str] | None = None

    def _get_params(self, n_classes: int) -> dict[str, Any]:
        """Get merged parameters from preset and overrides."""
        # Start with preset
        params = NGBOOST_PRESETS.get(self.preset, NGBOOST_PRESETS["endgame"]).copy()

        # Apply overrides
        if self.n_estimators is not None:
            params["n_estimators"] = self.n_estimators
        if self.learning_rate is not None:
            params["learning_rate"] = self.learning_rate
        if self.minibatch_frac is not None:
            params["minibatch_frac"] = self.minibatch_frac
        if self.col_sample is not None:
            params["col_sample"] = self.col_sample

        params["natural_gradient"] = self.natural_gradient

        # Set distribution based on number of classes
        if n_classes == 2:
            params["Dist"] = Bernoulli
        else:
            params["Dist"] = k_categorical(n_classes)

        # Score is always LogScore for classification
        params["Score"] = LogScore

        # Add base learner
        if self.base_learner is not None:
            params["Base"] = self.base_learner
        else:
            params["Base"] = DecisionTreeRegressor(
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
            )

        # Random state
        if self.random_state is not None:
            params["random_state"] = self.random_state

        # Verbose
        params["verbose"] = self.verbose

        # Additional kwargs
        params.update(self.kwargs)

        return params

    def fit(
        self,
        X,
        y,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        val_sample_weight: np.ndarray | None = None,
    ) -> "NGBoostClassifier":
        """Fit the NGBoost classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels.
        X_val : array-like, optional
            Validation features for early stopping.
        y_val : array-like, optional
            Validation labels for early stopping.
        sample_weight : array-like, optional
            Training sample weights.
        val_sample_weight : array-like, optional
            Validation sample weights.

        Returns
        -------
        self
        """
        X_arr = self._to_numpy(X)
        y_arr = np.asarray(y).ravel()

        # Store classes
        self.classes_ = np.unique(y_arr)
        self.n_classes_ = len(self.classes_)

        # Remap labels to contiguous 0..n-1 (required by ngboost k_categorical)
        self._label_remap = None
        if not np.array_equal(self.classes_, np.arange(self.n_classes_)):
            self._label_remap = {c: i for i, c in enumerate(self.classes_)}
            y_arr = np.array([self._label_remap[v] for v in y_arr])

        # Store feature names
        self._feature_names = self._get_feature_names(X, X_arr.shape[1])

        # Get parameters
        params = self._get_params(self.n_classes_)

        # Create model
        self.model_ = NGBClassifier(**params)

        # Prepare fit arguments
        fit_kwargs = {}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight

        # Early stopping
        if X_val is not None and y_val is not None:
            X_val_arr = self._to_numpy(X_val)
            y_val_arr = np.asarray(y_val).ravel()
            if self._label_remap is not None:
                y_val_arr = np.array([self._label_remap.get(v, v) for v in y_val_arr])
            fit_kwargs["X_val"] = X_val_arr
            fit_kwargs["Y_val"] = y_val_arr
            if val_sample_weight is not None:
                fit_kwargs["val_sample_weight"] = val_sample_weight
            if self.early_stopping_rounds is not None:
                fit_kwargs["early_stopping_rounds"] = self.early_stopping_rounds

        self._log(f"Training NGBoost classifier with {len(X_arr)} samples, {self.n_classes_} classes...")
        self.model_.fit(X_arr, y_arr, **fit_kwargs)

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
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        preds = self.model_.predict(X_arr)
        if self._label_remap is not None:
            preds = self.classes_[preds.astype(int)]
        return preds

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
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.predict_proba(X_arr)

    def pred_dist(self, X):
        """Predict the full distribution over classes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        dist : ngboost distribution
            Predicted distributions.
        """
        self._check_is_fitted()
        X_arr = self._to_numpy(X)
        return self.model_.pred_dist(X_arr)

    def score(self, X, y, sample_weight=None) -> float:
        """Return accuracy on the given data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        y : array-like of shape (n_samples,)
            True labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        score : float
            Accuracy score.
        """
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred, sample_weight=sample_weight)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances based on base learner splits."""
        self._check_is_fitted()
        return self.model_.feature_importances_

    def _get_feature_names(self, X, n_features: int) -> list[str]:
        """Extract feature names from input."""
        try:
            import pandas as pd
            if isinstance(X, pd.DataFrame):
                return list(X.columns)
        except ImportError:
            pass

        try:
            import polars as pl
            if isinstance(X, (pl.DataFrame, pl.LazyFrame)):
                if isinstance(X, pl.LazyFrame):
                    X = X.collect()
                return list(X.columns)
        except ImportError:
            pass

        return [f"f{i}" for i in range(n_features)]


# Convenience function for survival analysis (if needed in future)
def create_ngboost_survival(
    distribution: str = "exponential",
    **kwargs
) -> NGBoostRegressor:
    """Create an NGBoost model for survival analysis.

    Parameters
    ----------
    distribution : str, default='exponential'
        Distribution for survival times: 'exponential', 'lognormal'.
    **kwargs
        Additional parameters for NGBoostRegressor.

    Returns
    -------
    model : NGBoostRegressor
        Configured NGBoost model for survival analysis.

    Notes
    -----
    For survival analysis with censoring, use the ngboost library
    directly with NGBSurvival class.
    """
    return NGBoostRegressor(
        distribution=distribution,
        score="mle",
        **kwargs
    )
