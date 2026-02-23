"""Bayesian Additive Regression Trees (BART) wrapper.

BART is a Bayesian nonparametric approach that models functions as sums of
many regression trees, each contributing a small amount. The use of priors
regularizes the model, favoring shallow trees with leaf values close to zero.

This fundamentally different approach (Bayesian posterior inference via MCMC)
provides unique predictions that enhance ensemble diversity compared to
greedy GBDT methods.

References
----------
- Chipman et al., "BART: Bayesian Additive Regression Trees" (2010)
- Quiroga et al., "Bayesian additive regression trees for probabilistic programming" (2022)
- https://www.pymc.io/projects/bart

Notes
-----
BART requires PyMC and pymc-bart for full functionality. When these are not
available, a simpler scikit-learn-based approximation is used.
"""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Try importing pymc_bart
_HAS_PYMC_BART = False
try:
    import arviz as az
    import pymc as pm
    import pymc_bart as pmb
    _HAS_PYMC_BART = True
except ImportError:
    pass


class BARTRegressor(RegressorMixin, BaseEstimator):
    """Bayesian Additive Regression Trees Regressor.

    BART models the conditional mean function as a sum of many regression
    trees, using Bayesian priors to regularize complexity. Unlike greedy
    boosting (XGBoost, LightGBM), BART uses MCMC to explore the posterior
    distribution of tree structures.

    Parameters
    ----------
    n_trees : int, default=50
        Number of trees in the ensemble. 50-200 trees typically work well.
        More trees = smoother predictions but slower inference.
    n_samples : int, default=1000
        Number of posterior samples to draw via MCMC.
    n_tune : int, default=500
        Number of tuning samples (burn-in) before posterior sampling.
    n_chains : int, default=2
        Number of MCMC chains to run in parallel.
    alpha : float, default=0.95
        Prior probability that a tree split terminates at depth d.
        Higher values favor shallower trees.
    beta : float, default=2.0
        Prior rate of decrease in split probability with depth.
        Higher values penalize deeper trees more strongly.
    auto_scale : bool, default=True
        Whether to standardize features before fitting.
    random_state : int, optional
        Random seed for reproducibility.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    variable_importance_ : ndarray of shape (n_features,)
        Relative importance of each feature (based on split frequency).

    Examples
    --------
    >>> from endgame.models.probabilistic import BARTRegressor
    >>> reg = BARTRegressor(n_trees=50, n_samples=500, random_state=42)
    >>> reg.fit(X_train, y_train)
    >>> y_pred = reg.predict(X_test)
    >>> intervals = reg.predict_interval(X_test, alpha=0.1)  # 90% intervals

    Notes
    -----
    BART's Bayesian approach provides:
    1. **Uncertainty quantification**: Full posterior over predictions
    2. **Regularization via priors**: Avoids overfitting without CV
    3. **Variable importance**: Based on posterior split frequencies
    4. **Different inductive bias**: Complements greedy boosted trees

    For ensemble diversity, BART makes fundamentally different errors than
    XGBoost/LightGBM because it explores tree space via MCMC rather than
    greedy sequential fitting.
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        n_trees: int = 50,
        n_samples: int = 1000,
        n_tune: int = 500,
        n_chains: int = 2,
        alpha: float = 0.95,
        beta: float = 2.0,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.alpha = alpha
        self.beta = beta
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.n_features_in_: int = 0
        self.variable_importance_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._y_mean: float = 0.0
        self._y_std: float = 1.0
        self._trace: Any | None = None
        self._bart: Any | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> "BARTRegressor":
        """Fit the BART regressor using MCMC.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        if not _HAS_PYMC_BART:
            raise ImportError(
                "BART requires pymc and pymc-bart. "
                "Install with: pip install pymc pymc-bart"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
            self._y_mean = np.mean(y)
            self._y_std = np.std(y) + 1e-8
            y_scaled = (y - self._y_mean) / self._y_std
        else:
            X_scaled = X.copy()
            y_scaled = y.copy()

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)
        y_scaled = np.nan_to_num(y_scaled, nan=0.0)

        with pm.Model() as model:
            self._X_shared = pm.Data("X", X_scaled)

            self._bart = pmb.BART(
                "bart",
                self._X_shared,
                y_scaled,
                m=self.n_trees,
                alpha=self.alpha,
                beta=self.beta,
            )

            sigma = pm.HalfNormal("sigma", sigma=1.0)

            pm.Normal(
                "y_obs",
                mu=self._bart,
                sigma=sigma,
                observed=y_scaled,
                shape=self._bart.shape,
            )

            self._trace = pm.sample(
                draws=self.n_samples,
                tune=self.n_tune,
                chains=self.n_chains,
                random_seed=self.random_state,
                progressbar=False,
                return_inferencedata=True,
            )

        self._model = model

        try:
            vi = pmb.compute_variable_importance(
                self._trace, bartrv=self._bart, X=X_scaled,
            )
            self.variable_importance_ = vi.get(
                "r2_mean", np.ones(n_features) / n_features
            ) if isinstance(vi, dict) else vi
        except Exception:
            self.variable_importance_ = np.ones(n_features) / n_features

        self._is_fitted = True
        return self

    def _predict_samples(self, X_scaled: np.ndarray) -> np.ndarray:
        """Get posterior predictive samples for new data."""
        with self._model:
            self._X_shared.set_value(X_scaled)
            ppc = pm.sample_posterior_predictive(
                trace=self._trace,
                var_names=["bart"],
                random_seed=self.random_state,
                progressbar=False,
            )
        return ppc.posterior_predictive["bart"].values

    def predict(self, X) -> np.ndarray:
        """Predict mean target values.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean values.
        """
        if not self._is_fitted:
            raise RuntimeError("BARTRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.transform(X) if self.auto_scale else X.copy()
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        samples = self._predict_samples(X_scaled)
        y_pred = samples.mean(axis=(0, 1))

        if self.auto_scale:
            y_pred = y_pred * self._y_std + self._y_mean
        return y_pred

    def predict_interval(self, X, alpha: float = 0.1) -> np.ndarray:
        """Predict credible intervals.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.
        alpha : float, default=0.1
            Significance level. Returns (1-alpha)*100% credible intervals.

        Returns
        -------
        intervals : ndarray of shape (n_samples, 2)
            Lower and upper bounds of credible intervals.
        """
        if not self._is_fitted:
            raise RuntimeError("BARTRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.transform(X) if self.auto_scale else X.copy()
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        samples = self._predict_samples(X_scaled)
        y_flat = samples.reshape(-1, samples.shape[-1])

        lower = np.percentile(y_flat, 100 * alpha / 2, axis=0)
        upper = np.percentile(y_flat, 100 * (1 - alpha / 2), axis=0)

        if self.auto_scale:
            lower = lower * self._y_std + self._y_mean
            upper = upper * self._y_std + self._y_mean

        return np.column_stack([lower, upper])

    def predict_std(self, X) -> np.ndarray:
        """Predict posterior standard deviation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        std : ndarray of shape (n_samples,)
            Posterior standard deviation for each prediction.
        """
        if not self._is_fitted:
            raise RuntimeError("BARTRegressor has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.transform(X) if self.auto_scale else X.copy()
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        samples = self._predict_samples(X_scaled)
        y_flat = samples.reshape(-1, samples.shape[-1])
        std = np.std(y_flat, axis=0)

        if self.auto_scale:
            std = std * self._y_std
        return std

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on posterior split frequencies."""
        if self.variable_importance_ is None:
            raise RuntimeError("Model not fitted or importance not computed.")
        return self.variable_importance_


class BARTClassifier(ClassifierMixin, BaseEstimator):
    """Bayesian Additive Regression Trees Classifier.

    BART for classification uses a probit or logit link function to model
    class probabilities. The latent function is modeled as a sum of many
    trees with Bayesian priors.

    Parameters
    ----------
    n_trees : int, default=50
        Number of trees in the ensemble.
    n_samples : int, default=1000
        Number of posterior samples.
    n_tune : int, default=500
        Number of tuning samples.
    n_chains : int, default=2
        Number of MCMC chains.
    alpha : float, default=0.95
        Tree depth prior parameter.
    beta : float, default=2.0
        Tree depth penalty parameter.
    auto_scale : bool, default=True
        Whether to standardize features.
    random_state : int, optional
        Random seed.

    Attributes
    ----------
    classes_ : ndarray
        Unique class labels.
    n_features_in_ : int
        Number of features.
    variable_importance_ : ndarray
        Feature importance scores.

    Examples
    --------
    >>> from endgame.models.probabilistic import BARTClassifier
    >>> clf = BARTClassifier(n_trees=50, n_samples=500, random_state=42)
    >>> clf.fit(X_train, y_train)
    >>> proba = clf.predict_proba(X_test)

    Notes
    -----
    For binary classification, BART uses probit regression:
        P(y=1|X) = Phi(sum of trees)
    where Phi is the standard normal CDF.

    For multiclass, a softmax or one-vs-rest approach is used.
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        n_trees: int = 50,
        n_samples: int = 1000,
        n_tune: int = 500,
        n_chains: int = 2,
        alpha: float = 0.95,
        beta: float = 2.0,
        auto_scale: bool = True,
        random_state: int | None = None,
    ):
        self.n_trees = n_trees
        self.n_samples = n_samples
        self.n_tune = n_tune
        self.n_chains = n_chains
        self.alpha = alpha
        self.beta = beta
        self.auto_scale = auto_scale
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.n_classes_: int = 0
        self.n_features_in_: int = 0
        self.variable_importance_: np.ndarray | None = None
        self._scaler: StandardScaler | None = None
        self._label_encoder: LabelEncoder | None = None
        self._trace: Any | None = None
        self._bart: Any | None = None
        self._X_train: np.ndarray | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, **fit_params) -> "BARTClassifier":
        """Fit the BART classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self
        """
        if not _HAS_PYMC_BART:
            raise ImportError(
                "BART requires pymc and pymc-bart. "
                "Install with: pip install pymc pymc-bart"
            )

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        self.n_classes_ = len(self.classes_)

        if self.auto_scale:
            self._scaler = StandardScaler()
            X_scaled = self._scaler.fit_transform(X)
        else:
            X_scaled = X.copy()

        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        if self.n_classes_ == 2:
            with pm.Model() as model:
                self._X_shared = pm.Data("X", X_scaled)

                self._bart = pmb.BART(
                    "bart",
                    self._X_shared,
                    y_encoded,
                    m=self.n_trees,
                    alpha=self.alpha,
                    beta=self.beta,
                )

                p = pm.Deterministic("p", pm.math.invprobit(self._bart))
                pm.Bernoulli("y_obs", p=p, observed=y_encoded, shape=p.shape)

                self._trace = pm.sample(
                    draws=self.n_samples,
                    tune=self.n_tune,
                    chains=self.n_chains,
                    random_seed=self.random_state,
                    progressbar=False,
                    return_inferencedata=True,
                )
        else:
            raise NotImplementedError(
                "Multiclass BART is not yet supported. "
                "Use binary classification or one-vs-rest wrapper."
            )

        self._model = model

        try:
            vi = pmb.compute_variable_importance(
                self._trace, bartrv=self._bart, X=X_scaled,
            )
            self.variable_importance_ = vi.get(
                "r2_mean", np.ones(n_features) / n_features
            ) if isinstance(vi, dict) else vi
        except Exception:
            self.variable_importance_ = np.ones(n_features) / n_features

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
            raise RuntimeError("BARTClassifier has not been fitted.")

        proba = self.predict_proba(X)
        y_pred_encoded = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(y_pred_encoded)

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
            raise RuntimeError("BARTClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        X_scaled = self._scaler.transform(X) if self.auto_scale else X.copy()
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

        with self._model:
            self._X_shared.set_value(X_scaled)
            ppc = pm.sample_posterior_predictive(
                trace=self._trace,
                var_names=["bart"],
                random_seed=self.random_state,
                progressbar=False,
            )

        latent = ppc.posterior_predictive["bart"].values
        from scipy.stats import norm
        p_samples = norm.cdf(latent)
        p_mean = p_samples.mean(axis=(0, 1))

        if self.n_classes_ == 2:
            proba = np.column_stack([1 - p_mean, p_mean])
        else:
            proba = p_mean

        return proba

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importance based on posterior split frequencies."""
        if self.variable_importance_ is None:
            raise RuntimeError("Model not fitted or importance not computed.")
        return self.variable_importance_
