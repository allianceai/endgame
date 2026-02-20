"""Bayesian Model Averaging: Weight models by posterior probability.

Combines predictions from multiple fitted models by weighting each
model's contribution proportional to its posterior probability given
the data. Uses BIC (Bayesian Information Criterion) or AIC to
approximate the marginal likelihood.

Example
-------
>>> from endgame.ensemble import BayesianModelAveraging
>>> bma = BayesianModelAveraging(criterion="bic")
>>> bma.fit(
...     estimators=[rf, xgb, lr],
...     X_val=X_val,
...     y_val=y_val,
... )
>>> bma.predict(X_test)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import log_loss, mean_squared_error


class BayesianModelAveraging(BaseEstimator):
    """Bayesian Model Averaging using information-criterion weights.

    Parameters
    ----------
    criterion : {'bic', 'aic', 'aic_c'}, default='bic'
        - ``'bic'``: Bayesian Information Criterion (penalizes complexity more).
        - ``'aic'``: Akaike Information Criterion.
        - ``'aic_c'``: Corrected AIC for small samples.
    prior : {'uniform', 'complexity'} or array-like, optional
        Prior over models. Default is uniform.
    task : {'auto', 'classification', 'regression'}, default='auto'

    Attributes
    ----------
    weights_ : ndarray
        Posterior model weights summing to 1.
    ic_scores_ : ndarray
        Information criterion values for each model.
    estimators_ : list of estimator
        Fitted estimators (references, not clones).
    """

    def __init__(
        self,
        criterion: str = "bic",
        prior: str | Sequence[float] = "uniform",
        task: str = "auto",
    ):
        self.criterion = criterion
        self.prior = prior
        self.task = task

    def fit(
        self,
        estimators: list[BaseEstimator],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> BayesianModelAveraging:
        """Compute posterior weights from validation data.

        Parameters
        ----------
        estimators : list of fitted estimators
            Already-fitted models.
        X_val : array-like
            Validation features.
        y_val : array-like
            Validation target.

        Returns
        -------
        self
        """
        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val)
        n = len(y_val)
        self.estimators_ = list(estimators)
        m = len(estimators)

        # Auto-detect task
        if self.task == "auto":
            unique = np.unique(y_val)
            self._is_classifier = len(unique) <= 30 and (np.issubdtype(y_val.dtype, np.integer) or len(unique) <= 10)
        else:
            self._is_classifier = self.task == "classification"

        if self._is_classifier:
            self.classes_ = np.unique(y_val)

        # Compute negative log-likelihood for each model
        nll = np.zeros(m)
        n_params = np.zeros(m)

        for i, est in enumerate(estimators):
            if self._is_classifier:
                nll[i], n_params[i] = self._classification_nll(est, X_val, y_val, n)
            else:
                nll[i], n_params[i] = self._regression_nll(est, X_val, y_val, n)

        # Compute IC
        if self.criterion == "bic":
            self.ic_scores_ = 2 * nll + n_params * np.log(n)
        elif self.criterion == "aic":
            self.ic_scores_ = 2 * nll + 2 * n_params
        elif self.criterion == "aic_c":
            self.ic_scores_ = 2 * nll + 2 * n_params + (
                2 * n_params * (n_params + 1) / np.maximum(n - n_params - 1, 1)
            )
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

        # Convert IC to posterior weights via exp(-0.5 * delta_IC)
        delta = self.ic_scores_ - self.ic_scores_.min()
        log_weights = -0.5 * delta

        # Apply prior
        if isinstance(self.prior, str):
            if self.prior == "uniform":
                pass  # uniform = no change
            elif self.prior == "complexity":
                # Favor simpler models
                log_weights -= 0.01 * n_params
        else:
            log_prior = np.log(np.asarray(self.prior, dtype=float) + 1e-15)
            log_weights += log_prior

        # Softmax normalization
        log_weights -= log_weights.max()
        raw_weights = np.exp(log_weights)
        self.weights_ = raw_weights / raw_weights.sum()

        return self

    def _classification_nll(self, est, X, y, n):
        """Negative log-likelihood for classification."""
        try:
            proba = est.predict_proba(X)
            nll = n * log_loss(y, proba)
        except Exception:
            y_pred = est.predict(X)
            err_rate = float(np.mean(y_pred != y))
            nll = -n * (
                (1 - err_rate) * np.log(max(1 - err_rate, 1e-15))
                + err_rate * np.log(max(err_rate, 1e-15))
            )

        n_params = self._estimate_params(est)
        return float(nll), float(n_params)

    def _regression_nll(self, est, X, y, n):
        """Negative log-likelihood for regression (Gaussian)."""
        y_pred = est.predict(X)
        mse = mean_squared_error(y, y_pred)
        sigma2 = max(mse, 1e-15)
        nll = 0.5 * n * (np.log(2 * np.pi * sigma2) + 1.0)
        n_params = self._estimate_params(est)
        return float(nll), float(n_params)

    def _estimate_params(self, est):
        """Estimate effective number of parameters."""
        name = type(est).__name__

        # Known models
        if hasattr(est, "coef_"):
            coef = np.asarray(est.coef_).ravel()
            return len(coef) + (1 if hasattr(est, "intercept_") else 0)
        if hasattr(est, "n_estimators") and hasattr(est, "max_depth"):
            depth = est.max_depth or 6
            return getattr(est, "n_estimators", 100) * (2 ** depth)
        if hasattr(est, "tree_"):
            return est.tree_.node_count
        if hasattr(est, "n_estimators"):
            return getattr(est, "n_estimators", 100) * 10

        return 10  # fallback

    def predict(self, X):
        X = np.asarray(X)
        if self._is_classifier:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # Regression: weighted average
        pred = np.zeros(X.shape[0])
        for est, w in zip(self.estimators_, self.weights_):
            pred += w * est.predict(X)
        return pred

    def predict_proba(self, X):
        if not self._is_classifier:
            raise ValueError("predict_proba only for classification.")
        X = np.asarray(X)
        n = X.shape[0]
        n_classes = len(self.classes_)
        proba = np.zeros((n, n_classes))

        for est, w in zip(self.estimators_, self.weights_):
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X)
                proba[:, :p.shape[1]] += w * p
            else:
                preds = est.predict(X)
                le = {c: i for i, c in enumerate(self.classes_)}
                for j, pred in enumerate(preds):
                    if pred in le:
                        proba[j, le[pred]] += w

        # Normalize
        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        return proba / row_sums
