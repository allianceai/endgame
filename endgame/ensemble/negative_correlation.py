"""Negative Correlation Learning (NCL): Diversity-promoting ensemble.

Trains base learners simultaneously with a penalty term that encourages
diverse predictions. Each learner's loss includes a correlation penalty
that penalizes agreement with the ensemble, producing a set of
complementary models.

Theory: Liu & Yao (1999), "Ensemble Learning via Negative Correlation."

Example
-------
>>> from endgame.ensemble import NegativeCorrelationEnsemble
>>> nce = NegativeCorrelationEnsemble(
...     base_estimators=[Ridge(), Lasso(), ElasticNet()],
...     lambda_ncl=0.5,
...     n_iterations=20,
... )
>>> nce.fit(X_train, y_train)
>>> nce.predict(X_test)
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone


class NegativeCorrelationEnsemble(BaseEstimator, RegressorMixin):
    """Negative Correlation Learning ensemble for regression.

    Trains all base learners together, each with a modified loss that
    includes a diversity term penalizing correlation with the ensemble
    average. This produces models that are individually weaker but
    collectively stronger.

    Parameters
    ----------
    base_estimators : list of estimator
        Base regressors to train. Must support ``sample_weight``
        or partial fit.
    lambda_ncl : float, default=0.5
        Strength of the negative correlation penalty.
        - ``0``: independent training (standard ensemble).
        - ``1``: maximum diversity pressure.
    n_iterations : int, default=10
        Number of NCL training rounds.
    weights : array-like, optional
        Static model weights. Default is uniform.
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    estimators_ : list of estimator
        Fitted base regressors.
    weights_ : ndarray
        Model combination weights.
    diversity_ : float
        Measured ensemble diversity (average pairwise disagreement).

    References
    ----------
    Liu, Y. & Yao, X. (1999). Ensemble Learning via Negative
    Correlation. *Neural Networks*, 12(10), 1399-1404.
    """

    def __init__(
        self,
        base_estimators: list[BaseEstimator],
        lambda_ncl: float = 0.5,
        n_iterations: int = 10,
        weights: Sequence[float] | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.base_estimators = base_estimators
        self.lambda_ncl = lambda_ncl
        self.n_iterations = n_iterations
        self.weights = weights
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        """Fit with negative correlation learning.

        Each round:
        1. Compute ensemble prediction (average of all learners).
        2. For each learner i, compute modified sample weights
           that up-weight samples where learner i disagrees with
           the ensemble (promoting diversity).
        3. Refit each learner with the modified weights.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]
        n_models = len(self.base_estimators)

        rng = np.random.RandomState(self.random_state)

        # Initialize estimators
        self.estimators_ = [clone(est) for est in self.base_estimators]
        for est in self.estimators_:
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(0, 2**31)

        # Initial fit
        for est in self.estimators_:
            if sample_weight is not None:
                try:
                    est.fit(X, y, sample_weight=sample_weight)
                except TypeError:
                    est.fit(X, y)
            else:
                est.fit(X, y)

        # NCL iterations
        for iteration in range(self.n_iterations):
            # Get all predictions
            preds = np.array([est.predict(X) for est in self.estimators_])
            ensemble_pred = preds.mean(axis=0)

            for i, est in enumerate(self.estimators_):
                # Residual of learner i
                fi = preds[i]
                error_i = y - fi

                # Deviation from ensemble
                deviation = fi - ensemble_pred

                # NCL modified target: emphasize where this model
                # agrees with the ensemble (penalize correlation)
                # Modified loss gradient includes the correlation term
                ncl_weight = np.abs(error_i) + self.lambda_ncl * np.abs(deviation)
                ncl_weight = ncl_weight / (ncl_weight.sum() + 1e-15) * n_samples

                if sample_weight is not None:
                    ncl_weight *= sample_weight

                # Refit with NCL weights
                refitted = clone(self.base_estimators[i])
                if hasattr(refitted, "random_state"):
                    refitted.random_state = rng.randint(0, 2**31)
                try:
                    refitted.fit(X, y, sample_weight=ncl_weight)
                except TypeError:
                    refitted.fit(X, y)
                self.estimators_[i] = refitted

            if self.verbose:
                preds_new = np.array([est.predict(X) for est in self.estimators_])
                ens_new = preds_new.mean(axis=0)
                mse = float(np.mean((y - ens_new) ** 2))
                print(f"[NCL] Iteration {iteration + 1}/{self.n_iterations}, MSE={mse:.4f}")

        # Set weights
        if self.weights is not None:
            self.weights_ = np.asarray(self.weights, dtype=float)
            self.weights_ /= self.weights_.sum()
        else:
            self.weights_ = np.ones(n_models) / n_models

        # Compute diversity metric
        self.diversity_ = self._compute_diversity(X)

        return self

    def _compute_diversity(self, X):
        """Compute average pairwise disagreement (Q-statistic)."""
        preds = np.array([est.predict(X) for est in self.estimators_])
        n = len(self.estimators_)
        if n < 2:
            return 0.0
        total = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                corr = np.corrcoef(preds[i], preds[j])[0, 1]
                total += 1 - abs(corr)  # diversity = 1 - |correlation|
                count += 1
        return float(total / max(count, 1))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([est.predict(X) for est in self.estimators_])
        return (self.weights_[:, None] * preds).sum(axis=0)

    @property
    def feature_importances_(self):
        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise AttributeError("Base estimator has no feature_importances_.")
        return sum(
            est.feature_importances_ * w
            for est, w in zip(self.estimators_, self.weights_)
        )
