"""AdaBoost: Adaptive Boosting for classification and regression.

Implements SAMME (Stagewise Additive Modeling using a Multi-class
Exponential loss) for classification and SAMME.R for probability-
capable classifiers, plus AdaBoost.R2 for regression.

Example
-------
>>> from endgame.ensemble import AdaBoostClassifier
>>> boost = AdaBoostClassifier(
...     base_estimator=DecisionTreeClassifier(max_depth=1),
...     n_estimators=50,
...     learning_rate=1.0,
... )
>>> boost.fit(X_train, y_train)
>>> boost.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class AdaBoostClassifier(BaseEstimator, ClassifierMixin):
    """AdaBoost classifier (SAMME / SAMME.R).

    Parameters
    ----------
    base_estimator : estimator, optional
        Base learner. Default: ``DecisionTreeClassifier(max_depth=1)`` (stump).
    n_estimators : int, default=50
        Maximum number of boosting rounds.
    learning_rate : float, default=1.0
        Shrinkage applied to each estimator's weight. Lower values
        require more estimators but generalize better.
    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        - ``'SAMME'``: discrete AdaBoost using class labels.
        - ``'SAMME.R'``: real AdaBoost using class probabilities
          (requires ``predict_proba``).
    random_state : int or None, default=None

    Attributes
    ----------
    estimators_ : list of estimator
        Fitted weak learners.
    estimator_weights_ : ndarray
        Weight of each estimator (SAMME only).
    estimator_errors_ : ndarray
        Weighted error of each estimator.
    classes_ : ndarray
    n_classes_ : int
    feature_importances_ : ndarray
        Sum of feature importances weighted by estimator weight.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        algorithm: str = "SAMME.R",
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples = X.shape[0]

        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        le = {c: i for i, c in enumerate(self.classes_)}
        y_enc = np.array([le[v] for v in y])

        base = self.base_estimator or DecisionTreeClassifier(max_depth=1)

        if sample_weight is None:
            w = np.full(n_samples, 1.0 / n_samples)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).copy()
            w /= w.sum()

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        rng = np.random.RandomState(self.random_state)

        for t in range(self.n_estimators):
            est = clone(base)
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(0, 2**31)

            try:
                est.fit(X, y, sample_weight=w)
            except TypeError:
                est.fit(X, y)

            if self.algorithm == "SAMME.R" and hasattr(est, "predict_proba"):
                # SAMME.R: real-valued boosting using probabilities
                proba = est.predict_proba(X)
                proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
                log_proba = np.log(proba)

                # Update weights using the SAMME.R formula
                indicator = np.zeros_like(proba)
                indicator[np.arange(n_samples), y_enc] = 1.0
                estimator_weight = (
                    -1.0 * self.learning_rate
                    * ((self.n_classes_ - 1.0) / self.n_classes_)
                    * (indicator * log_proba).sum(axis=1)
                )

                # Error rate for monitoring
                y_pred = self.classes_[np.argmax(proba, axis=1)]
                incorrect = (y_pred != y)
                err = float(np.dot(w, incorrect))

                self.estimators_.append(est)
                self.estimator_weights_[t] = 1.0  # dummy for SAMME.R
                self.estimator_errors_[t] = err

                if err >= 1.0 - 1.0 / self.n_classes_:
                    break

                # Reweight samples
                w *= np.exp(estimator_weight - estimator_weight.max())
                w_sum = w.sum()
                if w_sum <= 0:
                    break
                w /= w_sum

            else:
                # SAMME: discrete boosting
                y_pred = est.predict(X)
                incorrect = (y_pred != y)
                err = float(np.dot(w, incorrect))

                if err > 1.0 - 1.0 / self.n_classes_:
                    if t == 0:
                        self.estimators_.append(est)
                        self.estimator_weights_[t] = 1.0
                        self.estimator_errors_[t] = err
                    break

                if err <= 0:
                    self.estimators_.append(est)
                    self.estimator_weights_[t] = 10.0  # large weight for perfect
                    self.estimator_errors_[t] = 0.0
                    break

                alpha = self.learning_rate * (
                    np.log((1.0 - err) / err) + np.log(self.n_classes_ - 1.0)
                )

                self.estimators_.append(est)
                self.estimator_weights_[t] = alpha
                self.estimator_errors_[t] = err

                # Reweight
                w *= np.exp(alpha * incorrect)
                w /= w.sum()

        n_fitted = len(self.estimators_)
        self.estimator_weights_ = self.estimator_weights_[:n_fitted]
        self.estimator_errors_ = self.estimator_errors_[:n_fitted]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if self.algorithm == "SAMME.R" and hasattr(self.estimators_[0], "predict_proba"):
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        # SAMME: weighted vote
        n = X.shape[0]
        scores = np.zeros((n, self.n_classes_))
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = est.predict(X)
            le = {c: i for i, c in enumerate(self.classes_)}
            for j in range(n):
                scores[j, le[preds[j]]] += alpha
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]

        if self.algorithm == "SAMME.R" and hasattr(self.estimators_[0], "predict_proba"):
            # SAMME.R: accumulate log probabilities
            log_proba_sum = np.zeros((n, self.n_classes_))
            for est in self.estimators_:
                proba = est.predict_proba(X)
                proba = np.clip(proba, 1e-15, 1.0 - 1e-15)
                log_p = np.log(proba)
                log_proba_sum += self.learning_rate * (
                    (self.n_classes_ - 1) * (log_p - log_p.mean(axis=1, keepdims=True))
                )
            # Softmax
            log_proba_sum -= log_proba_sum.max(axis=1, keepdims=True)
            proba = np.exp(log_proba_sum)
            proba /= proba.sum(axis=1, keepdims=True)
            return proba

        # SAMME: convert weighted votes to probabilities
        scores = np.zeros((n, self.n_classes_))
        le = {c: i for i, c in enumerate(self.classes_)}
        for est, alpha in zip(self.estimators_, self.estimator_weights_):
            preds = est.predict(X)
            for j in range(n):
                scores[j, le[preds[j]]] += alpha
        total = scores.sum(axis=1, keepdims=True)
        total = np.where(total == 0, 1, total)
        return scores / total

    @property
    def feature_importances_(self):
        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise AttributeError("Base estimator has no feature_importances_.")
        w = self.estimator_weights_
        if w.sum() == 0:
            w = np.ones(len(self.estimators_))
        norm_w = w / w.sum()
        imp = sum(
            est.feature_importances_ * nw
            for est, nw in zip(self.estimators_, norm_w)
        )
        return imp


class AdaBoostRegressor(BaseEstimator, RegressorMixin):
    """AdaBoost.R2 regressor.

    Parameters
    ----------
    base_estimator : estimator, optional
        Default: ``DecisionTreeRegressor(max_depth=3)``.
    n_estimators : int, default=50
    learning_rate : float, default=1.0
    loss : {'linear', 'square', 'exponential'}, default='linear'
        Loss function for computing sample weights.
    random_state : int or None, default=None

    Attributes
    ----------
    estimators_ : list of estimator
    estimator_weights_ : ndarray
    estimator_errors_ : ndarray
    feature_importances_ : ndarray
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        loss: str = "linear",
        random_state: int | None = None,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.loss = loss
        self.random_state = random_state

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples = X.shape[0]

        base = self.base_estimator or DecisionTreeRegressor(max_depth=3)
        rng = np.random.RandomState(self.random_state)

        if sample_weight is None:
            w = np.full(n_samples, 1.0 / n_samples)
        else:
            w = np.asarray(sample_weight, dtype=np.float64).copy()
            w /= w.sum()

        self.estimators_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators)
        self.estimator_errors_ = np.zeros(self.n_estimators)

        for t in range(self.n_estimators):
            est = clone(base)
            if hasattr(est, "random_state"):
                est.random_state = rng.randint(0, 2**31)

            try:
                est.fit(X, y, sample_weight=w)
            except TypeError:
                est.fit(X, y)

            y_pred = est.predict(X)

            # Compute error
            abs_error = np.abs(y - y_pred)
            max_err = abs_error.max()
            if max_err == 0:
                self.estimators_.append(est)
                self.estimator_weights_[t] = 1.0
                self.estimator_errors_[t] = 0.0
                break

            # Normalize error
            if self.loss == "linear":
                loss_arr = abs_error / max_err
            elif self.loss == "square":
                loss_arr = (abs_error / max_err) ** 2
            elif self.loss == "exponential":
                loss_arr = 1.0 - np.exp(-abs_error / max_err)
            else:
                raise ValueError(f"Unknown loss: {self.loss}")

            avg_loss = float(np.dot(w, loss_arr))

            if avg_loss >= 0.5:
                if t == 0:
                    self.estimators_.append(est)
                    self.estimator_weights_[t] = 1.0
                    self.estimator_errors_[t] = avg_loss
                break

            beta = avg_loss / (1.0 - avg_loss)
            alpha = self.learning_rate * np.log(1.0 / beta)

            self.estimators_.append(est)
            self.estimator_weights_[t] = alpha
            self.estimator_errors_[t] = avg_loss

            # Reweight
            w *= np.power(beta, (1.0 - loss_arr))
            w_sum = w.sum()
            if w_sum <= 0:
                break
            w /= w_sum

        n_fitted = len(self.estimators_)
        self.estimator_weights_ = self.estimator_weights_[:n_fitted]
        self.estimator_errors_ = self.estimator_errors_[:n_fitted]

        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        # Weighted median
        preds = np.array([est.predict(X) for est in self.estimators_])
        w = self.estimator_weights_
        if w.sum() == 0:
            return preds.mean(axis=0)

        n = X.shape[0]
        result = np.empty(n)
        for i in range(n):
            vals = preds[:, i]
            order = np.argsort(vals)
            sorted_vals = vals[order]
            sorted_w = w[order]
            cumw = np.cumsum(sorted_w)
            half = cumw[-1] / 2.0
            idx = np.searchsorted(cumw, half)
            idx = min(idx, len(sorted_vals) - 1)
            result[i] = sorted_vals[idx]
        return result

    @property
    def feature_importances_(self):
        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise AttributeError("Base estimator has no feature_importances_.")
        w = self.estimator_weights_
        if w.sum() == 0:
            w = np.ones(len(self.estimators_))
        norm_w = w / w.sum()
        return sum(
            est.feature_importances_ * nw
            for est, nw in zip(self.estimators_, norm_w)
        )
