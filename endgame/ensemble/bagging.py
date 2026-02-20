"""Bagging Ensemble: Bootstrap Aggregating for classification and regression.

Builds multiple copies of a base estimator, each trained on a random
bootstrap sample (and optionally a random feature subset), then combines
predictions via majority vote (classification) or averaging (regression).

Key hyper-parameters:
- ``max_samples``: fraction (or count) of training rows per bag.
- ``max_features``: fraction (or count) of features per bag ("random subspace").
- ``bootstrap`` / ``bootstrap_features``: with or without replacement.
- ``oob_score``: out-of-bag evaluation without a separate validation set.

Example
-------
>>> from endgame.ensemble import BaggingClassifier
>>> bag = BaggingClassifier(
...     base_estimator=DecisionTreeClassifier(max_depth=6),
...     n_estimators=50,
...     max_samples=0.8,
...     max_features=0.7,
...     oob_score=True,
... )
>>> bag.fit(X_train, y_train)
>>> print(f"OOB accuracy: {bag.oob_score_:.4f}")
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None  # type: ignore[assignment,misc]
    delayed = None


def _fit_bag(estimator, X, y, sample_weight, rng_seed, max_samples, max_features,
             bootstrap, bootstrap_features, n_samples, n_features):
    """Fit one bootstrap replica (joblib-friendly)."""
    rng = np.random.RandomState(rng_seed)

    # Row sampling
    if isinstance(max_samples, float):
        n_draw = max(1, int(max_samples * n_samples))
    else:
        n_draw = min(int(max_samples), n_samples)

    if bootstrap:
        row_idx = rng.randint(0, n_samples, size=n_draw)
    else:
        row_idx = rng.choice(n_samples, size=n_draw, replace=False)

    # Feature sampling
    if isinstance(max_features, float):
        n_feat_draw = max(1, int(max_features * n_features))
    else:
        n_feat_draw = min(int(max_features), n_features)

    if n_feat_draw < n_features:
        if bootstrap_features:
            col_idx = rng.randint(0, n_features, size=n_feat_draw)
        else:
            col_idx = rng.choice(n_features, size=n_feat_draw, replace=False)
    else:
        col_idx = np.arange(n_features)

    X_bag = X[np.ix_(row_idx, col_idx)]
    y_bag = y[row_idx]

    est = clone(estimator)
    if sample_weight is not None:
        w_bag = sample_weight[row_idx]
        try:
            est.fit(X_bag, y_bag, sample_weight=w_bag)
        except TypeError:
            est.fit(X_bag, y_bag)
    else:
        est.fit(X_bag, y_bag)

    return est, row_idx, col_idx


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    """Bootstrap Aggregating classifier.

    Parameters
    ----------
    base_estimator : estimator, optional
        Base learner to bag. Default: ``DecisionTreeClassifier()``.
    n_estimators : int, default=10
        Number of bootstrap replicas.
    max_samples : float or int, default=1.0
        Fraction (if float) or count (if int) of samples per bag.
    max_features : float or int, default=1.0
        Fraction (if float) or count (if int) of features per bag.
    bootstrap : bool, default=True
        Sample rows with replacement.
    bootstrap_features : bool, default=False
        Sample features with replacement.
    oob_score : bool, default=False
        Whether to compute out-of-bag accuracy.
    n_jobs : int or None, default=None
        Parallel jobs. ``-1`` uses all CPUs.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False

    Attributes
    ----------
    estimators_ : list of estimator
        Fitted base estimators.
    estimator_features_ : list of ndarray
        Feature indices used by each estimator.
    oob_score_ : float
        OOB accuracy (only if ``oob_score=True``).
    oob_decision_function_ : ndarray
        OOB predicted probabilities (only if ``oob_score=True``).
    classes_ : ndarray
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        n_estimators: int = 10,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        n_samples, n_features = X.shape
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)

        base = self.base_estimator or DecisionTreeClassifier()
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31, size=self.n_estimators)

        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_bag)(
                    base, X, y, sample_weight, int(seeds[i]),
                    self.max_samples, self.max_features,
                    self.bootstrap, self.bootstrap_features,
                    n_samples, n_features,
                )
                for i in range(self.n_estimators)
            )
        else:
            results = [
                _fit_bag(
                    base, X, y, sample_weight, int(seeds[i]),
                    self.max_samples, self.max_features,
                    self.bootstrap, self.bootstrap_features,
                    n_samples, n_features,
                )
                for i in range(self.n_estimators)
            ]

        self.estimators_ = [r[0] for r in results]
        self._row_indices = [r[1] for r in results]
        self.estimator_features_ = [r[2] for r in results]

        if self.oob_score:
            self._compute_oob_score(X, y, n_samples)

        return self

    def _compute_oob_score(self, X, y, n_samples):
        n_classes = self.n_classes_
        oob_decision = np.zeros((n_samples, n_classes))
        oob_counts = np.zeros(n_samples)
        le = {c: i for i, c in enumerate(self.classes_)}

        for est, row_idx, col_idx in zip(self.estimators_, self._row_indices, self.estimator_features_):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[row_idx] = False
            if not oob_mask.any():
                continue
            X_oob = X[np.ix_(np.where(oob_mask)[0], col_idx)]
            if hasattr(est, "predict_proba"):
                proba = est.predict_proba(X_oob)
                oob_decision[oob_mask, :proba.shape[1]] += proba
            else:
                preds = est.predict(X_oob)
                for j, p in zip(np.where(oob_mask)[0], preds):
                    if p in le:
                        oob_decision[j, le[p]] += 1
            oob_counts[oob_mask] += 1

        valid = oob_counts > 0
        oob_decision[valid] /= oob_counts[valid, None]
        self.oob_decision_function_ = oob_decision

        oob_pred = self.classes_[np.argmax(oob_decision[valid], axis=1)]
        self.oob_score_ = float(np.mean(oob_pred == y[valid]))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        all_proba = np.zeros((X.shape[0], self.n_classes_))
        for est, col_idx in zip(self.estimators_, self.estimator_features_):
            X_sub = X[:, col_idx]
            if hasattr(est, "predict_proba"):
                p = est.predict_proba(X_sub)
                all_proba[:, :p.shape[1]] += p
            else:
                le = {c: i for i, c in enumerate(self.classes_)}
                preds = est.predict(X_sub)
                for j, pred in enumerate(preds):
                    if pred in le:
                        all_proba[j, le[pred]] += 1
        all_proba /= self.n_estimators
        return all_proba

    @property
    def feature_importances_(self):
        """Average feature importances across bags (original feature space)."""
        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise AttributeError("Base estimator has no feature_importances_.")
        n_features = max(c.max() for c in self.estimator_features_) + 1
        imp = np.zeros(n_features)
        counts = np.zeros(n_features)
        for est, col_idx in zip(self.estimators_, self.estimator_features_):
            fi = est.feature_importances_
            for k, c in enumerate(col_idx):
                imp[c] += fi[k]
                counts[c] += 1
        mask = counts > 0
        imp[mask] /= counts[mask]
        return imp


class BaggingRegressor(BaseEstimator, RegressorMixin):
    """Bootstrap Aggregating regressor.

    Parameters
    ----------
    base_estimator : estimator, optional
        Base learner. Default: ``DecisionTreeRegressor()``.
    n_estimators : int, default=10
    max_samples : float or int, default=1.0
    max_features : float or int, default=1.0
    bootstrap : bool, default=True
    bootstrap_features : bool, default=False
    oob_score : bool, default=False
    n_jobs : int or None, default=None
    random_state : int or None, default=None
    verbose : bool, default=False

    Attributes
    ----------
    estimators_ : list of estimator
    estimator_features_ : list of ndarray
    oob_score_ : float
        OOB R² score (only if ``oob_score=True``).
    oob_prediction_ : ndarray
        OOB predictions (only if ``oob_score=True``).
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        n_estimators: int = 10,
        max_samples: float | int = 1.0,
        max_features: float | int = 1.0,
        bootstrap: bool = True,
        bootstrap_features: bool = False,
        oob_score: bool = False,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.bootstrap_features = bootstrap_features
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        n_samples, n_features = X.shape

        base = self.base_estimator or DecisionTreeRegressor()
        rng = np.random.RandomState(self.random_state)
        seeds = rng.randint(0, 2**31, size=self.n_estimators)

        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            results = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_bag)(
                    base, X, y, sample_weight, int(seeds[i]),
                    self.max_samples, self.max_features,
                    self.bootstrap, self.bootstrap_features,
                    n_samples, n_features,
                )
                for i in range(self.n_estimators)
            )
        else:
            results = [
                _fit_bag(
                    base, X, y, sample_weight, int(seeds[i]),
                    self.max_samples, self.max_features,
                    self.bootstrap, self.bootstrap_features,
                    n_samples, n_features,
                )
                for i in range(self.n_estimators)
            ]

        self.estimators_ = [r[0] for r in results]
        self._row_indices = [r[1] for r in results]
        self.estimator_features_ = [r[2] for r in results]

        if self.oob_score:
            self._compute_oob_score(X, y, n_samples)

        return self

    def _compute_oob_score(self, X, y, n_samples):
        oob_sum = np.zeros(n_samples)
        oob_counts = np.zeros(n_samples)

        for est, row_idx, col_idx in zip(self.estimators_, self._row_indices, self.estimator_features_):
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[row_idx] = False
            if not oob_mask.any():
                continue
            X_oob = X[np.ix_(np.where(oob_mask)[0], col_idx)]
            oob_sum[oob_mask] += est.predict(X_oob)
            oob_counts[oob_mask] += 1

        valid = oob_counts > 0
        self.oob_prediction_ = np.full(n_samples, np.nan)
        self.oob_prediction_[valid] = oob_sum[valid] / oob_counts[valid]

        from sklearn.metrics import r2_score
        self.oob_score_ = float(r2_score(y[valid], self.oob_prediction_[valid]))

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        preds = np.array([
            est.predict(X[:, col_idx])
            for est, col_idx in zip(self.estimators_, self.estimator_features_)
        ])
        return preds.mean(axis=0)

    @property
    def feature_importances_(self):
        if not hasattr(self.estimators_[0], "feature_importances_"):
            raise AttributeError("Base estimator has no feature_importances_.")
        n_features = max(c.max() for c in self.estimator_features_) + 1
        imp = np.zeros(n_features)
        counts = np.zeros(n_features)
        for est, col_idx in zip(self.estimators_, self.estimator_features_):
            fi = est.feature_importances_
            for k, c in enumerate(col_idx):
                imp[c] += fi[k]
                counts[c] += 1
        mask = counts > 0
        imp[mask] /= counts[mask]
        return imp
