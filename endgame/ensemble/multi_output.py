from __future__ import annotations

"""Multi-output wrappers: parallel and chained multi-target estimation.

Provides MultiOutputClassifier, MultiOutputRegressor, ClassifierChain,
and RegressorChain for multi-target learning tasks where Y has shape
(n_samples, n_outputs).
"""


import numpy as np
from sklearn.base import (
    BaseEstimator,
    ClassifierMixin,
    RegressorMixin,
    clone,
)

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None
    delayed = None


def _fit_single_estimator(estimator, X, y, sample_weight=None):
    """Fit a single estimator on (X, y).

    Parameters
    ----------
    estimator : estimator
        The estimator to fit.
    X : ndarray of shape (n_samples, n_features)
        Training features.
    y : ndarray of shape (n_samples,)
        Target values for a single output.
    sample_weight : ndarray, optional
        Sample weights.

    Returns
    -------
    estimator
        Fitted estimator.
    """
    if sample_weight is not None:
        try:
            estimator.fit(X, y, sample_weight=sample_weight)
        except TypeError:
            estimator.fit(X, y)
    else:
        estimator.fit(X, y)
    return estimator


class MultiOutputClassifier(BaseEstimator, ClassifierMixin):
    """Wraps a single-output classifier for multi-output classification.

    Fits one independent clone of the base classifier per output column.
    Supports parallel fitting via joblib.

    Parameters
    ----------
    estimator : estimator
        The base classifier to clone for each output. Must implement
        ``fit`` and ``predict``.
    n_jobs : int, optional
        Number of jobs for parallel fitting. ``None`` means 1 (sequential).
        ``-1`` means using all processors.
    random_state : int, optional
        Random seed. Passed to each cloned estimator if it accepts
        ``random_state``.
    verbose : bool, default=False
        Enable verbose output during fitting.

    Attributes
    ----------
    estimators_ : List[estimator]
        Fitted classifiers, one per output.
    classes_ : List[ndarray]
        Class labels for each output.
    n_outputs_ : int
        Number of output columns.

    Examples
    --------
    >>> from endgame.ensemble.multi_output import MultiOutputClassifier
    >>> from sklearn.tree import DecisionTreeClassifier
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> Y = np.random.randint(0, 3, size=(100, 3))
    >>> clf = MultiOutputClassifier(DecisionTreeClassifier(), n_jobs=-1)
    >>> clf.fit(X, Y)
    >>> preds = clf.predict(X)
    >>> preds.shape
    (100, 3)
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _make_estimator(self):
        """Clone the base estimator and inject random_state if supported."""
        est = clone(self.estimator)
        if self.random_state is not None and hasattr(est, "random_state"):
            est.random_state = self.random_state
        return est

    def fit(self, X, Y, sample_weight=None):
        """Fit one classifier per output column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples, n_outputs)
            Multi-output target matrix.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights passed to each estimator.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]
        self.classes_ = [np.unique(Y[:, i]) for i in range(self.n_outputs_)]

        estimators = [self._make_estimator() for _ in range(self.n_outputs_)]

        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(est, X, Y[:, i], sample_weight)
                for i, est in enumerate(estimators)
            )
        else:
            self.estimators_ = [
                _fit_single_estimator(est, X, Y[:, i], sample_weight)
                for i, est in enumerate(estimators)
            ]

        return self

    def predict(self, X):
        """Predict class labels for each output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Predicted class labels.
        """
        self._check_fitted()
        X = np.asarray(X)
        predictions = np.column_stack(
            [est.predict(X) for est in self.estimators_]
        )
        return predictions

    def predict_proba(self, X):
        """Predict class probabilities for each output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        list of ndarray
            List of length ``n_outputs_``, where each element is an array
            of shape ``(n_samples, n_classes_k)`` containing class
            probabilities for output k.
        """
        self._check_fitted()
        X = np.asarray(X)
        return [est.predict_proba(X) for est in self.estimators_]

    def score(self, X, Y, sample_weight=None):
        """Return the mean accuracy across all outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        Y : array-like of shape (n_samples, n_outputs)
            True labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Mean of per-output accuracy scores.
        """
        from sklearn.metrics import accuracy_score

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        preds = self.predict(X)
        scores = [
            accuracy_score(Y[:, i], preds[:, i], sample_weight=sample_weight)
            for i in range(self.n_outputs_)
        ]
        return float(np.mean(scores))

    def _check_fitted(self):
        """Raise if the estimator has not been fitted."""
        if not hasattr(self, "estimators_") or self.estimators_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )


class MultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Wraps a single-output regressor for multi-output regression.

    Fits one independent clone of the base regressor per output column.
    Supports parallel fitting via joblib.

    Parameters
    ----------
    estimator : estimator
        The base regressor to clone for each output. Must implement
        ``fit`` and ``predict``.
    n_jobs : int, optional
        Number of jobs for parallel fitting. ``None`` means 1 (sequential).
        ``-1`` means using all processors.
    random_state : int, optional
        Random seed. Passed to each cloned estimator if it accepts
        ``random_state``.
    verbose : bool, default=False
        Enable verbose output during fitting.

    Attributes
    ----------
    estimators_ : List[estimator]
        Fitted regressors, one per output.
    n_outputs_ : int
        Number of output columns.

    Examples
    --------
    >>> from endgame.ensemble.multi_output import MultiOutputRegressor
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(100, 5)
    >>> Y = np.random.randn(100, 3)
    >>> reg = MultiOutputRegressor(Ridge(), n_jobs=-1)
    >>> reg.fit(X, Y)
    >>> preds = reg.predict(X)
    >>> preds.shape
    (100, 3)
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _make_estimator(self):
        """Clone the base estimator and inject random_state if supported."""
        est = clone(self.estimator)
        if self.random_state is not None and hasattr(est, "random_state"):
            est.random_state = self.random_state
        return est

    def fit(self, X, Y, sample_weight=None):
        """Fit one regressor per output column.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples, n_outputs)
            Multi-output target matrix.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights passed to each estimator.

        Returns
        -------
        self
        """
        X = np.asarray(X)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]

        estimators = [self._make_estimator() for _ in range(self.n_outputs_)]

        if Parallel is not None and self.n_jobs is not None and self.n_jobs != 1:
            self.estimators_ = Parallel(n_jobs=self.n_jobs)(
                delayed(_fit_single_estimator)(est, X, Y[:, i], sample_weight)
                for i, est in enumerate(estimators)
            )
        else:
            self.estimators_ = [
                _fit_single_estimator(est, X, Y[:, i], sample_weight)
                for i, est in enumerate(estimators)
            ]

        return self

    def predict(self, X):
        """Predict target values for each output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Predicted values.
        """
        self._check_fitted()
        X = np.asarray(X)
        predictions = np.column_stack(
            [est.predict(X) for est in self.estimators_]
        )
        return predictions

    @property
    def feature_importances_(self):
        """Average feature importances across all output estimators.

        Returns
        -------
        ndarray of shape (n_features,)
            Mean of ``feature_importances_`` across fitted estimators.

        Raises
        ------
        AttributeError
            If the base estimators do not expose ``feature_importances_``.
        """
        self._check_fitted()
        all_importances = []
        for est in self.estimators_:
            if not hasattr(est, "feature_importances_"):
                raise AttributeError(
                    f"Base estimator {type(est).__name__} does not provide "
                    "feature_importances_."
                )
            all_importances.append(est.feature_importances_)
        return np.mean(all_importances, axis=0)

    def score(self, X, Y, sample_weight=None):
        """Return the mean R^2 score across all outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        Y : array-like of shape (n_samples, n_outputs)
            True target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Mean of per-output R^2 scores.
        """
        from sklearn.metrics import r2_score

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        preds = self.predict(X)
        scores = [
            r2_score(Y[:, i], preds[:, i], sample_weight=sample_weight)
            for i in range(self.n_outputs_)
        ]
        return float(np.mean(scores))

    def _check_fitted(self):
        """Raise if the estimator has not been fitted."""
        if not hasattr(self, "estimators_") or self.estimators_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )


def _determine_chain_order(Y, order, random_state=None):
    """Determine the chain ordering for output columns.

    Parameters
    ----------
    Y : ndarray of shape (n_samples, n_outputs)
        Target matrix.
    order : str or list of int
        Chain ordering strategy:
        - ``'auto'``: order by descending absolute pairwise correlation
          so that highly correlated outputs are adjacent.
        - ``'random'``: random permutation.
        - list of int: explicit ordering.
    random_state : int, optional
        Random seed for ``'random'`` ordering.

    Returns
    -------
    list of int
        Column indices in chain order.
    """
    n_outputs = Y.shape[1]

    if isinstance(order, (list, np.ndarray)):
        order_list = list(order)
        if sorted(order_list) != list(range(n_outputs)):
            raise ValueError(
                f"order must be a permutation of range({n_outputs}), "
                f"got {order_list}"
            )
        return order_list

    if order == "random":
        rng = np.random.RandomState(random_state)
        return list(rng.permutation(n_outputs))

    if order == "auto":
        # Greedy nearest-neighbour walk on the correlation matrix.
        # Start with the output that has highest mean absolute correlation
        # (a well-connected hub), then always pick the unvisited output most
        # correlated with the current one.
        corr = np.abs(np.corrcoef(Y.T))
        np.fill_diagonal(corr, 0.0)
        remaining = set(range(n_outputs))
        # Start with highest mean correlation column
        start = int(np.argmax(corr.mean(axis=1)))
        chain = [start]
        remaining.remove(start)
        while remaining:
            current = chain[-1]
            # Among remaining, pick the one most correlated with current
            best = max(remaining, key=lambda j: corr[current, j])
            chain.append(best)
            remaining.remove(best)
        return chain

    raise ValueError(
        f"order must be 'auto', 'random', or a list of int, got {order!r}"
    )


class ClassifierChain(BaseEstimator, ClassifierMixin):
    """Chain classifiers where each uses predictions of previous outputs as features.

    Each classifier in the chain receives the original feature matrix X
    augmented with the predictions from all preceding classifiers. This
    allows the chain to model dependencies between outputs.

    Parameters
    ----------
    estimator : estimator
        The base classifier to clone for each link in the chain.
    order : str or list of int, default='auto'
        Chain ordering strategy:
        - ``'auto'``: greedy ordering by pairwise correlation so that
          adjacent outputs in the chain are maximally correlated.
        - ``'random'``: random permutation (seeded by ``random_state``).
        - list of int: explicit column ordering.
    n_jobs : int, optional
        Not used directly (chain is inherently sequential), but stored
        for API consistency.
    random_state : int, optional
        Random seed for random ordering and estimator cloning.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    estimators_ : List[estimator]
        Fitted classifiers in chain order.
    order_ : list of int
        The resolved output ordering.
    classes_ : List[ndarray]
        Class labels for each output (in original column order).
    n_outputs_ : int
        Number of output columns.

    Examples
    --------
    >>> from endgame.ensemble.multi_output import ClassifierChain
    >>> from sklearn.linear_model import LogisticRegression
    >>> import numpy as np
    >>> X = np.random.randn(200, 5)
    >>> Y = np.random.randint(0, 2, size=(200, 3))
    >>> chain = ClassifierChain(LogisticRegression(), order='auto')
    >>> chain.fit(X, Y)
    >>> preds = chain.predict(X)
    >>> preds.shape
    (200, 3)
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        order: str | list[int] = "auto",
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.order = order
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _make_estimator(self):
        """Clone the base estimator and inject random_state if supported."""
        est = clone(self.estimator)
        if self.random_state is not None and hasattr(est, "random_state"):
            est.random_state = self.random_state
        return est

    def fit(self, X, Y, sample_weight=None):
        """Fit the classifier chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples, n_outputs)
            Multi-output target matrix.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]
        self.classes_ = [np.unique(Y[:, i]) for i in range(self.n_outputs_)]
        self.order_ = _determine_chain_order(Y, self.order, self.random_state)

        self.estimators_ = []
        X_aug = X.copy()

        for idx in self.order_:
            est = self._make_estimator()
            _fit_single_estimator(est, X_aug, Y[:, idx], sample_weight)
            self.estimators_.append(est)
            # Augment X with the true labels for this output (teacher forcing)
            X_aug = np.column_stack([X_aug, Y[:, idx]])

        return self

    def predict(self, X):
        """Predict class labels for each output.

        At prediction time, the chain uses its own predictions (rather
        than ground truth) for augmentation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Predicted class labels in original column order.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        # predictions_by_order[k] = prediction for chain position k
        predictions_by_order = {}
        X_aug = X.copy()

        for k, idx in enumerate(self.order_):
            est = self.estimators_[k]
            pred = est.predict(X_aug)
            predictions_by_order[idx] = pred
            X_aug = np.column_stack([X_aug, pred])

        # Reassemble in original column order
        result = np.column_stack(
            [predictions_by_order[i] for i in range(self.n_outputs_)]
        )
        return result

    def predict_proba(self, X):
        """Predict class probabilities for each output.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        list of ndarray
            List of length ``n_outputs_`` (in original column order),
            where each element is an array of shape
            ``(n_samples, n_classes_k)``.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        probas_by_order = {}
        preds_by_order = {}
        X_aug = X.copy()

        for k, idx in enumerate(self.order_):
            est = self.estimators_[k]
            probas_by_order[idx] = est.predict_proba(X_aug)
            pred = est.predict(X_aug)
            preds_by_order[idx] = pred
            X_aug = np.column_stack([X_aug, pred])

        return [probas_by_order[i] for i in range(self.n_outputs_)]

    def score(self, X, Y, sample_weight=None):
        """Return the mean accuracy across all outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        Y : array-like of shape (n_samples, n_outputs)
            True labels.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Mean of per-output accuracy scores.
        """
        from sklearn.metrics import accuracy_score

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        preds = self.predict(X)
        scores = [
            accuracy_score(Y[:, i], preds[:, i], sample_weight=sample_weight)
            for i in range(self.n_outputs_)
        ]
        return float(np.mean(scores))

    def _check_fitted(self):
        """Raise if the estimator has not been fitted."""
        if not hasattr(self, "estimators_") or self.estimators_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )


class RegressorChain(BaseEstimator, RegressorMixin):
    """Chain regressors where each uses predictions of previous outputs as features.

    Each regressor in the chain receives the original feature matrix X
    augmented with the predictions from all preceding regressors. This
    allows the chain to model dependencies between outputs.

    Parameters
    ----------
    estimator : estimator
        The base regressor to clone for each link in the chain.
    order : str or list of int, default='auto'
        Chain ordering strategy:
        - ``'auto'``: greedy ordering by pairwise correlation so that
          adjacent outputs in the chain are maximally correlated.
        - ``'random'``: random permutation (seeded by ``random_state``).
        - list of int: explicit column ordering.
    n_jobs : int, optional
        Not used directly (chain is inherently sequential), but stored
        for API consistency.
    random_state : int, optional
        Random seed for random ordering and estimator cloning.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    estimators_ : List[estimator]
        Fitted regressors in chain order.
    order_ : list of int
        The resolved output ordering.
    n_outputs_ : int
        Number of output columns.

    Examples
    --------
    >>> from endgame.ensemble.multi_output import RegressorChain
    >>> from sklearn.linear_model import Ridge
    >>> import numpy as np
    >>> X = np.random.randn(200, 5)
    >>> Y = np.random.randn(200, 3)
    >>> chain = RegressorChain(Ridge(), order='auto')
    >>> chain.fit(X, Y)
    >>> preds = chain.predict(X)
    >>> preds.shape
    (200, 3)
    """

    def __init__(
        self,
        estimator: BaseEstimator = None,
        order: str | list[int] = "auto",
        n_jobs: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.estimator = estimator
        self.order = order
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

    def _make_estimator(self):
        """Clone the base estimator and inject random_state if supported."""
        est = clone(self.estimator)
        if self.random_state is not None and hasattr(est, "random_state"):
            est.random_state = self.random_state
        return est

    def fit(self, X, Y, sample_weight=None):
        """Fit the regressor chain.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        Y : array-like of shape (n_samples, n_outputs)
            Multi-output target matrix.
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        self.n_outputs_ = Y.shape[1]
        self.order_ = _determine_chain_order(Y, self.order, self.random_state)

        self.estimators_ = []
        X_aug = X.copy()

        for idx in self.order_:
            est = self._make_estimator()
            _fit_single_estimator(est, X_aug, Y[:, idx], sample_weight)
            self.estimators_.append(est)
            # Augment X with the true values for this output (teacher forcing)
            X_aug = np.column_stack([X_aug, Y[:, idx]])

        return self

    def predict(self, X):
        """Predict target values for each output.

        At prediction time, the chain uses its own predictions (rather
        than ground truth) for augmentation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to predict.

        Returns
        -------
        ndarray of shape (n_samples, n_outputs)
            Predicted values in original column order.
        """
        self._check_fitted()
        X = np.asarray(X, dtype=np.float64)

        predictions_by_order = {}
        X_aug = X.copy()

        for k, idx in enumerate(self.order_):
            est = self.estimators_[k]
            pred = est.predict(X_aug)
            predictions_by_order[idx] = pred
            X_aug = np.column_stack([X_aug, pred])

        result = np.column_stack(
            [predictions_by_order[i] for i in range(self.n_outputs_)]
        )
        return result

    @property
    def feature_importances_(self):
        """Average feature importances across chain estimators.

        Only includes importances for the original features (not the
        chained predictions), averaged across all estimators.

        Returns
        -------
        ndarray of shape (n_features,)
            Mean feature importances for the original features.

        Raises
        ------
        AttributeError
            If the base estimators do not expose ``feature_importances_``.
        """
        self._check_fitted()
        # The first estimator sees n_features original features.
        # The k-th estimator sees n_features + k chained columns.
        # We extract only the first n_features importances from each.
        first_est = self.estimators_[0]
        if not hasattr(first_est, "feature_importances_"):
            raise AttributeError(
                f"Base estimator {type(first_est).__name__} does not provide "
                "feature_importances_."
            )
        n_features = len(first_est.feature_importances_)
        all_importances = []
        for est in self.estimators_:
            imp = est.feature_importances_[:n_features]
            all_importances.append(imp)
        return np.mean(all_importances, axis=0)

    def score(self, X, Y, sample_weight=None):
        """Return the mean R^2 score across all outputs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Test samples.
        Y : array-like of shape (n_samples, n_outputs)
            True target values.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        float
            Mean of per-output R^2 scores.
        """
        from sklearn.metrics import r2_score

        Y = np.asarray(Y)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        preds = self.predict(X)
        scores = [
            r2_score(Y[:, i], preds[:, i], sample_weight=sample_weight)
            for i in range(self.n_outputs_)
        ]
        return float(np.mean(scores))

    def _check_fitted(self):
        """Raise if the estimator has not been fitted."""
        if not hasattr(self, "estimators_") or self.estimators_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been fitted. "
                "Call 'fit' before using this estimator."
            )
