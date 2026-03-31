"""TSK with Privileged Information (LUPI).

Extends TSK fuzzy systems to leverage privileged features available
only at training time for improved rule formation.

References
----------
Vapnik, V., & Vashist, A. (2009). A new learning paradigm: Learning
Using Privileged Information. Neural Networks, 22(5-6), 544-557.

Example
-------
>>> from endgame.fuzzy.modern.tsk_plus import TSKPlusRegressor
>>> reg = TSKPlusRegressor(n_rules=10)
>>> reg.fit(X_train, y_train, X_privileged=X_priv_train)
>>> predictions = reg.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _init_tsk_params(
    X: np.ndarray,
    n_rules: int,
    n_mfs: int,
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initialize TSK parameters from data.

    Returns
    -------
    centers : ndarray of shape (n_rules, n_features)
    sigmas : ndarray of shape (n_rules, n_features)
    consequent_params : ndarray of shape (n_rules, n_features + 1)
    """
    n_samples, n_features = X.shape
    n_rules = min(n_rules, n_samples)

    indices = rng.choice(n_samples, size=n_rules, replace=False)
    centers = X[indices].copy()

    stds = np.std(X, axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    sigmas = np.tile(stds, (n_rules, 1))

    consequent_params = rng.randn(n_rules, n_features + 1) * 0.01

    return centers, sigmas, consequent_params


def _compute_firing(
    X: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Compute Gaussian firing strengths."""
    n_rules = centers.shape[0]
    firing = np.zeros((X.shape[0], n_rules))
    for r in range(n_rules):
        diff = X - centers[r]
        sigma_sq = sigmas[r] ** 2 + 1e-10
        firing[:, r] = np.exp(-0.5 * np.sum(diff ** 2 / sigma_sq, axis=1))
    return firing


def _tsk_predict(
    X: np.ndarray,
    centers: np.ndarray,
    sigmas: np.ndarray,
    consequent_params: np.ndarray,
) -> np.ndarray:
    """Compute TSK output."""
    firing = _compute_firing(X, centers, sigmas)
    fire_sum = firing.sum(axis=1, keepdims=True)
    fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
    normalized = firing / fire_sum

    X_aug = np.column_stack([X, np.ones(len(X))])
    n_rules = centers.shape[0]
    consequents = np.zeros((len(X), n_rules))
    for r in range(n_rules):
        consequents[:, r] = X_aug @ consequent_params[r]

    return np.sum(normalized * consequents, axis=1)


class TSKPlusRegressor(BaseEstimator, RegressorMixin):
    """TSK fuzzy regressor with Learning Using Privileged Information (LUPI).

    Extra features available only at training time guide rule structure
    learning, while consequent parameters use only regular features.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    order : int, default=1
        TSK order (0 or 1).
    n_mfs : int, default=3
        Number of membership functions per feature.
    n_epochs : int, default=50
        Training epochs for parameter refinement.
    lr : float, default=0.01
        Learning rate.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Rule antecedent centers (regular features only).
    sigmas_ : ndarray of shape (n_rules, n_features)
        Rule antecedent widths.
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        Rule consequent parameters.
    n_features_in_ : int
        Number of regular features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.tsk_plus import TSKPlusRegressor
    >>> X = np.random.randn(200, 5)
    >>> X_priv = np.random.randn(200, 3)  # privileged features
    >>> y = X[:, 0] + X_priv[:, 0] + np.random.randn(200) * 0.1
    >>> reg = TSKPlusRegressor(n_rules=5, n_epochs=30)
    >>> reg.fit(X, y, X_privileged=X_priv)
    TSKPlusRegressor(n_epochs=30, n_rules=5)
    >>> preds = reg.predict(X[:5])  # only regular features needed
    """

    def __init__(
        self,
        n_rules: int = 10,
        order: int = 1,
        n_mfs: int = 3,
        n_epochs: int = 50,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.order = order
        self.n_mfs = n_mfs
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Any,
        y: Any,
        X_privileged: np.ndarray | None = None,
    ) -> TSKPlusRegressor:
        """Fit the TSK+ model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Regular training features.
        y : array-like of shape (n_samples,)
            Target values.
        X_privileged : array-like of shape (n_samples, n_priv_features) or None
            Privileged features available only at training time.
            If None, behaves like a standard TSK system.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Step 1: Rule structure learning using combined features
        if X_privileged is not None:
            X_privileged = check_array(X_privileged)
            if X_privileged.shape[0] != n_samples:
                raise ValueError(
                    f"X_privileged has {X_privileged.shape[0]} samples "
                    f"but X has {n_samples}"
                )
            X_combined = np.column_stack([X, X_privileged])
        else:
            X_combined = X

        # Initialize rule centers from combined space
        n_rules = min(self.n_rules, n_samples)
        indices = rng.choice(n_samples, size=n_rules, replace=False)
        centers_combined = X_combined[indices].copy()

        stds_combined = np.std(X_combined, axis=0)
        stds_combined = np.where(stds_combined < 1e-6, 1.0, stds_combined)
        sigmas_combined = np.tile(stds_combined, (n_rules, 1))

        # Compute firing strengths using combined features
        firing_combined = _compute_firing(X_combined, centers_combined, sigmas_combined)

        # Step 2: Project rule centers back to regular feature space
        self.centers_ = centers_combined[:, :n_features].copy()
        self.sigmas_ = sigmas_combined[:, :n_features].copy()

        # Step 3: Learn consequent params using regular features only
        fire_sum = firing_combined.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing_combined / fire_sum

        X_aug = np.column_stack([X, np.ones(n_samples)])
        self.consequent_params_ = np.zeros((n_rules, n_features + 1))

        for r in range(n_rules):
            weights = normalized[:, r]
            W = np.diag(weights + 1e-10)
            try:
                self.consequent_params_[r] = np.linalg.lstsq(
                    W @ X_aug, W @ y, rcond=None
                )[0]
            except np.linalg.LinAlgError:
                self.consequent_params_[r] = rng.randn(n_features + 1) * 0.01

        # Step 4: Fine-tune with gradient descent on regular features
        for epoch in range(self.n_epochs):
            firing = _compute_firing(X, self.centers_, self.sigmas_)
            fire_sum = firing.sum(axis=1, keepdims=True)
            fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
            normalized = firing / fire_sum

            consequents = np.zeros((n_samples, n_rules))
            for r in range(n_rules):
                consequents[:, r] = X_aug @ self.consequent_params_[r]

            output = np.sum(normalized * consequents, axis=1)
            error = output - y
            loss = float(np.mean(error ** 2))

            if self.verbose and (epoch + 1) % 10 == 0:
                print(
                    f"[TSK+] Epoch {epoch+1}/{self.n_epochs}, Loss: {loss:.6f}"
                )

            # Update consequent params
            for r in range(n_rules):
                grad = (error * normalized[:, r])[:, None] * X_aug
                self.consequent_params_[r] -= self.lr * grad.mean(axis=0)

            # Update centers and sigmas
            total_output = np.sum(normalized * consequents, axis=1)
            for r in range(n_rules):
                diff = X - self.centers_[r]
                sigma_sq = self.sigmas_[r] ** 2 + 1e-10

                d_norm = (consequents[:, r] - total_output) / (
                    firing.sum(axis=1) + 1e-10
                )

                d_fire_center = firing[:, r:r+1] * diff / sigma_sq
                grad_center = (error * d_norm)[:, None] * d_fire_center
                self.centers_[r] -= self.lr * np.clip(
                    grad_center.mean(axis=0), -1.0, 1.0
                )

                d_fire_sigma = firing[:, r:r+1] * diff ** 2 / (
                    self.sigmas_[r] ** 3 + 1e-10
                )
                grad_sigma = (error * d_norm)[:, None] * d_fire_sigma
                self.sigmas_[r] -= self.lr * np.clip(
                    grad_sigma.mean(axis=0), -1.0, 1.0
                )
                self.sigmas_[r] = np.maximum(self.sigmas_[r], 1e-4)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values using regular features only.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, ["centers_", "sigmas_", "consequent_params_"])
        X = check_array(X)
        return _tsk_predict(X, self.centers_, self.sigmas_, self.consequent_params_)


class TSKPlusClassifier(BaseEstimator, ClassifierMixin):
    """TSK classifier with Learning Using Privileged Information (LUPI).

    Uses one-vs-rest decomposition with TSKPlusRegressor per class.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    order : int, default=1
        TSK order.
    n_mfs : int, default=3
        Number of membership functions per feature.
    n_epochs : int, default=50
        Training epochs.
    lr : float, default=0.01
        Learning rate.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of regular features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.tsk_plus import TSKPlusClassifier
    >>> X = np.random.randn(100, 5)
    >>> X_priv = np.random.randn(100, 3)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = TSKPlusClassifier(n_rules=5, n_epochs=20)
    >>> clf.fit(X, y, X_privileged=X_priv)
    TSKPlusClassifier(n_epochs=20, n_rules=5)
    >>> clf.predict(X[:3])
    array([...])
    """

    def __init__(
        self,
        n_rules: int = 10,
        order: int = 1,
        n_mfs: int = 3,
        n_epochs: int = 50,
        lr: float = 0.01,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.order = order
        self.n_mfs = n_mfs
        self.n_epochs = n_epochs
        self.lr = lr
        self.random_state = random_state
        self.verbose = verbose

    def fit(
        self,
        X: Any,
        y: Any,
        X_privileged: np.ndarray | None = None,
    ) -> TSKPlusClassifier:
        """Fit using one TSKPlusRegressor per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Regular training features.
        y : array-like of shape (n_samples,)
            Target class labels.
        X_privileged : array-like of shape (n_samples, n_priv_features) or None
            Privileged features (training only).

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        self.label_encoder_ = LabelEncoder()
        y_enc = self.label_encoder_.fit_transform(y)
        self.classes_ = self.label_encoder_.classes_

        self.regressors_ = []
        for c in range(len(self.classes_)):
            y_binary = (y_enc == c).astype(np.float64)
            reg = TSKPlusRegressor(
                n_rules=self.n_rules,
                order=self.order,
                n_mfs=self.n_mfs,
                n_epochs=self.n_epochs,
                lr=self.lr,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary, X_privileged=X_privileged)
            self.regressors_.append(reg)

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        ndarray of shape (n_samples, n_classes)
        """
        check_is_fitted(self, ["regressors_"])
        X = check_array(X)

        raw = np.column_stack([reg.predict(X) for reg in self.regressors_])
        raw_shifted = raw - raw.max(axis=1, keepdims=True)
        exp_raw = np.exp(raw_shifted)
        return exp_raw / exp_raw.sum(axis=1, keepdims=True)
