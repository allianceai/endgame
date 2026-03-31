"""FCM initialization with RDA training for TSK fuzzy systems.

Combines Fuzzy C-Means clustering for rule antecedent initialization
with MBGD-RDA training for parameter optimization.

References
----------
Bezdek, J.C. (1981). Pattern Recognition with Fuzzy Objective Function
Algorithms. Plenum Press.

Example
-------
>>> from endgame.fuzzy.modern.fcm_rdpa import FCMRDpARegressor
>>> reg = FCMRDpARegressor(n_rules=10, n_epochs=100)
>>> reg.fit(X_train, y_train)
>>> predictions = reg.predict(X_test)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from endgame.fuzzy.modern.mbgd_rda import MBGDRDATrainer


def _fuzzy_c_means(
    X: np.ndarray,
    n_clusters: int,
    fuzziness: float = 2.0,
    max_iter: int = 100,
    tol: float = 1e-5,
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run Fuzzy C-Means clustering.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Input data.
    n_clusters : int
        Number of clusters.
    fuzziness : float, default=2.0
        Fuzziness coefficient (m > 1).
    max_iter : int, default=100
        Maximum iterations.
    tol : float, default=1e-5
        Convergence tolerance.
    rng : RandomState or None
        Random number generator.

    Returns
    -------
    centers : ndarray of shape (n_clusters, n_features)
        Cluster centers.
    membership : ndarray of shape (n_samples, n_clusters)
        Fuzzy membership matrix.
    sigmas : ndarray of shape (n_clusters, n_features)
        Cluster spread (standard deviation) per feature.
    """
    if rng is None:
        rng = np.random.RandomState()

    n_samples, n_features = X.shape
    n_clusters = min(n_clusters, n_samples)

    # Initialize membership matrix randomly
    U = rng.rand(n_samples, n_clusters)
    U = U / U.sum(axis=1, keepdims=True)

    for _ in range(max_iter):
        # Update centers
        Um = U ** fuzziness
        um_sum = Um.sum(axis=0)  # (n_clusters,)
        um_sum = np.where(um_sum == 0, 1.0, um_sum)
        centers = (Um.T @ X) / um_sum[:, None]

        # Update membership
        dist = np.zeros((n_samples, n_clusters))
        for c in range(n_clusters):
            diff = X - centers[c]
            dist[:, c] = np.sum(diff ** 2, axis=1)
        dist = np.maximum(dist, 1e-10)

        U_new = np.zeros_like(U)
        exp = 2.0 / (fuzziness - 1.0)
        for c in range(n_clusters):
            denom = np.sum(
                (dist[:, c:c+1] / dist) ** exp,
                axis=1,
            )
            U_new[:, c] = 1.0 / denom

        if np.max(np.abs(U_new - U)) < tol:
            U = U_new
            break
        U = U_new

    # Compute cluster spreads
    Um = U ** fuzziness
    um_sum = Um.sum(axis=0)
    um_sum = np.where(um_sum == 0, 1.0, um_sum)
    sigmas = np.zeros((n_clusters, n_features))
    for c in range(n_clusters):
        diff = X - centers[c]
        sigmas[c] = np.sqrt(
            np.sum(Um[:, c:c+1] * diff ** 2, axis=0) / um_sum[c]
        )
    sigmas = np.maximum(sigmas, 1e-4)

    return centers, U, sigmas


class FCMRDpARegressor(BaseEstimator, RegressorMixin):
    """TSK regressor with FCM initialization and MBGD-RDA training.

    Initializes fuzzy rule antecedents via Fuzzy C-Means clustering,
    then refines all parameters using mini-batch gradient descent
    with DropRule regularization and AdaBound optimizer.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules (= number of FCM clusters).
    fuzziness : float, default=2.0
        FCM fuzziness coefficient.
    droprule_rate : float, default=0.5
        Fraction of rules to drop during training.
    n_epochs : int, default=100
        Number of MBGD-RDA training epochs.
    lr : float, default=0.01
        Learning rate.
    l2_reg : float, default=0.01
        L2 regularization strength.
    batch_size : int, default=32
        Mini-batch size.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    centers_ : ndarray of shape (n_rules, n_features)
        Rule antecedent centers (initialized by FCM).
    sigmas_ : ndarray of shape (n_rules, n_features)
        Rule antecedent widths (initialized from FCM spread).
    consequent_params_ : ndarray of shape (n_rules, n_features + 1)
        Rule consequent parameters.
    n_features_in_ : int
        Number of features seen during fit.
    loss_history_ : list of float
        Training loss per epoch.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.fcm_rdpa import FCMRDpARegressor
    >>> X = np.random.randn(200, 5)
    >>> y = X[:, 0] ** 2 + X[:, 1] + np.random.randn(200) * 0.1
    >>> reg = FCMRDpARegressor(n_rules=5, n_epochs=50)
    >>> reg.fit(X, y)
    FCMRDpARegressor(n_epochs=50, n_rules=5)
    >>> preds = reg.predict(X[:5])
    """

    def __init__(
        self,
        n_rules: int = 10,
        fuzziness: float = 2.0,
        droprule_rate: float = 0.5,
        n_epochs: int = 100,
        lr: float = 0.01,
        l2_reg: float = 0.01,
        batch_size: int = 32,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.fuzziness = fuzziness
        self.droprule_rate = droprule_rate
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FCMRDpARegressor:
        """Fit using FCM initialization followed by MBGD-RDA training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
            Fitted estimator.
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Step 1: FCM clustering for antecedent initialization
        self.centers_, _, self.sigmas_ = _fuzzy_c_means(
            X,
            n_clusters=self.n_rules,
            fuzziness=self.fuzziness,
            rng=rng,
        )

        # Step 2: Initialize consequent params via least squares
        firing = MBGDRDATrainer._compute_firing(X, self.centers_, self.sigmas_)
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum

        n_rules = self.centers_.shape[0]
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

        # Step 3: MBGD-RDA training
        trainer = MBGDRDATrainer(
            lr=self.lr,
            droprule_rate=self.droprule_rate,
            l2_reg=self.l2_reg,
            batch_size=self.batch_size,
            n_epochs=self.n_epochs,
            random_state=self.random_state,
            verbose=self.verbose,
        )
        self.loss_history_ = trainer.train(self, X, y)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict target values.

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

        firing = MBGDRDATrainer._compute_firing(X, self.centers_, self.sigmas_)
        fire_sum = firing.sum(axis=1, keepdims=True)
        fire_sum = np.where(fire_sum == 0, 1.0, fire_sum)
        normalized = firing / fire_sum

        X_aug = np.column_stack([X, np.ones(len(X))])
        n_rules = self.centers_.shape[0]
        consequents = np.zeros((len(X), n_rules))
        for r in range(n_rules):
            consequents[:, r] = X_aug @ self.consequent_params_[r]

        return np.sum(normalized * consequents, axis=1)


class FCMRDpAClassifier(BaseEstimator, ClassifierMixin):
    """TSK classifier with FCM initialization and MBGD-RDA training.

    Uses one-vs-rest decomposition with FCMRDpARegressor per class.

    Parameters
    ----------
    n_rules : int, default=10
        Number of fuzzy rules.
    fuzziness : float, default=2.0
        FCM fuzziness coefficient.
    droprule_rate : float, default=0.5
        Fraction of rules to drop during training.
    n_epochs : int, default=100
        Number of training epochs.
    lr : float, default=0.01
        Learning rate.
    l2_reg : float, default=0.01
        L2 regularization strength.
    batch_size : int, default=32
        Mini-batch size.
    random_state : int or None, default=None
        Random seed.
    verbose : bool, default=False
        Print training progress.

    Attributes
    ----------
    classes_ : ndarray of shape (n_classes,)
        Unique class labels.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.fuzzy.modern.fcm_rdpa import FCMRDpAClassifier
    >>> X = np.random.randn(100, 5)
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = FCMRDpAClassifier(n_rules=5, n_epochs=30)
    >>> clf.fit(X, y)
    FCMRDpAClassifier(n_epochs=30, n_rules=5)
    >>> clf.predict(X[:3])
    array([...])
    """

    def __init__(
        self,
        n_rules: int = 10,
        fuzziness: float = 2.0,
        droprule_rate: float = 0.5,
        n_epochs: int = 100,
        lr: float = 0.01,
        l2_reg: float = 0.01,
        batch_size: int = 32,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        self.n_rules = n_rules
        self.fuzziness = fuzziness
        self.droprule_rate = droprule_rate
        self.n_epochs = n_epochs
        self.lr = lr
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X: Any, y: Any) -> FCMRDpAClassifier:
        """Fit using one FCMRDpARegressor per class.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target class labels.

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
            reg = FCMRDpARegressor(
                n_rules=self.n_rules,
                fuzziness=self.fuzziness,
                droprule_rate=self.droprule_rate,
                n_epochs=self.n_epochs,
                lr=self.lr,
                l2_reg=self.l2_reg,
                batch_size=self.batch_size,
                random_state=self.random_state,
                verbose=self.verbose,
            )
            reg.fit(X, y_binary)
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
            Predicted class labels.
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
            Class probabilities.
        """
        check_is_fitted(self, ["regressors_"])
        X = check_array(X)

        raw = np.column_stack([reg.predict(X) for reg in self.regressors_])
        raw_shifted = raw - raw.max(axis=1, keepdims=True)
        exp_raw = np.exp(raw_shifted)
        return exp_raw / exp_raw.sum(axis=1, keepdims=True)
