"""Snapshot Ensemble: Ensemble from learning rate schedule snapshots.

Collects model snapshots at the end of each warm-restart cycle
(cosine annealing learning rate schedule) and averages their
predictions. Gets the diversity of an ensemble for the cost of
training a single model.

Theory: Huang et al. (2017), "Snapshot Ensembles: Train 1, Get M for Free."

Example
-------
>>> from endgame.ensemble import SnapshotEnsemble
>>> snap = SnapshotEnsemble(
...     base_estimator=MLPClassifier(max_iter=1),
...     n_snapshots=5,
...     epochs_per_cycle=40,
...     initial_lr=0.1,
... )
>>> snap.fit(X_train, y_train)
>>> snap.predict(X_test)
"""

from __future__ import annotations

import copy

import numpy as np
from sklearn.base import BaseEstimator, clone


class SnapshotEnsemble(BaseEstimator):
    """Snapshot Ensemble via cosine annealing warm restarts.

    Trains a single neural-network-like estimator with a cyclic
    learning rate schedule. At the end of each cycle (when LR reaches
    its minimum), takes a "snapshot" of the model. The final ensemble
    averages predictions across all snapshots.

    Parameters
    ----------
    base_estimator : estimator
        A model supporting ``partial_fit`` (e.g., ``MLPClassifier``,
        ``SGDClassifier``, ``SGDRegressor``). Must accept
        ``learning_rate_init`` or ``eta0``.
    n_snapshots : int, default=5
        Number of snapshots (cycles) to collect.
    epochs_per_cycle : int, default=40
        Training epochs per cosine annealing cycle.
    initial_lr : float, default=0.1
        Peak learning rate at the start of each cycle.
    min_lr : float, default=1e-5
        Minimum learning rate at end of each cycle (snapshot point).
    verbose : bool, default=False

    Attributes
    ----------
    snapshots_ : list of estimator
        Saved model snapshots.
    lr_history_ : list of float
        Learning rate at each epoch.
    is_classifier_ : bool

    References
    ----------
    Huang, G., Li, Y., Pleiss, G., Liu, Z., Hopcroft, J.E., &
    Weinberger, K.Q. (2017). Snapshot Ensembles: Train 1, Get M
    for Free. *ICLR*.
    """

    def __init__(
        self,
        base_estimator: BaseEstimator,
        n_snapshots: int = 5,
        epochs_per_cycle: int = 40,
        initial_lr: float = 0.1,
        min_lr: float = 1e-5,
        verbose: bool = False,
    ):
        self.base_estimator = base_estimator
        self.n_snapshots = n_snapshots
        self.epochs_per_cycle = epochs_per_cycle
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.verbose = verbose

    def _cosine_lr(self, epoch_in_cycle: int) -> float:
        """Cosine annealing learning rate for current epoch within a cycle."""
        t = epoch_in_cycle / max(self.epochs_per_cycle, 1)
        return self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + np.cos(np.pi * t))

    def fit(self, X, y, **fit_params):
        """Train with cyclic LR and collect snapshots.

        Parameters
        ----------
        X : array-like
        y : array-like
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Detect task type
        self.is_classifier_ = len(np.unique(y)) <= 30 and (
            np.issubdtype(y.dtype, np.integer) or len(np.unique(y)) <= 10
        )
        if self.is_classifier_:
            self.classes_ = np.unique(y)

        model = clone(self.base_estimator)

        # Check for partial_fit support
        if not hasattr(model, "partial_fit"):
            raise ValueError(
                "base_estimator must support partial_fit for snapshot ensemble. "
                "Use MLPClassifier, SGDClassifier, SGDRegressor, etc."
            )

        self.snapshots_ = []
        self.lr_history_ = []

        for cycle in range(self.n_snapshots):
            if self.verbose:
                print(f"[Snapshot] Cycle {cycle + 1}/{self.n_snapshots}")

            for epoch in range(self.epochs_per_cycle):
                lr = self._cosine_lr(epoch)
                self.lr_history_.append(lr)

                # Set learning rate
                if hasattr(model, "learning_rate_init"):
                    model.learning_rate_init = lr
                if hasattr(model, "eta0"):
                    model.eta0 = lr

                # partial_fit
                if self.is_classifier_ and cycle == 0 and epoch == 0 or self.is_classifier_:
                    model.partial_fit(X, y, classes=self.classes_)
                else:
                    model.partial_fit(X, y)

            # Take snapshot at end of cycle (minimum LR)
            snapshot = copy.deepcopy(model)
            self.snapshots_.append(snapshot)

            if self.verbose:
                if hasattr(snapshot, "score"):
                    score = snapshot.score(X, y)
                    print(f"  Snapshot score: {score:.4f}")

        return self

    def predict(self, X):
        X = np.asarray(X)
        if self.is_classifier_:
            proba = self.predict_proba(X)
            return self.classes_[np.argmax(proba, axis=1)]

        preds = np.array([s.predict(X) for s in self.snapshots_])
        return preds.mean(axis=0)

    def predict_proba(self, X):
        if not self.is_classifier_:
            raise ValueError("predict_proba only for classification.")
        X = np.asarray(X)
        probas = np.array([s.predict_proba(X) for s in self.snapshots_])
        return probas.mean(axis=0)
