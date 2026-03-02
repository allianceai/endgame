from __future__ import annotations

"""Pseudo-labeling for semi-supervised learning."""

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, clone

from endgame.core.base import EndgameEstimator


class PseudoLabelTrainer(EndgameEstimator):
    """Semi-supervised learning via pseudo-labeling.

    Algorithm:
    1. Train model on labeled data
    2. Predict on unlabeled data
    3. Add high-confidence predictions to training set
    4. Retrain

    Parameters
    ----------
    base_estimator : estimator
        Base model to use.
    confidence_threshold : float, default=0.9
        Minimum confidence to use pseudo-label.
    soft_labels : bool, default=True
        Use probability distribution instead of hard labels.
    n_iterations : int, default=2
        Number of pseudo-labeling rounds.
    sample_weight_decay : float, default=0.5
        Weight decay for pseudo-labeled samples relative to real labels.
    max_pseudo_samples : int, optional
        Maximum pseudo-labeled samples per iteration.

    Examples
    --------
    >>> pseudo = PseudoLabelTrainer(
    ...     base_estimator=TransformerClassifier(),
    ...     confidence_threshold=0.95
    ... )
    >>> pseudo.fit(model, X_train, y_train, X_unlabeled=X_test)
    """

    def __init__(
        self,
        base_estimator: BaseEstimator | None = None,
        confidence_threshold: float = 0.9,
        soft_labels: bool = True,
        n_iterations: int = 2,
        sample_weight_decay: float = 0.5,
        max_pseudo_samples: int | None = None,
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.base_estimator = base_estimator
        self.confidence_threshold = confidence_threshold
        self.soft_labels = soft_labels
        self.n_iterations = n_iterations
        self.sample_weight_decay = sample_weight_decay
        self.max_pseudo_samples = max_pseudo_samples

        self._model = None
        self._pseudo_labels: np.ndarray | None = None
        self._pseudo_indices: np.ndarray | None = None

    def fit(
        self,
        estimator: BaseEstimator | None,
        X_labeled: Any,
        y_labeled: np.ndarray,
        X_unlabeled: Any | None = None,
        **fit_params,
    ) -> PseudoLabelTrainer:
        """Train with pseudo-labeling.

        Parameters
        ----------
        estimator : estimator
            Pre-trained model or None to train from scratch.
        X_labeled : array-like
            Labeled training data.
        y_labeled : array-like
            Labels for training data.
        X_unlabeled : array-like, optional
            Unlabeled data for pseudo-labeling.

        Returns
        -------
        self
        """
        if estimator is not None:
            self._model = estimator
        elif self.base_estimator is not None:
            self._model = clone(self.base_estimator)
        else:
            raise ValueError("Either estimator or base_estimator must be provided")

        y_labeled = np.asarray(y_labeled)

        if X_unlabeled is None:
            # No unlabeled data, just fit normally
            self._model.fit(X_labeled, y_labeled, **fit_params)
            self._is_fitted = True
            return self

        # Iterative pseudo-labeling
        for iteration in range(self.n_iterations):
            self._log(f"Pseudo-labeling iteration {iteration + 1}/{self.n_iterations}")

            # Get predictions on unlabeled data
            if hasattr(self._model, "predict_proba"):
                proba = self._model.predict_proba(X_unlabeled)
                max_proba = proba.max(axis=1)
                pseudo_labels = proba.argmax(axis=1)
            else:
                pseudo_labels = self._model.predict(X_unlabeled)
                max_proba = np.ones(len(pseudo_labels))

            # Select high-confidence predictions
            confident_mask = max_proba >= self.confidence_threshold
            n_confident = confident_mask.sum()

            self._log(f"  Found {n_confident} confident samples")

            if n_confident == 0:
                self._log("  No confident samples, stopping early")
                break

            # Apply max samples limit
            if self.max_pseudo_samples and n_confident > self.max_pseudo_samples:
                confident_indices = np.where(confident_mask)[0]
                rng = np.random.RandomState(self.random_state)
                selected = rng.choice(
                    confident_indices,
                    self.max_pseudo_samples,
                    replace=False,
                )
                confident_mask = np.zeros(len(max_proba), dtype=bool)
                confident_mask[selected] = True

            self._pseudo_indices = np.where(confident_mask)[0]
            self._pseudo_labels = pseudo_labels[confident_mask]

            # Combine labeled and pseudo-labeled data
            if isinstance(X_labeled, list):
                X_combined = X_labeled + [X_unlabeled[i] for i in self._pseudo_indices]
            elif isinstance(X_labeled, np.ndarray):
                X_pseudo = X_unlabeled[self._pseudo_indices]
                X_combined = np.vstack([X_labeled, X_pseudo])
            else:
                # Assume it supports indexing
                X_combined = list(X_labeled) + [X_unlabeled[i] for i in self._pseudo_indices]

            y_combined = np.concatenate([y_labeled, self._pseudo_labels])

            # Sample weights (lower for pseudo-labels)
            sample_weight = np.ones(len(y_combined))
            sample_weight[len(y_labeled):] = self.sample_weight_decay ** (iteration + 1)

            # Retrain
            self._model = clone(self._model)

            try:
                self._model.fit(X_combined, y_combined, sample_weight=sample_weight, **fit_params)
            except TypeError:
                # Model doesn't support sample_weight
                self._model.fit(X_combined, y_combined, **fit_params)

        self._is_fitted = True
        return self

    def predict(self, X) -> np.ndarray:
        """Predict using trained model."""
        self._check_is_fitted()
        return self._model.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using trained model."""
        self._check_is_fitted()
        return self._model.predict_proba(X)

    @property
    def pseudo_labels_(self) -> np.ndarray | None:
        """Labels assigned during pseudo-labeling."""
        return self._pseudo_labels

    @property
    def pseudo_indices_(self) -> np.ndarray | None:
        """Indices of pseudo-labeled samples."""
        return self._pseudo_indices
