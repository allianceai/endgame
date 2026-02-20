"""Knowledge Distillation for model compression.

Compress ensemble or complex models into simpler, deployable models
while preserving most of the predictive performance.

Example
-------
>>> from endgame.ensemble import KnowledgeDistiller
>>> from sklearn.ensemble import GradientBoostingClassifier
>>> from sklearn.tree import DecisionTreeClassifier
>>>
>>> teacher = GradientBoostingClassifier(n_estimators=200).fit(X, y)
>>> distiller = KnowledgeDistiller(
...     teacher=teacher,
...     student=DecisionTreeClassifier(max_depth=6),
...     temperature=3.0
... )
>>> student = distiller.fit(X, y)
>>> predictions = student.predict(X_test)
"""

from __future__ import annotations

import numpy as np
from sklearn.base import (
    BaseEstimator,
    clone,
    is_classifier,
)
from sklearn.utils.validation import check_is_fitted


def _softmax_with_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Apply softmax with temperature scaling."""
    scaled = logits / temperature
    # Numerical stability
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / exp_scaled.sum(axis=1, keepdims=True)


def _munge_augment(
    X: np.ndarray,
    y: np.ndarray,
    n_augmented: int,
    swap_prob: float = 0.1,
    random_state: int | None = None,
) -> tuple:
    """MUNGE data augmentation for knowledge distillation.

    Creates synthetic training data by swapping feature values between
    nearest neighbor pairs with some probability.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
    y : ndarray of shape (n_samples,) or (n_samples, n_classes)
        Hard labels or soft probabilities.
    n_augmented : int
        Number of augmented samples to generate.
    swap_prob : float
        Probability of swapping each feature.
    random_state : int or None

    Returns
    -------
    X_aug, y_aug : augmented data
    """
    rng = np.random.RandomState(random_state)
    n_samples, n_features = X.shape

    X_aug = np.empty((n_augmented, n_features), dtype=X.dtype)
    if y.ndim == 1:
        y_aug = np.empty(n_augmented, dtype=y.dtype)
    else:
        y_aug = np.empty((n_augmented, y.shape[1]), dtype=y.dtype)

    for i in range(n_augmented):
        # Pick a random sample
        idx = rng.randint(0, n_samples)
        x_base = X[idx].copy()
        y_base = y[idx].copy()

        # Pick a random neighbor (simple random for efficiency)
        neighbor_idx = rng.randint(0, n_samples)
        while neighbor_idx == idx:
            neighbor_idx = rng.randint(0, n_samples)

        # Swap features with probability
        swap_mask = rng.random(n_features) < swap_prob
        x_base[swap_mask] = X[neighbor_idx, swap_mask]

        X_aug[i] = x_base
        y_aug[i] = y_base

    return X_aug, y_aug


class KnowledgeDistiller(BaseEstimator):
    """Knowledge distillation from teacher to student model.

    Trains a simpler student model to mimic the predictions of a
    complex teacher model (or ensemble), enabling deployment of
    lightweight models with minimal accuracy loss.

    Parameters
    ----------
    teacher : estimator
        Fitted teacher model. Must have predict_proba (classification)
        or predict (regression).
    student : estimator
        Unfitted student model to train.
    temperature : float, default=3.0
        Softmax temperature for soft label generation (classification only).
        Higher values produce softer probability distributions that reveal
        more about the teacher's learned relationships.
    alpha : float, default=0.7
        Weight for soft labels vs hard labels.
        Loss = alpha * soft_loss + (1 - alpha) * hard_loss.
        Set to 1.0 for pure distillation.
    augment : bool, default=False
        Whether to use MUNGE data augmentation to generate additional
        training data labeled by the teacher.
    augment_ratio : float, default=1.0
        Ratio of augmented samples to original samples.
    augment_swap_prob : float, default=0.1
        Feature swap probability for MUNGE augmentation.
    random_state : int or None, default=None
        Random state.

    Attributes
    ----------
    student_ : estimator
        The trained student model.
    teacher_score_ : float or None
        Teacher's accuracy/R2 on training data (for reference).
    student_score_ : float or None
        Student's accuracy/R2 on training data.
    is_classifier_ : bool
        Whether this is a classification task.

    Example
    -------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> teacher = RandomForestClassifier(n_estimators=500).fit(X, y)
    >>> distiller = KnowledgeDistiller(
    ...     teacher=teacher,
    ...     student=LogisticRegression(),
    ...     temperature=4.0,
    ...     alpha=0.8,
    ...     augment=True
    ... )
    >>> distiller.fit(X, y)
    >>> y_pred = distiller.predict(X_test)
    """

    def __init__(
        self,
        teacher,
        student,
        temperature: float = 3.0,
        alpha: float = 0.7,
        augment: bool = False,
        augment_ratio: float = 1.0,
        augment_swap_prob: float = 0.1,
        random_state: int | None = None,
    ):
        self.teacher = teacher
        self.student = student
        self.temperature = temperature
        self.alpha = alpha
        self.augment = augment
        self.augment_ratio = augment_ratio
        self.augment_swap_prob = augment_swap_prob
        self.random_state = random_state

    def fit(self, X, y, **fit_params) -> KnowledgeDistiller:
        """Train the student model using knowledge distillation.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            True labels (hard targets).

        Returns
        -------
        self
        """
        X = np.asarray(X)
        y = np.asarray(y)

        self.is_classifier_ = is_classifier(self.teacher)

        if self.is_classifier_:
            self._fit_classification(X, y, **fit_params)
        else:
            self._fit_regression(X, y, **fit_params)

        return self

    def _fit_classification(self, X, y, **fit_params):
        """Distillation for classification tasks."""
        # Get teacher's soft predictions
        teacher_proba = self.teacher.predict_proba(X)
        n_classes = teacher_proba.shape[1]

        # Generate soft labels with temperature
        if self.temperature != 1.0:
            # Convert probabilities to logits, then apply temperature
            eps = 1e-10
            logits = np.log(teacher_proba + eps)
            soft_labels = _softmax_with_temperature(logits, self.temperature)
        else:
            soft_labels = teacher_proba

        # Data augmentation with MUNGE
        if self.augment:
            n_aug = int(len(X) * self.augment_ratio)
            X_aug, _ = _munge_augment(
                X, y, n_aug,
                swap_prob=self.augment_swap_prob,
                random_state=self.random_state,
            )
            # Label augmented data with teacher
            teacher_proba_aug = self.teacher.predict_proba(X_aug)
            if self.temperature != 1.0:
                logits_aug = np.log(teacher_proba_aug + 1e-10)
                soft_labels_aug = _softmax_with_temperature(logits_aug, self.temperature)
            else:
                soft_labels_aug = teacher_proba_aug

            X_combined = np.vstack([X, X_aug])
            soft_combined = np.vstack([soft_labels, soft_labels_aug])
            # For hard labels, use teacher predictions on augmented data
            y_aug = np.argmax(teacher_proba_aug, axis=1)
            y_combined = np.concatenate([y, y_aug])
        else:
            X_combined = X
            soft_combined = soft_labels
            y_combined = y

        # Train student
        # Strategy: if alpha < 1, use blend of hard and soft labels
        # For most sklearn classifiers, we train on the teacher's hard predictions
        # (soft labels require specialized loss functions)
        if self.alpha >= 0.99:
            # Pure distillation: train on teacher's predictions
            y_teacher = np.argmax(soft_combined, axis=1)
            self.student_ = clone(self.student)

            # If student supports sample_weight, use confidence as weight
            if hasattr(self.student_, 'fit') and 'sample_weight' in \
               self.student_.fit.__code__.co_varnames:
                weights = np.max(soft_combined, axis=1)
                self.student_.fit(X_combined, y_teacher,
                                sample_weight=weights, **fit_params)
            else:
                self.student_.fit(X_combined, y_teacher, **fit_params)
        else:
            # Blend: mix hard labels with teacher predictions
            # Use teacher predictions weighted by alpha
            y_teacher = np.argmax(soft_combined, axis=1)
            # With probability (1-alpha), use true hard labels
            rng = np.random.RandomState(self.random_state)
            use_hard = rng.random(len(y_combined)) > self.alpha
            y_train = y_teacher.copy()
            y_train[use_hard] = y_combined[use_hard]

            self.student_ = clone(self.student)
            self.student_.fit(X_combined, y_train, **fit_params)

        # Record scores
        self.teacher_score_ = float(np.mean(
            self.teacher.predict(X) == y
        ))
        self.student_score_ = float(np.mean(
            self.student_.predict(X) == y
        ))

    def _fit_regression(self, X, y, **fit_params):
        """Distillation for regression tasks."""
        # Get teacher predictions
        teacher_preds = self.teacher.predict(X)

        if self.augment:
            n_aug = int(len(X) * self.augment_ratio)
            X_aug, _ = _munge_augment(
                X, y, n_aug,
                swap_prob=self.augment_swap_prob,
                random_state=self.random_state,
            )
            teacher_preds_aug = self.teacher.predict(X_aug)
            X_combined = np.vstack([X, X_aug])
            y_combined = np.concatenate([y, y])
            teacher_combined = np.concatenate([teacher_preds, teacher_preds_aug])
        else:
            X_combined = X
            y_combined = y
            teacher_combined = teacher_preds

        # Blend teacher predictions with true labels
        y_train = self.alpha * teacher_combined + (1 - self.alpha) * y_combined

        self.student_ = clone(self.student)
        self.student_.fit(X_combined, y_train, **fit_params)

        # Record scores
        from sklearn.metrics import r2_score
        self.teacher_score_ = r2_score(y, teacher_preds)
        self.student_score_ = r2_score(y, self.student_.predict(X))

    def predict(self, X) -> np.ndarray:
        """Predict using the trained student model."""
        check_is_fitted(self, 'student_')
        return self.student_.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Predict probabilities using the trained student model."""
        check_is_fitted(self, 'student_')
        if not self.is_classifier_:
            raise AttributeError("predict_proba is not available for regression.")
        return self.student_.predict_proba(X)

    @property
    def feature_importances_(self):
        """Feature importances from the student model."""
        check_is_fitted(self, 'student_')
        if hasattr(self.student_, 'feature_importances_'):
            return self.student_.feature_importances_
        raise AttributeError("Student model does not have feature_importances_.")

    def compression_report(self) -> dict:
        """Generate a report comparing teacher and student performance.

        Returns
        -------
        dict with keys:
            teacher_score, student_score, score_retention,
            teacher_type, student_type
        """
        check_is_fitted(self, 'student_')
        retention = (self.student_score_ / self.teacher_score_ * 100
                     if self.teacher_score_ > 0 else 0)
        return {
            "teacher_type": type(self.teacher).__name__,
            "student_type": type(self.student_).__name__,
            "teacher_score": self.teacher_score_,
            "student_score": self.student_score_,
            "score_retention_pct": round(retention, 1),
            "metric": "accuracy" if self.is_classifier_ else "r2",
        }
