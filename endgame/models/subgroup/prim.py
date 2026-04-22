from __future__ import annotations

"""PRIM: Patient Rule Induction Method for bump hunting.

PRIM is a subgroup discovery algorithm that finds rectangular regions
(boxes) in the feature space where the target variable has unusually
high (or low) mean values. It operates through iterative "peeling" and
optional "pasting" steps.

The algorithm is particularly useful for:
- Finding high-performing customer segments
- Identifying failure modes in manufacturing
- Scenario discovery in policy analysis
- Anomaly detection contexts

References
----------
- Friedman & Fisher, "Bump Hunting in High-Dimensional Data" (1999)
- RAND Corporation sdtoolkit
- Project-Platypus/PRIM Python implementation
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelEncoder

from endgame.core.glassbox import GlassboxMixin
from typing import Any


@dataclass
class Box:
    """A rectangular region (box) in feature space.

    Attributes
    ----------
    limits : Dict[int, Tuple[float, float]]
        Feature index -> (lower, upper) bound.
    coverage : float
        Fraction of data points inside the box.
    density : float
        Mean target value inside the box.
    support : int
        Number of data points inside the box.
    """

    limits: dict[int, tuple[float, float]] = field(default_factory=dict)
    coverage: float = 1.0
    density: float = 0.0
    support: int = 0

    def contains(self, X: np.ndarray) -> np.ndarray:
        """Check which points are inside the box.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data points to check.

        Returns
        -------
        mask : ndarray of shape (n_samples,)
            Boolean mask, True if point is inside box.
        """
        if not self.limits:
            return np.ones(len(X), dtype=bool)

        feat_indices = np.fromiter(self.limits.keys(), dtype=np.intp)
        bounds = np.array([self.limits[i] for i in feat_indices])
        X_sub = X[:, feat_indices]
        return np.all((X_sub >= bounds[:, 0]) & (X_sub <= bounds[:, 1]), axis=1)

    def to_rules(self, feature_names: list[str] | None = None) -> list[str]:
        """Convert box to human-readable rules.

        Parameters
        ----------
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        rules : list of str
            List of rule strings.
        """
        rules = []
        for feat_idx, (lower, upper) in sorted(self.limits.items()):
            name = feature_names[feat_idx] if feature_names else f"x{feat_idx}"
            rules.append(f"{lower:.4g} <= {name} <= {upper:.4g}")
        return rules

    def __repr__(self) -> str:
        return (
            f"Box(coverage={self.coverage:.3f}, density={self.density:.4f}, "
            f"support={self.support}, n_restrictions={len(self.limits)})"
        )


@dataclass
class PRIMResult:
    """Result of PRIM analysis.

    Attributes
    ----------
    boxes : List[Box]
        Sequence of boxes from peeling trajectory.
    peeling_trajectory : List[Dict]
        Statistics at each peeling step.
    selected_box : Box
        The selected box (based on some criterion).
    selected_idx : int
        Index of selected box in trajectory.
    """

    boxes: list[Box] = field(default_factory=list)
    peeling_trajectory: list[dict[str, float]] = field(default_factory=list)
    selected_box: Box | None = None
    selected_idx: int = -1

    def get_pareto_frontier(self) -> list[int]:
        """Get indices of boxes on the coverage-density Pareto frontier."""
        if not self.peeling_trajectory:
            return []

        coverages = np.array([t["coverage"] for t in self.peeling_trajectory])
        densities = np.array([t["density"] for t in self.peeling_trajectory])

        # Find Pareto-optimal points (max density for each coverage level)
        pareto_indices = []
        max_density = -np.inf
        for i in range(len(coverages) - 1, -1, -1):
            if densities[i] > max_density:
                pareto_indices.append(i)
                max_density = densities[i]

        return sorted(pareto_indices)


class PRIMRegressor(GlassboxMixin, RegressorMixin, BaseEstimator):
    """PRIM (Patient Rule Induction Method) for regression/continuous targets.

    Finds rectangular regions where the target variable has unusually
    high mean values. Uses iterative peeling to shrink boxes while
    increasing target density.

    Parameters
    ----------
    alpha : float, default=0.05
        Peeling fraction - proportion of data removed in each peel.
        Smaller values = more "patient" peeling.
    threshold_type : str, default='quantile'
        How to define "interesting" regions: 'quantile' or 'absolute'.
    threshold : float, default=0.9
        Threshold for defining interesting regions.
        If 'quantile', fraction of top values to consider interesting.
    min_support : int or float, default=20
        Minimum number of points in a box. If float, interpreted as fraction.
    pasting : bool, default=True
        Whether to apply pasting (box expansion) after peeling.
    paste_alpha : float, default=0.01
        Pasting fraction for box expansion.
    n_boxes : int, default=1
        Number of boxes to find (sequential covering).

    Attributes
    ----------
    result_ : PRIMResult
        Full PRIM analysis result.
    boxes_ : List[Box]
        The final boxes found.
    feature_names_in_ : ndarray
        Names of features.
    n_features_in_ : int
        Number of features.

    Examples
    --------
    >>> from endgame.models.subgroup import PRIMRegressor
    >>> prim = PRIMRegressor(alpha=0.05, min_support=30)
    >>> prim.fit(X, y)
    >>> print(prim.boxes_[0].to_rules())
    >>> mask = prim.predict(X)  # Boolean mask of points in box

    Notes
    -----
    PRIM works best when:
    1. You're looking for interpretable subgroups
    2. The target has heterogeneous behavior across the feature space
    3. You want rectangular (axis-aligned) regions
    """

    _estimator_type = "regressor"

    def __init__(
        self,
        alpha: float = 0.05,
        threshold_type: Literal["quantile", "absolute"] = "quantile",
        threshold: float = 0.9,
        min_support: int | float = 20,
        pasting: bool = True,
        paste_alpha: float = 0.01,
        n_boxes: int = 1,
    ):
        self.alpha = alpha
        self.threshold_type = threshold_type
        self.threshold = threshold
        self.min_support = min_support
        self.pasting = pasting
        self.paste_alpha = paste_alpha
        self.n_boxes = n_boxes

        self.result_: PRIMResult | None = None
        self.boxes_: list[Box] = []
        self.feature_names_in_: np.ndarray | None = None
        self.n_features_in_: int = 0
        self._is_fitted: bool = False

    def fit(self, X, y, feature_names: list[str] | None = None) -> PRIMRegressor:
        """Fit PRIM to find high-density regions.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values (higher = more interesting).
        feature_names : list of str, optional
            Names of features for interpretable output.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64) if not isinstance(X, np.ndarray) else X.astype(np.float64, copy=False)
        y = np.asarray(y, dtype=np.float64) if not isinstance(y, np.ndarray) else y.astype(np.float64, copy=False)

        n_samples, n_features = X.shape
        self.n_features_in_ = n_features

        if feature_names is not None:
            self.feature_names_in_ = np.array(feature_names)
        else:
            self.feature_names_in_ = np.array([f"x{i}" for i in range(n_features)])

        # Calculate minimum support
        if isinstance(self.min_support, float) and self.min_support < 1:
            min_support = int(self.min_support * n_samples)
        else:
            min_support = int(self.min_support)
        min_support = max(min_support, 2)

        # Store global mean for predict() fallback
        self._y_global_mean = float(y.mean())

        # Find boxes sequentially
        self.boxes_ = []
        remaining_mask = np.ones(n_samples, dtype=bool)

        for _ in range(self.n_boxes):
            X_remaining = X[remaining_mask]
            y_remaining = y[remaining_mask]

            if len(y_remaining) < min_support:
                break

            # Run PRIM on remaining data
            result = self._prim_one_box(
                X_remaining, y_remaining, min_support
            )

            if result.selected_box is not None:
                self.boxes_.append(result.selected_box)
                # Remove covered points for sequential covering
                box_mask = result.selected_box.contains(X)
                remaining_mask[box_mask] = False

        self.result_ = result if self.n_boxes == 1 else None
        self._is_fitted = True

        return self

    def _prim_one_box(
        self, X: np.ndarray, y: np.ndarray, min_support: int
    ) -> PRIMResult:
        """Find one box using PRIM algorithm."""
        n_samples, n_features = X.shape

        current_limits = {
            i: (X[:, i].min(), X[:, i].max()) for i in range(n_features)
        }
        current_mask = np.ones(n_samples, dtype=bool)

        boxes = []
        trajectory = []

        support = n_samples
        y_sum = y.sum()
        density = y_sum / support if support > 0 else 0.0
        coverage = 1.0

        boxes.append(
            Box(
                limits=current_limits.copy(),
                coverage=coverage,
                density=density,
                support=support,
            )
        )
        trajectory.append({"coverage": coverage, "density": density, "support": support})

        X_cols = [X[:, i] for i in range(n_features)]
        buf = np.empty(n_samples, dtype=bool)

        # Peeling phase
        while support > min_support:
            best_feat = -1
            best_side = ""
            best_threshold = 0.0
            best_density = density

            for feat_idx in range(n_features):
                col = X_cols[feat_idx]
                x_active = col[current_mask]
                n_active = len(x_active)
                if n_active <= min_support:
                    continue

                k_lo = max(1, int(self.alpha * n_active))
                k_hi = min(n_active - 1, int((1 - self.alpha) * n_active))

                partitioned = np.partition(x_active, (k_lo, k_hi))
                thresh_lo = partitioned[k_lo]
                thresh_hi = partitioned[k_hi]

                # Peel from bottom (in-place mask to avoid temporaries)
                np.greater(col, thresh_lo, out=buf)
                np.logical_and(current_mask, buf, out=buf)
                n_lo = buf.sum()
                if n_lo >= min_support:
                    d_lo = np.dot(y, buf) / n_lo
                    if d_lo > best_density:
                        best_density = d_lo
                        best_feat = feat_idx
                        best_side = "lower"
                        best_threshold = thresh_lo

                # Peel from top (reuse buf)
                np.less(col, thresh_hi, out=buf)
                np.logical_and(current_mask, buf, out=buf)
                n_hi = buf.sum()
                if n_hi >= min_support:
                    d_hi = np.dot(y, buf) / n_hi
                    if d_hi > best_density:
                        best_density = d_hi
                        best_feat = feat_idx
                        best_side = "upper"
                        best_threshold = thresh_hi

            if best_feat < 0:
                break

            col = X_cols[best_feat]
            if best_side == "lower":
                np.greater(col, best_threshold, out=buf)
                current_limits[best_feat] = (
                    best_threshold, current_limits[best_feat][1],
                )
            else:
                np.less(col, best_threshold, out=buf)
                current_limits[best_feat] = (
                    current_limits[best_feat][0], best_threshold,
                )
            np.logical_and(current_mask, buf, out=current_mask)

            support = current_mask.sum()
            density = np.dot(y, current_mask) / support
            coverage = support / n_samples

            boxes.append(
                Box(
                    limits=current_limits.copy(),
                    coverage=coverage,
                    density=density,
                    support=support,
                )
            )
            trajectory.append(
                {"coverage": coverage, "density": density, "support": support}
            )

        # Pasting phase (expand box boundaries if it improves density)
        if self.pasting and len(boxes) > 1:
            boxes, trajectory = self._paste(X, y, boxes, trajectory, min_support)

        # Select best box (maximum density with reasonable support)
        selected_idx = self._select_box(trajectory)

        return PRIMResult(
            boxes=boxes,
            peeling_trajectory=trajectory,
            selected_box=boxes[selected_idx] if boxes else None,
            selected_idx=selected_idx,
        )

    def _paste(
        self,
        X: np.ndarray,
        y: np.ndarray,
        boxes: list[Box],
        trajectory: list[dict],
        min_support: int,
    ) -> tuple[list[Box], list[dict]]:
        """Apply pasting to expand box boundaries."""
        if not boxes:
            return boxes, trajectory

        n_samples, n_features = X.shape
        X_cols = [X[:, i] for i in range(n_features)]

        best_idx = self._select_box(trajectory)
        current_box = boxes[best_idx]
        current_limits = current_box.limits.copy()
        current_mask = current_box.contains(X)
        current_n = current_mask.sum()
        current_ysum = np.dot(y, current_mask)

        improved = True
        while improved:
            improved = False
            current_density = current_ysum / current_n if current_n > 0 else 0.0

            for feat_idx in range(n_features):
                if feat_idx not in current_limits:
                    continue

                col = X_cols[feat_idx]
                lower, upper = current_limits[feat_idx]

                # Try expanding lower bound
                outside_lower = col < lower
                n_outside = outside_lower.sum()
                if n_outside > 0:
                    x_outside = col[outside_lower]
                    k = min(n_outside - 1, int((1 - self.paste_alpha) * n_outside))
                    expand_threshold = np.partition(x_outside, k)[k]
                    added = (col >= expand_threshold) & (col < lower) & ~current_mask
                    n_added = added.sum()
                    if n_added > 0:
                        new_n = current_n + n_added
                        new_ysum = current_ysum + np.dot(y, added)
                        new_density = new_ysum / new_n
                        if new_density > current_density:
                            current_limits[feat_idx] = (expand_threshold, upper)
                            current_mask = current_mask | added
                            current_n = new_n
                            current_ysum = new_ysum
                            improved = True

                lower, upper = current_limits[feat_idx]

                # Try expanding upper bound
                outside_upper = col > upper
                n_outside = outside_upper.sum()
                if n_outside > 0:
                    x_outside = col[outside_upper]
                    k = max(0, int(self.paste_alpha * n_outside))
                    expand_threshold = np.partition(x_outside, k)[k]
                    added = (col <= expand_threshold) & (col > upper) & ~current_mask
                    n_added = added.sum()
                    if n_added > 0:
                        new_n = current_n + n_added
                        new_ysum = current_ysum + np.dot(y, added)
                        new_density = new_ysum / new_n
                        if new_density > current_density:
                            current_limits[feat_idx] = (lower, expand_threshold)
                            current_mask = current_mask | added
                            current_n = new_n
                            current_ysum = new_ysum
                            improved = True

        if current_n != current_box.support:
            final_density = current_ysum / current_n if current_n > 0 else 0.0
            pasted_box = Box(
                limits=current_limits.copy(),
                coverage=current_n / n_samples,
                density=final_density,
                support=current_n,
            )
            boxes.append(pasted_box)
            trajectory.append(
                {
                    "coverage": pasted_box.coverage,
                    "density": pasted_box.density,
                    "support": pasted_box.support,
                }
            )

        return boxes, trajectory

    def _select_box(self, trajectory: list[dict]) -> int:
        """Select the best box from the peeling trajectory.

        Maximizes density subject to minimum coverage (0.01).
        """
        if not trajectory:
            return 0

        best_idx = 0
        best_density = -np.inf
        min_coverage = 0.01

        for i, t in enumerate(trajectory):
            if t["coverage"] >= min_coverage and t["density"] > best_density:
                best_density = t["density"]
                best_idx = i

        if best_density == -np.inf:
            return len(trajectory) - 1

        return best_idx

    def predict(self, X) -> np.ndarray:
        """Predict target values based on box membership.

        Points inside a box get that box's mean target density.
        Points outside all boxes get the global training mean.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted target values.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMRegressor has not been fitted.")

        X = np.asarray(X)
        predictions = np.full(len(X), self._y_global_mean)

        # Assign box density to points inside boxes (last box wins)
        for box in self.boxes_:
            mask = box.contains(X)
            predictions[mask] = box.density

        return predictions

    def predict_membership(self, X) -> np.ndarray:
        """Predict whether points fall in the found box(es).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        mask : ndarray of shape (n_samples,)
            Boolean mask, True if point is in any box.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMRegressor has not been fitted.")

        X = np.asarray(X)
        mask = np.zeros(len(X), dtype=bool)

        for box in self.boxes_:
            mask |= box.contains(X)

        return mask

    def score(self, X, y) -> float:
        """Score the model: mean target value in predicted boxes.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Target values.

        Returns
        -------
        score : float
            Mean target value in boxes minus overall mean.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMRegressor has not been fitted.")

        X = np.asarray(X)
        y = np.asarray(y)

        mask = self.predict(X)
        if mask.sum() == 0:
            return 0.0

        return y[mask].mean() - y.mean()

    def get_rules(self) -> list[list[str]]:
        """Get human-readable rules for all boxes.

        Returns
        -------
        rules : list of list of str
            Rules for each box.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMRegressor has not been fitted.")

        return [
            box.to_rules(list(self.feature_names_in_)) for box in self.boxes_
        ]


class PRIMClassifier(GlassboxMixin, ClassifierMixin, BaseEstimator):
    """PRIM for classification via one-vs-rest subgroup discovery.

    Trains a PRIM regressor per class (one-vs-rest) on the binary
    indicator for each class. At prediction time, the class whose
    box gives the highest density for a sample wins; samples not in
    any box are assigned the majority class.

    Parameters
    ----------
    alpha : float, default=0.05
        Peeling fraction.
    min_support : int or float, default=20
        Minimum number of points in a box.
    pasting : bool, default=True
        Whether to apply pasting after peeling.
    paste_alpha : float, default=0.01
        Pasting fraction.
    n_boxes : int, default=1
        Number of boxes to find per class.

    Examples
    --------
    >>> from endgame.models.subgroup import PRIMClassifier
    >>> prim = PRIMClassifier(alpha=0.05)
    >>> prim.fit(X, y)
    >>> preds = prim.predict(X)
    >>> print(prim.get_rules())
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        alpha: float = 0.05,
        min_support: int | float = 20,
        pasting: bool = True,
        paste_alpha: float = 0.01,
        n_boxes: int = 1,
    ):
        self.alpha = alpha
        self.min_support = min_support
        self.pasting = pasting
        self.paste_alpha = paste_alpha
        self.n_boxes = n_boxes

        self.classes_: np.ndarray | None = None
        self._prim_regressors: list[PRIMRegressor] = []
        self._base_rates: np.ndarray | None = None
        self._label_encoder: LabelEncoder | None = None
        self._majority_class_idx: int = 0
        self._is_fitted: bool = False

    def fit(self, X, y, feature_names: list[str] | None = None) -> PRIMClassifier:
        """Fit one PRIM model per class (one-vs-rest).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target labels.
        feature_names : list of str, optional
            Names of features.

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        n_classes = len(self.classes_)

        counts = np.bincount(y_encoded, minlength=n_classes)
        self._majority_class_idx = int(np.argmax(counts))
        self._base_rates = counts / len(y_encoded)

        self._prim_regressors = []
        for c in range(n_classes):
            y_binary = (y_encoded == c).astype(np.float64)
            reg = PRIMRegressor(
                alpha=self.alpha,
                min_support=self.min_support,
                pasting=self.pasting,
                paste_alpha=self.paste_alpha,
                n_boxes=self.n_boxes,
            )
            reg.fit(X, y_binary, feature_names=feature_names)
            self._prim_regressors.append(reg)

        self._is_fitted = True
        return self

    @property
    def boxes_(self) -> list[list[Box]]:
        """Get the discovered boxes for each class."""
        return [r.boxes_ for r in self._prim_regressors]

    @property
    def feature_names_in_(self) -> np.ndarray | None:
        """Get feature names."""
        if not self._prim_regressors:
            return None
        return self._prim_regressors[0].feature_names_in_

    @property
    def n_features_in_(self) -> int:
        """Get number of features."""
        if not self._prim_regressors:
            return 0
        return self._prim_regressors[0].n_features_in_

    def predict_proba(self, X) -> np.ndarray:
        """Estimate class probabilities based on box densities.

        For each sample, the probability for class *c* is the density
        of the best box that contains it, or the base rate if no box
        contains it. Probabilities are row-normalised.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")

        X = np.asarray(X, dtype=np.float64)
        n_samples = len(X)
        n_classes = len(self.classes_)

        proba = np.tile(self._base_rates, (n_samples, 1)).astype(np.float64)

        for c, reg in enumerate(self._prim_regressors):
            for box in reg.boxes_:
                mask = box.contains(X)
                if mask.any():
                    proba[mask, c] = max(box.density, proba[mask, c].max())

        row_sums = proba.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        proba /= row_sums
        return proba

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data points.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            Predicted class labels.
        """
        proba = self.predict_proba(X)
        indices = np.argmax(proba, axis=1)
        return self._label_encoder.inverse_transform(indices)

    def score(self, X, y) -> float:
        """Classification accuracy.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Target labels.

        Returns
        -------
        score : float
            Accuracy.
        """
        return float(np.mean(self.predict(X) == np.asarray(y)))

    def get_rules(self) -> list[list[list[str]]]:
        """Get human-readable rules for all classes and boxes.

        Returns
        -------
        rules : list[list[list[str]]]
            rules[class_idx][box_idx] is a list of rule strings.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")
        return [reg.get_rules() for reg in self._prim_regressors]


def _box_to_dict(box: Box, feature_names: list[str]) -> dict[str, Any]:
    limits = {
        (feature_names[i] if i < len(feature_names) else f"x{i}"): [
            float(lo) if np.isfinite(lo) else None,
            float(hi) if np.isfinite(hi) else None,
        ]
        for i, (lo, hi) in box.limits.items()
    }
    return {
        "limits": limits,
        "coverage": float(box.coverage),
        "density": float(box.density),
        "support": int(box.support),
        "rules": box.to_rules(feature_names),
    }


def _prim_regressor_structure(self) -> dict[str, Any]:
    if not self._is_fitted:
        raise RuntimeError("PRIMRegressor has not been fitted.")
    feature_names = list(self.feature_names_in_)
    return {
        "boxes": [_box_to_dict(b, feature_names) for b in self.boxes_],
        "n_boxes": len(self.boxes_),
        "objective": getattr(self, "objective", None),
    }


def _prim_classifier_structure(self) -> dict[str, Any]:
    if not self._is_fitted:
        raise RuntimeError("PRIMClassifier has not been fitted.")
    feature_names = list(self.feature_names_in_)
    per_class = []
    for class_idx, reg in enumerate(self._prim_regressors):
        per_class.append({
            "class_index": class_idx,
            "class": self.classes_[class_idx].item() if hasattr(self.classes_[class_idx], "item") else self.classes_[class_idx],
            "boxes": [_box_to_dict(b, feature_names) for b in reg.boxes_],
        })
    return {
        "per_class": per_class,
        "n_classes": len(self._prim_regressors),
    }


PRIMRegressor._structure_type = "boxes"
PRIMRegressor._structure_content = _prim_regressor_structure
PRIMClassifier._structure_type = "boxes"
PRIMClassifier._structure_content = _prim_classifier_structure
