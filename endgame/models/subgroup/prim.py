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


class PRIMRegressor(RegressorMixin, BaseEstimator):
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

    def fit(self, X, y, feature_names: list[str] | None = None) -> "PRIMRegressor":
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


class PRIMClassifier(ClassifierMixin, BaseEstimator):
    """PRIM for classification (finds regions with high class probability).

    Uses PRIM on the binary indicator of the target class to find
    regions where that class is most prevalent.

    Parameters
    ----------
    target_class : int or str, default=1
        The class to find high-density regions for.
        If 'minority', automatically selects the minority class.
    alpha : float, default=0.05
        Peeling fraction.
    min_support : int or float, default=20
        Minimum number of points in a box.
    pasting : bool, default=True
        Whether to apply pasting after peeling.
    paste_alpha : float, default=0.01
        Pasting fraction.
    n_boxes : int, default=1
        Number of boxes to find.

    Examples
    --------
    >>> from endgame.models.subgroup import PRIMClassifier
    >>> prim = PRIMClassifier(target_class='minority')
    >>> prim.fit(X, y)
    >>> print(prim.boxes_[0].to_rules())
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        target_class: int | str = 1,
        alpha: float = 0.05,
        min_support: int | float = 20,
        pasting: bool = True,
        paste_alpha: float = 0.01,
        n_boxes: int = 1,
    ):
        self.target_class = target_class
        self.alpha = alpha
        self.min_support = min_support
        self.pasting = pasting
        self.paste_alpha = paste_alpha
        self.n_boxes = n_boxes

        self.classes_: np.ndarray | None = None
        self._target_class_idx: int = 1
        self._prim_regressor: PRIMRegressor | None = None
        self._label_encoder: LabelEncoder | None = None
        self._is_fitted: bool = False

    def fit(self, X, y, feature_names: list[str] | None = None) -> "PRIMClassifier":
        """Fit PRIM to find regions with high target class probability.

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
        X = np.asarray(X)
        y = np.asarray(y)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        # Determine target class
        if self.target_class == "minority":
            counts = np.bincount(y_encoded)
            self._target_class_idx = np.argmin(counts)
        elif isinstance(self.target_class, int):
            self._target_class_idx = self.target_class
        else:
            self._target_class_idx = np.where(
                self.classes_ == self.target_class
            )[0][0]

        # Create binary target (1 if target class, 0 otherwise)
        y_binary = (y_encoded == self._target_class_idx).astype(np.float64)
        self._base_rate = y_binary.mean()

        # Fit PRIM regressor on binary target
        self._prim_regressor = PRIMRegressor(
            alpha=self.alpha,
            min_support=self.min_support,
            pasting=self.pasting,
            paste_alpha=self.paste_alpha,
            n_boxes=self.n_boxes,
        )
        self._prim_regressor.fit(X, y_binary, feature_names=feature_names)

        self._is_fitted = True
        return self

    @property
    def boxes_(self) -> list[Box]:
        """Get the discovered boxes."""
        if self._prim_regressor is None:
            return []
        return self._prim_regressor.boxes_

    @property
    def result_(self) -> PRIMResult | None:
        """Get full PRIM result."""
        if self._prim_regressor is None:
            return None
        return self._prim_regressor.result_

    @property
    def feature_names_in_(self) -> np.ndarray | None:
        """Get feature names."""
        if self._prim_regressor is None:
            return None
        return self._prim_regressor.feature_names_in_

    @property
    def n_features_in_(self) -> int:
        """Get number of features."""
        if self._prim_regressor is None:
            return 0
        return self._prim_regressor.n_features_in_

    def predict(self, X) -> np.ndarray:
        """Predict whether points fall in the found box(es).

        Parameters
        ----------
        X : array-like
            Data points.

        Returns
        -------
        mask : ndarray
            Boolean mask, True if point is in any box.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")
        return self._prim_regressor.predict(X)

    def predict_proba(self, X) -> np.ndarray:
        """Estimate class probability based on box membership.

        Points in box get the box's density as probability for target class.
        Points outside get the overall class frequency.

        Parameters
        ----------
        X : array-like
            Data points.

        Returns
        -------
        proba : ndarray of shape (n_samples, n_classes)
            Class probabilities.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")

        X = np.asarray(X)
        n_samples = len(X)
        n_classes = len(self.classes_)
        tc = self._target_class_idx
        oc = 1 - tc

        proba = np.empty((n_samples, n_classes))
        proba[:, tc] = self._base_rate
        proba[:, oc] = 1.0 - self._base_rate

        for box in self.boxes_:
            mask = box.contains(X)
            proba[mask, tc] = box.density
            proba[mask, oc] = 1.0 - box.density

        return proba

    def score(self, X, y) -> float:
        """Score the model: precision of target class in boxes.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Target labels.

        Returns
        -------
        score : float
            Precision of target class in predicted boxes.
        """
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")

        X = np.asarray(X)
        y = np.asarray(y)
        y_encoded = self._label_encoder.transform(y)

        mask = self.predict(X)
        if mask.sum() == 0:
            return 0.0

        # Precision: proportion of target class in boxes
        return (y_encoded[mask] == self._target_class_idx).mean()

    def get_rules(self) -> list[list[str]]:
        """Get human-readable rules for all boxes."""
        if not self._is_fitted:
            raise RuntimeError("PRIMClassifier has not been fitted.")
        return self._prim_regressor.get_rules()
