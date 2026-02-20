"""GritBot: Rule-based anomaly detection via recursive partitioning.

GritBot finds anomalies by partitioning data into homogeneous subsets
and identifying values that are surprising given the subset context.
This is a Python reimplementation of Quinlan's GritBot algorithm with
numba JIT compilation for performance.

Reference:
    Quinlan, J.R. (2010). GritBot GPL Edition. Rulequest Research.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_array, check_is_fitted

try:
    from numba import jit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range


# GritBot constants (from original implementation)
MAXFRAC = 0.01      # max proportion of outliers in a group
MAXNORM = 2.67      # max SDs for non-outlier
MAXTAIL = 5.34      # max SDs for 'ordinary' tail
MINCONTEXT = 25     # min cases to detect other difference


@dataclass
class AnomalyContext:
    """Context conditions that define when an anomaly occurs."""
    conditions: list[tuple[int, str, Any, Any]] = field(default_factory=list)
    # Each condition: (feature_idx, op, value1, value2)
    # op: 'le', 'gt', 'in_range', 'eq', 'in_set'


@dataclass
class Anomaly:
    """Detected anomaly with context."""
    case_idx: int
    feature_idx: int
    value: Any
    score: float  # Lower = more anomalous (probability-like)
    context: AnomalyContext
    group_size: int
    group_mean: float = 0.0
    group_std: float = 0.0
    expected_value: Any = None


@jit(nopython=True, cache=True)
def _trimmed_mean_std(values: np.ndarray, tail_frac: float = 0.01) -> tuple[float, float]:
    """Compute trimmed mean and SD, excluding extreme tails.

    Uses GritBot's robust estimation approach: exclude tail cases
    and adjust SD by factor (N + 3*Tail) / (N + Tail).
    """
    n = len(values)
    if n < 10:
        return np.mean(values), np.std(values) + 1e-10

    # Sort for trimming
    sorted_vals = np.sort(values)

    # Compute tail size (same as GritBot's MaxAnoms formula)
    tail = int(tail_frac * n + 2 * np.sqrt(n * tail_frac * (1 - tail_frac)) + 1)
    tail = min(tail, n // 4)  # Don't trim more than 25%

    if tail >= n // 2:
        return np.mean(values), np.std(values) + 1e-10

    # Trimmed statistics
    trimmed = sorted_vals[tail:n-tail]
    cases = len(trimmed)

    if cases < 5:
        return np.mean(values), np.std(values) + 1e-10

    mean = np.mean(trimmed)
    std = np.std(trimmed)

    # Adjust SD by factor (N + 3*Tail) / (N + Tail) per GritBot
    if cases > tail:
        std = std * (cases + 3.0 * tail) / (cases + tail)

    return mean, max(std, 1e-10)


@jit(nopython=True, cache=True)
def _find_continuous_outliers(
    values: np.ndarray,
    indices: np.ndarray,
    max_norm: float,
    min_abnorm: float,
    max_frac: float,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Find continuous outliers in a subset.

    Returns:
        outlier_indices: indices of outliers
        outlier_scores: anomaly scores (lower = more anomalous)
        mean: group mean
        std: group std
    """
    n = len(values)
    if n < 35:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0.0, 1.0

    # Robust mean/std estimation
    mean, std = _trimmed_mean_std(values, max_frac)

    if std < 1e-10:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), mean, std

    # Compute Z-scores
    z_scores = np.abs(values - mean) / std

    # Maximum allowed anomalies
    max_anoms = int(max_frac * n + 2 * np.sqrt(n * max_frac * (1 - max_frac)) + 1)

    # Sort by Z-score descending
    sorted_idx = np.argsort(-z_scores)

    # Find outliers: must have Z > min_abnorm and be separated from normal cases
    outlier_mask = np.zeros(n, dtype=np.bool_)
    n_outliers = 0

    for i in range(min(max_anoms, n)):
        idx = sorted_idx[i]
        z = z_scores[idx]

        if z < min_abnorm:
            break

        # Check gap from next case (simplified gap check)
        if i + 1 < n:
            next_z = z_scores[sorted_idx[i + 1]]
            if z - next_z < min_abnorm - max_norm:
                break

        outlier_mask[idx] = True
        n_outliers += 1

    if n_outliers == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), mean, std

    # Collect outliers
    outlier_indices = np.empty(n_outliers, dtype=np.int64)
    outlier_scores = np.empty(n_outliers, dtype=np.float64)

    j = 0
    for i in range(n):
        if outlier_mask[i]:
            outlier_indices[j] = indices[i]
            # Score using Chebyshev bound: P(|X-mu| >= k*sigma) <= 1/k^2
            z = z_scores[i]
            outlier_scores[j] = 1.0 / (z * z)
            j += 1

    return outlier_indices, outlier_scores, mean, std


@jit(nopython=True, cache=True)
def _find_discrete_outliers(
    values: np.ndarray,
    indices: np.ndarray,
    priors: np.ndarray,
    min_abnorm: float,
    max_frac: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Find discrete outliers in a subset.

    Anomalies are minority class values that are surprising given the
    class distribution in the subset.

    Returns:
        outlier_indices: indices of outliers
        outlier_scores: anomaly scores
        majority_class: most common value
    """
    n = len(values)
    if n < 35:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), 0

    # Count frequencies
    n_classes = len(priors)
    freqs = np.zeros(n_classes, dtype=np.int64)
    for v in values:
        if 0 <= v < n_classes:
            freqs[v] += 1

    # Find majority class
    majority = 0
    for i in range(n_classes):
        if freqs[i] > freqs[majority]:
            majority = i

    n_anoms = n - freqs[majority]
    max_anoms = int(max_frac * n + 2 * np.sqrt(n * max_frac * (1 - max_frac)) + 1)

    if n_anoms > max_anoms or n_anoms == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), majority

    # Threshold for surprise: impurity / prior < 1 / (MINABNORM^2)
    threshold = 1.0 / (min_abnorm * min_abnorm)

    # Find surprising minority cases
    outlier_list = []
    score_list = []

    for i in range(n):
        v = values[i]
        if v == majority:
            continue

        # Compute surprise score: (anoms / n) / prior
        prior = priors[v] if 0 <= v < n_classes else 1.0 / n_classes
        score = (n_anoms / n) / max(prior, 1e-10)

        if score <= threshold:
            outlier_list.append(indices[i])
            score_list.append(score)

    n_out = len(outlier_list)
    if n_out == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64), majority

    outlier_indices = np.empty(n_out, dtype=np.int64)
    outlier_scores = np.empty(n_out, dtype=np.float64)
    for i in range(n_out):
        outlier_indices[i] = outlier_list[i]
        outlier_scores[i] = score_list[i]

    return outlier_indices, outlier_scores, majority


@jit(nopython=True, cache=True, parallel=True)
def _compute_split_gain_continuous(
    target: np.ndarray,
    feature: np.ndarray,
    valid_mask: np.ndarray,
) -> tuple[float, float]:
    """Find best split point for continuous feature to minimize target variance."""
    valid_idx = np.where(valid_mask)[0]
    n = len(valid_idx)

    if n < 70:  # Need enough cases for split
        return -1.0, 0.0

    # Sort by feature value
    order = np.argsort(feature[valid_idx])
    sorted_target = target[valid_idx[order]]
    sorted_feature = feature[valid_idx[order]]

    # Compute cumulative stats for efficient variance calculation
    cum_sum = np.cumsum(sorted_target)
    cum_sum_sq = np.cumsum(sorted_target ** 2)

    total_sum = cum_sum[-1]
    total_sum_sq = cum_sum_sq[-1]
    total_var = total_sum_sq / n - (total_sum / n) ** 2

    if total_var < 1e-10:
        return -1.0, 0.0

    best_gain = -1.0
    best_cut = 0.0
    min_leaf = max(35, n // 10)

    for i in range(min_leaf, n - min_leaf):
        # Skip if same value
        if sorted_feature[i] == sorted_feature[i-1]:
            continue

        # Left variance
        n_left = i
        left_sum = cum_sum[i-1]
        left_sum_sq = cum_sum_sq[i-1]
        left_var = left_sum_sq / n_left - (left_sum / n_left) ** 2

        # Right variance
        n_right = n - i
        right_sum = total_sum - left_sum
        right_sum_sq = total_sum_sq - left_sum_sq
        right_var = right_sum_sq / n_right - (right_sum / n_right) ** 2

        # Weighted variance reduction
        weighted_var = (n_left * left_var + n_right * right_var) / n
        gain = total_var - weighted_var

        if gain > best_gain:
            best_gain = gain
            best_cut = (sorted_feature[i-1] + sorted_feature[i]) / 2

    return best_gain, best_cut


class GritBotDetector(BaseEstimator, OutlierMixin):
    """GritBot-style anomaly detection via recursive partitioning.

    GritBot finds anomalies by recursively partitioning data to find
    homogeneous subsets, then identifying values that are surprising
    given the subset context. This approach is particularly effective for:
    - Data with mixed attribute types
    - Context-dependent anomalies (value is only anomalous in certain contexts)
    - Interpretable anomaly explanations

    Parameters
    ----------
    max_conditions : int, default=4
        Maximum number of conditions (splits) defining a subset context.
    filtering_level : float, default=50.0
        Controls sensitivity (0-100). Higher = fewer but more confident anomalies.
        - 0: MINABNORM=4 (more sensitive)
        - 50: MINABNORM=8 (default)
        - 100: MINABNORM=20 (very conservative)
    contamination : float, default=0.01
        Maximum expected proportion of anomalies.
    min_cases : int or None, default=None
        Minimum cases in a subset to check for anomalies.
        None uses max(35, 0.5% of data).
    categorical_features : list or None, default=None
        Indices of categorical features. If None, auto-detected.
    n_jobs : int, default=1
        Parallel jobs (currently not used, reserved for future).
    random_state : int or None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    anomalies_ : list[Anomaly]
        Detected anomalies with full context.
    anomaly_indices_ : np.ndarray
        Indices of detected anomaly cases.
    anomaly_scores_ : np.ndarray
        Scores for each sample (higher = more anomalous).

    References
    ----------
    Quinlan, J.R. (2010). GritBot GPL Edition. Rulequest Research.

    Examples
    --------
    >>> from endgame.anomaly import GritBotDetector
    >>> detector = GritBotDetector(filtering_level=50)
    >>> detector.fit(X_train)
    >>> scores = detector.decision_function(X_test)
    >>> labels = detector.predict(X_test)  # 1 = anomaly
    >>>
    >>> # Get interpretable anomaly explanations
    >>> for anomaly in detector.anomalies_[:5]:
    ...     print(f"Case {anomaly.case_idx}: feature {anomaly.feature_idx}")
    ...     print(f"  Value: {anomaly.value}, Expected: {anomaly.expected_value}")
    ...     print(f"  Context: {anomaly.context.conditions}")
    """

    def __init__(
        self,
        max_conditions: int = 4,
        filtering_level: float = 50.0,
        contamination: float = 0.01,
        min_cases: int | None = None,
        categorical_features: list | None = None,
        n_jobs: int = 1,
        random_state: int | None = None,
    ):
        self.max_conditions = max_conditions
        self.filtering_level = filtering_level
        self.contamination = contamination
        self.min_cases = min_cases
        self.categorical_features = categorical_features
        self.n_jobs = n_jobs
        self.random_state = random_state

    def _get_minabnorm(self) -> float:
        """Compute MINABNORM from filtering level (per GritBot formula)."""
        cf = np.clip(self.filtering_level, 0, 100)
        if cf < 50:
            return 0.08 * cf + 4
        else:
            return 0.24 * cf - 4

    def _detect_categorical(self, X: np.ndarray) -> np.ndarray:
        """Auto-detect categorical features."""
        n_samples, n_features = X.shape
        is_categorical = np.zeros(n_features, dtype=bool)

        for j in range(n_features):
            col = X[:, j]
            unique = np.unique(col[~np.isnan(col)])

            # Heuristic: categorical if few unique values or all integers
            if len(unique) <= 20 or len(unique) <= 0.05 * n_samples and np.allclose(col[~np.isnan(col)], col[~np.isnan(col)].astype(int)):
                is_categorical[j] = True

        return is_categorical

    def _compute_priors(self, X: np.ndarray, is_categorical: np.ndarray) -> list[np.ndarray]:
        """Compute prior probabilities for categorical features."""
        n_samples, n_features = X.shape
        priors = []

        for j in range(n_features):
            if is_categorical[j]:
                col = X[:, j].astype(int)
                max_val = int(np.nanmax(col)) + 1
                counts = np.zeros(max_val)
                for v in col:
                    if not np.isnan(v) and 0 <= v < max_val:
                        counts[int(v)] += 1
                priors.append(counts / n_samples)
            else:
                priors.append(np.array([1.0]))

        return priors

    def fit(self, X: ArrayLike, y=None) -> GritBotDetector:
        """Fit the GritBot detector and find anomalies.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored
            Not used.

        Returns
        -------
        self : GritBotDetector
            Fitted detector.
        """
        X = check_array(X, accept_sparse=False, ensure_all_finite='allow-nan')
        n_samples, n_features = X.shape

        self.n_features_in_ = n_features

        # Set minimum cases
        if self.min_cases is None:
            self._min_cases = max(35, int(0.005 * n_samples))
        else:
            self._min_cases = self.min_cases

        # Detect categorical features
        if self.categorical_features is not None:
            self._is_categorical = np.zeros(n_features, dtype=bool)
            self._is_categorical[self.categorical_features] = True
        else:
            self._is_categorical = self._detect_categorical(X)

        # Compute priors for categorical features
        self._priors = self._compute_priors(X, self._is_categorical)

        # Get parameters
        minabnorm = self._get_minabnorm()

        # Store training data for anomaly detection
        self._X_train = X.copy()

        # Precompute statistics for each feature (for fast scoring)
        self._feature_stats = []
        for j in range(n_features):
            col = X[:, j]
            valid = col[~np.isnan(col)]
            if len(valid) >= self._min_cases:
                mean, std = _trimmed_mean_std(valid, MAXFRAC)
                self._feature_stats.append((mean, std, len(valid)))
            else:
                self._feature_stats.append((0.0, 1.0, 0))

        # Find anomalies using recursive partitioning
        self.anomalies_ = []
        self._anomaly_scores = np.zeros(n_samples)

        # Check each feature as the target attribute
        for target_att in range(n_features):
            self._check_attribute(
                X,
                np.arange(n_samples),
                target_att,
                [],  # conditions
                minabnorm,
            )

        # Collect results
        self.anomaly_indices_ = np.array([a.case_idx for a in self.anomalies_])

        # Compute threshold for predict
        if len(self.anomalies_) > 0:
            scores = self._anomaly_scores.copy()
            # Set threshold based on contamination
            self.threshold_ = np.percentile(
                scores[scores > 0],
                100 * (1 - self.contamination)
            ) if (scores > 0).sum() > 0 else 0.5
        else:
            self.threshold_ = 0.5

        return self

    def _check_attribute(
        self,
        X: np.ndarray,
        indices: np.ndarray,
        target_att: int,
        conditions: list,
        minabnorm: float,
    ):
        """Check an attribute for anomalies within a subset."""
        n = len(indices)
        if n < self._min_cases:
            return

        # Get target values (exclude missing)
        target_vals = X[indices, target_att]
        valid_mask = ~np.isnan(target_vals)
        valid_indices = indices[valid_mask]
        valid_vals = target_vals[valid_mask]

        if len(valid_vals) < self._min_cases:
            return

        # Find outliers in this subset
        if self._is_categorical[target_att]:
            outlier_idx, scores, majority = _find_discrete_outliers(
                valid_vals.astype(np.int64),
                valid_indices,
                self._priors[target_att],
                minabnorm,
                MAXFRAC,
            )
            mean, std = float(majority), 0.0
            expected = int(majority)
        else:
            outlier_idx, scores, mean, std = _find_continuous_outliers(
                valid_vals,
                valid_indices,
                MAXNORM,
                minabnorm,
                MAXFRAC,
            )
            expected = mean

        # Record anomalies
        for i, (idx, score) in enumerate(zip(outlier_idx, scores)):
            # Only record if this is the best score for this case/feature
            if score < self._anomaly_scores[idx] or self._anomaly_scores[idx] == 0:
                self._anomaly_scores[idx] = 1.0 - score  # Convert to higher=more anomalous

                ctx = AnomalyContext(conditions=conditions.copy())
                anomaly = Anomaly(
                    case_idx=int(idx),
                    feature_idx=target_att,
                    value=X[idx, target_att],
                    score=score,
                    context=ctx,
                    group_size=len(valid_vals),
                    group_mean=mean,
                    group_std=std,
                    expected_value=expected,
                )
                self.anomalies_.append(anomaly)

        # Recursive partitioning if we can add more conditions
        if len(conditions) >= self.max_conditions:
            return

        # Find best split on other attributes
        best_gain = 0.0
        best_att = -1
        best_cut = 0.0

        for split_att in range(X.shape[1]):
            if split_att == target_att:
                continue

            split_vals = X[valid_indices, split_att]
            split_mask = ~np.isnan(split_vals)

            if split_mask.sum() < 2 * self._min_cases:
                continue

            if self._is_categorical[split_att]:
                # For categorical, use information gain (simplified)
                continue  # Skip for now, focus on continuous splits
            else:
                gain, cut = _compute_split_gain_continuous(
                    valid_vals[split_mask],
                    split_vals[split_mask],
                    np.ones(split_mask.sum(), dtype=bool),
                )

                if gain > best_gain:
                    best_gain = gain
                    best_att = split_att
                    best_cut = cut

        # Recursively check if good split found
        if best_gain > 0 and best_att >= 0:
            split_vals = X[valid_indices, best_att]

            # Left branch (<=)
            left_mask = split_vals <= best_cut
            if left_mask.sum() >= self._min_cases:
                left_cond = conditions + [(best_att, 'le', best_cut, None)]
                self._check_attribute(
                    X, valid_indices[left_mask], target_att, left_cond, minabnorm
                )

            # Right branch (>)
            right_mask = split_vals > best_cut
            if right_mask.sum() >= self._min_cases:
                right_cond = conditions + [(best_att, 'gt', best_cut, None)]
                self._check_attribute(
                    X, valid_indices[right_mask], target_att, right_cond, minabnorm
                )

    def decision_function(self, X: ArrayLike) -> np.ndarray:
        """Compute anomaly scores for samples.

        Higher scores indicate more anomalous samples.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores.
        """
        check_is_fitted(self, ["anomalies_", "_feature_stats"])
        X = check_array(X, accept_sparse=False, ensure_all_finite='allow-nan')

        n_samples, n_features = X.shape
        minabnorm = self._get_minabnorm()

        # Vectorized scoring using precomputed statistics
        scores = np.zeros(n_samples)

        for j in range(n_features):
            mean, std, n_valid = self._feature_stats[j]
            if n_valid < self._min_cases:
                continue

            vals = X[:, j]
            valid_mask = ~np.isnan(vals)

            if self._is_categorical[j]:
                # Score based on rarity
                priors = self._priors[j]
                for i in np.where(valid_mask)[0]:
                    v = int(vals[i])
                    if 0 <= v < len(priors):
                        score = 1.0 - priors[v]
                    else:
                        score = 0.99
                    scores[i] = max(scores[i], score)
            else:
                # Vectorized Z-score calculation
                z = np.abs(vals - mean) / max(std, 1e-10)

                # Compute scores vectorized
                feat_scores = np.zeros(n_samples)
                high_z = z > minabnorm
                mid_z = (z > MAXNORM) & ~high_z

                feat_scores[high_z] = 1.0 - 1.0 / (z[high_z] ** 2)
                feat_scores[mid_z] = 0.5 * (z[mid_z] - MAXNORM) / (minabnorm - MAXNORM)

                # Apply mask and take max
                feat_scores[~valid_mask] = 0.0
                scores = np.maximum(scores, feat_scores)

        return scores

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Predict anomaly labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples to classify.

        Returns
        -------
        labels : ndarray of shape (n_samples,)
            1 for anomalies, 0 for normal samples.
        """
        scores = self.decision_function(X)
        return (scores >= self.threshold_).astype(int)

    def fit_predict(self, X: ArrayLike, y=None) -> np.ndarray:
        """Fit and return anomaly labels for training data."""
        self.fit(X)
        # For training data, use the stored scores
        return (self._anomaly_scores >= self.threshold_).astype(int)

    def get_anomaly_report(self, max_anomalies: int = 10) -> str:
        """Generate a human-readable anomaly report.

        Parameters
        ----------
        max_anomalies : int, default=10
            Maximum anomalies to include in report.

        Returns
        -------
        report : str
            Formatted anomaly report.
        """
        check_is_fitted(self, ["anomalies_"])

        if not self.anomalies_:
            return "No anomalies detected."

        # Sort by score (most anomalous first)
        sorted_anoms = sorted(self.anomalies_, key=lambda a: a.score)[:max_anomalies]

        lines = [f"GritBot Anomaly Report ({len(self.anomalies_)} total anomalies)"]
        lines.append("=" * 60)

        for i, anom in enumerate(sorted_anoms, 1):
            lines.append(f"\n{i}. Case {anom.case_idx} [score: {anom.score:.4f}]")
            lines.append(f"   Feature {anom.feature_idx}: {anom.value:.4f}" if isinstance(anom.value, float) else f"   Feature {anom.feature_idx}: {anom.value}")
            if anom.expected_value is not None:
                lines.append(f"   Expected: {anom.expected_value:.4f}" if isinstance(anom.expected_value, float) else f"   Expected: {anom.expected_value}")
            lines.append(f"   Group: {anom.group_size} cases, mean={anom.group_mean:.3f}, std={anom.group_std:.3f}")

            if anom.context.conditions:
                lines.append("   Context:")
                for feat, op, val1, val2 in anom.context.conditions:
                    if op == 'le':
                        lines.append(f"      Feature {feat} <= {val1:.3f}")
                    elif op == 'gt':
                        lines.append(f"      Feature {feat} > {val1:.3f}")
                    elif op == 'in_range':
                        lines.append(f"      Feature {feat} in [{val1:.3f}, {val2:.3f}]")

        return "\n".join(lines)
