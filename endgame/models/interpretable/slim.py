from __future__ import annotations

"""SLIM and FasterRisk: Sparse Integer Linear Models for Risk Scoring.

SLIM (Supersparse Linear Integer Models) and FasterRisk produce scoring
systems with small integer coefficients that can be computed by hand.
These are ideal for clinical risk calculators and other applications
requiring extreme interpretability.

References
----------
- Ustun & Rudin "Supersparse Linear Integer Models for Optimized
  Medical Scoring Systems" (Machine Learning 2016)
- Liu & Rudin "FasterRisk: Fast and Accurate Interpretable Risk Scores"
  (NeurIPS 2022)
- https://github.com/ustunb/slim-python
- https://github.com/jiachangliu/FasterRisk

Example
-------
>>> from endgame.models.interpretable import SLIMClassifier, FasterRiskClassifier
>>> clf = SLIMClassifier(max_coef=5)
>>> clf.fit(X_train, y_train)
>>> print(clf.get_scorecard())
>>> predictions = clf.predict(X_test)
"""


import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import KBinsDiscretizer, LabelEncoder
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y


def _round_to_integers(
    coefficients: np.ndarray,
    max_coef: int = 5,
    sparsity: int = None,
) -> np.ndarray:
    """Round coefficients to small integers.

    Uses a simple rounding scheme that preserves relative magnitudes.
    """
    if sparsity is not None and sparsity < len(coefficients):
        # Keep only top sparsity coefficients
        threshold = np.sort(np.abs(coefficients))[-sparsity]
        coefficients = np.where(np.abs(coefficients) >= threshold, coefficients, 0)

    # Scale to target range
    max_abs = np.abs(coefficients).max()
    if max_abs > 0:
        scaled = coefficients / max_abs * max_coef
    else:
        scaled = coefficients

    # Round to integers
    rounded = np.round(scaled).astype(int)

    return rounded


class SLIMClassifier(ClassifierMixin, BaseEstimator):
    """Supersparse Linear Integer Model Classifier.

    SLIM produces scoring systems with small integer coefficients that can
    be computed by hand. The model is a linear classifier with constraints
    on coefficient magnitude and sparsity.

    This is a simplified implementation using L1-regularized logistic
    regression followed by rounding. For the full SLIM optimization,
    use the original slim-python package.

    Parameters
    ----------
    max_coef : int, default=5
        Maximum absolute value for integer coefficients.
        Smaller values produce simpler scorecards.

    sparsity : int or None, default=None
        Maximum number of non-zero coefficients. If None, no sparsity
        constraint (uses L1 regularization instead).

    C : float, default=1.0
        Inverse of regularization strength for initial logistic regression.
        Smaller values produce sparser models.

    discretize : bool, default=True
        If True, discretize continuous features into bins.
        This often improves interpretability.

    n_bins : int, default=5
        Number of bins for discretization.

    threshold_style : str, default="quantile"
        Style for bin thresholds: "quantile", "uniform", or "kmeans".

    include_intercept : bool, default=True
        Whether to include an intercept term in the scorecard.

    random_state : int, optional
        Random seed.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    coef_ : ndarray
        Integer coefficients (after rounding).

    intercept_ : int
        Integer intercept.

    feature_names_ : list of str
        Names of features in the scorecard.

    scorecard_ : list of dict
        The scoring system as a list of (feature, points) pairs.

    Examples
    --------
    >>> from endgame.models.interpretable import SLIMClassifier
    >>> clf = SLIMClassifier(max_coef=5, sparsity=10)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.get_scorecard())
    >>> # Example output:
    >>> # Age >= 65:      +3 points
    >>> # Diabetes:       +2 points
    >>> # Heart disease:  +4 points
    >>> # Total score >= 5: High risk
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        max_coef: int = 5,
        sparsity: int | None = None,
        C: float = 1.0,
        discretize: bool = True,
        n_bins: int = 5,
        threshold_style: str = "quantile",
        include_intercept: bool = True,
        random_state: int | None = None,
    ):
        self.max_coef = max_coef
        self.sparsity = sparsity
        self.C = C
        self.discretize = discretize
        self.n_bins = n_bins
        self.threshold_style = threshold_style
        self.include_intercept = include_intercept
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> SLIMClassifier:
        """Fit the SLIM classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target labels (binary).
        feature_names : list of str, optional
            Names for features.
        sample_weight : array-like, optional
            Sample weights.

        Returns
        -------
        self : SLIMClassifier
        """
        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        if len(self.classes_) != 2:
            raise ValueError("SLIM only supports binary classification.")

        # Feature names
        if feature_names is not None:
            self._original_feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self._original_feature_names = list(X.columns)
        else:
            self._original_feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        # Discretize if needed
        if self.discretize:
            X_processed, self.feature_names_ = self._discretize(X)
        else:
            X_processed = X
            self.feature_names_ = self._original_feature_names
            self._discretizer = None

        # Fit logistic regression with L1 penalty
        self._fit_logistic(X_processed, y_encoded, sample_weight)

        # Round coefficients to integers
        self._round_coefficients()

        # Build scorecard
        self._build_scorecard()

        return self

    def _discretize(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Discretize features into binary indicators."""
        self._discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="onehot-dense",
            strategy=self.threshold_style,
            random_state=self.random_state,
        )

        X_binned = self._discretizer.fit_transform(X)

        # Generate descriptive names
        feature_names = []
        for feat_idx, edges in enumerate(self._discretizer.bin_edges_):
            feat_name = self._original_feature_names[feat_idx]
            for bin_idx in range(len(edges) - 1):
                if bin_idx == 0:
                    name = f"{feat_name} <= {edges[bin_idx + 1]:.2g}"
                elif bin_idx == len(edges) - 2:
                    name = f"{feat_name} > {edges[bin_idx]:.2g}"
                else:
                    name = f"{edges[bin_idx]:.2g} < {feat_name} <= {edges[bin_idx + 1]:.2g}"
                feature_names.append(name)

        return X_binned, feature_names

    def _fit_logistic(self, X, y, sample_weight):
        """Fit L1-regularized logistic regression."""
        self._logistic = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            C=self.C,
            random_state=self.random_state,
            max_iter=1000,
        )
        self._logistic.fit(X, y, sample_weight=sample_weight)

        self._raw_coef = self._logistic.coef_.ravel()
        self._raw_intercept = self._logistic.intercept_[0]

    def _round_coefficients(self):
        """Round coefficients to small integers."""
        self.coef_ = _round_to_integers(
            self._raw_coef,
            max_coef=self.max_coef,
            sparsity=self.sparsity,
        )

        # Round intercept similarly
        if self.include_intercept and self._raw_intercept != 0:
            max_raw = max(np.abs(self._raw_coef).max(), abs(self._raw_intercept))
            if max_raw > 0:
                scaled_intercept = self._raw_intercept / max_raw * self.max_coef
                self.intercept_ = int(np.round(scaled_intercept))
            else:
                self.intercept_ = 0
        else:
            self.intercept_ = 0

        # Compute threshold for classification
        # This is the score at which P(y=1) = 0.5
        self._threshold = -self.intercept_

    def _build_scorecard(self):
        """Build human-readable scorecard."""
        self.scorecard_ = []

        for i, (name, coef) in enumerate(zip(self.feature_names_, self.coef_)):
            if coef != 0:
                self.scorecard_.append({
                    "feature": name,
                    "points": int(coef),
                    "index": i,
                })

        # Sort by absolute points
        self.scorecard_.sort(key=lambda x: abs(x["points"]), reverse=True)

    def predict(self, X) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
        """
        check_is_fitted(self, "coef_")
        X = check_array(X)

        scores = self._compute_scores(X)
        y_pred = (scores >= self._threshold).astype(int)

        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities.

        Uses sigmoid of integer scores for probabilistic predictions.

        Parameters
        ----------
        X : array-like

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
        """
        check_is_fitted(self, "coef_")
        X = check_array(X)

        scores = self._compute_scores(X)

        # Convert scores to probabilities using sigmoid
        # Scale factor to approximate original logistic regression
        scale = np.abs(self._raw_coef).sum() / (np.abs(self.coef_).sum() + 1e-10)
        proba_1 = 1 / (1 + np.exp(-scores / max(scale, 0.1)))
        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute integer scores for samples."""
        if self.discretize and self._discretizer is not None:
            X_processed = self._discretizer.transform(X)
        else:
            X_processed = X

        scores = X_processed @ self.coef_ + self.intercept_
        return scores

    def get_scorecard(self) -> str:
        """Get a human-readable scorecard.

        Returns
        -------
        scorecard : str
            Formatted scoring system.
        """
        check_is_fitted(self, "scorecard_")

        lines = []
        lines.append("=" * 50)
        lines.append("SLIM Scoring System")
        lines.append("=" * 50)
        lines.append("")

        for item in self.scorecard_:
            points = item["points"]
            sign = "+" if points > 0 else ""
            lines.append(f"  {item['feature']:40s} {sign}{points} points")

        lines.append("")
        lines.append("-" * 50)
        lines.append(f"Base score: {self.intercept_}")
        lines.append(f"Classification threshold: {self._threshold}")
        lines.append(f"If total score >= {self._threshold}: predict {self.classes_[1]}")
        lines.append(f"If total score <  {self._threshold}: predict {self.classes_[0]}")
        lines.append("=" * 50)

        return "\n".join(lines)

    def score_sample(self, x: np.ndarray) -> tuple[int, list[tuple[str, int]]]:
        """Score a single sample and show the breakdown.

        Parameters
        ----------
        x : array-like of shape (n_features,)
            Single sample.

        Returns
        -------
        total_score : int
            Total integer score.
        breakdown : list of (feature_name, points)
            Score breakdown by feature.
        """
        check_is_fitted(self, "coef_")
        x = np.asarray(x).reshape(1, -1)

        if self.discretize and self._discretizer is not None:
            x_processed = self._discretizer.transform(x).ravel()
        else:
            x_processed = x.ravel()

        breakdown = []
        total = self.intercept_

        for i, (name, coef) in enumerate(zip(self.feature_names_, self.coef_)):
            if coef != 0 and x_processed[i] > 0:
                points = int(coef * x_processed[i])
                breakdown.append((name, points))
                total += points

        return total, breakdown


class FasterRiskClassifier(ClassifierMixin, BaseEstimator):
    """FasterRisk Classifier.

    Produces optimized sparse integer risk scores using beam search,
    diverse pool collection, and star ray search with auxiliary loss
    rounding. Native implementation based on:

    Liu & Rudin, "FasterRisk: Fast and Accurate Interpretable Risk
    Scores", NeurIPS 2022.

    Parameters
    ----------
    max_coef : int, default=5
        Maximum absolute value for integer coefficients.

    sparsity : int, default=10
        Maximum number of non-zero coefficients.

    n_models : int, default=50
        Number of candidate models to explore in beam search.

    parent_size : int, default=10
        Number of best models to keep at each beam search step.

    n_iters : int, default=1000
        Maximum iterations for coordinate descent.

    discretize : bool, default=True
        Whether to discretize continuous features.

    n_bins : int, default=5
        Number of bins for discretization.

    random_state : int, optional
        Random seed.

    Attributes
    ----------
    classes_ : ndarray
        Class labels.

    coef_ : ndarray
        Integer coefficients.

    intercept_ : int
        Integer intercept.

    multiplier_ : float
        Scaling factor for probability calibration.

    Examples
    --------
    >>> from endgame.models.interpretable import FasterRiskClassifier
    >>> clf = FasterRiskClassifier(max_coef=5, sparsity=10)
    >>> clf.fit(X_train, y_train)
    >>> print(clf.get_scorecard())
    """

    _estimator_type = "classifier"

    def __init__(
        self,
        max_coef: int = 5,
        sparsity: int = 10,
        n_models: int = 50,
        parent_size: int = 10,
        n_iters: int = 1000,
        discretize: bool = True,
        n_bins: int = 5,
        random_state: int | None = None,
    ):
        self.max_coef = max_coef
        self.sparsity = sparsity
        self.n_models = n_models
        self.parent_size = parent_size
        self.n_iters = n_iters
        self.discretize = discretize
        self.n_bins = n_bins
        self.random_state = random_state

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> FasterRiskClassifier:
        """Fit the FasterRisk classifier.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        feature_names : list of str, optional
        sample_weight : ignored

        Returns
        -------
        self : FasterRiskClassifier
        """
        from endgame.models.interpretable._fasterrisk_core import fasterrisk_fit

        X, y = check_X_y(X, y)
        self.n_features_in_ = X.shape[1]

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_

        if len(self.classes_) != 2:
            raise ValueError("FasterRisk only supports binary classification.")

        # Feature names
        if feature_names is not None:
            self._original_feature_names = list(feature_names)
        elif hasattr(X, "columns"):
            self._original_feature_names = list(X.columns)
        else:
            self._original_feature_names = [f"x{i}" for i in range(self.n_features_in_)]

        # Discretize
        if self.discretize:
            X_processed, self.feature_names_ = self._discretize(X)
        else:
            X_processed = X.astype(np.float64)
            self.feature_names_ = self._original_feature_names
            self._discretizer = None

        # Convert y to {-1, +1}
        y_binary = (2 * y_encoded - 1).astype(np.float64)

        # Fit using native FasterRisk algorithm
        self.multiplier_, beta0, betas = fasterrisk_fit(
            X=X_processed,
            y_binary=y_binary,
            k=self.sparsity,
            max_coef=self.max_coef,
            parent_size=self.parent_size,
            pool_size=self.n_models,
            num_rays=self.n_models,
            max_iter=self.n_iters,
        )

        self.coef_ = betas
        self.intercept_ = int(beta0)
        self._threshold = 0

        # Build scorecard
        self._build_scorecard()

        return self

    def _discretize(self, X: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Discretize features."""
        self._discretizer = KBinsDiscretizer(
            n_bins=self.n_bins,
            encode="onehot-dense",
            strategy="quantile",
            random_state=self.random_state,
        )

        X_binned = self._discretizer.fit_transform(X).astype(np.float64)

        feature_names = []
        for feat_idx, edges in enumerate(self._discretizer.bin_edges_):
            feat_name = self._original_feature_names[feat_idx]
            for bin_idx in range(len(edges) - 1):
                if bin_idx == 0:
                    name = f"{feat_name} <= {edges[bin_idx + 1]:.2g}"
                elif bin_idx == len(edges) - 2:
                    name = f"{feat_name} > {edges[bin_idx]:.2g}"
                else:
                    name = f"{edges[bin_idx]:.2g} < {feat_name} <= {edges[bin_idx + 1]:.2g}"
                feature_names.append(name)

        return X_binned, feature_names

    def _build_scorecard(self):
        """Build scorecard."""
        self.scorecard_ = []

        for i, (name, coef) in enumerate(zip(self.feature_names_, self.coef_)):
            if coef != 0:
                self.scorecard_.append({
                    "feature": name,
                    "points": int(coef),
                    "index": i,
                })

        self.scorecard_.sort(key=lambda x: abs(x["points"]), reverse=True)

    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        check_is_fitted(self, "coef_")
        X = check_array(X)

        scores = self._compute_scores(X)
        y_pred = (scores >= self._threshold).astype(int)

        return self._label_encoder.inverse_transform(y_pred)

    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        check_is_fitted(self, "coef_")
        X = check_array(X)

        scores = self._compute_scores(X)

        # Calibrated probability using FasterRisk's multiplier
        m = getattr(self, "multiplier_", 1.0)
        proba_1 = 1 / (1 + np.exp(-scores / m))
        proba_0 = 1 - proba_1

        return np.column_stack([proba_0, proba_1])

    def _compute_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute scores."""
        if self.discretize and self._discretizer is not None:
            X_processed = self._discretizer.transform(X)
        else:
            X_processed = X

        return X_processed @ self.coef_ + self.intercept_

    def get_scorecard(self) -> str:
        """Get human-readable scorecard."""
        check_is_fitted(self, "scorecard_")

        lines = []
        lines.append("=" * 50)
        lines.append("FasterRisk Scoring System")
        lines.append("=" * 50)
        lines.append("")

        for item in self.scorecard_:
            points = item["points"]
            sign = "+" if points > 0 else ""
            lines.append(f"  {item['feature']:40s} {sign}{points} points")

        lines.append("")
        lines.append("-" * 50)
        lines.append(f"Base score: {self.intercept_}")
        lines.append(f"If total score >= 0: predict {self.classes_[1]}")
        lines.append(f"If total score <  0: predict {self.classes_[0]}")
        lines.append("=" * 50)

        return "\n".join(lines)

    def score_sample(self, x: np.ndarray) -> tuple[int, list[tuple[str, int]]]:
        """Score a single sample with breakdown."""
        check_is_fitted(self, "coef_")
        x = np.asarray(x).reshape(1, -1)

        if self.discretize and self._discretizer is not None:
            x_processed = self._discretizer.transform(x).ravel()
        else:
            x_processed = x.ravel()

        breakdown = []
        total = self.intercept_

        for i, (name, coef) in enumerate(zip(self.feature_names_, self.coef_)):
            if coef != 0 and x_processed[i] > 0:
                points = int(coef * x_processed[i])
                breakdown.append((name, points))
                total += points

        return total, breakdown
