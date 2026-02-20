"""Oblique split direction finding methods for Oblique Random Forests.

This module provides multiple methods for finding linear combination split
directions in oblique decision trees. Each method produces candidate directions
that can capture linear decision boundaries more efficiently than axis-aligned splits.

Methods:
    - Ridge: Ridge regression on target labels (recommended default)
    - PCA: Principal Component Analysis directions
    - LDA: Linear Discriminant Analysis (classification only)
    - Random: Sparse random projections (fastest)
    - SVM: Linear SVM hyperplane directions
    - Householder: Householder reflection matrices
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass
class ObliqueSplit:
    """Represents an oblique (linear combination) split.

    Split condition: sum(coefficients[i] * X[:, feature_indices[i]]) <= threshold

    Attributes
    ----------
    feature_indices : ndarray of shape (n_features_in_split,)
        Indices of features involved in this split.
    coefficients : ndarray of shape (n_features_in_split,)
        Coefficients for the linear combination.
    threshold : float
        Split threshold value.
    impurity : float
        Impurity of the parent node before split.
    impurity_left : float
        Impurity of left child after split.
    impurity_right : float
        Impurity of right child after split.
    n_samples_left : int
        Number of samples going to left child.
    n_samples_right : int
        Number of samples going to right child.
    """

    feature_indices: np.ndarray
    coefficients: np.ndarray
    threshold: float
    impurity: float = 0.0
    impurity_left: float = 0.0
    impurity_right: float = 0.0
    n_samples_left: int = 0
    n_samples_right: int = 0

    def compute_projection(self, X: np.ndarray) -> np.ndarray:
        """Compute the linear combination for all samples.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        projection : ndarray of shape (n_samples,)
            X[:, feature_indices] @ coefficients
        """
        return X[:, self.feature_indices] @ self.coefficients

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Determine which samples go left (True) or right (False).

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        goes_left : ndarray of shape (n_samples,) dtype=bool
            True for samples that go to left child.
        """
        projection = self.compute_projection(X)
        return projection <= self.threshold

    def __str__(self) -> str:
        terms = []
        for idx, coef in zip(self.feature_indices, self.coefficients):
            if abs(coef) > 1e-10:
                terms.append(f"{coef:+.4f}*x{idx}")
        return f"({' '.join(terms)}) <= {self.threshold:.4f}"


# ---------------------------------------------------------------------------
# Direction Finding Methods
# ---------------------------------------------------------------------------


def get_ridge_directions(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    ridge_alpha: float = 1.0,
    n_directions: int = 1,
) -> np.ndarray:
    """Get split directions using Ridge regression.

    For classification: treats class labels as numeric and fits Ridge regression.
    The coefficient vector points in the direction of maximum class separation.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Target values (class labels or continuous).
    sample_weight : ndarray or None
        Sample weights.
    ridge_alpha : float
        Regularization strength.
    n_directions : int
        Number of directions to return.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Normalized direction vectors.
    """
    # Convert labels to numeric if needed
    if not np.issubdtype(y.dtype, np.floating):
        y_numeric = y.astype(np.float64)
    else:
        y_numeric = y

    # Fit Ridge regression
    ridge = Ridge(alpha=ridge_alpha, fit_intercept=True)
    ridge.fit(X, y_numeric, sample_weight=sample_weight)

    # Get coefficient vector
    direction = ridge.coef_.flatten()

    # Normalize
    norm = np.linalg.norm(direction)
    if norm > 1e-10:
        direction = direction / norm
    else:
        # Fallback to first axis if coefficients are zero
        direction = np.zeros(X.shape[1])
        direction[0] = 1.0

    return direction.reshape(1, -1)


def get_pca_directions(
    X: np.ndarray,
    y: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    n_directions: int = 1,
) -> np.ndarray:
    """Get split directions using Principal Component Analysis.

    Returns the top principal components as candidate directions.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ignored
        Not used, present for API consistency.
    sample_weight : ignored
        Not used, present for API consistency.
    n_directions : int
        Number of principal components to return.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Principal component directions.
    """
    n_samples, n_features = X.shape
    n_components = min(n_directions, n_features, n_samples - 1)

    if n_components < 1:
        return np.eye(n_features)[:1]

    # Center the data
    X_centered = X - X.mean(axis=0)

    # Compute covariance and eigenvectors
    try:
        cov = np.cov(X_centered, rowvar=False)
        if cov.ndim == 0:
            # Single feature case
            return np.array([[1.0]])

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        directions = eigenvectors[:, idx[:n_components]].T

        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions = directions / norms

        return directions

    except (np.linalg.LinAlgError, ValueError):
        # Fallback to identity
        return np.eye(n_features)[:n_directions]


def get_lda_directions(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    n_directions: int = 1,
) -> np.ndarray:
    """Get split directions using Linear Discriminant Analysis.

    LDA finds directions that maximize between-class variance
    relative to within-class variance.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Class labels.
    sample_weight : ndarray or None
        Sample weights (not used directly in this implementation).
    n_directions : int
        Number of discriminant directions.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Discriminant directions.
    """
    classes = np.unique(y)
    n_classes = len(classes)
    n_samples, n_features = X.shape

    if n_classes < 2:
        return np.eye(n_features)[:n_directions]

    n_components = min(n_directions, n_classes - 1, n_features)

    try:
        # Compute class means
        overall_mean = X.mean(axis=0)
        class_means = []
        class_counts = []

        for c in classes:
            mask = y == c
            class_means.append(X[mask].mean(axis=0))
            class_counts.append(np.sum(mask))

        class_means = np.array(class_means)
        class_counts = np.array(class_counts)

        # Between-class scatter matrix
        S_B = np.zeros((n_features, n_features))
        for i, (mean, count) in enumerate(zip(class_means, class_counts)):
            diff = (mean - overall_mean).reshape(-1, 1)
            S_B += count * (diff @ diff.T)

        # Within-class scatter matrix
        S_W = np.zeros((n_features, n_features))
        for c, mean in zip(classes, class_means):
            mask = y == c
            X_c = X[mask] - mean
            S_W += X_c.T @ X_c

        # Add regularization for numerical stability
        S_W += 1e-6 * np.eye(n_features)

        # Solve generalized eigenvalue problem
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.linalg.inv(S_W) @ S_B
        )

        # Sort by descending eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        directions = eigenvectors[:, idx[:n_components]].T

        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions = directions / norms

        return directions

    except (np.linalg.LinAlgError, ValueError):
        # Fallback to PCA if LDA fails
        return get_pca_directions(X, n_directions=n_directions)


def get_random_directions(
    X: np.ndarray,
    y: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    n_directions: int = 5,
    feature_combinations: int = 2,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Get split directions using random projections.

    Generates sparse random directions where each direction
    involves only a few features (controlled by feature_combinations).

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ignored
        Not used, present for API consistency.
    sample_weight : ignored
        Not used, present for API consistency.
    n_directions : int
        Number of random directions.
    feature_combinations : int
        Number of features per direction (sparsity).
    random_state : RandomState or None
        Random number generator.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Random direction vectors.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    n_features = X.shape[1]
    directions = []

    for _ in range(n_directions):
        # Select random subset of features
        n_select = min(feature_combinations, n_features)
        selected = random_state.choice(n_features, size=n_select, replace=False)

        # Random coefficients
        coeffs = random_state.randn(n_select)
        norm = np.linalg.norm(coeffs)
        if norm > 1e-10:
            coeffs = coeffs / norm

        # Create sparse direction vector
        direction = np.zeros(n_features)
        direction[selected] = coeffs

        directions.append(direction)

    return np.array(directions)


def get_svm_directions(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None = None,
    C: float = 1.0,
) -> np.ndarray:
    """Get split directions using Linear SVM.

    For multi-class, uses one-vs-rest and returns all hyperplanes.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ndarray of shape (n_samples,)
        Class labels.
    sample_weight : ndarray or None
        Sample weights.
    C : float
        SVM regularization parameter.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        SVM hyperplane directions.
    """
    from sklearn.svm import LinearSVC

    classes = np.unique(y)

    if len(classes) < 2:
        return np.eye(X.shape[1])[:1]

    try:
        # Scale data for SVM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        svm = LinearSVC(C=C, max_iter=2000, dual="auto", tol=1e-3)
        svm.fit(X_scaled, y, sample_weight=sample_weight)

        # Get coefficients (unscale them)
        directions = svm.coef_ / scaler.scale_

        # Normalize
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        directions = directions / norms

        return directions

    except Exception:
        # Fallback if SVM fails
        return get_ridge_directions(X, y, sample_weight)


def get_householder_directions(
    X: np.ndarray,
    y: np.ndarray | None = None,
    sample_weight: np.ndarray | None = None,
    n_directions: int = 5,
    random_state: np.random.RandomState | None = None,
) -> np.ndarray:
    """Get split directions using Householder reflections.

    Based on Menze et al. (2011). Generates orthogonal directions
    by reflecting random vectors.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    y : ignored
        Not used, present for API consistency.
    sample_weight : ignored
        Not used, present for API consistency.
    n_directions : int
        Number of directions.
    random_state : RandomState or None
        Random number generator.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Householder reflection directions.
    """
    if random_state is None:
        random_state = np.random.RandomState()

    n_features = X.shape[1]
    directions = []

    for _ in range(n_directions):
        # Random unit vector for Householder reflection
        v = random_state.randn(n_features)
        v = v / np.linalg.norm(v)

        # Random feature axis to reflect
        axis_idx = random_state.randint(n_features)
        e = np.zeros(n_features)
        e[axis_idx] = 1.0

        # Householder reflection: H @ e = e - 2*v*(v^T @ e)
        direction = e - 2 * v * v[axis_idx]
        norm = np.linalg.norm(direction)
        if norm > 1e-10:
            direction = direction / norm
        else:
            direction = e

        directions.append(direction)

    return np.array(directions)


# ---------------------------------------------------------------------------
# Direction Method Registry
# ---------------------------------------------------------------------------

DIRECTION_METHODS: dict[str, Callable] = {
    "ridge": get_ridge_directions,
    "pca": get_pca_directions,
    "lda": get_lda_directions,
    "random": get_random_directions,
    "svm": get_svm_directions,
    "householder": get_householder_directions,
}


def get_oblique_directions(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    method: str,
    random_state: np.random.RandomState | None = None,
    **kwargs,
) -> np.ndarray:
    """Get oblique split directions using the specified method.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Feature matrix for the current node.
    y : ndarray of shape (n_samples,)
        Target values.
    sample_weight : ndarray or None
        Sample weights.
    method : str
        One of 'ridge', 'pca', 'lda', 'random', 'svm', 'householder'.
    random_state : RandomState or None
        Random number generator.
    **kwargs : dict
        Method-specific parameters.

    Returns
    -------
    directions : ndarray of shape (n_directions, n_features)
        Candidate split directions.
    """
    if method not in DIRECTION_METHODS:
        raise ValueError(
            f"Unknown oblique_method: {method}. "
            f"Choose from {list(DIRECTION_METHODS.keys())}"
        )

    func = DIRECTION_METHODS[method]

    # Build kwargs based on method
    if method == "ridge":
        return func(
            X, y, sample_weight,
            ridge_alpha=kwargs.get("ridge_alpha", 1.0),
            n_directions=kwargs.get("n_directions", 1),
        )
    elif method == "pca" or method == "lda":
        return func(
            X, y, sample_weight,
            n_directions=kwargs.get("n_directions", 1),
        )
    elif method == "random":
        return func(
            X, y, sample_weight,
            n_directions=kwargs.get("n_directions", 5),
            feature_combinations=kwargs.get("feature_combinations", 2),
            random_state=random_state,
        )
    elif method == "svm":
        return func(
            X, y, sample_weight,
            C=kwargs.get("svm_C", 1.0),
        )
    elif method == "householder":
        return func(
            X, y, sample_weight,
            n_directions=kwargs.get("n_directions", 5),
            random_state=random_state,
        )

    return func(X, y, sample_weight, **kwargs)


# ---------------------------------------------------------------------------
# Impurity Functions
# ---------------------------------------------------------------------------


def compute_gini(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Compute Gini impurity.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels.
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    float
        Gini impurity (0 = pure, 0.5 = maximally impure for binary).
    """
    if len(y) == 0:
        return 0.0

    classes = np.unique(y)

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    total_weight = np.sum(sample_weight)
    if total_weight == 0:
        return 0.0

    gini = 1.0
    for c in classes:
        mask = y == c
        proportion = np.sum(sample_weight[mask]) / total_weight
        gini -= proportion ** 2

    return gini


def compute_entropy(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Compute entropy.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Class labels.
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    float
        Entropy.
    """
    if len(y) == 0:
        return 0.0

    classes = np.unique(y)

    if sample_weight is None:
        sample_weight = np.ones(len(y))

    total_weight = np.sum(sample_weight)
    if total_weight == 0:
        return 0.0

    entropy = 0.0
    for c in classes:
        mask = y == c
        proportion = np.sum(sample_weight[mask]) / total_weight
        if proportion > 0:
            entropy -= proportion * np.log2(proportion)

    return entropy


def compute_mse(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Compute mean squared error (variance) for regression.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    float
        MSE / variance.
    """
    if len(y) == 0:
        return 0.0

    if sample_weight is None:
        return np.var(y)

    total_weight = np.sum(sample_weight)
    if total_weight == 0:
        return 0.0

    mean = np.average(y, weights=sample_weight)
    mse = np.average((y - mean) ** 2, weights=sample_weight)
    return mse


def compute_mae(y: np.ndarray, sample_weight: np.ndarray | None = None) -> float:
    """Compute mean absolute error for regression.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target values.
    sample_weight : ndarray or None
        Sample weights.

    Returns
    -------
    float
        MAE.
    """
    if len(y) == 0:
        return 0.0

    if sample_weight is None:
        median = np.median(y)
        return np.mean(np.abs(y - median))

    # Weighted median
    sorted_idx = np.argsort(y)
    y_sorted = y[sorted_idx]
    weights_sorted = sample_weight[sorted_idx]
    cumsum = np.cumsum(weights_sorted)
    median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
    median = y_sorted[min(median_idx, len(y_sorted) - 1)]

    return np.average(np.abs(y - median), weights=sample_weight)


# ---------------------------------------------------------------------------
# Threshold Finding
# ---------------------------------------------------------------------------


def find_best_threshold(
    projection: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    feature_indices: np.ndarray,
    direction: np.ndarray,
    min_samples_leaf: int = 1,
    impurity_func: Callable = compute_gini,
) -> ObliqueSplit | None:
    """Find the best threshold for a given projection.

    Uses the standard CART approach applied to the projected (1D) data.

    Parameters
    ----------
    projection : ndarray of shape (n_samples,)
        Data projected onto split direction.
    y : ndarray of shape (n_samples,)
        Target values.
    sample_weight : ndarray or None
        Sample weights.
    feature_indices : ndarray
        Indices of features in the projection.
    direction : ndarray
        Coefficient vector for the projection.
    min_samples_leaf : int
        Minimum samples per leaf.
    impurity_func : callable
        Function to compute impurity (gini, entropy, mse).

    Returns
    -------
    best_split : ObliqueSplit or None
        Best split for this direction.
    """
    n_samples = len(y)

    if n_samples < 2 * min_samples_leaf:
        return None

    if sample_weight is None:
        sample_weight = np.ones(n_samples)

    # Sort by projection value
    sorted_indices = np.argsort(projection)
    projection_sorted = projection[sorted_indices]
    y_sorted = y[sorted_indices]
    weight_sorted = sample_weight[sorted_indices]

    # Parent impurity
    parent_impurity = impurity_func(y, sample_weight)

    if parent_impurity <= 0:
        return None

    best_threshold = None
    best_impurity_decrease = 0.0
    best_split_info = None

    # Find unique values and candidate thresholds
    unique_values, unique_indices = np.unique(projection_sorted, return_index=True)

    if len(unique_values) < 2:
        return None

    # Midpoint thresholds
    thresholds = (unique_values[:-1] + unique_values[1:]) / 2

    # For efficiency, we can use cumulative sums to compute impurities
    # But for correctness, let's do it directly for now
    for i, threshold in enumerate(thresholds):
        # Samples going left: projection <= threshold
        n_left = unique_indices[i + 1]
        n_right = n_samples - n_left

        if n_left < min_samples_leaf or n_right < min_samples_leaf:
            continue

        y_left = y_sorted[:n_left]
        y_right = y_sorted[n_left:]
        weight_left = weight_sorted[:n_left]
        weight_right = weight_sorted[n_left:]

        impurity_left = impurity_func(y_left, weight_left)
        impurity_right = impurity_func(y_right, weight_right)

        # Weighted impurity
        total_weight = np.sum(weight_sorted)
        weight_left_sum = np.sum(weight_left)
        weight_right_sum = np.sum(weight_right)

        weighted_impurity = (
            (weight_left_sum / total_weight) * impurity_left +
            (weight_right_sum / total_weight) * impurity_right
        )

        impurity_decrease = parent_impurity - weighted_impurity

        if impurity_decrease > best_impurity_decrease:
            best_impurity_decrease = impurity_decrease
            best_threshold = threshold
            best_split_info = (impurity_left, impurity_right, n_left, n_right)

    if best_threshold is None:
        return None

    return ObliqueSplit(
        feature_indices=feature_indices.copy(),
        coefficients=direction.copy(),
        threshold=best_threshold,
        impurity=parent_impurity,
        impurity_left=best_split_info[0],
        impurity_right=best_split_info[1],
        n_samples_left=best_split_info[2],
        n_samples_right=best_split_info[3],
    )


def find_best_oblique_split(
    X: np.ndarray,
    y: np.ndarray,
    sample_weight: np.ndarray | None,
    oblique_method: str,
    max_features: int,
    random_state: np.random.RandomState,
    min_samples_leaf: int = 1,
    impurity_func: Callable = compute_gini,
    include_axis_aligned: bool = True,
    **kwargs,
) -> ObliqueSplit | None:
    """Find the best oblique split for a node.

    Parameters
    ----------
    X : ndarray of shape (n_samples, n_features)
        Training data at this node.
    y : ndarray of shape (n_samples,)
        Target values.
    sample_weight : ndarray or None
        Sample weights.
    oblique_method : str
        Method for finding split direction.
    max_features : int
        Number of features to consider.
    random_state : RandomState
        Random number generator.
    min_samples_leaf : int
        Minimum samples per leaf.
    impurity_func : callable
        Impurity function.
    include_axis_aligned : bool
        Whether to also consider axis-aligned splits.
    **kwargs : dict
        Additional method-specific parameters.

    Returns
    -------
    best_split : ObliqueSplit or None
        Best split found, or None if no valid split exists.
    """
    n_samples, n_features = X.shape

    if n_samples < 2 * min_samples_leaf:
        return None

    # Select subset of features to consider
    if max_features < n_features:
        feature_indices = random_state.choice(
            n_features, size=max_features, replace=False
        )
    else:
        feature_indices = np.arange(n_features)

    X_subset = X[:, feature_indices]

    # Get candidate split directions based on method
    directions = get_oblique_directions(
        X_subset, y, sample_weight, oblique_method,
        random_state=random_state, **kwargs
    )

    # Also include axis-aligned splits (ensures we never do worse than standard RF)
    if include_axis_aligned:
        axis_aligned = np.eye(len(feature_indices))
        directions = np.vstack([directions, axis_aligned])

    # Find best split among all directions
    best_split = None
    best_impurity_decrease = 0.0

    parent_impurity = impurity_func(y, sample_weight)

    for direction in directions:
        # Normalize direction
        norm = np.linalg.norm(direction)
        if norm < 1e-10:
            continue
        direction = direction / norm

        # Project data onto this direction
        projection = X_subset @ direction

        # Find best threshold for this projection
        split = find_best_threshold(
            projection, y, sample_weight,
            feature_indices, direction,
            min_samples_leaf=min_samples_leaf,
            impurity_func=impurity_func,
        )

        if split is not None:
            impurity_decrease = parent_impurity - (
                (split.n_samples_left / n_samples) * split.impurity_left +
                (split.n_samples_right / n_samples) * split.impurity_right
            )

            if impurity_decrease > best_impurity_decrease:
                best_impurity_decrease = impurity_decrease
                best_split = split

    return best_split
