"""Synthetic dataset generators for controlled experiments.

Provides synthetic datasets with known structure for validating
rotation-based methods. These serve as control experiments
where the ground truth rotation is known.
"""


import numpy as np
from sklearn.datasets import make_blobs, make_classification

from endgame.benchmark.loader import DatasetInfo, TaskType


def _rotation_matrix_2d(angle_degrees: float) -> np.ndarray:
    """Create 2D rotation matrix for given angle.

    Parameters
    ----------
    angle_degrees : float
        Rotation angle in degrees.

    Returns
    -------
    ndarray of shape (2, 2)
        2D rotation matrix.
    """
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _random_orthogonal_matrix(
    n: int,
    random_state: int | None = None
) -> np.ndarray:
    """Generate random orthogonal matrix via QR decomposition.

    Parameters
    ----------
    n : int
        Matrix dimension.
    random_state : int, optional
        Random seed.

    Returns
    -------
    ndarray of shape (n, n)
        Random orthogonal matrix.
    """
    rng = np.random.RandomState(random_state)
    A = rng.randn(n, n)
    Q, _ = np.linalg.qr(A)
    return Q


def _block_rotation_matrix(
    n: int,
    block_size: int = 2,
    angle_degrees: float = 45.0,
) -> np.ndarray:
    """Create block-diagonal rotation matrix.

    Applies 2D rotations to consecutive pairs of features.

    Parameters
    ----------
    n : int
        Total dimension.
    block_size : int, default=2
        Size of each rotation block.
    angle_degrees : float, default=45.0
        Rotation angle for each block.

    Returns
    -------
    ndarray of shape (n, n)
        Block-diagonal rotation matrix.
    """
    W = np.eye(n)
    R = _rotation_matrix_2d(angle_degrees)

    for i in range(0, n - 1, block_size):
        end = min(i + block_size, n)
        if end - i == 2:
            W[i:end, i:end] = R

    return W


def make_rotated_blobs(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    rotation_angle: float = 45.0,
    cluster_std: float = 1.0,
    noise: float = 0.0,
    random_state: int | None = None,
) -> DatasetInfo:
    """Generate synthetic dataset with known rotation.

    Creates Gaussian blobs that are axis-aligned in a rotated coordinate
    system. rotation learning should be able to recover the rotation and achieve
    high accuracy by axis-aligned splits in the rotated space.

    This is the critical control experiment from the paper. Standard
    GBDTs fail on this because the decision boundaries are diagonal,
    while rotation learning should match MLP performance by learning the rotation.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=10
        Number of features.
    n_classes : int, default=3
        Number of classes (blob centers).
    rotation_angle : float, default=45.0
        Rotation angle in degrees applied pairwise to features.
    cluster_std : float, default=1.0
        Standard deviation of clusters before rotation.
    noise : float, default=0.0
        Additional Gaussian noise after rotation.
    random_state : int, optional
        Random seed.

    Returns
    -------
    DatasetInfo
        Synthetic dataset with metadata including ground truth rotation.

    Examples
    --------
    >>> from endgame.benchmark.synthetic import make_rotated_blobs
    >>> dataset = make_rotated_blobs(n_samples=500, rotation_angle=45.0)
    >>> print(dataset.name)
    synthetic_rotated_45
    >>> print(dataset.metadata['ground_truth_rotation'].shape)
    (10, 10)
    """
    rng = np.random.RandomState(random_state)

    # Generate axis-aligned blobs
    # Centers are along coordinate axes for easy axis-aligned separation
    centers = np.zeros((n_classes, n_features))
    for i in range(n_classes):
        # Place centers along first few dimensions
        dim = i % n_features
        centers[i, dim] = (i + 1) * 3.0  # Spread out centers

    X, y = make_blobs(
        n_samples=n_samples,
        n_features=n_features,
        centers=centers,
        cluster_std=cluster_std,
        random_state=random_state,
    )

    # Create and apply rotation matrix
    W = _block_rotation_matrix(n_features, block_size=2, angle_degrees=rotation_angle)
    X_rotated = X @ W

    # Add noise if specified
    if noise > 0:
        X_rotated += rng.randn(*X_rotated.shape) * noise

    # The inverse rotation is what rotation learning should learn
    W_inverse = W.T  # Orthogonal matrix: inverse = transpose

    return DatasetInfo(
        name=f"synthetic_rotated_{int(rotation_angle)}",
        task_type=TaskType.CLASSIFICATION,
        X=X_rotated.astype(np.float32),
        y=y,
        feature_names=[f"x{i}" for i in range(n_features)],
        categorical_indicator=[False] * n_features,
        source="synthetic",
        metadata={
            "ground_truth_rotation": W_inverse,
            "rotation_angle": rotation_angle,
            "cluster_std": cluster_std,
            "noise": noise,
            "description": (
                "Gaussian blobs rotated by a known angle. "
                "rotation learning should recover the rotation for axis-aligned separation."
            ),
        },
    )


def make_hidden_structure(
    n_samples: int = 1000,
    n_features: int = 20,
    n_informative: int = 5,
    structure_type: str = "diagonal",
    flip_y: float = 0.01,
    random_state: int | None = None,
) -> DatasetInfo:
    """Generate dataset with hidden linear structure.

    The true decision boundary is simple (axis-aligned) in a rotated
    coordinate system. This tests whether rotation learning can discover the
    useful feature combinations.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=20
        Total number of features.
    n_informative : int, default=5
        Number of truly informative features.
    structure_type : str, default='diagonal'
        Type of hidden structure:
        - 'diagonal': Linear combination of pairs
        - 'block': Block structure in feature space
        - 'random': Random orthogonal transformation
    flip_y : float, default=0.01
        Fraction of labels to flip (noise).
    random_state : int, optional
        Random seed.

    Returns
    -------
    DatasetInfo
        Synthetic dataset with hidden structure.
    """
    rng = np.random.RandomState(random_state)

    # Generate base classification problem
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_informative,
        n_informative=n_informative,
        n_redundant=0,
        n_clusters_per_class=2,
        flip_y=flip_y,
        random_state=random_state,
    )

    # Embed in higher dimension with noise features
    X_full = np.zeros((n_samples, n_features))
    X_full[:, :n_informative] = X
    X_full[:, n_informative:] = rng.randn(n_samples, n_features - n_informative) * 0.1

    # Apply rotation based on structure type
    if structure_type == "diagonal":
        W = _block_rotation_matrix(n_features, block_size=2, angle_degrees=45.0)
    elif structure_type == "block":
        W = _block_rotation_matrix(n_features, block_size=4, angle_degrees=30.0)
    elif structure_type == "random":
        W = _random_orthogonal_matrix(n_features, random_state=random_state)
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    X_transformed = X_full @ W
    W_inverse = W.T

    return DatasetInfo(
        name=f"synthetic_hidden_{structure_type}",
        task_type=TaskType.CLASSIFICATION,
        X=X_transformed.astype(np.float32),
        y=y,
        feature_names=[f"x{i}" for i in range(n_features)],
        categorical_indicator=[False] * n_features,
        source="synthetic",
        metadata={
            "ground_truth_rotation": W_inverse,
            "structure_type": structure_type,
            "n_informative": n_informative,
            "description": (
                f"Hidden {structure_type} structure. "
                f"Only {n_informative}/{n_features} features are informative "
                "in the rotated space."
            ),
        },
    )


def make_xor_rotated(
    n_samples: int = 1000,
    n_features: int = 10,
    rotation_angle: float = 45.0,
    noise: float = 0.1,
    random_state: int | None = None,
) -> DatasetInfo:
    """Generate XOR problem in rotated space.

    Classic XOR problem where the decision boundary is the product
    of two features, but rotated so that axis-aligned trees fail.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=10
        Total features (XOR uses first 2, rest are noise).
    rotation_angle : float, default=45.0
        Rotation angle for XOR features.
    noise : float, default=0.1
        Gaussian noise level.
    random_state : int, optional
        Random seed.

    Returns
    -------
    DatasetInfo
        XOR dataset with rotation.
    """
    rng = np.random.RandomState(random_state)

    # Generate base features
    X = rng.randn(n_samples, n_features).astype(np.float32)

    # XOR labels based on sign of product of first two features
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)

    # Apply rotation to first two features
    R = _rotation_matrix_2d(rotation_angle)
    X[:, :2] = X[:, :2] @ R

    # Add noise
    X += rng.randn(n_samples, n_features).astype(np.float32) * noise

    return DatasetInfo(
        name=f"synthetic_xor_{int(rotation_angle)}",
        task_type=TaskType.CLASSIFICATION,
        X=X,
        y=y,
        feature_names=[f"x{i}" for i in range(n_features)],
        categorical_indicator=[False] * n_features,
        source="synthetic",
        metadata={
            "rotation_angle": rotation_angle,
            "noise": noise,
            "description": (
                "Rotated XOR problem. Decision boundary is non-linear "
                "but becomes easier with proper rotation."
            ),
        },
    )


def make_regression_rotated(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    rotation_angle: float = 45.0,
    noise: float = 0.1,
    random_state: int | None = None,
) -> DatasetInfo:
    """Generate regression dataset with rotated structure.

    Linear regression problem where the true coefficients are
    axis-aligned in a rotated space.

    Parameters
    ----------
    n_samples : int, default=1000
        Number of samples.
    n_features : int, default=10
        Total features.
    n_informative : int, default=5
        Number of features with non-zero coefficients.
    rotation_angle : float, default=45.0
        Rotation angle.
    noise : float, default=0.1
        Target noise level.
    random_state : int, optional
        Random seed.

    Returns
    -------
    DatasetInfo
        Regression dataset.
    """
    rng = np.random.RandomState(random_state)

    # Generate features
    X = rng.randn(n_samples, n_features).astype(np.float32)

    # True coefficients (sparse in original space)
    coef = np.zeros(n_features)
    coef[:n_informative] = rng.randn(n_informative)

    # Target before rotation
    y_clean = X @ coef

    # Apply rotation to features
    W = _block_rotation_matrix(n_features, block_size=2, angle_degrees=rotation_angle)
    X_rotated = X @ W

    # Add noise to target
    y = y_clean + rng.randn(n_samples) * noise * np.std(y_clean)

    return DatasetInfo(
        name=f"synthetic_regression_{int(rotation_angle)}",
        task_type=TaskType.REGRESSION,
        X=X_rotated.astype(np.float32),
        y=y.astype(np.float32),
        feature_names=[f"x{i}" for i in range(n_features)],
        categorical_indicator=[False] * n_features,
        source="synthetic",
        metadata={
            "ground_truth_rotation": W.T,
            "true_coefficients": coef,
            "rotation_angle": rotation_angle,
            "noise": noise,
            "description": (
                "Linear regression with rotated features. "
                f"True coefficients are sparse ({n_informative}/{n_features} non-zero) "
                "in the original (pre-rotation) space."
            ),
        },
    )


def get_synthetic_suite(random_state: int = 42) -> dict[str, DatasetInfo]:
    """Get dictionary of all synthetic datasets for benchmarking.

    Returns a comprehensive suite of synthetic datasets designed to
    test rotation learning methods.

    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    Dict[str, DatasetInfo]
        Dictionary mapping dataset names to DatasetInfo objects.

    Examples
    --------
    >>> from endgame.benchmark.synthetic import get_synthetic_suite
    >>> suite = get_synthetic_suite()
    >>> for name, dataset in suite.items():
    ...     print(f"{name}: {dataset.n_samples} samples, {dataset.n_features} features")
    """
    return {
        # Rotated blobs at different angles
        "rotated_15": make_rotated_blobs(
            n_samples=2000, n_features=20, rotation_angle=15.0,
            random_state=random_state
        ),
        "rotated_45": make_rotated_blobs(
            n_samples=2000, n_features=20, rotation_angle=45.0,
            random_state=random_state
        ),
        "rotated_60": make_rotated_blobs(
            n_samples=2000, n_features=20, rotation_angle=60.0,
            random_state=random_state
        ),
        "rotated_90": make_rotated_blobs(
            n_samples=2000, n_features=20, rotation_angle=90.0,
            random_state=random_state
        ),

        # Large rotated blobs (paper's main control experiment)
        "rotated_control": make_rotated_blobs(
            n_samples=50000, n_features=20, n_classes=5,
            rotation_angle=45.0, cluster_std=1.5,
            random_state=random_state
        ),

        # Hidden structure datasets
        "hidden_diagonal": make_hidden_structure(
            n_samples=5000, n_features=30, n_informative=10,
            structure_type="diagonal", random_state=random_state
        ),
        "hidden_block": make_hidden_structure(
            n_samples=5000, n_features=30, n_informative=10,
            structure_type="block", random_state=random_state
        ),
        "hidden_random": make_hidden_structure(
            n_samples=5000, n_features=30, n_informative=10,
            structure_type="random", random_state=random_state
        ),

        # XOR problems
        "xor_45": make_xor_rotated(
            n_samples=2000, n_features=10, rotation_angle=45.0,
            random_state=random_state
        ),

        # Regression
        "regression_rotated": make_regression_rotated(
            n_samples=5000, n_features=20, n_informative=8,
            rotation_angle=45.0, random_state=random_state
        ),
    }


def get_control_dataset(random_state: int = 42) -> DatasetInfo:
    """Get the primary control dataset from the paper.

    This is the Synthetic Rotated dataset used as the critical control
    experiment. Standard GBDTs should fail here while rotation learning should
    recover the rotation and match MLP performance.

    Parameters
    ----------
    random_state : int, default=42
        Random seed.

    Returns
    -------
    DatasetInfo
        The control dataset.
    """
    return make_rotated_blobs(
        n_samples=50000,
        n_features=20,
        n_classes=5,
        rotation_angle=45.0,
        cluster_std=1.5,
        random_state=random_state,
    )
