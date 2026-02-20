"""Dataset loading utilities for benchmark suites.

Provides unified access to benchmark datasets from OpenML, sklearn, and custom sources.
"""

import warnings
from collections.abc import Generator
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import LabelEncoder

# Lazy imports for optional dependencies
HAS_OPENML = False
HAS_PANDAS = False

try:
    import openml
    HAS_OPENML = True
except ImportError:
    pass

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pass


class TaskType(str, Enum):
    """Type of machine learning task."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"


@dataclass
class DatasetInfo:
    """Container for dataset information and data.

    Attributes
    ----------
    name : str
        Name of the dataset.
    task_type : TaskType
        Type of ML task.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target variable.
    feature_names : List[str]
        Names of features.
    categorical_indicator : List[bool]
        Boolean mask indicating categorical features.
    n_samples : int
        Number of samples.
    n_features : int
        Number of features.
    n_classes : int
        Number of classes (for classification).
    class_distribution : Dict[Any, int]
        Distribution of classes.
    source : str
        Source of the dataset (e.g., 'openml', 'sklearn').
    openml_id : Optional[int]
        OpenML dataset ID if applicable.
    cv_splits : Optional[List[Tuple[np.ndarray, np.ndarray]]]
        Predefined cross-validation splits.
    metadata : Dict[str, Any]
        Additional metadata.
    """
    name: str
    task_type: TaskType
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str] = field(default_factory=list)
    categorical_indicator: list[bool] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0
    n_classes: int = 0
    class_distribution: dict[Any, int] = field(default_factory=dict)
    source: str = "unknown"
    openml_id: int | None = None
    cv_splits: list[tuple[np.ndarray, np.ndarray]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute derived attributes."""
        self.n_samples = self.X.shape[0]
        self.n_features = self.X.shape[1] if self.X.ndim > 1 else 1

        if not self.feature_names:
            self.feature_names = [f"f{i}" for i in range(self.n_features)]

        if not self.categorical_indicator:
            self.categorical_indicator = [False] * self.n_features

        if self.task_type in (TaskType.CLASSIFICATION, TaskType.MULTICLASS):
            unique, counts = np.unique(self.y, return_counts=True)
            self.n_classes = len(unique)
            self.class_distribution = dict(zip(unique.tolist(), counts.tolist()))

    @property
    def n_categorical(self) -> int:
        """Number of categorical features."""
        return sum(self.categorical_indicator)

    @property
    def n_numerical(self) -> int:
        """Number of numerical features."""
        return self.n_features - self.n_categorical

    @property
    def imbalance_ratio(self) -> float:
        """Class imbalance ratio (max_count / min_count)."""
        if self.task_type == TaskType.REGRESSION:
            return 1.0
        if not self.class_distribution:
            return 1.0
        counts = list(self.class_distribution.values())
        return max(counts) / min(counts) if min(counts) > 0 else float('inf')

    def get_cv_splits(
        self,
        n_splits: int = 10,
        shuffle: bool = True,
        random_state: int = 42,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Get cross-validation splits.

        Returns predefined splits if available, otherwise generates new ones.
        """
        if self.cv_splits is not None:
            return self.cv_splits

        if self.task_type in (TaskType.CLASSIFICATION, TaskType.MULTICLASS):
            cv = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        else:
            cv = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        return list(cv.split(self.X, self.y))


# Predefined benchmark suites
BUILTIN_SUITES: dict[str, dict[str, Any]] = {
    # OpenML suites
    "OpenML-CC18": {
        "type": "openml",
        "suite_id": 99,
        "description": "OpenML-CC18 classification benchmark (72 datasets)",
    },
    "OpenML-CTR23": {
        "type": "openml",
        "suite_id": 336,
        "description": "OpenML-CTR23 regression benchmark",
    },
    "AutoML-Benchmark": {
        "type": "openml",
        "suite_id": 271,
        "description": "AutoML benchmark suite",
    },
    # Sklearn built-in datasets
    "sklearn-classic": {
        "type": "sklearn",
        "datasets": ["iris", "wine", "breast_cancer", "digits", "diabetes"],
        "description": "Classic sklearn toy datasets",
    },
    "sklearn-classification": {
        "type": "sklearn",
        "datasets": ["iris", "wine", "breast_cancer", "digits"],
        "description": "Sklearn classification datasets",
    },
    "sklearn-regression": {
        "type": "sklearn",
        "datasets": ["diabetes", "california_housing"],
        "description": "Sklearn regression datasets",
    },
    # Popular UCI datasets via OpenML
    "uci-popular": {
        "type": "openml",
        "task_ids": [
            3,      # kr-vs-kp
            6,      # letter
            11,     # balance-scale
            12,     # mfeat-factors
            14,     # mfeat-fourier
            16,     # mfeat-karhunen
            18,     # mfeat-morphological
            22,     # mfeat-zernike
            28,     # optdigits
            32,     # pendigits
            37,     # diabetes
            44,     # spambase
            54,     # vehicle
            182,    # satimage
            188,    # eucalyptus
            1461,   # bank-marketing
            1464,   # blood-transfusion
            1480,   # ilpd
            1494,   # qsar-biodeg
            1510,   # wdbc
        ],
        "description": "Popular UCI ML Repository datasets",
    },
    # Small datasets for quick testing
    "quick-test": {
        "type": "sklearn",
        "datasets": ["iris", "wine", "breast_cancer"],
        "description": "Quick test suite with small datasets",
    },
    # Grinsztajn benchmark (NeurIPS 2022) — full 45 datasets across 4 suites
    # Reference: "Why do tree-based models still outperform deep learning
    # on typical tabular data?" (Grinsztajn et al., NeurIPS 2022)
    "grinsztajn": {
        "type": "openml",
        "suite_ids": [
            337,  # Classification on numerical features
            334,  # Classification on numerical + categorical features
            336,  # Regression on numerical features
            335,  # Regression on numerical + categorical features
        ],
        "description": "Grinsztajn et al. NeurIPS 2022 benchmark (~45 datasets, classification + regression)",
    },
    # Grinsztajn classification-only subset
    "grinsztajn-classif": {
        "type": "openml",
        "suite_ids": [
            337,  # Classification on numerical features
            334,  # Classification on numerical + categorical features
        ],
        "description": "Grinsztajn NeurIPS 2022 classification datasets only",
    },
    # Grinsztajn regression-only subset
    "grinsztajn-regression": {
        "type": "openml",
        "suite_ids": [
            336,  # Regression on numerical features
            335,  # Regression on numerical + categorical features
        ],
        "description": "Grinsztajn NeurIPS 2022 regression datasets only",
    },
}


class SuiteLoader:
    """Load benchmark datasets from various sources.

    Supports OpenML benchmark suites, sklearn built-in datasets, and custom datasets.
    Provides standardized interface for benchmark experiments.

    Parameters
    ----------
    suite : str or List[int]
        Suite name (e.g., "OpenML-CC18") or list of OpenML task IDs.
    max_datasets : int, optional
        Maximum number of datasets to load.
    max_samples : int, optional
        Maximum samples per dataset (larger datasets are sampled).
    max_features : int, optional
        Maximum features per dataset.
    cache_dir : str, optional
        Directory for caching downloaded datasets.
    random_state : int, default=42
        Random seed for sampling.
    verbose : bool, default=True
        Enable verbose output.

    Examples
    --------
    >>> loader = SuiteLoader(suite="sklearn-classic")
    >>> for dataset in loader.load():
    ...     print(f"{dataset.name}: {dataset.n_samples} samples, {dataset.n_features} features")

    >>> loader = SuiteLoader(suite="OpenML-CC18", max_datasets=10)
    >>> datasets = list(loader.load())
    """

    def __init__(
        self,
        suite: str | list[int] = "sklearn-classic",
        max_datasets: int | None = None,
        max_samples: int | None = None,
        max_features: int | None = None,
        cache_dir: str | None = None,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.suite = suite
        self.max_datasets = max_datasets
        self.max_samples = max_samples
        self.max_features = max_features
        self.cache_dir = cache_dir
        self.random_state = random_state
        self.verbose = verbose

        self._rng = np.random.RandomState(random_state)

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[SuiteLoader] {message}")

    def load(self) -> Generator[DatasetInfo, None, None]:
        """Load datasets from the suite.

        Yields
        ------
        DatasetInfo
            Dataset information and data.
        """
        if isinstance(self.suite, list):
            # List of OpenML task IDs
            yield from self._load_openml_tasks(self.suite)
        elif self.suite in BUILTIN_SUITES:
            suite_config = BUILTIN_SUITES[self.suite]
            if suite_config["type"] == "openml":
                if "suite_ids" in suite_config:
                    for sid in suite_config["suite_ids"]:
                        yield from self._load_openml_suite(sid)
                elif "suite_id" in suite_config:
                    yield from self._load_openml_suite(suite_config["suite_id"])
                elif "task_ids" in suite_config:
                    yield from self._load_openml_tasks(suite_config["task_ids"])
            elif suite_config["type"] == "sklearn":
                yield from self._load_sklearn_datasets(suite_config["datasets"])
            elif suite_config["type"] == "mixed":
                # Mixed suite with both OpenML and sklearn datasets
                if "openml_task_ids" in suite_config:
                    yield from self._load_openml_tasks(suite_config["openml_task_ids"])
                if "sklearn_datasets" in suite_config:
                    yield from self._load_sklearn_datasets(suite_config["sklearn_datasets"])
        else:
            raise ValueError(
                f"Unknown suite: {self.suite}. "
                f"Available: {list(BUILTIN_SUITES.keys())} or list of OpenML task IDs"
            )

    def _load_openml_suite(self, suite_id: int) -> Generator[DatasetInfo, None, None]:
        """Load datasets from an OpenML benchmark suite."""
        if not HAS_OPENML:
            raise ImportError(
                "openml is required for OpenML suites. "
                "Install with: pip install openml"
            )

        self._log(f"Loading OpenML suite {suite_id}...")

        import time as _time
        for _attempt in range(5):
            try:
                suite = openml.study.get_suite(suite_id)
                break
            except Exception as e:
                if _attempt < 4 and "connection" in str(e).lower():
                    wait = 10 * (2 ** _attempt)
                    self._log(f"OpenML server error, retrying in {wait}s... ({e})")
                    _time.sleep(wait)
                else:
                    raise
        task_ids = suite.tasks

        if self.max_datasets:
            task_ids = task_ids[:self.max_datasets]

        self._log(f"Found {len(task_ids)} tasks")

        yield from self._load_openml_tasks(task_ids)

    def _load_openml_tasks(self, task_ids: list[int]) -> Generator[DatasetInfo, None, None]:
        """Load datasets from OpenML task IDs."""
        if not HAS_OPENML:
            raise ImportError(
                "openml is required for OpenML datasets. "
                "Install with: pip install openml"
            )

        if self.max_datasets:
            task_ids = task_ids[:self.max_datasets]

        import time as _time
        for i, task_id in enumerate(task_ids):
            self._log(f"Loading task {task_id} ({i+1}/{len(task_ids)})...")
            dataset_info = None
            for _attempt in range(3):
                try:
                    dataset_info = self._load_openml_task(task_id)
                    break
                except Exception as e:
                    if _attempt < 2:
                        wait = 10 * (2 ** _attempt)
                        self._log(f"Task {task_id} failed ({e}), retrying in {wait}s...")
                        _time.sleep(wait)
                    else:
                        self._log(f"Failed to load task {task_id} after 3 attempts: {e}")
            if dataset_info is not None:
                yield dataset_info

    def _load_openml_task(self, task_id: int) -> DatasetInfo | None:
        """Load a single OpenML task."""
        try:
            task = openml.tasks.get_task(task_id)
            dataset = task.get_dataset()

            # Get data
            X, y, categorical_indicator, feature_names = dataset.get_data(
                target=dataset.default_target_attribute,
                dataset_format="array",
            )

            # Handle missing values
            if hasattr(X, 'toarray'):
                X = X.toarray()
            X = np.asarray(X, dtype=np.float64)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

            # Encode target if needed
            y = np.asarray(y)
            if y.dtype == object or (hasattr(y, 'dtype') and y.dtype.kind in ['U', 'S', 'O']):
                le = LabelEncoder()
                y = le.fit_transform(y.astype(str))
            y = np.nan_to_num(y, nan=0.0)

            # Determine task type (compare .value — OpenML enum != int)
            type_id = task.task_type_id
            if hasattr(type_id, "value"):
                type_id = type_id.value
            if type_id == 1:  # Classification
                n_classes = len(np.unique(y))
                task_type = TaskType.MULTICLASS if n_classes > 2 else TaskType.CLASSIFICATION
            elif type_id == 2:  # Regression
                task_type = TaskType.REGRESSION
            else:
                task_type = TaskType.CLASSIFICATION

            # Apply size limits
            X, y = self._apply_size_limits(X, y)

            # Handle categorical indicator
            if categorical_indicator is None:
                categorical_indicator = [False] * X.shape[1]
            else:
                categorical_indicator = list(categorical_indicator)[:X.shape[1]]

            # Handle feature names
            if feature_names is None:
                feature_names = [f"f{i}" for i in range(X.shape[1])]
            else:
                feature_names = list(feature_names)[:X.shape[1]]

            # Try to get CV splits
            cv_splits = None
            try:
                splits = task.get_split()
                cv_splits = []
                for fold in range(splits.get_maximum_folds()):
                    train_idx, test_idx = splits.get(fold=fold)
                    # Filter indices to valid range
                    train_idx = train_idx[train_idx < len(y)]
                    test_idx = test_idx[test_idx < len(y)]
                    if len(train_idx) > 0 and len(test_idx) > 0:
                        cv_splits.append((train_idx, test_idx))
            except Exception:
                pass  # Use default CV splits

            return DatasetInfo(
                name=dataset.name,
                task_type=task_type,
                X=X,
                y=y,
                feature_names=feature_names,
                categorical_indicator=categorical_indicator,
                source="openml",
                openml_id=dataset.dataset_id,
                cv_splits=cv_splits if cv_splits else None,
                metadata={
                    "task_id": task_id,
                    "openml_url": f"https://www.openml.org/d/{dataset.dataset_id}",
                },
            )

        except Exception as e:
            warnings.warn(f"Failed to load OpenML task {task_id}: {e}")
            return None

    def _load_sklearn_datasets(
        self,
        dataset_names: list[str],
    ) -> Generator[DatasetInfo, None, None]:
        """Load sklearn built-in datasets."""
        from sklearn import datasets

        sklearn_loaders = {
            "iris": (datasets.load_iris, TaskType.MULTICLASS),
            "wine": (datasets.load_wine, TaskType.MULTICLASS),
            "breast_cancer": (datasets.load_breast_cancer, TaskType.CLASSIFICATION),
            "digits": (datasets.load_digits, TaskType.MULTICLASS),
            "diabetes": (datasets.load_diabetes, TaskType.REGRESSION),
            "california_housing": (datasets.fetch_california_housing, TaskType.REGRESSION),
            "covtype": (datasets.fetch_covtype, TaskType.MULTICLASS),
        }

        if self.max_datasets:
            dataset_names = dataset_names[:self.max_datasets]

        for name in dataset_names:
            if name not in sklearn_loaders:
                self._log(f"Unknown sklearn dataset: {name}")
                continue

            try:
                self._log(f"Loading sklearn dataset: {name}")
                loader, task_type = sklearn_loaders[name]
                data = loader()

                X = np.asarray(data.data, dtype=np.float64)
                y = np.asarray(data.target)

                # Apply size limits
                X, y = self._apply_size_limits(X, y)

                feature_names = list(data.feature_names) if hasattr(data, 'feature_names') else None

                yield DatasetInfo(
                    name=name,
                    task_type=task_type,
                    X=X,
                    y=y,
                    feature_names=feature_names or [f"f{i}" for i in range(X.shape[1])],
                    categorical_indicator=[False] * X.shape[1],
                    source="sklearn",
                    metadata={"sklearn_name": name},
                )

            except Exception as e:
                self._log(f"Failed to load {name}: {e}")
                continue

    def _apply_size_limits(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply sample and feature limits."""
        # Sample limit
        if self.max_samples and X.shape[0] > self.max_samples:
            idx = self._rng.choice(X.shape[0], self.max_samples, replace=False)
            X = X[idx]
            y = y[idx]

        # Feature limit
        if self.max_features and X.shape[1] > self.max_features:
            # Use variance-based selection
            variances = np.var(X, axis=0)
            top_features = np.argsort(variances)[-self.max_features:]
            X = X[:, top_features]

        return X, y

    @staticmethod
    def list_suites() -> dict[str, str]:
        """List available benchmark suites."""
        return {name: config["description"] for name, config in BUILTIN_SUITES.items()}

    @staticmethod
    def get_suite_info(suite_name: str) -> dict[str, Any]:
        """Get detailed information about a suite."""
        if suite_name not in BUILTIN_SUITES:
            raise ValueError(f"Unknown suite: {suite_name}")
        return BUILTIN_SUITES[suite_name].copy()
