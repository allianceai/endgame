from __future__ import annotations

"""Dataset meta-feature extraction for meta-learning.

Extracts comprehensive meta-features that characterize datasets for predicting
optimal model/pipeline choices.
"""

import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Lazy imports for optional dependencies
HAS_PYMFE = False

try:
    from pymfe.mfe import MFE
    HAS_PYMFE = True
except ImportError:
    pass


class MetaFeatureGroup(str, Enum):
    """Groups of meta-features."""
    SIMPLE = "simple"           # Basic dataset properties
    STATISTICAL = "statistical" # Statistical properties
    INFO_THEORY = "info-theory" # Information-theoretic measures
    LANDMARKING = "landmarking" # Simple model performance
    MODEL_BASED = "model-based" # Tree-based meta-features
    COMPLEXITY = "complexity"   # Data complexity measures


# Meta-feature definitions
SIMPLE_META_FEATURES: list[str] = [
    "nr_inst",          # Number of instances
    "nr_attr",          # Number of attributes
    "nr_num",           # Number of numeric attributes
    "nr_cat",           # Number of categorical attributes
    "nr_class",         # Number of classes
    "nr_missing",       # Number of missing values
    "pct_missing",      # Percentage of missing values
    "inst_to_attr",     # Instance to attribute ratio
    "cat_to_num",       # Categorical to numeric ratio
    "class_imbalance",  # Class imbalance ratio
    "dimensionality",   # n_features / n_samples
]

STATISTICAL_META_FEATURES: list[str] = [
    "mean_mean",        # Mean of feature means
    "mean_std",         # Mean of feature std devs
    "mean_skewness",    # Mean skewness across features
    "mean_kurtosis",    # Mean kurtosis across features
    "mean_cor",         # Mean absolute correlation
    "max_cor",          # Maximum absolute correlation
    "eigenvalue_ratio", # Ratio of top eigenvalues (PCA)
    "outlier_ratio",    # Ratio of outliers (IQR method)
]

INFO_THEORY_META_FEATURES: list[str] = [
    "class_entropy",    # Entropy of class distribution
    "attr_entropy",     # Mean entropy of attributes
    "joint_entropy",    # Mean joint entropy (attr, class)
    "mutual_info",      # Mean mutual information
    "noise_signal",     # Noise-to-signal ratio
]

LANDMARKING_META_FEATURES: list[str] = [
    "lm_1nn",           # 1-Nearest Neighbor accuracy
    "lm_nb",            # Naive Bayes accuracy
    "lm_dt_stump",      # Decision stump accuracy
    "lm_linear",        # Linear model accuracy
    "lm_random",        # Random baseline
]


@dataclass
class MetaFeatureSet:
    """Container for extracted meta-features.

    Attributes
    ----------
    features : Dict[str, float]
        Dictionary of meta-feature name to value.
    groups : Dict[str, List[str]]
        Mapping from group name to feature names in that group.
    extraction_time : float
        Time taken to extract features (seconds).
    errors : List[str]
        Any errors encountered during extraction.
    """
    features: dict[str, float] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    extraction_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return self.features.copy()

    def to_array(self, feature_names: list[str] | None = None) -> np.ndarray:
        """Convert to numpy array.

        Parameters
        ----------
        feature_names : List[str], optional
            Specific features to include (in order).
            If None, uses all features in sorted order.
        """
        if feature_names is None:
            feature_names = sorted(self.features.keys())
        return np.array([self.features.get(f, np.nan) for f in feature_names])

    def get_group(self, group: str) -> dict[str, float]:
        """Get features from a specific group."""
        if group not in self.groups:
            return {}
        return {f: self.features[f] for f in self.groups[group] if f in self.features}


class MetaProfiler:
    """Extract meta-features from datasets for meta-learning.

    Uses pymfe when available, with fallback to pure numpy/sklearn implementations.

    Parameters
    ----------
    groups : List[str], optional
        Meta-feature groups to extract. Default: ["simple", "statistical", "info-theory"].
        Options: "simple", "statistical", "info-theory", "landmarking", "complexity".
    use_pymfe : bool, default=True
        Use pymfe library when available (more comprehensive features).
    landmarking_cv : int, default=3
        Number of CV folds for landmarking meta-features.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable verbose output.

    Examples
    --------
    >>> profiler = MetaProfiler(groups=["simple", "statistical"])
    >>> meta_features = profiler.profile(X, y)
    >>> print(meta_features.features)

    >>> # With landmarking
    >>> profiler = MetaProfiler(groups=["simple", "landmarking"])
    >>> meta_features = profiler.profile(X, y)
    """

    def __init__(
        self,
        groups: list[str] | None = None,
        use_pymfe: bool = True,
        landmarking_cv: int = 3,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.groups = groups or ["simple", "statistical", "info-theory"]
        self.use_pymfe = use_pymfe and HAS_PYMFE
        self.landmarking_cv = landmarking_cv
        self.random_state = random_state
        self.verbose = verbose

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[MetaProfiler] {message}")

    def profile(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indicator: list[bool] | None = None,
        task_type: str = "classification",
    ) -> MetaFeatureSet:
        """Extract meta-features from a dataset.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features).
        y : np.ndarray
            Target variable of shape (n_samples,).
        categorical_indicator : List[bool], optional
            Boolean mask indicating categorical features.
        task_type : str, default="classification"
            Type of task: "classification" or "regression".

        Returns
        -------
        MetaFeatureSet
            Extracted meta-features.
        """
        import time
        start_time = time.time()

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        if categorical_indicator is None:
            categorical_indicator = [False] * X.shape[1]

        features: dict[str, float] = {}
        group_features: dict[str, list[str]] = {}
        errors: list[str] = []

        # Try pymfe first
        if self.use_pymfe:
            try:
                pymfe_features, pymfe_groups = self._extract_pymfe(
                    X, y, categorical_indicator, task_type
                )
                features.update(pymfe_features)
                for g, fs in pymfe_groups.items():
                    group_features.setdefault(g, []).extend(fs)
            except Exception as e:
                errors.append(f"pymfe extraction failed: {e}")
                self._log(f"pymfe failed, using fallback: {e}")

        # Always compute our own simple features
        simple_features = self._extract_simple(X, y, categorical_indicator, task_type)
        features.update(simple_features)
        group_features.setdefault("simple", []).extend(simple_features.keys())

        # Compute statistical features
        if "statistical" in self.groups:
            try:
                stat_features = self._extract_statistical(X)
                features.update(stat_features)
                group_features.setdefault("statistical", []).extend(stat_features.keys())
            except Exception as e:
                errors.append(f"statistical extraction failed: {e}")

        # Compute info-theory features
        if "info-theory" in self.groups and task_type == "classification":
            try:
                info_features = self._extract_info_theory(X, y)
                features.update(info_features)
                group_features.setdefault("info-theory", []).extend(info_features.keys())
            except Exception as e:
                errors.append(f"info-theory extraction failed: {e}")

        # Compute landmarking features
        if "landmarking" in self.groups:
            try:
                lm_features = self._extract_landmarking(X, y, task_type)
                features.update(lm_features)
                group_features.setdefault("landmarking", []).extend(lm_features.keys())
            except Exception as e:
                errors.append(f"landmarking extraction failed: {e}")

        return MetaFeatureSet(
            features=features,
            groups=group_features,
            extraction_time=time.time() - start_time,
            errors=errors,
        )

    def _extract_pymfe(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indicator: list[bool],
        task_type: str,
    ) -> tuple[dict[str, float], dict[str, list[str]]]:
        """Extract meta-features using pymfe."""
        # Map our groups to pymfe groups
        pymfe_groups = []
        if "simple" in self.groups:
            pymfe_groups.append("general")
        if "statistical" in self.groups:
            pymfe_groups.append("statistical")
        if "info-theory" in self.groups:
            pymfe_groups.append("info-theory")
        if "landmarking" in self.groups:
            pymfe_groups.append("landmarking")
        if "model-based" in self.groups:
            pymfe_groups.append("model-based")
        if "complexity" in self.groups:
            pymfe_groups.append("complexity")

        if not pymfe_groups:
            return {}, {}

        mfe = MFE(groups=pymfe_groups, random_state=self.random_state)

        # Fit with categorical indicator
        cat_cols = [i for i, c in enumerate(categorical_indicator) if c]
        if cat_cols:
            mfe.fit(X, y, cat_cols=cat_cols)
        else:
            mfe.fit(X, y)

        # Extract
        names, values = mfe.extract()

        features = {}
        group_map = {}

        for name, value in zip(names, values):
            if value is not None and np.isfinite(value):
                features[name] = float(value)
                # Infer group from feature name
                for group in pymfe_groups:
                    group_map.setdefault(group, []).append(name)

        return features, group_map

    def _extract_simple(
        self,
        X: np.ndarray,
        y: np.ndarray,
        categorical_indicator: list[bool],
        task_type: str,
    ) -> dict[str, float]:
        """Extract simple meta-features."""
        n_samples, n_features = X.shape
        n_cat = sum(categorical_indicator)
        n_num = n_features - n_cat

        features = {
            "nr_inst": float(n_samples),
            "nr_attr": float(n_features),
            "nr_num": float(n_num),
            "nr_cat": float(n_cat),
            "inst_to_attr": n_samples / max(n_features, 1),
            "dimensionality": n_features / max(n_samples, 1),
        }

        # Categorical to numeric ratio
        features["cat_to_num"] = n_cat / max(n_num, 1)

        # Missing values
        n_missing = np.sum(np.isnan(X)) if np.any(np.isnan(X)) else 0
        features["nr_missing"] = float(n_missing)
        features["pct_missing"] = n_missing / (n_samples * n_features)

        # Class-specific features
        if task_type == "classification":
            unique, counts = np.unique(y, return_counts=True)
            features["nr_class"] = float(len(unique))
            features["class_imbalance"] = float(max(counts) / max(min(counts), 1))
        else:
            features["nr_class"] = 0.0
            features["class_imbalance"] = 1.0

        return features

    def _extract_statistical(self, X: np.ndarray) -> dict[str, float]:
        """Extract statistical meta-features."""
        features = {}

        # Feature-wise statistics
        means = np.mean(X, axis=0)
        stds = np.std(X, axis=0)

        features["mean_mean"] = float(np.mean(means))
        features["mean_std"] = float(np.mean(stds))

        # Skewness and kurtosis
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            skewness = stats.skew(X, axis=0, nan_policy='omit')
            kurtosis = stats.kurtosis(X, axis=0, nan_policy='omit')

        skewness = np.nan_to_num(skewness, nan=0.0)
        kurtosis = np.nan_to_num(kurtosis, nan=0.0)

        features["mean_skewness"] = float(np.mean(np.abs(skewness)))
        features["mean_kurtosis"] = float(np.mean(np.abs(kurtosis)))

        # Correlations
        if X.shape[1] > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                corr_matrix = np.corrcoef(X.T)
                corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

                # Get upper triangle (excluding diagonal)
                upper_tri = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]

                features["mean_cor"] = float(np.mean(np.abs(upper_tri)))
                features["max_cor"] = float(np.max(np.abs(upper_tri))) if len(upper_tri) > 0 else 0.0
        else:
            features["mean_cor"] = 0.0
            features["max_cor"] = 0.0

        # PCA-based eigenvalue ratio
        try:
            # Standardize
            X_scaled = StandardScaler().fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0)

            # Compute covariance eigenvalues
            cov = np.cov(X_scaled.T)
            if cov.ndim == 0:
                cov = np.array([[cov]])
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

            if len(eigenvalues) > 0 and eigenvalues.sum() > 0:
                features["eigenvalue_ratio"] = float(eigenvalues[0] / eigenvalues.sum())
            else:
                features["eigenvalue_ratio"] = 1.0
        except Exception:
            features["eigenvalue_ratio"] = 1.0

        # Outlier ratio (IQR method)
        outlier_count = 0
        for col in range(X.shape[1]):
            q1, q3 = np.percentile(X[:, col], [25, 75])
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outlier_count += np.sum((X[:, col] < lower) | (X[:, col] > upper))

        features["outlier_ratio"] = outlier_count / (X.shape[0] * X.shape[1])

        return features

    def _extract_info_theory(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> dict[str, float]:
        """Extract information-theoretic meta-features."""
        features = {}

        # Class entropy
        unique, counts = np.unique(y, return_counts=True)
        probs = counts / counts.sum()
        class_entropy = -np.sum(probs * np.log2(probs + 1e-10))
        features["class_entropy"] = float(class_entropy)

        # Attribute entropy (discretize continuous features)
        attr_entropies = []
        n_bins = min(10, int(np.sqrt(X.shape[0])))

        for col in range(X.shape[1]):
            try:
                # Discretize
                binned = np.digitize(X[:, col], np.linspace(X[:, col].min(), X[:, col].max(), n_bins))
                unique, counts = np.unique(binned, return_counts=True)
                probs = counts / counts.sum()
                entropy = -np.sum(probs * np.log2(probs + 1e-10))
                attr_entropies.append(entropy)
            except Exception:
                continue

        features["attr_entropy"] = float(np.mean(attr_entropies)) if attr_entropies else 0.0

        # Joint entropy and mutual information
        mutual_infos = []
        joint_entropies = []

        for col in range(min(X.shape[1], 20)):  # Limit for speed
            try:
                binned = np.digitize(X[:, col], np.linspace(X[:, col].min(), X[:, col].max(), n_bins))

                # Joint distribution
                joint_counts = {}
                for b, c in zip(binned, y):
                    key = (b, c)
                    joint_counts[key] = joint_counts.get(key, 0) + 1

                joint_probs = np.array(list(joint_counts.values())) / len(y)
                joint_entropy = -np.sum(joint_probs * np.log2(joint_probs + 1e-10))
                joint_entropies.append(joint_entropy)

                # Mutual information: I(X;Y) = H(X) + H(Y) - H(X,Y)
                attr_unique, attr_counts = np.unique(binned, return_counts=True)
                attr_probs = attr_counts / attr_counts.sum()
                attr_entropy = -np.sum(attr_probs * np.log2(attr_probs + 1e-10))

                mi = attr_entropy + class_entropy - joint_entropy
                mutual_infos.append(max(0, mi))  # MI should be non-negative

            except Exception:
                continue

        features["joint_entropy"] = float(np.mean(joint_entropies)) if joint_entropies else 0.0
        features["mutual_info"] = float(np.mean(mutual_infos)) if mutual_infos else 0.0

        # Noise-to-signal ratio
        if features["mutual_info"] > 0:
            features["noise_signal"] = features["attr_entropy"] / features["mutual_info"]
        else:
            features["noise_signal"] = float('inf')

        return features

    def _extract_landmarking(
        self,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> dict[str, float]:
        """Extract landmarking meta-features (simple model performance)."""
        from sklearn.dummy import DummyClassifier, DummyRegressor
        from sklearn.linear_model import LogisticRegression, Ridge
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

        features = {}

        # Prepare data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

        # Limit size for speed
        if X.shape[0] > 5000:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(X.shape[0], 5000, replace=False)
            X_scaled = X_scaled[idx]
            y_subset = y[idx]
        else:
            y_subset = y

        scoring = "accuracy" if task_type == "classification" else "r2"

        models: list[tuple[str, Any]] = []

        if task_type == "classification":
            models = [
                ("lm_1nn", KNeighborsClassifier(n_neighbors=1)),
                ("lm_nb", GaussianNB()),
                ("lm_dt_stump", DecisionTreeClassifier(max_depth=1, random_state=self.random_state)),
                ("lm_linear", LogisticRegression(max_iter=100, random_state=self.random_state)),
                ("lm_random", DummyClassifier(strategy="stratified", random_state=self.random_state)),
            ]
        else:
            models = [
                ("lm_1nn", KNeighborsRegressor(n_neighbors=1)),
                ("lm_dt_stump", DecisionTreeRegressor(max_depth=1, random_state=self.random_state)),
                ("lm_linear", Ridge(random_state=self.random_state)),
                ("lm_random", DummyRegressor(strategy="mean")),
            ]

        for name, model in models:
            try:
                scores = cross_val_score(
                    model,
                    X_scaled,
                    y_subset,
                    cv=self.landmarking_cv,
                    scoring=scoring,
                )
                features[name] = float(np.mean(scores))
            except Exception as e:
                self._log(f"Landmarking {name} failed: {e}")
                features[name] = np.nan

        return features

    def get_feature_names(self) -> list[str]:
        """Get list of all possible meta-feature names."""
        names = []
        if "simple" in self.groups:
            names.extend(SIMPLE_META_FEATURES)
        if "statistical" in self.groups:
            names.extend(STATISTICAL_META_FEATURES)
        if "info-theory" in self.groups:
            names.extend(INFO_THEORY_META_FEATURES)
        if "landmarking" in self.groups:
            names.extend(LANDMARKING_META_FEATURES)
        return names


def profile_dataset(
    X: np.ndarray,
    y: np.ndarray,
    groups: list[str] | None = None,
    **kwargs,
) -> dict[str, float]:
    """Convenience function to profile a dataset.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target variable.
    groups : List[str], optional
        Meta-feature groups to extract.
    **kwargs
        Additional arguments passed to MetaProfiler.

    Returns
    -------
    Dict[str, float]
        Dictionary of meta-features.
    """
    profiler = MetaProfiler(groups=groups, **kwargs)
    result = profiler.profile(X, y)
    return result.to_dict()
