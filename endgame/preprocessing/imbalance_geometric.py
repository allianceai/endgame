"""Modern geometric SMOTE extensions for class imbalance handling.

This module provides advanced oversampling methods that extend SMOTE with
geometric and statistical techniques. All methods use only numpy/scipy
(no optional dependencies).

Algorithms
----------
- MultivariateGaussianSMOTE: Local Gaussian sampling around minority points
- SimplicialSMOTE: Simplicial complex-based sampling with Dirichlet coordinates
- CVSMOTEResampler: Cross-validation guided synthetic sample selection
- OverlapRegionDetector: Overlap-aware meta-method for any base sampler

References
----------
- "Do we need rebalancing strategies?" (ICLR 2025)
- Simplicial complex extension of SMOTE (KDD 2025)
- CV-informed SMOTE (ICLR 2025)
- Overlap Region Detection (AAAI 2025)
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator, clone
from sklearn.utils.validation import check_X_y


def _compute_sampling_targets(
    y: np.ndarray,
    sampling_strategy: str | float | dict = "auto",
) -> dict:
    """Compute number of synthetic samples to generate per class.

    Interprets the ``sampling_strategy`` parameter following imblearn semantics
    and returns a dict ``{class_label: n_to_generate}``.

    Parameters
    ----------
    y : ndarray of shape (n_samples,)
        Target array.
    sampling_strategy : str, float, or dict
        - ``'auto'`` / ``'minority'`` / ``'not majority'``: up-sample relevant
          classes to match the majority count.
        - ``'all'``: up-sample every class to the majority count.
        - float: desired minority-to-majority ratio (0 < r <= 1).
        - dict: ``{class_label: desired_total_count}``.

    Returns
    -------
    targets : dict
        ``{class_label: n_synthetic_to_generate}`` (values >= 0).
    """
    unique, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique, counts))
    max_count = counts.max()
    majority_class = unique[np.argmax(counts)]

    if isinstance(sampling_strategy, dict):
        targets = {}
        for cls, desired in sampling_strategy.items():
            current = class_counts.get(cls, 0)
            targets[cls] = max(0, desired - current)
        return targets

    if isinstance(sampling_strategy, (int, float)) and not isinstance(
        sampling_strategy, bool
    ):
        # Float ratio: desired = ratio * majority_count
        desired = int(round(float(sampling_strategy) * max_count))
        targets = {}
        for cls, cnt in class_counts.items():
            if cls != majority_class and cnt < desired:
                targets[cls] = desired - cnt
        return targets

    # String strategies
    if sampling_strategy in ("auto", "minority", "not majority"):
        targets = {}
        for cls, cnt in class_counts.items():
            if cls != majority_class and cnt < max_count:
                targets[cls] = max_count - cnt
        return targets

    if sampling_strategy == "all":
        targets = {}
        for cls, cnt in class_counts.items():
            if cnt < max_count:
                targets[cls] = max_count - cnt
        return targets

    raise ValueError(
        f"Unknown sampling_strategy: {sampling_strategy!r}. "
        "Expected 'auto', 'minority', 'not majority', 'all', a float, or a dict."
    )


# =============================================================================
# MultivariateGaussianSMOTE
# =============================================================================


class MultivariateGaussianSMOTE(BaseEstimator):
    """Multivariate Gaussian SMOTE oversampler.

    For each minority sample, fits a local multivariate Gaussian from its
    k-nearest minority neighbours and samples new points from it.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`_compute_sampling_targets` for semantics.
    k_neighbors : int, default=5
        Number of nearest minority neighbours for covariance estimation.
    regularization : float, default=1e-6
        Ridge added to the diagonal of local covariance matrices to
        ensure positive-definiteness.
    random_state : int or None, default=None
        Random seed.

    References
    ----------
    "Do we need rebalancing strategies?" (ICLR 2025)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        k_neighbors: int = 5,
        regularization: float = 1e-6,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.regularization = regularization
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> MultivariateGaussianSMOTE:
        """Fit the sampler (validates input and computes targets).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        X_resampled : ndarray
        y_resampled : ndarray
        """
        from scipy.spatial import KDTree

        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            n_cls = len(X_cls)
            if n_cls == 0:
                continue

            k = min(self.k_neighbors, n_cls - 1)
            if k < 1:
                # Only one sample — replicate with small noise
                noise = rng.randn(n_synthetic, X.shape[1]) * self.regularization
                synthetic_X.append(np.tile(X_cls[0], (n_synthetic, 1)) + noise)
                synthetic_y.append(np.full(n_synthetic, cls))
                continue

            tree = KDTree(X_cls)

            # Distribute samples uniformly across minority points
            samples_per_point = np.full(n_cls, n_synthetic // n_cls)
            samples_per_point[: n_synthetic % n_cls] += 1

            for i, n_gen in enumerate(samples_per_point):
                if n_gen == 0:
                    continue

                # k+1 because query includes the point itself
                _, nn_idx = tree.query(X_cls[i], k=k + 1)
                nn_idx = nn_idx[1:]  # exclude self

                neighbours = X_cls[nn_idx]
                mean = neighbours.mean(axis=0)
                cov = np.cov(neighbours, rowvar=False)
                if cov.ndim == 0:
                    cov = np.array([[cov]])
                cov += np.eye(cov.shape[0]) * self.regularization

                samples = rng.multivariate_normal(mean, cov, size=int(n_gen))
                synthetic_X.append(samples)
                synthetic_y.append(np.full(int(n_gen), cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# SimplicialSMOTE
# =============================================================================


class SimplicialSMOTE(BaseEstimator):
    """Simplicial complex SMOTE oversampler.

    Builds simplicial complexes from the k-NN graph of minority samples and
    generates new points inside simplices using Dirichlet-distributed
    barycentric coordinates.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`_compute_sampling_targets`.
    k_neighbors : int, default=5
        Number of nearest neighbours for graph construction.
    simplex_dim : int, default=2
        Dimension of the simplices to sample from (2 = triangles,
        3 = tetrahedra). Clamped to ``min(simplex_dim, k_neighbors)``.
    random_state : int or None, default=None
        Random seed.

    References
    ----------
    Simplicial complex extension of SMOTE (KDD 2025)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        k_neighbors: int = 5,
        simplex_dim: int = 2,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.simplex_dim = simplex_dim
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> SimplicialSMOTE:
        """Fit the sampler."""
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def _build_simplices(
        self,
        X_cls: np.ndarray,
        k: int,
        simplex_dim: int,
        rng: np.random.RandomState,
    ) -> list[np.ndarray]:
        """Build simplices from k-NN graph.

        Returns list of index arrays, each of length ``simplex_dim + 1``.
        """
        from scipy.spatial import KDTree

        n = len(X_cls)
        tree = KDTree(X_cls)
        _, nn_indices = tree.query(X_cls, k=min(k + 1, n))

        simplices = []
        for i in range(n):
            neighbours = nn_indices[i, 1:]  # exclude self
            if len(neighbours) < simplex_dim:
                continue
            # Form simplices from point i and simplex_dim of its neighbours
            for _ in range(max(1, len(neighbours) // simplex_dim)):
                chosen = rng.choice(neighbours, size=simplex_dim, replace=False)
                simplices.append(np.concatenate([[i], chosen]))

        return simplices

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample."""
        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            n_cls = len(X_cls)
            if n_cls == 0:
                continue

            k = min(self.k_neighbors, n_cls - 1)
            effective_dim = min(self.simplex_dim, k)

            if k < 1 or effective_dim < 1:
                # Fallback: replicate with noise
                noise = rng.randn(n_synthetic, X.shape[1]) * 1e-6
                synthetic_X.append(np.tile(X_cls[0], (n_synthetic, 1)) + noise)
                synthetic_y.append(np.full(n_synthetic, cls))
                continue

            simplices = self._build_simplices(X_cls, k, effective_dim, rng)
            if not simplices:
                continue

            generated = 0
            batch: list[np.ndarray] = []
            while generated < n_synthetic:
                simplex_idx = simplices[rng.randint(len(simplices))]
                vertices = X_cls[simplex_idx]  # (simplex_dim+1, n_features)

                # Dirichlet-distributed barycentric coordinates
                weights = rng.dirichlet(np.ones(len(simplex_idx)))
                point = weights @ vertices
                batch.append(point)
                generated += 1

            synthetic_X.append(np.array(batch))
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# CVSMOTEResampler
# =============================================================================


class CVSMOTEResampler(BaseEstimator):
    """Cross-validation guided SMOTE oversampler.

    Generates a pool of candidate synthetic samples via SMOTE-style
    interpolation, then uses cross-validation to retain only those that
    improve a scorer metric.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`_compute_sampling_targets`.
    k_neighbors : int, default=5
        Nearest neighbours for SMOTE interpolation.
    cv : int, default=3
        Number of cross-validation folds for candidate evaluation.
    estimator : estimator or None, default=None
        Classifier used to score candidate batches. Defaults to
        ``LogisticRegression(max_iter=500)``.
    scoring : str, default='f1_macro'
        Scoring metric for cross-validation (sklearn convention).
    candidate_pool_factor : float, default=2.0
        Generate this many times the required synthetic samples as
        candidates, then keep the best subset.
    random_state : int or None, default=None
        Random seed.

    References
    ----------
    CV-informed SMOTE (ICLR 2025)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        k_neighbors: int = 5,
        cv: int = 3,
        estimator: Any = None,
        scoring: str = "f1_macro",
        candidate_pool_factor: float = 2.0,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.cv = cv
        self.estimator = estimator
        self.scoring = scoring
        self.candidate_pool_factor = candidate_pool_factor
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> CVSMOTEResampler:
        """Fit the sampler."""
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    @staticmethod
    def _smote_interpolate(
        X_cls: np.ndarray,
        n_synthetic: int,
        k: int,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate SMOTE-style interpolated samples."""
        from scipy.spatial import KDTree

        n_cls = len(X_cls)
        tree = KDTree(X_cls)
        _, nn_indices = tree.query(X_cls, k=min(k + 1, n_cls))

        samples = []
        for _ in range(n_synthetic):
            idx = rng.randint(n_cls)
            neighbours = nn_indices[idx, 1:]
            nn = neighbours[rng.randint(len(neighbours))]
            lam = rng.uniform()
            samples.append(X_cls[idx] + lam * (X_cls[nn] - X_cls[idx]))

        return np.array(samples)

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold, cross_val_score

        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        estimator = self.estimator
        if estimator is None:
            estimator = LogisticRegression(
                max_iter=500, random_state=self.random_state
            )

        synthetic_X: list[np.ndarray] = []
        synthetic_y: list[np.ndarray] = []

        for cls, n_synthetic in self.targets_.items():
            if n_synthetic <= 0:
                continue

            X_cls = X[y == cls]
            n_cls = len(X_cls)
            if n_cls == 0:
                continue

            k = min(self.k_neighbors, n_cls - 1)
            if k < 1:
                synthetic_X.append(np.tile(X_cls[0], (n_synthetic, 1)))
                synthetic_y.append(np.full(n_synthetic, cls))
                continue

            # Generate candidate pool
            n_candidates = int(n_synthetic * self.candidate_pool_factor)
            candidates = self._smote_interpolate(X_cls, n_candidates, k, rng)

            # Score each candidate by adding it and running CV
            # For efficiency, score in batches rather than one-by-one
            batch_size = max(1, n_synthetic // 5)
            rng.shuffle(candidates)

            # Baseline score
            cv_split = StratifiedKFold(
                n_splits=min(self.cv, min(np.bincount(y.astype(int)))),
                shuffle=True,
                random_state=self.random_state,
            )
            baseline = np.mean(
                cross_val_score(
                    clone(estimator), X, y, cv=cv_split, scoring=self.scoring
                )
            )

            best_candidates = []
            for start in range(0, len(candidates), batch_size):
                batch = candidates[start : start + batch_size]
                X_aug = np.vstack([X] + [np.array(c).reshape(1, -1) for c in best_candidates] + [batch])
                y_aug = np.concatenate(
                    [y]
                    + [np.full(len(best_candidates), cls)]
                    + [np.full(len(batch), cls)]
                )

                cv_split_aug = StratifiedKFold(
                    n_splits=min(self.cv, min(np.bincount(y_aug.astype(int)))),
                    shuffle=True,
                    random_state=self.random_state,
                )
                score = np.mean(
                    cross_val_score(
                        clone(estimator),
                        X_aug,
                        y_aug,
                        cv=cv_split_aug,
                        scoring=self.scoring,
                    )
                )

                if score >= baseline:
                    best_candidates.extend(batch)
                    baseline = score

                if len(best_candidates) >= n_synthetic:
                    break

            # If we didn't get enough from CV selection, fill with remaining
            if len(best_candidates) < n_synthetic:
                shortfall = n_synthetic - len(best_candidates)
                extra = self._smote_interpolate(X_cls, shortfall, k, rng)
                best_candidates.extend(extra)

            final = np.array(best_candidates[:n_synthetic])
            synthetic_X.append(final)
            synthetic_y.append(np.full(n_synthetic, cls))

        if synthetic_X:
            X_out = np.vstack([X] + synthetic_X)
            y_out = np.concatenate([y] + synthetic_y)
        else:
            X_out, y_out = X.copy(), y.copy()

        return X_out, y_out


# =============================================================================
# OverlapRegionDetector
# =============================================================================


class OverlapRegionDetector(BaseEstimator):
    """Overlap Region Detection meta-method for class imbalance.

    Identifies the overlap region between classes using classifier uncertainty,
    then applies a base sampler with overlap awareness.

    Algorithm
    ---------
    1. Train a classifier to get predicted probabilities.
    2. Samples with high uncertainty (max prob < 1 - threshold) are labelled
       as "overlap".
    3. Apply the base sampler on the augmented label space.
    4. Map generated samples back to original labels.

    Parameters
    ----------
    sampling_strategy : str, float, or dict, default='auto'
        See :func:`_compute_sampling_targets`.
    base_sampler : str or estimator, default='smote'
        Base oversampling method. If a string, looked up in the combined
        sampler registries. Otherwise must support ``fit_resample(X, y)``.
    overlap_estimator : estimator or None, default=None
        Classifier for overlap detection. Defaults to
        ``RandomForestClassifier(n_estimators=100)``.
    k_neighbors : int, default=5
        Passed to base sampler when constructed from string.
    threshold : float, default=0.3
        Uncertainty threshold: a sample is in the overlap region if
        ``max(predicted_proba) < 1 - threshold``.
    random_state : int or None, default=None
        Random seed.

    References
    ----------
    Overlap Region Detection (AAAI 2025)
    """

    def __init__(
        self,
        sampling_strategy: str | float | dict = "auto",
        base_sampler: str | Any = "smote",
        overlap_estimator: Any = None,
        k_neighbors: int = 5,
        threshold: float = 0.3,
        random_state: int | None = None,
    ):
        self.sampling_strategy = sampling_strategy
        self.base_sampler = base_sampler
        self.overlap_estimator = overlap_estimator
        self.k_neighbors = k_neighbors
        self.threshold = threshold
        self.random_state = random_state

    def fit(self, X: ArrayLike, y: ArrayLike) -> OverlapRegionDetector:
        """Fit the sampler."""
        X, y = check_X_y(X, y)
        self.targets_ = _compute_sampling_targets(y, self.sampling_strategy)
        return self

    def _get_base_sampler(self):
        """Resolve the base sampler from string or return as-is."""
        if isinstance(self.base_sampler, str):
            from endgame.preprocessing.imbalance import ALL_SAMPLERS

            key = self.base_sampler
            if key not in ALL_SAMPLERS:
                raise ValueError(
                    f"Unknown base_sampler '{key}'. "
                    f"Available: {list(ALL_SAMPLERS.keys())}"
                )
            SamplerClass = ALL_SAMPLERS[key]
            import inspect

            sig = inspect.signature(SamplerClass.__init__)
            params: dict[str, Any] = {}
            if "random_state" in sig.parameters:
                params["random_state"] = self.random_state
            if "k_neighbors" in sig.parameters:
                params["k_neighbors"] = self.k_neighbors
            if "sampling_strategy" in sig.parameters:
                params["sampling_strategy"] = "auto"
            return SamplerClass(**params)
        return clone(self.base_sampler)

    def fit_resample(
        self, X: ArrayLike, y: ArrayLike
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fit and resample with overlap awareness."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import cross_val_predict

        self.fit(X, y)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)

        # Step 1: Detect overlap region
        overlap_est = self.overlap_estimator
        if overlap_est is None:
            overlap_est = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, n_jobs=-1
            )

        # Use cross-validated predictions to avoid overfitting
        n_cv = min(5, min(np.bincount(y.astype(int))))
        n_cv = max(2, n_cv)
        proba = cross_val_predict(
            clone(overlap_est), X, y, cv=n_cv, method="predict_proba"
        )

        # Step 2: Identify overlap samples (high uncertainty)
        max_proba = proba.max(axis=1)
        overlap_mask = max_proba < (1.0 - self.threshold)
        self.n_overlap_ = int(overlap_mask.sum())

        # Step 3: Create augmented labels
        # Use a sentinel value that won't collide with existing classes
        unique_classes = np.unique(y)
        overlap_label = unique_classes.max() + 1 if len(unique_classes) > 0 else -1

        y_augmented = y.copy()
        y_augmented[overlap_mask] = overlap_label

        # Step 4: Apply base sampler
        base = self._get_base_sampler()

        # If all minority samples are in the overlap region, fall back
        # to standard resampling on original labels
        unique_aug = np.unique(y_augmented)
        min_class_counts = []
        for cls in unique_aug:
            min_class_counts.append(np.sum(y_augmented == cls))
        if min(min_class_counts) < 2:
            # Not enough samples per augmented class; fall back
            base_fallback = self._get_base_sampler()
            # Override sampling strategy to 'auto' on original labels
            return base_fallback.fit_resample(X, y)

        try:
            X_res, y_res_aug = base.fit_resample(X, y_augmented)
        except Exception:
            # Fallback on any error with augmented labels
            base_fallback = self._get_base_sampler()
            return base_fallback.fit_resample(X, y)

        # Step 5: Map overlap label back to original minority classes
        # Assign overlap-generated samples to the nearest original class
        overlap_generated = y_res_aug == overlap_label
        if overlap_generated.any():
            # For overlap-generated samples, assign to nearest original
            # minority class centre
            from scipy.spatial import KDTree

            # Build centres for original classes
            class_centres = {}
            for cls in unique_classes:
                mask = y == cls
                if mask.any():
                    class_centres[cls] = X[mask].mean(axis=0)

            if class_centres:
                centre_labels = list(class_centres.keys())
                centre_points = np.array(
                    [class_centres[c] for c in centre_labels]
                )
                tree = KDTree(centre_points)
                _, nearest_idx = tree.query(X_res[overlap_generated])
                y_res_aug[overlap_generated] = np.array(centre_labels)[
                    nearest_idx
                ]

        return X_res, y_res_aug


# Category dict for registration
GEOMETRIC_SAMPLERS = {
    "multivariate_gaussian_smote": MultivariateGaussianSMOTE,
    "simplicial_smote": SimplicialSMOTE,
    "cv_smote": CVSMOTEResampler,
    "overlap_region_detector": OverlapRegionDetector,
}
