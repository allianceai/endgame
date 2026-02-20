"""Cross-validation splitters for competition-specific scenarios."""

from collections.abc import Generator
from itertools import combinations
from typing import Any

import numpy as np
from sklearn.model_selection import BaseCrossValidator, StratifiedKFold

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class PurgedTimeSeriesSplit(BaseCrossValidator):
    """Time series CV with purging and embargo to prevent lookahead bias.

    Essential for financial competitions (Optiver, Jane Street) where
    temporal leakage can severely overfit models.

    Purging removes samples between train and validation that might
    contain information about the validation period.

    Embargo adds a gap after validation to prevent using future information.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    purge_gap : int, default=0
        Number of samples to purge between train and validation.
    embargo_pct : float, default=0.01
        Percentage of test data to embargo after each split.
    max_train_size : int, optional
        Maximum size of training set (rolling window).

    Examples
    --------
    >>> cv = PurgedTimeSeriesSplit(n_splits=5, purge_gap=10, embargo_pct=0.01)
    >>> for train_idx, val_idx in cv.split(X):
    ...     # train_idx ends purge_gap samples before val_idx starts
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_gap: int = 0,
        embargo_pct: float = 0.01,
        max_train_size: int | None = None,
    ):
        self.n_splits = n_splits
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct
        self.max_train_size = max_train_size

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate train/validation indices with purging and embargo.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable (ignored).
        groups : array-like, optional
            Group labels (ignored).

        Yields
        ------
        train_idx : ndarray
            Training indices for this fold.
        val_idx : ndarray
            Validation indices for this fold.
        """
        n_samples = len(X)
        indices = np.arange(n_samples)

        # Compute embargo size
        embargo_size = int(n_samples * self.embargo_pct)

        # Compute fold boundaries
        fold_size = n_samples // (self.n_splits + 1)

        for fold in range(self.n_splits):
            # Validation start and end
            val_start = (fold + 1) * fold_size
            val_end = val_start + fold_size

            if fold == self.n_splits - 1:
                val_end = n_samples

            # Training ends before purge gap
            train_end = val_start - self.purge_gap

            if train_end <= 0:
                continue

            train_idx = indices[:train_end]

            # Apply max_train_size (rolling window)
            if self.max_train_size is not None and len(train_idx) > self.max_train_size:
                train_idx = train_idx[-self.max_train_size:]

            # Validation with embargo
            val_idx = indices[val_start:val_end]

            # Apply embargo to training (remove samples too close to future val)
            if embargo_size > 0 and fold > 0:
                embargo_start = val_start - embargo_size
                train_idx = train_idx[train_idx < embargo_start]

            if len(train_idx) > 0 and len(val_idx) > 0:
                yield train_idx, val_idx


class StratifiedGroupKFold(BaseCrossValidator):
    """Stratified K-Fold that respects groups.

    Combines stratification (maintaining class balance) with group constraints
    (keeping all samples from a group in the same fold).

    Essential when samples are related (e.g., patient_id, user_id) to prevent
    data leakage.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle groups before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> cv = StratifiedGroupKFold(n_splits=5)
    >>> for train_idx, val_idx in cv.split(X, y, groups=patient_ids):
    ...     # No patient appears in both train and val
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any,
        groups: Any,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate stratified group-aware train/validation indices.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target variable for stratification.
        groups : array-like
            Group labels (e.g., patient_id).

        Yields
        ------
        train_idx : ndarray
            Training indices for this fold.
        val_idx : ndarray
            Validation indices for this fold.
        """
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Get unique groups and their properties
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)

        # Compute dominant class for each group (for stratification)
        group_to_class = {}
        for group in unique_groups:
            group_mask = groups == group
            group_y = y[group_mask]
            # Use most common class in group
            unique, counts = np.unique(group_y, return_counts=True)
            group_to_class[group] = unique[np.argmax(counts)]

        # Create array of group classes for stratification
        group_classes = np.array([group_to_class[g] for g in unique_groups])

        # Stratified split of groups
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=self.shuffle,
            random_state=self.random_state,
        )

        for train_group_idx, val_group_idx in skf.split(unique_groups, group_classes):
            train_groups = set(unique_groups[train_group_idx])
            val_groups = set(unique_groups[val_group_idx])

            train_idx = np.where([g in train_groups for g in groups])[0]
            val_idx = np.where([g in val_groups for g in groups])[0]

            yield train_idx, val_idx


class RepeatedStratifiedGroupKFold(BaseCrossValidator):
    """Repeated Stratified Group K-Fold.

    Runs multiple iterations of StratifiedGroupKFold with different
    random seeds for more robust CV estimates.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds per repeat.
    n_repeats : int, default=3
        Number of times to repeat the splits.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_splits: int = 5,
        n_repeats: int = 3,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the total number of splits."""
        return self.n_splits * self.n_repeats

    def split(
        self,
        X: Any,
        y: Any,
        groups: Any,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate repeated stratified group-aware splits."""
        rng = np.random.RandomState(self.random_state)

        for repeat in range(self.n_repeats):
            cv = StratifiedGroupKFold(
                n_splits=self.n_splits,
                shuffle=True,
                random_state=rng.randint(0, 2**31),
            )
            yield from cv.split(X, y, groups)


class MultilabelStratifiedKFold(BaseCrossValidator):
    """Stratified K-Fold for multilabel classification.

    Maintains label distribution across folds for multilabel problems
    using iterative stratification.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    shuffle : bool, default=True
        Whether to shuffle before splitting.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> # y is shape (n_samples, n_labels) with binary labels
    >>> cv = MultilabelStratifiedKFold(n_splits=5)
    >>> for train_idx, val_idx in cv.split(X, y):
    ...     # Label proportions maintained across folds
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def split(
        self,
        X: Any,
        y: Any,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate multilabel-stratified train/validation indices.

        Uses iterative stratification algorithm to maintain label proportions.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like of shape (n_samples, n_labels)
            Multilabel target matrix.
        groups : array-like, optional
            Ignored.

        Yields
        ------
        train_idx : ndarray
            Training indices for this fold.
        val_idx : ndarray
            Validation indices for this fold.
        """
        y = np.asarray(y)
        if y.ndim == 1:
            y = y.reshape(-1, 1)

        n_samples, n_labels = y.shape
        indices = np.arange(n_samples)

        rng = np.random.RandomState(self.random_state)

        if self.shuffle:
            rng.shuffle(indices)
            y = y[indices]

        # Initialize folds
        folds = [[] for _ in range(self.n_splits)]
        fold_label_counts = np.zeros((self.n_splits, n_labels))

        # Desired samples per fold per label
        label_totals = y.sum(axis=0)
        desired_per_fold = label_totals / self.n_splits

        # Iterative stratification
        for i in range(n_samples):
            sample_labels = y[i]

            # Find fold with most need for this sample's labels
            scores = np.zeros(self.n_splits)
            for fold in range(self.n_splits):
                for label in range(n_labels):
                    if sample_labels[label] == 1:
                        deficit = desired_per_fold[label] - fold_label_counts[fold, label]
                        scores[fold] += deficit

            # Add sample to fold with highest score (most deficit)
            # Break ties by fold size (prefer smaller)
            fold_sizes = np.array([len(f) for f in folds])
            # Slightly prefer smaller folds
            scores = scores - 0.001 * fold_sizes
            best_fold = np.argmax(scores)

            folds[best_fold].append(indices[i])
            fold_label_counts[best_fold] += sample_labels

        # Generate splits
        for fold in range(self.n_splits):
            val_idx = np.array(folds[fold])
            train_idx = np.concatenate([
                np.array(folds[f]) for f in range(self.n_splits) if f != fold
            ])
            yield train_idx, val_idx


class AdversarialKFold(BaseCrossValidator):
    """K-Fold that weights folds by test-similarity.

    Uses adversarial validation to identify training samples that
    look most like test data, then ensures each fold has similar
    proportions of test-like samples.

    Parameters
    ----------
    n_splits : int, default=5
        Number of folds.
    test_similarity_threshold : float, default=0.5
        Threshold for considering a sample "test-like".
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> cv = AdversarialKFold(n_splits=5)
    >>> for train_idx, val_idx in cv.split(X_train, y, X_test=X_test):
    ...     # Each fold has similar proportion of test-like samples
    ...     pass
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_similarity_threshold: float = 0.5,
        random_state: int | None = None,
    ):
        self.n_splits = n_splits
        self.test_similarity_threshold = test_similarity_threshold
        self.random_state = random_state
        self._test_similarity: np.ndarray | None = None

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def fit(self, X_train: Any, X_test: Any) -> "AdversarialKFold":
        """Compute test similarity scores for training samples.

        Parameters
        ----------
        X_train : array-like
            Training features.
        X_test : array-like
            Test features.

        Returns
        -------
        self
        """
        from endgame.validation.adversarial import AdversarialValidator

        av = AdversarialValidator(
            cv=3,
            random_state=self.random_state,
        )
        av.check_drift(X_train, X_test)

        # Get test-likeness scores
        X_train_arr = np.asarray(X_train)
        if X_train_arr.ndim == 1:
            X_train_arr = X_train_arr.reshape(-1, 1)
        X_train_arr = np.nan_to_num(X_train_arr, nan=0.0, posinf=0.0, neginf=0.0)

        self._test_similarity = av._estimator.predict_proba(X_train_arr)[:, 1]
        return self

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
        X_test: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate adversarial-aware train/validation indices.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Target variable.
        groups : array-like, optional
            Ignored.
        X_test : array-like, optional
            Test data for computing similarity (if not already fit).

        Yields
        ------
        train_idx : ndarray
            Training indices for this fold.
        val_idx : ndarray
            Validation indices for this fold.
        """
        n_samples = len(X)

        # Compute similarity if not already done
        if self._test_similarity is None:
            if X_test is None:
                raise ValueError(
                    "X_test must be provided or fit() must be called first"
                )
            self.fit(X, X_test)

        # Split samples into test-like and non-test-like
        test_like_mask = self._test_similarity >= self.test_similarity_threshold
        test_like_idx = np.where(test_like_mask)[0]
        non_test_like_idx = np.where(~test_like_mask)[0]

        rng = np.random.RandomState(self.random_state)
        rng.shuffle(test_like_idx)
        rng.shuffle(non_test_like_idx)

        # Split both groups into folds
        test_like_folds = np.array_split(test_like_idx, self.n_splits)
        non_test_like_folds = np.array_split(non_test_like_idx, self.n_splits)

        for fold in range(self.n_splits):
            val_idx = np.concatenate([
                test_like_folds[fold],
                non_test_like_folds[fold],
            ])

            train_idx = np.concatenate([
                np.concatenate([test_like_folds[f] for f in range(self.n_splits) if f != fold]),
                np.concatenate([non_test_like_folds[f] for f in range(self.n_splits) if f != fold]),
            ])

            yield train_idx, val_idx


class CombinatorialPurgedKFold(BaseCrossValidator):
    """Combinatorial Purged Cross-Validation for time series/financial data.

    Implements the CPCV method from Marcos López de Prado's "Advances in
    Financial Machine Learning" (Chapter 12). This method:

    1. Divides data into N sequential groups (folds)
    2. Uses combinations of k groups as test sets (C(N,k) total splits)
    3. Applies purging to remove training samples that overlap with test labels
    4. Applies embargo to remove training samples too close to test periods

    This generates multiple "backtest paths" that can be recombined to compute
    statistics like the distribution of Sharpe ratios, enabling detection of
    backtest overfitting.

    Parameters
    ----------
    n_folds : int, default=10
        Number of sequential groups to divide the data into.
        Must be >= 3.

    n_test_folds : int, default=2
        Number of folds to use as test set in each split.
        Must be >= 1 and < n_folds.
        Total number of splits = C(n_folds, n_test_folds).

    purge_gap : int, default=0
        Number of samples to purge (remove) from training set at boundaries
        with test set. These are samples whose labels might overlap with
        the test period.

    embargo_pct : float, default=0.0
        Percentage of total samples to embargo after each test period.
        Embargo removes training samples that occur immediately after test
        samples to prevent lookahead bias from label leakage.

    Attributes
    ----------
    n_splits : int
        Total number of train/test splits = C(n_folds, n_test_folds).

    n_test_paths : int
        Number of reconstructible test paths from combinations.

    fold_bounds_ : List[Tuple[int, int]]
        Start and end indices for each fold (set after split is called).

    Notes
    -----
    The key insight of CPCV is that standard k-fold CV produces only ONE
    backtest path (the concatenation of all test folds). CPCV produces
    MULTIPLE backtest paths by using combinations of test folds, enabling
    statistical analysis of strategy performance across different scenarios.

    For example, with n_folds=6 and n_test_folds=2:
    - Standard KFold: 6 splits, 1 backtest path
    - CPCV: C(6,2)=15 splits, multiple backtest paths

    References
    ----------
    López de Prado, M. (2018). "Advances in Financial Machine Learning".
    Chapter 12: Backtesting through Cross-Validation.

    Examples
    --------
    >>> from endgame.validation import CombinatorialPurgedKFold
    >>> import numpy as np
    >>>
    >>> # Financial time series with 1000 samples
    >>> X = np.random.randn(1000, 10)
    >>> y = np.random.randn(1000)
    >>>
    >>> # Use 6 folds, 2 test folds per split, with purging and embargo
    >>> cpcv = CombinatorialPurgedKFold(
    ...     n_folds=6,
    ...     n_test_folds=2,
    ...     purge_gap=10,
    ...     embargo_pct=0.01,
    ... )
    >>>
    >>> print(f"Number of splits: {cpcv.get_n_splits()}")  # 15 splits
    >>>
    >>> for train_idx, test_idx in cpcv.split(X):
    ...     # Train model on train_idx, evaluate on test_idx
    ...     pass
    >>>
    >>> # Get backtest paths for strategy analysis
    >>> paths = cpcv.get_test_paths(X)
    >>> print(f"Number of backtest paths: {len(paths)}")
    """

    def __init__(
        self,
        n_folds: int = 10,
        n_test_folds: int = 2,
        purge_gap: int = 0,
        embargo_pct: float = 0.0,
    ):
        if n_folds < 3:
            raise ValueError("n_folds must be >= 3")
        if n_test_folds < 1:
            raise ValueError("n_test_folds must be >= 1")
        if n_test_folds >= n_folds:
            raise ValueError("n_test_folds must be < n_folds")

        self.n_folds = n_folds
        self.n_test_folds = n_test_folds
        self.purge_gap = purge_gap
        self.embargo_pct = embargo_pct

        self.fold_bounds_: list[tuple[int, int]] | None = None
        self._n_samples: int | None = None

    @property
    def n_splits(self) -> int:
        """Total number of train/test splits."""
        from math import comb
        return comb(self.n_folds, self.n_test_folds)

    @property
    def n_test_paths(self) -> int:
        """Number of reconstructible backtest paths.

        Each path is a complete sequence through the data using different
        combinations of the test folds.
        """
        from math import comb
        # Number of ways to arrange test folds into paths
        return comb(self.n_folds - 1, self.n_test_folds - 1)

    def get_n_splits(
        self,
        X: Any | None = None,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> int:
        """Return the number of splits."""
        return self.n_splits

    def _compute_fold_bounds(self, n_samples: int) -> list[tuple[int, int]]:
        """Compute the start and end indices for each fold."""
        fold_size = n_samples // self.n_folds
        bounds = []

        for i in range(self.n_folds):
            start = i * fold_size
            end = (i + 1) * fold_size if i < self.n_folds - 1 else n_samples
            bounds.append((start, end))

        return bounds

    def _get_embargo_size(self, n_samples: int) -> int:
        """Compute embargo size in number of samples."""
        return int(n_samples * self.embargo_pct)

    def _apply_purging_and_embargo(
        self,
        train_indices: np.ndarray,
        test_fold_indices: list[int],
        fold_bounds: list[tuple[int, int]],
        n_samples: int,
    ) -> np.ndarray:
        """Apply purging and embargo to training indices.

        Purging: Remove training samples whose indices are within purge_gap
        of any test fold boundary.

        Embargo: Remove training samples that occur within embargo period
        after any test fold.

        Parameters
        ----------
        train_indices : np.ndarray
            Original training indices.
        test_fold_indices : List[int]
            Indices of folds used for testing.
        fold_bounds : List[Tuple[int, int]]
            Start and end indices for each fold.
        n_samples : int
            Total number of samples.

        Returns
        -------
        np.ndarray
            Purged and embargoed training indices.
        """
        if self.purge_gap == 0 and self.embargo_pct == 0.0:
            return train_indices

        embargo_size = self._get_embargo_size(n_samples)
        mask = np.ones(len(train_indices), dtype=bool)

        for test_fold_idx in test_fold_indices:
            test_start, test_end = fold_bounds[test_fold_idx]

            for i, train_idx in enumerate(train_indices):
                # Purging: Remove samples too close to test boundaries
                if self.purge_gap > 0:
                    # Sample is within purge_gap before test start
                    if test_start - self.purge_gap <= train_idx < test_start or test_end <= train_idx < test_end + self.purge_gap:
                        mask[i] = False

                # Embargo: Remove samples in embargo period after test
                if embargo_size > 0:
                    if test_end <= train_idx < test_end + embargo_size:
                        mask[i] = False

        return train_indices[mask]

    def split(
        self,
        X: Any,
        y: Any | None = None,
        groups: Any | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate combinatorial purged train/test splits.

        Parameters
        ----------
        X : array-like
            Training data. Used only to determine the number of samples.
        y : array-like, optional
            Target variable (ignored, but accepted for sklearn compatibility).
        groups : array-like, optional
            Group labels (ignored).

        Yields
        ------
        train_idx : np.ndarray
            Training indices for this split (purged and embargoed).
        test_idx : np.ndarray
            Test indices for this split.
        """
        n_samples = len(X)
        self._n_samples = n_samples
        indices = np.arange(n_samples)

        # Compute fold boundaries
        fold_bounds = self._compute_fold_bounds(n_samples)
        self.fold_bounds_ = fold_bounds

        # Generate all combinations of test folds
        test_fold_combinations = list(combinations(range(self.n_folds), self.n_test_folds))

        for test_fold_indices in test_fold_combinations:
            # Determine train fold indices
            train_fold_indices = [i for i in range(self.n_folds) if i not in test_fold_indices]

            # Build test indices
            test_idx_parts = []
            for fold_idx in test_fold_indices:
                start, end = fold_bounds[fold_idx]
                test_idx_parts.append(indices[start:end])
            test_idx = np.concatenate(test_idx_parts)

            # Build train indices
            train_idx_parts = []
            for fold_idx in train_fold_indices:
                start, end = fold_bounds[fold_idx]
                train_idx_parts.append(indices[start:end])
            train_idx = np.concatenate(train_idx_parts)

            # Apply purging and embargo
            train_idx = self._apply_purging_and_embargo(
                train_idx,
                list(test_fold_indices),
                fold_bounds,
                n_samples,
            )

            if len(train_idx) > 0 and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_test_paths(self, X: Any) -> list[list[np.ndarray]]:
        """Reconstruct all possible backtest paths from the splits.

        A backtest path is a sequence of test sets that together cover
        the entire dataset in temporal order. CPCV allows reconstructing
        multiple such paths from the combinatorial splits.

        Parameters
        ----------
        X : array-like
            Training data (used only to determine size).

        Returns
        -------
        List[List[np.ndarray]]
            List of paths, where each path is a list of test index arrays
            that together form a complete pass through the data.
        """
        n_samples = len(X)
        if self.fold_bounds_ is None:
            self.fold_bounds_ = self._compute_fold_bounds(n_samples)

        indices = np.arange(n_samples)
        paths = []

        # Generate paths by selecting which fold goes into which position
        # For n_test_folds positions, we need to select from remaining folds
        # This is a simplified path reconstruction

        # Each path consists of test_folds in sequence
        # We enumerate paths by choosing which n_test_folds to use
        test_fold_combinations = list(combinations(range(self.n_folds), self.n_test_folds))

        for test_fold_indices in test_fold_combinations:
            # Sort test folds by their temporal order
            sorted_test_folds = sorted(test_fold_indices)
            path = []
            for fold_idx in sorted_test_folds:
                start, end = self.fold_bounds_[fold_idx]
                path.append(indices[start:end])
            paths.append(path)

        return paths

    def get_fold_info(self, X: Any) -> dict[str, Any]:
        """Get detailed information about the fold structure.

        Parameters
        ----------
        X : array-like
            Training data.

        Returns
        -------
        Dict[str, Any]
            Dictionary containing:
            - n_samples: Total number of samples
            - n_folds: Number of folds
            - n_test_folds: Number of test folds per split
            - n_splits: Total number of splits
            - n_test_paths: Number of backtest paths
            - fold_sizes: List of fold sizes
            - purge_gap: Purge gap setting
            - embargo_size: Embargo size in samples
        """
        n_samples = len(X)
        fold_bounds = self._compute_fold_bounds(n_samples)
        fold_sizes = [end - start for start, end in fold_bounds]

        return {
            "n_samples": n_samples,
            "n_folds": self.n_folds,
            "n_test_folds": self.n_test_folds,
            "n_splits": self.n_splits,
            "n_test_paths": self.n_test_paths,
            "fold_sizes": fold_sizes,
            "purge_gap": self.purge_gap,
            "embargo_size": self._get_embargo_size(n_samples),
        }

    def __repr__(self) -> str:
        return (
            f"CombinatorialPurgedKFold(n_folds={self.n_folds}, "
            f"n_test_folds={self.n_test_folds}, "
            f"purge_gap={self.purge_gap}, "
            f"embargo_pct={self.embargo_pct})"
        )
