"""Target and categorical encoding transformers."""

from typing import Any

import numpy as np
from sklearn.model_selection import KFold

from endgame.core.base import PolarsTransformer
from endgame.core.polars_ops import (
    HAS_PANDAS,
    HAS_POLARS,
    infer_categorical_columns,
)

if HAS_POLARS:
    import polars as pl
if HAS_PANDAS:
    pass


class SafeTargetEncoder(PolarsTransformer):
    """Target encoding with M-estimate smoothing and inner-fold encoding.

    Prevents target leakage through nested cross-validation during fit
    and applies smoothing for rare categories.

    Implements: S_i = (n_i × μ_i + m × μ_global) / (n_i + m)

    Parameters
    ----------
    cols : List[str], optional
        Columns to encode. If None, encodes all categorical columns.
    smoothing : float, default=10
        Smoothing parameter (m) for rare categories.
        Higher values = more regularization toward global mean.
    cv : int, default=5
        Number of folds for inner-fold encoding during fit.
    min_samples_leaf : int, default=1
        Minimum samples required to compute category statistic.
    noise_level : float, default=0.0
        Gaussian noise std to add for regularization.
    handle_unknown : str, default='global_mean'
        Strategy for unseen categories: 'global_mean', 'nan', 'error'.
    output_format : str, default='auto'
        Output format: 'auto', 'polars', 'pandas', 'numpy'.
    random_state : int, optional
        Random seed for cross-validation and noise.

    Examples
    --------
    >>> from endgame.preprocessing import SafeTargetEncoder
    >>> encoder = SafeTargetEncoder(smoothing=10, cv=5)
    >>> X_encoded = encoder.fit_transform(X, y)
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        smoothing: float = 10.0,
        cv: int = 5,
        min_samples_leaf: int = 1,
        noise_level: float = 0.0,
        handle_unknown: str = "global_mean",
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.smoothing = smoothing
        self.cv = cv
        self.min_samples_leaf = min_samples_leaf
        self.noise_level = noise_level
        self.handle_unknown = handle_unknown

        self._encodings: dict[str, dict[Any, float]] = {}
        self._global_mean: float = 0.0
        self._target_cols: list[str] = []

    def fit(self, X, y, **fit_params) -> "SafeTargetEncoder":
        """Fit the target encoder.

        Uses inner-fold encoding to prevent leakage during training.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        y = np.asarray(y).astype(np.float64)
        self._global_mean = float(np.mean(y))

        # Determine columns to encode
        if self.cols is None:
            self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        self._log(f"Encoding {len(self._target_cols)} columns with target encoding")

        # Compute encodings using full data (for transform)
        for col in self._target_cols:
            col_data = df[col].to_numpy()
            self._encodings[col] = self._compute_encoding(col_data, y)

        self._is_fitted = True
        return self

    def _compute_encoding(
        self,
        col_data: np.ndarray,
        y: np.ndarray,
    ) -> dict[Any, float]:
        """Compute target encoding for a single column."""
        encodings = {}

        # Replace None with a sentinel before finding uniques to avoid
        # comparison errors between str and NoneType during sorting.
        _MISSING = "__MISSING__"
        is_none = np.array([v is None or (isinstance(v, float) and np.isnan(v))
                            for v in col_data])
        safe_data = np.where(is_none, _MISSING, col_data)
        unique_values = np.unique(safe_data)

        for value in unique_values:
            mask = safe_data == value
            n_samples = mask.sum()

            if n_samples < self.min_samples_leaf:
                encodings[value] = self._global_mean
            else:
                category_mean = y[mask].mean()
                # M-estimate smoothing
                smoothed = (
                    n_samples * category_mean + self.smoothing * self._global_mean
                ) / (n_samples + self.smoothing)
                encodings[value] = smoothed

        return encodings

    def transform(self, X) -> Any:
        """Transform data using learned encodings.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Transformed data with encoded columns.
        """
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                # Apply encoding
                col_data = df[col].to_numpy()
                encoded = np.array([
                    self._encodings[col].get(v, self._handle_unknown_value(col, v))
                    for v in col_data
                ])

                # Add noise if specified
                if self.noise_level > 0:
                    rng = np.random.RandomState(self.random_state)
                    encoded = encoded + rng.normal(0, self.noise_level, len(encoded))

                result_dict[col] = encoded
            else:
                result_dict[col] = df[col].to_numpy()

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())

    def _handle_unknown_value(self, col: str, value: Any) -> float:
        """Handle unknown category values."""
        if self.handle_unknown == "global_mean":
            return self._global_mean
        elif self.handle_unknown == "nan":
            return np.nan
        elif self.handle_unknown == "error":
            raise ValueError(f"Unknown value '{value}' in column '{col}'")
        else:
            raise ValueError(f"Unknown handle_unknown strategy: {self.handle_unknown}")

    def fit_transform(self, X, y, **fit_params) -> Any:
        """Fit and transform with inner-fold encoding to prevent leakage.

        During fit_transform, uses cross-validation to compute encodings
        without leakage. Each sample is encoded using statistics computed
        only from other samples.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like
            Target values.

        Returns
        -------
        X_transformed : array-like
            Transformed training data.
        """
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        y = np.asarray(y).astype(np.float64)
        self._global_mean = float(np.mean(y))

        # Determine columns to encode
        if self.cols is None:
            self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        n_samples = len(df)
        result_dict = {col: np.zeros(n_samples) for col in self._target_cols}

        # Copy non-encoded columns
        for col in df.columns:
            if col not in self._target_cols:
                result_dict[col] = df[col].to_numpy()

        # Inner-fold encoding
        kf = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        for train_idx, val_idx in kf.split(df):
            for col in self._target_cols:
                col_data = df[col].to_numpy()

                # Compute encoding from train fold only
                train_encoding = self._compute_encoding(
                    col_data[train_idx],
                    y[train_idx],
                )

                # Apply to validation fold
                for idx in val_idx:
                    value = col_data[idx]
                    encoded = train_encoding.get(value, self._global_mean)
                    result_dict[col][idx] = encoded

        # Store encodings computed from full data for transform
        for col in self._target_cols:
            col_data = df[col].to_numpy()
            self._encodings[col] = self._compute_encoding(col_data, y)

        # Add noise if specified
        if self.noise_level > 0:
            rng = np.random.RandomState(self.random_state)
            for col in self._target_cols:
                result_dict[col] = result_dict[col] + rng.normal(
                    0, self.noise_level, n_samples
                )

        self._is_fitted = True

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names (same as input for target encoding)."""
        self._check_is_fitted()
        return self._feature_names_in or []


class LeaveOneOutEncoder(PolarsTransformer):
    """Leave-One-Out target encoding for online settings.

    Each sample's encoding excludes its own target value, preventing
    direct leakage while still using all available data.

    Parameters
    ----------
    cols : List[str], optional
        Columns to encode. If None, encodes all categorical columns.
    smoothing : float, default=1.0
        Smoothing parameter for regularization.
    handle_unknown : str, default='global_mean'
        Strategy for unseen categories.
    random_state : int, optional
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        smoothing: float = 1.0,
        handle_unknown: str = "global_mean",
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.smoothing = smoothing
        self.handle_unknown = handle_unknown

        self._encodings: dict[str, dict[Any, tuple]] = {}  # value -> (sum, count)
        self._global_mean: float = 0.0
        self._target_cols: list[str] = []

    def fit(self, X, y, **fit_params) -> "LeaveOneOutEncoder":
        """Fit the LOO encoder."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        y = np.asarray(y).astype(np.float64)
        self._global_mean = float(np.mean(y))

        if self.cols is None:
            self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        # Store sum and count for each category
        for col in self._target_cols:
            col_data = df[col].to_numpy()
            self._encodings[col] = {}

            for value in np.unique(col_data):
                mask = col_data == value
                self._encodings[col][value] = (y[mask].sum(), mask.sum())

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Transform using stored statistics (no LOO at test time)."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                col_data = df[col].to_numpy()
                encoded = np.zeros(len(col_data))

                for i, value in enumerate(col_data):
                    if value in self._encodings[col]:
                        total, count = self._encodings[col][value]
                        encoded[i] = (total + self.smoothing * self._global_mean) / (
                            count + self.smoothing
                        )
                    else:
                        encoded[i] = self._global_mean

                result_dict[col] = encoded
            else:
                result_dict[col] = df[col].to_numpy()

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())

    def fit_transform(self, X, y, **fit_params) -> Any:
        """Fit and transform with LOO to prevent leakage."""
        self.fit(X, y, **fit_params)

        lf = self._to_lazyframe(X)
        df = lf.collect()
        y = np.asarray(y).astype(np.float64)

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                col_data = df[col].to_numpy()
                encoded = np.zeros(len(col_data))

                for i, value in enumerate(col_data):
                    if value in self._encodings[col]:
                        total, count = self._encodings[col][value]
                        # Leave out current sample
                        loo_total = total - y[i]
                        loo_count = count - 1

                        if loo_count > 0:
                            encoded[i] = (
                                loo_total + self.smoothing * self._global_mean
                            ) / (loo_count + self.smoothing)
                        else:
                            encoded[i] = self._global_mean
                    else:
                        encoded[i] = self._global_mean

                result_dict[col] = encoded
            else:
                result_dict[col] = df[col].to_numpy()

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())


class CatBoostEncoder(PolarsTransformer):
    """CatBoost-style ordered target encoding.

    Encodes based only on preceding samples, mimicking CatBoost's
    internal target statistic computation. Prevents leakage by
    using only "past" information for each sample.

    Parameters
    ----------
    cols : List[str], optional
        Columns to encode.
    smoothing : float, default=1.0
        Smoothing parameter.
    random_state : int, optional
        Random seed for sample ordering.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        smoothing: float = 1.0,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.smoothing = smoothing

        self._encodings: dict[str, dict[Any, float]] = {}
        self._global_mean: float = 0.0
        self._target_cols: list[str] = []

    def fit(self, X, y, **fit_params) -> "CatBoostEncoder":
        """Fit encoder (stores final statistics for transform)."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        y = np.asarray(y).astype(np.float64)
        self._global_mean = float(np.mean(y))

        if self.cols is None:
            self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        # Compute final encodings for transform
        for col in self._target_cols:
            col_data = df[col].to_numpy()
            self._encodings[col] = {}

            for value in np.unique(col_data):
                mask = col_data == value
                n = mask.sum()
                if n > 0:
                    mean = y[mask].mean()
                    self._encodings[col][value] = (
                        n * mean + self.smoothing * self._global_mean
                    ) / (n + self.smoothing)
                else:
                    self._encodings[col][value] = self._global_mean

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Transform using final statistics."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                col_data = df[col].to_numpy()
                encoded = np.array([
                    self._encodings[col].get(v, self._global_mean)
                    for v in col_data
                ])
                result_dict[col] = encoded
            else:
                result_dict[col] = df[col].to_numpy()

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())

    def fit_transform(self, X, y, **fit_params) -> Any:
        """Fit and transform with ordered encoding."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        y = np.asarray(y).astype(np.float64)
        self._global_mean = float(np.mean(y))

        if self.cols is None:
            self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        n_samples = len(df)

        # Random permutation for ordering
        rng = np.random.RandomState(self.random_state)
        perm = rng.permutation(n_samples)
        inv_perm = np.argsort(perm)

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                col_data = df[col].to_numpy()[perm]
                y_perm = y[perm]
                encoded = np.zeros(n_samples)

                # Running statistics
                running_sum: dict[Any, float] = {}
                running_count: dict[Any, int] = {}

                for i in range(n_samples):
                    value = col_data[i]

                    # Use running statistics up to this point
                    if value in running_count and running_count[value] > 0:
                        prior_sum = running_sum[value]
                        prior_count = running_count[value]
                        encoded[i] = (
                            prior_sum + self.smoothing * self._global_mean
                        ) / (prior_count + self.smoothing)
                    else:
                        encoded[i] = self._global_mean

                    # Update running statistics
                    if value not in running_sum:
                        running_sum[value] = 0.0
                        running_count[value] = 0
                    running_sum[value] += y_perm[i]
                    running_count[value] += 1

                # Restore original order
                result_dict[col] = encoded[inv_perm]
            else:
                result_dict[col] = df[col].to_numpy()

        # Store final encodings for transform
        for col in self._target_cols:
            col_data = df[col].to_numpy()
            self._encodings[col] = {}
            for value in np.unique(col_data):
                mask = col_data == value
                n = mask.sum()
                if n > 0:
                    mean = y[mask].mean()
                    self._encodings[col][value] = (
                        n * mean + self.smoothing * self._global_mean
                    ) / (n + self.smoothing)

        self._is_fitted = True

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())


class FrequencyEncoder(PolarsTransformer):
    """Frequency encoding for categorical features.

    Replaces categories with their frequency (count or proportion).
    Simple but effective encoding that doesn't require target values.

    Parameters
    ----------
    cols : List[str], optional
        Columns to encode. If None, encodes all categorical columns.
    normalize : bool, default=True
        If True, use proportions. If False, use raw counts.
    handle_unknown : str, default='zero'
        Strategy for unseen categories: 'zero', 'nan', 'error'.
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        normalize: bool = True,
        handle_unknown: str = "zero",
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.cols = cols
        self.normalize = normalize
        self.handle_unknown = handle_unknown

        self._frequencies: dict[str, dict[Any, float]] = {}
        self._target_cols: list[str] = []
        self._n_samples: int = 0

    def fit(self, X, y=None, **fit_params) -> "FrequencyEncoder":
        """Compute frequencies from training data."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        self._n_samples = len(df)

        if self.cols is None:
            # For numpy arrays, encode all columns by default
            if isinstance(X, np.ndarray):
                self._target_cols = list(df.columns)
            else:
                self._target_cols = infer_categorical_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        for col in self._target_cols:
            col_data = df[col].to_numpy().flatten()  # Ensure 1D
            unique, counts = np.unique(col_data, return_counts=True)

            if self.normalize:
                frequencies = counts / self._n_samples
            else:
                frequencies = counts.astype(float)

            self._frequencies[col] = dict(zip(unique, frequencies))

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Apply frequency encoding."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        result_dict = {}

        for col in df.columns:
            if col in self._target_cols:
                col_data = df[col].to_numpy().flatten()  # Ensure 1D

                if self.handle_unknown == "zero":
                    default = 0.0
                elif self.handle_unknown == "nan":
                    default = np.nan
                else:
                    default = None

                encoded = np.zeros(len(col_data), dtype=np.float64)
                for i, v in enumerate(col_data):
                    if v in self._frequencies[col]:
                        encoded[i] = self._frequencies[col][v]
                    elif default is not None:
                        encoded[i] = default
                    else:
                        raise ValueError(f"Unknown value '{v}' in column '{col}'")

                result_dict[col] = encoded
            else:
                col_arr = df[col].to_numpy()
                # Ensure 1D array
                if col_arr.ndim > 1:
                    col_arr = col_arr.flatten()
                result_dict[col] = col_arr

        result_df = pl.DataFrame(result_dict)
        return self._from_lazyframe(result_df.lazy())
