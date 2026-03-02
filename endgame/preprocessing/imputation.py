from __future__ import annotations

"""Missing data imputation transformers.

Provides sklearn-compatible imputers with competition-winning defaults:

- SimpleImputer: Mean/median/mode/constant with better defaults than sklearn
- IndicatorImputer: Adds binary missing-indicator columns alongside imputed values
- KNNImputer: K-Nearest Neighbors imputation with competition defaults
- MICEImputer: Multiple Imputation by Chained Equations (IterativeImputer)
- MissForestImputer: Random Forest-based iterative imputation
- AutoImputer: Automatic strategy selection based on missingness patterns

All imputers accept numpy arrays and pandas DataFrames, preserving
column names when possible.

Examples
--------
>>> import numpy as np
>>> from endgame.preprocessing.imputation import AutoImputer
>>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan]])
>>> imputer = AutoImputer()
>>> X_imputed = imputer.fit_transform(X)
"""

from typing import Any

import numpy as np
from sklearn.base import TransformerMixin

from endgame.core.base import EndgameEstimator

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _to_numpy(X: Any) -> np.ndarray:
    """Convert input to numpy array, handling pandas DataFrames."""
    if isinstance(X, np.ndarray):
        return X
    if HAS_PANDAS and isinstance(X, pd.DataFrame):
        return X.values
    return np.asarray(X, dtype=np.float64)


def _extract_column_names(X: Any) -> list[str] | None:
    """Extract column names from input if available."""
    if HAS_PANDAS and isinstance(X, pd.DataFrame):
        return list(X.columns)
    return None


def _restore_dataframe(
    X_out: np.ndarray,
    columns: list[str] | None,
    was_dataframe: bool,
) -> Any:
    """Restore pandas DataFrame if the input was a DataFrame."""
    if was_dataframe and HAS_PANDAS and columns is not None:
        return pd.DataFrame(X_out, columns=columns)
    return X_out


class SimpleImputer(EndgameEstimator, TransformerMixin):
    """Simple imputation with mean, median, mode, or constant fill.

    Thin wrapper around sklearn.impute.SimpleImputer with better defaults
    for competition settings (median instead of mean, which is more robust
    to outliers).

    Parameters
    ----------
    strategy : str, default='median'
        Imputation strategy:
        - 'mean': Replace with column mean
        - 'median': Replace with column median (default, outlier-robust)
        - 'most_frequent': Replace with mode
        - 'constant': Replace with ``fill_value``
    fill_value : float or str, optional
        Value to use when ``strategy='constant'``. Default is 0.
    add_indicator : bool, default=False
        If True, append binary missing-indicator columns.
    copy : bool, default=True
        If True, create a copy of X before imputing.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        The imputation fill value for each feature.
    indicator_ : MissingIndicator or None
        Indicator used to add binary indicators for missing values.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import SimpleImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan]])
    >>> imp = SimpleImputer(strategy='median')
    >>> imp.fit_transform(X)
    array([[1. , 2. ],
           [4. , 3. ],
           [7. , 2.5]])
    """

    def __init__(
        self,
        strategy: str = "median",
        fill_value: float | str | None = None,
        add_indicator: bool = False,
        copy: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_indicator = add_indicator
        self.copy = copy

    def fit(self, X, y=None, **fit_params) -> SimpleImputer:
        """Fit the imputer on training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data with missing values (np.nan).
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.impute import SimpleImputer as _SklearnSimpleImputer

        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        fill_value = self.fill_value if self.fill_value is not None else 0
        self._imputer = _SklearnSimpleImputer(
            strategy=self.strategy,
            fill_value=fill_value,
            add_indicator=self.add_indicator,
            copy=self.copy,
        )
        self._imputer.fit(_to_numpy(X))

        self.statistics_ = self._imputer.statistics_
        self.n_features_in_ = self._imputer.n_features_in_
        self.indicator_ = getattr(self._imputer, "indicator_", None)

        self._is_fitted = True
        self._log(f"Fitted SimpleImputer with strategy='{self.strategy}'")
        return self

    def transform(self, X) -> Any:
        """Impute missing values in X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : ndarray or DataFrame of shape (n_samples, n_features)
            Imputed data.
        """
        self._check_is_fitted()
        was_df = HAS_PANDAS and isinstance(X, pd.DataFrame)
        cols = _extract_column_names(X) if was_df else self._column_names

        X_out = self._imputer.transform(_to_numpy(X))

        if self.add_indicator and cols is not None:
            n_orig = len(cols)
            indicator_cols = [f"{c}_missing" for c in cols]
            cols = list(cols) + indicator_cols

        return _restore_dataframe(X_out, cols, was_df)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        names = input_features or self._column_names or []
        if self.add_indicator and names:
            names = list(names) + [f"{c}_missing" for c in names]
        return names


class IndicatorImputer(EndgameEstimator, TransformerMixin):
    """Imputer that adds binary missing-indicator columns alongside imputed values.

    For each feature with missing values, appends a binary column indicating
    which rows were originally missing. This is a common Kaggle trick that
    lets tree-based models learn different splits for missing vs. non-missing.

    Parameters
    ----------
    base_strategy : str, default='median'
        Strategy for filling missing values: 'mean', 'median', 'most_frequent',
        'constant'.
    fill_value : float, optional
        Fill value when base_strategy='constant'.
    only_missing : bool, default=True
        If True, only add indicators for features that have missing values
        in the training data. If False, add indicators for all features.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    statistics_ : ndarray of shape (n_features,)
        The imputation fill value for each feature.
    missing_features_ : list of int
        Indices of features that had missing values during fit.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import IndicatorImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan]])
    >>> imp = IndicatorImputer(base_strategy='median')
    >>> X_out = imp.fit_transform(X)
    >>> X_out.shape
    (3, 4)
    """

    def __init__(
        self,
        base_strategy: str = "median",
        fill_value: float | None = None,
        only_missing: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.base_strategy = base_strategy
        self.fill_value = fill_value
        self.only_missing = only_missing

    def fit(self, X, y=None, **fit_params) -> IndicatorImputer:
        """Fit the indicator imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.impute import SimpleImputer as _SklearnSimpleImputer

        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        X_np = _to_numpy(X)
        self.n_features_in_ = X_np.shape[1]

        # Identify features with missing values
        missing_mask = np.isnan(X_np)
        if self.only_missing:
            self.missing_features_ = list(
                np.where(missing_mask.any(axis=0))[0]
            )
        else:
            self.missing_features_ = list(range(X_np.shape[1]))

        # Fit the base imputer
        fill_value = self.fill_value if self.fill_value is not None else 0
        self._imputer = _SklearnSimpleImputer(
            strategy=self.base_strategy,
            fill_value=fill_value,
        )
        self._imputer.fit(X_np)
        self.statistics_ = self._imputer.statistics_

        self._is_fitted = True
        self._log(
            f"Fitted IndicatorImputer: {len(self.missing_features_)} features "
            f"with missing values"
        )
        return self

    def transform(self, X) -> Any:
        """Impute and add indicator columns.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_out : ndarray or DataFrame of shape (n_samples, n_features + n_indicators)
            Imputed data with binary indicator columns appended.
        """
        self._check_is_fitted()
        was_df = HAS_PANDAS and isinstance(X, pd.DataFrame)
        cols = _extract_column_names(X) if was_df else self._column_names

        X_np = _to_numpy(X)

        # Create indicator columns BEFORE imputation
        indicators = np.isnan(X_np[:, self.missing_features_]).astype(np.float64)

        # Impute
        X_imputed = self._imputer.transform(X_np)

        # Concatenate
        X_out = np.hstack([X_imputed, indicators])

        # Build output column names
        out_cols = None
        if cols is not None:
            indicator_names = [f"{cols[i]}_missing" for i in self.missing_features_]
            out_cols = list(cols) + indicator_names

        return _restore_dataframe(X_out, out_cols, was_df)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        names = input_features or self._column_names or []
        if names:
            indicator_names = [f"{names[i]}_missing" for i in self.missing_features_]
            return list(names) + indicator_names
        return []


class KNNImputer(EndgameEstimator, TransformerMixin):
    """K-Nearest Neighbors imputation with competition defaults.

    Wraps sklearn.impute.KNNImputer with defaults tuned for tabular
    competitions: n_neighbors=5, uniform weights, nan_euclidean distance.

    Parameters
    ----------
    n_neighbors : int, default=5
        Number of nearest neighbors to use.
    weights : str, default='uniform'
        Weight function for prediction: 'uniform' or 'distance'.
    metric : str, default='nan_euclidean'
        Distance metric for finding neighbors.
    add_indicator : bool, default=False
        If True, append binary missing-indicator columns.
    copy : bool, default=True
        If True, create a copy of X.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import KNNImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, 6], [5, np.nan]])
    >>> imp = KNNImputer(n_neighbors=2)
    >>> imp.fit_transform(X)
    array([[1. , 2. ],
           [3. , 3. ],
           [7. , 6. ],
           [5. , 4. ]])
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        weights: str = "uniform",
        metric: str = "nan_euclidean",
        add_indicator: bool = False,
        copy: bool = True,
        verbose: bool = False,
    ):
        super().__init__(verbose=verbose)
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.add_indicator = add_indicator
        self.copy = copy

    def fit(self, X, y=None, **fit_params) -> KNNImputer:
        """Fit the KNN imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.impute import KNNImputer as _SklearnKNNImputer

        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        self._imputer = _SklearnKNNImputer(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            add_indicator=self.add_indicator,
            copy=self.copy,
        )
        self._imputer.fit(_to_numpy(X))

        self.n_features_in_ = self._imputer.n_features_in_

        self._is_fitted = True
        self._log(f"Fitted KNNImputer with n_neighbors={self.n_neighbors}")
        return self

    def transform(self, X) -> Any:
        """Impute missing values using KNN.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : ndarray or DataFrame
            Imputed data.
        """
        self._check_is_fitted()
        was_df = HAS_PANDAS and isinstance(X, pd.DataFrame)
        cols = _extract_column_names(X) if was_df else self._column_names

        X_out = self._imputer.transform(_to_numpy(X))

        if self.add_indicator and cols is not None:
            indicator_cols = [f"{c}_missing" for c in cols]
            cols = list(cols) + indicator_cols

        return _restore_dataframe(X_out, cols, was_df)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        names = input_features or self._column_names or []
        if self.add_indicator and names:
            names = list(names) + [f"{c}_missing" for c in names]
        return names


class MICEImputer(EndgameEstimator, TransformerMixin):
    """Multiple Imputation by Chained Equations.

    Uses sklearn.impute.IterativeImputer with BayesianRidge as the default
    estimator, which is the standard MICE implementation. Iteratively models
    each feature as a function of all other features.

    Parameters
    ----------
    estimator : estimator, optional
        The estimator to predict each feature from all others. Default is
        BayesianRidge, which provides the standard MICE formulation.
    max_iter : int, default=10
        Maximum number of imputation rounds.
    tol : float, default=1e-3
        Convergence tolerance.
    initial_strategy : str, default='median'
        Strategy for initial imputation before iterating: 'mean', 'median',
        'most_frequent', 'constant'.
    sample_posterior : bool, default=False
        If True, sample from the predictive posterior for each imputation.
        Provides proper multiple imputations when True.
    random_state : int, default=42
        Random seed for reproducibility. Default set for deterministic results
        in competition settings.
    add_indicator : bool, default=False
        If True, append binary missing-indicator columns.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import MICEImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [5, 4]])
    >>> imp = MICEImputer(max_iter=10, random_state=42)
    >>> X_imputed = imp.fit_transform(X)
    """

    def __init__(
        self,
        estimator: Any | None = None,
        max_iter: int = 10,
        tol: float = 1e-3,
        initial_strategy: str = "median",
        sample_posterior: bool = False,
        random_state: int | None = 42,
        add_indicator: bool = False,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.estimator = estimator
        self.max_iter = max_iter
        self.tol = tol
        self.initial_strategy = initial_strategy
        self.sample_posterior = sample_posterior
        self.add_indicator = add_indicator

    def fit(self, X, y=None, **fit_params) -> MICEImputer:
        """Fit the MICE imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        estimator = self.estimator
        if estimator is None:
            from sklearn.linear_model import BayesianRidge

            estimator = BayesianRidge()

        self._imputer = IterativeImputer(
            estimator=estimator,
            max_iter=self.max_iter,
            tol=self.tol,
            initial_strategy=self.initial_strategy,
            sample_posterior=self.sample_posterior,
            random_state=self.random_state,
            add_indicator=self.add_indicator,
        )
        self._imputer.fit(_to_numpy(X))

        self.n_features_in_ = self._imputer.n_features_in_
        self.n_iter_ = self._imputer.n_iter_

        self._is_fitted = True
        self._log(
            f"Fitted MICEImputer: converged in {self.n_iter_} iterations"
        )
        return self

    def transform(self, X) -> Any:
        """Impute missing values using MICE.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : ndarray or DataFrame
            Imputed data.
        """
        self._check_is_fitted()
        was_df = HAS_PANDAS and isinstance(X, pd.DataFrame)
        cols = _extract_column_names(X) if was_df else self._column_names

        X_out = self._imputer.transform(_to_numpy(X))

        if self.add_indicator and cols is not None:
            indicator_cols = [f"{c}_missing" for c in cols]
            cols = list(cols) + indicator_cols

        return _restore_dataframe(X_out, cols, was_df)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        names = input_features or self._column_names or []
        if self.add_indicator and names:
            names = list(names) + [f"{c}_missing" for c in names]
        return names


class MissForestImputer(EndgameEstimator, TransformerMixin):
    """Random Forest-based iterative imputation (MissForest algorithm).

    Uses sklearn.impute.IterativeImputer with a RandomForestRegressor
    as the base estimator. This non-parametric approach handles non-linear
    relationships and interactions between features effectively.

    Parameters
    ----------
    n_estimators : int, default=100
        Number of trees in the random forest estimator.
    max_iter : int, default=10
        Maximum number of imputation rounds.
    max_depth : int or None, default=None
        Maximum depth of each tree. None means nodes are expanded until
        all leaves are pure or contain fewer than min_samples_split samples.
    max_features : str or float, default='sqrt'
        Number of features considered at each split.
    initial_strategy : str, default='median'
        Strategy for initial imputation before iterating.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs for the random forest. -1 uses all cores.
    add_indicator : bool, default=False
        If True, append binary missing-indicator columns.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during fit.
    n_iter_ : int
        Number of iterations performed.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import MissForestImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [5, 4]])
    >>> imp = MissForestImputer(n_estimators=50, random_state=42)
    >>> X_imputed = imp.fit_transform(X)
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_iter: int = 10,
        max_depth: int | None = None,
        max_features: str | float = "sqrt",
        initial_strategy: str = "median",
        random_state: int | None = 42,
        n_jobs: int = -1,
        add_indicator: bool = False,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.max_features = max_features
        self.initial_strategy = initial_strategy
        self.n_jobs = n_jobs
        self.add_indicator = add_indicator

    def fit(self, X, y=None, **fit_params) -> MissForestImputer:
        """Fit the MissForest imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer

        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        rf_estimator = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )

        self._imputer = IterativeImputer(
            estimator=rf_estimator,
            max_iter=self.max_iter,
            initial_strategy=self.initial_strategy,
            random_state=self.random_state,
            add_indicator=self.add_indicator,
        )
        self._imputer.fit(_to_numpy(X))

        self.n_features_in_ = self._imputer.n_features_in_
        self.n_iter_ = self._imputer.n_iter_

        self._is_fitted = True
        self._log(
            f"Fitted MissForestImputer with {self.n_estimators} trees, "
            f"converged in {self.n_iter_} iterations"
        )
        return self

    def transform(self, X) -> Any:
        """Impute missing values using MissForest.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : ndarray or DataFrame
            Imputed data.
        """
        self._check_is_fitted()
        was_df = HAS_PANDAS and isinstance(X, pd.DataFrame)
        cols = _extract_column_names(X) if was_df else self._column_names

        X_out = self._imputer.transform(_to_numpy(X))

        if self.add_indicator and cols is not None:
            indicator_cols = [f"{c}_missing" for c in cols]
            cols = list(cols) + indicator_cols

        return _restore_dataframe(X_out, cols, was_df)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        names = input_features or self._column_names or []
        if self.add_indicator and names:
            names = list(names) + [f"{c}_missing" for c in names]
        return names


class AutoImputer(EndgameEstimator, TransformerMixin):
    """Automatic imputation strategy selection based on missingness patterns.

    Analyzes the missingness structure in the data and selects an appropriate
    imputation strategy:

    - <5% missing -> SimpleImputer (fast, sufficient for low missingness)
    - 5-30% missing -> KNNImputer (captures local structure)
    - >30% missing -> MICEImputer (models complex dependencies)

    Also performs an approximate Little's MCAR test to characterize the
    missingness mechanism (MCAR, MAR, or MNAR).

    Parameters
    ----------
    strategy : str, default='auto'
        Imputation strategy:
        - 'auto': Automatically select based on missingness percentage
        - 'simple': Force SimpleImputer
        - 'knn': Force KNNImputer
        - 'mice': Force MICEImputer
        - 'missforest': Force MissForestImputer
    low_threshold : float, default=0.05
        Missingness fraction below which SimpleImputer is used (in auto mode).
    high_threshold : float, default=0.30
        Missingness fraction above which MICEImputer is used (in auto mode).
    random_state : int, default=42
        Random seed for reproducibility.
    add_indicator : bool, default=False
        If True, append binary missing-indicator columns.
    verbose : bool, default=False
        Enable verbose output.

    Attributes
    ----------
    missingness_fraction_ : float
        Overall fraction of missing values in the training data.
    missingness_type_ : str
        Detected missingness mechanism: 'MCAR', 'MAR', or 'MNAR'.
    selected_strategy_ : str
        The imputation strategy that was selected.
    imputer_ : estimator
        The fitted imputer instance.
    n_features_in_ : int
        Number of features seen during fit.

    Examples
    --------
    >>> import numpy as np
    >>> from endgame.preprocessing.imputation import AutoImputer
    >>> X = np.array([[1, 2], [np.nan, 3], [7, np.nan], [5, 4]])
    >>> imp = AutoImputer(strategy='auto', random_state=42)
    >>> X_imputed = imp.fit_transform(X)
    >>> imp.selected_strategy_
    'knn'
    """

    def __init__(
        self,
        strategy: str = "auto",
        low_threshold: float = 0.05,
        high_threshold: float = 0.30,
        random_state: int | None = 42,
        add_indicator: bool = False,
        verbose: bool = False,
    ):
        super().__init__(random_state=random_state, verbose=verbose)
        self.strategy = strategy
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.add_indicator = add_indicator

    @staticmethod
    def _littles_mcar_test_approx(X: np.ndarray) -> tuple[str, float]:
        """Approximate Little's MCAR test using correlation between missingness
        patterns and observed values.

        This is a lightweight approximation. The full Little's MCAR test uses a
        chi-squared statistic on the EM-estimated covariance matrix; here we
        instead compute point-biserial correlations between each feature's
        missingness indicator and all other observed features, then aggregate.

        A high aggregate correlation suggests the missingness depends on
        observed values (MAR or MNAR). If correlations are uniformly low,
        missingness is likely MCAR.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data with missing values (np.nan).

        Returns
        -------
        missingness_type : str
            One of 'MCAR', 'MAR', 'MNAR'.
        test_statistic : float
            Aggregate correlation score. Higher values indicate structured
            missingness.
        """
        n_samples, n_features = X.shape
        missing_mask = np.isnan(X)

        # Features that have some (but not all) missing values
        cols_with_missing = []
        for j in range(n_features):
            n_miss = missing_mask[:, j].sum()
            if 0 < n_miss < n_samples:
                cols_with_missing.append(j)

        if len(cols_with_missing) == 0:
            return "MCAR", 0.0

        correlations = []

        for j in cols_with_missing:
            indicator = missing_mask[:, j].astype(np.float64)
            for k in range(n_features):
                if k == j:
                    continue
                # Use only rows where feature k is observed
                observed_mask = ~missing_mask[:, k]
                if observed_mask.sum() < 10:
                    continue

                vals = X[observed_mask, k]
                ind = indicator[observed_mask]

                # Need variance in both to compute correlation
                if np.std(vals) < 1e-12 or np.std(ind) < 1e-12:
                    continue

                corr = np.abs(np.corrcoef(vals, ind)[0, 1])
                if not np.isnan(corr):
                    correlations.append(corr)

        if len(correlations) == 0:
            return "MCAR", 0.0

        mean_corr = float(np.mean(correlations))
        max_corr = float(np.max(correlations))

        # Heuristic thresholds:
        # - mean_corr < 0.05 => MCAR (no systematic relationship)
        # - mean_corr < 0.15 and max_corr < 0.3 => MAR (mild dependence)
        # - otherwise => MNAR (strong dependence on observed or unobserved)
        if mean_corr < 0.05:
            return "MCAR", mean_corr
        elif mean_corr < 0.15 and max_corr < 0.30:
            return "MAR", mean_corr
        else:
            return "MNAR", mean_corr

    def fit(self, X, y=None, **fit_params) -> AutoImputer:
        """Fit the auto imputer.

        Analyzes missingness patterns and selects the appropriate strategy,
        then fits the chosen imputer.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : ignored

        Returns
        -------
        self
        """
        self._column_names = _extract_column_names(X)
        self._was_dataframe = HAS_PANDAS and isinstance(X, pd.DataFrame)

        X_np = _to_numpy(X)
        n_samples, n_features = X_np.shape
        self.n_features_in_ = n_features

        # Compute missingness statistics
        total_cells = n_samples * n_features
        n_missing = np.isnan(X_np).sum()
        self.missingness_fraction_ = float(n_missing / total_cells) if total_cells > 0 else 0.0

        # Approximate Little's MCAR test
        self.missingness_type_, self._test_statistic = self._littles_mcar_test_approx(X_np)

        # Select strategy
        if self.strategy == "auto":
            if self.missingness_fraction_ < self.low_threshold:
                self.selected_strategy_ = "simple"
            elif self.missingness_fraction_ < self.high_threshold:
                self.selected_strategy_ = "knn"
            else:
                self.selected_strategy_ = "mice"
        else:
            self.selected_strategy_ = self.strategy

        self._log(
            f"Missingness: {self.missingness_fraction_:.1%} "
            f"(type={self.missingness_type_}), "
            f"selected strategy: {self.selected_strategy_}"
        )

        # Build and fit the selected imputer
        if self.selected_strategy_ == "simple":
            self.imputer_ = SimpleImputer(
                strategy="median",
                add_indicator=self.add_indicator,
                verbose=self.verbose,
            )
        elif self.selected_strategy_ == "knn":
            self.imputer_ = KNNImputer(
                n_neighbors=5,
                add_indicator=self.add_indicator,
                verbose=self.verbose,
            )
        elif self.selected_strategy_ == "mice":
            self.imputer_ = MICEImputer(
                max_iter=10,
                random_state=self.random_state,
                add_indicator=self.add_indicator,
                verbose=self.verbose,
            )
        elif self.selected_strategy_ == "missforest":
            self.imputer_ = MissForestImputer(
                random_state=self.random_state,
                add_indicator=self.add_indicator,
                verbose=self.verbose,
            )
        else:
            raise ValueError(
                f"Unknown strategy: '{self.strategy}'. "
                "Use 'auto', 'simple', 'knn', 'mice', or 'missforest'."
            )

        self.imputer_.fit(X, y)

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Impute missing values using the selected strategy.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data with missing values.

        Returns
        -------
        X_imputed : ndarray or DataFrame
            Imputed data.
        """
        self._check_is_fitted()
        return self.imputer_.transform(X)

    def get_feature_names_out(
        self, input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()
        return self.imputer_.get_feature_names_out(input_features)
