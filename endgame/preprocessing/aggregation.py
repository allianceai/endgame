"""Aggregation and interaction feature generation."""

from collections.abc import Sequence
from typing import Any

from endgame.core.base import PolarsTransformer
from endgame.core.polars_ops import (
    HAS_PANDAS,
    HAS_POLARS,
    compute_statistics,
    infer_numeric_columns,
)

if HAS_POLARS:
    import polars as pl
if HAS_PANDAS:
    pass


class AutoAggregator(PolarsTransformer):
    """Generates "Magic Feature" aggregations used in winning solutions.

    Creates group-level statistics that capture relationships between
    entities. Key technique from Optiver 1st place and many tabular wins.

    Parameters
    ----------
    group_cols : List[str]
        Columns to group by (e.g., ['customer_id', 'store_id']).
    agg_cols : List[str], optional
        Columns to aggregate (e.g., ['amount', 'quantity']).
        If None, aggregates all numeric columns.
    methods : List[str], default=['mean', 'std', 'min', 'max']
        Aggregation methods: 'mean', 'std', 'min', 'max', 'sum',
        'count', 'median', 'skew', 'kurtosis', 'first', 'last', 'nunique'.
    rank_features : bool, default=True
        Whether to compute rank features within groups.
        Key technique from Optiver 1st place solution.
    diff_features : bool, default=False
        Whether to compute difference from group mean.
    ratio_features : bool, default=False
        Whether to compute ratio to group mean.
    prefix : str, optional
        Prefix for generated feature names.

    Examples
    --------
    >>> agg = AutoAggregator(
    ...     group_cols=['customer_id'],
    ...     agg_cols=['amount'],
    ...     methods=['mean', 'std', 'skew'],
    ...     rank_features=True
    ... )
    >>> X_agg = agg.fit_transform(X)
    """

    def __init__(
        self,
        group_cols: list[str],
        agg_cols: list[str] | None = None,
        methods: Sequence[str] = ("mean", "std", "min", "max"),
        rank_features: bool = True,
        diff_features: bool = False,
        ratio_features: bool = False,
        prefix: str | None = None,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.group_cols = group_cols
        self.agg_cols = agg_cols
        self.methods = list(methods)
        self.rank_features = rank_features
        self.diff_features = diff_features
        self.ratio_features = ratio_features
        self.prefix = prefix

        self._agg_stats: pl.DataFrame | None = None
        self._target_agg_cols: list[str] = []
        self._new_feature_names: list[str] = []

    def fit(self, X, y=None, **fit_params) -> "AutoAggregator":
        """Compute aggregation statistics from training data.

        Parameters
        ----------
        X : array-like
            Training data.
        y : array-like, optional
            Ignored.

        Returns
        -------
        self
        """
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        # Validate group columns exist
        for col in self.group_cols:
            if col not in df.columns:
                raise ValueError(f"Group column '{col}' not found in data")

        # Determine columns to aggregate
        if self.agg_cols is None:
            self._target_agg_cols = [
                c for c in infer_numeric_columns(X)
                if c not in self.group_cols
            ]
        else:
            self._target_agg_cols = [
                c for c in self.agg_cols
                if c in df.columns and c not in self.group_cols
            ]

        self._log(f"Aggregating {len(self._target_agg_cols)} columns by {self.group_cols}")

        # Compute aggregations
        agg_lf = compute_statistics(
            lf,
            group_cols=self.group_cols,
            agg_cols=self._target_agg_cols,
            methods=self.methods,
        )

        self._agg_stats = agg_lf.collect()
        self._new_feature_names = [
            c for c in self._agg_stats.columns if c not in self.group_cols
        ]

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Apply aggregation features to data.

        Parameters
        ----------
        X : array-like
            Data to transform.

        Returns
        -------
        X_transformed : array-like
            Original data with aggregation features added.
        """
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        # Join aggregation statistics
        result = df.join(
            self._agg_stats,
            on=self.group_cols,
            how="left",
        )

        # Add rank features
        if self.rank_features:
            for col in self._target_agg_cols:
                rank_col = f"{col}_rank" if self.prefix is None else f"{self.prefix}_{col}_rank"
                result = result.with_columns(
                    pl.col(col).rank().over(self.group_cols).alias(rank_col)
                )
                # Percentage rank
                pct_rank_col = f"{col}_pct_rank" if self.prefix is None else f"{self.prefix}_{col}_pct_rank"
                result = result.with_columns(
                    (pl.col(rank_col) / pl.col(rank_col).max().over(self.group_cols)).alias(pct_rank_col)
                )

        # Add difference features
        if self.diff_features:
            for col in self._target_agg_cols:
                mean_col = f"{col}_mean"
                if mean_col in result.columns:
                    diff_col = f"{col}_diff_from_mean" if self.prefix is None else f"{self.prefix}_{col}_diff"
                    result = result.with_columns(
                        (pl.col(col) - pl.col(mean_col)).alias(diff_col)
                    )

        # Add ratio features
        if self.ratio_features:
            for col in self._target_agg_cols:
                mean_col = f"{col}_mean"
                if mean_col in result.columns:
                    ratio_col = f"{col}_ratio_to_mean" if self.prefix is None else f"{self.prefix}_{col}_ratio"
                    result = result.with_columns(
                        pl.when(pl.col(mean_col) != 0)
                        .then(pl.col(col) / pl.col(mean_col))
                        .otherwise(1.0)
                        .alias(ratio_col)
                    )

        return self._from_lazyframe(result.lazy())

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names including generated aggregations."""
        self._check_is_fitted()

        if input_features is None:
            input_features = self._feature_names_in or []

        output_features = list(input_features) + self._new_feature_names

        # Add rank features
        if self.rank_features:
            for col in self._target_agg_cols:
                output_features.append(f"{col}_rank")
                output_features.append(f"{col}_pct_rank")

        if self.diff_features:
            for col in self._target_agg_cols:
                output_features.append(f"{col}_diff_from_mean")

        if self.ratio_features:
            for col in self._target_agg_cols:
                output_features.append(f"{col}_ratio_to_mean")

        return output_features


class InteractionFeatures(PolarsTransformer):
    """Generates interaction features between specified columns.

    Creates arithmetic combinations (multiply, divide, add, subtract)
    between pairs of numeric features.

    Parameters
    ----------
    interaction_pairs : List[Tuple[str, str]], optional
        Specific pairs to create. If None, creates all pairs.
    operations : List[str], default=['multiply', 'divide']
        Operations: 'multiply', 'divide', 'add', 'subtract'.
    max_interactions : int, default=100
        Maximum number of interactions to create.
    include_cols : List[str], optional
        Only consider these columns for interactions.
    exclude_cols : List[str], optional
        Exclude these columns from interactions.

    Examples
    --------
    >>> inter = InteractionFeatures(
    ...     operations=['multiply', 'divide'],
    ...     max_interactions=50
    ... )
    >>> X_inter = inter.fit_transform(X)
    """

    def __init__(
        self,
        interaction_pairs: list[tuple[str, str]] | None = None,
        operations: Sequence[str] = ("multiply", "divide"),
        max_interactions: int = 100,
        include_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        output_format: str = "auto",
        random_state: int | None = None,
        verbose: bool = False,
    ):
        super().__init__(
            output_format=output_format,
            random_state=random_state,
            verbose=verbose,
        )
        self.interaction_pairs = interaction_pairs
        self.operations = list(operations)
        self.max_interactions = max_interactions
        self.include_cols = include_cols
        self.exclude_cols = exclude_cols

        self._pairs: list[tuple[str, str]] = []
        self._new_feature_names: list[str] = []

    def fit(self, X, y=None, **fit_params) -> "InteractionFeatures":
        """Determine interaction pairs from training data."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        # Determine columns for interactions
        numeric_cols = infer_numeric_columns(X)

        if self.include_cols:
            numeric_cols = [c for c in numeric_cols if c in self.include_cols]

        if self.exclude_cols:
            numeric_cols = [c for c in numeric_cols if c not in self.exclude_cols]

        # Generate pairs
        if self.interaction_pairs:
            self._pairs = [
                (a, b) for a, b in self.interaction_pairs
                if a in df.columns and b in df.columns
            ]
        else:
            # All pairs
            self._pairs = []
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1:]:
                    if len(self._pairs) * len(self.operations) >= self.max_interactions:
                        break
                    self._pairs.append((col1, col2))
                if len(self._pairs) * len(self.operations) >= self.max_interactions:
                    break

        self._log(f"Created {len(self._pairs)} interaction pairs")

        # Generate feature names
        self._new_feature_names = []
        for col1, col2 in self._pairs:
            for op in self.operations:
                op_symbol = {"multiply": "*", "divide": "/", "add": "+", "subtract": "-"}[op]
                self._new_feature_names.append(f"{col1}{op_symbol}{col2}")

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Create interaction features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        new_cols = []

        for col1, col2 in self._pairs:
            for op in self.operations:
                if op == "multiply":
                    new_col = (pl.col(col1) * pl.col(col2)).alias(f"{col1}*{col2}")
                elif op == "divide":
                    new_col = (
                        pl.when(pl.col(col2) != 0)
                        .then(pl.col(col1) / pl.col(col2))
                        .otherwise(0.0)
                        .alias(f"{col1}/{col2}")
                    )
                elif op == "add":
                    new_col = (pl.col(col1) + pl.col(col2)).alias(f"{col1}+{col2}")
                elif op == "subtract":
                    new_col = (pl.col(col1) - pl.col(col2)).alias(f"{col1}-{col2}")
                else:
                    raise ValueError(f"Unknown operation: {op}")

                new_cols.append(new_col)

        if new_cols:
            result = df.with_columns(new_cols)
        else:
            result = df

        return self._from_lazyframe(result.lazy())

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()

        if input_features is None:
            input_features = self._feature_names_in or []

        return list(input_features) + self._new_feature_names


class RankFeatures(PolarsTransformer):
    """Compute rank-based features.

    Converts numeric values to ranks, which can be more robust to
    outliers and non-linear relationships.

    Parameters
    ----------
    cols : List[str], optional
        Columns to rank. If None, ranks all numeric columns.
    method : str, default='average'
        Ranking method: 'average', 'min', 'max', 'dense', 'ordinal'.
    pct : bool, default=True
        Whether to return percentile ranks (0-1).
    suffix : str, default='_rank'
        Suffix for ranked column names.

    Examples
    --------
    >>> ranker = RankFeatures(pct=True)
    >>> X_ranked = ranker.fit_transform(X)
    """

    def __init__(
        self,
        cols: list[str] | None = None,
        method: str = "average",
        pct: bool = True,
        suffix: str = "_rank",
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
        self.method = method
        self.pct = pct
        self.suffix = suffix

        self._target_cols: list[str] = []

    def fit(self, X, y=None, **fit_params) -> "RankFeatures":
        """Identify columns to rank."""
        lf = self._to_lazyframe(X, store_metadata=True)
        df = lf.collect()

        if self.cols is None:
            self._target_cols = infer_numeric_columns(X)
        else:
            self._target_cols = [c for c in self.cols if c in df.columns]

        self._is_fitted = True
        return self

    def transform(self, X) -> Any:
        """Compute rank features."""
        self._check_is_fitted()

        lf = self._to_lazyframe(X)
        df = lf.collect()

        rank_cols = []

        for col in self._target_cols:
            rank_expr = pl.col(col).rank(method=self.method)

            if self.pct:
                # Normalize to 0-1
                rank_expr = rank_expr / pl.col(col).count()

            rank_cols.append(rank_expr.alias(f"{col}{self.suffix}"))

        if rank_cols:
            result = df.with_columns(rank_cols)
        else:
            result = df

        return self._from_lazyframe(result.lazy())

    def get_feature_names_out(
        self,
        input_features: list[str] | None = None,
    ) -> list[str]:
        """Get output feature names."""
        self._check_is_fitted()

        if input_features is None:
            input_features = self._feature_names_in or []

        rank_names = [f"{col}{self.suffix}" for col in self._target_cols]
        return list(input_features) + rank_names
