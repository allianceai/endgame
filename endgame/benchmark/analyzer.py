from __future__ import annotations

"""Results analysis and visualization for benchmarks.

Provides tools for analyzing, ranking, and visualizing benchmark results.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

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

from endgame.benchmark.tracker import ExperimentTracker


class RankingMethod(str, Enum):
    """Methods for ranking models."""
    MEAN_SCORE = "mean_score"       # Mean metric value across datasets
    MEAN_RANK = "mean_rank"         # Mean rank across datasets
    WIN_COUNT = "win_count"         # Number of datasets where model was best
    BORDA_COUNT = "borda_count"     # Borda count ranking
    FRIEDMAN = "friedman"           # Friedman test ranking


@dataclass
class ModelComparison:
    """Comparison between two models.

    Attributes
    ----------
    model_a : str
        Name of first model.
    model_b : str
        Name of second model.
    wins_a : int
        Datasets where model A won.
    wins_b : int
        Datasets where model B won.
    ties : int
        Datasets where models tied.
    p_value : float
        P-value from statistical test.
    effect_size : float
        Effect size (Cohen's d or similar).
    is_significant : bool
        Whether difference is statistically significant.
    """
    model_a: str
    model_b: str
    wins_a: int = 0
    wins_b: int = 0
    ties: int = 0
    p_value: float = 1.0
    effect_size: float = 0.0
    is_significant: bool = False


class ResultsAnalyzer:
    """Analyze and compare benchmark results.

    Provides methods for:
    - Ranking models across datasets
    - Statistical significance testing
    - Critical difference diagrams
    - Performance profiles
    - Meta-feature correlation analysis

    Parameters
    ----------
    tracker : ExperimentTracker
        Tracker containing experiment results.
    metric : str, default="accuracy"
        Primary metric for comparisons.
    higher_is_better : bool, default=True
        Whether higher metric values are better.
    significance_level : float, default=0.05
        Alpha level for statistical tests.

    Examples
    --------
    >>> analyzer = ResultsAnalyzer(tracker, metric="accuracy")
    >>> rankings = analyzer.rank_models()
    >>> print(rankings)
    >>>
    >>> # Statistical comparison
    >>> comparison = analyzer.compare_models("RF", "XGBoost")
    >>> print(f"P-value: {comparison.p_value}")
    """

    def __init__(
        self,
        tracker: ExperimentTracker,
        metric: str = "accuracy",
        higher_is_better: bool = True,
        significance_level: float = 0.05,
    ):
        self.tracker = tracker
        self.metric = metric
        self.higher_is_better = higher_is_better
        self.significance_level = significance_level

        self._df = None
        self._pivot_table = None

    @classmethod
    def from_pivot(
        cls,
        pivot: dict[str, dict[str, float]],
        metric: str = "accuracy",
        higher_is_better: bool = True,
        significance_level: float = 0.05,
    ) -> ResultsAnalyzer:
        """Create a ResultsAnalyzer from a pivot dict.

        Convenience factory for external experiment systems that already have
        results in {dataset: {method: score}} form.

        Parameters
        ----------
        pivot : Dict[str, Dict[str, float]]
            Mapping of dataset_name -> {method_name: score}.
        metric : str, default="accuracy"
            Name of the metric the scores represent.
        higher_is_better : bool, default=True
            Whether higher metric values are better.
        significance_level : float, default=0.05
            Alpha level for statistical tests.

        Returns
        -------
        ResultsAnalyzer
            Analyzer ready for ranking, comparison, and statistical tests.

        Examples
        --------
        >>> pivot = {
        ...     "iris": {"RF": 0.95, "XGB": 0.96},
        ...     "wine": {"RF": 0.97, "XGB": 0.95},
        ... }
        >>> analyzer = ResultsAnalyzer.from_pivot(pivot, metric="accuracy")
        >>> print(analyzer.summary_table())
        """
        tracker = ExperimentTracker(name="from_pivot")
        for dataset_name, method_scores in pivot.items():
            for method_name, score in method_scores.items():
                if score is not None:
                    tracker.log_experiment(
                        dataset_name=dataset_name,
                        model_name=method_name,
                        metrics={metric: score},
                    )
        return cls(
            tracker=tracker,
            metric=metric,
            higher_is_better=higher_is_better,
            significance_level=significance_level,
        )

    @property
    def df(self):
        """Get results as DataFrame."""
        if self._df is None:
            self._df = self.tracker.to_dataframe()
        return self._df

    def get_pivot_table(self, metric: str | None = None):
        """Get pivot table of models vs datasets.

        Parameters
        ----------
        metric : str, optional
            Metric to use. If None, uses default metric.

        Returns
        -------
        DataFrame
            Pivot table with models as rows, datasets as columns.
        """
        metric = metric or self.metric
        metric_col = f"metric_{metric}"

        df = self.df

        # Filter to successful experiments
        if HAS_POLARS and isinstance(df, pl.DataFrame):
            df = df.filter(pl.col("status") == "success")

            # Create pivot table
            pivot = df.pivot(
                values=metric_col,
                index="model_name",
                on="dataset_name",
            )
        else:
            df = df[df["status"] == "success"]
            pivot = df.pivot_table(
                values=metric_col,
                index="model_name",
                columns="dataset_name",
                aggfunc="mean",
            )

        return pivot

    def rank_models(
        self,
        method: str | RankingMethod = RankingMethod.MEAN_RANK,
        metric: str | None = None,
    ) -> dict[str, float]:
        """Rank models across all datasets.

        Parameters
        ----------
        method : RankingMethod
            Ranking method to use.
        metric : str, optional
            Metric to rank by.

        Returns
        -------
        Dict[str, float]
            Model name to rank/score mapping (sorted).
        """
        method = RankingMethod(method) if isinstance(method, str) else method
        metric = metric or self.metric

        pivot = self.get_pivot_table(metric)

        if HAS_POLARS and isinstance(pivot, pl.DataFrame):
            models = pivot["model_name"].to_list()
            data = pivot.drop("model_name").to_numpy()
        else:
            models = list(pivot.index)
            data = pivot.values

        # Handle NaN values
        data = np.nan_to_num(data, nan=np.nanmin(data) - 1 if self.higher_is_better else np.nanmax(data) + 1)

        if method == RankingMethod.MEAN_SCORE:
            scores = np.nanmean(data, axis=1)
            if not self.higher_is_better:
                scores = -scores

        elif method == RankingMethod.MEAN_RANK:
            # Compute rank for each dataset
            ranks = np.zeros_like(data)
            for j in range(data.shape[1]):
                col = data[:, j]
                if self.higher_is_better:
                    ranks[:, j] = stats.rankdata(-col)  # Higher is better = lower rank
                else:
                    ranks[:, j] = stats.rankdata(col)
            scores = -np.nanmean(ranks, axis=1)  # Negative because lower rank is better

        elif method == RankingMethod.WIN_COUNT:
            wins = np.zeros(len(models))
            for j in range(data.shape[1]):
                col = data[:, j]
                if self.higher_is_better:
                    best_idx = np.argmax(col)
                else:
                    best_idx = np.argmin(col)
                wins[best_idx] += 1
            scores = wins

        elif method == RankingMethod.BORDA_COUNT:
            # Borda count: points based on position in each ranking
            n_models = len(models)
            points = np.zeros(len(models))

            for j in range(data.shape[1]):
                col = data[:, j]
                if self.higher_is_better:
                    order = np.argsort(-col)
                else:
                    order = np.argsort(col)

                for rank, idx in enumerate(order):
                    points[idx] += n_models - rank - 1

            scores = points

        elif method == RankingMethod.FRIEDMAN:
            # Use Friedman ranks
            ranks = np.zeros_like(data)
            for j in range(data.shape[1]):
                col = data[:, j]
                if self.higher_is_better:
                    ranks[:, j] = stats.rankdata(-col)
                else:
                    ranks[:, j] = stats.rankdata(col)
            scores = -np.nanmean(ranks, axis=1)

        else:
            raise ValueError(f"Unknown ranking method: {method}")

        # Create sorted dictionary
        ranking = dict(sorted(
            zip(models, scores),
            key=lambda x: x[1],
            reverse=True,
        ))

        return ranking

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        metric: str | None = None,
        test: str = "wilcoxon",
    ) -> ModelComparison:
        """Compare two models statistically.

        Parameters
        ----------
        model_a : str
            Name of first model.
        model_b : str
            Name of second model.
        metric : str, optional
            Metric to compare on.
        test : str, default="wilcoxon"
            Statistical test: "wilcoxon", "paired_t", "sign".

        Returns
        -------
        ModelComparison
            Comparison results.
        """
        metric = metric or self.metric
        pivot = self.get_pivot_table(metric)

        if HAS_POLARS and isinstance(pivot, pl.DataFrame):
            pivot_pd = pivot.to_pandas().set_index("model_name")
        else:
            pivot_pd = pivot

        if model_a not in pivot_pd.index or model_b not in pivot_pd.index:
            raise ValueError("Model not found in results")

        scores_a = pivot_pd.loc[model_a].values
        scores_b = pivot_pd.loc[model_b].values

        # Remove NaN pairs
        valid_mask = ~(np.isnan(scores_a) | np.isnan(scores_b))
        scores_a = scores_a[valid_mask]
        scores_b = scores_b[valid_mask]

        if len(scores_a) < 2:
            return ModelComparison(model_a=model_a, model_b=model_b)

        # Count wins
        if self.higher_is_better:
            wins_a = np.sum(scores_a > scores_b)
            wins_b = np.sum(scores_b > scores_a)
        else:
            wins_a = np.sum(scores_a < scores_b)
            wins_b = np.sum(scores_b < scores_a)
        ties = len(scores_a) - wins_a - wins_b

        # Statistical test
        if test == "wilcoxon":
            try:
                stat, p_value = stats.wilcoxon(scores_a, scores_b)
            except ValueError:
                p_value = 1.0
        elif test == "paired_t":
            stat, p_value = stats.ttest_rel(scores_a, scores_b)
        elif test == "sign":
            # Sign test
            diff = scores_a - scores_b
            if self.higher_is_better:
                n_plus = np.sum(diff > 0)
            else:
                n_plus = np.sum(diff < 0)
            n = np.sum(diff != 0)
            p_value = stats.binom_test(n_plus, n, 0.5) if n > 0 else 1.0
        else:
            raise ValueError(f"Unknown test: {test}")

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.var(scores_a) + np.var(scores_b)) / 2)
        if pooled_std > 0:
            effect_size = (np.mean(scores_a) - np.mean(scores_b)) / pooled_std
        else:
            effect_size = 0.0

        return ModelComparison(
            model_a=model_a,
            model_b=model_b,
            wins_a=int(wins_a),
            wins_b=int(wins_b),
            ties=int(ties),
            p_value=float(p_value),
            effect_size=float(effect_size),
            is_significant=p_value < self.significance_level,
        )

    def friedman_test(self, metric: str | None = None) -> tuple[float, float]:
        """Perform Friedman test across all models.

        Parameters
        ----------
        metric : str, optional
            Metric to test on.

        Returns
        -------
        Tuple[float, float]
            (chi2 statistic, p-value)
        """
        metric = metric or self.metric
        pivot = self.get_pivot_table(metric)

        if HAS_POLARS and isinstance(pivot, pl.DataFrame):
            data = pivot.drop("model_name").to_numpy()
        else:
            data = pivot.values

        # Remove datasets with missing values
        valid_cols = ~np.any(np.isnan(data), axis=0)
        data = data[:, valid_cols]

        if data.shape[1] < 2:
            return 0.0, 1.0

        # Friedman test
        chi2, p_value = stats.friedmanchisquare(*data)

        return float(chi2), float(p_value)

    def nemenyi_critical_difference(
        self,
        alpha: float = 0.05,
    ) -> float:
        """Compute critical difference for Nemenyi test.

        Parameters
        ----------
        alpha : float, default=0.05
            Significance level.

        Returns
        -------
        float
            Critical difference value.
        """
        pivot = self.get_pivot_table()

        if HAS_POLARS and isinstance(pivot, pl.DataFrame):
            n_models = len(pivot["model_name"])
            n_datasets = len(pivot.columns) - 1
        else:
            n_models = len(pivot.index)
            n_datasets = len(pivot.columns)

        # Critical values for Nemenyi test (q_alpha)
        # These are approximations for common alpha levels
        q_alpha_table = {
            2: {0.05: 1.960, 0.10: 1.645},
            3: {0.05: 2.343, 0.10: 2.052},
            4: {0.05: 2.569, 0.10: 2.291},
            5: {0.05: 2.728, 0.10: 2.459},
            6: {0.05: 2.850, 0.10: 2.589},
            7: {0.05: 2.949, 0.10: 2.693},
            8: {0.05: 3.031, 0.10: 2.780},
            9: {0.05: 3.102, 0.10: 2.855},
            10: {0.05: 3.164, 0.10: 2.920},
        }

        k = min(n_models, 10)
        q_alpha = q_alpha_table.get(k, {}).get(alpha, 2.569)  # Default to k=4

        cd = q_alpha * np.sqrt(n_models * (n_models + 1) / (6 * n_datasets))

        return cd

    def get_model_summary(
        self,
        model_name: str,
        metric: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed summary for a specific model.

        Parameters
        ----------
        model_name : str
            Name of the model.
        metric : str, optional
            Metric to summarize.

        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        metric = metric or self.metric
        metric_col = f"metric_{metric}"

        df = self.df

        if HAS_POLARS and isinstance(df, pl.DataFrame):
            model_df = df.filter(
                (pl.col("model_name") == model_name) &
                (pl.col("status") == "success")
            )

            if len(model_df) == 0:
                return {"error": "No successful experiments for this model"}

            scores = model_df[metric_col].to_numpy()
            datasets = model_df["dataset_name"].to_list()
            fit_times = model_df["fit_time"].to_numpy()
        else:
            model_df = df[(df["model_name"] == model_name) & (df["status"] == "success")]

            if len(model_df) == 0:
                return {"error": "No successful experiments for this model"}

            scores = model_df[metric_col].values
            datasets = model_df["dataset_name"].tolist()
            fit_times = model_df["fit_time"].values

        # Remove NaN
        valid = ~np.isnan(scores)
        scores = scores[valid]

        return {
            "model_name": model_name,
            "metric": metric,
            "n_datasets": len(datasets),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "median_score": float(np.median(scores)),
            "mean_fit_time": float(np.mean(fit_times)),
            "total_fit_time": float(np.sum(fit_times)),
            "best_dataset": datasets[np.argmax(scores)] if len(scores) > 0 else None,
            "worst_dataset": datasets[np.argmin(scores)] if len(scores) > 0 else None,
        }

    def get_dataset_summary(
        self,
        dataset_name: str,
        metric: str | None = None,
    ) -> dict[str, Any]:
        """Get detailed summary for a specific dataset.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset.
        metric : str, optional
            Metric to summarize.

        Returns
        -------
        Dict[str, Any]
            Summary statistics.
        """
        metric = metric or self.metric
        metric_col = f"metric_{metric}"

        df = self.df

        if HAS_POLARS and isinstance(df, pl.DataFrame):
            ds_df = df.filter(
                (pl.col("dataset_name") == dataset_name) &
                (pl.col("status") == "success")
            )

            if len(ds_df) == 0:
                return {"error": "No successful experiments for this dataset"}

            scores = ds_df[metric_col].to_numpy()
            models = ds_df["model_name"].to_list()
            n_samples = int(ds_df["n_samples"][0])
            n_features = int(ds_df["n_features"][0])
            task_type = str(ds_df["task_type"][0])
        else:
            ds_df = df[(df["dataset_name"] == dataset_name) & (df["status"] == "success")]

            if len(ds_df) == 0:
                return {"error": "No successful experiments for this dataset"}

            scores = ds_df[metric_col].values
            models = ds_df["model_name"].tolist()
            n_samples = int(ds_df["n_samples"].iloc[0])
            n_features = int(ds_df["n_features"].iloc[0])
            task_type = str(ds_df["task_type"].iloc[0])

        # Remove NaN
        valid = ~np.isnan(scores)
        scores = scores[valid]
        models_valid = [m for m, v in zip(models, valid) if v]

        best_idx = np.argmax(scores) if self.higher_is_better else np.argmin(scores)

        return {
            "dataset_name": dataset_name,
            "metric": metric,
            "n_models": len(models),
            "n_samples": n_samples,
            "n_features": n_features,
            "task_type": task_type,
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "best_score": float(np.max(scores)) if self.higher_is_better else float(np.min(scores)),
            "best_model": models_valid[best_idx] if len(models_valid) > 0 else None,
            "score_range": float(np.max(scores) - np.min(scores)),
        }

    def summary_table(
        self,
        metric: str | None = None,
        sort_by: str = "mean_rank",
    ) -> str:
        """Generate formatted summary table.

        Parameters
        ----------
        metric : str, optional
            Metric to summarize.
        sort_by : str, default="mean_rank"
            Column to sort by.

        Returns
        -------
        str
            Formatted table string.
        """
        metric = metric or self.metric

        rankings = self.rank_models(RankingMethod.MEAN_RANK, metric)
        scores = self.rank_models(RankingMethod.MEAN_SCORE, metric)
        wins = self.rank_models(RankingMethod.WIN_COUNT, metric)

        lines = [
            f"Benchmark Summary (metric: {metric})",
            "=" * 70,
            f"{'Model':<25} {'Mean Rank':<12} {'Mean Score':<12} {'Wins':<8}",
            "-" * 70,
        ]

        for model in rankings.keys():
            mean_rank = -rankings[model]  # Convert back to positive rank
            mean_score = scores.get(model, 0)
            win_count = wins.get(model, 0)
            lines.append(f"{model:<25} {mean_rank:<12.2f} {mean_score:<12.4f} {int(win_count):<8}")

        lines.append("-" * 70)

        # Add Friedman test result
        chi2, p_value = self.friedman_test(metric)
        lines.append(f"Friedman test: chi2={chi2:.2f}, p={p_value:.4f}")

        if p_value < self.significance_level:
            cd = self.nemenyi_critical_difference()
            lines.append(f"Critical difference (Nemenyi): {cd:.3f}")

        return "\n".join(lines)

    def meta_feature_correlation(
        self,
        metric: str | None = None,
        model_name: str | None = None,
    ) -> dict[str, float]:
        """Compute correlation between meta-features and performance.

        Parameters
        ----------
        metric : str, optional
            Performance metric.
        model_name : str, optional
            Specific model to analyze. If None, averages across models.

        Returns
        -------
        Dict[str, float]
            Meta-feature name to correlation mapping.
        """
        metric = metric or self.metric
        metric_col = f"metric_{metric}"

        df = self.df

        if HAS_POLARS and isinstance(df, pl.DataFrame):
            if model_name:
                df = df.filter(pl.col("model_name") == model_name)
            df = df.filter(pl.col("status") == "success")

            # Get meta-feature columns
            mf_cols = [c for c in df.columns if c.startswith("mf_")]

            if not mf_cols or metric_col not in df.columns:
                return {}

            scores = df[metric_col].to_numpy()
            correlations = {}

            for col in mf_cols:
                mf_values = df[col].to_numpy()
                valid = ~(np.isnan(mf_values) | np.isnan(scores))
                if np.sum(valid) > 2:
                    corr, _ = stats.pearsonr(mf_values[valid], scores[valid])
                    if np.isfinite(corr):
                        correlations[col[3:]] = float(corr)  # Remove "mf_" prefix
        else:
            if model_name:
                df = df[df["model_name"] == model_name]
            df = df[df["status"] == "success"]

            mf_cols = [c for c in df.columns if c.startswith("mf_")]

            if not mf_cols or metric_col not in df.columns:
                return {}

            scores = df[metric_col].values
            correlations = {}

            for col in mf_cols:
                mf_values = df[col].values
                valid = ~(np.isnan(mf_values) | np.isnan(scores))
                if np.sum(valid) > 2:
                    corr, _ = stats.pearsonr(mf_values[valid], scores[valid])
                    if np.isfinite(corr):
                        correlations[col[3:]] = float(corr)

        # Sort by absolute correlation
        correlations = dict(sorted(
            correlations.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        ))

        return correlations
