from __future__ import annotations

"""Learning Curve Experiments following LCDB protocol.

Generates learning curves at standard anchor points for systematic
comparison of sample efficiency across models.

References
----------
- Mohr et al. "LCDB 1.0: An Extensive Learning Curves Database" (2022)
"""

import gc
import time
import warnings
from dataclasses import dataclass, field

# Suppress harmless sklearn/LightGBM feature name warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier

# np.trapz was removed in NumPy 2.0, replaced by np.trapezoid (added in 1.25)
_trapz = getattr(np, "trapezoid", None) or np.trapz
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder

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

from endgame.benchmark.loader import DatasetInfo, SuiteLoader, TaskType

# Default LCDB anchor points (fractions of training set)
LCDB_ANCHORS = [0.025, 0.05, 0.10, 0.20, 0.40, 0.80, 1.0]

# Reduced anchors for faster experiments
FAST_ANCHORS = [0.1, 0.25, 0.5, 1.0]


@dataclass
class LearningCurveConfig:
    """Configuration for learning curve experiments.

    Parameters
    ----------
    anchors : List[float]
        Training set fractions (LCDB protocol default).
    n_seeds : int
        Number of random seeds per anchor point.
    cv_folds : int
        Cross-validation folds per seed (0 = holdout only).
    test_fraction : float
        Holdout test set fraction.
    metrics_classification : List[str]
        Metrics for classification tasks.
    metrics_regression : List[str]
        Metrics for regression tasks.
    timeout_per_fit : int
        Timeout per model fit in seconds.
    random_state : int
        Base random seed.
    verbose : bool
        Enable verbose output.
    """
    anchors: list[float] = field(default_factory=lambda: LCDB_ANCHORS.copy())
    n_seeds: int = 5
    cv_folds: int = 0  # 0 = holdout only (faster)
    test_fraction: float = 0.2
    metrics_classification: list[str] = field(default_factory=lambda: [
        "accuracy", "balanced_accuracy", "f1_weighted"
    ])
    metrics_regression: list[str] = field(default_factory=lambda: [
        "r2", "neg_mean_squared_error"
    ])
    timeout_per_fit: int = 600
    random_state: int = 42
    verbose: bool = True


@dataclass
class LearningCurveRecord:
    """Single learning curve data point.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    anchor : float
        Training set fraction.
    n_train : int
        Actual number of training samples.
    seed : int
        Random seed used.
    metrics : Dict[str, float]
        Performance metrics.
    fit_time : float
        Training time in seconds.
    status : str
        'success' or 'error'.
    error_message : str, optional
        Error message if failed.
    """
    dataset_name: str
    model_name: str
    anchor: float
    n_train: int
    seed: int
    metrics: dict[str, float]
    fit_time: float
    status: str = "success"
    error_message: str | None = None


@dataclass
class LearningCurveResults:
    """Container for learning curve results with analysis methods.

    Attributes
    ----------
    records : List[LearningCurveRecord]
        All experiment records.
    config : LearningCurveConfig
        Configuration used.
    """
    records: list[LearningCurveRecord] = field(default_factory=list)
    config: LearningCurveConfig | None = None

    def add_record(self, record: LearningCurveRecord):
        """Add a record to results."""
        self.records.append(record)

    def to_dataframe(self):
        """Convert results to DataFrame.

        Returns
        -------
        DataFrame
            Results in tabular format.
        """
        rows = []
        for r in self.records:
            row = {
                "dataset": r.dataset_name,
                "model": r.model_name,
                "anchor": r.anchor,
                "n_train": r.n_train,
                "seed": r.seed,
                "fit_time": r.fit_time,
                "status": r.status,
            }
            row.update(r.metrics)
            if r.error_message:
                row["error"] = r.error_message
            rows.append(row)

        if HAS_POLARS:
            return pl.DataFrame(rows)
        elif HAS_PANDAS:
            return pd.DataFrame(rows)
        else:
            return rows

    def save(self, path: str):
        """Save results to file.

        Parameters
        ----------
        path : str
            Output path (.parquet, .csv, or .json).
        """
        df = self.to_dataframe()

        if path.endswith('.parquet'):
            if HAS_POLARS:
                df.write_parquet(path)
            elif HAS_PANDAS:
                df.to_parquet(path, index=False)
        elif path.endswith('.csv'):
            if HAS_POLARS:
                df.write_csv(path)
            elif HAS_PANDAS:
                df.to_csv(path, index=False)
        else:
            raise ValueError(f"Unsupported format: {path}")

    def get_learning_curve(
        self,
        dataset: str,
        model: str,
        metric: str = "accuracy",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get learning curve for a specific dataset/model.

        Parameters
        ----------
        dataset : str
            Dataset name.
        model : str
            Model name.
        metric : str
            Metric to retrieve.

        Returns
        -------
        anchors : ndarray
            Training fractions.
        means : ndarray
            Mean metric values.
        stds : ndarray
            Standard deviations.
        """
        # Filter records
        filtered = [
            r for r in self.records
            if r.dataset_name == dataset
            and r.model_name == model
            and r.status == "success"
        ]

        if not filtered:
            return np.array([]), np.array([]), np.array([])

        # Group by anchor
        anchor_values = {}
        for r in filtered:
            if r.anchor not in anchor_values:
                anchor_values[r.anchor] = []
            if metric in r.metrics:
                anchor_values[r.anchor].append(r.metrics[metric])

        anchors = sorted(anchor_values.keys())
        means = [np.mean(anchor_values[a]) for a in anchors]
        stds = [np.std(anchor_values[a]) for a in anchors]

        return np.array(anchors), np.array(means), np.array(stds)

    def compute_aulc(
        self,
        dataset: str,
        model: str,
        metric: str = "accuracy",
    ) -> float:
        """Compute Area Under Learning Curve.

        Higher AULC indicates better sample efficiency (learns faster).

        Parameters
        ----------
        dataset : str
            Dataset name.
        model : str
            Model name.
        metric : str
            Metric to use.

        Returns
        -------
        float
            Area under learning curve (normalized to [0, 1]).
        """
        anchors, means, _ = self.get_learning_curve(dataset, model, metric)

        if len(anchors) < 2:
            return 0.0

        # Trapezoidal integration
        aulc = _trapz(means, anchors)

        # Normalize by max possible area (1.0 * anchor range)
        max_area = anchors[-1] - anchors[0]
        return aulc / max_area if max_area > 0 else 0.0

    def summary(self, metric: str = "accuracy") -> dict[str, dict[str, float]]:
        """Generate summary statistics.

        Parameters
        ----------
        metric : str
            Primary metric for summary.

        Returns
        -------
        dict
            Summary with AULC and final performance per model.
        """
        # Get unique datasets and models
        datasets = set(r.dataset_name for r in self.records)
        models = set(r.model_name for r in self.records)

        summary = {}
        for model in models:
            model_stats = {"aulc": [], "final": []}

            for dataset in datasets:
                aulc = self.compute_aulc(dataset, model, metric)
                if aulc > 0:
                    model_stats["aulc"].append(aulc)

                # Final performance (anchor = 1.0)
                anchors, means, _ = self.get_learning_curve(dataset, model, metric)
                if len(means) > 0:
                    model_stats["final"].append(means[-1])

            summary[model] = {
                "mean_aulc": np.mean(model_stats["aulc"]) if model_stats["aulc"] else 0,
                "mean_final": np.mean(model_stats["final"]) if model_stats["final"] else 0,
                "n_datasets": len(model_stats["aulc"]),
            }

        return summary

    def plot_learning_curves(
        self,
        dataset: str,
        metric: str = "accuracy",
        models: list[str] | None = None,
        ax=None,
        **kwargs,
    ):
        """Plot learning curves for a dataset.

        Parameters
        ----------
        dataset : str
            Dataset name.
        metric : str
            Metric to plot.
        models : List[str], optional
            Models to include (default: all).
        ax : matplotlib.axes.Axes, optional
            Axes to plot on.
        **kwargs
            Additional arguments to plt.plot.

        Returns
        -------
        ax : matplotlib.axes.Axes
            The axes with the plot.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 6))

        all_models = set(r.model_name for r in self.records if r.dataset_name == dataset)
        if models is None:
            models = sorted(all_models)
        else:
            models = [m for m in models if m in all_models]

        for model in models:
            anchors, means, stds = self.get_learning_curve(dataset, model, metric)
            if len(anchors) == 0:
                continue

            line, = ax.plot(anchors, means, marker='o', label=model, **kwargs)
            ax.fill_between(
                anchors,
                means - stds,
                means + stds,
                alpha=0.2,
                color=line.get_color(),
            )

        ax.set_xlabel("Training Set Fraction")
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.set_title(f"Learning Curves: {dataset}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax


class LearningCurveExperiment:
    """Run learning curve experiments across datasets.

    Implements the LCDB (Learning Curve Database) protocol for
    systematic evaluation of sample efficiency.

    Parameters
    ----------
    suite : str or List[DatasetInfo]
        Benchmark suite name or list of datasets.
    config : LearningCurveConfig, optional
        Experiment configuration.
    max_datasets : int, optional
        Maximum number of datasets.
    verbose : bool
        Enable verbose output.

    Examples
    --------
    >>> from endgame.benchmark import LearningCurveExperiment, LearningCurveConfig
    >>> from endgame.models import LGBMWrapper
    >>>
    >>> config = LearningCurveConfig(anchors=[0.1, 0.5, 1.0], n_seeds=3)
    >>> exp = LearningCurveExperiment(suite="sklearn-classic", config=config)
    >>>
    >>> models = [
    ...     ("LGBM", LGBMWrapper(preset="fast")),
    ... ]
    >>> results = exp.run(models)
    >>> print(results.summary())
    """

    def __init__(
        self,
        suite: str | list[DatasetInfo],
        config: LearningCurveConfig | None = None,
        max_datasets: int | None = None,
        verbose: bool = True,
    ):
        self.suite = suite
        self.config = config or LearningCurveConfig()
        self.max_datasets = max_datasets
        self.verbose = verbose

        self._datasets: list[DatasetInfo] = []

    def _log(self, msg: str):
        """Print message if verbose."""
        if self.verbose:
            print(f"[LearningCurve] {msg}")

    def _load_datasets(self) -> list[DatasetInfo]:
        """Load datasets from suite."""
        if isinstance(self.suite, list):
            return self.suite[:self.max_datasets] if self.max_datasets else self.suite

        loader = SuiteLoader(
            suite=self.suite,
            max_datasets=self.max_datasets,
            random_state=self.config.random_state,
            verbose=self.verbose,
        )
        return list(loader.load())

    def _subsample_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        fraction: float,
        seed: int,
        stratify: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stratified subsampling for anchor point.

        Parameters
        ----------
        X : ndarray
            Features.
        y : ndarray
            Targets.
        fraction : float
            Fraction to use for training.
        seed : int
            Random seed.
        stratify : bool
            Use stratified sampling.

        Returns
        -------
        X_train, X_test, y_train, y_test
        """
        n_samples = len(X)
        n_train = max(10, int(n_samples * fraction * (1 - self.config.test_fraction)))
        n_test = max(10, int(n_samples * self.config.test_fraction))

        # Ensure we don't exceed available samples
        if n_train + n_test > n_samples:
            n_test = max(10, n_samples // 5)
            n_train = n_samples - n_test

        if stratify and len(np.unique(y)) > 1:
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                test_size=n_test,
                train_size=n_train,
                random_state=seed,
            )
        else:
            splitter = ShuffleSplit(
                n_splits=1,
                test_size=n_test,
                train_size=n_train,
                random_state=seed,
            )

        train_idx, test_idx = next(splitter.split(X, y))
        return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

    def _evaluate_model(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        is_classification: bool,
    ) -> tuple[dict[str, float], float]:
        """Fit model and compute metrics.

        Returns
        -------
        metrics : dict
            Performance metrics.
        fit_time : float
            Training time.
        """
        start = time.time()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X_train, y_train)

        fit_time = time.time() - start

        metrics = {}

        if is_classification:
            y_pred = model.predict(X_test)

            metrics["accuracy"] = accuracy_score(y_test, y_pred)
            metrics["balanced_accuracy"] = balanced_accuracy_score(y_test, y_pred)

            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                metrics["f1"] = f1_score(y_test, y_pred, zero_division=0)
            else:
                metrics["f1_weighted"] = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )

            # Probability-based metrics
            if hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)
                    if n_classes == 2:
                        metrics["roc_auc"] = roc_auc_score(y_test, y_proba[:, 1])
                    else:
                        metrics["roc_auc_ovr"] = roc_auc_score(
                            y_test, y_proba, multi_class="ovr", average="weighted"
                        )
                    metrics["log_loss"] = log_loss(y_test, y_proba)
                except Exception:
                    pass
        else:
            y_pred = model.predict(X_test)
            metrics["r2"] = r2_score(y_test, y_pred)
            metrics["mse"] = mean_squared_error(y_test, y_pred)
            metrics["rmse"] = np.sqrt(metrics["mse"])

        return metrics, fit_time

    def run(
        self,
        models: list[tuple[str, BaseEstimator]],
        output_file: str | None = None,
        continue_on_error: bool = True,
    ) -> LearningCurveResults:
        """Run learning curve experiments.

        Parameters
        ----------
        models : List[Tuple[str, BaseEstimator]]
            List of (name, model) tuples.
        output_file : str, optional
            Path to save results.
        continue_on_error : bool
            Continue if a model fails.

        Returns
        -------
        LearningCurveResults
            Experiment results.
        """
        self._log("Starting learning curve experiment")
        self._log(f"Anchors: {self.config.anchors}")
        self._log(f"Seeds per anchor: {self.config.n_seeds}")

        # Load datasets
        self._datasets = self._load_datasets()
        self._log(f"Loaded {len(self._datasets)} datasets")

        results = LearningCurveResults(config=self.config)

        total_experiments = (
            len(self._datasets) *
            len(models) *
            len(self.config.anchors) *
            self.config.n_seeds
        )
        completed = 0

        for dataset in self._datasets:
            self._log(f"\nDataset: {dataset.name} ({dataset.n_samples} samples)")
            is_classification = dataset.task_type != TaskType.REGRESSION

            # Prepare data
            X = dataset.X.copy()
            y = dataset.y.copy()

            if is_classification:
                le = LabelEncoder()
                y = le.fit_transform(y)

            for model_name, model_template in models:
                self._log(f"  Model: {model_name}")

                for anchor in self.config.anchors:
                    for seed in range(self.config.n_seeds):
                        completed += 1
                        actual_seed = self.config.random_state + seed

                        try:
                            # Subsample data
                            X_train, X_test, y_train, y_test = self._subsample_data(
                                X, y, anchor, actual_seed,
                                stratify=is_classification,
                            )

                            # Clone model
                            model = clone(model_template)

                            # Fit and evaluate
                            metrics, fit_time = self._evaluate_model(
                                model, X_train, y_train, X_test, y_test,
                                is_classification,
                            )

                            record = LearningCurveRecord(
                                dataset_name=dataset.name,
                                model_name=model_name,
                                anchor=anchor,
                                n_train=len(X_train),
                                seed=seed,
                                metrics=metrics,
                                fit_time=fit_time,
                                status="success",
                            )

                            if self.verbose and seed == 0:
                                primary_metric = "accuracy" if is_classification else "r2"
                                val = metrics.get(primary_metric, 0)
                                self._log(
                                    f"    anchor={anchor:.0%}, n={len(X_train)}: "
                                    f"{primary_metric}={val:.4f}"
                                )

                        except Exception as e:
                            record = LearningCurveRecord(
                                dataset_name=dataset.name,
                                model_name=model_name,
                                anchor=anchor,
                                n_train=0,
                                seed=seed,
                                metrics={},
                                fit_time=0.0,
                                status="error",
                                error_message=str(e),
                            )

                            if not continue_on_error:
                                raise

                            if self.verbose:
                                self._log(f"    ERROR at anchor={anchor:.0%}: {e}")

                        results.add_record(record)

                # Clean up
                gc.collect()

        # Save results
        if output_file:
            results.save(output_file)
            self._log(f"\nResults saved to: {output_file}")

        self._log(f"\nExperiment complete! {len(results.records)} records.")
        return results


def quick_learning_curve(
    model: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    anchors: list[float] | None = None,
    n_seeds: int = 3,
    test_fraction: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Quick learning curve for a single model/dataset.

    Parameters
    ----------
    model : BaseEstimator
        Model to evaluate.
    X : ndarray
        Features.
    y : ndarray
        Targets.
    anchors : List[float], optional
        Training fractions.
    n_seeds : int
        Seeds per anchor.
    test_fraction : float
        Test set fraction.
    random_state : int
        Random seed.

    Returns
    -------
    anchors : ndarray
        Training fractions.
    means : ndarray
        Mean accuracies.
    stds : ndarray
        Standard deviations.
    """
    if anchors is None:
        anchors = FAST_ANCHORS

    is_clf = is_classifier(model)
    results = {a: [] for a in anchors}

    for anchor in anchors:
        for seed in range(n_seeds):
            actual_seed = random_state + seed

            # Split data
            n_samples = len(X)
            n_test = max(10, int(n_samples * test_fraction))
            n_train = max(10, int(n_samples * anchor * (1 - test_fraction)))

            if is_clf:
                splitter = StratifiedShuffleSplit(
                    n_splits=1, test_size=n_test, train_size=n_train,
                    random_state=actual_seed,
                )
            else:
                splitter = ShuffleSplit(
                    n_splits=1, test_size=n_test, train_size=n_train,
                    random_state=actual_seed,
                )

            train_idx, test_idx = next(splitter.split(X, y))

            # Fit and score
            m = clone(model)
            m.fit(X[train_idx], y[train_idx])

            if is_clf:
                score = accuracy_score(y[test_idx], m.predict(X[test_idx]))
            else:
                score = r2_score(y[test_idx], m.predict(X[test_idx]))

            results[anchor].append(score)

    anchors_arr = np.array(anchors)
    means = np.array([np.mean(results[a]) for a in anchors])
    stds = np.array([np.std(results[a]) for a in anchors])

    return anchors_arr, means, stds
