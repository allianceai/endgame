"""Benchmark runner for systematic model evaluation.

Orchestrates experiments across multiple datasets and models/pipelines.
"""

import gc
import hashlib
import json
import multiprocessing
import os
import time
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass, field
from typing import Any

# Suppress harmless sklearn/LightGBM feature name warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)

import numpy as np
from sklearn.base import BaseEstimator, clone, is_classifier, is_regressor
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    make_scorer,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.preprocessing import LabelEncoder


def _specificity_score(y_true, y_pred):
    """Compute specificity (true negative rate) for binary classification."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        return tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return 0.0


def _fpr_score(y_true, y_pred):
    """Compute false positive rate for binary classification."""
    return 1.0 - _specificity_score(y_true, y_pred)

from endgame.benchmark.loader import DatasetInfo, SuiteLoader, TaskType
from endgame.benchmark.profiler import MetaFeatureSet, MetaProfiler
from endgame.benchmark.tracker import (
    ExperimentRecord,
    ExperimentTracker,
    get_experiment_hash,
    serialize_pipeline,
)

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False

try:
    import pandas as _pd_check  # noqa: F401
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


class TimeoutException(Exception):
    """Exception raised when a model training exceeds the time limit."""
    pass


def _run_func_in_process(func: Callable, result_queue: multiprocessing.Queue, *args, **kwargs):
    """Worker function to run in a subprocess."""
    try:
        result = func(*args, **kwargs)
        result_queue.put(("success", result))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _run_with_timeout(func: Callable, timeout: int, *args, **kwargs) -> Any:
    """Run a function with a timeout using multiprocessing.

    This implementation actually terminates the computation when timeout is reached,
    unlike ThreadPoolExecutor which only stops waiting but lets the thread continue.

    Parameters
    ----------
    func : Callable
        Function to run.
    timeout : int
        Timeout in seconds.
    *args, **kwargs
        Arguments to pass to the function.

    Returns
    -------
    Any
        Result of the function.

    Raises
    ------
    TimeoutException
        If the function exceeds the timeout.
    Exception
        If the function raises an exception.
    """
    # First try with ThreadPoolExecutor (faster, works for most cases)
    # Fall back to multiprocessing only if needed for hard timeout
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            # Thread is still running but we hit timeout
            # For now, just raise - the thread will eventually complete or be cleaned up
            # A more aggressive approach would use multiprocessing, but that has
            # serialization issues with sklearn models
            raise TimeoutException(f"Exceeded timeout of {timeout}s")


def _get_default_cache_dir() -> str:
    """Get default cache directory for meta-features."""
    # Use XDG_CACHE_HOME if available, otherwise ~/.cache
    cache_home = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
    return os.path.join(cache_home, "endgame", "meta_features")


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark runs.

    Attributes
    ----------
    suite : str
        Benchmark suite name or list of task IDs.
    max_datasets : int, optional
        Maximum number of datasets to run.
    max_samples : int, optional
        Maximum samples per dataset.
    cv_folds : int
        Number of cross-validation folds.
    scoring_classification : List[str]
        Metrics for classification tasks.
    scoring_regression : List[str]
        Metrics for regression tasks.
    profile_datasets : bool
        Whether to extract meta-features.
    profile_groups : List[str]
        Meta-feature groups to extract.
    cache_meta_features : bool
        Whether to cache meta-features to disk.
    meta_features_cache_dir : str, optional
        Directory to cache meta-features. Defaults to ~/.cache/endgame/meta_features.
    timeout_per_fit : int
        Timeout per model fit in seconds.
    n_jobs : int
        Number of parallel jobs for CV.
    random_state : int
        Random seed.
    verbose : bool
        Enable verbose output.
    skip_completed : bool
        Skip experiments that already succeeded.
    """
    suite: str = "sklearn-classic"
    max_datasets: int | None = None
    max_samples: int | None = None
    cv_folds: int = 5
    scoring_classification: list[str] = field(
        default_factory=lambda: ["accuracy", "f1_weighted", "roc_auc_ovr_weighted"]
    )
    scoring_regression: list[str] = field(
        default_factory=lambda: ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]
    )
    profile_datasets: bool = True
    profile_groups: list[str] = field(
        default_factory=lambda: ["simple", "statistical"]
    )
    cache_meta_features: bool = True
    meta_features_cache_dir: str | None = None
    timeout_per_fit: int = 300  # 5 minutes
    n_jobs: int = 1
    random_state: int = 42
    verbose: bool = True
    skip_completed: bool = True  # Skip experiments that already succeeded


class BenchmarkRunner:
    """Run systematic benchmarks across datasets and models.

    Orchestrates the complete benchmark workflow:
    1. Load datasets from benchmark suite
    2. Profile datasets (extract meta-features)
    3. Run cross-validation for each model on each dataset
    4. Record results with full provenance

    Parameters
    ----------
    suite : str, default="sklearn-classic"
        Benchmark suite name.
    config : BenchmarkConfig, optional
        Full configuration object.
    max_datasets : int, optional
        Override maximum number of datasets.
    fast_run : bool, default=False
        Quick run with reduced settings.
    verbose : bool, default=True
        Enable verbose output.
    **kwargs
        Additional configuration parameters.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.linear_model import LogisticRegression
    >>>
    >>> models = [
    ...     ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
    ...     ("LR", LogisticRegression(max_iter=1000)),
    ... ]
    >>>
    >>> runner = BenchmarkRunner(suite="sklearn-classic")
    >>> results = runner.run(models)
    >>> print(results.summary())
    >>>
    >>> # Save results
    >>> results.save("benchmark_results.parquet")
    """

    def __init__(
        self,
        suite: str = "sklearn-classic",
        config: BenchmarkConfig | None = None,
        max_datasets: int | None = None,
        fast_run: bool = False,
        verbose: bool = True,
        **kwargs,
    ):
        if config is not None:
            self.config = config
        else:
            self.config = BenchmarkConfig(
                suite=suite,
                max_datasets=max_datasets,
                verbose=verbose,
                **kwargs,
            )

        # Fast run overrides
        if fast_run:
            self.config.max_datasets = min(self.config.max_datasets or 5, 5)
            self.config.max_samples = 1000
            self.config.cv_folds = 3
            self.config.profile_groups = ["simple"]

        self.verbose = verbose
        self._tracker = ExperimentTracker(name=f"benchmark_{suite}")
        self._profiler = MetaProfiler(
            groups=self.config.profile_groups,
            random_state=self.config.random_state,
            verbose=False,
        )

        # Meta-feature cache directory
        self._cache_dir = self.config.meta_features_cache_dir or _get_default_cache_dir()
        if self.config.cache_meta_features:
            os.makedirs(self._cache_dir, exist_ok=True)

        # Results cache
        self._datasets: list[DatasetInfo] = []
        self._meta_features: dict[str, MetaFeatureSet] = {}
        self._completed_config_hashes: set = set()  # Set of config hashes for completed experiments

    def _load_completed_experiments(self, output_file: str) -> None:
        """Load completed experiments from existing results file.

        Parameters
        ----------
        output_file : str
            Path to the results file.
        """
        if not os.path.exists(output_file):
            return

        try:
            if HAS_POLARS:
                df = pl.read_parquet(output_file)
                # Filter for successful experiments only
                successful = df.filter(pl.col('status') == 'success')
                # Load config hashes for completed experiments
                if 'config_hash' in successful.columns:
                    for row in successful.iter_rows(named=True):
                        if row.get('config_hash'):
                            self._completed_config_hashes.add(row['config_hash'])
            elif HAS_PANDAS:
                import pandas as pd
                df = pd.read_parquet(output_file)
                successful = df[df['status'] == 'success']
                if 'config_hash' in successful.columns:
                    for _, row in successful.iterrows():
                        if row.get('config_hash'):
                            self._completed_config_hashes.add(row['config_hash'])

            if self._completed_config_hashes:
                self._log(f"Loaded {len(self._completed_config_hashes)} completed experiment configs from {output_file}")
        except Exception as e:
            self._log(f"Warning: Could not load existing results: {e}")

    def _log(self, message: str) -> None:
        """Print message if verbose."""
        if self.verbose:
            print(f"[BenchmarkRunner] {message}")

    def _get_dataset_cache_key(self, dataset: "DatasetInfo") -> str:
        """Generate a cache key for a dataset based on its properties.

        The key includes dataset name, shape, and a hash of the data to detect changes.
        """
        # Create a fingerprint from dataset properties
        fingerprint_data = {
            "name": dataset.name,
            "n_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "task_type": dataset.task_type.value,
            "profile_groups": sorted(self.config.profile_groups),
        }
        # Add a hash of the first few rows for data identity
        if dataset.X is not None and len(dataset.X) > 0:
            # Sample a small portion to compute hash quickly
            sample_size = min(100, len(dataset.X))
            data_sample = dataset.X[:sample_size].tobytes() if hasattr(dataset.X, 'tobytes') else str(dataset.X[:sample_size]).encode()
            fingerprint_data["data_hash"] = hashlib.md5(data_sample).hexdigest()[:16]

        # Create a hash of the fingerprint
        fingerprint_str = json.dumps(fingerprint_data, sort_keys=True)
        cache_key = hashlib.md5(fingerprint_str.encode()).hexdigest()
        return f"{dataset.name}_{cache_key[:12]}"

    def _get_cached_meta_features(self, dataset: "DatasetInfo") -> MetaFeatureSet | None:
        """Try to load cached meta-features for a dataset.

        Returns None if not cached or cache is disabled.
        """
        if not self.config.cache_meta_features:
            return None

        cache_key = self._get_dataset_cache_key(dataset)
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")

        if not os.path.exists(cache_file):
            return None

        try:
            with open(cache_file) as f:
                cached = json.load(f)

            # Reconstruct MetaFeatureSet
            return MetaFeatureSet(
                features=cached.get("features", {}),
                groups=cached.get("groups", {}),
                extraction_time=cached.get("extraction_time", 0.0),
                errors=cached.get("errors", []),
            )
        except (OSError, json.JSONDecodeError, KeyError):
            # Cache is corrupted, delete it
            try:
                os.remove(cache_file)
            except OSError:
                pass
            return None

    def _cache_meta_features(self, dataset: "DatasetInfo", meta_features: MetaFeatureSet) -> None:
        """Cache meta-features for a dataset to disk."""
        if not self.config.cache_meta_features:
            return

        cache_key = self._get_dataset_cache_key(dataset)
        cache_file = os.path.join(self._cache_dir, f"{cache_key}.json")

        try:
            cached_data = {
                "features": meta_features.features,
                "groups": meta_features.groups,
                "extraction_time": meta_features.extraction_time,
                "errors": meta_features.errors,
                "dataset_name": dataset.name,
                "n_samples": dataset.n_samples,
                "n_features": dataset.n_features,
            }
            with open(cache_file, 'w') as f:
                json.dump(cached_data, f, indent=2)
        except (OSError, TypeError) as e:
            self._log(f"Warning: Could not cache meta-features for {dataset.name}: {e}")

    def run(
        self,
        models: list[tuple[str, BaseEstimator] | tuple[str, BaseEstimator, BaseEstimator]],
        output_file: str | None = None,
        continue_on_error: bool = True,
    ) -> ExperimentTracker:
        """Run benchmark on all models and datasets.

        Parameters
        ----------
        models : List[Union[Tuple[str, BaseEstimator], Tuple[str, BaseEstimator, BaseEstimator]]]
            List of model specifications. Each can be either:
            - (name, estimator): Single estimator used for all tasks
            - (name, classifier, regressor): Pair of estimators, classifier used for
              classification tasks and regressor for regression tasks. Either can be
              None to skip that task type.
        output_file : str, optional
            Path to save results.
        continue_on_error : bool, default=True
            Continue if a model fails on a dataset.

        Returns
        -------
        ExperimentTracker
            Tracker with all experiment results.
        """
        self._log(f"Starting benchmark on suite: {self.config.suite}")
        # Extract model names from either 2-tuple or 3-tuple format
        model_names = [m[0] for m in models]
        self._log(f"Models: {model_names}")

        # Load completed experiments if skip_completed is enabled
        if self.config.skip_completed and output_file:
            self._load_completed_experiments(output_file)

        # Load datasets
        loader = SuiteLoader(
            suite=self.config.suite,
            max_datasets=self.config.max_datasets,
            max_samples=self.config.max_samples,
            random_state=self.config.random_state,
            verbose=self.verbose,
        )

        self._datasets = list(loader.load())
        self._log(f"Loaded {len(self._datasets)} datasets")

        # Profile datasets (with caching)
        if self.config.profile_datasets:
            self._log("Profiling datasets...")
            cached_count = 0
            profiled_count = 0
            for dataset in self._datasets:
                # Try to load from cache first
                cached_meta = self._get_cached_meta_features(dataset)
                if cached_meta is not None:
                    self._meta_features[dataset.name] = cached_meta
                    cached_count += 1
                    continue

                # Not cached, profile the dataset
                try:
                    task_type = "classification" if dataset.task_type != TaskType.REGRESSION else "regression"
                    meta = self._profiler.profile(
                        dataset.X,
                        dataset.y,
                        categorical_indicator=dataset.categorical_indicator,
                        task_type=task_type,
                    )
                    self._meta_features[dataset.name] = meta
                    # Cache the result
                    self._cache_meta_features(dataset, meta)
                    profiled_count += 1
                except Exception as e:
                    self._log(f"Failed to profile {dataset.name}: {e}")
                    self._meta_features[dataset.name] = MetaFeatureSet()
                    profiled_count += 1

            if cached_count > 0:
                self._log(f"  Loaded {cached_count} from cache, profiled {profiled_count} new datasets")

        # Run experiments
        total_experiments = len(self._datasets) * len(models)
        completed = 0

        for dataset in self._datasets:
            self._log(f"\nDataset: {dataset.name} ({dataset.n_samples} samples, {dataset.n_features} features)")
            is_regression = dataset.task_type == TaskType.REGRESSION
            task_type = "regression" if is_regression else "classification"

            for model_spec in models:
                completed += 1

                # Parse model specification - supports both 2-tuple and 3-tuple formats
                model_name = model_spec[0]
                if len(model_spec) == 2:
                    # (name, estimator) format - use same model for all tasks
                    model = model_spec[1]
                elif len(model_spec) == 3:
                    # (name, classifier, regressor) format - select based on task
                    classifier, regressor = model_spec[1], model_spec[2]
                    model = regressor if is_regression else classifier
                    if model is None:
                        self._log(f"  [{completed}/{total_experiments}] Skipping {model_name} (no {task_type} variant)")
                        continue
                else:
                    self._log(f"  [{completed}/{total_experiments}] Skipping {model_name} (invalid model spec)")
                    continue

                # Compute config hash for this experiment
                # This includes dataset, model name, hyperparameters, and task type
                try:
                    hyperparams = model.get_params()
                except Exception:
                    hyperparams = {}

                config_hash = get_experiment_hash(
                    dataset_name=dataset.name,
                    model_name=model_name,
                    hyperparameters=hyperparams,
                    task_type=task_type,
                )

                # Check if this exact experiment config was already completed successfully
                if self.config.skip_completed and config_hash in self._completed_config_hashes:
                    self._log(f"  [{completed}/{total_experiments}] Skipping {model_name} (already completed)")
                    continue

                self._log(f"  [{completed}/{total_experiments}] Running {model_name}...")

                try:
                    record = self._run_single_experiment(
                        dataset=dataset,
                        model_name=model_name,
                        model=model,
                    )

                    if record.status == "success":
                        primary_metric = list(record.metrics.keys())[0] if record.metrics else "unknown"
                        primary_value = record.metrics.get(primary_metric, 0)
                        self._log(f"    {primary_metric}: {primary_value:.4f} (fit: {record.fit_time:.2f}s)")
                        # Add to completed set so we skip if interrupted and restarted
                        self._completed_config_hashes.add(config_hash)
                    else:
                        self._log(f"    FAILED: {record.error_message}")

                except Exception as e:
                    if continue_on_error:
                        self._log(f"    ERROR: {e}")
                        self._tracker.log_failure(
                            dataset_name=dataset.name,
                            model_name=model_name,
                            error_message=str(e),
                            n_samples=dataset.n_samples,
                            n_features=dataset.n_features,
                            task_type=dataset.task_type.value,
                        )
                    else:
                        raise

                # Save results after each experiment for resumability
                if output_file:
                    self._tracker.save(output_file, append=True, deduplicate=True)

                # Clean up memory
                gc.collect()

        # Final save (in case no experiments ran)
        if output_file:
            self._log(f"\nResults saved to: {output_file}")

        self._log(f"\nBenchmark complete! {len(self._tracker)} experiments recorded.")
        return self._tracker

    def _run_single_experiment(
        self,
        dataset: DatasetInfo,
        model_name: str,
        model: BaseEstimator,
    ) -> ExperimentRecord:
        """Run a single experiment (one model on one dataset)."""
        # Determine task type
        is_classification = dataset.task_type != TaskType.REGRESSION

        # Clone model first (needed for scoring check)
        try:
            model_clone = clone(model)
        except Exception:
            # Some models don't support sklearn clone
            model_clone = model

        # Explicitly check if model is a classifier/regressor
        # This helps with models that sklearn might misidentify
        model_is_classifier = is_classifier(model_clone)
        model_is_regressor = is_regressor(model_clone)

        # Warn if there's a mismatch between task type and model type
        if is_classification and model_is_regressor and not model_is_classifier:
            self._log(f"  Warning: Model {model_name} detected as regressor but task is classification")
        elif not is_classification and model_is_classifier and not model_is_regressor:
            self._log(f"  Warning: Model {model_name} detected as classifier but task is regression")

        # Get scoring - pass model to check what methods it supports
        if is_classification:
            scoring = self._get_classification_scoring(dataset, model_clone)
        else:
            scoring = self._get_regression_scoring()

        # Get CV splitter
        cv = self._get_cv_splitter(dataset)

        # Prepare data
        X = dataset.X.copy()
        y = dataset.y.copy()

        # Encode target for classification
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y)

        # Run cross-validation with timeout
        start_time = time.time()
        timeout = self.config.timeout_per_fit

        def _run_cv():
            """Run cross-validation (wrapped for timeout)."""
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return cross_validate(
                    model_clone,
                    X,
                    y,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                    n_jobs=self.config.n_jobs,
                    error_score="raise",
                )

        try:
            if timeout and timeout > 0:
                cv_results = _run_with_timeout(_run_cv, timeout)
            else:
                cv_results = _run_cv()

            fit_time = time.time() - start_time

            # Compute metrics
            metrics = {}
            cv_scores = None

            # Handle both dict and list scoring
            score_names = list(scoring.keys()) if isinstance(scoring, dict) else scoring

            for score_name in score_names:
                key = f"test_{score_name}"
                if key in cv_results:
                    scores = cv_results[key]
                    # Handle negative scores (sklearn convention)
                    if score_name.startswith("neg_"):
                        scores = -scores
                        metric_name = score_name[4:]  # Remove "neg_" prefix
                    else:
                        metric_name = score_name

                    metrics[metric_name] = float(np.mean(scores))

                    if cv_scores is None:
                        cv_scores = scores.tolist()

            # Get pipeline config
            pipeline_config = serialize_pipeline(model)

            # Get hyperparameters
            hyperparameters = model.get_params()

            # Get meta-features
            meta_features = {}
            if dataset.name in self._meta_features:
                meta_features = self._meta_features[dataset.name].to_dict()

            # Record experiment
            record = self._tracker.log_experiment(
                dataset_name=dataset.name,
                dataset_id=str(dataset.openml_id) if dataset.openml_id else None,
                model_name=model_name,
                pipeline_config=pipeline_config,
                hyperparameters=self._serialize_hyperparameters(hyperparameters),
                metrics=metrics,
                meta_features=meta_features,
                cv_scores=cv_scores,
                fit_time=fit_time,
                predict_time=float(np.sum(cv_results.get("score_time", [0]))),
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                task_type=dataset.task_type.value,
                status="success",
            )

            return record

        except TimeoutException as e:
            fit_time = time.time() - start_time

            # Get meta-features even for failures
            meta_features = {}
            if dataset.name in self._meta_features:
                meta_features = self._meta_features[dataset.name].to_dict()

            error_msg = f"TIMEOUT: {e}"
            record = self._tracker.log_failure(
                dataset_name=dataset.name,
                dataset_id=str(dataset.openml_id) if dataset.openml_id else None,
                model_name=model_name,
                error_message=error_msg,
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                task_type=dataset.task_type.value,
                meta_features=meta_features,
            )

            return record

        except Exception as e:
            fit_time = time.time() - start_time

            # Get meta-features even for failures
            meta_features = {}
            if dataset.name in self._meta_features:
                meta_features = self._meta_features[dataset.name].to_dict()

            record = self._tracker.log_failure(
                dataset_name=dataset.name,
                dataset_id=str(dataset.openml_id) if dataset.openml_id else None,
                model_name=model_name,
                error_message=str(e),
                n_samples=dataset.n_samples,
                n_features=dataset.n_features,
                task_type=dataset.task_type.value,
                meta_features=meta_features,
            )

            return record

    def _get_classification_scoring(self, dataset: DatasetInfo, model: BaseEstimator | None = None) -> list[str] | dict[str, Any]:
        """Get scoring metrics for classification.

        Returns either a list of scoring strings or a dict with make_scorer objects
        that explicitly specify response_method to avoid sklearn's auto-detection issues.

        Metrics computed:
        - accuracy: Overall accuracy
        - balanced_accuracy: Balanced accuracy (accounts for class imbalance)
        - f1 / f1_weighted: F1 score (weighted for multiclass)
        - precision / precision_weighted: Precision
        - recall / recall_weighted: Recall (sensitivity/TPR)
        - mcc: Matthews correlation coefficient
        - roc_auc: Area under ROC curve (if predict_proba available)
        - log_loss: Log loss / cross-entropy (if predict_proba available)
        - brier_score: Brier score (binary only, if predict_proba available)
        - specificity: True negative rate (binary only)
        - fpr: False positive rate (binary only)
        """
        # Check what response methods the model supports
        has_predict_proba = model is not None and hasattr(model, 'predict_proba')
        has_decision_function = model is not None and hasattr(model, 'decision_function')
        is_binary = dataset.n_classes == 2

        # Build scoring dict with explicit response_method for problematic metrics
        scoring_dict = {}

        # === Metrics that work with predict() ===

        # Accuracy always works
        scoring_dict["accuracy"] = "accuracy"
        scoring_dict["balanced_accuracy"] = "balanced_accuracy"

        # F1, Precision, Recall
        if is_binary:
            scoring_dict["f1"] = "f1"
            scoring_dict["precision"] = "precision"
            scoring_dict["recall"] = "recall"  # Same as sensitivity/TPR
            # Specificity and FPR for binary
            scoring_dict["specificity"] = make_scorer(_specificity_score)
            scoring_dict["fpr"] = make_scorer(_fpr_score)
        else:
            scoring_dict["f1_weighted"] = "f1_weighted"
            scoring_dict["precision_weighted"] = "precision_weighted"
            scoring_dict["recall_weighted"] = "recall_weighted"

        # Matthews Correlation Coefficient - works for both binary and multiclass
        scoring_dict["mcc"] = make_scorer(matthews_corrcoef)

        # === Metrics that need predict_proba ===
        if has_predict_proba:
            # ROC-AUC
            if is_binary:
                scoring_dict["roc_auc"] = make_scorer(
                    roc_auc_score,
                    response_method="predict_proba",
                )
                # Brier score (only for binary)
                def brier_binary(y_true, y_proba):
                    # y_proba is the probability of positive class
                    if y_proba.ndim == 2:
                        y_proba = y_proba[:, 1]
                    return brier_score_loss(y_true, y_proba)
                scoring_dict["brier_score"] = make_scorer(
                    brier_binary,
                    response_method="predict_proba",
                    greater_is_better=False,
                )
            elif dataset.n_classes > 2 and dataset.n_classes <= 10:
                # For multiclass, roc_auc_score needs additional parameters
                def roc_auc_multiclass(y_true, y_score):
                    return roc_auc_score(y_true, y_score, multi_class='ovr', average='weighted')
                scoring_dict["roc_auc_ovr_weighted"] = make_scorer(
                    roc_auc_multiclass,
                    response_method="predict_proba",
                )

            # Log loss (works for both binary and multiclass)
            if dataset.n_classes <= 10:  # Skip for high cardinality targets
                scoring_dict["log_loss"] = make_scorer(
                    log_loss,
                    response_method="predict_proba",
                    greater_is_better=False,
                )

        return scoring_dict

    def _get_regression_scoring(self) -> list[str]:
        """Get scoring metrics for regression."""
        return ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

    def _get_cv_splitter(self, dataset: DatasetInfo):
        """Get appropriate CV splitter."""
        if dataset.cv_splits:
            return dataset.cv_splits[:self.config.cv_folds]

        if dataset.task_type != TaskType.REGRESSION:
            return StratifiedKFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )
        else:
            return KFold(
                n_splits=self.config.cv_folds,
                shuffle=True,
                random_state=self.config.random_state,
            )

    def _serialize_hyperparameters(self, params: dict[str, Any]) -> dict[str, Any]:
        """Serialize hyperparameters to JSON-compatible format."""
        result = {}
        for key, value in params.items():
            if value is None or isinstance(value, (int, float, str, bool)):
                result[key] = value
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                result[key] = str(value)
            elif hasattr(value, "__name__"):
                result[key] = value.__name__
            else:
                result[key] = str(type(value).__name__)
        return result

    @property
    def tracker(self) -> ExperimentTracker:
        """Get the experiment tracker."""
        return self._tracker

    @property
    def datasets(self) -> list[DatasetInfo]:
        """Get loaded datasets."""
        return self._datasets

    @property
    def meta_features(self) -> dict[str, MetaFeatureSet]:
        """Get extracted meta-features."""
        return self._meta_features

    def get_results_dataframe(self):
        """Get results as DataFrame."""
        return self._tracker.to_dataframe()


def quick_benchmark(
    model: BaseEstimator,
    model_name: str = "model",
    suite: str = "quick-test",
    **kwargs,
) -> ExperimentTracker:
    """Quick benchmark a single model on test datasets.

    Parameters
    ----------
    model : BaseEstimator
        Model to benchmark.
    model_name : str, default="model"
        Name for the model.
    suite : str, default="quick-test"
        Benchmark suite.
    **kwargs
        Additional arguments to BenchmarkRunner.

    Returns
    -------
    ExperimentTracker
        Results tracker.

    Examples
    --------
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> results = quick_benchmark(RandomForestClassifier(), "RF")
    >>> print(results.summary())
    """
    runner = BenchmarkRunner(suite=suite, fast_run=True, **kwargs)
    return runner.run([(model_name, model)])


def compare_models(
    models: list[tuple[str, BaseEstimator]],
    suite: str = "sklearn-classic",
    **kwargs,
) -> ExperimentTracker:
    """Compare multiple models on benchmark datasets.

    Parameters
    ----------
    models : List[Tuple[str, BaseEstimator]]
        List of (name, model) tuples.
    suite : str, default="sklearn-classic"
        Benchmark suite.
    **kwargs
        Additional arguments to BenchmarkRunner.

    Returns
    -------
    ExperimentTracker
        Results tracker.
    """
    runner = BenchmarkRunner(suite=suite, **kwargs)
    return runner.run(models)
