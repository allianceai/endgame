"""Benchmarking and Meta-Learning Engine for Endgame.

This module provides tools for systematic model evaluation across standard
benchmark datasets, meta-feature extraction, and meta-learning to predict
optimal pipelines for new datasets.

Example
-------
>>> import endgame as eg
>>> from endgame.benchmark import BenchmarkRunner, SuiteLoader
>>> from sklearn.ensemble import RandomForestClassifier
>>> from sklearn.linear_model import LogisticRegression
>>>
>>> # Define models to benchmark
>>> models = [
...     ("RF", RandomForestClassifier(n_estimators=100, random_state=42)),
...     ("LR", LogisticRegression(max_iter=1000)),
... ]
>>>
>>> # Run benchmark on OpenML-CC18 suite
>>> runner = BenchmarkRunner(suite="OpenML-CC18", max_datasets=10)
>>> results = runner.run(models)
>>>
>>> # Analyze results
>>> print(results.summary())
>>> results.to_parquet("benchmark_results.parquet")
"""

from endgame.benchmark.analyzer import (
    RankingMethod,
    ResultsAnalyzer,
)
from endgame.benchmark.learning_curve import (
    FAST_ANCHORS,
    LCDB_ANCHORS,
    LearningCurveConfig,
    LearningCurveExperiment,
    LearningCurveRecord,
    LearningCurveResults,
    quick_learning_curve,
)
from endgame.benchmark.loader import (
    BUILTIN_SUITES,
    DatasetInfo,
    SuiteLoader,
)
from endgame.benchmark.metalearner import (
    MetaLearner,
    PipelineRecommender,
)
from endgame.benchmark.profiler import (
    INFO_THEORY_META_FEATURES,
    LANDMARKING_META_FEATURES,
    SIMPLE_META_FEATURES,
    STATISTICAL_META_FEATURES,
    MetaFeatureSet,
    MetaProfiler,
)
from endgame.benchmark.report import (
    BenchmarkReportGenerator,
    extract_interpretability_outputs,
)
from endgame.benchmark.runner import (
    BenchmarkConfig,
    BenchmarkRunner,
    compare_models,
    quick_benchmark,
)
from endgame.benchmark.synthetic import (
    get_control_dataset,
    get_synthetic_suite,
    make_hidden_structure,
    make_regression_rotated,
    make_rotated_blobs,
    make_xor_rotated,
)
from endgame.benchmark.tracker import (
    DEFAULT_MASTER_DB_PATH,
    ExperimentRecord,
    ExperimentTracker,
    get_experiment_hash,
)

__all__ = [
    # Loader
    "SuiteLoader",
    "DatasetInfo",
    "BUILTIN_SUITES",
    # Profiler
    "MetaProfiler",
    "MetaFeatureSet",
    "SIMPLE_META_FEATURES",
    "STATISTICAL_META_FEATURES",
    "INFO_THEORY_META_FEATURES",
    "LANDMARKING_META_FEATURES",
    # Tracker
    "ExperimentTracker",
    "ExperimentRecord",
    "DEFAULT_MASTER_DB_PATH",
    "get_experiment_hash",
    # Runner
    "BenchmarkRunner",
    "BenchmarkConfig",
    "quick_benchmark",
    "compare_models",
    # Analyzer
    "ResultsAnalyzer",
    "RankingMethod",
    # Meta-learning
    "MetaLearner",
    "PipelineRecommender",
    # Reporting
    "BenchmarkReportGenerator",
    "extract_interpretability_outputs",
    # Learning curves
    "LearningCurveExperiment",
    "LearningCurveConfig",
    "LearningCurveResults",
    "LearningCurveRecord",
    "LCDB_ANCHORS",
    "FAST_ANCHORS",
    "quick_learning_curve",
    # Synthetic datasets
    "make_rotated_blobs",
    "make_hidden_structure",
    "make_xor_rotated",
    "make_regression_rotated",
    "get_synthetic_suite",
    "get_control_dataset",
]
