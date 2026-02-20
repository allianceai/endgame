"""Pipeline orchestrator for AutoML.

This module coordinates the execution of all AutoML pipeline stages
with intelligent time budget management and graceful degradation.
"""

import atexit
import logging
import os
import time
import weakref
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from endgame.automl.presets import PRESETS, PresetConfig
from endgame.automl.search.base import PipelineConfig, SearchResult
from endgame.automl.time_manager import TimeBudgetManager

logger = logging.getLogger(__name__)

# Track non-daemon child processes so they are cleaned up on parent exit.
_active_children: list[weakref.ref] = []


def _cleanup_children():
    import multiprocessing as _mp
    for ref in _active_children:
        p = ref()
        if p is not None and p.is_alive():
            try:
                p.terminate()
                p.join(timeout=3)
                if p.is_alive():
                    p.kill()
            except Exception:
                pass


atexit.register(_cleanup_children)


# ── Process-based model training with hard kill ─────────────────────

def _train_worker(func, args, kwargs, result_queue):
    """Target for the child process.  Runs ``func(*args, **kwargs)``
    and puts the result (or exception) on ``result_queue``."""
    import os
    import signal
    import traceback

    # Allow the child to use most available cores.  We run one model
    # at a time so there's no contention — let BLAS/OpenMP parallelise.
    n_cpus = str(max(1, os.cpu_count() - 2))
    for key in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = n_cpus

    # Force CPU-only mode unless GPU is explicitly enabled via kwargs.
    # CUDA can't re-initialize after fork(), so forked processes always
    # disable CUDA.  GPU training uses the thread-based fallback path.
    use_gpu = kwargs.pop("_use_gpu", False)
    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    signal.signal(signal.SIGTERM, lambda *_: os._exit(1))
    try:
        result = func(*args, **kwargs)
        # result is (oof_pred, score) — pure numpy/float, always picklable.
        # The model is never sent through the queue; the parent refits
        # in-process to avoid pickle/segfault issues with PyTorch etc.
        result_queue.put(("ok", result))
    except BaseException as e:
        try:
            result_queue.put(("error", e))
        except Exception:
            result_queue.put((
                "error",
                RuntimeError(
                    f"{type(e).__name__}: {e}\n"
                    f"{traceback.format_exc()}"
                ),
            ))


def _train_with_timeout(func, *args, sample_weights=None, use_gpu=False, **kwargs):
    """Run *func* in a child **process** with a hard timeout.

    If the child exceeds the deadline, ``Process.kill()`` sends
    SIGKILL which immediately frees all RAM — including any
    joblib / loky workers the model spawned.

    Falls back to thread-based execution if fork is unavailable
    (e.g. macOS with spawn-only context) or if ``use_gpu=True``
    (CUDA cannot re-initialize after fork).

    Parameters match ``_train_model(config, X, y, task_type,
    time_budget, sample_weights=...)``.  The ``time_budget`` arg
    (5th positional) doubles as the deadline.
    """
    import multiprocessing as mp
    import sys

    time_budget = args[4]  # 5th positional = time_budget
    deadline = time_budget + min(30, time_budget * 0.3)

    # Prefer fork (child inherits memory = fast, supports bound methods).
    # Fall back to thread if fork unavailable or GPU mode is active
    # (CUDA cannot re-init after fork).
    use_process = True
    if use_gpu:
        use_process = False
    else:
        try:
            ctx = mp.get_context("fork")
        except ValueError:
            use_process = False

    if not use_process:
        # Thread-based execution (required for GPU, fallback for no-fork)
        import os
        from concurrent.futures import ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        _env_backup = {}
        n_cpus = str(max(1, (os.cpu_count() or 4) - 2))
        for key in ("LOKY_MAX_CPU_COUNT", "OMP_NUM_THREADS",
                     "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            _env_backup[key] = os.environ.get(key)
            os.environ[key] = n_cpus

        # Only force CPU when GPU is not requested
        if not use_gpu:
            _env_backup["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(func, *args, sample_weights=sample_weights)
        try:
            return future.result(timeout=deadline)
        except FuturesTimeout:
            raise TimeoutError(
                f"Model training exceeded {deadline:.0f}s deadline"
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
            for key, val in _env_backup.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    q = ctx.Queue()
    p = ctx.Process(
        target=_train_worker,
        args=(func, args, {"sample_weights": sample_weights, "_use_gpu": use_gpu}, q),
        daemon=False,
    )
    p.start()
    _active_children.append(weakref.ref(p))

    # Drain the queue BEFORE joining.  The child sends a single message:
    #   ("ok", (oof_pred, score))  or  ("error", exception)
    # No fitted model is sent — the parent refits in-process.
    result = None
    try:
        result = q.get(timeout=deadline)
    except KeyboardInterrupt:
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        raise
    except Exception:
        pass

    p.join(timeout=30)
    if p.is_alive():
        p.terminate()
        p.join(timeout=5)
        if p.is_alive():
            p.kill()
            p.join(timeout=2)
        if result is None:
            raise TimeoutError(f"Model training exceeded {deadline:.0f}s deadline")

    if result is None:
        if p.exitcode is not None and p.exitcode != 0:
            detail = f"exit code {p.exitcode}"
            if p.exitcode < 0:
                import signal as _signal
                try:
                    detail = f"signal {_signal.Signals(-p.exitcode).name}"
                except (ValueError, AttributeError):
                    detail = f"signal {-p.exitcode}"
            raise RuntimeError(f"Model training process died ({detail})")

        raise RuntimeError(
            "Model training process exited normally but produced no result "
            "(likely an unhandled exception in the child)"
        )

    status, payload = result
    if status == "error":
        raise payload
    return payload


@dataclass
class StageResult:
    """Result from executing a pipeline stage.

    Attributes
    ----------
    stage_name : str
        Name of the stage.
    success : bool
        Whether the stage completed successfully.
    duration : float
        Time taken in seconds.
    output : Any
        Stage output data.
    error : str, optional
        Error message if stage failed.
    metadata : dict
        Additional metadata.
    """

    stage_name: str
    success: bool
    duration: float
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    """Complete result from running the AutoML pipeline.

    Attributes
    ----------
    best_model : Any
        Best trained model.
    ensemble : Any
        Final ensemble (if built).
    score : float
        Best validation score.
    scores : dict
        All evaluation metrics.
    stage_results : dict
        Results from each pipeline stage.
    total_time : float
        Total execution time in seconds.
    metadata : dict
        Additional pipeline metadata.
    """

    best_model: Any
    ensemble: Any
    score: float
    scores: dict[str, float]
    stage_results: dict[str, StageResult]
    total_time: float
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseStageExecutor(ABC):
    """Base class for pipeline stage executors."""

    @abstractmethod
    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Execute the stage.

        Parameters
        ----------
        context : dict
            Pipeline context with data and intermediate results.
        time_budget : float
            Time budget for this stage in seconds.

        Returns
        -------
        StageResult
            Result of stage execution.
        """
        pass


class ProfilingExecutor(BaseStageExecutor):
    """Executes the data profiling stage."""

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Profile the dataset to extract meta-features.

        Parameters
        ----------
        context : dict
            Must contain 'X' and 'y' keys.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains meta_features dict in output.
        """
        start_time = time.time()

        try:
            X = context["X"]
            y = context["y"]
            task_type = context.get("task_type", "classification")

            # Try to use MetaProfiler if available
            try:
                from endgame.benchmark.profiler import MetaProfiler

                profiler = MetaProfiler()
                meta_features = profiler.profile(X, y, task_type=task_type)
                meta_features_dict = meta_features.features
            except (ImportError, AttributeError):
                meta_features_dict = self._basic_profile(X, y, task_type)
            except Exception:
                # MetaProfiler failed (e.g. mixed dtypes), fall back
                meta_features_dict = self._basic_profile(X, y, task_type)

            duration = time.time() - start_time

            return StageResult(
                stage_name="profiling",
                success=True,
                duration=duration,
                output={"meta_features": meta_features_dict},
                metadata={"n_features_computed": len(meta_features_dict)},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Profiling failed: {e}")

            return StageResult(
                stage_name="profiling",
                success=False,
                duration=duration,
                output={"meta_features": {}},
                error=str(e),
            )

    def _basic_profile(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        task_type: str,
    ) -> dict[str, float]:
        """Basic profiling when MetaProfiler is unavailable.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        task_type : str
            Task type.

        Returns
        -------
        dict
            Basic meta-features.
        """
        if isinstance(X, pd.DataFrame):
            n_cat = len(X.select_dtypes(include=["object", "category"]).columns)
            X_num = X.select_dtypes(include=[np.number])
            n_samples, n_features = X.shape
        else:
            X_num = None
            n_cat = 0
            n_samples, n_features = X.shape

        features = {
            "nr_inst": float(n_samples),
            "nr_attr": float(n_features),
            "nr_cat": float(n_cat),
            "nr_num": float(n_features - n_cat),
            "dimensionality": n_features / max(n_samples, 1),
        }

        # Missing value statistics
        if isinstance(X, pd.DataFrame):
            features["pct_missing"] = X.isna().mean().mean()
        elif np.issubdtype(X.dtype, np.floating):
            features["pct_missing"] = float(np.isnan(X).mean())
        else:
            features["pct_missing"] = 0.0

        # Missing value rates per column
        if isinstance(X, pd.DataFrame):
            missing_rate = X.isna().mean().mean()
        elif np.issubdtype(X.dtype, np.floating):
            missing_rate = float(np.isnan(X).mean())
        else:
            missing_rate = 0.0
        features["missing_rate"] = missing_rate

        # Class imbalance for classification
        if task_type == "classification":
            _, counts = np.unique(y, return_counts=True)
            features["class_imbalance"] = counts.max() / max(counts.min(), 1)
            features["nr_class"] = float(len(counts))
            features["class_imbalance_ratio"] = float(counts.min()) / float(max(counts.max(), 1))

        # Time series detection: check for datetime columns
        n_datetime_cols = 0
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if pd.api.types.is_datetime64_any_dtype(X[col]):
                    n_datetime_cols += 1
                elif X[col].dtype == "object":
                    try:
                        pd.to_datetime(X[col].dropna().head(20))
                        n_datetime_cols += 1
                    except (ValueError, TypeError):
                        pass
        features["n_datetime_cols"] = float(n_datetime_cols)
        features["is_time_series"] = float(n_datetime_cols > 0)

        # Boolean columns
        n_boolean = 0
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if pd.api.types.is_bool_dtype(X[col]) or (X[col].nunique() == 2 and pd.api.types.is_numeric_dtype(X[col]) and set(X[col].dropna().unique()).issubset({0, 1})):
                    n_boolean += 1
        features["n_boolean"] = float(n_boolean)

        # Target skewness for regression
        if task_type == "regression":
            try:
                from scipy.stats import skew
                features["target_skewness"] = float(skew(y))
            except ImportError:
                features["target_skewness"] = 0.0

        return features


class DataCleaningExecutor(BaseStageExecutor):
    """Executes the data cleaning stage.

    Handles outlier removal and label noise detection based on the
    feature engineering level.
    """

    def __init__(self, feature_engineering: str = "none"):
        self.feature_engineering = feature_engineering

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Clean data by removing outliers and noisy labels.

        Parameters
        ----------
        context : dict
            Must contain 'X' and 'y' keys.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains X_cleaned, y_cleaned, clean_mask in output.
        """
        start_time = time.time()
        level = self.feature_engineering

        X = context["X"]
        y = context["y"]

        if level in ("none", "light"):
            duration = time.time() - start_time
            return StageResult(
                stage_name="data_cleaning",
                success=True,
                duration=duration,
                output={"X_cleaned": X, "y_cleaned": y, "clean_mask": None},
                metadata={"skipped": True, "level": level},
            )

        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else X
            n_samples = X_arr.shape[0]
            clean_mask = np.ones(n_samples, dtype=bool)
            outliers_removed = 0
            noisy_removed = 0

            # Moderate+: Outlier detection via IsolationForest
            if level in ("moderate", "aggressive"):
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.6:
                    try:
                        from endgame.anomaly.isolation_forest import IsolationForestDetector
                        detector = IsolationForestDetector(
                            n_estimators=100, contamination="auto",
                        )
                        outlier_preds = detector.fit_predict(X_arr)
                        outlier_mask = outlier_preds == -1
                        outlier_frac = outlier_mask.sum() / n_samples

                        # Safety guard: only remove if <10% flagged
                        if outlier_frac < 0.10:
                            clean_mask &= ~outlier_mask
                            outliers_removed = int(outlier_mask.sum())
                            logger.info(
                                f"DataCleaning: removed {outliers_removed} outliers "
                                f"({outlier_frac:.1%})"
                            )
                        else:
                            logger.info(
                                f"DataCleaning: skipping outlier removal "
                                f"({outlier_frac:.1%} > 10% threshold)"
                            )
                    except ImportError:
                        logger.debug("IsolationForestDetector not available, skipping outlier removal")
                    except Exception as e:
                        logger.debug(f"Outlier detection failed: {e}")

            # Aggressive: Label noise detection
            if level == "aggressive":
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.9:
                    try:
                        from endgame.preprocessing.noise_detection import ConfidentLearningFilter
                        clf = ConfidentLearningFilter(base_estimator="rf", cv=3)
                        noisy_mask = clf.fit_predict(X_arr[clean_mask], y[clean_mask])
                        noisy_frac = noisy_mask.sum() / clean_mask.sum()

                        # Safety guard: only remove if <5% flagged
                        if noisy_frac < 0.05:
                            # Map back to original indices
                            clean_indices = np.where(clean_mask)[0]
                            noisy_indices = clean_indices[noisy_mask]
                            clean_mask[noisy_indices] = False
                            noisy_removed = int(noisy_mask.sum())
                            logger.info(
                                f"DataCleaning: removed {noisy_removed} noisy labels "
                                f"({noisy_frac:.1%})"
                            )
                        else:
                            logger.info(
                                f"DataCleaning: skipping noise removal "
                                f"({noisy_frac:.1%} > 5% threshold)"
                            )
                    except ImportError:
                        logger.debug("ConfidentLearningFilter not available, skipping noise detection")
                    except Exception as e:
                        logger.debug(f"Noise detection failed: {e}")

            # Apply mask
            if isinstance(X, pd.DataFrame):
                X_cleaned = X.iloc[clean_mask].reset_index(drop=True)
            else:
                X_cleaned = X_arr[clean_mask]
            y_cleaned = y[clean_mask]

            duration = time.time() - start_time
            return StageResult(
                stage_name="data_cleaning",
                success=True,
                duration=duration,
                output={
                    "X_cleaned": X_cleaned,
                    "y_cleaned": y_cleaned,
                    "clean_mask": clean_mask,
                },
                metadata={
                    "level": level,
                    "outliers_removed": outliers_removed,
                    "noisy_removed": noisy_removed,
                    "samples_remaining": int(clean_mask.sum()),
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Data cleaning failed: {e}")
            return StageResult(
                stage_name="data_cleaning",
                success=False,
                duration=duration,
                output={"X_cleaned": X, "y_cleaned": y, "clean_mask": None},
                error=str(e),
            )


class PreprocessingExecutor(BaseStageExecutor):
    """Executes the preprocessing stage.

    Uses endgame's advanced preprocessing modules when the preset's
    feature_engineering level is 'light', 'moderate', or 'aggressive'.
    Falls back to basic sklearn for 'none'/'fast' levels.
    """

    def __init__(self, feature_engineering: str = "none"):
        """Initialize preprocessing executor.

        Parameters
        ----------
        feature_engineering : str, default="none"
            Feature engineering level: "none", "light", "moderate", "aggressive".
        """
        self.feature_engineering = feature_engineering

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Build and fit preprocessing pipeline.

        Parameters
        ----------
        context : dict
            Must contain 'X', 'y', and 'meta_features'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains preprocessor and X_processed in output.
        """
        start_time = time.time()

        try:
            X = context["X"]
            y = context["y"]
            meta_features = context.get("meta_features", {})
            task_type = context.get("task_type", "classification")

            # Build preprocessing pipeline
            preprocessor = self._build_preprocessor(
                X, y, meta_features, task_type,
            )

            # Fit and transform
            X_processed = preprocessor.fit_transform(X, y)

            duration = time.time() - start_time

            return StageResult(
                stage_name="preprocessing",
                success=True,
                duration=duration,
                output={
                    "preprocessor": preprocessor,
                    "X_processed": X_processed,
                },
                metadata={
                    "n_steps": len(preprocessor.steps) if hasattr(preprocessor, "steps") else 1,
                    "feature_engineering": self.feature_engineering,
                },
            )

        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")

            # Fallback: try basic preprocessor before giving up
            if self.feature_engineering not in ("none", "fast"):
                try:
                    X = context["X"]
                    y = context["y"]
                    if isinstance(X, pd.DataFrame):
                        numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
                        categorical_cols = X.select_dtypes(
                            include=["object", "category"]
                        ).columns.tolist()
                    else:
                        numeric_cols = list(range(X.shape[1]))
                        categorical_cols = []

                    preprocessor = self._build_basic_preprocessor(
                        numeric_cols, categorical_cols, X,
                    )
                    X_processed = preprocessor.fit_transform(X, y)

                    duration = time.time() - start_time
                    logger.info("Fell back to basic preprocessor successfully")
                    return StageResult(
                        stage_name="preprocessing",
                        success=True,
                        duration=duration,
                        output={
                            "preprocessor": preprocessor,
                            "X_processed": X_processed,
                        },
                        metadata={
                            "fallback": True,
                            "original_error": str(e),
                        },
                    )
                except Exception as fallback_e:
                    logger.error(f"Fallback preprocessing also failed: {fallback_e}")

            duration = time.time() - start_time
            return StageResult(
                stage_name="preprocessing",
                success=False,
                duration=duration,
                output={
                    "preprocessor": None,
                    "X_processed": context["X"],
                },
                error=str(e),
            )

    def _build_preprocessor(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        meta_features: dict[str, float],
        task_type: str = "classification",
    ):
        """Build preprocessing pipeline based on data characteristics and preset.

        The pipeline varies by feature_engineering level:
        - none/fast: SimpleImputer(median) + StandardScaler + OneHotEncoder
        - light: KNNImputer for numeric, SafeTargetEncoder for high-cardinality
          categoricals (>20 unique), OneHotEncoder for low-cardinality
        - moderate: MICEImputer, SafeTargetEncoder, AutoBalancer when
          minority class ratio < 0.2
        - aggressive: All of moderate + CorrelationSelector(threshold=0.95)

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        meta_features : dict
            Dataset meta-features.
        task_type : str, default="classification"
            Task type.

        Returns
        -------
        Pipeline
            Preprocessing pipeline.
        """

        level = self.feature_engineering

        # Identify column types
        if isinstance(X, pd.DataFrame):
            numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            numeric_cols = list(range(X.shape[1]))
            categorical_cols = []

        # Split categoricals into high/low cardinality for target encoding
        high_card_cols = []
        low_card_cols = []
        if isinstance(X, pd.DataFrame) and categorical_cols:
            for col in categorical_cols:
                if X[col].nunique() > 20:
                    high_card_cols.append(col)
                else:
                    low_card_cols.append(col)
        else:
            low_card_cols = categorical_cols

        # === Build the ColumnTransformer ===
        if level in ("none", "fast"):
            return self._build_basic_preprocessor(
                numeric_cols, categorical_cols, X,
            )

        if level == "light":
            return self._build_light_preprocessor(
                numeric_cols, low_card_cols, high_card_cols, X,
            )

        if level == "moderate":
            return self._build_moderate_preprocessor(
                numeric_cols, low_card_cols, high_card_cols,
                X, y, meta_features, task_type,
            )

        if level == "aggressive":
            return self._build_aggressive_preprocessor(
                numeric_cols, low_card_cols, high_card_cols,
                X, y, meta_features, task_type,
            )

        # Default fallback
        return self._build_basic_preprocessor(numeric_cols, categorical_cols, X)

    def _build_basic_preprocessor(self, numeric_cols, categorical_cols, X):
        """Build basic sklearn preprocessor (none/fast level)."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        transformers = []

        if numeric_cols:
            numeric_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if categorical_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("categorical", categorical_pipeline, categorical_cols))

        if transformers:
            return ColumnTransformer(transformers, remainder="passthrough")
        else:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

    def _build_light_preprocessor(self, numeric_cols, low_card_cols, high_card_cols, X):
        """Build light preprocessor with KNNImputer and SafeTargetEncoder."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        transformers = []

        if numeric_cols:
            try:
                from endgame.preprocessing.imputation import KNNImputer as EgKNNImputer
                numeric_pipeline = Pipeline([
                    ("imputer", EgKNNImputer(n_neighbors=5)),
                    ("scaler", StandardScaler()),
                ])
            except ImportError:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if low_card_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat_low", categorical_pipeline, low_card_cols))

        if high_card_cols:
            try:
                from endgame.preprocessing.encoding import SafeTargetEncoder
                transformers.append(("cat_high", SafeTargetEncoder(cols=high_card_cols), high_card_cols))
            except ImportError:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
                ])
                transformers.append(("cat_high", categorical_pipeline, high_card_cols))

        if transformers:
            return ColumnTransformer(transformers, remainder="passthrough")
        else:
            return Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ])

    def _build_moderate_preprocessor(
        self, numeric_cols, low_card_cols, high_card_cols,
        X, y, meta_features, task_type,
    ):
        """Build moderate preprocessor with MICEImputer, SafeTargetEncoder, AutoBalancer."""
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler

        transformers = []

        if numeric_cols:
            try:
                from endgame.preprocessing.imputation import MICEImputer
                numeric_pipeline = Pipeline([
                    ("imputer", MICEImputer(max_iter=10)),
                    ("scaler", StandardScaler()),
                ])
            except ImportError:
                numeric_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ])
            transformers.append(("numeric", numeric_pipeline, numeric_cols))

        if low_card_cols:
            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ])
            transformers.append(("cat_low", categorical_pipeline, low_card_cols))

        if high_card_cols:
            try:
                from endgame.preprocessing.encoding import SafeTargetEncoder
                transformers.append(("cat_high", SafeTargetEncoder(cols=high_card_cols), high_card_cols))
            except ImportError:
                categorical_pipeline = Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
                ])
                transformers.append(("cat_high", categorical_pipeline, high_card_cols))

        if transformers:
            return ColumnTransformer(transformers, remainder="passthrough")

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

    def _build_aggressive_preprocessor(
        self, numeric_cols, low_card_cols, high_card_cols,
        X, y, meta_features, task_type,
    ):
        """Build aggressive preprocessor: moderate + CorrelationSelector."""
        moderate_preprocessor = self._build_moderate_preprocessor(
            numeric_cols, low_card_cols, high_card_cols,
            X, y, meta_features, task_type,
        )

        # Wrap with CorrelationSelector
        from sklearn.pipeline import Pipeline

        try:
            from endgame.feature_selection.filter.correlation import CorrelationSelector
            return Pipeline([
                ("preprocessing", moderate_preprocessor),
                ("feature_selection", CorrelationSelector(threshold=0.95)),
            ])
        except ImportError:
            return moderate_preprocessor


class AdvancedFeatureEngineeringExecutor(BaseStageExecutor):
    """Executes the advanced feature engineering stage.

    Applies PCA, MRMR, Boruta, DAE, and AutoCluster features based
    on the feature engineering level.
    """

    class _FeatureAugmenter:
        """Wraps a transformer so that transform() concatenates new features with input."""

        def __init__(self, transformer, method: str = "transform"):
            self.transformer = transformer
            self.method = method

        def transform(self, X):
            func = getattr(self.transformer, self.method)
            new_features = func(X)
            if new_features.ndim == 1:
                new_features = new_features.reshape(-1, 1)
            return np.hstack([X, new_features])

    def __init__(self, feature_engineering: str = "none"):
        self.feature_engineering = feature_engineering

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Apply advanced feature engineering.

        Parameters
        ----------
        context : dict
            Must contain 'X_processed' and 'y'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains X_engineered and feature_transformers in output.
        """
        start_time = time.time()
        level = self.feature_engineering

        # Pick the most-processed data available
        X = context.get("X_processed", context.get("X_cleaned", context.get("X")))
        y = context.get("y_cleaned", context.get("y"))

        if level in ("none", "light"):
            duration = time.time() - start_time
            return StageResult(
                stage_name="feature_engineering",
                success=True,
                duration=duration,
                output={"X_engineered": X, "feature_transformers": []},
                metadata={"skipped": True, "level": level},
            )

        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            n_samples, n_features = X_arr.shape
            feature_transformers = []

            # PCA for high-dimensional data (d > 500)
            if n_features > 500:
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.3:
                    try:
                        from endgame.dimensionality_reduction.linear import PCAReducer
                        pca = PCAReducer(n_components=0.95)
                        X_arr = pca.fit_transform(X_arr)
                        feature_transformers.append(("pca", pca))
                        logger.info(
                            f"FeatureEngineering: PCA reduced {n_features} -> "
                            f"{X_arr.shape[1]} features"
                        )
                    except ImportError:
                        logger.debug("PCAReducer not available, skipping PCA")
                    except Exception as e:
                        logger.debug(f"PCA failed: {e}")

            if level == "moderate":
                # MRMR feature selection
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.8:
                    try:
                        from endgame.feature_selection.filter.mrmr import MRMRSelector
                        n_select = max(10, X_arr.shape[1] // 2)
                        mrmr = MRMRSelector(n_features=n_select)
                        X_arr = mrmr.fit_transform(X_arr, y)
                        feature_transformers.append(("mrmr", mrmr))
                        logger.info(f"FeatureEngineering: MRMR selected {X_arr.shape[1]} features")
                    except ImportError:
                        logger.debug("MRMRSelector not available, skipping MRMR")
                    except Exception as e:
                        logger.debug(f"MRMR failed: {e}")

            elif level == "aggressive":
                # Boruta feature selection
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.5:
                    try:
                        from endgame.feature_selection.wrapper.boruta import BorutaSelector
                        boruta = BorutaSelector(max_iter=50)
                        X_arr = boruta.fit_transform(X_arr, y)
                        feature_transformers.append(("boruta", boruta))
                        logger.info(f"FeatureEngineering: Boruta selected {X_arr.shape[1]} features")
                    except ImportError:
                        logger.debug("BorutaSelector not available, skipping Boruta")
                    except Exception as e:
                        logger.debug(f"Boruta failed: {e}")

                # Denoising autoencoder features
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.75:
                    try:
                        from endgame.preprocessing.dae import DenoisingAutoEncoder
                        dae = DenoisingAutoEncoder(hidden_dims=[128, 64])
                        dae_features = dae.fit_transform(X_arr)
                        X_arr = np.hstack([X_arr, dae_features])
                        feature_transformers.append(("dae", self._FeatureAugmenter(dae)))
                        logger.info(f"FeatureEngineering: DAE added {dae_features.shape[1]} features")
                    except ImportError:
                        logger.debug("DenoisingAutoEncoder not available, skipping DAE")
                    except Exception as e:
                        logger.debug(f"DAE failed: {e}")

                # AutoCluster features
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.9:
                    try:
                        from endgame.clustering.auto import AutoCluster
                        clusterer = AutoCluster()
                        cluster_labels = clusterer.fit_predict(X_arr).reshape(-1, 1)
                        X_arr = np.hstack([X_arr, cluster_labels])
                        feature_transformers.append(("cluster", self._FeatureAugmenter(clusterer, method="predict")))
                        logger.info("FeatureEngineering: added cluster label feature")
                    except ImportError:
                        logger.debug("AutoCluster not available, skipping clustering")
                    except Exception as e:
                        logger.debug(f"AutoCluster failed: {e}")

            duration = time.time() - start_time
            return StageResult(
                stage_name="feature_engineering",
                success=True,
                duration=duration,
                output={
                    "X_engineered": X_arr,
                    "feature_transformers": feature_transformers,
                },
                metadata={
                    "level": level,
                    "n_transformers": len(feature_transformers),
                    "output_features": X_arr.shape[1],
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Feature engineering failed: {e}")
            return StageResult(
                stage_name="feature_engineering",
                success=False,
                duration=duration,
                output={"X_engineered": X, "feature_transformers": []},
                error=str(e),
            )


class DataAugmentationExecutor(BaseStageExecutor):
    """Executes the data augmentation stage.

    Uses triage-informed SMOTE and sample weighting for imbalanced
    classification tasks.
    """

    def __init__(self, feature_engineering: str = "none"):
        self.feature_engineering = feature_engineering

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Apply data augmentation for imbalanced classification.

        Parameters
        ----------
        context : dict
            Must contain feature data and 'y'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains X_augmented, y_augmented, sample_weights in output.
        """
        start_time = time.time()
        level = self.feature_engineering
        task_type = context.get("task_type", "classification")

        # Pick the most-processed data
        X = context.get("X_engineered", context.get("X_processed",
            context.get("X_cleaned", context.get("X"))))
        y = context.get("y_cleaned", context.get("y"))

        # Skip for regression or none/light level
        if task_type == "regression" or level in ("none", "light"):
            duration = time.time() - start_time
            return StageResult(
                stage_name="data_augmentation",
                success=True,
                duration=duration,
                output={
                    "X_augmented": X,
                    "y_augmented": y,
                    "sample_weights": None,
                },
                metadata={"skipped": True, "reason": "regression or light level"},
            )

        # Check class imbalance
        meta_features = context.get("meta_features", {})
        imbalance_ratio = meta_features.get("class_imbalance_ratio", 1.0)

        if imbalance_ratio >= 0.3:
            duration = time.time() - start_time
            return StageResult(
                stage_name="data_augmentation",
                success=True,
                duration=duration,
                output={
                    "X_augmented": X,
                    "y_augmented": y,
                    "sample_weights": None,
                },
                metadata={"skipped": True, "reason": "balanced data"},
            )

        try:
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            sample_weights = None

            if level == "moderate":
                try:
                    from endgame.preprocessing.imbalance import AutoBalancer
                    balancer = AutoBalancer(strategy="auto")
                    X_arr, y = balancer.fit_resample(X_arr, y)
                    logger.info(f"DataAugmentation: AutoBalancer -> {len(y)} samples")
                except ImportError:
                    logger.debug("No augmentation modules available")

            elif level == "aggressive":
                # Aggressive augmentation: apply class weighting
                logger.debug("Aggressive augmentation: using class weights")

            duration = time.time() - start_time
            return StageResult(
                stage_name="data_augmentation",
                success=True,
                duration=duration,
                output={
                    "X_augmented": X_arr,
                    "y_augmented": y,
                    "sample_weights": sample_weights,
                },
                metadata={"level": level, "n_samples": len(y)},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Data augmentation failed: {e}")
            return StageResult(
                stage_name="data_augmentation",
                success=False,
                duration=duration,
                output={
                    "X_augmented": X,
                    "y_augmented": y,
                    "sample_weights": None,
                },
                error=str(e),
            )


class ModelSelectionExecutor(BaseStageExecutor):
    """Executes the model selection stage."""

    def __init__(self, search_strategy=None):
        """Initialize model selection executor.

        Parameters
        ----------
        search_strategy : BaseSearchStrategy, optional
            Search strategy to use. If None, uses PortfolioSearch.
        """
        self.search_strategy = search_strategy

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Select model configurations to train.

        Parameters
        ----------
        context : dict
            Must contain 'meta_features' and 'max_models'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains model_configs in output.
        """
        start_time = time.time()

        try:
            meta_features = context.get("meta_features", {})
            max_models = context.get("max_models", 5)
            task_type = context.get("task_type", "classification")

            # Initialize search strategy if needed
            if self.search_strategy is None:
                from endgame.automl.search.portfolio import PortfolioSearch

                self.search_strategy = PortfolioSearch(
                    task_type=task_type,
                    preset=context.get("preset", "medium_quality"),
                )

            # Get model configurations
            configs = self.search_strategy.suggest(
                meta_features=meta_features,
                n_suggestions=max_models,
            )

            duration = time.time() - start_time

            return StageResult(
                stage_name="model_selection",
                success=True,
                duration=duration,
                output={"model_configs": configs},
                metadata={"n_models_selected": len(configs)},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Model selection failed: {e}")

            # Fallback to LGBM
            fallback_config = PipelineConfig(
                model_name="lgbm",
                model_params={"n_estimators": 1000, "learning_rate": 0.05},
            )

            return StageResult(
                stage_name="model_selection",
                success=False,
                duration=duration,
                output={"model_configs": [fallback_config]},
                error=str(e),
            )


class ModelTrainingExecutor(BaseStageExecutor):
    """Executes the model training stage."""

    def __init__(
        self,
        cv_folds: int = 5,
        parallel: bool = True,
        feature_engineering: str = "none",
        verbose: int = 1,
        min_model_time: float = 300.0,
        max_model_time: float = 600.0,
        eval_metric: str = "auto",
        early_stopping_rounds: int = 50,
        use_gpu: bool = False,
    ):
        """Initialize model training executor.

        Parameters
        ----------
        cv_folds : int, default=5
            Number of cross-validation folds.
        parallel : bool, default=True
            Whether to train models in parallel.
        feature_engineering : str, default="none"
            Feature engineering level (used to decide target transform).
        verbose : int, default=1
            Verbosity level.
        min_model_time : float, default=300.0
            Minimum time budget (seconds) guaranteed to each model.
            Models are trained in priority order; once the remaining
            stage budget drops below this floor, training stops.
        max_model_time : float, default=600.0
            Hard ceiling (seconds) for any single model. If a model
            exceeds this, its training thread is abandoned and the
            pipeline moves on to the next model.
        eval_metric : str, default="auto"
            Evaluation metric for scoring models.
        early_stopping_rounds : int, default=50
            Early stopping patience for GBDT models during CV.
        use_gpu : bool, default=False
            Whether to enable GPU acceleration.
        """
        self.cv_folds = cv_folds
        self.parallel = parallel
        self.feature_engineering = feature_engineering
        self.verbose = verbose
        self.min_model_time = min_model_time
        self.max_model_time = max_model_time
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Train models with cross-validation.

        Parameters
        ----------
        context : dict
            Must contain 'X_processed', 'y', 'model_configs'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains trained_models and oof_predictions in output.
        """
        # Prevent CUDA from being initialized in the parent process.
        # _refit_model runs here and PyTorch models can lazily init CUDA
        # which would poison all subsequent fork()ed children with
        # "Cannot re-initialize CUDA in forked subprocess".
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

        # Auto-select CV strategy based on data characteristics
        self._cv_strategy = self._select_cv_strategy(context)

        start_time = time.time()

        # Initialise outside try so partial results survive exceptions
        trained_models: dict[str, Any] = {}
        oof_predictions: dict[str, np.ndarray] = {}
        results: list = []

        try:
            # Use most-processed data available
            X = context.get("X_augmented",
                context.get("X_engineered",
                    context.get("X_processed",
                        context.get("X_cleaned", context.get("X")))))
            y = context.get("y_augmented",
                context.get("y_cleaned", context.get("y")))
            # Keep a reference to raw (pre-preprocessing) data so that
            # configs with their own imputer/encoder can start from
            # unprocessed features instead of the global pipeline output.
            X_raw = context.get("X_cleaned", context.get("X"))
            configs = context["model_configs"]
            task_type = context.get("task_type", "classification")
            sample_weights = context.get("sample_weights")

            # --- Budget planning ---
            # Each model is guaranteed at least min_model_time.  If the
            # stage budget can't accommodate every model at that rate,
            # we train as many as we can in priority order (configs are
            # already sorted by the search strategy).  Remaining models
            # will be tried in the continuous loop.
            if time_budget == float("inf") or time_budget <= 0:
                max_trainable = len(configs)
            else:
                max_trainable = max(1, int(time_budget / self.min_model_time))
            if max_trainable < len(configs):
                if self.verbose > 0:
                    print(
                        f"    Budget ({time_budget:.0f}s) allows "
                        f"{max_trainable}/{len(configs)} models "
                        f"(≥{self.min_model_time:.0f}s each); "
                        f"rest deferred to continuous loop"
                    )
                configs = configs[:max_trainable]

            for i, config in enumerate(configs):
                model_start = time.time()

                # Stop if remaining budget can't give the next model
                # a meaningful amount of time
                if time_budget == float("inf"):
                    remaining = float("inf")
                else:
                    remaining = time_budget - (time.time() - start_time)
                if remaining < self.min_model_time and trained_models:
                    if self.verbose > 0:
                        print(
                            f"    Remaining budget ({remaining:.0f}s) < "
                            f"min_model_time ({self.min_model_time:.0f}s) — "
                            f"stopping after {i}/{len(configs)} models"
                        )
                    break

                # Per-model budget: fair share of what's left, clamped
                # between min_model_time and max_model_time.
                if remaining == float("inf"):
                    model_budget = self.max_model_time
                else:
                    models_left = max(len(configs) - i, 1)
                    model_budget = max(remaining / models_left, self.min_model_time)
                    model_budget = min(model_budget, remaining, self.max_model_time)

                if self.verbose > 0:
                    print(
                        f"    [{i+1}/{len(configs)}] Training {config.model_name} "
                        f"(budget {model_budget:.0f}s)...",
                        end="", flush=True,
                    )

                try:
                    # If the config has its own imputer or encoder, use raw
                    # data so the per-config preprocessing can apply its own
                    # strategy.  Otherwise use globally-preprocessed data.
                    _has_preproc = any(
                        s[0] in ("imputer", "encoder")
                        for s in (config.preprocessing or [])
                    )
                    X_for_config = X_raw if (_has_preproc and X_raw is not None) else X

                    oof_pred, score = _train_with_timeout(
                        self._cv_score_model,
                        config, X_for_config, y, task_type, model_budget,
                        sample_weights=sample_weights,
                        use_gpu=self.use_gpu,
                    )

                    oof_predictions[config.model_name] = oof_pred

                    # Skip refit for partial-data bandit rungs — these
                    # configs are just being scored for selection, not
                    # used for final prediction.
                    frac = config.metadata.get("data_fraction", 1.0)
                    if frac < 1.0:
                        if self.verbose > 0:
                            print(
                                f" score={score:.4f} "
                                f"({time.time()-model_start:.1f}s, "
                                f"{frac:.0%} data, skip refit)",
                            )
                    else:
                        # Refit on all data in the parent process — no
                        # pickle boundary, so PyTorch/C-extension models
                        # that segfault during serialization work fine.
                        refit_start = time.time()
                        try:
                            model = self._refit_model(
                                config, X_for_config, y, task_type,
                                sample_weights=sample_weights,
                            )
                            trained_models[config.model_name] = model
                            refit_time = time.time() - refit_start
                            print(
                                f" refit {refit_time:.0f}s",
                                end="", flush=True,
                            )
                        except Exception as refit_err:
                            logger.warning(
                                f"Refit failed for {config.model_name}: {refit_err}"
                            )

                    result = SearchResult(
                        config=config,
                        score=score,
                        scores={"primary": score},
                        fit_time=time.time() - model_start,
                        oof_predictions=oof_pred,
                        success=True,
                    )
                    results.append(result)

                    if self.verbose > 0 and frac >= 1.0:
                        print(f" score={score:.4f} ({time.time()-model_start:.1f}s)")
                    logger.debug(f"Trained {config.model_name}: score={score:.4f}")

                except TimeoutError:
                    elapsed = time.time() - model_start
                    if self.verbose > 0:
                        print(f" KILLED after {elapsed:.0f}s (budget was {model_budget:.0f}s)")
                    logger.warning(
                        f"Model {config.model_name} exceeded time budget "
                        f"({elapsed:.0f}s > {model_budget:.0f}s), killed"
                    )
                    result = SearchResult(
                        config=config,
                        score=float("-inf"),
                        fit_time=elapsed,
                        success=False,
                        error=f"Killed after {elapsed:.0f}s",
                    )
                    results.append(result)

                except RuntimeError as e:
                    err_str = str(e)
                    is_cuda_oom = "CUDA out of memory" in err_str
                    if is_cuda_oom and self.use_gpu:
                        if self.verbose > 0:
                            print(" CUDA OOM — falling back to CPU")
                        logger.warning(
                            f"CUDA OOM for {config.model_name}, "
                            f"falling back to CPU retraining"
                        )
                        # Retry with GPU disabled for this model
                        try:
                            _prev = os.environ.get("CUDA_VISIBLE_DEVICES")
                            os.environ["CUDA_VISIBLE_DEVICES"] = ""
                            oof_pred, score = self._cv_score_model(
                                config, X_for_config, y, task_type,
                                model_budget,
                                sample_weights=sample_weights,
                            )
                            oof_predictions[config.model_name] = oof_pred
                            model = self._refit_model(
                                config, X_for_config, y, task_type,
                                sample_weights=sample_weights,
                            )
                            trained_models[config.model_name] = model
                            result = SearchResult(
                                config=config,
                                score=score,
                                scores={"primary": score},
                                fit_time=time.time() - model_start,
                                oof_predictions=oof_pred,
                                success=True,
                            )
                            results.append(result)
                            if self.verbose > 0:
                                print(f" score={score:.4f} (CPU fallback, {time.time()-model_start:.1f}s)")
                        except Exception as cpu_err:
                            if self.verbose > 0:
                                print(f" CPU fallback FAILED ({cpu_err})")
                            logger.warning(f"CPU fallback failed for {config.model_name}: {cpu_err}")
                            result = SearchResult(
                                config=config,
                                score=float("-inf"),
                                fit_time=time.time() - model_start,
                                success=False,
                                error=str(cpu_err),
                            )
                            results.append(result)
                        finally:
                            if _prev is None:
                                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                            else:
                                os.environ["CUDA_VISIBLE_DEVICES"] = _prev
                        continue
                    # Not a CUDA OOM — fall through to generic handler
                    if self.verbose > 0:
                        print(f" FAILED ({e})")
                    logger.warning(f"Failed to train {config.model_name}: {e}")
                    result = SearchResult(
                        config=config,
                        score=float("-inf"),
                        fit_time=time.time() - model_start,
                        success=False,
                        error=str(e),
                    )
                    results.append(result)

                except Exception as e:
                    if self.verbose > 0:
                        print(f" FAILED ({e})")
                    logger.warning(f"Failed to train {config.model_name}: {e}")
                    result = SearchResult(
                        config=config,
                        score=float("-inf"),
                        fit_time=time.time() - model_start,
                        success=False,
                        error=str(e),
                    )
                    results.append(result)

        except Exception as e:
            logger.error(f"Model training failed: {e}")

        # --- Fallback: if all models failed/killed, train a fast model ---
        if not trained_models and results:
            logger.warning(
                "All models failed or were killed. "
                "Training fast fallback model (HistGradientBoosting)."
            )
            if self.verbose > 0:
                print("    All models killed/failed — training fallback model...", end="", flush=True)
            try:
                X = context.get("X_augmented",
                    context.get("X_engineered",
                        context.get("X_processed",
                            context.get("X_cleaned", context.get("X")))))
                y = context.get("y_augmented",
                    context.get("y_cleaned", context.get("y")))
                task_type = context.get("task_type", "classification")

                if task_type == "regression":
                    from sklearn.ensemble import HistGradientBoostingRegressor
                    fallback = HistGradientBoostingRegressor(
                        max_iter=100, max_depth=6, random_state=42,
                    )
                else:
                    from sklearn.ensemble import HistGradientBoostingClassifier
                    fallback = HistGradientBoostingClassifier(
                        max_iter=100, max_depth=6, random_state=42,
                    )

                fallback.fit(X, y)
                trained_models["fallback_hgb"] = fallback
                if self.verbose > 0:
                    print(" OK (fallback)")
                logger.info("Fallback model trained successfully.")
            except Exception as fallback_err:
                if self.verbose > 0:
                    print(f" FAILED ({fallback_err})")
                logger.error(f"Fallback model also failed: {fallback_err}")

        duration = time.time() - start_time

        return StageResult(
            stage_name="model_training",
            success=len(trained_models) > 0,
            duration=duration,
            output={
                "trained_models": trained_models,
                "oof_predictions": oof_predictions,
                "results": results,
            },
            metadata={
                "n_models_trained": len(trained_models),
                "n_models_failed": max(0, len(results) - len(trained_models)),
            },
        )

    def _cv_score_model(
        self,
        config: PipelineConfig,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        time_budget: float,
        sample_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Score a model via cross-validation (no final fit).

        Runs in a forked child process.  Returns only numpy/float data
        so there are never pickling issues.  The parent calls
        ``_refit_model`` afterwards in-process.

        If the config has ``metadata["data_fraction"] < 1.0`` (set by
        BanditSearch), only that fraction of data is used for CV.
        """
        import inspect

        from sklearn.base import clone
        from sklearn.model_selection import KFold, StratifiedKFold

        # ── Data fraction subsampling (BanditSearch support) ────────
        data_fraction = config.metadata.get("data_fraction", 1.0)
        if data_fraction < 1.0:
            n = len(X)
            n_sub = max(20, int(n * data_fraction))
            if n_sub < n:
                rng = np.random.RandomState(42)
                if task_type in ("classification", "binary", "multiclass"):
                    # Stratified subsample to preserve class balance
                    from sklearn.model_selection import StratifiedShuffleSplit
                    sss = StratifiedShuffleSplit(
                        n_splits=1, train_size=n_sub, random_state=42,
                    )
                    idx, _ = next(sss.split(X, y))
                else:
                    idx = rng.choice(n, size=n_sub, replace=False)
                    idx.sort()
                X = X[idx]
                y = y[idx]
                if sample_weights is not None:
                    sample_weights = sample_weights[idx]

        model = self._instantiate_model(config, task_type)

        if task_type == "regression" and self.feature_engineering in ("moderate", "aggressive"):
            try:
                from scipy.stats import skew
                target_skewness = skew(y)
                if abs(target_skewness) > 1.0:
                    from endgame.preprocessing.target_transform import TargetTransformer
                    model = TargetTransformer(regressor=model, method="auto")
            except ImportError:
                pass

        model = self._wrap_with_preprocessing(model, config)

        supports_sw = False
        if sample_weights is not None:
            try:
                fit_sig = inspect.signature(model.fit)
                supports_sw = "sample_weight" in fit_sig.parameters
            except (ValueError, TypeError):
                pass

        # Use the intelligently-selected CV strategy (set by execute())
        cv = getattr(self, "_cv_strategy", None)
        if cv is None:
            if task_type == "classification":
                cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            else:
                cv = KFold(n_splits=self.cv_folds, shuffle=True, random_state=42)

        use_proba = task_type == "classification" and hasattr(model, "predict_proba")

        # ── Detect GBDT models for early stopping ────────────────────
        _GBDT_NAMES = {"lgbm", "xgb", "catboost", "ngboost"}
        is_gbdt = config.model_name in _GBDT_NAMES
        es_rounds = getattr(self, "early_stopping_rounds", 50) if is_gbdt else 0

        oof_pred = None
        cv_start = time.time()

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            fold_start = time.time()
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            fold_model = clone(model)

            fit_kwargs: dict[str, Any] = {}
            if supports_sw and sample_weights is not None:
                fit_kwargs["sample_weight"] = sample_weights[train_idx]

            # ── Early stopping for GBDTs ─────────────────────────────
            if es_rounds > 0:
                try:
                    fit_kwargs.update(
                        self._get_early_stopping_kwargs(
                            fold_model, config.model_name,
                            X_val, y_val, es_rounds,
                        )
                    )
                except Exception:
                    pass  # Fall back to training without early stopping

            fold_model.fit(X_tr, y_tr, **fit_kwargs)

            if use_proba:
                preds = fold_model.predict_proba(X_val)
            else:
                preds = fold_model.predict(X_val)

            if oof_pred is None:
                if preds.ndim > 1:
                    oof_pred = np.zeros((len(y), preds.shape[1]), dtype=preds.dtype)
                else:
                    oof_pred = np.zeros(len(y), dtype=preds.dtype)
            oof_pred[val_idx] = preds

            fold_time = time.time() - fold_start
            n_total_folds = cv.get_n_splits(X, y) if hasattr(cv, "get_n_splits") else self.cv_folds
            print(
                f" fold {fold_idx + 1}/{n_total_folds} {fold_time:.0f}s",
                end="", flush=True,
            )

        score = self._score_oof(oof_pred, y, task_type)
        return oof_pred, score

    def _refit_model(
        self,
        config: PipelineConfig,
        X: np.ndarray,
        y: np.ndarray,
        task_type: str,
        sample_weights: np.ndarray | None = None,
    ) -> Any:
        """Refit a model on all data in the parent process (no pickle)."""
        import inspect

        model = self._instantiate_model(config, task_type)

        if task_type == "regression" and self.feature_engineering in ("moderate", "aggressive"):
            try:
                from scipy.stats import skew
                target_skewness = skew(y)
                if abs(target_skewness) > 1.0:
                    from endgame.preprocessing.target_transform import TargetTransformer
                    model = TargetTransformer(regressor=model, method="auto")
            except ImportError:
                pass

        model = self._wrap_with_preprocessing(model, config)

        supports_sw = False
        if sample_weights is not None:
            try:
                fit_sig = inspect.signature(model.fit)
                supports_sw = "sample_weight" in fit_sig.parameters
            except (ValueError, TypeError):
                pass

        if supports_sw and sample_weights is not None:
            model.fit(X, y, sample_weight=sample_weights)
        else:
            model.fit(X, y)

        return model

    def _select_cv_strategy(self, context: dict[str, Any]) -> Any:
        """Auto-select a CV splitter based on data characteristics.

        Examines meta-features and context to choose the most appropriate
        cross-validation strategy:

        - Time series data -> PurgedTimeSeriesSplit (if time column detected)
        - Grouped data -> StratifiedGroupKFold (if groups present)
        - Small datasets (< 500 samples) -> RepeatedStratifiedKFold / RepeatedKFold
        - Imbalanced classification (minority < 10%) -> StratifiedKFold
        - Default classification -> StratifiedKFold
        - Default regression -> KFold
        """
        from sklearn.model_selection import (
            KFold,
            RepeatedKFold,
            RepeatedStratifiedKFold,
            StratifiedKFold,
        )

        task_type = context.get("task_type", "classification")
        meta_features = context.get("meta_features", {})
        y = context.get("y_augmented", context.get("y_cleaned", context.get("y")))
        n_folds = self.cv_folds
        is_clf = task_type in ("classification", "binary", "multiclass")

        n_samples = len(y) if y is not None else meta_features.get("nr_inst", 10_000)

        # ── Time series: use purged time series split ───────────────
        if meta_features.get("is_timeseries") or context.get("time_column"):
            try:
                from endgame.validation import PurgedTimeSeriesSplit
                cv = PurgedTimeSeriesSplit(
                    n_splits=n_folds,
                    embargo_pct=0.01,
                )
                if self.verbose > 0:
                    print(f"    CV strategy: PurgedTimeSeriesSplit (n_splits={n_folds})")
                return cv
            except ImportError:
                pass

        # ── Grouped data: use group-aware splitting ─────────────────
        groups = context.get("groups")
        if groups is not None:
            try:
                from endgame.validation import StratifiedGroupKFold as SGKFold
                cv = SGKFold(n_splits=min(n_folds, len(np.unique(groups))))
                if self.verbose > 0:
                    print(f"    CV strategy: StratifiedGroupKFold (n_splits={cv.n_splits})")
                return cv
            except ImportError:
                from sklearn.model_selection import GroupKFold
                cv = GroupKFold(n_splits=min(n_folds, len(np.unique(groups))))
                if self.verbose > 0:
                    print(f"    CV strategy: GroupKFold (n_splits={cv.n_splits})")
                return cv

        # ── Small datasets: use repeated k-fold for more stable estimates
        if n_samples < 500:
            n_repeats = 3
            if is_clf:
                cv = RepeatedStratifiedKFold(
                    n_splits=n_folds, n_repeats=n_repeats, random_state=42,
                )
                if self.verbose > 0:
                    print(
                        f"    CV strategy: RepeatedStratifiedKFold "
                        f"({n_folds}x{n_repeats}, small dataset: {n_samples} samples)"
                    )
            else:
                cv = RepeatedKFold(
                    n_splits=n_folds, n_repeats=n_repeats, random_state=42,
                )
                if self.verbose > 0:
                    print(
                        f"    CV strategy: RepeatedKFold "
                        f"({n_folds}x{n_repeats}, small dataset: {n_samples} samples)"
                    )
            return cv

        # ── Default: stratified for classification, plain for regression
        if is_clf:
            cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
            if self.verbose > 1:
                print(f"    CV strategy: StratifiedKFold (n_splits={n_folds})")
        else:
            cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)
            if self.verbose > 1:
                print(f"    CV strategy: KFold (n_splits={n_folds})")

        return cv

    def _score_oof(
        self, oof_pred: np.ndarray, y: np.ndarray, task_type: str,
    ) -> float:
        """Score OOF predictions using the configured eval_metric."""
        metric = self.eval_metric
        is_clf = task_type in ("classification", "binary", "multiclass")

        if metric == "auto":
            metric = "roc_auc" if is_clf else "rmse"

        if is_clf and oof_pred.ndim > 1:
            # For binary: use probability of positive class
            if oof_pred.shape[1] == 2:
                oof_proba = oof_pred[:, 1]
            else:
                oof_proba = oof_pred
            pred_labels = np.argmax(oof_pred, axis=1)
        elif is_clf:
            oof_proba = oof_pred
            pred_labels = (oof_pred > 0.5).astype(int)
        else:
            oof_proba = None
            pred_labels = None

        if metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            try:
                if oof_proba is not None and oof_proba.ndim > 1:
                    return roc_auc_score(y, oof_proba, multi_class="ovr")
                return roc_auc_score(y, oof_proba)
            except ValueError:
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, pred_labels)
        elif metric == "log_loss":
            from sklearn.metrics import log_loss
            return -log_loss(y, oof_proba if oof_proba is not None else oof_pred)
        elif metric == "accuracy":
            from sklearn.metrics import accuracy_score
            return accuracy_score(y, pred_labels)
        elif metric == "f1":
            from sklearn.metrics import f1_score
            return f1_score(y, pred_labels, average="binary" if len(np.unique(y)) == 2 else "macro")
        elif metric in ("rmse", "neg_rmse"):
            from sklearn.metrics import root_mean_squared_error
            return -root_mean_squared_error(y, oof_pred)
        elif metric in ("r2", "r_squared"):
            from sklearn.metrics import r2_score
            return r2_score(y, oof_pred)
        elif metric == "mae":
            from sklearn.metrics import mean_absolute_error
            return -mean_absolute_error(y, oof_pred)
        else:
            if is_clf:
                from sklearn.metrics import accuracy_score
                return accuracy_score(y, pred_labels)
            from sklearn.metrics import r2_score
            return r2_score(y, oof_pred)

    @staticmethod
    def _get_early_stopping_kwargs(
        model: Any,
        model_name: str,
        X_val: np.ndarray,
        y_val: np.ndarray,
        early_stopping_rounds: int,
    ) -> dict[str, Any]:
        """Build early-stopping fit kwargs for GBDT models.

        Handles both bare estimators and sklearn Pipelines (where
        preprocessing steps must transform X_val before it reaches the
        final estimator).

        Returns a dict of keyword arguments to pass to ``model.fit()``.
        """
        from sklearn.pipeline import Pipeline

        # If the model is wrapped in a Pipeline, we need to transform
        # X_val through the preprocessing steps and prefix the kwarg
        # names with the final step name.
        if isinstance(model, Pipeline):
            # Transform X_val through all steps except the final estimator
            X_val_transformed = X_val
            for step_name, step_transformer in model.steps[:-1]:
                if hasattr(step_transformer, "transform"):
                    X_val_transformed = step_transformer.transform(X_val_transformed)
            final_step_name = model.steps[-1][0]
            prefix = f"{final_step_name}__"
        else:
            X_val_transformed = X_val
            prefix = ""

        kwargs: dict[str, Any] = {}

        if model_name == "lgbm":
            kwargs[f"{prefix}eval_set"] = [(X_val_transformed, y_val)]
            try:
                import lightgbm as lgb
                kwargs[f"{prefix}callbacks"] = [
                    lgb.early_stopping(early_stopping_rounds, verbose=False),
                    lgb.log_evaluation(period=0),
                ]
            except ImportError:
                # LightGBM not available — skip early stopping
                return {}
        elif model_name == "xgb":
            kwargs[f"{prefix}eval_set"] = [(X_val_transformed, y_val)]
            kwargs[f"{prefix}verbose"] = False
        elif model_name == "catboost":
            kwargs[f"{prefix}eval_set"] = (X_val_transformed, y_val)
            kwargs[f"{prefix}early_stopping_rounds"] = early_stopping_rounds
            kwargs[f"{prefix}verbose"] = 0
        elif model_name == "ngboost":
            kwargs[f"{prefix}X_val"] = X_val_transformed
            kwargs[f"{prefix}Y_val"] = y_val
            kwargs[f"{prefix}early_stopping_rounds"] = early_stopping_rounds
        else:
            return {}

        return kwargs

    @staticmethod
    def _wrap_with_preprocessing(
        model: Any,
        config: "PipelineConfig",
    ) -> Any:
        """Wrap *model* in a sklearn Pipeline if the config specifies
        per-model preprocessing or feature-selection steps.

        The global preprocessing (run by ``PreprocessingExecutor``) has
        already been applied to X.  These per-config steps add *extra*
        transformations so that different pipeline configs can explore
        different scaling, encoding, or feature-selection strategies.
        """
        steps_spec = config.preprocessing or []
        if not steps_spec:
            return model

        from sklearn.pipeline import Pipeline

        pipeline_steps: list[tuple[str, Any]] = []

        for step_name, params in steps_spec:
            try:
                transformer = _build_preprocessing_step(step_name, params)
                if transformer is not None:
                    pipeline_steps.append((step_name, transformer))
            except Exception as e:
                logger.debug(f"Could not build preprocessing step {step_name}: {e}")

        if not pipeline_steps:
            return model

        pipeline_steps.append(("model", model))
        return Pipeline(pipeline_steps)

    def _instantiate_model(self, config: PipelineConfig, task_type: str):
        """Instantiate a model from configuration.

        Parameters
        ----------
        config : PipelineConfig
            Model configuration.
        task_type : str
            Task type.

        Returns
        -------
        model
            Instantiated model.
        """
        from endgame.automl.model_registry import get_model_class

        model_class = get_model_class(config.model_name)

        if model_class is None:
            # Fallback to sklearn
            if config.model_name == "lgbm":
                if task_type == "classification":
                    from lightgbm import LGBMClassifier

                    model_class = LGBMClassifier
                else:
                    from lightgbm import LGBMRegressor

                    model_class = LGBMRegressor
            else:
                raise ValueError(f"Unknown model: {config.model_name}")

        return model_class(**config.model_params)


class _ImportanceMaskSelector:
    """Sklearn-compatible feature selector using a precomputed boolean mask.

    Used by iterative feature selection: the mask is derived from
    aggregate feature importances of previously trained models.
    """

    def __init__(self, mask: list[bool]):
        self._mask = np.asarray(mask, dtype=bool)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "iloc"):
            return X.iloc[:, self._mask[:X.shape[1]]]
        return X[:, self._mask[:X.shape[1]]]

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_support(self, indices=False):
        if indices:
            return np.where(self._mask)[0]
        return self._mask


def _build_preprocessing_step(step_name: str, params: dict) -> Any:
    """Build a single sklearn-compatible preprocessing transformer.

    Called by ``ModelTrainingExecutor._wrap_with_preprocessing`` to
    construct per-config pipeline steps.  Returns ``None`` if the step
    cannot be built (missing dependency, unknown name, etc.).
    """
    p = params or {}

    if step_name == "scaler":
        method = p.get("method", "standard")
        if method == "standard":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        elif method == "quantile":
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution="normal", random_state=42)
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()

    elif step_name == "feature_selection":
        method = p.get("method", "variance_threshold")
        if method == "variance_threshold":
            from sklearn.feature_selection import VarianceThreshold
            return VarianceThreshold(threshold=p.get("threshold", 0.01))
        elif method == "mutual_info":
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            return SelectKBest(mutual_info_classif, k=min(p.get("k", 20), 50))
        elif method == "boruta":
            try:
                from endgame.preprocessing.feature_selection import BorutaSelector
                return BorutaSelector(max_iter=30)
            except ImportError:
                return None
        elif method == "importance_mask":
            # Importance-based mask from iterative feature selection feedback
            mask = p.get("mask")
            if mask is not None:
                return _ImportanceMaskSelector(mask)

    elif step_name == "imputer":
        strategy = p.get("strategy", "median")
        if strategy == "knn":
            from sklearn.impute import KNNImputer
            return KNNImputer(n_neighbors=p.get("n_neighbors", 5))
        else:
            from sklearn.impute import SimpleImputer
            return SimpleImputer(strategy=strategy)

    elif step_name == "encoder":
        method = p.get("method", "ordinal")
        if method == "ordinal":
            from sklearn.preprocessing import OrdinalEncoder
            return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        elif method == "onehot":
            from sklearn.preprocessing import OneHotEncoder
            return OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
        elif method == "target":
            try:
                from endgame.preprocessing.encoding import SafeTargetEncoder
                return SafeTargetEncoder()
            except ImportError:
                from sklearn.preprocessing import OrdinalEncoder
                return OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    elif step_name == "dim_reduction":
        method = p.get("method", "pca")
        n_components = p.get("n_components", 0.95)
        if method == "pca":
            from sklearn.decomposition import PCA
            if isinstance(n_components, float) and 0 < n_components < 1:
                return PCA(n_components=n_components, random_state=42)
            # Integer n_components — wrap to avoid exceeding feature count
            from endgame.automl._safe_dim_reduce import SafePCA
            return SafePCA(n_components=int(n_components), random_state=42)
        elif method == "truncated_svd":
            from endgame.automl._safe_dim_reduce import SafeTruncatedSVD
            n_comp = int(n_components) if isinstance(n_components, (int, float)) and n_components >= 1 else 10
            return SafeTruncatedSVD(n_components=n_comp, random_state=42)

    return None


class EnsemblingExecutor(BaseStageExecutor):
    """Executes the ensembling stage.

    When ``method="auto"``, tries all available strategies (hill climbing,
    stacking, averaging) and picks the one with the best OOF score.
    """

    def __init__(self, method: str = "hill_climbing"):
        """Initialize ensembling executor.

        Parameters
        ----------
        method : str, default="hill_climbing"
            Ensemble method: "none", "hill_climbing", "stacking", "auto".
            "auto" tries all and picks the best.
        """
        self.method = method

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Build ensemble from trained models."""
        start_time = time.time()

        try:
            trained_models = context["trained_models"]
            oof_predictions = context["oof_predictions"]
            y = context.get("y_augmented", context.get("y_cleaned", context["y"]))
            task_type = context.get("task_type", "classification")

            if not trained_models:
                duration = time.time() - start_time
                return StageResult(
                    stage_name="ensembling",
                    success=False,
                    duration=duration,
                    output={"ensemble": None, "weights": {}},
                    error="No trained models available for ensembling",
                )

            if len(trained_models) < 2 or self.method == "none":
                results_list = context.get("results", [])
                score_map = {}
                if isinstance(results_list, list):
                    for r in results_list:
                        if hasattr(r, "config") and hasattr(r, "score") and r.success:
                            score_map[r.config.model_name] = r.score
                best_model_name = max(
                    trained_models.keys(),
                    key=lambda k: score_map.get(k, 0),
                )
                ensemble = trained_models.get(best_model_name)

                duration = time.time() - start_time
                return StageResult(
                    stage_name="ensembling",
                    success=True,
                    duration=duration,
                    output={"ensemble": ensemble, "weights": {best_model_name: 1.0}},
                    metadata={"method": "single_best"},
                )

            # Try multiple methods and pick the best
            if self.method == "auto":
                ensemble, weights, chosen_method = self._auto_ensemble(
                    trained_models, oof_predictions, y, task_type,
                )
            elif self.method == "hill_climbing":
                ensemble, weights = self._hill_climbing_ensemble(
                    trained_models, oof_predictions, y, task_type,
                )
                chosen_method = "hill_climbing"
            elif self.method == "stacking":
                ensemble, weights = self._stacking_ensemble(
                    trained_models, oof_predictions, y, task_type,
                )
                chosen_method = "stacking"
            else:
                ensemble, weights = self._average_ensemble(trained_models)
                chosen_method = "averaging"

            duration = time.time() - start_time

            return StageResult(
                stage_name="ensembling",
                success=True,
                duration=duration,
                output={"ensemble": ensemble, "weights": weights},
                metadata={"method": chosen_method, "n_models": len(weights)},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Ensembling failed: {e}")

            if "trained_models" in context and context["trained_models"]:
                best_model = list(context["trained_models"].values())[0]
            else:
                best_model = None

            return StageResult(
                stage_name="ensembling",
                success=False,
                duration=duration,
                output={"ensemble": best_model, "weights": {}},
                error=str(e),
            )

    def _auto_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float], str]:
        """Try all ensemble methods and return the best by OOF score."""
        from sklearn.metrics import r2_score, roc_auc_score

        is_clf = task_type in ("classification", "binary", "multiclass")

        def _safe_roc_auc(y_true, y_pred):
            try:
                return roc_auc_score(y_true, y_pred)
            except ValueError:
                from sklearn.metrics import accuracy_score
                return accuracy_score(y_true, y_pred)

        score_fn = _safe_roc_auc if is_clf else r2_score

        candidates: list[tuple[Any, dict[str, float], str, float]] = []

        valid_oof = {
            k: v for k, v in oof_predictions.items()
            if isinstance(v, np.ndarray) and len(v) == len(y)
        }

        # Hill climbing
        try:
            ens, wts = self._hill_climbing_ensemble(
                trained_models, valid_oof, y, task_type,
            )
            candidates.append((ens, wts, "hill_climbing", 0.0))
        except Exception as e:
            logger.debug(f"Hill climbing ensemble failed: {e}")

        # Stacking
        try:
            ens, wts = self._stacking_ensemble(
                trained_models, valid_oof, y, task_type,
            )
            candidates.append((ens, wts, "stacking", 0.0))
        except Exception as e:
            logger.debug(f"Stacking ensemble failed: {e}")

        # Optuna-optimized blending
        try:
            ens, wts = self._optimized_blend_ensemble(
                trained_models, valid_oof, y, task_type,
            )
            candidates.append((ens, wts, "optimized_blend", 0.0))
        except Exception as e:
            logger.debug(f"Optimized blend ensemble failed: {e}")

        # Power-weighted blending
        try:
            ens, wts = self._power_blend_ensemble(
                trained_models, valid_oof, y, task_type,
            )
            candidates.append((ens, wts, "power_blend", 0.0))
        except Exception as e:
            logger.debug(f"Power blend ensemble failed: {e}")

        # Rank averaging
        try:
            ens, wts = self._rank_average_ensemble(
                trained_models, valid_oof, y, task_type,
            )
            candidates.append((ens, wts, "rank_average", 0.0))
        except Exception as e:
            logger.debug(f"Rank average ensemble failed: {e}")

        # Uniform averaging (always available as a baseline)
        try:
            ens, wts = self._average_ensemble(trained_models)
            candidates.append((ens, wts, "averaging", 0.0))
        except Exception as e:
            logger.debug(f"Averaging ensemble failed: {e}")

        # Score each candidate using OOF predictions
        for i, (ens, wts, method, _) in enumerate(candidates):
            try:
                model_names = list(valid_oof.keys())
                oof_pred = None

                # Stacking: use the meta-estimator on the OOF stack
                if hasattr(ens, "meta_estimator") and hasattr(ens, "model_order"):
                    meta_X = np.column_stack(
                        [valid_oof[n] for n in ens.model_order if n in valid_oof]
                    )
                    oof_pred = ens.meta_estimator.predict(meta_X)

                # Rank averaging: use the blender's rank transform
                elif hasattr(ens, "blender") and hasattr(ens.blender, "blend"):
                    oof_preds_list = [
                        valid_oof[n] for n in ens.model_order
                        if n in valid_oof
                    ]
                    if oof_preds_list:
                        blended = ens.blender.blend(oof_preds_list)
                        if is_clf:
                            oof_pred = (blended > 0.5).astype(int)
                        else:
                            oof_pred = blended

                # Weighted ensembles (hill climbing, optimized blend, power blend, averaging)
                elif hasattr(ens, "weights"):
                    total_w = sum(wts.get(n, 0) for n in model_names)
                    if total_w > 0 and is_clf:
                        proba = None
                        for name in model_names:
                            w = wts.get(name, 0) / total_w
                            if w > 0 and name in valid_oof:
                                p = valid_oof[name]
                                if p.ndim == 1:
                                    p = np.column_stack([1 - p, p])
                                proba = p * w if proba is None else proba + p * w
                        oof_pred = (
                            np.argmax(proba, axis=1)
                            if proba is not None
                            else np.zeros(len(y))
                        )
                    elif total_w > 0:
                        oof_pred = sum(
                            wts.get(n, 0) / total_w * valid_oof[n]
                            for n in model_names
                            if n in valid_oof and wts.get(n, 0) > 0
                        )

                if oof_pred is not None:
                    score = score_fn(y, oof_pred)
                    candidates[i] = (ens, wts, method, score)
            except Exception as e:
                logger.debug(f"Ensemble scoring failed for {method}: {e}")

        if not candidates:
            return self._average_ensemble(trained_models) + ("averaging",)

        # Pick the best
        best = max(candidates, key=lambda x: x[3])
        logger.info(
            f"Auto-ensemble: tried {[c[2] for c in candidates]}, "
            f"scores={[f'{c[3]:.4f}' for c in candidates]}, "
            f"chose {best[2]}"
        )
        return best[0], best[1], best[2]

    def _hill_climbing_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float]]:
        """Build hill climbing ensemble.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        oof_predictions : dict
            OOF predictions for each model.
        y : array-like
            Target vector.
        task_type : str
            Task type.

        Returns
        -------
        tuple
            (ensemble, weights)
        """
        try:
            from endgame.ensemble.hill_climbing import HillClimbingEnsemble

            preds_list = list(oof_predictions.values())
            model_names = list(oof_predictions.keys())

            hc = HillClimbingEnsemble(
                metric="roc_auc" if task_type == "classification" else "r2",
                n_iterations=100,
            )
            hc.fit(preds_list, y)

            # hc.weights_ is {int_index: float_weight} — map to model names
            weights = {
                model_names[idx]: w
                for idx, w in hc.weights_.items()
                if idx < len(model_names)
            }

            ensemble = _WeightedEnsemble(
                models=trained_models,
                weights=weights,
                task_type=task_type,
            )

            return ensemble, weights

        except ImportError:
            return self._average_ensemble(trained_models)

    def _stacking_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float]]:
        """Build stacking ensemble using OOF predictions as meta-features.

        Parameters
        ----------
        trained_models : dict
            Trained models.
        oof_predictions : dict
            OOF predictions for each model.
        y : array-like
            Target vector.
        task_type : str
            Task type.

        Returns
        -------
        tuple
            (ensemble, weights)
        """
        try:
            from sklearn.linear_model import LogisticRegression, Ridge

            model_names = list(oof_predictions.keys())
            meta_X = np.column_stack(list(oof_predictions.values()))

            is_clf = task_type in ("classification", "binary", "multiclass")
            if is_clf:
                meta_est = LogisticRegression(max_iter=1000, solver="lbfgs")
            else:
                meta_est = Ridge(alpha=1.0)

            meta_est.fit(meta_X, y)

            weights = {name: 1.0 / len(model_names) for name in model_names}

            ensemble = _StackingEnsembleWrapper(
                models=trained_models,
                model_order=model_names,
                meta_estimator=meta_est,
                task_type=task_type,
            )

            return ensemble, weights

        except Exception:
            return self._average_ensemble(trained_models)

    def _average_ensemble(
        self,
        trained_models: dict[str, Any],
    ) -> tuple[Any, dict[str, float]]:
        """Build simple averaging ensemble.

        Parameters
        ----------
        trained_models : dict
            Trained models.

        Returns
        -------
        tuple
            (ensemble, weights)
        """
        weights = {name: 1.0 / len(trained_models) for name in trained_models}

        ensemble = _WeightedEnsemble(
            models=trained_models,
            weights=weights,
            task_type="classification",
        )

        return ensemble, weights

    def _optimized_blend_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float]]:
        """Optuna-optimized blend weights."""
        from endgame.ensemble.blending import OptimizedBlender

        preds_list = list(oof_predictions.values())
        model_names = list(oof_predictions.keys())

        is_clf = task_type in ("classification", "binary", "multiclass")
        blender = OptimizedBlender(
            metric="roc_auc" if is_clf else "rmse",
            n_trials=30,
            maximize=is_clf,
            verbose=False,
        )
        blender.fit(preds_list, y)

        weights = {
            model_names[idx]: w
            for idx, w in blender.weights_.items()
            if idx < len(model_names)
        }

        ensemble = _WeightedEnsemble(
            models=trained_models,
            weights=weights,
            task_type=task_type,
        )
        return ensemble, weights

    def _power_blend_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float]]:
        """Power-weighted blending based on individual OOF scores."""
        from sklearn.metrics import r2_score, roc_auc_score

        from endgame.ensemble.blending import PowerBlender

        preds_list = list(oof_predictions.values())
        model_names = list(oof_predictions.keys())

        is_clf = task_type in ("classification", "binary", "multiclass")
        score_fn = roc_auc_score if is_clf else r2_score

        scores = []
        for pred in preds_list:
            try:
                scores.append(score_fn(y, pred))
            except Exception:
                scores.append(0.5 if is_clf else 0.0)

        blender = PowerBlender(scores=scores, power=3.0, higher_is_better=True)
        blender.fit()

        weights = {
            model_names[idx]: w
            for idx, w in blender.weights_.items()
            if idx < len(model_names)
        }

        ensemble = _WeightedEnsemble(
            models=trained_models,
            weights=weights,
            task_type=task_type,
        )
        return ensemble, weights

    def _rank_average_ensemble(
        self,
        trained_models: dict[str, Any],
        oof_predictions: dict[str, np.ndarray],
        y: np.ndarray,
        task_type: str,
    ) -> tuple[Any, dict[str, float]]:
        """Rank-based averaging — robust to different prediction scales."""
        from endgame.ensemble.blending import RankAverageBlender

        preds_list = list(oof_predictions.values())
        model_names = list(oof_predictions.keys())

        blender = RankAverageBlender()
        blender.fit()

        weights = {name: 1.0 / len(model_names) for name in model_names}

        ensemble = _RankAverageEnsembleWrapper(
            models=trained_models,
            model_order=model_names,
            blender=blender,
            task_type=task_type,
        )
        return ensemble, weights


class _RankAverageEnsembleWrapper:
    """Wrapper for rank-average ensemble that applies rank transform at predict time."""

    def __init__(
        self,
        models: dict[str, Any],
        model_order: list[str],
        blender: Any,
        task_type: str = "classification",
    ):
        self.models = models
        self.model_order = model_order
        self.blender = blender
        self.task_type = task_type
        self.weights = {name: 1.0 / len(model_order) for name in model_order}

    def predict(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for name in self.model_order:
            if name in self.models:
                try:
                    if self.task_type == "classification" and hasattr(self.models[name], "predict_proba"):
                        p = self.models[name].predict_proba(X)[:, 1]
                    else:
                        p = self.models[name].predict(X)
                    preds.append(p)
                except Exception:
                    continue

        if not preds:
            return np.zeros(X.shape[0])

        blended = self.blender.blend(preds)
        if self.task_type == "classification":
            return (blended > 0.5).astype(int)
        return blended

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        preds = []
        for name in self.model_order:
            if name in self.models:
                try:
                    if hasattr(self.models[name], "predict_proba"):
                        p = self.models[name].predict_proba(X)[:, 1]
                    else:
                        p = self.models[name].predict(X)
                    preds.append(p)
                except Exception:
                    continue

        if not preds:
            p1 = np.full(X.shape[0], 0.5)
            return np.column_stack([1 - p1, p1])

        blended = self.blender.blend(preds)
        return np.column_stack([1 - blended, blended])


class _WeightedEnsemble:
    """Simple weighted ensemble wrapper."""

    def __init__(
        self,
        models: dict[str, Any],
        weights: dict[str, float],
        task_type: str = "classification",
    ):
        self.models = models
        self.weights = weights
        self.task_type = task_type

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        array-like
            Predictions.
        """
        if self.task_type == "classification":
            proba = self.predict_proba(X)
            return np.argmax(proba, axis=1)
        else:
            preds = []
            total_weight = sum(self.weights.values())
            if total_weight == 0:
                total_weight = len(self.models)
                self.weights = {n: 1.0 for n in self.models}

            for name, model in self.models.items():
                weight = self.weights.get(name, 0) / total_weight
                if weight > 0:
                    preds.append(weight * model.predict(X))

            return sum(preds) if preds else list(self.models.values())[0].predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : array-like
            Feature matrix.

        Returns
        -------
        array-like
            Class probabilities.
        """
        proba = None
        total_weight = sum(self.weights.values())

        # Fall back to equal weights if all weights are zero
        if total_weight == 0:
            total_weight = len(self.models)
            self.weights = {n: 1.0 for n in self.models}

        for name, model in self.models.items():
            weight = self.weights.get(name, 0) / total_weight

            if weight > 0 and hasattr(model, "predict_proba"):
                model_proba = model.predict_proba(X)

                if proba is None:
                    proba = weight * model_proba
                else:
                    proba += weight * model_proba

        return proba if proba is not None else np.zeros((X.shape[0], 2))


class _StackingEnsembleWrapper:
    """Stacking ensemble: base model predictions -> meta-estimator."""

    def __init__(
        self,
        models: dict[str, Any],
        model_order: list[str],
        meta_estimator: Any,
        task_type: str = "classification",
    ):
        self.models = models
        self.model_order = model_order
        self.meta_estimator = meta_estimator
        self.task_type = task_type

    def _meta_features(self, X: np.ndarray) -> np.ndarray:
        cols = []
        for name in self.model_order:
            model = self.models[name]
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)
                cols.append(p[:, 1] if p.ndim == 2 and p.shape[1] == 2 else p)
            else:
                cols.append(model.predict(X))
        return np.column_stack(cols)

    def predict(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._meta_features(X)
        return self.meta_estimator.predict(meta_X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        meta_X = self._meta_features(X)
        if hasattr(self.meta_estimator, "predict_proba"):
            return self.meta_estimator.predict_proba(meta_X)
        decision = self.meta_estimator.predict(meta_X)
        proba = np.column_stack([1 - decision, decision])
        return proba


class CalibrationExecutor(BaseStageExecutor):
    """Executes the calibration stage."""

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Calibrate model probabilities.

        Parameters
        ----------
        context : dict
            Must contain 'ensemble', 'X_val', 'y_val'.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains calibrator in output.
        """
        start_time = time.time()

        try:
            ensemble = context.get("ensemble")
            X_val = context.get("X_val")
            y_val = context.get("y_val")
            task_type = context.get("task_type", "classification")

            if ensemble is None or task_type != "classification" or X_val is None:
                duration = time.time() - start_time
                return StageResult(
                    stage_name="calibration",
                    success=True,
                    duration=duration,
                    output={"calibrator": None},
                    metadata={"skipped": True},
                )

            # Get uncalibrated probabilities
            if hasattr(ensemble, "predict_proba"):
                uncalibrated_proba = ensemble.predict_proba(X_val)
            else:
                duration = time.time() - start_time
                return StageResult(
                    stage_name="calibration",
                    success=True,
                    duration=duration,
                    output={"calibrator": None},
                    metadata={"skipped": True, "reason": "no predict_proba"},
                )

            # Try different calibration methods
            calibrator = self._select_best_calibrator(uncalibrated_proba, y_val)

            duration = time.time() - start_time

            return StageResult(
                stage_name="calibration",
                success=True,
                duration=duration,
                output={"calibrator": calibrator},
                metadata={"method": type(calibrator).__name__ if calibrator else None},
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Calibration failed: {e}")

            return StageResult(
                stage_name="calibration",
                success=False,
                duration=duration,
                output={"calibrator": None},
                error=str(e),
            )

    def _select_best_calibrator(
        self,
        proba: np.ndarray,
        y: np.ndarray,
    ):
        """Select best calibration method using ECE-based 3-fold CV.

        Tries PlattScaling, IsotonicCalibration, and BetaCalibration
        from endgame.calibration, selecting the one with lowest ECE.
        Falls back to sklearn LogisticRegression if endgame calibration
        is unavailable.

        Parameters
        ----------
        proba : array-like
            Uncalibrated probabilities.
        y : array-like
            True labels.

        Returns
        -------
        calibrator
            Best calibrator or None.
        """
        # Extract 1D probabilities for binary case
        if proba.ndim > 1:
            proba_1d = proba[:, 1] if proba.shape[1] == 2 else proba
        else:
            proba_1d = proba

        try:
            from sklearn.model_selection import KFold

            from endgame.calibration.analysis import expected_calibration_error
            from endgame.calibration.scaling import (
                BetaCalibration,
                IsotonicCalibration,
                PlattScaling,
            )

            candidates = [
                ("PlattScaling", PlattScaling()),
                ("IsotonicCalibration", IsotonicCalibration()),
                ("BetaCalibration", BetaCalibration()),
            ]

            # Add additional calibrators if available
            try:
                from endgame.calibration.venn_abers import VennABERS
                candidates.append(("VennABERS", VennABERS()))
            except ImportError:
                pass

            try:
                from endgame.calibration.scaling import TemperatureScaling
                candidates.append(("TemperatureScaling", TemperatureScaling()))
            except ImportError:
                pass

            try:
                from endgame.calibration.scaling import HistogramBinning
                candidates.append(("HistogramBinning", HistogramBinning()))
            except ImportError:
                pass

            best_calibrator = None
            best_ece = float("inf")
            best_name = None

            for name, calibrator_template in candidates:
                try:
                    # 3-fold CV to estimate ECE
                    ece_scores = []
                    kf = KFold(n_splits=3, shuffle=True, random_state=42)

                    for train_idx, val_idx in kf.split(proba_1d):
                        p_train, p_val = proba_1d[train_idx], proba_1d[val_idx]
                        y_train, y_val = y[train_idx], y[val_idx]

                        # Clone calibrator for each fold
                        from sklearn.base import clone
                        cal = clone(calibrator_template)
                        cal.fit(p_train, y_train)
                        p_cal = cal.transform(p_val)

                        # Ensure 1D for ECE computation
                        if p_cal.ndim > 1:
                            p_cal = p_cal[:, 1] if p_cal.shape[1] == 2 else p_cal.ravel()

                        ece = expected_calibration_error(y_val, p_cal)
                        ece_scores.append(ece)

                    mean_ece = np.mean(ece_scores)
                    if mean_ece < best_ece:
                        best_ece = mean_ece
                        best_name = name
                        best_calibrator = clone(calibrator_template)

                except Exception as e:
                    logger.debug(f"Calibrator {name} failed during CV: {e}")
                    continue

            if best_calibrator is not None:
                # Fit on full data
                best_calibrator.fit(proba_1d, y)
                logger.info(f"Selected calibrator: {best_name} (ECE={best_ece:.4f})")
                return best_calibrator

        except ImportError:
            logger.debug("endgame.calibration not available, using sklearn fallback")

        # Fallback to sklearn LogisticRegression
        try:
            from sklearn.linear_model import LogisticRegression

            calibrator = LogisticRegression()
            calibrator.fit(proba_1d.reshape(-1, 1), y)
            return calibrator

        except Exception:
            return None


class PostTrainingExecutor(BaseStageExecutor):
    """Executes the post-training stage.

    Handles knowledge distillation and conformal prediction.
    """

    def __init__(self, feature_engineering: str = "none"):
        self.feature_engineering = feature_engineering

    def execute(
        self,
        context: dict[str, Any],
        time_budget: float,
    ) -> StageResult:
        """Apply post-training optimizations.

        Parameters
        ----------
        context : dict
            Must contain 'ensemble' and optionally validation data.
        time_budget : float
            Time budget in seconds.

        Returns
        -------
        StageResult
            Contains distilled_model and conformal_predictor in output.
        """
        start_time = time.time()
        level = self.feature_engineering
        task_type = context.get("task_type", "classification")

        if level in ("none", "light"):
            duration = time.time() - start_time
            return StageResult(
                stage_name="post_training",
                success=True,
                duration=duration,
                output={"distilled_model": None, "conformal_predictor": None},
                metadata={"skipped": True, "level": level},
            )

        try:
            ensemble = context.get("ensemble")
            X_val = context.get("X_val")
            y_val = context.get("y_val")
            distilled_model = None
            conformal_predictor = None

            # Aggressive: Knowledge distillation
            if level == "aggressive" and ensemble is not None:
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.5:
                    try:
                        from endgame.ensemble.distillation import KnowledgeDistiller
                        distiller = KnowledgeDistiller(
                            teacher=ensemble,
                            student=None,  # Use default (LGBMClassifier)
                            temperature=3.0,
                            alpha=0.7,
                        )
                        # Use training data for distillation
                        X_train = context.get("X_augmented",
                            context.get("X_engineered",
                                context.get("X_processed",
                                    context.get("X_cleaned", context.get("X")))))
                        y_train = context.get("y_augmented",
                            context.get("y_cleaned", context.get("y")))
                        distilled_model = distiller.fit(X_train, y_train)
                        logger.info("PostTraining: knowledge distillation complete")
                    except ImportError:
                        logger.debug("KnowledgeDistiller not available")
                    except Exception as e:
                        logger.debug(f"Knowledge distillation failed: {e}")

            # Moderate+Aggressive: Conformal prediction (if validation data available)
            if level in ("moderate", "aggressive") and X_val is not None and y_val is not None:
                elapsed = time.time() - start_time
                if elapsed < time_budget * 0.9:
                    try:
                        if task_type == "classification":
                            from endgame.calibration.conformal import ConformalClassifier
                            conformal_predictor = ConformalClassifier(
                                method="lac", alpha=0.1,
                            )
                        else:
                            from endgame.calibration.conformal import ConformalRegressor
                            conformal_predictor = ConformalRegressor(
                                method="adaptive", alpha=0.1,
                            )
                        if ensemble is not None:
                            conformal_predictor.fit(ensemble, X_val, y_val)
                            logger.info("PostTraining: conformal prediction calibrated")
                    except ImportError:
                        logger.debug("Conformal prediction not available")
                        conformal_predictor = None
                    except Exception as e:
                        logger.debug(f"Conformal prediction failed: {e}")
                        conformal_predictor = None

            duration = time.time() - start_time
            return StageResult(
                stage_name="post_training",
                success=True,
                duration=duration,
                output={
                    "distilled_model": distilled_model,
                    "conformal_predictor": conformal_predictor,
                },
                metadata={
                    "level": level,
                    "has_distilled": distilled_model is not None,
                    "has_conformal": conformal_predictor is not None,
                },
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Post-training failed: {e}")
            return StageResult(
                stage_name="post_training",
                success=False,
                duration=duration,
                output={"distilled_model": None, "conformal_predictor": None},
                error=str(e),
            )


class PipelineOrchestrator:
    """Coordinates AutoML pipeline stages with time budget management.

    The orchestrator manages the execution of all pipeline stages,
    handles time allocation, and provides graceful degradation when
    stages fail or time runs out.

    Parameters
    ----------
    preset : str or PresetConfig, default="medium_quality"
        Preset configuration to use.
    time_limit : int, optional
        Total time budget in seconds. Overrides preset default.
    search_strategy : BaseSearchStrategy, optional
        Search strategy for model selection.
    verbose : int, default=1
        Verbosity level.

    Attributes
    ----------
    stage_results_ : dict
        Results from each executed stage.
    time_manager_ : TimeBudgetManager
        Time budget manager.

    Examples
    --------
    >>> orchestrator = PipelineOrchestrator(preset="medium_quality")
    >>> result = orchestrator.run(X_train, y_train, X_val, y_val)
    >>> print(result.score)
    """

    # Default stage order and time allocations
    DEFAULT_STAGES = [
        ("profiling", 0.01),
        ("quality_guardrails", 0.02),
        ("data_cleaning", 0.02),
        ("preprocessing", 0.05),
        ("feature_engineering", 0.03),
        ("data_augmentation", 0.02),
        ("model_selection", 0.04),
        ("model_training", 0.40),
        ("constraint_check", 0.01),
        ("hyperparameter_tuning", 0.20),
        ("ensembling", 0.06),
        ("threshold_opt", 0.02),
        ("calibration", 0.03),
        ("post_training", 0.02),
        ("explainability", 0.02),
        ("persistence", 0.01),
    ]

    def __init__(
        self,
        preset: str | PresetConfig = "medium_quality",
        time_limit: int | None = None,
        search_strategy=None,
        verbose: int = 1,
        checkpoint_callback=None,
        keep_training: bool = False,
        patience: int = 5,
        min_improvement: float = 1e-4,
        min_model_time: float = 300.0,
        max_model_time: float = 600.0,
        eval_metric: str = "auto",
        excluded_models: list[str] | None = None,
        early_stopping_rounds: int = 50,
        use_gpu: bool = False,
    ):
        if isinstance(preset, str):
            self.preset = PRESETS.get(preset, PRESETS["medium_quality"])
        else:
            self.preset = preset
        self.eval_metric = eval_metric
        self.excluded_models = set(excluded_models or [])

        # time_limit semantics:
        #   positive int  → hard budget in seconds
        #   0             → unlimited (sentinel large value)
        #   None          → fall back to preset default, then 900s
        _UNLIMITED = 10 ** 9  # ~31 years
        if time_limit is not None and time_limit > 0:
            self.time_limit = time_limit
        elif time_limit == 0:
            self.time_limit = _UNLIMITED
        else:
            self.time_limit = self.preset.default_time_limit or 900
        self.search_strategy = search_strategy
        self.verbose = verbose
        self.checkpoint_callback = checkpoint_callback
        self.keep_training = keep_training
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_model_time = min_model_time
        self.max_model_time = max_model_time
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu

        # Configure logging based on verbosity
        if verbose >= 3:
            logger.setLevel(logging.DEBUG)
        elif verbose >= 2:
            logger.setLevel(logging.INFO)
        elif verbose >= 1:
            logger.setLevel(logging.WARNING)
        else:
            logger.setLevel(logging.ERROR)

        # Ensure at least one handler exists so messages are visible
        if not logger.handlers and not logging.getLogger().handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)

        # Initialize executors
        feature_eng = getattr(self.preset, "feature_engineering", "none")
        self._executors = {
            "profiling": ProfilingExecutor(),
            "data_cleaning": DataCleaningExecutor(feature_engineering=feature_eng),
            "preprocessing": PreprocessingExecutor(feature_engineering=feature_eng),
            "feature_engineering": AdvancedFeatureEngineeringExecutor(feature_engineering=feature_eng),
            "data_augmentation": DataAugmentationExecutor(feature_engineering=feature_eng),
            "model_selection": ModelSelectionExecutor(search_strategy),
            "model_training": ModelTrainingExecutor(
                cv_folds=self.preset.cv_folds,
                feature_engineering=feature_eng,
                verbose=self.verbose,
                min_model_time=self.min_model_time,
                max_model_time=self.max_model_time,
                eval_metric=self.eval_metric,
                early_stopping_rounds=self.early_stopping_rounds,
                use_gpu=self.use_gpu,
            ),
            "ensembling": EnsemblingExecutor(method=self.preset.ensemble_method),
            "calibration": CalibrationExecutor(),
            "post_training": PostTrainingExecutor(feature_engineering=feature_eng),
        }

        # Register new stage executors
        from endgame.automl.executors import (
            ConstraintCheckExecutor,
            ExplainabilityExecutor,
            HyperparameterTuningExecutor,
            PersistenceExecutor,
            ThresholdOptimizationExecutor,
        )
        from endgame.automl.guardrails import QualityGuardrailsExecutor

        guardrails_strict = getattr(self.preset, "guardrails_strict", False)
        self._executors["quality_guardrails"] = QualityGuardrailsExecutor(
            strict=guardrails_strict,
        )
        self._executors["hyperparameter_tuning"] = HyperparameterTuningExecutor(
            top_n=3,
            cv_folds=self.preset.cv_folds,
        )
        self._executors["threshold_opt"] = ThresholdOptimizationExecutor()
        self._executors["explainability"] = ExplainabilityExecutor()

        # Persistence: auto-save when output_path is configured
        # The output_path is passed via preset or constructor kwargs.
        output_path = getattr(self.preset, "output_path", None)
        self._executors["persistence"] = PersistenceExecutor(
            output_dir=output_path,
        )

        constraints = getattr(self.preset, "constraints", None)
        self._executors["constraint_check"] = ConstraintCheckExecutor(
            constraints=constraints,
        )

        # State
        self.stage_results_: dict[str, StageResult] = {}
        self.time_manager_: TimeBudgetManager | None = None

    def run(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        task_type: str = "classification",
    ) -> PipelineResult:
        """Execute the full AutoML pipeline.

        Parameters
        ----------
        X : array-like
            Training feature matrix.
        y : array-like
            Training target vector.
        X_val : array-like, optional
            Validation feature matrix.
        y_val : array-like, optional
            Validation target vector.
        task_type : str, default="classification"
            Task type.

        Returns
        -------
        PipelineResult
            Complete pipeline results.
        """
        start_time = time.time()

        # Initialize time manager
        allocations = dict(self.preset.time_allocations or self.DEFAULT_STAGES)
        self.time_manager_ = TimeBudgetManager(
            total_budget=self.time_limit,
            allocations=allocations,
        )
        self.time_manager_.start()

        # Initialize context
        context = {
            "X": X,
            "y": y,
            "X_val": X_val,
            "y_val": y_val,
            "task_type": task_type,
            "preset": self.preset.name,
            "max_models": len(self.preset.model_pool),
        }

        # Execute stages
        fail_fast = False
        for stage_name, _ in self.DEFAULT_STAGES:
            if stage_name not in self._executors:
                continue

            # Abort early on critical guardrail failures
            if fail_fast:
                logger.warning(
                    f"Skipping {stage_name}: pipeline aborted due to critical "
                    "quality guardrail failure (guardrails_strict=True)"
                )
                break

            # Skip calibration if not enabled
            if stage_name == "calibration" and not self.preset.calibrate:
                continue

            # Skip HPO if not enabled in preset
            if stage_name == "hyperparameter_tuning" and not self.preset.hyperparameter_tune:
                continue

            # Skip stages with 0.0 time allocation
            stage_alloc = allocations.get(stage_name, 0.0)
            if stage_alloc <= 0.0 and stage_name not in ("profiling",):
                continue

            # Begin stage
            stage_budget = self.time_manager_.begin_stage(stage_name)

            budget_str = "unlimited" if stage_budget >= 10**7 else f"{stage_budget:.1f}s"
            if self.verbose > 0:
                print(f"  [AutoML] Starting stage: {stage_name} (budget: {budget_str})")
            logger.debug(f"Starting stage: {stage_name} (budget: {budget_str})")

            try:
                # Execute stage
                result = self._executors[stage_name].execute(context, stage_budget)
                self.stage_results_[stage_name] = result

                # Update context with stage output
                if result.output:
                    context.update(result.output)

                # Propagate preprocessing / feature-engineering to X_val
                # so that downstream stages (calibration, etc.) get
                # correctly transformed validation data.
                if stage_name == "preprocessing" and result.success:
                    preprocessor = result.output.get("preprocessor") if result.output else None
                    if preprocessor is not None and context.get("X_val") is not None:
                        try:
                            context["X_val"] = preprocessor.transform(context["X_val"])
                        except Exception as e:
                            logger.debug(f"Could not preprocess X_val: {e}")

                if stage_name == "feature_engineering" and result.success:
                    fe_transforms = result.output.get("feature_transformers", []) if result.output else []
                    if fe_transforms and context.get("X_val") is not None:
                        try:
                            X_val_arr = (
                                context["X_val"].values
                                if isinstance(context["X_val"], pd.DataFrame)
                                else np.asarray(context["X_val"])
                            )
                            for _name, transformer in fe_transforms:
                                X_val_arr = transformer.transform(X_val_arr)
                            context["X_val"] = X_val_arr
                        except Exception as e:
                            logger.debug(f"Could not feature-engineer X_val: {e}")

                # Check for fail_fast signal from guardrails
                if result.output and result.output.get("fail_fast"):
                    fail_fast = True

                status = "OK" if result.success else "FAILED"
                if self.verbose > 0:
                    print(f"  [AutoML] Completed stage: {stage_name} [{status}] ({result.duration:.1f}s)")
                logger.debug(f"Completed stage: {stage_name} [{status}] ({result.duration:.1f}s)")

                # Checkpoint after heavyweight stages
                if stage_name in (
                    "model_training", "hyperparameter_tuning",
                    "ensembling", "calibration",
                ):
                    self._checkpoint(context, f"post_{stage_name}")

            except Exception as e:
                if self.verbose > 0:
                    print(f"  [AutoML] Stage {stage_name} FAILED: {e}")
                logger.error(f"Stage {stage_name} failed with exception: {e}")
                self.stage_results_[stage_name] = StageResult(
                    stage_name=stage_name,
                    success=False,
                    duration=0,
                    error=str(e),
                )

            finally:
                # End stage and redistribute unused time
                self.time_manager_.end_stage()

        # Checkpoint after initial pipeline
        self._checkpoint(context, "post_pipeline")

        # --- Feedback loop: iterative search if time permits ---
        if (
            not fail_fast
            and self.search_strategy is not None
            and self.time_manager_ is not None
            and self.time_manager_.remaining_budget() > 60
        ):
            self._run_feedback_loop(context, allocations)
            self._checkpoint(context, "post_feedback")

        # --- HPO stage on top models (if enabled and time permits) ---
        if (
            not fail_fast
            and self.preset.hyperparameter_tune
            and "hyperparameter_tuning" in self._executors
            and self.time_manager_ is not None
            and self.time_manager_.remaining_budget() > 30
        ):
            stage_budget = self.time_manager_.begin_stage("hyperparameter_tuning")
            if self.verbose > 0:
                print(f"  [AutoML] Starting stage: hyperparameter_tuning (budget: {stage_budget:.1f}s)")
            try:
                result = self._executors["hyperparameter_tuning"].execute(context, stage_budget)
                self.stage_results_["hyperparameter_tuning"] = result
                if result.output:
                    context.update(result.output)
                if self.verbose > 0:
                    status = "OK" if result.success else "FAILED"
                    print(f"  [AutoML] Completed stage: hyperparameter_tuning [{status}] ({result.duration:.1f}s)")
            except Exception as e:
                logger.error(f"HPO stage failed: {e}")
            finally:
                self.time_manager_.end_stage()
            self._checkpoint(context, "post_hpo")

        # --- Continuous optimization loop (keep_training mode) ---
        if self.keep_training and not fail_fast:
            self._run_continuous_loop(context)

        # Store context so the result builder can access continuous-loop
        # models/results that were not in the initial training stage.
        self._final_context = context

        # Build final result
        total_time = time.time() - start_time

        # Get best score
        training_result = self.stage_results_.get("model_training")
        # Collect ALL results including from the continuous loop
        all_results: list = []
        if training_result and training_result.output:
            all_results = training_result.output.get("results", [])

        # Also include results stored in context by the continuous loop
        ctx = getattr(self, "_final_context", {})
        ctx_results = ctx.get("results", [])
        seen_ids = {id(r) for r in all_results}
        for r in ctx_results:
            if id(r) not in seen_ids:
                all_results.append(r)

        successful_results = [r for r in all_results if r.success]
        best_score = max((r.score for r in successful_results), default=0.0)

        # Get ensemble
        ensemble_result = self.stage_results_.get("ensembling")
        ensemble = ensemble_result.output.get("ensemble") if ensemble_result and ensemble_result.output else None

        # Get best model from ALL trained models (initial + continuous)
        trained_models: dict = {}
        if training_result and training_result.output:
            trained_models.update(training_result.output.get("trained_models", {}))
        ctx_models = ctx.get("trained_models", {})
        trained_models.update(ctx_models)

        best_model = None
        if trained_models and successful_results:
            score_map: dict[str, float] = {}
            for r in successful_results:
                name = r.config.model_name
                if name in trained_models:
                    score_map[name] = max(score_map.get(name, -float("inf")), r.score)
            if score_map:
                best_name = max(score_map, key=score_map.get)
                best_model = trained_models[best_name]
        elif trained_models:
            best_model = next(iter(trained_models.values()))

        return PipelineResult(
            best_model=best_model,
            ensemble=ensemble,
            score=best_score,
            scores={"primary": best_score},
            stage_results=self.stage_results_,
            total_time=total_time,
            metadata={
                "preset": self.preset.name,
                "time_limit": self.time_limit,
                "task_type": task_type,
            },
        )

    def _run_feedback_loop(
        self,
        context: dict[str, Any],
        allocations: dict[str, float],
    ) -> None:
        """Run iterative feedback loop with remaining time budget.

        Suggests and trains new model configurations (including HPO
        variants once the initial sweep is complete).  Re-runs ensembling
        if new models are added.  This loop is no longer gated on the
        HPO preset flag — it runs whenever time permits.
        """
        max_iterations = 5
        strategy = self.search_strategy
        results = context.get("results", [])
        trained_models = context.get("trained_models", {})
        meta_features = context.get("meta_features", {})

        if self.verbose > 0:
            remaining = self.time_manager_.remaining_budget()
            print(f"  [AutoML] Starting feedback loop ({remaining:.0f}s remaining)")

        # Sync strategy with all results so far
        already_synced = {r.config.config_id for r in strategy.results_} if strategy.results_ else set()
        for r in results:
            if r.config.config_id not in already_synced:
                try:
                    strategy.update(r)
                except Exception:
                    pass

        new_models_added = False
        for iteration in range(max_iterations):
            remaining = self.time_manager_.remaining_budget()
            if remaining < 30:
                break

            try:
                new_configs = strategy.suggest(meta_features, n_suggestions=2)
            except Exception as e:
                logger.debug(f"Feedback loop suggest failed: {e}")
                break

            if not new_configs:
                break

            # For non-variant configs, skip models already trained.
            # For HPO variants, always allow (they have unique config_ids).
            new_configs = [
                c for c in new_configs
                if c.metadata.get("source") == "portfolio_hpo_variant"
                or c.model_name not in trained_models
            ]
            if not new_configs:
                break

            train_budget = min(
                remaining * 0.5,
                self.min_model_time * max(len(new_configs), 1),
            )
            if self.verbose > 0:
                names = [c.model_name for c in new_configs]
                print(
                    f"  [AutoML] Feedback iteration {iteration + 1}: "
                    f"training {names} ({train_budget:.0f}s)"
                )

            trainer = self._executors.get("model_training")
            if trainer is None:
                break

            context["model_configs"] = new_configs
            try:
                train_result = trainer.execute(context, train_budget)
                if train_result.success and train_result.output:
                    new_trained = train_result.output.get("trained_models", {})
                    new_results = train_result.output.get("results", [])

                    for name, model in new_trained.items():
                        trained_models[name] = model
                        new_models_added = True

                    results.extend(new_results)

                    for r in new_results:
                        try:
                            strategy.update(r)
                        except Exception:
                            pass
            except Exception as e:
                logger.warning(f"Feedback loop training failed: {e}")
                break

        if new_models_added:
            context["trained_models"] = trained_models
            context["results"] = results
            ensembler = self._executors.get("ensembling")
            if ensembler is not None:
                remaining = self.time_manager_.remaining_budget()
                if remaining > 10:
                    try:
                        ensemble_result = ensembler.execute(context, remaining * 0.5)
                        if ensemble_result.success:
                            self.stage_results_["ensembling"] = ensemble_result
                            if ensemble_result.output:
                                context.update(ensemble_result.output)
                    except Exception as e:
                        logger.warning(f"Feedback loop ensembling failed: {e}")

    def _update_feature_selection_feedback(
        self,
        context: dict[str, Any],
        trained_models: dict[str, Any],
        results: list,
    ) -> None:
        """Compute aggregate feature importances from trained models and
        store an informed feature mask in context.

        This enables iterative feature selection: future pipeline configs
        can reference ``context["important_feature_mask"]`` to focus on
        features that matter, dropping noise columns.
        """
        try:
            importances: list[np.ndarray] = []
            for r in results:
                if not r.success:
                    continue
                model = trained_models.get(r.config.model_name)
                if model is None:
                    continue
                # Try to get feature importances from model
                fi = getattr(model, "feature_importances_", None)
                if fi is None and hasattr(model, "named_steps"):
                    # Pipeline wrapper — get from final estimator
                    final = model.named_steps.get("model", model)
                    fi = getattr(final, "feature_importances_", None)
                if fi is not None and len(fi) > 0:
                    # Normalise to sum=1 so different models are comparable
                    total = np.sum(np.abs(fi))
                    if total > 0:
                        importances.append(np.abs(fi) / total)

            if len(importances) < 2:
                return

            # Stack and average across models
            min_len = min(len(fi) for fi in importances)
            stacked = np.stack([fi[:min_len] for fi in importances])
            avg_importance = np.mean(stacked, axis=0)

            # Mark features with >1% average importance as "important"
            threshold = 0.01 / min_len if min_len > 0 else 0.01
            mask = avg_importance > threshold
            n_kept = int(np.sum(mask))

            if 0 < n_kept < min_len:
                context["important_feature_mask"] = mask
                context["feature_importances_aggregated"] = avg_importance
                if self.verbose > 1:
                    print(
                        f"  [AutoML] Feature selection feedback: "
                        f"keeping {n_kept}/{min_len} features "
                        f"(threshold={threshold:.6f})"
                    )
        except Exception as e:
            logger.debug(f"Feature selection feedback failed: {e}")

    def _checkpoint(self, context: dict[str, Any], label: str = "") -> None:
        """Invoke the checkpoint callback if one is registered."""
        if self.checkpoint_callback is not None:
            try:
                self.checkpoint_callback(
                    stage_results=self.stage_results_,
                    context=context,
                    label=label,
                )
            except Exception as e:
                logger.warning(f"Checkpoint callback failed ({label}): {e}")

    def _run_continuous_loop(self, context: dict[str, Any]) -> None:
        """Run continuous optimization until convergence or interruption.

        This is the core AutoML loop.  It alternates between:

        1. **Model search** — ask the search strategy for new configs
           (new model types during the initial sweep, then HPO variants
           of the top performers).
        2. **Training** — fit the suggested configurations.
        3. **Optional HPO** — run Optuna on the best models if the HPO
           executor is available and time permits.
        4. **Re-ensembling** — rebuild the ensemble with the expanded
           model pool.

        Stopping criteria:
        - ``patience`` consecutive rounds without improvement exceeding
          ``min_improvement``.
        - ``time_limit`` reached (if set).
        - ``KeyboardInterrupt`` (saves checkpoint and exits).
        """
        strategy = self.search_strategy
        if strategy is None:
            return

        trained_models = context.get("trained_models", {})
        oof_predictions = context.get("oof_predictions", {})
        results: list = context.get("results", [])
        meta_features = context.get("meta_features", {})

        # Sync strategy with all results collected so far
        already_synced = {r.config.config_id for r in strategy.results_} if strategy.results_ else set()
        for r in results:
            if r.config.config_id not in already_synced:
                try:
                    strategy.update(r)
                except Exception:
                    pass

        best_score = max((r.score for r in results if r.success), default=0.0)
        rounds_without_improvement = 0
        iteration = 0
        total_new_models = 0

        is_genetic = hasattr(strategy, "_evolve")  # duck-type GeneticSearch
        is_bandit = hasattr(strategy, "current_rung")  # duck-type BanditSearch
        is_adaptive = hasattr(strategy, "current_phase")  # duck-type AdaptiveSearch

        if self.verbose > 0:
            if is_adaptive:
                strategy_label = f"adaptive ({getattr(strategy, 'phase_name', '?')})"
            elif is_bandit:
                strategy_label = "bandit (successive halving)"
            elif is_genetic:
                strategy_label = "evolutionary"
            else:
                phase = "model sweep" if not getattr(strategy, "initial_sweep_done", True) else "HPO variants"
                strategy_label = f"portfolio ({phase})"
            print(
                f"\n  [AutoML] Entering continuous optimization "
                f"(patience={self.patience}, strategy={strategy_label})"
            )

        trainer = self._executors.get("model_training")
        ensembler = self._executors.get("ensembling")
        hpo_executor = self._executors.get("hyperparameter_tuning")

        try:
            while self.patience == 0 or rounds_without_improvement < self.patience:
                iteration += 1

                # ── Time budget check ────────────────────────────────
                remaining = (
                    self.time_manager_.remaining_budget()
                    if self.time_manager_ is not None
                    else float("inf")
                )
                if remaining < 30:
                    if self.verbose > 0:
                        print("  [AutoML] Time budget exhausted, stopping")
                    break

                # ── Step 1: Get new configs from strategy ────────────
                n_suggest = 5 if (is_genetic or is_bandit) else 3
                new_configs = None
                for _suggest_attempt in range(3):
                    try:
                        new_configs = strategy.suggest(
                            meta_features, n_suggestions=n_suggest,
                        )
                        break
                    except Exception as e:
                        logger.warning(
                            f"Strategy suggest failed (attempt "
                            f"{_suggest_attempt + 1}/3): {e}"
                        )
                        if self.verbose > 0:
                            print(
                                f"  [AutoML] ⚠ suggest() error: {e} "
                                f"(retry {_suggest_attempt + 1}/3)"
                            )
                        import traceback
                        traceback.print_exc()

                if not new_configs:
                    # Bandit search returns empty when all rungs are complete
                    if is_bandit and hasattr(strategy, "should_stop") and strategy.should_stop():
                        if self.verbose > 0:
                            print("  [AutoML] Bandit search completed all rungs, stopping")
                        break
                    # Genetic search returns empty when should_stop is True
                    if is_genetic and hasattr(strategy, "should_stop") and strategy.should_stop():
                        if self.verbose > 0:
                            print("  [AutoML] Evolutionary search converged, stopping")
                        break
                    if self.verbose > 0:
                        print("  [AutoML] No new candidates from strategy, stopping")
                    break

                source = new_configs[0].metadata.get("source", "")
                if is_bandit:
                    rung = new_configs[0].metadata.get("rung", "?")
                    frac = new_configs[0].metadata.get("data_fraction", 1.0)
                    phase_label = f"rung {rung} ({frac:.0%} data)"
                elif is_genetic:
                    gen = new_configs[0].metadata.get("generation", "?")
                    phase_label = f"gen {gen}"
                elif source == "portfolio_hpo_variant":
                    phase_label = "HPO variant"
                else:
                    phase_label = "new model"

                if self.verbose > 0:
                    names = []
                    for c in new_configs:
                        v = c.metadata.get("variant_num")
                        names.append(f"{c.model_name}(v{v})" if v else c.model_name)
                    rem = (
                        self.time_manager_.remaining_budget()
                        if self.time_manager_
                        else float("inf")
                    )
                    remaining = "unlimited" if rem == float("inf") else f"{rem:.0f}s left"
                    print(
                        f"  [AutoML] Iter {iteration} [{phase_label}]: "
                        f"training {names}  ({remaining})"
                    )

                # ── Step 2: Train ────────────────────────────────────
                if trainer is None:
                    break

                n_new = max(len(new_configs), 1)
                remaining_budget = (
                    self.time_manager_.remaining_budget()
                    if self.time_manager_ is not None
                    else float("inf")
                )
                if remaining_budget == float("inf"):
                    iter_budget = self.max_model_time * n_new
                else:
                    iter_budget = max(
                        self.min_model_time * n_new,
                        min(remaining_budget * 0.4, self.max_model_time * n_new),
                    )

                context["model_configs"] = new_configs
                try:
                    train_result = trainer.execute(context, iter_budget)
                except Exception as e:
                    logger.warning(f"Continuous loop training failed: {e}")
                    rounds_without_improvement += 1
                    continue

                if not train_result.success or not train_result.output:
                    rounds_without_improvement += 1
                    continue

                new_trained = train_result.output.get("trained_models", {})
                new_results = train_result.output.get("results", [])
                new_oof = train_result.output.get("oof_predictions", {})

                # For HPO variants, use a unique key so they don't overwrite
                # the original model.
                for name, model in new_trained.items():
                    if source == "portfolio_hpo_variant" and name in trained_models:
                        # Only replace if the variant is better
                        old_score = next(
                            (r.score for r in results
                             if r.success and r.config.model_name == name),
                            -float("inf"),
                        )
                        new_score = next(
                            (r.score for r in new_results
                             if r.success and r.config.model_name == name),
                            -float("inf"),
                        )
                        if new_score > old_score:
                            trained_models[name] = model
                            if self.verbose > 0:
                                print(
                                    f"    ↑ {name} improved: "
                                    f"{old_score:.4f} → {new_score:.4f}"
                                )
                    else:
                        trained_models[name] = model

                oof_predictions.update(new_oof)
                results.extend(new_results)
                total_new_models += len(new_trained)

                for r in new_results:
                    try:
                        strategy.update(r)
                    except Exception:
                        pass

                # ── Step 3: Check improvement ────────────────────────
                round_best = max(
                    (r.score for r in new_results if r.success),
                    default=0.0,
                )
                if round_best > best_score + self.min_improvement:
                    improvement = round_best - best_score
                    best_score = round_best
                    rounds_without_improvement = 0
                    if self.verbose > 0:
                        print(
                            f"  [AutoML] ★ New best score: {best_score:.4f} "
                            f"(+{improvement:.4f})"
                        )
                else:
                    rounds_without_improvement += 1
                    if self.verbose > 0:
                        print(
                            f"  [AutoML] No improvement "
                            f"({rounds_without_improvement}/{self.patience})"
                        )

                # ── Step 3b: Iterative feature selection feedback ────
                # Every 3 iterations, compute aggregate feature importances
                # from successful models and inject an informed feature
                # mask into context so the next search iteration can use it.
                if iteration % 3 == 0 and any(r.success for r in results):
                    self._update_feature_selection_feedback(
                        context, trained_models, results,
                    )
                    # Pass feedback to the search strategy if it supports it
                    mask = context.get("important_feature_mask")
                    scores = context.get("feature_importances_aggregated")
                    if mask is not None and hasattr(strategy, "set_feature_importance_feedback"):
                        strategy.set_feature_importance_feedback(mask, scores)

                # ── Step 4: Re-ensemble ──────────────────────────────
                context["trained_models"] = trained_models
                context["oof_predictions"] = oof_predictions
                context["results"] = results

                if ensembler is not None and len(trained_models) >= 2:
                    try:
                        ens_result = ensembler.execute(context, 60)
                        if ens_result.success:
                            self.stage_results_["ensembling"] = ens_result
                            if ens_result.output:
                                context.update(ens_result.output)
                    except Exception:
                        pass

                # ── Step 5: Periodic HPO on top models ───────────────
                # Skip for genetic search — it evolves HPs through mutation
                if (
                    not is_genetic
                    and hpo_executor is not None
                    and self.preset.hyperparameter_tune
                    and iteration % 3 == 0  # every 3rd iteration
                    and (self.time_manager_ is None
                         or self.time_manager_.remaining_budget() > 120)
                ):
                    hpo_budget = min(
                        120.0,
                        self.time_manager_.remaining_budget() * 0.2
                        if self.time_manager_
                        else 120.0,
                    )
                    if self.verbose > 0:
                        print(f"  [AutoML] Running HPO stage ({hpo_budget:.0f}s budget)")
                    try:
                        hpo_result = hpo_executor.execute(context, hpo_budget)
                        if hpo_result.success and hpo_result.output:
                            context.update(hpo_result.output)
                            tuning_results = hpo_result.output.get("tuning_results", [])
                            improved = [t for t in tuning_results if t.get("improved")]
                            if improved and self.verbose > 0:
                                for t in improved:
                                    print(
                                        f"    ↑ HPO improved {t['model']}: "
                                        f"{t['original_score']:.4f} → {t['tuned_score']:.4f}"
                                    )
                    except Exception as e:
                        logger.debug(f"HPO in continuous loop failed: {e}")

                self._checkpoint(context, f"continuous_iter_{iteration}")

        except KeyboardInterrupt:
            if self.verbose > 0:
                print(
                    f"\n  [AutoML] Interrupted by user after {iteration} iterations"
                )
                print(
                    f"  [AutoML] Best score so far: {best_score:.4f}  "
                    f"({total_new_models} models trained)"
                )
                print("  [AutoML] Saving best pipelines to checkpoint…")
            self._checkpoint(context, "interrupted")
            if self.verbose > 0:
                print("  [AutoML] Done — checkpoint saved.")

        else:
            if self.verbose > 0:
                print(
                    f"  [AutoML] Continuous optimization complete: "
                    f"{iteration} iters, {total_new_models} models trained, "
                    f"best={best_score:.4f}"
                )

    def get_stage_summary(self) -> pd.DataFrame:
        """Get summary of stage execution.

        Returns
        -------
        pd.DataFrame
            Summary table with stage statistics.
        """
        rows = []
        for stage_name, result in self.stage_results_.items():
            rows.append({
                "stage": stage_name,
                "success": result.success,
                "duration": result.duration,
                "error": result.error,
            })

        return pd.DataFrame(rows)
