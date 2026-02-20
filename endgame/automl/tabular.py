"""Tabular data AutoML predictor.

This module provides the TabularPredictor class for automated machine learning
on tabular (structured) data.
"""

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from endgame.automl.base import BasePredictor, DataInput, FitSummary
from endgame.automl.executors.constraint_check import DeploymentConstraints
from endgame.automl.orchestrator import PipelineOrchestrator
from endgame.automl.presets import get_preset
from endgame.automl.report import AutoMLReport, ReportGenerator
from endgame.automl.utils.data_loader import (
    DataLoader,
    infer_column_types,
    load_data,
)

if TYPE_CHECKING:
    from endgame.tracking.base import ExperimentLogger

logger = logging.getLogger(__name__)


class TabularPredictor(BasePredictor):
    """AutoML predictor for tabular data.

    This predictor automates the complete machine learning pipeline for
    structured/tabular data, including preprocessing, model selection,
    hyperparameter tuning, and ensemble building.

    Parameters
    ----------
    label : str
        Name of the target column.
    problem_type : str, default="auto"
        Type of problem: "classification", "regression", "multiclass", or "auto".
    eval_metric : str, default="auto"
        Evaluation metric. "auto" selects based on problem type.
    presets : str, default="medium_quality"
        Quality preset: "best_quality", "high_quality", "good_quality",
        "medium_quality", "fast", "exhaustive", "interpretable".
        "exhaustive" uses all 100+ models with evolutionary search.
    time_limit : int, optional
        Time limit in seconds. If None, uses preset default.
    search_strategy : str, default="portfolio"
        Search strategy: "portfolio", "heuristic", "genetic", "random",
        "bayesian", "bandit", "adaptive".
    track_experiments : bool, default=True
        Whether to track experiments to the meta-learning database.
    output_path : str, optional
        Path to save outputs (models, logs, etc.).
    random_state : int, default=42
        Random seed for reproducibility.
    verbosity : int, default=2
        Verbosity level (0=silent, 1=progress, 2=detailed, 3=debug).
    min_model_time : float, default=300.0
        Minimum time budget (in seconds) for each individual model.
        If the remaining stage budget is less than this value (and at
        least one model has already been trained), training stops
        rather than giving models an inadequately short budget.
    max_model_time : float, default=600.0
        Hard ceiling (in seconds) for any single model. If a model
        exceeds this time, its training is abandoned and the pipeline
        moves on. Prevents slow models from monopolizing the budget.
    early_stopping_rounds : int, default=50
        Early stopping patience for GBDT models (LightGBM, XGBoost,
        CatBoost, NGBoost) during cross-validation. Training halts
        when no improvement is seen for this many consecutive rounds.
        Only applies during CV scoring — the final refit uses all
        boosting rounds.
    use_gpu : bool, default=False
        Enable GPU acceleration for supported models. When True,
        training uses thread-based execution (instead of fork) to
        avoid CUDA re-initialization issues. Models that encounter
        CUDA out-of-memory errors automatically fall back to CPU.

    Attributes
    ----------
    is_fitted_ : bool
        Whether the predictor has been fitted.
    fit_summary_ : FitSummary
        Summary of the fitting process.
    problem_type_ : str
        Detected or specified problem type.
    classes_ : np.ndarray
        Class labels for classification problems.
    feature_names_ : list of str
        Names of input features.
    meta_features_ : dict
        Dataset meta-features computed during fitting.
    leaderboard_ : pd.DataFrame
        Model performance leaderboard.

    Examples
    --------
    Basic usage with CSV file:

    >>> predictor = TabularPredictor(label="target")
    >>> predictor.fit("train.csv")
    >>> predictions = predictor.predict("test.csv")

    Using presets:

    >>> predictor = TabularPredictor(label="price", presets="best_quality")
    >>> predictor.fit(train_df, time_limit=3600)
    >>> predictions = predictor.predict(test_df)

    With explicit problem type:

    >>> predictor = TabularPredictor(
    ...     label="survived",
    ...     problem_type="binary",
    ...     eval_metric="roc_auc",
    ... )
    >>> predictor.fit(train_df)
    >>> proba = predictor.predict_proba(test_df)
    """

    def __init__(
        self,
        label: str,
        problem_type: str = "auto",
        eval_metric: str = "auto",
        presets: str = "medium_quality",
        time_limit: int | None = None,
        search_strategy: str = "portfolio",
        track_experiments: bool = True,
        output_path: str | None = None,
        random_state: int = 42,
        verbosity: int = 2,
        logger: "ExperimentLogger | None" = None,
        constraints: DeploymentConstraints | None = None,
        guardrails_strict: bool = False,
        checkpoint_dir: str | None = None,
        keep_training: bool = False,
        patience: int = 5,
        min_improvement: float = 1e-4,
        min_model_time: float = 300.0,
        max_model_time: float = 600.0,
        excluded_models: list[str] | None = None,
        early_stopping_rounds: int = 50,
        use_gpu: bool = False,
    ):
        super().__init__(
            label=label,
            problem_type=problem_type,
            eval_metric=eval_metric,
            presets=presets,
            time_limit=time_limit,
            search_strategy=search_strategy,
            track_experiments=track_experiments,
            output_path=output_path,
            random_state=random_state,
            verbosity=verbosity,
            logger=logger,
        )

        # Deployment constraints and guardrail settings
        self.constraints = constraints
        self.guardrails_strict = guardrails_strict

        # Incremental checkpointing
        self.checkpoint_dir = checkpoint_dir
        self.keep_training = keep_training
        self.patience = patience
        self.min_improvement = min_improvement
        self.min_model_time = min_model_time
        self.max_model_time = max_model_time
        self.excluded_models = excluded_models or []
        self.early_stopping_rounds = early_stopping_rounds
        self.use_gpu = use_gpu

        # Tabular-specific state
        self.meta_features_: dict[str, float] | None = None
        self.leaderboard_: pd.DataFrame | None = None
        self._data_loader: DataLoader | None = None
        self._column_types: dict[str, list[str]] | None = None

        # Label encoding for string targets
        self._label_encoder: LabelEncoder | None = None

        # New artifacts from deep integration
        self._distilled_model: Any | None = None
        self._conformal_predictor: Any | None = None
        self._feature_transformers: list[Any] | None = None
        self._clean_mask: np.ndarray | None = None
        self._threshold_optimizer: Any | None = None
        self._explanations: dict[str, Any] | None = None
        self.report_: AutoMLReport | None = None

        # Stored reference for refit_full()
        self._train_data_ref: DataInput | None = None

    def fit(
        self,
        train_data: DataInput,
        tuning_data: DataInput | None = None,
        time_limit: int | None = None,
        presets: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        interpretable_only: bool = False,
        **kwargs,
    ) -> "TabularPredictor":
        """Fit the AutoML predictor on tabular data.

        Parameters
        ----------
        train_data : str, Path, DataFrame, or ndarray
            Training data. Can be a file path, DataFrame, or array.
        tuning_data : optional
            Validation/tuning data. If None, uses internal holdout.
        time_limit : int, optional
            Override the time limit.
        presets : str, optional
            Override the preset.
        hyperparameters : dict, optional
            Override hyperparameters for specific models.
        interpretable_only : bool, default=False
            If True, only use interpretable models. This limits the model
            search to glass-box models that provide human-understandable
            explanations, including:
            - GAM-style models: EBM, GAM, NAM, NODE-GAM, GAMI-Net
            - Rule-based models: CORELS, RuleFit, FURIA, Symbolic Regression
            - Sparse linear models: SLIM, FasterRisk, Linear, MARS
            - Interpretable trees: GOSDT, C5.0
            - Bayesian models: Naive Bayes, TAN, LDA
        **kwargs
            Additional arguments.

        Returns
        -------
        TabularPredictor
            The fitted predictor.
        """
        start_time = time.time()

        # Override time limit if specified
        if time_limit is not None:
            self.time_limit = time_limit

        # Override presets if specified
        if presets is not None:
            self.presets = presets
            self._preset_config = get_preset(presets)

        # Store interpretable_only setting
        self._interpretable_only = interpretable_only

        # Validate GPU availability if requested
        if self.use_gpu:
            try:
                import torch
                if not torch.cuda.is_available():
                    logger.warning(
                        "use_gpu=True but CUDA is not available. "
                        "Models will fall back to CPU."
                    )
                    if self.verbosity > 0:
                        print("  WARNING: CUDA not available, GPU models will fall back to CPU")
            except ImportError:
                logger.warning(
                    "use_gpu=True but PyTorch is not installed. "
                    "GPU acceleration requires PyTorch with CUDA support."
                )
                if self.verbosity > 0:
                    print("  WARNING: PyTorch not installed, GPU acceleration unavailable")

        if self.verbosity > 0:
            print("Beginning AutoML training for tabular data")
            print(f"  Label: {self.label}")
            print(f"  Presets: {self.presets}")
            tl_str = "unlimited" if self.time_limit == 0 else f"{self.time_limit}s"
            print(f"  Time limit: {tl_str}")
            if interpretable_only:
                print("  Mode: Interpretable models only")
            if self.use_gpu:
                print("  GPU: enabled")

        # Store data reference for refit_full()
        self._train_data_ref = train_data

        # 1. Load and validate data
        X_train, y_train = self._load_train_data(train_data)

        # 2. Load tuning data if provided
        X_val, y_val = None, None
        if tuning_data is not None:
            X_val, y_val = self._load_tuning_data(tuning_data)
        elif self._preset_config.use_holdout:
            # Create holdout from training data
            X_train, X_val, y_train, y_val = self._create_holdout(
                X_train, y_train, self._preset_config.holdout_frac
            )

        # 3. Detect problem type
        self.problem_type_ = self._detect_problem_type(y_train)

        if self.verbosity > 0:
            print(f"  Problem type: {self.problem_type_}")
            print(f"  Training samples: {len(X_train)}")
            print(f"  Features: {X_train.shape[1]}")

        # Store class labels for classification
        if self.problem_type_ in ("binary", "multiclass"):
            self.classes_ = np.unique(y_train)
            if self.verbosity > 0:
                print(f"  Classes: {len(self.classes_)}")

        # Store feature names
        if isinstance(X_train, pd.DataFrame):
            self.feature_names_ = X_train.columns.tolist()
        else:
            self.feature_names_ = [f"feature_{i}" for i in range(X_train.shape[1])]

        # Log fit start if logger provided
        self._log_fit_start(n_samples=len(X_train), n_features=X_train.shape[1])

        # 4. Initialize search strategy
        # Auto-select genetic search for the exhaustive preset
        if self.presets == "exhaustive" and self.search_strategy == "portfolio":
            self.search_strategy = "genetic"
        search_strategy = self._get_search_strategy()

        # 5. Create and run orchestrator
        # Attach constraints and guardrails settings to preset for orchestrator
        if self.constraints is not None:
            self._preset_config.constraints = self.constraints
        if self.guardrails_strict:
            self._preset_config.guardrails_strict = True

        # Build checkpoint callback
        checkpoint_cb = None
        if self.checkpoint_dir:
            _ckpt_path = Path(self.checkpoint_dir)
            _ckpt_path.mkdir(parents=True, exist_ok=True)

            def _save_checkpoint(stage_results, context, label=""):
                self._incremental_save(stage_results, context, _ckpt_path, label)

            checkpoint_cb = _save_checkpoint

        orchestrator = PipelineOrchestrator(
            preset=self._preset_config,
            time_limit=self.time_limit,
            search_strategy=search_strategy,
            verbose=self.verbosity,
            checkpoint_callback=checkpoint_cb,
            keep_training=self.keep_training,
            patience=self.patience,
            min_improvement=self.min_improvement,
            min_model_time=self.min_model_time,
            max_model_time=self.max_model_time,
            eval_metric=self.eval_metric,
            excluded_models=self.excluded_models,
            early_stopping_rounds=self.early_stopping_rounds,
            use_gpu=self.use_gpu,
        )

        # Set persistence output directory if configured
        if self.output_path and "persistence" in orchestrator._executors:
            orchestrator._executors["persistence"].output_dir = Path(self.output_path)

        # Determine task type for orchestrator
        task_type = "regression" if self.problem_type_ == "regression" else "classification"

        # Run the pipeline
        result = orchestrator.run(
            X=X_train,
            y=y_train,
            X_val=X_val,
            y_val=y_val,
            task_type=task_type,
        )

        # 6. Store results
        self._store_results(result, orchestrator)

        # 7. Track experiment if enabled
        if self.track_experiments:
            self._track_experiment(result)

        # 8. Build fit summary
        total_time = time.time() - start_time
        self._build_fit_summary(result, total_time)

        self.is_fitted_ = True

        # 9. Generate report (not a pipeline stage, runs after orchestrator)
        try:
            generator = ReportGenerator()
            self.report_ = generator.generate(result, orchestrator, self._models)
        except Exception as e:
            logger.warning(f"Report generation failed: {e}")

        # Log fit end if logger provided
        self._log_fit_end()

        if self.verbosity > 0:
            print("\nTraining complete!")
            print(f"  Total time: {total_time:.1f}s")
            print(f"  Best score: {self.fit_summary_.best_score:.4f}")
            print(f"  Best model: {self.fit_summary_.best_model}")

        if self.verbosity >= 2 and self.report_ is not None:
            # Show quality warnings if any
            if self.report_.quality_warnings:
                print(f"\n  Quality warnings: {len(self.report_.quality_warnings)}")
                for w in self.report_.quality_warnings[:5]:
                    print(f"    [{w.severity.upper()}] {w.message}")
            # Show top features if available
            if (
                self.report_.feature_importances is not None
                and not self.report_.feature_importances.empty
            ):
                top = self.report_.feature_importances.head(5)
                print("\n  Top features:")
                for _, row in top.iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")

        return self

    def predict(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate predictions.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data to predict on.
        model : str, optional
            Specific model to use. If None, uses the ensemble.

        Returns
        -------
        np.ndarray
            Predictions.
        """
        self._check_is_fitted()

        if not self._models and self._ensemble is None:
            raise RuntimeError(
                "No trained models available. All models failed during training. "
                "Check the training logs for details."
            )

        X = self._prepare_prediction_input(data)

        # Get predictions
        if model is not None and model in self._models:
            estimator = self._models[model].get("estimator")
            if estimator is None:
                raise ValueError(f"No estimator found for model '{model}'")
            predictions = estimator.predict(X)
        elif self._ensemble is not None:
            predictions = self._ensemble.predict(X)
        else:
            # Use best model
            best_model = self.fit_summary_.best_model
            if not best_model or best_model not in self._models:
                raise RuntimeError(
                    "No trained models available. All models failed during training. "
                    "Check the training logs for details."
                )
            estimator = self._models[best_model].get("estimator")
            predictions = estimator.predict(X)

        # Apply threshold optimization if available (classification only)
        if (
            self._threshold_optimizer is not None
            and self.problem_type_ in ("binary", "multiclass")
            and model is None  # Only apply to default predictions
        ):
            try:
                # Get probability predictions from the same source
                if self._ensemble is not None and hasattr(self._ensemble, "predict_proba"):
                    proba = self._ensemble.predict_proba(X)
                else:
                    best_name = self.fit_summary_.best_model
                    best_est = self._models[best_name].get("estimator")
                    if best_est is not None and hasattr(best_est, "predict_proba"):
                        proba = best_est.predict_proba(X)
                    else:
                        proba = None
                if proba is not None:
                    predictions = self._threshold_optimizer.predict(proba)
            except Exception:
                pass  # Fall back to original predictions

        # Inverse-transform encoded labels back to original strings
        if self._label_encoder is not None:
            try:
                predictions = self._label_encoder.inverse_transform(
                    predictions.astype(int)
                )
            except (ValueError, TypeError):
                pass

        return predictions

    def predict_proba(
        self,
        data: DataInput,
        model: str | None = None,
    ) -> np.ndarray:
        """Generate probability predictions (classification only).

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data.
        model : str, optional
            Specific model to use.

        Returns
        -------
        np.ndarray
            Probability predictions with shape (n_samples, n_classes).
        """
        self._check_is_fitted()

        if self.problem_type_ == "regression":
            raise ValueError("predict_proba is not available for regression")

        if not self._models and self._ensemble is None:
            raise RuntimeError(
                "No trained models available. All models failed during training."
            )

        X = self._prepare_prediction_input(data)

        # Get probability predictions
        if model is not None and model in self._models:
            estimator = self._models[model].get("estimator")
            if estimator is None:
                raise ValueError(f"No estimator found for model '{model}'")
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
            else:
                raise ValueError(f"Model '{model}' does not support predict_proba")
        elif self._ensemble is not None and hasattr(self._ensemble, "predict_proba"):
            proba = self._ensemble.predict_proba(X)
        else:
            # Use best model
            best_model = self.fit_summary_.best_model
            if not best_model or best_model not in self._models:
                raise RuntimeError(
                    "No trained models available. All models failed during training."
                )
            estimator = self._models[best_model].get("estimator")
            if hasattr(estimator, "predict_proba"):
                proba = estimator.predict_proba(X)
            else:
                raise ValueError("No model with predict_proba available")

        # Apply calibration if available
        if self._calibrator is not None:
            if proba.ndim == 2 and proba.shape[1] == 2:
                proba_1d = proba[:, 1]

                # endgame calibrators use transform() API (TransformerMixin)
                if hasattr(self._calibrator, "transform"):
                    calibrated = self._calibrator.transform(proba_1d)
                else:
                    # sklearn-style calibrators use predict_proba()
                    calibrated = self._calibrator.predict_proba(proba_1d.reshape(-1, 1))

                if calibrated.ndim == 2:
                    proba = calibrated
                else:
                    calibrated = np.asarray(calibrated).ravel()
                    proba = proba.copy()
                    proba[:, 1] = calibrated
                    proba[:, 0] = 1 - calibrated

        return proba

    def predict_distilled(
        self,
        data: DataInput,
    ) -> np.ndarray:
        """Generate predictions using the distilled model.

        The distilled model is a lightweight student model trained via
        knowledge distillation from the ensemble teacher. It provides
        faster inference while approximating ensemble accuracy.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data to predict on.

        Returns
        -------
        np.ndarray
            Predictions from the distilled model.

        Raises
        ------
        RuntimeError
            If predictor is not fitted or no distilled model is available.
        """
        self._check_is_fitted()

        if self._distilled_model is None:
            raise RuntimeError(
                "No distilled model available. Use preset='best_quality' with "
                "feature_engineering='aggressive' to enable knowledge distillation."
            )

        X = self._prepare_prediction_input(data)

        return self._distilled_model.predict(X)

    def predict_sets(
        self,
        data: DataInput,
        alpha: float = 0.1,
    ) -> np.ndarray:
        """Generate prediction sets/intervals using conformal prediction.

        For classification, returns prediction sets (sets of plausible labels).
        For regression, returns prediction intervals.

        Parameters
        ----------
        data : str, Path, DataFrame, or ndarray
            Input data to predict on.
        alpha : float, default=0.1
            Significance level (1 - coverage). E.g., alpha=0.1 gives 90% coverage.

        Returns
        -------
        np.ndarray
            For classification: boolean array of shape (n_samples, n_classes)
            indicating which classes are in each prediction set.
            For regression: array of shape (n_samples, 2) with [lower, upper] bounds.

        Raises
        ------
        RuntimeError
            If predictor is not fitted or no conformal predictor is available.
        """
        self._check_is_fitted()

        if self._conformal_predictor is None:
            raise RuntimeError(
                "No conformal predictor available. Use preset='best_quality' or "
                "'high_quality' with validation data to enable conformal prediction."
            )

        X = self._prepare_prediction_input(data)

        return self._conformal_predictor.predict(X, alpha=alpha)

    def explain(self) -> dict[str, Any]:
        """Get model explanations computed during fitting.

        Returns the SHAP-based feature importances and optionally
        interaction effects that were computed by the explainability
        stage.

        Returns
        -------
        dict
            Explanation results including ``feature_importance_df``,
            ``top_features``, and optionally ``shap_explanation`` and
            ``feature_interactions``.

        Raises
        ------
        RuntimeError
            If predictor is not fitted or no explanations available.
        """
        self._check_is_fitted()

        if self._explanations is None:
            raise RuntimeError(
                "No explanations available. Ensure the explainability stage "
                "has a non-zero time allocation in your preset."
            )
        return self._explanations

    def display_models(
        self,
        *,
        top_rules: int = 15,
        top_features: int = 10,
        print_output: bool = True,
    ) -> str:
        """Display learned structures for all trained interpretable models.

        Prints rules, trees, equations, scorecards, coefficients, and
        feature importances for every model that was trained.

        Parameters
        ----------
        top_rules : int, default=15
            Maximum rules/terms per model.
        top_features : int, default=10
            Maximum features per importance display.
        print_output : bool, default=True
            If True, print to stdout. Always returns the full text.

        Returns
        -------
        str
            Complete formatted text for all models.

        Example
        -------
        >>> predictor = TabularPredictor(label="target", presets="interpretable")
        >>> predictor.fit(train_df)
        >>> predictor.display_models()
        """
        self._check_is_fitted()
        from endgame.automl.display import display_models as _display

        fn = getattr(self, "feature_names_", None)
        models = {name: info["estimator"] for name, info in self._models.items()
                  if info.get("estimator") is not None}

        if not models:
            msg = "No trained models to display."
            if print_output:
                print(msg)
            return msg

        return _display(
            models,
            feature_names=fn,
            top_rules=top_rules,
            top_features=top_features,
            print_output=print_output,
        )

    def display_model(
        self,
        model_name: str,
        *,
        top_rules: int = 15,
        top_features: int = 10,
        print_output: bool = True,
    ) -> str:
        """Display the learned structure of a single trained model.

        Parameters
        ----------
        model_name : str
            Name of the model (as shown in the leaderboard).
        top_rules : int, default=15
            Maximum rules/terms to display.
        top_features : int, default=10
            Maximum features per importance display.
        print_output : bool, default=True
            If True, print to stdout.

        Returns
        -------
        str
            Formatted display text.

        Example
        -------
        >>> predictor.display_model("ebm")
        """
        self._check_is_fitted()
        from endgame.automl.display import display_model as _display

        info = self._models.get(model_name)
        if info is None:
            avail = ", ".join(self._models.keys()) or "(none)"
            raise KeyError(
                f"Model '{model_name}' not found. Available: {avail}"
            )

        model = info.get("estimator")
        if model is None:
            raise RuntimeError(f"Model '{model_name}' has no fitted estimator.")

        fn = getattr(self, "feature_names_", None)

        return _display(
            model_name,
            model,
            feature_names=fn,
            top_rules=top_rules,
            top_features=top_features,
            print_output=print_output,
        )

    def report(self) -> AutoMLReport:
        """Get the AutoML performance report.

        Returns
        -------
        AutoMLReport
            Structured report with summary, leaderboard, warnings, etc.

        Raises
        ------
        RuntimeError
            If predictor is not fitted or no report available.
        """
        self._check_is_fitted()

        if self.report_ is None:
            raise RuntimeError("No report available.")
        return self.report_

    def _incremental_save(
        self,
        stage_results: dict,
        context: dict,
        ckpt_path: Path,
        label: str = "",
        keep_top_n: int = 5,
    ) -> None:
        """Save the best-N pipelines to disk, replacing the previous snapshot.

        Called by the orchestrator after key stages / each continuous-loop
        iteration.  Only the *current* top-N models (by score) are kept on
        disk — stale models from earlier iterations are removed so the
        checkpoint directory doesn't grow unboundedly during long runs.

        The ensemble and preprocessor are always saved alongside.
        """
        import pickle as _pkl
        import shutil

        try:
            # ── Metadata ────────────────────────────────────────────
            meta = {
                "label": self.label,
                "problem_type": getattr(self, "problem_type_", self.problem_type),
                "presets": self.presets,
                "checkpoint_label": label,
                "checkpoint_time": time.time(),
            }
            with open(ckpt_path / "checkpoint_meta.pkl", "wb") as f:
                _pkl.dump(meta, f)

            # ── Identify top-N models ───────────────────────────────
            results = context.get("results", [])
            trained = context.get("trained_models", {})
            successful = sorted(
                [r for r in results if r.success and r.config.model_name in trained],
                key=lambda r: r.score,
                reverse=True,
            )
            # De-duplicate by model name (keep highest score)
            seen = set()
            top = []
            for r in successful:
                if r.config.model_name not in seen:
                    top.append(r)
                    seen.add(r.config.model_name)
                if len(top) >= keep_top_n:
                    break

            top_names = {r.config.model_name for r in top}

            # ── Save top-N models (atomic swap) ─────────────────────
            models_dir = ckpt_path / "models"
            models_tmp = ckpt_path / "models_new"
            if models_tmp.exists():
                shutil.rmtree(models_tmp, ignore_errors=True)
            models_tmp.mkdir(parents=True, exist_ok=True)

            from endgame.persistence import save as eg_save

            for r in top:
                name = r.config.model_name
                model = trained.get(name)
                if model is None:
                    continue
                model_dir = models_tmp / name
                model_dir.mkdir(parents=True, exist_ok=True)
                try:
                    eg_save(model, str(model_dir / "model"))
                except Exception:
                    pass

            # Swap: old models dir → delete, new → models
            models_old = ckpt_path / "models_old"
            if models_dir.exists():
                models_dir.rename(models_old)
            models_tmp.rename(models_dir)
            if models_old.exists():
                shutil.rmtree(models_old, ignore_errors=True)

            # ── Save ensemble ───────────────────────────────────────
            ensemble = context.get("ensemble")
            if ensemble is not None:
                try:
                    eg_save(ensemble, str(ckpt_path / "ensemble"))
                except Exception:
                    pass

            # ── Save preprocessor (once) ────────────────────────────
            preprocessor = context.get("preprocessor")
            if preprocessor is not None:
                try:
                    eg_save(preprocessor, str(ckpt_path / "preprocessor"))
                except Exception:
                    pass

            # ── Leaderboard CSV (full history for analysis) ─────────
            if results:
                rows = []
                for r in results:
                    if hasattr(r, "config"):
                        rows.append({
                            "model": r.config.model_name,
                            "config_id": getattr(r.config, "config_id", ""),
                            "score": r.score,
                            "fit_time": r.fit_time,
                            "success": r.success,
                            "saved": r.config.model_name in top_names,
                        })
                if rows:
                    import pandas as _pd
                    df = _pd.DataFrame(rows)
                    df.to_csv(ckpt_path / "leaderboard.csv", index=False)
                    self._append_results_to_parquet(rows, label)

            if self.verbosity >= 2:
                names = [f"{r.config.model_name}({r.score:.4f})" for r in top]
                print(
                    f"  [Checkpoint] Saved top-{len(top)} to {ckpt_path}: "
                    + ", ".join(names)
                )

        except Exception as e:
            logger.warning(f"Incremental save failed: {e}")

    def _append_results_to_parquet(self, rows: list[dict], label: str = "") -> None:
        """Append checkpoint results to benchmark_results.parquet."""
        try:
            from datetime import datetime, timezone
            from pathlib import Path as _P

            parquet_path = _P("benchmark_results.parquet")
            ts = datetime.now(timezone.utc).isoformat()

            records = []
            for r in rows:
                records.append({
                    "timestamp": ts,
                    "script": f"automl_checkpoint_{label}",
                    "dataset": getattr(self, "_dataset_name", "unknown"),
                    "model": r.get("model", ""),
                    "status": "ok" if r.get("success", False) else "failed",
                    "accuracy": r.get("score"),
                    "auc": None,
                    "fit_time_s": r.get("fit_time"),
                    "error": None,
                })

            new_df = pd.DataFrame(records)

            if parquet_path.exists():
                try:
                    existing = pd.read_parquet(parquet_path)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                except Exception:
                    combined = new_df
            else:
                combined = new_df

            combined.to_parquet(parquet_path, index=False)
        except Exception as e:
            logger.debug(f"Parquet append failed during checkpoint: {e}")

    def _prepare_prediction_input(self, data: DataInput) -> np.ndarray | pd.DataFrame:
        """Load data for prediction and apply all preprocessing/feature transforms.

        Parameters
        ----------
        data : DataInput
            Input data.

        Returns
        -------
        array-like
            Transformed features ready for model prediction.
        """
        X = self._load_prediction_data(data)

        if self._preprocessor is not None:
            X = self._preprocessor.transform(X)

        if self._feature_transformers:
            X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
            for name, transformer in self._feature_transformers:
                X_arr = transformer.transform(X_arr)
            X = X_arr

        return X

    def _load_train_data(
        self,
        data: DataInput,
    ) -> tuple:
        """Load and validate training data.

        Parameters
        ----------
        data : various
            Input training data.

        Returns
        -------
        tuple
            (X_train, y_train)
        """
        X, y = load_data(data, label=self.label)

        if y is None:
            raise ValueError(f"Target column '{self.label}' not found in training data")

        # Infer column types
        if isinstance(X, pd.DataFrame):
            self._column_types = infer_column_types(X)

        if isinstance(y, pd.Series):
            needs_encoding = not pd.api.types.is_numeric_dtype(y)
            y_arr = y.values
        else:
            y_arr = y
            needs_encoding = y_arr.dtype.kind in ('O', 'U', 'S')

        if needs_encoding:
            self._label_encoder = LabelEncoder()
            y_arr = self._label_encoder.fit_transform(y_arr)

        return X, y_arr

    def _load_tuning_data(
        self,
        data: DataInput,
    ) -> tuple:
        """Load tuning/validation data.

        Parameters
        ----------
        data : various
            Input tuning data.

        Returns
        -------
        tuple
            (X_val, y_val)
        """
        X, y = load_data(data, label=self.label)

        if y is None:
            raise ValueError(f"Target column '{self.label}' not found in tuning data")

        y_arr = y.values if isinstance(y, pd.Series) else y

        if self._label_encoder is not None:
            y_arr = self._label_encoder.transform(y_arr)

        return X, y_arr

    def _load_prediction_data(
        self,
        data: DataInput,
    ) -> pd.DataFrame | np.ndarray:
        """Load data for prediction.

        Parameters
        ----------
        data : various
            Input data.

        Returns
        -------
        array-like
            Features for prediction.
        """
        if isinstance(data, (str, Path)):
            X, _ = load_data(data, label=None)
            # Remove label column if present
            if isinstance(X, pd.DataFrame) and self.label in X.columns:
                X = X.drop(columns=[self.label])
        elif isinstance(data, pd.DataFrame):
            X = data.copy()
            if self.label in X.columns:
                X = X.drop(columns=[self.label])
        elif isinstance(data, np.ndarray):
            X = data
        else:
            raise ValueError(f"Unsupported data type: {type(data)}")

        return X

    def _create_holdout(
        self,
        X: pd.DataFrame | np.ndarray,
        y: np.ndarray,
        holdout_frac: float,
    ) -> tuple:
        """Create holdout validation set.

        Parameters
        ----------
        X : array-like
            Features.
        y : array-like
            Target.
        holdout_frac : float
            Fraction of data for holdout.

        Returns
        -------
        tuple
            (X_train, X_val, y_train, y_val)
        """
        from sklearn.model_selection import train_test_split

        stratify = y if self.problem_type != "regression" else None

        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=holdout_frac,
                random_state=self.random_state,
                stratify=stratify,
            )
        except ValueError:
            # Stratification may fail for small datasets
            X_train, X_val, y_train, y_val = train_test_split(
                X, y,
                test_size=holdout_frac,
                random_state=self.random_state,
            )

        return X_train, X_val, y_train, y_val

    def _get_search_strategy(self):
        """Get the search strategy instance.

        Returns
        -------
        BaseSearchStrategy
            Search strategy.
        """
        task_type = "regression" if self.problem_type_ == "regression" else "classification"
        interpretable_only = getattr(self, "_interpretable_only", False)

        if self.search_strategy == "portfolio":
            from endgame.automl.search.portfolio import PortfolioSearch

            return PortfolioSearch(
                task_type=task_type,
                preset=self.presets,
                eval_metric=self._get_eval_metric(),
                verbose=self.verbosity,
                interpretable_only=interpretable_only,
            )

        elif self.search_strategy == "heuristic":
            try:
                from endgame.automl.search.heuristic import HeuristicSearch

                return HeuristicSearch(
                    task_type=task_type,
                    eval_metric=self._get_eval_metric(),
                )
            except ImportError:
                logger.warning("HeuristicSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        elif self.search_strategy == "genetic":
            try:
                from endgame.automl.search.genetic import GeneticSearch

                n_gens = 100_000 if self.patience == 0 else 100
                return GeneticSearch(
                    task_type=task_type,
                    eval_metric=self._get_eval_metric(),
                    model_pool=self._preset_config.model_pool,
                    population_size=max(20, len(self._preset_config.model_pool) // 2),
                    n_generations=n_gens,
                    patience=self.patience,
                    verbose=self.verbosity,
                    random_state=self.random_state,
                    excluded_models=set(self.excluded_models),
                    max_model_time=self.max_model_time,
                )
            except ImportError:
                logger.warning("GeneticSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        elif self.search_strategy == "random":
            try:
                from endgame.automl.search.random import RandomSearch

                return RandomSearch(
                    task_type=task_type,
                    eval_metric=self._get_eval_metric(),
                )
            except ImportError:
                logger.warning("RandomSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        elif self.search_strategy == "bayesian":
            try:
                from endgame.automl.search.bayesian import BayesianSearch

                return BayesianSearch(
                    task_type=task_type,
                    eval_metric=self._get_eval_metric(),
                    model_pool=self._preset_config.model_pool,
                    random_state=self.random_state,
                    verbose=self.verbosity,
                    excluded_models=set(self.excluded_models),
                )
            except ImportError:
                logger.warning("BayesianSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        elif self.search_strategy == "adaptive":
            try:
                from endgame.automl.search.adaptive import AdaptiveSearch
                from endgame.automl.search.portfolio import PortfolioSearch

                # Phase 1: Portfolio for diverse coverage
                portfolio = PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    eval_metric=self._get_eval_metric(),
                    interpretable_only=interpretable_only,
                )
                # Phase 2: Bayesian for focused HPO
                try:
                    from endgame.automl.search.bayesian import BayesianSearch
                    bayesian = BayesianSearch(
                        task_type=task_type,
                        eval_metric=self._get_eval_metric(),
                        model_pool=self._preset_config.model_pool,
                        random_state=self.random_state,
                        verbose=self.verbosity,
                        excluded_models=set(self.excluded_models),
                    )
                except ImportError:
                    bayesian = None

                phases = [(portfolio, 15)]
                if bayesian:
                    phases.append((bayesian, 0))

                return AdaptiveSearch(
                    strategies=phases,
                    switch_patience=5,
                    verbose=self.verbosity,
                )
            except ImportError:
                logger.warning("AdaptiveSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        elif self.search_strategy == "bandit":
            try:
                from endgame.automl.search.bandit import BanditSearch

                return BanditSearch(
                    task_type=task_type,
                    eval_metric=self._get_eval_metric(),
                    model_pool=self._preset_config.model_pool,
                    max_configs=27,
                    reduction_factor=3,
                    random_state=self.random_state,
                    verbose=self.verbosity,
                    excluded_models=set(self.excluded_models),
                )
            except ImportError:
                logger.warning("BanditSearch not available, falling back to PortfolioSearch")
                from endgame.automl.search.portfolio import PortfolioSearch

                return PortfolioSearch(
                    task_type=task_type,
                    preset=self.presets,
                    interpretable_only=interpretable_only,
                )

        else:
            raise ValueError(f"Unknown search strategy: {self.search_strategy}")

    def _store_results(
        self,
        result,
        orchestrator: PipelineOrchestrator,
    ) -> None:
        """Store results from the pipeline execution.

        Parameters
        ----------
        result : PipelineResult
            Pipeline execution result.
        orchestrator : PipelineOrchestrator
            The orchestrator that ran the pipeline.
        """
        # Store preprocessor
        preprocessing_result = orchestrator.stage_results_.get("preprocessing")
        if preprocessing_result and preprocessing_result.output:
            self._preprocessor = preprocessing_result.output.get("preprocessor")

        # Store meta-features
        profiling_result = orchestrator.stage_results_.get("profiling")
        if profiling_result and profiling_result.output:
            self.meta_features_ = profiling_result.output.get("meta_features", {})

        # Store trained models — merge initial training + continuous loop
        training_result = orchestrator.stage_results_.get("model_training")

        all_trained: dict = {}
        all_oof: dict = {}
        all_results: list = []

        if training_result and training_result.output:
            all_trained.update(training_result.output.get("trained_models", {}))
            all_oof.update(training_result.output.get("oof_predictions", {}))
            all_results.extend(training_result.output.get("results", []))

        ctx = getattr(orchestrator, "_final_context", {})
        if ctx:
            all_trained.update(ctx.get("trained_models", {}))
            all_oof.update(ctx.get("oof_predictions", {}))
            seen = {id(r) for r in all_results}
            for r in ctx.get("results", []):
                if id(r) not in seen:
                    all_results.append(r)

        for model_name, model in all_trained.items():
            best_result = max(
                (r for r in all_results
                 if r.success and r.config.model_name == model_name),
                key=lambda r: r.score,
                default=None,
            )

            self._models[model_name] = {
                "estimator": model,
                "score": best_result.score if best_result else 0.0,
                "fit_time": best_result.fit_time if best_result else 0.0,
                "oof_predictions": all_oof.get(model_name),
                "n_features": len(self.feature_names_) if self.feature_names_ else 0,
            }

        # Store ensemble
        ensemble_result = orchestrator.stage_results_.get("ensembling")
        if ensemble_result and ensemble_result.output:
            self._ensemble = ensemble_result.output.get("ensemble")

        # Store calibrator
        calibration_result = orchestrator.stage_results_.get("calibration")
        if calibration_result and calibration_result.output:
            self._calibrator = calibration_result.output.get("calibrator")

        # Store data cleaning mask
        cleaning_result = orchestrator.stage_results_.get("data_cleaning")
        if cleaning_result and cleaning_result.output:
            self._clean_mask = cleaning_result.output.get("clean_mask")

        # Store feature transformers
        fe_result = orchestrator.stage_results_.get("feature_engineering")
        if fe_result and fe_result.output:
            self._feature_transformers = fe_result.output.get("feature_transformers", [])

        # Store post-training artifacts
        post_result = orchestrator.stage_results_.get("post_training")
        if post_result and post_result.output:
            self._distilled_model = post_result.output.get("distilled_model")
            self._conformal_predictor = post_result.output.get("conformal_predictor")

        # Store threshold optimizer
        threshold_result = orchestrator.stage_results_.get("threshold_opt")
        if threshold_result and threshold_result.output:
            self._threshold_optimizer = threshold_result.output.get("threshold_optimizer")

        # Store explanations
        explain_result = orchestrator.stage_results_.get("explainability")
        if explain_result and explain_result.output:
            self._explanations = explain_result.output.get("explanations")

        # Build leaderboard
        self._build_leaderboard()

    def _build_leaderboard(self) -> None:
        """Build the model leaderboard."""
        rows = []
        for name, info in self._models.items():
            rows.append({
                "model": name,
                "score": info.get("score", 0.0),
                "fit_time": info.get("fit_time", 0.0),
            })

        self.leaderboard_ = pd.DataFrame(rows)
        if len(rows) > 0:
            self.leaderboard_ = self.leaderboard_.sort_values(
                "score", ascending=False
            ).reset_index(drop=True)

    def _build_fit_summary(
        self,
        result,
        total_time: float,
    ) -> None:
        """Build the fit summary.

        Parameters
        ----------
        result : PipelineResult
            Pipeline execution result.
        total_time : float
            Total training time.
        """
        # Count models
        n_trained = len(self._models)
        n_failed = result.metadata.get("n_models_failed", 0)

        # Find best model
        best_model = ""
        best_score = 0.0
        if self._models:
            best_model = max(
                self._models.keys(),
                key=lambda k: self._models[k].get("score", 0.0)
            )
            best_score = self._models[best_model].get("score", 0.0)

        # Get stage times
        stage_times = {}
        for stage_name, stage_result in result.stage_results.items():
            stage_times[stage_name] = stage_result.duration

        self.fit_summary_ = FitSummary(
            total_time=total_time,
            n_models_trained=n_trained,
            n_models_failed=n_failed,
            best_model=best_model,
            best_score=best_score,
            cv_score=result.score,
            stage_times=stage_times,
        )

    def _track_experiment(self, result) -> None:
        """Track the experiment to the meta-learning database and benchmark parquet.

        Parameters
        ----------
        result : PipelineResult
            Pipeline execution result.
        """
        try:
            from endgame.benchmark.tracker import ExperimentTracker

            tracker = ExperimentTracker(name="automl", auto_save=True)

            for model_name, model_info in self._models.items():
                tracker.log_experiment(
                    dataset_name=f"automl_{id(self)}",
                    model_name=model_name,
                    hyperparameters=model_info.get("estimator", {}).get_params()
                    if hasattr(model_info.get("estimator", {}), "get_params")
                    else {},
                    metrics={"score": model_info.get("score", 0.0)},
                    meta_features=self.meta_features_ or {},
                    fit_time=model_info.get("fit_time", 0.0),
                    task_type=self.problem_type_,
                )

            tracker.save_to_master()

        except ImportError:
            logger.debug("ExperimentTracker not available, skipping experiment tracking")
        except Exception as e:
            logger.warning(f"Failed to track experiment: {e}")

        self._save_to_benchmark_parquet()

    def _save_to_benchmark_parquet(self) -> None:
        """Append individual model results to benchmark_results.parquet."""
        try:
            from datetime import datetime, timezone
            from pathlib import Path

            import pandas as pd

            parquet_path = Path("benchmark_results.parquet")
            ts = datetime.now(timezone.utc).isoformat()

            rows = []
            for model_name, model_info in self._models.items():
                rows.append({
                    "timestamp": ts,
                    "script": "automl",
                    "dataset": getattr(self, "_dataset_name", "unknown"),
                    "model": model_name,
                    "status": "ok" if model_info.get("score") is not None else "failed",
                    "accuracy": model_info.get("score"),
                    "auc": None,
                    "fit_time_s": model_info.get("fit_time"),
                    "error": None,
                })

            new_df = pd.DataFrame(rows)

            if parquet_path.exists():
                try:
                    existing = pd.read_parquet(parquet_path)
                    combined = pd.concat([existing, new_df], ignore_index=True)
                except Exception:
                    combined = new_df
            else:
                combined = new_df

            combined.to_parquet(parquet_path, index=False)
            logger.info(
                f"Model results appended to {parquet_path} ({len(combined)} total rows)"
            )
        except Exception as e:
            logger.warning(f"Failed to save to benchmark parquet: {e}")

    def get_model_names(self) -> list[str]:
        """Get names of all trained models.

        Returns
        -------
        list of str
            Model names.
        """
        self._check_is_fitted()
        return list(self._models.keys())

    def get_model(self, name: str):
        """Get a specific trained model.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        estimator
            The trained model.
        """
        self._check_is_fitted()

        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")

        return self._models[name].get("estimator")

    def refit_full(self, data: DataInput | None = None) -> "TabularPredictor":
        """Retrain best model(s) on all available data (train + validation).

        After cross-validation identifies the best model and hyperparameters,
        this method retrains on the full dataset for maximum deployment
        performance.

        Parameters
        ----------
        data : DataInput, optional
            Full dataset including the label column. If None, uses the
            training data from the last ``fit()`` call.

        Returns
        -------
        TabularPredictor
            Self with models retrained on full data.
        """
        self._check_is_fitted()

        if not self._models:
            raise RuntimeError("No models available to refit.")

        # Resolve data source
        if data is None:
            data = self._train_data_ref
        if data is None:
            raise ValueError(
                "No data available for refitting. Pass data explicitly "
                "or ensure fit() was called with a reusable data reference."
            )

        # Load full data (no holdout split)
        X_full, y_full = self._load_train_data(data)

        # Apply preprocessing and feature transforms
        if self._preprocessor is not None:
            X_full = self._preprocessor.transform(X_full)

        if self._feature_transformers:
            X_arr = X_full.values if isinstance(X_full, pd.DataFrame) else np.asarray(X_full)
            for name, transformer in self._feature_transformers:
                X_arr = transformer.transform(X_arr)
            X_full = X_arr

        logger.warning(
            "refit_full(): Retraining on all data. "
            "The resulting model cannot be evaluated with a holdout set."
        )

        # Refit each model on the full dataset
        for name, model_info in self._models.items():
            estimator = model_info.get("estimator")
            if estimator is None:
                continue

            try:
                estimator.fit(X_full, y_full)
                if self.verbosity > 0:
                    print(f"  Refitted {name} on {len(X_full)} samples")
            except Exception as e:
                logger.warning(f"Failed to refit model '{name}': {e}")

        # Refit ensemble if present
        if self._ensemble is not None:
            try:
                self._ensemble.fit(X_full, y_full)
            except Exception as e:
                logger.warning(f"Failed to refit ensemble: {e}")

        return self
